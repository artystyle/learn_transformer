import torch
import torch.nn as nn
import math

# ------------------------------------------------------------
# 1. Механизм многоголового внимания (Multi-Head Attention)
# ------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    Класс реализует механизм многоголового внимания (из статьи "Attention Is All You Need").
    
    Принцип работы:
    Входные данные (Q, K, V) проецируются в пространства запросов, ключей и значений.
    Затем они разделяются на nhead "голов", где каждая голова независимо вычисляет внимание.
    Результаты всех голов конкатенируются и снова проецируются линейным слоем.
    
    Args:
        d_model: Размерность входных векторов (обычно 512).
        nhead: Количество "голов" внимания (обычно 8).
    """
    def __init__(self, d_model=512, nhead=8):  # ИСПРАВЛЕНО: было **init**
        super().__init__()
        assert d_model % nhead == 0, "d_model должно делиться на nhead без остатка"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead   # Размерность векторов для одной головы (d_model / nhead)
        
        # Линейные слои для проекции Q, K, V.
        # Они трансформируют вход d_model -> d_model, чтобы затем разделить на головы.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Выходной линейный слой после объединения результатов всех голов
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 1) Линейное преобразование и изменение формы (reshape)
        # Исходная форма: [batch, seq_len, d_model]
        # Превращаем в: [batch, seq_len, nhead, d_k]
        Q = self.W_q(Q).view(batch_size, -1, self.nhead, self.d_k)
        K = self.W_k(K).view(batch_size, -1, self.nhead, self.d_k)
        V = self.W_v(V).view(batch_size, -1, self.nhead, self.d_k)
        
        # 2) Транспонирование для удобства матричного умножения
        # Меняем местами оси seq_len и nhead, чтобы получить форму:
        # [batch, nhead, seq_len, d_k]
        # Это позволяет выполнить матричное умножение Q @ K^T для всех голов сразу.
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 3) Вычисление "сырых" оценок внимания (scores)
        # Формула: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
        # Деление на sqrt(d_k) стабилизирует градиенты при обучении.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 4) Применение маски (если передана)
        # Маска используется, чтобы игнорировать паддинги или будущие токены.
        # Там, где маска равна 0, мы ставим -inf, чтобы после softmax там было 0.
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # 5) Нормализация весов через Softmax по последней оси (по длине последовательности)
        attn_weights = torch.softmax(scores, dim=-1)   # Форма: [batch, nhead, seq_len_q, seq_len_k]
        
        # 6) Взвешенная сумма значений V
        out = torch.matmul(attn_weights, V)            # Форма: [batch, nhead, seq_len_q, d_k]
        
        # 7) Объединение голов обратно в единую матрицу
        # Транспонируем обратно: [batch, seq_len, nhead, d_k]
        out = out.transpose(1, 2).contiguous()
        # Сглаживаем последние два измерения: [batch, seq_len, d_model]
        out = out.view(batch_size, -1, self.d_model)
        
        # 8) Финальная линейная проекция
        out = self.out_linear(out)
        return out

# ------------------------------------------------------------
# 2. Позиционное кодирование (Sinusoidal Positional Encoding)
# ------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Добавляет информацию о порядке токенов в последовательности.
    
    Поскольку механизм внимания перестановочно-инвариантен (порядок не важен для него),
    мы явно добавляем векторы позиций. Используются синусоидальные функции,
    так как они позволяют модели легко обучаться относительным позициям.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):  # ИСПРАВЛЕНО: было **init**
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Создаем матрицу позиционных кодирований [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # Вектор позиций [0, 1, ..., max_len-1], форма [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Вычисляем знаменатель для аргумента синуса/косинуса:
        # div_term[i] = 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        
        # Заполняем четные индексы синусом, нечетные — косинусом
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Добавляем размерность batch (чтобы потом можно было сложить с эмбеддингами)
        pe = pe.unsqueeze(0)   # Форма: [1, max_len, d_model]
        
        # Регистрируем как буфер, чтобы он сохранялся в state_dict, но не был параметром для обучения
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # Берем нужные строки из матрицы позиций (до длины seq_len)
        # и прибавляем к эмбеддингам токенов.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ------------------------------------------------------------
# 3. Блок энкодера (Encoder Block)
# ------------------------------------------------------------
class EncoderBlock(nn.Module):
    """
    Один слой энкодера Transformer.
    Архитектура:
        Input -> [MultiHeadSelfAttention] -> Add & Norm -> [FeedForward] -> Add & Norm -> Output
    
    Остаточные соединения (Add) и нормализация слоев (Norm) помогают стабильно обучать глубокие сети.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):  # ИСПРАВЛЕНО: было **init**
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-Forward Network (FFN): два линейных слоя с активацией ReLU
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1) Self-attention с остаточным соединением
        # Q, K, V берутся из одного источника x
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 2) Feed-forward с остаточным соединением
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x

# ------------------------------------------------------------
# 4. Блок декодера (Decoder Block)
# ------------------------------------------------------------
class DecoderBlock(nn.Module):
    """
    Один слой декодера Transformer.
    Архитектура:
        1. Masked Self-Attention (видит только предыдущие токены)
        2. Cross-Attention (видит выход энкодера)
        3. Feed-Forward
    
    Каждый блок имеет свой LayerNorm и остаточное соединение.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):  # ИСПРАВЛЕНО: было **init**
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1) Masked Self-attention
        # Внутреннее внимание декодера. tgt_mask запрещает видеть будущие токены.
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        
        # 2) Cross-attention
        # Запросы (Q) идут из декодера (x), а Ключи (K) и Значения (V) — из энкодера.
        # Это позволяет декодеру "смотреть" на контекст исходного предложения.
        cross_attn_out = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        
        # 3) Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        
        return x

# ------------------------------------------------------------
# 5. Полный Transformer (Encoder + Decoder)
# ------------------------------------------------------------
class Transformer(nn.Module):
    """
    Полная модель Transformer для задач перевода (seq2seq).
    """
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, vocab_size=10000, max_len=5000):  # ИСПРАВЛЕНО: было **init**
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Слой эмбеддингов: переводит индексы слов в векторы
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Позиционное кодирование (общее для энкодера и декодера)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Создаем стек слоев энкодера
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)  # ИСПРАВЛЕНО: было for _in
        ])
        
        # Создаем стек слоев декодера
        self.decoder = nn.ModuleList([
            DecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)  # ИСПРАВЛЕНО: было for _in
        ])
        
        # Финальный линейный слой: проецирует скрытые состояния декодера на размер словаря (логиты)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Инициализация весов методом Xavier Uniform
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: [batch, src_len] – индексы слов исходного языка.
            tgt: [batch, tgt_len] – индексы слов целевого языка (обычно сдвинуты).
            src_mask: маска паддингов для источника.
            tgt_mask: маска "последующих токенов" + паддинги для цели.
        """
        # 1) Эмбеддинг и позиционное кодирование
        # Умножаем на sqrt(d_model) для стабилизации масштаба (стандарт в оригинальной статье)
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)   # [batch, src_len, d_model]
        
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)   # [batch, tgt_len, d_model]
        
        # 2) Проход через энкодер
        enc_out = src_emb
        for layer in self.encoder:
            enc_out = layer(enc_out, src_mask)
            
        # 3) Проход через декодер
        dec_out = tgt_emb
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
            
        # 4) Проекция на словарь
        logits = self.output_layer(dec_out)   # [batch, tgt_len, vocab_size]
        return logits

# ------------------------------------------------------------
# 6. Вспомогательные функции для масок
# ------------------------------------------------------------
def generate_square_subsequent_mask(seq_len, device='cpu'):
    """
    Генерирует маску треугольной формы для авторегрессивной генерации.
    Запрещает токенам видеть будущие токены (нижний треугольник = 1, верхний = 0).
    """
    # torch.triu возвращает верхний треугольник. Мы инвертируем его, чтобы получить разрешенные связи.
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    mask = ~mask  # True = можно смотреть, False = нельзя
    return mask

def create_padding_mask(seq, pad_idx=0):
    """
    Создает маску, где 1 стоит там, где реальные токены, и 0 там, где паддинг (pad_idx).
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)   # [batch, 1, 1, seq_len]
    return mask

# ------------------------------------------------------------
# 7. Учебный пример использования
# ------------------------------------------------------------
if __name__ == "__main__":  # ИСПРАВЛЕНО: было **name**
    # Гиперпараметры для быстрого теста
    d_model = 128
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    vocab_size = 50       # Маленький словарь
    batch_size = 4
    src_len = 10
    tgt_len = 8
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Инициализация модели
    model = Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        vocab_size=vocab_size,
        max_len=100
    ).to(device)
    
    # Случайные данные
    src = torch.randint(0, vocab_size, (batch_size, src_len)).to(device)
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len)).to(device)
    
    # Маски
    src_padding_mask = create_padding_mask(src, pad_idx=0)
    
    # Комбинированная маска для декодера:
    # 1. Маска последующих токенов (чтобы не видеть будущее)
    tgt_subsequent_mask = generate_square_subsequent_mask(tgt_len, device)
    # 2. Маска паддингов целевой последовательности
    tgt_padding_mask = create_padding_mask(tgt, pad_idx=0)
    
    # Объединение масок
    # tgt_subsequent_mask имеет форму [tgt_len, tgt_len]
    # tgt_padding_mask имеет форму [batch, 1, 1, tgt_len]
    # Нам нужно [batch, nhead, tgt_len, tgt_len] для механизма внимания, 
    # но в простой реализации достаточно логического умножения, если batch не влияет на последовательность
    # Для корректности broadcast-инга:
    tgt_mask_sub = tgt_subsequent_mask.unsqueeze(0).unsqueeze(0) # [1, 1, L, L]
    tgt_mask_pad = tgt_padding_mask.unsqueeze(-1)                # [B, 1, 1, L] -> [B, 1, L, L] после умножения? 
    # Упрощенный вариант для демонстрации (предполагает, что паддинги совпадают или их нет в начале):
    tgt_mask = tgt_mask_sub & tgt_padding_mask.unsqueeze(-1) 

    # Прямой проход (Forward Pass)
    logits = model(src, tgt, src_mask=src_padding_mask, tgt_mask=tgt_mask)
    
    print("Форма входного src (индексы):", src.shape)       # [4, 10]
    print("Форма входного tgt (индексы):", tgt.shape)       # [4, 8]
    print("Форма выходных логитов:", logits.shape)          # [4, 8, 50]
    
    # Пример вычисления потерь
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    loss = loss_fn(logits.view(-1, vocab_size), tgt.view(-1))
    print("Значение loss для случайных данных:", loss.item())