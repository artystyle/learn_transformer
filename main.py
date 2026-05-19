import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Компоненты архитектуры
# ==========================================

class PositionalEncoding(nn.Module):
    """
    Добавляет информацию о позиции токена в последовательности.
    Без этого трансформер не различал бы порядок слов, так как у него нет рекуррентности.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: размерность эмбеддингов (должна совпадать с размерностью модели)
            max_len: максимальная длина последовательности, которую поддерживаем
            dropout: вероятность обнуления для регуляризации
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Создаем матрицу позиционных кодирований [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # Позиции: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Делитель для частот: exp(-log(10000) * (2i/d_model))
        # Для четных и нечетных индексов используются разные частоты
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Четные индексы - синус
        pe[:, 0::2] = torch.sin(position * div_term)
        # Нечетные индексы - косинус
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Добавляем batch-измерение: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Регистрируем как буфер (не обучаемый параметр, но часть модуля)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [B, L, D] - входная последовательность эмбеддингов
        Returns:
            [B, L, D] - эмбеддинги с добавленной позиционной информацией
        """
        # Добавляем позиционное кодирование к входу (обрезаем до нужной длины)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Многоголовое внимание - ключевой механизм трансформера.
    Позволяет модели фокусироваться на разных частях последовательности одновременно.
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        """
        Args:
            d_model: размерность модели (должна делиться на nhead)
            nhead: количество голов внимания
            dropout: dropout для весов внимания
        """
        super().__init__()
        assert d_model % nhead == 0, "d_model должен делиться на nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead  # Размерность каждой головы
        
        # Линейные слои для проекций Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Финальная проекция после конкатенации голов
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Вычисляет внимание между query и key-value парами.
        
        Args:
            query: [B, L_q, D] - запросы (обычно из декодера)
            key:   [B, L_k, D] - ключи (обычно из энкодера или того же запроса)
            value: [B, L_k, D] - значения (обычно из энкодера или того же запроса)
            attn_mask: [L_q, L_k] or [B, L_q, L_k] - маска для внимания (например, look-ahead)
            key_padding_mask: [B, L_k] - маска паддингов (True = игнорировать)
        
        Returns:
            [B, L_q, D] - результат внимания
        """
        B, L_q = query.size(0), query.size(1)
        L_k = key.size(1)
        
        # Шаг 1: Линейные проекции
        Q = self.W_q(query)  # [B, L_q, D]
        K = self.W_k(key)    # [B, L_k, D]
        V = self.W_v(value)  # [B, L_k, D]
        
        # Шаг 2: Разделение на головы и перестановка измерений
        # [B, L, H, D_k] -> [B, H, L, D_k]
        Q = Q.view(B, L_q, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(B, L_k, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(B, L_k, self.nhead, self.d_k).transpose(1, 2)
        
        # Шаг 3: Вычисление весов внимания (scaled dot-product)
        # scores: [B, H, L_q, L_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Шаг 4: Применение масок
        
        # 4a. Маска внимания (например, look-ahead для декодера)
        if attn_mask is not None:
            # Приводим attn_mask к правильной размерности
            if attn_mask.dim() == 2:
                # [L_q, L_k] -> [1, 1, L_q, L_k]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # [B, L_q, L_k] -> [B, 1, L_q, L_k]
                attn_mask = attn_mask.unsqueeze(1)
            # Заполняем -inf, чтобы после softmax эти позиции стали нулями
            scores = scores.masked_fill(attn_mask == float('-inf'), float('-inf'))
        
        # 4b. Маска паддингов (игнорируем специальные токены <pad>)
        if key_padding_mask is not None:
            # key_padding_mask: [B, L_k] -> [B, 1, 1, L_k]
            kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(kpm, float('-inf'))
        
        # Шаг 5: Softmax и dropout
        attn_weights = nn.functional.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Шаг 6: Применяем веса к значениям
        out = torch.matmul(attn_weights, V)  # [B, H, L_q, D_k]
        
        # Шаг 7: Объединяем головы обратно
        # [B, H, L_q, D_k] -> [B, L_q, H, D_k] -> [B, L_q, D]
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        
        # Шаг 8: Финальная проекция
        out = self.W_o(out)
        return out


class TransformerEncoderLayer(nn.Module):
    """
    Один слой энкодера трансформера.
    Состоит из: Self-Attention -> Add & Norm -> Feed-Forward -> Add & Norm
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Self-attention (каждый токен взаимодействует со всеми токенами входа)
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        # Feed-forward network (два линейных слоя с активацией)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization для стабилизации обучения
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout для регуляризации
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: [B, L, D] - входная последовательность
            src_mask: [L, L] - маска внимания (обычно None для энкодера)
            src_key_padding_mask: [B, L] - маска паддингов
        """
        # Подблок 1: Self-attention с остаточным соединением и нормой
        attn_out = self.self_attn(
            src, src, src,  # Q=src, K=src, V=src - self-attention
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        # Остаточное соединение: добавляем исходный вход
        src = self.norm1(src + self.dropout1(attn_out))
        
        # Подблок 2: Feed-forward network с остаточным соединением и нормой
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff_out))
        
        return src


class TransformerDecoderLayer(nn.Module):
    """
    Один слой декодера трансформера.
    Состоит из: Masked Self-Attention -> Cross-Attention -> Feed-Forward
    Каждый с добавлением остаточных соединений и LayerNorm.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Замаскированное self-attention (декодер не видит будущие токены)
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        # Cross-attention (декодер смотрит на выход энкодера)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer Normalization (три штуки - для каждого подблока)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout для регуляризации
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: [B, L_t, D] - целевая последовательность (декодер)
            memory: [B, L_m, D] - выход энкодера
            tgt_mask: [L_t, L_t] - look-ahead маска для self-attention
            memory_mask: [L_t, L_m] - маска для cross-attention (обычно None)
            tgt_key_padding_mask: [B, L_t] - маска паддингов в tgt
            memory_key_padding_mask: [B, L_m] - маска паддингов в memory
        """
        # Подблок 1: Masked Self-attention
        # Декодер смотрит на уже сгенерированные токены, не видя будущие
        tgt2 = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        
        # Подблок 2: Cross-attention
        # Декодер (query) смотрит на память энкодера (key, value)
        tgt2 = self.multihead_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        
        # Подблок 3: Feed-forward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        
        return tgt


class TransformerEncoder(nn.Module):
    """
    Стек слоёв энкодера.
    Просто последовательно применяет N одинаковых слоёв.
    """
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # Создаем список из num_layers копий одного слоя
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output


class TransformerDecoder(nn.Module):
    """
    Стек слоёв декодера.
    Последовательно применяет N одинаковых слоёв.
    """
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        return output


class Transformer(nn.Module):
    """
    Полная модель Transformer (энкодер + декодер).
    Для задач seq2seq: перевод, суммирование, генерация.
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Входной слой: преобразуем индексы токенов в эмбеддинги
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Позиционное кодирование
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Стек энкодера
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Стек декодера
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Выходной слой: проецируем обратно на размер словаря
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов для стабильного обучения"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Полный проход модели (обычно для обучения).
        
        Args:
            src: [B, L_src] - исходные токены
            tgt: [B, L_tgt] - целевые токены (сдвинутые для teacher forcing)
            src_mask, tgt_mask, memory_mask: различные маски внимания
            src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask: маски паддингов
        """
        # Энкодер: входная последовательность
        src = self.embedding(src) * math.sqrt(self.d_model)  # Масштабирование эмбеддингов
        src = self.pos_encoder(src)
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        # Декодер: целевая последовательность с использованием памяти энкодера
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Выход: логиты для каждого токена словаря
        return self.output_layer(output)

    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Только энкодер (для инференса).
        Кэшируем память для последующего использования декодером.
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        return self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None,
               tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Только декодер (для инференса).
        Использует заранее вычисленную память энкодера.
        """
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.output_layer(output)


# ==========================================
# 2. Токенизация
# ==========================================

class CharTokenizer:
    """
    Простой символьный токенизатор.
    Каждый символ - отдельный токен.
    Идеален для задач, работающих с текстом на уровне символов.
    """
    def __init__(self):
        # 26 букв + пробел
        chars = list("abcdefghijklmnopqrstuvwxyz ")
        
        # Специальные токены
        self.specials = ['<pad>', '<sos>', '<eos>', '<unk>']
        
        # Словарь: индекс <-> символ
        self.idx2char = self.specials + chars
        self.char2idx = {c: i for i, c in enumerate(self.idx2char)}
        
        # Удобные константы
        self.pad_idx = self.char2idx['<pad>']
        self.sos_idx = self.char2idx['<sos>']
        self.eos_idx = self.char2idx['<eos>']
        self.unk_idx = self.char2idx['<unk>']

    def encode(self, text):
        """
        Преобразует строку в список индексов.
        Добавляет <sos> в начало и <eos> в конец.
        """
        return [self.sos_idx] + [self.char2idx.get(c, self.unk_idx) for c in text] + [self.eos_idx]

    def decode(self, indices, skip_special=True):
        """
        Преобразует список индексов обратно в строку.
        По умолчанию пропускает специальные токены.
        """
        chars = []
        for i in indices:
            c = self.idx2char[i]
            if skip_special and c in self.specials:
                continue
            chars.append(c)
        return ''.join(chars)

    def __len__(self):
        return len(self.idx2char)


# ==========================================
# 3. Датасет: обращение строк
# ==========================================

class ReverseStringDataset(Dataset):
    """
    Датасет для задачи обращения строк.
    Генерирует пары (исходная_строка, перевернутая_строка).
    """
    def __init__(self, num_samples=10000, min_len=3, max_len=10, tokenizer=None):
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data = []
        
        chars = "abcdefghijklmnopqrstuvwxyz "
        
        # Генерация случайных примеров
        for _ in range(num_samples):
            length = np.random.randint(min_len, max_len + 1)
            src_str = ''.join(np.random.choice(list(chars), size=length))
            tgt_str = src_str[::-1]  # Обращение строки
            src_ids = tokenizer.encode(src_str)
            tgt_ids = tokenizer.encode(tgt_str)
            self.data.append((src_ids, tgt_ids))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.data[idx]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_fn(batch, pad_idx):
    """
    Функция для объединения примеров в батч.
    Добавляет паддинги до максимальной длины в батче.
    """
    src_list, tgt_list = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


# ==========================================
# 4. Вспомогательные функции
# ==========================================

def generate_square_subsequent_mask(sz):
    """
    Создает look-ahead маску для декодера.
    Запрещает модели видеть будущие токены на каждом шаге.
    
    Пример для sz=4:
    [[0, -inf, -inf, -inf],
     [0,   0, -inf, -inf],
     [0,   0,   0, -inf],
     [0,   0,   0,   0]]
    
    Где 0 = разрешено, -inf = запрещено (после softmax станет 0)
    """
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


def greedy_decode(model, src, src_key_padding_mask, tokenizer, max_len=50, device='cpu'):
    """
    Жадное декодирование на инференсе.
    На каждом шаге выбирает токен с максимальной вероятностью.
    
    Процесс:
    1. Кодируем исходную последовательность один раз (получаем memory)
    2. Начинаем с токена <sos>
    3. На каждом шаге подаем всю сгенерированную последовательность в декодер
    4. Выбираем самый вероятный следующий токен
    5. Добавляем его к выходу
    6. Останавливаемся при генерации <eos> или достижении max_len
    """
    model.eval()  # Переключаем в режим оценки (отключаем dropout)
    
    # Кодируем вход (один раз)
    memory = model.encode(src, src_key_padding_mask=src_key_padding_mask)
    
    # Начинаем с токена начала последовательности
    ys = torch.tensor([[tokenizer.sos_idx]], device=device)
    
    for _ in range(max_len - 1):
        # Маска, чтобы не видеть будущие токены
        tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
        
        # Декодируем текущую последовательность
        out = model.decode(ys, memory, tgt_mask=tgt_mask, 
                          memory_key_padding_mask=src_key_padding_mask)
        
        # Берем последний токен и выбираем наиболее вероятный
        prob = out[:, -1, :]
        next_token = prob.argmax(dim=-1, keepdim=True)
        
        # Добавляем к выходу
        ys = torch.cat([ys, next_token], dim=1)
        
        # Если сгенерировали <eos> - останавливаемся
        if next_token.item() == tokenizer.eos_idx:
            break
    
    return ys[0].tolist()


# ==========================================
# 5. Параметры и инициализация
# ==========================================

# Гиперпараметры (уменьшены для быстрого обучения на CPU)
BATCH_SIZE = 64
EPOCHS = 15
LR = 0.001
D_MODEL = 128          # Размерность модели
NHEAD = 4              # Количество голов внимания
NUM_LAYERS = 3         # Количество слоев в энкодере/декодере
DROPOUT = 0.1          # Вероятность dropout
FEEDFORWARD_DIM = 256  # Размер скрытого слоя в FFN

# Создаем токенизатор
tokenizer = CharTokenizer()
vocab_size = len(tokenizer)
print(f"Размер словаря: {vocab_size}")

# Создаем датасеты
train_dataset = ReverseStringDataset(
    num_samples=20000, min_len=3, max_len=12, tokenizer=tokenizer
)
val_dataset = ReverseStringDataset(
    num_samples=200, min_len=3, max_len=12, tokenizer=tokenizer
)

# Создаем DataLoader'ы
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=lambda b: collate_fn(b, tokenizer.pad_idx)
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=lambda b: collate_fn(b, tokenizer.pad_idx)
)

# Определяем устройство (GPU если доступно)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Создаем модель
model = Transformer(
    vocab_size=vocab_size, d_model=D_MODEL, nhead=NHEAD, 
    num_layers=NUM_LAYERS, dim_feedforward=FEEDFORWARD_DIM,
    dropout=DROPOUT
).to(device)

# Функция потерь (игнорируем паддинги при подсчете ошибки)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

# Оптимизатор (Adam - хороший выбор для трансформеров)
optimizer = optim.Adam(model.parameters(), lr=LR)


# ==========================================
# 6. Цикл обучения
# ==========================================

print("Начинаем обучение...\n")

for epoch in range(EPOCHS):
    model.train()  # Переключаем в режим обучения
    total_loss = 0
    
    for batch_idx, (src, tgt) in enumerate(train_loader):
        # Перемещаем данные на устройство (GPU/CPU)
        src, tgt = src.to(device), tgt.to(device)
        
        # Teacher forcing: вход для декодера = все токены кроме последнего
        # Целевые выходы = все токены кроме первого
        tgt_input = tgt[:, :-1]   # [B, L_tgt-1] (без <eos>)
        tgt_output = tgt[:, 1:]   # [B, L_tgt-1] (без <sos>)
        
        # Создаем маски
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        src_key_padding_mask = (src == tokenizer.pad_idx)  # True для паддингов
        tgt_key_padding_mask = (tgt_input == tokenizer.pad_idx)
        
        # Forward pass
        output = model(
            src, tgt_input,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # Вычисляем loss (сравниваем с реальными выходами)
        # output: [B, L_tgt-1, vocab_size] -> [B*(L_tgt-1), vocab_size]
        # tgt_output: [B, L_tgt-1] -> [B*(L_tgt-1)]
        loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Обрезка градиентов для стабильности обучения
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Логирование
        if batch_idx % 200 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx} | Loss: {loss.item():.4f}")
    
    # Средний loss за эпоху
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} завершена. Средний loss: {avg_loss:.4f}")
    
    # Валидация на одном примере после каждой эпохи
    model.eval()
    with torch.no_grad():
        src_sample, tgt_sample = val_dataset[0]
        src_tensor = src_sample.unsqueeze(0).to(device)  # Добавляем batch dimension
        src_mask = (src_tensor == tokenizer.pad_idx)
        
        # Генерируем предсказание
        pred_ids = greedy_decode(model, src_tensor, src_mask, tokenizer, device=device)
        
        # Декодируем для вывода
        src_text = tokenizer.decode(src_sample.tolist())
        tgt_text = tokenizer.decode(tgt_sample.tolist())
        pred_text = tokenizer.decode(pred_ids)
        
        print(f"  Пример: '{src_text}' -> Истина: '{tgt_text}' | Предсказание: '{pred_text}'\n")


# ==========================================
# 7. Тестирование на новых примерах
# ==========================================

print("\n--- Тестирование ---")

test_samples = ["hello world", "transformer", "deep learning", "pytorch"]

for s in test_samples:
    # Токенизируем вход
    src_ids = tokenizer.encode(s)
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
    src_mask = (src_tensor == tokenizer.pad_idx)
    
    # Генерируем предсказание
    pred_ids = greedy_decode(model, src_tensor, src_mask, tokenizer, device=device)
    pred_str = tokenizer.decode(pred_ids)
    
    # Выводим результат
    print(f"'{s}' -> '{pred_str}' (ожидалось: '{s[::-1]}')")

print("\nОбучение завершено!")
