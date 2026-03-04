# KV Cache Analizi

## Genel Bakış

Bu dosya, `LLMs-from-scratch` projesinin Chapter 4 bölümündeki KV Cache implementasyonunu detaylı olarak açıklamaktadır.

## KV Cache Nedir?

KV Cache (Key-Value Önbellek), LLM inference sırasında kullanılan bir **performans optimizasyonu** tekniğidir. Temel fikir:

- **Cache yok**: Her yeni token üretiminde tüm bağlam tekrar işlenir
- **KV Cache**: Sadece ilk seferde K ve V hesapla, sonraki adımlarda yeniden kullan

---

## Akış Diyagramı

### Cache Yok (Orijinal)

```
Token Üretimi:
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: "Hello"                                                │
│  Input: ["Hello"]                                               │
│  Model: Forward Pass → Tüm K,V hesapla → ["Hello", "I"]        │
├─────────────────────────────────────────────────────────────────┤
│  Step 2: "Hello I"                                              │
│  Input: ["Hello", "I"]  ← TÜM BAĞLAM TEKRAR                    │
│  Model: Forward Pass → K,V YENİDEN hesapla → ["Hello", "I", "am"]
├─────────────────────────────────────────────────────────────────┤
│  Step 3: "Hello I am"                                           │
│  Input: ["Hello", "I", "am"]  ← TÜM BAĞLAM TEKRAR              │
│  Model: Forward Pass → K,V YENİDEN hesapla → ...               │
└─────────────────────────────────────────────────────────────────┘
⏱️ Her adımda O(n²) karmaşıklık
```

### KV Cache'li (Optimize)

```
┌─────────────────────────────────────────────────────────────────┐
│  BAŞLANGIÇ: Prompt "Hello, I am"                                │
│  Input: ["Hello", ",", "I", "am"]                               │
│  Model: Forward Pass → K,V HESAPLA → CACHE'A KAYDET            │
│                                                                 │
│  cache_k = [K1, K2, K3, K4]                                    │
│  cache_v = [V1, V2, V3, V4]                                    │
│  current_pos = 4                                               │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Yeni token üret                                        │
│  Input: ["am"] (sadece son token!)                             │
│  Query: Q5 hesapla                                              │
│  Keys: cache_k + K5  → [K1,K2,K3,K4,K5]                        │
│  Values: cache_v + V5 → [V1,V2,V3,V4,V5]                       │
│  Attention(Q5, [K1..K5], [V1..V5]) → Yeni token               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Kod Değişiklikleri (Satır Satır)

### 1. MultiHeadAttention Sınıfı

**`__init__` (satır 34-39):**
```python
self.register_buffer("cache_k", None, persistent=False)
self.register_buffer("cache_v", None, persistent=False)
self.ptr_current_pos = 0
```
- K ve V için cache bufferları oluşturuldu
- `ptr_current_pos`: Mevcut pozisyonu takip eder (causal mask için)

**`forward()` (satır 56-65):**
```python
if use_cache:
    if self.cache_k is None:
        self.cache_k, self.cache_v = keys_new, values_new
    else:
        self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
        self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
    keys, values = self.cache_k, self.cache_v
else:
    keys, values = keys_new, values_new
```
- İlk çağrıda cache başlatılır
- Sonraki çağrılarda yeni K,V eklenir (concatenate)

**Mask hesaplaması (satır 77-87):**
```python
if use_cache:
    mask_bool = self.mask.bool()[
        self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
    ]
    self.ptr_current_pos += num_tokens_Q
else:
    mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]
```
- Cache kullanırken pozisyon bazlı mask uygulanır

**`reset_cache()` metodu (satır 106-109):**
```python
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
```
- Cache'i temizler, yeni üretim için hazır hale getirir

---

### 2. TransformerBlock Sınıfı

```python
def forward(self, x, use_cache=False):
    x = self.att(x, use_cache=use_cache)  # use_cache aktarılıyor
```

---

### 3. GPTModel Sınıfı

**ModuleList değişikliği:**
```python
self.trf_blocks = nn.ModuleList(
    [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
```
- `nn.Sequential` yerine `ModuleList` - bloklara index ile erişim için

**`forward()` değişikliği:**
```python
if use_cache:
    pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, ...)
    self.current_pos += seq_len
else:
    pos_ids = torch.arange(0, seq_len, ...)

for blk in self.trf_blocks:
    x = blk(x, use_cache=use_cache)
```
- Position embedding dinamik olarak hesaplanır (cache durumuna göre)

**`reset_kv_cache()` metodu:**
```python
def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0
```
- Tüm katmanların cache'lerini temizler

---

### 4. generate_text_simple_cached() Fonksiyonu

```python
def generate_text_simple_cached(model, idx, max_new_tokens, context_size=None, use_cache=True):
    with torch.no_grad():
        if use_cache:
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)  # İlk prompt
            
            for _ in range(max_new_tokens):
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)
                logits = model(next_idx, use_cache=True)  # Sadece yeni token
```

**Akış:**
1. Önce prompt işlenir, K,V cache'lenir
2. Her iteration'da sadece 1 yeni token modele verilir
3. Model bunu işler, K ve V'ye ekler, yeni token'ı tahmin eder

---

## Performans Karşılaştırması

| Aspect | Cache Yok | KV Cache |
|--------|-----------|----------|
| **Input** | Tüm sekans | Sadece 1 token |
| **K,V Hesapla** | Her adımda | Sadece 1. kez |
| **Maliyet** | O(n²) her adım | O(1) her adım |
| **Bellek** | Düşük | Yüksek (K,V saklanır) |

---

## Cached Forward Pass Detay

```
                    Token Embedding
                         │
                    Position Embedding
                         │
              ┌──────────┴──────────┐
              │  Transformer Block  │
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
         LayerNorm          MultiHeadAttention
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              W_key(x)       W_value(x)      W_query(x)
                    │              │              │
                    ▼              ▼              ▼
              keys_new       values_new       queries
                    │              │
        ┌───────────┴───────────┐  │
        ▼                       ▼  ▼
   cache_k yok mu?          concat        [Sadece bu adım]
        │                       │           hesaplanır
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │  keys = cache_k + K   │
        │  values = cache_v + V │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   Attention(Q,K,V)    │
        │   + Causal Mask       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │      Output logits    │
        └───────────────────────┘
```

---

## Basit Anlatım

**Cache yok:** 
> "Her cümle kurarken, önceki her şeyi tekrar tekrar okuyup anlıyorum"

**Cache var:**
> "İlk kez okuyup anladığımı hafızada tutuyorum, sonra sadece yeni kelimeyi işliyorum"

---

## Tensor Shape Değişimleri (batch, num_token, embed_size)

### Cache Yok (Orijinal)

Her adımda input uzunluğu artar:

```
Step 1: Input: ["Hello"]
        Shape: (batch=1, num_tokens=1, embed_size=768)
        ─────────────────────────────────────────
        
Step 2: Input: ["Hello", "I"]
        Shape: (batch=1, num_tokens=2, embed_size=768)
        ─────────────────────────────────────────

Step 3: Input: ["Hello", "I", "am"]
        Shape: (batch=1, num_tokens=3, embed_size=768)
        ─────────────────────────────────────────
        
Step N: Input: [token_1, token_2, ..., token_N]
        Shape: (batch=1, num_tokens=N, embed_size=768)
        
⏫ Her adımda num_tokens artıyor! (O(n) boyut)
```

### KV Cache'li (Optimize)

Sadece 1 token input olarak verilir:

```
┌─────────────────────────────────────────────────────────────────┐
│  BAŞLANGIÇ: Prompt "Hello, I am" (4 token)                   │
│                                                                 │
│  Input: ["Hello", ",", "I", "am"]                              │
│  Shape: (batch=1, num_tokens=4, embed_size=768)              │
│                                                                 │
│  cache_k.shape: (1, 4, 12, 64)  → (batch, num_tokens, heads, dim)│
│  cache_v.shape: (1, 4, 12, 64)                                 │
│  current_pos = 4                                                │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Yeni token üret                                       │
│                                                                 │
│  Input: ["am"]  ← SADECE 1 TOKEN!                              │
│  Shape: (batch=1, num_tokens=1, embed_size=768)               │
│                                                                 │
│  queries.shape: (1, 1, 12, 64)  ← sadece 1 token için Q      │
│  keys.shape:     (1, 5, 12, 64)  ← cache + yeni token        │
│  values.shape:   (1, 5, 12, 64)  ← cache + yeni token        │
│                                                                 │
│  current_pos = 5                                                │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2:                                                        │
│                                                                 │
│  Input: [token_5]                                               │
│  Shape: (batch=1, num_tokens=1, embed_size=768)               │
│                                                                 │
│  queries.shape: (1, 1, 12, 64)                                 │
│  keys.shape:     (1, 6, 12, 64)  ← cache büyümeye devam       │
│  values.shape:   (1, 6, 12, 64)                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  ÖZET:                                                          │
│                                                                 │
│  ┌──────────────┬──────────────┬────────────────────────────┐   │
│  │   Değişken   │   Cache Yok  │      KV Cache             │   │
│  ├──────────────┼──────────────┼────────────────────────────┤   │
│  │ Input Token  │    Artar     │    Hep 1 (sabit)          │   │
│  │ batch        │    1         │    1                      │   │
│  │ num_tokens   │    N         │    1                      │   │
│  │ embed_size   │    768       │    768                    │   │
│  │ keys/values  │    N         │    N (cache'de birikir)   │   │
│  └──────────────┴──────────────┴────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Detaylı Shape Analizi

```
Attention Matris Hesaplaması:

Cache Yok:
──────────
queries:     (batch, num_heads, Q_tokens, head_dim)
keys:         (batch, num_heads, K_tokens, head_dim)
attention:   (batch, num_heads, Q_tokens, K_tokens)

Örnek (Step 3):
queries:     (1, 12, 3, 64)
keys:        (1, 12, 3, 64)
attention:   (1, 12, 3, 3)   ← 3x3 = 9 hesaplama

KV Cache (Step 3):
───────────────────
queries:     (1, 12, 1, 64)    ← sadece 1 Q!
keys:        (1, 12, 5, 64)     ← 5 K (4 cache + 1 yeni)
attention:   (1, 12, 1, 5)      ← 1x5 = 5 hesaplama!

Maliyet: 9 → 5 (neredeyse yarı!)
```

---

## Position Embedding ve Maskeleme Kontrolü

### Problem

Transformer'larda position embedding önemlidir:
- Token "am" 1. pozisyonda mı 5. pozisyonda mı farklı anlam taşır
- Cache kullanırken pozisyonlar kaybolmasın diye manuel kontrol gerekir

### Çözüm: current_pos Takibi

```
┌─────────────────────────────────────────────────────────────────┐
│                    Position Embedding Kontrolü                 │
└─────────────────────────────────────────────────────────────────┘

GPTModel.forward() içinde:

if use_cache:
    # Dinamik pozisyon IDsı hesapla
    pos_ids = torch.arange(
        self.current_pos,                 # ← Başlangıç: 0, sonra 4, 5, 6...
        self.current_pos + seq_len,        # ← offset ekle
        device=in_idx.device,
        dtype=torch.long
    )
    self.current_pos += seq_len           # ← Pozisyonu güncelle
else:
    # Normal: her zaman 0'dan başla
    pos_ids = torch.arange(0, seq_len, ...)

pos_embeds = self.pos_emb(pos_ids)
```

### Örnek Akış

```
Prompt: "Hello, I am" (4 token)
────────────────────────────────

Step 0 (Prompt işleniyor):
  current_pos = 0
  seq_len = 4
  pos_ids = [0, 1, 2, 3]       ← Normal
  After: current_pos = 4

Step 1 (1 token üretiliyor):
  current_pos = 4
  seq_len = 1
  pos_ids = [4]                ← 4'ten devam ediyor!
  After: current_pos = 5

Step 2:
  current_pos = 5
  seq_ids = [5]                ← 5'ten devam ediyor!
  After: current_pos = 6
```

### Causal Mask (Alt Üçgen) Kontrolü

Attention'da geleceğe bakmayı engellemek için causal mask kullanılır:

```
Maske Matrisi (5 token için):
       K0  K1  K2  K3  K4
    ┌─────────────────────┐
Q0  │ ✓   -   -   -   -   │  ← Q0 sadece K0'a bakabilir
Q1  │ ✓  ✓   -   -   -   │  ← Q1, K0 ve K1'e bakabilir
Q2  │ ✓  ✓  ✓   -   -   │  ← Q2, K0, K1, K2'ye bakabilir
Q3  │ ✓  ✓  ✓  ✓   -   │  ← Q3, K0, K1, K2, K3'e bakabilir
Q4  │ ✓  ✓  ✓  ✓  ✓   │  ← Q4, hepsine bakabilir
    └─────────────────────┘
```

### Cache ile Maske Sorunu

Cache kullanırken pozisyon kaydığı için mask de kaydırılmalı:

```
┌─────────────────────────────────────────────────────────────────┐
│           ptr_current_pos ile Maske Kaydırma                   │
└─────────────────────────────────────────────────────────────────┘

MultiHeadAttention içinde:

# Cache YOKSA:
mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]

Örnek: 3 token
mask[0:3, 0:3] → İlk 3x3 alt üçgen


# Cache VARSA:
mask_bool = self.mask.bool()[
    self.ptr_current_pos : self.ptr_current_pos + num_tokens_Q,  # ← KAYDIR
    :num_tokens_K
]

Örnek: current_pos=4, Q=1, K=5
mask[4:5, 0:5] → 4. satırdaki alt üçgen (5x5'lik matrisin 4. satırı)

┌─────────────────────────────────────┐
│  mask (6x6 için, sadece satır 4):  │
│                                     │
│  [False True True True True True]  │
│       ↑                             │
│       Bu satır kullanılıyor         │
│                                     │
│  Yani: Q@4 sadece K0..K4'e erişir  │
└─────────────────────────────────────┘
```

### Akış Diyagramı: Position + Mask

```
┌─────────────────────────────────────────────────────────────────┐
│                    FORWARD PASS (use_cache=True)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Token Embedding                                            │
│     Input: (batch, 1, embed)  ← sadece 1 token                 │
│            │                                                     │
│     ┌──────┴──────┐                                            │
│     ▼             ▼                                             │
│  2. Position Embedding                                         │
│     current_pos oku → [4]                                      │
│     pos_emb = pos_emb(pos_ids)                                 │
│     current_pos += 1 → 5                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. MultiHeadAttention                                         │
│                                                                 │
│     Q = W_query(x)  → (1, 1, 12, 64)  ← sadece 1 token         │
│     K = W_key(x)    → (1, 1, 12, 64)                           │
│     V = W_value(x)  → (1, 1, 12, 64)                           │
│                                                                 │
│     ┌──────────────────────────────────────┐                   │
│     │ CACHE İŞLEMLERİ:                      │                   │
│     │                                      │                   │
│     │ cache_k = (1, 4, 12, 64)  ← 4 token │                   │
│     │ cache_v = (1, 4, 12, 64)             │                   │
│     │                                      │                   │
│     │ cache_k NEW = (1, 1, 12, 64)         │                   │
│     │ cache_k = cat([cache_k, NEW], dim=1)│                   │
│     │            → (1, 5, 12, 64)          │                   │
│     └──────────────────────────────────────┘                   │
│                              │                                  │
│     ┌──────────────────────────────────────┐                   │
│     │ MASK HESAPLAMA:                      │                   │
│     │                                      │                   │
│     │ ptr_current_pos = 4                 │                   │
│     │ num_tokens_Q = 1                     │                   │
│     │ num_tokens_K = 5                     │                   │
│     │                                      │                   │
│     │ mask[4:5, 0:5]                       │                   │
│     │   → [False True True True True]      │                   │
│     │                                      │                   │
│     │ ptr_current_pos += 1 → 5            │                   │
│     └──────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Attention Hesapla                                          │
│                                                                 │
│     attn_scores = Q @ K.T  → (1, 12, 1, 5)                     │
│     attn_scores.masked_fill_(mask, -inf)                       │
│     attn_weights = softmax(...)                                 │
│     context = attn_weights @ V  → (1, 1, 12, 64)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. Output                                                      │
│     (batch, 1, vocab_size)                                      │
│                                                                 │
│  SONRAKİ ADIM İÇİN HAZIRLIK:                                   │
│  - cache_k ve cache_v bir sonraki adım için hazır              │
│  - current_pos = 5 (bir sonraki token için)                   │
└─────────────────────────────────────────────────────────────────┘
```

### Özet Tablo

| Adım | Input Tokens | current_pos | ptr_current_pos | Cache Size |
|------|--------------|-------------|------------------|-------------|
| Prompt | 4 | 0→4 | 0→4 | 4 |
| +1 token | 1 | 4→5 | 4→5 | 5 |
| +1 token | 1 | 5→6 | 5→6 | 6 |
| +1 token | 1 | 6→7 | 6→7 | 7 |
| ... | ... | ... | ... | ... |

---

## num_tokens_Q ve num_tokens_K: Mask Boyutlandırma

### Kod

```python
num_tokens_Q = queries.shape[-2]  # Query token sayısı
num_tokens_K = keys.shape[-2]     # Key token sayısı
```

### Ne İşe Yarar?

Attention mask'i **doğru boyutlandırmak** için kullanılır. KV Cache kullanırken Q ve K'nın token sayıları farklıdır!

### Query vs Key Kavramı

```
Attention mekanizmasında:

Query (Q): "Ben hangi bilgiye bakmalıyım?"
Key (K):   "Bu pozisyonda ne var?"

Örnek: Cümle okurken
- Q: "Şu an okuduğum kelime ne?"
- K: "Cümledeki her kelime ne anlama geliyor?"
- Attention: Q ile K'lar arasındaki ilişkiyi hesapla
```

### Shape Farklılığı (Cache Etkisi)

```
┌─────────────────────────────────────────────────────────────────┐
│  AŞAMA 1: Prompt "Hello, I am" (4 token) işleniyor           │
│                                                                 │
│  queries.shape: (1, 12, 4, 64)  → num_tokens_Q = 4          │
│  keys.shape:     (1, 12, 4, 64)  → num_tokens_K = 4          │
│  values.shape:   (1, 12, 4, 64)                               │
│                                                                 │
│  Mask: 4x4 = 16 hücre                                          │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  AŞAMA 2: Cache ile yeni token üretimi                        │
│                                                                 │
│  Input: Sadece 1 token (son üretilen)                         │
│                                                                 │
│  queries.shape: (1, 12, 1, 64)  → num_tokens_Q = 1           │
│              ↑                                                  │
│              SADECE 1 token için query!                       │
│                                                                 │
│  keys.shape:     (1, 12, 5, 64)  → num_tokens_K = 5          │
│              ↑                                                  │
│              4 (cache) + 1 (yeni) = 5 key!                   │
│                                                                 │
│  values.shape:   (1, 12, 5, 64)                               │
│                                                                 │
│  Mask: 1x5 = 5 hücre  ← Q sayısı x K sayısı                │
└─────────────────────────────────────────────────────────────────┘
```

### Neden Farklı?

```
┌─────────────────────────────────────────────────────────────────┐
│  SEBEP: Cache sayesinde K'ları yeniden kullanıyoruz            │
└─────────────────────────────────────────────────────────────────┘

Cache YOKSA:
  - Her adımda tüm sekansı işle
  - Q sayısı = K sayısı = toplam token

Cache VARSA:
  - Sadece yeni token için Q hesapla (1 tane)
  - K ve V'leri cache'den al (4, 5, 6... tane)
  - Q sayısı (1) ≠ K sayısı (N)
```

### Mask Matrisi Boyutları

```
Cache YOK (3 token):
┌─────────────────┐
│ Q1-K1, Q1-K2, Q1-K3 │  ← 3x3
│ Q2-K1, Q2-K2, Q2-K3 │
│ Q3-K1, Q3-K2, Q3-K3 │
└─────────────────┘
shape: (3, 3)


Cache VAR (Q=1, K=5):
┌───────────────────────────┐
│ Q1-K1, Q1-K2, Q1-K3, Q1-K4, Q1-K5 │  ← 1x5
└───────────────────────────┘
shape: (1, 5)
```

### Kodda Kullanımı

```python
# Attention scores hesapla
attn_scores = queries @ keys.transpose(2, 3)  # (batch, heads, Q, K)

# Mask'i Q ve K boyutuna göre kes
mask_bool = self.mask.bool()[
    self.ptr_current_pos : self.ptr_current_pos + num_tokens_Q,  # Q boyutu
    :num_tokens_K                                             # K boyutu
]

# Mask uygula
attn_scores.masked_fill_(mask_bool, -torch.inf)
```

### Özet Tablo

| Değişken | Anlamı | Cache'siz | Cache'li |
|----------|--------|-----------|----------|
| `num_tokens_Q` | Query sayısı | N | 1 |
| `num_tokens_K` | Key sayısı | N | N+1 |
| **Mask shape** | | (N, N) | (1, N+1) |

---

## Transformer Block'lar Arasında KV Cache

### Her Block'un Kendi Cache'i Var

GPT modelinde 12 transformer block var. **Her block'un kendi MultiHeadAttention'ı** ve dolayısıyla **kendi KV cache'i** var.

```
┌─────────────────────────────────────────────────────────────────┐
│                     GPT Model (12 Layer)                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Block 0                                                        │
│  ┌─────────────────────────────────────────┐                   │
│  │ MultiHeadAttention                      │                   │
│  │   cache_k_0: [K1, K2, K3, ..., KN]      │  ← Block 0'in     │
│  │   cache_v_0: [V1, V2, V3, ..., VN]      │      cache'i      │
│  └─────────────────────────────────────────┘                   │
│  ┌─────────────────────────────────────────┐                   │
│  │ FeedForward                              │                   │
│  └─────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Block 1                                                        │
│  ┌─────────────────────────────────────────┐                   │
│  │ MultiHeadAttention                      │                   │
│  │   cache_k_1: [K1, K2, K3, ..., KN]      │  ← Block 1'in     │
│  │   cache_v_1: [V1, V2, V3, ..., VN]      │      cache'i      │
│  └─────────────────────────────────────────┘                   │
│  ┌─────────────────────────────────────────┐                   │
│  │ FeedForward                              │                   │
│  └─────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Block 2 ... Block 11                                           │
│  (Her birinin ayrı cache_k ve cache_v'si)                      │
└─────────────────────────────────────────────────────────────────┘
```

### Önemli: Block'lar Arasında Cache Aktarılmaz!

```
┌─────────────────────────────────────────────────────────────────┐
│  YANLIŞ DÜŞÜNCE:                                               │
│                                                                 │
│  Block 0'ın çıktısı → Block 1'e giderken KV cache de gider   │
│                                                                 │
│  ✗ DOĞRU DEĞİL!                                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  DOĞRU:                                                         │
│                                                                 │
│  Block 0: Kendi K0,V0 hesaplar, cache'ler                      │
│  Block 1: Kendi K1,V1 hesaplar, AYRI cache'ler                 │
│  Block 2: Kendi K2,V2 hesaplar, AYRI cache'ler                 │
│                                                                 │
│  Her block BAĞIMSIZ cache tutar!                               │
└─────────────────────────────────────────────────────────────────┘
```

### Forward Pass Sırasında Cache Akışı

```
1 token üretiliyor (use_cache=True):

Input: [yeni_token]

Layer 0:
  Input: [yeni_token] → Q0 hesapla
  + cache_k_0 + cache_v_0 kullan
  Output: [hidden_0] → Layer 1'e git

Layer 1:
  Input: [hidden_0] → Q1 hesapla  
  + cache_k_1 + cache_v_1 kullan (BAĞIMSIZ!)
  Output: [hidden_1] → Layer 2'ye git
  ...
```

### Block Başına Cache Durumu

```
AŞAMA: "Hello, I am" (4 token) Prompt işleniyor
──────────────────────────────────────────────────

Her block için cache büyüklükleri:

Block 0: cache_k.shape = (1, 4, 12, 64)  ← 4 token
Block 1: cache_k.shape = (1, 4, 12, 64)  ← 4 token
Block 2: cache_k.shape = (1, 4, 12, 64)  ← 4 token
...
Block 11: cache_k.shape = (1, 4, 12, 64) ← 4 token

Toplam KV cache bellek: 12 layer × 2 (K,V) × 4 token × 12 head × 64 dim


AŞAMA: +1 token üretildi
────────────────────────────────

Block 0: cache_k.shape = (1, 5, 12, 64)  ← 5 token
Block 1: cache_k.shape = (1, 5, 12, 64)  ← 5 token
...
Block 11: cache_k.shape = (1, 5, 12, 64) ← 5 token


AŞAMA: +10 token üretildi (toplam 14 token)
────────────────────────────────────────────

Block 0: cache_k.shape = (1, 14, 12, 64) ← 14 token
Block 1: cache_k.shape = (1, 14, 12, 64) ← 14 token
...
Block 11: cache_k.shape = (1, 14, 12, 64) ← 14 token
```

### reset_kv_cache() Nasıl Çalışır?

```python
def reset_kv_cache(self):
    # Tüm transformer block'ların...
    for blk in self.trf_blocks:
        # ...attention module'larının cache'lerini temizle
        blk.att.reset_cache()
    # Position counter'ı sıfırla
    self.current_pos = 0
```

Her block'un cache'i **ayrı ayrı sıfırlanır**:

```
reset_kv_cache() çağrılınca:

Block 0: cache_k = None, cache_v = None, ptr = 0
Block 1: cache_k = None, cache_v = None, ptr = 0
Block 2: cache_k = None, cache_v = None, ptr = 0
...
Block 11: cache_k = None, cache_v = None, ptr = 0
```

### Özet

| Özellik | Açıklama |
|---------|----------|
| **Her block'un cache'i** | Evet, bağımsız |
| **Block'lar arası aktarım** | Hayır, yok |
| **Sıfırlama** | `reset_kv_cache()` ile hepsi birden |
| **Bellek kullanımı** | Layer sayısı × 2 × Token sayısı × Head × Dim |

---

## current_pos: Position Embedding İzleme

### Kod

```python
# GPTModel.__init__
self.current_pos = 0

# GPTModel.forward()
if use_cache:
    pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len)
    self.current_pos += seq_len
else:
    pos_ids = torch.arange(0, seq_len)
```

### Problem

```
Normal mod (cache yok):
┌────────────────────────────────────────┐
│  Input: ["Hello", "I"]                │
│  pos_ids = [0, 1]  ← Her zaman 0'dan │
└────────────────────────────────────────┘

KV Cache mod:
┌────────────────────────────────────────┐
│  Step 1: Input: ["Hello", ",", "I", "am"]
│          pos_ids = [0, 1, 2, 3]       │
│          current_pos = 0 → 4           │
├────────────────────────────────────────┤
│  Step 2: Input: ["am"]  (sadece 1 token!)
│          pos_ids = ?                   │
│          ↑                              │
│          Kaçıncı pozisyonda? 4 mü 0 mı?│
└────────────────────────────────────────┘
```

### Çözüm: current_pos

```python
if use_cache:
    # Dinamik pozisyon: 4'ten devam et
    pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len)
    self.current_pos += seq_len  # Güncelle
else:
    # Normal: 0'dan başla
    pos_ids = torch.arange(0, seq_len)
```

### Akış Örneği

```
Prompt: "Hello, I am" (4 token)
────────────────────────────────

current_pos = 0

┌─────────────────────────────────────────────────────────┐
│ Step 0: Prompt işleniyor                                │
│                                                         │
│ Input: ["Hello", ",", "I", "am"] → 4 token            │
│ seq_len = 4                                            │
│                                                         │
│ pos_ids = arange(0, 0+4) = [0, 1, 2, 3]              │
│ current_pos = 0 + 4 = 4                                │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ Step 1: +1 token üret                                   │
│                                                         │
│ Input: [yeni_token] → 1 token                         │
│ seq_len = 1                                            │
│                                                         │
│ pos_ids = arange(4, 4+1) = [4]  ← 4'ten devam!       │
│ current_pos = 4 + 1 = 5                                │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: +1 token üret                                   │
│                                                         │
│ Input: [yeni_token2] → 1 token                         │
│ seq_len = 1                                            │
│                                                         │
│ pos_ids = arange(5, 5+1) = [5]  ← 5'ten devam!       │
│ current_pos = 5 + 1 = 6                                │
└─────────────────────────────────────────────────────────┘
```

### Neden Önemli?

Position embedding olmazsa:

| Pozisyon | Embedding | Anlam |
|----------|-----------|-------|
| 0 | "am" | 1. kelime |
| 5 | "am" | 6. kelime |

Aynı token ama **farklı anlam**! Model "am" kelimesinin cümlede nerede olduğunu bilmeli.

### reset_kv_cache() ile Sıfırlama

```python
def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0  # ← Position counter da sıfırlanır
```

Yeni metin üretimi için `current_pos` 0'a döner.

---

## Neden Her Block'un Ayrı Cache'i Var?

### Temel Sebep: Her Block Farklı K ve V Hesaplar

```
┌─────────────────────────────────────────────────────────────────┐
│  Transformer Block'ların Yapısı                                  │
└─────────────────────────────────────────────────────────────────┘

Input (embeddings)
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Block 0:                                                       │
│    LayerNorm → MultiHeadAttention(K,V hesapla!) → Output_0    │
│                   ↑                                             │
│                   W_key_0, W_value_0 (AĞIRLIKLAR)              │
└─────────────────────────────────────────────────────────────────┘
       │ Output_0 (yeni temsil)
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Block 1:                                                       │
│    LayerNorm → MultiHeadAttention(K,V hesapla!) → Output_1    │
│                   ↑                                             │
│                   W_key_1, W_value_1 (FARKLI AĞIRLIKLAR!)      │
└─────────────────────────────────────────────────────────────────┘
       │ Output_1
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Block 2:                                                       │
│    LayerNorm → MultiHeadAttention(K,V hesapla!) → Output_2    │
│                   ↑                                             │
│                   W_key_2, W_value_2 (FARKLI AĞIRLIKLAR!)      │
└─────────────────────────────────────────────────────────────────┘
```

### Önemli: Her Block'un Kendi Ağırlıkları Var!

```python
# Block 0'ın ağırlıkları
self.att.W_key = nn.Linear(768, 768)  # Block 0
self.att.W_value = nn.Linear(768, 768)  # Block 0

# Block 1'in ağırlıkları (TAMAMEN FARKLI!)
self.att.W_key = nn.Linear(768, 768)  # Block 1 (ayrı weight!)
self.att.W_value = nn.Linear(768, 768)  # Block 1 (ayrı weight!)
```

### Bu Ne Anlama Geliyor?

```
Block 0'ın K0'ı:
  K0 = W_key_0(input)  → Block 0'ın yorumu

Block 1'in K1'i:
  K1 = W_key_1(output_0)  → Block 1'in yorumu (farklı!)

Bu ikisi FARKLI şeyleri temsil ediyor!
```

### Örnek: "Cat" Kelimesi

```
Giriş: "The cat sat"

Block 0 (İlk katman):
  "cat" → K0 = [0.1, 0.5, 0.3, ...]  ← Yüzeysel özellikler (kedi hayvanı)
  
Block 1 (İkinci katman):
  "cat" → K1 = [0.8, 0.2, 0.9, ...]  ← Daha soyut (hayvan + nesne + canlı)
  
Block 2 (Üçüncü katman):
  "cat" → K2 = [0.4, 0.7, 0.1, ...]  ← Bağlamsal (cümle içindeki anlam)

Aynı kelime ama HER BLOCK'TA FARKLI temsil!
```

### Eğer Cache'leri Paylaşsaydık Ne Olurdu?

```
┌─────────────────────────────────────────────────────────────────┐
│  YANLIŞ: Tüm block'lar tek cache kullansa                       │
└─────────────────────────────────────────────────────────────────┘

Block 0: Input → K0 hesapla → Cache'e koy
Block 1: Input → K1 hesapla → HATA! K0'ı K1 yerine kullanıyor!

┌─────────────────────────────────────────────────────────────────┐
│  Sorun:                                                         │
│                                                                 │
│  - Block 1, Block 0'ın K0'ını kullanıyor                       │
│  - Ama Block 1'in W_key_1'i yok sayılıyor                      │
│  - Doğru K1 hesaplanamıyor!                                    │
│                                                                 │
│  Sonuç: Model BOZUK çıktı üretir!                              │
└─────────────────────────────────────────────────────────────────┘
```

### Doğru: Her Block'un Ayrı Cache'i

```
┌─────────────────────────────────────────────────────────────────┐
│  DOĞRU: Her block kendi cache'ini tutar                        │
└─────────────────────────────────────────────────────────────────┘

Block 0: Input → K0 hesapla → cache_0'a koy
Block 1: Input_0 → K1 hesapla → cache_1'e koy  ← Farklı hesaplama!
Block 2: Input_1 → K2 hesapla → cache_2'a koy  ← Farklı hesaplama!

Her block:
  - Kendi ağırlıklarını (W_key, W_value) kullanır
  - Kendi cache'ini tutar
  - Bağımsız çalışır
```

### Görsel Özet

```
                    [Prompt: 4 token]
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
      Block 0         Block 1         Block 2
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │W_key_0  │      │W_key_1  │      │W_key_2  │
    │cache_k_0│      │cache_k_1│      │cache_k_2│  ← HEPSİ FARKLI!
    └─────────┘      └─────────┘      └─────────┘
```

### Özet

| Soru | Cevap |
|------|-------|
| Neden ayrı cache? | Her block'un farklı W_key, W_value ağırlıkları var |
| Aynı olsaydı ne olurdu? | Yanlış K/V kullanılır, model bozulur |
| Her block ne saklıyor? | O block'un katmanındaki K ve V değerleri |

---

## Her Block Farklı Özellikleri Mi Öğreniyor?

### Evet! Transformer'lar Hiyerarşik Öğrenme Yapar

```
┌─────────────────────────────────────────────────────────────────┐
│  Block'lar Kelimelerin Farklı Yönlerini Öğrenir                │
└─────────────────────────────────────────────────────────────────┘

Input: "The cat sat on the mat"

┌─────────────────────────────────────────────────────────────────┐
│  Block 0 (İlk katman):                                        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  Kelimelerin TEMEL özellikleri:                               │ │
│  "cat" → Kedi, hayvan, 4 ayak, miyav...                     │ │
│  "sat" → Oturmak, eylem, geçmiş...                          │ │
│  "mat" → Halı, düz, yere serili...                          │ │
│  → Yüzeysel, somut bilgiler                                  │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Block 1 (Orta katman):                                       │
│  ┌───────────────────────────────────────────────────────────┐ │
│  Kelimelerin İLİŞKİSEL özellikleri:                          │ │
│  "cat" → "sat" ile ilişkili (kedi oturuyor)                 │ │
│  "sat" → "mat" ile ilişkili (halı üzerinde)                  │ │
│  "the" → Her şeyi niteliyor                                  │ │
│  → Bağlamsal, cümle içindeki ilişkiler                      │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Block 2 ... Block 11 (Son katmanlar):                       │
│  ┌───────────────────────────────────────────────────────────┐ │
│  Kelimelerin SOYUT anlamları:                                │ │
│  "cat sat on mat" → "Kedi halının üzerinde oturuyor"        │ │
│  → Cümlenin tam anlamı, gramer yapısı, ifade...             │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Her Block Neden Farklı?

```
┌─────────────────────────────────────────────────────────────────┐
│  Matematiksel Açıklama:                                        │
└─────────────────────────────────────────────────────────────────┘

Block 0:
  K0 = W_key_0(input)          → İlk dönüşüm
  Bu K0, Block 0'in ağırlıklarıyla üretildi

Block 1:
  input_1 = Block_0_çıktısı
  K1 = W_key_1(input_1)        → İkinci dönüşüm
  Bu K1, Block 1'in ağırlıklarıyla üretildi (FARKLI!)

Block 2:
  input_2 = Block_1_çıktısı
  K2 = W_key_2(input_2)        → Üçüncü dönüşüm
  ...

Her block'un W_key, W_value AĞIRLIKLARI FARKLI olduğu için,
her block FARKLI türde bilgi çıkarır!
```

### Görselleştirme

```
Block 0:  "cat" → [Kedi, Hayvan, Tüylü, 4 Ayak]
                ↓
Block 1:  "cat" → [Evcil Hayvan, Ev Kedisi, Owner: Human]
                ↓
Block 2:  "cat" → [Konu, Özne, Tekil]
                ↓
...
Block 11: "cat" → [Hayvan varlığı, Canlı, Nesne]

Aynı kelime ama her katmanda FARKLI temsil!
```

### Neden Önemli?

```
┌─────────────────────────────────────────────────────────────────┐
│  Her katman farklı bilgiye ihtiyaç duyar:                      │
└─────────────────────────────────────────────────────────────────┘

Katman 0: "Bu kelime ne?" (Ne olduğu)
Katman 1: "Bu kelime neyle ilgili?" (İlişki)
Katman 2: "Bu kelime cümlede ne yapıyor?" (Rol)
...
Katman N: "Cümlenin anlamı ne?" (Genel anlam)

Eğer Block 0'ın cache'ini Block 1'de kullansaydık:
- Block 1, Block 0'ın yüzeysel bilgisini kullanırdı
- Block 1'in öğrenmesi gereken ilişkisel bilgi kaybolurdu
- Model doğru çıktı üretemezdi!
```

### Özet

| Block | Öğrendiği | Örnek |
|-------|-----------|-------|
| Block 0 | Yüzeysel | "cat" = kedi, hayvan |
| Block 1 | İlişkisel | "cat" + "sat" = kedi oturuyor |
| Block 2+ | Soyut | Cümle anlamı, dilbilgisi |

Her block **farklı şeyler öğrendiği** için cache'leri de ayrı olmak zorunda!

---

## Kaynaklar

- [Karpathy's LLMs from Scratch](https://github.com/karpathy/llm.c)
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
