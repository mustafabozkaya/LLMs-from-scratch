# Bölüm 5.3: Rastgelelik Kontrolü - Decoding Strategies

## Genel Bakış

Bu bölümde, LLM'lerin metin üretirken kullandığı **decoding strategies** (kod çözme stratejileri) hakkında bilgi ediniyoruz. Bu stratejiler, modelin çıktısındaki rastgelelik ve çeşitliliği kontrol etmemizi sağlar.

---

## 5.3.1 Temperature Scaling (Sıcaklık Ölçeklendirme)

### Problem

Varsayılan olarak `torch.argmax` kullanarak her zaman en yüksek olasılıklı token'ı seçiyoruz. Bu **Greedy Search** (açgözlü arama) olarak bilinir ve her zaman aynı çıktıyı üretir.

### Çözüm: Probability Distribution'dan Sampling

```python
# En yüksek olasılıklı token'ı seç (Greedy)
next_token_id = torch.argmax(probas, dim=-1)

# Veya olasılık dağılımından random seçim
next_token_id = torch.multinomial(probas, num_samples=1)
```

### Örnek

Küçük bir vocabulary ile açıklama:

```python
vocab = { 
    "closer": 0,
    "every": 1, 
    "effort": 2, 
    "forward": 3,
    "inches": 4,
    "moves": 5, 
    "pizza": 6,
    "toward": 7,
    "you": 8,
} 

# LLM'in ürettiği logits
next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])

# Softmax ile olasılıklara dönüştürme
probas = torch.softmax(next_token_logits, dim=0)
# Sonuç: [0.06, 0.001, 0.0001, 0.57, 0.003, 0.0001, 0.0001, 0.36, 0.004]
```

### torch.multinomial Nedir?

Bu fonksiyon, verilen olasılık dağılımından **random sampling** yapar:

- Her token'ın seçilme olasılığı, o token'ın olasılık değeri kadardır
- `num_samples=1` = 1 token seç

```python
# 1000 kez sampling yapalım
torch.manual_seed(123)
sample = [torch.multinomial(probas, num_samples=1).item() for _ in range(1000)]

# Sonuçlar (yaklaşık):
# forward: 544 kez (%54.4) - gerçek olasılık: %57.2
# toward:  376 kez (%37.6) - gerçek olasılık: %35.8
# closer:   71 kez (%7.1)  - gerçek olasılık: %6.1
```

---

## Temperature Scaling

**Temperature scaling**, softmax çıktısını etkileyerek dağılımı **düzenler**:

```python
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)
```

### Temperature Değerlerinin Etkisi

| Temperature | Etkisi | Örnek |
|-------------|---------|-------|
| **T > 1** | Daha düzgün (uniform) dağılım. Daha fazla çeşitlilik | T=5: Tüm token'lara yakın olasılık |
| **T = 1** | Original softmax | Normal dağılım |
| **T < 1** | Daha keskin (peaky) dağılım. Daha güvenli tahminler | T=0.1: En yüksek olasılıklı token neredeyse her zaman seçilir |
| **T = 0** | Greedy search ile aynı (argmax) | Sadece en yüksek olasılıklı token |

### Görselleştirme

```
Temperature = 0.1 (Çok Keskin):
┌────────────────────────────────────────┐
│ forward ████████████████████████████ 99%│
│ toward  ▏                             1%│
│ diğerleri ▏                           0%│
└────────────────────────────────────────┘

Temperature = 1.0 (Normal):
┌────────────────────────────────────────┐
│ forward ████████████████████ 57%        │
│ toward  ████████████████ 36%            │
│ closer ██ 6%                           │
│ diğerleri ▏                           1%│
└────────────────────────────────────────┘

Temperature = 5.0 (Düzgün):
┌────────────────────────────────────────┐
│ forward ████ 15%                       │
│ toward  ████ 14%                       │
│ closer ███ 11%                         │
│ ...                                  ~10%│
└────────────────────────────────────────┘
```

### Sezgisel Açıklama

- `logits / 0.1` → Farklar büyür → En yüksek olasılık baskınlaşır
- `logits / 5.0` → Farklar küçülür → Her token yaklaşık eşit şanslı

---

## 5.3.2 Top-K Sampling

### Problem

Yüksek temperature kullanınca, düşük olasılıklı token'lar da seçilebilir ve **anlamsız metin** üretebilir.

### Çözüm

Sadece **en iyi K token** arasından seçim yap:

```python
top_k = 3

# En yüksek K logit'i seç
top_logits, top_pos = torch.topk(next_token_logits, top_k)

# Diğerlerini -inf yap (olasılık = 0)
new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float("-inf")), 
    other=next_token_logits
)

# Artık sadece top-K token'lardan biri seçilebilir
topk_probas = torch.softmax(new_logits, dim=0)
```

### Örnek

```
Orijinal logits: [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
                 (closer, every, effort, forward, inches, moves, pizza, toward, you)

Top-K = 3:
┌─────────────────────────────────────────────────┐
│ Position 3: forward  → 6.75 → max               │
│ Position 7: toward  → 6.28                     │
│ Position 0: closer   → 4.51                     │
│ Diğerleri          → -inf → seçilemez         │
└─────────────────────────────────────────────────┘

Yeni olasılık dağılımı:
- forward:  %61.5 (6.75 → 0.577)
- toward:   %38.5 (6.28 → 0.361)
- closer:    %0.0  (-inf → 0)
- Diğerleri: %0.0  (-inf → 0)
```

---

## 5.3.3 Advanced Text Generation Function

Temperature ve Top-K'ı birleştiren gelişmiş üretim fonksiyonu:

```python
def generate(model, idx, max_new_tokens, context_size, 
             temperature=0.0, top_k=None, eos_id=None):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]  # Sadece son token'ı al

        # 1. Top-K filtering
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, 
                torch.tensor(float("-inf")).to(logits.device), 
                logits
            )

        # 2. Temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            # Numerik stabilite için max'ı çıkar
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # EOS token kontrolü
        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

### Kullanım Örnekleri

```python
# GÜVENLİ - Düşük sıcaklık, greedy'e yakın
token_ids = generate(
    model=gpt,
    idx=encoded,
    max_new_tokens=50,
    context_size=1024,
    temperature=0.0,  # Greedy
    top_k=None
)

# ORTALAMA - Biraz çeşitlilik
token_ids = generate(
    model=gpt,
    idx=encoded,
    max_new_tokens=50,
    context_size=1024,
    temperature=0.7,
    top_k=50
)

# YARATICI - Yüksek çeşitlilik
token_ids = generate(
    model=gpt,
    idx=encoded,
    max_new_tokens=50,
    context_size=1024,
    temperature=1.5,
    top_k=100
)
```

---

## Özet Tablo

| Strateji | Açıklama | Avantajlar | Dezavantajlar |
|----------|-----------|------------|----------------|
| **Greedy (T=0)** | Her zaman en yüksek olasılıklı token | Tutarlı, hızlı | Sıkıcı, tekrarlayan |
| **Random Sampling (T=1)** | Olasılıklara göre random | Çeşitlilik | Anlamsız çıktılar |
| **Temperature (T>1)** | Dağılımı düzleştir | Çok çeşitli | Riskli |
| **Temperature (T<1)** | Dağılımı keskinleştir | Güvenli | Az çeşitli |
| **Top-K** | K token ile sınırla | Anlamsız token'ları engeller | K sabit olabilir |
| **Top-P (Nucleus)** | Cumulative olasılıkla sınırla | Dinamik K | Daha karmaşık |

---

## En İyi Uygulamalar

### 💡 Önerilen Başlangıç Değerleri

```python
# Gpt-2 için tipik değerler
temperature = 0.7   # 0.7-1.0 arası iyi çalışır
top_k = 40          # 40-100 arası
```

### 🎯 Kullanım Senaryoları

| Senaryo | Temperature | Top-K | Açıklama |
|---------|-------------|-------|----------|
| Kod üretimi | 0.2-0.4 | 20-40 | Tutarlılık önemli |
| Yaratıcı yazı | 0.8-1.2 | 80-100 | Çeşitlilik önemli |
| Soru-Cevap | 0.3-0.5 | 30-50 | Doğruluk önemli |
| Translation | 0.4-0.7 | 40-60 | Denge |

---

## Kaynaklar

- [LLMs from Scratch GitHub](https://github.com/rasbt/LLMs-from-scratch)
- [Temperature Sampling](https://arxiv.org/abs/1909.05858)
- [Nucleus Sampling](https://arxiv.org/abs/1904.09751)
