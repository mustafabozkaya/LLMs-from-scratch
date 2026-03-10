# Bölüm 5.4: PyTorch'ta Model Ağırlıklarını Kaydetme ve Yükleme

## Genel Bakış

LLM'leri eğitmek hesaplama açısından oldukça pahalıdır. Bu nedenle, model ağırlıklarını kaydedebilmek ve gerektiğinde yükleyebilmek kritik öneme sahiptir.

![Model Kaydetme/Yükleme](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/16.webp)

---

## 1. Sadece Model Ağırlıklarını Kaydetme

### Neden state_dict?

PyTorch'ta en yaygın ve önerilen yöntem, modelin `state_dict` metodunu kullanmaktır. Bu, modelin tüm öğrenilebilir parametrelerini bir Python dictionary'si olarak döndürür.

```python
torch.save(model.state_dict(), "model.pth")
```

### state_dict Nedir?

`state_dict`, modelin tüm parametrelerini (ağırlıklar ve bias'lar) içeren bir dictionary'dir:

| Anahtar | Açıklama | Örnek Shape |
|---------|----------|-------------|
| `tok_emb.weight` | Token embedding tablosu | (50257, 768) |
| `pos_emb.weight` | Position embedding tablosu | (256, 768) |
| `trf_blocks.X.att.W_query.weight` | Query ağırlıkları | (768, 768) |
| `trf_blocks.X.att.W_key.weight` | Key ağırlıkları | (768, 768) |
| `trf_blocks.X.att.W_value.weight` | Value ağırlıkları | (768, 768) |
| `trf_blocks.X.att.out_proj.weight` | Attention output | (768, 768) |
| `trf_blocks.X.ff.layers.0.weight` | FFN ilk katman | (3072, 768) |
| `trf_blocks.X.ff.layers.2.weight` | FFN çıkış katmanı | (768, 3072) |
| `trf_blocks.X.norm1.scale` | LayerNorm gamma | (768,) |
| `final_norm.scale` | Final LayerNorm | (768,) |
| `out_head.weight` | Çıkış head | (50257, 768) |

---

## 2. Model Ağırlıklarını Yükleme

```python
import torch
from previous_chapters import GPTModel

# Modeli oluştur
model = GPTModel(GPT_CONFIG_124M)

# Cihazı belirle
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Ağırlıkları yükle
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
model.eval()  # Inference için dropout'u kapat
```

### Önemli Noktalar:

1. **`map_location`**: Ağırlıkları farklı bir cihaza taşımak için kullanılır
2. **`weights_only=True`**: Güvenlik açısından önerilir
3. **`model.eval()`**: Inference sırasında dropout vb. katmanları devre dışı bırakır

---

## 3. Model + Optimizer Birlikte Kaydetme

LLM'leri eğitirken genellikle **Adam** veya **AdamW** gibi adaptif optimizer'lar kullanırız. Bu optimizer'lar her ağırlık için ek parametreler saklar:

- **exp_avg**: Birinci moment tahmini (momentum)
- **exp_avg_sq**: İkinci moment tahmini (variance)

Eğitime devam etmek istediğimizde bu optimizer durumlarını da kaydetmeliyiz:

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "model_and_optimizer.pth")
```

### AdamW Optimizer Durumu

```python
optimizer_state = optimizer.state_dict()

print(f"Learning Rate: {optimizer_state['param_groups'][0]['lr']}")
print(f"Weight Decay: {optimizer_state['param_groups'][0]['weight_decay']}")
print(f"State anahtar sayısı: {len(optimizer_state['state'])}")
```

**Optimizer State İçeriği:**

| Değişken | Açıklama |
|----------|----------|
| `exp_avg` | Gradyanların 1. momenti (momentum) - yön bilgisi |
| `exp_avg_sq` | Gradyanların 2. momenti (variance) - öğrenme hızı adaptasyonu |
| `max_squared_grad` | (Opsiyonel) Gradient clipping için |

---

## 4. Model + Optimizer Birlikte Yükleme

```python
# Checkpoint'i yükle
checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)

# Modeli oluştur
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])

# Optimizer'ı oluştur ve durumunu yükle
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# Eğitime hazır!
model.train()
```

---

## 5. Kaydedilen Dosyaların Boyutları

124M parametreli GPT-2 modeli için tahmini boyutlar:

| Dosya | Açıklama | Tahmini Boyut |
|-------|----------|---------------|
| `model.pth` | Sadece model ağırlıkları | ~500 MB |
| `model_and_optimizer.pth` | Model + AdamW state | ~1 GB |

**Detaylı Boyut Hesabı:**

```python
# Model ağırlıkları (float32 = 4 byte)
total_model_params = sum(p.numel() * 4 for p in model.state_dict().values())

# Optimizer durumu (exp_avg + exp_avg_sq)
total_optimizer_params = sum(
    s['exp_avg'].numel() * 4 + s['exp_avg_sq'].numel() * 4 
    for s in optimizer.state_dict().state.values()
)

print(f"Model: ~{total_model_params / (1024**2):.2f} MB")
print(f"Optimizer: ~{total_optimizer_params / (1024**2):.2f} MB")
```

---

## 6. Görselleştirme: Model Ağırlık Dağılımları

Model ağırlıklarının istatistiksel analizi:

```python
import matplotlib.pyplot as plt
import numpy as np

# Tüm parametreleri topla
all_weights = []
for param in model.parameters():
    all_weights.extend(param.cpu().detach().numpy().flatten())

plt.hist(all_weights, bins=100, alpha=0.7, edgecolor='black')
plt.xlabel('Ağırlık Değeri')
plt.ylabel('Frekans')
plt.title('Model Ağırlık Dağılımı')
plt.axvline(x=0, color='r', linestyle='--')
plt.show()
```

---

## 7. Görselleştirme: AdamW Moment Tahminleri

```python
# İlk parametre için optimizer durumunu al
first_key = list(optimizer.state_dict()['state'].keys())[0]
state = optimizer.state_dict()['state'][first_key]

exp_avg = state['exp_avg'].cpu().numpy()
exp_avg_sq = state['exp_avg_sq'].cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# exp_avg (momentum)
axes[0].hist(exp_avg.flatten(), bins=50, color='green', alpha=0.7)
axes[0].set_title('exp_avg (1. Moment - Momentum)')
axes[0].set_xlabel('Değer')

# exp_avg_sq (variance)
axes[1].hist(exp_avg_sq.flatten(), bins=50, color='orange', alpha=0.7)
axes[1].set_title('exp_avg_sq (2. Moment - Variance)')
axes[1].set_xlabel('Değer')

plt.tight_layout()
plt.show()
```

---

## 8. En İyi Uygulamalar

### ✅ Yapılması Gerekenler

1. **Checkpoint'leri düzenli kaydedin**: Her N epoch'ta bir kaydedin
2. **Optimizer state'ini de kaydedin**: Eğitime devam etmek için gerekli
3. **Farklı dosya adları kullanın**: `model_epoch_1.pth`, `model_epoch_2.pth` gibi
4. **Metadata ekleyin**: Epoch sayısı, loss değeri, config bilgileri

### ❌ Yapılmaması Gerekenler

1. **Tüm model nesnesini kaydetmeyin** (`torch.save(model, ...)`)
   - Çünkü model sınıfı değişirse yükleme başarısız olur
2. **Ağırlıkları güvensiz kaynaklardan yüklemeyin**
3. **`weights_only=True` kullanmadan yüklemeyin** (güvenlik)

---

## Özet

| Yöntem | Kullanım | Boyut |
|--------|----------|-------|
| `model.state_dict()` | Sadece ağırlıklar | ~500 MB |
| `model.state_dict() + optimizer.state_dict()` | Ağırlıklar + Optimizer | ~1 GB |
| `model.state_dict() + optimizer.state_dict() + epoch/loss` | Full checkpoint | ~1 GB |

**Avantajlar:**
- Eğitime devam edebilme
- Farklı ortamlarda paylaşım
- Model versiyonlama

---

## Kaynaklar

- [PyTorch Serialization Docs](https://pytorch.org/docs/stable/notes/serialization.html)
- [LLMs from Scratch GitHub](https://github.com/rasbt/LLMs-from-scratch)
