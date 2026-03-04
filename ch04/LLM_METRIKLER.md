# LLM Performans Metrikleri - Açıklama Rehberi

Bu belge, büyük dil modellerinin (LLM) performansını ölçmek için kullanılan temel metrikleri açıklar.

---

## 1. ⏱️ Total Time (Toplam Süre)

**Tanım:** Modelin prompt'u (istemi) alıp, cevabı tamamen üretene kadar geçen toplam süre.

```
Sen: "Merhaba, nasılsın?"
Model: "İyiyim, teşekkür ederim..." → 35.73 saniye sonra biter
```

### Özellikleri

- **Dahil:** Token üretme süresi
- **Dahil değil:** Modeli VRAM'e yükleme süresi
- **Birim:** Saniye (s)

### Ne Anlama Geliyor?

| Süre | Değerlendirme |
|------|---------------|
| < 10 saniye | ⚡ Çok hızlı |
| 10-30 saniye | ✅ Normal |
| 30-60 saniye | ⏳ Yavaş |
| > 60 saniye | 🐢 Çok yavaş |

### Nelere Bağlı?

- Model boyutu (parametre sayısı)
- GPU gücü
- Batch size
- Sequence uzunluğu

---

## 2. ⚡ Tokens Per Second (Saniyedeki Token Sayısı)

**Tanım:** Modelin saniyede ürettiği token sayısı.

```
10.19 token/s = Saniyede ~7-8 kelime yazıyor
```

### Token Nedir?

Modeller kelimeleri bütün olarak değil, "token" adı verilen küçük parçalar olarak üretir:

```
"Merhaba" → ["Mer", "haba"]
"Çalışma" → ["Çalış", "ma"]
```

- 1 token ≈ 0.75 kelime (İngilizce)
- 1 token ≈ 0.5 kelime (Türkçe)

### İnsan Karşılaştırması

| Okuma/Yazma Hızı | Değerlendirme |
|------------------|---------------|
| 3-5 token/s | İnsan ortalama okuma hızı |
| 10-15 token/s | ✅ LLM için iyi |
| 20-30 token/s | ⚡ Çok iyi |
| 50+ token/s | 🚀 Mükemmel |

### Nelere Bağlı?

- **GPU VRAM:** Daha fazla = Daha hızlı
- **Model Boyutu:** Büyük model = Daha yavaş
- **Quantization:** INT8 = 2x daha hızlı
- **GPU Modeli:** RTX 4090, A100 gibi üst seviye = Daha hızlı

### Örnek Hesaplama

```
Toplam Süre: 35.73 saniye
Hız: 10.19 token/s

Üretilen Token = 35.73 × 10.19 = ~364 token
Kelime Sayısı = 364 × 0.75 = ~273 kelime
```

---

## 3. 🎯 Time To First Token (TTFT) - İlk Tokene Kadar Geçen Süre

**Tanım:** Prompt gönderildikten sonra, modelin ilk token'ı üretmesine kadar geçen süre.

```
Sen: "Merhaba" 
→ [5ms] → "İ" → "iy" → "iyi" → ...
```

### Neden Önemli?

**Kullanıcı deneyimi (UX) için en kritik metriktir!**

- Kullanıcı anında yanıt görmeye başlar
- 100ms altı = Anlık hissettirir
- 1s üstü = "takılmış" hissi verir

### Nelere Bağlı?

| Faktör | Etkisi |
|--------|--------|
| **Context Length** | Uzun context = Daha uzun TTFT |
| **KV Cache** | Evet = Çok daha hızlı |
| **GPU Memory Bandwidth** | Yüksek bant genişliği = Daha hızlı |

### Değerlendirme Tablosu

| TTFT | Değerlendirme |
|------|---------------|
| < 50ms | 🚀 Anlık |
| 50-200ms | ⚡ Hızlı |
| 200-500ms | ✅ Normal |
| 500ms-1s | ⏳ Yavaş |
| > 1s | 🐢 Çok yavaş |

---

## 4. 🔢 FLOPs (Floating Point Operations)

**Tanım:** Modelin tek bir forward pass için yaptığı toplam hesaplama sayısı.

```
GPT-2 Small (124M): 510.000.000.000 FLOPs
```

### Basitçe Anlatım

FLOPs = "Virgüllü İşlem Sayısı"

Bilgisayarın matematik işlemi yapması gibi düşün:

```
2 + 3 = 5          → 1 işlem
2 × 3 = 6          → 1 işlem
768 × 768 = ...    → 589.824 işlem!
```

Neural network'lar milyarlarca bu tür işlem yaparlar!

### Büyük Sayıları Okuma

| Yazılış | Okunuşu | Değeri |
|---------|---------|--------|
| 5.1e+11 | 510 milyar | 510.000.000.000 |
| 1.4e+12 | 1.4 trilyon | 1.400.000.000.000 |
| 3.2e+12 | 3.2 trilyon | 3.200.000.000.000 |
| 6.4e+12 | 6.4 trilyon | 6.400.000.000.000 |

### Daha Okunaklı Format

```python
def format_flops(flops):
    if flops >= 1e12:
        return f"{flops/1e12:.1f} Trilyon"  # 6.4 Trilyon
    elif flops >= 1e9:
        return f"{flops/1e9:.0f} Milyar"      # 510 Milyar
    return str(flops)
```

### Örnek Değerler (1024 Token, Batch 2)

| Model | Parametre | FLOPs | Okunuş |
|-------|-----------|-------|---------|
| GPT-2 Small | 124M | 5.1e+11 | 510 Milyar |
| GPT-2 Medium | 355M | 1.4e+12 | 1.4 Trilyon |
| GPT-2 Large | 774M | 3.2e+12 | 3.2 Trilyon |
| GPT-2 XL | 1558M | 6.4e+12 | 6.4 Trilyon |

### FLOPs Ne Anlama Geliyor?

Model ne kadar "zor çalışıyor" gösterir:

```
Daha fazla FLOPs = Daha güçlü GPU gerekli
```

### Hesaplama Süresi

Formül:
```
Süre (saniye) = Toplam FLOPs ÷ GPU Gücü (FLOPs/s)
```

Örnek:
```
6.4 Trilyon FLOPs ÷ 30 Trilyon FLOPs/s = 0.21 saniye
```

### GPU Güçleri

| GPU | FP16 Gücü |
|-----|-----------|
| RTX 3060 | 12.7 TFLOPs/s |
| RTX 3070 | 29.8 TFLOPs/s |
| RTX 3080 | 29.8 TFLOPs/s |
| RTX 3090 | 35.6 TFLOPs/s |
| RTX 4090 | 82.6 TFLOPs/s |
| A100 | 78.0 TFLOPs/s |
| H100 | 205.0 TFLOPs/s |

### MACs vs FLOPs

Notebook'larda şöyle geçer:
```python
macs, params = profile(model, inputs=(input_tensor,), verbose=False)
flops = 2 * macs  # MACs × 2 = FLOPs
```

Neden 2 katı?
```
MACs = Multiply-Accumulate Operations
Örnek: y = ax + b
  → çarpma (a×x) + toplama (+b)
  → 2 işlem ama 1 MAC olarak sayılır
```

---

## 5. 📊 MFU (Model FLOPs Utilization)

**Tanım:** GPU'nun teorik maksimum gücünün ne kadarının kullanıldığını gösteren verimlilik oranı.

```
MFU = Gerçek Performans ÷ Teorik Maksimum
```

### Ne Demek?

| MFU | Anlamı |
|-----|---------|
| %100 | Mükemmel (mümkün değil!) |
| %75-85 | Çok iyi |
| %50-75 | İyi |
| %25-50 | Orta |
| < %25 | Kötü |

### Neden %100 Olmaz?

1. **Memory transfer:** Veri GPU'ya taşınırken bekleme
2. **Kernel overhead:** Küçük işlemler için hazırlık süresi
3. **Synchronization:** İşlemler arası bekleme
4. **Not all cores busy:** Her zaman %100 kullanılamaz

### Örnek Değerler

GPU: A100, Batch: 8, Sequence: 1024

| Model | MFU |
|-------|------|
| GPT-2 Small (124M) | %60 |
| GPT-2 Medium (355M) | %61 |
| GPT-2 Large (774M) | %74 |
| GPT-2 XL (1558M) | %81 |

**Görüldüğü gibi:** Büyük modeller daha verimli çalışıyor!

---

## 6. 💾 Memory (VRAM) Gereksinimleri

**Tanım:** Modelin çalışması için gereken GPU belleği.

### Precision Başına Bellek

| Precision | Bayt/Parametre | Örnek |
|-----------|-----------------|-------|
| FP32 | 4 byte | 124M × 4 = ~500MB |
| FP16/BF16 | 2 byte | 124M × 2 = ~250MB |
| INT8 | 1 byte | 124M × 1 = ~125MB |
| INT4 | 0.5 byte | 124M × 0.5 = ~62MB |

### Inference vs Training

| Mod | Gereken VRAM |
|-----|-------------|
| **Inference (FP16)** | Model + KV Cache |
| **Training (FP16)** | Model + Gradients + Optimizer States |

### Örnek: GPT-2 Small (124M)

| Mod | Precision | VRAM |
|-----|-----------|------|
| Inference | FP16 | ~1.5 GB |
| Inference | INT8 | ~0.8 GB |
| Training | FP16 | ~6 GB |
| Training | FP32 | ~12 GB |

---

## 📊 Metrik İlişkileri

```
Prompt Gönderimi
      ↓
[ TTFT: 5ms ] ← İlk token
      ↓
[ 10 token/s ] ← Sürekli üretim
      ↓
[ Toplam: 35s ] ← Tamamlandı
      ↓
[ FLOPs: 510M ] ← Her pass'te yapılan işlem
```

Formül:
```
Toplam Süre = TTFT + (Üretilen Token Sayısı ÷ Token/s)
```

Örnek:
```
35s = 5ms + (364 token ÷ 10.19 token/s)
35s ≈ 0.005s + 35s ✓
```

---

## 💡 Performans İpuçları

### Daha Hızlı İçin

1. **Quantization kullan:**
   - FP16 → 2x hız
   - INT8 → 4x hız

2. **KV Cache etkinleştir:**
   - Tekrarlayan prompt'larda 10x hız

3. **GPU upgrade et:**
   - RTX 3070 → RTX 4090 = 3x hız

4. **Batch size küçült:**
   - Daha az bekleme süresi

### Daha Az VRAM İçin

1. **Quantization:** INT8 = 4x tasarruf
2. **Gradient checkpointing:** ~30% tasarruf
3. **Model parallelism:** VRAM'i böl

---

## 🎯 Hedef Değerler

| Kullanım | Token/s | TTFT | Toplam Süre | MFU |
|----------|---------|------|-------------|-----|
| **İdeal** | 30+ | < 100ms | < 5s | %75+ |
| **İyi** | 15-30 | < 200ms | < 15s | %50+ |
| **Kabul edilebilir** | 10-15 | < 500ms | < 30s | %30+ |
| **Yavaş** | < 10 | > 500ms | > 30s | < %30 |

---

## 🔧 Ölçüm Araçları

### Python ile Ölçüm

```python
import time
import torch
from thop import profile

# Modeli yükle
model = load_model()

# Prompt
prompt = "Merhaba, nasılsın?"

# Ölçüm başla
start = time.time()

# Üret
output = model.generate(prompt)

# Ölçüm bitir
end = time.time()

total_time = end - start
print(f"Toplam Süre: {total_time:.2f}s")
```

### FLOPs Ölçümü

```python
from thop import profile

input_tensor = torch.randint(0, 50257, (1, 1024))

macs, params = profile(model, inputs=(input_tensor,), verbose=False)
flops = 2 * macs  # MACs × 2 = FLOPs

print(f"FLOPs: {flops:.2e}")
```

---

## 📋 Hızlı Referans Tablosu

| Metrik | Açıklama | İyi Değer |
|--------|----------|----------|
| **Total Time** | Toplam üretim süresi | < 30s |
| **Tokens/s** | Yazma hızı | 20+ |
| **TTFT** | İlk token süresi | < 100ms |
| **FLOPs** | Hesaplama miktarı | Model boyutuna bağlı |
| **MFU** | GPU verimliliği | %50+ |
| **VRAM** | GPU belleği | Model + Batch'a bağlı |

---

## ✅ Özet

1. **Toplam Süre** = Ne kadar bekleyeceğin
2. **Token/s** = Modelin yazma hızı
3. **TTFT** = İlk yanıtı görme süresi (en önemli!)
4. **FLOPs** = Modelin ne kadar "zor çalıştığı"
5. **MFU** = GPU verimliliği
6. **VRAM** = Ne kadar bellek gerektiği

---

*Bu rehber, kişisel öğrenme ve araştırma amaçlıdır.*
