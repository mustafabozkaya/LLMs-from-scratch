# GPT-2 Ağırlıklarını Yükleme (Weight Loading) Yöntemleri

Büyük Dil Modellerini (LLM) sıfırdan oluştururken, mimariyi kurduktan sonraki en kritik aşama önceden eğitilmiş model ağırlıklarını (parametleri) kendi kurduğumuz ağa entegre etmektir. Bölüm 5'te OpenAI tarafından yayınlanan orijinal GPT-2 ağırlıklarını PyTorch modelimize yüklemek için iki farklı yöntem gösterilmiştir.

## Yöntem 1: Orijinal TensorFlow Ağırlıklarını Dönüştürerek Yüklemek

OpenAI, GPT-2 modelini başlangıçta **TensorFlow** kullanarak eğitmiş ve dağıtmıştır. Bu nedenle, ham model dosyaları PyTorch ile doğrudan uyumlu değildir. `gpt_download.py` betiği bu engeli aşmak için geliştirilmiş bir köprü aracıdır.

### İşleyiş Adımları:
1. **İndirme (Download):** `download_and_load_gpt2` fonksiyonu, OpenAI sunucularından 7 temel dosyayı indirir:
   - `model.ckpt.data-00000-of-00001`: Gerçek ağırlıkların (Weight ve Bias) tutulduğu en büyük ikili dosya (~500 MB).
   - `model.ckpt.index` ve `model.ckpt.meta`: TensorFlow dizin haritaları ve grafiği.
   - `hparams.json`: Modelin yapısal ölçülerini (D_Model, katman sayısı, kafa sayısı vb.) içeren sözlük dosyası.
   - Diğer BPE Tokenizer dosyaları.
2. **Dönüştürme (Conversion):** `load_gpt2_params_from_tf_ckpt` fonksiyonu çalışır. TensorFlow kontrol noktasındaki (checkpoint) isimleri (örn: "h0") okur, fazladan boşlukları atar (`np.squeeze`) ve ağırlıkları PyTorch'un kavrayabileceği iç içe geçmiş bir **NumPy Sözlüğüne** (Dictionary) formatlar.
3. **Sonuç:** Fonksiyon geriye model mimarisini kurmak için `settings` sözlüğünü ve ağırlıkların olduğu devasa `params` tablosunu döndürür. Bu NumPy tablosu daha sonra bizim yazdığımız özel fonksiyonlarla PyTorch tensörlerine dönüştürülüp `model.load_state_dict()` yapısına uyarlanarak modele takılır.

---

## Yöntem 2: Alternatif PyTorch (.pth) Dosyalarını Doğrudan Yüklemek

TensorFlow dönüştürme süreciyle uğraşmak istemeyenler için Bölüm 5 bonus kodlarında (ve Hugging Face üzerinde) alternatif bir yol sunulmaktadır. Kitabın yazarı TensorFlow dönüşümlerini önceden sizin yerinize yapmış ve salt PyTorch durum sözlüklerini (state dicts) `.pth` uzantılı dosyalar olarak yayınlamıştır (örn: `gpt2-small-124M.pth`).

### İşleyiş Adımları:
1. **İndirme (Download):** Gerekli parametre dosyasını (`gpt2-small-124M.pth`) normal bir HTTP isteği (requests) çalıştırarak (Hugging Face ve B2 Cloud üzerinden) diskinize indirirsiniz.
2. **Doğrudan Yükleme:** Dosya yerel PyTorch formatında (`.pth`) olduğu için hiçbir isim değiştirme veya NumPy dizilerinden tensöre dönüşüm işlemine (TensorFlow'da olduğu gibi) gerek kalmaz. Doğrudan ağırlık sözlüğü (State Dict) olarak okunabilir.

### Kod Örneği:
```python
import torch

# PyTorch .pth dosyasını doğrudan RAM'e okuyun
state_dict = torch.load("gpt2-small-124M.pth")

# Modelinize doğrudan entegre edin
model.load_state_dict(state_dict) 
```

---

## Yöntemlerin Karşılaştırması

| Özellik | Yöntem 1 (TensorFlow ile gpt_download.py) | Yöntem 2 (.pth Doğrudan PyTorch) |
| :--- | :--- | :--- |
| **Kaynak Sunucu** | Orijinal OpenAI Sunucuları | Raschka (Hugging Face / Backblaze Cloud) |
| **Ön Hazırlık / Hız** | Uzun sürer ve TensorFlow bağımlılığı gerektirir (ayrıştırma+dönüşüm karmaşası). | Önceden dönüştürülüp paketlendiği için çok daha hızlı kodlanır  ve kolay yüklenir. |
| **Kullanım Kolaylığı**| Karmaşıktır, TensorFlow model hiyerarşisini PyTorch State_Dict sistemine isim isim, boyut boyut eşleyen kodlar barındırır. | Son derece basittir (Klasik PyTorch yüklemesidir). |
| **Öğreticilik** | Orijinal TF tabanlı eski dil modeli yapılarının dönüşüm mühendisliğinde kaputun altını görmek için eşsiz bir antrenmandır. | Öğreticilikten ziyade salt modeli ayağa kaldırmak ve hızlıca test aşamasına geçmek isteyenler içindir. |

> **Özet:** Öğrenme ve "kaputun altını" görme amacıyla TensorFlow ağırlıklarını dönüştürmek son derece kıymetli bir mühendislik pratiğidir. Ancak sadece modeli ayağa kaldırmak ve bir an önce Infenrence (Çıkarım) veya Finetuning (İnce Ayar) işlemlerine geçmek için alternatif `.pth` yüklemesi (Yöntem 2) kesinlikle ideal ve hatasız bir kod deneyimi sunar.
