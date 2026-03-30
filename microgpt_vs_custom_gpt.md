# Karşılaştırma Raporu: `microgpt` vs. Bizzat Kodladığınız [GPTModel](file:///c:/Users/kurtar/Desktop/Sebastian%20rashgca/LLMs-from-scratch/ch05/01_main-chapter-code/previous_chapters.py#190-213)

Bu rapor, Karpathy'nin 200 satırlık saf Python ile yazdığı `microgpt` projesi ile sizin `LLMs-from-scratch/previous_chapters.py` dosyasında PyTorch kullanarak yazdığınız GPT modelinin (ve eğitim araçlarının) arasındaki mimari ve fonksiyonel farkları listelemektedir.

Özetle: `microgpt` her şeyi **en alt seviyede (scratch)** ve saf matematikle yaparken, sizin kodunuz **PyTorch'un sağladığı optimize edilmiş modern yapıtaşlarını** kullanmaktadır.

## 1. Veri İşleme (Dataset & Tokenizer)

| Özellik | `microgpt` (Karpathy) | Sizin Modeliniz ([GPTDatasetV1](file:///c:/Users/kurtar/Desktop/Sebastian%20rashgca/LLMs-from-scratch/ch05/01_main-chapter-code/previous_chapters.py#20-40) vb.) |
| :--- | :--- | :--- |
| **Tokenizer Türü** | Karakter Bazlı (Character-level). Tüm harfleri `a-z` indexler. | Alt-Kelime Bazlı (BPE - Byte Pair Encoding). OpenAI'ın `tiktoken` kütüphanesini (`gpt2`) kullanır. |
| **Kelime Dağarcığı (Vocab Size)** | 27 (26 harf + 1 adet `<BOS>` token'ı) | 50.257 (Tüm modern İngilizce kelime parçacıkları) |
| **Veri Yükleyici (Dataloader)** | Basit bir Python listesi ve [for](file:///c:/Users/kurtar/Desktop/Sebastian%20rashgca/LLMs-from-scratch/ch05/01_main-chapter-code/previous_chapters.py#203-213) döngüsü. Bir kelime (isim) alıp başına/sonuna `<BOS>` ekler. | PyTorch [Dataset](file:///c:/Users/kurtar/Desktop/Sebastian%20rashgca/LLMs-from-scratch/ch05/01_main-chapter-code/previous_chapters.py#20-40) ve `DataLoader` kullanır. Metni `max_length` (örn: 256) boyutunda *sliding window (kayan pencere)* ile parçalar ve batch'ler halinde modele sunar (`batch_size=4`). |

## 2. Model Mimarisi (Architecture & Katmanlar)

| Özellik | `microgpt` (Karpathy) | Sizin Modeliniz ([GPTModel](file:///c:/Users/kurtar/Desktop/Sebastian%20rashgca/LLMs-from-scratch/ch05/01_main-chapter-code/previous_chapters.py#190-213)) |
| :--- | :--- | :--- |
| **Matris Çarpımları (Linear)** | Python list comprehension ile yazılmış `linear(x, w)` fonksiyonu kullanır. `sum(wi * xi)` yapar. | PyTorch `nn.Linear` kullanır. Arka planda optimize C++/CUDA matris çarpımlarını barındırır. |
| **Normalizasyon (Norm)** | Basitleştirilmiş `rmsnorm(x)` kullanır. Vektörü karekök ortalamasına göre ölçekler, *bias/shift* parametresi yoktur. | Orijinal GPT-2'ye sadık kalarak [LayerNorm](file:///c:/Users/kurtar/Desktop/Sebastian%20rashgca/LLMs-from-scratch/ch05/01_main-chapter-code/previous_chapters.py#119-131) (özel sınıf) kullanır. `scale` ve `shift` adında öğrenilebilir parametreler barındırır. |
| **Aktivasyon Fonksiyonu** | En basit olan `ReLU` kullanır (`max(0, x)`). | Orijinal GPT formuna sadık kalarak matematiksel hesaplarla baştan yazılmış [GELU](file:///c:/Users/kurtar/Desktop/Sebastian%20rashgca/LLMs-from-scratch/ch05/01_main-chapter-code/previous_chapters.py#133-142) sınıfını kullanır. |
| **Positional Embedding** | Öğrenilebilir (Learned) pozisyon tablosu kullanır. `token_emb + pos_emb` toplanır. | Tam olarak aynı mantığı kullanır: `nn.Embedding` ile token ve pozisyon tablosu oluşturulur ve toplanır. |

## 3. Attention (Dikkat Mekanizması)

Buradaki fark en belirgin olandır. Sizin modeliniz modern, optimize ve paralel çalışan bir mimariye sahipken, Karpathy'nin kodu temel mantığı göstermek için zaman adımlarında (sequence) tek tek döner.

| Özellik | `microgpt` (Karpathy) | Sizin Modeliniz ([MultiHeadAttention](file:///c:/Users/kurtar/Desktop/Sebastian%20rashgca/LLMs-from-scratch/ch05/01_main-chapter-code/previous_chapters.py#60-114)) |
| :--- | :--- | :--- |
| **Nasıl Çalışır?** | Tek bir for döngüsü (zaman adımı - t) içinde çalışır. Token token ilerler (Parallel sequence işleme yoktur). | Koca bir metin dizisini (Sequence) aynı anda matris çarpımına (`queries @ keys.transpose()`) sokarak işler. |
| **KV Cache** | Eğitimin (Training) tam ortasında şaşırtıcı şekilde manuel olarak `keys` ve `values` listeleri tutarak *KV Cache* kullanır. Çünkü sekansları paralel değil, tek tek işler. | Eğitim sırasında KV Cache **kullanmaz**, "causal mask" (`torch.triu`) kullanarak matrisin üst yarısını `-Inf` ile doldurup önceki token'ların geleceğe bakmasını tek bir matris operasyonuyla engeller. Modern eğitim standartı budur. |
| **Softmax** | Kendi yazdığı basit `.exp() / total` serisi (`softmax` metodu) çalıştırır. | PyTorch kütüphanesinden optimize edilmiş `torch.softmax` fonksiyonunu kullanır. |

## 4. Eğitim ve Türev Süreci (Autograd)

En büyük mucize buradadır.
*   `microgpt` **Autograd** işlemini kendi yazdığı `Value` sınıfı üzerinden yapar. `Value` adındaki bu sınıf, Python'un toplama, çarpma, üst alma (`__add__`, `__mul__`, vb.) özelliklerini manipüle ederek bir "hesaplama grafiği (computation graph)" çizer. Modele bir şey öğretileceğinde, Karpathy kendi kurguladığı `backward()` fonksiyonunu çağırarak bu zinciri tersine işletir ve zincir kuralı (chain rule) ile `dL/dw` hesaplar.
*   Sizin kodunuzda ise bu iş PyTorch'un `torch.Tensor` nesnelerine devredilmiştir. Arka planda milyarlarca derivasyon anında ve optimize edilmiş bir şekilde (çoğunlukla C++ ve CUDA ile bilgisayarın GPU'sunda) C-kütüphaneleri vasıtasıyla hesaplanır.

## 5. Çıkarım (Inference - Üretim)

| Özellik | `microgpt` (Karpathy) | Sizin Modeliniz ([generate_text_simple](file:///c:/Users/kurtar/Desktop/Sebastian%20rashgca/LLMs-from-scratch/ch05/01_main-chapter-code/previous_chapters.py#215-239)) |
| :--- | :--- | :--- |
| **Sırası (Decoding)** | `softmax([l / temperature for l in logits])` ile tahmin yaratıcılığını (temperature) belirler. Üstelik Python'un `random.choices` modülünü kullanarak bu olasılıklara göre kelime çeker (Sampling / Rastgele Örneklem). | [generate_text_simple](file:///c:/Users/kurtar/Desktop/Sebastian%20rashgca/LLMs-from-scratch/ch05/01_main-chapter-code/previous_chapters.py#215-239) içinde sadece `torch.argmax(logits)` kullanır. Yani hep en yüksek ihtimalli token'ı (Greedy Decoding) seçer. (Siz `temperature` veya rastlantısallık henüz eklemediniz). |

---
**Sonuç:** `microgpt` aslında sizin PyTorch ile `nn.Linear`, `nn.Embedding` veya `.backward()` diyerek bir satırda çağırdığınız işlerin matematikte neye denk geldiğini sıfırdan ve sadece Python listeleriyle yaparak bir "eğitimsel kanıt" sunmaktadır. Sizin yazdığınız kod ise bu temel formülleri alıp büyük veri kümelerinde eğitilebilecek gerçekçi (production-ready) sürümlere hazırlamaktadır.
