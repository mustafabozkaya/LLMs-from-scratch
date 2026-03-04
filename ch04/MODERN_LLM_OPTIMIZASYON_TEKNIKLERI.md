# Modern LLM Optimizasyon Teknikleri: Karşılaştırmalı Analiz

Bu doküman, büyük dil modellerinin (LLM) bellek tüketimini azaltmak, inference (çıkarım) hızını artırmak ve hesaplama maliyetlerini düşürmek için kullanılan **5 temel modern optimizasyon tekniğinin** (ve temel KV-Cache mekanizmasının) detaylı analizini, avantaj/dezavantajlarını ve benchmark karşılaştırmalarını içermektedir.

---

## 1. KV Cache (Key-Value Cache)

*Tüm modern hızlandırma tekniklerinin temelidir.*

- **Çözdüğü Problem:** Model yeni bir kelime (token) üretirken, geçmişteki tüm kelimelerin Key (K) ve Value (V) tensörlerini her seferinde baştan hesaplamak zorundadır. Bu durum hesaplama (compute) açısından $O(N^2)$ zorluk yaratır ve süreci inanılmaz yavaşlatır.
- **Nasıl Çalışır:** Üretilen her yeni token'ın Key ve Value değerleri GPU belleğine (VRAM) kaydedilir. Bir sonraki adımda eski kelimeler baştan hesaplanmaz, sadece yeni kelimenin K ve V'si hesaplanıp mevcut önbelleğe (cache) eklenir.
- **Avantajları:** Çıkarım (inference) hızında muazzam bir artış sağlar (Örn: 5x hızlanma). İşlemci/GPU hesaplama yükünü $O(N^2)$'den $O(N)$'e düşürür.
- **Dezavantajları:** Metin uzadıkça VRAM kullanımı doğrusal olarak artar. Çok uzun metinlerde "Out of Memory" (OOM) hatasına sebep olur.
- **Benchmark:** Küçük bir modelde (124M param) 200 token üretimi:
  - Cache YOK: **27 token/saniye**
  - Cache VAR: **166 token/saniye** (~6x daha hızlı)

---

## 2. GQA (Grouped-Query Attention)

*(Llama 2, Llama 3, Qwen modellerinde kullanılır)*

- **Çözdüğü Problem:** Standart Multi-Head Attention (MHA) mekanizmasında, her Query (Q) başlığının kendine ait bir Key (K) ve Value (V) başlığı vardır. KV-Cache devreye girdiğinde, bu kadar çok K ve V başlığını VRAM'de tutmak hafızayı çok hızlı doldurur.
- **Nasıl Çalışır:** Query başlıkları gruplara ayrılır (Örn: 8 Query başlığı 4 gruba bölünür). Her gruptaki Query'ler (örn. 2 adet Q), **sadece 1 adet ortak Key ve Value** matrisini paylaşır.
- **Avantajları:** Modelin kalitesini (modelling performance) Multi-Query Attention'a (MQA) kıyasla çok daha iyi korurken, VRAM (KV Cache) tüketimini büyük oranda düşürür.
- **Dezavantajları:** Q başlıkları K ve V'yi paylaştığı için modelin her detaya ayrı ayrı odaklanma kapasitesinden çok ince bir miktar da olsa ödün verilir.
- **Benchmark:** 32 Katmanlı, 32 Head'li, 32K Token Context:
  - Standart MHA KV Cache: **17.18 GB**
  - GQA (4 Grup) KV Cache: **4.29 GB** *(%75 Tasarruf)*

---

## 3. MLA (Multi-Head Latent Attention)

*(DeepSeek V2, DeepSeek V3 ve R1 modellerinde kullanılır)*

- **Çözdüğü Problem:** GQA'nın başlıkları gruplayarak yarattığı o ufak kalite/odak kaybını engellemek ve VRAM kullanımını en az GQA kadar düşürmek hedeflenir.
- **Nasıl Çalışır:** K ve V başlıklarının sayısı azaltılmaz. Bunun yerine K ve V verisi (matrisler), KV Cache'e yazılmadan önce **çok daha küçük bir boyuta (Latent Vektöre)** sıkıştırılır. Model attention işlemi yapacağı zaman, bu küçük vektörü okuyarak tekrar eski devasa boyutuna genişletir (Up-projection).
- **Avantajları:** Başlıklar arası gruplama/paylaşım olmadığı için GQA'dan daha yüksek metin anlama/üretme (modelling) kalitesi sunar. VRAM tasarrufu GQA ile aynı veya daha iyidir.
- **Dezavantajları:** Sıkıştırma (Down-proj) ve Geri Açma (Up-proj) işlemleri fazladan matris çarpımı gerektirdiği için GPU işlem gücü (Compute) açısından minik bir ek maliyeti vardır.
- **Benchmark:** 48 Katmanlı, 24 Head'li, 8K Token Context:
  - Standart MHA KV Cache: **3.25 GB**
  - MLA (1/8 Sıkıştırma): **0.81 GB** *(%75 Tasarruf, Kalite Kaybı Yok)*

---

## 4. SWA (Sliding Window Attention)

*(Gemma 2, Gemma 3 modellerinde kullanılır)*

- **Çözdüğü Problem:** MHA'da model o anki kelimeyi üretirken geçmişteki 100.000 kelimenin tamamıyla attention (dikkat) hesabı yapar. Bu da bağlam ne kadar uzarsa, hesaplama ve VRAM maliyetini o kadar artırır.
- **Nasıl Çalışır:** Global attention yerine Lokal attention kullanılır. Model sadece kendisine en yakın, belirli sayıdaki (örneğin son 1024) kelimeye odaklanır. "Pencere" her kelimede bir adım kayar. Uzak geçmişin unutulmaması için bazı modellerde (örn. Gemma 3) **Hibrit** kullanılır: 5 katman lokal SWA yaparken, 1 katman Global (tüm geçmişe) bakar.
- **Avantajları:** KV Cache hafıza ihtiyacı metin uzadıkça **artmaz**, pencere boyutu (W) kadar sabit kalır. İnanılmaz bir VRAM ve hesaplama hızı tasarrufu sağlar.
- **Dezavantajları:** Sadece yerel bağlama odaklandığı için, hibrit şekilde (aralara Global katmanlar eklenerek) kullanılmazsa uzun metinlerde model geçmiş bilgileri tamamen unutur.
- **Benchmark:** 32 Katmanlı, 32K Token Context:
  - Standart MHA KV Cache: **17.18 GB**
  - SWA (5:1 Hibrit, 1024 Pencere): **3.14 GB** *(Büyük oranda lokalizasyon tasarrufu)*

---

## 5. MoE (Mixture of Experts)

*(Mixtral 8x7B, DeepSeek V3, Qwen MoE modellerinde kullanılır)*

- **Çözdüğü Problem:** Çok akıllı bir model yapmak için milyarlarca parametre eklemek gerekir. Ancak yüz milyarlarca parametreli dev bir model (Dense) tek parça olursa, her kelimede GPU bu dev kütlenin tamamını hesaplamak zorunda kalır ve hız çok düşer.
- **Nasıl Çalışır:** Dev Feed-Forward (İleri Besleme) katmanı, alt parçalara (Örn: 256 Uzman/Expert) bölünür. Modelin başına bir "Yönlendirici (Router)" konur. Her gelen kelime (token) için sadece gerekli uzmanlar (Örn: O anki kelime için sadece 2 uzman) çalıştırılır, diğerleri inaktif kalır.
- **Avantajları:** Muazzam büyüklükte bir "Toplam Parametre" (Örn: 671 Milyar) kapasitesi ile zeka artırılırken, her kelimede harcanan "Aktif Parametre" (Örn: 37 Milyar) küçük kaldığından hız yüksek ve Compute maliyeti düşük kalır.
- **Dezavantajları:** Tüm uzmanlar o an çalışmasa bile ağırlıkları (weights) VRAM'de durmak zorundadır, bu yüzden modeli yüklemek için yüksek VRAM kapasiteli donanım gerektirir. Küçük modellerde yönlendirme kararı (Router) ekstra gecikme yaratır.
- **Benchmark:** ~308M Parametrelik Tek Bir Katman İçin:
  - Dense FFN (Tek Parça) Aktif Parametre/Token: **308 Milyon** (Süre: 0.75 ms)
  - MoE (8 Uzmanlı, Top 2 Seçim) Aktif Parametre/Token: **77 Milyon** (Aktif yük 4 kat azaldı)

---

## 6. Gated DeltaNet (Linear Attention)

*(Qwen3-Next, Kimi Linear modellerinde kullanılır)*

- **Çözdüğü Problem:** Standart Attention'ın her kelimeyi matris olarak diğer her kelimeyle çarpması, $O(N^2)$ (Karesel) bir zorluk yaratır. Milyon token seviyelerinde bu matematiksel olarak imkansızlaşır.
- **Nasıl Çalışır:** Klasik RNN'ler (Tekrarlayan Sinir Ağları) ve Mamba gibi State-Space modellerinden ilham alınmıştır. Model geçmişi büyük bir dikkat matrisinde (NxN) tutmak yerine, sabit boyutlu bir "Durum (State - S)" değişkenine sıkıştırır. Kapılar (Gates) sayesinde eski bilgilerin ne kadarının unutulacağına (Decay) ve yeni bilginin ne kadar önemli olduğuna (Update) karar verilerek Durum güncellenir.
- **Avantajları:** Hesaplama karmaşıklığını ve VRAM tüketimini Karesel $O(N^2)$'den **Lineer $O(N)$**'e düşürür. KV Cache boyutu okunan token sayısından bağımsız olarak daima **sabit** kalır. Sınırsız context length potansiyeli yaratır.
- **Dezavantajları:** Karışık geçmişi tek bir "Durum" değişkenine sıkıştırmak zorundadır. Bu sebeple spesifik bilgileri, MHA'nın net hatırlama gücü kadar iyi hatırlayamayabilir. (Bu yüzden Qwen3, bunu 3:1 Linear/Global hibrit oranıyla dengeler).
- **Benchmark:** VRAM Karşılaştırması Formülü:
  - MHA KV Cache = `Uzunluk(N) * Boyut(D)` -> N arttıkça şişer.
  - DeltaNet KV Cache = Sadece `Boyut(D) * Boyut(D)` -> N ile artmaz, sabittir!

---

## Genel Özet (Ne Zaman Hangisi?)

| Teknik             | Çözdüğü Asıl Sorun                                  | Feda Edilen / Dezavantaj                                                        | Nerede Popüler?  |
| :----------------- | :-------------------------------------------------------- | :------------------------------------------------------------------------------ | :---------------- |
| **GQA**      | KV Cache'in yarattığı VRAM darboğazı                 | Grup birleştirme sebebiyle çok ufak kalite kaybı                             | Llama 3, Qwen 2.5 |
| **MLA**      | VRAM darboğazı (Kaliteden ödün vermeden)              | Sıkıştırma/Açma sebebiyle ufak hesaplama yükü (Compute)                  | DeepSeek V3 / R1  |
| **SWA**      | Çok uzun bağlamlarda KV Cache'in şişmesi              | Eskiyi unutma riski (Global katmanlarla çözülür)                            | Gemma 3           |
| **MoE**      | Model zekasını artırırken yavaşlamayı önleme       | Tüm uzmanları hafızada tutmak için fiziksel dev donanım (VRAM) gereksinimi | Mixtral, DeepSeek |
| **DeltaNet** | Matrix çarpımının Quadratic ($O(N^2)$) sınırları | Uzak geçmişi hatırlama/odak kaybı (Bilgi sıkışması)                     | Qwen3-Next, Mamba |

## 7. FlashAttention (Donanım Optimizasyonu)

*(Tüm Modern LLM'lerde kullanılır - Llama 3, GPT-4, Claude 3)*

- **Çözdüğü Problem:** Standart Attention hesaplamasında, ara sonuçlar (Attention Scores Matrisi) GPU'nun hızlı ama küçük olan SRAM belleği ile yavaş ama büyük olan HBM (VRAM) belleği arasında sürekli gidip gelir. Bu veri transferi (IO-bound), hesaplamadan (Compute-bound) çok daha fazla zaman alır ve VRAM'i şişirir.
- **Nasıl Çalışır:** Donanım seviyesinde bir optimizasyondur. GPU'nun SRAM belleği içine sığabilecek bloklar (Tiles) halinde hesaplama yapar. Ara sonuçları asla HBM'e (VRAM) yazmaz. Hesaplamayı bitirir, softmax uygular ve sonucu doğrudan bir sonraki katmana (FFN) gönderir. Buna **Tiling** denir.
- **Avantajları:**
  - **Hız:** Veri transferi minimuma indiği için %2-4 kat arası hız artışı sağlar.
  - **VRAM:** Ara matrisleri kaydetmediği için, aynı donanımda çok daha uzun bağlam (Context Length) işleyebilir.
- **Dezavantajları:** Donanıma bağımlıdır. Sadece NVIDIA Ampere (RTX 3000) ve sonrası GPU'larda tam verimlilikle çalışır.
- **Benchmark:** 80 Katmanlı, 128K Token Context:
  - Standart Attention: **Çalışmaz** (VRAM yetersiz kalır).
  - FlashAttention: **Çalışır** (Çok daha az VRAM kullanır).

## 8. RMSNorm (Root Mean Square Normalization)

*(Llama 3, Gemma, Mistral ve birçok modern modelde kullanılır)*

- **Çözdüğü Problem:** Standart LayerNorm, ortalamayı (Mean) sıfıra çekerken veriyi bir de standart sapmaya (Variance) göre normalize eder. Bu ek hesaplamalar (özellikle karekök alma) ve bias terimi, her katmanda ufak da olsa zaman kaybına yol açar.
- **Nasıl Çalışır:** Sadece verinin büyüklüğüne (Magnitude) odaklanır. Veriyi ortalamasını almadan, doğrudan kareköküne (RMS) bölerek normalize eder. Bias terimi tamamen kaldırılmıştır.
- **Avantajları:** Hesaplama basittir, daha az matris çarpımı gerektirir. Bu sayede hem eğitim (Training) hem de çıkarım (Inference) hızını artırır.
- **Dezavantajları:** LayerNorm kadar güçlü bir regülasyon (aşırı öğrenmeyi önleme) sağlamayabilir. Ancak modern modellerde bu eksiklik, Dropout oranlarının düşürülmesi veya ek regülasyon teknikleriyle telafi edilir.
- **Benchmark:** 70B Parametreli Model Eğitimi:
  - LayerNorm: **100%** (Referans)
  - RMSNorm: **%95** (Aynı doğruluk, %5 daha hızlı)

---

## 🚀 Ek: GPT Mimarisi Analizi (Neden Sadece Decoder?)

**GPT (Generative Pre-trained Transformer)** model ailesinin mimarisi tamamen **Decoder (Çözücü) tabanlıdır** (Decoder-only architecture). Çeviri yapmak için tasarlanan orijinal Transformer modelindeki *Encoder (Gömücü)* bölmesi GPT'de bulunmaz.

Bunun temel sebebi şudur:

* **Orijinal Transformer (Encoder + Decoder):** Encoder metni bütünüyle okur, bağlamı kavrar. Decoder ise bu bağlamdan yeni bir metin (örn. başka dilde çeviri) üretir.
* **BERT (Sadece Encoder):** Metni her iki yönden (sağdan ve soldan) okuyarak anlama ve sınıflandırma (Örn. duygu analizi, boşluk doldurma) işlerinde mükemmeldir. Ancak kelime kelime akıcı hikaye üretemez.
* **GPT (Sadece Decoder):** Sadece "kendi solundaki geçmiş kelimelere" bakarak sağdaki **İLK YENİ KELİMEYİ** tahmin etme (Next-token prediction) üzerine eğitilmiştir. Bu "Maskeli (Causal) Attention" kısıtlaması, GPT'yi metin üretimi (Generation) konusunda bir numara yapar.

**Sonuç:** Llama, Claude, DeepSeek ve GPT serisi gibi sıfırdan şiir yazan, kod üreten, sohbet eden genel amaçlı LLM'lerin %99'u **Decoder-Only** (Sadece Çözücü) mimarisine sahiptir.

## Görselleştirme

[Modern LLM Optimizasyon Teknikleri](https://gemini.google.com/share/27a4cc8cdc39)
[Modern LLM Optimizasyon Teknikleri AI Studio](https://ai.studio/apps/4cb94799-dcd2-4e60-82eb-030e1de2acc8)
