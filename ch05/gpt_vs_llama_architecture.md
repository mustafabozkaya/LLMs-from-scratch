# GPT ve LLaMA Mimarileri: Karşılaştırmalı Analiz ve "Hidden State" Kavramı

Büyük Dil Modelleri (LLM) dünyasında, orijinal GPT-2/GPT-3 mimarisi ("Build a Large Language Model From Scratch" kitabında sıfırdan inşa ettiğimiz model) uzun süre altın standart olarak kabul edildi. Ancak Meta AI'ın LLaMA (Llama 2, Llama 3) modelleri, bu temel "Decoder-Only Transformer" yapısını alıp kritik matematiksel ve yapısal optimizasyonlar yaparak günümüzün açık kaynak lideri haline geldi.

Bu raporda, her iki mimarinin temel farklarını, LLaMA'nın getirdiği devrimsel yenilikleri ve **Hidden State** (Gizli Durum) verisinin bu iki mimarideki yolculuğunu inceleyeceğiz.

---

## 1. Mimari Karşılaştırma Özeti

| Özellik | Orijinal GPT (GPT-2) | LLaMA (Llama 2 & 3) | LLaMA'nın Avantajı |
| :--- | :--- | :--- | :--- |
| **Normalizasyon** | [LayerNorm](file:///c:/Users/kurtar/Desktop/Sebastian%20rashgca/LLMs-from-scratch/ch04/04_gqa/gpt_with_kv_gqa.py#129-141) | `RMSNorm` | Matematiksel olarak daha sade, %10-20 daha hızlı hesaplanır. |
| **Aktivasyon Fonk.** | [GELU](file:///c:/Users/kurtar/Desktop/Sebastian%20rashgca/LLMs-from-scratch/ch04/03_kv-cache/gpt_with_kv_cache_optimized.py#146-155) | `SwiGLU` | Daha karmaşık (Gated) bir yapıya sahip, doğrusal olmayan öğrenmeyi artırır. |
| **Pozisyonlama** | Mutlak (Absolute) / Öğrenilebilir | `RoPE` (Rotary Positional Embeddings) | Uzun metinleri çok daha iyi anlar, dışarıdan yeni bağlam uzunluklarına daha kolay adapte olur (Extrapolation). |
| **Dikkat (Attention)** | Multi-Head Attention (MHA) | Grouped-Query Attention (GQA) / MHA | Hafıza (VRAM) kullanımını inanılmaz derecede düşürür (Özellikle KV-Cache sırasında). |
| **Bias Kullanımı** | Linear katmanlarda `bias=True` | Doğrusal (Linear) katmanlarda `bias=False` | Daha az parametre, daha stabil ve hızlı eğitim. |

---

## 2. Mimari Diyagramlar: Hidden State'in Yolculuğu

Aşağıdaki diyagramda, kelimelerin modele girdiği andan itibaren **Hidden State** olarak adlandırıldığı bölümleri ve GPT ile LLaMA blokları arasındaki devasa farkı görebilirsiniz.

```mermaid
graph TD
    classDef input fill:#f9d0c4,stroke:#333,stroke-width:2px;
    classDef hidden fill:#d4e6f1,stroke:#333,stroke-width:2px;
    classDef block fill:#d5f5e3,stroke:#333,stroke-width:2px;
    classDef output fill:#fcf3cf,stroke:#333,stroke-width:2px;

    subgraph GPT ["Klasik GPT Mimarisi"]
        G_In["Girdi Kelime IDleri"] ::: input
        G_Emb["H_0 Ilk Hidden State\nToken Embd + Absolute Pos Embd"] ::: hidden
        
        G_Block["Transformer Block x12 Katman"] ::: block
        G_Block_Det["1. LayerNorm\n2. Multi-Head Attention\n3. Add Residual\n4. LayerNorm\n5. FeedForward GELU\n6. Add Residual"]
        G_Block -.-> G_Block_Det
        
        G_Out_Hidden["Final Hidden State H_12"] ::: hidden
        G_OutNorm["Final LayerNorm"] ::: block
        G_Out["Linear Katman Logits\n50257 Kelime Puani"] ::: output

        G_In --> G_Emb --> G_Block --> G_Out_Hidden --> G_OutNorm --> G_Out
    end

    subgraph LLAMA ["LLaMA 2 ve 3 Mimarisi"]
        L_In["Girdi Kelime IDleri"] ::: input
        L_Emb["H_0 Ilk Hidden State\nSADECE Token Embd"] ::: hidden
        
        L_Block["Llama Block x32 Katman"] ::: block
        L_Block_Det["1. RMSNorm\n2. RoPE + Grouped-Query Attention\n3. Add Residual\n4. RMSNorm\n5. FeedForward SwiGLU\n6. Add Residual"]
        L_Block -.-> L_Block_Det

        L_Out_Hidden["Final Hidden State H_32"] ::: hidden
        L_OutNorm["Final RMSNorm"] ::: block
        L_Out["Linear Katman Logits\n128000 Kelime Puani"] ::: output

        L_In --> L_Emb --> L_Block --> L_Out_Hidden --> L_OutNorm --> L_Out
    end
```

### "Hidden State" Tam Olarak Nerededir?
Yukarıdaki diyagramda **Mavi renkli (`H_0` ve `H_Final`) ve Yeşil renkli (`Blok içi`)** kısımlarda akan veri, Hidden State'tir. Modelin "kendi içindeki karanlık odasıdır". Bu veri; 
* Sadece sayılardan oluşur (Örn: `[Batch=1, Seq=1024, Embed_Dim=4096]`).
* Logits aşamasında, kelime sözlüğündeki (Vocab) kelimelerle çarpıştırılıp olasılıklara dönüştürüldüğü an **Hidden State ölür, yerine İhtimaller doğar.**

---

## 3. LLaMA'nın Getirdiği 4 Büyük Yenilik (Detaylı Analiz)

### 3.1. LayerNorm Yerine `RMSNorm` (Root Mean Square Normalization)
*   **GPT'nin Yöntemi:** [LayerNorm](file:///c:/Users/kurtar/Desktop/Sebastian%20rashgca/LLMs-from-scratch/ch04/04_gqa/gpt_with_kv_gqa.py#129-141), her veriyi ortalamasını (mean) bularak sıfıra çeker ve ardından varyansına bölerek standartlaştırır. (Ortalama hesaplamak sistem yorar).
*   **LLaMA'nın Yöntemi:** Araştırmalar gösterdi ki, normalizasyondaki başarının asıl sebebi "ortalama almak" değil, değerleri ölçeklendirmektir. `RMSNorm`, ortalama hesaplamayı tamamen çöpe atar. Sadece karelerin ortalamasının karekökünü (RMS) alarak scale (ölçekleme) yapar.
*   **Sonuç:** Matematiksel olarak daha ucuz ve eğitim sırasında %10-20'ye varan hız artışı sağlar.

### 3.2. GELU Yerine `SwiGLU` (Swish Gated Linear Unit)
Akıllı bir nöral ağ, doğrusal olmayan (non-linear) karmaşık kalıpları öğrenebilmek için Aktivasyon fonksiyonlarına muhtaçtır.
*   **GPT'nin Yöntemi:** [GELU](file:///c:/Users/kurtar/Desktop/Sebastian%20rashgca/LLMs-from-scratch/ch04/03_kv-cache/gpt_with_kv_cache_optimized.py#146-155) (Gaussian Error Linear Unit). Basit ve etkilidir.
*   **LLaMA'nın Yöntemi:** `SwiGLU`. Burada veriyi iki ayrı koldan geçirir. Bir kolu Swish aktivasyonundan geçirirken, diğer kolu onsuz ilerletir ve sonda bu ikisini birleştirir (element-wise multiplication). Buna **"Gated" (Kapılı)** mimari denir. Hangi bilginin geçip hangisinin geçmeyeceğine (forget/remember) model matris çarpımıyla karar verir.
*   **Sonuç:** Modelin parametre sayısı burada hafifçe artsa da (çünkü fazladan matris tutar), başarısı muazzam derecede yukarı fırlar.

### 3.3. Dönen Pozisyonlar: `RoPE` (Rotary Positional Embeddings)
Bu, GPT mimarisinden en radikal kopuştur.
*   **GPT'nin Yöntemi (Mutlak Pozisyonlama):** Modele kelime girerken (*"Benim adım..."*), "Benim" kelimesinin üstüne "Sen 1. sıradasın", "adım" kelimesinin üstüne "Sen 2. sıradasın" yazan devasa bir statik matris eklenir ($H_0$ oluşturulurken).
*   **LLaMA'nın Yöntemi:** `RoPE`'ta kelimeler modele girerken **hiçbir pozisyon bilgisi almazlar!** ($H_0$ sadece Token Embedding'den ibarettir). Peki tokenlar sırasını nasıl bilecek? 
    * Pozisyon bilgisi, Multi-Head Attention'ın tam İÇİNDE, **Query (Soru) ve Key (Anahtar)** vektörleri birbirleriyle çarpıştırılırken verilir.
    * Her bir kelime vektörü, bulunduğu pozisyona göre sanal bir dairesel uzayda (Complex Matrix) belirli bir **açı (derece) kadar dairesel döndürülür (Rotary)**. 
*   **Sonuç:** Bu sayede kelimeler sadece birbirinin nerede olduğunu değil, **"Aramızda kaç kelime mesafe var?" (Relative Position)** sorusuna yanıt verebilir hale geldi. Bu da LLaMA modellerinin 100 bin kelimelik devasa metinleri okuyabilmesini sağlayan en kritik keşiftir.

### 3.4. GQA (Grouped-Query Attention) ve KV-Cache Optimizasyonu
*   **GPT'nin Yöntemi:** GPT metin üretirken her yeni kelimede geçmişi tekrar tekrar hesaplamamak için KV-Cache kullanır. Ancak GPT'nin çok sayıda kafası (Heads) vardır (Örn: 96 kafa). Her bir kafa için geçmiş Query, Key, Value değerlerini RAM'de tutmak devasa bir bellek (VRAM - Ekran kartı hafızası) tüketir.
*   **LLaMA 2 ve 3 Yöntemi (GQA):** "Neden 96 ayrı kafa için 96 ayrı Key-Value tutalım ki?" diyerek Kafaları gruplarlar. Örneğin 8 Query kafası, sadece 1 adet Key-Value kafasını ortaklaşa kullanır.
*   **Sonuç:** Aynı zeka seviyesini korurken, VRAM tüketiminde (özellikle uzun metinlerde) devasa bir tasarruf (8 kata varan oranlarda) elde edildi. Bu sayede Llama 3 modelleri evdeki ortalama bilgisayarlarda bile çalışabilmektedir.

---

## Özet
Ağırlık yükleme bölümünden de görebileceğiniz gibi, orijinal GPT kodu hala temel "baba" mimariyi temsil etmektedir. Ancak Meta ekibi, LLaMA serisinde o sadeliği alıp donanım verimliliğini (RMSNorm ve GQA) ve öğrenme kapasitesini (SwiGLU ve RoPE) en üst düzeye çıkaran modern standartları oluşturmuştur. Tüm modern modeller (Mistral, Qwen, DeepSeek vb.) bugün temel GPT yerine bu LLaMA-benzeri ("Llama-like") iyileştirmeleri kullanmaktadır.
