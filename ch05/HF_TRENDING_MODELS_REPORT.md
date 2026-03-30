# Hugging Face Gerçek Zamanlı Trend Modeller Analiz Raporu (Mart 2026)

Bu rapor, Hugging Face Hub üzerindeki anlık popülerleşen (trending) modelleri, bu modellerin mimari yeniliklerini, "Uncensored/Distilled" gibi varyasyonların nasıl üretildiğini ve kullanılan veri setlerini incelemektedir.

---

## 1. Trend Listesi Özeti (Top 10 Focus)

Şu an listede zirveyi çeken ilk 10 model ve kategorileri:

1. **QWEN 3.5 (27B/9B/35B-A3B):** Listenin tartışmasız lideri. Hem yoğun (dense) hem de MoE versiyonlarıyla trendlerde.
2. **NVIDIA NEMOTRON 3 SUPER (120B):** Hibrit mimarisiyle devasa bağlam (1M token) sunan model.
3. **LTX-2.3 (Lightricks):** Video üretim dünyasında yeni standart haline gelen difüzyon modeli.
4. **S2-PRO (Fish Audio):** Ses ve konuşma sentezleme (TTS) alanında en çok ilgi gören model.
5. **TADA-1B/3B (Hume AI):** Duygusal zeka ve sesli etkileşim odaklı küçük dil modelleri.
6. **SARVAM-105B:** Hindistan odaklı, çok dilli ve devasa parametreli yerel model atağı.

---

## 2. Varyasyon Analizi: "Uncensored" ve "Distilled" Nedir?

Trend listesindeki isimlendirmelerde sıkça geçen bu terimler, modellerin "post-training" (eğitim sonrası) aşamalarını ifade eder.

### 2.1. Uncensored (Sansürsüz) Varyasyonlar
*   **Teknik Süreç:** Standart modeller (Llama, Qwen vb.) eğitim sonunda **RLHF** (Human Feedback) ile etik sınırlar içine hapsedilir. "Uncensored" modeller, bu hizalama (alignment) katmanının kaldırıldığı veya reddetme davranışının öğretilmediği veri setleri ile yeniden eğitilir.
*   **Örnek:** `HAUHAUCS/QWEN3.5-9B-UNCENSORED`

### 2.2. Distilled (Damıtılmış) Varyasyonlar
*   **Teknik Süreç:** Bilgi Damıtma (Knowledge Distillation) tekniğinde, "Öğretmen" (Teacher) olan devasa bir modelin (örn: Claude 4.6 Opus veya DeepSeek-R1) ürettiği mantık yürütme (reasoning) yolları, "Öğrenci" (Student) olan daha küçük bir modele öğretilir.
*   **Örnek:** `JACKRONG/QWEN3.5-27B-CLAUDE-4.6-OPUS-REASONING-DISTILLED`

---

## 3. Teknik Uygulama: Bu Varyasyonlar Nasıl Üretiliyor?

Trend olan modellerin neredeyse tamamı **QLoRA (Quantized Low-Rank Adaptation)** yöntemiyle finetune edilmiştir.

### 3.1. Eğitim Metodolojisi: QLoRA
*   **Neden Tercih Ediliyor?** 27B veya 72B gibi modelleri tam parametreyle eğitmek devasa A100/H100 kümeleri gerektirir. QLoRA, ana modeli 4-bit (NF4) olarak dondurur ve sadece üzerine eklenen küçük "adapter" katmanlarını eğitir. Bu sayede 72B'lik bir model, tek bir 24GB VRAM'li GPU'da finetune edilebilir hale gelir.
*   **Kullanılan Araçlar:** Unsloth (hızlandırılmış eğitim), Axolotl (konfigürasyon tabanlı eğitim) ve bitsandbytes.

### 3.2. Kullanılan Kritik Veri Setleri
Varyasyonun türüne göre kullanılan popüler veri setleri şunlardır:

| Varyasyon Türü | Popüler Veri Seti | Amacı |
| :--- | :--- | :--- |
| **Uncensored** | *Eric Hartford's Wizard-Vicuna-Unfiltered* | Reddetme (refusal) davranışını temizlemek. |
| **Reasoning / Distilled** | *DeepSeek-R1-Distill-Data-110k* | "Chain of Thought" (Düşünce Zinciri) yeteneği kazandırmak. |
| **General Purpose** | *OpenHermes-2.5 / ShareGPT* | Çok yönlü sohbet ve talimat takip yeteneği. |
| **Coding** | *Magicoder-OSS-Instruct-75K* | Karmaşık programlama yeteneklerini geliştirmek. |

### 3.3. Adım Adım Üretim Akışı
1.  **Model Seçimi:** Güçlü bir temel (Base) model seçilir (örn: Qwen 3.5 9B).
2.  **Veri Sentezi (Distillation için):** Öğretmen modelden (örn: GPT-4o) binlerce soruya "Adım adım düşünerek" cevap vermesi istenir. Bu sentetik veri kaydedilir.
3.  **Hizalama (Alignment) Kararı:**
    *   *Uncensored yapılacaksa:* Güvenlik filtreleri içeren veriler çıkartılır, saf bilgi verilir.
    *   *Reasoning yapılacaksa:* Mantık hatalarını cezalandıran **DPO (Direct Preference Optimization)** algoritması uygulanır.
4.  **QLoRA Eğitimi:** Model 4-bit'e sıkıştırılır ve veri setiyle eğitilir.

---

## 4. Mimari Devrim: Standart Transformer'ın Ötesi

Trend modeller artık saf Transformer mimarisinden uzaklaşıp hibrit yapılara geçiyor.

### 4.1. Qwen 3.5 Mimarisi (Hibrit Dikkat)
*   **Gated DeltaNet (Linear Attention):** Katmanların %75'inde kullanılır. Sonsuz context ve sabit RAM kullanımı sağlar.
*   **GQA (Grouped-Query Attention):** Katmanların %25'inde kullanılır. DeltaNet'in uzak geçmişi unutma sorununu çözer.

### 4.2. NVIDIA Nemotron 3 Super (Mamba-Transformer MoE)
1.  **Mamba-2 Layers:** Veriyi doğrusal zamanda işler.
2.  **Transformer Attention:** Kritik noktaları yakalar.
3.  **Latent MoE:** 120B parametrenin sadece 12B'sini aktif ederek verimlilik sağlar.

---

## 5. Teknik Karşılaştırma Tablosu

| Özellik | Qwen 3.5 Serisi | NVIDIA Nemotron 3 | Llama 3.2 (Referans) |
| :--- | :--- | :--- | :--- |
| **Ana Mimari** | Hybrid (DeltaNet + GQA) | Hybrid (Mamba + Transf.) | Pure Transformer |
| **Context Window** | 256K+ Token | 1M+ Token | 128K Token |
| **VRAM Verimi** | Çok Yüksek (Linear) | Çok Yüksek (SSM) | Orta (Quadratic) |
| **Finetune Türü** | QLoRA / Unsloth | Full / Distributed | QLoRA / LoRA |

---

## 6. Sonuç ve Öngörü

Mart 2026 trendleri, **QLoRA**'nın demokratikleştirici gücü sayesinde topluluğun devasa modelleri (27B+) kendi ihtiyaçlarına göre (Uncensored, Distilled) hızla özelleştirebildiğini göstermektedir. Özellikle **DeepSeek-R1**'den damıtılan (distilled) sentetik veri setleri, küçük modellerin zeka seviyesini bir üst lige taşımıştır.
