# Bölüm 2: Metin Verilerini LLM İçin Hazırlama (Özet)

## Ana Fikir

Bu bölüm, ham metin verisinin, bir Büyük Dil Modeli (LLM) tarafından anlaşılabilecek ve işlenebilecek sayısal tensörlere nasıl dönüştürüldüğünü adım adım açıklamaktadır. Bu süreç, modelin "sonraki kelimeyi tahmin etme" görevini öğrenebilmesi için temel veri hazırlama hattını (data pipeline) oluşturur.

### Temel Kavramlar ve Süreç

1.  **Tokenizasyon (Metni Parçalara Ayırma):**
    *   **Neden?** Modeller metinleri değil, sayıları anlar. Metinler, anlamlı en küçük birimlere (token'lara) ayrılmalıdır.
    *   **Nasıl?** `tiktoken` kütüphanesi kullanılarak GPT-2'nin BPE (Byte-Pair Encoding) tokenizasyon yöntemi uygulanır. Bu yöntem, kelimeleri ve hatta bilinmeyen kelimeleri daha küçük alt kelime birimlerine ayırarak `vocab_size` (kelime dağarcığı boyutu) sorununu çözer. Metindeki her bir parça, bir kimlik numarasına (ID) dönüştürülür.

2.  **Veri Kümesi Oluşturma (Input-Target Pairs):**
    *   **Neden?** Modeli eğitmek için ona "girdi" (input) ve bu girdiye karşılık gelmesi beklenen "hedef" (target) sunulmalıdır.
    *   **Nasıl?** "Kayan Pencere" (Sliding Window) tekniği kullanılır. `max_length` (bağlam penceresi) boyutunda bir metin parçası alınır.
        *   **Girdi (X):** `token_ids[i: i + max_length]`
        *   **Hedef (Y):** `token_ids[i+1: i + max_length + 1]`
    *   Bu yöntemle, model her bir pozisyon için bir sonraki token'ı tahmin etmeyi öğrenir.

3.  **Veri Yükleme (Batching):**
    *   **Neden?** Veriyi tek tek işlemek yerine, GPU'da paralel hesaplama gücünden yararlanmak için veriler gruplar (batch) halinde işlenir.
    *   **Nasıl?** PyTorch'un `DataLoader`'ı, oluşturulan (girdi, hedef) çiftlerini `batch_size` ile belirtilen sayıda bir araya getirerek modelin eğitimi için hazır tensör grupları oluşturur.

4.  **Embedding (Anlamsal ve Konumsal Vektörler):**
    *   **Token Embedding:** Token ID'leri (örneğin, 50257 boyutlu bir sözlükteki sıra numaraları), modelin anlamsal ilişkileri öğrenebileceği yoğun vektörlere (`output_dim` boyutlu, örn: 256) dönüştürülür. Bu işlem bir arama tablosu (`nn.Embedding`) gibi çalışır.
    *   **Positional Embedding (Konumsal Gömme):** Transformer mimarisi, kelimelerin sırasına doğal olarak duyarlı değildir. Bu yüzden her bir token'ın vektörüne, o token'ın dizideki konumunu belirten bir "konum vektörü" eklenir. Bu sayede "kedi fareyi yedi" ile "fare kediyi yedi" arasındaki fark model tarafından anlaşılabilir hale gelir.

### Teknik Uygulama (`ch02.ipynb`)

*   **Tokenizasyon:** `tiktoken.get_encoding("gpt2")` ile GPT-2 tokenizer'ı yüklenir ve `.encode()` metoduyla metinler ID dizilerine çevrilir.
*   **Veri Kümesi:** `GPTDatasetV1` sınıfı, token'lanmış tüm metni alır ve kayan pencere (`stride` ile adımlar atlayarak) yöntemiyle girdi ve hedef ID'lerini oluşturur.
*   **Veri Yükleyici:** `create_dataloader_v1` fonksiyonu, `GPTDatasetV1`'i kullanarak, veriyi `batch_size`'a göre gruplayan bir `DataLoader` nesnesi döndürür.
*   **Embedding Katmanı:** `torch.nn.Embedding(vocab_size, output_dim)` ile bir embedding katmanı oluşturulur. Bu katman, token ID'lerini alıp onlara karşılık gelen embedding vektörlerini döndürür.

### Nihai Çıktı

Bölümün sonunda, elimizde ham metinden yola çıkarak oluşturulmuş, `[batch_size, max_length]` boyutunda token ID'leri içeren tensörler (girdiler ve hedefler) bulunur. Bu tensörler, `nn.Embedding` katmanından geçirilerek `[batch_size, max_length, output_dim]` boyutunda, anlamsal ve konumsal bilgiyle zenginleştirilmiş, modelin bir sonraki bölümlerde işleyeceği nihai girdi haline gelir.
