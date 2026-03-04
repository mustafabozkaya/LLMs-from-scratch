# Bonus: BPE Tokenizer Implementasyonlarını Karşılaştırma

## Ana Fikir

Bu not defteri, GPT-2 için kullanılan farklı Byte-Pair Encoding (BPE) tokenizer implementasyonlarının performansını karşılaştırır. Temel amaç, OpenAI'nin Rust tabanlı `tiktoken` kütüphanesinin, orijinal Python tabanlı veya diğer kütüphane uygulamalarına kıyasla ne kadar verimli olduğunu göstermektir.

### Karşılaştırılan Implementasyonlar

Bu bölümde dört ana BPE tokenizer uygulaması incelenmiş ve `%timeit` ile hızları test edilmiştir:

1.  **`tiktoken`:** OpenAI tarafından geliştirilen, çekirdeği Rust ile yazılmış yüksek performanslı tokenizer.
2.  **Orijinal GPT-2 BPE:** GPT-2'nin ilk çıktığı zamanlarda kullanılan saf Python tabanlı `encoder.py` script'i.
3.  **Hugging Face `transformers`:**
    *   `GPT2Tokenizer`: Python tabanlı, yavaş implementasyon.
    *   `GPT2TokenizerFast`: Rust tabanlı, `tiktoken`'e benzer şekilde hızlı bir implementasyon.
4.  **Sıfırdan BPE:** Kitabın `05_bpe-from-scratch` bölümünde eğitim amaçlı yazılan Python tabanlı yavaş implementasyon.

### Teknik Uygulama (`compare-bpe-tiktoken.ipynb`)

*   Her bir tokenizer ( `tiktoken`, `bpe_openai_gpt2`, `transformers.GPT2Tokenizer`, `transformers.GPT2TokenizerFast` ve `BPETokenizerSimple`) aynı metin (`the-verdict.txt`) üzerinde çalıştırılır.
*   `%timeit` magic komutu kullanılarak her birinin `.encode()` metodunun ortalama çalışma süresi ölçülür.
*   Tüm implementasyonların aynı metin için **aynı token ID'lerini** ürettiği, yani işlevsel olarak doğru çalıştıkları doğrulanır.

### Nihai Çıktı ve Temel Sonuç

*   **Hız Kralı `tiktoken`:** Test sonuçları, `tiktoken`'in (~1.59 ms) saf Python tabanlı orijinal GPT-2 implementasyonundan (~6.82 ms) ve yavaş Hugging Face tokenizer'ından (~16.7 ms) **önemli ölçüde daha hızlı** olduğunu açıkça göstermektedir.
*   **Rust'ın Gücü:** `tiktoken` ve `GPT2TokenizerFast` gibi Rust tabanlı tokenizer'ların hızı, büyük veri kümeleriyle çalışırken veri ön işleme (preprocessing) adımlarını ne kadar hızlandırdığını kanıtlar.
*   **Sonuç:** Büyük dil modelleri eğitilirken veya kullanılırken, veri hazırlama hattının (data pipeline) verimliliği kritik öneme sahiptir. `tiktoken` gibi optimize edilmiş bir kütüphane kullanmak, bu süreçte hem zamandan hem de hesaplama kaynaklarından tasarruf sağlar.
