# Bonus: Sıfırdan Byte-Pair Encoding (BPE) Tokenizer Yazmak

## Ana Fikir

Bu not defteri, GPT-2 gibi modern LLM'lerde kullanılan Byte-Pair Encoding (BPE) tokenizasyon algoritmasının, eğitim ve kullanım mantığını sıfırdan bir Python sınıfı (`BPETokenizerSimple`) içinde kodlayarak öğretmeyi amaçlar. Amaç, hazır kütüphanelerin (örn: `tiktoken`) arkasındaki "sihri" anlamak ve kendi veri setimiz için nasıl özel bir tokenizer eğitebileceğimizi görmektir.

### BPE Algoritmasının Adım Adım Kodlanması

Algoritma, en sık tekrar eden bitişik karakter veya token çiftlerini bularak ve bunları yeni bir token olarak birleştirerek çalışır.

1.  **Başlangıç Sözlüğü (Initial Vocab):** Eğitim, metindeki tüm benzersiz karakterleri içeren bir başlangıç sözlüğü ile başlar (genellikle ilk 256 byte değeri).
2.  **En Sık Tekrar Eden Çifti Bul (`find_freq_pair`):** Token'lanmış metin taranır ve en sık tekrar eden ardışık token çifti (bigram) bulunur.
3.  **Çifti Birleştir ve Değiştir (`replace_pair`):** Metindeki bu çiftin tüm tekrarları, sözlüğe eklenecek yeni bir token ID'si ile değiştirilir.
4.  **Birleştirmeyi Kaydet (`bpe_merges`):** Hangi çiftin hangi yeni ID ile birleştirildiği bir "birleştirme kuralları" (merges) listesine kaydedilir.
5.  **Tekrarlama:** Bu süreç, `vocab_size` ile belirlenen hedef kelime dağarcığı boyutuna ulaşılana kadar tekrarlanır.

### Teknik Uygulama (`BPETokenizerSimple` Sınıfı)

Not defteri, bu mantığı kapsayan bir sınıf sunar:

*   **`train(text, vocab_size)`:**
    *   Yukarıda açıklanan BPE eğitim sürecini uygular.
    *   Verilen metni alır, en sık tekrar eden çiftleri bularak `vocab_size` boyutunda bir sözlük (`vocab`) ve birleştirme kuralları listesi (`bpe_merges`) oluşturur.

*   **`encode(text)`:**
    *   Bir metni token ID'lerine dönüştürür.
    *   Metni önce karakterlere ayırır, ardından eğitimde öğrenilen `bpe_merges` kurallarını en yüksek öncelikliden başlayarak (veya GPT-2 için rank'a göre) uygular.
    *   Mümkün olan en uzun alt kelimeler (subwords) oluşana kadar birleştirmeye devam eder ve sonuçta bir ID listesi döndürür.

*   **`decode(token_ids)`:**
    *   Token ID listesini tekrar okunabilir metne dönüştürür.
    *   `vocab` sözlüğünü kullanarak her bir ID'yi karşılık gelen karaktere, alt kelimeye veya kelimeye çevirir.

*   **`load_vocab_and_merges_from_openai(...)`:**
    *   Sıfırdan eğitmek yerine, OpenAI'nin orijinal GPT-2 tokenizer'ına ait `encoder.json` (vocab) ve `vocab.bpe` (merges) dosyalarını yükleyerek hazır bir tokenizer kullanma yeteneği sağlar.

### Nihai Çıktı ve Temel Sonuç

Bu not defteri sayesinde, BPE'nin sadece teorik bir kavram olmadığı, aynı zamanda saf Python ile adım adım kodlanabilir bir algoritma olduğu anlaşılır. Çıktı olarak, herhangi bir metin üzerinde eğitilebilen, kaydedilip yüklenebilen ve hem `encode` hem de `decode` yapabilen işlevsel bir tokenizer elde edilir. Bu, özel diller veya alanlar için LLM'ler geliştirirken tokenizer'ların nasıl özelleştirilebileceğine dair temel bir anlayış sağlar.
