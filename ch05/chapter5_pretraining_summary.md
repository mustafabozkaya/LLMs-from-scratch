# Bölüm 5: Etiketlenmemiş Veriler Üzerinde Ön Eğitim (Pretraining) Analizi

Bu bölümde, sıfırdan oluşturduğumuz GPT modelini ufak bir metin veri seti üzerinde eğitmeyi (pretraining), kayıp (loss) değerlerini hesaplamayı, metin üretiminde tutarlılığı artırmayı sağlayan çözümleme (decoding) stratejilerini ve önceden eğitilmiş devasa OpenAI GPT-2 ağırlıklarının modelimize nasıl yüklendiğini adım adım ele alıyoruz.

---

## 1. Veri Kümesi ve DataLoader ile Batch Yönetimi
Modelin öğrenmesi için sadece bir metin okuması yetmez, bunu sistematik olarak yapması gerekir.
- **Veri Yükleme:** Küçük bir roman/hikaye olan `the-verdict.txt` kullanılmıştır (yaklaşık 5145 token). Veri, Eğitim (%90) ve Doğrulama (Validation - %10) olarak ikiye bölünür.
- **DataLoader ve Kaydırma (Stride):** Modeli "sıradaki kelimeyi (next-token) bulma" üzerine eğittiğimiz için veri, diziler (sequence) halinde modele sokulur. Eğitim seti için dizileri hazırlarken `stride = context_length` kullanılır (yani pencereler atlamalı gider), ancak hedef (target) matrisleri, girdi matrisinin sağa doğru **1 kelime kaydırılmış** halidir.

## 1. İlk Metin Üretimi ve Loss İhtiyacı

Öncelikle daha önce yazdığımız GPT modelini, herhangi bir eğitim yapmadan (rastgele ağırlıklarla) bir metin üretmesi test edilir. 

### Modelin Başlatılması ve İlk Çıktı

```python
import torch
import tiktoken
from previous_chapters import GPTModel, generate_text_simple

# 124 Milyon parametreli GPT konfigürasyonu (Kitap kolay çalışsın diye context'i 256'ya indirilmiş)
GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Sözlük boyutu
    "context_length": 256, # Orjinali 1024, eğitim hızlı olsun diye kısaltılmış
    "emb_dim": 768,        # Gömme boyutu (Her kelimeyi 768 boyutlu vektör yapar)
    "n_heads": 12,         # Multi-Head Attention'daki kafa sayısı
    "n_layers": 12,        # Transformer bloğu sayısı
    "drop_rate": 0.1,      # Aşırı ezberlemeyi önleyen Dropout
    "qkv_bias": False      # Modern LLM'lerde QKV matrislerinde Bias kullanılmaz
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval();  # Çıkarım (inference) moduna alınır, dropout'lar kapatılır.

start_context = "Every effort moves you" # Başlangıç cümlemiz
tokenizer = tiktoken.get_encoding("gpt2") # Kelimeleri ID'lere çeviren sözlük

# Metni tokenize edip (1, n) boyutunda tensör yaparız
encoded = tokenizer.encode(start_context, allowed_special={'<|endoftext|>'})
encoded_tensor = torch.tensor(encoded).unsqueeze(0) 

# Rastgele ağırlıklara sahip modelden 10 kelime üretmesini isteriz
token_ids = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

# Çıkan ID'leri tekrar metne çeviri:
flat = token_ids.squeeze(0) 
print(tokenizer.decode(flat.tolist()))
# Çıktı: "Every effort moves you rentingetic wasnم refres RexMeCHicular stren"
```
**Açıklama:** Model daha eğitilmediği için (ağırlıklar rastgele atandığı için) "rentingetic" gibi anlamsız kelimeler veya harf dizilimleri üretti. Modeli eğitmemiz (Pretraining) ve başarısını ölçebilmemiz (Loss) gerektiği buradan anlaşılıyor.

---

## 2. Text Generation Loss: Çapraz Entropi ve Perplexity (Kafa Karışıklığı)
Bir modelin eğitiminin başarılı olup olmadığını bilmek için "Ne kadar kötü üretiyor?" sorusunu matematiksel olarak cevaplamamız gerekir.
1. **Olasılık Seçimi:** Model bir girdi aldığında Sözlük Boyutu (Vocab Size) kadar yani 50.257 boyutunda "Ham Olasılık (Logits)" basar. Hedef kelimemizin indeksi hangisiyse, logits matrisindeki o indekse denk gelen sayıyı Logaritmik olarak alırız. İdealde olasılığın en yüksek yani "0" (log 1) olmasını hedefleriz. Uzaklaştıkça Değer negatifleşir (Örn: -10).
2. **Çapraz Entropi (Cross-Entropy):** Derin öğrenmede negatif sayıyı sıfıra yükseltmek yerine, formülü (+) ile çarpıp, "Kayıp'ı sıfıra düşürmeyi" (minimize etmeyi) hedefleriz. Yani negatif ortalama olasılık bizim **Loss** değerimizdir. `torch.nn.functional.cross_entropy()` bu işlemi otomatik yapar.
3. **Perplexity:** Cross-Entropy Loss değerinin exponansiyeline ($e^{loss}$) denir. Anlamı, "Model bir sonraki kelimeyi seçerken kaç farklı kelime arasında kaldığı/kararsızlık yaşadığıdır". Eğitim başlarında `Perplexity: 48000` iken (yani kelimelerin tümünden birini rastgele çekiyorken), eğitimle beraber bu modelin eminliği arttıkça sayı düşer.

## 3. Basit Eğitim Döngüsü (Training Loop)
Eğitim dediğimiz süreç, ağırlıkların sırasıyla güncellenmesi döngüsüdür:
- **Optimizer:** `AdamW` optimizasyon algoritması kullanılmıştır. Kayıp (Loss) değerleri (`loss.backward()`) kullanılarak geriye doğru türevleme/gradyanlar hesaplanır ve `optimizer.step()` ile her bir Nöron ağırlığı optimum noktaya doğru yavaşça adım atar.
- **Optimizasyon/Yenileme:** Her yeni batch döngüsünde `optimizer.zero_grad()` ile bir önceki adımın momentumu sıfırlanmalıdır.
- Model eğitildikçe çıkan metnin dili İngilizceye yaklaşır (fakat veri küçük olduğu için overfit -ezberleme- yaşamaya başlar).

## 4. Decoding Stratejileri (Sıcaklık ve Top-K Örneklemesi)
Model eğitildikten sonra sürekli aynı `argmax(logits)` (en mantıklı sıradaki kelime) yöntemini seçersek model kendini tekrarlayan monoton, robotik metinler yazar. Farklılık katmak için "Decoding" manipülasyonları uygulanır.

### Sıcaklık Ölçeklendirme (Temperature Scaling)
- Ham logitleri `Softmax` fonksiyonuna göndermeden önce sabit bir "Sıcaklık (T)" değerine böleriz.
- **T < 1.0 (Örn 0.1):** Yüksek güvenilirlik (Daha keskin, en garanti kelimeleri basar. Çıldırma/halüsinasyon riski sıfıra yakındır). Kodlama için iyidir.
- **T > 1.0 (Örn 5.0):** Düşük güven, yüksek çeşitlilik (Olasılıkları eşitleyerek riskli kelime seçimini artırır. Yaratıcı hikayelerde kullanılır. Ancak yüksek tutulursa saçma kelimeler türetebilir).

### Top-K Örneklemesi
- Sıcaklık ile modelin farklı kelimelere gitmesini istiyoruz ancak olasılığı çok çok düşük saçma sapan kelimeleri de engellememiz gerek.
- Eğer **Top-K = 50** dersek: En yüksek olasılıklı 50 kelime hariç geri kalan 50.200 kelimenin olasılığını `-Infinity` yapar.
- Softmax sonucunda o diğer gereksiz ihtimallerin şansı matematiken 0 olur. Modele "Bir çeşitlilik yarat ama sadece en tutarlı bu 50 seçenek arasından kura çekerek (Multinomial) yap" demiş oluruz.

## 5. Model Ağırlıklarını Kaydetme ve OpenAI (GPT-2) Yükleme
Ev bilgisayarlarında küçük veri setleriyle bir zeka oluşturmak imkansıza yakındır. Dev modeller şirketlerin dev GPU çiftliklerinde aylarca eğitilir. 
- *Ağırlıkları kaydetme:* O anki nöron bağlarımızı `torch.save(model.state_dict(), "model.pth")` ile kaydederiz. (Adam optimizer state de kaydedilebilir). 
- *OpenAI Ağırlıklarını Çekme:* İşi baştan yapmak yerine, OpenAI şirketinin eğittiği *GPT2 (124 Milyar parametre) / Medium / Large veya XL* versiyonlarının ağırlıklarını indirebiliriz. 
- Çekilen bu Tensör N-boyutlu matrislerini, Bölüm 4'te yazdığımız Nöronlarımızın üzerine dikkatlice tek tek yerleştiririz (Ağırlıkları transfer edersek). Böylece sıfırdan kurduğumuz model, birden OpenAI'ın orijinal zekasına kavuşmuş halde düzgün, akıcı ve grameri iyi cümleler üretmeye başlar.
