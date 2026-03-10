# Bölüm 6: Sınıflandırma İçin İnce Ayar (Fine-Tuning) Özeti

Bu doküman, büyük dil modellerinin (LLM) metin sınıflandırma görevleri (örneğin, spam tespiti) için nasıl ince ayar (fine-tune) edildiğini özetlemektedir.

## 1. Veri Seti Hazırlığı (SpamDataset)

Modeli sınıflandırma görevi için eğitmeden önce verilerin hazırlanması gerekir:
- **Veri Seti:** SMS Spam Collection veri seti kullanılmıştır (Spam ve Not Spam/Ham).
- **Dengesizlik Giderme:** Orijinal veri setinde %87 ham, %13 spam mesajı bulunuyordu. Eğitimin daha dengeli olması için "ham" (spam olmayan) mesajlar altörnekleme (undersampling) yöntemiyle azaltılarak sınıflar eşitlenmiştir.
- **Etiketleme:** Sınıflar tam sayı etiketlere dönüştürülmüştür (`spam: 1`, `not spam: 0`).
- **Tokenizasyon ve Dolgulama (Padding):** Tüm mesajlar aynı uzunlukta olacak şekilde en uzun mesaja göre `<|endoftext|>` tokeni ile dolgulanmıştır (padding). Bu sayede veriler DataLoader ile toplu (batch) olarak işlenebilmektedir.

## 2. Model Mimarisindeki Değişiklikler (Bölüm 5'e Kadarki Hali vs. Bölüm 6)

### Bölüm 5'e Kadarki Hali (Üretken LLM)
Bölüm 5'e kadar geliştirdiğimiz GPT modeli, **sonraki kelimeyi (token) tahmin etme** üzerine eğitilmiş üretken bir modeldi:
- **Çıkış Katmanı (Output Head):** Vektör boyutundan (`emb_dim=768`) kelime dağarcığı boyutuna (`vocab_size=50257`) eşleme yapıyordu.
- **Odak Noktası:** Her bir giriş tokeni için karşılık gelen bir çıkış vektörü üretilir (hedef, bir sonraki tokeni bulmaktır).

### Bölüm 6 Hali (Sınıflandırıcı LLM)
Sınıflandırma görevinde artık yeni tokenler üretmek yerine, metnin tamamını değerlendirip bir sınıf etiketi (spam veya değil) vermesini istiyoruz:
- **Yeni Çıkış Katmanı:** Modelin orijinal çıkış katmanı silinip yerine sadece sınıf sayısı kadar (`num_classes=2`) çıkış veren yeni bir `Linear` katman eklendi: `(emb_dim=768) -> (num_classes=2)`.
- **Dondurma (Freezing):** Önceden eğitilmiş ağırlıkların bozulmaması ve eğitimin hızlanması için gövde katmanlarındaki parametreler donduruldu (`requires_grad = False`).
- **Eğitilebilir Katmanlar:** Sadece yeni eklenen son çıkış katmanı (`out_head`), son Transformer bloğu (`trf_blocks[-1]`) ve son katman normalizasyonu (`final_norm`) eğitilebilir bırakıldı (`requires_grad = True`).
- **Son Token Stratejisi:** Öz-dikkat (Self-Attention) katmanında nedensel maske (causal mask) kullanıldığı için, dizideki **en son token**, kendisinden önceki tüm tokenlerin bağlam bilgisini içerir. Sınıflandırma kararını vermek için tüm tokenlerin çıkışı yerine sadece dizinin **en son tokeninin** çıkışı (`outputs[:, -1, :]`) kullanılır.

## 3. Kayıp (Loss) ve Doğruluk (Accuracy) Hesaplama

- **Doğruluk:** Modelin ürettiği çıkışlardan (logits) en yüksek değerli olanın indeksi `argmax` ile seçilir. Eğer bu indeks, gerçek etiket ile eşleşiyorsa tahmin doğru kabul edilir. Toplam doğru tahminlerin toplam örneklere bölümü ile doğruluk yüzdesi bulunur.
- **Kayıp Fonksiyonu:** Doğruluk (accuracy) diferansiyellenebilir bir fonksiyon olmadığı için, onun yerine sinir ağlarında sınıflandırma görevleri için standart olan **Çapraz Entropi Kaybı (Cross-Entropy Loss)** minimize edilir. Girdi olarak sadece dizinin en son tokeninin logit vektörü verilir.

## 4. İnce Ayar Eğitimi (Finetuning Loop)

Sınıflandırma eğitim döngüsü, Bölüm 5'teki ön eğitim (pretraining) döngüsüne çok benzerdir. Sadece şu farklılıklar vardır:
- Modeldeki ağırlıkların tamamı yerine kısmi bir kümesi (son katmanlar) eğitilir.
- Döngü her `adımda (step)` ağa batch'ler halinde veriyi besler ve hata gradyanlarını hesaplayıp `AdamW` optimizasyon algoritması ile ağırlıkları günceller.
- İlerlemeyi görmek için belirli aralıklarla `evaluate_model` fonksiyonu kullanılarak hem Eğitim (Train) hem de Doğrulama (Validation) kümeleri üzerindeki Hata (Loss) ve Doğruluk (Accuracy) değerleri hesaplanıp takip edilir.

Son aşamada, başarıyla eğitilen bu sınıflandırıcı model ağırlıkları `.pth` dosyası olarak kaydedilebilir ve başka uygulamalarda tekrar kullanılabilir.
