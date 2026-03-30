# Derin Öğrenme İşlem Akışı: Forward Pass, Loss, Backpropagation ve Optimizer

Bu doküman, LLM eğitiminde gerçekleşen temel işlem adımlarını açıklar.

---

## 1) İleri Yayılım (Forward Pass): Aktivasyonlar Nerede Çalışıyor?

`loss = calc_loss_batch(input_batch, target_batch, model, device)` fonksiyonu çağrıldığında ilk çalışan şey **İleri Yayılımdır (Forward Pass)**. Tensörler ağ içine girer.

### Aktivasyon Fonksiyonları (GELU, Softmax vb.)
- **Ne işe yararlar?** Girdiler Linear katmanlardan (matris çarpımları) çıktıktan sonra, sonucu bir şekle sokup doğrusal olmayan kalıpları ("non-linear" özellikleri) öğrenmesini sağlarlar.
- **Nerede Çalışırlar?** MultiHeadAttention'dan (Dikkat Mekanizması) sonra devreye girerler. Cümlelerin mantıksal soyutlamalarını çıkartıp bir sonraki bloğa iletirler.
- **Kim Hesaplıyor?** PyTorch motoru (ön planda PyTorch, arka planda C++ ve CUDA GPU çekirdekleri) anlık olarak tensörler içinden geçirirken aktivasyonların matematiksel denklemlerini hesaplar.

---

## 2) Hata (Loss) Hesaplanması

Girdiler en son katmana (Linear Head'e) geldi. Örneğin, spam/not spam sınıflandırması için iki tane oran üretti.

- **Ne Yapılıyor?** Üretilen oran (`logits`) ile Doğru Etiket (Ground Truth, `target_batch`) karşılaştırılıyor.
- **Kim Hesaplıyor?** **Cross Entropy** formülü hesaplıyor. Sonuç olarak bir skaler değer ("Ne kadar hatalıyız?") üretiyor.

### Cross-Entropy Loss Formülü
```
Loss = -log(doğru_kelime_olasılığı)
```

---

## 3) Gradyanlar (Türevler) Nasıl ve Nerede Hesaplanıyor?

Hemen ardından kodda `loss.backward()` fonksiyonu tetiklenir. Buna **Geri Yayılım (Backpropagation)** denir.

### Gradyan (Gradient) Nedir?
"Bu hatayı (loss) küçültmek için modeldeki yüzlerce ağırlığı (weight/bias) hangi yöne, ne kadar miktar kaydırmalıyım?" sorusunun Türev ile hesaplanmış "yön vektörü"dür.

### Kim Hesaplıyor?
PyTorch'un `Autograd` (Otomatik Türev) motoru. Model eğitilirken (İleri yayılımda) PyTorch gizlice bir hesaplama grafiği (Computational Graph) oluşturur. `backward()` dediğimiz an, bu grafik zincir kuralı (chain rule) ile baştan sona (sondan başa doğru) otomatikman türevlenir.

### Nerede Çalışır?
Ağırlıkları güncelleyecek bilgiyi, dondurulmamış (unfreeze edilmiş) tüm katmanlardaki ağırlıkların o andaki `param.grad` isminde bir tampon hafızaya atar.

---

## 4) Optimizer (AdamW) Hangi Aşamada Çalışıyor?

`loss.backward()` işini bitirdi. Artık her ağırlığın *ne kadar güncelleneceğini içeren bir gradyan yönü* var. Ardından kodda şu satır çalışır:

```python
optimizer.step()
```

### Ne yapıyor?
Optimizer (bizdeki AdamW algoritması), modelin kapısını çalar, az önce Autograd'ın hesaplayıp hafızaya bıraktığı Gradyan (hata yön göstergesi) vektörüne bakar ve ilgili tüm parametrelerin (*Ağırlık ve Biasların*) gerçek değerlerini eksi yönde **fiziksel olarak günceller**.

Model o an "Eğitilmiş" ya da "Öğrenmiş" olur. Öğrenme Oranı (Learning rate) ve Weight Decay (L2 zayıflatması) kurallarını da güncellerken uygular.

---

## 5) Özet: İşlem Akışı (Fabrika Bandı)

```
┌─────────────────────────────────────────────────────────────┐
│  1. FORWARD PASS (İleri Yayılım)                          │
│     • Input → Linear → GELU → Linear → ... → Output        │
│     • PyTorch Autograd grafiği oluşturulur                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  2. LOSS HESAPLAMA                                         │
│     • CrossEntropy(logits, targets)                        │
│     • Skaler hata değeri üretilir                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  3. BACKWARD PASS (Geri Yayılım)                          │
│     • loss.backward()                                      │
│     • Zincir kuralı ile türevler hesaplanır                │
│     • Her param.grad içine gradyanlar yazılır             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  4. OPTIMIZER STEP                                        │
│     • optimizer.zero_grad()     → Eski gradyanları sil     │
│     • optimizer.step()          → Ağırlıkları güncelle     │
│     • Learning rate + Weight decay uygulanır               │
└─────────────────────────────────────────────────────────────┘
                           ↓
                   [Yeni Batch ile Tekrarla]
```

---

## Önemli Not

Bir döngü bittikten sonra yeni tensör grubu (batch) geldiğinde eski gradyanlar işleme karışmasın diye eğitimin başında `optimizer.zero_grad()` ile o hafızaya atılan türevler sıfırlanıp fabrika bandı tekrar başlatılır.
