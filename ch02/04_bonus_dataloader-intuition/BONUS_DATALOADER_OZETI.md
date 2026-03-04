# Bonus: DataLoader'ın Sezgisel Anlatımı

## Ana Fikir

Bu not defteri, `DataLoader`'ın ve `GPTDatasetV1` içinde kullanılan "kayan pencere" (sliding window) mekanizmasının nasıl çalıştığını, karmaşık metinler yerine basit bir sayı dizisi (`0, 1, 2, 3, ...`) kullanarak sezgisel bir şekilde açıklamayı amaçlar. Bu sayede, modelin "bir sonrakini tahmin etme" görevini öğrenmesi için verinin nasıl hazırlandığı netleşir.

### Temel Kavram: Kayan Pencere (Sliding Window)

`DataLoader`'ın kalbinde yatan veri hazırlama mantığı, ardışık bir diziden eğitim örnekleri oluşturmaktır.

*   **Girdi (Input - X):** Diziden `max_length` (pencere boyutu) kadar eleman alınır.
*   **Hedef (Target - Y):** Girdinin bir birim sağa kaydırılmış halidir.

**Örnek (`max_length=4`, `stride=1`):**

Sayı Dizimiz: `[0, 1, 2, 3, 4, 5, 6, ...]`

1.  **İlk Pencere:**
    *   Girdi (X): `[0, 1, 2, 3]`
    *   Hedef (Y): `[1, 2, 3, 4]`
2.  **Pencere 1 birim kaydırılır:**
    *   Girdi (X): `[1, 2, 3, 4]`
    *   Hedef (Y): `[2, 3, 4, 5]`
3.  **Pencere tekrar 1 birim kaydırılır:**
    *   Girdi (X): `[2, 3, 4, 5]`
    *   Hedef (Y): `[3, 4, 5, 6]`
    ... ve bu süreç tüm veri bitene kadar devam eder.

### Teknik Uygulama (`dataloader-intuition.ipynb`)

*   **Veri:** Token'lanmış metin yerine `0`'dan `1000`'e kadar olan sayıları içeren bir `number-data.txt` dosyası oluşturulur.
*   **`GPTDatasetV1` Modifikasyonu:** `tokenizer.encode()` satırı, dosyadan sayıları okuyan basit bir `[int(i) for i in txt.strip().split()]` satırıyla değiştirilir.
*   **`create_dataloader_v1`:**
    *   `stride`: Pencerenin her adımda kaç birim kayacağını belirler. `stride=1` maksimum örtüşme sağlarken, `stride=max_length` hiç örtüşme olmamasını sağlar.
    *   `shuffle`: `True` yapıldığında, oluşturulan (X, Y) çiftlerinin hangi sırayla modele verileceğini karıştırır. Bu, modelin ezberlemesini önlemek için standart bir tekniktir.
    *   `batch_size`: `DataLoader`'ın her seferinde kaç tane (X, Y) çiftini bir araya getirip tek bir tensör grubu olarak sunacağını belirler.

### Nihai Çıktı ve Temel Sonuç

Bu basit sayısal örnek, Büyük Dil Modelleri'nin eğitiminde kullanılan `DataLoader`'ın aslında ne kadar basit bir temel mantığa dayandığını gösterir. Karmaşık gibi görünen veri hazırlama süreci, özünde bir diziyi alıp, bir sonraki elemanı tahmin etme görevini oluşturmak için onu sistematik olarak kaydırmaktan ibarettir. Bu temel yapı, daha sonra metin verileri ve token'lar ile çalışmak için de aynen kullanılır.
