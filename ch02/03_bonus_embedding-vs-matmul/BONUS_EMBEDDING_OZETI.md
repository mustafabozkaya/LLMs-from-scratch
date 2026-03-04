# Bonus: Embedding Katmanı, One-Hot ve Lineer Katman Arasındaki İlişki

## Ana Fikir

Bu not defteri, PyTorch'taki `nn.Embedding` katmanının aslında ne olduğunu ve neden bu kadar verimli çalıştığını kanıtlamaktadır. Temel fikir şudur: Bir `nn.Embedding` katmanı, bir token'ın "one-hot" kodlanmış haline uygulanan bir `nn.Linear` katmanıyla matematiksel olarak tamamen aynı işi yapar, ancak bunu çok daha hızlı bir "arama" (lookup) işlemiyle gerçekleştirir.

### Kavramsal ve Matematiksel Kanıt

1. **Yöntem 1: One-Hot Encoding + Lineer Katman (Yavaş ve Verimsiz Yol)**

   * **Adım A: One-Hot Encoding:** Bir token ID'si (örneğin, `3`), `vocab_size` boyutunda bir vektöre dönüştürülür. Bu vektörün 3. indeksi `1`, geri kalan tüm elemanları `0` olur (`[0, 0, 0, 1, 0, ...]`).
   * **Adım B: Lineer Katman:** `nn.Linear(vocab_size, embedding_dim)` boyutunda bir lineer katman (aslında bir ağırlık matrisi) oluşturulur.
   * **Adım C: Matris Çarpımı:** One-hot vektörü, bu lineer katman ile matris çarpımına (`@`) sokulur. Bu çarpma işlemi, one-hot vektöründeki `1`'in olduğu pozisyona denk gelen matris satırını seçer ve diğer tüm satırlar sıfırla çarpıldığı için yok sayılır.
2. **Yöntem 2: `nn.Embedding` (Hızlı ve Verimli Yol)**

   * **Adım A: Arama Tablosu:** `nn.Embedding(vocab_size, embedding_dim)` katmanı, aslında `[vocab_size, embedding_dim]` boyutunda bir ağırlık matrisinden (arama tablosu) ibarettir.
   * **Adım B: Doğrudan Erişim (Lookup):** Katmana bir token ID'si (örneğin `3`) verildiğinde, matris çarpımı yapmak yerine doğrudan bu tablonun 3. satırını seçer ve döndürür.

### Teknik Uygulama (`embeddings-and-linear-layers.ipynb`)

* Not defteri, aynı başlangıç ağırlıklarına (`torch.manual_seed(123)`) sahip bir `nn.Embedding` katmanı ve bir `nn.Linear` katmanı oluşturur.
* `nn.Linear` katmanının ağırlık matrisinin, `nn.Embedding` katmanının ağırlık matrisinin **transpozesi** (`.T`) olduğu gösterilir.
* Aynı token ID'leri kullanılarak her iki yöntemle de embedding işlemi yapılır ve çıktılarının **birebir aynı** olduğu `torch.allclose()` ile kanıtlanır.

### Nihai Çıktı ve Temel Sonuç

* **Matematiksel Eşdeğerlik:** `nn.Embedding` katmanı, one-hot + lineer katman çarpımının optimize edilmiş, verimli bir kısayoludur.
* **Performans Farkı:** `nn.Embedding`'in doğrudan bir arama (lookup) işlemi yapması, onu büyük kelime dağarcıkları (`vocab_size`) ile çalışırken seyrek (sparse) one-hot vektörleriyle yapılan gereksiz sıfır çarpmalarından kurtarır. Bu da hem bellek kullanımını (VRAM) azaltır hem de hesaplama hızını büyük ölçüde artırır. Bu nedenle, pratikte her zaman `nn.Embedding` katmanı tercih edilir.
