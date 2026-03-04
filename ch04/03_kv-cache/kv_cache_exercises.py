"""
KV Cache & Sliding Window Attention
Egzersiz Notebook'u - Python Script Versiyonu
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("KV CACHE & SLIDING WINDOW ATTENTION")
print("=" * 60)

# =============================================================================
# BÖLÜM 1: KV CACHE TEMELLERİ
# =============================================================================

print("\n" + "=" * 60)
print("BÖLÜM 1: KV CACHE TEMELLERİ")
print("=" * 60)

# -----------------------------------------------------------------------------
# 1.1 Query, Key, Value Nedir?
# -----------------------------------------------------------------------------

print("\n--- 1.1 Query, Key, Value ---")

# Basit bir örnek oluşturalım
# 3 token var: ["The", "cat", "sat"]

# Her token için basit embedding (gerçek modelde learnable)
embeddings = torch.tensor(
    [
        [1.0, 0.0, 0.0],  # "The"
        [0.0, 1.0, 0.0],  # "cat"
        [0.0, 0.0, 1.0],  # "sat"
    ]
)

print("Token Embeddings (3 token, 3 boyut):")
print(f"Shape: {embeddings.shape}")
print(embeddings)

# -----------------------------------------------------------------------------
# 1.2 Q, K, V Hesaplama
# -----------------------------------------------------------------------------

print("\n--- 1.2 Q, K, V Hesaplama ---")

# W_query, W_key, W_value ağırlıkları (basit örnek için identity matrix)
W_query = torch.eye(3)
W_key = torch.eye(3)
W_value = torch.eye(3)

# Q, K, V hesapla
Q = embeddings @ W_query.T  # (3, 3)
K = embeddings @ W_key.T
V = embeddings @ W_value.T

print("Query (Q):")
print(Q)
print("\nKey (K):")
print(K)
print("\nValue (V):")
print(V)

# -----------------------------------------------------------------------------
# 1.3 Attention Score Hesaplama
# -----------------------------------------------------------------------------

print("\n--- 1.3 Attention Score Hesaplama ---")

# Attention scores: Q @ K.T
attn_scores = Q @ K.T  # (3, 3)

print("Attention Scores (Q @ K.T):")
print(attn_scores)
print("\nBu matris neyi gösteriyor?")
print("Satır: Hangi Query")
print("Sütun: Hangi Key ile ilişki")
print(f"\nÖrn: attn_scores[1,2] = {attn_scores[1, 2].item()}")
print("Bu: 'cat' query'sinin 'sat' key ile ilişkisi")

# -----------------------------------------------------------------------------
# 1.4 Causal Mask
# -----------------------------------------------------------------------------

print("\n--- 1.4 Causal Mask ---")

# Causal mask: Üst üçgeni -inf yap
seq_len = 3
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

print("Causal Mask (True = gizle):")
print(causal_mask)

# Mask'i uygula
attn_scores_masked = attn_scores.masked_fill(causal_mask, float("-inf"))

print("\nMasked Attention Scores:")
print(attn_scores_masked)

# -----------------------------------------------------------------------------
# 1.5 Softmax ve Context Vector
# -----------------------------------------------------------------------------

print("\n--- 1.5 Softmax ve Context Vector ---")

# Softmax ile olasılık dağılımı
attn_weights = torch.softmax(attn_scores_masked, dim=-1)

print("Attention Weights (Olasılıklar):")
print(attn_weights)
print("\nHer satır toplamı 1 olmalı:")
print(attn_weights.sum(dim=-1))

# Context vector: attention @ value
context = attn_weights @ V

print("\nContext Vectors:")
print(context)

# =============================================================================
# BÖLÜM 2: KV CACHE - CACHE OLMADAN VS CACHE İLE
# =============================================================================

print("\n" + "=" * 60)
print("BÖLÜM 2: KV CACHE - CACHE OLMADAN VS CACHE İLE")
print("=" * 60)

# -----------------------------------------------------------------------------
# 2.1 Cache Olmadan (Naive)
# -----------------------------------------------------------------------------

print("\n--- 2.1 Cache Olmadan ---")

"""
Cache OLMADAN:
- Her yeni token için TÜM sekans tekrar işlenir
- K ve V yeniden hesaplanır
"""

print("=== CACHE OLMADAN ===\n")

num_layers = 2  # Basitlik için 2 layer
total_steps = 5

for step in range(total_steps):
    # Her adımda tüm önceki token'lar + yeni token
    current_seq_len = step + 1

    # K ve V hesapla (tüm sekans için)
    k_computed = current_seq_len * num_layers  # Her layer'da K hesaplanır
    v_computed = current_seq_len * num_layers

    print(f"Step {step + 1}: seq_len={current_seq_len}, K hesaplanan: {k_computed}, V hesaplanan: {v_computed}")

# -----------------------------------------------------------------------------
# 2.2 Cache İle (KV Cache)
# -----------------------------------------------------------------------------

print("\n--- 2.2 Cache İle ---")

"""
Cache İLE:
- İlk prompt için K,V hesapla ve cache'e kaydet
- Sonraki adımlarda sadece yeni token için Q hesapla
- K ve V cache'den al
"""

print("=== KV CACHE İLE ===\n")

prompt_len = 3  # Prompt: 3 token
generate_len = 5  # 5 token üretilecek
num_layers = 2

print(f"1. Prompt işleniyor ({prompt_len} token):")
k_cache = prompt_len * num_layers
v_cache = prompt_len * num_layers
print(f"   K hesaplandı: {k_cache}, V hesaplandı: {v_cache}")
print(f"   Cache'e kaydedildi: {k_cache} K, {v_cache} V\n")

total_k_computed = k_cache
total_v_computed = v_cache

print("2. Token üretimi:")
for step in range(generate_len):
    # Sadece Q hesapla (1 token için)
    q_computed = 1 * num_layers
    # K ve V cache'den al (hesaplanmadı!)
    k_from_cache = prompt_len + step
    v_from_cache = prompt_len + step

    total_k_computed += 0  # Cache'den alındı
    total_v_computed += 0

    print(f"   Step {step + 1}: Q={q_computed}, K=cached({k_from_cache}), V=cached({v_from_cache})")

print(f"\nToplam K hesaplama: {total_k_computed}")
print("(Cache olsaydı: 3 + 1 + 1 + 1 + 1 + 1 = 8 olurdu)")

# -----------------------------------------------------------------------------
# 2.3 Karşılaştırma
# -----------------------------------------------------------------------------

print("\n--- 2.3 Karşılaştırma ---")

# Karşılaştırma grafiği
steps = list(range(1, 11))

# Cache yok: Her adımda tüm sekans
no_cache = [i * 2 for i in steps]  # 2 layer * i token

# Cache var: İlk = i, sonra = 1
with_cache = [3 * 2 + (i - 1) * 2 for i in steps]  # İlk prompt + (i-1) * 1

plt.figure(figsize=(10, 5))
plt.plot(steps, no_cache, "r-o", label="Cache Yok (Her adımda tüm sekans)")
plt.plot(steps, with_cache, "g-o", label="KV Cache (Sadece yeni token)")
plt.xlabel("Üretilen Token Sayısı")
plt.ylabel("K/V Hesaplama Sayısı")
plt.title("KV Cache Performans Karşılaştırması")
plt.legend()
plt.grid(True)
plt.show()

print("Görüldüğü gibi cache ile hesaplama sayısı çok daha az!")

# =============================================================================
# BÖLÜM 3: SLIDING WINDOW ATTENTION
# =============================================================================

print("\n" + "=" * 60)
print("BÖLÜM 3: SLIDING WINDOW ATTENTION")
print("=" * 60)

# -----------------------------------------------------------------------------
# 3.1 Problem: Çok Uzun Sekanslar
# -----------------------------------------------------------------------------

print("\n--- 3.1 Problem: Çok Uzun Sekanslar ---")

# Sliding Window kavramını göster
window_size = 5

print(f"Window Size: {window_size} token\n")

# Her adımda cache'i simüle et
cache_state = []

for step in range(10):
    new_token = f"T{step}"

    # Yeni token ekle
    cache_state.append(new_token)

    # Window size'ı aştıysa eskileri at
    if len(cache_state) > window_size:
        removed = cache_state.pop(0)

    print(f"Step {step + 1}: Cache = {cache_state}")
    print(f"         Length: {len(cache_state)}")

# -----------------------------------------------------------------------------
# 3.2 Kod: Sliding Window Cache Implementasyonu
# -----------------------------------------------------------------------------

print("\n--- 3.2 Sliding Window Cache Implementasyonu ---")


class SlidingWindowKVCache:
    """Sliding Window KV Cache Implementasyonu"""

    def __init__(self, window_size=5, num_heads=2, head_dim=4):
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Pre-allocated cache
        self.cache_k = torch.zeros(window_size, num_heads, head_dim)
        self.cache_v = torch.zeros(window_size, num_heads, head_dim)
        self.ptr_cur = 0  # Sıradaki boş pozisyon

    def add(self, keys_new, values_new):
        """Yeni K, V ekle"""
        num_tokens = keys_new.shape[0]

        # Overflow kontrolü
        if self.ptr_cur + num_tokens > self.window_size:
            # Taşma miktarı
            overflow = self.ptr_cur + num_tokens - self.window_size

            # Sola kaydır (shift)
            self.cache_k[:-overflow] = self.cache_k[overflow:].clone()
            self.cache_v[:-overflow] = self.cache_v[overflow:].clone()
            self.ptr_cur -= overflow

            print(f"   ↩️  Overflow! {overflow} token kaydırıldı")

        # Yeni K, V'yi cache'e yaz
        self.cache_k[self.ptr_cur : self.ptr_cur + num_tokens] = keys_new
        self.cache_v[self.ptr_cur : self.ptr_cur + num_tokens] = values_new
        self.ptr_cur += num_tokens

        return self.cache_k[: self.ptr_cur], self.cache_v[: self.ptr_cur]

    def reset(self):
        """Cache'i sıfırla"""
        self.ptr_cur = 0

    def __repr__(self):
        return f"SlidingWindowKVCache(ptr={self.ptr_cur}/{self.window_size})"


# Test
cache = SlidingWindowKVCache(window_size=5, num_heads=1, head_dim=3)

print("\n=== Sliding Window Cache Test ===\n")

for i in range(8):
    # Her step için 1 token (basit örnek)
    key = torch.ones(1, 1, 3) * (i + 1)  # Farklı değerler
    value = torch.ones(1, 1, 3) * (i + 1)

    print(f"Step {i + 1}: Yeni token eklenecek")
    k, v = cache.add(key, value)
    print(f"   {cache}")
    print(f"   Cache K değerleri: {k.squeeze().tolist()}")
    print()

# -----------------------------------------------------------------------------
# 3.3 Pointer (ptr_cur) Açıklaması
# -----------------------------------------------------------------------------

print("\n--- 3.3 Pointer Mekanizması ---")

# Pointer mekanizmasını görselleştir

window_size = 5

print("=== Pointer Mekanizması ===\n")

stages = [
    ("Başlangıç", 0, None),
    ("3 token eklendi", 3, ["K0", "K1", "K2"]),
    ("5 token eklendi (dolu)", 5, ["K0", "K1", "K2", "K3", "K4"]),
    ("Overflow! 6. token (shift)", 5, ["K1", "K2", "K3", "K4", "K5"]),
    ("Overflow! 7. token (shift)", 5, ["K2", "K3", "K4", "K5", "K6"]),
]

for stage_name, ptr_val, cache_content in stages:
    print(f"{stage_name}:")
    print(f"  ptr_cur = {ptr_val}")

    # Cache görselleştirme
    cache_viz = ["_"] * window_size
    if cache_content:
        for i, c in enumerate(cache_content):
            if i < window_size:
                cache_viz[i] = c

    print(f"  Cache: [{' | '.join(cache_viz)}]")
    print()

# =============================================================================
# BÖLÜM 4: MASK HESAPLAMA
# =============================================================================

print("\n" + "=" * 60)
print("BÖLÜM 4: MASK HESAPLAMA")
print("=" * 60)

# -----------------------------------------------------------------------------
# 4.1 Problem: Offset
# -----------------------------------------------------------------------------

print("\n--- 4.1 Problem: Offset ---")


def create_causal_mask(num_tokens_q, num_tokens_k, offset=0):
    """
    Causal mask oluştur

    Args:
        num_tokens_q: Query token sayısı
        num_tokens_k: Key token sayısı
        offset: Cache'deki mevcut token sayısı
    """
    # Satır indeksleri
    row_idx = torch.arange(num_tokens_q).unsqueeze(1)  # (Q, 1)
    # Sütun indeksleri
    col_idx = torch.arange(num_tokens_k).unsqueeze(0)  # (1, K)

    # Offset ekle
    # Mask: j > i + offset ise True (gizle)
    mask = row_idx + offset < col_idx

    return mask


print("=== Offset ile Mask Hesaplama ===\n")

# Durum 1: Cache yok (offset = 0)
print("Durum 1: 3 token, cache yok")
mask1 = create_causal_mask(3, 3, offset=0)
print(mask1)

# Durum 2: Cache'de 4 token var, 1 yeni token
print("\nDurum 2: Q=1 token, K=5 token (4 cache + 1 yeni), offset=4")
mask2 = create_causal_mask(1, 5, offset=4)
print(mask2)

# Durum 3: Cache'de 5 token var, 1 yeni token
print("\nDurum 3: Q=1 token, K=6 token (5 cache + 1 yeni), offset=5")
mask3 = create_causal_mask(1, 6, offset=5)
print(mask3)

# =============================================================================
# BÖLÜM 5: EGZERSİZLER
# =============================================================================

print("\n" + "=" * 60)
print("BÖLÜM 5: EGZERSİZLER")
print("=" * 60)

# -----------------------------------------------------------------------------
# Egzersiz 1: Basic KV Cache
# -----------------------------------------------------------------------------

print("\n" + "-" * 40)
print("EGZERSİZ 1: Basic KV Cache")
print("-" * 40)


# Çözüm
class BasicKVCache:
    """Basic KV Cache"""

    def __init__(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, keys_new, values_new, use_cache=False):
        """
        Args:
            keys_new: Yeni key değerleri
            values_new: Yeni value değerleri
            use_cache: Cache kullanılsın mı?
        """
        if use_cache:
            # İlk çağrı mı?
            if self.cache_k is None:
                # Cache'i keys_new ile başlat
                self.cache_k = keys_new
                self.cache_v = values_new
            else:
                # Cache'e yeni değerleri ekle
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)

            keys = self.cache_k
            values = self.cache_v
        else:
            keys = keys_new
            values = values_new

        return keys, values

    def reset(self):
        """Cache'i sıfırla"""
        self.cache_k = None
        self.cache_v = None


# Test
cache = BasicKVCache()

# 3 token'lık input
keys = torch.randn(1, 3, 8)  # (batch, tokens, dim)
values = torch.randn(1, 3, 8)

# Cache ile forward
k, v = cache.forward(keys, values, use_cache=True)
print(f"İlk çağrı: k.shape = {k.shape}")

# 1 token daha ekle
new_key = torch.randn(1, 1, 8)
new_value = torch.randn(1, 1, 8)

k2, v2 = cache.forward(new_key, new_value, use_cache=True)
print(f"İkinci çağrı: k2.shape = {k2.shape}")

print("\n✅ Egzersiz 1 tamamlandı!" if k2.shape[1] == 4 else "❌ Hata var!")

# -----------------------------------------------------------------------------
# Egzersiz 2: Sliding Window Overflow
# -----------------------------------------------------------------------------

print("\n" + "-" * 40)
print("EGZERSİZ 2: Sliding Window Overflow")
print("-" * 40)


# Çözüm
class SlidingWindowKVCacheExercise:
    """Sliding Window Cache - Overflow handling"""

    def __init__(self, window_size=4):
        self.window_size = window_size
        self.cache_k = torch.zeros(window_size, 4)  # window_size x dim
        self.cache_v = torch.zeros(window_size, 4)
        self.ptr_cur = 0

    def add_with_overflow(self, keys_new, values_new):
        """
        Yeni K, V ekle + overflow handling

        keys_new: (num_tokens, dim)
        values_new: (num_tokens, dim)
        """
        num_tokens = keys_new.shape[0]

        # Overflow kontrolü
        if self.ptr_cur + num_tokens > self.window_size:
            # Taşma miktarını hesapla
            overflow = self.ptr_cur + num_tokens - self.window_size

            # Cache'i sola kaydır
            self.cache_k[:-overflow] = self.cache_k[overflow:].clone()
            self.cache_v[:-overflow] = self.cache_v[overflow:].clone()

            # Pointer'ı güncelle
            self.ptr_cur -= overflow
            print(f"   ↩️  Overflow! {overflow} token atıldı")

        # Yeni değerleri cache'e yaz
        self.cache_k[self.ptr_cur : self.ptr_cur + num_tokens] = keys_new
        self.cache_v[self.ptr_cur : self.ptr_cur + num_tokens] = values_new

        # Pointer'ı ilerlet
        self.ptr_cur += num_tokens

        return self.cache_k[: self.ptr_cur], self.cache_v[: self.ptr_cur]


# Test
cache = SlidingWindowKVCacheExercise(window_size=4)

print("\n=== Sliding Window Overflow Test ===\n")

for i in range(6):
    key = torch.tensor([i * 10, i * 10 + 1, i * 10 + 2, i * 10 + 3]).float().unsqueeze(0)
    value = key.clone()

    print(f"Step {i + 1}: Yeni token ekleniyor")
    k, v = cache.add_with_overflow(key, value)
    print(f"   ptr: {cache.ptr_cur}, Cache: {k.squeeze().tolist()}")
    print()

print("✅ Egzersiz 2 tamamlandı!")

# -----------------------------------------------------------------------------
# Egzersiz 3: Window Size Karşılaştırması
# -----------------------------------------------------------------------------

print("\n" + "-" * 40)
print("EGZERSİZ 3: Window Size Karşılaştırması")
print("-" * 40)


def simulate_window(window_size, total_tokens):
    """Verilen window size ile cache simülasyonu"""
    cache_state = []

    for i in range(total_tokens):
        cache_state.append(i)

        if len(cache_state) > window_size:
            cache_state.pop(0)

    return cache_state


# Farklı window size'larda karşılaştır
total_tokens = 20
window_sizes = [3, 5, 10, 20]  # Sonsuz

plt.figure(figsize=(12, 6))

for ws in window_sizes:
    cache_state = simulate_window(ws, total_tokens)
    plt.plot(range(total_tokens), cache_state, label=f"Window={ws}")

plt.xlabel("Token Pozisyonu")
plt.ylabel("Cache deki token indeksi")
plt.title("Farklı Window Size'ların Karşılaştırması")
plt.legend()
plt.grid(True)
plt.show()

print("\n=== SONUÇ ===")
print("Window size küçükse: Eski token'lar çabuk unutulur")
print("Window size büyükse: Daha fazla bağlam korunur")
print("\nBu, modelin 'uzun vadeli hafızasını' belirler!")

# -----------------------------------------------------------------------------
# Egzersiz 4 (Bonus): Gerçek Attention + Mask
# -----------------------------------------------------------------------------

print("\n" + "-" * 40)
print("EGZERSİZ 4 (Bonus): Attention + Mask")
print("-" * 40)


def attention_with_cache_and_mask(queries, keys, values, use_cache=False, offset=0):
    """
    Attention hesapla + causal mask

    queries: (batch, heads, Q_tokens, head_dim)
    keys:    (batch, heads, K_tokens, head_dim)
    values:  (batch, heads, V_tokens, head_dim)
    offset:  Cache'deki mevcut token sayısı
    """
    # 1. Attention scores
    attn_scores = queries @ keys.transpose(-2, -1)

    # 2. Boyutları al
    num_q = queries.shape[-2]
    num_k = keys.shape[-2]

    # 3. Mask oluştur
    row_idx = torch.arange(num_q).unsqueeze(1)
    col_idx = torch.arange(num_k).unsqueeze(0)

    # Offset kullanarak mask oluştur
    mask = row_idx + offset < col_idx

    # 4. Mask'i uygula
    attn_scores = attn_scores.masked_fill(mask, float("-inf"))

    # 5. Softmax
    attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)

    # 6. Context
    context = attn_weights @ values

    return context


# Test
print("\n=== Attention + Cache + Mask Test ===\n")

# Durum 1: Cache yok (3 token)
Q = torch.randn(1, 1, 3, 8)
K = torch.randn(1, 1, 3, 8)
V = torch.randn(1, 1, 3, 8)

context1 = attention_with_cache_and_mask(Q, K, V, use_cache=False, offset=0)
print(f"Durum 1 (Cache yok): {context1.shape}")

# Durum 2: Cache'de 4 token var, 1 yeni token (Q=1, K=5)
Q = torch.randn(1, 1, 1, 8)  # 1 query
K = torch.randn(1, 1, 5, 8)  # 5 keys (4 cache + 1 new)
V = torch.randn(1, 1, 5, 8)  # 5 values

context2 = attention_with_cache_and_mask(Q, K, V, use_cache=True, offset=4)
print(f"Durum 2 (Cache var, offset=4): {context2.shape}")

print("\n✅ Egzersiz 4 tamamlandı!")

# =============================================================================
# ÖZET
# =============================================================================

print("\n" + "=" * 60)
print("ÖZET")
print("=" * 60)

print("""
Bu script'te öğrendiklerimiz:

1. KV Cache: K ve V değerlerini saklayarak tekrar hesaplamayı önleme

2. Sliding Window: Bellek sınırlı tutmak için eski token'ları atma

3. Pointer (ptr_cur): Cache'in mevcut pozisyonunu takip etme

4. Mask Hesaplama: Offset kullanarak causal mask'i dinamik oluşturma

=== Anahtar Kavramlar ===

| Kavram     | Açıklama                          |
|------------|----------------------------------|
| Query      | "Ne arıyorum?"                   |
| Key        | "Bu pozisyonda ne var?"          |
| Value      | "Bu bilginin değeri"             |
| Cache      | K ve V'yi saklama                |
| Sliding Window | Sınırlı cache boyutu         |
| Offset     | Cache'deki token sayısı         |
| ptr_cur    | Sıradaki boş slot                |

""")

print("=" * 60)
print("Script'i başarıyla tamamladınız! 🎉")
print("=" * 60)
