# Bölüm 5.5: OpenAI'den Önceden Eğitilmiş Ağırlıkları Yükleme

## Genel Bakış

Bu bölümde, sıfırdan eğitmek yerine OpenAI'nin **GPT-2** modelinin önceden eğitilmiş ağırlıklarını yüklüyoruz. Bu sayede:

- Yüzbinlerce dolarlık eğitim maliyetinden kaçınıyoruz
- Anında **anlamlı metin üretebilen** bir model elde ediyoruz

> **Önemli Not:** TensorFlow uyumluluk sorunları yaşayanlar için alternatif bir yöntem de mevcut

---

## Adımlar

### Adım 1: Gerekli Kütüphanelerin Kurulumu

```python
pip install tensorflow tqdm
```

- **TensorFlow**: OpenAI, GPT-2 ağırlıklarını TensorFlow formatında yayınladı
- **tqdm**: İndirme işlemi için ilerleme çubuğu

### Adım 2: GPT-2 Ağırlıklarını İndirme

```python
from gpt_download import download_and_load_gpt2

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

Bu fonksiyon:
- `gpt2` klasörüne model dosyalarını indirir
- **124M, 355M, 774M, 1558M** gibi farklı boyutlardan seçim yapılabilir

İndirilen dosyalar:

| Dosya | Boyut | Açıklama |
|-------|-------|-----------|
| checkpoint | 77 KB | Model kontrol noktası |
| encoder.json | 1.04 MB | Tokenizer encoder |
| hparams.json | 90 KB | Hiperparametreler |
| model.ckpt.data | 498 MB | **Ana ağırlıklar** |
| vocab.bpe | 456 KB | BPE vocabulary |

### Adım 3: Settings (Hiperparametreler)

```python
print("Settings:", settings)
# Çıktı: {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}
```

- **n_vocab**: 50,257 kelime vocabulary
- **n_ctx**: 1024 token context length
- **n_embd**: 768 embedding boyutu
- **n_head**: 12 attention head
- **n_layer**: 12 transformer katmanı

### Adım 4: Parametre Sözlüğü Yapısı

```python
print("Parameter dictionary keys:", params.keys())
# dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])
```

| Anahtar | Açıklama |
|---------|-----------|
| `blocks` | 12 transformer bloğu |
| `b` | Final layer norm bias |
| `g` | Final layer norm gamma (scale) |
| `wpe` | Position embedding ağırlıkları |
| `wte` | Token embedding ağırlıkları |

### Adım 5: Model Konfigürasyonu

Farklı GPT-2 boyutları:

| Model | Parametre | Embedding | Layers | Heads |
|-------|-----------|-----------|--------|-------|
| small | 124M | 768 | 12 | 12 |
| medium | 355M | 1024 | 24 | 16 |
| large | 774M | 1280 | 36 | 20 |
| xl | 1558M | 1600 | 48 | 25 |

```python
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs["gpt2-small (124M)"])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})
```

> **Not:** OpenAI orijinal modelde `qkv_bias` kullandığı için `qkv_bias: True` ayarlanmalı

### Adım 6: Ağırlıkları GPTModel'e Aktarma

```python
def load_weights_into_gpt(gpt, params):
    # Position ve Token Embeddings
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        # Attention: Query, Key, Value ağırlıkları
        q_w, k_w, v_w = np.split(
            params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )
        
        # Attention output projection
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        
        # Feed Forward Network
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        
        # Layer Norm
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"]
        )
    
    # Final LayerNorm ve Output Head
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params['g'])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params['b'])
    gpt.out_head.weight = assign(gpt.out_head.weight, params['wte'])
```

> **Önemli:** TensorFlow ağırlıkları PyTorch'a **transpose** edilerek aktarılmalı (`.T`)

### Adım 7: Model ile Metin Üretme

```python
torch.manual_seed(123)

token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```

**Örnek Çıktı:**
```
Every effort moves you as far as the hand can go until the end of your turn unless something interrupts your control flow. As you may observe I
```

---

## Alternatif Yöntem: Hugging Face Üzerinden PyTorch Ağırlıkları

TensorFlow sorunları yaşayanlar için, önceden PyTorch'a dönüştürülmüş ağırlıkları Hugging Face'den indirebilirsiniz:

```python
file_name = "gpt2-small-124M.pth"
# file_name = "gpt2-medium-355M.pth"
# file_name = "gpt2-large-774M.pth"
# file_name = "gpt2-xl-1558M.pth"

url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"

if not os.path.exists(file_name):
    urllib.request.urlretrieve(url, file_name)
    print(f"Downloaded to {file_name}")

gpt = GPTModel(BASE_CONFIG)
gpt.load_state_dict(torch.load(file_name, weights_only=True))
gpt.eval()
```

---

## Özet

1. **TensorFlow** ile OpenAI'nin GPT-2 ağırlıklarını indiriyoruz
2. Parametreleri **PyTorch formatına dönüştürüyoruz** (transpose vb.)
3. Kendi `GPTModel` yapımıza yüklüyoruz
4. Artık **anlamlı metin üretebilen** bir modele sahibiz!

Bu, sıfırdan eğitimden çok daha iyi sonuçlar verir çünkü model milyarlarca token üzerinde önceden eğitilmiş.

---

## Kaynaklar

- [LLMs from Scratch GitHub](https://github.com/rasbt/LLMs-from-scratch)
- [OpenAI GPT-2](https://openai.com/research/gpt-2)
- [Hugging Face Model](https://huggingface.co/rasbt/gpt2-from-scratch-pytorch)
