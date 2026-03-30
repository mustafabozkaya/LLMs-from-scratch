---
language: en
license: mit
tags:
- text-classification
- spam-detection
- gpt2
- pytorch
- llm-from-scratch
datasets:
- sms-spam-collection
metrics:
- accuracy
model-index:
- name: review_classifier
  results:
  - task:
      type: text-classification
      name: Spam Detection
    dataset:
      name: SMS Spam Collection
      type: sms-spam-collection
    metrics:
    - type: accuracy
      value: 95.67
      name: Test Accuracy
---

# Review Classifier (Spam Detection GPT-2)

Bu model, **"Build a Large Language Model (From Scratch)"** kitabının 6. bölümü kapsamında, GPT-2 (124M) mimarisi üzerine inşa edilmiş bir SMS spam sınıflandırıcısıdır. Model, metinlerin "spam" (istenmeyen) veya "not spam" (ham) olup olmadığını yüksek doğrulukla tespit etmek için eğitilmiştir.

## Model Detayları

- **Model Adı:** review_classifier
- **Temel Mimari:** GPT-2 Small (124M parametre)
- **Görev:** İkili Sınıflandırma (Spam / Ham)
- **Girdi:** Metin (Maksimum 120 token)
- **Çıktı:** 0 (Ham), 1 (Spam)

## Veri Seti: SMS Spam Collection

Eğitimde UCI Machine Learning Repository'den alınan **SMS Spam Collection** veri seti kullanılmıştır.
- **Dengeleme:** Veri seti, 747 ham ve 747 spam mesajı içerecek şekilde dengelenmiştir (Toplam 1494 örnek).
- **Split:** %70 Eğitim, %10 Doğrulama, %20 Test.

## Eğitim Performansı

Model 5 epoch boyunca AdamW optimizer (LR=5e-5, Weight Decay=0.1) kullanılarak eğitilmiştir.

### Metrikler
| Split      | Accuracy (%) |
| ----------- | ----------- |
| **Training** | 97.21%      |
| **Validation**| 97.32%      |
| **Test**      | 95.67%      |

### Eğitim Kaybı ve Doğruluk Grafikleri
Model eğitimi sırasında elde edilen kayıp (loss) ve doğruluk (accuracy) grafikleri aşağıdadır:

**Loss Plot:**
![Training Loss](loss-plot.pdf)

**Accuracy Plot:**
![Accuracy](accuracy-plot.pdf)

*(Not: Grafikler eğitim sırasında otomatik olarak .pdf formatında kaydedilmektedir.)*

## Nasıl Kullanılır?

Modeli yüklemek ve tahmin yürütmek için aşağıdaki kod bloğunu kullanabilirsiniz:

```python
import torch
from model_architecture import GPTModel # Kitaptaki model tanımı

# Modelin yüklenmesi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpamClassifier() # Classification head eklenmiş hali
model.load_state_dict(torch.load("review_classifier.pth", map_location=device))
model.to(device)
model.eval()

# Tahmin Fonksiyonu
def predict(text):
    # Tokenization and forward pass...
    pass
```

## Kaynaklar
- Sebastian Raschka - [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- Veri Seti: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
