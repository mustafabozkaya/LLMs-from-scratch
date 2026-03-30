---
language: en
tags:
- pytorch
- text-classification
- spam-detection
- gpt2
- huggingface
- finetuning
datasets:
- sms-spam-collection
metrics:
- accuracy
pipeline_tag: text-classification
---
# GPT-2 Spam Detector (Fine-tuned with Hugging Face Transformers)

This is a fine-tuned GPT-2 model for spam detection.

## Model Overview

| Property          | Value                                         |
| ----------------- | --------------------------------------------- |
| Base Model        | openai-community/gpt2 (124M parameters)       |
| Task              | Binary Text Classification (Spam vs Not Spam) |
| Framework         | Hugging Face Transformers (PyTorch)           |
| Training Approach | Freezing + Selective Unfreezing |

## Architecture

1. **Base Model**: GPT-2 (openai-community/gpt2)
2. **Modified Layer**: Added `out_head = nn.Linear(768, 2)` classification head
3. **Unfrozen Layers**:
   - Last transformer block (`model.h[-1]`)
   - Final LayerNorm (`model.ln_f`)
   - Classification head (`out_head`)
4. **Classification Method**: Using the **last token** hidden state for prediction

## Training Details

- **Dataset**: SMS Spam Collection (UCI)
  - Total samples: 1,494 SMS messages
  - Training set: 1,045 samples
  - Validation set: 149 samples
  - Test set: 300 samples
- **Epochs**: 5
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW
- **Loss Function**: CrossEntropyLoss

## Results

| Metric              | Score |
| ------------------- | ----- |
| Test Accuracy       | ~95%  |
| Validation Accuracy | ~95%+ |

## Usage with Hugging Face Pipeline

```python
from transformers import pipeline

# Load the model directly via pipeline
classifier = pipeline("text-classification", model="mustafaege/spam-detector-gpt2-hf")

# Test with spam message
result = classifier("Congratulations! You've won a free prize! Click here now!")
print(result)
# Output: [{'label': 'LABEL_1', 'score': 0.98}]

# Test with normal message
result = classifier("Hey, are we still meeting tomorrow?")
print(result)
# Output: [{'label': 'LABEL_0', 'score': 0.95}]
```

## Manual Usage

```python
from transformers import GPT2Model, GPT2Tokenizer
import torch
import torch.nn as nn

# Load model and tokenizer
model_name = "mustafaege/spam-detector-gpt2-hf"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)

# Add classification head (if not saved with it)
model.out_head = nn.Linear(768, 2)

# Prepare input
text = "Free prize! Click here now!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

# Get last token hidden state
outputs = model(**inputs)
last_token_hidden = outputs.last_hidden_state[:, -1, :]

# Get prediction
logits = model.out_head(last_token_hidden)
prediction = torch.argmax(logits, dim=-1)

print("SPAM" if prediction.item() == 1 else "NOT SPAM")
```

## Labels

- `LABEL_0`: Not Spam (ham)
- `LABEL_1`: Spam

## Files

- `model.safetensors` - Model weights
- `config.json` - Model configuration
- `vocab.json`, `merges.txt` - Tokenizer files
- `tokenizer_config.json` - Tokenizer configuration
- `special_tokens_map.json` - Special tokens mapping

## Training Visualization

See `training_plots.png` for loss and accuracy curves during training.

## License

Educational use. Dataset: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

## References

- Hugging Face Transformers: https://huggingface.co/transformers
- Dataset: https://huggingface.co/datasets/mustafaege/sms-spam-balanced
