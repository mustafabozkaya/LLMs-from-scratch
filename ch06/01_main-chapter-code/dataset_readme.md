---
annotations_creators:
- no-annotation
language:
- en
language_details: English SMS messages
license: other
multilinguality:
- monolingual
size_categories:
- n<1K
source_datasets:
- original
---

# SMS Spam Collection (Balanced)

A balanced SMS spam dataset for text classification.

## Overview

| Property | Value |
|----------|-------|
| Total Samples | 1,494 |
| Train | 1,045 (70%) |
| Validation | 149 (10%) |
| Test | 300 (20%) |
| Classes | ham (0), spam (1) |

## Dataset Description

This is a balanced version of the UCI SMS Spam Collection dataset. Originally, the dataset had 5,572 messages with an imbalanced distribution (4,825 ham, 747 spam). We balanced it to 747 ham and 747 spam for training efficiency.

### Label Distribution (Balanced)

| Label | Count |
|-------|-------|
| ham (0) | 747 |
| spam (1) | 747 |

### Split Distribution

| Split | ham | spam | Total |
|-------|-----|------|-------|
| train | ~522 | ~523 | 1,045 |
| validation | ~75 | ~74 | 149 |
| test | ~150 | ~150 | 300 |

## How It Was Created

1. **Downloaded** from UCI Machine Learning Repository
2. **Balanced** by undersampling ham messages to match spam count (747 each)
3. **Split** into train (70%), validation (10%), test (20%) sets
4. **Converted** labels to integers: ham=0, spam=1

## Usage with Hugging Face

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("mustafaege/sms-spam-balanced")

# Access splits
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Example
print(train_data[0])
# {'Text': 'Go until jurong point, crazy..', 'Label': 0}
```

## Usage with Pandas

```python
from datasets import load_dataset

dataset = load_dataset("mustafaege/sms-spam-balanced")

# Convert to pandas
train_df = dataset['train'].to_pandas()
print(train_df.head())
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| Text | string | SMS message text |
| Label | int | 0 = ham (not spam), 1 = spam |

## Example Messages

### Ham (Label 0)
- "Go until jurong point, crazy.."
- "Ok lar... Joking wif u oni..."
- "U dun say so early hor..."

### Spam (Label 1)
- "Free entry in 2 a wkly comp to win FA Cup final..."
- "Had your contract mobile 11 Mnths? Latest Moto..."
- "This is the 2nd time we have tried to contact u..."

## For Fine-tuning

This dataset is designed for fine-tuning LLMs for spam detection. 

### Training Details Used

- **Epochs**: 5
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW
- **Loss Function**: CrossEntropyLoss

### Results

| Metric | Score |
|--------|-------|
| Test Accuracy | ~95% |
| Validation Accuracy | ~95%+ |

### Companion Model

**Model:** [mustafaege/spam-detector-gpt2-hf](https://huggingface.co/mustafaege/spam-detector-gpt2-hf)

## License

The original UCI SMS Spam Collection dataset is available under the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) terms.

## References

- Original Dataset: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
