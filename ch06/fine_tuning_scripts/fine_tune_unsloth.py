"""
Fine-tune a language model using Unsloth with LoRA adaptation.
This script demonstrates fine-tuning TinyLlama on a small custom dataset.
"""

import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

# 1. Model and tokenizer loading
model_name = "unsloth/TinyLlama-V1.1-Chat"
max_seq_length = 2048
dtype = torch.float16
load_in_4bit = True  # Use 4-bit quantization to reduce memory usage

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # if using a gated model
)

# 2. Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Rank
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    max_seq_length=max_seq_length,
)

# 3. Prepare a small dataset for demonstration
# In practice, replace this with your own dataset
data = [
    {
        "instruction": "Merhaba, nasılsın?",
        "response": "İyiyim, teşekkür ederim! Sen nasılsın?",
    },
    {
        "instruction": "Türkiye'nin başkenti neresidir?",
        "response": "Türkiye'nin başkenti Ankara'dır.",
    },
    {
        "instruction": "Python'da bir listeyi nasıl sıralarsın?",
        "response": "Python'da bir listeyi sorted() fonksiyonu veya list.sort() metodu kullanarak sıralayabilirsin.",
    },
    {
        "instruction": "E=mc² formülüsünü açıklayabilir misin?",
        "response": "E=mc² formülü, Einstein'ın görecelilik teorisinden kuuluşur ve enerji (E), kütle (m) ve ışık hızının karesi (c²) arasındaki ilişkiyi gösterir.",
    },
]


# Convert to Hugging Face Dataset
def format_instruction(example):
    return {
        "text": f"### Kullanıcı:\n{example['instruction']}\n\n### Asistan:\n{example['response']}"
    }


dataset = Dataset.from_list(data)
dataset = dataset.map(format_instruction, remove_columns=["instruction", "response"])

# 4. Configure trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can set to True for faster training
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,  # Set to 1 for demo; increase for real training
        learning_rate=2e-4,
        fp16=not load_in_4bit,  # Use FP16 if not 4-bit
        bf16=False,  # Set to True if using Ampere or newer GPU
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # Use for Weights and Biases
    ),
)

# 5. Train
print("Starting training...")
trainer.train()

# 6. Save the model
print("Saving model...")
model.save_pretrained("fine_tuned_model")  # Local saving
tokenizer.save_pretrained("fine_tuned_model")

# Optional: Save to Hugging Face Hub
# model.push_to_hub("your_username/fine_tuned_model", token = "hf_...")
# tokenizer.push_to_hub("your_username/fine_tuned_model", token = "hf_...")

print("Fine-tuning completed! Model saved to 'fine_tuned_model' directory.")
