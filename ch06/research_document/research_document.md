# Comprehensive Guide: Fine-tuning LLMs with Unsloth, Converting to GGUF, and Optimizing with AutoKernel/Triton

## Overview
This document provides a complete workflow for:
1. Fine-tuning language models using Unsloth with LoRA/QLoRA adaptations
2. Converting fine-tuned models to GGUF format for use with llama.cpp
3. Optimizing model execution locally using AutoKernel with Triton kernel optimizations
4. Running the optimized models on local hardware

## Part 1: Fine-tuning with Unsloth

### Why Unsloth?
Unsloth is an open-source library that makes fine-tuning LLMs faster and more memory-efficient by:
- Using optimized 4-bit quantization that maintains accuracy while reducing memory usage
- Implementing LoRA (Low-Rank Adaptation) and QLoRA for parameter-efficient fine-tuning
- Providing custom Triton kernels for faster attention and MLP operations
- Supporting a wide range of models (Llama, Mistral, Gemma, etc.)

### Installation
```bash
# Install Unsloth and dependencies
pip install unsloth torch transformers datasets trl peft

# For best performance, ensure you have the latest CUDA drivers and PyTorch with CUDA support
```

### Basic Fine-tuning Workflow
Here's a complete example of fine-tuning a model using Unsloth:

```python
import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

# 1. Load model in 4-bit quantization
model_name = "unsloth/TinyLlama-V1.1-Chat"  # or any other supported model
max_seq_length = 2048
dtype = torch.float16
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 2. Apply LoRA for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=max_seq_length,
)

# 3. Prepare dataset
# Format: [{ "instruction": "...", "response": "..." }, ...]
data = [
    {"instruction": "Merhaba, nasılsın?", "response": "İyiyim, teşekkür ederim! Sen nasılsın?"},
    {"instruction": "Türkiye'nin başkenti neresidir?", "response": "Türkiye'nin başkenti Ankara'dır."},
]

def format_instruction(example):
    return {"text": f"### Kullanıcı:\n{example['instruction']}\n\n### Asistan:\n{example['response']}"}

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
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not load_in_4bit,
        bf16=False,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

# 5. Train
trainer.train()

# 6. Save model
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
```

### Advanced Unsloth Features
- **QLoRA**: Combine 4-bit quantization with LoRA for maximum memory efficiency
- **RoPE Scaling**: Extend context length beyond the model's original training
- **Dynamic Quantization**: Adjust precision during inference for speed/quality trade-offs
- **Speculative Decoding**: Use a smaller draft model to accelerate generation

## Part 2: Converting to GGUF Format

### What is GGUF?
GGUF (GPT-Generated Unified Format) is the latest format used by llama.cpp for storing models. It replaces the older GGML format and offers:
- Better extensibility
- Improved memory mapping
- Support for various quantization methods
- Embedded metadata and tokenizer

### Conversion Process
To convert a Hugging Face model to GGUF format:

#### Method 1: Using llama.cpp's conversion script
```bash
# 1. Clone llama.cpp if you haven't already
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 2. Install required Python packages
pip install -r requirements.txt
pip install -r requirements-convert-hf-to-gguf.txt

# 3. Convert the model
python convert_hf_to_gguf.py ./path/to/your/fine_tuned_model \
  --outtype f16  # or q4_0, q5_1, q8_0 for quantization
```

#### Method 2: Using our custom conversion script
We've created a `convert_to_gguf.py` script that simplifies this process:
```bash
python convert_to_gguf.py ./fine_tuned_model --outtype q4_0
```

### Quantization Options
When converting to GGUF, you can choose different quantization types:
- `f16` or `f32`: Float16/Float32 (no quantization, highest quality)
- `q4_0`: 4-bit quantization (good balance of size/quality)
- `q4_1`: 4-bit quantization with higher accuracy
- `q5_0`: 5-bit quantization
- `q5_1`: 5-bit quantization with higher accuracy
- `q8_0`: 8-bit quantization (near lossless)

For local deployment on consumer hardware, `q4_0` or `q5_1` are often good choices.

## Part 3: Running with llama.cpp

### llama.cpp Overview
llama.cpp is a C/C++ inference engine for LLMs that:
- Requires no Python or CUDA for basic operation
- Uses memory mapping for efficient loading
- Supports GGUF format natively
- Provides both CLI and server interfaces

### Running with llama-cli
```bash
# Basic usage
./llama-cli -m ./fine_tuned_model/model.q4_0.gguf -p "Merhaba, nasılsın?" -n 50

# Interactive mode
./llama-cli -m ./fine_tuned_model/model.q4_0.gguf -i

# With specific parameters
./llama-cli -m ./fine_tuned_model/model.q4_0.gguf \
  -p "Türkiye'nin başkenti neresidir?" \
  -n 100 \
  --temp 0.7 \
  --top_p 0.9 \
  --repeat_penalty 1.1
```

### Running with llama-server (API mode)
```bash
# Start the server
./llama-server -m ./fine_tuned_model/model.q4_0.gguf --port 8080

# Then use it like an OpenAI-compatible API
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "prompt": "Merhaba, nasılsın?",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

## Part 4: Optimizing with AutoKernel and Triton

### What is AutoKernel?
AutoKernel is a tool for automatically generating optimized kernels for deep learning operations using Triton. It can:
- Profile existing implementations to identify bottlenecks
- Generate optimized Triton kernels for attention, MLP, and other operations
- Automatically tune kernel parameters for your specific hardware
- Integrate with PyTorch models

### What is Triton?
Triton is a programming language and compiler for writing highly efficient GPU kernels. It:
- Provides a Python-like syntax for GPU programming
- Automatically handles many low-level CUDA details
- Often produces kernels that match or exceed hand-optimized CUDA
- Is particularly effective for operations like attention and matrix multiplication

### Integrating AutoKernel with Our Workflow
AutoKernel works best with models defined in pure PyTorch (like our GPT-2 implementation). Here's how to use it:

#### Step 1: Prepare a PyTorch model
We already have a minimal GPT-2 implementation in `autokernel/models/gpt2.py` that doesn't require the transformers library.

#### Step 2: Profile the model
```bash
# Profile the GPT-2 model to identify bottlenecks
uv run profile.py --model models/gpt2.py --class-name GPT2 --input-shape 1,1024 --dtype float16
```

#### Step 3: Generate optimized kernels
Based on the profiling results, AutoKernel can generate optimized Triton kernels for the bottleneck operations (typically attention and MLP).

#### Step 4: Integrate the optimized kernels
Replace the standard PyTorch operations in your model with the optimized Triton kernels generated by AutoKernel.

#### Step 5: Benchmark the optimized model
```bash
# Benchmark the optimized model
uv run bench.py --model models/gpt2.py --class-name GPT2 --input-shape 1,1024 --dtype float16
```

### Practical Example: Optimizing Attention
Let's look at how we might optimize the attention mechanism in our GPT-2 model:

Original attention computation in `gpt2.py`:
```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
att = F.softmax(att, dim=-1)
y = att @ v
```

AutoKernel might generate a Triton kernel that fuses these operations for better memory efficiency.

### Benefits of Triton Optimization
- **Reduced Memory Access**: Fused operations reduce trips to global memory
- **Better Occupancy**: Optimized block sizes and warp utilization
- **Specialized Instructions**: Use of tensor cores and other GPU features
- **Automatic Tuning**: Kernel parameters optimized for your specific GPU architecture

## Part 5: Complete Local Deployment Workflow

Here's the complete end-to-end workflow for creating an optimized, locally deployable LLM:

### Step 1: Select and Prepare Base Model
Choose a base model suitable for your task (e.g., TinyLlama, Phi-3, etc.)

### Step 2: Fine-tune with Unsloth
```python
# Use the fine_tune_with_unsloth.py script we created
python fine_tune_with_unsloth.py
```

### Step 3: Convert to GGUF
```bash
# Convert to GGUF with desired quantization
python convert_to_gguf.py ./fine_tuned_model --outtype q4_0
```

### Step 4: Test with llama.cpp
```bash
# Quick test
./llama-cli -m ./fine_tuned_model/model.q4_0.gguf -p "Merhaba" -n 10
```

### Step 5: Optimize with AutoKernel/Triton (for PyTorch deployment)
If you prefer to run the model in PyTorch rather than llama.cpp:
```bash
# Profile the model
uv run profile.py --model models/gpt2.py --class-name GPT2 --input-shape 1,1024 --dtype float16

# Generate optimized kernels (this would be done by AutoKernel based on profiling)
# Integrate the optimized kernels into your model

# Benchmark optimized version
uv run bench.py --model models/gpt2.py --class-name GPT2 --input-shape 1,1024 --dtype float16
```

### Step 6: Deploy Locally
You now have several options for local deployment:
1. **llama.cpp**: Best for minimal dependencies and CPU/GPU flexibility
2. **PyTorch with AutoKernel optimizations**: Best for maximum performance on compatible hardware
3. **Hybrid approach**: Use llama.cpp for development/testing, PyTorch+AutoKernel for production

## Performance Considerations

### Memory Usage
- **Unsloth 4-bit + LoRA**: Can fine-tune a 7B model on a single consumer GPU (8-16GB VRAM)
- **GGUF q4_0**: A 7B model uses ~3.5-4GB RAM/VRAM
- **GGUF f16**: A 7B model uses ~13-14GB RAM/VRAM

### Inference Speed
- **llama.cpp**: Good CPU performance, excellent GPU performance when built with CUDA
- **PyTorch + AutoKernel/Triton**: Can exceed llama.cpp performance on NVIDIA GPUs with optimized kernels
- **Quantization trade-offs**: Lower bitrates (q4_0) are faster but may slightly reduce quality

### Hardware Recommendations
- **Minimum for experimentation**: GTX 1660 or RTX 2060 (6GB VRAM)
- **Recommended for comfortable use**: RTX 3060/3070 (8GB+) or RTX 40-series
- **For larger models**: RTX 3090/4090 (24GB) or multiple GPUs
- **CPU fallback**: All methods work on CPU, but much slower

## Troubleshooting Common Issues

### Unsloth Issues
- **"CUDA out of memory"**: Reduce batch size, increase gradient accumulation, or use 4-bit quantization
- **"trust_remote_code required"**: This is normal for models with custom code (like s2-pro)
- **"Version mismatch"**: Ensure you have compatible versions of torch, transformers, etc.

### llama.cpp Issues
- **"model file is corrupted"**: Redownload the GGUF file or check conversion process
- **"illegal instruction"**: Make sure you built llama.cpp for your specific CPU architecture
- **"segmentation fault"**: Often due to insufficient RAM/VRAM or incompatible binaries

### AutoKernel/Triton Issues
- **"kernel compilation failed"**: Check Triton version compatibility with your CUDA toolkit
- **"numerical mismatch"**: Optimized kernels may have slight numerical differences (usually acceptable)
- **"no improvement seen"**: The bottleneck may be elsewhere (memory bandwidth, etc.)

## Security and Ethical Considerations

### Model Safety
- Always evaluate fine-tuned models for harmful biases or unsafe outputs
- Consider implementing content filters for deployed models
- Be aware of the data used in fine-tuning and its potential biases

### Privacy
- When fine-tuning on private data, ensure proper data handling and deletion after training
- Consider federated learning approaches for highly sensitive data
- Be aware that GGUF models can be easily shared and redistributed

### Responsible AI
- Clearly disclose when users are interacting with AI-generated content
- Avoid using models for high-stakes decisions without proper validation
- Respect copyright and licensing when using training data

## Conclusion

This workflow provides a powerful pathway from base model to locally deployed, optimized LLM:
1. **Unsloth** makes fine-tuning accessible on consumer hardware
2. **GGUF + llama.cpp** enables local deployment with minimal dependencies
3. **AutoKernel/Triton** can provide additional performance gains for PyTorch deployments

By combining these technologies, you can create highly efficient, customized language models that run locally on your machine, ensuring privacy, reducing latency, and eliminating dependency on external APIs.

The key is to experiment with different combinations to find what works best for your specific use case, hardware, and requirements.