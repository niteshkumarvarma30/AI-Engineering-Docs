# Lesson 18 — Hugging Face TRL & SFTTrainer

> **Goal:** Transition from standard PyTorch training loops to using Hugging Face's `TRL` (Transformer Reinforcement Learning) library and the highly optimized `SFTTrainer`.

---

## 1. The Standard Trainer vs SFTTrainer

In standard Hugging Face (Lesson 5), we use the `Trainer` class. However, `Trainer` requires you to:
1. Manually tokenize your dataset.
2. Manually apply chat templates.
3. Handle chunking and padding manually.
4. Set up complex data collators.

### Enter TRL (Transformer Reinforcement Learning)
TRL is an advanced library built on top of Transformers designed specifically for fine-tuning LLMs. It includes a specialized class: **`SFTTrainer`** (Supervised Fine-Tuning Trainer).

`SFTTrainer` acts as a massive wrapper that automates the tedious parts of text-based fine-tuning.

```text
Dataset (Raw Text or Dicts)
      ↓
SFTTrainer 
  ├─ Automates Tokenization
  ├─ Applies Chat Templates
  ├─ Handles Loss Masking (via Data Collators)
  └─ Injects LoRA adapters (via PEFT integration)
      ↓
Fine-Tuned Model
```

---

## 2. Key Features of SFTTrainer

### A. Packing (Constant Length Chunking)
If your dataset contains short responses (e.g., 50 tokens), passing them one by one wastes GPU compute because modern GPUs are optimized for large sequence lengths (e.g., 2048 tokens).

`SFTTrainer` supports **packing**: it concatenates multiple short sequences together (separated by an `<EOS>` token) to perfectly fill the context window.

```text
Without Packing (Wasted Compute):
[User 1 + AI 1] [ PAD PAD PAD PAD PAD ... ]
[User 2 + AI 2] [ PAD PAD PAD PAD PAD ... ]

With Packing (High Efficiency):
[User 1 + AI 1] <EOS> [User 2 + AI 2] <EOS> [User 3 + AI 3]
```

### B. Direct PEFT Integration
You don't need to manually wrap your model in `get_peft_model()`. You simply pass your `LoraConfig` into the `SFTTrainer`, and it injects the adapters automatically.

---

## 3. The SFTTrainer Code Skeleton

Here is the exact boilerplate used by professional AI engineers to set up a fine-tuning job.

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# 1. Load Model and Tokenizer
model_id = "meta-llama/Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# 2. Configure LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10
)

# 4. Load Dataset
dataset = load_dataset("json", data_files="my_chat_data.json", split="train")

# 5. Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",  # Column containing formatted text
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
    packing=True                # Enable high-efficiency batching
)

# 6. Train!
trainer.train()
```

---

## 4. Formatting Functions

If your dataset isn't already a single string, you can pass a `formatting_func` to `SFTTrainer` instead of `dataset_text_field`.

```python
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"User: {example['instruction'][i]}\nAI: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

trainer = SFTTrainer(
    ...
    formatting_func=formatting_prompts_func,
    ...
)
```
This maps your raw JSON structure directly into text on the fly, saving disk space and pre-processing time.
