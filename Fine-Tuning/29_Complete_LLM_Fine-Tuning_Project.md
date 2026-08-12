# Lesson 29 — Complete LLM Fine-Tuning Project

> **Goal:** Synthesize all 28 previous lessons into a single, production-ready, highly optimized AI engineering script.

---

## 1. The Architecture

We will build a complete script to fine-tune `Llama-3-8B-Instruct` on a custom conversational dataset. We will utilize:
1. **Unsloth** (Lesson 22) for 2x faster Triton-optimized kernels.
2. **QLoRA (4-bit NF4)** (Lessons 20 & 21) to fit the training into 8GB of VRAM.
3. **Chat Templates & Loss Masking** (Lesson 17) to ensure perfect conversational formatting.
4. **SFTTrainer** (Lesson 18) to orchestrate the pipeline.

---

## 2. The Production Script

```python
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments

# ==========================================
# 1. Configuration & Unsloth Initialization
# ==========================================
max_seq_length = 2048 
model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,           # Auto-detects BF16
    load_in_4bit = True,    # Uses QLoRA (NF4 + Double Quantization)
)

# ==========================================
# 2. Inject LoRA Adapters
# ==========================================
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", # 50% VRAM saving
)

# ==========================================
# 3. Data Preparation & Formatting
# ==========================================
dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train")

def format_chat_template(example):
    # Applies Llama 3's exact chat template `<|start_header_id|>`
    example["text"] = tokenizer.apply_chat_template(
        example["messages"], 
        tokenize=False
    )
    return example

formatted_dataset = dataset.map(format_chat_template)

# ==========================================
# 4. Loss Masking (Data Collator)
# ==========================================
# We only want to train the model on its own responses, not the user prompts.
response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template, 
    tokenizer=tokenizer
)

# ==========================================
# 5. SFTTrainer Execution
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = formatted_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = collator,
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        output_dir = "unsloth-llama-3-finetuned",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Effective batch size = 8
        learning_rate = 2e-4,
        lr_scheduler_type = "linear",
        warmup_steps = 5,
        max_steps = 60,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",            # 8-bit optimizer to save VRAM
        weight_decay = 0.01,
        seed = 3407,
    ),
)

print("Starting Highly Optimized Training...")
trainer.train()

# ==========================================
# 6. Save Final Model
# ==========================================
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("Training Complete. Model Saved.")
```

### Conclusion
You have now reached the cutting edge of AI Engineering. You understand how to gather data, build complex training architectures, overcome physical memory limits through Quantization and FSDP, align models to human values using DPO, and orchestrate it all using high-performance custom kernels. 
