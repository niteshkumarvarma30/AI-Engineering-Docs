# Lesson 19 — Custom Dataset to Complete SFT Pipeline

> **Goal:** Build an end-to-end Supervised Fine-Tuning pipeline. Take a raw JSONL file, convert it to a standard conversational format, apply LoRA, train with `SFTTrainer`, and run inference.

---

## 1. Preparing the Custom Dataset

The hardest part of fine-tuning is often data preparation.
Suppose you have a custom dataset of customer support tickets in a raw `tickets.csv` or JSONL format.

```json
{"ticket_id": 101, "customer_issue": "My router is blinking red.", "agent_reply": "A blinking red light means the router cannot connect to the internet. Please restart it."}
```

### The Standard Conversational Format (ShareGPT / OpenAI)
`SFTTrainer` expects your data to be formatted in a standard conversational array (often referred to as the ShareGPT or OpenAI format).

```python
[
    {"role": "system", "content": "You are a tech support agent."},
    {"role": "user", "content": "My router is blinking red."},
    {"role": "assistant", "content": "A blinking red light means..."}
]
```

### Mapping the Data
We use the Hugging Face `datasets` library to load and map our raw data into this standard structure.

```python
from datasets import load_dataset

raw_dataset = load_dataset("json", data_files="tickets.jsonl", split="train")

def format_conversation(example):
    return {
        "messages": [
            {"role": "system", "content": "You are a tech support agent."},
            {"role": "user", "content": example["customer_issue"]},
            {"role": "assistant", "content": example["agent_reply"]}
        ]
    }

formatted_dataset = raw_dataset.map(format_conversation, remove_columns=raw_dataset.column_names)
```

---

## 2. The Complete End-to-End Pipeline

Now we combine everything we've learned: **LoRA**, **Chat Templates**, **Loss Masking**, and **SFTTrainer**.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

model_id = "meta-llama/Llama-3-8B-Instruct"

# 1. Load Tokenizer & Apply Chat Template setup
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Load Model in bf16
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 3. Setup LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. Setup Loss Masking (Data Collator)
# Find the exact string the tokenizer uses for the assistant's turn
response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template, 
    tokenizer=tokenizer
)

# 5. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./support_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    optim="adamw_torch",
    save_strategy="epoch"
)

# 6. Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=peft_config,
    dataset_kwargs={"skip_prepare_dataset": True}, # We already formatted it
    data_collator=collator,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args
)

# 7. Start Training
trainer.train()

# 8. Save the final LoRA adapter
trainer.model.save_pretrained("./final_support_lora")
```

---

## 3. Inference: Running the Fine-Tuned Model

After training, you have two folders:
1. The original base model (`Llama-3-8B-Instruct`)
2. Your LoRA adapter weights (`./final_support_lora`)

To run inference, you load the base model and dynamically merge the adapter.

```python
from peft import PeftModel

# Load Base
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# Load and Merge Adapter
model = PeftModel.from_pretrained(base_model, "./final_support_lora")
model = model.merge_and_unload() # Fuses LoRA weights into the base model permanently

# Prepare Input
test_messages = [
    {"role": "system", "content": "You are a tech support agent."},
    {"role": "user", "content": "My screen is totally black."}
]

inputs = tokenizer.apply_chat_template(test_messages, return_tensors="pt", add_generation_prompt=True).to("cuda")

# Generate
outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

By completing this lesson, you have officially mastered the standard Supervised Fine-Tuning pipeline. In the next lessons, we will explore how to do this on hardware that is far too small to hold the base model (Quantization).
