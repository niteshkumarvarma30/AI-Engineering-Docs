# Lesson 11 — Practical QLoRA Fine-Tuning with Hugging Face

> **Goal:** Move from understanding LoRA/QLoRA mathematically to building a complete, practical QLoRA fine-tuning pipeline using the Hugging Face ecosystem.

---

# 1. What We Are Building

In the previous lessons, we learned:

```text
Fine-Tuning
    ↓
PEFT
    ↓
LoRA
    ↓
Quantization
    ↓
QLoRA
```

Now we will actually implement the workflow:

```text
Pretrained LLM
      ↓
Tokenizer
      ↓
Dataset
      ↓
4-bit Quantized Model
      ↓
LoRA Configuration
      ↓
LoRA Adapters
      ↓
Training Configuration
      ↓
Trainer
      ↓
Fine-Tuning
      ↓
Save LoRA Adapter
      ↓
Load Adapter
      ↓
Inference
```

The purpose of this lesson is not just to copy code.

You should understand **what every major component is doing**.

---

# 2. The Hugging Face Ecosystem

For practical QLoRA fine-tuning, several libraries commonly work together.

```text
Transformers
     +
Datasets
     +
PEFT
     +
BitsAndBytes
     +
Accelerate
```

Each library has a different responsibility.

---

# 3. Transformers

Hugging Face `transformers` provides:

- Pretrained models
- Tokenizers
- Model architectures
- Training utilities
- Generation utilities

Conceptually:

```text
Transformers
      ↓
Model + Tokenizer
```

Example:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
```

---

# 4. Datasets

The Hugging Face `datasets` library provides tools for loading and processing datasets.

Example:

```python
from datasets import load_dataset
```

Conceptually:

```text
Dataset
   ↓
Load
   ↓
Process
   ↓
Tokenize
   ↓
Training
```

---

# 5. PEFT

PEFT stands for:

\[
\boxed{\text{Parameter-Efficient Fine-Tuning}}
\]

We learned that LoRA is a PEFT method.

The Hugging Face PEFT library provides implementations for methods such as:

- LoRA
- Adapters
- Other parameter-efficient approaches

Example:

```python
from peft import LoraConfig, get_peft_model
```

---

# 6. BitsAndBytes

`bitsandbytes` is commonly used with Transformers for low-bit model loading and quantization.

For QLoRA, we can configure a model to load using 4-bit quantization.

Example:

```python
from transformers import BitsAndBytesConfig
```

---

# 7. Accelerate

Hugging Face Accelerate helps manage device placement and training across different hardware configurations.

It becomes particularly useful when working with:

- GPUs
- Mixed precision
- Distributed training
- Larger models

We will use it indirectly through the Hugging Face training ecosystem in this lesson.

---

# 8. Installing the Main Libraries

A typical environment can be prepared with:

```bash
pip install transformers datasets peft accelerate bitsandbytes
```

Depending on the model and training setup, additional libraries may be required.

---

# 9. The Complete Pipeline

Before writing code, understand the complete flow:

```text
1. Select pretrained model
        ↓
2. Load tokenizer
        ↓
3. Load dataset
        ↓
4. Preprocess dataset
        ↓
5. Configure 4-bit quantization
        ↓
6. Load model
        ↓
7. Configure LoRA
        ↓
8. Attach LoRA
        ↓
9. Check trainable parameters
        ↓
10. Configure training
        ↓
11. Train
        ↓
12. Save adapter
        ↓
13. Load adapter
        ↓
14. Generate output
```

---

# 10. Step 1 — Select a Pretrained Model

For causal language-model fine-tuning, we use a pretrained causal LM.

Conceptually:

```python
model_name = "YOUR_MODEL_ID"
```

For learning, use a model that is small enough for your available hardware.

The exact model is a practical choice based on:

- GPU VRAM
- Model size
- License
- Language
- Task
- Dataset
- Hardware support

---

# 11. Step 2 — Load the Tokenizer

A tokenizer converts text into tokens.

For example:

```text
"Hello world"
       ↓
Tokenizer
       ↓
Token IDs
```

Load it with:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)
```

---

# 12. Why Do We Need a Tokenizer?

Neural networks do not directly consume strings.

The model needs numerical representations.

The pipeline is approximately:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embeddings
 ↓
Transformer
```

---

# 13. Tokenizer and Model Must Match

A very important rule:

> **Use the tokenizer associated with the model unless you have a specific reason to use another compatible tokenizer.**

Conceptually:

```text
Model A
   +
Tokenizer A
```

is the normal setup.

Do not arbitrarily combine unrelated tokenizers and models.

---

# 14. Padding

Some causal language models do not have a dedicated padding token.

In such cases, a common setup is:

```python
tokenizer.pad_token = tokenizer.eos_token
```

But this should be done only when appropriate for the selected model.

Always understand the tokenizer configuration of the specific model.

---

# 15. Step 3 — Prepare the Dataset

A fine-tuning dataset contains examples from which the model can learn.

For example:

```text
Question:
What is machine learning?

Answer:
Machine learning is a branch of AI...
```

Or:

```text
Instruction:
Explain gradient descent.

Response:
Gradient descent is an optimization algorithm...
```

---

# 16. Dataset Formats

A dataset may contain columns such as:

```text
instruction
input
output
```

or:

```text
prompt
response
```

or:

```text
messages
```

The exact format depends on the task.

Later, when we study **custom datasets, instruction tuning, and chat templates**, we will go deeper into dataset formatting.

---

# 17. Loading a Dataset

Example:

```python
from datasets import load_dataset

dataset = load_dataset(
    "your_dataset_name"
)
```

For a local file, the approach depends on the file format.

For example, a JSON dataset might contain:

```json
[
  {
    "instruction": "Explain LoRA.",
    "response": "LoRA is a parameter-efficient..."
  }
]
```

---

# 18. Inspect the Dataset

Before training, always inspect your dataset.

For example:

```python
print(dataset)
```

and:

```python
print(dataset["train"][0])
```

You should know:

- What columns exist
- How many examples exist
- Whether examples are valid
- Whether there are missing values
- How the text is structured

Never blindly start training without inspecting the data.

---

# 19. Why Dataset Quality Matters

A model learns from the training data.

Therefore:

```text
Bad Dataset
     ↓
Bad Training Signal
     ↓
Poor Fine-Tuned Model
```

Whereas:

```text
High-quality Dataset
     ↓
Useful Training Signal
     ↓
Better Adaptation
```

QLoRA does not magically fix a poor dataset.

---

# 20. Step 4 — Format the Dataset

Suppose we have:

```text
instruction
response
```

We can combine them into a training text.

For example:

```text
### Instruction:
Explain LoRA.

### Response:
LoRA freezes the pretrained weights and learns...
```

Conceptually:

```python
def format_example(example):
    return {
        "text": (
            "### Instruction:\n"
            + example["instruction"]
            + "\n\n"
            + "### Response:\n"
            + example["response"]
        )
    }
```

This is a simple instructional format.

Later we will learn **chat templates**, which are often more appropriate for chat/instruct models.

---

# 21. Step 5 — Configure 4-Bit Quantization

Now we reach the QLoRA part.

Import:

```python
from transformers import BitsAndBytesConfig
```

Create the configuration:

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True
)
```

---

# 22. Understand the Configuration

## `load_in_4bit`

```python
load_in_4bit=True
```

Means:

> Load the model using a 4-bit representation.

---

## `bnb_4bit_quant_type`

```python
bnb_4bit_quant_type="nf4"
```

Uses NF4 quantization.

Recall:

\[
\boxed{\text{NF4}=\text{NormalFloat 4}}
\]

---

## `bnb_4bit_compute_dtype`

```python
bnb_4bit_compute_dtype="bfloat16"
```

Specifies the compute dtype.

Conceptually:

```text
Storage → 4-bit
Computation → BF16
```

The exact dtype should match what your hardware supports.

---

## `bnb_4bit_use_double_quant`

```python
bnb_4bit_use_double_quant=True
```

Enables double quantization to reduce quantization-related memory overhead.

---

# 23. Important Hardware Consideration

Do not blindly use:

```python
bfloat16
```

on every machine.

Your hardware must support the desired dtype efficiently.

For example, depending on your GPU, you may choose:

```text
BF16
```

or:

```text
FP16
```

This is one reason hardware knowledge matters when doing practical fine-tuning.

---

# 24. Step 6 — Load the Model

Now load the pretrained causal language model.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
```

The exact loading arguments can vary by model and Transformers version.

---

# 25. What Happened Here?

We started with:

```text
Pretrained Model
```

Then:

```text
Pretrained Model
       ↓
4-bit Quantization
       ↓
Loaded into available device(s)
```

The model is now ready to be prepared for LoRA.

---

# 26. Step 7 — Prepare the Model for k-Bit Training

When using quantized models with PEFT, a preparation step is commonly used.

For example:

```python
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
```

The purpose is to prepare the quantized model for parameter-efficient training.

Remember:

```text
Base model
↓
Quantized
↓
Prepared
↓
LoRA
```

---

# 27. Step 8 — Create the LoRA Configuration

Now define LoRA.

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

# 28. Understand Every LoRA Parameter

## `r`

```python
r=8
```

This is the LoRA rank.

Recall:

\[
r\ll d
\]

Smaller rank:

```text
Fewer trainable parameters
```

Larger rank:

```text
More adaptation capacity
```

---

# 29. `lora_alpha`

```python
lora_alpha=16
```

The common LoRA scaling is:

\[
\frac{\alpha}{r}
\]

Therefore:

\[
\frac{16}{8}=2
\]

So the LoRA contribution is scaled accordingly.

---

# 30. `lora_dropout`

```python
lora_dropout=0.05
```

This provides dropout regularization for the LoRA adaptation.

---

# 31. `target_modules`

Example:

```python
target_modules=["q_proj", "v_proj"]
```

This tells PEFT which modules should receive LoRA adapters.

For attention:

\[
Q=XW_Q
\]

\[
V=XW_V
\]

The exact module names depend on the model architecture.

For some models, the relevant names may be different.

---

# 32. `task_type`

```python
task_type="CAUSAL_LM"
```

This tells PEFT that we are adapting a causal language model.

---

# 33. Step 9 — Attach LoRA

Now attach the LoRA configuration to the model.

```python
from peft import get_peft_model

model = get_peft_model(
    model,
    lora_config
)
```

Conceptually:

```text
Quantized Frozen Model
        +
LoRA
        ↓
QLoRA Model
```

---

# 34. Step 10 — Check Trainable Parameters

This is one of the most important checks.

Run:

```python
model.print_trainable_parameters()
```

You may see something conceptually like:

```text
trainable params: 8M
all params: 7B
trainable%: 0.11%
```

The exact values depend on:

- Model size
- Rank
- Target modules
- Number of layers

---

# 35. Why Check Trainable Parameters?

Because you want to confirm that LoRA was actually applied.

You expect:

```text
Total parameters
        ↓
Very large

Trainable parameters
        ↓
Small fraction
```

If almost the entire model is trainable, something is wrong with your setup.

---

# 36. Step 11 — Training Arguments

Now we need to tell the training system how training should happen.

Hugging Face provides:

```python
from transformers import TrainingArguments
```

A simplified configuration might look like:

```python
training_args = TrainingArguments(
    output_dir="./qlora-output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    report_to="none"
)
```

The exact values are hardware- and task-dependent.

---

# 37. Understanding `output_dir`

```python
output_dir="./qlora-output"
```

Specifies where checkpoints and outputs are stored.

---

# 38. Understanding `num_train_epochs`

```python
num_train_epochs=1
```

One epoch means the model sees the training dataset once.

For example:

```text
1000 examples
1 epoch
↓
Model processes approximately 1000 examples
```

The actual number of optimizer steps depends on batch size, gradient accumulation, sequence lengths, and other settings.

---

# 39. Understanding `per_device_train_batch_size`

```python
per_device_train_batch_size=1
```

This is the number of examples processed at once on each device.

A small batch size is often necessary when GPU memory is limited.

---

# 40. Understanding `gradient_accumulation_steps`

Suppose:

```python
per_device_train_batch_size=1
```

and:

```python
gradient_accumulation_steps=8
```

Then the effective batch size is approximately:

\[
1\times8=8
\]

for a single device, ignoring additional factors such as the number of devices.

The model processes several micro-batches before performing an optimizer update.

---

# 41. Understanding `learning_rate`

Example:

```python
learning_rate=2e-4
```

The learning rate controls how strongly trainable parameters are updated.

For LoRA, learning rates are often larger than typical full-model fine-tuning learning rates, but the appropriate value depends on the model, dataset, optimizer, and training setup.

Do not treat `2e-4` as a universal rule.

---

# 42. `logging_steps`

```python
logging_steps=10
```

Controls how frequently training metrics are logged.

---

# 43. `save_steps`

```python
save_steps=100
```

Controls how frequently checkpoints are saved.

---

# 44. FP16 vs BF16

You may see:

```python
fp16=True
```

or:

```python
bf16=True
```

Do not enable both.

Conceptually:

```text
Supported hardware
      ↓
Choose suitable compute dtype
      ↓
FP16 or BF16
```

BF16 is often preferred on hardware with strong BF16 support.

---

# 45. Step 12 — Trainer

Hugging Face provides:

```python
from transformers import Trainer
```

A simplified trainer setup can look like:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)
```

The exact dataset and data-collator setup depends on how your dataset is formatted.

---

# 46. Important: Tokenization Must Happen Before Training

The trainer cannot simply send arbitrary strings into the model.

The model needs token IDs.

Therefore:

```text
Text
 ↓
Tokenizer
 ↓
input_ids
 ↓
attention_mask
 ↓
Model
```

---

# 47. Tokenizing a Dataset

A simplified example:

```python
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512
    )
```

Then:

```python
tokenized_dataset = dataset.map(
    tokenize_function
)
```

---

# 48. What Does `input_ids` Mean?

Suppose:

```text
"Hello"
```

becomes:

```text
[15496]
```

The exact IDs depend on the tokenizer.

For a sentence:

```text
"Hello world"
```

the tokenizer may produce something like:

```text
[15496, 995]
```

These are numerical token IDs.

---

# 49. What Does `attention_mask` Mean?

The attention mask tells the model which positions contain actual tokens and which positions are padding.

Conceptually:

```text
Tokens:
[A, B, C, PAD, PAD]

Mask:
[1, 1, 1, 0, 0]
```

The exact behavior depends on the model and attention implementation.

---

# 50. Labels for Causal Language Modeling

For causal language modeling, the model learns to predict the next token.

Conceptually:

```text
Input:

The cat is

Target:

cat is sleeping
```

More precisely, the training objective shifts the sequence so that each position predicts the next token.

For example:

```text
Input:
The cat is

Prediction:
cat is sleeping
```

The exact label shifting may be handled by the model/data collator.

---

# 51. The Causal Language Modeling Objective

Suppose the sequence is:

```text
I love machine learning
```

The model learns approximately:

```text
I
↓
predict love

I love
↓
predict machine

I love machine
↓
predict learning
```

Mathematically, the model learns:

\[
P(x_t\mid x_1,\ldots,x_{t-1})
\]

and the sequence probability can be factorized as:

\[
P(x_1,\ldots,x_T)
=
\prod_{t=1}^{T}
P(x_t\mid x_1,\ldots,x_{t-1})
\]

---

# 52. Step 13 — Start Training

Once everything is prepared:

```python
trainer.train()
```

The training flow becomes:

```text
Dataset
   ↓
Tokenizer
   ↓
Token IDs
   ↓
QLoRA Model
   ↓
Forward Pass
   ↓
Loss
   ↓
Backward Pass
   ↓
LoRA Gradients
   ↓
Optimizer
   ↓
Update LoRA
```

---

# 53. What Is Actually Being Updated?

Remember Lesson 10.

```text
Base model
↓
Quantized
↓
Frozen

LoRA
↓
Trainable
↓
Updated
```

So the important idea is:

\[
\boxed{
\text{QLoRA training updates LoRA parameters}
}
\]

not the frozen base model.

---

# 54. What Happens to the Base Model During Training?

The base model still performs the forward computation.

But its parameters are not updated.

Conceptually:

```text
Forward:

Quantized Base
     +
LoRA
     ↓
Prediction
     ↓
Loss

Backward:

     ↓
LoRA gradients
     ↓
Update LoRA
```

---

# 55. Step 14 — Save the Adapter

After training, you can save the PEFT adapter.

Conceptually:

```python
model.save_pretrained(
    "./my-lora-adapter"
)

tokenizer.save_pretrained(
    "./my-lora-adapter"
)
```

The exact saving approach can depend on the training setup.

---

# 56. What Does the Adapter Contain?

The adapter contains the learned LoRA parameters.

Conceptually:

```text
Base Model
+
LoRA Adapter
```

The adapter itself is much smaller than a complete copy of the base model.

This is one of the major practical advantages of PEFT.

---

# 57. Why Save Only the Adapter?

Suppose you fine-tune the same base model for:

```text
Medical
Legal
Coding
Customer Support
```

You could have:

```text
                Base Model
               /    |     \
              ↓     ↓      ↓
           Adapter Adapter Adapter
           Medical  Legal  Coding
```

Instead of storing a complete model copy for each task.

---

# 58. Step 15 — Load the Adapter

Conceptually:

```python
from peft import PeftModel

model = PeftModel.from_pretrained(
    base_model,
    "./my-lora-adapter"
)
```

The idea is:

```text
Base Model
    +
Saved LoRA Adapter
    ↓
Fine-Tuned Model
```

---

# 59. Step 16 — Inference

After loading the base model and adapter:

```text
User Input
    ↓
Tokenizer
    ↓
Fine-Tuned Model
    ↓
Generated Tokens
    ↓
Tokenizer Decode
    ↓
Text Response
```

Conceptually:

```python
inputs = tokenizer(
    "Explain LoRA",
    return_tensors="pt"
)

outputs = model.generate(
    **inputs,
    max_new_tokens=100
)

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)
```

The exact generation settings depend on the model.

---

# 60. Complete Conceptual QLoRA Pipeline

Put everything together:

```text
                 Pretrained LLM
                       ↓
                    Tokenizer
                       ↓
                    Dataset
                       ↓
                 Format Dataset
                       ↓
                    Tokenize
                       ↓
              4-bit Quantization
                       ↓
               Frozen Base Model
                       +
                  LoRA Adapter
                       ↓
                Trainable Params
                       ↓
                    Trainer
                       ↓
                  Fine-Tuning
                       ↓
                Save LoRA Adapter
                       ↓
              Load Base + Adapter
                       ↓
                    Inference
```

---

# 61. Minimal Code Skeleton

The following is a conceptual skeleton.

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

from datasets import load_dataset


# --------------------------------
# 1. Model
# --------------------------------

model_name = "YOUR_MODEL_ID"


# --------------------------------
# 2. Tokenizer
# --------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


# --------------------------------
# 3. Dataset
# --------------------------------

dataset = load_dataset(
    "YOUR_DATASET"
)


# --------------------------------
# 4. Quantization
# --------------------------------

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True
)


# --------------------------------
# 5. Load model
# --------------------------------

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)


# --------------------------------
# 6. Prepare model
# --------------------------------

model = prepare_model_for_kbit_training(
    model
)


# --------------------------------
# 7. LoRA
# --------------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)


# --------------------------------
# 8. Attach LoRA
# --------------------------------

model = get_peft_model(
    model,
    lora_config
)


# --------------------------------
# 9. Check trainable parameters
# --------------------------------

model.print_trainable_parameters()


# --------------------------------
# 10. Training arguments
# --------------------------------

training_args = TrainingArguments(
    output_dir="./qlora-output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    report_to="none"
)


# --------------------------------
# 11. Trainer
# --------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)


# --------------------------------
# 12. Train
# --------------------------------

trainer.train()


# --------------------------------
# 13. Save
# --------------------------------

model.save_pretrained(
    "./my-lora-adapter"
)

tokenizer.save_pretrained(
    "./my-lora-adapter"
)
```

> **Important:** This is a learning skeleton, not a guaranteed drop-in script for every model/dataset. Real training requires the correct dataset formatting, tokenization, data collator, model-specific target modules, hardware-compatible precision, and appropriate training arguments.

---

# 62. Understanding the Code as a Story

Instead of memorizing the code, understand it as:

```text
I have a pretrained model.
        ↓
I need its tokenizer.
        ↓
I have training data.
        ↓
I want to reduce model memory.
        ↓
Load the base model in 4-bit.
        ↓
I don't want to train the base model.
        ↓
Freeze/prepare it.
        ↓
I need task-specific adaptation.
        ↓
Add LoRA.
        ↓
Check that only a small number of parameters are trainable.
        ↓
Configure training.
        ↓
Train.
        ↓
Save the small adapter.
        ↓
Load base + adapter.
        ↓
Use the fine-tuned model.
```

This mental model is more important than memorizing individual function calls.

---

# 63. Important Practical Issue — Dataset Formatting

The biggest missing piece in the skeleton above is proper dataset formatting.

Real LLM fine-tuning often uses structures such as:

```text
instruction
input
response
```

or:

```text
messages
```

For example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Explain LoRA."
    },
    {
      "role": "assistant",
      "content": "LoRA is a parameter-efficient..."
    }
  ]
}
```

Modern chat/instruction models often expect a particular chat format.

This is why **chat templates** are extremely important.

We will study them in a later lesson.

---

# 64. Another Important Issue — Data Collators

A `Trainer` often needs a data collator to construct batches correctly.

For causal language modeling, you may use a suitable language-modeling data collator or a trainer-specific mechanism.

The exact choice depends on:

- Model architecture
- Dataset format
- Whether examples are already tokenized
- Whether labels need to be created
- Whether padding is required

Do not assume one data collator works for every LLM.

---

# 65. Another Important Issue — Sequence Length

Suppose:

```text
max_length = 512
```

Then examples longer than the selected length may be truncated.

Sequence length has a large effect on memory.

Very roughly:

```text
Sequence length ↑
        ↓
Activation memory ↑
        ↓
Training memory ↑
```

Therefore:

```text
512 tokens
```

can be dramatically cheaper than:

```text
4096 tokens
```

depending on the architecture and training configuration.

---

# 66. Another Important Issue — Batch Size

GPU memory is also affected by batch size.

```text
Batch size ↑
     ↓
More examples processed simultaneously
     ↓
More memory
```

When GPU memory is limited:

```text
Small micro-batch
+
Gradient accumulation
```

is often useful.

---

# 67. Another Important Issue — Learning Rate

Do not blindly copy:

```python
learning_rate=2e-4
```

The appropriate learning rate depends on:

- Model
- Dataset size
- Dataset quality
- LoRA rank
- Number of target modules
- Training duration
- Optimizer
- Task

Treat example values as starting points, not universal rules.

---

# 68. Another Important Issue — Overfitting

Fine-tuning a model on a small dataset can lead to overfitting.

For example:

```text
Small dataset
      ↓
Many training epochs
      ↓
Model memorizes examples
      ↓
Poor generalization
```

Monitor:

- Training loss
- Validation loss
- Generated outputs
- Task-specific evaluation metrics

---

# 69. Another Important Issue — Evaluation

Training loss alone is not enough.

Suppose training loss decreases:

```text
Loss:
2.5
 ↓
1.8
 ↓
1.2
 ↓
0.7
```

That does not automatically mean the model is becoming better for the real task.

You should evaluate the fine-tuned model using a suitable validation/test dataset and task-specific metrics.

---

# 70. Common Mistakes

## Mistake 1 — Training the Entire Model Accidentally

You intended to use LoRA but made most parameters trainable.

Check:

```python
model.print_trainable_parameters()
```

---

## Mistake 2 — Wrong Target Module Names

You use:

```python
target_modules=["q_proj", "v_proj"]
```

but the selected model uses different names.

Always inspect the model architecture.

---

## Mistake 3 — Wrong Tokenizer

Using an incompatible tokenizer can break the training pipeline.

Use the tokenizer associated with the selected model unless there is a specific reason not to.

---

## Mistake 4 — Wrong Padding Configuration

Some causal LMs require explicit handling of padding.

Check:

```python
tokenizer.pad_token
```

---

## Mistake 5 — Dataset Is Not Tokenized

The model needs token IDs, not raw strings.

```text
Raw text
↓
Tokenizer
↓
input_ids
```

---

## Mistake 6 — Sequence Length Too Large

If you get CUDA out-of-memory errors:

```text
Reduce sequence length
Reduce micro-batch size
Use gradient accumulation
Use gradient checkpointing
```

---

## Mistake 7 — Unsupported BF16

If your hardware does not support BF16 appropriately, use a compatible precision configuration.

---

## Mistake 8 — Expecting QLoRA to Make Any Model Fit

QLoRA greatly reduces memory requirements, but it does not make arbitrarily large models fit into arbitrary GPUs.

---

# 71. Debugging Checklist

When QLoRA training fails, check in this order:

```text
1. Is the model ID correct?
        ↓
2. Is the model architecture supported?
        ↓
3. Does the tokenizer load?
        ↓
4. Does the dataset load?
        ↓
5. Is the dataset format correct?
        ↓
6. Is the model loaded in the intended dtype?
        ↓
7. Is 4-bit quantization working?
        ↓
8. Are target_modules correct?
        ↓
9. Are only LoRA parameters trainable?
        ↓
10. Is the sequence length reasonable?
        ↓
11. Is the batch size small enough?
        ↓
12. Is the GPU memory sufficient?
```

---

# 72. Questions & Answers

## Q1. What libraries are commonly used for QLoRA?

### Answer

A common stack is:

```text
Transformers
Datasets
PEFT
BitsAndBytes
Accelerate
```

---

## Q2. What does Transformers provide?

### Answer

It provides pretrained models, tokenizers, generation utilities, and training-related components.

---

## Q3. What does PEFT provide?

### Answer

PEFT provides parameter-efficient fine-tuning methods such as LoRA.

---

## Q4. What does BitsAndBytes provide?

### Answer

It is commonly used for low-bit quantization and quantized model loading with Transformers.

---

## Q5. What does `load_in_4bit=True` do?

### Answer

It requests loading the model using a 4-bit quantized representation where supported.

---

## Q6. What is `r` in `LoraConfig`?

### Answer

It is the LoRA rank.

```python
r=8
```

means the low-rank intermediate dimension is 8.

---

## Q7. What does `target_modules` do?

### Answer

It tells PEFT which model modules should receive LoRA adapters.

---

## Q8. Why should we call `print_trainable_parameters()`?

### Answer

To verify that only a small fraction of the model is trainable.

---

## Q9. What does `gradient_accumulation_steps` do?

### Answer

It accumulates gradients across multiple micro-batches before performing an optimizer update.

---

## Q10. Why is gradient accumulation useful?

### Answer

It allows a larger effective batch size while keeping the actual per-device micro-batch small enough to fit in GPU memory.

---

## Q11. Why do we tokenize the dataset?

### Answer

Because the model accepts numerical token IDs rather than raw text strings.

---

## Q12. What does a causal language model learn?

### Answer

It learns to predict the next token based on previous tokens:

\[
P(x_t\mid x_1,\ldots,x_{t-1})
\]

---

## Q13. What is saved after LoRA training?

### Answer

Typically, the learned LoRA adapter parameters can be saved separately from the base model.

---

## Q14. Why is saving the adapter useful?

### Answer

The adapter is much smaller than a complete model copy and can be reused with the corresponding base model.

---

# 73. Self-Test

Try answering these without looking back:

1. What is the complete QLoRA pipeline?
2. What does Transformers provide?
3. What does PEFT provide?
4. What does BitsAndBytes do?
5. Why do we need a tokenizer?
6. Why must the tokenizer generally match the model?
7. What is a dataset format?
8. Why should we inspect the dataset before training?
9. What does `load_in_4bit=True` mean?
10. What is NF4?
11. What is `bnb_4bit_compute_dtype`?
12. What is `prepare_model_for_kbit_training()` used for?
13. What is `LoraConfig`?
14. What does `r` mean?
15. What does `lora_alpha` mean?
16. What does `target_modules` mean?
17. Why should we check trainable parameters?
18. What does `TrainingArguments` configure?
19. What does gradient accumulation do?
20. What happens during `trainer.train()`?
21. Which parameters are updated during QLoRA?
22. Why do we tokenize before training?
23. Why is sequence length important for GPU memory?
24. Why can a model overfit a small fine-tuning dataset?
25. Why is evaluation necessary?
26. Why can the exact code differ between different LLMs?

---

# 74. Understanding Check

Before moving to the next lesson, make sure you can explain this entire process:

```text
I have a pretrained LLM.
        ↓
I load its tokenizer.
        ↓
I prepare my dataset.
        ↓
I load the base model in 4-bit.
        ↓
I prepare the quantized model.
        ↓
I configure LoRA.
        ↓
I attach LoRA.
        ↓
I check trainable parameters.
        ↓
I configure the training process.
        ↓
I train only the LoRA parameters.
        ↓
I save the LoRA adapter.
        ↓
I load the base model + adapter.
        ↓
I run inference.
```

If you can explain why each step exists, you understand the practical QLoRA workflow.

---

# 75. Final Mental Model

Remember:

```text
                  PRETRAINED LLM
                        │
                        ↓
                  4-BIT QUANTIZATION
                        │
                        ↓
                  FROZEN BASE MODEL
                        │
                        +
                   LoRA ADAPTER
                   A + B
                        │
                        ↓
                  TRAIN LoRA ONLY
                        │
                        ↓
                  TASK ADAPTATION
                        │
                        ↓
                  SAVE ADAPTER
                        │
                        ↓
              BASE MODEL + ADAPTER
                        │
                        ↓
                     INFERENCE
```

The most important idea is:

\[
\boxed{
\text{QLoRA training}
=
\text{4-bit frozen base}
+
\text{trainable LoRA adapters}
}
\]

---

# 76. Lesson 11 Summary

## Main libraries

```text
Transformers
→ Models + Tokenizers + Training utilities

Datasets
→ Dataset loading + processing

PEFT
→ LoRA and other PEFT methods

BitsAndBytes
→ Low-bit quantization

Accelerate
→ Hardware/training acceleration support
```

## Main workflow

```text
Dataset
↓
Format
↓
Tokenizer
↓
4-bit model
↓
Prepare k-bit model
↓
LoRA configuration
↓
Attach LoRA
↓
Check trainable parameters
↓
TrainingArguments
↓
Trainer
↓
Train
↓
Save adapter
↓
Load adapter
↓
Inference
```

## Most important concepts

\[
\boxed{
\text{Base Model}=\text{Frozen}
}
\]

\[
\boxed{
\text{Base Model}=\text{Quantized}
}
\]

\[
\boxed{
\text{LoRA}=\text{Trainable}
}
\]

and:

\[
\boxed{
W_{\text{effective}}
=
W+
\frac{\alpha}{r}BA
}
\]

---

# 77. What Comes Next?

## Lesson 12 — Custom Datasets for LLM Fine-Tuning

Now that you understand the QLoRA implementation pipeline, the next major question is:

> **How do we create and prepare our own dataset for fine-tuning?**

We will learn:

```text
Raw Dataset
    ↓
Understand dataset structure
    ↓
Clean data
    ↓
Instruction / Response format
    ↓
Train / Validation split
    ↓
Formatting
    ↓
Tokenization
    ↓
Chat Templates
    ↓
Ready for SFT / QLoRA
```

This is an important step because:

\[
\boxed{
\text{Good Fine-Tuning}
=
\text{Good Model}
+
\text{Good Dataset}
+
\text{Correct Training Setup}
}
\]

The next lesson will focus on **custom datasets**, before moving into **chat templates, instruction tuning, and SFT**.
