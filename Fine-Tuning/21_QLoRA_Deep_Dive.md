# Lesson 21 — QLoRA Deep Dive

> **Goal:** Understand QLoRA (Quantized Low-Rank Adaptation). Learn the three mathematical innovations (NF4, Double Quantization, Paged Optimizers) that make fine-tuning a 65B parameter model on a single GPU possible.

---

## 1. What is QLoRA?

In Lesson 9, we learned **LoRA**: freezing the base model and injecting small trainable matrices ($A$ and $B$).
In Lesson 20, we learned **Quantization**: reducing the base model's precision to 8-bit to save memory.

**QLoRA = Quantization + LoRA.**

Specifically, QLoRA loads the base model in **4-bit precision** (drastically reducing VRAM), but keeps the LoRA adapters in **16-bit precision** (BF16).

```text
Input 
  ↓
Base Model (Frozen, 4-bit)  --->  Adapter A (Trainable, 16-bit)
  ↓                                ↓
  +  <-------------------------  Adapter B (Trainable, 16-bit)
  ↓
Output
```

By doing this, the researchers managed to fine-tune a massive 65 Billion parameter model on a single 48GB GPU—retaining 99% of full 16-bit performance.

---

## 2. The Three Innovations of QLoRA

Standard 4-bit quantization usually ruins the model's performance. QLoRA solves this using three mathematical tricks.

### Innovation 1: NormalFloat 4 (NF4)
Standard 4-bit integers (`0` to `15`) assume the model's weights are distributed evenly. But neural network weights usually follow a **Normal (Gaussian) Distribution**—most weights are near `0.0`.

NF4 is a new data type designed specifically for neural networks. Instead of spacing the 16 available buckets evenly, it spaces them based on the standard normal distribution curve. 

There are more "buckets" near zero, and fewer buckets at the tails. This completely eliminates the quantization error for the vast majority of the weights.

### Innovation 2: Double Quantization
In quantization, we need to save the **Scale Factor ($S$)** and **Zero Point ($Z$)** for every single block of 64 weights.
For a 65B model, storing these scale factors takes up **0.5 GB of VRAM**.

**Double Quantization** means the researchers simply applied quantization *again* to the scale factors themselves (converting 32-bit floats to 8-bit floats). 
This saved an average of **0.37 GB per model**, which is the difference between an Out-Of-Memory error and a successful training run.

### Innovation 3: Paged Optimizers
During backpropagation, the Optimizer (like Adam) stores momentum and variance for every single parameter. This takes a massive amount of VRAM.
QLoRA introduced **Paged Optimizers**, utilizing the NVIDIA Unified Memory feature.

When the GPU memory fills up, QLoRA seamlessly pages the optimizer states out to the much slower **CPU RAM**, and pulls them back into VRAM only when needed for the weight update step. 
This prevents GPU spikes from crashing the training job.

---

## 3. QLoRA in Code (Hugging Face)

Implementing QLoRA requires combining the `bitsandbytes` library with `peft`.

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# 1. Configure the 4-bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,      # Double Quantization
    bnb_4bit_quant_type="nf4",           # NormalFloat 4
    bnb_4bit_compute_dtype=torch.bfloat16 # Computation is done in 16-bit
)

# 2. Load the base model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. Prepare the model for training (freezes base, sets gradients)
model = prepare_model_for_kbit_training(model)

# 4. Inject 16-bit LoRA adapters
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

print(model.print_trainable_parameters())
# Output: trainable params: 2% | all params: 100%
```

With this exact code block, you can fine-tune Llama-3-8B on a completely free Google Colab GPU (15GB T4).
