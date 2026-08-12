# Lesson 24 — Direct Preference Optimization (DPO)

> **Goal:** Understand how Direct Preference Optimization (DPO) bypasses complex Reinforcement Learning and mathematically simplifies preference alignment into a single loss function.

---

## 1. The Complexity of RLHF

Before 2023, the only way to do preference alignment was **RLHF**.
RLHF requires you to:
1. Train a base model.
2. SFT fine-tune it.
3. Train a separate **Reward Model** to score responses.
4. Use **PPO** (Proximal Policy Optimization) to update the main model based on the Reward Model's score.

This is highly unstable, requires double the VRAM, and is notoriously difficult to converge.

---

## 2. The Mathematical Breakthrough of DPO

In 2023, researchers at Stanford published **Direct Preference Optimization (DPO)**.

They made a brilliant mathematical observation: 
**You can map the Reward Model directly to the Language Model's implicit probabilities.**

Instead of training a separate Reward Model, DPO proves that the LLM *itself* acts as a reward model. If the LLM assigns a high probability to a sequence of words, that *is* its implicit reward.

### The DPO Loss Function

To implement DPO, you need two models loaded into memory:
1. **The Policy Model** ($\pi_\theta$): The model you are actively training (usually with LoRA).
2. **The Reference Model** ($\pi_{\text{ref}}$): A frozen copy of the model *before* DPO training.

The DPO Loss Function compares how both models rate the Chosen ($y_w$) and Rejected ($y_l$) responses:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right)
$$

### Breaking Down the Math:
1. $\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}$: Is our training model predicting the *Chosen* response more often than the frozen reference model? (We want this to go UP).
2. $\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}$: Is our training model predicting the *Rejected* response more often than the frozen reference model? (We want this to go DOWN).
3. $\beta$: A temperature parameter that controls how much the training model is allowed to deviate from the reference model (prevents the model from destroying its grammar to maximize the score).
4. $\sigma$: The sigmoid function, pushing the difference into a 0-1 probability.

By minimizing this loss, we directly increase the likelihood of the chosen response and decrease the likelihood of the rejected response—in a single, stable training loop!

---

## 3. DPO in Practice (Hugging Face TRL)

Just like `SFTTrainer`, Hugging Face provides a `DPOTrainer`.

### The DPO Dataset Format
DPO requires three columns: `prompt`, `chosen`, and `rejected`.

```python
dataset = [
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris.",
        "rejected": "Paris."
    }
]
```

### The DPOTrainer Code

```python
import torch
from transformers import AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
from peft import LoraConfig

# Load the model we want to train
model = AutoModelForCausalLM.from_pretrained("my-sft-model", device_map="auto")

# Load a frozen copy of the exact same model to act as the Reference
ref_model = AutoModelForCausalLM.from_pretrained("my-sft-model", device_map="auto")

peft_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="./dpo_results",
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    remove_unused_columns=False # Crucial for DPO
)

dpo_trainer = DPOTrainer(
    model,
    ref_model,                   # Pass the frozen reference model
    args=training_args,
    beta=0.1,                    # The beta parameter from the math formula
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config      # Apply LoRA to the training model
)

dpo_trainer.train()
```

DPO has almost completely replaced RLHF in the open-source community due to its mathematical simplicity and absolute stability.
