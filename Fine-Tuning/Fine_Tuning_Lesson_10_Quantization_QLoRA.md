# Lesson 10 — Quantization & QLoRA

> **Goal:** Understand quantization and QLoRA from first principles and see how they solve a different problem from LoRA.

---

# 1. Where We Are in the Fine-Tuning Roadmap

So far:

```text
Lesson 1
Why Fine-Tuning?
        ↓
Lessons 2–6
Loss → Backpropagation → Optimization
        ↓
Lesson 7
Learning Rate → Optimizer → AdamW → Scheduler
        ↓
Lesson 8
GPU Memory → Gradient Accumulation
→ Mixed Precision → Gradient Checkpointing
        ↓
Lesson 9
LoRA + PEFT
        ↓
Lesson 10
Quantization + QLoRA
```

You already learned that LoRA reduces the number of **trainable parameters**.

Now we solve another problem:

> **What if the frozen base model itself is too large to fit comfortably in GPU memory?**

---

# 2. The Problem With Large Models

Suppose we have a pretrained model containing:

\[
7\text{ billion parameters}
\]

With LoRA, we freeze these parameters.

But frozen does **not** mean they disappear.

The base model still needs to be loaded into memory.

Therefore:

```text
LoRA
↓
Reduces trainable parameters
↓
But the base model still has to be stored
```

This leads to the next question:

> **Can we store the frozen base model using fewer bits?**

Yes.

This is where **quantization** comes in.

---

# 3. What Is Quantization?

Quantization means:

> **Representing numerical values using fewer bits.**

For example:

```text
FP32
 ↓
FP16
 ↓
INT8
 ↓
INT4
```

Generally:

```text
Fewer bits
     ↓
Less memory
```

However, there is a trade-off:

```text
Fewer bits
     ↓
Potentially less numerical precision
```

So quantization is a balance between **memory efficiency** and **numerical accuracy**.

---

# 4. FP32

FP32 means:

\[
\boxed{\text{32-bit floating point}}
\]

Each parameter uses:

\[
32\text{ bits}=4\text{ bytes}
\]

Therefore:

```text
1 parameter
↓
4 bytes
```

---

# 5. FP16

FP16 means:

\[
\boxed{\text{16-bit floating point}}
\]

Each parameter uses:

\[
16\text{ bits}=2\text{ bytes}
\]

Compared with FP32:

```text
FP32 → 4 bytes
FP16 → 2 bytes
```

So the raw storage for the parameter values is approximately halved.

---

# 6. BF16

BF16 means:

\[
\boxed{\text{Brain Floating Point 16}}
\]

It is also a 16-bit floating-point representation.

Therefore:

\[
16\text{ bits}=2\text{ bytes}
\]

BF16 has a different allocation of bits between the exponent and significand compared with FP16, giving it a larger exponent range.

For modern hardware that supports it, BF16 is often useful for deep-learning computation.

---

# 7. Integer Quantization

Quantization can also use integer representations.

For example:

### INT8

\[
8\text{ bits}=1\text{ byte}
\]

### INT4

\[
4\text{ bits}=0.5\text{ byte}
\]

So:

```text
FP32 → 32 bits
FP16 → 16 bits
INT8 → 8 bits
INT4 → 4 bits
```

Ignoring overhead, fewer bits mean less storage.

---

# 8. Memory Comparison

Suppose we have:

\[
1\text{ billion parameters}
\]

Ignoring metadata and other runtime overhead:

| Representation | Bits / Parameter | Approx. Raw Memory |
|---|---:|---:|
| FP32 | 32 | 4 GB |
| FP16 | 16 | 2 GB |
| INT8 | 8 | 1 GB |
| INT4 | 4 | 0.5 GB |

The exact memory usage in real systems is higher because of:

- Metadata
- Tensor layout
- Runtime buffers
- Activations
- Other model components

But this table gives the core intuition.

---

# 9. Why Not Use INT4 for Everything?

If fewer bits reduce memory, you might ask:

> **Why not represent everything using the smallest possible number of bits?**

Because lower precision can cause information loss.

For example:

```text
Original value:

0.73458291
```

A low-precision representation cannot preserve every numerical detail.

Therefore:

\[
\boxed{
\text{Less precision}
\rightarrow
\text{Potential information loss}
}
\]

Quantization is therefore a trade-off:

```text
Fewer bits
    ↓
Less memory
    +
Potentially less numerical precision
```

---

# 10. Quantization Is More Than Simple Rounding

A beginner may think:

> "Quantization simply means rounding every number."

That is useful as a basic intuition, but modern quantization methods use more sophisticated mappings.

A simplified view is:

```text
High-precision values
        ↓
Quantization scheme
        ↓
Lower-precision representation
```

The exact mapping depends on the quantization method.

---

# 11. Quantizing Neural Network Weights

Suppose a layer contains:

\[
W=
\begin{bmatrix}
0.82 & -0.41\\
0.15 & 0.73
\end{bmatrix}
\]

Instead of storing every value using a high-precision representation:

```text
Original W
    ↓
Quantization
    ↓
Quantized W
```

The quantized representation uses fewer bits.

This reduces the memory required to store the model.

---

# 12. Quantization of the Frozen Base Model

Now connect this with LoRA.

Recall LoRA:

```text
Base Model
   ↓
Frozen

LoRA
   ↓
Trainable
```

Add quantization:

```text
Base Model
   ↓
Quantized
   ↓
Frozen

LoRA
   ↓
Trainable
```

This is the foundation of QLoRA.

---

# 13. What Is QLoRA?

QLoRA stands for:

\[
\boxed{\text{Quantized Low-Rank Adaptation}}
\]

The basic idea is:

> **Load the frozen pretrained model in a quantized representation and train LoRA adapters on top of it.**

Conceptually:

```text
Pretrained Model
       ↓
Quantization
       ↓
Frozen Base Model
       +
LoRA Adapters
       ↓
Train LoRA only
```

Therefore:

\[
\boxed{
QLoRA
=
\text{Quantized Base Model}
+
\text{LoRA}
}
\]

---

# 14. LoRA vs QLoRA

## LoRA

```text
Base Model
   ↓
FP16/BF16
   ↓
Frozen
   ↓
LoRA
   ↓
Train LoRA
```

## QLoRA

```text
Base Model
   ↓
Quantization
   ↓
Frozen
   ↓
LoRA
   ↓
Train LoRA
```

The major difference is the representation of the frozen base model.

---

# 15. What Problem Does Each Technique Solve?

### Full Fine-Tuning

Problem:

> Updating billions of parameters is expensive.

```text
Full Fine-Tuning
↓
Train everything
```

---

### LoRA

Problem:

> We do not want to train billions of parameters.

```text
LoRA
↓
Freeze base
↓
Train small low-rank adapters
```

---

### Quantization

Problem:

> We do not want to store the model using unnecessarily high precision.

```text
Quantization
↓
Use fewer bits
↓
Reduce memory
```

---

### QLoRA

Combines both:

```text
Quantization
      +
LoRA
      ↓
Memory-efficient fine-tuning
```

---

# 16. Why QLoRA Can Be More Memory Efficient

Suppose:

```text
Base Model
7B parameters
```

With ordinary LoRA:

```text
7B parameters
↓
Loaded in FP16/BF16
↓
Frozen
+
LoRA parameters
```

With QLoRA:

```text
7B parameters
↓
Loaded in a low-bit quantized representation
↓
Frozen
+
LoRA parameters
```

Therefore the memory needed for the frozen base model can be substantially reduced.

---

# 17. Important: QLoRA Does Not Train the Quantized Base Model

Typically:

```text
Quantized Base Model
        ↓
      Frozen
```

while:

```text
LoRA Adapters
        ↓
     Trainable
```

Therefore:

\[
\boxed{
\text{Train LoRA, not the quantized base weights}
}
\]

---

# 18. QLoRA Training Flow

```text
Input
  ↓
Quantized Frozen Base Model
  +
LoRA Adapter
  ↓
Forward Pass
  ↓
Prediction
  ↓
Loss
  ↓
Backward Pass
  ↓
Gradients for LoRA
  ↓
Update LoRA
  ↓
Repeat
```

The base model remains frozen throughout training.

---

# 19. 4-Bit Quantization

QLoRA is strongly associated with **4-bit quantization** of the frozen base model.

Why 4-bit?

Because it provides a large reduction in memory compared with FP16/BF16 while still allowing useful fine-tuning when combined with appropriate quantization methods and LoRA.

Conceptually:

```text
FP16
 ↓
2 bytes / parameter

INT4
 ↓
0.5 byte / represented value
```

Actual memory usage includes overhead.

---

# 20. BitsAndBytes

One commonly used library for quantized Transformer loading is:

\[
\boxed{\text{bitsandbytes}}
\]

It integrates with tools such as Hugging Face Transformers.

Typical installation:

```bash
pip install bitsandbytes
```

---

# 21. Loading a Model in 4-Bit

A typical Hugging Face workflow uses:

```python
from transformers import BitsAndBytesConfig
```

Then:

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True
)
```

This tells the model-loading process to use 4-bit quantization where supported.

---

# 22. Combining 4-Bit Quantization With LoRA

Conceptually:

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig
```

Define quantization:

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True
)
```

Then define LoRA:

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
```

Then attach the LoRA configuration to the model.

The exact code depends on the model architecture and library versions.

---

# 23. Typical QLoRA Configuration

A more complete example may look like:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True
)
```

Let's understand each component.

---

# 24. `load_in_4bit=True`

```python
load_in_4bit=True
```

means:

> Load the base model using 4-bit quantization.

This is the main memory-saving component of this configuration.

---

# 25. What Is NF4?

NF4 stands for:

\[
\boxed{\text{NormalFloat 4}}
\]

It is a 4-bit quantization data type designed for neural-network weights whose distributions are approximately normal.

Instead of treating the 4-bit representation as a generic integer representation, NF4 uses a quantization scheme designed for neural-network weights.

Conceptually:

```text
Neural-network weights
       ↓
NF4 quantization
       ↓
Efficient 4-bit representation
```

---

# 26. Why NF4?

Neural-network weights often have distributions that are approximately centered around zero and can resemble normal-like distributions.

NF4 is designed to allocate its representable values appropriately for such distributions.

Therefore:

```text
Neural-network weights
       ↓
NF4
       ↓
Efficient low-bit representation
```

---

# 27. `bnb_4bit_compute_dtype`

Example:

```python
bnb_4bit_compute_dtype="bfloat16"
```

This controls the compute dtype used for operations involving the quantized model.

Conceptually:

```text
Storage
  ↓
4-bit

Computation
  ↓
BF16
```

This gives an important distinction:

> **The model can be stored using a low-bit representation while computation can use a higher-precision format.**

---

# 28. Storage Precision vs Computation Precision

Do not assume:

> "If the model is 4-bit, every computation is performed using 4-bit values."

Not necessarily.

You can have:

```text
Model storage
      ↓
4-bit

Computation
      ↓
BF16 / FP16
```

This can provide memory savings while maintaining a more suitable numerical format for computation.

---

# 29. `bnb_4bit_use_double_quant`

Example:

```python
bnb_4bit_use_double_quant=True
```

This enables an additional quantization step for quantization constants/metadata.

Its purpose is to reduce memory overhead further.

Conceptually:

```text
4-bit quantization
      +
Quantization of quantization constants
      ↓
Additional memory savings
```

This is commonly used in QLoRA configurations.

---

# 30. The Four Important QLoRA Concepts

Remember:

```text
1. 4-bit quantization
2. NF4
3. Compute dtype
4. Double quantization
```

These are important when moving from basic LoRA to practical QLoRA.

---

# 31. QLoRA Architecture

A simplified architecture:

```text
                         Input
                           │
                           ↓
                ┌────────────────────┐
                │ Quantized Base LLM │
                │                    │
                │      FROZEN        │
                └─────────┬──────────┘
                          │
                          +
                    LoRA Adapters
                    A + B
                          │
                          ↓
                       Output
                          │
                          ↓
                         Loss
                          │
                          ↓
                    Backpropagation
                          │
                          ↓
                  Update LoRA only
```

---

# 32. Compare the Three Approaches

## Full Fine-Tuning

```text
Base Model
   ↓
FP32/FP16/BF16
   ↓
Train entire model
```

Main problem:

```text
Very high memory
Very high compute
```

---

## LoRA

```text
Base Model
   ↓
FP16/BF16
   ↓
Freeze
   ↓
Train LoRA
```

Main advantage:

```text
Much fewer trainable parameters
```

---

## QLoRA

```text
Base Model
   ↓
4-bit quantization
   ↓
Freeze
   ↓
Train LoRA
```

Main advantage:

```text
Lower base-model memory
+
Few trainable parameters
```

---

# 33. Complete Comparison

| Method | Base Model | Base Trainable? | Adapter | Main Advantage |
|---|---|---|---|---|
| Full FT | Full precision | Yes | No | Maximum direct adaptation |
| LoRA | Usually FP16/BF16 | No | Yes | Fewer trainable parameters |
| QLoRA | Quantized | No | Yes | Lower memory + parameter-efficient training |

---

# 34. Why QLoRA Is Useful on Consumer GPUs

Suppose you have a relatively limited GPU.

For example:

```text
8 GB VRAM
```

A large model may not fit comfortably in FP16.

QLoRA can reduce the memory required by the frozen base model:

```text
Large model
     ↓
4-bit loading
     ↓
Smaller memory footprint
     +
LoRA
     ↓
Train adapter
```

Whether a specific model fits depends on:

- Number of parameters
- Sequence length
- Batch size
- GPU memory
- Quantization implementation
- Runtime overhead

---

# 35. QLoRA Does Not Mean "4-Bit Training Everything"

A common misunderstanding is:

```text
Everything → INT4
```

That is not the correct mental model.

Instead:

```text
Base model
↓
Quantized representation
↓
Frozen

LoRA parameters
↓
Trainable
↓
Typically maintained in a suitable floating-point representation
```

The computation can also use an appropriate higher-precision dtype.

---

# 36. Why LoRA Parameters Are Not Quantized in the Same Way

LoRA parameters are the parameters we actually want to optimize.

Therefore, they generally use a suitable floating-point representation for training.

Conceptually:

```text
Base Model
↓
4-bit
↓
Frozen

LoRA
↓
FP16/BF16
↓
Trainable
```

This is a useful mental model for QLoRA.

---

# 37. Quantization Trade-Off

Quantization gives:

```text
Memory ↓
```

but can introduce:

```text
Precision loss
```

Therefore good quantization methods try to minimize the impact on model quality.

QLoRA is designed to make this trade-off practical for fine-tuning.

---

# 38. Important Memory Insight

Training memory consists of more than just model weights:

\[
Memory
\approx
Parameters
+
Gradients
+
Optimizer\ States
+
Activations
+
Temporary\ Memory
\]

QLoRA mainly attacks the **base model parameter storage** problem.

LoRA reduces the trainable-parameter-related memory.

Other techniques can still be useful:

```text
Gradient checkpointing
Gradient accumulation
Mixed precision
Sequence-length reduction
```

---

# 39. QLoRA + Gradient Checkpointing

You can combine:

```text
QLoRA
+
Gradient Checkpointing
```

Conceptually:

```text
4-bit Base Model
      ↓
Frozen
      +
LoRA
      ↓
Trainable

Gradient Checkpointing
      ↓
Reduce activation memory
```

Different techniques solve different memory problems.

---

# 40. QLoRA + Gradient Accumulation

You can also use:

```text
QLoRA
+
Gradient Accumulation
```

For example:

```text
Mini-batch = 2
Accumulation = 8
```

Effective batch size:

\[
2\times8=16
\]

This allows a larger effective batch without requiring the full batch to fit in memory simultaneously.

---

# 41. QLoRA + Mixed Precision

You may also use:

```text
4-bit storage
+
BF16 computation
```

Conceptually:

```text
Storage
↓
4-bit

Computation
↓
BF16
```

This combination is one reason QLoRA can be practical.

---

# 42. Complete Modern Fine-Tuning Stack

A memory-conscious LLM fine-tuning pipeline can look like:

```text
                Pretrained LLM
                       ↓
                 4-bit Quantization
                       ↓
                 Frozen Base Model
                       +
                    LoRA
                       ↓
                Train LoRA Parameters
                       +
                Gradient Checkpointing
                       +
                Gradient Accumulation
                       +
                BF16/FP16 Computation
                       ↓
                 Task-specific model
```

---

# 43. What Happens During Training?

Let's follow the process carefully.

### Step 1 — Load the pretrained model

```text
Pretrained LLM
```

### Step 2 — Quantize the base model

```text
Pretrained LLM
↓
4-bit representation
```

### Step 3 — Freeze the base model

```text
4-bit Base
↓
Frozen
```

### Step 4 — Add LoRA adapters

```text
Frozen Base
+
Trainable LoRA
```

### Step 5 — Input enters the model

```text
Input
↓
Quantized base
+
LoRA
```

### Step 6 — Compute prediction

```text
Prediction
```

### Step 7 — Calculate loss

```text
Prediction
↓
Loss
```

### Step 8 — Backpropagation

```text
Loss
↓
Gradients
↓
LoRA parameters
```

### Step 9 — Update LoRA

```text
A, B
↓
Updated
```

The quantized base model remains frozen.

---

# 44. What Happens to the Quantized Base During Backpropagation?

The base model participates in the forward computation.

But we generally do not update its quantized weights.

Conceptually:

```text
Forward:
Base + LoRA
      ↓
Prediction

Backward:
      ↓
Compute gradients needed for LoRA
      ↓
Update LoRA
```

The base remains frozen.

---

# 45. Simple Analogy

Imagine a huge encyclopedia.

```text
Huge Encyclopedia
```

You want to adapt it into a specialized assistant.

### Full Fine-Tuning

Rewrite huge portions of the encyclopedia.

### LoRA

Keep the encyclopedia and add a small set of learned notes.

### QLoRA

Compress the encyclopedia first and then add the learned notes.

```text
Full FT:
Encyclopedia → Rewrite

LoRA:
Encyclopedia + Notes

QLoRA:
Compressed Encyclopedia + Notes
```

This is only an intuition, not a mathematical equivalence.

---

# 46. The Three Questions You Should Ask

Whenever you see a fine-tuning method, ask:

### Question 1

**What happens to the base model?**

```text
Train?
Freeze?
Quantize?
```

### Question 2

**What parameters are trainable?**

```text
All?
LoRA?
Adapters?
```

### Question 3

**How is memory reduced?**

```text
Fewer trainable parameters?
Lower precision?
Gradient checkpointing?
Smaller batch?
```

For QLoRA:

```text
Base → Quantized + Frozen
Trainable → LoRA
Memory → Reduced through quantization + PEFT
```

---

# 47. Questions & Answers

## Q1. What is quantization?

### Answer

Quantization represents numerical values using fewer bits.

For example:

```text
FP32 → FP16 → INT8 → INT4
```

Lower-bit representations generally require less memory.

---

## Q2. Why does quantization save memory?

### Answer

Because each parameter requires fewer bits to represent.

For example:

\[
FP16=16\text{ bits}
\]

while:

\[
INT4=4\text{ bits}
\]

Ignoring overhead, INT4 requires one-quarter as many bits as FP16 for the represented values.

---

## Q3. What is QLoRA?

### Answer

QLoRA combines a quantized frozen base model with trainable LoRA adapters.

\[
\boxed{
QLoRA
=
Quantized\ Base\ Model
+
LoRA
}
\]

---

## Q4. What problem does LoRA solve?

### Answer

LoRA reduces the number of trainable parameters by freezing the base model and learning a low-rank adaptation.

---

## Q5. What problem does quantization solve?

### Answer

Quantization reduces the memory needed to represent model parameters by using fewer bits.

---

## Q6. What problem does QLoRA solve?

### Answer

QLoRA combines both ideas:

```text
Quantization
↓
Reduce base-model memory

LoRA
↓
Reduce trainable parameters
```

---

## Q7. Does QLoRA train the quantized base model?

### Answer

Typically, no.

The quantized base model is frozen, while the LoRA adapters are trained.

---

## Q8. What is NF4?

### Answer

NF4, or NormalFloat 4, is a 4-bit quantization data type designed for neural-network weights with approximately normal distributions.

---

## Q9. What is `bnb_4bit_compute_dtype`?

### Answer

It specifies the numerical dtype used for computation involving the quantized model.

For example:

```python
bnb_4bit_compute_dtype="bfloat16"
```

can use BF16 for computation while the model weights are stored in a 4-bit representation.

---

## Q10. What is double quantization?

### Answer

It applies an additional quantization step to quantization constants/metadata to reduce memory overhead.

---

## Q11. Is a 4-bit model performing every operation in 4-bit?

### Answer

No.

The storage representation and computation dtype can be different.

For example:

```text
Storage → 4-bit
Computation → BF16
```

---

## Q12. Why is QLoRA useful?

### Answer

It can make fine-tuning large models possible with substantially less GPU memory than full fine-tuning, especially when combined with other memory-saving techniques.

---

## Q13. Does QLoRA eliminate GPU memory requirements?

### Answer

No.

The model still requires memory for:

```text
Quantized base model
LoRA parameters
Activations
Temporary tensors
Training infrastructure
```

---

## Q14. Can gradient accumulation be combined with QLoRA?

### Answer

Yes.

Gradient accumulation can be combined with QLoRA to simulate a larger effective batch size.

---

## Q15. Can gradient checkpointing be combined with QLoRA?

### Answer

Yes.

Gradient checkpointing can reduce activation memory and can be combined with QLoRA.

---

# 48. Self-Test

Try answering these without looking back:

1. What is quantization?
2. Why does using fewer bits reduce memory?
3. What is the difference between FP32, FP16, INT8, and INT4?
4. Why can't we simply use the lowest possible number of bits?
5. What is QLoRA?
6. What is the difference between LoRA and QLoRA?
7. What happens to the base model in QLoRA?
8. What happens to the LoRA parameters in QLoRA?
9. What is NF4?
10. Why is NF4 useful for neural-network weights?
11. What is `bnb_4bit_compute_dtype`?
12. Why can storage precision and computation precision be different?
13. What is double quantization?
14. Why is 4-bit quantization useful for large LLMs?
15. Does QLoRA train the quantized base weights?
16. How does QLoRA reduce GPU memory?
17. How is QLoRA different from full fine-tuning?
18. Can gradient accumulation be combined with QLoRA?
19. Can gradient checkpointing be combined with QLoRA?
20. Why might someone choose QLoRA over ordinary LoRA?

---

# 49. Understanding Check

Before moving forward, make sure you can explain this:

## Full Fine-Tuning

```text
Base Model
↓
Train everything
```

## LoRA

```text
Base Model
↓
Freeze
+
LoRA
↓
Train LoRA
```

## QLoRA

```text
Base Model
↓
Quantize
↓
Freeze
+
LoRA
↓
Train LoRA
```

Therefore:

\[
\boxed{
\text{Full FT}
\neq
\text{LoRA}
\neq
\text{QLoRA}
}
\]

---

# 50. Final Mental Model

Remember this hierarchy:

```text
                    Fine-Tuning
                        │
          ┌─────────────┴─────────────┐
          ↓                           ↓
     Full Fine-Tuning              PEFT
                                      │
                                      ↓
                                    LoRA
                                      │
                                      ↓
                                   QLoRA
```

And remember what each one solves:

```text
Full Fine-Tuning
↓
Update everything


LoRA
↓
Don't update everything
↓
Freeze base + train low-rank adapters


Quantization
↓
Don't store everything at full precision
↓
Use fewer bits


QLoRA
↓
Quantized base
+
LoRA adapters
↓
Memory-efficient fine-tuning
```

The most important equation from this lesson is:

\[
\boxed{
QLoRA
=
\text{Quantized Frozen Base Model}
+
\text{Trainable LoRA}
}
\]

---

# 51. Lesson 10 Summary

## Quantization

\[
\boxed{
\text{Fewer bits}
\rightarrow
\text{Lower memory}
}
\]

## LoRA

\[
\boxed{
W_{\text{effective}}
=
W+
\frac{\alpha}{r}BA
}
\]

## QLoRA

\[
\boxed{
\text{Quantized Base}
+
\text{LoRA}
}
\]

## Practical QLoRA

```text
Base Model
    ↓
4-bit Quantization
    ↓
Frozen
    ↓
LoRA
    ↓
Train LoRA
    ↓
Task-specific model
```

---

# 52. What Comes Next?

## Lesson 11 — Practical QLoRA Fine-Tuning With Hugging Face

Now we move from **understanding the concepts** to actually building one.

We will connect:

```text
Transformers
        +
Datasets
        +
BitsAndBytes
        +
PEFT
        +
LoRA
        +
Trainer
```

and build the complete pipeline:

```text
Dataset
   ↓
Tokenizer
   ↓
Load 4-bit model
   ↓
Configure LoRA
   ↓
Attach LoRA adapters
   ↓
Check trainable parameters
   ↓
TrainingArguments
   ↓
Trainer
   ↓
Fine-tune
   ↓
Save adapter
   ↓
Load adapter
   ↓
Inference
```

This will be your first complete **practical LLM fine-tuning workflow**.
