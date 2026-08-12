# Fine-Tuning — Lesson 8: Practical Fine-Tuning, Checkpoints, Gradient Accumulation, Mixed Precision & GPU Memory

> **Goal:** Understand the practical techniques used to fine-tune Transformer models when GPU memory and compute are limited.

---

# 1. Why GPU Memory Matters

Transformer models can contain millions or billions of parameters.

During training, GPU memory is required for much more than model parameters.

A simplified view is:

```text
GPU Memory
├── Model Parameters
├── Gradients
├── Optimizer States
├── Activations
└── Temporary tensors
```

Therefore:

\[
\boxed{
\text{Training memory} > \text{Model parameter memory}
}
\]

---

# 2. Parameters Are Not the Only Memory Cost

Suppose a model contains:

```text
1 billion parameters
```

We cannot assume that only the memory required for those parameters is needed.

During training, we also need:

```text
Parameters
+
Gradients
+
Optimizer states
+
Activations
```

This is why a model that fits in GPU memory during inference may not fit during full fine-tuning.

---

# 3. Training vs Inference Memory

## Inference

During inference:

```text
Input
 ↓
Model
 ↓
Prediction
```

We generally do not need to retain the complete backward computational graph.

## Training

During training:

```text
Input
 ↓
Forward pass
 ↓
Loss
 ↓
Backward pass
 ↓
Gradients
 ↓
Parameter update
```

The backward pass requires additional memory.

Therefore:

\[
\boxed{
\text{Training usually requires substantially more memory than inference}
}
\]

---

# 4. Batch Size

Batch size means:

> **How many training examples are processed together before an optimizer update.**

Example:

```python
per_device_train_batch_size=8
```

means each device processes 8 examples per mini-batch.

---

# 5. Why Larger Batch Sizes Need More Memory

Suppose:

```text
Batch size = 8
```

The GPU processes 8 examples.

Now:

```text
Batch size = 32
```

The GPU processes 32 examples.

More examples generally require more activation memory.

Therefore:

\[
\boxed{
\text{Larger batch size} \rightarrow \text{More GPU memory}
}
\]

---

# 6. What If the GPU Cannot Handle the Desired Batch Size?

Suppose you want:

```text
Batch size = 32
```

but your GPU can only fit:

```text
Batch size = 8
```

You can use:

\[
\boxed{\text{Gradient Accumulation}}
\]

---

# 7. Gradient Accumulation

Gradient accumulation means:

> Process several smaller mini-batches and accumulate their gradients before performing one optimizer update.

Suppose:

```text
Batch size = 8
Gradient accumulation steps = 4
```

Then:

```text
Batch 1 → calculate gradients
Batch 2 → accumulate gradients
Batch 3 → accumulate gradients
Batch 4 → accumulate gradients
                  ↓
           optimizer.step()
```

The effective batch size is approximately:

\[
8\times4=32
\]

Therefore:

\[
\boxed{
\text{Effective Batch Size}
=
\text{Batch Size}
\times
\text{Gradient Accumulation Steps}
}
\]

For multiple devices:

\[
\boxed{
\text{Effective Batch Size}
=
\text{Per-device Batch Size}
\times
\text{Number of Devices}
\times
\text{Gradient Accumulation Steps}
}
\]

---

# 8. Example

Suppose:

```text
per_device_train_batch_size = 4
gradient_accumulation_steps = 8
```

For one device:

\[
4\times8=32
\]

So the effective batch size is approximately:

\[
\boxed{32}
\]

---

# 9. Why Gradient Accumulation Saves Memory

Without gradient accumulation:

```text
Batch = 32
```

The GPU must process all 32 examples in one mini-batch.

With accumulation:

```text
Batch = 8
Accumulation = 4
```

the GPU processes only 8 examples at a time:

```text
8 → gradients
8 → gradients
8 → gradients
8 → gradients
       ↓
optimizer update
```

Therefore, the instantaneous memory requirement can be much smaller.

---

# 10. Optimizer Step vs Mini-Batch

This distinction is important.

With:

```text
batch size = 8
gradient accumulation = 4
```

there are:

```text
4 forward/backward passes
```

before:

```text
1 optimizer update
```

So:

```text
Mini-batch
   ↓
Forward
   ↓
Backward
   ↓
Accumulate

Mini-batch
   ↓
Forward
   ↓
Backward
   ↓
Accumulate

...

After 4 mini-batches
   ↓
Optimizer step
```

---

# 11. PyTorch Gradient Accumulation

A simplified implementation:

```python
optimizer.zero_grad()

for step, batch in enumerate(dataloader):

    outputs = model(**batch)

    loss = outputs.loss

    loss = loss / accumulation_steps

    loss.backward()

    if (step + 1) % accumulation_steps == 0:

        optimizer.step()

        optimizer.zero_grad()
```

The important idea is:

```python
loss.backward()
```

happens multiple times before:

```python
optimizer.step()
```

---

# 12. Why Divide the Loss?

You will often see:

```python
loss = loss / accumulation_steps
```

Without this division, accumulated gradients can become larger simply because multiple mini-batches were summed.

Dividing the loss keeps the accumulated gradient approximately aligned with the gradient of the effective batch average.

---

# 13. Gradient Accumulation in Hugging Face

With `Trainer`:

```python
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8
)
```

Conceptually:

```text
4 examples × 8 accumulation steps
=
effective batch size ≈ 32
```

---

# 14. Gradient Accumulation Does Not Increase GPU VRAM

This is important.

Suppose:

```text
GPU VRAM = 8 GB
```

Gradient accumulation does not turn it into:

```text
32 GB VRAM
```

Instead, it allows you to simulate a larger effective batch by processing smaller batches sequentially.

---

# 15. What Is Gradient Clipping?

During training, gradients can sometimes become extremely large.

This can cause unstable updates.

This is called:

\[
\boxed{\text{Exploding Gradients}}
\]

Gradient clipping limits the gradient magnitude.

---

# 16. Gradient Clipping Intuition

Suppose the gradient norm is extremely large.

Instead of allowing an enormous update:

```text
Huge gradient
     ↓
Gradient clipping
     ↓
Controlled gradient
```

A common PyTorch operation is:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

This limits the overall gradient norm.

---

# 17. Gradient Clipping in a Training Loop

A common ordering is:

```python
loss.backward()

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

optimizer.step()
optimizer.zero_grad()
```

So:

```text
Backward
   ↓
Gradients
   ↓
Clip gradients
   ↓
Optimizer update
```

---

# 18. What Is Mixed Precision?

Modern GPUs can perform calculations using lower numerical precision than FP32.

Common formats include:

```text
FP32
FP16
BF16
```

Mixed precision means using different numerical precisions strategically to improve efficiency.

---

# 19. FP32

FP32 means:

\[
\boxed{\text{32-bit floating point}}
\]

It is commonly called:

```text
float32
```

It provides relatively high numerical precision, but requires more memory than lower-precision formats.

---

# 20. FP16

FP16 means:

\[
\boxed{\text{16-bit floating point}}
\]

It uses fewer bits than FP32.

Therefore:

```text
FP32
 ↓
More memory

FP16
 ↓
Less memory
```

FP16 can also provide faster computation on supported hardware.

---

# 21. BF16

BF16 means:

\[
\boxed{\text{Brain Floating Point 16}}
\]

It is also a 16-bit floating-point format.

Compared with FP16, BF16 has a larger exponent range.

BF16 is particularly useful on hardware that supports it.

---

# 22. FP32 vs FP16 vs BF16

| Format | Bits | Memory | Key characteristic |
|---|---:|---:|---|
| FP32 | 32 | Higher | Higher numerical precision |
| FP16 | 16 | Lower | Lower memory usage |
| BF16 | 16 | Lower | Larger exponent range than FP16 |

The exact performance depends on the GPU, framework, and workload.

---

# 23. Why Use Mixed Precision?

Using appropriate lower-precision calculations can provide:

```text
↓ GPU memory usage
↑ Throughput
↑ Training speed
```

when supported by the hardware and workload.

---

# 24. Why Not Always Use FP16?

Lower precision introduces numerical considerations.

Very small values can underflow, while very large values can overflow.

Therefore, frameworks use techniques such as:

```text
Automatic mixed precision
Loss scaling
```

to make low-precision training more reliable.

---

# 25. Automatic Mixed Precision

PyTorch provides automatic mixed precision through:

```python
torch.autocast(...)
```

For example:

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.float16
):
    outputs = model(**batch)
    loss = outputs.loss
```

The framework chooses appropriate precision for supported operations.

---

# 26. Gradient Scaling

For FP16 training, gradient scaling can help prevent numerical underflow.

Conceptually:

```text
Loss
 ↓
Scale loss
 ↓
Backward
 ↓
Scaled gradients
 ↓
Unscale
 ↓
Optimizer step
```

PyTorch provides tools for this purpose.

The exact API can vary with the PyTorch version and training setup.

---

# 27. BF16 vs FP16

A practical rule:

```text
If hardware supports BF16 well:
    BF16 is often a convenient choice.

Otherwise:
    FP16 may be useful.
```

The exact choice depends on:

- GPU architecture
- Framework version
- Model
- Numerical stability
- Performance

---

# 28. Mixed Precision in Hugging Face

With `TrainingArguments`, you may see:

```python
training_args = TrainingArguments(
    output_dir="./results",
    bf16=True
)
```

or:

```python
training_args = TrainingArguments(
    output_dir="./results",
    fp16=True
)
```

Do not enable both blindly.

The available option depends on your hardware.

---

# 29. GPU Memory Components

A simplified training memory model is:

\[
\boxed{
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
}
\]

Reducing only parameter precision does not solve every memory problem.

---

# 30. Activations

During the forward pass, intermediate values are produced.

For example:

```text
Input
 ↓
Embedding
 ↓
Attention
 ↓
Feed Forward
 ↓
Output
```

Intermediate results may need to be retained for backpropagation.

These are called **activations**.

---

# 31. Why Sequence Length Matters

Suppose:

```text
Sequence length = 128
```

versus:

```text
Sequence length = 2048
```

The longer sequence can require significantly more memory.

For standard Transformer self-attention, attention-score computation has approximately:

\[
O(n^2)
\]

complexity with respect to sequence length \(n\).

Therefore:

```text
Longer sequence
      ↓
More attention computation
      ↓
More memory pressure
```

---

# 32. Batch Size + Sequence Length

GPU memory depends strongly on both:

```text
Batch Size
+
Sequence Length
```

For example:

```text
Batch = 8
Sequence = 128
```

may fit comfortably.

But:

```text
Batch = 8
Sequence = 2048
```

can require much more memory.

Therefore, when GPU memory is limited, reducing sequence length can be highly effective.

---

# 33. Padding and Memory

Suppose a batch contains:

```text
"The cat sat"

"A very long sentence containing many more tokens..."
```

If padded to the longest sequence, the shorter example receives padding tokens.

Therefore, inefficient padding can waste computation and memory.

Dynamic padding can help:

```text
Batch
 ↓
Find longest sequence in this batch
 ↓
Pad only to that length
```

This is one reason data collators are useful.

---

# 34. Checkpointing

There are two different concepts that are both commonly called "checkpointing":

1. **Saving model/training checkpoints**
2. **Gradient/activation checkpointing**

They are not the same thing.

---

# 35. Model Training Checkpoint

A model checkpoint is a saved state of the model during training.

Examples:

```text
checkpoint-500
checkpoint-1000
checkpoint-1500
```

A checkpoint can contain information such as:

```text
Model weights
Optimizer state
Scheduler state
Training state
```

This allows training to resume later.

---

# 36. Why Save Checkpoints?

Suppose training takes:

```text
10 hours
```

and the machine crashes after:

```text
8 hours
```

Without checkpoints:

```text
Potentially restart from the beginning
```

With checkpoints:

```text
Resume from saved state
```

Therefore:

\[
\boxed{\text{Checkpointing protects training progress}}
\]

---

# 37. Hugging Face Checkpoints

Example:

```python
training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="steps",
    save_steps=500
)
```

This can save checkpoints periodically.

You can also use:

```python
save_strategy="epoch"
```

to save after each epoch.

---

# 38. Loading a Checkpoint

Conceptually:

```python
trainer.train(
    resume_from_checkpoint="./results/checkpoint-1000"
)
```

This allows training to continue from the saved state.

---

# 39. Best Checkpoint

Checkpoint saving can also be combined with evaluation.

For example:

```python
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)
```

The trainer can select the best checkpoint according to the evaluation configuration.

This connects directly to Lesson 6.

---

# 40. Gradient/Activation Checkpointing

This is different from saving checkpoints.

**Gradient checkpointing** is a memory-saving technique used during training.

Instead of storing every activation needed for backpropagation:

```text
Store everything
```

the model stores fewer activations and recomputes some of them during the backward pass.

---

# 41. Why Does Gradient Checkpointing Save Memory?

Normally:

```text
Forward
 ↓
Store many activations
 ↓
Backward
```

With gradient checkpointing:

```text
Forward
 ↓
Store fewer activations
 ↓
Backward
 ↓
Recompute some activations
```

Therefore:

```text
Memory ↓
Computation ↑
```

This is a classic trade-off:

\[
\boxed{
\text{Less memory at the cost of additional computation}
}
\]

---

# 42. Enabling Gradient Checkpointing

With Hugging Face models:

```python
model.gradient_checkpointing_enable()
```

Or in some training configurations:

```python
training_args = TrainingArguments(
    output_dir="./results",
    gradient_checkpointing=True
)
```

The exact API depends on the Transformers version and training setup.

---

# 43. Checkpoint vs Gradient Checkpointing

This distinction is extremely important.

### Training checkpoint

```text
Save model state
     ↓
Resume training later
```

### Gradient checkpointing

```text
Save fewer activations
     ↓
Reduce GPU memory
     ↓
Recompute during backward
```

They solve different problems.

---

# 44. Practical Memory Optimization Strategy

If a Transformer does not fit in GPU memory, possible strategies include:

```text
1. Reduce batch size
        ↓
2. Use gradient accumulation
        ↓
3. Use mixed precision
        ↓
4. Reduce sequence length
        ↓
5. Enable gradient checkpointing
        ↓
6. Consider parameter-efficient fine-tuning
```

The exact order depends on the model and hardware.

---

# 45. Example: Limited GPU

Suppose:

```text
GPU VRAM = 8 GB
```

You want:

```text
Effective batch size = 32
```

but batch 32 doesn't fit.

You could try:

```python
training_args = TrainingArguments(
    output_dir="./results",

    per_device_train_batch_size=4,

    gradient_accumulation_steps=8,

    fp16=True,

    gradient_checkpointing=True
)
```

Conceptually:

```text
Batch size = 4
Accumulation = 8

Effective batch ≈ 32

Mixed precision
       ↓
Less memory

Gradient checkpointing
       ↓
Less activation memory
```

---

# 46. What If It Still Doesn't Fit?

You may need to reduce:

```text
Batch size
```

or:

```text
Sequence length
```

For example:

```text
max_length = 512
```

instead of:

```text
max_length = 1024
```

You could also consider:

```text
LoRA
QLoRA
Quantization
```

These will be covered later.

---

# 47. Gradient Accumulation and Effective Batch Size

For one device:

\[
\boxed{
Effective\ Batch\ Size
=
PerDeviceBatchSize
\times
GradientAccumulationSteps
}
\]

For multiple devices:

\[
\boxed{
Effective\ Batch\ Size
=
PerDeviceBatchSize
\times
NumberOfDevices
\times
GradientAccumulationSteps
}
\]

Example:

```text
Per-device batch = 4
Devices = 2
Accumulation = 4
```

Then:

\[
4\times2\times4=32
\]

effective examples per optimizer update.

---

# 48. Important: Batch Size Does Not Mean Optimizer Update

Suppose:

```text
batch = 4
accumulation = 8
```

There are:

```text
8 mini-batches
```

before one optimizer update.

Therefore:

```text
4 examples
×
8 mini-batches
=
32 effective examples
```

---

# 49. Training Step vs Optimizer Step

These terms can be confusing.

A **forward/backward iteration** processes one mini-batch.

An **optimizer step** updates parameters.

With gradient accumulation:

```text
Mini-batch 1 → backward
Mini-batch 2 → backward
Mini-batch 3 → backward
Mini-batch 4 → backward
       ↓
optimizer.step()
```

So one optimizer step may correspond to multiple mini-batches.

---

# 50. Practical Hugging Face Configuration

A memory-conscious setup might look like:

```python
training_args = TrainingArguments(
    output_dir="./results",

    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    gradient_accumulation_steps=8,

    learning_rate=2e-5,

    weight_decay=0.01,

    num_train_epochs=3,

    warmup_ratio=0.1,

    fp16=True,

    gradient_checkpointing=True,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True
)
```

This is an example, not a universal configuration.

Your GPU and model determine which options are appropriate.

---

# 51. Complete Practical Pipeline

```text
                 Dataset
                    ↓
               Tokenization
                    ↓
              Dynamic Padding
                    ↓
                 Batch
                    ↓
          Mixed Precision
                    ↓
               Forward Pass
                    ↓
                  Loss
                    ↓
             Backward Pass
                    ↓
            Gradient Clipping
                    ↓
        Gradient Accumulation
                    ↓
             Optimizer Step
                    ↓
            Scheduler Step
                    ↓
               Checkpoint
                    ↓
               Evaluation
                    ↓
                 Repeat
```

---

# 52. Memory Optimization Mental Model

Think of GPU memory as a limited budget.

```text
                 GPU VRAM
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
   Parameters   Activations  Optimizer
                               States
```

You can reduce pressure by:

```text
Parameters
   ↓
LoRA / Quantization

Activations
   ↓
Smaller batch
   ↓
Shorter sequence
   ↓
Gradient checkpointing
   ↓
Mixed precision

Optimizer states
   ↓
Memory-efficient optimizers / PEFT
```

---

# 53. Important Questions & Answers

## Q1. What is gradient accumulation?

### Answer

Gradient accumulation processes several smaller mini-batches and accumulates their gradients before performing one optimizer update.

---

## Q2. What is effective batch size?

### Answer

For one device:

\[
Effective\ Batch\ Size
=
Batch\ Size
\times
Accumulation\ Steps
\]

For multiple devices:

\[
Effective\ Batch\ Size
=
Batch\ Size
\times
Devices
\times
Accumulation\ Steps
\]

---

## Q3. Why does gradient accumulation help?

### Answer

It allows you to simulate a larger effective batch while keeping each individual mini-batch small enough to fit in GPU memory.

---

## Q4. Does gradient accumulation increase GPU VRAM?

### Answer

No.

It reduces the instantaneous mini-batch size while accumulating gradients across multiple batches.

---

## Q5. What is mixed precision?

### Answer

Mixed precision uses lower-precision numerical formats such as FP16 or BF16 for appropriate operations while maintaining training correctness through framework techniques.

---

## Q6. What is FP32?

### Answer

FP32 is 32-bit floating-point representation.

---

## Q7. What is FP16?

### Answer

FP16 is 16-bit floating-point representation. It uses less memory than FP32.

---

## Q8. What is BF16?

### Answer

BF16 is a 16-bit floating-point format with a larger exponent range than FP16. It is particularly useful on hardware that supports it.

---

## Q9. Why use mixed precision?

### Answer

Potential benefits include:

```text
Lower memory usage
Higher throughput
Faster training
```

when supported by the hardware and workload.

---

## Q10. What is gradient clipping?

### Answer

Gradient clipping limits the magnitude of gradients to prevent excessively large updates.

Example:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

---

## Q11. What is a training checkpoint?

### Answer

A training checkpoint saves model and training state so training can be resumed later.

---

## Q12. What is gradient checkpointing?

### Answer

Gradient checkpointing is a memory-saving technique that stores fewer activations and recomputes some of them during backpropagation.

---

## Q13. Are training checkpoints and gradient checkpointing the same?

### Answer

No.

```text
Training checkpoint
→ Save training state

Gradient checkpointing
→ Save GPU memory during training
```

---

## Q14. Why does sequence length affect memory?

### Answer

Transformer self-attention has approximately quadratic complexity with respect to sequence length:

\[
O(n^2)
\]

Therefore, increasing sequence length can significantly increase computation and memory requirements.

---

## Q15. What should you try if a model does not fit in GPU memory?

### Answer

Possible strategies include:

```text
Reduce batch size
↓
Use gradient accumulation
↓
Use FP16/BF16
↓
Reduce sequence length
↓
Use gradient checkpointing
↓
Use LoRA/QLoRA
↓
Use quantization
```

---

# 54. Self-Test

Try answering these without looking back:

1. Why does training require more memory than inference?
2. What is batch size?
3. What is gradient accumulation?
4. How do you calculate effective batch size?
5. Why does gradient accumulation help with limited VRAM?
6. What is the difference between a mini-batch and an optimizer step?
7. What is gradient clipping?
8. What is mixed precision?
9. What is FP32?
10. What is FP16?
11. What is BF16?
12. Why might BF16 be preferred over FP16 on supported hardware?
13. What is a training checkpoint?
14. What is gradient checkpointing?
15. What is the difference between checkpoint saving and gradient checkpointing?
16. Why does sequence length affect Transformer memory?
17. How can you reduce GPU memory usage during fine-tuning?
18. What happens when gradient accumulation is increased?
19. Does gradient accumulation change the number of examples processed?
20. Why might LoRA become useful when full fine-tuning doesn't fit in memory?

---

# 55. Final Mental Model

The complete practical fine-tuning process is:

```text
                    PRETRAINED MODEL
                           ↓
                        Dataset
                           ↓
                      Tokenization
                           ↓
                      Mini-Batches
                           ↓
                    ┌──────────────┐
                    │ Forward Pass │
                    └──────┬───────┘
                           ↓
                          Loss
                           ↓
                   loss.backward()
                           ↓
                       Gradients
                           ↓
                  Gradient Clipping
                           ↓
                Gradient Accumulation
                           ↓
                    Optimizer Step
                           ↓
                   Scheduler Step
                           ↓
                     Checkpoint
                           ↓
                     Evaluation
                           ↓
                        Repeat
```

Main memory-saving tools:

\[
\boxed{\text{Smaller Batch}}
\]

\[
\boxed{\text{Gradient Accumulation}}
\]

\[
\boxed{\text{Mixed Precision}}
\]

\[
\boxed{\text{Shorter Sequence Length}}
\]

\[
\boxed{\text{Gradient Checkpointing}}
\]

\[
\boxed{\text{LoRA / QLoRA}}
\]

---

# 56. What You Should Remember From Lesson 8

## 1. Gradient Accumulation

\[
\boxed{
Effective\ Batch
=
MiniBatch
\times
Accumulation
\times
Devices
}
\]

## 2. Mixed Precision

```text
FP32
 ↓
FP16 / BF16
 ↓
Lower memory + potentially faster computation
```

## 3. Gradient Clipping

```text
Huge gradients
      ↓
Clip
      ↓
Controlled updates
```

## 4. Gradient Checkpointing

```text
Store fewer activations
      ↓
Recompute during backward
      ↓
Lower memory
      +
More computation
```

## 5. Training Checkpoints

```text
Save training state
      ↓
Resume training later
```

---

# 57. Next Lesson

## Lesson 9 — LoRA & PEFT: Parameter-Efficient Fine-Tuning

This is one of the most important lessons for modern LLM fine-tuning.

We will learn:

```text
Full Fine-Tuning
       ↓
Why it becomes expensive
       ↓
Freeze pretrained model
       ↓
LoRA
       ↓
Low-Rank Matrices
       ↓
Train only small number of parameters
       ↓
PEFT
       ↓
QLoRA
```

The central question will be:

> **If a Transformer has billions of parameters, do we really need to update all of them to adapt it to a new task?**

This leads directly to **LoRA (Low-Rank Adaptation)** and modern parameter-efficient fine-tuning.
