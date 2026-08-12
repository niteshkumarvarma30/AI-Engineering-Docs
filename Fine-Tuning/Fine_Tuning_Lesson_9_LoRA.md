# Lesson 9 — LoRA: Low-Rank Adaptation & PEFT

> **Goal:** Understand LoRA from first principles, without starting with complicated mathematics.
>
> The central question is:
>
> **If a pretrained model has billions of parameters, do we really need to update all of them to adapt the model to a new task?**

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
```

LoRA is one of the most important techniques used for modern LLM fine-tuning.

---

# 2. The Problem With Full Fine-Tuning

Suppose we have a pretrained model containing billions of parameters.

For simplicity, imagine one layer contains a weight matrix:

\[
W
\]

During normal/full fine-tuning:

```text
Pretrained W
     ↓
Training
     ↓
Gradient
     ↓
W changes
```

Mathematically:

\[
W_{\text{new}} = W + \Delta W
\]

where:

- \(W\) = original pretrained weight matrix
- \(\Delta W\) = learned change/update

The problem is that \(W\) can be extremely large.

---

# 3. Why Full Fine-Tuning Is Expensive

Suppose:

\[
W\in\mathbb{R}^{4096\times4096}
\]

Then the number of parameters is:

\[
4096\times4096=16,777,216
\]

That is more than **16.7 million parameters in just one matrix**.

A large language model contains many such matrices.

Therefore, full fine-tuning may require:

```text
Huge number of parameters
        ↓
Gradients for those parameters
        ↓
Optimizer states for those parameters
        ↓
Large GPU memory requirement
        ↓
High computational cost
```

This leads to the central question:

> **Do we really need to learn a completely new huge weight matrix to adapt the model?**

---

# 4. The Basic Idea Behind PEFT

PEFT stands for:

\[
\boxed{\text{Parameter-Efficient Fine-Tuning}}
\]

The idea is:

> **Freeze most of the pretrained model and train only a small number of additional or selected parameters.**

Instead of:

```text
Entire model
     ↓
Train everything
```

we do:

```text
Pretrained model
     ↓
Freeze most parameters
     ↓
Add/train small parameter set
     ↓
Fine-tune
```

---

# 5. What Does "Freeze" Mean?

Suppose a parameter has value:

\[
W=0.5
\]

If that parameter is frozen:

```text
Gradient
   ↓
Parameter is NOT updated
```

So:

\[
W=0.5
\]

remains unchanged during fine-tuning.

In PyTorch, a parameter can be frozen with:

```python
parameter.requires_grad = False
```

---

# 6. Frozen Does NOT Mean Unused

This is extremely important.

A frozen parameter is still used during the forward pass.

"Frozen" only means:

> **The parameter is not updated during training.**

Therefore:

```text
Frozen parameters
       ↓
Still used in forward pass
       ↓
But not updated
```

---

# 7. Full Fine-Tuning vs PEFT

## Full Fine-Tuning

```text
Pretrained Model
       ↓
All/most parameters trainable
       ↓
Gradients
       ↓
Update parameters
```

## PEFT

```text
Pretrained Model
       ↓
Freeze most parameters
       ↓
Small trainable component
       ↓
Update only that component
```

Therefore:

\[
\boxed{
\text{PEFT} \rightarrow \text{Much fewer trainable parameters}
}
\]

---

# 8. What Is LoRA?

LoRA stands for:

\[
\boxed{\text{Low-Rank Adaptation}}
\]

LoRA is a parameter-efficient fine-tuning technique.

The key idea is:

> **Keep the original pretrained weight matrix frozen and learn a small low-rank update instead.**

Suppose the original weight matrix is:

\[
W
\]

Normal fine-tuning learns:

\[
\Delta W
\]

directly.

LoRA instead represents that update as:

\[
\boxed{
\Delta W=BA
}
\]

Therefore:

\[
\boxed{
W_{\text{effective}}=W+BA
}
\]

This is the central equation of LoRA.

---

# 9. The Simplest Way to Understand LoRA

Normal fine-tuning:

```text
W
↓
TRAIN
↓
W changes
```

LoRA:

```text
W
↓
FREEZE
↓
W does not change

A + B
↓
TRAIN
↓
Learn the adaptation
```

So:

\[
\boxed{
\text{Full FT: train }W
}
\]

while:

\[
\boxed{
\text{LoRA: freeze }W,\text{ train }A,B
}
\]

---

# 10. Why Does LoRA Work If W Is Frozen?

This is one of the most important questions.

You may wonder:

> If the original model weights \(W\) never change, how can the model learn?

Because the effective weight used by the layer is:

\[
W_{\text{effective}}=W+BA
\]

Although \(W\) is frozen:

```text
W → unchanged
```

the LoRA matrices change:

```text
A → changes
B → changes
```

Therefore:

\[
BA
\]

changes.

Consequently:

\[
W+BA
\]

changes.

So the **effective behavior of the model changes** even though the original pretrained weights remain frozen.

---

# 11. The Core LoRA Picture

```text
                 PRETRAINED MODEL
                       │
                       │
                 W = FROZEN
                       │
                       ↓
              ┌─────────────────┐
Input ───────→│ Original Layer  │
              └────────┬────────┘
                       │
                       │
              ┌────────┴────────┐
              │                 │
              ↓                 ↓
             Wx                BAx
              │                 │
              └────────┬────────┘
                       ↓
                    Wx + BAx
                       ↓
                     Output
```

The original model contributes:

\[
Wx
\]

while LoRA contributes:

\[
BAx
\]

The final output is:

\[
\boxed{
y=Wx+BAx
}
\]

which can also be written as:

\[
\boxed{
y=(W+BA)x
}
\]

---

# 12. Why Two Matrices A and B?

Instead of directly learning the large update:

\[
\Delta W
\]

LoRA learns two smaller matrices:

\[
A
\]

and:

\[
B
\]

such that:

\[
\Delta W=BA
\]

The dimensions are chosen so that \(BA\) has exactly the same shape as \(W\).

---

# 13. Matrix Dimensions

Suppose:

\[
W\in\mathbb{R}^{d\times k}
\]

LoRA introduces:

\[
A\in\mathbb{R}^{r\times k}
\]

and:

\[
B\in\mathbb{R}^{d\times r}
\]

where:

\[
r\ll d,k
\]

Now multiply:

\[
BA
\]

The dimensions are:

\[
(d\times r)(r\times k)
\]

Therefore:

\[
BA\in\mathbb{R}^{d\times k}
\]

which is exactly the same shape as \(W\).

So we can add:

\[
W+BA
\]

---

# 14. Why Is It Called "Low Rank"?

The value:

\[
r
\]

is called the **rank** or LoRA rank.

The important idea is:

\[
r\ll d,k
\]

For example:

```text
Original matrix
4096 × 4096

LoRA rank
r = 8
```

Instead of learning a huge \(4096\times4096\) update directly, LoRA learns:

```text
A = 8 × 4096
B = 4096 × 8
```

The narrow dimension:

\[
r=8
\]

makes the update low-rank.

---

# 15. Parameter Comparison

Suppose:

\[
W\in\mathbb{R}^{4096\times4096}
\]

### Full update

A complete update matrix would contain:

\[
4096\times4096
\]

parameters:

\[
16,777,216
\]

### LoRA with \(r=8\)

Matrix \(A\):

\[
8\times4096=32,768
\]

Matrix \(B\):

\[
4096\times8=32,768
\]

Total:

\[
32,768+32,768
=
65,536
\]

Compare:

```text
Full update
≈ 16.8 million parameters

LoRA
≈ 65.5 thousand parameters
```

That is dramatically smaller.

---

# 16. Why This Saves Memory

With full fine-tuning:

```text
Large pretrained parameters
        ↓
Gradients for many parameters
        ↓
Optimizer states for many parameters
        ↓
Large memory requirement
```

With LoRA:

```text
Large pretrained parameters
        ↓
Frozen
        ↓
No trainable gradients for them

Small LoRA parameters
        ↓
Trainable
        ↓
Gradients + optimizer states
```

Therefore:

\[
\boxed{
\text{Much fewer trainable parameters}
}
\]

and potentially much lower training memory.

---

# 17. LoRA Does NOT Replace the Base Model

A common misunderstanding is:

> "LoRA replaces the original Transformer with two tiny matrices."

That is incorrect.

The original model is still there.

The correct picture is:

```text
Original Transformer
        +
Small LoRA Updates
        ↓
Task-specific behavior
```

The pretrained model remains the foundation.

---

# 18. LoRA Does NOT Mean Only A and B Exist

The original model still contains all of its parameters.

For example:

```text
Original model:

W1
W2
W3
W4
...
W1000
```

During LoRA fine-tuning:

```text
W1
W2
W3
...
W1000
↓
FROZEN

+

A1 B1
A2 B2
A3 B3
...
↓
TRAINABLE
```

Only the LoRA parameters are trained.

---

# 19. Numerical Example

Suppose:

\[
W=
\begin{bmatrix}
1&2\\
3&4
\end{bmatrix}
\]

LoRA learns:

\[
A=
\begin{bmatrix}
0.1&0.2
\end{bmatrix}
\]

and:

\[
B=
\begin{bmatrix}
0.3\\
0.4
\end{bmatrix}
\]

Now calculate:

\[
BA
=
\begin{bmatrix}
0.3\\
0.4
\end{bmatrix}
\begin{bmatrix}
0.1&0.2
\end{bmatrix}
\]

Therefore:

\[
BA=
\begin{bmatrix}
0.03&0.06\\
0.04&0.08
\end{bmatrix}
\]

The effective weight becomes:

\[
W_{\text{effective}}=W+BA
\]

Therefore:

\[
W_{\text{effective}}
=
\begin{bmatrix}
1.03&2.06\\
3.04&4.08
\end{bmatrix}
\]

Notice:

```text
Original W
    ↓
Frozen

BA
    ↓
Learned adaptation

W + BA
    ↓
Effective transformation
```

---

# 20. LoRA Forward Pass

Suppose a normal layer performs:

\[
y=Wx
\]

With LoRA:

\[
y=(W+BA)x
\]

Expanding:

\[
y=Wx+BAx
\]

Therefore:

```text
Original pathway
       ↓
      Wx

LoRA pathway
       ↓
      BAx

Combine
       ↓
    Wx + BAx
```

This is the mathematical heart of LoRA.

---

# 21. LoRA Scaling

A common LoRA formulation includes a scaling factor:

\[
\boxed{
W_{\text{effective}}
=
W+
\frac{\alpha}{r}BA
}
\]

where:

- \(r\) = LoRA rank
- \(\alpha\) = LoRA scaling parameter

Therefore:

\[
y=
Wx+
\frac{\alpha}{r}BAx
\]

The simplified equation:

\[
W+BA
\]

is useful for understanding the concept.

The more complete common formulation includes:

\[
\frac{\alpha}{r}
\]

---

# 22. LoRA Rank \(r\)

The rank controls the capacity of the LoRA update.

Common example values include:

```text
r = 4
r = 8
r = 16
r = 32
r = 64
```

Generally:

### Smaller \(r\)

```text
Fewer parameters
↓
Less memory
↓
Less adaptation capacity
```

### Larger \(r\)

```text
More parameters
↓
More memory
↓
More adaptation capacity
```

The best rank depends on the model, dataset, and task.

---

# 23. LoRA Alpha

A common scaling factor is:

\[
\frac{\alpha}{r}
\]

For example:

\[
r=8
\]

and:

\[
\alpha=16
\]

Then:

\[
\frac{\alpha}{r}
=
\frac{16}{8}
=
2
\]

So the LoRA update is scaled by 2 in this example.

---

# 24. LoRA Dropout

LoRA can also use dropout:

```python
lora_dropout = 0.05
```

The purpose is to provide regularization to the trainable adaptation and potentially reduce overfitting.

---

# 25. Where Is LoRA Applied in a Transformer?

Recall the Transformer attention projections:

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

LoRA can be applied to selected projection matrices.

For example:

```text
Attention
├── Query projection
├── Key projection
├── Value projection
└── Output projection
```

A common configuration may target:

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

The exact target module names depend on the model architecture.

---

# 26. LoRA Applied to Query Projection

Original:

\[
Q=XW_Q
\]

With LoRA:

\[
\boxed{
Q=X(W_Q+B_QA_Q)
}
\]

Expanding:

\[
Q=XW_Q+XB_QA_Q
\]

Therefore:

```text
Original Q transformation
        +
LoRA Q adaptation
        ↓
New Q
```

---

# 27. LoRA Applied to Value Projection

Original:

\[
V=XW_V
\]

With LoRA:

\[
\boxed{
V=X(W_V+B_VA_V)
}
\]

Again, the original \(W_V\) remains frozen while the LoRA matrices are trainable.

---

# 28. Connecting LoRA to Self-Attention

You already know:

\[
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

Suppose LoRA modifies the Query and Value projections.

Then:

```text
X
 ↓
LoRA-adjusted Q projection
 ↓
Q

X
 ↓
LoRA-adjusted V projection
 ↓
V

Q, K, V
 ↓
Self-Attention
 ↓
Output
```

The model can therefore learn task-specific changes to its attention transformations.

---

# 29. What Happens During LoRA Training?

Consider one training example:

```text
Input
 ↓
LoRA-enabled model
 ↓
Prediction
 ↓
Loss
```

Then:

```text
Loss
 ↓
Backward pass
 ↓
Gradients
```

But:

```text
Original W
↓
Frozen
```

So the trainable parameters are primarily:

```text
A
B
```

Then:

```text
optimizer.step()
```

updates the LoRA parameters.

---

# 30. Complete LoRA Training Flow

```text
Input
 ↓
Frozen Pretrained Model
 +
LoRA Adapters
 ↓
Prediction
 ↓
Loss
 ↓
Backward
 ↓
Gradients
 ↓
LoRA Parameters
 ↓
Optimizer
 ↓
Update A and B
 ↓
Repeat
```

This is the main training process to remember.

---

# 31. LoRA and the Optimizer

With full fine-tuning:

```text
Optimizer
     ↓
States for huge number of trainable parameters
```

With LoRA:

```text
Optimizer
     ↓
States mainly for trainable LoRA parameters
```

Therefore optimizer memory can be substantially reduced.

---

# 32. LoRA and GPU Memory

LoRA can reduce training memory because:

```text
Most pretrained parameters
        ↓
Frozen

Fewer gradients
        ↓
Less gradient memory

Fewer trainable parameters
        ↓
Fewer optimizer states
        ↓
Less optimizer memory
```

However:

> **LoRA does not eliminate GPU memory requirements.**

The base model still needs to be loaded, and activations still require memory.

---

# 33. LoRA Does Not Automatically Solve Every Memory Problem

Even with LoRA, you may still need:

```text
Mixed Precision
Gradient Checkpointing
Gradient Accumulation
Quantization
```

For very large models.

Modern fine-tuning can combine:

```text
LoRA
+
BF16 / FP16
+
Gradient Checkpointing
+
Gradient Accumulation
```

---

# 34. LoRA and Multiple Tasks

Suppose we have one base model:

```text
Base LLM
```

and want to create:

```text
Medical assistant
Legal assistant
Coding assistant
Customer support assistant
```

With full fine-tuning, we may need separate complete model copies.

With LoRA:

```text
                    Base Model
                   /    |     \
                  /     |      \
                 ↓      ↓       ↓
             LoRA A  LoRA B  LoRA C
             Medical   Legal   Coding
```

The same base model can be shared while different LoRA adapters provide task-specific adaptations.

---

# 35. What Is a LoRA Adapter?

The trainable LoRA component is commonly called an:

\[
\boxed{\text{Adapter}}
\]

Conceptually:

```text
Base Model
     +
LoRA Adapter
     ↓
Task-specific model
```

The adapter contains the learned low-rank matrices.

---

# 36. PEFT Library

Hugging Face provides a library called:

\[
\boxed{\text{PEFT}}
\]

which makes parameter-efficient fine-tuning easier.

A common stack is:

```text
Transformers
+
PEFT
+
LoRA
```

---

# 37. Installing PEFT

A typical installation is:

```bash
pip install peft
```

You may also need:

```bash
pip install transformers datasets accelerate
```

depending on your project.

---

# 38. Basic LoRA Configuration

A typical PEFT configuration looks like:

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

The exact configuration depends on the model and task.

---

# 39. Applying LoRA to a Model

Conceptually:

```python
from peft import get_peft_model

model = get_peft_model(
    model,
    lora_config
)
```

The model now contains the LoRA adapters.

---

# 40. Checking Trainable Parameters

A very useful operation is:

```python
model.print_trainable_parameters()
```

You may see something conceptually like:

```text
trainable params: 8M
all params: 7B
trainable%: 0.11%
```

The actual numbers depend on:

- Model architecture
- LoRA rank
- Target modules
- Number of layers

---

# 41. Why This Is Powerful

Suppose:

```text
Total parameters = 7,000,000,000

Trainable parameters = 8,000,000
```

Then:

\[
\frac{8,000,000}{7,000,000,000}\times100
\approx0.114\%
\]

So only about:

\[
\boxed{0.114\%}
\]

of the parameters are trainable in this example.

The remaining parameters remain frozen.

---

# 42. Full Fine-Tuning vs LoRA

| Feature | Full Fine-Tuning | LoRA |
|---|---|---|
| Pretrained weights | Trainable | Frozen |
| Additional parameters | None | Small LoRA matrices |
| Trainable parameters | Very large | Much smaller |
| GPU memory | High | Lower |
| Training cost | High | Lower |
| Task-specific storage | Large | Small adapter |
| Multiple adapters | Expensive | Convenient |
| Base model knowledge | Updated directly | Preserved in frozen weights |

---

# 43. What LoRA Actually Learns

Suppose the pretrained model already knows:

```text
English
Grammar
Syntax
General representations
World knowledge
Reasoning patterns
```

Now we want:

```text
Medical question answering
```

Full fine-tuning could update the entire model.

LoRA instead tries to learn:

```text
Pretrained knowledge
        +
Small task-specific adaptation
        ↓
Medical behavior
```

So the original pretrained knowledge remains in the frozen weights while LoRA learns a task-specific adjustment.

---

# 44. LoRA vs QLoRA

This is important because these terms are often confused.

## LoRA

```text
Base Model
↓
Usually FP16/BF16
↓
Frozen
↓
LoRA trained
```

## QLoRA

```text
Base Model
↓
Quantized
↓
Frozen
↓
LoRA trained
```

Therefore:

\[
\boxed{
QLoRA
=
Quantized\ Base\ Model
+
LoRA
}
\]

QLoRA will be studied in greater detail later.

---

# 45. Quantization Preview

Quantization means representing numerical values using fewer bits.

For example:

```text
FP32
 ↓
FP16 / BF16
 ↓
INT8
 ↓
INT4
```

Lower-bit representations can substantially reduce memory requirements.

However, quantization introduces numerical and accuracy considerations.

---

# 46. Why QLoRA Exists

Suppose a model is too large to fit comfortably in GPU memory even with ordinary LoRA.

We can reduce the memory used by the frozen base model through quantization.

Conceptually:

```text
Large pretrained model
        ↓
Quantize base model
        ↓
Freeze base model
        ↓
Add LoRA
        ↓
Train LoRA
```

This is the basic idea behind QLoRA.

---

# 47. When Should You Use Full Fine-Tuning?

Full fine-tuning can make sense when:

```text
Sufficient GPU resources
+
Large/high-quality dataset
+
Need for maximum adaptation
+
Model size is manageable
```

LoRA is not universally better.

It is an efficiency technique.

---

# 48. When Should You Use LoRA?

LoRA is especially attractive when:

```text
Large model
+
Limited GPU memory
+
Small/moderate dataset
+
Need efficient task adaptation
+
Want multiple task-specific adapters
```

---

# 49. Simple Fine-Tuning Decision Tree

```text
Need to adapt a pretrained model?
              │
              ↓
       Can full fine-tuning
       comfortably fit?
          /          \
        Yes           No
        ↓              ↓
 Full FT          Consider PEFT
                       ↓
                     LoRA
                       ↓
             Still memory constrained?
                       ↓
                    QLoRA
```

This is a simplified decision process, not a strict rule.

---

# 50. The Most Important Concept

If you remember only one idea from this lesson, remember:

```text
Normal Fine-Tuning:

W
↓
Update W


LoRA:

W
↓
FREEZE W

A + B
↓
TRAIN A and B

Effective weight:

W + BA
```

Therefore:

\[
\boxed{
W_{\text{effective}}
=
W_{\text{frozen}}
+
BA
}
\]

And in the commonly used scaled formulation:

\[
\boxed{
W_{\text{effective}}
=
W+
\frac{\alpha}{r}BA
}
\]

---

# 51. One More Intuitive Analogy

Imagine you have a large pretrained machine.

```text
BIG MACHINE
```

The machine already contains a huge amount of learned capability.

You want it to perform a new specialized task.

### Full fine-tuning

You rebuild or modify many internal components:

```text
BIG MACHINE
↓
Modify many components
```

### LoRA

You keep the machine intact and attach a small adjustable component:

```text
BIG MACHINE
     +
SMALL ADAPTER
     ↓
Specialized behavior
```

The big machine is still doing most of the work.

The adapter provides the task-specific adjustment.

---

# 52. Common Misunderstandings

## Misunderstanding 1

> "LoRA removes the original model."

❌ Incorrect.

The original model remains.

---

## Misunderstanding 2

> "Frozen means the model doesn't use those parameters."

❌ Incorrect.

Frozen parameters are still used during forward computation.

They simply are not updated.

---

## Misunderstanding 3

> "LoRA trains the entire model."

❌ Incorrect.

LoRA trains the small LoRA parameter set while keeping the base parameters frozen.

---

## Misunderstanding 4

> "LoRA means A and B replace W."

❌ Incorrect.

They create an update:

\[
\Delta W=BA
\]

which is combined with \(W\):

\[
W_{\text{effective}}=W+BA
\]

---

## Misunderstanding 5

> "LoRA requires no GPU memory."

❌ Incorrect.

The base model still has to be loaded, and training still requires memory for activations and other tensors.

---

# 53. Questions & Answers

## Q1. What is PEFT?

### Answer

PEFT stands for Parameter-Efficient Fine-Tuning.

It adapts a pretrained model by training only a small subset of parameters rather than updating the entire model.

---

## Q2. What is LoRA?

### Answer

LoRA stands for Low-Rank Adaptation.

It freezes the pretrained weights and learns a low-rank update:

\[
\Delta W=BA
\]

so that:

\[
W_{\text{effective}}=W+BA
\]

---

## Q3. What does "freeze the model" mean?

### Answer

It means the pretrained parameters are not updated during training.

They are still used during the forward pass.

---

## Q4. Are frozen parameters used during inference?

### Answer

Yes.

Frozen parameters remain part of the model and are used during forward computation.

---

## Q5. Why is LoRA memory efficient?

### Answer

Because most pretrained parameters remain frozen, so gradients and optimizer states are needed for far fewer trainable parameters.

---

## Q6. What is the main LoRA equation?

### Answer

Simplified:

\[
\boxed{
W'=W+BA
}
\]

Common scaled formulation:

\[
\boxed{
W'=W+\frac{\alpha}{r}BA
}
\]

---

## Q7. What are A and B?

### Answer

\(A\) and \(B\) are small trainable matrices whose product:

\[
BA
\]

represents the task-specific update to the frozen weight matrix.

---

## Q8. What is LoRA rank \(r\)?

### Answer

\(r\) is the low-rank dimension used by the LoRA matrices.

A smaller \(r\) generally means fewer trainable parameters.

---

## Q9. What happens if the LoRA rank increases?

### Answer

Generally:

```text
Higher rank
↓
More trainable parameters
↓
More memory/computation
↓
Greater adaptation capacity
```

---

## Q10. What are target modules?

### Answer

Target modules specify which layers or projections receive LoRA adapters.

For Transformer models, these are often attention projection layers such as:

```text
q_proj
k_proj
v_proj
o_proj
```

The exact names depend on the model architecture.

---

## Q11. Why is LoRA commonly applied to attention projections?

### Answer

Transformer attention uses learned projection matrices:

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

These projections strongly influence how the model transforms and relates information, making them useful places for task-specific adaptation.

---

## Q12. What is an adapter?

### Answer

An adapter is a relatively small trainable component added to a pretrained model to provide task-specific adaptation.

LoRA provides adapters using low-rank matrices.

---

## Q13. What is QLoRA?

### Answer

QLoRA combines a quantized frozen base model with trainable LoRA adapters.

\[
\boxed{
QLoRA
=
Quantization
+
LoRA
}
\]

---

## Q14. Does LoRA modify the original pretrained weights?

### Answer

The original pretrained weights remain frozen.

LoRA learns a separate update:

\[
\Delta W=BA
\]

The effective transformation is:

\[
W+\Delta W
\]

---

## Q15. Can multiple LoRA adapters use the same base model?

### Answer

Yes.

Conceptually:

```text
              Base Model
             /    |     \
            ↓     ↓      ↓
        Adapter A B      C
        Medical  Legal  Coding
```

This allows different task-specific adaptations to share the same base model.

---

# 54. Self-Test

Try answering these without looking back:

1. Why is full fine-tuning expensive?
2. What does PEFT stand for?
3. What does LoRA stand for?
4. What does freezing a parameter mean?
5. Are frozen parameters still used during the forward pass?
6. What is the main LoRA equation?
7. What are \(A\) and \(B\)?
8. What is LoRA rank \(r\)?
9. Why does a smaller rank reduce trainable parameters?
10. What are target modules?
11. Why can LoRA be applied to Transformer attention projections?
12. What is the difference between full fine-tuning and LoRA?
13. What is an adapter?
14. What is QLoRA?
15. How does quantization help QLoRA?
16. Why can LoRA reduce optimizer memory?
17. Does LoRA eliminate all GPU memory requirements?
18. Can multiple LoRA adapters share one base model?
19. Why might you choose LoRA instead of full fine-tuning?
20. What is the difference between LoRA and QLoRA?

---

# 55. Understanding Check

Before moving to QLoRA, make sure you can answer these three questions:

### Question 1

In normal fine-tuning, what happens to \(W\)?

### Answer

\(W\) is trainable and is updated using gradients.

---

### Question 2

In LoRA, what happens to \(W\), and what are \(A\) and \(B\) used for?

### Answer

\(W\) is frozen.

\(A\) and \(B\) are trainable matrices used to create a low-rank update:

\[
\Delta W=BA
\]

Therefore:

\[
W_{\text{effective}}=W+BA
\]

---

### Question 3

If:

\[
W\in\mathbb{R}^{4096\times4096}
\]

and:

\[
r=8
\]

what are the dimensions of \(A\) and \(B\)?

### Answer

\[
A\in\mathbb{R}^{8\times4096}
\]

and:

\[
B\in\mathbb{R}^{4096\times8}
\]

Then:

\[
BA
=
(4096\times8)(8\times4096)
\]

giving:

\[
BA\in\mathbb{R}^{4096\times4096}
\]

So \(BA\) has the same shape as \(W\), allowing:

\[
W+BA
\]

---

# 56. Final Mental Model

Keep this picture in your mind:

```text
                 PRETRAINED MODEL
                        │
                        │
                   W FROZEN
                        │
                        │
               ┌────────┴────────┐
               │                 │
               ↓                 ↓
          Original Path      LoRA Path
               │                 │
              Wx                BAx
               │                 │
               └────────┬────────┘
                        ↓
                    Wx + BAx
                        ↓
                     Output
                        ↓
                       Loss
                        ↓
                    Backward
                        ↓
                  Update A, B
```

The original model remains frozen.

The LoRA matrices learn the task-specific adaptation.

The effective layer behaves as though its weight is:

\[
\boxed{
W_{\text{effective}}
=
W_{\text{frozen}}
+
\frac{\alpha}{r}BA
}
\]

---

# 57. Lesson 9 Summary

## Full Fine-Tuning

\[
\boxed{
\text{Train the pretrained weights}
}
\]

## PEFT

\[
\boxed{
\text{Freeze most parameters + train a small parameter set}
}
\]

## LoRA

\[
\boxed{
\Delta W=BA
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

## Main Components

```text
W
↓
Frozen pretrained weights

A
↓
Trainable low-rank matrix

B
↓
Trainable low-rank matrix

r
↓
LoRA rank

α
↓
Scaling factor
```

## Main Advantage

```text
Full Fine-Tuning
        ↓
Train huge number of parameters

LoRA
        ↓
Freeze base model
        ↓
Train small number of parameters
        ↓
Lower training memory/cost
        ↓
Small task-specific adapter
```

---

# 58. What Comes Next?

## Lesson 10 — Quantization & QLoRA

Now that the basic LoRA idea is clear, the next question is:

> **What if the base model itself is so large that even loading it in FP16/BF16 is difficult on our GPU?**

That leads to:

```text
Quantization
     ↓
FP32
     ↓
FP16 / BF16
     ↓
INT8
     ↓
INT4
     ↓
Lower memory
     ↓
QLoRA
     ↓
Quantized Base Model
+
LoRA Adapter
     ↓
Memory-efficient fine-tuning
```

The next lesson will explain **quantization, 4-bit models, BitsAndBytes, and QLoRA** step by step.
