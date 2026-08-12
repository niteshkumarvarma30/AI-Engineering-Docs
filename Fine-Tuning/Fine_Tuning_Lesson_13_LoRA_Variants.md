# Lesson 13 — LoRA Variants

> **Goal:** Understand the important LoRA variants introduced in the supplied fine-tuning material and how each differs from standard LoRA.

---

# 1. Recap: Standard LoRA

Suppose the pretrained weight matrix is:

\[
W\in\mathbb{R}^{d_{out}\times d_{in}}
\]

Standard LoRA introduces two low-rank matrices:

\[
A\in\mathbb{R}^{r\times d_{in}}
\]

and:

\[
B\in\mathbb{R}^{d_{out}\times r}
\]

where:

\[
r\ll d
\]

The effective weight is:

\[
\boxed{
W'=W+\frac{\alpha}{r}BA
}
\]

where:

- \(W\) = original pretrained weight
- \(A\) = trainable low-rank matrix
- \(B\) = trainable low-rank matrix
- \(r\) = LoRA rank
- \(\alpha\) = scaling factor

During standard LoRA training:

```text
W
↓
Frozen

A
↓
Trainable

B
↓
Trainable
```

The LoRA update is:

\[
\boxed{\Delta W=BA}
\]

---

# 2. Why Do We Need LoRA Variants?

Standard LoRA already reduces the number of trainable parameters significantly.

However, researchers have explored ways to make LoRA:

- More memory efficient
- More parameter efficient
- Better optimized
- Faster to train
- More flexible

The supplied material introduces four important LoRA-related techniques:

```text
Standard LoRA
     │
     ├── LoRA+
     ├── LoRA-FA
     ├── VeRA
     └── Delta-LoRA
```

The goal is not to memorize the names. The goal is to understand **what changes compared with standard LoRA**.

---

# 3. Standard LoRA as the Baseline

Standard LoRA learns:

\[
\boxed{\Delta W=BA}
\]

and:

\[
\boxed{
W'=W+\frac{\alpha}{r}BA
}
\]

Both \(A\) and \(B\) are trainable:

```text
A → Trainable
B → Trainable
```

This is our baseline for comparing the variants.

---

# 4. LoRA+

## 4.1 The Problem

In standard LoRA, \(A\) and \(B\) can be optimized using the same learning-rate setup.

For example:

```python
optimizer = Adam(
    [
        {"params": A},
        {"params": B}
    ],
    lr=1e-4
)
```

Conceptually:

\[
LR_A=LR_B
\]

However, \(A\) and \(B\) do not necessarily play identical roles during optimization.

---

# 5. LoRA+ Idea

LoRA+ uses **different learning rates** for the two LoRA matrices.

Instead of:

\[
LR_A=LR_B
\]

we use:

\[
\boxed{LR_A\neq LR_B}
\]

For example:

\[
LR_A=10^{-4}
\]

and:

\[
LR_B=10^{-3}
\]

The exact learning-rate ratio depends on the training setup.

---

# 6. Why Separate Learning Rates?

Recall:

\[
\Delta W=BA
\]

Both matrices jointly determine the update.

LoRA+ changes the optimization strategy so that \(A\) and \(B\) can be optimized at different rates.

Conceptually:

```text
                 LoRA
                   │
              ┌────┴────┐
              ↓         ↓
              A         B
              │         │
           LR_A       LR_B
              │         │
              └────┬────┘
                   ↓
                  BA
                   ↓
             LoRA Update
```

The basic LoRA equation remains:

\[
W'=W+\frac{\alpha}{r}BA
\]

---

# 7. LoRA+ in PyTorch

The idea can be represented using different optimizer parameter groups:

```python
optimizer = torch.optim.AdamW(
    [
        {
            "params": model.A,
            "lr": 1e-4
        },
        {
            "params": model.B,
            "lr": 1e-3
        }
    ]
)
```

Now:

```text
A → learning rate = 1e-4
B → learning rate = 1e-3
```

instead of giving both the same learning rate.

---

# 8. Key Point About LoRA+

Do not think:

> LoRA+ completely changes the LoRA architecture.

The low-rank structure remains:

\[
\boxed{\Delta W=BA}
\]

The major change is the **optimization strategy**, particularly the learning rates used for \(A\) and \(B\).

---

# 9. LoRA-FA

LoRA-FA refers to:

> **LoRA with Frozen A**

The key difference is which LoRA parameter is trainable.

### Standard LoRA

```text
A → Trainable
B → Trainable
```

### LoRA-FA

```text
A → Frozen
B → Trainable
```

So:

\[
\boxed{
A_{\text{frozen}},\quad B_{\text{trainable}}
}
\]

---

# 10. Standard LoRA vs LoRA-FA

### Standard LoRA

\[
W'=W+\frac{\alpha}{r}BA
\]

and:

```text
A → Trainable
B → Trainable
```

### LoRA-FA

The \(A\) matrix is fixed/frozen:

```text
A → Frozen
B → Trainable
```

The optimization therefore focuses on \(B\).

---

# 11. Why Freeze A?

One important motivation is **memory efficiency during training**.

Training a neural network requires memory for more than just parameters.

Typical memory components include:

```text
Parameters
Gradients
Optimizer states
Activations
```

Freezing \(A\) can reduce some training-related memory requirements associated with optimizing it.

This can be useful when fine-tuning large models.

---

# 12. LoRA-FA Diagram

```text
                    LoRA-FA

                       A
                       │
                    Frozen
                       │
                       ↓
Input ───────────────→ A
                       │
                       ↓
                       B
                       │
                    Trainable
                       │
                       ↓
                      BA
                       │
                       ↓
                 LoRA Update
```

The key idea is:

\[
\boxed{
A\text{ is fixed},\quad B\text{ is trained}
}
\]

---

# 13. LoRA+ vs LoRA-FA

This is an important distinction.

### LoRA+

Changes:

\[
\boxed{\text{Learning rates}}
\]

Both \(A\) and \(B\) remain trainable.

```text
A → Trainable
B → Trainable

but:

LR_A ≠ LR_B
```

### LoRA-FA

Changes:

\[
\boxed{\text{Which parameters are trained}}
\]

```text
A → Frozen
B → Trainable
```

---

# 14. VeRA

VeRA is a more substantial change.

The supplied material presents VeRA as:

> **Vector-based Random Matrix Adaptation**

The basic idea is to reduce the number of trainable parameters even further.

Recall standard LoRA:

\[
\Delta W=BA
\]

where both \(A\) and \(B\) are trainable.

VeRA instead uses **fixed/shared random matrices** and learns much smaller scaling vectors.

---

# 15. Standard LoRA Parameter Cost

Suppose:

\[
A\in\mathbb{R}^{r\times d}
\]

and:

\[
B\in\mathbb{R}^{d\times r}
\]

Then the trainable parameter count is:

\[
rd+dr
\]

or:

\[
\boxed{2dr}
\]

For a large model, this can still be substantial.

---

# 16. VeRA's Idea

Instead of learning the entire \(A\) and \(B\) matrices for every adapted layer, VeRA uses fixed random matrices and learns small vectors that scale them.

Conceptually:

```text
Standard LoRA:

Trainable A
     ×
Trainable B
     ↓
   Update


VeRA:

Fixed random matrices
        +
Trainable vectors
        ↓
      Update
```

---

# 17. Conceptual VeRA Equation

A simplified conceptual representation is:

\[
\Delta W
=
\Lambda_B B_{\text{fixed}}
A_{\text{fixed}}\Lambda_A
\]

where the important idea is:

```text
A_fixed
B_fixed
   ↓
Not trained

Lambda_A
Lambda_B
   ↓
Trainable
```

The exact implementation has additional architectural details, but this captures the core idea.

---

# 18. Why VeRA Is Efficient

Standard LoRA:

\[
A,B
\]

are trainable.

VeRA:

\[
A_{\text{fixed}},B_{\text{fixed}}
\]

can be shared/fixed, while only small vectors are learned.

Conceptually:

```text
LoRA

Large trainable matrices
        ↓
More trainable parameters


VeRA

Fixed matrices
+
Small trainable vectors
        ↓
Much fewer trainable parameters
```

---

# 19. VeRA Mental Model

A useful intuition is:

### LoRA

> Learn the low-rank matrices that define the task-specific update.

### VeRA

> Use predetermined random directions and learn how strongly to use them.

This is why VeRA can reduce the number of trainable parameters substantially.

---

# 20. Delta-LoRA

The fourth technique in the supplied material is:

> **Delta-LoRA**

First remember standard LoRA:

\[
W_{\text{effective}}
=
W+\Delta W
\]

where:

\[
\Delta W=\frac{\alpha}{r}BA
\]

The original \(W\) stays frozen in standard LoRA.

---

# 21. Standard LoRA

During training:

```text
W
↓
Frozen

A + B
↓
Trainable

Output
=
W + BA
```

The LoRA branch remains separate from the frozen base weight.

---

# 22. Delta-LoRA Idea

Delta-LoRA uses the LoRA update to modify the base weight during training.

Conceptually:

```text
LoRA update
     ↓
   ΔW
     ↓
Update W
     ↓
W ← W + ΔW
```

So unlike standard LoRA, the learned LoRA-derived update is incorporated into the base weight during training.

---

# 23. Why Is It Called Delta-LoRA?

The central object is the weight change:

\[
\Delta W
\]

Recall:

\[
\Delta W=BA
\]

Delta-LoRA uses this learned update more directly in updating the base weight.

---

# 24. Conceptual Comparison

### Standard LoRA

```text
W ────────────────→ Output
│
│ frozen
│
A → B ────────────→ Output
```

The LoRA branch remains separate.

### Delta-LoRA

```text
A → B
  ↓
 ΔW
  ↓
Update W
  ↓
New W
```

The update is incorporated into the base weight during training.

---

# 25. The Four Techniques Compared

| Method | A | B | Main idea |
|---|---|---|---|
| **LoRA** | Trainable | Trainable | Low-rank adaptation |
| **LoRA+** | Trainable | Trainable | Different learning rates |
| **LoRA-FA** | Frozen | Trainable | Freeze A for efficiency |
| **VeRA** | Fixed/shared matrix | Fixed/shared matrix | Learn small scaling vectors |
| **Delta-LoRA** | Used to produce updates | Used to produce updates | Feed LoRA updates into base weights |

---

# 26. The Most Important Difference

Remember each technique with one sentence.

### LoRA

> Learn two small matrices.

\[
\boxed{A+B}
\]

### LoRA+

> Learn both matrices but optimize them differently.

\[
\boxed{LR_A\neq LR_B}
\]

### LoRA-FA

> Freeze \(A\), train \(B\).

\[
\boxed{
A_{\text{frozen}}+B_{\text{trainable}}
}
\]

### VeRA

> Keep large random matrices fixed and learn small scaling vectors.

\[
\boxed{
\text{Fixed matrices}+\text{trainable vectors}
}
\]

### Delta-LoRA

> Use the learned LoRA delta to update the base weights.

\[
\boxed{
W\leftarrow W+\Delta W
}
\]

---

# 27. One Big Picture

```text
                         LoRA
                          │
             W' = W + α/r · BA
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ↓                 ↓                 ↓
     LoRA+             LoRA-FA            VeRA
        │                 │                 │
 Different LR       Freeze A          Fixed matrices
 for A and B             │                 │
        │                ↓              Small vectors
        │             Train B               │
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ↓
                     Delta-LoRA
                          │
                          ↓
                 Update base weights
```

---

# 28. Connecting the Variants to PyTorch

The PyTorch concepts from Lesson 12 help explain the variants.

### Standard LoRA

```python
A.requires_grad = True
B.requires_grad = True
```

### LoRA-FA

```python
A.requires_grad = False
B.requires_grad = True
```

### LoRA+

Both remain trainable:

```python
A.requires_grad = True
B.requires_grad = True
```

but optimizer parameter groups use different learning rates:

```python
optimizer = torch.optim.AdamW([
    {"params": A, "lr": lr_A},
    {"params": B, "lr": lr_B}
])
```

This connects LoRA variants directly to:

- `requires_grad`
- Optimizers
- Parameter groups
- Learning rates
- Autograd

---

# 29. Parameter Efficiency Has Different Dimensions

When we say a method is "more efficient," we should ask:

> Efficient in what way?

Efficiency can mean:

### Parameter efficiency

Fewer trainable parameters.

### Memory efficiency

Less GPU memory required during training.

### Computational efficiency

Less computation.

### Optimization efficiency

Better training behavior or convergence.

Different LoRA variants target different aspects of efficiency.

---

# 30. Do Not Memorize the Variants Yet

At this stage, remember this mental map:

```text
LoRA
│
├── LoRA+
│   └── Different learning rates
│
├── LoRA-FA
│   └── Freeze A
│
├── VeRA
│   └── Fixed random matrices + small trainable vectors
│
└── Delta-LoRA
    └── Use LoRA delta to update base weights
```

The detailed implementation can be studied when we encounter these methods practically.

---

# 31. Questions & Answers

## Q1. What is the main difference between LoRA and LoRA+?

LoRA+ keeps the low-rank architecture but uses **different learning rates** for \(A\) and \(B\).

---

## Q2. What does LoRA-FA freeze?

\[
\boxed{A}
\]

Therefore:

```text
A → Frozen
B → Trainable
```

---

## Q3. What is the main idea behind VeRA?

VeRA uses fixed/shared random matrices and learns much smaller scaling vectors.

---

## Q4. What is the main idea behind Delta-LoRA?

It uses the LoRA-derived weight update:

\[
\Delta W
\]

to update the base weight during training.

---

## Q5. Which method changes the learning-rate strategy?

\[
\boxed{\text{LoRA+}}
\]

---

## Q6. Which method freezes \(A\)?

\[
\boxed{\text{LoRA-FA}}
\]

---

## Q7. Which method focuses on fixed random matrices and small trainable vectors?

\[
\boxed{\text{VeRA}}
\]

---

## Q8. Which method uses the LoRA delta to update the base weights?

\[
\boxed{\text{Delta-LoRA}}
\]

---

# 32. Self-Test

Try answering these without looking back:

1. What is the standard LoRA equation?
2. Which matrices are trainable in standard LoRA?
3. What does LoRA+ change?
4. Why might using different learning rates for \(A\) and \(B\) be useful?
5. What does the "FA" in LoRA-FA refer to?
6. Which LoRA-FA parameter is frozen?
7. What is the main parameter-efficiency idea behind VeRA?
8. What are the large matrices doing in VeRA?
9. What are the trainable components in VeRA?
10. What does Delta-LoRA do differently from standard LoRA?
11. Which technique primarily changes optimization rather than the basic LoRA equation?
12. Which technique freezes one of the LoRA matrices?
13. Which technique uses fixed random matrices?
14. Which technique incorporates the LoRA delta into the base weights?

---

# 33. Lesson 13 Summary

The progression is:

\[
\boxed{
\text{LoRA}
\rightarrow
\text{LoRA+}
\rightarrow
\text{LoRA-FA}
\rightarrow
\text{VeRA}
\rightarrow
\text{Delta-LoRA}
}
\]

The key ideas are:

```text
LoRA
↓
Train A and B


LoRA+
↓
Train A and B
but use different learning rates


LoRA-FA
↓
Freeze A
Train B


VeRA
↓
Fixed/shared random matrices
+
small trainable vectors


Delta-LoRA
↓
Use LoRA delta to update base weights
```

---

# 34. Connection to the Next Phase

So far we have focused on **how to modify the model**:

```text
Full Fine-Tuning
        ↓
LoRA
        ↓
QLoRA
        ↓
LoRA Variants
```

Now we move toward **what data we actually give the model**.

The next phase is:

```text
Raw Dataset
     ↓
Clean / Format
     ↓
Instruction Dataset
     ↓
Input → Response
     ↓
Tokenization
     ↓
Training
```

---

# 35. Next Lesson

## Lesson 14 — Fine-Tuning LLMs With Custom Datasets

We will learn:

- What a fine-tuning dataset looks like
- Dataset formats
- Instruction-response pairs
- Custom datasets
- JSON / JSONL datasets
- Dataset preprocessing
- Tokenization
- Train/validation splits
- How labels are created
- How custom data enters the fine-tuning pipeline
- Connection between custom datasets and IFT
- Preparation for SFT

Then we will progress toward:

```text
Custom Dataset
      ↓
IFT
      ↓
Chat Templates
      ↓
SFT
      ↓
RFT
      ↓
GRPO
      ↓
Reasoning LLMs
      ↓
OpenEnv
      ↓
ART
```
