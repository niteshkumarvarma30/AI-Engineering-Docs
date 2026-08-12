# Lesson 12 — Implementing LoRA From Scratch with PyTorch

> **Goal:** Understand and implement the core LoRA mechanism yourself in PyTorch instead of treating PEFT as a black box.

---

# 1. Why Implement LoRA From Scratch?

Normally, with PEFT, we can write:

```python
model = get_peft_model(model, lora_config)
```

PEFT handles the implementation for us.

But understanding the underlying implementation helps us understand:

- What PEFT is doing
- Why LoRA saves parameters
- Why the base model is frozen
- Where gradients flow
- How LoRA modifies a Linear layer
- How LoRA connects to PyTorch autograd
- How LoRA is applied to Transformer projections

The goal is:

```text
LoRA Mathematics
       ↓
PyTorch Implementation
       ↓
Autograd
       ↓
Transformer Linear Layers
       ↓
PEFT
```

---

# 2. Start With a Normal Linear Layer

A standard linear layer performs:

\[
y = xW^T + b
\]

In PyTorch:

```python
import torch
import torch.nn as nn

linear = nn.Linear(
    in_features=4,
    out_features=3
)
```

The weight matrix has approximately:

```text
out_features × in_features
```

So here:

\[
W \in \mathbb{R}^{3\times4}
\]

and the number of weight parameters is:

\[
3\times4=12
\]

There are also 3 bias parameters.

---

# 3. What Happens During Normal Fine-Tuning?

Suppose:

\[
W\in\mathbb{R}^{3\times4}
\]

Traditional fine-tuning updates the parameters of \(W\):

```text
W
↓
Trainable
↓
Gradient
↓
Optimizer
↓
Updated W
```

For an LLM, there can be many enormous matrices.

For example:

\[
W\in\mathbb{R}^{4096\times4096}
\]

contains:

\[
4096\times4096
=
16,777,216
\]

parameters in just one matrix.

LLMs contain many such matrices.

---

# 4. LoRA's Main Idea

Instead of directly learning a complete weight update:

\[
\Delta W
\]

LoRA approximates the update using two smaller matrices:

\[
\boxed{\Delta W = BA}
\]

Therefore:

\[
\boxed{
W' = W + \frac{\alpha}{r}BA
}
\]

where:

- \(W\) = original pretrained weight
- \(A\) = trainable low-rank matrix
- \(B\) = trainable low-rank matrix
- \(r\) = LoRA rank
- \(\alpha\) = scaling factor

---

# 5. Why Is It Called Low-Rank?

Suppose:

\[
W\in\mathbb{R}^{4096\times4096}
\]

A complete update would require:

\[
4096\times4096
\]

parameters.

Instead choose:

\[
r=8
\]

Then:

\[
A\in\mathbb{R}^{8\times4096}
\]

and:

\[
B\in\mathbb{R}^{4096\times8}
\]

The trainable parameter count becomes:

\[
8\times4096 + 4096\times8
\]

\[
=32768+32768
\]

\[
=\boxed{65,536}
\]

instead of:

\[
16,777,216
\]

This is the main efficiency advantage of LoRA.

---

# 6. Visualizing the Architecture

A normal Linear layer:

```text
Input ─────────────→ W ─────────────→ Output
```

LoRA:

```text
                    Frozen W
                 ┌──────────────→
                 │
Input ───────────┤
                 │
                 └────→ A → B ───→
                           LoRA

Final Output
=
Base Output
+
LoRA Output
```

Mathematically:

\[
y=xW^T+\frac{\alpha}{r}x(BA)^T
\]

---

# 7. Important: The Base Weight Is Frozen

One of the most important ideas in LoRA is:

```text
W
↓
Frozen
```

while:

```text
A
↓
Trainable

B
↓
Trainable
```

Therefore:

```text
Original knowledge
       ↓
       W
     frozen

Task adaptation
       ↓
      A + B
    trainable
```

The pretrained model continues to provide its existing capabilities, while LoRA learns a task-specific update.

---

# 8. Creating the LoRA Matrices

We can implement a LoRA linear layer in PyTorch.

```python
import torch
import torch.nn as nn


class LoRALinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        rank=4,
        alpha=8
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        self.scaling = alpha / rank

        # Base weight
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features)
        )

        # LoRA matrices
        self.A = nn.Parameter(
            torch.randn(rank, in_features)
        )

        self.B = nn.Parameter(
            torch.randn(out_features, rank)
        )
```

There is an important problem here:

The base weight is still trainable.

We need to freeze it.

---

# 9. Freezing the Base Weight

We can write:

```python
self.weight.requires_grad = False
```

or create it directly as:

```python
self.weight = nn.Parameter(
    torch.randn(out_features, in_features),
    requires_grad=False
)
```

Now:

```text
weight
↓
Frozen

A
↓
Trainable

B
↓
Trainable
```

---

# 10. The Forward Pass

LoRA computes:

\[
y=xW^T+\frac{\alpha}{r}x(BA)^T
\]

A direct PyTorch implementation is:

```python
def forward(self, x):

    base_output = x @ self.weight.T

    lora_output = (
        x @ self.A.T
        @ self.B.T
    )

    return (
        base_output
        + self.scaling * lora_output
    )
```

---

# 11. Dimension Analysis

Suppose:

```text
in_features = 4
out_features = 3
rank = 2
```

Then:

\[
W\in\mathbb{R}^{3\times4}
\]

\[
A\in\mathbb{R}^{2\times4}
\]

\[
B\in\mathbb{R}^{3\times2}
\]

Suppose:

\[
x\in\mathbb{R}^{batch\times4}
\]

Then:

\[
xA^T
\]

has shape:

\[
(batch\times4)(4\times2)
\]

so:

\[
=batch\times2
\]

Then:

\[
(xA^T)B^T
\]

has shape:

\[
(batch\times2)(2\times3)
\]

so:

\[
=batch\times3
\]

This is exactly the same output shape as the original Linear layer.

---

# 12. Complete `LoRALinear`

```python
import torch
import torch.nn as nn


class LoRALinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        rank=4,
        alpha=8
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        self.scaling = alpha / rank

        # Frozen pretrained weight
        self.weight = nn.Parameter(
            torch.randn(
                out_features,
                in_features
            ),
            requires_grad=False
        )

        # Trainable LoRA matrices
        self.A = nn.Parameter(
            torch.randn(
                rank,
                in_features
            )
        )

        self.B = nn.Parameter(
            torch.randn(
                out_features,
                rank
            )
        )

    def forward(self, x):

        base_output = x @ self.weight.T

        lora_output = (
            x @ self.A.T
            @ self.B.T
        )

        return (
            base_output
            + self.scaling * lora_output
        )
```

---

# 13. Important LoRA Initialization Detail

The simplified implementation above initializes both \(A\) and \(B\) randomly.

A common LoRA initialization instead uses:

```text
A → random initialization
B → zero initialization
```

Why?

Initially:

\[
B=0
\]

Therefore:

\[
BA=0
\]

and:

\[
W'=W
\]

at initialization.

Thus the LoRA branch initially contributes no update, so the model initially behaves like the original pretrained model.

---

# 14. Better Initialization

We can use:

```python
nn.init.kaiming_uniform_(
    self.A,
    a=5**0.5
)

nn.init.zeros_(self.B)
```

So:

```text
A
↓
Random

B
↓
Zeros
```

Initially:

\[
BA=0
\]

Therefore:

\[
W'=W
\]

---

# 15. Improved Implementation

```python
import torch
import torch.nn as nn


class LoRALinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        rank=4,
        alpha=8
    ):
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen base weight
        self.weight = nn.Parameter(
            torch.randn(
                out_features,
                in_features
            ),
            requires_grad=False
        )

        # Trainable LoRA matrices
        self.A = nn.Parameter(
            torch.empty(
                rank,
                in_features
            )
        )

        self.B = nn.Parameter(
            torch.zeros(
                out_features,
                rank
            )
        )

        # Initialize A
        nn.init.kaiming_uniform_(
            self.A,
            a=5**0.5
        )

    def forward(self, x):

        base_output = x @ self.weight.T

        lora_output = (
            x @ self.A.T
            @ self.B.T
        )

        return (
            base_output
            + self.scaling * lora_output
        )
```

---

# 16. Test the Layer

Create the layer:

```python
layer = LoRALinear(
    in_features=4,
    out_features=3,
    rank=2,
    alpha=4
)
```

Create an input:

```python
x = torch.randn(5, 4)
```

Here:

```text
5 = batch size
4 = input features
```

Run:

```python
output = layer(x)
```

Check:

```python
print(output.shape)
```

Expected shape:

```text
torch.Size([5, 3])
```

because:

```text
5 × 4
 ↓
Linear
 ↓
5 × 3
```

---

# 17. Check Trainable Parameters

Run:

```python
for name, parameter in layer.named_parameters():

    print(
        name,
        parameter.shape,
        parameter.requires_grad
    )
```

Conceptually:

```text
weight   [3, 4]   False
A        [2, 4]   True
B        [3, 2]   True
```

This confirms that:

```text
Base weight → frozen
A            → trainable
B            → trainable
```

---

# 18. Count Parameters

For:

```text
out_features = 3
in_features = 4
rank = 2
```

Base weight:

\[
3\times4=12
\]

LoRA parameters:

\[
A:2\times4=8
\]

\[
B:3\times2=6
\]

Total trainable LoRA parameters:

\[
8+6=14
\]

For this tiny example, LoRA is not necessarily smaller than the original matrix.

LoRA becomes useful when:

\[
d\gg r
\]

---

# 19. General Parameter Comparison

Original matrix:

\[
d_{out}d_{in}
\]

LoRA:

\[
rd_{in}+d_{out}r
\]

If:

\[
d_{in}=d_{out}=d
\]

then:

\[
\text{LoRA parameters}=2dr
\]

while:

\[
\text{Original parameters}=d^2
\]

Therefore:

\[
\frac{\text{LoRA parameters}}
{\text{Original parameters}}
=
\frac{2dr}{d^2}
=
\frac{2r}{d}
\]

For:

\[
d=4096,\quad r=8
\]

we get:

\[
\frac{16}{4096}
=
0.00390625
\]

or approximately:

\[
\boxed{0.39\%}
\]

of the original matrix's parameter count.

---

# 20. What Happens During Backpropagation?

Consider:

```text
Input
  ↓
      ┌── Frozen W ──┐
      │              │
      └── A → B ─────┘
              ↓
            Output
              ↓
             Loss
```

During backpropagation:

```text
Loss
 ↓
Gradients
 ↓
A and B
 ↓
Optimizer
 ↓
Update
```

But:

```text
W
↓
requires_grad=False
↓
No parameter update
```

---

# 21. Why `requires_grad=False` Matters

PyTorch uses:

```python
parameter.requires_grad
```

to determine whether gradients should be tracked for that parameter.

For example:

```python
W.requires_grad = False
```

means the parameter is frozen and is not updated through gradient descent.

While:

```python
A.requires_grad = True
B.requires_grad = True
```

means those parameters participate in gradient-based learning.

---

# 22. Tiny Training Experiment

Train only the LoRA parameters:

```python
layer = LoRALinear(
    4,
    3,
    rank=2,
    alpha=4
)

optimizer = torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        layer.parameters()
    ),
    lr=0.01
)
```

Notice:

```python
filter(
    lambda p: p.requires_grad,
    layer.parameters()
)
```

selects only trainable parameters.

---

# 23. Training Loop

```python
for step in range(100):

    x = torch.randn(8, 4)
    target = torch.randn(8, 3)

    output = layer(x)

    loss = nn.functional.mse_loss(
        output,
        target
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 10 == 0:
        print(
            step,
            loss.item()
        )
```

This is ordinary PyTorch training.

The important difference is:

\[
\boxed{\text{Only A and B are trainable}}
\]

---

# 24. What Happens at `loss.backward()`?

PyTorch calculates gradients through the computational graph.

Conceptually:

```text
Loss
 ↓
Output
 ↓
LoRA branch
 ↓
B
 ↓
A
```

The base weight:

```text
W
```

is frozen.

Therefore, it is not updated as a trainable parameter.

After:

```python
loss.backward()
```

you can inspect:

```python
print(layer.weight.grad)
print(layer.A.grad)
print(layer.B.grad)
```

Conceptually:

```text
weight.grad → None

A.grad → tensor(...)

B.grad → tensor(...)
```

---

# 25. Connecting LoRA to Autograd

PyTorch autograd builds a computational graph during the forward pass.

For LoRA:

\[
y=xW^T+\frac{\alpha}{r}x(BA)^T
\]

The computation involves \(A\) and \(B\).

Because \(W\) does not require gradients, PyTorch does not calculate a parameter gradient for updating \(W\).

Then:

```python
loss.backward()
```

propagates gradients through the graph to the trainable parameters.

---

# 26. The Most Important Concept

LoRA does **not** mean:

> Create a completely different neural network.

Instead:

\[
\boxed{
\text{Original Layer}
+
\text{Small Trainable Update}
}
\]

The original layer remains.

The adaptation is added alongside it.

---

# 27. LoRA in a Transformer

Now connect this to the Transformer concepts you already learned.

Recall:

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

These are linear projections.

LoRA can be applied to these projection matrices.

For example:

\[
W_Q'
=
W_Q+
\frac{\alpha}{r}B_QA_Q
\]

and:

\[
W_V'
=
W_V+
\frac{\alpha}{r}B_VA_V
\]

Conceptually:

```text
Transformer Attention
       │
       ├── WQ ──→ LoRA
       │
       ├── WK
       │
       ├── WV ──→ LoRA
       │
       └── WO
```

The actual target modules depend on the model architecture and configuration.

---

# 28. This Explains `target_modules`

Earlier we saw:

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

Now you can interpret it.

It means:

> Add LoRA adapters to the model's Query and Value projection modules.

Conceptually:

```text
q_proj
   ↓
WQ + LoRA

v_proj
   ↓
WV + LoRA
```

---

# 29. LoRA Does Not Have to Modify Only Q and V

LoRA can be applied to other compatible modules depending on the model and training strategy.

For example:

```text
q_proj
k_proj
v_proj
o_proj
```

and sometimes MLP projections.

The exact module names are model-specific.

Therefore, do not blindly copy:

```python
target_modules=["q_proj", "v_proj"]
```

without inspecting the architecture.

---

# 30. LoRA vs Full Fine-Tuning

## Full Fine-Tuning

```text
Pretrained Model
       ↓
All parameters trainable
       ↓
Large memory requirements
       ↓
Large optimizer state
       ↓
Expensive
```

## LoRA

```text
Pretrained Model
       ↓
Base parameters frozen
       +
Small LoRA matrices
       ↓
Only LoRA trainable
       ↓
Much fewer trainable parameters
```

---

# 31. LoRA vs QLoRA

This distinction is important.

## LoRA

```text
Base model
↓
Usually kept in standard precision
↓
Frozen

+
LoRA
↓
Trainable
```

## QLoRA

```text
Base model
↓
Quantized, commonly 4-bit
↓
Frozen

+
LoRA
↓
Trainable
```

Therefore:

\[
\boxed{
\text{QLoRA}
=
\text{Quantization}
+
\text{LoRA}
}
\]

QLoRA is not a completely different low-rank method.

---

# 32. LoRA Mathematical Summary

Starting with:

\[
y=xW^T
\]

Traditional fine-tuning learns:

\[
W'=W+\Delta W
\]

LoRA approximates:

\[
\Delta W\approx BA
\]

Therefore:

\[
\boxed{
y
=
xW^T+
\frac{\alpha}{r}x(BA)^T
}
\]

with:

\[
W\text{ frozen}
\]

and:

\[
A,B\text{ trainable}
\]

---

# 33. Why Does This Work?

The central idea behind LoRA is that useful task adaptation can often be represented by a low-rank update rather than requiring an unrestricted update to every weight.

Instead of learning:

\[
\Delta W
\]

directly, we learn:

\[
A,B
\]

where:

\[
r\ll d
\]

This dramatically reduces the number of trainable parameters.

---

# 34. Important Correction

Do not interpret LoRA as:

> "The original weight matrix is replaced by \(BA\)."

That is incorrect.

The correct formulation is:

\[
\boxed{
W_{\text{effective}}
=
W+
\frac{\alpha}{r}BA
}
\]

The original \(W\) remains part of the computation.

---

# 35. Mental Model

Think of the pretrained model as:

```text
                 PRETRAINED KNOWLEDGE
                         │
                         ↓
                         W
                      FROZEN
                         │
                         │
Input ───────────────────┤
                         │
                         +
                         │
                    LoRA Update
                         │
                     A → B
                         │
                     TRAINABLE
                         ↓
                       Output
```

The base model provides the existing capability.

LoRA provides the task-specific adjustment.

---

# 36. Questions & Answers

## Q1. What is frozen in LoRA?

**Answer:**

The pretrained model weights \(W\).

---

## Q2. What is trainable?

**Answer:**

The LoRA matrices \(A\) and \(B\).

---

## Q3. What does LoRA learn?

It learns an approximation of the weight update:

\[
\Delta W\approx BA
\]

---

## Q4. Why use two matrices?

Instead of learning:

\[
d\times d
\]

parameters, we learn:

\[
d\times r+r\times d
\]

with:

\[
r\ll d
\]

---

## Q5. Why is \(r\) small?

To make the number of trainable parameters much smaller.

---

## Q6. Why is \(B\) commonly initialized to zero?

A common initialization makes the initial LoRA update zero:

\[
B=0
\]

therefore:

\[
BA=0
\]

and:

\[
W'=W
\]

initially.

---

## Q7. Does LoRA change the original model weights during training?

No. The original weights remain frozen.

---

## Q8. Does LoRA create a completely new model?

No. It adds a trainable low-rank adaptation to existing layers.

---

## Q9. What does `target_modules` mean?

It identifies which modules receive LoRA adapters.

---

## Q10. What is the difference between LoRA and QLoRA?

LoRA uses low-rank adapters.

QLoRA additionally quantizes the frozen base model to reduce memory usage.

---

# 37. Self-Test

Try answering these without looking back:

1. What does a normal `nn.Linear` layer calculate?
2. What is the problem with directly learning \(\Delta W\)?
3. What is LoRA's approximation of \(\Delta W\)?
4. What are the dimensions of \(A\) and \(B\)?
5. Why must \(r\ll d\)?
6. Which parameters are frozen?
7. Which parameters are trainable?
8. Why is \(B\) commonly initialized to zero?
9. What does `requires_grad=False` do?
10. What happens during `loss.backward()`?
11. Why does `target_modules=["q_proj", "v_proj"]` make sense in a Transformer?
12. What is the difference between LoRA and QLoRA?

---

# 38. Final Picture

You can now connect:

```text
                    TRANSFORMER
                         │
               ┌─────────┴─────────┐
               ↓                   ↓
             WQ                  WV
               │                   │
               ↓                   ↓
          Frozen WQ            Frozen WV
               +                   +
               ↓                   ↓
            LoRA Q              LoRA V
             A,B                 A,B
               │                   │
               └─────────┬─────────┘
                         ↓
                  Attention Output
```

Mathematically:

\[
\boxed{
W_Q'
=
W_Q+
\frac{\alpha}{r}B_QA_Q
}
\]

\[
\boxed{
W_V'
=
W_V+
\frac{\alpha}{r}B_VA_V
}
\]

This connects:

```text
Transformer
     ↓
Linear Projections
     ↓
LoRA
     ↓
PyTorch Autograd
     ↓
LLM Fine-Tuning
```

---

# 39. Lesson 12 Summary

The most important concepts are:

### Normal fine-tuning

\[
W'=W+\Delta W
\]

### LoRA

\[
\Delta W\approx BA
\]

### Effective weight

\[
\boxed{
W'=W+\frac{\alpha}{r}BA
}
\]

### Parameter efficiency

Original:

\[
d^2
\]

LoRA:

\[
2dr
\]

when the original matrix is \(d\times d\).

### Training

```text
Base W
↓
Frozen

A
↓
Trainable

B
↓
Trainable
```

### Transformer connection

\[
Q=XW_Q
\]

\[
V=XW_V
\]

can become:

\[
Q=X
\left(
W_Q+\frac{\alpha}{r}B_QA_Q
\right)
\]

and:

\[
V=X
\left(
W_V+\frac{\alpha}{r}B_VA_V
\right)
\]

---

# 40. Next Lesson

## Lesson 13 — LoRA Variants

The fine-tuning material you provided introduces four important LoRA-related techniques:

```text
1. LoRA+
2. LoRA-FA
3. VeRA
4. Delta-LoRA
```

We will compare each against standard LoRA and understand:

- What problem it addresses
- What changes mathematically
- Which parameters are trained
- Why it can be more efficient
- How it differs from standard LoRA
- When it is useful

After that, we will move into:

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
