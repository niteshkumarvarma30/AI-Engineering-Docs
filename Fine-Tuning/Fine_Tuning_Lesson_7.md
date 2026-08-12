# Fine-Tuning — Lesson 7: Learning Rate, Optimizers, Weight Decay & Learning-Rate Schedulers

> **Goal:** Understand how pretrained parameters are actually changed during fine-tuning and how learning rate, optimizers, weight decay, schedulers, and warmup control that process.

---

# 1. The Core Idea of Fine-Tuning

During fine-tuning:

```text
Pretrained Model
       ↓
Already learned parameters
       ↓
Fine-tuning
       ↓
Small parameter updates
       ↓
Task-specific model
```

We generally don't want to completely destroy the knowledge learned during pretraining.

Therefore, fine-tuning commonly uses a **small learning rate**.

---

# 2. What Is a Parameter?

A neural network contains learnable parameters.

For a simple linear layer:

\[
y=Wx+b
\]

where:

- \(W\) = weights
- \(b\) = bias

These are parameters learned during training.

For a Transformer, there are millions or billions of parameters distributed across many layers.

---

# 3. What Does Training Actually Change?

Suppose a parameter initially has:

\[
W=0.50
\]

After calculating the loss, backpropagation gives:

\[
\frac{\partial L}{\partial W}=0.20
\]

A basic gradient-descent update is:

\[
\boxed{
W_{new}=W_{old}-\eta\frac{\partial L}{\partial W}
}
\]

where:

\[
\eta=\text{learning rate}
\]

---

# 4. Simple Numerical Example

Suppose:

\[
W=0.50
\]

and:

\[
\frac{\partial L}{\partial W}=0.20
\]

Learning rate:

\[
\eta=0.01
\]

Then:

\[
W_{new}
=
0.50-(0.01)(0.20)
\]

\[
W_{new}=0.498
\]

So:

```text
Before = 0.500
After  = 0.498
```

The parameter changed by a small amount.

---

# 5. What Is the Learning Rate?

The learning rate controls:

> **How large a parameter update should be.**

It is usually represented by:

\[
\boxed{\eta}
\]

Examples:

```text
0.1
0.01
0.001
0.00001
```

A larger learning rate generally means larger updates.

A smaller learning rate generally means smaller updates.

---

# 6. Why Is Learning Rate Important?

Imagine trying to reach the lowest point of a valley.

```text
        \          /
         \        /
          \      /
           \____/
```

### Very large steps

You may jump over the minimum.

### Very small steps

You may eventually reach it, but training can be very slow.

Therefore, we need an appropriate learning rate.

---

# 7. Learning Rate Too Large

If:

\[
\eta=1
\]

updates may become very large.

Possible result:

```text
Loss
 ↑
 |      /\    /\
 |     /  \  /  \
 |____/    \/    \____
 +----------------------→ Training
```

The optimization process may become unstable.

The loss can oscillate or even diverge.

---

# 8. Learning Rate Too Small

If:

\[
\eta=10^{-8}
\]

updates may become extremely small.

Training may be:

- Very slow
- Inefficient
- Unable to make meaningful progress within the available training time

---

# 9. Learning Rate in Fine-Tuning

A pretrained Transformer already contains useful language representations.

If we use an extremely large learning rate:

```text
Pretrained knowledge
       ↓
Huge updates
       ↓
Knowledge can be badly disrupted
```

With a smaller learning rate:

```text
Small controlled updates
       ↓
Adapt to task
```

Typical fine-tuning learning rates might be around:

```text
1e-5
2e-5
3e-5
5e-5
```

These are examples, not universal rules.

The appropriate value depends on:

- Model
- Dataset size
- Task
- Batch size
- Optimizer
- Number of epochs
- Full fine-tuning vs parameter-efficient fine-tuning

---

# 10. Gradient Descent

The simplest optimizer is gradient descent.

The update rule is:

\[
\boxed{
\theta_{t+1}
=
\theta_t
-
\eta\nabla_\theta L
}
\]

where:

- \(\theta\) = model parameters
- \(L\) = loss
- \(\nabla_\theta L\) = gradient
- \(\eta\) = learning rate

This equation is fundamental to understanding optimization.

---

# 11. Connecting This to PyTorch

You have already seen:

```python
loss.backward()
```

This calculates gradients.

Then:

```python
optimizer.step()
```

uses those gradients to update parameters.

Simplified flow:

```text
loss.backward()
      ↓
∂Loss/∂Parameter
      ↓
optimizer.step()
      ↓
Parameter update
```

---

# 12. What Does `loss.backward()` Do?

Suppose:

```python
loss = ...
```

Then:

```python
loss.backward()
```

uses automatic differentiation to calculate gradients.

For a parameter \(W\):

\[
\frac{\partial L}{\partial W}
\]

The gradient tells the optimizer:

> **How does changing this parameter affect the loss?**

---

# 13. What Does `optimizer.step()` Do?

For example:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
```

After:

```python
loss.backward()
```

we call:

```python
optimizer.step()
```

The optimizer updates parameters using their gradients.

Conceptually:

\[
W_{new}
=
W_{old}
-
\eta
\frac{\partial L}{\partial W}
\]

---

# 14. Why Do We Need an Optimizer?

The optimizer decides:

> **How should the parameters be updated using the gradients?**

Common optimizers include:

```text
SGD
Adam
AdamW
```

For Transformer fine-tuning, **AdamW** is especially common.

---

# 15. SGD

SGD stands for:

\[
\boxed{\text{Stochastic Gradient Descent}}
\]

The basic update is:

\[
\theta_{t+1}
=
\theta_t
-
\eta g_t
\]

where:

\[
g_t=\nabla_\theta L
\]

A PyTorch example:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
```

---

# 16. Why Adam?

Adam maintains additional information about gradients to adapt the update for each parameter.

Adam stands for:

\[
\boxed{\text{Adaptive Moment Estimation}}
\]

It maintains estimates related to:

- First moment of gradients
- Second moment of gradients

---

# 17. First Moment

Adam maintains an exponential moving average of gradients:

\[
m_t
=
\beta_1m_{t-1}
+
(1-\beta_1)g_t
\]

where:

- \(m_t\) = first moment estimate
- \(g_t\) = current gradient
- \(\beta_1\) = decay parameter

This captures information about the direction of gradients over time.

---

# 18. Second Moment

Adam also maintains an exponential moving average of squared gradients:

\[
v_t
=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
\]

where:

- \(v_t\) = second moment estimate
- \(\beta_2\) = decay parameter

This provides information about the magnitude/scale of gradients.

---

# 19. Bias Correction

Adam applies bias correction:

\[
\hat m_t
=
\frac{m_t}{1-\beta_1^t}
\]

and:

\[
\hat v_t
=
\frac{v_t}{1-\beta_2^t}
\]

Then the parameter update is approximately:

\[
\boxed{
\theta_t
=
\theta_{t-1}
-
\eta
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
}
\]

This is the core idea behind Adam.

---

# 20. Why Adam Is Useful

Adam adapts updates according to gradient behavior.

Instead of every parameter receiving exactly the same raw gradient-based step, Adam uses running gradient statistics to adapt the update.

This often makes optimization easier in deep neural networks.

---

# 21. Adam vs AdamW

This distinction is important.

You will commonly encounter:

```text
Adam
AdamW
```

They are related, but their treatment of **weight decay** differs.

AdamW was introduced to decouple weight decay from the adaptive gradient update.

---

# 22. What Is Weight Decay?

Weight decay is a regularization technique.

The basic idea is:

> Prevent model parameters from growing unnecessarily large.

Conceptually:

```text
Without regularization
        ↓
Parameters may become excessively large
        ↓
Potential overfitting

With weight decay
        ↓
Parameters are gently pushed toward smaller values
```

---

# 23. Simple Intuition

Suppose:

\[
W=10
\]

Weight decay applies a small shrinking effect.

Conceptually:

\[
W\rightarrow9.99
\]

The exact update depends on the optimizer and implementation.

Important idea:

\[
\boxed{\text{Weight decay gently shrinks parameters}}
\]

---

# 24. Why Use Weight Decay During Fine-Tuning?

Fine-tuning datasets can be much smaller than the data used for pretraining.

Therefore:

```text
Huge pretrained model
        ↓
Small task dataset
        ↓
Risk of overfitting
```

Weight decay can act as a regularizer.

---

# 25. AdamW

A simplified view of AdamW is:

```text
Gradient-based Adam update
          +
Decoupled weight decay
```

In PyTorch:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01
)
```

Here:

```text
lr = 2e-5
```

controls the learning rate.

```text
weight_decay = 0.01
```

controls the strength of weight decay.

---

# 26. Learning Rate vs Weight Decay

Do not confuse them.

### Learning rate

Controls:

> How large are parameter updates?

### Weight decay

Controls:

> How strongly are parameters regularized/shrunk?

Therefore:

```text
Learning Rate → Update magnitude
Weight Decay  → Regularization
```

---

# 27. Learning Rate Scheduler

Instead of keeping the learning rate constant throughout training:

```text
Learning rate
|
|-------------------------
|
+------------------------→ steps
```

we can change it over time:

```text
Learning rate
|
|\
| \
|  \
|   \____
|
+------------------------→ steps
```

This is called **learning-rate scheduling**.

---

# 28. Why Change the Learning Rate?

Early in training, larger updates may be useful.

Later, smaller updates may be better for fine adjustment.

Therefore:

```text
Early training
     ↓
Larger updates

Later training
     ↓
Smaller updates
```

A scheduler controls this change.

---

# 29. Learning Rate Schedule Example

Suppose:

```text
Initial LR = 0.001
```

A scheduler may produce:

```text
Step       Learning Rate

0          0.0010
100        0.0009
200        0.0008
300        0.0007
...
```

The exact schedule depends on the scheduler.

---

# 30. Warmup

An important technique in Transformer training is:

\[
\boxed{\text{Learning Rate Warmup}}
\]

Instead of immediately starting at the maximum learning rate, we gradually increase it during the first part of training.

Example:

```text
Learning Rate

     /---------
    /
   /
  /
 /
+----------------→ Steps
```

---

# 31. Why Warmup?

At the beginning of training, the optimization process may be sensitive.

Warmup provides a gradual start:

```text
Step 0
  ↓
Very small LR
  ↓
Gradually increase
  ↓
Target LR
```

This can improve optimization stability.

---

# 32. Warmup Example

Suppose:

```text
Target learning rate = 2e-5
Warmup steps = 1000
```

Conceptually:

```text
Step 0     → 0
Step 250   → 0.5e-5
Step 500   → 1.0e-5
Step 750   → 1.5e-5
Step 1000  → 2.0e-5
```

The exact implementation depends on the scheduler.

---

# 33. Warmup + Decay

A common conceptual schedule is:

```text
Learning Rate

       /\
      /  \
     /    \
    /      \
___/        \________
+--------------------→ Steps
  Warmup      Decay
```

First:

\[
LR\uparrow
\]

Then:

\[
LR\downarrow
\]

This combines:

```text
Warmup
+
Learning-rate decay
```

---

# 34. Why Is This Useful for Fine-Tuning?

Fine-tuning is often sensitive because we begin from a pretrained model.

We don't necessarily want a huge update immediately.

Instead:

```text
Small initial updates
       ↓
Stable optimization
       ↓
Normal learning rate
       ↓
Gradual reduction
       ↓
Fine adjustment
```

---

# 35. Hugging Face `TrainingArguments`

In Hugging Face Transformers:

```python
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    warmup_ratio=0.1
)
```

Here:

```text
learning_rate
→ base learning rate

weight_decay
→ regularization

num_train_epochs
→ number of passes through training data

warmup_ratio
→ fraction of training steps used for warmup
```

These values are examples, not universal values.

---

# 36. Learning Rate Scheduler in Trainer

Hugging Face Trainer can manage the optimizer and scheduler for you.

Conceptually:

```text
Trainer
   ↓
Optimizer
   ↓
AdamW
   ↓
Scheduler
   ↓
Parameter updates
```

You generally don't need to manually write:

```python
optimizer.step()
```

when using `Trainer`.

Trainer handles the training loop internally.

---

# 37. Manual PyTorch Training Loop

To understand what Trainer is doing internally:

```python
for batch in dataloader:

    optimizer.zero_grad()

    outputs = model(**batch)

    loss = outputs.loss

    loss.backward()

    optimizer.step()

    scheduler.step()
```

---

# 38. `optimizer.zero_grad()`

```python
optimizer.zero_grad()
```

clears previously accumulated gradients.

PyTorch gradients accumulate by default.

Without clearing them:

```text
Current gradient
      +
Previous gradient
      +
Older gradient
```

could accumulate unintentionally.

Therefore, normally:

```python
optimizer.zero_grad()
```

is called before the next update.

---

# 39. Forward Pass

```python
outputs = model(**batch)
```

The model processes the batch.

Conceptually:

```text
Input
 ↓
Transformer
 ↓
Logits
 ↓
Loss
```

If labels are provided, many Hugging Face models automatically calculate the loss.

---

# 40. Backward Pass

```python
loss.backward()
```

PyTorch calculates gradients.

Conceptually:

```text
Loss
 ↓
Computational Graph
 ↓
Gradients
 ↓
∂L/∂θ
```

---

# 41. Optimizer Step

```python
optimizer.step()
```

uses the gradients to update parameters.

Conceptually:

\[
\theta
\rightarrow
\theta+\Delta\theta
\]

where \(\Delta\theta\) is determined by the optimizer.

---

# 42. Scheduler Step

```python
scheduler.step()
```

updates the learning rate according to the scheduler.

Therefore:

```text
optimizer.step()
     ↓
Parameters change

scheduler.step()
     ↓
Learning rate changes
```

This distinction is important.

---

# 43. Complete Training Loop

```python
for batch in dataloader:

    optimizer.zero_grad()

    outputs = model(**batch)

    loss = outputs.loss

    loss.backward()

    optimizer.step()

    scheduler.step()
```

Mental model:

```text
zero_grad()
     ↓
forward()
     ↓
loss
     ↓
backward()
     ↓
optimizer.step()
     ↓
scheduler.step()
```

---

# 44. What Happens to the Pretrained Parameters?

Suppose a pretrained parameter is:

\[
\theta=0.500000
\]

After one update:

\[
\theta=0.499998
\]

After another:

\[
\theta=0.500003
\]

After many updates:

```text
Pretrained parameter
        ↓
small update
        ↓
small update
        ↓
small update
        ↓
task-adapted parameter
```

The model gradually adapts to the new task.

---

# 45. Does Fine-Tuning Change Every Parameter?

In **full fine-tuning**, normally the trainable parameters of the model are updated.

Conceptually:

```text
Transformer
├── Layer 1 → updated
├── Layer 2 → updated
├── Layer 3 → updated
├── ...
└── Layer N → updated
```

In parameter-efficient fine-tuning methods such as LoRA, most original parameters are frozen and only additional trainable parameters are updated.

LoRA will be covered later.

---

# 46. Why Use a Small Learning Rate?

Suppose:

```text
Pretrained Model
     ↓
Excellent general representation
```

If we apply enormous updates:

```text
Huge LR
   ↓
Large parameter changes
   ↓
Pretrained representation can be disrupted
```

With a smaller learning rate:

```text
Small LR
   ↓
Small controlled changes
   ↓
Adapt to task
```

This is one reason fine-tuning commonly uses a smaller learning rate than training from scratch.

---

# 47. Learning Rate vs Number of Epochs

These are different concepts.

### Learning rate

Controls:

> How much parameters move during each update.

### Epochs

Controls:

> How many times the model sees the training dataset.

Example:

```text
LR = 2e-5
Epochs = 3
```

means:

```text
Small parameter updates
+
Three passes through training data
```

---

# 48. What If We Increase Epochs Too Much?

Example:

```text
Epoch 1 → good
Epoch 2 → better
Epoch 3 → best
Epoch 4 → overfitting
Epoch 5 → more overfitting
```

More training is not automatically better.

This connects directly to Lesson 6:

```text
Training longer
       ↓
Training loss may decrease
       ↓
Validation performance may worsen
```

---

# 49. Learning Rate and Batch Size

Batch size also affects optimization.

Suppose:

```text
Batch size = 8
```

The model processes 8 examples before an update.

With:

```text
Batch size = 32
```

the model processes 32 examples before an update.

Changing batch size changes optimization dynamics.

Therefore, learning rate and batch size should not always be considered independently.

---

# 50. Gradient Accumulation Preview

Suppose your GPU cannot fit:

```text
Batch size = 32
```

You might instead use:

```text
Batch size = 8
Gradient accumulation steps = 4
```

Conceptually:

```text
Batch 1 → gradients
Batch 2 → accumulate
Batch 3 → accumulate
Batch 4 → accumulate
              ↓
        optimizer.step()
```

This creates an effective larger batch.

We will study this more deeply in Lesson 8.

---

# 51. Important Hyperparameters

For fine-tuning, pay particular attention to:

```text
learning_rate
weight_decay
num_train_epochs
batch_size
warmup_ratio
scheduler
gradient_accumulation_steps
```

These can strongly affect training behavior.

---

# 52. Typical Fine-Tuning Configuration

A conceptual example:

```python
training_args = TrainingArguments(
    output_dir="./results",

    learning_rate=2e-5,

    weight_decay=0.01,

    num_train_epochs=3,

    warmup_ratio=0.1,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True
)
```

These values are examples only.

Do not memorize them as universal values.

---

# 53. Complete Mental Model

```text
                 Input
                   ↓
                Model
                   ↓
              Prediction
                   ↓
                  Loss
                   ↓
           loss.backward()
                   ↓
              Gradients
                   ↓
              Optimizer
                   ↓
            Parameter Update
                   ↓
           Scheduler controls
            learning rate
                   ↓
               Repeat
```

---

# 54. The Most Important Distinction

Remember:

```text
Gradient
   ↓
Tells us the direction/sensitivity of loss

Learning Rate
   ↓
Controls update magnitude

Optimizer
   ↓
Determines how gradients are used

Weight Decay
   ↓
Regularizes/shrinks parameters

Scheduler
   ↓
Changes learning rate over training
```

---

# 55. Questions & Answers

## Q1. What is a learning rate?

### Answer

The learning rate controls the magnitude of parameter updates.

\[
\theta_{new}
=
\theta_{old}
-
\eta\nabla_\theta L
\]

---

## Q2. Why is learning rate important during fine-tuning?

### Answer

A very large learning rate can make large parameter changes and potentially disrupt useful pretrained representations.

Fine-tuning therefore often uses relatively small learning rates.

---

## Q3. What does `loss.backward()` do?

### Answer

It calculates gradients of the loss with respect to the model's trainable parameters using automatic differentiation.

---

## Q4. What does `optimizer.step()` do?

### Answer

It uses the calculated gradients to update the model parameters according to the optimizer's update rule.

---

## Q5. What does `optimizer.zero_grad()` do?

### Answer

It clears previously accumulated gradients.

---

## Q6. What is Adam?

### Answer

Adam is an adaptive optimization algorithm that maintains running estimates related to the first and second moments of gradients.

---

## Q7. What is AdamW?

### Answer

AdamW is an Adam-based optimizer that decouples weight decay from the adaptive gradient update.

It is widely used for Transformer training and fine-tuning.

---

## Q8. What is weight decay?

### Answer

Weight decay is a regularization mechanism that encourages parameters to remain smaller and can help reduce overfitting.

---

## Q9. What is the difference between learning rate and weight decay?

### Answer

Learning rate controls the magnitude of optimization updates.

Weight decay controls the strength of parameter regularization/shrinkage.

---

## Q10. What is a learning-rate scheduler?

### Answer

A scheduler changes the learning rate during training according to a predefined schedule.

---

## Q11. What is warmup?

### Answer

Warmup gradually increases the learning rate from a small value to the target learning rate during the initial training steps.

---

## Q12. Why is warmup useful?

### Answer

It can make early optimization more stable, especially in large Transformer models.

---

## Q13. What is the difference between `optimizer.step()` and `scheduler.step()`?

### Answer

```text
optimizer.step()
→ Updates model parameters

scheduler.step()
→ Updates the learning rate
```

---

## Q14. Does a smaller learning rate always produce a better model?

### Answer

No.

If it is too small, learning can become extremely slow or ineffective.

The learning rate must be appropriate for the model, dataset, optimizer, and training setup.

---

## Q15. Does fine-tuning change pretrained parameters?

### Answer

In full fine-tuning, trainable pretrained parameters are updated using gradients.

In parameter-efficient methods such as LoRA, most pretrained parameters can remain frozen while additional parameters are trained.

---

# 56. Self-Test

Try answering these without looking back:

1. What is a learning rate?
2. What does a gradient represent?
3. What does `loss.backward()` do?
4. What does `optimizer.step()` do?
5. Why do we call `optimizer.zero_grad()`?
6. What is SGD?
7. What is Adam?
8. What is AdamW?
9. What is weight decay?
10. How is weight decay different from learning rate?
11. What is a learning-rate scheduler?
12. What is warmup?
13. Why might fine-tuning use a smaller learning rate?
14. What happens if the learning rate is too large?
15. What happens if the learning rate is too small?
16. What is the difference between `optimizer.step()` and `scheduler.step()`?
17. What happens to pretrained parameters during full fine-tuning?
18. Why shouldn't we simply train for a huge number of epochs?

---

# 57. Final Summary

The most important equations are:

### Gradient Descent

\[
\boxed{
\theta_{t+1}
=
\theta_t
-
\eta\nabla_\theta L
}
\]

### Adam First Moment

\[
\boxed{
m_t
=
\beta_1m_{t-1}
+
(1-\beta_1)g_t
}
\]

### Adam Second Moment

\[
\boxed{
v_t
=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
}
\]

### Adam Update

\[
\boxed{
\theta_t
=
\theta_{t-1}
-
\eta
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
}
\]

The practical PyTorch training flow is:

```python
optimizer.zero_grad()

outputs = model(**batch)

loss = outputs.loss

loss.backward()

optimizer.step()

scheduler.step()
```

Remember:

\[
\boxed{
\text{Gradient}=\text{What direction should parameters move?}
}
\]

\[
\boxed{
\text{Learning Rate}=\text{How large should the update be?}
}
\]

\[
\boxed{
\text{Optimizer}=\text{How should gradients be used?}
}
\]

\[
\boxed{
\text{Weight Decay}=\text{Regularization}
}
\]

\[
\boxed{
\text{Scheduler}=\text{How learning rate changes over time}
}
\]

---

# 58. Next Lesson

## Lesson 8 — Practical Fine-Tuning: Checkpoints, Gradient Accumulation, Mixed Precision & GPU Memory

Next we will move from the theory of optimization to practical GPU fine-tuning.

We will learn:

```text
Checkpointing
      ↓
Gradient Accumulation
      ↓
Effective Batch Size
      ↓
Mixed Precision
      ↓
FP32 / FP16 / BF16
      ↓
GPU Memory
      ↓
Gradient Clipping
      ↓
Practical Training Configuration
```

This lesson will explain how to fine-tune Transformer models when GPU VRAM is limited.
