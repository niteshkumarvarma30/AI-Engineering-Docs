# Lesson 16 — Supervised Fine-Tuning (SFT)

> **Goal:** Understand how Supervised Fine-Tuning works internally, including instruction/response datasets, tokenization, labels, label shifting, loss masking, logits, cross-entropy loss, backpropagation, LoRA, QLoRA, chat templates, attention masks, batching, and the complete SFT pipeline.

---

# 1. What Is SFT?

**SFT = Supervised Fine-Tuning.**

The word **supervised** is important.

We provide the model with:

```text
Input
+
Expected Output
```

For example:

```text
Instruction:
What is LoRA?

Expected response:
LoRA is a parameter-efficient fine-tuning method.
```

The model generates a prediction and compares it with the expected answer.

The difference becomes the **loss**.

Then backpropagation is used to update the trainable parameters.

---

# 2. SFT in One Equation

Suppose:

\[
x=\text{input}
\]

and:

\[
y=\text{target response}
\]

The model learns:

\[
\boxed{
P(y|x)
}
\]

Because an autoregressive LLM generates tokens one at a time:

\[
P(y|x)
=
\prod_{t=1}^{T}
P(y_t|x,y_{<t})
\]

where:

- \(x\) = instruction/context
- \(y_t\) = target token at position \(t\)
- \(y_{<t}\) = previous target tokens

The training objective is to maximize the probability of the correct response.

Equivalently, we minimize negative log-likelihood:

\[
\boxed{
\mathcal{L}
=
-\sum_{t=1}^{T}
\log P(y_t|x,y_{<t})
}
\]

---

# 3. Simple Example

Suppose:

```text
Instruction:
What is LoRA?

Response:
LoRA is a parameter-efficient fine-tuning method.
```

The tokenizer converts this into tokens.

Conceptually:

```text
Input:
[What, is, LoRA, ?]

Target:
[LoRA, is, a, parameter-efficient, ...]
```

The model predicts each target token.

For example:

```text
What is LoRA?
       ↓
     "LoRA"

What is LoRA? LoRA
       ↓
      "is"

What is LoRA? LoRA is
       ↓
       "a"
```

The model is trained to increase the probability of the correct next token.

---

# 4. SFT Is Still Next-Token Prediction

This is extremely important.

SFT does **not** replace the language-model objective.

The model is still fundamentally doing:

\[
\boxed{
\text{Predict the next token}
}
\]

The difference is the **data distribution**.

During pretraining:

```text
Huge general corpus
       ↓
Next-token prediction
```

During SFT:

```text
Instruction + desired response
       ↓
Next-token prediction
```

So:

```text
Same basic objective
+
Different training data
=
Instruction-following behavior
```

---

# 5. Pretraining vs SFT

| | Pretraining | SFT |
|---|---|---|
| Data | Huge general corpus | Curated supervised examples |
| Input | General text | Instructions/conversations |
| Target | Next token | Desired response tokens |
| Goal | Learn general language | Learn desired behavior/tasks |
| Typical scale | Extremely large | Much smaller |
| Supervision | Usually self-supervised | Supervised |

---

# 6. What Does "Supervised" Mean Here?

Consider:

```text
Input:
Explain gradient descent.

Target:
Gradient descent is an optimization algorithm...
```

The target response acts as the supervision signal.

The model predicts:

```text
Gradient
```

and the training data tells it:

```text
Correct token = Gradient
```

Then:

```text
prediction
    ↓
compare with target
    ↓
loss
    ↓
gradient
    ↓
parameter update
```

---

# 7. The SFT Training Loop

At a high level:

```text
for each batch:

    input_ids
        ↓
    model
        ↓
    logits
        ↓
    calculate loss
        ↓
    loss.backward()
        ↓
    optimizer.step()
        ↓
    optimizer.zero_grad()
```

This is the core training mechanism.

---

# 8. Forward Pass

Suppose the input token IDs are:

```python
input_ids = [
    [101, 205, 304, 502]
]
```

The model processes them:

```text
input_ids
    ↓
Embedding
    ↓
Transformer layers
    ↓
Final hidden states
    ↓
LM Head
    ↓
Logits
```

The output is a tensor of logits.

---

# 9. What Are Logits?

Suppose the vocabulary has:

\[
V=50,000
\]

tokens.

For each input position, the model produces a score for every vocabulary token.

Therefore, the output shape is conceptually:

\[
\boxed{
(batch,\ sequence,\ vocabulary)
}
\]

For example:

```text
(batch_size, sequence_length, vocab_size)
```

If:

```text
batch_size = 2
sequence_length = 128
vocab_size = 50,000
```

then:

\[
\text{logits shape}
=
(2,128,50000)
\]

---

# 10. Logits vs Probabilities

The model initially produces **logits**.

Example:

```text
Token A → 2.1
Token B → 0.4
Token C → 5.7
Token D → -1.2
```

Softmax converts logits into probabilities:

\[
P_i=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
\]

Conceptually:

```text
Logits
  ↓
Softmax
  ↓
Probabilities
```

The highest-probability token is the model's most likely prediction.

---

# 11. Cross-Entropy Loss

The predicted probability is compared with the correct target.

Suppose the correct token is:

```text
Token C
```

and the model predicts:

```text
A → 0.10
B → 0.05
C → 0.80
D → 0.05
```

Then the loss for this token is:

\[
-\log(0.80)
\]

If the model predicts the correct token with high probability:

\[
P(\text{correct})\rightarrow1
\]

then:

\[
Loss\rightarrow0
\]

If it assigns low probability:

\[
P(\text{correct})\rightarrow0
\]

then the loss becomes large.

---

# 12. Cross-Entropy Formula

For a single target token:

\[
\boxed{
L=-\log P(y)
}
\]

For the entire response:

\[
\boxed{
L
=
-\sum_{t=1}^{T}
\log P(y_t|x,y_{<t})
}
\]

Often the implementation uses the mean over valid tokens rather than a raw sum.

---

# 13. The Important Question: What Is the Input and What Is the Label?

Suppose:

```text
User:
What is LoRA?

Assistant:
LoRA is a parameter-efficient fine-tuning method.
```

We combine them into a sequence:

```text
User:
What is LoRA?

Assistant:
LoRA is a parameter-efficient fine-tuning method.
```

But the model should generally be trained to predict the **assistant response**, not simply reproduce every part of the prompt.

This leads to **loss masking**.

---

# 14. Labels

The model receives token IDs.

For example, conceptually:

```text
Tokens:

[user] What is LoRA?
[assistant] LoRA is a parameter-efficient method.
```

We create labels corresponding to the target tokens.

Conceptually:

```text
input_ids:
[user] What is LoRA? [assistant] LoRA is a parameter-efficient method.

labels:
   -100   -100   -100  -100     -100      LoRA is a parameter-efficient method.
```

The exact tokenization depends on the tokenizer and chat template.

The important idea is:

\[
\boxed{
\text{Only selected tokens contribute to the loss}
}
\]

---

# 15. What Does `-100` Mean?

In common PyTorch/Hugging Face cross-entropy workflows, the label value:

```python
-100
```

is used as an **ignore index**.

So if:

```text
labels = [-100, -100, 45, 67, 89]
```

the loss ignores the first two positions.

Conceptually:

```text
Token       Label       Loss?
--------------------------------
User        -100        ❌
Question    -100        ❌
Assistant   -100        ❌
LoRA        45          ✅
is          67          ✅
efficient   89          ✅
```

This is called **loss masking**.

---

# 16. Why Loss Masking Is Important

Suppose the model sees:

```text
User:
Explain LoRA.

Assistant:
LoRA is a parameter-efficient method.
```

We usually don't want the model's training loss to encourage it to generate the user prompt.

We want it to learn:

```text
Given the conversation,
generate a good assistant response.
```

Therefore:

```text
Prompt tokens
     ↓
Ignored for loss

Assistant tokens
     ↓
Used for loss
```

---

# 17. SFT Loss Masking

Conceptually:

```text
                 Full Sequence
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
      User tokens              Assistant tokens
          ↓                         ↓
       Ignore                    Train
          ↓                         ↓
       -100                  Cross-Entropy
```

This is a major concept in modern conversational SFT.

---

# 18. Not Every SFT Setup Uses Exactly the Same Mask

The exact loss masking strategy depends on:

- Dataset format
- Model architecture
- Training library
- Chat template
- Whether system/user tokens are included in the loss
- Whether the objective is full-sequence causal LM training or assistant-only training

The important principle is:

> **Determine which tokens are intended to contribute to the supervised objective.**

---

# 19. Chat Template

Modern chat models often use structured messages:

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful AI tutor."
    },
    {
        "role": "user",
        "content": "Explain LoRA."
    },
    {
        "role": "assistant",
        "content": "LoRA is a parameter-efficient fine-tuning method."
    }
]
```

The tokenizer converts these messages into the model-specific format.

Conceptually:

```text
Messages
   ↓
Chat Template
   ↓
Special Tokens
   ↓
Token IDs
```

---

# 20. Why Not Manually Build the Chat Format?

Different models may use different control tokens.

For example, one model may conceptually expect:

```text
<|user|>
...
<|assistant|>
...
```

while another may use a different format.

Therefore, use the model's tokenizer/chat template when available.

Conceptually:

```python
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False
)
```

---

# 21. SFT With a Chat Template

The pipeline becomes:

```text
Messages
   ↓
Chat Template
   ↓
Formatted Conversation
   ↓
Tokenizer
   ↓
Input IDs
   ↓
Labels
   ↓
Loss Masking
   ↓
Model
```

---

# 22. The Shifted-Token Concept

Suppose:

```text
Tokens:

A B C D
```

For autoregressive training, the model predicts:

```text
A → B
B → C
C → D
```

So conceptually:

```text
Input:

A B C

Target:

B C D
```

This is called **shifting** the labels.

---

# 23. Why Shift the Labels?

At position \(t\), the model should predict:

\[
x_{t+1}
\]

using:

\[
x_{\leq t}
\]

So:

```text
Input:
<BOS> LoRA is

Target:
LoRA is efficient
```

The model learns:

```text
<BOS> → LoRA
LoRA → is
LoRA is → efficient
```

---

# 24. Causal Mask + Label Shift

These two mechanisms work together.

### Causal mask

Prevents looking into the future.

### Label shift

Defines which next token should be predicted.

Together:

```text
Previous tokens
      ↓
Causal attention
      ↓
Predict next token
      ↓
Compare with shifted label
      ↓
Loss
```

---

# 25. Complete SFT Example

Suppose:

```text
User:
What is LoRA?

Assistant:
LoRA is a parameter-efficient fine-tuning method.
```

Conceptually:

```text
Conversation:

[USER] What is LoRA? [ASSISTANT] LoRA is a parameter-efficient fine-tuning method.
```

After tokenization:

```text
input_ids:
[u1, u2, u3, u4, a1, a2, a3, a4, a5, ...]
```

Labels:

```text
labels:
[-100, -100, -100, -100, a1, a2, a3, a4, a5, ...]
```

The model predicts next tokens.

Cross-entropy is calculated only where labels are valid.

---

# 26. SFT With Full Fine-Tuning

If we perform full fine-tuning:

```text
Base Model
     ↓
All/most model parameters
     ↓
Trainable
```

Training updates the model weights.

---

# 27. SFT With LoRA

With LoRA:

```text
Base Model
     ↓
Frozen

LoRA A
     ↓
Trainable

LoRA B
     ↓
Trainable
```

The SFT objective remains the same.

Only the trainable parameter set changes.

This is a crucial distinction.

---

# 28. SFT + LoRA

Conceptually:

```text
Instruction Dataset
        ↓
Chat Template
        ↓
Tokenization
        ↓
Labels
        ↓
Loss Masking
        ↓
Forward Pass
        ↓
Cross-Entropy
        ↓
Backpropagation
        ↓
Gradients
        ↓
LoRA A/B
        ↓
Optimizer Update
```

The base model weights remain frozen.

---

# 29. SFT + QLoRA

QLoRA adds quantization.

Conceptually:

```text
Pretrained Model
      ↓
Quantized Base Weights
      ↓
Frozen

LoRA A + B
      ↓
Trainable
```

The training objective remains supervised next-token prediction.

So:

```text
SFT
=
Training objective/data

QLoRA
=
Parameter + memory efficient adaptation strategy
```

---

# 30. Role of the Optimizer

After computing the loss:

```python
loss.backward()
```

PyTorch calculates gradients.

Then:

```python
optimizer.step()
```

updates the trainable parameters.

Then:

```python
optimizer.zero_grad()
```

clears the gradients.

So:

```text
Forward
   ↓
Loss
   ↓
Backward
   ↓
Gradients
   ↓
Optimizer
   ↓
Updated parameters
```

---

# 31. SFT Training Loop in PyTorch

A simplified conceptual loop:

```python
for batch in dataloader:

    optimizer.zero_grad()

    outputs = model(
        input_ids=batch["input_ids"],
        labels=batch["labels"]
    )

    loss = outputs.loss

    loss.backward()

    optimizer.step()
```

This is the core training mechanism.

---

# 32. What Happens Inside `model(..., labels=...)`?

For many causal language models, passing:

```python
labels=labels
```

allows the model implementation to calculate the language-modeling loss.

Conceptually:

```text
input_ids
    ↓
Transformer
    ↓
Logits
    ↓
Shift logits / labels
    ↓
Cross-Entropy
    ↓
loss
```

The exact implementation depends on the model architecture/library.

---

# 33. Small Mathematical Example

Suppose the vocabulary contains only four tokens:

```text
A
B
C
D
```

At one position the model produces logits:

\[
z=[1.0,2.0,0.5,0.0]
\]

Softmax gives probabilities.

If the correct target is `B`, the loss is:

\[
L=-\log P(B)
\]

If the model assigns a high probability to `B`, loss is low.

If it assigns a low probability to `B`, loss is high.

---

# 34. Why Cross-Entropy Works

Suppose:

\[
P(\text{correct})=0.9
\]

Then:

\[
L=-\log(0.9)
\]

which is small.

But if:

\[
P(\text{correct})=0.1
\]

then:

\[
L=-\log(0.1)
\]

which is much larger.

Therefore, minimizing cross-entropy encourages the model to assign higher probability to correct tokens.

---

# 35. SFT and Autograd

SFT uses the same PyTorch `autograd` mechanism you previously learned.

```text
Loss
 ↓
autograd
 ↓
∂L/∂θ
 ↓
Gradients
```

where \(\theta\) represents trainable parameters.

With full fine-tuning:

\[
\theta=W
\]

With LoRA:

\[
\theta=\{A,B\}
\]

while:

\[
W
\]

is frozen.

---

# 36. The Most Important Connection

### SFT determines the objective:

\[
\boxed{
\text{Predict the desired response tokens}
}
\]

### LoRA determines the trainable parameters:

\[
\boxed{
\text{Train }A,B
}
\]

### Dataset determines the examples:

\[
\boxed{
\text{Instruction + Response}
}
\]

### Chat template determines the representation:

\[
\boxed{
\text{Messages}\rightarrow\text{Model-specific token format}
}
\]

These are separate concepts that work together.

---

# 37. SFT Data Collator

When training batches, we need to prepare:

```text
input_ids
attention_mask
labels
```

A data collator can handle batching and padding.

Conceptually:

```text
Dataset
   ↓
Data Collator
   ↓
Batch
   ├── input_ids
   ├── attention_mask
   └── labels
```

The exact collator depends on the training setup.

---

# 38. `attention_mask`

The attention mask tells the model which positions are valid versus padding or otherwise masked positions.

Example:

```text
input_ids:
[101, 200, 300, 0, 0]

attention_mask:
[ 1,   1,   1,  0, 0]
```

Conceptually:

```text
1 → valid token
0 → masked/padding position
```

This is distinct from the **causal attention mask** used to prevent future-token access.

---

# 39. Two Different Types of Masking

This is a common source of confusion.

## Padding / Attention Mask

Deals with padding or valid input positions.

```text
1 → attend/use
0 → ignore padding
```

## Causal Mask

Prevents looking at future tokens.

```text
Current token
      ↓
Can see past
Cannot see future
```

They solve different problems.

---

# 40. Loss Mask vs Attention Mask

Another important distinction:

### Attention mask

Controls what the model attends to.

### Loss mask

Controls which tokens contribute to the loss.

For example:

```text
Attention mask:
[1, 1, 1, 1, 1, 1]

Loss mask:
[-100, -100, -100, 45, 67, 89]
```

The model can process all relevant tokens, but only selected tokens contribute to the supervised loss.

---

# 41. Complete SFT Data Flow

```text
Raw Dataset
     ↓
Clean
     ↓
Instruction / Response
     ↓
Chat Template
     ↓
Tokenization
     ↓
input_ids
     ↓
Padding / Truncation
     ↓
attention_mask
     ↓
Create labels
     ↓
Loss masking
     ↓
Batch
     ↓
LLM
     ↓
Logits
     ↓
Shift logits / labels
     ↓
Cross-Entropy
     ↓
Loss
     ↓
Backpropagation
     ↓
Trainable Parameters
     ↓
Optimizer
```

---

# 42. SFT With Hugging Face TRL

Later we will use **TRL** to simplify this process.

Conceptually:

```text
Dataset
   ↓
SFTTrainer
   ↓
Tokenizer / Processor
   ↓
Batching
   ↓
Loss
   ↓
Backpropagation
   ↓
Optimizer
```

Instead of manually implementing every component.

However, understanding the underlying pipeline first is important.

---

# 43. Why Learn the Manual Pipeline First?

If you immediately use:

```python
SFTTrainer(...)
```

you may get a working model without understanding what is happening.

You should understand:

```text
input_ids
labels
attention_mask
loss
gradients
optimizer
LoRA
```

before relying on a high-level trainer.

Then the high-level API becomes much easier to understand.

---

# 44. SFT With a Custom Dataset

A realistic pipeline might look like:

```text
custom_dataset.json
        ↓
load_dataset()
        ↓
clean
        ↓
format
        ↓
chat template
        ↓
tokenizer
        ↓
train/validation split
        ↓
SFTTrainer
        ↓
LoRA
        ↓
trained adapter
```

---

# 45. What Does the Final Model Learn?

Suppose your dataset contains:

```text
Instruction:
Explain Python decorators.

Response:
A decorator is a function that modifies the behavior
of another function...
```

After training, the model's parameters are adjusted so that instructions similar to:

```text
Explain Python decorators.
What are decorators in Python?
How do Python decorators work?
```

are more likely to produce useful responses.

This is **generalization**.

---

# 46. SFT Does Not Simply Memorize

Ideally:

```text
Training examples
      ↓
Learn patterns
      ↓
Generalize
      ↓
New instructions
```

However, excessive training or poor data can lead to memorization or overfitting.

This is why validation and evaluation are important.

---

# 47. Overfitting During SFT

Suppose the dataset is very small:

```text
100 examples
```

and you train for too many epochs.

The model may become overly specialized to those examples.

Conceptually:

```text
Training performance
       ↑
       │
       │       ______
       │      /
       │     /
       │____/
       │
       └────────────────→

Validation performance
       ↑
       │      /\
       │     /  \
       │____/    \____
       │
       └────────────────→
             Training
```

Training loss may continue decreasing while validation performance deteriorates.

---

# 48. Epoch

An **epoch** means one complete pass through the training dataset.

If:

```text
Dataset = 10,000 examples
```

then:

```text
1 epoch
=
model sees all 10,000 examples once
```

If:

```text
3 epochs
```

the model sees the training examples approximately three times.

---

# 49. Batch Size

Suppose:

```text
batch_size = 8
```

The model processes eight examples in one batch before the optimizer update, depending on the training setup.

Conceptually:

```text
Dataset
   ↓
Batch 1 → 8 examples
   ↓
Loss
   ↓
Gradient
   ↓
Update

Batch 2 → 8 examples
   ↓
...
```

---

# 50. Gradient Accumulation

If GPU memory cannot fit a large batch, gradient accumulation can simulate a larger effective batch.

For example:

```text
micro_batch_size = 2
gradient_accumulation_steps = 4
```

Approximate effective batch:

\[
2\times4=8
\]

Conceptually:

```text
Batch 1
 ↓
Gradient

Batch 2
 ↓
Accumulate

Batch 3
 ↓
Accumulate

Batch 4
 ↓
Accumulate

Optimizer Step
```

This becomes important for large LLM fine-tuning.

---

# 51. SFT Hyperparameters

Common parameters include:

```text
learning_rate
batch_size
gradient_accumulation_steps
num_train_epochs
max_seq_length
warmup_steps / warmup_ratio
weight_decay
```

With LoRA, also:

```text
lora_rank
lora_alpha
lora_dropout
target_modules
```

These determine how the training process behaves.

---

# 52. Practical SFT Configuration

Conceptually:

```python
training_args = {
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 512
}
```

These are example values for understanding the configuration, not universal recommendations.

---

# 53. SFT + LoRA Configuration

Conceptually:

```python
lora_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05
}
```

Recall:

\[
\Delta W=
\frac{\alpha}{r}BA
\]

So if:

\[
r=16
\]

and:

\[
\alpha=32
\]

then:

\[
\frac{\alpha}{r}=2
\]

and:

\[
\Delta W=2BA
\]

---

# 54. Hyperparameters Are Not Universal

Do not memorize:

```text
r=16
alpha=32
learning_rate=2e-5
```

as the "correct" values.

They depend on:

- Model
- Dataset
- Task
- Hardware
- Batch size
- Training duration
- Target modules

The important thing is understanding what each parameter controls.

---

# 55. Questions & Answers

## Q1. What is SFT?

Supervised Fine-Tuning trains a pretrained model using supervised examples consisting of inputs and desired outputs.

---

## Q2. What does "supervised" mean?

The training data provides a target response that acts as the supervision signal.

---

## Q3. Is SFT still next-token prediction?

Yes.

The model still predicts the next token, but the training examples are curated supervised examples.

---

## Q4. What is the SFT loss?

A common objective is cross-entropy / negative log-likelihood over the target tokens:

\[
\boxed{
\mathcal{L}
=
-\sum_t\log P(y_t|x,y_{<t})
}
\]

---

## Q5. What are labels?

Labels identify the target token that the model should predict at each position.

---

## Q6. Why use `-100` in labels?

In common PyTorch/Hugging Face cross-entropy implementations, `-100` is used as an ignore index so those positions do not contribute to the loss.

---

## Q7. What is loss masking?

Loss masking determines which tokens contribute to the training loss.

---

## Q8. Is loss masking the same as attention masking?

No.

Attention masking controls which positions can participate in attention.

Loss masking controls which positions contribute to the loss.

---

## Q9. What does LoRA change in SFT?

LoRA changes which parameters are trained.

The base model weights are frozen and the LoRA parameters are trained.

---

## Q10. Does LoRA change the SFT objective?

No.

The supervised next-token prediction objective remains the same.

---

## Q11. What is the purpose of a chat template?

It converts structured messages into the model-specific token format expected by a conversational model.

---

## Q12. What is an epoch?

One complete pass through the training dataset.

---

## Q13. What is gradient accumulation?

It accumulates gradients over multiple smaller batches before performing an optimizer update, allowing a larger effective batch size.

---

## Q14. What are logits?

The raw scores produced by the model for possible next tokens before converting them into probabilities.

---

## Q15. Why do we need a validation set?

To monitor generalization and help detect problems such as overfitting.

---

# 56. Self-Test

Try answering these without looking back:

1. What does SFT stand for?
2. Why is SFT called supervised?
3. Is SFT still based on next-token prediction?
4. What is the mathematical SFT objective?
5. What are logits?
6. What does cross-entropy measure?
7. What are labels?
8. Why is label shifting necessary?
9. What is loss masking?
10. Why is `-100` commonly used?
11. What is the difference between loss masking and attention masking?
12. What is a chat template?
13. What happens during `loss.backward()`?
14. What does the optimizer do?
15. What changes when SFT uses LoRA?
16. What remains the same when SFT uses LoRA?
17. What is an epoch?
18. What is batch size?
19. What is gradient accumulation?
20. Why can excessive SFT cause overfitting?

---

# 57. Final Mental Model

The complete SFT pipeline is:

```text
                    CUSTOM DATASET
                          ↓
                 Instruction / Response
                          ↓
                    Chat Template
                          ↓
                      Tokenizer
                          ↓
                     input_ids
                          ↓
                 Padding / Truncation
                          ↓
                  attention_mask
                          ↓
                      Labels
                          ↓
                   Loss Masking
                          ↓
                    Forward Pass
                          ↓
                       Logits
                          ↓
                 Shifted Prediction
                          ↓
                  Cross-Entropy Loss
                          ↓
                    loss.backward()
                          ↓
                       Gradients
                          ↓
                     Optimizer
                          ↓
                 Parameter Updates
                          ↓
                  Fine-Tuned Model
```

With LoRA:

```text
Base Model
    ↓
Frozen

LoRA A
    ↓
Trainable

LoRA B
    ↓
Trainable
```

---

# 58. Four Concepts You Must Keep Separate

## Dataset

\[
\boxed{
\text{What examples do we train on?}
}
\]

## SFT / IFT

\[
\boxed{
\text{What supervised behavior do we train?}
}
\]

## Chat Template

\[
\boxed{
\text{How are conversations represented to the model?}
}
\]

## LoRA / QLoRA

\[
\boxed{
\text{How do we efficiently update the model?}
}
\]

Together:

```text
Dataset
   ↓
Chat Template
   ↓
Tokenization
   ↓
SFT
   ↓
LoRA / QLoRA
   ↓
Fine-Tuned LLM
```

---

# 59. Lesson 16 Summary

You learned:

- What SFT means
- Why SFT is supervised
- SFT vs pretraining
- SFT as next-token prediction
- The SFT mathematical objective
- Logits
- Softmax
- Cross-entropy
- Labels
- Label shifting
- Loss masking
- `-100` ignore index
- Chat templates
- Attention masks
- Causal masks
- Difference between attention masking and loss masking
- SFT with full fine-tuning
- SFT with LoRA
- SFT with QLoRA
- Autograd in SFT
- Optimizer updates
- Data collators
- Epochs
- Batch size
- Gradient accumulation
- SFT hyperparameters
- Overfitting
- SFT with custom datasets

The central idea is:

\[
\boxed{
\text{SFT}=
\text{supervised next-token training on desired responses}
}
\]

while:

\[
\boxed{
\text{LoRA}=
\text{efficiently choosing which parameters to train}
}
\]

---

# 60. Next Lesson

## Lesson 17 — Chat Templates + Loss Masking in Practice

Next we will make the previous concepts practical.

We will build the complete transformation:

```text
messages
   ↓
chat_template
   ↓
formatted conversation
   ↓
input_ids
   ↓
labels
   ↓
assistant-only loss mask
   ↓
model
   ↓
cross-entropy loss
```

Then we will inspect the actual tensors:

```text
input_ids
attention_mask
labels
```

and understand **exactly which tokens are being trained and which tokens are ignored**.

After that, we can move into **TRL**, `SFTTrainer`, and practical LLM fine-tuning.
