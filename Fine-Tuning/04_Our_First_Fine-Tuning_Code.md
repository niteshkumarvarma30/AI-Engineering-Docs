# Fine-Tuning — Lesson 4: Our First Fine-Tuning Code

> **Goal:** Move from fine-tuning theory to an actual implementation using PyTorch and Hugging Face Transformers.

---

# 1. What Are We Going to Build?

We will create a simple:

> **Movie Review Sentiment Classifier**

Example:

```text
"I absolutely loved this movie!"
        ↓
Positive
```

and:

```text
"The movie was terrible."
        ↓
Negative
```

The architecture is:

```text
                Text
                 ↓
              Tokenizer
                 ↓
        input_ids + attention_mask
                 ↓
        Pretrained DistilBERT
                 ↓
       Classification Head
                 ↓
               Logits
                 ↓
                Loss
                 ↓
          Backpropagation
                 ↓
            Optimizer
                 ↓
          Updated Parameters
```

The main objective is **not to memorize the code**.

The objective is to understand what every important line is doing.

---

# 2. Required Libraries

We will use:

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets

Basic imports:

```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
```

### PyTorch

```python
import torch
```

PyTorch provides:

- Tensors
- Autograd
- Neural-network operations
- Optimizers
- GPU support
- Training utilities

### Hugging Face Transformers

```python
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
```

Transformers provides pretrained Transformer architectures and utilities.

---

# 3. Install the Libraries

Install the required packages:

```bash
pip install torch transformers datasets
```

Check PyTorch:

```python
import torch

print(torch.__version__)
```

Check Transformers:

```python
import transformers

print(transformers.__version__)
```

---

# 4. Select a Pretrained Model

We will use:

```python
model_name = "distilbert-base-uncased"
```

DistilBERT is a pretrained Transformer model.

The important point is:

```text
We are NOT creating a Transformer from random weights.
```

Instead:

```text
Pretrained DistilBERT
        ↓
Fine-tune on our task
```

This connects directly to Lesson 2:

\[
\boxed{
\theta_{\text{pretrained}}
\rightarrow
\theta_{\text{fine-tuned}}
}
\]

---

# 5. Load the Tokenizer

```python
from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
```

The tokenizer knows:

- Vocabulary
- Tokenization rules
- Special tokens
- Token IDs
- Padding
- Truncation

Conceptually:

```text
Raw Text
   ↓
Tokenizer
   ↓
Tokens
   ↓
Token IDs
```

---

# 6. Test the Tokenizer

Let's give it a sentence:

```python
text = "I loved this movie!"

tokens = tokenizer.tokenize(text)

print(tokens)
```

You may get something similar to:

```text
['i', 'loved', 'this', 'movie', '!']
```

The exact result depends on the tokenizer.

---

# 7. Convert Tokens Into IDs

```python
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(token_ids)
```

Conceptually:

```text
Tokens
   ↓
['i', 'loved', 'this', 'movie', '!']
   ↓
Token IDs
   ↓
[1045, 3866, 2023, 3185, 999]
```

The exact IDs depend on the tokenizer.

---

# 8. Use the Tokenizer Directly

Normally, we don't need to manually tokenize and convert IDs.

We can simply write:

```python
encoded = tokenizer(text)

print(encoded)
```

This can return fields such as:

```python
{
    "input_ids": [...],
    "attention_mask": [...]
}
```

---

# 9. Understanding `input_ids`

```python
print(encoded["input_ids"])
```

`input_ids` are the numerical IDs representing the tokens.

Conceptually:

```text
Text
 ↓
Tokenizer
 ↓
input_ids
```

These are integers, not embeddings.

The model later converts them into dense representations through its embedding layer.

---

# 10. Understanding `attention_mask`

```python
print(encoded["attention_mask"])
```

For a sentence without padding, you might see:

```text
[1, 1, 1, 1, 1, 1, ...]
```

If padding exists:

```text
[1, 1, 1, 1, 0, 0]
```

Typically:

```text
1 → valid token position
0 → padding position
```

---

# 11. Padding and Truncation

When working with batches, we usually specify:

```python
encoded = tokenizer(
    text,
    padding=True,
    truncation=True
)
```

### `padding=True`

Pads sequences when necessary.

### `truncation=True`

Cuts sequences that exceed the allowed maximum length.

---

# 12. Maximum Sequence Length

We can explicitly specify:

```python
max_length=128
```

For example:

```python
encoded = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=128
)
```

Now the sequence will have:

```text
128 positions
```

If the sentence is shorter:

```text
tokens + PAD + PAD + ...
```

If it is longer:

```text
first 128 tokens
```

---

# 13. Why Do We Need Truncation?

Suppose:

```text
Maximum length = 128
```

but the input contains:

```text
200 tokens
```

Then:

```text
200 tokens
    ↓
Truncation
    ↓
128 tokens
```

This prevents sequences from exceeding the chosen maximum length.

---

# 14. Create a Tiny Dataset

Let's start with a very small dataset so we can understand the mechanics.

```python
texts = [
    "I loved this movie!",
    "This movie was amazing.",
    "I really enjoyed the film.",
    "This was fantastic.",
    "I hated this movie.",
    "This movie was terrible.",
    "The film was boring.",
    "I did not enjoy this movie."
]
```

Labels:

```python
labels = [
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0
]
```

We define:

```text
1 → Positive
0 → Negative
```

Therefore:

```text
"I loved this movie!"      → 1
"This movie was terrible." → 0
```

---

# 15. Tokenize the Dataset

We can tokenize all sentences:

```python
encodings = tokenizer(
    texts,
    padding=True,
    truncation=True
)
```

Now:

```python
encodings["input_ids"]
```

contains token IDs for all examples.

And:

```python
encodings["attention_mask"]
```

contains the corresponding attention masks.

---

# 16. Understanding Tensor Shapes

Suppose we have:

```text
8 sentences
```

and after padding:

```text
64 tokens
```

Then:

```python
input_ids.shape
```

might be:

```text
torch.Size([8, 64])
```

Meaning:

\[
\boxed{
[batch\_size,\ sequence\_length]
}
\]

So:

```text
8  → number of examples
64 → tokens per example
```

---

# 17. Convert to PyTorch Tensors

```python
import torch

input_ids = torch.tensor(encodings["input_ids"])
attention_mask = torch.tensor(encodings["attention_mask"])
labels_tensor = torch.tensor(labels)
```

Now:

```python
print(input_ids.shape)
print(attention_mask.shape)
print(labels_tensor.shape)
```

Conceptually:

```text
input_ids:
[8, 64]

attention_mask:
[8, 64]

labels:
[8]
```

---

# 18. Load the Pretrained Model

Now we load the pretrained model:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
```

Conceptually:

```text
Pretrained DistilBERT
        ↓
Classification Head
        ↓
2 output classes
```

---

# 19. What Does `num_labels=2` Mean?

We have two classes:

```text
0 → Negative
1 → Positive
```

Therefore:

```python
num_labels=2
```

The model produces:

```text
2 logits
```

for every example.

If:

```text
batch_size = 8
```

then:

```text
logits shape = [8, 2]
```

---

# 20. Forward Pass

Now send the tensors through the model:

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask
)
```

The model returns an output object containing information such as logits.

Access them:

```python
logits = outputs.logits

print(logits.shape)
```

Expected shape:

```text
[8, 2]
```

---

# 21. Understanding `[8, 2]`

Suppose:

```text
batch_size = 8
num_labels = 2
```

Then:

\[
\text{logits}\in\mathbb{R}^{8\times2}
\]

Example:

```text
Example 1 → [ 2.4, -0.8 ]
Example 2 → [ 1.7, -0.2 ]
Example 3 → [-0.5,  2.1 ]
...
```

Each row corresponds to one example.

Each column corresponds to one class.

```text
Column 0 → Negative
Column 1 → Positive
```

---

# 22. Before Fine-Tuning

At this point, the model is essentially:

```text
Pretrained Model
      +
Classification Head
```

The classification head has not yet been properly adapted to our dataset.

Therefore, the model may produce poor predictions.

That is expected.

Fine-tuning will adapt the trainable parameters to our task.

---

# 23. Calculate Loss

Now provide the labels:

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels_tensor
)
```

The model can now calculate a classification loss.

Access it:

```python
loss = outputs.loss

print(loss)
```

Conceptually:

```text
Logits
   +
Labels
   ↓
Cross-Entropy Loss
```

---

# 24. Important: We Did Not Apply Softmax

Notice that we did not write:

```python
softmax(logits)
```

before calculating the loss.

That is intentional.

For classification using cross-entropy, we normally provide the raw logits.

The cross-entropy implementation internally performs the appropriate normalization.

So remember:

```text
Logits
   ↓
Cross-Entropy Loss
```

rather than manually doing:

```text
Logits
   ↓
Softmax
   ↓
Cross-Entropy Loss
```

when using the standard PyTorch/Hugging Face cross-entropy implementation.

---

# 25. Backpropagation

Now:

```python
loss.backward()
```

calculates gradients for trainable parameters.

Conceptually:

\[
\frac{\partial L}{\partial\theta}
\]

where:

- \(L\) = loss
- \(\theta\) = trainable parameters

The flow is:

```text
Loss
 ↓
Backpropagation
 ↓
Gradients
 ↓
Model Parameters
```

This connects directly to PyTorch Autograd.

---

# 26. Optimizer

We need an optimizer to update the parameters.

For example:

```python
from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=5e-5
)
```

Here:

```text
AdamW
```

is the optimizer.

And:

```text
5e-5
```

is the learning rate.

---

# 27. Update Parameters

After:

```python
loss.backward()
```

we have gradients.

Now:

```python
optimizer.step()
```

updates the parameters.

Then:

```python
optimizer.zero_grad()
```

clears the old gradients before the next training step.

One training iteration therefore looks like:

```python
optimizer.zero_grad()

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels_tensor
)

loss = outputs.loss

loss.backward()

optimizer.step()
```

---

# 28. The Complete Training Step

The process is:

```text
             Input Data
                 ↓
               Model
                 ↓
               Logits
                 ↓
                Loss
                 ↓
         loss.backward()
                 ↓
              Gradients
                 ↓
          optimizer.step()
                 ↓
        Updated Parameters
```

This is the heart of fine-tuning.

---

# 29. Why Do We Call `zero_grad()`?

This is an important PyTorch concept.

PyTorch gradients accumulate by default.

Suppose:

```python
loss.backward()
```

produces a gradient:

\[
G_1
\]

A later backward pass can accumulate another gradient:

\[
G_1+G_2
\]

If we do not clear the gradients, they accumulate.

Therefore, we normally do:

```python
optimizer.zero_grad()
```

before the next training step.

Conceptually:

```text
Previous gradients
       ↓
Clear
       ↓
New forward pass
       ↓
New loss
       ↓
New gradients
```

---

# 30. One Complete Training Loop

A very basic training loop is:

```python
for epoch in range(3):

    optimizer.zero_grad()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels_tensor
    )

    loss = outputs.loss

    loss.backward()

    optimizer.step()

    print("Loss:", loss.item())
```

This loop is intentionally simple.

Its purpose is to make the mechanics of fine-tuning visible.

Later, Hugging Face `Trainer` will automate much of this.

---

# 31. What Is an Epoch?

An **epoch** means one complete pass through the training dataset.

If the dataset contains:

```text
8 examples
```

then:

```text
1 epoch
=
model sees all 8 examples once
```

If:

```python
for epoch in range(3):
```

the model sees the dataset three times.

---

# 32. What Is a Batch?

Suppose we have:

```text
1,000 examples
```

and:

```text
batch_size = 32
```

Instead of processing all 1,000 simultaneously:

```text
1,000 examples
      ↓
split into batches
      ↓
32 examples
32 examples
32 examples
...
```

Each batch produces a loss and generally leads to an optimizer update.

---

# 33. Epoch vs Batch vs Training Step

These three concepts are important.

### Epoch

One complete pass through the dataset.

### Batch

A group of training examples processed together.

### Training Step

One optimizer update, usually corresponding to one batch.

Example:

```text
Dataset = 1,000 examples
Batch size = 100
```

Approximately:

```text
10 batches
```

Therefore:

```text
1 epoch
≈
10 optimizer steps
```

assuming no special handling such as gradient accumulation or dropped batches.

---

# 34. Why Don't We Train Everything at Once?

Because of memory constraints.

Suppose:

```text
1,000,000 examples
```

Trying to process everything simultaneously would require enormous memory.

Instead:

```text
Dataset
 ↓
Batch 1
 ↓
Forward
 ↓
Loss
 ↓
Backward
 ↓
Update

Batch 2
 ↓
Forward
 ↓
Loss
 ↓
Backward
 ↓
Update

...
```

This makes training computationally manageable.

---

# 35. What Actually Gets Updated?

In ordinary **full fine-tuning**, trainable parameters throughout the model can be updated.

Depending on the model/configuration, this can include:

- Transformer weights
- Attention projection weights
- Feed-forward weights
- Embeddings
- Classification head

The exact set of parameters being updated is determined by which parameters have:

```python
requires_grad=True
```

---

# 36. Check Trainable Parameters

In PyTorch:

```python
for name, param in model.named_parameters():

    if param.requires_grad:
        print(name)
```

The key concept is:

\[
\boxed{
\texttt{requires\_grad=True}
\Rightarrow
\text{parameter can receive gradients}
}
\]

---

# 37. Freezing Parameters

Later, we will learn parameter-efficient fine-tuning.

For example:

```python
for param in model.base_model.parameters():
    param.requires_grad = False
```

Now the base model is frozen.

Only selected parameters remain trainable.

This leads to techniques such as:

- LoRA
- QLoRA
- Adapters

> **Do not use these techniques yet.** First understand ordinary full fine-tuning.

---

# 38. Why Are We Learning the Manual Loop First?

Hugging Face provides:

```python
Trainer
```

which automates much of the training process.

Eventually you may write:

```python
trainer.train()
```

But if you only memorize:

```python
trainer.train()
```

you may not understand what is happening underneath.

Our learning sequence is:

```text
Manual PyTorch Training
        ↓
Understand Forward Pass
        ↓
Understand Loss
        ↓
Understand Backward Pass
        ↓
Understand Optimizer
        ↓
Understand Batching
        ↓
Understand Evaluation
        ↓
Hugging Face Trainer
```

This gives you a stronger understanding of fine-tuning.

---

# 39. The Full Manual Pipeline

Putting everything together:

```text
                RAW TEXT
                   ↓
               Tokenizer
                   ↓
        ┌────────────────────┐
        │ input_ids          │
        │ attention_mask     │
        │ labels             │
        └─────────┬──────────┘
                  ↓
          PRETRAINED MODEL
                  ↓
               Logits
                  ↓
                Loss
                  ↓
          loss.backward()
                  ↓
              Gradients
                  ↓
          optimizer.step()
                  ↓
        UPDATED PARAMETERS
```

---

# 40. The Most Important Code

The most important section to understand is:

```python
optimizer.zero_grad()

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels_tensor
)

loss = outputs.loss

loss.backward()

optimizer.step()
```

Translate it into English:

```text
Clear old gradients
        ↓
Give training data to model
        ↓
Calculate predictions
        ↓
Calculate loss
        ↓
Calculate gradients
        ↓
Update parameters
```

---

# 41. What Is Happening Mathematically?

The model starts with:

\[
\theta_{\text{pretrained}}
\]

Forward pass:

\[
\hat{y}=f_\theta(x)
\]

Loss:

\[
L(y,\hat{y})
\]

Backpropagation:

\[
\nabla_\theta L
\]

Parameter update:

\[
\boxed{
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta\nabla_\theta L
}
\]

Therefore:

\[
\boxed{
\theta_{\text{pretrained}}
\rightarrow
\theta_{\text{fine-tuned}}
}
\]

This is the mathematical core of fine-tuning.

---

# 42. Training vs Inference

This distinction is extremely important.

## Training

During training:

```text
Input
 ↓
Prediction
 ↓
Loss
 ↓
Backward
 ↓
Update parameters
```

Parameters change.

## Inference

During inference:

```text
Input
 ↓
Prediction
 ↓
Output
```

Normally:

```text
No backward
No optimizer step
No parameter update
```

Therefore:

\[
\boxed{
\text{Training} \neq \text{Inference}
}
\]

---

# 43. `model.train()` vs `model.eval()`

PyTorch models have different modes.

### Training mode

```python
model.train()
```

Used during training.

### Evaluation mode

```python
model.eval()
```

Used during evaluation and inference.

For evaluation, we commonly also use:

```python
with torch.no_grad():
    ...
```

This prevents gradient computation during that block.

---

# 44. Example Evaluation

After training:

```python
model.eval()

with torch.no_grad():

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    predictions = torch.argmax(
        outputs.logits,
        dim=-1
    )

print(predictions)
```

`argmax` selects the class with the highest logit.

---

# 45. Training vs Evaluation

```text
TRAINING

model.train()
     ↓
Forward
     ↓
Loss
     ↓
Backward
     ↓
Optimizer
     ↓
Update


EVALUATION

model.eval()
     ↓
Forward
     ↓
Prediction
     ↓
Metrics

No parameter update
```

---

# 46. Loss Is Not Accuracy

Suppose:

```text
Loss = 0.25
```

This does **not** mean:

```text
Accuracy = 75%
```

Loss and accuracy are different metrics.

### Loss

Measures how well predictions align with the training objective.

### Accuracy

Measures the proportion of predictions that are correct.

For example:

```text
100 examples
80 predicted correctly
```

Then:

\[
Accuracy=\frac{80}{100}=0.80
\]

or:

\[
80\%
\]

---

# 47. A Simple Accuracy Calculation

```python
predictions = torch.argmax(
    outputs.logits,
    dim=-1
)

accuracy = (
    predictions == labels_tensor
).float().mean()

print(accuracy.item())
```

Conceptually:

```text
Predictions
     ↓
Compare with labels
     ↓
Correct / Incorrect
     ↓
Mean
     ↓
Accuracy
```

---

# 48. Important Practical Issue: The Tiny Dataset

The tiny dataset in this lesson is only for learning.

It is **not a suitable real-world fine-tuning dataset**.

A real project should generally have:

```text
Training set
Validation set
Test set
```

For example:

```text
Dataset
   ↓
Train       80%
Validation  10%
Test        10%
```

The exact split depends on the problem and dataset.

---

# 49. Why Do We Need Validation Data?

Suppose training loss keeps decreasing:

```text
Epoch 1 → 0.50
Epoch 2 → 0.30
Epoch 3 → 0.15
Epoch 4 → 0.05
```

That looks good.

But validation performance might do:

```text
Epoch 1 → 0.70
Epoch 2 → 0.75
Epoch 3 → 0.73
Epoch 4 → 0.62
```

The model may be **overfitting**.

Therefore, we should not evaluate only on the training data.

---

# 50. Real Fine-Tuning Workflow

A more realistic workflow is:

```text
Raw Dataset
     ↓
Clean Dataset
     ↓
Train / Validation / Test Split
     ↓
Tokenizer
     ↓
DataLoader
     ↓
Pretrained Model
     ↓
Training
     ↓
Validation
     ↓
Hyperparameter Adjustment
     ↓
Final Test Evaluation
```

Later we will implement this properly.

---

# 51. Complete Minimal Example

Here is the complete simplified manual example:

```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from torch.optim import AdamW


# --------------------------------
# 1. Model name
# --------------------------------

model_name = "distilbert-base-uncased"


# --------------------------------
# 2. Tokenizer
# --------------------------------

tokenizer = AutoTokenizer.from_pretrained(model_name)


# --------------------------------
# 3. Tiny dataset
# --------------------------------

texts = [
    "I loved this movie!",
    "This movie was amazing.",
    "I really enjoyed the film.",
    "This was fantastic.",
    "I hated this movie.",
    "This movie was terrible.",
    "The film was boring.",
    "I did not enjoy this movie."
]

labels = [
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0
]


# --------------------------------
# 4. Tokenization
# --------------------------------

encodings = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)


# --------------------------------
# 5. Labels
# --------------------------------

labels_tensor = torch.tensor(labels)


# --------------------------------
# 6. Load pretrained model
# --------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


# --------------------------------
# 7. Optimizer
# --------------------------------

optimizer = AdamW(
    model.parameters(),
    lr=5e-5
)


# --------------------------------
# 8. Training
# --------------------------------

model.train()

for epoch in range(3):

    optimizer.zero_grad()

    outputs = model(
        input_ids=encodings["input_ids"],
        attention_mask=encodings["attention_mask"],
        labels=labels_tensor
    )

    loss = outputs.loss

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch + 1}, "
        f"Loss: {loss.item():.4f}"
    )


# --------------------------------
# 9. Evaluation
# --------------------------------

model.eval()

with torch.no_grad():

    outputs = model(
        input_ids=encodings["input_ids"],
        attention_mask=encodings["attention_mask"]
    )

    predictions = torch.argmax(
        outputs.logits,
        dim=-1
    )

print("Predictions:", predictions)
print("Labels:", labels_tensor)
```

> **Important:** This is intentionally a minimal educational example. It is not a production-quality training pipeline. It uses the same examples for training and evaluation, which is not appropriate for measuring real-world generalization.

---

# 52. Code Flow Explained

The code can be summarized as:

```text
1. Choose pretrained model
        ↓
2. Load tokenizer
        ↓
3. Create dataset
        ↓
4. Tokenize dataset
        ↓
5. Convert labels to tensors
        ↓
6. Load pretrained Transformer
        ↓
7. Create optimizer
        ↓
8. Training loop
        ↓
9. Forward pass
        ↓
10. Calculate loss
        ↓
11. Backpropagation
        ↓
12. Update parameters
        ↓
13. Evaluation
```

---

# 53. Lesson 4 Summary

The key concepts are:

1. We can load a pretrained Transformer using Hugging Face.
2. `AutoTokenizer` converts text into model-compatible inputs.
3. `input_ids` represent token IDs.
4. `attention_mask` identifies valid/padded positions.
5. `AutoModelForSequenceClassification` configures a classification head.
6. The model produces logits.
7. Labels are used to calculate loss.
8. `loss.backward()` calculates gradients.
9. `optimizer.step()` updates trainable parameters.
10. `optimizer.zero_grad()` clears accumulated gradients.
11. An epoch is one complete pass through the dataset.
12. A batch is a group of examples processed together.
13. A training step generally corresponds to an optimizer update.
14. `model.train()` puts the model into training mode.
15. `model.eval()` puts the model into evaluation mode.
16. `torch.no_grad()` prevents unnecessary gradient tracking during evaluation.
17. Training changes parameters; inference normally does not.
18. Accuracy and loss are different metrics.
19. A real fine-tuning workflow should include validation and test evaluation.
20. The manual training loop helps us understand what higher-level APIs such as `Trainer` automate.

---

# 54. Questions & Answers

## Q1. What is the purpose of `AutoTokenizer`?

### Answer

It converts raw text into the numerical representation expected by the pretrained Transformer.

It handles things such as:

- Tokenization
- Vocabulary lookup
- Token IDs
- Special tokens
- Padding
- Truncation

Conceptually:

```text
Text
 ↓
AutoTokenizer
 ↓
input_ids + attention_mask
```

---

## Q2. What does `AutoModelForSequenceClassification` do?

### Answer

It loads a pretrained Transformer configured for sequence classification.

Conceptually:

```text
Pretrained Transformer
        +
Classification Head
        ↓
Class Logits
```

For example:

```python
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
```

means the model is configured to output two class scores.

---

## Q3. What happens during the forward pass?

### Answer

The input is passed through the model:

```text
Input IDs
    ↓
Embeddings
    ↓
Transformer layers
    ↓
Contextual representations
    ↓
Classification head
    ↓
Logits
```

Mathematically:

\[
\hat{y}=f_\theta(x)
\]

---

## Q4. What does `loss.backward()` do?

### Answer

It performs backpropagation and calculates gradients of the loss with respect to trainable parameters.

Conceptually:

\[
\boxed{
loss.backward()
\rightarrow
\nabla_\theta L
}
\]

These gradients tell the optimizer how the parameters should be adjusted.

---

## Q5. What does `optimizer.step()` do?

### Answer

It updates trainable model parameters using the calculated gradients.

Conceptually:

\[
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta\nabla_\theta L
\]

---

## Q6. Why do we call `optimizer.zero_grad()`?

### Answer

PyTorch gradients accumulate by default.

Therefore, before calculating gradients for a new training step, we normally clear the previous gradients:

```python
optimizer.zero_grad()
```

The basic sequence is:

```text
Clear gradients
      ↓
Forward
      ↓
Loss
      ↓
Backward
      ↓
Update
```

---

## Q7. What is an epoch?

### Answer

An epoch is one complete pass through the training dataset.

If there are 1,000 training examples:

```text
1 epoch
=
model processes all 1,000 examples once
```

---

## Q8. What is a batch?

### Answer

A batch is a group of training examples processed together.

For example:

```text
Dataset = 1,000 examples
Batch size = 32
```

The model processes approximately:

```text
32
32
32
...
```

rather than all 1,000 simultaneously.

---

## Q9. What is the difference between training and inference?

### Answer

### Training

```text
Input
 ↓
Prediction
 ↓
Loss
 ↓
Backward
 ↓
Optimizer
 ↓
Parameter update
```

### Inference

```text
Input
 ↓
Prediction
 ↓
Output
```

During normal inference, model parameters are not updated.

---

## Q10. Why do we use `model.eval()` during evaluation?

### Answer

`model.eval()` puts the model into evaluation mode.

This matters because some neural-network layers behave differently during training and evaluation.

For example, dropout is active during training but disabled during evaluation.

---

## Q11. Why do we use `torch.no_grad()` during evaluation?

### Answer

We normally don't need gradients when evaluating a model.

Therefore:

```python
with torch.no_grad():
    ...
```

prevents gradient tracking in that block.

This reduces unnecessary memory usage and computation.

---

## Q12. Why isn't a tiny dataset suitable for a real fine-tuning project?

### Answer

The tiny dataset in this lesson is only for understanding the mechanics.

A real fine-tuning project needs enough representative data to generalize to unseen examples.

Usually we also separate data into:

```text
Training
Validation
Test
```

This allows us to monitor overfitting and evaluate generalization.

---

# 55. Self-Test

Before moving forward, try answering these without looking back:

1. What does `AutoTokenizer.from_pretrained()` do?
2. What does `AutoModelForSequenceClassification.from_pretrained()` do?
3. What is a forward pass?
4. What are logits?
5. What does `labels=labels_tensor` allow the model to calculate?
6. What does `loss.backward()` do?
7. What does `optimizer.step()` do?
8. Why is `optimizer.zero_grad()` necessary?
9. What is an epoch?
10. What is a batch?
11. What is the difference between `model.train()` and `model.eval()`?
12. Why do we use `torch.no_grad()` during evaluation?
13. Why should we have a validation set?
14. Explain the entire training loop in your own words.

---

# 56. Next Lesson

## Lesson 5 — Hugging Face `Dataset`, `DataLoader`, and `Trainer`

Now that we understand the **manual training loop**, we can make the implementation cleaner and more realistic.

We will learn:

```text
Hugging Face Dataset
        ↓
Train / Validation Split
        ↓
Tokenizer
        ↓
Data Collator
        ↓
DataLoader
        ↓
Pretrained Model
        ↓
Trainer
        ↓
Training
        ↓
Evaluation
```

We will also understand what:

```python
trainer.train()
```

is actually doing internally instead of treating it as a black box.

---

# 57. Final Mental Model

You should now be able to look at the following code:

```python
optimizer.zero_grad()

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)

loss = outputs.loss

loss.backward()

optimizer.step()
```

and mentally translate it into:

```text
Clear previous gradients
        ↓
Run the pretrained model
        ↓
Produce logits
        ↓
Compare logits with correct labels
        ↓
Calculate loss
        ↓
Backpropagate the loss
        ↓
Calculate gradients
        ↓
Update model parameters
```

That is the core mechanism of **full fine-tuning**.
