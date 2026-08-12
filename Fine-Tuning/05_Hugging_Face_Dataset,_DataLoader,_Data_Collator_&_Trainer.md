# Fine-Tuning — Lesson 5: Hugging Face Dataset, DataLoader, Data Collator & Trainer

> **Goal:** Move from the manual fine-tuning loop learned in Lesson 4 to a more realistic and scalable Hugging Face training pipeline.

---

# 1. Why Do We Need Lesson 5?

In Lesson 4, we manually created:

```python
input_ids
attention_mask
labels
```

and then manually wrote:

```python
for epoch in range(3):

    optimizer.zero_grad()

    outputs = model(...)

    loss = outputs.loss

    loss.backward()

    optimizer.step()
```

This was excellent for understanding the fundamentals.

But real datasets may contain:

```text
10,000 examples
100,000 examples
1,000,000 examples
```

We don't want to manually manage every tensor.

Therefore, we need a better data pipeline.

---

# 2. The New Pipeline

The more realistic pipeline is:

```text
Raw Dataset
     ↓
Hugging Face Dataset
     ↓
Train / Validation Split
     ↓
Tokenizer
     ↓
Tokenized Dataset
     ↓
Data Collator
     ↓
DataLoader
     ↓
Pretrained Transformer
     ↓
Loss
     ↓
Backpropagation
     ↓
Optimizer
     ↓
Updated Parameters
```

And eventually:

```text
Dataset
   ↓
Tokenizer
   ↓
Data Collator
   ↓
Trainer
   ↓
Fine-Tuned Model
```

---

# 3. What Is a Dataset?

A dataset is simply a collection of examples.

For sentiment classification:

```text
Text                         Label

"I loved this movie."          1
"This was terrible."           0
"This film was amazing."       1
"The movie was boring."        0
```

Each row contains:

```text
Input + Target
```

---

# 4. Hugging Face `Dataset`

Hugging Face provides a `Dataset` abstraction through the `datasets` library.

Install it:

```bash
pip install datasets
```

Import:

```python
from datasets import Dataset
```

We can create a dataset from Python data:

```python
data = {
    "text": [
        "I loved this movie!",
        "This movie was amazing.",
        "I hated this movie.",
        "This movie was terrible."
    ],
    "label": [
        1,
        1,
        0,
        0
    ]
}

dataset = Dataset.from_dict(data)
```

Now:

```python
print(dataset)
```

will show information about the dataset.

---

# 5. Dataset Structure

Conceptually:

```text
┌───────────────────────────────┐
│ text                    label │
├───────────────────────────────┤
│ I loved this movie!       1   │
│ This movie was amazing.   1   │
│ I hated this movie.       0   │
│ This movie was terrible.  0   │
└───────────────────────────────┘
```

This is much easier to manage than manually maintaining separate Python lists.

---

# 6. Accessing Examples

We can access an example:

```python
print(dataset[0])
```

Conceptually:

```python
{
    "text": "I loved this movie!",
    "label": 1
}
```

We can access a column:

```python
print(dataset["text"])
```

---

# 7. Train / Validation Split

We shouldn't train and evaluate on exactly the same examples.

Instead:

```text
Dataset
   ↓
Train
Validation
```

We can use:

```python
split_dataset = dataset.train_test_split(
    test_size=0.2
)
```

Now:

```python
train_dataset = split_dataset["train"]
validation_dataset = split_dataset["test"]
```

Conceptually:

```text
Original Dataset
      ↓
   ┌───────┐
   ↓       ↓
 Train   Validation
  80%       20%
```

---

# 8. Why Is Validation Important?

Suppose:

```text
Training accuracy = 99%
Validation accuracy = 70%
```

This suggests the model may be memorizing the training data.

This is called:

\[
\boxed{\text{Overfitting}}
\]

Validation data helps us estimate how well the model generalizes to unseen examples.

---

# 9. Reproducible Splitting

We can provide a seed:

```python
split_dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42
)
```

The seed helps make the split reproducible.

---

# 10. Load the Tokenizer

We already learned this in Lesson 4:

```python
from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
```

---

# 11. Tokenization Function

Now we'll create a function:

```python
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True
    )
```

This function receives one dataset example.

For example:

```python
{
    "text": "I loved this movie!",
    "label": 1
}
```

The tokenizer processes:

```text
"I loved this movie!"
```

and produces:

```text
input_ids
attention_mask
```

---

# 12. Apply Tokenization to the Dataset

Hugging Face Dataset provides `.map()`.

```python
tokenized_dataset = dataset.map(
    tokenize_function
)
```

This applies the tokenizer to the examples.

Conceptually:

```text
Dataset
   ↓
map(tokenize_function)
   ↓
Tokenized Dataset
```

---

# 13. Why Use `.map()`?

Instead of manually doing:

```python
for text in texts:
    tokenizer(text)
```

we can let the Dataset system apply the function to every example.

This is particularly useful for large datasets.

---

# 14. Tokenized Dataset

Before tokenization:

```text
text
label
```

After tokenization:

```text
text
label
input_ids
attention_mask
```

Conceptually:

```text
┌────────────────────────────────────────────┐
│ text       label   input_ids   mask        │
├────────────────────────────────────────────┤
│ I love...    1     [101,...]   [1,...]    │
│ This...      0     [101,...]   [1,...]    │
└────────────────────────────────────────────┘
```

---

# 15. Important: Dynamic Padding

In Lesson 4, we used:

```python
padding="max_length"
```

This means every example gets padded to the same maximum length.

For example:

```text
max_length = 128
```

Even if a sentence has only 10 tokens:

```text
10 real tokens + 118 PAD tokens
```

This can waste computation.

Instead, we can use **dynamic padding**.

---

# 16. What Is Dynamic Padding?

Suppose a batch contains:

```text
Sentence A → 10 tokens
Sentence B → 20 tokens
Sentence C → 15 tokens
```

Instead of padding all of them to 128:

```text
128
128
128
```

we can pad them to the longest sequence in that batch:

```text
20
20
20
```

Conceptually:

```text
Before:

A → 10
B → 20
C → 15

Dynamic padding:

A → 20
B → 20
C → 20
```

This reduces unnecessary computation.

---

# 17. Data Collator

Hugging Face provides a **data collator** for this purpose.

For sequence classification:

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)
```

Its job is to prepare individual examples into a batch.

Conceptually:

```text
Individual examples
       ↓
Data Collator
       ↓
Padding
       ↓
Batch
```

---

# 18. Why Is the Data Collator Useful?

Suppose:

```text
Example 1 → 8 tokens
Example 2 → 14 tokens
Example 3 → 11 tokens
```

The collator can dynamically create:

```text
Example 1 → 14 tokens
Example 2 → 14 tokens
Example 3 → 14 tokens
```

using padding.

This is more efficient than always padding to a large fixed maximum.

---

# 19. What Is a DataLoader?

Now we need to understand PyTorch's `DataLoader`.

A `DataLoader` takes a dataset and produces batches.

Conceptually:

```text
Dataset
   ↓
DataLoader
   ↓
Batch 1
Batch 2
Batch 3
...
```

Import:

```python
from torch.utils.data import DataLoader
```

---

# 20. Basic DataLoader Example

```python
loader = DataLoader(
    dataset,
    batch_size=2
)
```

Suppose we have:

```text
8 examples
```

and:

```text
batch_size = 2
```

Then approximately:

```text
Batch 1 → 2 examples
Batch 2 → 2 examples
Batch 3 → 2 examples
Batch 4 → 2 examples
```

---

# 21. Why Do We Need DataLoader?

Without a DataLoader:

```text
Dataset
   ↓
Manually create batches
```

With DataLoader:

```text
Dataset
   ↓
DataLoader
   ↓
Automatic batching
```

It can also handle things such as:

- Shuffling
- Batch creation
- Parallel data loading
- Iteration

---

# 22. DataLoader + Data Collator

For Transformers, the typical idea is:

```text
Dataset
   ↓
DataLoader
   ↓
Data Collator
   ↓
Padded Batch
```

For example:

```python
loader = DataLoader(
    tokenized_dataset,
    batch_size=8,
    collate_fn=data_collator
)
```

The `collate_fn` tells the DataLoader how to combine individual examples into a batch.

---

# 23. Why Can't We Just Stack Variable-Length Tensors?

Suppose:

```text
Example 1:
[101, 20, 30]

Example 2:
[101, 20, 30, 40, 50]
```

These tensors have different lengths.

A normal tensor batch needs a rectangular shape.

Therefore:

```text
[101, 20, 30]
[101, 20, 30, 40, 50]
```

cannot simply be stacked as-is.

We need padding:

```text
[101, 20, 30, PAD, PAD]
[101, 20, 30, 40, 50]
```

The data collator handles this.

---

# 24. The Data Pipeline So Far

We now have:

```text
Raw Text
   ↓
Hugging Face Dataset
   ↓
Train / Validation Split
   ↓
Tokenizer
   ↓
.map()
   ↓
Tokenized Dataset
   ↓
Data Collator
   ↓
DataLoader
   ↓
Batches
```

---

# 25. Load the Model

We already know:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
```

So now we have:

```text
Tokenized Dataset
        +
Pretrained Model
```

---

# 26. Manual DataLoader Training

Before using `Trainer`, let's connect everything manually.

```python
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

train_loader = DataLoader(
    tokenized_train,
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator
)
```

Now:

```python
for batch in train_loader:
    print(batch)
```

The DataLoader produces batches such as:

```text
input_ids
attention_mask
labels
```

---

# 27. Why `shuffle=True`?

During training, we generally shuffle the training examples.

Why?

Suppose the dataset is ordered:

```text
Positive
Positive
Positive
Positive
Negative
Negative
Negative
Negative
```

The model might receive batches with highly correlated labels.

Shuffling produces a more mixed ordering:

```text
Positive
Negative
Positive
Negative
...
```

This generally gives better stochastic training behavior.

---

# 28. Should Validation Data Be Shuffled?

Usually:

```python
train_loader = DataLoader(
    train_dataset,
    shuffle=True
)
```

but validation generally:

```python
validation_loader = DataLoader(
    validation_dataset,
    shuffle=False
)
```

because the ordering doesn't need to be randomized for evaluation.

---

# 29. Manual Training With DataLoader

Now our previous training loop becomes:

```python
for epoch in range(3):

    model.train()

    for batch in train_loader:

        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()
```

This is much closer to a real training pipeline.

---

# 30. Notice What Changed

Previously:

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels_tensor
)
```

Now:

```python
outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"]
)
```

The DataLoader gives us the batch.

---

# 31. Where Does the Batch Come From?

The complete flow is:

```text
Dataset
   ↓
DataLoader
   ↓
Data Collator
   ↓
Batch
   ↓
Model
```

For example:

```text
Batch:

input_ids
[8, 20]

attention_mask
[8, 20]

labels
[8]
```

Meaning:

```text
8 examples
20 tokens each after dynamic padding
8 labels
```

---

# 32. Device: CPU vs GPU

Now we introduce another important concept.

Models can run on:

```text
CPU
```

or:

```text
GPU
```

We can select the device:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Then:

```python
model.to(device)
```

moves the model to the selected device.

---

# 33. Move the Batch to the Device

If the model is on GPU, the tensors must also be on GPU.

We can do:

```python
batch = {
    key: value.to(device)
    for key, value in batch.items()
}
```

Then:

```python
outputs = model(**batch)
```

Now model and tensors are on the same device.

---

# 34. What Does `**batch` Mean?

Suppose:

```python
batch = {
    "input_ids": ...,
    "attention_mask": ...,
    "labels": ...
}
```

Then:

```python
model(**batch)
```

is equivalent to:

```python
model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"]
)
```

This is a very common Python pattern in Hugging Face code.

---

# 35. Complete Manual DataLoader Training

A simplified version is:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

optimizer = AdamW(
    model.parameters(),
    lr=5e-5
)

for epoch in range(3):

    model.train()

    for batch in train_loader:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(**batch)

        loss = outputs.loss

        loss.backward()

        optimizer.step()

    print(
        f"Epoch {epoch + 1}, "
        f"Loss: {loss.item():.4f}"
    )
```

This is already a proper conceptual training pipeline.

---

# 36. But We Still Have to Write a Lot of Code

We currently need to manually manage:

- DataLoader
- Device
- Optimizer
- Training loop
- Evaluation
- Logging
- Number of epochs
- Saving
- Evaluation metrics

Hugging Face provides a higher-level API:

\[
\boxed{\texttt{Trainer}}
\]

---

# 37. What Is Hugging Face `Trainer`?

`Trainer` is a high-level training abstraction from Hugging Face Transformers.

Instead of manually writing:

```python
for epoch in range(...):

    for batch in train_loader:

        optimizer.zero_grad()

        outputs = model(...)

        loss = outputs.loss

        loss.backward()

        optimizer.step()
```

we can configure a trainer and call:

```python
trainer.train()
```

---

# 38. Important Mental Model

Do **not** think:

```text
trainer.train()
```

magically trains the model.

Think:

```text
Trainer
   ↓
Creates/manages training loop
   ↓
Gets batches
   ↓
Calls model
   ↓
Calculates loss
   ↓
Backpropagation
   ↓
Optimizer update
   ↓
Repeats
```

So `Trainer` is essentially an abstraction over much of the training infrastructure.

---

# 39. TrainingArguments

First we configure the training process:

```python
from transformers import TrainingArguments
```

Example:

```python
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5
)
```

Let's understand each parameter.

---

# 40. `output_dir`

```python
output_dir="./results"
```

Specifies where training outputs/checkpoints can be stored.

---

# 41. `num_train_epochs`

```python
num_train_epochs=3
```

Means:

```text
Train for 3 epochs.
```

---

# 42. `per_device_train_batch_size`

```python
per_device_train_batch_size=8
```

Means:

```text
8 examples per device per training batch
```

If using one GPU:

```text
≈ 8 examples/batch
```

With multiple devices, the effective total batch size can be larger.

---

# 43. `per_device_eval_batch_size`

```python
per_device_eval_batch_size=8
```

Controls the evaluation batch size per device.

---

# 44. `learning_rate`

```python
learning_rate=5e-5
```

Controls the size of parameter updates.

Conceptually:

\[
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta\nabla_\theta L
\]

where:

\[
\eta
\]

is the learning rate.

---

# 45. Evaluation Strategy

Depending on the Transformers version, evaluation configuration can be specified through arguments such as:

```python
eval_strategy="epoch"
```

This means evaluation is performed after each epoch.

For example:

```text
Epoch 1
 ↓
Training
 ↓
Evaluation

Epoch 2
 ↓
Training
 ↓
Evaluation

Epoch 3
 ↓
Training
 ↓
Evaluation
```

> Hugging Face has changed some `TrainingArguments` parameter names across library versions. Check the installed version's documentation if an argument is rejected.

---

# 46. Save Strategy

We can configure checkpoint saving, for example:

```python
save_strategy="epoch"
```

This can save a checkpoint after each epoch.

Conceptually:

```text
Epoch 1 → Save checkpoint
Epoch 2 → Save checkpoint
Epoch 3 → Save checkpoint
```

---

# 47. Create the Trainer

We can now create:

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)
```

Depending on your installed Transformers version, the tokenizer argument may be represented differently. The exact API can vary by version.

---

# 48. Start Training

Now:

```python
trainer.train()
```

That's it.

But internally, Trainer is performing a process conceptually similar to:

```text
for epoch in range(...):

    for batch in train_dataloader:

        optimizer.zero_grad()

        outputs = model(batch)

        loss = outputs.loss

        loss.backward()

        optimizer.step()
```

plus additional functionality.

---

# 49. What Does Trainer Handle?

Depending on configuration and version, `Trainer` can manage much of:

- Batching
- Training loop
- Evaluation
- Optimizer
- Learning-rate scheduling
- Checkpointing
- Logging
- Device placement
- Gradient accumulation
- Mixed precision
- Distributed training

This is why `Trainer` is useful.

---

# 50. Trainer Does Not Remove the Fundamentals

This is very important.

When you use:

```python
trainer.train()
```

the underlying concepts remain:

```text
Forward pass
Loss
Backward pass
Gradients
Optimizer
Parameter updates
```

`Trainer` simply automates much of the surrounding engineering.

---

# 51. Evaluate the Model

After training:

```python
metrics = trainer.evaluate()

print(metrics)
```

You may get values such as:

```text
eval_loss
```

and other configured metrics.

---

# 52. Prediction

We can also use:

```python
predictions = trainer.predict(
    validation_dataset
)
```

This gives prediction outputs and evaluation information.

---

# 53. Saving the Model

After fine-tuning:

```python
trainer.save_model(
    "./fine_tuned_model"
)
```

We can also save the tokenizer:

```python
tokenizer.save_pretrained(
    "./fine_tuned_model"
)
```

Now we have:

```text
fine_tuned_model/
    ├── model files
    ├── configuration
    └── tokenizer files
```

---

# 54. Loading the Fine-Tuned Model Later

We can load it again:

```python
tokenizer = AutoTokenizer.from_pretrained(
    "./fine_tuned_model"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "./fine_tuned_model"
)
```

So:

```text
Fine-Tuned Model
       ↓
Save
       ↓
Disk
       ↓
Load later
```

---

# 55. Complete Conceptual Pipeline

Now our entire workflow is:

```text
                    RAW DATA
                       ↓
              Hugging Face Dataset
                       ↓
              Train / Validation Split
                       ↓
                    Tokenizer
                       ↓
                    .map()
                       ↓
               Tokenized Dataset
                       ↓
                 Data Collator
                       ↓
                    Batches
                       ↓
              Pretrained Transformer
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
                       ↓
                Validation / Test
                       ↓
                Fine-Tuned Model
```

With `Trainer`:

```text
Dataset
   ↓
Tokenizer
   ↓
Data Collator
   ↓
Trainer
   ↓
Training Loop
   ↓
Evaluation
   ↓
Saved Model
```

---

# 56. Complete Example

A simplified realistic example:

```python
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)


# -----------------------------------------
# 1. Model
# -----------------------------------------

model_name = "distilbert-base-uncased"


# -----------------------------------------
# 2. Dataset
# -----------------------------------------

data = {
    "text": [
        "I loved this movie!",
        "This movie was amazing.",
        "I really enjoyed the film.",
        "This was fantastic.",
        "I hated this movie.",
        "This movie was terrible.",
        "The film was boring.",
        "I did not enjoy this movie."
    ],

    "label": [
        1, 1, 1, 1,
        0, 0, 0, 0
    ]
}

dataset = Dataset.from_dict(data)


# -----------------------------------------
# 3. Train / Validation Split
# -----------------------------------------

split_dataset = dataset.train_test_split(
    test_size=0.25,
    seed=42
)

train_dataset = split_dataset["train"]
validation_dataset = split_dataset["test"]


# -----------------------------------------
# 4. Tokenizer
# -----------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


# -----------------------------------------
# 5. Tokenization
# -----------------------------------------

def tokenize_function(example):

    return tokenizer(
        example["text"],
        truncation=True
    )


train_dataset = train_dataset.map(
    tokenize_function
)

validation_dataset = validation_dataset.map(
    tokenize_function
)


# -----------------------------------------
# 6. Data Collator
# -----------------------------------------

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# -----------------------------------------
# 7. Model
# -----------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


# -----------------------------------------
# 8. Training Arguments
# -----------------------------------------

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=5e-5,
    eval_strategy="epoch",
    save_strategy="epoch"
)


# -----------------------------------------
# 9. Trainer
# -----------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)


# -----------------------------------------
# 10. Train
# -----------------------------------------

trainer.train()


# -----------------------------------------
# 11. Evaluate
# -----------------------------------------

metrics = trainer.evaluate()

print(metrics)


# -----------------------------------------
# 12. Save
# -----------------------------------------

trainer.save_model(
    "./fine_tuned_model"
)

tokenizer.save_pretrained(
    "./fine_tuned_model"
)
```

> **Important:** This tiny dataset is educational only. It is far too small to produce a meaningful real-world sentiment classifier.

---

# 57. Understanding the Code in One Picture

```text
Dataset.from_dict()
        ↓
      Dataset
        ↓
train_test_split()
        ↓
┌───────────────┐
│ Train         │
│ Validation    │
└───────┬───────┘
        ↓
      Tokenizer
        ↓
       .map()
        ↓
 Tokenized Dataset
        ↓
 DataCollatorWithPadding
        ↓
      Trainer
        ↓
 ┌───────────────┐
 │ Forward       │
 │ Loss          │
 │ Backward      │
 │ Optimizer     │
 │ Evaluation    │
 │ Checkpoints   │
 └───────┬───────┘
         ↓
  Fine-Tuned Model
```

---

# 58. `Dataset` vs `DataLoader` vs `DataCollator` vs `Trainer`

This is one of the most important sections of this lesson.

## Dataset

Stores examples.

```text
Dataset
 ↓
Examples
```

---

## DataLoader

Creates batches and manages iteration.

```text
Dataset
 ↓
DataLoader
 ↓
Batches
```

---

## DataCollator

Combines individual examples into a batch and can perform dynamic padding.

```text
Examples
 ↓
Data Collator
 ↓
Padded Batch
```

---

## Trainer

Manages much of the training infrastructure.

```text
Dataset
 ↓
DataLoader / batching
 ↓
Model
 ↓
Loss
 ↓
Backward
 ↓
Optimizer
 ↓
Evaluation
 ↓
Checkpoints
```

---

# 59. A Simple Analogy

Imagine a restaurant.

### Dataset

The raw ingredients.

```text
Dataset = ingredients
```

### Tokenizer

Prepares ingredients into usable pieces.

```text
Tokenizer = chopping/preparation
```

### Data Collator

Organizes ingredients into a plate/batch.

```text
Data Collator = plate preparation
```

### DataLoader

Brings batches to the kitchen.

```text
DataLoader = delivery of batches
```

### Model

Cooks the food.

```text
Model = cooking
```

### Loss

Tells you how far the result is from what you wanted.

```text
Loss = quality feedback
```

### Optimizer

Changes the cooking process based on feedback.

```text
Optimizer = adjustment mechanism
```

### Trainer

Manages the whole kitchen.

```text
Trainer = kitchen manager
```

The analogy is only for intuition; mathematically, the actual process is the tensor/optimization pipeline described above.

---

# 60. Why Dynamic Padding Matters

Suppose the maximum sequence length is:

\[
128
\]

But your batch contains:

```text
12 tokens
18 tokens
20 tokens
15 tokens
```

With fixed padding:

```text
128
128
128
128
```

Total token positions:

\[
4\times128=512
\]

With dynamic padding:

```text
20
20
20
20
```

Total:

\[
4\times20=80
\]

So dynamic padding can greatly reduce unnecessary computation for batches containing short sequences.

---

# 61. One Important Distinction: Padding vs Truncation

### Padding

Makes sequences longer so they have compatible lengths within a batch.

```text
[1,2,3]
 ↓
[1,2,3,PAD,PAD]
```

### Truncation

Makes sequences shorter when they exceed a specified maximum.

```text
[1,2,3,4,5,6,...]
 ↓
[1,2,3,4]
```

Therefore:

\[
\boxed{
\text{Padding increases length}
}
\]

\[
\boxed{
\text{Truncation decreases length}
}
\]

---

# 62. What Happens Inside `Trainer`?

Conceptually, a simplified Trainer loop resembles:

```python
for epoch in range(num_epochs):

    model.train()

    for batch in train_dataloader:

        optimizer.zero_grad()

        outputs = model(**batch)

        loss = outputs.loss

        loss.backward()

        optimizer.step()

    model.eval()

    evaluate()
```

The real implementation contains considerably more functionality, but this is the fundamental idea.

---

# 63. Why Learn Both Manual Training and Trainer?

Because they serve different purposes.

### Manual PyTorch

Best for understanding:

- Forward pass
- Loss
- Autograd
- Gradients
- Optimizers
- Parameter updates
- Tensor movement
- Training mechanics

### Hugging Face Trainer

Best for efficiently handling:

- Training loops
- Evaluation
- Logging
- Checkpoints
- Scheduling
- Device management
- Larger experiments

Therefore:

\[
\boxed{
\text{Understand manually}
\rightarrow
\text{Automate with Trainer}
}
\]

---

# 64. Lesson 5 Summary

The most important concepts are:

1. `Dataset` stores structured training examples.
2. `train_test_split()` can create training and validation sets.
3. `.map()` applies a tokenization function across dataset examples.
4. `DataLoader` creates batches.
5. `DataCollatorWithPadding` can dynamically pad examples in a batch.
6. Dynamic padding can reduce unnecessary computation.
7. `shuffle=True` is commonly used for training.
8. Validation data is normally not shuffled.
9. Model and tensors must be on the same device.
10. `**batch` expands a dictionary into keyword arguments.
11. `Trainer` is a high-level training abstraction.
12. `TrainingArguments` configures the training process.
13. `trainer.train()` runs the training process.
14. `trainer.evaluate()` evaluates the model.
15. `trainer.predict()` can generate predictions.
16. `trainer.save_model()` saves the fine-tuned model.
17. A tiny dataset is useful for learning but not for meaningful real-world fine-tuning.
18. `Dataset`, `DataLoader`, `DataCollator`, and `Trainer` solve different problems.

---

# 65. Questions & Answers

## Q1. What is a Hugging Face `Dataset`?

### Answer

It is a data structure provided by the `datasets` library for storing and processing structured datasets.

For example:

```text
text                         label
"I loved this movie."          1
"This was terrible."           0
```

---

## Q2. Why do we split data into training and validation sets?

### Answer

The training set is used to update model parameters.

The validation set is used to evaluate how well the model performs on examples that were not used for those parameter updates.

This helps detect overfitting.

---

## Q3. What does `.map()` do?

### Answer

`.map()` applies a function to dataset examples.

For tokenization:

```python
tokenized_dataset = dataset.map(
    tokenize_function
)
```

Conceptually:

```text
Dataset
 ↓
tokenize_function
 ↓
Tokenized Dataset
```

---

## Q4. What is dynamic padding?

### Answer

Dynamic padding pads examples only to the longest sequence in the current batch instead of always padding to a fixed maximum length.

Example:

```text
10 tokens
20 tokens
15 tokens
```

becomes:

```text
20
20
20
```

rather than:

```text
128
128
128
```

if the maximum length is 128.

---

## Q5. What is a Data Collator?

### Answer

A data collator takes individual examples and combines them into a batch.

For Transformers, it can also perform dynamic padding.

Example:

```python
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)
```

---

## Q6. What is a DataLoader?

### Answer

A PyTorch `DataLoader` iterates through a dataset and creates batches.

For example:

```python
loader = DataLoader(
    dataset,
    batch_size=8
)
```

It produces groups of 8 examples.

---

## Q7. Why do we use `shuffle=True` for the training DataLoader?

### Answer

It randomizes the order of training examples.

This prevents the model from repeatedly seeing the training data in the same order and generally improves stochastic training behavior.

---

## Q8. What is the difference between Dataset and DataLoader?

### Answer

### Dataset

Stores the examples.

```text
Dataset
 ↓
Examples
```

### DataLoader

Iterates over those examples and creates batches.

```text
Dataset
 ↓
DataLoader
 ↓
Batches
```

---

## Q9. What is `Trainer`?

### Answer

`Trainer` is a high-level Hugging Face abstraction that manages much of the training and evaluation infrastructure.

It can handle things such as:

- Training loop
- Evaluation
- Optimizer setup
- Scheduling
- Checkpoints
- Logging
- Device management

---

## Q10. Does `Trainer` replace backpropagation?

### Answer

No.

Backpropagation still happens.

`Trainer` automates the training process that includes operations conceptually equivalent to:

```python
loss.backward()
```

and:

```python
optimizer.step()
```

---

## Q11. What does `trainer.train()` actually do?

### Answer

Conceptually:

```text
Get batch
   ↓
Forward pass
   ↓
Calculate loss
   ↓
Backpropagation
   ↓
Optimizer update
   ↓
Repeat
```

It performs the training loop and associated infrastructure.

---

## Q12. What is `TrainingArguments`?

### Answer

It configures how training should be performed.

For example:

```python
TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-5
)
```

These specify things such as:

- Number of epochs
- Batch size
- Learning rate
- Output directory

---

## Q13. What does `per_device_train_batch_size` mean?

### Answer

It specifies how many examples are processed per device in each training batch.

For example:

```python
per_device_train_batch_size=8
```

means approximately 8 examples per device per batch.

---

## Q14. Why do model and tensors need to be on the same device?

### Answer

PyTorch operations generally require the participating tensors and model parameters to be on compatible devices.

For example, if the model is on GPU:

```text
Model → GPU
```

the input tensors should also be on GPU:

```text
Input → GPU
```

Otherwise, you can get a device mismatch error.

---

## Q15. What does `model(**batch)` mean?

### Answer

If:

```python
batch = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "labels": labels
}
```

then:

```python
model(**batch)
```

is equivalent to:

```python
model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)
```

This is Python dictionary unpacking.

---

## Q16. Why is dynamic padding often more efficient?

### Answer

Because it avoids padding every sequence to a potentially large global maximum.

For example:

```text
Fixed padding:
128 + 128 + 128 + 128

Dynamic padding:
20 + 20 + 20 + 20
```

The second case contains fewer unnecessary padding positions.

---

# 66. Self-Test

Try answering these without looking back:

1. What is a Hugging Face `Dataset`?
2. What does `train_test_split()` do?
3. What does `.map()` do?
4. What is dynamic padding?
5. What does `DataCollatorWithPadding` do?
6. What does a PyTorch `DataLoader` do?
7. Why is training data commonly shuffled?
8. Why isn't validation data normally shuffled?
9. What is `Trainer`?
10. What does `trainer.train()` conceptually do?
11. What is `TrainingArguments`?
12. What does `per_device_train_batch_size` mean?
13. Why must model and input tensors be on the same device?
14. What does `model(**batch)` mean?
15. Why is dynamic padding useful?
16. Explain the complete pipeline from raw dataset to fine-tuned model.

---

# 67. Important Mental Model

You should now understand these four components separately:

```text
Dataset
    ↓
Stores examples

DataLoader
    ↓
Creates/iterates batches

Data Collator
    ↓
Combines examples into correctly padded batches

Trainer
    ↓
Manages much of the training/evaluation process
```

Do **not** think of them as interchangeable.

They solve different problems.

---

# 68. Final Mental Model

The complete fine-tuning stack is:

```text
                    RAW DATA
                       ↓
                Hugging Face Dataset
                       ↓
               Train / Validation Split
                       ↓
                    Tokenizer
                       ↓
                  Tokenized Data
                       ↓
                Data Collator
                       ↓
                    DataLoader
                       ↓
               Batches of Tensors
                       ↓
              Pretrained Transformer
                       ↓
                     Logits
                       ↓
                      Loss
                       ↓
                 Backpropagation
                       ↓
                    Gradients
                       ↓
                    Optimizer
                       ↓
              Updated Parameters
                       ↓
                 Validation
                       ↓
               Fine-Tuned Model
```

And when using the high-level API:

```text
Dataset
   ↓
Tokenizer
   ↓
Data Collator
   ↓
Trainer
   ↓
trainer.train()
   ↓
Fine-Tuned Model
```

The important thing is that **`Trainer` is an abstraction over the training process you already learned manually in Lesson 4.**

---

# 69. Next Lesson

## Lesson 6 — Evaluation, Metrics, Overfitting & Model Selection

Next we will focus on:

> **How do we know whether our fine-tuned model is actually good?**

We will learn:

```text
Training Loss
Validation Loss
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Overfitting
Underfitting
Early Stopping
Best Model Selection
```

We will also learn why:

```text
Training accuracy = 95%
```

does **not necessarily mean**:

```text
Model is good
```

and how validation/test performance gives us a much more meaningful picture.
