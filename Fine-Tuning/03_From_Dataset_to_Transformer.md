# Fine-Tuning — Lesson 3: From Dataset to Transformer

> **Goal:** Understand what actually happens to data when we fine-tune a pretrained Transformer, from raw text all the way to loss, backpropagation, and parameter updates.

---

# 1. The Complete Pipeline

The fundamental fine-tuning pipeline is:

```text
Raw Dataset
    ↓
Tokenizer
    ↓
Input IDs
    ↓
Attention Mask
    ↓
Labels
    ↓
Pretrained Transformer
    ↓
Logits
    ↓
Loss
    ↓
Backpropagation
    ↓
Parameter Update
```

This lesson is important because later, when you see Hugging Face code such as:

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)
```

you should understand exactly what each argument represents.

---

# 2. Start With Raw Text

Suppose our dataset contains:

```text
"I loved this movie!"
"This movie was terrible."
"I really enjoyed it."
"I hated the ending."
```

Humans understand these sentences directly.

A Transformer does not process raw text directly.

A Transformer ultimately operates on **numbers**.

Therefore:

```text
Text
 ↓
Numbers
```

The first major step is **tokenization**.

---

# 3. Tokenization

A tokenizer converts text into smaller units called **tokens**.

For example:

```text
"I love AI"
```

could conceptually become:

```text
["I", "love", "AI"]
```

Modern tokenizers often use **subword tokenization**, so the exact tokens depend on the tokenizer.

For example, a word might be split into multiple pieces:

```text
"unbelievable"
      ↓
["un", "believ", "able"]
```

The exact splitting depends on the tokenizer vocabulary and algorithm.

---

# 4. Token IDs

The Transformer still cannot directly process strings such as:

```text
["I", "love", "AI"]
```

Each token is therefore mapped to an integer ID.

Conceptually:

```text
"I"     → 101
"love"  → 2293
"AI"    → 993
```

So:

```text
"I love AI"
      ↓
Tokenizer
      ↓
[101, 2293, 993]
```

These are called:

\[
\boxed{\text{input\_ids}}
\]

The exact IDs depend on the tokenizer.

---

# 5. Why Do We Need Token IDs?

The Transformer performs mathematical operations such as:

\[
XW_Q
\]

\[
QK^T
\]

\[
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
\]

These operations require numerical tensors.

Therefore:

```text
Text
 ↓
Tokens
 ↓
Token IDs
 ↓
Embeddings
 ↓
Transformer
```

---

# 6. Token IDs Are Not Embeddings

This distinction is extremely important.

Suppose:

```text
"I love AI"
```

becomes:

```text
[101, 2293, 993]
```

These integers are **token IDs**, not meaningful dense vectors.

The model has an embedding layer that maps token IDs to vectors.

Conceptually:

```text
Token ID
   ↓
Embedding Lookup
   ↓
Dense Vector
```

For example:

```text
2293
 ↓
[0.21, -0.14, 0.73, ...]
```

Therefore:

\[
\boxed{
\text{Token ID} \neq \text{Embedding}
}
\]

---

# 7. The Embedding Process

Suppose the vocabulary contains:

```text
Token      ID

I          101
love       2293
AI         993
```

The embedding matrix contains a vector for every vocabulary token.

Conceptually:

\[
E\in\mathbb{R}^{V\times D}
\]

where:

- \(V\) = vocabulary size
- \(D\) = embedding dimension

For example, if:

\[
V=50,000
\]

and:

\[
D=768
\]

then:

\[
E\in\mathbb{R}^{50000\times768}
\]

The token ID selects the corresponding row from the embedding matrix.

---

# 8. Padding

Now suppose we have sentences with different lengths:

```text
"I love AI"

"I really love artificial intelligence"
```

Their token counts are different.

When processing a batch, we generally need compatible tensor dimensions.

Therefore, shorter sequences can be padded.

Conceptually:

```text
Sentence 1:
[I, love, AI]

Sentence 2:
[I, really, love, artificial, intelligence]
```

might become:

```text
[I, love, AI, PAD, PAD]

[I, really, love, artificial, intelligence]
```

The special token:

```text
PAD
```

is used to make sequence lengths compatible within the batch.

---

# 9. Attention Mask

Now we have a problem.

The model should not treat padding tokens as actual meaningful words.

This is where the **attention mask** is useful.

Conceptually:

```text
Input IDs:

[I, love, AI, PAD, PAD]

Attention Mask:

[1,   1,    1,   0,   0]
```

Typically:

```text
1 → real token
0 → padding token
```

Therefore:

\[
\boxed{
\text{attention\_mask}=1
\Rightarrow
\text{real token}
}
\]

\[
\boxed{
\text{attention\_mask}=0
\Rightarrow
\text{padding position}
}
\]

---

# 10. Why Is It Called an Attention Mask?

The attention mask provides information about which positions should be treated as valid input positions during attention.

For example:

```text
Input:

I love AI PAD PAD

Mask:

1  1   1   0   0
```

The model should focus on:

```text
I
love
AI
```

and ignore the padding positions.

> **Important:** This is different from the causal mask used to prevent future-token access in autoregressive decoder-style self-attention.

---

# 11. Labels

Now we need to tell the model what the correct answer is.

Suppose we are doing sentiment classification:

```text
"I loved this movie."      → Positive
"This movie was terrible." → Negative
```

We might encode:

```text
Positive → 1
Negative → 0
```

So:

```text
Text                       Label

"I loved this movie"         1
"This was terrible"          0
```

The labels represent the **target output**.

---

# 12. Input vs Label

This distinction is fundamental.

### Input

What we give the model:

```text
"I loved this movie."
```

### Label

What we want the model to predict:

```text
Positive
```

So:

```text
Input
  ↓
Model
  ↓
Prediction
  ↓
Compare with Label
  ↓
Loss
```

---

# 13. Complete Classification Example

Let's combine everything.

Raw dataset:

```text
"I loved this movie." → Positive
```

### Step 1 — Tokenization

```text
"I loved this movie."
        ↓
["I", "loved", "this", "movie", "."]
```

### Step 2 — Token IDs

```text
[101, 1045, 3866, 2023, 3185, 1012]
```

> The IDs above are illustrative. Exact IDs depend on the tokenizer.

### Step 3 — Attention Mask

```text
[1, 1, 1, 1, 1, 1]
```

### Step 4 — Label

```text
1
```

So the model receives something conceptually like:

```text
input_ids:
[101, 1045, 3866, 2023, 3185, 1012]

attention_mask:
[1, 1, 1, 1, 1, 1]

labels:
1
```

---

# 14. What Happens Inside the Transformer?

Now the data enters the pretrained Transformer.

Conceptually:

```text
Input IDs
    ↓
Embedding Layer
    ↓
Position Information
    ↓
Transformer Layers
    ↓
Contextual Representations
    ↓
Task Head
    ↓
Logits
```

The important point is:

> The token IDs themselves are not what the attention mechanism directly operates on.

The conceptual flow is:

```text
Token IDs
   ↓
Embeddings
   ↓
Transformer
   ↓
Contextual representations
```

---

# 15. What Are Logits?

The model eventually produces numerical scores representing its predictions.

These raw scores are called:

\[
\boxed{\text{logits}}
\]

Suppose we have two classes:

```text
Positive
Negative
```

The model might produce:

```text
Positive → 2.8
Negative → -1.2
```

These are logits.

They are **not probabilities yet**.

---

# 16. Logits → Probabilities

We can apply Softmax:

\[
P_i=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
\]

where \(z_i\) is a logit.

For example:

```text
Logits:

Positive = 2.8
Negative = -1.2
```

Softmax converts them into probabilities approximately like:

```text
Positive → 0.982
Negative → 0.018
```

Therefore:

```text
Logits
  ↓
Softmax
  ↓
Probabilities
```

### Important PyTorch Note

When using `CrossEntropyLoss` in PyTorch, you normally pass the **raw logits** to the loss function rather than applying Softmax yourself.

PyTorch's cross-entropy implementation handles the required log-softmax operation internally.

---

# 17. Prediction

The model can select the class with the highest score.

For example:

```text
Positive → 2.8
Negative → -1.2
```

Therefore:

```text
Prediction = Positive
```

But we still need to determine how good the prediction is.

---

# 18. Loss

We compare:

```text
Prediction
      vs
Correct Label
```

Suppose:

```text
Prediction:
Positive

Actual label:
Positive
```

The loss should be relatively low if the model assigned high probability to the correct class.

If:

```text
Prediction:
Negative

Actual label:
Positive
```

the loss should be higher.

For classification, a common choice is **cross-entropy loss**.

---

# 19. Backpropagation

Now we connect everything to PyTorch.

We have:

```text
Input
 ↓
Transformer
 ↓
Logits
 ↓
Loss
```

Then:

```python
loss.backward()
```

calculates gradients.

Conceptually:

\[
\frac{\partial L}{\partial\theta}
\]

where \(\theta\) represents the trainable parameters.

Then:

```python
optimizer.step()
```

updates the trainable parameters.

So:

```text
Dataset
   ↓
Tokenizer
   ↓
Transformer
   ↓
Logits
   ↓
Loss
   ↓
Backward
   ↓
Gradients
   ↓
Optimizer
   ↓
Updated Model
```

This is the actual fine-tuning loop.

---

# 20. Where Does the Pretrained Model Come In?

This connects directly to Lesson 2.

We don't start with:

```text
Random Transformer
```

Instead:

```text
Pretrained Transformer
```

So:

```text
Pretrained Parameters
        ↓
Task Dataset
        ↓
Forward Pass
        ↓
Loss
        ↓
Backpropagation
        ↓
Parameter Updates
```

This is fine-tuning.

---

# 21. A Hugging Face Example

Eventually you will see code such as:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
```

Let's understand each part.

---

## `AutoTokenizer`

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

This loads the tokenizer associated with the pretrained model.

It knows things such as:

- Vocabulary
- Tokenization rules
- Special tokens
- Padding behavior
- Token-to-ID mapping

---

## `AutoModelForSequenceClassification`

```python
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
```

This loads a pretrained Transformer and configures a classification head.

Conceptually:

```text
Pretrained Transformer
        ↓
Classification Head
        ↓
2 output classes
```

For example:

```text
0 → Negative
1 → Positive
```

---

# 22. Tokenizing Text in Code

For example:

```python
text = "I loved this movie."

encoded = tokenizer(
    text,
    padding=True,
    truncation=True
)
```

The tokenizer can produce fields such as:

```python
encoded["input_ids"]
encoded["attention_mask"]
```

Conceptually:

```text
input_ids:
[101, 1045, ..., 1012]

attention_mask:
[1, 1, ..., 1]
```

The exact values depend on the tokenizer.

---

# 23. Batching

In real training, we usually don't send one sentence at a time.

We use batches.

For example:

```text
Batch

Sentence 1
Sentence 2
Sentence 3
Sentence 4
```

The tokenizer converts them into tensors such as:

```text
input_ids
    shape:
    [batch_size, sequence_length]

attention_mask
    shape:
    [batch_size, sequence_length]
```

For example:

\[
(4,128)
\]

could mean:

```text
4 examples
128 tokens maximum per example
```

---

# 24. Why Tensor Shapes Matter

This connects directly to PyTorch.

Suppose:

\[
X\in\mathbb{R}^{B\times S}
\]

where:

- \(B\) = batch size
- \(S\) = sequence length

After embedding:

\[
X\in\mathbb{R}^{B\times S\times D}
\]

where:

- \(D\) = hidden/embedding dimension

For example:

```text
input_ids:

[32, 128]
```

means:

```text
32 sequences
128 tokens per sequence
```

After embedding:

```text
[32, 128, 768]
```

means:

```text
32 sequences
128 tokens per sequence
768-dimensional representation per token
```

Understanding these tensor shapes is essential for PyTorch and Transformer work.

---

# 25. What Is the Classification Head?

The pretrained Transformer produces contextual representations.

For classification, we need a component that converts those representations into class scores.

Conceptually:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Transformer
 ↓
Contextual Representation
 ↓
Classification Head
 ↓
Logits
```

If:

\[
D=768
\]

and:

\[
\text{num\_labels}=2
\]

the classification head can conceptually perform:

\[
z=XW+b
\]

where the output contains scores for the classes.

---

# 26. Why Do We Add a Task Head?

The pretrained Transformer was not necessarily trained specifically for our new task.

For example:

```text
Pretrained Transformer
```

may have been pretrained using a general language objective.

We want:

```text
Sentiment Classification
```

So we configure a task-specific head:

```text
Pretrained Transformer
        +
Classification Head
        ↓
Sentiment Model
```

The head allows the model to produce outputs appropriate for the new task.

---

# 27. Full Fine-Tuning Flow

Now we can see the whole process:

```text
                    RAW DATA
                       ↓
                  Tokenization
                       ↓
                    Input IDs
                       ↓
                 Attention Mask
                       ↓
                     Labels
                       ↓
             ┌───────────────────┐
             │ Pretrained        │
             │ Transformer       │
             └─────────┬─────────┘
                       ↓
                Contextual
               Representations
                       ↓
                Classification
                     Head
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
```

That is the fundamental fine-tuning pipeline.

---

# 28. What You Should Understand Before Coding

At this point, you should be able to distinguish:

### `input_ids`

Integer IDs representing tokens.

### `attention_mask`

Indicates which positions are valid tokens versus padding, and can be used by the model to control attention.

### `labels`

The desired target output used to calculate the loss.

### `logits`

Raw model scores before probability normalization.

### `loss`

Measures how far the predictions are from the desired targets.

### `gradients`

Indicate how the trainable parameters should change to reduce the loss.

---

# 29. The Most Important Pipeline

Conceptually:

\[
\boxed{
\text{Text}
\rightarrow
\text{Tokens}
\rightarrow
\text{Token IDs}
\rightarrow
\text{Embeddings}
\rightarrow
\text{Transformer}
\rightarrow
\text{Logits}
\rightarrow
\text{Loss}
}
\]

Then:

\[
\boxed{
\text{Loss}
\rightarrow
\text{Backpropagation}
\rightarrow
\text{Gradients}
\rightarrow
\text{Parameter Updates}
}
\]

Combined:

```text
Text
 ↓
Tokenizer
 ↓
Input IDs + Attention Mask
 ↓
Pretrained Transformer
 ↓
Logits
 ↓
Loss
 ↓
Backward
 ↓
Gradients
 ↓
Optimizer
 ↓
Fine-Tuned Model
```

---

# 30. Padding Mask vs Causal Mask

Do not confuse these two concepts.

## Padding Attention Mask

Example:

```text
Tokens:
I love AI PAD PAD

Mask:
1  1   1   0   0
```

Purpose:

> Ignore padding positions.

## Causal Mask

Example:

```text
✓ ✗ ✗ ✗
✓ ✓ ✗ ✗
✓ ✓ ✓ ✗
✓ ✓ ✓ ✓
```

Purpose:

> Prevent a token from attending to future tokens.

The causal mask is especially important for autoregressive decoder-style models.

---

# 31. Lesson 3 Summary

The key points are:

1. Transformers process numerical tensors, not raw text.
2. A **tokenizer** converts text into tokens.
3. Tokens are mapped to integer **token IDs**.
4. Token IDs are not embeddings.
5. An embedding layer converts token IDs into dense vectors.
6. Padding allows sequences of different lengths to form batches.
7. `attention_mask` can identify real tokens versus padding.
8. `labels` represent the desired outputs.
9. The Transformer produces contextual representations and ultimately task-specific **logits**.
10. Logits are raw scores, not probabilities.
11. Cross-entropy can be calculated from logits and labels.
12. `loss.backward()` calculates gradients.
13. `optimizer.step()` updates trainable parameters.
14. Fine-tuning is:

\[
\boxed{
\text{Pretrained Model}
+
\text{Task Data}
+
\text{Optimization}
\rightarrow
\text{Fine-Tuned Model}
}
\]

---

# 32. Questions & Answers

## Q1. Why can't we directly give text such as `"I love AI"` to a Transformer?

### Answer

A Transformer performs numerical operations.

Therefore, text must first be converted into tokens and then into numerical token IDs.

```text
Text
 ↓
Tokens
 ↓
Token IDs
 ↓
Embeddings
 ↓
Transformer
```

---

## Q2. What is `input_ids`?

### Answer

`input_ids` are integer IDs representing the tokens in the input sequence.

For example:

```text
"I love AI"
```

might become:

```text
[101, 2293, 993]
```

The exact IDs depend on the tokenizer.

---

## Q3. Are token IDs the same as embeddings?

### Answer

No.

Token IDs are integers used to identify tokens.

Embeddings are dense numerical vectors.

```text
Token ID
   ↓
Embedding Lookup
   ↓
Dense Vector
```

Therefore:

\[
\boxed{
\text{Token ID} \neq \text{Embedding}
}
\]

---

## Q4. Why do we need an attention mask?

### Answer

One common use is to tell the model which positions contain real tokens and which positions are padding.

For example:

```text
Input:
[I, love, AI, PAD, PAD]

Mask:
[1, 1, 1, 0, 0]
```

The mask helps prevent padding positions from being treated as meaningful input.

---

## Q5. Is a padding mask the same as a causal mask?

### Answer

No.

### Padding mask

Handles padding:

```text
1 1 1 0 0
```

### Causal mask

Prevents future-token access:

```text
✓ ✗ ✗
✓ ✓ ✗
✓ ✓ ✓
```

They solve different problems.

---

## Q6. What are logits?

### Answer

Logits are the raw output scores produced by a model before probability normalization.

For example:

```text
Positive → 2.8
Negative → -1.2
```

These can be converted into probabilities using Softmax.

For classification with PyTorch `CrossEntropyLoss`, you normally provide the raw logits directly rather than applying Softmax manually.

---

## Q7. What are labels?

### Answer

Labels represent the correct target output.

For sentiment classification:

```text
"I loved this movie." → 1
"This was terrible."  → 0
```

The loss compares the model's predictions against these labels.

---

## Q8. What is the complete fine-tuning pipeline?

### Answer

```text
Raw Text
   ↓
Tokenizer
   ↓
Input IDs
   ↓
Attention Mask
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
```

---

# 33. Self-Test Before Lesson 4

Try answering these without looking back:

1. What is tokenization?
2. What are token IDs?
3. What is the difference between token IDs and embeddings?
4. What does `input_ids` contain?
5. What does `attention_mask` do?
6. What are `labels`?
7. What are logits?
8. Why don't we normally apply Softmax before `CrossEntropyLoss`?
9. What is the difference between a padding mask and a causal mask?
10. What happens after `loss.backward()`?
11. What does `optimizer.step()` do?
12. Explain the entire pipeline from raw text to updated model parameters.

---

# 34. Next Lesson

## Lesson 4 — Our First Fine-Tuning Code

Now we will move from theory to implementation.

We will use:

```text
Python
   ↓
PyTorch
   ↓
Hugging Face Transformers
   ↓
Tokenizer
   ↓
Pretrained DistilBERT
   ↓
Small Dataset
   ↓
Fine-Tuning
   ↓
Evaluation
```

We will first understand the important steps without hiding everything behind high-level APIs.

Then we will learn the Hugging Face `Trainer` approach.

The goal is that when you eventually see:

```python
trainer.train()
```

you understand what is happening underneath it—not just how to call the function.
