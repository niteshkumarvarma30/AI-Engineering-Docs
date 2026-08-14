# Lesson 09 — Pre-training Objectives

## Complete LLM Training Note

### Core question

How does a Transformer actually learn to become a language model?

For modern decoder-only LLMs, the central idea is:

\[
\boxed{\text{Next-token prediction}}
\]

The model repeatedly receives previous tokens and learns to assign high probability to the correct next token.

---

# 1. What Does Training an LLM Mean?

Suppose the training text is:

```text
I love machine learning
```

The model learns next-token relationships:

```text
Input: I
Target: love
```

```text
Input: I love
Target: machine
```

```text
Input: I love machine
Target: learning
```

So the fundamental task is:

\[
\boxed{\text{Given previous tokens, predict the next token}}
\]

---

# 2. Next-Token Prediction

Suppose tokenization produces:

```text
["I", "love", "machine", "learning"]
```

The training sequence can be viewed as:

```text
Input:
I       love       machine

Target:
love    machine    learning
```

Each position predicts the token immediately after it.

Importantly, the Transformer can calculate all these training predictions in one forward pass.

---

# 3. Why Can Training Be Parallel?

You might wonder:

> If generation happens one token at a time, why isn't training also one token at a time?

Because training uses **causal masking**.

For:

```text
I love machine learning
```

the attention pattern is approximately:

```text
          I   love   machine   learning
I         ✓    ✗       ✗          ✗
love      ✓    ✓       ✗          ✗
machine   ✓    ✓       ✓          ✗
learning  ✓    ✓       ✓          ✓
```

Therefore:

```text
Position 1 → predicts token 2
Position 2 → predicts token 3
Position 3 → predicts token 4
```

All prediction positions can be computed in parallel.

---

# 4. Causal Language Modeling

This objective is called:

\[
\boxed{\text{Causal Language Modeling (CLM)}}
\]

"Causal" means the prediction at position \(t\) cannot use future tokens.

The autoregressive factorization is:

\[
\boxed{
P(x_1,\ldots,x_T)
=
\prod_{t=1}^{T}
P(x_t|x_1,\ldots,x_{t-1})
}
\]

The model learns:

\[
P(x_t|x_{<t})
\]

---

# 5. Decoder-Only LLM Training Flow

A simplified GPT-style training pipeline:

```text
Training Text
     ↓
Tokenizer
     ↓
Token IDs
     ↓
Embedding
     ↓
Positional Information / RoPE
     ↓
Transformer Blocks
     │
     ├── Causal Self-Attention
     ├── FFN / MoE
     └── Residual + Normalization
     ↓
Hidden Representations
     ↓
LM Head
     ↓
Logits
     ↓
Cross-Entropy Loss
     ↓
Backpropagation
     ↓
Optimizer Update
```

This is repeated over many training batches.

---

# 6. What Is the LM Head?

The Transformer produces a hidden representation:

\[
h_t
\]

for every position.

The model needs to convert this representation into a score for every vocabulary token.

A simplified output layer is:

\[
z_t=W_oh_t+b
\]

where:

\[
z_t\in\mathbb{R}^{V}
\]

and \(V\) is the vocabulary size.

For example:

```text
Vocabulary = 50,000 tokens

Hidden state
     ↓
LM Head
     ↓
50,000 logits
```

---

# 7. What Are Logits?

Suppose the vocabulary contains:

```text
["cat", "dog", "car", "food"]
```

The model might produce:

```text
cat  → 2.1
dog  → 5.4
car  → 0.7
food → 1.2
```

These values are **logits**.

Logits are not probabilities.

A larger logit generally means the model currently favors that token more strongly.

---

# 8. Softmax

Softmax converts logits into probabilities:

\[
P_i=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
\]

For example:

```text
cat  → 0.08
dog  → 0.72
car  → 0.04
food → 0.16
```

The probabilities sum to:

\[
1
\]

---

# 9. Correct Token and Loss

Suppose the correct next token is:

```text
dog
```

and the model predicts:

```text
cat  → 0.08
dog  → 0.72
car  → 0.04
food → 0.16
```

Then:

\[
P(\text{correct})=0.72
\]

The single-token cross-entropy loss is:

\[
\boxed{
L=-\log P(\text{correct})
}
\]

Therefore:

\[
L=-\log(0.72)
\]

\[
\boxed{L\approx0.329}
\]

---

# 10. Numerical Loss Example

Suppose:

```text
Correct token = dog
```

Prediction:

```text
cat  → 0.10
dog  → 0.70
car  → 0.05
food → 0.15
```

Then:

\[
L=-\log(0.70)
\]

\[
\boxed{L\approx0.357}
\]

If the model improves:

```text
cat  → 0.05
dog  → 0.90
car  → 0.02
food → 0.03
```

then:

\[
L=-\log(0.90)
\]

\[
\boxed{L\approx0.105}
\]

So:

```text
Higher probability for correct token
          ↓
Lower loss
```

---

# 11. Why Cross-Entropy?

Cross-entropy strongly penalizes the model when it assigns very low probability to the correct token.

For example:

\[
P(\text{correct})=0.9
\]

gives:

\[
L=-\log(0.9)\approx0.105
\]

while:

\[
P(\text{correct})=0.1
\]

gives:

\[
L=-\log(0.1)\approx2.303
\]

Therefore:

```text
Correct token probability ↑
        ↓
Loss ↓
```

---

# 12. Complete Training Example

Text:

```text
The cat is sleeping
```

Suppose tokenization gives:

```text
["The", "cat", "is", "sleeping"]
```

The model learns:

```text
The              → cat
The cat          → is
The cat is       → sleeping
```

Conceptually:

```text
                Transformer
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
     Predict       Predict      Predict
       cat           is        sleeping
        │            │            │
        ▼            ▼            ▼
      Loss 1       Loss 2       Loss 3
        └────────────┼────────────┘
                     ▼
                Total Loss
```

---

# 13. Shifted Inputs and Labels

A typical implementation conceptually uses:

```text
Input:
The      cat       is       sleeping

Label:
cat      is        sleeping <EOS>
```

The labels are shifted by one position.

For a sequence:

\[
[x_1,x_2,x_3,x_4]
\]

the model predicts:

\[
[x_2,x_3,x_4,\ldots]
\]

The exact handling of the final position depends on the tokenizer and training setup.

---

# 14. Teacher Forcing

During autoregressive training, the model is given the **ground-truth previous tokens**.

Suppose the correct sequence is:

```text
I love machine learning
```

Training uses:

```text
I
I love
I love machine
```

rather than repeatedly feeding the model's own sampled mistakes.

This is commonly described as **teacher forcing**.

It makes training much more efficient and stable.

---

# 15. Training vs Generation

This distinction is extremely important.

## Training

```text
Ground-truth sequence
        ↓
Causal mask
        ↓
Many next-token predictions in parallel
        ↓
Loss
        ↓
Backpropagation
```

## Inference

```text
Prompt
  ↓
Predict next token
  ↓
Append generated token
  ↓
Predict next token
  ↓
Append
  ↓
Repeat
```

Therefore:

\[
\boxed{\text{Training = highly parallel}}
\]

while:

\[
\boxed{\text{Autoregressive inference = sequential}}
\]

---

# 16. Why Causal Masking Is Necessary

Without a causal mask, the model could see the future answer.

Suppose:

```text
The cat is sleeping
```

If the model is predicting:

```text
sleeping
```

it must not be allowed to attend to the actual future token `sleeping`.

Otherwise training would leak the answer.

The rule is:

```text
Current position
      ↓
Can see previous/current positions
      ↓
Cannot see future positions
```

---

# 17. Causal Attention Matrix

For four tokens:

```text
          1   2   3   4

1         ✓   ✗   ✗   ✗
2         ✓   ✓   ✗   ✗
3         ✓   ✓   ✓   ✗
4         ✓   ✓   ✓   ✓
```

A numerical mask can be represented as:

\[
M=
\begin{bmatrix}
0 & -\infty & -\infty & -\infty\\
0 & 0 & -\infty & -\infty\\
0 & 0 & 0 & -\infty\\
0 & 0 & 0 & 0
\end{bmatrix}
\]

The mask is added to attention scores before softmax.

Because:

\[
e^{-\infty}=0
\]

future positions receive zero attention probability.

---

# 18. Causal vs Bidirectional Attention

There are two major language-modeling paradigms.

## Causal

Used by GPT-style decoder-only LLMs.

```text
Token 1 → sees 1
Token 2 → sees 1,2
Token 3 → sees 1,2,3
Token 4 → sees 1,2,3,4
```

Future tokens are hidden.

## Bidirectional

Used by encoder-style models such as BERT during masked-language-model pretraining.

A token can use information from both left and right context.

```text
Token 1 ↔ Token 2 ↔ Token 3 ↔ Token 4
```

---

# 19. Masked Language Modeling

BERT-style pretraining commonly uses:

\[
\boxed{\text{Masked Language Modeling (MLM)}}
\]

Example:

```text
The cat is sleeping
```

Mask a token:

```text
The cat is [MASK]
```

The model predicts:

```text
sleeping
```

Unlike causal LM, the model can generally use both left and right context.

---

# 20. Causal LM vs Masked LM

| Property | Causal LM | Masked LM |
|---|---|---|
| Typical example | GPT-style | BERT-style |
| Objective | Next-token prediction | Predict masked tokens |
| Attention | Causal | Usually bidirectional |
| Future tokens visible? | No | Yes, except target information is masked |
| Natural text generation | Directly suited | Not directly autoregressive |
| Typical architecture | Decoder-only | Encoder-only |

---

# 21. Why GPT Uses Causal LM

GPT-style models are designed for autoregressive generation.

At inference:

```text
Prompt
 ↓
Predict next token
 ↓
Append token
 ↓
Predict next token
 ↓
Append token
 ↓
Repeat
```

Therefore the training objective naturally matches generation:

\[
\boxed{
P(\text{next token}|\text{previous tokens})
}
\]

---

# 22. What Gets Updated?

The model contains many trainable parameters, such as:

```text
Embedding parameters
Attention Q/K/V projections
Attention output projections
FFN parameters
MoE router parameters
MoE expert parameters
Normalization parameters
LM head parameters
```

During training:

```text
Prediction
   ↓
Loss
   ↓
Backpropagation
   ↓
Gradients
   ↓
Optimizer
   ↓
Parameter updates
```

For a parameter \(\theta\):

\[
\boxed{
\theta_{new}
=
\theta_{old}
-
\eta
\frac{\partial L}{\partial\theta}
}
\]

where:

- \(L\) = loss
- \(\eta\) = learning rate

---

# 23. Complete Training Loop

```text
             Training Dataset
                    │
                    ▼
                Tokenization
                    │
                    ▼
                 Token IDs
                    │
                    ▼
                Embeddings
                    │
                    ▼
             Positional Information
                    │
                    ▼
             Transformer Blocks
                    │
          ┌─────────┴─────────┐
          │                   │
   Causal Attention       FFN / MoE
          │                   │
          └─────────┬─────────┘
                    ▼
                 Logits
                    │
                    ▼
             Cross-Entropy Loss
                    │
                    ▼
              Backpropagation
                    │
                    ▼
             Optimizer Update
                    │
                    ▼
              Updated Model
```

Then another batch is processed.

---

# 24. Batch Training

LLMs normally train on batches rather than one sentence at a time.

Conceptually:

```text
Batch
 ├── Sequence 1
 ├── Sequence 2
 ├── Sequence 3
 └── Sequence 4
```

A simplified tensor flow is:

```text
Token IDs
[batch_size, sequence_length]
          ↓
Transformer
          ↓
Logits
[batch_size, sequence_length, vocabulary_size]
          ↓
Cross-Entropy
          ↓
Loss
```

Padding/attention masks and exact tensor layouts depend on the implementation.

---

# 25. Mathematical Training Objective

For:

\[
x_1,x_2,\ldots,x_T
\]

the autoregressive language-model objective is:

\[
\boxed{
P(x_1,\ldots,x_T)
=
\prod_{t=1}^{T}
P(x_t|x_1,\ldots,x_{t-1})
}
\]

The negative log-likelihood loss is:

\[
\boxed{
L
=
-\sum_{t=1}^{T}
\log P(x_t|x_{<t})
}
\]

Often the loss is averaged across valid prediction positions:

\[
\boxed{
L
=
-\frac{1}{T}
\sum_{t=1}^{T}
\log P(x_t|x_{<t})
}
\]

This is the mathematical heart of decoder-only LLM pretraining.

---

# 26. What Does the Model Learn?

Suppose the model repeatedly encounters:

```text
The cat is sleeping.
The cat is sleeping.
The cat is sleeping.
```

It learns statistical relationships that increase the probability of appropriate continuations.

Eventually:

```text
"The cat is"
      ↓
sleeping → high probability
```

But the model does not necessarily store an explicit symbolic rule:

```text
"The cat is" → "sleeping"
```

Instead, these behaviors emerge from learned neural-network representations and parameters.

---

# 27. One Training Step

A single optimization step is:

```text
Training batch
     ↓
Forward pass
     ↓
Logits
     ↓
Cross-entropy loss
     ↓
Backward pass
     ↓
Gradients
     ↓
Optimizer
     ↓
Parameter update
```

Then the next batch.

---

# 28. Epochs and Steps

Suppose the dataset is divided into:

```text
Batch 1
Batch 2
Batch 3
...
Batch N
```

Processing the complete dataset once is approximately one:

\[
\boxed{\text{Epoch}}
\]

A training **step** generally refers to one optimizer update, although terminology can vary with gradient accumulation and distributed training.

Large LLM pretraining typically uses enormous datasets and very large numbers of optimization steps.

---

# 29. Pretraining vs Fine-Tuning

Do not confuse these.

## Pretraining

Goal:

> Learn broad language and world representations from large-scale data.

For a decoder-only LLM, the core objective is typically:

\[
\boxed{\text{Next-token prediction}}
\]

## Fine-tuning

Goal:

> Adapt an already pretrained model to a particular behavior, task, domain, or instruction format.

Examples:

```text
Instruction following
Question answering
Code
Domain-specific tasks
Tool use
```

---

# 30. What Does "Pretrained" Mean?

A pretrained LLM has already learned its parameters by repeatedly optimizing its training objective.

Conceptually:

```text
Huge text dataset
       ↓
Tokenization
       ↓
Next-token prediction
       ↓
Loss
       ↓
Backpropagation
       ↓
Parameter update
       ↓
Repeat over enormous numbers of tokens
       ↓
Pretrained LLM
```

---

# 31. Complete Connection With Previous Lessons

You have now connected:

```text
Text
  ↓
Tokenization
  ↓
Token IDs
  ↓
Embeddings
  ↓
Positional Information / RoPE
  ↓
Q/K/V
  ↓
Self-Attention
  ↓
Causal Mask
  ↓
MHA / GQA / MQA
  ↓
FFN / MoE
  ↓
Hidden States
  ↓
LM Head
  ↓
Logits
  ↓
Softmax
  ↓
Cross-Entropy
  ↓
Backpropagation
  ↓
Parameter Update
```

This is the complete conceptual path from training text to learning.

---

# 32. The Most Important Formula

For a decoder-only LLM:

\[
\boxed{
L
=
-\sum_t
\log
P(x_t|x_1,\ldots,x_{t-1})
}
\]

In words:

> Given all tokens before position \(t\), make the correct token at position \(t\) as probable as possible.

---

# 33. Training vs Inference — Complete Comparison

| Stage | Training | Inference |
|---|---|---|
| Input | Ground-truth token sequence | Prompt + generated tokens |
| Attention | Causal | Causal |
| Predictions | Many positions in parallel | One new token at a time |
| Previous tokens | Ground truth | Model-generated history |
| Loss | Yes | No parameter-training loss |
| Backpropagation | Yes | No |
| Parameter update | Yes | No |
| KV cache | Usually not required in standard training | Very important |
| Goal | Learn parameters | Generate text |

---

# 34. Why Training Is Parallel but Inference Is Sequential

This is one of the most important LLM concepts.

### Training

```text
"I love machine learning"

Position 1 → predict "love"
Position 2 → predict "machine"
Position 3 → predict "learning"

All three predictions
       ↓
computed in parallel
```

### Inference

```text
"I"
 ↓
"love"
 ↓
"machine"
 ↓
"learning"
```

The generated token is unknown until the previous prediction has been made.

Therefore autoregressive generation is sequential.

---

# 35. Connection to KV Cache

During inference:

```text
Prompt
  ↓
Prefill
  ↓
KV Cache
  ↓
Generate token
  ↓
Add new K/V
  ↓
Generate next token
  ↓
Repeat
```

During training, the full sequence is available, so the model can calculate the causal attention computation for many positions in parallel.

This is why:

\[
\boxed{
Training\ parallelism
\neq
Inference\ autoregressive\ generation
}
\]

---

# 36. Self-Check Questions

Before moving to Lesson 10, make sure you can answer:

1. What is an LLM training objective?
2. What is next-token prediction?
3. What is causal language modeling?
4. Why is causal masking necessary?
5. Why can decoder-only LLM training process many positions in parallel?
6. Why is autoregressive inference sequential?
7. What is the LM head?
8. What are logits?
9. Why do we use softmax?
10. What is cross-entropy loss?
11. Why does low probability for the correct token produce high loss?
12. What is teacher forcing?
13. What is the difference between training and inference?
14. What is the mathematical autoregressive factorization?
15. What is the difference between causal LM and masked LM?
16. Why is BERT-style MLM different from GPT-style next-token prediction?
17. What is backpropagation doing during LLM training?
18. Which parameters get updated?
19. What is pretraining?
20. What is the difference between pretraining and fine-tuning?
21. Why cannot a causal model see future tokens during training?
22. What is the relationship between input tokens and shifted labels?
23. Why is KV cache especially important during autoregressive inference?

---

# 37. Next Lesson — Logits, Softmax & Temperature

Next we go deeper into what happens after the Transformer produces its hidden states:

```text
Hidden State
     ↓
LM Head
     ↓
Logits
     ↓
Temperature
     ↓
Softmax
     ↓
Probability Distribution
     ↓
Sampling / Selection
     ↓
Next Token
```

We will study:

```text
1. What logits really represent
2. Why logits can be negative
3. Softmax mathematically
4. Numerical softmax example
5. Log probabilities
6. Temperature
7. Temperature < 1
8. Temperature = 1
9. Temperature > 1
10. Why temperature changes randomness
11. Relationship between logits and probabilities
12. Why sampling happens after logits/probabilities
```

The key transition is:

\[
\boxed{
\text{Transformer hidden state}
\rightarrow
\text{logits}
\rightarrow
\text{probabilities}
\rightarrow
\text{next token}
}
\]
