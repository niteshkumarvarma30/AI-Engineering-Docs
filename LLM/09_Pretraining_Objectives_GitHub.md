# Lesson 09 - Pre-training Objectives

## Complete LLM Training Note

> GitHub-safe version: formulas use plain ASCII notation. No LaTeX syntax or special mathematical symbols are used.

---

## 1. Core Question

How does a Transformer actually learn to become a language model?

For modern decoder-only LLMs, the central idea is:

```text
Next-token prediction
```

The model repeatedly receives previous tokens and learns to assign high probability to the correct next token.

---

# 2. What Does Training an LLM Mean?

Suppose the training text is:

```text
I love machine learning
```

The model learns next-token relationships:

```text
Input:  I
Target: love
```

```text
Input:  I love
Target: machine
```

```text
Input:  I love machine
Target: learning
```

The fundamental task is:

```text
Given previous tokens, predict the next token.
```

---

# 3. Next-Token Prediction

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

Importantly, the Transformer can calculate all these training predictions in one forward pass because causal masking prevents information from future positions from leaking backward.

---

# 4. Why Can Training Be Parallel?

You might wonder:

> If generation happens one token at a time, why is training not also one token at a time?

The answer is **causal masking**.

For:

```text
I love machine learning
```

the attention pattern is approximately:

```text
          I   love   machine   learning
I         X
love      X    X
machine   X    X       X
learning  X    X       X          X
```

Therefore:

```text
Position 1 -> predicts token 2
Position 2 -> predicts token 3
Position 3 -> predicts token 4
```

All prediction positions can be computed in parallel.

This is called **teacher-forced parallel training**: the complete ground-truth sequence is available during the forward pass, while the causal mask prevents each position from seeing future tokens.

---

# 5. Causal Language Modeling

This objective is called:

```text
Causal Language Modeling (CLM)
```

"Causal" means the prediction at position `t` cannot use future tokens.

The autoregressive factorization is:

```text
P(x1, x2, ..., xT)
=
P(x1)
x
P(x2 | x1)
x
P(x3 | x1, x2)
x
...
x
P(xT | x1, ..., xT-1)
```

More compactly:

```text
P(x1, ..., xT)
=
product over t of P(xt | x1, ..., x(t-1))
```

The model learns:

```text
P(xt | x<t)
```

where `x<t` means all tokens before position `t`.

---

# 6. Decoder-Only LLM Training Flow

A simplified GPT-style training pipeline is:

```text
Training Text
     |
     v
Tokenizer
     |
     v
Token IDs
     |
     v
Embedding
     |
     v
Positional Information / RoPE
     |
     v
Transformer Blocks
     |
     +--> Causal Self-Attention
     |
     +--> FFN / MoE
     |
     +--> Residual Connections + Normalization
     |
     v
Hidden Representations
     |
     v
LM Head
     |
     v
Logits
     |
     v
Cross-Entropy Loss
     |
     v
Backpropagation
     |
     v
Optimizer Update
```

This process is repeated over many training batches.

---

# 7. What Is the LM Head?

The Transformer produces a hidden representation:

```text
h_t
```

for every position.

The model then needs to convert each hidden representation into a score for every vocabulary token.

A simplified output layer is:

```text
z_t = W_o h_t + b
```

where:

```text
z_t = logit vector
V   = vocabulary size
z_t has V values
```

For example:

```text
Vocabulary = 50,000 tokens

Hidden state
     |
     v
LM Head
     |
     v
50,000 logits
```

The LM head maps the model's internal representation to the vocabulary space.

---

# 8. What Are Logits?

Suppose the vocabulary contains:

```text
["cat", "dog", "car", "food"]
```

The model might produce:

```text
cat  -> 2.1
dog  -> 5.4
car  -> 0.7
food -> 1.2
```

These values are called **logits**.

Logits are not probabilities.

A larger logit generally means the model currently favors that token more strongly.

Important:

```text
logits can be positive
logits can be negative
logits do not need to sum to 1
```

Softmax later converts them into a probability distribution.

---

# 9. Softmax

Softmax converts logits into probabilities.

For logit `z_i`:

```text
P_i = exp(z_i) / sum_j exp(z_j)
```

For example:

```text
Logits:
cat  -> 2.1
dog  -> 5.4
car  -> 0.7
food -> 1.2
```

After softmax, we obtain probabilities such as:

```text
cat  -> 0.08
dog  -> 0.72
car  -> 0.04
food -> 0.16
```

The probabilities sum to approximately:

```text
1.0
```

The exact values depend on the logits.

---

# 10. Correct Token and Loss

Suppose the correct next token is:

```text
dog
```

and the model predicts:

```text
cat  -> 0.08
dog  -> 0.72
car  -> 0.04
food -> 0.16
```

Then:

```text
P(correct) = 0.72
```

The single-token cross-entropy loss is:

```text
L = -log(P(correct))
```

Therefore:

```text
L = -log(0.72)
  ~= 0.329
```

The model is rewarded for assigning high probability to the correct token.

---

# 11. Numerical Loss Example

Suppose:

```text
Correct token = dog
```

Prediction:

```text
cat  -> 0.10
dog  -> 0.70
car  -> 0.05
food -> 0.15
```

Then:

```text
L = -log(0.70)
  ~= 0.357
```

If the model improves:

```text
cat  -> 0.05
dog  -> 0.90
car  -> 0.02
food -> 0.03
```

then:

```text
L = -log(0.90)
  ~= 0.105
```

So:

```text
Higher probability for correct token
              |
              v
          Lower loss
```

---

# 12. Why Cross-Entropy?

Cross-entropy strongly penalizes the model when it assigns very low probability to the correct token.

For example:

```text
P(correct) = 0.9

L = -log(0.9)
  ~= 0.105
```

while:

```text
P(correct) = 0.1

L = -log(0.1)
  ~= 2.303
```

Therefore:

```text
Correct-token probability increases
              |
              v
            Loss decreases
```

This gives the optimizer a strong training signal.

---

# 13. Complete Training Example

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
The              -> cat
The cat          -> is
The cat is       -> sleeping
```

Conceptually:

```text
                Transformer
                     |
        +------------+------------+
        v            v            v
     Predict       Predict      Predict
       cat           is        sleeping
        |            |            |
        v            v            v
      Loss 1       Loss 2       Loss 3
        +------------+------------+
                     |
                     v
                Total Loss
```

---

# 14. Shifted Inputs and Labels

A typical implementation conceptually uses:

```text
Input:
The      cat       is       sleeping

Label:
cat      is        sleeping <EOS>
```

The labels are shifted by one position.

For a sequence:

```text
[x1, x2, x3, x4]
```

the model predicts:

```text
[x2, x3, x4, ...]
```

The exact handling of the final position depends on the tokenizer and training setup.

In many implementations, the final input position has no next-token target inside the same sequence, so that position is ignored for the loss.

---

# 15. Teacher Forcing

During autoregressive training, the model is given the **ground-truth previous tokens**.

Suppose the correct sequence is:

```text
I love machine learning
```

Training uses the known sequence:

```text
I
I love
I love machine
```

rather than repeatedly feeding the model's own sampled mistakes.

This is commonly described as **teacher forcing**.

It makes training much more efficient and stable.

---

# 16. Training vs Generation

This distinction is extremely important.

## Training

```text
Ground-truth sequence
        |
        v
Causal mask
        |
        v
Many next-token predictions in parallel
        |
        v
Loss
        |
        v
Backpropagation
```

## Inference

```text
Prompt
  |
  v
Predict next token
  |
  v
Append generated token
  |
  v
Predict next token
  |
  v
Append
  |
  v
Repeat
```

Therefore:

```text
Training = highly parallel
Inference = autoregressive and sequential
```

---

# 17. Why Causal Masking Is Necessary

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

Otherwise the training signal would leak the answer.

The rule is:

```text
Current position
      |
      v
Can see previous/current positions
      |
      v
Cannot see future positions
```

---

# 18. Causal Attention Matrix

For four tokens:

```text
          1   2   3   4

1         X   .   .   .
2         X   X   .   .
3         X   X   X   .
4         X   X   X   X
```

A numerical causal mask can be represented as:

```text
M =
[
  [0,   -inf, -inf, -inf],
  [0,    0,   -inf, -inf],
  [0,    0,    0,   -inf],
  [0,    0,    0,    0]
]
```

The mask is added to attention scores before softmax.

Conceptually:

```text
exp(-inf) -> 0
```

Therefore future positions receive zero attention probability.

---

# 19. Causal vs Bidirectional Attention

There are two major language-modeling paradigms.

## Causal

Used by GPT-style decoder-only LLMs.

```text
Token 1 -> sees 1
Token 2 -> sees 1,2
Token 3 -> sees 1,2,3
Token 4 -> sees 1,2,3,4
```

Future tokens are hidden.

## Bidirectional

Used by encoder-style models such as BERT during masked-language-model pretraining.

A token can use information from both left and right context.

Conceptually:

```text
Token 1 <-> Token 2 <-> Token 3 <-> Token 4
```

The exact attention implementation can vary, but the key distinction is that the model is not restricted to left-to-right context in the same way as causal language modeling.

---

# 20. Masked Language Modeling

BERT-style pretraining commonly uses:

```text
Masked Language Modeling (MLM)
```

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

Unlike causal LM, the model can generally use both left and right context around the masked position.

---

# 21. Causal LM vs Masked LM

| Property | Causal LM | Masked LM |
|---|---|---|
| Typical example | GPT-style | BERT-style |
| Objective | Next-token prediction | Predict masked tokens |
| Attention | Causal | Usually bidirectional |
| Future context available? | No | Yes, except target information is masked |
| Natural text generation | Directly suited | Not directly autoregressive |
| Typical architecture | Decoder-only | Encoder-only |

---

# 22. Why GPT Uses Causal LM

GPT-style models are designed for autoregressive generation.

At inference:

```text
Prompt
 |
 v
Predict next token
 |
 v
Append token
 |
 v
Predict next token
 |
 v
Append token
 |
 v
Repeat
```

Therefore the training objective naturally matches generation:

```text
P(next token | previous tokens)
```

---

# 23. What Gets Updated?

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
   |
   v
Loss
   |
   v
Backpropagation
   |
   v
Gradients
   |
   v
Optimizer
   |
   v
Parameter updates
```

For a parameter `theta`:

```text
theta_new
=
theta_old
-
learning_rate * dL/dtheta
```

where:

```text
L = loss
dL/dtheta = gradient of loss with respect to theta
```

The exact optimizer update can be more sophisticated than this simple gradient-descent expression. For example, Adam and AdamW maintain additional optimizer state.

---

# 24. Complete Training Loop

```text
             Training Dataset
                    |
                    v
                Tokenization
                    |
                    v
                 Token IDs
                    |
                    v
                Embeddings
                    |
                    v
             Positional Information
                    |
                    v
             Transformer Blocks
                    |
          +---------+---------+
          |                   |
   Causal Attention       FFN / MoE
          |                   |
          +---------+---------+
                    |
                    v
                 Logits
                    |
                    v
             Cross-Entropy Loss
                    |
                    v
              Backpropagation
                    |
                    v
             Optimizer Update
                    |
                    v
              Updated Model
```

Then another batch is processed.

---

# 25. Batch Training

LLMs normally train on batches rather than one sentence at a time.

Conceptually:

```text
Batch
 +-- Sequence 1
 +-- Sequence 2
 +-- Sequence 3
 +-- Sequence 4
```

A simplified tensor flow is:

```text
Token IDs
[batch_size, sequence_length]
          |
          v
Transformer
          |
          v
Logits
[batch_size, sequence_length, vocabulary_size]
          |
          v
Cross-Entropy
          |
          v
Loss
```

Padding masks, sequence packing, and exact tensor layouts depend on the implementation.

---

# 26. Mathematical Training Objective

For:

```text
x1, x2, ..., xT
```

the autoregressive language-model objective is:

```text
P(x1, ..., xT)
=
product over t of P(xt | x1, ..., x(t-1))
```

The negative log-likelihood loss is:

```text
L
=
-sum over t of log P(xt | x<t)
```

Often the loss is averaged across valid prediction positions:

```text
L
=
-(1/T) * sum over t of log P(xt | x<t)
```

This is the mathematical heart of decoder-only LLM pretraining.

---

# 27. What Does the Model Learn?

Suppose the model repeatedly encounters:

```text
The cat is sleeping.
The cat is sleeping.
The cat is sleeping.
```

It learns statistical relationships that increase the probability of appropriate continuations.

Eventually, for a context such as:

```text
"The cat is"
```

the token:

```text
sleeping
```

may receive a relatively high probability.

The model does not necessarily store an explicit symbolic rule:

```text
"The cat is" -> "sleeping"
```

Instead, these behaviors emerge from learned neural-network representations and parameters.

---

# 28. One Training Step

A single optimization step is:

```text
Training batch
     |
     v
Forward pass
     |
     v
Logits
     |
     v
Cross-entropy loss
     |
     v
Backward pass
     |
     v
Gradients
     |
     v
Optimizer
     |
     v
Parameter update
```

Then the next batch is processed.

---

# 29. Epochs and Steps

Suppose the dataset is divided into:

```text
Batch 1
Batch 2
Batch 3
...
Batch N
```

Processing the complete dataset once is approximately one:

```text
Epoch
```

A training step generally refers to one optimizer update, although terminology can vary when gradient accumulation and distributed training are used.

Large LLM pretraining typically uses enormous datasets and very large numbers of optimization steps.

---

# 30. Pretraining vs Fine-Tuning

Do not confuse these.

## Pretraining

Goal:

```text
Learn broad language and world representations
from large-scale data.
```

For a decoder-only LLM, the core objective is typically:

```text
Next-token prediction
```

## Fine-tuning

Goal:

```text
Adapt an already pretrained model to a particular
behavior, task, domain, or instruction format.
```

Examples:

```text
Instruction following
Question answering
Code
Domain-specific tasks
Tool use
```

---

# 31. What Does "Pretrained" Mean?

A pretrained LLM has already learned its parameters by repeatedly optimizing its training objective.

Conceptually:

```text
Huge text dataset
       |
       v
Tokenization
       |
       v
Next-token prediction
       |
       v
Loss
       |
       v
Backpropagation
       |
       v
Parameter update
       |
       v
Repeat over enormous numbers of tokens
       |
       v
Pretrained LLM
```

---

# 32. Complete Connection With Previous Lessons

You have now connected:

```text
Text
  |
  v
Tokenization
  |
  v
Token IDs
  |
  v
Embeddings
  |
  v
Positional Information / RoPE
  |
  v
Q/K/V
  |
  v
Self-Attention
  |
  v
Causal Mask
  |
  v
MHA / GQA / MQA
  |
  v
FFN / MoE
  |
  v
Hidden States
  |
  v
LM Head
  |
  v
Logits
  |
  v
Softmax
  |
  v
Cross-Entropy
  |
  v
Backpropagation
  |
  v
Parameter Update
```

This is the complete conceptual path from training text to learning.

---

# 33. The Most Important Formula

For a decoder-only LLM:

```text
L
=
-sum over t of log P(xt | x1, ..., x(t-1))
```

In words:

> Given all tokens before position `t`, make the correct token at position `t` as probable as possible.

---

# 34. Training vs Inference - Complete Comparison

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

Note that optimized training systems may use specialized memory-saving techniques, but the conceptual distinction remains: training computes gradients over known sequences, while inference uses the learned parameters without updating them.

---

# 35. Why Training Is Parallel but Inference Is Sequential

This is one of the most important LLM concepts.

## Training

```text
"I love machine learning"

Position 1 -> predict "love"
Position 2 -> predict "machine"
Position 3 -> predict "learning"

All three predictions
       |
       v
computed in parallel
```

## Inference

```text
"I"
 |
 v
"love"
 |
 v
"machine"
 |
 v
"learning"
```

The generated token is unknown until the previous prediction has been made.

Therefore autoregressive generation is sequential.

---

# 36. Connection to KV Cache

During inference:

```text
Prompt
  |
  v
Prefill
  |
  v
KV Cache
  |
  v
Generate token
  |
  v
Add new K/V
  |
  v
Generate next token
  |
  v
Repeat
```

During training, the full sequence is available, so the model can calculate the causal attention computation for many positions in parallel.

This is why:

```text
Training parallelism
is different from
autoregressive inference
```

---

# 37. Self-Check Questions

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

# 38. Next Lesson - Logits, Softmax & Temperature

Next we go deeper into what happens after the Transformer produces its hidden states:

```text
Hidden State
     |
     v
LM Head
     |
     v
Logits
     |
     v
Temperature
     |
     v
Softmax
     |
     v
Probability Distribution
     |
     v
Sampling / Selection
     |
     v
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

```text
Transformer hidden state
        |
        v
      logits
        |
        v
  probabilities
        |
        v
    next token
```

---

# End of Lesson 09
