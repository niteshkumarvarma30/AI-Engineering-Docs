# Lesson 05 — LLM Architecture Internals

## Complete Modern Decoder-Only Transformer Architecture

This lesson connects the previous concepts:

```text
Tokenization
     ↓
Token IDs
     ↓
Token Embeddings
     ↓
Positional Information / RoPE
     ↓
Transformer Blocks
     ↓
Final Hidden States
     ↓
LM Head
     ↓
Logits
     ↓
Softmax
     ↓
Next Token
```

The goal is to understand what happens to text inside a modern decoder-only LLM and how **training** differs from **inference**.

---

# Table of Contents

1. [Big Picture](#1-big-picture)
2. [Decoder-Only Transformer](#2-decoder-only-transformer)
3. [Why Is It Called Decoder-Only?](#3-why-is-it-called-decoder-only)
4. [Input Example](#4-input-example)
5. [Token Embedding](#5-token-embedding)
6. [Position Information](#6-position-information)
7. [Transformer Block](#7-transformer-block)
8. [Why So Many Layers?](#8-why-so-many-layers)
9. [RMSNorm](#9-rmsnorm)
10. [Why Normalize?](#10-why-normalize)
11. [Causal Self-Attention](#11-causal-self-attention)
12. [Why Causal Masking?](#12-why-causal-masking)
13. [Q K V](#13-q-k-v)
14. [Attention Calculation](#14-attention-calculation)
15. [Multi-Head Attention](#15-multi-head-attention)
16. [Why Multiple Heads?](#16-why-multiple-heads)
17. [Feed-Forward Network](#17-feed-forward-network)
18. [Why Do We Need the FFN?](#18-why-do-we-need-the-ffn)
19. [Residual Connections](#19-residual-connections)
20. [One Complete Transformer Block](#20-one-complete-transformer-block)
21. [What Happens After the Last Layer?](#21-what-happens-after-the-last-layer)
22. [LM Head](#22-lm-head)
23. [Logits](#23-logits)
24. [Softmax](#24-softmax)
25. [Next-Token Prediction](#25-next-token-prediction)
26. [Training vs Inference](#26-training-vs-inference)
27. [Training Parallelism](#27-training-parallelism)
28. [Cross-Entropy Loss](#28-cross-entropy-loss)
29. [Backpropagation](#29-backpropagation)
30. [Inference Is Different](#30-inference-is-different)
31. [Why Inference Is Sequential](#31-why-inference-is-sequential)
32. [KV Cache](#32-kv-cache)
33. [Why KV Cache Matters](#33-why-kv-cache-matters)
34. [Complete Modern LLM Flow](#34-complete-modern-llm-flow)
35. [Architecture vs Training vs Inference](#35-architecture-vs-training-vs-inference)
36. [One-Sentence Mental Model](#36-one-sentence-mental-model)
37. [What You Must Know](#37-what-you-must-know)
38. [Self-Check Questions](#38-self-check-questions)
39. [Next Lesson](#39-next-lesson)

---

# 1. Big Picture

A simplified modern decoder-only LLM looks like:

```text
                    INPUT TEXT
                        │
                        ▼
                    Tokenizer
                        │
                        ▼
                    Token IDs
                        │
                        ▼
                 Token Embeddings
                        │
                        ▼
                  Position / RoPE
                        │
                        ▼
              ┌─────────────────────┐
              │   Transformer Block │
              │                     │
              │   RMSNorm           │
              │      ↓              │
              │   Causal            │
              │   Self-Attention    │
              │      ↓              │
              │   Residual          │
              │      ↓              │
              │   RMSNorm           │
              │      ↓              │
              │   FFN / SwiGLU       │
              │      ↓              │
              │   Residual          │
              └─────────────────────┘
                        │
                        ▼
                  Repeat N times
                        │
                        ▼
                    Final RMSNorm
                        │
                        ▼
                      LM Head
                        │
                        ▼
                     Logits
                        │
                        ▼
                    Softmax
                        │
                        ▼
                  Next Token
```

This is the high-level architecture you should understand before studying advanced inference optimization.

---

# 2. Decoder-Only Transformer

There are three broad Transformer architecture families:

```text
Encoder-only
Decoder-only
Encoder-Decoder
```

Examples:

```text
BERT       → Encoder-only
GPT-style  → Decoder-only
T5         → Encoder-Decoder
```

Modern generative LLMs are predominantly **decoder-only Transformers**.

---

# 3. Why Is It Called Decoder-Only?

A decoder-only LLM predicts the next token using previous tokens.

Example:

```text
The cat is
```

The model predicts:

```text
sleeping
```

Then the sequence becomes:

```text
The cat is sleeping
```

The model predicts another token.

The fundamental probability is:

\[
\boxed{
P(x_t\mid x_1,x_2,\ldots,x_{t-1})
}
\]

This is **causal language modeling**.

---

# 4. Input Example

Suppose:

```text
"I love cats"
```

Tokenization:

```text
["I", "love", "cats"]
```

Token IDs:

```text
[10, 20, 30]
```

The model does not directly perform neural computation on these integers.

They first go through an embedding layer.

---

# 5. Token Embedding

The embedding matrix is:

\[
E\in\mathbb{R}^{V\times d}
\]

where:

- \(V\) = vocabulary size
- \(d\) = hidden dimension

For each token:

\[
x_i=E[token_i]
\]

So:

```text
10 → embedding vector
20 → embedding vector
30 → embedding vector
```

The resulting sequence has shape:

\[
X\in\mathbb{R}^{T\times d}
\]

where:

- \(T\) = sequence length
- \(d\) = hidden dimension

---

# 6. Position Information

A Transformer needs information about token order.

Modern LLMs commonly use **RoPE — Rotary Position Embeddings**.

A simplified older approach could be:

```text
Token embedding
      +
Position embedding
      ↓
Transformer input
```

RoPE is different. It applies position-dependent rotations to query/key representations used by attention.

Conceptually:

```text
Token representations
        ↓
Q/K projections
        ↓
RoPE
        ↓
Attention
```

RoPE will be studied in detail in the positional encoding lesson.

---

# 7. Transformer Block

A modern Transformer block can be represented approximately as:

```text
Input
  │
  ▼
RMSNorm
  │
  ▼
Causal Self-Attention
  │
  ▼
Residual Addition
  │
  ▼
RMSNorm
  │
  ▼
Feed-Forward Network
  │
  ▼
Residual Addition
  │
  ▼
Output
```

This block is repeated many times.

The exact ordering varies between architectures, but this is a useful modern pre-norm mental model.

---

# 8. Why So Many Layers?

Suppose an LLM has:

```text
32 Transformer layers
```

Then:

```text
Embedding
   ↓
Layer 1
   ↓
Layer 2
   ↓
Layer 3
   ↓
...
   ↓
Layer 32
   ↓
Final hidden states
```

Each layer progressively transforms the representations.

The deeper layers can build increasingly abstract and task-relevant representations.

---

# 9. RMSNorm

Before attention, many modern LLMs normalize the hidden representation.

A simplified RMSNorm calculation is:

\[
RMS(x)
=
\sqrt{
\frac{1}{d}
\sum_{i=1}^{d}x_i^2
+
\epsilon
}
\]

Then:

\[
\boxed{
RMSNorm(x)
=
\frac{x}{RMS(x)}
\odot\gamma
}
\]

where:

- \(\gamma\) = learned scale
- \(\epsilon\) = small numerical stability constant

---

# 10. Why Normalize?

Without normalization, activations can become poorly scaled as they pass through many layers.

Normalization helps maintain stable numerical behavior.

Modern LLMs commonly use RMSNorm rather than the original LayerNorm formulation.

---

# 11. Causal Self-Attention

This is one of the most important concepts in a decoder-only LLM.

For:

```text
I love cats
```

the model must not allow the representation at an earlier position to see future tokens during causal language modeling.

Attention pattern:

```text
             I    love   cats

I            ✓     ✗      ✗
love         ✓     ✓      ✗
cats         ✓     ✓      ✓
```

This is a **causal mask**.

---

# 12. Why Causal Masking?

During training, we want:

```text
I        → predict love
I love   → predict cats
```

The model must not see the target token from the future.

Therefore:

\[
\boxed{
Token_t\ can\ attend\ only\ to\ positions\leq t
}
\]

In a causal attention score matrix, forbidden future positions are effectively assigned a very large negative value, commonly represented as:

\[
-\infty
\]

Before softmax:

```text
Allowed position → normal score
Future position  → -∞
```

After softmax:

\[
e^{-\infty}=0
\]

Therefore future tokens receive zero attention probability.

---

# 13. Q, K, V

After normalization, the model creates:

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

where:

- \(Q\) = Query
- \(K\) = Key
- \(V\) = Value

A useful intuition:

### Query

> What information am I looking for?

### Key

> How should another token match against me?

### Value

> What information should I provide if I am attended to?

These projections are learned parameters.

---

# 14. Attention Calculation

The basic attention formula is:

\[
\boxed{
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
+
M
\right)V
}
\]

where:

- \(QK^T\) produces attention compatibility scores
- \(d_k\) is the key dimension
- \(M\) is the attention mask

For causal attention, future positions in \(M\) receive a large negative value.

---

# 15. Multi-Head Attention

Modern Transformers normally use multiple attention heads.

Suppose:

```text
Hidden dimension = 4096
Number of heads = 32
```

Then, in a standard equal partition:

\[
d_{head}
=
\frac{4096}{32}
=
128
\]

Conceptually:

```text
                  Hidden State
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        Head 1       Head 2       Head 32
          │            │            │
      Attention    Attention    Attention
          │            │            │
          └────────────┼────────────┘
                       ▼
                   Concatenate
                       │
                       ▼
                Output Projection
```

---

# 16. Why Multiple Heads?

Different heads can learn different attention patterns.

For example, one head may learn strong relationships between:

```text
pronoun ↔ noun
```

Another may focus on:

```text
verb ↔ subject
```

Another may capture:

```text
nearby tokens
```

Another may capture:

```text
long-range relationships
```

These roles are not manually assigned. The model learns useful patterns during training.

---

# 17. Feed-Forward Network

After attention, the representation passes through a feed-forward network.

A simple FFN can be represented as:

\[
FFN(x)
=
W_2\sigma(W_1x+b_1)+b_2
\]

Modern LLMs often use **SwiGLU-style** feed-forward networks instead of a simple ReLU FFN.

Conceptually:

```text
Hidden state
     ↓
Linear projections
     ↓
Activation + gating
     ↓
Projection
     ↓
Output
```

---

# 18. Why Do We Need the FFN?

A useful conceptual distinction is:

```text
Attention:
"Which other tokens should I use information from?"

FFN:
"How should I transform the information I now have?"
```

Attention allows information to move between token positions.

The FFN performs nonlinear transformation of the resulting representation.

---

# 19. Residual Connections

After attention, a residual connection adds the block input back to the transformed output.

Conceptually:

```text
              ┌──────────────────┐
              │                  │
Input ────────┼──────► Attention │
  │           │                  │
  │           └────────┬─────────┘
  │                    │
  └────────── Add ◄────┘
                    │
                    ▼
                 Output
```

Mathematically, in simplified form:

\[
X_1=X+Attention(X)
\]

The same idea is used around the FFN.

Residual connections help information and gradients flow through deep networks.

---

# 20. One Complete Transformer Block

A simplified modern pre-norm block:

```text
                 X
                 │
                 ├─────────────────────┐
                 │                     │
                 ▼                     │
              RMSNorm                 │
                 │                     │
                 ▼                     │
        Causal Self-Attention          │
                 │                     │
                 └──────────► ADD ◄────┘
                              │
                              ▼
                             X'
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         │
              RMSNorm                     │
                 │                         │
                 ▼                         │
             SwiGLU / FFN                 │
                 │                         │
                 └────────────► ADD ◄──────┘
                              │
                              ▼
                             X''
```

This structure is repeated across the model.

---

# 21. What Happens After the Last Layer?

Suppose there are \(N\) Transformer layers.

After layer \(N\):

\[
H\in\mathbb{R}^{T\times d}
\]

These are the final hidden states.

Then:

```text
Final hidden states
       ↓
Final RMSNorm
       ↓
LM Head
```

---

# 22. LM Head

The LM head converts hidden representations into vocabulary scores.

If:

```text
Hidden dimension = d
Vocabulary size = V
```

then conceptually:

\[
W_{LM}\in\mathbb{R}^{d\times V}
\]

and:

\[
\boxed{
Logits=HW_{LM}
}
\]

The output shape is:

\[
[T,V]
\]

Each token position gets one score for every vocabulary token.

---

# 23. What Are Logits?

Suppose the vocabulary contains:

```text
cat
dog
car
tree
...
```

The model might produce:

```text
cat  → 4.2
dog  → 2.1
car  → 0.7
tree → -0.5
```

These are **logits**.

They are raw scores, not probabilities.

---

# 24. Softmax

Softmax converts logits into probabilities:

\[
P_i=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
\]

For example:

```text
cat  → 0.72
dog  → 0.18
car  → 0.07
tree → 0.03
```

The probabilities sum to:

\[
1
\]

---

# 25. Next-Token Prediction

The model then uses a decoding strategy.

Common strategies include:

```text
Greedy
Temperature sampling
Top-k
Top-p
Min-p
```

Example:

```text
"The cat is"
       ↓
Transformer
       ↓
Logits
       ↓
Probabilities
       ↓
"sleeping"
```

The generated token is appended to the sequence.

---

# 26. Training vs Inference

This distinction is extremely important.

## Training

The model receives a sequence containing the target tokens.

Example:

```text
Input:
I love cats
```

Training pairs can be viewed as:

```text
Context          Target

I                love
I love           cats
I love cats      <EOS>
```

The causal mask ensures that each prediction only uses information available at or before that position.

---

# 27. Training Parallelism

Suppose:

```text
Tokens:

[I, love, cats, today]
```

The model can process the complete sequence in one forward pass.

The causal pattern is:

```text
I       → I
love    → I, love
cats    → I, love, cats
today   → I, love, cats, today
```

The model simultaneously produces predictions for all positions.

Therefore Transformer training is highly parallelizable across sequence positions.

This is a major difference from autoregressive generation.

---

# 28. Cross-Entropy Loss

For next-token prediction:

\[
\boxed{
\mathcal{L}
=
-\sum_t
\log P(x_t\mid x_{<t})
}
\]

Usually the loss is averaged over valid prediction positions.

Example:

```text
Input:
I love cats

Targets:
love cats <EOS>
```

The model produces:

```text
Position 1 → probability distribution for love
Position 2 → probability distribution for cats
Position 3 → probability distribution for <EOS>
```

The loss measures how much probability the model assigned to the correct targets.

---

# 29. Backpropagation

After calculating the loss:

```text
Loss
 ↓
Backpropagation
 ↓
Gradients
 ↓
Optimizer
 ↓
Updated parameters
```

Parameters updated during training include:

```text
Token embeddings
Attention projections
FFN weights
Normalization parameters
LM head
```

and other architecture-specific parameters.

---

# 30. Inference Is Different

Suppose the prompt is:

```text
I love
```

The model predicts:

```text
cats
```

Now the sequence becomes:

```text
I love cats
```

Then it predicts another token:

```text
I love cats today
```

Then:

```text
I love cats today <EOS>
```

Generation is therefore **autoregressive**.

---

# 31. Why Inference Is Sequential

During inference, the correct future tokens are unknown.

The model has to generate them one at a time:

```text
Step 1 → generate token 1
Step 2 → generate token 2
Step 3 → generate token 3
...
```

Therefore generation cannot simply process all future positions in parallel in the same way as training.

This sequential dependency is a major source of generation latency.

---

# 32. KV Cache

KV Cache is one of the most important inference concepts.

Without KV caching, when generating a new token, the model would repeatedly recompute the keys and values for previous tokens.

Instead, the model stores them.

```text
Previous tokens
      ↓
Past K and V
      ↓
KV Cache
```

When a new token arrives:

```text
New token
   ↓
New Q, K, V
   ↓
Q attends to cached K
   ↓
Use cached V
```

The exact implementation depends on the model and attention variant, but this is the core idea.

---

# 33. Why KV Cache Matters

Suppose the generated context is:

```text
I love cats
```

The cache conceptually stores:

```text
I       → K,V
love    → K,V
cats    → K,V
```

For the next generated token, the model does not need to recompute the previous tokens' K/V projections from scratch.

It computes the new token's K/V and uses the stored history.

Therefore:

\[
\boxed{
KV\ Cache
=
stored\ past\ Key/Value\ representations
}
\]

---

# 34. Complete Modern LLM Flow

```text
                    TEXT
                     │
                     ▼
                 TOKENIZER
                     │
                     ▼
                  TOKEN IDs
                     │
                     ▼
              TOKEN EMBEDDINGS
                     │
                     ▼
               POSITION / RoPE
                     │
                     ▼
          ┌────────────────────────┐
          │ Transformer Layer 1    │
          │                        │
          │ RMSNorm                │
          │ ↓                      │
          │ Causal Self-Attention  │
          │ ↓                      │
          │ Residual               │
          │ ↓                      │
          │ RMSNorm                │
          │ ↓                      │
          │ SwiGLU / FFN           │
          │ ↓                      │
          │ Residual               │
          └───────────┬────────────┘
                      ▼
                 Layer 2
                      │
                     ...
                      │
                      ▼
                 Layer N
                      │
                      ▼
                 Final RMSNorm
                      │
                      ▼
                    LM Head
                      │
                      ▼
                    Logits
                      │
                      ▼
                   Softmax
                      │
                      ▼
                Next Token
```

---

# 35. Architecture vs Training vs Inference

Keep these three concepts separate.

## Architecture

What components exist?

```text
Embedding
RoPE / positional mechanism
Attention
RMSNorm
FFN
Residual connections
LM Head
```

## Training

How are parameters learned?

```text
Tokens
 ↓
Forward pass
 ↓
Logits
 ↓
Loss
 ↓
Backpropagation
 ↓
Optimizer
 ↓
Updated weights
```

## Inference

How does the trained model generate text?

```text
Prompt
 ↓
Forward pass
 ↓
Logits
 ↓
Sampling / decoding
 ↓
New token
 ↓
KV Cache
 ↓
Repeat
```

---

# 36. One-Sentence Mental Model

A modern decoder-only LLM:

> **Turns token IDs into vectors, repeatedly mixes contextual information using causal self-attention and transforms it with feed-forward networks, then converts the final representations into vocabulary logits to predict the next token.**

---

# 37. What You Must Know

You should be able to explain this complete chain:

```text
Tokenization
      ↓
Token IDs
      ↓
Embedding
      ↓
Position / RoPE
      ↓
RMSNorm
      ↓
Causal Self-Attention
      ↓
Multi-Head Attention
      ↓
Residual
      ↓
RMSNorm
      ↓
SwiGLU / FFN
      ↓
Residual
      ↓
Repeat N times
      ↓
Final RMSNorm
      ↓
LM Head
      ↓
Logits
      ↓
Softmax
      ↓
Next Token
```

And you should understand the major training/inference distinction:

```text
TRAINING

Whole sequence available
        ↓
Causal mask
        ↓
Parallel forward pass
        ↓
Loss
        ↓
Backpropagation
        ↓
Weight update


INFERENCE

Prompt
  ↓
Generate one token
  ↓
KV Cache
  ↓
Generate next token
  ↓
KV Cache
  ↓
Generate next token
  ↓
...
```

---

# 38. Self-Check Questions

Before moving on, make sure you can answer:

1. What is a decoder-only Transformer?
2. Why are modern generative LLMs commonly decoder-only?
3. What is causal language modeling?
4. Why do we need causal masking?
5. What is the difference between causal masking and padding masking?
6. What is RMSNorm?
7. Why is normalization used?
8. What are Q, K, and V?
9. What does \(QK^T\) calculate?
10. Why divide attention scores by \(\sqrt{d_k}\)?
11. What does the causal mask do before softmax?
12. What is multi-head attention?
13. Why use multiple heads?
14. What is the role of the FFN?
15. What is SwiGLU?
16. Why are residual connections used?
17. What is an LM head?
18. What are logits?
19. Why are logits not probabilities?
20. What does softmax do?
21. What is next-token prediction?
22. Why can Transformer training process many positions in parallel?
23. Why is autoregressive inference sequential?
24. What is KV Cache?
25. Why does KV Cache make inference faster?
26. What parameters are updated during training?
27. What is the difference between architecture, training, and inference?

---

# 39. Next Lesson

## Lesson 06 — Positional Encodings

The next lesson focuses deeply on **how Transformers represent position**.

We will study:

```text
1. Why Transformers need position
2. Absolute positional encoding
3. Learned positional embeddings
4. Sinusoidal positional encoding
5. Relative position
6. Rotary Position Embeddings (RoPE)
7. RoPE numerical example
8. Why RoPE rotates Q and K
9. RoPE frequency calculation
10. Long-context behavior
11. Context extension
12. RoPE vs sinusoidal vs learned positions
```

The most important modern concept will be:

\[
\boxed{
RoPE
}
\]

because it is widely used in modern decoder-only LLM architectures.
