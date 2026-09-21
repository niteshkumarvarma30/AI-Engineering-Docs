# Lesson 04 — Numerical Examples: Embeddings, Positional Information, Self-Attention & Contextual Representations

## Purpose

This lesson builds numerical intuition for the transformation:

```text
Token
  ↓
Token ID
  ↓
Token Embedding
  ↓
Positional Information
  ↓
Q, K, V
  ↓
Self-Attention
  ↓
Contextual Representation
```

The numbers are intentionally tiny and simplified so that every calculation can be followed by hand. Real LLMs use much larger dimensions, learned matrices, multiple attention heads, normalization, residual connections, feed-forward networks, and many Transformer layers.

---

# 1. Start With a Sentence

Consider:

```text
I love cats
```

Suppose tokenization produces:

```text
["I", "love", "cats"]
```

For this toy example, assign token IDs:

```text
I     → 10
love  → 20
cats  → 30
```

Therefore:

\[
input\_ids=[10,20,30]
\]

---

# 2. Token IDs Are Just Indices

Suppose:

```text
cats → 30
```

The number `30` does **not** mean:

```text
30 units of meaning
```

It simply tells the model which vocabulary/embedding entry to retrieve.

If `E` is the embedding matrix:

\[
E[30]
\]

means:

> Retrieve the embedding vector stored at index 30.

A token ID is therefore an **index**, not a semantic vector.

---

# 3. Create a Tiny Embedding Matrix

Suppose:

```text
Vocabulary size V = 100
Embedding dimension d = 4
```

Then the embedding matrix has shape:

\[
E\in\mathbb{R}^{100\times4}
\]

We only need three rows for this example:

| Token | Token ID | Embedding |
|---|---:|---|
| I | 10 | `[1, 0, 0, 1]` |
| love | 20 | `[0, 1, 1, 0]` |
| cats | 30 | `[1, 1, 0, 0]` |

---

# 4. Embedding Lookup

The input IDs are:

```text
[10, 20, 30]
```

Look up each row:

\[
E[10]=[1,0,0,1]
\]

\[
E[20]=[0,1,1,0]
\]

\[
E[30]=[1,1,0,0]
\]

Therefore:

\[
X=
\begin{bmatrix}
1&0&0&1\\
0&1&1&0\\
1&1&0&0
\end{bmatrix}
\]

The shape is:

\[
[3,4]
\]

Meaning:

```text
3 tokens
×
4 features per token
```

---

# 5. What Is a Token Embedding?

For `cats`:

```text
Token ID:
30

Initial token embedding:
[1, 1, 0, 0]
```

The embedding is a learned vector associated with that token ID.

It is the model's **initial numerical representation** of the token before the Transformer layers contextualize it.

The basic operation is:

```text
Token ID
   ↓
Embedding lookup
   ↓
Token embedding
```

---

# 6. Why Do We Need Positional Information?

Consider:

```text
dog bites man
```

and:

```text
man bites dog
```

The same words occur, but their order is different.

A Transformer needs some mechanism that lets it distinguish token positions.

In this toy example, suppose we use learned positional vectors:

```text
Position 0 → [0.1, 0.1, 0.1, 0.1]
Position 1 → [0.2, 0.2, 0.2, 0.2]
Position 2 → [0.3, 0.3, 0.3, 0.3]
```

---

# 7. Add Token Embedding + Position

For an architecture that uses additive positional embeddings:

\[
X_i=E_i+P_i
\]

Here:

- \(E_i\) = token embedding
- \(P_i\) = positional embedding
- \(X_i\) = Transformer input representation at position \(i\)

## Position 0: `I`

Token embedding:

\[
[1,0,0,1]
\]

Position vector:

\[
[0.1,0.1,0.1,0.1]
\]

Add them:

\[
[1,0,0,1]+[0.1,0.1,0.1,0.1]
\]

\[
\boxed{[1.1,0.1,0.1,1.1]}
\]

---

## Position 1: `love`

Token embedding:

\[
[0,1,1,0]
\]

Position vector:

\[
[0.2,0.2,0.2,0.2]
\]

Add:

\[
[0,1,1,0]+[0.2,0.2,0.2,0.2]
\]

\[
\boxed{[0.2,1.2,1.2,0.2]}
\]

---

## Position 2: `cats`

Token embedding:

\[
[1,1,0,0]
\]

Position vector:

\[
[0.3,0.3,0.3,0.3]
\]

Add:

\[
[1,1,0,0]+[0.3,0.3,0.3,0.3]
\]

\[
\boxed{[1.3,1.3,0.3,0.3]}
\]

---

# 8. Transformer Input

The resulting input matrix is:

\[
X=
\begin{bmatrix}
1.1&0.1&0.1&1.1\\
0.2&1.2&1.2&0.2\\
1.3&1.3&0.3&0.3
\end{bmatrix}
\]

Shape:

\[
[3,4]
\]

This means:

```text
3 token positions
×
4-dimensional representation
```

In this simplified example, this is the representation entering the Transformer.

---

# 9. Important Modern LLM Note: Additive Position Is Only One Approach

The calculation:

\[
X=E_{token}+E_{position}
\]

is useful for learning because it is easy to calculate.

However, modern LLMs do not all use additive positional embeddings.

Many modern decoder-only LLM architectures use **RoPE (Rotary Position Embeddings)**.

RoPE does not simply add a learned position vector to the token embedding.

Conceptually:

```text
Token representations
       ↓
Q/K projections
       ↓
Position-dependent rotation
       ↓
Attention
```

RoPE will be studied separately in the positional encoding lesson.

---

# 10. Self-Attention

Self-attention allows each token position to interact with other token positions.

For:

```text
I love cats
```

the `love` position can gather information from:

```text
I
love
cats
```

Conceptually:

```text
I     ← information from I, love, cats
love  ← information from I, love, cats
cats  ← information from I, love, cats
```

The amount of information gathered from each position is controlled by attention weights.

---

# 11. Q, K, and V

Given the input matrix \(X\), the model creates:

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
- \(W_Q,W_K,W_V\) = learned projection matrices

The standard scaled dot-product attention formula is:

\[
\boxed{
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
}
\]

---

# 12. What Do Q, K, and V Mean?

A useful intuition is:

### Query

> What information am I looking for?

### Key

> What information do I provide for matching?

### Value

> What information should be passed forward if I am attended to?

These are only intuitions. In the actual Transformer, Q, K, and V are learned linear projections of the hidden states.

---

# 13. Simple Attention Score Example

Suppose that for the query corresponding to `love`, the scaled attention scores are:

```text
          I     love    cats
love →   1.0    2.0     0.5
```

These are assumed values for the simplified example.

In a real Transformer they come from:

\[
\frac{QK^T}{\sqrt{d_k}}
\]

The larger the score, the stronger the preference for that key before softmax.

---

# 14. Apply Softmax

Softmax is:

\[
softmax(z_i)
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
\]

For:

```text
[1.0, 2.0, 0.5]
```

the approximate softmax output is:

```text
[0.231, 0.629, 0.140]
```

Check that the values sum to approximately 1:

\[
0.231+0.629+0.140=1.000
\]

Therefore the `love` position assigns approximately:

```text
23.1% → I
62.9% → love
14.0% → cats
```

---

# 15. Attention Weights

Write these weights as:

\[
A_{love}=[0.231,0.629,0.140]
\]

These are **not** the final representation.

They tell us how strongly the `love` position should use the corresponding Value vectors.

---

# 16. Create Simple Value Vectors

For illustration, suppose:

\[
V_I=[1,0,0,0]
\]

\[
V_{love}=[0,1,0,0]
\]

\[
V_{cats}=[0,0,1,0]
\]

In a real model, the Value vectors are produced by:

\[
V=XW_V
\]

and will generally contain many dimensions.

---

# 17. Weighted Value Combination

The attention output for `love` is:

\[
0.231V_I
+
0.629V_{love}
+
0.140V_{cats}
\]

Calculate each contribution.

### Contribution from `I`

\[
0.231[1,0,0,0]
=
[0.231,0,0,0]
\]

### Contribution from `love`

\[
0.629[0,1,0,0]
=
[0,0.629,0,0]
\]

### Contribution from `cats`

\[
0.140[0,0,1,0]
=
[0,0,0.140,0]
\]

Add them:

\[
\boxed{
[0.231,0.629,0.140,0]
}
\]

This is the simplified attention output for the `love` query.

---

# 18. What Did Attention Do?

Before attention, the `love` position had its own representation.

During attention:

```text
love
  ↓
Compare its Query with Keys
  ↓
Obtain attention scores
  ↓
Apply softmax
  ↓
Obtain attention weights
  ↓
Use the weights to combine Values
  ↓
Create an updated representation
```

This is the beginning of **contextualization**.

---

# 19. What Is a Contextual Representation?

A contextual representation is a hidden representation whose value depends on the surrounding tokens.

Consider:

```text
I sat near the bank
```

versus:

```text
I deposited money in the bank
```

The token `bank` is the same token in both sentences.

Therefore, its initial token embedding is the same.

But the surrounding context is different.

After Transformer processing, the hidden representation associated with `bank` can become different in the two sentences.

---

# 20. Initial Embedding of `bank`

Suppose:

```text
bank → token ID 500
```

and:

\[
E_{bank}
=
[0.4,0.7,0.2,0.9]
\]

For both sentences, the initial embedding lookup gives:

\[
[0.4,0.7,0.2,0.9]
\]

So:

```text
Sentence A:
I sat near the bank
                ↓
        same initial embedding

Sentence B:
I deposited money in the bank
                         ↓
                 same initial embedding
```

---

# 21. Context A — River Meaning

Sentence:

```text
I sat near the bank
```

Words such as:

```text
sat
near
```

provide contextual information.

Suppose, purely for illustration, that after Transformer processing the `bank` hidden state becomes:

\[
\boxed{
H_{bank}^{river}
=
[0.8,0.6,0.1,0.3]
}
\]

These numbers are hypothetical and are used only to demonstrate the concept.

---

# 22. Context B — Financial Meaning

Sentence:

```text
I deposited money in the bank
```

Words such as:

```text
deposited
money
```

provide a different context.

Suppose the resulting hidden state becomes:

\[
\boxed{
H_{bank}^{finance}
=
[0.2,0.3,0.9,0.8]
}
\]

Again, these values are illustrative.

---

# 23. Compare the Two

Initial embedding:

\[
\boxed{
E_{bank}
=
[0.4,0.7,0.2,0.9]
}
\]

After river-related context:

\[
\boxed{
[0.8,0.6,0.1,0.3]
}
\]

After financial context:

\[
\boxed{
[0.2,0.3,0.9,0.8]
}
\]

The conceptual flow is:

```text
                    bank
                     │
             Same token ID
                     │
          Same initial embedding
                     │
               Transformer
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   River-related context   Financial context
          ▼                     ▼
 [0.8,0.6,0.1,0.3]      [0.2,0.3,0.9,0.8]
```

This is the core idea of a contextual representation.

---

# 24. Why Does Context Change the Representation?

Self-attention allows a token position to interact with other positions.

For example, in:

```text
I deposited money in the bank
```

suppose the `bank` position assigns these illustrative attention weights:

```text
deposited → 0.5
money     → 0.3
bank      → 0.2
```

They sum to:

\[
0.5+0.3+0.2=1
\]

The exact values in a real model are learned from the model's parameters and depend on the input.

---

# 25. Numerical Context Gathering

Suppose:

\[
V_{deposited}=[1,0,0,0]
\]

\[
V_{money}=[0,1,0,0]
\]

\[
V_{bank}=[0,0,1,0]
\]

Then:

\[
0.5V_{deposited}
+
0.3V_{money}
+
0.2V_{bank}
\]

Calculate:

\[
0.5[1,0,0,0]
=
[0.5,0,0,0]
\]

\[
0.3[0,1,0,0]
=
[0,0.3,0,0]
\]

\[
0.2[0,0,1,0]
=
[0,0,0.2,0]
\]

Add them:

\[
\boxed{
[0.5,0.3,0.2,0]
}
\]

This demonstrates how attention can gather information from surrounding positions.

The actual Transformer performs additional learned transformations and combines the attention output with residual connections and feed-forward processing.

---

# 26. Important: Raw Attention Output Is Not Automatically the Final Contextual Embedding

The simplified calculation:

\[
H=AV
\]

demonstrates the core weighted-value operation.

A real Transformer block contains additional operations.

A simplified modern block can be viewed as:

```text
Input
  ↓
Normalization
  ↓
Q/K/V projections
  ↓
Self-Attention
  ↓
Output projection
  ↓
Residual connection
  ↓
Normalization
  ↓
Feed-Forward Network
  ↓
Residual connection
  ↓
Output
```

The exact ordering and normalization strategy depend on the architecture.

Therefore:

> The raw weighted sum of Values is not necessarily the final contextual hidden state.

---

# 27. Contextualization Happens Layer by Layer

Suppose a Transformer has several layers:

```text
Token Embedding
      ↓
Position Information
      ↓
Transformer Layer 1
      ↓
Hidden State 1
      ↓
Transformer Layer 2
      ↓
Hidden State 2
      ↓
Transformer Layer 3
      ↓
Hidden State 3
      ↓
...
      ↓
Transformer Layer N
      ↓
Final Hidden State
```

The representations are repeatedly transformed and contextualized.

---

# 28. Full Numerical Flow for `I love cats`

### Step 1 — Token IDs

```text
[10, 20, 30]
```

### Step 2 — Token embeddings

```text
I     → [1,0,0,1]
love  → [0,1,1,0]
cats  → [1,1,0,0]
```

### Step 3 — Position vectors

```text
0 → [0.1,0.1,0.1,0.1]
1 → [0.2,0.2,0.2,0.2]
2 → [0.3,0.3,0.3,0.3]
```

### Step 4 — Add token and position information

```text
I     → [1.1,0.1,0.1,1.1]
love  → [0.2,1.2,1.2,0.2]
cats  → [1.3,1.3,0.3,0.3]
```

### Step 5 — Q, K, V

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

### Step 6 — Attention scores

\[
S=\frac{QK^T}{\sqrt{d_k}}
\]

### Step 7 — Softmax

\[
A=softmax(S)
\]

### Step 8 — Weighted Values

\[
H_{attention}=AV
\]

### Step 9 — Remaining Transformer operations

The result then passes through output projection, residual connections, normalization, feed-forward processing, and subsequent Transformer layers.

---

# 29. A Useful Mental Model

Think of these concepts as three different questions.

## Token Embedding

```text
"What token is this?"
```

Example:

```text
bank
  ↓
[0.4,0.7,0.2,0.9]
```

## Position

```text
"Where is this token?"
```

Example:

```text
bank at position 4
```

## Contextual Representation

```text
"What does this token represent
given the surrounding tokens?"
```

Example:

```text
bank + river-related context
          ↓
[0.8,0.6,0.1,0.3]
```

versus:

```text
bank + financial context
          ↓
[0.2,0.3,0.9,0.8]
```

---

# 30. Token Embedding vs Contextual Representation

| Property | Token Embedding | Contextual Representation |
|---|---|---|
| Obtained from | Embedding lookup | Transformer processing |
| Depends on surrounding context? | No, at initial lookup | Yes |
| Same token in different sentences | Same initial vector | Can become different |
| Main mechanism | Embedding matrix | Attention + FFN + Transformer layers |
| Example | `bank → E[bank]` | `bank + context → H_bank` |

---

# 31. Token Embedding vs Sentence Embedding

Do not confuse:

```text
Token embedding
```

with:

```text
Sentence embedding
```

A token embedding represents one token:

```text
cat
 ↓
one token vector
```

A sentence embedding represents an entire text as one vector:

```text
"I love cats"
 ↓
one vector for the entire text
```

Sentence embeddings are commonly used for:

- Semantic search
- Retrieval
- RAG
- Clustering
- Similarity comparison

A sentence embedding is not simply the same thing as the raw token embedding matrix.

---

# 32. Tensor Shapes

Suppose:

```text
Batch size B = 8
Sequence length T = 128
Hidden/embedding dimension d = 4096
```

Then the input hidden-state tensor has shape:

\[
X\in\mathbb{R}^{8\times128\times4096}
\]

Meaning:

```text
8    → sequences in the batch
128  → token positions per sequence
4096 → features per token
```

For self-attention, Q, K, and V preserve the batch/sequence structure while projecting the hidden dimension into the attention representation.

In multi-head attention, the hidden dimension is also divided across multiple heads.

---

# 33. Embedding Matrix Parameter Count

Suppose:

\[
V=50,000
\]

and:

\[
d=4096
\]

Then:

\[
E\in\mathbb{R}^{50000\times4096}
\]

The number of embedding parameters is:

\[
50,000\times4096
=
204,800,000
\]

So the embedding matrix contains approximately:

\[
\boxed{205\text{ million parameters}}
\]

This is why the vocabulary size and hidden dimension can contribute significantly to model size.

---

# 34. Why Is This Numerical Example Simplified?

Real LLMs differ from this toy example in many ways:

- Embedding dimensions are much larger.
- \(W_Q,W_K,W_V\) are learned matrices.
- Multi-head attention is used.
- Decoder-only LLMs use causal masking during generation.
- Normalization is used.
- Residual connections are used.
- Feed-forward networks are used.
- Modern architectures may use RoPE rather than additive positional embeddings.
- There may be dozens or hundreds of Transformer layers.
- The hidden states are not generally interpreted as individual human-readable features.

The toy calculations are intended to expose the mechanics, not reproduce a production LLM numerically.

---

# 35. Complete Mental Diagram

```text
                       "I love cats"
                              │
                              ▼
                         TOKENIZER
                              │
                              ▼
                       TOKEN IDs
                       [10,20,30]
                              │
                              ▼
                     EMBEDDING LOOKUP
                              │
                              ▼
                      TOKEN EMBEDDINGS
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
             Token information   Position information
                    │                   │
                    └─────────┬─────────┘
                              ▼
                     TRANSFORMER INPUT
                              │
                              ▼
                       Q, K, V PROJECTIONS
                              │
                              ▼
                        SELF-ATTENTION
                              │
                              ▼
                       CONTEXT MIXING
                              │
                              ▼
                  CONTEXTUAL REPRESENTATION
                              │
                              ▼
                     FEED-FORWARD NETWORK
                              │
                              ▼
                    NEXT TRANSFORMER LAYER
                              │
                             ...
                              │
                              ▼
                     FINAL HIDDEN STATES
                              │
                              ▼
                           LM HEAD
                              │
                              ▼
                           LOGITS
                              │
                              ▼
                     NEXT-TOKEN PREDICTION
```

---

# 36. Key Formulas

### Embedding Matrix

\[
\boxed{
E\in\mathbb{R}^{V\times d}
}
\]

### Embedding Lookup

\[
\boxed{
e_i=E[i]
}
\]

### Additive Positional Embedding

\[
\boxed{
X_i=E_i+P_i
}
\]

### Query

\[
\boxed{
Q=XW_Q
}
\]

### Key

\[
\boxed{
K=XW_K
}
\]

### Value

\[
\boxed{
V=XW_V
}
\]

### Scaled Dot-Product Attention

\[
\boxed{
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
}
\]

### Contextual Representation

Conceptually:

\[
\boxed{
Contextual\ representation
=
Transformer(Input\ representation,\ Context)
}
\]

---

# 37. Most Important Takeaway

The numerical flow can be summarized as:

```text
Token ID
   ↓
Embedding lookup
   ↓
Initial token vector
   ↓
Position information
   ↓
Q/K/V projections
   ↓
Attention scores
   ↓
Softmax
   ↓
Attention weights
   ↓
Weighted Values
   ↓
Context mixing
   ↓
Transformer layers
   ↓
Contextual hidden state
```

The central example is:

```text
Same token:

bank
 ↓
[0.4,0.7,0.2,0.9]

              ↓ Transformer processing

River-related context:
[0.8,0.6,0.1,0.3]

Financial context:
[0.2,0.3,0.9,0.8]
```

Therefore:

\[
\boxed{
Same\ initial\ token\ embedding
\rightarrow
Different\ contextual\ representations
}
\]

because the surrounding context is different.

---

# 38. Self-Check Questions

1. What is the difference between a token and a token ID?
2. Why is a token ID not a semantic vector?
3. What is an embedding matrix?
4. How does embedding lookup work?
5. What does the embedding dimension mean?
6. Why do we need positional information?
7. How does additive positional embedding work numerically?
8. Why is additive positional embedding not universal in modern LLMs?
9. What is RoPE conceptually?
10. What are Q, K, and V?
11. What does \(QK^T\) calculate?
12. Why do we divide by \(\sqrt{d_k}\)?
13. What does softmax do?
14. What are attention weights?
15. How are Values combined using attention weights?
16. What is a contextual representation?
17. Why can the same token have different contextual representations?
18. Explain the `bank` example numerically.
19. What happens to representations across Transformer layers?
20. Why is the raw attention output not necessarily the final contextual representation?
21. What is the difference between token embedding and sentence embedding?
22. What is the shape of a batched Transformer input?
23. How do embedding parameters contribute to model size?

---

# 39. Next Lesson

## Lesson 05 — LLM Architecture Internals

Next we combine these concepts into the architecture of a modern decoder-only LLM:

```text
Token IDs
   ↓
Token Embedding
   ↓
RoPE
   ↓
RMSNorm
   ↓
Causal Self-Attention
   ↓
Residual Connection
   ↓
RMSNorm
   ↓
SwiGLU / FFN
   ↓
Residual Connection
   ↓
Repeat N times
   ↓
Final RMSNorm
   ↓
LM Head
   ↓
Logits
   ↓
Next-token prediction
```

We will also learn **KV Cache**, which explains why inference is computationally different from training.
