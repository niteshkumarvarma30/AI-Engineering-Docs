# Lesson 04 — Numerical Examples: Embeddings, Positional Information, Self-Attention & Contextual Representations

## Purpose

This note focuses specifically on the **numerical intuition** behind:

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

The numbers below are deliberately tiny and simplified. Real LLMs use hundreds/thousands of dimensions and learned matrices.

---

# 1. Start With a Sentence

Use:

```text
I love cats
```

Suppose tokenization produces:

```text
["I", "love", "cats"]
```

Assign toy token IDs:

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
cat → 30
```

The number `30` does not mean:

```text
30 units of cat meaning
```

It simply means:

> Look at vocabulary/embedding row 30.

For example:

\[
E[30]
\]

retrieves the embedding for token ID 30.

---

# 3. Create a Tiny Embedding Matrix

Suppose:

```text
Vocabulary size V = 100
Embedding dimension d = 4
```

Then:

\[
E\in\mathbb{R}^{100\times4}
\]

We only need three rows:

```text
Token     ID       Embedding
--------------------------------
I         10       [1, 0, 0, 1]
love      20       [0, 1, 1, 0]
cats      30       [1, 1, 0, 0]
```

---

# 4. Embedding Lookup

Input:

```text
[10,20,30]
```

Lookup:

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

Shape:

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

Embedding:
[1,1,0,0]
```

The embedding is a learned vector.

It is the model's **initial numerical representation** of that token.

Important:

```text
Token ID
   ↓
Embedding lookup
   ↓
Token embedding
```

---

# 6. Why Do We Need Position?

Compare:

```text
dog bites man
```

and:

```text
man bites dog
```

The same tokens appear, but their order differs.

The model needs positional information.

For this toy example, suppose we use learned positional vectors.

```text
Position 0 → [0.1,0.1,0.1,0.1]
Position 1 → [0.2,0.2,0.2,0.2]
Position 2 → [0.3,0.3,0.3,0.3]
```

---

# 7. Add Token Embedding + Position

For an architecture using additive positional embeddings:

\[
X_i=E_i+P_i
\]

## Position 0: "I"

Token:

\[
[1,0,0,1]
\]

Position:

\[
[0.1,0.1,0.1,0.1]
\]

Add:

\[
[1,0,0,1]+[0.1,0.1,0.1,0.1]
\]

\[
\boxed{
[1.1,0.1,0.1,1.1]
}
\]

---

## Position 1: "love"

Token:

\[
[0,1,1,0]
\]

Position:

\[
[0.2,0.2,0.2,0.2]
\]

Add:

\[
[0,1,1,0]+[0.2,0.2,0.2,0.2]
\]

\[
\boxed{
[0.2,1.2,1.2,0.2]
}
\]

---

## Position 2: "cats"

Token:

\[
[1,1,0,0]
\]

Position:

\[
[0.3,0.3,0.3,0.3]
\]

Add:

\[
[1,1,0,0]+[0.3,0.3,0.3,0.3]
\]

\[
\boxed{
[1.3,1.3,0.3,0.3]
}
\]

---

# 8. Transformer Input

Therefore:

\[
X=
\begin{bmatrix}
1.1&0.1&0.1&1.1\\
0.2&1.2&1.2&0.2\\
1.3&1.3&0.3&0.3
\end{bmatrix}
\]

This is the representation entering the Transformer in this simplified additive-position example.

---

# 9. Important Modern LLM Note

The above:

\[
X=E_{token}+E_{position}
\]

is a simplified example.

Modern LLMs often use **RoPE (Rotary Position Embeddings)**.

RoPE does not simply add a learned position vector to token embeddings.

Conceptually:

```text
Token representation
       ↓
Q/K projections
       ↓
Position-dependent rotation
       ↓
Attention
```

RoPE will be studied separately in the positional encoding lesson.

For this numerical lesson, additive position is used because it is easier to calculate manually.

---

# 10. Self-Attention

Self-attention allows each token to gather information from other tokens.

For:

```text
I love cats
```

conceptually:

```text
I     ← I, love, cats
love  ← I, love, cats
cats  ← I, love, cats
```

The amount of information gathered from each token is determined by attention weights.

---

# 11. Q, K, V

Given:

\[
X
\]

the model creates:

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

- Q = Query
- K = Key
- V = Value

The full attention formula is:

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

# 12. What Do Q, K, and V Intuitively Mean?

A useful intuition:

### Query

> What information am I looking for?

### Key

> What information do I contain / how should I be matched?

### Value

> What information should actually be passed forward?

The model learns the projections \(W_Q,W_K,W_V\).

---

# 13. Simple Attention Score Example

Suppose for the query corresponding to `love`, the model produces scaled scores:

```text
          I     love    cats
love →   1.0    2.0     0.5
```

These are the values after:

\[
\frac{QK^T}{\sqrt{d_k}}
\]

for this simplified example.

---

# 14. Apply Softmax

Softmax:

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

we get approximately:

```text
[0.231, 0.629, 0.140]
```

Check:

\[
0.231+0.629+0.140=1
\]

Therefore `love` pays approximately:

```text
23.1% attention to I
62.9% attention to love
14.0% attention to cats
```

---

# 15. Attention Weights

Write the weights as:

\[
A_{love}
=
[0.231,0.629,0.140]
\]

These are not the final representation.

They tell us **how much information to take from each Value vector**.

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

The actual model's values are produced by:

\[
V=XW_V
\]

and are not normally this simple.

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

Calculate each term.

### Contribution from I

\[
0.231[1,0,0,0]
=
[0.231,0,0,0]
\]

### Contribution from love

\[
0.629[0,1,0,0]
=
[0,0.629,0,0]
\]

### Contribution from cats

\[
0.140[0,0,1,0]
=
[0,0,0.140,0]
\]

Add:

\[
\boxed{
[0.231,0.629,0.140,0]
}
\]

This is the simplified attention output for the `love` query.

---

# 18. What Did Attention Do?

Before attention, the `love` representation mainly represented its own token plus its position.

After attention:

```text
love
 ↓
Looks at I
Looks at love
Looks at cats
 ↓
Combines information
 ↓
Updated representation
```

This is the beginning of **contextualization**.

---

# 19. Contextual Representation

A contextual representation is a representation that depends on surrounding tokens.

Consider:

```text
I sat near the bank
```

versus:

```text
I deposited money in the bank
```

The token `bank` is the same token in both sentences.

Therefore the initial token embedding is the same.

But the context differs.

---

# 20. Bank Initial Embedding

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

Then:

```text
Sentence A:
I sat near the bank
        ↓
bank initial embedding
[0.4,0.7,0.2,0.9]
```

and:

```text
Sentence B:
I deposited money in the bank
                         ↓
bank initial embedding
[0.4,0.7,0.2,0.9]
```

Same initial vector.

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

provide context.

Suppose after Transformer processing the representation becomes:

\[
\boxed{
H_{bank}^{river}
=
[0.8,0.6,0.1,0.3]
}
\]

This is a hypothetical number used to demonstrate the concept.

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

provide different context.

Suppose after Transformer processing:

\[
\boxed{
H_{bank}^{finance}
=
[0.2,0.3,0.9,0.8]
}
\]

Again, these are illustrative numbers.

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

After context A:

\[
\boxed{
[0.8,0.6,0.1,0.3]
}
\]

After context B:

\[
\boxed{
[0.2,0.3,0.9,0.8]
}
\]

Therefore:

```text
                    bank
                     │
             Same token ID
                     │
             Same initial embedding
                     │
               Self-Attention
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   River context          Money context
          ▼                     ▼
 [0.8,0.6,0.1,0.3]    [0.2,0.3,0.9,0.8]
```

This is the core idea of contextual representations.

---

# 24. Why Does Context Change the Vector?

Self-attention produces weights that determine how strongly each token interacts with other tokens.

For example, for `bank` in:

```text
I deposited money in the bank
```

suppose attention weights are:

```text
deposited → 0.5
money     → 0.3
bank      → 0.2
```

These sum to:

\[
1
\]

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

Add:

\[
\boxed{
[0.5,0.3,0.2,0]
}
\]

This is contextual information gathered through attention.

The real Transformer combines this with learned projections, residual connections, normalization, and feed-forward transformations.

---

# 26. Important: Attention Output Is Not Automatically the Final Contextual Embedding

The simplified calculation:

\[
H=AV
\]

shows the attention operation.

But a real Transformer block includes more:

```text
Input
 ↓
Normalization
 ↓
Q/K/V projections
 ↓
Attention
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

The exact ordering depends on the architecture.

Therefore the final contextual hidden state is more than just the raw weighted value sum.

---

# 27. Contextualization Happens Layer by Layer

Suppose the model has several Transformer layers.

```text
Token Embedding
      ↓
Position Information
      ↓
Layer 1
      ↓
Hidden State 1
      ↓
Layer 2
      ↓
Hidden State 2
      ↓
Layer 3
      ↓
Hidden State 3
      ↓
...
      ↓
Layer N
      ↓
Final Hidden State
```

Representations become progressively transformed and contextualized.

---

# 28. Full Numerical Flow

For:

```text
I love cats
```

we have:

### Token IDs

```text
[10,20,30]
```

### Token embeddings

```text
I     → [1,0,0,1]
love  → [0,1,1,0]
cats  → [1,1,0,0]
```

### Position vectors

```text
0 → [0.1,0.1,0.1,0.1]
1 → [0.2,0.2,0.2,0.2]
2 → [0.3,0.3,0.3,0.3]
```

### Transformer input

```text
I     → [1.1,0.1,0.1,1.1]
love  → [0.2,1.2,1.2,0.2]
cats  → [1.3,1.3,0.3,0.3]
```

### Q/K/V

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

### Attention

\[
A=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
\]

### Contextual information

\[
H=AV
\]

Then the output goes through the remaining Transformer operations and subsequent layers.

---

# 29. A Useful Mental Model

Think of the three concepts as three questions.

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
bank + river context
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

| | Token Embedding | Contextual Representation |
|---|---|---|
| Obtained from | Embedding lookup | Transformer processing |
| Depends on context? | No, at initial lookup | Yes |
| Same token, different sentence | Same initial vector | Can become different |
| Main mechanism | Embedding matrix | Attention + FFN + layers |
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

Token embedding:

```text
cat
 ↓
one token vector
```

Sentence embedding:

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
- Similarity

---

# 32. Tensor Shapes

Suppose:

```text
Batch size B = 8
Sequence length T = 128
Embedding dimension d = 4096
```

Then:

\[
X\in\mathbb{R}^{8\times128\times4096}
\]

Meaning:

```text
8    → sequences
128  → tokens per sequence
4096 → features per token
```

For self-attention, Q/K/V commonly preserve the batch and sequence dimensions while projecting the hidden dimension.

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

Parameter count:

\[
50,000\times4096
=
204,800,000
\]

So the embedding matrix alone contains approximately:

\[
\boxed{
205\text{ million parameters}
}
\]

---

# 34. Why the Numerical Example Is Simplified

Real LLMs differ from this toy example in several ways:

- Embedding dimensions are much larger.
- \(W_Q,W_K,W_V\) are learned matrices.
- Multi-head attention is used.
- Causal masks are used in decoder-only LLMs.
- Normalization is used.
- Residual connections are used.
- Feed-forward networks are used.
- Modern models may use RoPE instead of additive positional embeddings.
- There may be dozens or hundreds of Transformer layers.

The toy calculations are designed to show the underlying mechanics.

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

The entire numerical idea can be summarized as:

```text
Token ID
   ↓
Embedding lookup
   ↓
Initial token vector
   ↓
Position information
   ↓
Q/K/V
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
Contextual hidden state
```

And the most important example is:

```text
Same token:

bank
 ↓
[0.4,0.7,0.2,0.9]

              ↓ self-attention

River context:
[0.8,0.6,0.1,0.3]

Financial context:
[0.2,0.3,0.9,0.8]
```

So:

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

Next we combine everything into the architecture of a modern decoder-only LLM:

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
