# Lesson 04 - Numerical Examples: Embeddings, Positional Information, Self-Attention and Contextual Representations

## Purpose

This lesson builds numerical intuition for the following pipeline:

```text
Token
  |
  v
Token ID
  |
  v
Token Embedding
  |
  v
Positional Information
  |
  v
Q, K, V
  |
  v
Self-Attention
  |
  v
Contextual Representation
```

The numbers in this lesson are intentionally tiny and simplified so that the calculations can be followed by hand.

Real LLMs use much larger dimensions, learned matrices, multiple attention heads, normalization, residual connections, feed-forward networks, and many Transformer layers.

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
I     -> 10
love  -> 20
cats  -> 30
```

Therefore:

```text
input_ids = [10, 20, 30]
```

---

# 2. Token IDs Are Just Indices

Suppose:

```text
cats -> 30
```

The number `30` does NOT mean:

```text
30 units of meaning
```

It simply tells the model which vocabulary or embedding entry to retrieve.

If `E` is the embedding matrix:

```text
E[30]
```

means:

```text
Retrieve the embedding stored at index 30.
```

A token ID is therefore an index, not a semantic vector.

---

# 3. Create a Tiny Embedding Matrix

Suppose:

```text
Vocabulary size V = 100
Embedding dimension d = 4
```

Then the embedding matrix has shape:

```text
E: 100 x 4
```

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

```text
E[10] = [1, 0, 0, 1]

E[20] = [0, 1, 1, 0]

E[30] = [1, 1, 0, 0]
```

Therefore:

```text
X =
[
  [1, 0, 0, 1],
  [0, 1, 1, 0],
  [1, 1, 0, 0]
]
```

Shape:

```text
3 x 4
```

Meaning:

```text
3 tokens
x
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

It is the model's initial numerical representation of the token before the Transformer layers contextualize it.

The basic operation is:

```text
Token ID
   |
   v
Embedding lookup
   |
   v
Token embedding
```

Important:

```text
Token ID != Token Embedding
```

A token ID is an integer index.

A token embedding is a vector of learned numbers.

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

The same words appear, but their order is different.

A Transformer therefore needs some mechanism that provides information about token positions.

For this toy example, suppose we use learned positional vectors:

```text
Position 0 -> [0.1, 0.1, 0.1, 0.1]

Position 1 -> [0.2, 0.2, 0.2, 0.2]

Position 2 -> [0.3, 0.3, 0.3, 0.3]
```

---

# 7. Add Token Embedding and Position

For an architecture using additive positional embeddings:

```text
X_i = E_i + P_i
```

where:

```text
E_i = token embedding
P_i = positional embedding
X_i = resulting input representation
```

## Position 0: "I"

Token embedding:

```text
[1, 0, 0, 1]
```

Position vector:

```text
[0.1, 0.1, 0.1, 0.1]
```

Add them:

```text
[1, 0, 0, 1]
+
[0.1, 0.1, 0.1, 0.1]
=
[1.1, 0.1, 0.1, 1.1]
```

---

## Position 1: "love"

Token embedding:

```text
[0, 1, 1, 0]
```

Position vector:

```text
[0.2, 0.2, 0.2, 0.2]
```

Add:

```text
[0, 1, 1, 0]
+
[0.2, 0.2, 0.2, 0.2]
=
[0.2, 1.2, 1.2, 0.2]
```

---

## Position 2: "cats"

Token embedding:

```text
[1, 1, 0, 0]
```

Position vector:

```text
[0.3, 0.3, 0.3, 0.3]
```

Add:

```text
[1, 1, 0, 0]
+
[0.3, 0.3, 0.3, 0.3]
=
[1.3, 1.3, 0.3, 0.3]
```

---

# 8. Transformer Input

The resulting input matrix is:

```text
X =
[
  [1.1, 0.1, 0.1, 1.1],
  [0.2, 1.2, 1.2, 0.2],
  [1.3, 1.3, 0.3, 0.3]
]
```

Shape:

```text
3 x 4
```

Meaning:

```text
3 token positions
x
4-dimensional representation
```

In this simplified additive-position example, this is the representation entering the Transformer.

---

# 9. Important Modern LLM Note: Additive Position Is Only One Approach

The calculation:

```text
X = Token Embedding + Position Embedding
```

is useful for learning because it is easy to calculate.

However, modern LLMs do not all use additive positional embeddings.

Many modern decoder-only LLM architectures use:

```text
RoPE = Rotary Position Embeddings
```

RoPE does not simply add a learned position vector to the token embedding.

Conceptually:

```text
Token representations
       |
       v
Q/K projections
       |
       v
Position-dependent rotation
       |
       v
Attention
```

RoPE will be studied separately.

For this lesson, additive positional information is used because it makes the numerical calculation easy to follow.

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
I     <- information from I, love, cats
love  <- information from I, love, cats
cats  <- information from I, love, cats
```

The amount of information gathered from each position is controlled by attention weights.

---

# 11. Q, K, and V

Given the input matrix `X`, the model creates:

```text
Q = X W_Q

K = X W_K

V = X W_V
```

where:

```text
Q = Query
K = Key
V = Value
```

`W_Q`, `W_K`, and `W_V` are learned projection matrices.

The standard scaled dot-product attention formula is:

```text
Attention(Q, K, V)
    =
softmax(
    (Q K^T) / sqrt(d_k)
) V
```

This formula is one of the most important formulas in Transformer architecture.

---

# 12. What Do Q, K, and V Mean?

A useful intuition is:

## Query

```text
What information am I looking for?
```

## Key

```text
What information do I provide for matching?
```

## Value

```text
What information should actually be passed forward?
```

These are only intuitions.

In the actual Transformer, Q, K, and V are learned linear projections of hidden states.

---

# 13. Simple Attention Score Example

Suppose that for the query corresponding to `love`, the scaled attention scores are:

```text
          I     love    cats
love ->  1.0    2.0     0.5
```

These are assumed values for our simplified example.

In a real Transformer, they come from:

```text
(Q K^T) / sqrt(d_k)
```

A larger score means a stronger match before softmax.

---

# 14. Apply Softmax

Softmax is:

```text
softmax(z_i) = exp(z_i) / sum(exp(z_j))
```

For:

```text
[1.0, 2.0, 0.5]
```

the approximate result is:

```text
[0.231, 0.629, 0.140]
```

Check:

```text
0.231 + 0.629 + 0.140 = 1.000
```

Therefore the `love` position assigns approximately:

```text
23.1% -> I
62.9% -> love
14.0% -> cats
```

---

# 15. Attention Weights

Write these weights as:

```text
A_love = [0.231, 0.629, 0.140]
```

These are NOT the final representation.

They tell us how strongly the `love` position should use the corresponding Value vectors.

---

# 16. Create Simple Value Vectors

For illustration, suppose:

```text
V_I    = [1, 0, 0, 0]

V_love = [0, 1, 0, 0]

V_cats = [0, 0, 1, 0]
```

In a real model, the Value vectors are produced by:

```text
V = X W_V
```

and usually contain many dimensions.

---

# 17. Weighted Value Combination

The attention output for `love` is:

```text
0.231 * V_I
+
0.629 * V_love
+
0.140 * V_cats
```

Calculate each contribution.

### Contribution from I

```text
0.231 * [1, 0, 0, 0]
=
[0.231, 0, 0, 0]
```

### Contribution from love

```text
0.629 * [0, 1, 0, 0]
=
[0, 0.629, 0, 0]
```

### Contribution from cats

```text
0.140 * [0, 0, 1, 0]
=
[0, 0, 0.140, 0]
```

Add them:

```text
[0.231, 0, 0, 0]
+
[0, 0.629, 0, 0]
+
[0, 0, 0.140, 0]
=
[0.231, 0.629, 0.140, 0]
```

Therefore:

```text
Attention output for "love":

[0.231, 0.629, 0.140, 0]
```

---

# 18. What Did Attention Do?

Before attention, the `love` position had its own representation.

During attention:

```text
love
  |
  v
Compare Query with Keys
  |
  v
Calculate attention scores
  |
  v
Apply softmax
  |
  v
Obtain attention weights
  |
  v
Combine Value vectors
  |
  v
Create updated representation
```

This is the beginning of contextualization.

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

Therefore its initial token embedding is the same.

But the surrounding context is different.

After Transformer processing, the hidden representation associated with `bank` can become different in the two sentences.

---

# 20. Initial Embedding of "bank"

Suppose:

```text
bank -> token ID 500
```

and:

```text
E_bank = [0.4, 0.7, 0.2, 0.9]
```

For both sentences, the initial embedding lookup gives:

```text
[0.4, 0.7, 0.2, 0.9]
```

So:

```text
Sentence A:
I sat near the bank
                |
                v
        same initial embedding


Sentence B:
I deposited money in the bank
                         |
                         v
                 same initial embedding
```

---

# 21. Context A - River-Related Meaning

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

```text
H_bank_river = [0.8, 0.6, 0.1, 0.3]
```

These numbers are hypothetical.

They are only being used to demonstrate the idea of contextual representations.

---

# 22. Context B - Financial Meaning

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

```text
H_bank_finance = [0.2, 0.3, 0.9, 0.8]
```

Again, these values are illustrative.

---

# 23. Compare the Two

Initial embedding:

```text
E_bank = [0.4, 0.7, 0.2, 0.9]
```

After river-related context:

```text
[0.8, 0.6, 0.1, 0.3]
```

After financial context:

```text
[0.2, 0.3, 0.9, 0.8]
```

Conceptually:

```text
                    bank
                     |
              Same token ID
                     |
          Same initial embedding
                     |
               Transformer
                     |
          +----------+----------+
          |                     |
          v                     v
   River-related context   Financial context
          |                     |
          v                     v
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
deposited -> 0.5
money     -> 0.3
bank      -> 0.2
```

They sum to:

```text
0.5 + 0.3 + 0.2 = 1
```

The exact values in a real model are learned and depend on the input.

---

# 25. Numerical Context Gathering

Suppose:

```text
V_deposited = [1, 0, 0, 0]

V_money     = [0, 1, 0, 0]

V_bank      = [0, 0, 1, 0]
```

Then:

```text
0.5 * V_deposited
+
0.3 * V_money
+
0.2 * V_bank
```

Calculate:

```text
0.5 * [1, 0, 0, 0]
=
[0.5, 0, 0, 0]
```

```text
0.3 * [0, 1, 0, 0]
=
[0, 0.3, 0, 0]
```

```text
0.2 * [0, 0, 1, 0]
=
[0, 0, 0.2, 0]
```

Add them:

```text
[0.5, 0, 0, 0]
+
[0, 0.3, 0, 0]
+
[0, 0, 0.2, 0]
=
[0.5, 0.3, 0.2, 0]
```

This demonstrates how attention can gather information from surrounding positions.

The actual Transformer performs additional learned transformations and combines the attention output with residual connections and feed-forward processing.

---

# 26. Important: Raw Attention Output Is Not Automatically the Final Contextual Representation

The simplified calculation:

```text
H_attention = A V
```

demonstrates the core weighted-Value operation.

A real Transformer block contains additional operations.

A simplified block can be viewed as:

```text
Input
  |
  v
Normalization
  |
  v
Q/K/V projections
  |
  v
Self-Attention
  |
  v
Output projection
  |
  v
Residual connection
  |
  v
Normalization
  |
  v
Feed-Forward Network
  |
  v
Residual connection
  |
  v
Output
```

The exact ordering and normalization strategy depend on the architecture.

Therefore:

```text
Raw attention output
        !=
Final contextual hidden state
```

The attention output is one important intermediate result.

---

# 27. Contextualization Happens Layer by Layer

Suppose a Transformer has several layers:

```text
Token Embedding
      |
      v
Position Information
      |
      v
Transformer Layer 1
      |
      v
Hidden State 1
      |
      v
Transformer Layer 2
      |
      v
Hidden State 2
      |
      v
Transformer Layer 3
      |
      v
Hidden State 3
      |
     ...
      |
      v
Transformer Layer N
      |
      v
Final Hidden State
```

The representations are repeatedly transformed and contextualized.

---

# 28. Full Numerical Flow for "I love cats"

## Step 1 - Token IDs

```text
[10, 20, 30]
```

## Step 2 - Token embeddings

```text
I     -> [1, 0, 0, 1]
love  -> [0, 1, 1, 0]
cats  -> [1, 1, 0, 0]
```

## Step 3 - Position vectors

```text
0 -> [0.1, 0.1, 0.1, 0.1]
1 -> [0.2, 0.2, 0.2, 0.2]
2 -> [0.3, 0.3, 0.3, 0.3]
```

## Step 4 - Add token and position information

```text
I     -> [1.1, 0.1, 0.1, 1.1]
love  -> [0.2, 1.2, 1.2, 0.2]
cats  -> [1.3, 1.3, 0.3, 0.3]
```

## Step 5 - Q, K, V

```text
Q = X W_Q
K = X W_K
V = X W_V
```

## Step 6 - Attention scores

```text
S = (Q K^T) / sqrt(d_k)
```

## Step 7 - Softmax

```text
A = softmax(S)
```

## Step 8 - Weighted Values

```text
H_attention = A V
```

## Step 9 - Remaining Transformer operations

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
  |
  v
[0.4, 0.7, 0.2, 0.9]
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
          |
          v
[0.8, 0.6, 0.1, 0.3]
```

versus:

```text
bank + financial context
          |
          v
[0.2, 0.3, 0.9, 0.8]
```

---

# 30. Token Embedding vs Contextual Representation

| Property | Token Embedding | Contextual Representation |
|---|---|---|
| Obtained from | Embedding lookup | Transformer processing |
| Depends on surrounding context? | No, at initial lookup | Yes |
| Same token in different sentences | Same initial vector | Can become different |
| Main mechanism | Embedding matrix | Attention + FFN + Transformer layers |
| Example | `bank -> E[bank]` | `bank + context -> H_bank` |

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
  |
  v
one token vector
```

A sentence embedding represents an entire text as one vector:

```text
"I love cats"
      |
      v
one vector representing the text
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
Hidden dimension d = 4096
```

Then the input hidden-state tensor has shape:

```text
8 x 128 x 4096
```

Meaning:

```text
8    -> sequences in the batch
128  -> token positions per sequence
4096 -> features per token
```

For self-attention, Q, K, and V preserve the batch and sequence structure while projecting the hidden dimension into attention representations.

In multi-head attention, the hidden dimension is divided across multiple heads.

---

# 33. Embedding Matrix Parameter Count

Suppose:

```text
Vocabulary size V = 50,000

Embedding dimension d = 4,096
```

Then:

```text
Embedding matrix = 50,000 x 4,096
```

Parameter count:

```text
50,000 x 4,096
= 204,800,000
```

So the embedding matrix contains approximately:

```text
205 million parameters
```

This shows why vocabulary size and hidden dimension can contribute significantly to model size.

---

# 34. Why Is This Numerical Example Simplified?

Real LLMs differ from this toy example in many ways:

- Embedding dimensions are much larger.
- `W_Q`, `W_K`, and `W_V` are learned matrices.
- Multi-head attention is used.
- Decoder-only LLMs use causal masking during generation.
- Normalization is used.
- Residual connections are used.
- Feed-forward networks are used.
- Modern architectures may use RoPE rather than additive positional embeddings.
- There may be dozens or hundreds of Transformer layers.
- Hidden states are not generally interpretable as individual human-readable features.
- Attention output is only one part of a Transformer block.

The toy calculations are designed to explain the mechanics, not reproduce a production LLM numerically.

---

# 35. Complete Mental Diagram

```text
                       "I love cats"
                              |
                              v
                         TOKENIZER
                              |
                              v
                       TOKEN IDs
                       [10, 20, 30]
                              |
                              v
                     EMBEDDING LOOKUP
                              |
                              v
                      TOKEN EMBEDDINGS
                              |
                    +---------+---------+
                    |                   |
                    v                   v
             Token information   Position information
                    |                   |
                    +---------+---------+
                              |
                              v
                     TRANSFORMER INPUT
                              |
                              v
                       Q, K, V PROJECTIONS
                              |
                              v
                        SELF-ATTENTION
                              |
                              v
                       CONTEXT MIXING
                              |
                              v
                  CONTEXTUAL REPRESENTATION
                              |
                              v
                     FEED-FORWARD NETWORK
                              |
                              v
                    NEXT TRANSFORMER LAYER
                              |
                             ...
                              |
                              v
                     FINAL HIDDEN STATES
                              |
                              v
                           LM HEAD
                              |
                              v
                           LOGITS
                              |
                              v
                     NEXT-TOKEN PREDICTION
```

---

# 36. Key Formulas

## Embedding Matrix

```text
E has shape:

V x d
```

where:

```text
V = vocabulary size
d = embedding dimension
```

## Embedding Lookup

```text
e_i = E[i]
```

## Additive Positional Embedding

```text
X_i = E_i + P_i
```

## Query

```text
Q = X W_Q
```

## Key

```text
K = X W_K
```

## Value

```text
V = X W_V
```

## Scaled Dot-Product Attention

```text
Attention(Q, K, V)
=
softmax(
    (Q K^T) / sqrt(d_k)
) V
```

## Contextual Representation

Conceptually:

```text
Contextual representation
=
Transformer processing of
the input representation in context
```

---

# 37. Most Important Takeaway

The numerical flow can be summarized as:

```text
Token ID
   |
   v
Embedding lookup
   |
   v
Initial token vector
   |
   v
Position information
   |
   v
Q/K/V projections
   |
   v
Attention scores
   |
   v
Softmax
   |
   v
Attention weights
   |
   v
Weighted Values
   |
   v
Context mixing
   |
   v
Transformer layers
   |
   v
Contextual hidden state
```

The central example is:

```text
Same token:

bank
  |
  v
[0.4, 0.7, 0.2, 0.9]

        Transformer processing

River-related context:
[0.8, 0.6, 0.1, 0.3]

Financial context:
[0.2, 0.3, 0.9, 0.8]
```

Therefore:

```text
Same initial token embedding
              |
              v
Different contextual representations
```

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
11. What does `Q K^T` calculate?
12. Why do we divide by `sqrt(d_k)`?
13. What does softmax do?
14. What are attention weights?
15. How are Value vectors combined using attention weights?
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

## Lesson 05 - LLM Architecture Internals

Next we combine these concepts into the architecture of a modern decoder-only LLM:

```text
Token IDs
   |
   v
Token Embedding
   |
   v
RoPE
   |
   v
RMSNorm
   |
   v
Causal Self-Attention
   |
   v
Residual Connection
   |
   v
RMSNorm
   |
   v
SwiGLU / FFN
   |
   v
Residual Connection
   |
   v
Repeat N times
   |
   v
Final RMSNorm
   |
   v
LM Head
   |
   v
Logits
   |
   v
Next-token prediction
```

We will also learn **KV Cache**, which explains why inference is computationally different from training.
