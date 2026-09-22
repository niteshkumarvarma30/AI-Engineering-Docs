# Positional Encoding vs RoPE - Complete Conceptual Note

## Core Idea

A traditional additive positional encoding approach looks like this:

```text
Token IDs
   |
   v
Token Embedding
   |
   v
Add Positional Encoding
   |
   v
X
   |
   v
Q, K, V
   |
   v
Self-Attention
```

The central operation is:

```text
X = E + PE
```

where:

```text
E  = token embeddings
PE = positional encoding
X  = representation passed to the Transformer
```

RoPE (Rotary Position Embeddings) uses a different strategy:

```text
Token IDs
   |
   v
Token Embedding
   |
   v
X
   |
   v
Q, K, V
   |
   +------> Q -> RoPE -> Q_rot
   |
   +------> K -> RoPE -> K_rot
   |
   +------> V -> unchanged
   |
   v
Self-Attention
```

The key difference is:

```text
Additive positional encoding:
Position is added to X.

RoPE:
Position-dependent rotation is applied to Q and K.
```

---

# 1. Normal Transformer Flow

For:

```text
I love cats
```

tokenization might produce:

```text
["I", "love", "cats"]
```

Then:

```text
["I", "love", "cats"]
        |
        v
     Token IDs
        |
        v
  Embedding Lookup
        |
        v
 Token Embeddings X
```

The model then creates:

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

---

# 2. Why Is Position Needed?

Compare:

```text
The dog chased the cat
```

and:

```text
The cat chased the dog
```

The same words occur, but their order changes the meaning.

A self-attention mechanism therefore needs positional information.

Without a positional mechanism, the model would have difficulty distinguishing token roles based on sequence position.

---

# 3. Additive Positional Encoding

A traditional approach adds positional information to the token representation:

```text
X = E + PE
```

Then:

```text
Q = X W_Q

K = X W_K

V = X W_V
```

and attention is calculated as:

```text
Attention(Q, K, V)
=
softmax(
    (Q K^T) / sqrt(d_k)
) V
```

The important point is that positional information is already present in `X` before Q, K, and V are calculated.

---

# 4. Numerical Additive PE Example

Suppose:

```text
E_cats = [1, 1, 0, 0]
```

and position 2 has:

```text
PE_2 = [0.3, 0.3, 0.3, 0.3]
```

Then:

```text
X_cats = E_cats + PE_2
```

Therefore:

```text
[1, 1, 0, 0]
+
[0.3, 0.3, 0.3, 0.3]
=
[1.3, 1.3, 0.3, 0.3]
```

So:

```text
X_cats = [1.3, 1.3, 0.3, 0.3]
```

This representation is then used to calculate Q, K, and V.

---

# 5. What Is RoPE?

RoPE means:

```text
Rotary Position Embeddings
```

Instead of adding a positional vector to `X`, RoPE applies a position-dependent rotation to Q and K.

Correct flow:

```text
                 Token Embedding X
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
             Q           K           V
             |           |           |
             v           v           |
           RoPE        RoPE          |
             |           |           |
             v           v           |
          Q_rot       K_rot           |
             +-----+-----+            |
                   |                  |
                   v                  |
              Q_rot K_rot^T           |
                   |                  |
                   v                  |
             Attention scores         |
                   |                  |
                   v                  |
                 Softmax              |
                   |                  |
                   v                  |
             Attention weights -------+
                                      |
                                      v
                              Weighted Values
```

Important:

> RoPE does NOT mean that V disappears.

You still calculate Q, K, and V.

Only Q and K receive the rotary transformation in the standard RoPE formulation.

---

# 6. Q, K, and V Intuition

A useful intuition:

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
What information should I provide if another token attends to me?
```

Therefore:

```text
Q + K
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
V
  |
  v
Attention output
```

RoPE changes the positional relationship represented in Q and K.

---

# 7. Why Is RoPE Applied to Q and K?

The attention scores are based on:

```text
Q K^T
```

With RoPE, this becomes:

```text
Q_rot K_rot^T
```

where the rotation applied to Q and K depends on their positions.

Therefore positional information directly influences the attention relationship between token positions.

---

# 8. Understanding 2D Rotation

RoPE can be understood using a simple 2D rotation.

Start with:

```text
q = [x, y]
```

A 2D rotation by angle theta is:

```text
R(theta) =
[
  [cos(theta), -sin(theta)],
  [sin(theta),  cos(theta)]
]
```

The rotated vector is:

```text
q_rot = R(theta) q
```

This is the mathematical foundation behind the intuitive idea of RoPE.

---

# 9. Numerical Rotation Example

Suppose:

```text
q = [1, 0]
```

and:

```text
theta = 90 degrees
```

Then:

```text
R(90 degrees) =
[
  [0, -1],
  [1,  0]
]
```

Multiply:

```text
q_rot =
[
  [0, -1],
  [1,  0]
]
*
[
  1
  0
]
```

Result:

```text
q_rot =
[
  0
  1
]
```

Therefore:

```text
q_rot = [0, 1]
```

The vector has been rotated by 90 degrees.

---

# 10. Position Controls the Rotation

Suppose:

```text
q = [1, 0]
```

and the position-dependent angles are:

```text
Position 0 -> 0 degrees
Position 1 -> 30 degrees
Position 2 -> 60 degrees
```

Then:

```text
Position 0:

q_0 = R(0 degrees) q
    = [1, 0]
```

For position 1:

```text
q_1 = R(30 degrees) q
    = [0.866, 0.5]
```

For position 2:

```text
q_2 = R(60 degrees) q
    = [0.5, 0.866]
```

Therefore:

```text
Same underlying vector
        |
        v
Different position
        |
        v
Different rotation
        |
        v
Different Q/K representation
```

The same position-dependent operation is also applied to K.

---

# 11. Relative Position Intuition

Suppose:

```text
Query position = 5
Key position   = 3
```

The relative distance is:

```text
5 - 3 = 2
```

Another pair:

```text
Query position = 10
Key position   = 8
```

The relative distance is also:

```text
10 - 8 = 2
```

A major property of RoPE is that the interaction between rotated Q and K has a natural dependence on relative positional differences.

Conceptually:

```text
RoPE makes attention sensitive to relative position.
```

This does not mean RoPE explicitly stores a simple scalar such as "distance = 2" in every vector. Rather, the rotation structure makes relative position appear naturally in the Q/K dot-product relationship.

---

# 12. RoPE Does Not Mean "Skip V"

This is a common misunderstanding.

Incorrect:

```text
Embedding
   |
   v
Q, K
   |
   v
Attention
```

Correct:

```text
Embedding X
     |
     +----> Q ----> RoPE ----> Q_rot
     |
     +----> K ----> RoPE ----> K_rot
     |
     +----> V ---------------> V
```

Then:

```text
Q_rot + K_rot
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
V
       |
       v
Attention output
```

So Q, K, and V are still all present.

---

# 13. Why Is V Normally Not Rotated?

Attention can be viewed conceptually as two stages.

## Stage 1 - Decide where to attend

```text
Q + K
  |
  v
Attention scores
  |
  v
Attention weights
```

## Stage 2 - Gather information

```text
Attention weights + V
          |
          v
   Attention output
```

RoPE is used to encode positional relationships in the Q/K interaction.

Therefore, in the standard RoPE formulation, Q and K are rotated while V is left unchanged by RoPE.

---

# 14. Additive PE vs RoPE

## Additive PE

```text
Token IDs
   |
   v
Embedding E
   |
   v
E + PE
   |
   v
X
   |
   v
Q, K, V
   |
   v
Attention
```

The central operation is:

```text
X = E + PE
```

## RoPE

```text
Token IDs
   |
   v
Embedding X
   |
   v
Q, K, V
   |
   +----> Q -> RoPE -> Q_rot
   |
   +----> K -> RoPE -> K_rot
   |
   +----> V -> unchanged
   |
   v
Attention
```

The central operations are:

```text
Q = X W_Q
K = X W_K
V = X W_V

Q_rot = RoPE(Q, position)
K_rot = RoPE(K, position)
```

Then:

```text
Attention
=
softmax(
    (Q_rot K_rot^T) / sqrt(d_k)
) V
```

---

# 15. Side-by-Side Comparison

```text
          ADDITIVE PE                         RoPE

          Token IDs                          Token IDs
              |                                 |
              v                                 v
          Embedding                         Embedding
              |                                 |
              v                                 v
              E                                 X
              |                                 |
              +-- PE                            +----> Q
              |                                 |       |
              v                                 |     RoPE
              X                                 |       |
              |                                 |       v
           +--+--+                              |     Q_rot
           |  |  |                              |
           v  v  v                              +----> K
           Q  K  V                              |       |
           |  |  |                              |     RoPE
           +--+--+                              |       |
              |                                 |       v
              v                                 |     K_rot
          Attention                             |
                                                +----> V
                                                        |
                                                        v
                                               Q_rot, K_rot, V
                                                        |
                                                        v
                                                   Attention
```

---

# 16. RoPE in Multi-Head Attention

Suppose:

```text
Hidden size = 4096
Number of attention heads = 32
Head dimension = 128
```

The hidden representation is projected into multiple attention heads.

Conceptually:

```text
Hidden representation
        |
        v
   Multi-Head Q/K/V
        |
   +----+----+----------------+
   |         |                |
   v         v                v
 Head 1    Head 2           Head 32
   |         |                |
   v         v                v
  Q K V     Q K V            Q K V
   |         |                |
   v         v                v
 RoPE      RoPE              RoPE
   |         |                |
   v         v                v
Attention Attention        Attention
```

RoPE is applied to the relevant Q and K dimensions within each attention head.

---

# 17. RoPE Works on Dimension Pairs

Suppose one attention head has dimension 8:

```text
[x1, x2, x3, x4, x5, x6, x7, x8]
```

Conceptually, pair the dimensions:

```text
(x1, x2)
(x3, x4)
(x5, x6)
(x7, x8)
```

Each pair can be treated as a 2D vector.

For example:

```text
(x1, x2)
```

can be rotated using a 2D rotation matrix.

The same idea is applied across the other pairs.

---

# 18. Different Frequencies

Different dimension pairs use different rotation frequencies.

Conceptually:

```text
Pair 1 -> frequency 1
Pair 2 -> frequency 2
Pair 3 -> frequency 3
...
```

This allows different dimensions to represent positional patterns at different scales.

Some components change more rapidly with position, while others change more slowly.

This multi-frequency structure is important for representing positional relationships across different distances.

---

# 19. Real LLMs

The 2D examples above are only for intuition.

Real models may use head dimensions such as:

```text
64
96
128
...
```

RoPE operates across pairs of dimensions.

The exact RoPE implementation can differ between model architectures, including details such as the frequency base, scaling strategy, and which dimensions receive rotation.

---

# 20. RoPE and KV Cache

RoPE is closely related to inference and the KV Cache.

During autoregressive generation, previously computed Key and Value states can be cached.

Conceptually:

```text
Previous tokens
      |
      v
Past K and V
      |
      v
KV Cache
```

For a new token:

```text
New token
   |
   v
Q, K, V
   |
   +----> Q -> RoPE -> Q_rot
   |
   +----> K -> RoPE -> K_rot
   |
   +----> V -> unchanged
```

Then:

```text
New Q_rot
    |
    v
attends to cached K
    |
    v
uses cached/new V
    |
    v
Attention output
```

The exact implementation depends on the model architecture and KV-cache convention.

The key idea is:

```text
KV Cache = stored past Key and Value states
```

---

# 21. Complete Modern Attention Flow

A simplified modern decoder-only attention block can be visualized as:

```text
                       Token IDs
                           |
                           v
                    Token Embeddings
                           |
                           v
                           X
                           |
              +------------+------------+
              |            |            |
              v            v            v
              Q            K            V
              |            |            |
              v            v            |
            RoPE         RoPE           |
              |            |            |
              v            v            |
            Q_rot        K_rot          |
              |            |            |
              +-----+------+            |
                    |
                    v
               Q_rot K_rot^T
                    |
                    v
             Divide by sqrt(d_k)
                    |
                    v
               Causal Mask
                    |
                    v
                  Softmax
                    |
                    v
             Attention Weights
                    |
                    +------------------+
                                       |
                                       v
                                 Weighted V
                                       |
                                       v
                               Output Projection
                                       |
                                       v
                                  Residual Path
```

The exact block ordering depends on the model architecture.

---

# 22. Mathematical Summary

## Additive Positional Encoding

Start with token embeddings:

```text
E = Token Embeddings
```

Add position:

```text
X = E + PE
```

Then:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

Attention:

```text
Attention
=
softmax(
    (Q K^T) / sqrt(d_k)
) V
```

## RoPE

Start with token embeddings:

```text
X = E
```

Create Q, K, V:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

Apply position-dependent rotation:

```text
Q_rot = RoPE(Q, position)

K_rot = RoPE(K, position)
```

V remains unchanged by RoPE:

```text
V = unchanged by RoPE
```

Then:

```text
Attention
=
softmax(
    (Q_rot K_rot^T) / sqrt(d_k)
) V
```

For a decoder-only LLM, a causal mask is also applied so that a token cannot attend to future tokens.

---

# 23. The Most Important Difference

Remember:

```text
Traditional additive PE:

Position
   |
   v
Added to X
   |
   v
Q, K, V
   |
   v
Attention
```

versus:

```text
RoPE:

X
   |
   v
Q, K, V
   |
   +----> Q + position-dependent rotation
   |
   +----> K + position-dependent rotation
   |
   +----> V unchanged by RoPE
   |
   v
Attention
```

Therefore:

```text
Additive PE -> positional information enters X.

RoPE -> positional information enters the Q/K attention relationship.
```

---

# 24. Important Precision

A common statement is:

> "In RoPE, we directly go from embedding to Q and K."

That statement is incomplete.

The correct statement is:

> After embedding, the model creates Q, K, and V. RoPE is then applied to Q and K before the attention calculation.

Correct:

```text
Embedding X
     |
     +----> Q ----> RoPE ----> Q_rot
     |
     +----> K ----> RoPE ----> K_rot
     |
     +----> V ---------------> V
```

Then:

```text
Q_rot + K_rot
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
V
       |
       v
Attention output
```

---

# 25. Final Mental Model

```text
                 TEXT
                  |
                  v
              Tokenizer
                  |
                  v
              Token IDs
                  |
                  v
           Token Embeddings
                  |
                  v
                  X
                  |
        +---------+---------+
        |         |         |
        v         v         v
        Q         K         V
        |         |         |
       RoPE      RoPE       |
        |         |         |
        v         v         |
      Q_rot     K_rot       V
        |         |         |
        +----+----+         |
             |              |
             v              |
       Attention Scores     |
             |              |
             v              |
        Causal Mask         |
             |              |
             v              |
           Softmax          |
             |              |
             v              |
      Attention Weights ----+
             |
             v
        Weighted V
             |
             v
      Output Projection
             |
             v
          Residual
             |
             v
        FFN / next block
```

---

# 26. Final Takeaway

If you remember only one thing:

> In additive positional encoding, positional information is added to the input representation X before Q, K, and V projections. In a RoPE-based Transformer, Q, K, and V are still all created from X, but RoPE applies position-dependent rotations to Q and K before the QK-transpose attention scores are calculated. V is normally left unrotated by RoPE.

Compact comparison:

```text
Additive PE:

E
 |
 + PE
 |
 v
X
 |
 +----> Q
 +----> K
 +----> V
 |
 v
Attention
```

versus:

```text
RoPE:

E
 |
 v
X
 |
 +----> Q ----> RoPE ----> Q_rot
 |
 +----> K ----> RoPE ----> K_rot
 |
 +----> V ---------------> V
                              |
                              v
                 Attention(Q_rot, K_rot, V)
```

This is the core mental model for understanding RoPE-based Transformer architectures.

---

# 27. Self-Check Questions

1. Why does a Transformer need positional information?
2. What is additive positional encoding?
3. What does `X = E + PE` mean?
4. After adding PE, how are Q, K, and V calculated?
5. What is RoPE?
6. Does RoPE eliminate Q, K, or V?
7. Which of Q, K, and V receive RoPE?
8. Why is V normally not rotated?
9. What is the 2D rotation matrix?
10. How does position determine the rotation angle?
11. Why are dimensions paired in RoPE?
12. Why do different dimension pairs use different frequencies?
13. How does RoPE affect `Q K^T`?
14. Why does RoPE naturally encode relative position?
15. What is the difference between additive PE and RoPE?
16. Where does RoPE appear in a modern Transformer attention block?
17. How is RoPE related to KV Cache?
18. Can you draw the complete flow from token embedding to attention output?
19. Explain: "RoPE modifies Q/K, not X."
20. Why is "RoPE goes directly from embedding to Q and K" incomplete?

---

# 28. Next Lesson

## Lesson 07 - Mixture of Experts (MoE)

Next we will study how some modern LLMs use sparse expert networks instead of sending every token through the same feed-forward network.

```text
Dense Transformer
      |
      v
Every token uses the same FFN
      |
      v
MoE Transformer
      |
      v
Router
      |
      v
Select top-k experts
      |
      v
Only selected experts process each token
      |
      v
Combine expert outputs
```

Topics:

- Dense vs sparse models
- Experts
- Router
- Gating scores
- Top-k routing
- Expert selection
- Load balancing
- Auxiliary routing loss
- Expert capacity
- Active parameters vs total parameters
- MoE inference
- Why MoE can scale model capacity efficiently
