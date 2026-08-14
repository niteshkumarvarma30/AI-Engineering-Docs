# Positional Encoding vs RoPE — Complete Conceptual Note

## Core Idea

Traditional additive positional encoding:

```text
Token IDs
   ↓
Token Embedding
   ↓
Add Positional Encoding
   ↓
X
   ↓
Q, K, V
   ↓
Self-Attention
```

Mathematically:

$$X=E+PE$$

RoPE:

```text
Token IDs
   ↓
Token Embedding
   ↓
X
   ↓
Q, K, V
   ↓
RoPE applied to Q and K
   ↓
Q_rot, K_rot, V
   ↓
Self-Attention
```

The key difference is:

$$\boxed{\text{Additive PE modifies }X}$$

while:

$$\boxed{\text{RoPE modifies }Q\text{ and }K}$$

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
        ↓
Token IDs
        ↓
Embedding Lookup
        ↓
Token Embeddings X
```

The model then creates:

$$Q=XW_Q$$

$$K=XW_K$$

$$V=XW_V$$

---

# 2. Why Position Is Needed

Compare:

```text
The dog chased the cat
```

and:

```text
The cat chased the dog
```

The same words can occur, but their order changes the meaning.

Self-attention by itself needs an explicit positional mechanism.

---

# 3. Additive Positional Encoding

A traditional approach adds position to the token representation:

$$\boxed{X=E+PE}$$

Then:

$$Q=XW_Q$$

$$K=XW_K$$

$$V=XW_V$$

and:

$$Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

---

# 4. Numerical Additive PE Example

Suppose:

$$E_{cats}=[1,1,0,0]$$

and position 2 has:

$$PE_2=[0.3,0.3,0.3,0.3]$$

Then:

$$X_{cats}=E_{cats}+PE_2$$

$$=[1,1,0,0]+[0.3,0.3,0.3,0.3]$$

Therefore:

$$\boxed{X_{cats}=[1.3,1.3,0.3,0.3]}$$

This representation is then used to create Q, K, and V.

---

# 5. RoPE

RoPE means **Rotary Position Embeddings**.

Instead of adding a positional vector to X, RoPE applies a position-dependent rotation to Q and K.

Correct flow:

```text
                 Token Embedding X
                         │
             ┌───────────┼───────────┐
             ↓           ↓           ↓
            Q            K           V
             ↓           ↓           │
           RoPE         RoPE         │
             ↓           ↓           │
          Q_rot        K_rot          │
             └──────┬────┘           │
                    ↓                │
                 Q_rot K_rotᵀ        │
                    ↓                │
                  Softmax             │
                    ↓                │
             Attention Weights ──────┘
                    ↓
                Weighted V
```

Important:

> RoPE does **not** mean that V disappears.

You still calculate Q, K, and V.

Only Q and K receive the rotary transformation.

---

# 6. Q, K, V Intuition

### Query

> What information am I looking for?

### Key

> How should another token match against me?

### Value

> What information should I provide if I am attended to?

Therefore:

```text
Q + K
  ↓
Attention weights

Attention weights + V
  ↓
Attention output
```

RoPE modifies the Q/K relationship.

---

# 7. Why RoPE Is Applied to Q and K

Attention scores come from:

$$QK^T$$

RoPE changes this to:

$$Q_{rot}K_{rot}^T$$

where Q and K have position-dependent rotations.

Thus position directly influences the attention relationship.

---

# 8. 2D Rotation

Start with:

$$q=\begin{bmatrix}x\\y\end{bmatrix}$$

A 2D rotation matrix is:

$$
R(\theta)=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}
$$

Then:

$$q'=R(\theta)q$$

---

# 9. Numerical Rotation Example

Suppose:

$$q=\begin{bmatrix}1\\0\end{bmatrix}$$

and:

$$\theta=90^\circ$$

Then:

$$
R(90^\circ)=
\begin{bmatrix}
0&-1\\
1&0
\end{bmatrix}
$$

Therefore:

$$
q'=
\begin{bmatrix}
0&-1\\
1&0
\end{bmatrix}
\begin{bmatrix}
1\\
0
\end{bmatrix}
=
\begin{bmatrix}
0\\
1
\end{bmatrix}
$$

So:

$$\boxed{q'=[0,1]}$$

---

# 10. Position Controls the Rotation

Suppose:

$$q=[1,0]$$

and:

$$\theta=30^\circ$$

Position 0:

$$q_0=R(0^\circ)q=[1,0]$$

Position 1:

$$q_1=R(30^\circ)q=[0.866,0.5]$$

Position 2:

$$q_2=R(60^\circ)q=[0.5,0.866]$$

Therefore:

```text
Same underlying vector
        ↓
Different position
        ↓
Different rotation
        ↓
Different Q/K representation
```

The same position-dependent operation is applied to K.

---

# 11. Relative Position Intuition

Suppose:

```text
Query position = 5
Key position   = 3
```

Relative distance:

$$5-3=2$$

Another pair:

```text
Query position = 10
Key position   = 8
```

Relative distance:

$$10-8=2$$

The mathematical structure of RoPE makes the Q/K interaction naturally sensitive to relative positional differences.

Conceptually:

$$\boxed{RoPE\ makes\ attention\ sensitive\ to\ relative\ position}$$

---

# 12. RoPE Does Not Mean "Skip V"

This is a common misunderstanding.

Incorrect:

```text
Embedding
   ↓
Q, K
   ↓
Attention
```

Correct:

```text
Embedding X
     │
     ├──→ Q ──→ RoPE ──→ Q_rot
     │
     ├──→ K ──→ RoPE ──→ K_rot
     │
     └──→ V ─────────────→ V
```

Then:

```text
Q_rot + K_rot
       ↓
Attention scores
       ↓
Softmax
       ↓
Attention weights
       ↓
V
       ↓
Attention output
```

---

# 13. Why V Is Normally Not Rotated

Attention has two conceptual stages.

### Stage 1

$$Q+K\rightarrow Attention\ weights$$

### Stage 2

$$Attention\ weights+V\rightarrow Output$$

RoPE is used to encode positional relationships in Stage 1.

Therefore the standard formulation applies it to Q and K, not V.

---

# 14. Additive PE vs RoPE

## Additive PE

```text
Token IDs
   ↓
Embedding E
   ↓
E + PE
   ↓
X
   ↓
Q, K, V
   ↓
Attention
```

Mathematically:

$$X=E+PE$$

---

## RoPE

```text
Token IDs
   ↓
Embedding X
   ↓
Q, K, V
   ↓
Q → RoPE
K → RoPE
V → unchanged
   ↓
Attention
```

Mathematically:

$$Q=XW_Q$$

$$K=XW_K$$

$$V=XW_V$$

then:

$$Q_{rot}=R(pos)Q$$

$$K_{rot}=R(pos)K$$

and:

$$\boxed{Attention(Q_{rot},K_{rot},V)}$$

---

# 15. Side-by-Side

```text
       ADDITIVE PE                         RoPE

      Token IDs                           Token IDs
          ↓                                  ↓
      Embedding                            Embedding
          ↓                                  ↓
          E                                  X
          │                                  │
          +                                  ├──→ Q
          │                                  │     ↓
         PE                                  │   RoPE
          │                                  │
          ↓                                  ├──→ K
          X                                  │     ↓
          │                                  │   RoPE
       ┌──┼──┐                               │
       ↓  ↓  ↓                               └──→ V
       Q  K  V                                    │
       │  │  │                                    │
       └──┼──┘                                    │
          ↓                                       │
      Attention                         Q_rot, K_rot, V
                                                  ↓
                                             Attention
```

---

# 16. RoPE in Multi-Head Attention

Suppose:

```text
Hidden size = 4096
Number of heads = 32
Head dimension = 128
```

Each head has its own Q/K/V representations.

Conceptually:

```text
Hidden representation
        ↓
   Multi-Head Q/K/V
        ↓
   ┌────┼────┐
   ↓    ↓    ↓
 Head1 Head2 ... Head32
   ↓    ↓         ↓
  QKV  QKV       QKV
   ↓    ↓         ↓
  RoPE RoPE      RoPE
   ↓    ↓         ↓
Attention Attention
```

RoPE operates on the relevant Q/K dimensions within each head.

---

# 17. RoPE Works on Dimension Pairs

Suppose a head has dimension 8:

```text
[x1, x2, x3, x4, x5, x6, x7, x8]
```

Conceptually group it as:

```text
(x1, x2)
(x3, x4)
(x5, x6)
(x7, x8)
```

Each pair behaves like a 2D vector and can be rotated.

Different pairs use different frequencies.

---

# 18. Different Frequencies

Different dimension pairs use different angular frequencies.

Conceptually:

```text
Pair 1 → θ₁
Pair 2 → θ₂
Pair 3 → θ₃
...
```

This allows the representation to encode positional patterns at different scales.

Some components vary faster with position and others more slowly.

---

# 19. Real LLMs

The examples use two dimensions because the rotation is easy to visualize.

Real models may have head dimensions such as:

```text
64
96
128
...
```

RoPE works across pairs of dimensions.

---

# 20. RoPE and KV Cache

This connects directly to inference.

During generation:

```text
Previous tokens
      ↓
Past K and V
      ↓
RoPE on K
      ↓
Cached K
```

For a new token:

```text
New token
   ↓
Q, K, V
   ↓
RoPE on Q and K
   ↓
New Q, New K
```

Then:

```text
New Q
  ↓
attends to cached K
  ↓
cached/new V
```

The exact implementation depends on the architecture, but the high-level relationship is:

$$\boxed{KV\ Cache=stored\ past\ Key/Value\ states}$$

---

# 21. Complete Modern Attention Flow

```text
                       Token IDs
                           │
                           ▼
                    Token Embeddings
                           │
                           ▼
                           X
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
              Q            K            V
              │            │            │
              ▼            ▼            │
            RoPE         RoPE           │
              │            │            │
              ▼            ▼            │
            Q_rot        K_rot          │
              │            │            │
              └──────┬─────┘            │
                     ▼                  │
                 Q_rot K_rotᵀ          │
                     │                  │
                     ▼                  │
               Divide by √dₖ           │
                     │                  │
                     ▼                  │
                 Causal Mask            │
                     │                  │
                     ▼                  │
                   Softmax              │
                     │                  │
                     ▼                  │
              Attention Weights         │
                     │                  │
                     └─────────┬────────┘
                               ▼
                          Weighted V
                               │
                               ▼
                       Output Projection
                               │
                               ▼
                            Residual
```

---

# 22. Mathematical Summary

### Additive PE

$$\boxed{X=E+PE}$$

Then:

$$Q=XW_Q$$

$$K=XW_K$$

$$V=XW_V$$

and:

$$
Attention=
softmax\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

---

### RoPE

Start with token representation:

$$X=E$$

Then:

$$Q=XW_Q$$

$$K=XW_K$$

$$V=XW_V$$

Apply position-dependent rotations:

$$\boxed{Q_{rot}=R(pos)Q}$$

$$\boxed{K_{rot}=R(pos)K}$$

Then:

$$
\boxed{
Attention=
softmax\left(
\frac{Q_{rot}K_{rot}^T}{\sqrt{d_k}}
\right)V
}
$$

For a decoder-only LLM, the causal mask is also applied.

---

# 23. The Most Important Difference

Remember:

```text
Traditional additive PE:

Position
   ↓
added to X
   ↓
Q/K/V
   ↓
Attention
```

versus:

```text
RoPE:

X
   ↓
Q/K/V
   ↓
Position rotates Q/K
   ↓
Attention
```

Therefore:

$$\boxed{\text{Additive PE → position enters }X}$$

$$\boxed{\text{RoPE → position enters Q/K}}$$

---

# 24. Important Precision

The statement:

> "In RoPE, we directly go to Q and K after embedding."

is incomplete.

The correct statement is:

> **After embedding, the model creates Q, K, and V. RoPE is then applied to Q and K before the attention calculation.**

Correct:

```text
Embedding X
     │
     ├──→ Q ──→ RoPE ──→ Q_rot
     │
     ├──→ K ──→ RoPE ──→ K_rot
     │
     └──→ V ─────────────→ V
```

Then:

```text
Q_rot + K_rot
       ↓
Attention scores
       ↓
Softmax
       ↓
Attention weights
       ↓
V
       ↓
Attention output
```

---

# 25. Final Mental Model

```text
                 TEXT
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
                  X
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
        Q         K         V
        │         │         │
       RoPE      RoPE       │
        │         │         │
        ▼         ▼         │
      Q_rot     K_rot       V
        │         │         │
        └────┬────┘         │
             ▼              │
        Attention Scores    │
             │              │
             ▼              │
       Causal Mask          │
             │              │
             ▼              │
          Softmax           │
             │              │
             ▼              │
      Attention Weights ────┘
             │
             ▼
        Weighted V
             │
             ▼
      Output Projection
             │
             ▼
          Residual
             │
             ▼
          FFN etc.
```

---

# 26. Final Takeaway

If you remember only one thing:

> **In additive positional encoding, positional information is added to the input representation X before Q/K/V projections. In a RoPE-based Transformer, Q, K, and V are still all created from X, but RoPE applies position-dependent rotations to Q and K before QKᵀ is calculated. V is normally left unrotated.**

In compact mathematical form:

$$
\boxed{
\text{Additive PE: }E+PE\rightarrow X\rightarrow Q,K,V
}
$$

versus:

$$
\boxed{
\text{RoPE: }E\rightarrow Q,K,V\rightarrow(Q_{rot},K_{rot},V)
}
$$

This is the correct mental model for modern RoPE-based decoder-only LLMs.

---

# 27. Self-Check Questions

1. Why does a Transformer need positional information?
2. What is additive positional encoding?
3. What does \(X=E+PE\) mean?
4. After adding PE, how are Q, K, and V calculated?
5. What is RoPE?
6. Does RoPE eliminate Q, K, or V?
7. Which of Q, K, and V receive RoPE?
8. Why is V normally not rotated?
9. What is the 2D rotation matrix?
10. How does position determine the rotation angle?
11. Why are dimensions paired in RoPE?
12. Why do different dimension pairs use different frequencies?
13. How does RoPE affect \(QK^T\)?
14. Why does RoPE naturally encode relative position?
15. What is the difference between additive PE and RoPE?
16. Where does RoPE appear in a modern Transformer block?
17. How is RoPE related to KV Cache?
18. Can you draw the complete flow from token embedding to attention output?
19. Explain: "RoPE modifies Q/K, not X."
20. Why is "RoPE goes directly from embedding to Q and K" incomplete?

---

# 28. Next Lesson

## Lesson 07 — Mixture of Experts (MoE)

Next:

```text
Dense Transformer
      ↓
Every token uses the same FFN
      ↓
MoE Transformer
      ↓
Router
      ↓
Select top-k experts
      ↓
Only selected experts process each token
      ↓
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
