# Lesson 08 — Context Window & Attention Patterns

## Complete LLM Engineering Note

### Core question

How can modern LLMs process long contexts efficiently when standard self-attention has quadratic token-to-token interactions and the KV cache grows with sequence length?

The major concepts are:

```text
Context Window
      ↓
Self-Attention
      ↓
O(n²) attention
      ↓
KV Cache
      ↓
MHA → MQA → GQA
      ↓
MLA
      ↓
Flash Attention
```

---

# 1. Context Window

The **context window** is the maximum number of tokens that a model can consider in one context.

```text
Text
 ↓
Tokenizer
 ↓
Tokens
 ↓
Context Window
```

Context can include:

```text
System instructions
+
Conversation history
+
Retrieved documents
+
Tool results
+
Current user message
```

The context limit is measured in **tokens**, not characters.

---

# 2. Why Context Length Matters

A larger context can contain:

- Long documents
- Codebases
- Long conversations
- Research papers
- Retrieved RAG documents
- Tool outputs
- Agent trajectories

But increasing context creates computational and memory challenges.

---

# 3. Self-Attention Recap

Self-attention is:

$$
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

For \(n\) tokens:

$$
Q\in\mathbb{R}^{n\times d}
$$

and:

$$
K\in\mathbb{R}^{n\times d}
$$

Therefore:

$$
QK^T
$$

has shape:

$$
(n\times d)(d\times n)=n\times n
$$

So the attention matrix has \(n^2\) entries.

---

# 4. Why Attention Is O(n²)

The number of token-to-token interactions grows approximately as:

$$
\boxed{O(n^2)}
$$

Examples:

| Tokens | Attention entries |
|---:|---:|
| 10 | 100 |
| 100 | 10,000 |
| 1,000 | 1,000,000 |
| 10,000 | 100,000,000 |
| 100,000 | 10,000,000,000 |

Therefore, doubling sequence length approximately quadruples the number of pairwise attention interactions.

---

# 5. Numerical Attention Example

For:

$$
n=4
$$

we get:

$$
QK^T\in\mathbb{R}^{4\times4}
$$

Conceptually:

```text
        K1 K2 K3 K4
      ┌─────────────┐
Q1    │ •  •  •  •  │
Q2    │ •  •  •  •  │
Q3    │ •  •  •  •  │
Q4    │ •  •  •  •  │
      └─────────────┘
```

Every query can potentially interact with every key.

For 1,000 tokens:

$$
1000\times1000=1,000,000
$$

attention scores.

---

# 6. Training vs Inference

## Training

Many positions can be processed in parallel:

```text
Input tokens
    ↓
Embedding
    ↓
Q/K/V
    ↓
Attention
    ↓
Loss
    ↓
Backpropagation
```

## Autoregressive inference

Tokens are generated sequentially:

```text
Prompt
  ↓
Token 1
  ↓
Token 2
  ↓
Token 3
  ↓
...
```

This is where the KV cache becomes extremely important.

---

# 7. KV Cache

Suppose the model has already processed:

```text
The cat sat
```

The attention states include:

```text
K1 K2 K3
V1 V2 V3
```

Instead of recomputing these every time a new token is generated, they are cached.

```text
Past tokens
    ↓
Past K/V
    ↓
KV Cache
```

For the new token:

```text
New token
    ↓
Q_new
K_new
V_new
```

The new query can attend to:

```text
K1 K2 K3 K_new
```

using:

```text
V1 V2 V3 V_new
```

---

# 8. KV Cache Memory

For a simplified model with:

- \(L\) = sequence length
- \(h_{KV}\) = number of KV heads
- \(d_h\) = head dimension

the cache contains approximately:

$$
K\in\mathbb{R}^{L\times h_{KV}\times d_h}
$$

and:

$$
V\in\mathbb{R}^{L\times h_{KV}\times d_h}
$$

Therefore, KV-cache memory grows approximately linearly with sequence length:

$$
\boxed{O(L)}
$$

But it can still become very large because the cache exists across many layers and potentially many sequences in a batch.

---

# 9. Multi-Head Attention (MHA)

Traditional Multi-Head Attention uses separate Q, K, and V heads.

For example:

```text
8 Q heads
8 K heads
8 V heads
```

Therefore:

$$
h_Q=h_K=h_V
$$

Conceptually:

```text
Q1 → K1,V1
Q2 → K2,V2
Q3 → K3,V3
...
Q8 → K8,V8
```

This provides many independent attention representations but requires a larger KV cache.

---

# 10. Multi-Query Attention (MQA)

MQA keeps many Q heads but shares one K head and one V head.

Example:

```text
Q:
Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8

K:
K1

V:
V1
```

Therefore:

$$
h_Q=8
$$

but:

$$
h_K=h_V=1
$$

Conceptually:

```text
Q1 ─┐
Q2 ─┤
Q3 ─┤
Q4 ─┤
Q5 ─┤
Q6 ─┤
Q7 ─┤
Q8 ─┘
     ↓
   K1,V1
```

### Advantage

Much smaller KV cache.

### Trade-off

All query heads share the same K/V representation, reducing K/V diversity.

---

# 11. Grouped-Query Attention (GQA)

GQA is between MHA and MQA.

Example:

```text
8 Q heads
4 KV heads
```

Group them:

```text
Q1 Q2 → K1,V1

Q3 Q4 → K2,V2

Q5 Q6 → K3,V3

Q7 Q8 → K4,V4
```

Therefore:

$$
h_Q=8
$$

and:

$$
h_K=h_V=4
$$

---

# 12. MHA vs GQA vs MQA

Suppose there are 8 Q heads.

### MHA

```text
Q = 8
K = 8
V = 8
```

### GQA

```text
Q = 8
K = 4
V = 4
```

### MQA

```text
Q = 8
K = 1
V = 1
```

Therefore:

$$
\boxed{
MHA \rightarrow GQA \rightarrow MQA
}
$$

moves toward progressively more K/V sharing.

---

# 13. Numerical KV-Cache Comparison

Suppose:

```text
Layers = 32
Sequence length = 10,000
Head dimension = 128
Q heads = 32
```

Ignore datatype and batch size.

### MHA

If:

```text
KV heads = 32
```

the approximate number of cached scalar values is:

$$
2\times32\times10,000\times32\times128
$$

The first 2 represents K and V.

### GQA

If:

```text
KV heads = 8
```

then:

$$
2\times8\times10,000\times32\times128
$$

The ratio is:

$$
\frac{8}{32}=\frac14
$$

So GQA in this example uses approximately one-quarter of the KV storage of MHA.

Thus:

$$
\boxed{\text{About 4× less KV storage}}
$$

for the same sequence length, layers, and head dimension.

---

# 14. Why GQA Is Useful

GQA provides a middle ground:

```text
MHA
 ↓
Many independent K/V heads
 ↓
Large KV cache


GQA
 ↓
Several K/V groups
 ↓
Smaller KV cache
 ↓
Good representation diversity


MQA
 ↓
One K/V group
 ↓
Smallest KV cache
```

This makes GQA especially useful for efficient LLM inference.

---

# 15. Multi-Head Latent Attention (MLA)

**Multi-Head Latent Attention (MLA)** uses a different approach to reducing KV-cache requirements.

At a high level:

```text
Hidden representation
        ↓
KV compression / latent representation
        ↓
Store compressed latent
        ↓
Use latent representation during attention
```

The objective is:

$$
\boxed{\text{Reduce KV-cache memory while preserving useful attention information}}
$$

MLA is more sophisticated than simply grouping K/V heads.

---

# 16. Flash Attention

Flash Attention is different from GQA and MQA.

It is an **efficient exact attention algorithm**.

The mathematical attention remains:

$$
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

Flash Attention changes **how this calculation is performed**, not the definition of attention.

---

# 17. Why Flash Attention Helps

A naive implementation may materialize the large \(n\times n\) attention matrix in GPU memory.

Conceptually:

```text
Q
 ↓
QKᵀ
 ↓
Large attention matrix
 ↓
Softmax
 ↓
Multiply by V
```

Flash Attention uses blocking/tiling and efficient memory access so that the full attention matrix does not need to be naively materialized in high-bandwidth memory.

Conceptually:

```text
Q/K/V
  ↓
Split into blocks
  ↓
Block-wise attention computation
  ↓
Efficient online softmax / accumulation
  ↓
Output
```

The result is mathematically equivalent to exact attention, subject to normal floating-point implementation details.

---

# 18. Flash Attention vs GQA

This distinction is essential.

### GQA

Changes the architecture:

```text
Many Q heads
+
Fewer K/V heads
```

Goal:

```text
Smaller KV cache
```

### Flash Attention

Changes the implementation:

```text
Compute attention efficiently
```

Goal:

```text
Less memory traffic
Better GPU utilization
Lower memory usage for attention computation
```

So:

$$
\boxed{
GQA=\text{attention architecture}
}
$$

while:

$$
\boxed{
Flash\ Attention=\text{efficient attention algorithm}
}
$$

---

# 19. Context Window vs KV Cache

Do not confuse these.

### Context Window

How many tokens the model can consider.

```text
Context Window
      ↓
Maximum context tokens
```

### KV Cache

Stores previous K/V states during autoregressive inference.

```text
KV Cache
    ↓
Stored past attention states
```

They are related but not identical.

---

# 20. Prefill vs Decode

Modern inference is commonly divided into two phases.

## Prefill

The entire prompt is processed:

```text
20,000-token prompt
        ↓
Transformer
        ↓
Compute K/V
        ↓
KV Cache
```

Prefill is often compute-heavy because many prompt tokens are processed together.

## Decode

New tokens are generated one at a time:

```text
KV Cache + new token
        ↓
Transformer
        ↓
Next token
        ↓
Update KV Cache
        ↓
Repeat
```

Decode is often strongly affected by memory bandwidth and KV-cache access.

---

# 21. Why GQA Helps Decode

Suppose:

```text
Q heads = 32
KV heads = 8
```

Each KV head serves multiple Q heads.

Instead of storing:

```text
32 K heads
32 V heads
```

we store:

```text
8 K heads
8 V heads
```

Therefore:

```text
Fewer K/V states
      ↓
Smaller cache
      ↓
Less memory traffic
      ↓
More efficient decoding
```

---

# 22. Long Context Challenges

Long context introduces several challenges:

### Compute

Naive attention has:

$$
O(n^2)
$$

pairwise interactions.

### Memory

Large attention intermediates and KV caches require memory.

### Latency

More tokens require more processing.

### Cost

More computation and memory increase serving cost.

### Quality

A larger context window does not automatically mean the model can perfectly use every token.

---

# 23. Context Window and RAG

This is especially important for RAG.

A naive RAG pipeline might retrieve too many chunks:

```text
Query
 ↓
Retrieve 100 chunks
 ↓
Put all 100 into prompt
 ↓
LLM
```

This can cause:

```text
Large prompt
   ↓
Higher token usage
   ↓
Higher latency
   ↓
Larger KV cache
   ↓
Potentially worse information utilization
```

A better pipeline often looks like:

```text
Query
  ↓
Retrieval
  ↓
Reranking
  ↓
Select best chunks
  ↓
Context compression / organization
  ↓
LLM
```

This becomes important later when learning RAG and context engineering.

---

# 24. Efficient Attention — Big Picture

Think of the optimizations as solving different problems:

```text
                 Efficient LLM Attention
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   Architecture         KV Cache          Computation
        │                  │                  │
     GQA/MQA              MLA           Flash Attention
```

---

# 25. RoPE, MHA, GQA, MQA, MLA, Flash Attention

Do not confuse these technologies.

| Technology | Main purpose |
|---|---|
| RoPE | Position information |
| MHA | Multiple independent Q/K/V heads |
| MQA | One shared K/V pair across Q heads |
| GQA | Groups of Q heads share K/V heads |
| MLA | Compress/restructure KV representation |
| Flash Attention | Compute attention efficiently |
| KV Cache | Store past K/V during generation |
| Context Window | Maximum context tokens |

---

# 26. Complete Modern Attention Picture

```text
                         Token
                           │
                           ▼
                      Embedding
                           │
                           ▼
                       Q, K, V
                           │
                           ▼
                         RoPE
                    (Q and K)
                           │
                           ▼
                 Attention Architecture
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
             MHA          GQA          MQA
              │            │            │
          many K/V      grouped K/V   shared K/V
              │            │            │
              └────────────┼────────────┘
                           ▼
                  Efficient Attention
                           │
                    Flash Attention
                           │
                           ▼
                       Output
                           │
                           ▼
                    KV Cache
                  (during inference)
```

The exact ordering and implementation vary by model architecture.

---

# 27. Training vs Inference

## Training

```text
Input tokens
    ↓
Embedding
    ↓
Q/K/V
    ↓
RoPE
    ↓
Attention
    ↓
FFN / MoE
    ↓
Loss
    ↓
Backpropagation
```

Many positions can be processed in parallel.

## Inference

```text
Prompt
  ↓
Prefill
  ↓
Compute K/V
  ↓
Store KV Cache
  ↓
Generate next token
  ↓
New Q/K/V
  ↓
Use cached K/V
  ↓
Generate next token
  ↓
Repeat
```

---

# 28. One Important Conceptual Separation

Remember these three ideas:

```text
RoPE
 ↓
Where is the token?
 ↓
Position information


GQA / MQA / MLA
 ↓
How much K/V information must be stored?
 ↓
KV-cache efficiency


Flash Attention
 ↓
How can attention be computed efficiently?
 ↓
Memory-efficient computation
```

This separation will prevent many architecture misconceptions.

---

# 29. Final Mental Model

```text
                     Decoder-only LLM
                            │
                            ▼
                       Embeddings
                            │
                            ▼
                          Q/K/V
                            │
                            ▼
                           RoPE
                            │
                            ▼
                     Attention Design
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
             MHA           GQA           MQA
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                      Flash Attention
                            │
                            ▼
                         Output
                            │
                            ▼
                       KV Cache
                    during inference
```

---

# 30. Essential Formulas

### Attention

$$
\boxed{
Attention(Q,K,V)=
softmax\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
}
$$

### Attention matrix shape

For \(n\) tokens:

$$
\boxed{
QK^T\in\mathbb{R}^{n\times n}
}
$$

### Naive attention complexity

$$
\boxed{O(n^2)}
$$

### KV cache shape

Approximately:

$$
\boxed{
K,V\in\mathbb{R}^{L\times h_{KV}\times d_h}
}
$$

per layer, ignoring batch and implementation-specific layouts.

### GQA cache ratio

If MHA has \(h\) KV heads and GQA has \(g\):

$$
\boxed{
\frac{\text{GQA KV storage}}
{\text{MHA KV storage}}
\approx
\frac{g}{h}
}
$$

assuming the other dimensions are equal.

---

# 31. What You Must Remember

### Context Window

$$
\boxed{\text{Maximum number of context tokens}}
$$

### Attention

$$
\boxed{QK^T\rightarrow n\times n}
$$

therefore naive attention has quadratic token-interaction growth.

### MHA

```text
Many Q
Many K
Many V
```

### MQA

```text
Many Q
One K
One V
```

### GQA

```text
Many Q
Several K
Several V
```

### MLA

```text
Compress/restructure KV representation
```

### Flash Attention

```text
Compute the same attention more efficiently
```

### KV Cache

```text
Store past K/V during autoregressive generation
```

---

# 32. Self-Check Questions

1. What is a context window?
2. Why is context measured in tokens?
3. Why does naive self-attention have \(O(n^2)\) token interactions?
4. What is the shape of \(QK^T\) for \(n\) tokens?
5. Why does long context increase memory requirements?
6. What is the KV cache?
7. Why is KV cache important during inference?
8. What is Multi-Head Attention?
9. What is Multi-Query Attention?
10. What is Grouped-Query Attention?
11. What is the difference between MHA, GQA, and MQA?
12. Why does GQA reduce KV-cache memory?
13. What is MLA?
14. What problem does MLA attempt to solve?
15. What is Flash Attention?
16. Does Flash Attention change the mathematical definition of attention?
17. What is the difference between Flash Attention and GQA?
18. What is prefill?
19. What is decode?
20. Why is GQA useful during decoding?
21. How are RoPE, GQA, MLA, and Flash Attention different?
22. Why does a larger context window not automatically mean better model performance?
23. Why is context management important in RAG?

---

# 33. Next Lesson — Pre-training Objectives

The next lesson explains **how an LLM actually learns language**.

Topics:

```text
Training objective
      ↓
Next-token prediction
      ↓
Causal Language Modeling
      ↓
Cross-entropy loss
      ↓
Teacher-forcing style training
      ↓
Masked Language Modeling
      ↓
Bidirectional vs causal attention
      ↓
GPT-style vs BERT-style training
      ↓
Why decoder-only LLMs predict the next token
```

The key transition is:

```text
Transformer Architecture
        ↓
Attention
        ↓
LLM Architecture
        ↓
Training Objective
        ↓
Next-token prediction
        ↓
Loss
        ↓
Backpropagation
        ↓
Learned language model
```
