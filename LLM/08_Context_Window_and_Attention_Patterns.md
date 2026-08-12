# Lesson 8: Context Window & Attention Patterns

Welcome to Lesson 8 of the AI Engineering Interview Preparation series. In this lesson, we will dive deep into the heart of modern Large Language Models (LLMs): the Attention Mechanism. Specifically, we will explore how models manage large context windows, the evolution of attention patterns from standard Multi-Head Attention to Multi-Query and Grouped-Query Attention, the hardware-aware optimizations introduced by Flash Attention, and the cutting-edge Multi-head Latent Attention (MLA) popularized by models like DeepSeek.

Understanding these concepts is absolutely critical for AI Engineering interviews, as they dictate the memory footprint, inference speed, and overall efficiency of deploying LLMs in production.

---

## 1. The Bottleneck: The KV Cache

Before discussing attention patterns, we must understand the core problem they aim to solve: the **KV Cache** memory bottleneck during inference.

In autoregressive generation, a model predicts the next token based on all previous tokens. To avoid recomputing the attention scores for past tokens at every step, we store their Key ($K$) and Value ($V$) tensors in memory. This is called the KV Cache.

### Math Formulation
Let $x \in \mathbb{R}^{d}$ be an input token embedding. We compute Queries ($Q$), Keys ($K$), and Values ($V$) via linear projections:
$$
Q = x W_Q, \quad K = x W_K, \quad V = x W_V
$$

The attention output is computed as:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

During generation, for the $t$-th token, we only need a single query $q_t$, but we must attend to all previous keys and values: $K_{1:t}$ and $V_{1:t}$.

### Memory Complexity
For a model with $L$ layers, $h$ heads, and a head dimension $d_k$, the size of the KV cache for a context window of length $N$ is:
$$
\text{Memory}_{\text{KV}} = 2 \times N \times L \times h \times d_k \times \text{bytes\_per\_parameter}
$$
*(The factor of 2 accounts for both Keys and Values).*

As the context window $N$ grows (e.g., to 128k or 1M tokens), the KV cache becomes enormous, often exceeding the memory required for the model weights themselves, severely limiting batch sizes and throughput.

---

## 2. Multi-Head Attention (MHA) vs. MQA vs. GQA

To mitigate the KV cache bottleneck, researchers have proposed structural variations to the attention mechanism.

### 2.1 Multi-Head Attention (MHA)

In standard MHA (introduced in "Attention Is All You Need"), there are $h$ Query heads, $h$ Key heads, and $h$ Value heads. Each Query head attends to its own corresponding Key and Value head.

**Characteristics:**
- High expressive power.
- Very high memory usage for the KV cache.
- Low arithmetic intensity during decoding (memory bandwidth bound).

**ASCII Diagram:**
```text
Queries:  [ Q1 ] [ Q2 ] [ Q3 ] [ Q4 ]
Keys:     [ K1 ] [ K2 ] [ K3 ] [ K4 ]
Values:   [ V1 ] [ V2 ] [ V3 ] [ V4 ]
```

### 2.2 Multi-Query Attention (MQA)

Multi-Query Attention drastically reduces the KV cache size by sharing a *single* Key head and a *single* Value head across all $h$ Query heads.

**Characteristics:**
- KV cache size is reduced by a factor of $h$.
- Greatly increases inference throughput and allows larger batch sizes.
- Can lead to a noticeable drop in model quality/capacity compared to MHA.

**ASCII Diagram:**
```text
Queries:  [ Q1 ] [ Q2 ] [ Q3 ] [ Q4 ]
             \      |      |      /
Keys:        -------[ K_shared ]-------
             -------[ V_shared ]-------
Values: 
```

### 2.3 Grouped-Query Attention (GQA)

Grouped-Query Attention interpolates between MHA and MQA. It divides the $h$ Query heads into $G$ groups. Each group shares a single Key and Value head.
If $G = h$, it is MHA. If $G = 1$, it is MQA. Typical values are $G = 8$ (e.g., Llama 2 uses 32 Q heads and 8 KV heads).

**Characteristics:**
- Strikes a balance between the high quality of MHA and the fast inference of MQA.
- KV cache size is reduced by a factor of $h/G$.
- Currently the industry standard for state-of-the-art open weights models (Llama 3, Mistral, etc.).

**ASCII Diagram (G=2, h=4):**
```text
Group 1:                   Group 2:
Queries:  [ Q1 ] [ Q2 ]    Queries:  [ Q3 ] [ Q4 ]
             \      /                   \      /
Keys:        [ K_g1 ]                   [ K_g2 ]
Values:      [ V_g1 ]                   [ V_g2 ]
```

---

## 3. Hardware-Aware Optimizations: Flash Attention

While MQA/GQA solve the KV cache size problem structurally, **Flash Attention** solves the attention computation problem algorithmically, making it hardware-aware.

### 3.1 The Memory Wall

In modern GPUs, computing is extremely fast (FLOPs), but moving data to the compute units is slow. GPUs have a memory hierarchy:
1. **SRAM (Shared Memory):** Very fast, but very small (e.g., 20MB per GPU).
2. **HBM (High Bandwidth Memory):** Slower, but large (e.g., 80GB on an A100).

Standard attention involves computing the attention matrix $S = Q K^T$ and materializing it in HBM. For a sequence length $N$, this matrix requires $O(N^2)$ memory.
This $O(N^2)$ HBM read/write operation creates a massive bottleneck.

### 3.2 Tiling and Avoiding Materialization

Flash Attention computes the exact same output as standard attention but without ever materializing the $O(N^2)$ attention matrix in HBM. It uses a technique called **Tiling**.

1. Load blocks of $Q$, $K$, and $V$ from HBM into the fast SRAM.
2. Compute the attention scores for that block entirely in SRAM.
3. Update the output incrementally.
4. Write the final output back to HBM.

Because the softmax function requires the denominator (the sum of exponentiated scores over the entire row), calculating it block-by-block seems impossible. Flash Attention uses the **online softmax trick**: it keeps track of running maximums and normalizers, allowing mathematically exact softmax computation in a single pass.

### 3.3 Complexity and Benefits

- **Standard Attention HBM accesses:** $O(N d + N^2)$
- **Flash Attention HBM accesses:** $O(N d)$

By reducing memory reads/writes, Flash Attention achieves a 2x-4x speedup during training and dramatically reduces VRAM usage, enabling the massive context windows (100k+) we see today.

**ASCII Diagram of Tiling:**
```text
HBM (Global Memory)                           SRAM (Fast, On-Chip)
+-------------------+                         +-------------------+
|  Q, K, V (N x d)  |  --- Block Load --->    |  q_block, k_block |
|                   |                         |  v_block          |
|                   |                         |                   |
|                   |  <--- Write Output --   | Compute Softmax   |
+-------------------+                         +-------------------+
```

---

## 4. Multi-Head Latent Attention (MLA)

Multi-head Latent Attention (MLA) is a novel architecture introduced in models like DeepSeek-V2 and DeepSeek-V3. It aims to achieve the inference efficiency of MQA while maintaining (or exceeding) the expressiveness of MHA.

### 4.1 The Mechanism of MLA

Instead of caching the full Keys and Values, MLA projects the input token embedding $x_t$ into a low-dimensional **latent vector** $c_t \in \mathbb{R}^{d_c}$, where $d_c \ll d \times h$.

$$
c_t = x_t W_{c}
$$

During inference, we **only cache this latent vector $c_t$**, which acts as a compressed representation of both Key and Value.

When generating the next token, the model needs the Keys and Values. It computes them by up-projecting the cached latent vector:
$$
k_{t, h} = c_t W_{UK, h}, \quad v_{t, h} = c_t W_{UV, h}
$$

However, doing this naive up-projection would defeat the purpose, as it would cost compute and memory bandwidth at every step.

### 4.2 The ROPE Decoupling and Matrix Absorption Trick

The brilliance of MLA lies in how it interacts with Rotary Position Embeddings (RoPE).
If we directly substitute the up-projection into the attention equation, we can absorb the projection matrices into the query projection, avoiding the need to explicitly compute $K$ and $V$.

Let $q_h = x W_{Q, h}$. The attention score before softmax is:
$$
\text{score} = q_h k_h^T = q_h (c W_{UK, h})^T = (q_h W_{UK, h}^T) c^T = q'_h c^T
$$
Where $q'_h$ is the absorbed query. Thus, we can compute attention directly between the transformed query and the cached latent vector!

**The RoPE Complication:**
RoPE requires applying position-dependent rotations to $Q$ and $K$. Because RoPE is non-linear with respect to matrix multiplication, we cannot absorb the up-projection matrix if RoPE is applied *after* up-projection.

**The Solution:**
DeepSeek decouples the Keys into two parts:
1. A content-based part generated from the latent vector $c_t$ (without RoPE).
2. A separate, very small Key vector explicitly for RoPE: $k^{R}_t$.

Thus, the KV Cache for MLA consists only of:
$$
\text{KV Cache} = \{ c_t, k^{R}_t \}
$$

### 4.3 Memory Savings

By compressing the KV cache into a low-dimensional latent space, MLA achieves a KV cache size significantly smaller than even MQA, while empirically outperforming MQA in reasoning and generation tasks.

---

## 5. Typical Interview Questions

Here are some typical questions you might face regarding context windows and attention patterns, along with guidelines for answering.

### Q1: Why does inference batch size decrease as the sequence length increases in standard Transformers?
**Answer:** The primary reason is the KV Cache. For standard Multi-Head Attention, the KV cache grows linearly with both sequence length and batch size. As sequence length increases, the memory required to store the KV cache for a single sequence balloons. Because GPU VRAM is strictly limited, to fit the larger KV cache of longer sequences, you must decrease the batch size to avoid OOM (Out of Memory) errors.

### Q2: Compare and contrast MHA, MQA, and GQA. When would you use which?
**Answer:**
- **MHA:** 1 Q head to 1 KV head. Highest quality, highest memory footprint. Used in older models (GPT-3, early Llama) or when KV cache size isn't a bottleneck (e.g., short contexts).
- **MQA:** N Q heads to 1 KV head. Fastest inference, smallest memory footprint, but can suffer quality degradation. Used when throughput and large batch sizes are the absolute top priority (e.g., some coding assistants).
- **GQA:** N Q heads to G KV heads (where 1 < G < N). The sweet spot. Provides nearly the quality of MHA with nearly the speed of MQA. This is the current default for modern LLMs (Llama-3, Mixtral).

### Q3: How does Flash Attention speed up the model if the number of FLOPs remains exactly the same?
**Answer:** Flash Attention is an IO-aware algorithm. In standard attention, the bottleneck isn't the compute (FLOPs); it's the memory bandwidth—the time it takes to read/write the $O(N^2)$ attention matrix from the slow High Bandwidth Memory (HBM) to the fast SRAM. Flash Attention uses tiling to compute the attention block-by-block directly in SRAM, bypassing the need to materialize the $O(N^2)$ matrix in HBM. By drastically reducing memory IO, it achieves massive speedups.

### Q4: Can you explain the "online softmax" trick conceptually?
**Answer:** In standard softmax, you need the sum of $e^{x_i}$ for all elements in a row to calculate the denominator. This implies a two-pass algorithm: one to find the max (for numerical stability) and compute the sum, and another to divide. In Flash Attention's tiling, we process the row in chunks. The online softmax trick allows us to keep track of a "running maximum" and a "running sum" for the chunks seen so far. When a new chunk arrives, we can mathematically rescale the previous running sum based on the new maximum, allowing us to compute exact softmax in a single pass without storing the intermediate exponentiated scores.

### Q5: What is the core innovation of DeepSeek's Multi-head Latent Attention (MLA)?
**Answer:** MLA solves the KV cache bottleneck by compressing the Keys and Values into a single, low-dimensional latent vector. During inference, only this latent vector is cached. The model uses mathematical matrix absorption to compute attention scores directly against this latent vector without explicitly decompressing it into full Keys and Values at every step. To support Rotary Position Embeddings (RoPE), which break matrix absorption, MLA decouples a small, separate RoPE-specific key vector that is cached alongside the latent vector.

---
*End of Lesson 8*
