# Lesson 6: Positional Encodings in Large Language Models

Welcome to Lesson 6 of the AI Engineering Interview Preparation course. In this lesson, we will dive deep into **Positional Encodings** (PE), an absolutely crucial component of the Transformer architecture. Without positional encodings, a Transformer is essentially a bag-of-words model. 

---

## 1. The Core Problem: Permutation Equivariance of Attention

### 1.1 The Nature of Self-Attention
To understand why positional encodings exist, we first need to look at the core self-attention equation:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

In this operation, each token computes its attention score with every other token using dot products between the queries ($Q$) and keys ($K$). It then computes a weighted sum of the values ($V$). 

Notice what is missing from this equation: **sequence order**. 

If you shuffle the input tokens, the attention mechanism will compute the exact same attention scores for the corresponding pairs, and the output will simply be the shuffled version of the original output. We say that the standard self-attention mechanism is **permutation equivariant**.

Let $X = [x_1, x_2, \dots, x_N]^T$ be an input sequence of length $N$. Let $\pi$ be a permutation operator.
If we apply standard self-attention $f_{attn}$, we observe:
$$ f_{attn}(\pi(X)) = \pi(f_{attn}(X)) $$

### 1.2 The Need for Sequence Information
Human language is highly dependent on word order. Consider the sentences:
1. "The dog chased the cat."
2. "The cat chased the dog."

These contain the exact same tokens, but entirely different meanings. Because the standard attention mechanism is permutation equivariant, it cannot distinguish between these two sentences on its own. It doesn't know who is the subject and who is the object based on word order.

To solve this, we must inject positional information into the embeddings before or during the attention computation. This is the role of Positional Encodings.

---

## 2. Absolute Positional Encodings

The earliest and most straightforward approach to providing sequence information is **Absolute Positional Encoding**. This involves generating a unique vector for each position index $t$ in the sequence and adding it to the token's embedding $x_t$.

$$ \tilde{x}_t = x_t + P_t $$

Where $P_t \in \mathbb{R}^d$ is the positional encoding for position $t$, and $d$ is the model's hidden dimension.

### 2.1 Sinusoidal Positional Encodings (Vaswani et al., 2017)
In the original Transformer paper ("Attention Is All You Need"), the authors proposed a deterministic, fixed sinusoidal function to generate $P_t$. 

For a given position $t$ and dimension index $i \in [0, d/2 - 1]$, the positional encoding is defined as:

$$ P_{t, 2i} = \sin\left(\frac{t}{10000^{2i/d}}\right) $$
$$ P_{t, 2i+1} = \cos\left(\frac{t}{10000^{2i/d}}\right) $$

**Why use this specific formula?**
1. **Uniqueness:** Each position $t$ produces a unique vector.
2. **Bounded Values:** Sine and cosine values are bounded between $[-1, 1]$, preventing the positional embeddings from dominating the semantic token embeddings.
3. **Relative Position Inference:** The authors noted that for any fixed offset $k$, $P_{t+k}$ can be represented as a linear function of $P_t$. This allows the model to easily learn to attend by relative positions.
4. **Generalization:** Theoretically, the model can extrapolate to sequence lengths longer than those seen during training (though in practice, absolute encodings struggle with this).

### 2.2 Visualizing Sinusoidal Encodings

Imagine the positional encoding as a set of gears turning at different speeds. 
The lowest dimension ($i=0$) has the highest frequency and completes a full cycle rapidly. As $i$ increases, the wavelength increases geometrically, meaning the higher dimensions vary extremely slowly across the sequence length.

```text
Dimension 0 & 1 (Fastest)  : ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~
Dimension 2 & 3 (Medium)   : ~~~~~~ ~~~~~~ ~~~~~~ ~~~~~~
...
Dimension d-2 & d-1 (Slow) : ~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

### 2.3 Learned Absolute Positional Embeddings
Instead of using fixed sinusoidal functions, models like BERT and early versions of GPT opted for **Learned Absolute Positional Embeddings**. 
In this approach, $P_t$ is simply a trainable parameter matrix of size $(L_{max}, d)$, where $L_{max}$ is the maximum sequence length. 
- **Pros:** Highly flexible; the model learns exactly what it needs.
- **Cons:** Strictly limited to $L_{max}$. The model completely fails if given a sequence of length $L_{max} + 1$ during inference because $P_{L_{max}+1}$ was never initialized or trained.

---

## 3. The Shift to Relative Positional Encodings

Absolute encodings tell the model *where* a token is in the overall sequence (e.g., "I am token #5"). However, linguistic relationships typically depend on *relative* distances (e.g., "This adjective is 2 tokens before that noun").

Relative Positional Encodings (RPE) inject distance information directly into the attention mechanism rather than adding it to the input embeddings. 

Instead of computing $q_m \cdot k_n$ (where $m$ and $n$ are absolute positions), RPE models compute attention based on $m - n$. Let's look at the modern standard for this: RoPE.

---

## 4. Rotary Positional Embeddings (RoPE)

Introduced in the paper *RoFormer: Enhanced Transformer with Rotary Position Embedding* (Su et al., 2021), **RoPE** has become the defacto standard in modern LLMs (used in LLaMA, PaLM, Mistral, Qwen, etc.).

### 4.1 The Core Idea of RoPE
RoPE encodes absolute position with a rotation matrix and simultaneously incorporates explicit relative position dependency in the self-attention formulation.

The primary goal of RoPE is to find a function $f$ for queries and keys such that their dot product is solely a function of their embeddings and their relative distance $m-n$:

$$ \langle f_q(x_m, m), f_k(x_n, n) \rangle = g(x_m, x_n, m - n) $$

### 4.2 Mathematical Deep Dive: 2D Complex Plane
Let's consider a simplified 2D case where our embedding dimension $d=2$. We can represent our query vector $q = \begin{pmatrix} q_1 \\ q_2 \end{pmatrix}$ as a complex number: 
$$ q = q_1 + iq_2 = |q| e^{i\theta_q} $$

To encode position $m$, RoPE simply rotates this vector in the complex plane by an angle $m\theta$, where $\theta$ is a predefined constant base angle.

$$ f_q(x_m, m) = q e^{im\theta} $$
$$ f_k(x_n, n) = k e^{in\theta} $$

Now, let's look at their inner (dot) product. In complex numbers, the real part of the inner product $q^* k$ is equivalent to the vector dot product:

$$ \text{Re}( (q e^{im\theta})^* (k e^{in\theta}) ) = \text{Re}( q^* e^{-im\theta} k e^{in\theta} ) = \text{Re}( q^* k e^{i(n-m)\theta} ) $$

As you can see, the dot product between the rotated query at position $m$ and the rotated key at position $n$ explicitly depends on $(n-m)$, which is their **relative distance**. 

### 4.3 Generalizing to d-Dimensions
For a $d$-dimensional embedding vector, we divide the vector into $d/2$ pairs of 2D vectors. We apply the 2D rotation to each pair using a different frequency $\theta_i$.

The frequencies are defined identically to the Vaswani sinusoidal encodings:
$$ \theta_i = 10000^{-2i/d} \quad \text{for} \quad i \in [0, d/2 - 1] $$

The rotation matrix $R_{\Theta, m}^d$ for position $m$ is a block-diagonal matrix:

$$
R_{\Theta, m}^d = 
\begin{pmatrix}
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1}
\end{pmatrix}
$$

Because the matrix is highly sparse, we do not perform actual matrix multiplication in practice. Instead, we use a computationally efficient Hadamard (element-wise) product formulation.

```text
Algorithm for applying RoPE:
1. Reshape vector from [d] to [d/2, 2]
2. Form the complex vectors: v_complex = v[..., 0] + i * v[..., 1]
3. Create position frequencies: freqs = exp(i * m * theta)
4. Multiply: rotated_complex = v_complex * freqs
5. Flatten back to [d] by extracting real and imaginary parts
```

### 4.4 Why RoPE Won
1. **Unifies Absolute and Relative:** It applies an absolute transformation to individual tokens, but achieves a relative distance effect in the attention scores.
2. **No Extra Memory:** Unlike standard relative position biases that require an $O(N^2)$ bias matrix, RoPE modifies the $Q$ and $K$ vectors directly in $O(N \cdot d)$ time and space.
3. **Extrapolatable:** Because it's based on continuous rotations, it's easier to manipulate the frequencies to extend the context window post-training.

---

## 5. ALiBi (Attention with Linear Biases)

While RoPE is dominant, **ALiBi** (Press et al., 2021) is an incredibly elegant alternative used in models like MPT and BLOOM.

### 5.1 The Core Idea of ALiBi
ALiBi completely removes positional embeddings from the input layer. Instead, it adds a static, non-learned bias directly to the attention scores *before* the softmax operation. The bias is linearly proportional to the distance between the query and the key.

For a query at position $m$ and a key at position $n$, the pre-softmax attention score becomes:

$$ \text{Score}(m, n) = q_m \cdot k_n - m \cdot |m - n| $$

Where $m$ is a head-specific slope scalar. 

### 5.2 The Head-Specific Slope
If we only had one attention head, adding $- |m-n|$ would strictly penalize distant tokens. To allow the model to capture both local and long-range dependencies, ALiBi uses a different slope $m$ for each attention head.

For $H$ attention heads, the slopes form a geometric sequence. For example, with 8 heads, the slopes might be:
$$ m \in \left\{ \frac{1}{2^1}, \frac{1}{2^2}, \frac{1}{2^3}, \dots, \frac{1}{2^8} \right\} $$

```text
Attention Score Matrix (Distance Penalty) for one head (m):
      k0   k1   k2   k3
q0 [   0,  -m, -2m, -3m ]
q1 [  -m,   0,  -m, -2m ]
q2 [ -2m,  -m,   0,  -m ]
q3 [ -3m, -2m,  -m,   0 ]
```

### 5.3 Pros and Cons of ALiBi
- **Pros:** 
  - **Zero-shot Extrapolation:** ALiBi is exceptionally good at generalizing to sequence lengths longer than it was trained on. Because the penalty is just a linear function of distance, it scales naturally.
  - **Speed:** No complex numbers or trigonometric functions to compute, making it slightly faster and highly memory efficient.
- **Cons:** 
  - **Strict Inductive Bias:** It forces a strong assumption that "closer tokens are more relevant", which is generally true for language but might limit the model's ability to learn complex, non-monotonic long-range retrieval tasks compared to RoPE.

---

## 6. Context Window Extension Methods

One of the biggest areas of research in LLMs is taking a model trained on context length $L$ (e.g., 4k) and extending it to $L'$ (e.g., 32k or 128k) without training from scratch.

Because RoPE relies on frequencies and wavelengths, directly feeding position $m > L$ into a trained RoPE model causes massive performance degradation. The model has never seen the high-frequency dimensions rotated to such extreme angles.

### 6.1 Position Interpolation (PI)
Instead of extrapolating to unseen positions, **Position Interpolation** (Chen et al., 2023) scales the position indices down so they fit within the original trained context window.

If extending from $L=4k$ to $L'=16k$ (a scale factor of $s=4$), we map the new positions $[0, 16000]$ to the range $[0, 4000]$ by dividing the position index by $s$:

$$ \text{RoPE}(x, m/s) $$

**Result:** The model only ever sees rotation angles it observed during training. However, the *resolution* between adjacent tokens is squeezed, requiring a small amount of fine-tuning (usually 1000 steps) for the model to adapt to this "compressed" spacing.

### 6.2 NTK-Aware Scaled RoPE
Position Interpolation scales all dimensions equally. However, high-frequency dimensions (which capture local relationships) are heavily distorted when squeezed, while low-frequency dimensions (which capture long-range structure) need the scaling to avoid extrapolation.

**Neural Tangent Kernel (NTK)-Aware Scaling** dynamically changes the scale based on the dimension frequency. 
- High frequencies are NOT scaled (preserving local token relationships).
- Low frequencies are heavily scaled (interpolating the long-range relationships).

This is achieved mathematically by simply altering the base constant in the RoPE frequency formula (e.g., changing $10000$ to an effectively larger base depending on the extension factor). This method famously allowed zero-shot context extension (no fine-tuning required).

### 6.3 YaRN (Yet another RoPE extensioN method)
YaRN (Peng et al., 2023) improves upon NTK-aware scaling by explicitly separating the dimensions into three groups:
1. **High-frequency:** No interpolation (extrapolate only).
2. **Low-frequency:** Pure linear interpolation.
3. **Mid-frequency:** A smooth transition blend between extrapolation and interpolation.

YaRN also introduces a temperature scalar to the attention softmax to counteract the entropy shift caused by having more tokens in the context window. YaRN is currently considered state-of-the-art for post-training context extension.

---

## 7. Typical Interview Questions

Here are common interview questions you might face regarding positional encodings, along with how to structure your answers.

### Q1: Why can't we just use a single scalar value from 0 to 1 for positional encoding?
**Candidate Answer:** "If we assign position 0 as 0.0 and position $L_{max}$ as 1.0, the delta between adjacent tokens becomes dependent on the sequence length. A model wouldn't have a consistent definition of 'one token apart.' If we instead use $0, 1, 2, \dots, L$, the values grow unbounded, which destabilizes the gradients and dominates the token embeddings. We need a representation that provides unique positions, bounded values, and consistent relative distances—which is why high-dimensional sinusoidal or rotary embeddings are used."

### Q2: How does RoPE differ from standard Absolute Positional Encodings?
**Candidate Answer:** "Standard absolute positional encodings (like Vaswani's sinusoidal) are *added* to the token embeddings right at the input layer. The model has to learn to infer relative distances through standard projection matrices. RoPE, on the other hand, is applied via multiplication (rotations in the complex plane) specifically to the Queries and Keys at *every* attention layer. By rotating the $Q$ and $K$ vectors by their absolute positions, their dot-product inherently computes the relative distance between them."

### Q3: How would you extend the context window of a LLaMA model from 4k to 32k without retraining from scratch?
**Candidate Answer:** "Since LLaMA uses RoPE, I would not just feed it longer sequences, as it would fail on the out-of-distribution rotation angles in the lower frequencies. Instead, I would use an interpolation method. A basic approach is Position Interpolation (Linear Scaling), which divides the position indices by a factor of 8 ($32k/4k$) and requires a brief fine-tuning phase. A more advanced and effective approach would be YaRN or NTK-Aware Scaling, which scales the low frequencies to interpolate while leaving high frequencies untouched to preserve local semantics. YaRN often allows for near zero-shot extension or requires very minimal fine-tuning."

### Q4: In what scenario would you choose ALiBi over RoPE?
**Candidate Answer:** "I would choose ALiBi if my primary design goal is robust, zero-shot length extrapolation without complex frequency manipulation. ALiBi models naturally generalize to longer sequences because the attention penalty is just a linear scalar $-m|x-y|$. It is also slightly more computationally efficient during inference as it avoids trigonometric operations. However, if my model relies heavily on complex, non-local retrieval tasks, RoPE often demonstrates slightly better empirical performance at the target context length."

---
*End of Lesson 6. Proceed to Lesson 7 for Key-Value Cache and Memory Optimizations.*
