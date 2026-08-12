# Lesson 5: LLM Architecture Internals

Welcome to Lesson 5 of the AI Engineering Interview Preparation course. In this lesson, we will peel back the layers of modern Large Language Models (LLMs) and examine their foundational architecture. While the original Transformer paper ("Attention Is All You Need") revolutionized NLP, today's state-of-the-art models like LLaMA, GPT-4, and Mistral employ several critical modifications to improve efficiency, stability, and scaling. 

This lesson will focus on:
1. The shift from Encoder-Decoder to Decoder-Only architectures.
2. Modern normalization techniques (RMSNorm vs. LayerNorm).
3. Advanced activation functions (SwiGLU).
4. The mechanics and mathematics of the KV Cache.

By understanding these internals, you will be well-equipped to discuss model architecture choices, memory bottlenecks, and inference optimization strategies during your AI Engineering interviews.

---

## 1. Architectural Paradigms: Decoder-Only vs. Encoder-Decoder

The original Transformer was introduced in 2017 for machine translation, a classic sequence-to-sequence (seq2seq) task. It featured an Encoder to process the source language and a Decoder to generate the target language. However, the landscape has largely shifted towards Decoder-only models for most generative AI tasks.

### 1.1 The Encoder-Decoder Architecture

An Encoder-Decoder model (e.g., T5, BART) consists of two separate stacks of transformer blocks.

**The Encoder:**
- Processes the input sequence comprehensively.
- Uses **bidirectional self-attention** (unmasked attention). Every token can attend to every other token (past and future) in the input sequence.
- Output is a sequence of contextualized embeddings representing the entire input.

**The Decoder:**
- Generates the output sequence autoregressively (one token at a time).
- Uses **masked self-attention** (causal attention) to prevent tokens from "looking ahead" at future tokens during generation.
- Uses **cross-attention** to attend to the Encoder's output representations.

**ASCII Diagram: Encoder-Decoder**
```text
[Input Text] -> (Encoder: Bidirectional Attention) -> [Context Vectors]
                                                             |
                                                             v
[Target Prefix] -> (Decoder: Masked Self-Attention) -> (Cross-Attention) -> [Output Probabilities]
```

**Pros:** Excellent for tasks where the entire input context is needed simultaneously before generation starts (e.g., translation, summarization).
**Cons:** More complex, requires maintaining two separate sets of parameters, less efficient for open-ended text generation.

### 1.2 The Decoder-Only Architecture

The Decoder-only architecture (e.g., GPT series, LLaMA series, Claude) simplifies the design by dropping the Encoder entirely.

- It consists of a single stack of transformer blocks.
- It relies exclusively on **masked self-attention** (causal attention). 
- To perform a task, the input prompt is simply prepended to the generated sequence. The model reads the prompt (using causal attention, though practically this can be optimized during the "pre-fill" phase) and continues predicting the next token.

**ASCII Diagram: Decoder-Only**
```text
[Prompt + Generated Text so far] -> (Masked Self-Attention) -> (Feed Forward) -> [Next Token Probabilities]
```

### 1.3 Why Did the Industry Shift to Decoder-Only?

1. **Simplicity and Scaling:** Training a single stack of layers is conceptually simpler and easier to optimize across large distributed systems.
2. **Zero-Shot/Few-Shot Capabilities:** It turns out that language modeling (predicting the next word) on massive datasets is a sufficiently powerful objective to learn world knowledge and reason. A decoder-only model naturally handles in-context learning.
3. **Efficiency:** During generation, cross-attention in Encoder-Decoder models requires attending back to the encoder's outputs at every step. Decoder-only models unify the prompt and generated text into a single KV cache (discussed later), streamlining the inference process.
4. **The "Prefix" Optimization:** Modern implementations can process the initial prompt in parallel (like an encoder) because all prompt tokens are known. This is known as the "pre-fill" phase, blurring the efficiency gap between bidirectional and causal attention for the input context.

---

## 2. Modern Architectural Choices

To train models at the scale of tens or hundreds of billions of parameters, researchers had to modify the original Transformer to improve training stability and computational efficiency. Two of the most significant changes are the adoption of RMSNorm and SwiGLU.

### 2.1 Normalization: LayerNorm vs. RMSNorm

Normalization is crucial in deep neural networks to prevent vanishing or exploding gradients and to ensure stable training.

#### Layer Normalization (LayerNorm)
The original Transformer used LayerNorm. For an input vector $$x$$, LayerNorm computes the mean $$\mu$$ and variance $$\sigma^2$$ of the elements in $$x$$, normalizes $$x$$, and then scales and shifts using learnable parameters $$\gamma$$ and $$\beta$$.

$$ \mu = \frac{1}{d} \sum_{i=1}^{d} x_i $$
$$ \sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2 $$
$$ LayerNorm(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta $$

#### Root Mean Square Normalization (RMSNorm)
RMSNorm is a simplified variant of LayerNorm that has become the standard in modern LLMs (e.g., LLaMA). The key insight behind RMSNorm is that the **scaling** aspect of normalization is crucial for success, while the **centering** (shifting by the mean) is often unnecessary and computationally expensive.

RMSNorm drops the mean calculation and normalizes solely by the Root Mean Square (RMS).

$$ RMS(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2} $$
$$ RMSNorm(x) = \frac{x}{RMS(x) + \epsilon} \odot \gamma $$

**Why RMSNorm?**
- **Computational Efficiency:** It removes the calculation of the mean and the subtraction of the mean from every element, saving memory bandwidth and compute cycles. It is roughly 10-50% faster than LayerNorm in practice.
- **Equivalent Performance:** Empirically, RMSNorm provides the same training stability and final model performance as LayerNorm.
- **Pre-Norm vs. Post-Norm:** Note that modern LLMs almost universally use "Pre-Norm" (applying RMSNorm *before* the Attention and Feed-Forward sub-layers) rather than "Post-Norm" (applying it after the residual addition) to vastly improve training stability at scale.

### 2.2 Activation Functions: SwiGLU

The Feed-Forward Network (FFN) in a Transformer traditionally used ReLU (Rectified Linear Unit) or GELU (Gaussian Error Linear Unit). Modern architectures have largely shifted to GLU (Gated Linear Unit) variants, specifically SwiGLU.

#### The Traditional FFN
A standard FFN consists of two linear transformations with an activation function in between:

$$ FFN(x) = \text{Activation}(x W_1 + b_1) W_2 + b_2 $$

#### Gated Linear Units (GLU)
A GLU introduces a gating mechanism. Instead of a single projection before the activation, it uses two parallel projections. One projection acts as the gate, controlling the information flow of the other projection via an element-wise multiplication.

$$ GLU(x, W, V, b, c) = \sigma(xW + b) \odot (xV + c) $$

#### SwiGLU
SwiGLU replaces the Sigmoid activation $$\sigma$$ in a standard GLU with the Swish activation function (specifically, Swish with $$\beta=1$$, which is also called SiLU - Sigmoid Linear Unit).

$$ \text{Swish}(x) = x \cdot \sigma(\beta x) $$
$$ \text{SwiGLU}(x, W, V) = \text{Swish}(xW) \odot (xV) $$

The final FFN layer using SwiGLU (omitting biases, which are often removed in modern LLMs) looks like:

$$ FFN_{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xV_1) W_2 $$

*Note: Because SwiGLU requires three weight matrices ($$W_1, V_1, W_2$$) instead of two, the hidden dimension of the FFN is typically reduced (e.g., from $$4d$$ to $$\frac{8}{3}d$$) to keep the total parameter count constant compared to a standard FFN.*

**Why SwiGLU?**
- **Empirical Superiority:** Papers like "GLU Variants Improve Transformer" (Shazeer, 2020) demonstrated that GLU variants, and SwiGLU in particular, consistently outperform ReLU and GELU across various tasks and scales.
- **Smoothness and Non-Monotonicity:** The Swish function is smooth and non-monotonic (it dips slightly below zero for small negative inputs). This property helps gradients flow better during training.
- **Multiplicative Interactions:** The gating mechanism ($$\odot$$) allows the model to learn more complex, multiplicative relationships between features within the FFN.

---

## 3. Deep Dive: The KV Cache

Understanding the KV (Key-Value) Cache is arguably the most important concept for anyone deploying or optimizing LLM inference. It is the primary bottleneck for memory and throughput in production systems.

### 3.1 The Autoregressive Generation Problem

LLMs generate text autoregressively: they predict token $$t$$ based on tokens $$1$$ to $$t-1$$. Then, to predict token $$t+1$$, they use tokens $$1$$ to $$t$$.

A naive implementation of generation would recalculate the self-attention for *all* previous tokens at every step.

**Naive Generation (Step-by-Step):**
- Step 1: Input `["The", "cat"]`. Model computes Keys ($$K$$) and Values ($$V$$) for "The" and "cat". Computes Query ($$Q$$) for "cat". Attends. Outputs "sat".
- Step 2: Input `["The", "cat", "sat"]`. Model computes $$K$$ and $$V$$ for "The", "cat", and "sat". Computes $$Q$$ for "sat". Attends. Outputs "on".
- Step 3: Input `["The", "cat", "sat", "on"]`. Model computes $$K$$ and $$V$$ for "The", "cat", "sat", "and "on". Computes $$Q$$ for "on". Attends. Outputs "the".

Notice the massive redundant computation. At Step 3, we are recomputing the Keys and Values for "The", "cat", and "sat", which we already computed in Steps 1 and 2!

### 3.2 What is the KV Cache?

The KV Cache is an inference optimization technique that stores the previously computed Keys and Values for all past tokens in the sequence. 

Because causal attention ensures that past tokens cannot attend to future tokens, the Keys and Values for token $$i$$ depend *only* on the input up to token $$i$$. Therefore, once we compute $$K_i$$ and $$V_i$$, they will never change during the generation of the rest of the sequence.

**Optimized Generation with KV Cache:**
- Step 1 (Prefill): Input `["The", "cat"]`. Model computes $$K, V$$ for "The", "cat". **Caches them.** Outputs "sat".
- Step 2 (Decode): Input `["sat"]` (only the new token!). Model computes $$K, V$$ for "sat". **Appends to Cache.** Model computes $$Q$$ for "sat". Computes attention using $$Q_{sat}$$ and the *entire* cached $$K, V$$ (`["The", "cat", "sat"]`). Outputs "on".
- Step 3 (Decode): Input `["on"]`. Model computes $$K, V$$ for "on". **Appends to Cache.** Model computes $$Q$$ for "on". Computes attention using $$Q_{on}$$ and cached $$K, V$$ (`["The", "cat", "sat", "on"]`). Outputs "the".

**ASCII Diagram: KV Cache Mechanism**
```text
Time Step T (Generating token T+1):

Current Token (T) ---> [Linear Projections] ---> Q_T, K_T, V_T
                                                   |    |
                                                   v    v
KV Cache Storage:                               [Append to Cache]
[K_1, K_2, ..., K_T-1] <----------------------- [K_1, K_2, ..., K_T-1, K_T]
[V_1, V_2, ..., V_T-1] <----------------------- [V_1, V_2, ..., V_T-1, V_T]

Attention Calculation:
Attention(Q_T, Cache_K, Cache_V) = Softmax( Q_T * (Cache_K)^T / sqrt(d) ) * Cache_V
```

### 3.3 The Two Phases of Inference

Because of the KV Cache, LLM inference is fundamentally split into two distinct phases with very different computational profiles:

1. **The Prefill Phase (Time to First Token - TTFT):**
   - Processes the entire input prompt at once.
   - Computes the initial KV cache for all prompt tokens.
   - **Compute-Bound:** This phase relies on large matrix multiplications. It scales well with GPU Tensor Cores.
2. **The Decode Phase (Time Per Output Token - TPOT):**
   - Generates tokens one by one autoregressively.
   - At each step, it processes a single token, updates the cache, and computes attention against the growing cache.
   - **Memory-Bandwidth Bound:** The GPU compute is underutilized because it's waiting to load the massive KV Cache from HBM (High Bandwidth Memory) to the compute cores for *every single token generated*.

### 3.4 KV Cache Memory Math

The size of the KV cache grows linearly with the sequence length and batch size. Let's calculate the memory required.

**Parameters per token in the cache:**
For every token, we must store a Key vector and a Value vector for every layer and every attention head.

$$ \text{Memory per token} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{bytes\_per\_param} $$

Since $$n_{\text{heads}} \times d_{\text{head}} = d_{\text{model}}$$ (the hidden dimension), this simplifies to:

$$ \text{Memory per token} = 2 \times n_{\text{layers}} \times d_{\text{model}} \times \text{bytes\_per\_param} $$

*(Note: The factor of 2 is because we store both Keys and Values).*

**Example Calculation: LLaMA-2 70B**
- $$n_{\text{layers}} = 80$$
- $$d_{\text{model}} = 8192$$
- Data type: FP16 (2 bytes per parameter)

$$ \text{Memory per token} = 2 \times 80 \times 8192 \times 2 \text{ bytes} = 2,621,440 \text{ bytes} \approx 2.6 \text{ MB/token} $$

If you want to process a batch of 32 requests, each with a context length of 4,096 tokens:

$$ \text{Total KV Cache Memory} = 32 \times 4096 \times 2.6 \text{ MB} \approx 340 \text{ GB} $$

This is massive! The model weights for LLaMA-2 70B in FP16 take about 140 GB. In this scenario, the KV cache takes more than double the memory of the model weights themselves. This is why techniques like Grouped-Query Attention (GQA), Multi-Query Attention (MQA), PagedAttention (vLLM), and quantization are critical for serving LLMs efficiently.

---

## 4. Typical Interview Questions

1. **"Explain the difference between an Encoder-Decoder and a Decoder-only architecture. Why did OpenAI and Meta choose Decoder-only for GPT-4 and LLaMA?"**
   *Look for:* Understanding of bidirectional vs. causal attention. Mention of training simplicity, zero-shot scaling laws, and the efficiency of unifying the prefix and generation phases under a single architecture.

2. **"What is the KV cache? Why is it essential for LLM inference?"**
   *Look for:* Explanation of autoregressive generation. Without KV cache, time complexity per token is $$O(N^2)$$ where N is sequence length. KV cache reduces redundant computation by storing past keys and values, changing the bottleneck from compute to memory bandwidth.

3. **"How does the memory footprint of the KV cache scale? Can you derive the formula?"**
   *Look for:* Mention of the formula: `2 * Layers * Hidden_Dim * Precision_Bytes * Seq_Len * Batch_Size`. Understanding that memory grows linearly with sequence length and batch size, eventually overtaking model weight memory in long-context or high-concurrency scenarios.

4. **"Why do modern models use RMSNorm instead of LayerNorm?"**
   *Look for:* Understanding that LayerNorm does mean-centering and variance-scaling. RMSNorm drops the mean-centering, which is computationally expensive and empirically unnecessary for training stability, resulting in a 10-50% speedup for the normalization operation.

5. **"What is the bottleneck during the prefill phase vs. the decode phase of LLM inference?"**
   *Look for:* Prefill is compute-bound (large matrix multiplications for the prompt). Decode is memory-bandwidth bound (loading the ever-growing KV cache from GPU memory to compute units for a single token matrix-vector multiplication).

6. **"How does SwiGLU differ from a standard ReLU Feed-Forward Network?"**
   *Look for:* Explanation of the Gated Linear Unit concept (two linear projections, one acting as a gate). Mention of the Swish activation function and how the gating mechanism allows for more expressive multiplicative interactions, leading to empirically better downstream performance.
