# Lesson — Inference Optimization

## Complete LLM Inference Optimization Notes

Inference optimization is the process of making a trained LLM generate responses **faster, with less memory, higher throughput, and lower cost**, while maintaining acceptable output quality.

> **Core idea:** Do less work, move fewer bytes, reuse previous work, and avoid using an expensive model when a cheaper one is sufficient.

---

## 1. What Is LLM Inference?

**Inference** means using a trained model to generate an output for a given input.

```text
Training
Data → Model → Learn Parameters → Trained Model

Inference
User Prompt → Trained Model → Generated Tokens → Response
```

| Training | Inference |
|---|---|
| Learns model parameters | Uses learned parameters |
| Forward + backward pass | Mainly forward pass |
| Usually compute-intensive | Often latency/memory intensive |
| Happens during model development | Happens for every model request |
| Goal: improve parameters | Goal: generate output efficiently |

---

## 2. Why LLM Inference Can Be Memory-Bound

A large LLM may contain billions of parameters.

For a **70B parameter model** stored in FP16:

```text
70 billion × 2 bytes
≈ 140 GB
```

This is only the weight memory. Real inference also needs memory for:

- KV cache
- Activations
- CUDA/runtime overhead
- Temporary buffers
- Batching

A simplified memory-bandwidth intuition is:

```text
Time ≈ Amount of data moved / Memory bandwidth
```

Example:

```text
Model size ≈ 140 GB
GPU bandwidth ≈ 3.35 TB/s

140 GB / 3.35 TB/s
≈ 42 ms
```

This is only a simplified lower-bound intuition. Actual performance depends on the GPU, kernels, model architecture, batch size, context length, quantization, framework, and workload.

### Key idea

During autoregressive decoding, inference can become **memory-bandwidth-bound** rather than purely compute-bound.

---

## 3. Prefill vs Decode

LLM inference has two major stages:

```text
                LLM Inference
                     |
             +-------+-------+
             |               |
          Prefill          Decode
```

### Prefill

Prefill processes the user's input/prompt.

```text
Prompt tokens
     ↓
Transformer
     ↓
Hidden states
     ↓
KV cache
```

Many input tokens can be processed in parallel, so prefill is relatively parallelizable.

### Decode

Decode generates one token at a time:

```text
Prompt
  ↓
Token 1
  ↓
Token 2
  ↓
Token 3
  ↓
Token 4
  ↓
...
```

Autoregressive generation follows:

```text
P(xₜ | x₁, x₂, ..., xₜ₋₁)
```

Because each new token depends on previous tokens, decode is sequential and is especially important for interactive latency.

---

## 4. KV Cache

KV cache is one of the most important LLM inference optimizations.

Transformer attention uses:

```text
Q = Query
K = Key
V = Value
```

During generation, previous tokens do not change. Recomputing their Key and Value representations is therefore wasteful.

### Without KV Cache

A naive implementation can repeatedly recompute information for earlier tokens:

```text
Token 5 → recompute previous K/V
Token 6 → recompute previous K/V
Token 7 → recompute previous K/V
...
```

### With KV Cache

The model stores previous K/V tensors:

```text
Token 1 → K₁,V₁ → Cache
Token 2 → K₂,V₂ → Cache
Token 3 → K₃,V₃ → Cache
Token 4 → K₄,V₄ → Cache
```

For the next token:

```text
New Query
   ↓
Attend to cached K/V
   ↓
Generate next token
```

### Important precision

KV caching avoids recomputing **past K/V projections**. It does **not** make all attention O(N). Each new token still attends over the growing cached context.

> The key benefit is reusing information that has already been computed.

---

## 5. KV Cache Memory Cost

A simplified relationship is:

```text
KV Cache Memory
∝
Number of layers
× Number of KV heads
× Head dimension
× Sequence length
× Bytes per element
```

KV cache becomes especially important when:

- Context windows are large
- Many requests are served simultaneously
- Batch size is large
- The model has many layers

This is why modern serving systems use:

- GQA
- MQA
- PagedAttention
- Prefix caching
- KV-cache quantization

---

## 6. Grouped-Query Attention (GQA)

GQA means **Grouped-Query Attention**.

Instead of having a separate K/V head for every Query head, multiple Query heads share K/V heads.

Example:

```text
Query heads:
Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8

K/V groups:
K1 V1
K2 V2
K3 V3
K4 V4
```

Conceptually:

```text
Q1 ─┐
Q2 ─┘ → K1,V1

Q3 ─┐
Q4 ─┘ → K2,V2

Q5 ─┐
Q6 ─┘ → K3,V3

Q7 ─┐
Q8 ─┘ → K4,V4
```

Fewer K/V heads means:

```text
Less KV cache
     ↓
Less GPU memory
     ↓
Better serving efficiency
```

---

## 7. Quantization

Quantization reduces the number of bits used to represent model values.

Typical formats:

| Format | Bits | Approx. 70B weight memory |
|---|---:|---:|
| FP32 | 32 | 280 GB |
| FP16 | 16 | 140 GB |
| INT8 | 8 | 70 GB |
| INT4 | 4 | 35 GB |

These are simplified **weights-only** calculations.

For example:

```text
70B × 2 bytes ≈ 140 GB  → FP16

70B × 1 byte  ≈ 70 GB   → INT8

70B × 0.5 byte ≈ 35 GB  → INT4
```

### Quantization trade-off

```text
Lower precision
      ↓
Less memory
      ↓
Potentially faster/cheaper inference
      ↓
Possible numerical/quality loss
```

Actual quality impact depends on the model, task, calibration data, quantization method, and hardware.

---

## 8. AWQ

AWQ stands for **Activation-aware Weight Quantization**.

The basic idea is to identify weights that are especially important for preserving model behavior and protect them more carefully during quantization.

```text
All weights
    ↓
Identify important weights
    ↓
Protect important weights
    ↓
Quantize efficiently
```

This can provide a useful quality-efficiency trade-off for low-bit inference.

---

## 9. Speculative Decoding

Speculative decoding uses a smaller model to propose tokens and a larger model to verify them.

```text
                 Prompt
                    ↓
             Small Draft Model
                    ↓
             Draft several tokens
                    ↓
             Large Target Model
                    ↓
               Verify tokens
                    ↓
             Accept / Correct
```

Example:

```text
Small model:
"The capital of France is Paris"

Draft:
The | capital | of | France | is | Paris

Large model:
Verify the proposed tokens
```

If many proposed tokens are accepted, fewer expensive target-model decoding steps are required.

### Why it can be faster

```text
Small model drafts multiple tokens
            ↓
Large model verifies efficiently
            ↓
More accepted tokens per expensive pass
            ↓
Lower generation latency
```

Properly implemented speculative sampling can preserve the target model's sampling distribution.

---

## 10. Batching

Batching processes multiple requests together.

Instead of:

```text
A → GPU
B → GPU
C → GPU
D → GPU
```

we can process:

```text
A ─┐
B ─┤
C ─┼──→ GPU
D ─┘
```

This can improve hardware utilization and throughput.

---

## 11. Static Batching

Static batching waits for a batch to form.

```text
Request A arrives
     ↓
Wait

Request B arrives
     ↓
Wait

Request C arrives
     ↓
Batch formed
     ↓
GPU processing
```

The waiting time can increase latency.

---

## 12. Continuous Batching

Continuous batching dynamically fills available inference slots as requests finish.

```text
Initial:
[A][B][C]

A finishes:
[D][B][C]

B finishes:
[D][E][C]
```

The GPU does not need to wait for the entire original batch to finish before admitting new work.

> **Continuous batching primarily improves serving throughput and GPU utilization.**

---

## 13. Latency vs Throughput

These are different metrics.

### Latency

How long one request takes.

```text
Request → Response
       100 ms
```

### Throughput

How much work can be processed per unit time.

```text
10,000 tokens/sec
```

An optimization can improve throughput without improving single-request latency.

Always ask:

```text
Are we optimizing latency or throughput?
```

---

## 14. Flash Attention

Standard attention is:

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √d)V
```

The attention score matrix has approximately:

```text
N × N
```

elements for sequence length N.

Materializing this large intermediate matrix can create substantial memory traffic.

### Flash Attention

Flash Attention uses memory-efficient techniques such as:

- Tiling
- On-chip SRAM usage
- Reduced memory movement
- Avoiding materialization of the full attention matrix in HBM

Conceptually:

```text
Large attention computation
        ↓
Process smaller tiles
        ↓
Keep useful intermediate values closer to compute
        ↓
Reduce HBM traffic
```

### Important precision

Flash Attention does **not** simply change standard attention's mathematical complexity from O(N²) to O(N).

Its major benefit is reducing memory usage and memory movement while efficiently implementing the same attention computation.

---

## 15. Prefix Caching

Many applications repeatedly use the same prompt prefix.

Example:

```text
System Prompt:
"You are an AI assistant for Company X..."

User 1:
"How do I reset my password?"

User 2:
"How do I change my email?"

User 3:
"Where can I see my invoices?"
```

Without prefix caching:

```text
Request 1 → Compute system prompt
Request 2 → Compute system prompt again
Request 3 → Compute system prompt again
```

With prefix caching:

```text
System Prompt
     ↓
Compute once
     ↓
Store KV state
     ↓
Reuse across requests
```

Useful for:

- Long system prompts
- Few-shot examples
- Agent instructions
- Tool definitions
- Repeated document prefixes

---

## 16. PagedAttention

PagedAttention manages KV-cache memory in blocks/pages.

A large KV cache can suffer from fragmentation if memory must be allocated as large contiguous regions.

Conceptually:

```text
KV Cache

Page 1 → Request A
Page 2 → Request C
Page 3 → Request A
Page 4 → Request B
Page 5 → Request D
```

Benefits:

```text
Paged KV Cache
      ↓
Better memory utilization
      ↓
Less fragmentation
      ↓
More concurrent requests
```

PagedAttention is primarily a **KV-cache memory-management technique**, not a fundamentally different attention algorithm.

It is strongly associated with efficient LLM serving systems such as vLLM.

---

## 17. Small Language Models (SLMs)

A Small Language Model is a relatively small model designed for lower compute and memory requirements.

There is no universally fixed parameter boundary, but models in roughly the **1B–15B** range are often discussed as small models depending on context.

SLMs are useful for:

- Classification
- Intent detection
- Extraction
- Routing
- Formatting
- Simple summarization
- Simple support tasks
- Lightweight agents

Example:

```text
User Request
     ↓
Small Model
     ↓
Simple?
  /     Yes      No
 ↓        ↓
Answer   Large LLM
```

---

## 18. Model Cascade

A model cascade routes requests to different models depending on difficulty.

Instead of:

```text
Every request
      ↓
Large model
```

use:

```text
                 User Request
                      ↓
                Small Model
                      ↓
              Confidence / Router
                /                         Easy            Hard
              ↓                ↓
         Small Model       Large Model
```

### Cost example

Suppose:

```text
Small model cost = 0.1
Large model cost = 1.0
```

If the small model handles 70% of requests:

```text
Small-model cost:
0.1

Large-model fallback:
0.30 × 1.0 = 0.3

Total:
0.1 + 0.3 = 0.4
```

Compared with always using the large model:

```text
Cost = 1.0
```

Under these simplified assumptions, the cascade uses 40% of the original cost, or about 60% lower cost.

Actual production savings depend on token lengths, routing overhead, hardware, and pricing.

---

## 19. Model Cascade and Calibration

The router needs to decide:

```text
Can the small model handle this?
```

A confidence threshold can be used:

```text
Confidence ≥ threshold
        ↓
Use small model

Confidence < threshold
        ↓
Use large model
```

This is where **calibration** matters.

A confidence score of 0.9 should have a meaningful relationship with actual correctness over comparable predictions.

The threshold should ideally be selected using a labeled validation set containing real application queries.

---

## 20. Serving Frameworks

### vLLM

Focuses on high-throughput LLM serving.

Important concepts:

- PagedAttention
- Continuous batching
- Efficient KV-cache management
- High-throughput serving

### SGLang

Designed for efficient LLM programs and structured/agentic workloads.

Important ideas:

- Structured generation
- Efficient scheduling
- Prefix/radix caching
- Optimized serving

### TensorRT-LLM

NVIDIA's high-performance inference stack.

Focuses on:

- GPU optimization
- Quantization
- Optimized kernels
- High-performance inference

### Ollama

Popular for local model experimentation and development.

Useful for:

- Local development
- Prototyping
- Running quantized models
- Consumer-hardware experimentation

---

## 21. Combining Inference Optimizations

Production systems often combine several techniques.

```text
                    User Request
                         ↓
                 Small Model Router
                    /                           Easy           Hard
                  ↓               ↓
             Small Model       Large Model
                                  ↓
                         Quantized Model
                                  ↓
                         Prefix / KV Cache
                                  ↓
                        Continuous Batching
                                  ↓
                         Optimized Kernels
                                  ↓
                              Response
```

| Optimization | Main idea |
|---|---|
| KV Cache | Avoid recomputing past K/V |
| Prefix Cache | Reuse shared prefixes |
| GQA | Reduce KV-cache size |
| Quantization | Use fewer bits |
| Speculative Decoding | Draft with a small model |
| Continuous Batching | Improve GPU utilization |
| Flash Attention | Reduce memory traffic |
| PagedAttention | Manage KV memory efficiently |
| SLM/Cascade | Avoid expensive models |
| Optimized kernels | Execute operations efficiently |

---

## 22. Four Core Strategies

### 1. Don't Recompute

```text
KV Cache
Prefix Cache
```

Reuse work that has already been performed.

### 2. Move Fewer Bytes

```text
Quantization
GQA
Flash Attention
```

Reduce memory usage and memory movement.

### 3. Amortize Expensive Work

```text
Batching
Speculative Decoding
```

Do more useful work per expensive model execution.

### 4. Avoid Expensive Models

```text
SLMs
Model Cascades
Routing
```

Use a large model only when necessary.

---

## 23. Example Production Architecture

Imagine a customer-support chatbot.

```text
                       User
                        ↓
                  API Gateway
                        ↓
                  Request Router
                        ↓
                 Small Classifier
                        ↓
             +----------+----------+
             |                     |
          Simple                 Complex
             |                     |
             ↓                     ↓
        Small LLM              Large LLM
                                   |
                            Quantized Model
                                   |
                            Prefix Cache
                                   |
                               KV Cache
                                   |
                         Continuous Batching
                                   |
                              GPU Serving
                                   |
                                Output
```

This architecture can reduce cost and latency while improving throughput and concurrency.

---

## 24. What Should You Optimize First?

Use a measurement-driven workflow:

```text
1. Measure
   ↓
2. Identify bottleneck
   ↓
3. Optimize
   ↓
4. Benchmark again
   ↓
5. Check quality
   ↓
6. Deploy
```

Measure:

### Latency

- Time to First Token (TTFT)
- Inter-Token Latency (ITL)
- End-to-End Latency

### Throughput

- Tokens/sec
- Requests/sec
- Concurrent users

### Memory

- Model memory
- KV-cache memory
- GPU utilization
- Memory utilization

### Quality

- Task accuracy
- Human evaluation
- Application-specific evaluation
- Regression tests

### Cost

```text
Cost per request
Cost per 1M tokens
GPU cost
```

---

## 25. Important Inference Metrics

### Time to First Token (TTFT)

Time from receiving the request until the first generated token appears.

```text
Request
  ↓
[ Prefill ]
  ↓
First Token

<---- TTFT ---->
```

TTFT is influenced by:

- Prompt length
- Prefill computation
- Queueing
- Batch size
- GPU performance

### Inter-Token Latency (ITL)

Time between generated tokens.

```text
Token 1
   ↓
Token 2
   ↓
Token 3
   ↓
Token 4
```

ITL affects streaming responsiveness.

### End-to-End Latency

```text
Request
   ↓
Queue
   ↓
Prefill
   ↓
Decode
   ↓
Response
```

Total time is the end-to-end latency.

---

## 26. Common Misconceptions

### Misconception 1

> KV cache makes attention O(N).

Not exactly. KV cache avoids recomputing past K/V projections, but every new token still attends over the growing cached context.

### Misconception 2

> Flash Attention makes attention O(N).

Not for standard attention. Its main benefit is reducing memory usage and memory traffic.

### Misconception 3

> Quantization always makes models faster.

Not necessarily. It can reduce memory footprint and may improve speed, but actual performance depends on hardware and kernels.

### Misconception 4

> PagedAttention changes the attention algorithm.

Not fundamentally. It primarily improves KV-cache memory allocation and management.

### Misconception 5

> Speculative decoding always gives a speedup.

No. It works best when the draft model is much cheaper, acceptance rates are high, and verification is efficient.

### Misconception 6

> Bigger batch size always means lower latency.

No. Batching often improves throughput, but larger batches or queueing can increase individual request latency.

---

## 27. Interview Questions

### Q1. Why can LLM inference be memory-bound?

Large models require moving substantial amounts of model data through GPU memory during generation. During autoregressive decoding, memory bandwidth can become a limiting factor relative to compute capacity.

### Q2. What is KV cache?

KV cache stores previously computed Key and Value tensors so they do not need to be recomputed for every generated token.

### Q3. What problem does GQA solve?

GQA reduces the number of Key/Value heads, lowering KV-cache memory requirements while retaining multiple Query heads.

### Q4. What is quantization?

Quantization represents model values using fewer bits, such as INT8 or INT4, reducing memory usage and potentially improving inference efficiency.

### Q5. What is speculative decoding?

A smaller draft model proposes several tokens, and a larger target model verifies them, potentially reducing expensive target-model decoding steps.

### Q6. What is continuous batching?

Continuous batching dynamically adds new requests into available inference slots as existing requests finish, improving GPU utilization and throughput.

### Q7. What is Flash Attention?

A memory-efficient attention implementation that uses tiling and optimized memory access to reduce memory traffic and avoid materializing the full attention matrix in HBM.

### Q8. What is prefix caching?

Prefix caching stores the KV state for a shared prompt prefix so it can be reused across multiple requests.

### Q9. What is PagedAttention?

A KV-cache memory-management technique that organizes cache storage into blocks/pages, reducing fragmentation and improving memory utilization.

### Q10. What is a model cascade?

A routing architecture where a smaller/cheaper model handles easier requests while harder or uncertain requests are sent to a larger model.

---

## 28. Final Mental Model

Think of LLM inference optimization like optimizing a factory.

### Problem 1: Repeating work

```text
Solution:
KV Cache
Prefix Cache
```

### Problem 2: Moving too much data

```text
Solution:
Quantization
GQA
Flash Attention
```

### Problem 3: GPU sitting idle

```text
Solution:
Batching
Continuous Batching
Optimized Kernels
```

### Problem 4: Large model is expensive

```text
Solution:
SLM
Model Cascade
Speculative Decoding
```

### Problem 5: GPU memory is fragmented

```text
Solution:
PagedAttention
```

Overall:

```text
                 INFERENCE OPTIMIZATION
                          |
        +-----------------+-----------------+
        |                 |                 |
   Do less work      Move fewer bytes   Avoid expensive work
        |                 |                 |
   KV Cache          Quantization       SLMs
   Prefix Cache      GQA                Cascades
                     Flash Attention    Speculative Decoding
        |
        +-------------------------+
                                  |
                         Amortize expensive work
                                  |
                       Batching / Continuous Batching
```

---

## 29. Key Takeaways

1. **Inference** is using a trained model to generate outputs.
2. LLM decoding can become **memory-bandwidth-bound**.
3. **Prefill** processes the prompt; **decode** generates tokens autoregressively.
4. **KV cache** prevents repeated computation of past Key/Value states.
5. **GQA** reduces KV-cache memory by sharing K/V heads.
6. **Quantization** reduces the number of bits used to represent model values.
7. **AWQ** uses activation-aware information to preserve important weights during quantization.
8. **Speculative decoding** uses a small model to draft and a large model to verify.
9. **Continuous batching** improves serving throughput by dynamically filling inference slots.
10. **Flash Attention** reduces memory traffic and avoids materializing the full attention matrix in HBM.
11. **Prefix caching** reuses KV states for repeated prompt prefixes.
12. **PagedAttention** improves KV-cache memory management.
13. **SLMs** can handle simple tasks cheaply.
14. **Model cascades** route difficult requests to larger models.
15. Production optimization requires measuring **latency, throughput, memory, quality, and cost**.
16. The goal is the right balance of **quality + latency + throughput + memory + cost**.

---

## One-Line Summary

> **LLM inference optimization means doing less computation, moving fewer bytes, reusing previous work, keeping GPUs efficiently utilized, and using expensive models only when necessary.**
