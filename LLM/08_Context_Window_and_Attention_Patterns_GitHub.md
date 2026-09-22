# Lesson 08 - Context Window & Attention Patterns

## Complete LLM Engineering Note

> GitHub-safe version: formulas use plain ASCII notation. No LaTeX syntax or special mathematical symbols are used.

---

## 1. Core Question

How can modern LLMs process long contexts efficiently when standard self-attention has quadratic token-to-token interactions and the KV cache grows with sequence length?

```text
Context Window
      |
      v
Self-Attention
      |
      v
O(n^2) attention
      |
      v
KV Cache
      |
      v
MHA -> MQA -> GQA
      |
      v
MLA
      |
      v
Flash Attention
```

---

# 2. Context Window

The context window is the maximum number of tokens that a model can process as part of one model context.

A context can contain:

```text
System instructions
        +
Conversation history
        +
User prompt
        +
Retrieved documents
        +
Tool results
        +
Generated tokens
```

All of these consume context tokens according to the exact model and serving system.

## 2.1 Tokens, Not Characters

Context windows are measured in tokens, not characters.

```text
characters != tokens
words != tokens
```

Token count depends on the tokenizer.

For English text, one token may be a whole short word, part of a word, punctuation, or another learned text fragment. This is only an approximation.

## 2.2 Context Window Is Not the Same as Memory

A context window is the amount of information available inside the current model computation.

Long-term memory is a separate system-level concept.

```text
Conversation from yesterday
        |
        v
Database / vector store / memory system
        |
        v
Retrieve relevant information
        |
        v
Insert retrieved information into current context
        |
        v
LLM
```

Therefore:

```text
Context window = current working context
External memory = information stored outside the current context
```

This distinction is important for RAG and memory-based chatbots.

---

# 3. Why Context Length Matters

Suppose:

```text
n = sequence length
```

Standard full self-attention computes an attention score for every query against every key.

Therefore the number of query-key interactions grows approximately as:

```text
n x n = n^2
```

Examples:

```text
Sequence length       Pairwise interactions

10                    100
100                   10,000
1,000                 1,000,000
10,000                100,000,000
100,000               10,000,000,000
```

Doubling the sequence length makes the pairwise interaction count approximately 4x larger.

---

# 4. Self-Attention Refresher

For:

```text
X = [x1, x2, ..., xn]
```

the Transformer creates:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

where:

```text
Q = Queries
K = Keys
V = Values
```

The attention operation is:

```text
Attention(Q,K,V)
=
softmax((Q K^T) / sqrt(d_k)) V
```

For causal language models, a causal mask prevents a token from attending to future tokens.

```text
        Keys
        1  2  3  4
Q1      X
Q2      X  X
Q3      X  X  X
Q4      X  X  X  X
```

This triangular pattern is causal attention.

---

# 5. Why Standard Attention Is O(n^2)

The main interaction matrix has shape:

```text
Q:     n x d_k
K^T:   d_k x n

Q K^T:
       n x n
```

Examples:

```text
n = 4
4 x 4 = 16

n = 1,000
1,000 x 1,000 = 1,000,000

n = 10,000
10,000 x 10,000 = 100,000,000
```

The key intuition:

```text
n doubles
    |
    v
pairwise interactions become about 4x
```

---

# 6. What Exactly Is Expensive?

There are two related but different problems.

## 6.1 Attention computation

The QK^T operation creates interactions between queries and keys.

This produces an n x n score matrix for full attention.

## 6.2 KV-cache memory

During autoregressive generation, previously computed keys and values can be stored so they do not need to be recomputed for every new token.

Therefore:

```text
Attention interaction complexity -> roughly O(n^2)

KV cache size -> roughly O(n)
```

The KV cache is linear in sequence length, but can still become a major GPU-memory bottleneck for long contexts and large batches.

---

# 7. Autoregressive Generation

Suppose the model is generating:

```text
The capital of France is
```

and needs to predict:

```text
Paris
```

At each generation step, the new token creates a new query.

Previously generated tokens already have keys and values.

Without a KV cache:

```text
Past K/V would repeatedly be recomputed.
```

With a KV cache:

```text
Past tokens
    |
    +--> cached K
    +--> cached V

New token
    |
    +--> new Q
    +--> new K
    +--> new V

New Q
    |
    v
attends to cached K + new K
    |
    v
weighted values
    |
    v
next-token prediction
```

---

# 8. What Is the KV Cache?

KV cache stores previously computed:

```text
Keys
Values
```

for the attention layers.

Typically:

```text
Q -> not cached across generation steps
K -> cached
V -> cached
```

Why is Q usually not cached?

Because the query is specific to the newly generated token and is used immediately to retrieve information from the previous context.

Example:

```text
Past context:
K1, K2, K3, K4
V1, V2, V3, V4

New token:
Q5, K5, V5

Attention:
Q5 attends to
K1, K2, K3, K4, K5
```

---

# 9. KV-Cache Memory Formula

Let:

```text
L = number of Transformer layers
n = number of cached tokens
h_kv = number of KV heads
d = head dimension
b = bytes per stored element
```

A simplified KV-cache memory estimate is:

```text
KV memory
~ 2 x L x n x h_kv x d x b
```

The factor 2 is because both K and V are stored.

This is a simplified estimate. Real systems can add memory for batching, padding, metadata, cache management, quantization choices, and implementation details.

---

# 10. Numerical KV-Cache Example

Suppose:

```text
Layers = 32
Sequence length = 8,000
KV heads = 32
Head dimension = 128
Data type = FP16
```

FP16 uses:

```text
2 bytes per value
```

Approximate memory:

```text
2 x 32 x 8,000 x 32 x 128 x 2 bytes
= 4,194,304,000 bytes
```

Approximately:

```text
4.19 GB
```

for one sequence under this simplified setup.

This demonstrates why KV-cache optimization matters for long-context inference.

---

# 11. Multi-Head Attention (MHA)

Traditional Transformer attention commonly uses Multi-Head Attention.

Suppose:

```text
Number of query heads = 32
Number of key heads   = 32
Number of value heads = 32
```

Then:

```text
Q heads = 32
K heads = 32
V heads = 32
```

Conceptually:

```text
Q1 -> K1,V1
Q2 -> K2,V2
Q3 -> K3,V3
...
Q32 -> K32,V32
```

This provides multiple attention subspaces, but also means many K/V vectors must be stored in the KV cache.

---

# 12. Why Reduce K/V Heads?

During autoregressive generation:

```text
Q changes every generation step
```

while:

```text
past K and V can be reused
```

Therefore, reducing the number of K/V heads can substantially reduce KV-cache memory.

This motivates:

```text
MHA
 |
 v
MQA
 |
 v
GQA
```

---

# 13. Multi-Query Attention (MQA)

Multi-Query Attention keeps many query heads but shares a small number of K/V heads.

The extreme case is:

```text
Q heads = 32
K heads = 1
V heads = 1
```

Conceptually:

```text
Q1 Q2  Q3   ...   ---> shared K,V
Q32  /
```

The query heads remain separate, but the K/V representation is shared.

## 13.1 MQA Cache Reduction

If:

```text
MHA: K/V heads = 32
MQA: K/V heads = 1
```

then the K/V head count is reduced by:

```text
32 / 1 = 32x
```

under the same other assumptions.

MQA can therefore dramatically reduce KV-cache memory, although sharing K/V can reduce representational flexibility compared with full MHA.

---

# 14. Grouped-Query Attention (GQA)

Grouped-Query Attention sits between MHA and MQA.

Example:

```text
Q heads = 32
K heads = 8
V heads = 8
```

Each K/V head serves a group of query heads:

```text
32 / 8 = 4
```

So:

```text
Q1 Q2 Q3 Q4   -> K1,V1
Q5 Q6 Q7 Q8   -> K2,V2
...
Q29 Q30 Q31 Q32 -> K8,V8
```

---

# 15. MHA vs MQA vs GQA

| Method | Q Heads | K Heads | V Heads | KV Cache |
|---|---:|---:|---:|---|
| MHA | many | many | many | largest |
| MQA | many | 1 | 1 | smallest |
| GQA | many | fewer | fewer | middle |

Mental model:

```text
MHA:
Every query head has its own K/V head.

MQA:
All query heads share K/V.

GQA:
Groups of query heads share K/V.
```

---

# 16. Numerical GQA Example

Suppose:

```text
Query heads = 32
KV heads = 8
Head dimension = 128
```

Each KV head serves:

```text
32 / 8 = 4
```

query heads.

Compared with MHA:

```text
32 KV heads -> 8 KV heads
```

So the K/V cache associated with head count is reduced by:

```text
32 / 8 = 4x
```

under the same other assumptions.

---

# 17. MQA vs GQA Mental Model

Think about a classroom.

MHA:

```text
Student 1 -> Teacher 1
Student 2 -> Teacher 2
Student 3 -> Teacher 3
...
```

MQA:

```text
Students 1..32 -> Teacher 1
```

GQA:

```text
Students 1..4   -> Teacher 1
Students 5..8   -> Teacher 2
Students 9..12  -> Teacher 3
...
```

GQA provides a middle ground between maximum K/V independence and maximum K/V sharing.

---

# 18. Multi-Head Latent Attention (MLA)

Multi-Head Latent Attention is another approach to reducing KV-cache memory.

The basic idea differs from simply reducing the number of K/V heads.

Instead of storing the full K/V representations for every token, MLA introduces a lower-dimensional latent representation that can be cached.

Conceptually:

```text
Hidden state
    |
    v
Compression / latent representation
    |
    v
Cached latent state
    |
    v
Used by attention
```

The exact implementation depends on the architecture.

Key intuition:

```text
MHA:
Cache large K/V representations

GQA:
Cache fewer K/V heads

MLA:
Cache a compressed latent representation
```

---

# 19. Why MLA Can Reduce KV Memory

If the necessary attention information can be represented using a smaller latent vector:

```text
Full K/V representation
        |
        v
Large memory

Compressed latent representation
        |
        v
Smaller memory
```

then long-context inference can use less cache memory.

MLA is therefore a KV-cache compression strategy, rather than simply a head-sharing strategy.

---

# 20. MHA, MQA, GQA and MLA

```text
MHA
|
|-- Many Q heads
|-- Many K/V heads
|-- High KV-cache memory
|
v
MQA
|
|-- Many Q heads
|-- One K/V head
|-- Very low KV-cache memory
|
v
GQA
|
|-- Many Q heads
|-- Several K/V heads
|-- Middle ground
|
v
MLA
|
|-- Cache compressed latent representations
|-- Different mechanism from head sharing
|-- Can reduce KV-cache memory
```

Important distinction:

```text
MQA/GQA -> reduce K/V head count

MLA -> compress the information that must be cached
```

---

# 21. Attention Computation vs Memory Optimization

This distinction is critical.

Changing:

```text
MHA -> GQA
```

mainly reduces:

```text
K/V cache size
```

It does not turn full attention into linear-time attention.

Similarly:

```text
KV cache
```

does not remove the fundamental quadratic interaction structure of full attention during training or prompt processing.

Therefore:

```text
Attention complexity
        and
KV-cache memory
```

are related but different problems.

---

# 22. Prefill vs Decode

Modern LLM inference has two major stages.

## Prefill

The model processes the input prompt.

Example:

```text
10,000-token prompt
        |
        v
Transformer
        |
        +--> K cache
        +--> V cache
```

## Decode

The model generates new tokens one by one.

```text
Token 10,001
Token 10,002
Token 10,003
...
```

During decode:

```text
New Q
   |
   v
Attend to cached K/V
   |
   v
Generate next token
```

This distinction is extremely important for inference optimization.

---

# 23. Why KV Cache Helps Decode

Suppose the context already contains:

```text
10,000 tokens
```

Without KV caching, the model would repeatedly compute K and V for the existing context during generation.

With KV caching:

```text
Old K/V -> reused

Only the new token's
Q/K/V need to be computed
```

Then:

```text
New Q
  |
  v
Old K/V + New K/V
  |
  v
Attention
```

---

# 24. Important Limitation of KV Cache

KV caching does not make memory constant.

If:

```text
context length increases
```

then:

```text
number of cached K/V vectors increases
```

Therefore:

```text
KV cache memory ~= O(n)
```

This is why long-context serving can become memory-bound.

---

# 25. Flash Attention

Flash Attention is an algorithmic and systems-level optimization for attention computation.

The central idea is not to change the mathematical definition of attention, but to change how attention is computed and how data moves through GPU memory.

Naive attention conceptually does:

```text
Q K^T
    |
    v
large attention score matrix
    |
    v
softmax
    |
    v
attention probabilities
    |
    v
multiply by V
```

For long sequences, explicitly materializing the entire attention matrix can create substantial memory traffic.

Flash Attention uses tiling/blocking and efficient GPU memory access. It computes attention in blocks and uses an online softmax strategy so the complete attention matrix does not need to be stored in high-bandwidth memory in the naive way.

---

# 26. Why Memory Movement Matters

A simplified GPU memory hierarchy is:

```text
GPU registers
    |
    v
GPU shared memory / SRAM
    |
    v
GPU high-bandwidth memory
```

Moving data between memory levels costs time and bandwidth.

A computation can have the same mathematical complexity but run much faster when memory access is organized efficiently.

Flash Attention focuses heavily on this issue.

---

# 27. Tiling Intuition

Instead of computing attention over the entire sequence at once:

```text
Q ---------------------------->
K ---------------------------->
V ---------------------------->

Huge attention matrix
```

Flash Attention processes blocks:

```text
Q block 1 <-> K block 1
Q block 1 <-> K block 2
Q block 1 <-> K block 3

Q block 2 <-> K block 1
Q block 2 <-> K block 2
...
```

The algorithm keeps useful intermediate data in faster memory as much as possible and avoids storing the entire attention matrix in the usual way.

---

# 28. Does Flash Attention Change Attention?

The conceptual equation remains:

```text
Attention(Q,K,V)
=
softmax((Q K^T) / sqrt(d_k)) V
```

Flash Attention changes the implementation strategy.

Therefore:

```text
Standard attention:
same math + potentially inefficient memory behavior

Flash Attention:
same attention result conceptually + more memory-efficient computation
```

---

# 29. Flash Attention and Complexity

A common misunderstanding is:

```text
Flash Attention removes O(n^2)
```

For standard full attention, that is not the correct mental model.

Full attention still has a fundamentally quadratic number of pairwise interactions.

Flash Attention improves:

```text
GPU memory usage
Memory traffic
Runtime
IO efficiency
```

The attention matrix does not need to be materialized in the same way as in a naive implementation.

---

# 30. Standard Attention vs Flash Attention

| Property | Standard / Naive Attention | Flash Attention |
|---|---|---|
| Attention mathematics | Same | Same |
| Full pairwise interaction | Yes | Yes |
| Avoids large attention matrix materialization | Usually no | Yes |
| Memory traffic | Higher | Lower |
| GPU efficiency | Lower | Higher |
| Full-attention interaction count | O(n^2) | O(n^2) |
| Main benefit | Simplicity | Memory and speed efficiency |

---

# 31. Efficient Attention: Four Different Problems

It helps to separate the optimization targets.

## Problem 1: Too many token-to-token interactions

```text
O(n^2)
```

Potential approaches:

```text
Sparse attention
Local attention
Sliding-window attention
Linear attention
Other approximate attention mechanisms
```

These change the attention pattern or mathematical strategy.

## Problem 2: KV cache is too large

Potential approaches:

```text
MQA
GQA
MLA
KV quantization
Paged KV-cache management
```

## Problem 3: Attention uses too much GPU memory bandwidth

Potential approaches:

```text
Flash Attention
Tiling
Kernel fusion
Memory-efficient kernels
```

## Problem 4: Context is too large to fit efficiently

Potential approaches:

```text
Chunking
RAG
Summarization
Memory systems
Retrieval
Long-context architectures
```

These are different engineering problems.

---

# 32. Local / Sliding-Window Attention

Full attention allows:

```text
Every token -> every previous token
```

Local attention limits the range.

For example, with a window of 3:

```text
Token 1 -> Token 1

Token 2 -> Token 1, Token 2

Token 3 -> Token 1, Token 2, Token 3

Token 4 -> Token 2, Token 3, Token 4

Token 5 -> Token 3, Token 4, Token 5
```

Instead of attending to the entire history, each token attends to a local neighborhood.

This can reduce the number of attention interactions.

---

# 33. Full Attention vs Sliding Window

Full causal attention:

```text
Q1 -> K1
Q2 -> K1,K2
Q3 -> K1,K2,K3
Q4 -> K1,K2,K3,K4
Q5 -> K1,K2,K3,K4,K5
```

Sliding-window attention:

```text
Q1 -> K1
Q2 -> K1,K2
Q3 -> K1,K2,K3
Q4 -> K2,K3,K4
Q5 -> K3,K4,K5
```

The second pattern restricts the receptive field.

---

# 34. Sparse Attention

Sparse attention means that a token does not attend to every other token.

```text
All possible edges
       |
       v
Select useful attention edges
       |
       v
Fewer interactions
```

Possible patterns include:

```text
Local attention
Global tokens
Block attention
Strided attention
Dilated patterns
Hybrid patterns
```

The goal is to reduce computation while preserving useful information flow.

---

# 35. Long-Context Challenges

Increasing context length creates several engineering problems.

## 35.1 Compute

Full attention has approximately:

```text
O(n^2)
```

pairwise interactions.

## 35.2 Memory

KV cache grows approximately:

```text
O(n)
```

during autoregressive decoding.

## 35.3 Latency

Longer prompts require more computation before generation can start.

## 35.4 Throughput

Large KV caches reduce how many concurrent requests can fit into GPU memory.

## 35.5 Information dilution

Even if a model technically supports a very long context, putting more information into the context does not guarantee that every piece will be used equally effectively.

Retrieval and context organization can remain important.

---

# 36. Long Context Does Not Mean Infinite Memory

Suppose a model supports:

```text
128K tokens
```

That does not mean:

```text
The model permanently remembers 128K tokens.
```

It means the model can process a context of approximately that size under its supported limits.

When application state exceeds the available context, the system may need to:

```text
Truncate
Summarize
Retrieve
Compress
Store externally
```

This is why production LLM systems often combine:

```text
LLM
+
Context management
+
Memory
+
RAG
```

---

# 37. Context Management in RAG

Suppose a user asks:

```text
"What does the third chapter say about vector databases?"
```

A naive system might put the entire book into the context.

A RAG system can instead:

```text
User query
    |
    v
Embedding / retrieval
    |
    v
Relevant chunks
    |
    v
Reranking / filtering
    |
    v
Prompt construction
    |
    v
LLM
```

The goal is not always:

```text
maximize context length
```

The goal is:

```text
provide the right information within the available context
```

---

# 38. Context Window vs KV Cache

These terms are often confused.

## Context window

The maximum amount of token context the model can process.

## KV cache

The stored K/V representations used during autoregressive inference.

Relationship:

```text
Larger context
      |
      v
More tokens
      |
      v
More cached K/V
      |
      v
Larger KV cache
```

They are related but not identical concepts.

---

# 39. Context Window vs Attention Complexity

Another common confusion:

```text
Context window = maximum supported length

Attention complexity = computational scaling with sequence length
```

Two models can have the same context length but different attention architectures, kernels, cache formats, and inference memory requirements.

---

# 40. Context Window vs Model Parameters

These are also different properties.

```text
Parameter count
    |
    +--> model capacity

Context length
    |
    +--> amount of token context processed

KV cache
    |
    +--> inference memory for cached attention states
```

A larger parameter count does not automatically imply a larger context window.

---

# 41. Big Numerical Comparison

Suppose:

```text
Model:
32 layers
32 query heads
128 head dimension
FP16
```

Compare:

```text
MHA:
KV heads = 32

GQA:
KV heads = 8

MQA:
KV heads = 1
```

Ignoring other memory overheads:

```text
MHA cache factor = 32
GQA cache factor = 8
MQA cache factor = 1
```

Relative to MHA:

```text
GQA:
32 / 8 = 4x smaller KV head storage

MQA:
32 / 1 = 32x smaller KV head storage
```

These are relative head-count effects, not universal end-to-end memory savings. Other model components and implementation details also consume memory.

---

# 42. One Complete Generation Example

Prompt:

```text
"The capital of France is"
```

Tokenized sequence:

```text
t1 t2 t3 t4 t5
```

During prefill:

```text
t1..t5
 |
 v
Transformer
 |
 +--> K1..K5
 +--> V1..V5
```

These K/V states are cached.

Now the model generates:

```text
"Paris"
```

For the next generation step:

```text
New token
   |
   v
Q6, K6, V6
```

The query attends to:

```text
K1 K2 K3 K4 K5 K6
```

and uses:

```text
V1 V2 V3 V4 V5 V6
```

Then:

```text
K6 -> added to cache
V6 -> added to cache
```

The cache becomes:

```text
K1..K6
V1..V6
```

This repeats for every generated token.

---

# 43. Where RoPE Fits

From Lesson 06:

```text
X
 |
 +--> Q
 +--> K
 +--> V
```

RoPE applies position-dependent rotation to:

```text
Q
K
```

not normally to V.

A simplified decoder attention flow:

```text
Hidden states
      |
      v
   Q K V
      |
      +--> RoPE(Q)
      |
      +--> RoPE(K)
      |
      v
Causal attention
      |
      v
Attention output
```

During decoding:

```text
Past K/V -> cached
New Q/K -> computed for current position
```

The exact cache representation depends on the model implementation.

---

# 44. Where GQA Fits

A simplified modern attention pipeline can be:

```text
Hidden state
     |
     v
Q, K, V projections
     |
     v
RoPE on Q/K
     |
     v
GQA head grouping
     |
     v
Causal attention
     |
     v
Attention output
```

Important:

```text
RoPE
```

addresses positional representation.

```text
GQA
```

reduces K/V head redundancy and cache memory.

They solve different problems and can be used together.

---

# 45. Where Flash Attention Fits

Flash Attention is primarily an implementation optimization around attention computation.

A conceptual stack:

```text
Transformer architecture
        |
        +--> Attention mathematics
        |
        +--> RoPE
        |
        +--> MHA / MQA / GQA / other attention design
        |
        +--> Flash Attention implementation
```

Do not confuse:

```text
GQA
```

with:

```text
Flash Attention
```

GQA changes the attention representation/head structure.

Flash Attention changes how the computation is executed efficiently.

---

# 46. Complete Modern Attention Picture

```text
                  Input tokens
                       |
                       v
                Token embeddings
                       |
                       v
                Hidden states
                       |
             +---------+---------+
             |         |         |
             v         v         v
             Q         K         V
             |         |
             v         v
            RoPE      RoPE
             |         |
             +----+----+
                  |
                  v
          MHA / MQA / GQA / MLA
                  |
                  v
          Causal attention
                  |
                  v
          Flash Attention
       (efficient implementation)
                  |
                  v
            Attention output
                  |
                  v
             FFN / MoE
                  |
                  v
           Transformer block
                  |
                  v
                Logits
                  |
                  v
              Next token
```

Note:

```text
MLA is an architectural approach to attention/KV representation.
Flash Attention is an efficient attention implementation.
```

---

# 47. Practical LLM Engineer Checklist

When evaluating a long-context LLM, ask:

```text
Context:
What is the maximum context length?

Tokenization:
How many tokens does my data consume?

Attention:
Is attention full, local, sparse, or hybrid?

KV cache:
How large is the KV cache per request?

Attention heads:
Does the model use MHA, MQA, GQA, MLA, or another design?

Position encoding:
Does it use RoPE or another positional mechanism?

Kernel:
Does inference use Flash Attention or another optimized kernel?

Serving:
How many concurrent sequences can fit into GPU memory?

Context management:
Does the application use RAG, summarization, memory, chunking, or compression?
```

---

# 48. Common Misconceptions

## Misconception 1

"A 128K context model has 128K tokens of permanent memory."

False.

Context is working context, not permanent memory.

## Misconception 2

"KV cache makes attention O(n)."

Not exactly.

KV caching avoids repeatedly recomputing past K/V during autoregressive decoding, but full attention still attends over the available context.

## Misconception 3

"GQA makes attention linear."

False.

GQA primarily reduces K/V head count and therefore KV-cache memory.

## Misconception 4

"Flash Attention removes quadratic attention."

False for standard full attention.

Flash Attention improves the implementation's memory behavior and runtime while preserving full attention.

## Misconception 5

"MQA and MLA are the same."

False.

MQA shares K/V heads.

MLA uses a latent/compressed representation strategy for attention and caching.

## Misconception 6

"Longer context is always better."

Not necessarily.

More context can increase compute, memory, latency, cost, and irrelevant information.

Good context selection is often more useful than simply adding more tokens.

---

# 49. Final Mental Model

Remember these five ideas:

```text
1. Context Window
   = how many tokens can be processed in the current context.

2. Full Self-Attention
   = roughly O(n^2) pairwise token interactions.

3. KV Cache
   = stores past K/V states to make autoregressive decoding efficient.
   Cache size grows roughly O(n).

4. MHA -> MQA -> GQA
   = progressively reduce K/V head redundancy.
   MQA uses maximum sharing.
   GQA provides a middle ground.

5. MLA + Flash Attention
   = different optimization directions.
   MLA reduces/compresses cached attention information.
   Flash Attention improves the efficiency of attention computation.
```

The big picture:

```text
Long context creates two major pressures:

        COMPUTE
           |
           v
      O(n^2) attention

        MEMORY
           |
           v
      O(n) KV cache


Different techniques attack different pressures:

Full/sparse/local attention
        -> attention computation pattern

MHA / MQA / GQA
        -> KV head count

MLA
        -> KV representation compression

Flash Attention
        -> memory traffic and kernel efficiency

RAG / memory / summarization
        -> reduce unnecessary context
```

---

# 50. Interview Questions

## Q1. Why is full self-attention O(n^2)?

Because every query can interact with every key:

```text
n queries x n keys = n^2 interactions
```

## Q2. What does the KV cache store?

Previously computed keys and values for attention layers during autoregressive generation.

## Q3. Why is Q usually not cached?

The current query is associated with the newly generated token and is used immediately to attend to the cached context.

## Q4. How does GQA reduce memory?

It uses fewer K/V heads than query heads, so fewer K/V representations need to be cached.

## Q5. What is the difference between MQA and GQA?

```text
MQA:
many Q heads -> one K/V head

GQA:
many Q heads -> several K/V heads
```

## Q6. What does Flash Attention optimize?

It optimizes the implementation of attention by reducing unnecessary memory movement and avoiding materialization of the full attention matrix in the naive way.

## Q7. Does Flash Attention change the attention equation?

Conceptually, no. It is primarily an efficient implementation of the same attention computation.

## Q8. Does GQA remove O(n^2) attention?

No. It primarily reduces K/V head count and KV-cache memory.

## Q9. What is the difference between context window and KV cache?

```text
Context window:
maximum/current token context.

KV cache:
stored K/V representations used during decoding.
```

## Q10. Why is long-context inference difficult?

Because longer sequences can increase:

```text
attention computation
KV-cache memory
latency
GPU memory pressure
serving cost
```

---

# 51. Lesson Summary

You should now understand:

```text
Context Window
       |
       v
maximum token context

Full Attention
       |
       v
O(n^2) interactions

KV Cache
       |
       v
stores past K/V for decoding

MHA
       |
       v
many K/V heads

MQA
       |
       v
one shared K/V head

GQA
       |
       v
groups of Q heads share K/V

MLA
       |
       v
compressed latent attention/cache representation

Flash Attention
       |
       v
efficient memory-aware attention implementation

RAG / Memory
       |
       v
avoid filling context with unnecessary information
```

---

# 52. Connection to Previous Lessons

```text
Lesson 03
Tokenization
    |
    v
Tokens

Lesson 04
Embeddings
    |
    v
Vector representations

Lesson 05
Transformer architecture
    |
    v
Q/K/V + attention + FFN

Lesson 06
Positional information
    |
    v
RoPE

Lesson 07
Mixture of Experts
    |
    v
Sparse FFN computation

Lesson 08
Context and efficient attention
    |
    +--> Context window
    +--> O(n^2) attention
    +--> KV cache
    +--> MHA
    +--> MQA
    +--> GQA
    +--> MLA
    +--> Flash Attention
    +--> Long-context strategies
```

---

# 53. Next Lesson

## Lesson 09 - Pre-training Objectives

Next, learn how an LLM actually learns from text.

Topics:

```text
Language modeling
      |
      v
Next-token prediction
      |
      v
Training targets
      |
      v
Cross-entropy loss
      |
      v
Logits
      |
      v
Softmax
      |
      v
Backpropagation
      |
      v
Parameter updates
```

You will connect the architecture from Lessons 05-08 to the actual training objective that teaches the model to predict the next token.

---

# End of Lesson 08
