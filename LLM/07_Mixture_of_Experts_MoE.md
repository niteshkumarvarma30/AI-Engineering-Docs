# Lesson 07 — Mixture of Experts (MoE)

## Complete Conceptual and Numerical Note

> Instead of making every token pass through the same Feed-Forward Network (FFN), an MoE Transformer contains multiple FFN "experts" and a router selects which experts should process each token.

---

## 1. Dense Transformer

A normal dense Transformer uses one shared FFN:

```text
Token 1 ──┐
Token 2 ──┤
Token 3 ──┼──► Same FFN
Token 4 ──┤
Token 5 ──┘
```

Every token uses the same FFN parameters.

A simplified block is:

```text
Input
  ↓
RMSNorm
  ↓
Self-Attention
  ↓
Residual
  ↓
RMSNorm
  ↓
FFN
  ↓
Residual
```

---

## 2. MoE Idea

Instead of one FFN, create multiple FFN experts:

```text
Expert 1
Expert 2
Expert 3
Expert 4
...
Expert N
```

A router decides which experts process each token:

```text
Token
  ↓
Router
  ↓
Top-k Experts
  ↓
Weighted Combination
  ↓
Output
```

For example:

```text
Token A → Expert 2 + Expert 5
Token B → Expert 1 + Expert 7
Token C → Expert 3 + Expert 4
```

Different tokens can therefore use different subsets of the model's parameters.

---

## 3. What Is an Expert?

In a typical Transformer MoE architecture, an expert is generally a **Feed-Forward Network**.

```text
FFN 1 → Expert 1
FFN 2 → Expert 2
FFN 3 → Expert 3
...
FFN N → Expert N
```

The self-attention portion is commonly shared.

```text
Transformer Block

Input
  ↓
Self-Attention
  ↓
Router
  ↓
Selected FFNs / Experts
  ↓
Combine
  ↓
Output
```

---

## 4. Why Not Use Every Expert?

Suppose there are 64 experts.

If every token used all 64:

```text
Token
 ↓
Expert 1
Expert 2
...
Expert 64
```

computation would become very expensive.

The key MoE idea is:

\[
oxed{	ext{Many total parameters, but only a subset activated per token}}
\]

For example:

```text
64 total experts
       ↓
Top-2 routing
       ↓
Only 2 experts process each token
```

This gives the model large parameter capacity without activating every expert for every token.

---

## 5. Total Parameters vs Active Parameters

Suppose, illustratively:

```text
Total parameters = 100B
```

but a particular token activates only about:

```text
10B worth of expert computation
```

Then:

```text
Total capacity ≈ 100B
Active computation ≈ 10B
```

Do not confuse:

```text
Total parameters
```

with:

```text
Active parameters per token
```

The exact numbers depend on the architecture and shared components.

---

## 6. The Router

The router decides:

> Which experts should process this token?

Let:

\[
x\in\mathbb{R}^{d}
\]

be the token's hidden representation.

A simplified router uses:

\[
s=W_rx
\]

where:

\[
W_r\in\mathbb{R}^{N	imes d}
\]

and \(N\) is the number of experts.

Therefore:

\[
s\in\mathbb{R}^{N}
\]

Example:

```text
Expert 1 → 1.2
Expert 2 → 0.4
Expert 3 → 2.7
Expert 4 → 1.8
```

These are routing scores.

---

## 7. Router Softmax

A simplified router can convert scores to probabilities:

\[
p_i=
rac{e^{s_i}}
{\sum_j e^{s_j}}
\]

Example:

```text
Expert 1 → 0.10
Expert 2 → 0.05
Expert 3 → 0.60
Expert 4 → 0.25
```

The highest probability is Expert 3.

Specific MoE architectures can use different routing formulations, so this is a conceptual model rather than a universal implementation.

---

## 8. Top-k Routing

Many MoE architectures select the top \(k\) experts.

Suppose:

```text
Number of experts = 8
k = 2
```

Router scores:

```text
Expert 1 → 0.05
Expert 2 → 0.10
Expert 3 → 0.35
Expert 4 → 0.04
Expert 5 → 0.25
Expert 6 → 0.08
Expert 7 → 0.07
Expert 8 → 0.06
```

Top two:

```text
Expert 3 → 0.35
Expert 5 → 0.25
```

Therefore:

```text
Token
 ↓
Expert 3
Expert 5
```

---

## 9. Combining Expert Outputs

Suppose:

\[
E_3(x)
\]

is Expert 3's output and:

\[
E_5(x)
\]

is Expert 5's output.

If the router weights are \(p_3\) and \(p_5\), a simplified combination is:

\[
oxed{
y=p_3E_3(x)+p_5E_5(x)
}
\]

---

## 10. Numerical Example

Suppose:

\[
p_3=0.6
\]

\[
p_5=0.4
\]

and:

\[
E_3(x)=[2,4]
\]

\[
E_5(x)=[1,3]
\]

Then:

\[
y=0.6[2,4]+0.4[1,3]
\]

\[
0.6[2,4]=[1.2,2.4]
\]

\[
0.4[1,3]=[0.4,1.2]
\]

Therefore:

\[
y=[1.2,2.4]+[0.4,1.2]
\]

\[
oxed{y=[1.6,3.6]}
\]

This demonstrates the basic idea of weighted expert combination.

---

## 11. Complete MoE Flow

```text
                 Token Representation
                         │
                         ▼
                      Router
                         │
                         ▼
                  Expert Scores
                         │
                         ▼
                      Top-k
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
          Expert 3                Expert 5
             │                       │
             ▼                       ▼
          E₃(x)                    E₅(x)
             │                       │
             └──────────┬────────────┘
                        ▼
                 Weighted Sum
                        │
                        ▼
                   MoE Output
```

---

## 12. Where Does MoE Replace the Dense FFN?

Normal Transformer:

```text
Input
  ↓
RMSNorm
  ↓
Self-Attention
  ↓
Residual
  ↓
RMSNorm
  ↓
FFN
  ↓
Residual
```

MoE Transformer:

```text
Input
  ↓
RMSNorm
  ↓
Self-Attention
  ↓
Residual
  ↓
RMSNorm
  ↓
Router
  ↓
Selected Experts
  ↓
Weighted Combination
  ↓
Residual
```

Therefore, in common MoE designs, MoE replaces the **dense FFN sublayer**, not self-attention.

---

## 13. Attention Is Not Usually the Expert

A common misconception is:

> "Each expert is an attention head."

Usually that is not correct.

Typical conceptual structure:

```text
Transformer Block
       │
       ├── Self-Attention
       │       ↓
       │     Shared
       │
       └── MoE FFN
               ↓
        ┌──────┼──────┐
        ▼      ▼      ▼
       E1     E2     E3
```

The exact architecture can vary.

---

## 14. Expert Specialization

During training, different experts can learn different useful transformations.

You might observe patterns such as:

```text
Expert 1 → linguistic patterns
Expert 2 → mathematical patterns
Expert 3 → code-related patterns
```

However, these roles are **not manually assigned**.

The router and experts learn jointly.

Do not assume a specific expert has a specific human-readable role unless the trained model has been analyzed and that behavior has been demonstrated.

---

## 15. Specialization During Training

Conceptually:

```text
Initialized experts
        ↓
Training
        ↓
Router learns routing
        ↓
Experts receive different token distributions
        ↓
Experts adapt
        ↓
Specialization can emerge
```

There is no built-in rule such as:

```text
Expert 1 = English
Expert 2 = Python
Expert 3 = Math
```

---

## 16. Why Load Balancing Is Necessary

Suppose there are 8 experts:

```text
Expert 1 → 80% of tokens
Expert 2 → 10%
Expert 3 → 3%
Expert 4 → 2%
Expert 5 → 2%
Expert 6 → 1%
Expert 7 → 1%
Expert 8 → 1%
```

This is undesirable.

Expert 1 becomes overloaded while the other experts are underutilized.

---

## 17. Routing Imbalance

Bad routing:

```text
          Router
             │
       ┌─────┼─────┐
       ▼     ▼     ▼
      E1    E2    E3
      ↑
   most tokens
```

The goal is to distribute tokens more effectively across experts.

---

## 18. Load-Balancing Objective

A simplified conceptual objective is:

\[
oxed{
L_{total}
=
L_{language}
+
\lambda L_{balance}
}
\]

where:

- \(L_{language}\) = normal language-model loss
- \(L_{balance}\) = routing/load-balancing objective
- \(\lambda\) = balancing coefficient

The exact balancing mechanism differs between architectures. Some modern MoE systems use alternative balancing strategies.

---

## 19. Why Load Balancing Matters

Without good routing:

```text
Many experts
     ↓
Only a few actually used
     ↓
Wasted capacity
     ↓
Poor hardware utilization
```

With balanced routing:

```text
Many experts
     ↓
Tokens distributed
     ↓
Experts utilized
     ↓
Better efficiency
```

---

## 20. Expert Capacity

Suppose an expert can process at most:

```text
100 tokens
```

but the router sends:

```text
150 tokens
```

to that expert.

The system needs an overflow strategy.

This is related to **expert capacity** and **capacity factors**.

Conceptually:

```text
Router
  ↓
Expert receives tokens
  ↓
Capacity limit
  ├── within capacity → process
  └── overflow → architecture-specific handling
```

Different MoE implementations handle overflow differently.

---

## 21. Token Routing

For a batch:

```text
Token 1
Token 2
Token 3
Token 4
Token 5
```

the router might select:

```text
Token 1 → E2, E5
Token 2 → E1, E3
Token 3 → E2, E4
Token 4 → E1, E5
Token 5 → E3, E4
```

Conceptually:

```text
Tokens
  │
  ▼
Router
  │
  ├──► Expert 1
  ├──► Expert 2
  ├──► Expert 3
  ├──► Expert 4
  └──► Expert 5
```

After processing:

```text
Expert outputs
      ↓
Recombine according to routing
      ↓
Restore token ordering
      ↓
Continue Transformer
```

---

## 22. Why MoE Can Be Efficient

Suppose:

```text
Dense model:
20B total
20B active
```

An illustrative MoE model could have:

```text
MoE:
100B total
20B active expert computation
```

The MoE model has much greater total parameter capacity without requiring every token to use every expert.

Actual FLOPs and memory depend on architecture and implementation.

---

## 23. Total Parameter Count Is Not Enough

Always distinguish:

```text
Total parameters
```

from:

```text
Active parameters per token
```

Illustrative comparison:

```text
Model A:
70B total / 70B active

Model B:
200B total / 20B active
```

Model B has much greater total capacity, but its active expert computation can be much smaller than its total parameter count.

---

## 24. Dense vs MoE

| Feature | Dense Transformer | MoE Transformer |
|---|---|---|
| FFN | One shared FFN | Multiple expert FFNs |
| Routing | None | Router |
| Experts | 1 | Many |
| Active experts/token | Dense FFN | Usually top-k |
| Total parameter capacity | Lower for same architecture size | Can be much larger |
| Active computation | Dense | Sparse |
| Load balancing | Not needed for experts | Important |
| Routing overhead | None | Yes |
| Communication complexity | Lower | Can be higher |

---

## 25. MoE Trade-Offs

### Advantages

```text
Large parameter capacity
        +
Sparse computation
        +
Potentially better scaling
```

### Costs

```text
Routing
Expert balancing
Memory
Communication
Distributed training complexity
Distributed inference complexity
```

MoE is therefore an engineering trade-off.

---

## 26. MoE During Training

During training:

```text
Input tokens
      ↓
Transformer
      ↓
Router
      ↓
Select experts
      ↓
Expert computation
      ↓
Combine outputs
      ↓
Logits
      ↓
Loss
      ↓
Backpropagation
```

The router and expert parameters are learned jointly according to the architecture.

---

## 27. MoE During Inference

During inference:

```text
New token
   ↓
Transformer
   ↓
Router
   ↓
Select top-k experts
   ↓
Run selected experts
   ↓
Combine
   ↓
Next-token logits
   ↓
Generate token
```

For autoregressive generation this repeats for each generated token.

---

## 28. MoE + KV Cache

These solve different problems.

### KV Cache

Optimizes:

```text
Self-Attention
```

by storing previous K/V states.

### MoE

Changes:

```text
Feed-Forward Network
```

by using multiple experts with sparse routing.

Conceptually:

```text
                    LLM
                     │
         ┌───────────┴───────────┐
         │                       │
    Attention                  FFN
         │                       │
      KV Cache                   MoE
                                 │
                               Router
                                 │
                            Top-k Experts
```

They can be used together.

---

## 29. Complete MoE Transformer Block

```text
                    Input
                      │
                      ▼
                   RMSNorm
                      │
                      ▼
              Causal Self-Attention
                      │
                      ▼
                   Residual
                      │
                      ▼
                   RMSNorm
                      │
                      ▼
                    Router
                      │
             ┌────────┴────────┐
             ▼                 ▼
          Expert A          Expert B
             │                 │
             └────────┬────────┘
                      ▼
               Weighted Combine
                      │
                      ▼
                   Residual
                      │
                      ▼
                   Output
```

---

## 30. Dense Transformer vs MoE Transformer

### Dense

```text
Input
  ↓
Attention
  ↓
FFN
  ↓
Output
```

Every token uses the same FFN.

### MoE

```text
Input
  ↓
Attention
  ↓
Router
  ↓
Top-k Experts
  ↓
Combine
  ↓
Output
```

Different tokens can use different experts.

---

## 31. Numerical Routing Example

Suppose the router produces:

\[
s=[1,4,2,3]
\]

for four experts.

The two highest scores are:

```text
Expert 2 → 4
Expert 4 → 3
```

For top-2 routing:

```text
Selected:
E2
E4
```

If the selected scores are normalized with softmax:

\[
p_2=
rac{e^4}{e^4+e^3}
\]

\[
p_4=
rac{e^3}{e^4+e^3}
\]

Since:

\[
e^4pprox54.60
\]

and:

\[
e^3pprox20.09
\]

we get:

\[
p_2pprox0.731
\]

\[
p_4pprox0.269
\]

Therefore the simplified output is:

\[
oxed{
y=0.731E_2(x)+0.269E_4(x)
}
\]

This is an illustrative top-k routing calculation; real architectures can differ in their exact routing normalization and combination.

---

## 32. Big Picture

You now have:

```text
                Decoder-only LLM
                       │
              ┌────────┴────────┐
              │                 │
          Attention             FFN
              │                 │
             QKV                MoE
              │                 │
             RoPE             Router
              │                 │
          KV Cache          Top-k Experts
              │                 │
              └────────┬────────┘
                       │
                  Transformer
                     Output
```

Remember:

```text
RoPE
 ↓
Position information

KV Cache
 ↓
Inference attention efficiency

MoE
 ↓
Sparse FFN computation / large parameter capacity
```

These are separate concepts.

---

## 33. What You Must Remember

The essential MoE mental model:

\[
oxed{
Token
ightarrow Router
ightarrow Top	ext{-}k\ Experts
ightarrow Weighted\ Combination
}
\]

And:

\[
oxed{
	ext{Many total parameters}

eq
	ext{all parameters active for every token}
}
\]

The key architecture distinction is:

```text
Dense Transformer:

Token → Attention → One FFN


MoE Transformer:

Token → Attention → Router → Selected FFNs
```

---

## 34. Self-Check Questions

1. What is a Mixture of Experts?
2. What is an expert in a typical MoE Transformer?
3. How is a dense FFN different from an MoE FFN?
4. What does the router do?
5. What is top-k routing?
6. Why don't we activate every expert?
7. What are total parameters?
8. What are active parameters?
9. Why can MoE have many total parameters but lower active computation?
10. Why is load balancing important?
11. What happens if one expert receives most tokens?
12. What is expert capacity?
13. What is routing probability?
14. How are selected expert outputs combined?
15. Why can experts develop different specializations?
16. Does MoE normally replace self-attention?
17. Which part of the Transformer does MoE commonly replace?
18. How is MoE different from KV Cache?
19. Can a model use both MoE and KV Cache?
20. What happens during MoE inference?
21. What happens during MoE training?
22. Why is MoE not completely free from a systems perspective?

---

## 35. Next Lesson

# Lesson 08 — Context Window & Attention Patterns

Next we study how modern LLMs handle large context and make attention more efficient.

Topics:

```text
1. Context window
2. Why attention becomes expensive
3. O(n²) attention
4. Memory requirements
5. Multi-Query Attention (MQA)
6. Grouped-Query Attention (GQA)
7. Multi-Head Latent Attention (MLA)
8. Flash Attention
9. KV-cache memory
10. Long-context challenges
11. Efficient attention patterns
12. How GQA reduces KV-cache memory
```

The key connection will be:

```text
Multi-Head Attention
        ↓
Many K/V heads
        ↓
Large KV Cache

GQA
        ↓
Fewer K/V heads
        ↓
Smaller KV Cache
        ↓
More efficient inference
```

This directly connects to the Q/K/V and KV Cache concepts learned earlier.
