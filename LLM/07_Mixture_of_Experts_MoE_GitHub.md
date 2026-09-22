# Lesson 07 - Mixture of Experts (MoE)

## Complete Conceptual and Numerical Note

> Instead of making every token pass through the same Feed-Forward Network (FFN), an MoE Transformer contains multiple FFN "experts", and a router selects which experts should process each token.

---

# 1. Dense Transformer

A normal dense Transformer uses one shared FFN:

```text
Token 1 ----+
Token 2 ----+
Token 3 ----+----> Same FFN
Token 4 ----+
Token 5 ----+
```

Every token uses the same FFN parameters.

A simplified Transformer block is:

```text
Input
  |
  v
RMSNorm
  |
  v
Self-Attention
  |
  v
Residual Connection
  |
  v
RMSNorm
  |
  v
FFN
  |
  v
Residual Connection
```

---

# 2. MoE Idea

Instead of one FFN, create multiple FFN experts:

```text
Expert 1
Expert 2
Expert 3
Expert 4
...
Expert N
```

A router decides which experts should process each token:

```text
Token
  |
  v
Router
  |
  v
Top-k Experts
  |
  v
Weighted Combination
  |
  v
Output
```

For example:

```text
Token A -> Expert 2 + Expert 5
Token B -> Expert 1 + Expert 7
Token C -> Expert 3 + Expert 4
```

Different tokens can therefore use different subsets of the model's parameters.

---

# 3. What Is an Expert?

In a typical Transformer MoE architecture, an expert is generally a Feed-Forward Network.

Conceptually:

```text
FFN 1 -> Expert 1
FFN 2 -> Expert 2
FFN 3 -> Expert 3
...
FFN N -> Expert N
```

The self-attention portion is commonly shared.

```text
Transformer Block

Input
  |
  v
Self-Attention
  |
  v
Router
  |
  v
Selected FFNs / Experts
  |
  v
Combine
  |
  v
Output
```

The exact placement of the MoE sublayer depends on the architecture.

---

# 4. Why Not Use Every Expert?

Suppose there are 64 experts.

If every token used all 64:

```text
Token
  |
  +--> Expert 1
  +--> Expert 2
  +--> ...
  +--> Expert 64
```

The computation would become very expensive.

The key MoE idea is:

```text
Many total parameters
        +
Only a subset of experts activated per token
```

For example:

```text
64 total experts
       |
       v
Top-2 routing
       |
       v
Only 2 experts process each token
```

This gives the model a large parameter capacity without activating every expert for every token.

---

# 5. Total Parameters vs Active Parameters

Suppose, illustratively:

```text
Total parameters = 100B
```

but a particular token activates only about:

```text
10B worth of parameter computation
```

Then:

```text
Total capacity       ~= 100B
Active computation   ~= 10B
```

Do not confuse:

```text
Total parameters
```

with:

```text
Active parameters per token
```

The exact active parameter count depends on the architecture, shared parameters, routing strategy, and expert sizes.

Also note that "active parameters" and "FLOPs" are related but are not exactly the same thing.

---

# 6. The Router

The router decides:

```text
Which experts should process this token?
```

Let:

```text
x = token hidden representation
```

and suppose:

```text
x has dimension d
```

A simplified router can calculate:

```text
s = W_r x
```

where:

```text
W_r has shape N x d
N = number of experts
```

Therefore:

```text
s has N values
```

One value is produced for each expert.

Example:

```text
Expert 1 -> 1.2
Expert 2 -> 0.4
Expert 3 -> 2.7
Expert 4 -> 1.8
```

These are routing scores.

---

# 7. Router Softmax

A simplified router can convert scores into probabilities:

```text
p_i = exp(s_i) / sum_j exp(s_j)
```

For example, suppose the normalized probabilities are:

```text
Expert 1 -> 0.10
Expert 2 -> 0.05
Expert 3 -> 0.60
Expert 4 -> 0.25
```

The highest probability is:

```text
Expert 3 -> 0.60
```

Specific MoE architectures can use different routing formulations, so this is a conceptual model rather than a universal implementation.

---

# 8. Top-k Routing

Many MoE architectures select the top k experts.

Suppose:

```text
Number of experts = 8
k = 2
```

Router probabilities:

```text
Expert 1 -> 0.05
Expert 2 -> 0.10
Expert 3 -> 0.35
Expert 4 -> 0.04
Expert 5 -> 0.25
Expert 6 -> 0.08
Expert 7 -> 0.07
Expert 8 -> 0.06
```

The two highest values are:

```text
Expert 3 -> 0.35
Expert 5 -> 0.25
```

Therefore:

```text
Token
  |
  +--> Expert 3
  |
  +--> Expert 5
```

The selected routing weights may then be renormalized before combining the expert outputs, depending on the implementation.

---

# 9. Combining Expert Outputs

Suppose:

```text
E3(x) = output of Expert 3
E5(x) = output of Expert 5
```

If the routing weights are `p3` and `p5`, a simplified combination is:

```text
y = p3 * E3(x) + p5 * E5(x)
```

This produces the MoE output for that token.

---

# 10. Numerical Example

Suppose:

```text
p3 = 0.6
p5 = 0.4
```

and:

```text
E3(x) = [2, 4]

E5(x) = [1, 3]
```

Then:

```text
y = 0.6 * [2, 4] + 0.4 * [1, 3]
```

Calculate the first contribution:

```text
0.6 * [2, 4]
=
[1.2, 2.4]
```

Calculate the second contribution:

```text
0.4 * [1, 3]
=
[0.4, 1.2]
```

Add them:

```text
[1.2, 2.4]
+
[0.4, 1.2]
=
[1.6, 3.6]
```

Therefore:

```text
MoE output = [1.6, 3.6]
```

This demonstrates the basic idea of weighted expert combination.

---

# 11. Complete MoE Flow

```text
                 Token Representation
                         |
                         v
                      Router
                         |
                         v
                  Expert Scores
                         |
                         v
                      Top-k
                         |
             +-----------+-----------+
             |                       |
             v                       v
          Expert 3                Expert 5
             |                       |
             v                       v
          E3(x)                    E5(x)
             |                       |
             +-----------+-----------+
                         |
                         v
                  Weighted Sum
                         |
                         v
                     MoE Output
```

For a whole batch, many tokens can be routed to different experts simultaneously.

---

# 12. Where Does MoE Replace the Dense FFN?

A common Transformer design looks like:

```text
Input
  |
  v
RMSNorm
  |
  v
Self-Attention
  |
  v
Residual Connection
  |
  v
RMSNorm
  |
  v
FFN
  |
  v
Residual Connection
```

In an MoE version, the dense FFN sublayer can be replaced by an MoE layer:

```text
Input
  |
  v
RMSNorm
  |
  v
Self-Attention
  |
  v
Residual Connection
  |
  v
RMSNorm
  |
  v
Router
  |
  v
Selected Experts
  |
  v
Weighted Combination
  |
  v
Residual Connection
```

Therefore, in common MoE designs:

```text
MoE replaces the dense FFN sublayer.
```

It does not normally replace self-attention.

---

# 13. Attention Is Not Usually the Expert

A common misconception is:

> "Each expert is an attention head."

Usually, that is not correct.

A typical conceptual structure is:

```text
Transformer Block
       |
       +--> Self-Attention
       |       |
       |       +--> Shared attention mechanism
       |
       +--> MoE FFN
               |
        +------+------+
        |      |      |
        v      v      v
       E1     E2     E3
```

Attention heads are subdivisions of the attention mechanism.

MoE experts are generally separate FFN sub-networks.

The exact architecture can vary.

---

# 14. Expert Specialization

During training, different experts can learn different useful transformations.

You might observe patterns such as:

```text
Expert 1 -> certain linguistic patterns
Expert 2 -> mathematical patterns
Expert 3 -> code-related patterns
```

However, these roles are not manually assigned.

The router and experts learn jointly.

Do not assume that a specific expert has a specific human-readable role unless the trained model has actually been analyzed and that behavior has been demonstrated.

---

# 15. Specialization During Training

Conceptually:

```text
Initialized experts
        |
        v
Training
        |
        v
Router learns routing
        |
        v
Experts receive different token distributions
        |
        v
Experts adapt
        |
        v
Specialization can emerge
```

There is no built-in rule such as:

```text
Expert 1 = English
Expert 2 = Python
Expert 3 = Math
```

Specialization, if it emerges, is learned from training.

---

# 16. Why Load Balancing Is Necessary

Suppose there are 8 experts:

```text
Expert 1 -> 80% of tokens
Expert 2 -> 10%
Expert 3 -> 3%
Expert 4 -> 2%
Expert 5 -> 2%
Expert 6 -> 1%
Expert 7 -> 1%
Expert 8 -> 1%
```

This creates a routing imbalance.

Expert 1 becomes overloaded while the other experts are underutilized.

---

# 17. Routing Imbalance

Bad routing can look like:

```text
                 Router
                    |
       +------------+------------+
       |            |            |
       v            v            v
      E1           E2           E3
      ^
      |
  Most tokens
```

The goal is to distribute tokens more effectively across experts.

This matters both for efficient computation and for making use of the available expert capacity.

---

# 18. Load-Balancing Objective

A simplified conceptual objective is:

```text
L_total
=
L_language
+
lambda * L_balance
```

where:

```text
L_language = normal language-model loss
L_balance  = routing/load-balancing objective
lambda     = balancing coefficient
```

The exact balancing mechanism differs between architectures.

Some modern MoE systems use different routing and balancing strategies, so the equation above should be treated as a conceptual representation rather than a universal MoE loss.

---

# 19. Why Load Balancing Matters

Without good routing:

```text
Many experts
     |
     v
Only a few receive most tokens
     |
     v
Some experts become overloaded
     |
     v
Other experts are underutilized
     |
     v
Poor hardware utilization
```

With more balanced routing:

```text
Many experts
     |
     v
Tokens distributed more evenly
     |
     v
Experts utilized
     |
     v
Better efficiency
```

Balanced routing is especially important in distributed MoE systems.

---

# 20. Expert Capacity

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

This is related to:

```text
Expert capacity
Capacity factor
Token dropping or rerouting
```

Conceptually:

```text
Router
  |
  v
Expert receives tokens
  |
  v
Capacity limit
  |
  +---- within capacity --> process
  |
  +---- overflow --------> architecture-specific handling
```

Different MoE implementations handle overflow differently.

---

# 21. Token Routing

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
Token 1 -> E2, E5
Token 2 -> E1, E3
Token 3 -> E2, E4
Token 4 -> E1, E5
Token 5 -> E3, E4
```

Conceptually:

```text
Tokens
  |
  v
Router
  |
  +----> Expert 1
  +----> Expert 2
  +----> Expert 3
  +----> Expert 4
  +----> Expert 5
```

After processing:

```text
Expert outputs
      |
      v
Recombine according to routing weights
      |
      v
Restore token ordering
      |
      v
Continue through the Transformer
```

---

# 22. Why MoE Can Be Efficient

Suppose, illustratively:

```text
Dense model:
20B total parameters
20B active parameters per token
```

An MoE model could have:

```text
MoE:
100B total parameters
Only a subset of expert parameters active for each token
```

The MoE model can therefore provide much larger total parameter capacity without requiring every token to use every expert.

Actual compute, memory, and communication costs depend on the architecture and implementation.

---

# 23. Total Parameter Count Is Not Enough

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

Model B has much greater total parameter capacity, but its expert computation for an individual token can be much smaller than its total parameter count.

Also remember:

```text
Active parameters != exact inference FLOPs
```

because shared layers, routing, attention, and implementation details also contribute to computation.

---

# 24. Dense vs MoE

| Feature | Dense Transformer | MoE Transformer |
|---|---|---|
| FFN | One shared FFN | Multiple expert FFNs |
| Routing | None | Router |
| Experts | One FFN | Many FFNs |
| Active experts per token | One shared FFN | Usually top-k experts |
| Total parameter capacity | Usually lower for comparable active compute | Can be much larger |
| Active computation | Dense | Sparse in expert layer |
| Load balancing | Not needed for experts | Important |
| Routing overhead | None | Yes |
| Communication complexity | Generally simpler | Can be higher in distributed systems |

---

# 25. MoE Trade-Offs

## Advantages

```text
Large parameter capacity
        +
Sparse expert computation
        +
Potentially better scaling
```

## Costs

```text
Routing
Expert balancing
Memory
Communication
Distributed training complexity
Distributed inference complexity
```

MoE is therefore an engineering trade-off rather than a free reduction in computation.

---

# 26. MoE During Training

During training:

```text
Input tokens
      |
      v
Transformer
      |
      v
Router
      |
      v
Select experts
      |
      v
Expert computation
      |
      v
Combine outputs
      |
      v
Logits
      |
      v
Language-model loss
      |
      v
Backpropagation
```

The router and expert parameters are learned jointly according to the architecture.

---

# 27. MoE During Inference

During inference:

```text
New token
   |
   v
Transformer
   |
   v
Router
   |
   v
Select top-k experts
   |
   v
Run selected experts
   |
   v
Combine outputs
   |
   v
Next-token logits
   |
   v
Generate token
```

For autoregressive generation, this process repeats as new tokens are generated.

---

# 28. MoE and KV Cache

MoE and KV Cache solve different problems.

## KV Cache

KV Cache optimizes repeated self-attention computation during autoregressive generation by storing previously computed Key and Value states.

```text
Past tokens
    |
    v
Past K and V
    |
    v
KV Cache
```

## MoE

MoE changes the feed-forward part of the Transformer:

```text
Dense FFN
   |
   v
Multiple expert FFNs
   |
   v
Router selects experts
```

Conceptually:

```text
                    LLM
                     |
         +-----------+-----------+
         |                       |
     Attention                  FFN
         |                       |
      KV Cache                   MoE
                                 |
                               Router
                                 |
                            Top-k Experts
```

They can be used together.

---

# 29. Complete MoE Transformer Block

A simplified block can look like:

```text
                    Input
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
                    Router
                      |
             +--------+--------+
             |                 |
             v                 v
          Expert A          Expert B
             |                 |
             +--------+--------+
                      |
                      v
               Weighted Combine
                      |
                      v
              Residual Connection
                      |
                      v
                    Output
```

The exact normalization and residual ordering depends on the model architecture.

---

# 30. Dense Transformer vs MoE Transformer

## Dense

```text
Input
  |
  v
Attention
  |
  v
FFN
  |
  v
Output
```

Every token uses the same FFN parameters.

## MoE

```text
Input
  |
  v
Attention
  |
  v
Router
  |
  v
Top-k Experts
  |
  v
Combine
  |
  v
Output
```

Different tokens can use different experts.

---

# 31. Numerical Routing Example

Suppose the router produces scores:

```text
s = [1, 4, 2, 3]
```

for four experts.

The two highest scores are:

```text
Expert 2 -> 4
Expert 4 -> 3
```

For top-2 routing:

```text
Selected experts:
E2
E4
```

If the selected scores are normalized with softmax:

```text
p2 = exp(4) / (exp(4) + exp(3))

p4 = exp(3) / (exp(4) + exp(3))
```

Using:

```text
exp(4) ~= 54.60
exp(3) ~= 20.09
```

we get approximately:

```text
p2 ~= 0.731

p4 ~= 0.269
```

Therefore the simplified output is:

```text
y = 0.731 * E2(x) + 0.269 * E4(x)
```

This is an illustrative top-k routing calculation.

Real architectures can differ in their routing normalization, auxiliary losses, capacity handling, and combination strategy.

---

# 32. Big Picture

You now have three separate concepts:

```text
                Decoder-only LLM
                       |
              +--------+--------+
              |                 |
          Attention             FFN
              |                 |
             QKV                MoE
              |                 |
             RoPE             Router
              |                 |
          KV Cache          Top-k Experts
              |                 |
              +--------+--------+
                       |
                  Transformer
                     Output
```

Remember:

```text
RoPE
  |
  v
Position information in Q/K

KV Cache
  |
  v
Efficient autoregressive attention

MoE
  |
  v
Sparse FFN computation and large parameter capacity
```

These are separate concepts, although they can coexist in the same model.

---

# 33. What You Must Remember

The essential MoE mental model is:

```text
Token
  |
  v
Router
  |
  v
Top-k Experts
  |
  v
Weighted Combination
  |
  v
MoE Output
```

And:

```text
Many total parameters
        !=
All parameters active for every token
```

The key architecture distinction is:

```text
Dense Transformer:

Token -> Attention -> One shared FFN


MoE Transformer:

Token -> Attention -> Router -> Selected FFNs
```

---

# 34. Self-Check Questions

1. What is a Mixture of Experts?
2. What is an expert in a typical MoE Transformer?
3. How is a dense FFN different from an MoE FFN?
4. What does the router do?
5. What is top-k routing?
6. Why do we not activate every expert?
7. What are total parameters?
8. What are active parameters?
9. Why can MoE have many total parameters but lower active expert computation?
10. Why is load balancing important?
11. What happens if one expert receives most tokens?
12. What is expert capacity?
13. What is a routing probability?
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

# 35. Next Lesson

# Lesson 08 - Context Window and Attention Patterns

Next we study how modern LLMs handle large context and make attention more efficient.

Topics:

```text
1. Context window
2. Why attention becomes expensive
3. O(n^2) attention
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
        |
        v
Many K/V heads
        |
        v
Large KV Cache

GQA
        |
        v
Fewer K/V heads
        |
        v
Smaller KV Cache
        |
        v
More efficient inference
```

This directly connects to the Q/K/V and KV Cache concepts learned earlier.
