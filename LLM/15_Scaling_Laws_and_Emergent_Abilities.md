# Scaling Laws & Emergent Abilities

## Complete LLM Scaling Notes

Scaling laws explain how LLM performance changes as we increase model parameters, training data, training compute, post-training compute, and test-time compute.

> **Core idea:** LLM capability does not scale simply by making models bigger. Modern systems improve by allocating computation intelligently across data, architecture, post-training, and inference.

---

## 1. The Big Picture

Imagine an AI lab has a fixed compute budget:

```text
                    Compute Budget
                          |
          +---------------+---------------+
          |               |               |
      Pre-training    Post-training   Test-time
          |               |               |
     Parameters        RLHF / GRPO     Reasoning
     Data              Alignment       More tokens
     Architecture      Tool use       Verification
```

The older strategy was largely:

```text
Make the model bigger
        ↓
Train with more compute
        ↓
Get better performance
```

The modern strategy is increasingly:

```text
Scale smarter
    ↓
Better data
    ↓
Better architecture
    ↓
Better post-training
    ↓
More inference-time reasoning
```

---

## 2. What Are Scaling Laws?

Scaling laws are empirical relationships describing how model performance changes as we increase model size, training data, and training compute.

A simplified relationship is:

```text
L ∝ C^(-α)
```

where:

- `L` = training loss
- `C` = training compute
- `α` = scaling exponent

A simplified example is:

```text
L ∝ C^(-0.05)
```

The exact exponent is not universal. It depends on the model family, dataset, training regime, and measurement.

> **More compute generally reduces training loss, but the marginal improvement becomes smaller as scale increases.**

---

## 3. What Is a Power Law?

Suppose compute increases by 10×. It does not mean the model becomes 10× better.

Using `L ∝ C^(-0.05)`:

```text
10^(-0.05) ≈ 0.891
```

So the remaining loss is approximately 89.1% of the original loss:

```text
10× compute
      ↓
~11% lower loss
```

For 2× compute:

```text
2^(-0.05) ≈ 0.966
```

Therefore:

```text
2× compute
      ↓
~3.4% lower loss
```

This is **diminishing returns**.

---

## 4. Why Scaling Laws Matter

Training a frontier LLM can require enormous resources. Researchers can first run smaller experiments:

```text
Small training runs
        ↓
Measure loss
        ↓
Fit scaling relationship
        ↓
Forecast larger training run
        ↓
Choose compute budget
```

Scaling laws help answer:

- How much compute should we use?
- Should we increase model size?
- Should we add more data?
- What loss might we achieve?
- Is the additional compute worth the cost?

---

## 5. Training Loss ≠ Capability

Suppose:

```text
Model A
Loss = 2.0

Model B
Loss = 1.8
```

Model B has lower loss, but we cannot automatically conclude that it is 10% better at reasoning or can solve 10% more problems.

Training loss measures a specific predictive objective. It does not directly measure every capability users care about.

```text
Training loss
      ↓
Smooth improvement

Downstream capability
      ↓
May change differently
```

A small reduction in loss might unlock a useful behavior, or produce little noticeable change.

---

## 6. Chinchilla: Compute-Optimal Training

The Chinchilla work showed that model size and training data should be balanced more carefully than the earlier strategy of simply making models as large as possible.

Simplified comparison:

```text
Gopher
280B parameters
300B tokens
≈ 1 token / parameter
```

versus:

```text
Chinchilla
70B parameters
1.4T tokens
≈ 20 tokens / parameter
```

At comparable training compute, the smaller Chinchilla model performed better on the reported benchmarks.

> **A smaller model trained on substantially more data can be more compute-efficient than a much larger model trained on too little data.**

---

## 7. What Does 20 Tokens per Parameter Mean?

For a 70B-parameter model:

```text
70B × 20 = 1.4T tokens
```

So approximately:

```text
70B parameters
+
1.4 trillion training tokens
```

Important:

> **20 tokens per parameter is not a universal law. It is an approximate compute-optimal result under the assumptions of the Chinchilla analysis.**

---

## 8. Compute-Optimal vs Inference-Optimal

### Compute-optimal

Question:

> How should I spend training compute efficiently?

Approximate Chinchilla-style idea:

```text
~20 tokens / parameter
```

### Inference-optimal

Question:

> How should I train a model that will be cheap to serve at very large scale?

A possible strategy is:

```text
Smaller model
+
Much more training data
```

Example:

```text
70B parameters

20:1
↓
1.4T tokens

200:1
↓
14T tokens
```

The second model requires much more training data, but the resulting model can remain cheaper to serve than a substantially larger model.

---

## 9. Why Inference Cost Changes the Ratio

Training usually happens once. Inference happens repeatedly.

```text
Training
   ↓
Pay once

Inference
   ↓
Pay for every request
```

For millions or billions of requests, inference cost can dominate lifetime economics.

Therefore:

```text
Higher training cost
        +
Smaller inference model
        ↓
Potentially lower total lifetime cost
```

This motivates deliberately training smaller models on more data.

---

## 10. What Is Over-Training?

In this context, over-training means deliberately training a model on substantially more tokens than the original compute-optimal recipe would suggest for its parameter count.

It does not mean training incorrectly.

Example:

```text
70B model

Chinchilla-style:
~1.4T tokens

More heavily trained:
many more trillions of tokens
```

The goal can be:

```text
Improve a smaller model
      ↓
Keep inference cost low
      ↓
Serve at large scale
```

---

## 11. Emergent Abilities

An **emergent ability** is a capability that appears to improve sharply once a model reaches a certain scale.

Example:

```text
Model Size      Accuracy

1B              5%
5B              6%
10B             7%
20B             8%
40B             80%
```

It can look like a sudden transition.

---

## 12. Water-Boiling Analogy

```text
20°C → Liquid
40°C → Liquid
60°C → Liquid
80°C → Liquid
99°C → Liquid
100°C → Boiling
```

Some LLM benchmark results have similarly appeared to show capabilities changing sharply at a scale threshold.

This led to the idea of **emergent abilities**.

---

## 13. The Emergence Debate

The key question is:

> Are capabilities genuinely appearing suddenly?

Some research reported many apparently emergent abilities. Later work argued that some apparent jumps can result from the evaluation metric itself.

### Metric example

Suppose underlying capability improves gradually:

```text
20%
30%
40%
50%
60%
70%
```

But the evaluation is all-or-nothing:

```text
Correct = 1
Wrong   = 0
```

The measured result might look like:

```text
0%
0%
0%
0%
100%
100%
```

The model may be improving smoothly while the metric creates an apparent jump.

> **Some apparent emergence may therefore be a measurement artifact rather than a truly discontinuous change in the underlying capability.**

The practical improvement can still be very real: going from 5% to 75% accuracy is a major user-visible change.

---

## 14. Why Emergence Matters

If a 3B model fails a task, do not automatically conclude that the architecture can never solve it.

Evaluate multiple scales:

```text
3B  → weak
7B  → moderate
13B → stronger
30B → strong
```

This helps distinguish a genuine scale threshold from gradual improvement.

---

## 15. Why Can't We Keep Scaling Forever?

Raw scaling faces several constraints:

```text
Data
Compute
Energy
Hardware
Money
```

These are the major scaling bottlenecks.

---

## 16. The Data Wall

Traditional training data includes:

- Websites
- Books
- Wikipedia
- Code
- Articles
- Documentation
- Public documents

High-quality human-generated data is finite.

```text
High-quality data
        ↓
Most useful data consumed
        ↓
Less new high-quality data
        ↓
Data becomes a bottleneck
```

This is the **data wall**.

---

## 17. Why Not Reuse the Same Data?

Repeatedly training on identical data has limitations such as:

- Overfitting
- Memorization
- Reduced diversity
- Lower marginal value per token

Therefore, modern training increasingly uses:

```text
Data filtering
Data deduplication
Synthetic data
Generated reasoning traces
High-quality curated datasets
```

---

## 18. Synthetic Data

Synthetic data is data generated by models or automated systems.

Example:

```text
Strong model
      ↓
Generate math problems
      ↓
Generate solutions
      ↓
Verify / filter
      ↓
Training dataset
      ↓
Train another model
```

Synthetic data can extend the available training corpus, but correctness, diversity, and contamination risks must be managed carefully.

---

## 19. Compute and Energy Walls

Frontier training requires enormous computation:

```text
More parameters
      +
More training tokens
      +
More training steps
      ↓
More GPU compute
      ↓
More infrastructure
      ↓
Higher cost
```

Large-scale AI also requires substantial electricity and cooling:

```text
GPU computation
     ↓
Electricity
     ↓
Heat
     ↓
Cooling
```

A large training cluster also needs:

- Accelerators
- High-bandwidth memory
- Fast networking
- Storage
- Power
- Cooling

---

## 20. Scale Smarter, Not Just Bigger

When raw pre-training scaling becomes increasingly expensive, other scaling axes become important:

```text
1. Architecture
2. Post-training
3. Test-time compute
```

---

## 21. Mixture of Experts (MoE)

MoE stands for **Mixture of Experts**.

Instead of activating the entire model for every token, an MoE model contains multiple expert networks and a routing mechanism.

```text
                Input Token
                     ↓
                   Router
                /    |    \
               ↓     ↓     ↓
           Expert A Expert B Expert C
```

The router selects which experts should process the token.

---

## 22. Dense Model vs MoE

A dense model might have:

```text
100B parameters
```

with approximately all of them participating for every token.

An MoE model might have:

```text
500B total parameters
```

while only a subset is active for each token.

Therefore:

```text
MoE

Total parameters = very large
Active parameters per token = much smaller
```

This can provide:

```text
Large total capacity
        +
Lower active computation
```

---

## 23. Why MoE Helps

The main idea is:

```text
Total model capacity ↑
        +
Active computation per token ↓
```

Potential benefits include:

- High total capacity
- Lower active compute per token
- Better capability-to-compute trade-offs

Engineering challenges include:

- Expert routing
- Load balancing
- Inter-device communication
- Memory requirements
- Serving complexity

MoE is therefore not automatically cheaper for every workload.

---

## 24. Post-Training

Pre-training provides broad language and world knowledge. Post-training can improve how the model behaves.

```text
Pre-training
     ↓
Base Model
     ↓
Instruction tuning
     ↓
Preference / RL training
     ↓
Better Assistant
```

Post-training can improve:

- Instruction following
- Reasoning behavior
- Safety
- Tool use
- Coding behavior
- Helpfulness
- Response style

---

## 25. RLHF

RLHF means **Reinforcement Learning from Human Feedback**.

Simplified pipeline:

```text
Prompt
   ↓
Model generates responses
   ↓
Humans compare / rank responses
   ↓
Preference signal
   ↓
Reward / RL optimization
   ↓
Improved model behavior
```

The key idea is:

> The model is optimized not only to predict text, but also toward behaviors humans prefer.

---

## 26. GRPO

GRPO means **Group Relative Policy Optimization**.

A simplified intuition is:

```text
Prompt
   ↓
Generate multiple candidate responses
   ↓
Evaluate responses
   ↓
Compare candidates relative to each other
   ↓
Reinforce stronger responses
```

For a reasoning problem:

```text
Question
   |
   +--> Answer A → Incorrect
   |
   +--> Answer B → Correct
   |
   +--> Answer C → Partially correct
   |
   +--> Answer D → Correct
```

The relative quality of generated responses can provide a learning signal.

---

## 27. RLHF vs GRPO

| Aspect | RLHF | GRPO |
|---|---|---|
| Full name | Reinforcement Learning from Human Feedback | Group Relative Policy Optimization |
| Main idea | Optimize toward preference feedback | Compare multiple generated responses |
| Feedback | Often human preference-derived | Often relative/group-based reward |
| Common use | Alignment and instruction following | Reasoning-focused optimization |
| Core idea | Learn preferred behavior | Improve responses relative to alternatives |

These are not mutually exclusive concepts; they describe different parts of the post-training landscape.

---

## 28. Test-Time Compute

Test-time compute means:

> **Using additional computation while answering a particular query.**

Traditional generation:

```text
Question
   ↓
Model
   ↓
Answer
```

Test-time reasoning:

```text
Question
   ↓
Generate reasoning
   ↓
Check / explore / verify
   ↓
Generate more reasoning if needed
   ↓
Final answer
```

Instead of increasing model parameter count, we increase computation spent on the problem.

---

## 29. Easy vs Hard Problems

Not every question requires the same amount of reasoning.

```text
Easy task
→ Less test-time compute

Hard task
→ More test-time compute
```

For a difficult problem, the model may:

```text
Generate solution
      ↓
Check solution
      ↓
Try another approach
      ↓
Compare
      ↓
Final answer
```

This is a form of adaptive computation.

---

## 30. Training-Time vs Test-Time Scaling

### Training-time scaling

```text
Training compute ↑
       ↓
Model/data scale ↑
       ↓
Potentially better model
```

### Test-time scaling

```text
Inference compute ↑
       ↓
More reasoning / search / verification
       ↓
Potentially better answer
```

---

## 31. Multi-Dimensional Scaling

The old mental model was:

```text
Bigger model
      ↓
Better model
```

The modern mental model is:

```text
                 Better AI
                    ↑
       +------------+------------+
       |            |            |
   Pre-training  Post-training  Test-time
       |            |            |
   Data/Params   RLHF/GRPO     Reasoning
       |
      MoE
```

This is **multi-dimensional scaling**.

---

## 32. Comparing the Main Scaling Strategies

| Strategy | What is scaled? | Main purpose |
|---|---|---|
| Bigger dense model | Parameters | Increase model capacity |
| More training data | Tokens | Improve learning |
| Chinchilla-style training | Parameters + tokens | Allocate training compute efficiently |
| Over-training | Tokens relative to parameters | Improve smaller inference models |
| MoE | Total capacity with sparse activation | Increase capacity efficiently |
| RLHF | Post-training optimization | Improve desired behavior |
| GRPO | Relative RL optimization | Improve reasoning behavior |
| Test-time compute | Inference computation | Spend more compute on hard problems |

---

## 33. Full LLM Scaling Pipeline

```text
                    Compute Budget
                          ↓
                  Scaling Laws
                          ↓
              Predict Training Loss
                          ↓
             Allocate Parameters/Data
                          ↓
                     Pre-training
                          ↓
                  Base Language Model
                          ↓
                  Architecture Choices
                          ↓
                       MoE / Dense
                          ↓
                    Post-training
                    /           \
                 RLHF          GRPO
                    \           /
                     Better Model
                          ↓
                   User Query
                          ↓
              Test-Time Compute
                          ↓
               Reason / Verify / Search
                          ↓
                     Final Answer
```

---

## 34. Practical Example

Suppose an AI company wants to build a model for millions of users.

### Option A: Huge Dense Model

```text
Very large parameter count
        ↓
Very expensive training
        ↓
Very expensive inference
```

### Option B: Smaller Model + More Data

```text
Smaller parameter count
        ↓
More training tokens
        ↓
Higher training investment
        ↓
Cheaper inference
```

### Option C: MoE

```text
Large total capacity
        ↓
Only selected experts active
        ↓
Lower active computation
```

### Option D: Post-training

```text
Base model
    ↓
RLHF / GRPO
    ↓
Improve behavior / reasoning
```

### Option E: Test-time reasoning

```text
Easy query
→ little reasoning

Hard query
→ more reasoning
```

A modern system can combine all of these.

---

## 35. Common Misconceptions

### Misconception 1: 10× compute means 10× better model

Wrong. Scaling laws imply diminishing returns.

```text
10× compute
→ ~11% lower loss in the simplified example
```

not:

```text
10× compute
→ 10× capability
```

### Misconception 2: Lower loss automatically means better reasoning

Not necessarily.

```text
Lower training loss
≠
Guaranteed downstream capability improvement
```

### Misconception 3: Chinchilla says every model must use exactly 20 tokens per parameter

Wrong. It is an approximate compute-optimal result under a particular analysis.

### Misconception 4: Over-training means the model was trained badly

No. Here it means deliberately using more training tokens relative to model size.

### Misconception 5: Emergent abilities prove sudden intelligence at a fixed parameter count

Not necessarily. Some apparent emergence can be caused by discontinuous evaluation metrics.

### Misconception 6: MoE means only a small model exists

No. An MoE can have a very large total parameter count while activating only a subset per token.

### Misconception 7: Test-time compute means increasing model parameters

No. It means allocating more computation during inference for a particular problem.

### Misconception 8: More test-time compute always improves every task

No. It is most useful when additional reasoning, search, verification, or sampling can improve the solution.

---

## 36. Interview Questions

### Q1. What are scaling laws?

Scaling laws are empirical power-law relationships showing how training loss changes as model parameters, data, and compute increase. They help researchers forecast model behavior and allocate training budgets.

### Q2. What does diminishing returns mean in scaling?

It means that increasing compute continues to improve training loss, but each additional unit of compute produces a smaller marginal improvement than earlier compute.

### Q3. What did Chinchilla demonstrate?

Chinchilla demonstrated that compute-optimal training requires balancing model size and training data. A smaller model trained on substantially more tokens can outperform a larger model trained on insufficient data at comparable training compute.

### Q4. What is the difference between compute-optimal and inference-optimal training?

Compute-optimal training focuses on minimizing training compute for a target loss, while inference-optimal training considers lifetime serving cost and may deliberately train a smaller model on more tokens.

### Q5. What is an emergent ability?

An emergent ability is a capability that appears to improve sharply after a model reaches a certain scale. Some apparent discontinuities can result from the evaluation metric rather than a discontinuous underlying capability.

### Q6. Why can't we keep scaling indefinitely?

Because scaling faces diminishing returns and constraints involving high-quality data, compute, energy, hardware, infrastructure, and cost.

### Q7. What is MoE?

Mixture-of-Experts is an architecture in which a router selects a subset of expert networks for each token, allowing a model to have large total capacity while keeping active computation lower than a dense model of the same total size.

### Q8. What is RLHF?

RLHF stands for Reinforcement Learning from Human Feedback. It uses preference information to optimize model behavior toward desired responses.

### Q9. What is GRPO?

GRPO stands for Group Relative Policy Optimization. It can compare multiple candidate responses and use their relative rewards to optimize the model.

### Q10. What is test-time compute?

Test-time compute is additional computation allocated during inference, such as reasoning, sampling, search, or verification, to improve performance on a particular query.

---

## 37. Final Mental Model

Think of LLM scaling as a set of knobs:

```text
                         LLM CAPABILITY
                              ↑
             +----------------+----------------+
             |                |                |
        Pre-training     Post-training     Test-time
             |                |                |
       +-----+-----+      RLHF / GRPO      Reasoning
       |           |
   Parameters     Data
       |
      MoE
```

The old strategy:

```text
Turn the parameter knob to maximum
```

The smarter strategy:

```text
Find the bottleneck
      ↓
Choose the right scaling axis
      ↓
Spend compute where it creates the most value
```

---

## 38. One-Line Summary of Each Concept

| Concept | Simple meaning |
|---|---|
| Scaling Laws | More scale improves loss predictably but with diminishing returns |
| Power Law | Improvement follows a nonlinear relationship with scale |
| Diminishing Returns | More compute produces smaller marginal gains |
| Chinchilla | Balance parameters and training tokens for compute-efficient training |
| 20 Tokens/Parameter | Approximate Chinchilla-style compute-optimal ratio |
| Over-training | Deliberately train a smaller model on many more tokens |
| Inference-optimal | Optimize the model for lifetime serving cost |
| Emergent Ability | Ability that appears to improve sharply at scale |
| Metric Artifact | Evaluation method can make smooth progress look sudden |
| Data Wall | High-quality training data is finite |
| Compute Wall | Frontier training requires enormous compute |
| Energy Wall | Large-scale AI requires substantial electricity and cooling |
| MoE | Activate only selected experts for each token |
| RLHF | Optimize behavior using human preference feedback |
| GRPO | Improve policy using relative comparison of generated responses |
| Test-time Compute | Spend more inference computation on difficult problems |
| Scale Smarter | Use multiple scaling axes instead of only increasing parameters |

---

## 39. Final Takeaways

1. **Scaling laws** show that training loss generally follows predictable power-law relationships with scale.
2. **Diminishing returns** mean that 10× more compute does not produce 10× better performance.
3. A relationship such as `L ∝ C^-0.05` is an example, not a universal constant.
4. **Training loss is not the same as downstream capability.**
5. **Chinchilla** showed the importance of balancing model parameters and training tokens.
6. The approximate **20 tokens/parameter** rule is a compute-optimal result under a particular training regime, not a universal requirement.
7. Production objectives can favor **over-training smaller models** because inference costs accumulate across many requests.
8. **Emergent abilities** describe capabilities that appear to change sharply at certain scales.
9. Some apparent emergence can result from **discontinuous evaluation metrics**.
10. High-quality training data is finite, creating a **data bottleneck**.
11. Frontier training also faces **compute, energy, hardware, and economic constraints**.
12. **MoE** increases total model capacity while activating only a subset of experts per token.
13. **RLHF** improves behavior using preference feedback.
14. **GRPO** can improve reasoning through relative comparison of multiple generated responses.
15. **Test-time compute** increases computation during inference rather than simply increasing parameter count.
16. Modern LLM development is increasingly **multi-dimensional scaling**.
17. The key strategy is:

```text
Scale smarter,
not just bigger.
```

---

## 40. Final Mental Shortcut

```text
SCALING LAWS
     ↓
More compute → Lower loss
     ↓
Diminishing returns
     ↓
CHINCHILLA
     ↓
Balance parameters + data
     ↓
PRODUCTION
     ↓
Train smaller models longer
     ↓
EMERGENCE
     ↓
Loss ≠ Capability
     ↓
DATA / COMPUTE / ENERGY WALLS
     ↓
SCALE SMARTER
     ↓
MoE + Post-training + Test-time Compute
```

## Core Idea

> **The future of LLM scaling is not simply making models bigger. It is deciding where additional computation creates the most useful capability: during pre-training, in the architecture, during post-training, or at inference time.**
