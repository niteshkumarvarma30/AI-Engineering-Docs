# Reasoning Patterns: Chain-of-Thought, Self-Consistency, and Structured Reasoning

> **Module**: Agentic AI & Multi-Agent Systems  
> **Topic**: Reasoning Patterns  
> **Source**: [AIML Companion — Reasoning Patterns](https://aimlcompanion.ai/module/aiAgents/reasoningPatterns)

---

## 🎯 In This Module

- **The Reasoning Ladder**: Understanding the cost-accuracy spectrum across **Single-shot**, **Chain-of-Thought (CoT)**, and **Self-Consistency (SC)**.
- **Chain-of-Thought (CoT)**: Forcing models to write intermediate calculations into context (+30% to +50% accuracy boost on math and logic).
- **Self-Consistency**: Sampling diverse reasoning trajectories at elevated temperatures ($\tau = 0.7 - 1.0$) and taking majority consensus.
- **Zero-shot CoT**: What appending *"Let's think step by step"* physically does during token generation.
- **Budget-Aware Routing**: Directing tasks to the right reasoning rung to slash operational costs by ~70%.
- **Failure Modes & Tool Offloading**: Why reasoning cannot fix arithmetic hallucination or correlated training bias, and why calculators beat pure thinking.

---

## 1. The Big Picture: The Reasoning Ladder

Reasoning strategies form a progressive ladder where each rung trades additional token cost and latency for higher accuracy. **Climb only as far as the problem requires.**

```
+-------------------------------------------------------------------------------+
|                             THE REASONING LADDER                              |
|                                                                               |
|   [RUNG 3: SELF-CONSISTENCY]   Sample 5-10 CoT paths & majority vote          |
|   (5x - 15x Token Cost)        High-stakes decisions (Medical, Finance)       |
|            ▲                   Gain: +5% to +10% over single CoT              |
|            │                                                                  |
|   [RUNG 2: CHAIN-OF-THOUGHT]   Forces step-by-step intermediate tokens        |
|   (2x - 3x Token Cost)         Math, multi-hop logic, causal reasoning        |
|            ▲                   Gain: +30% to +50% on quantitative tasks       |
|            │                                                                  |
|   [RUNG 1: SINGLE-SHOT]        Answer directly from parametric weights        |
|   (1x Baseline Cost)           Factual recall, simple classification, lookup  |
+-------------------------------------------------------------------------------+
```

### The Core Strategy Formula
$$\mathbf{\text{Single-Shot (Facts)} \longrightarrow \text{Chain-of-Thought (Steps)} \longrightarrow \text{Self-Consistency (High Stakes)}}$$

* Use **Single-shot** when the model already possesses the static knowledge.
* Add **Chain-of-Thought** when the answer requires multi-step derivation.
* Reach for **Self-Consistency** only when the financial or operational cost of a wrong answer vastly exceeds the cost of multiplying token usage by $5\times - 10\times$.
* When a step is pure arithmetic, **a tool (calculator/code) beats any amount of reasoning**.

---

## 2. Walkthrough: One Problem Across All Three Rungs

**Problem**:
> *"A cafe sells 3 coffees at \$4 each and 2 muffins at \$3 each. With a 10% discount on the total, what is the final bill?"*

```
RUNG 1: Single-Shot (Direct Answer)
-----------------------------------
Prompt: "Calculate total: 3 coffees at $4, 2 muffins at $3, 10% discount."
Model Output: "$18.00"
Result: ❌ FAILED.
The model performed 3*4 + 2*3 = 18 in its internal activations, but rushed
to emit the final token and omitted the 10% discount step entirely.

RUNG 2: Chain-of-Thought (Step-by-Step)
---------------------------------------
Prompt: "Calculate total... Let's think step by step."
Model Output:
"1. Coffee subtotal: 3 * $4 = $12.
 2. Muffin subtotal: 2 * $3 = $6.
 3. Gross total: $12 + $6 = $18.
 4. Discount: 10% of $18 = $1.80.
 5. Final total: $18.00 - $1.80 = $16.20."
Result: ✅ SUCCESS (Winner).
Writing intermediate steps forces each calculation into the context window,
allowing the attention mechanism to attend to $18 when computing the discount.

RUNG 3: Self-Consistency (5x Sample + Majority Vote)
-----------------------------------------------------
Path 1: Computes steps -> $16.20
Path 2: Computes steps -> $16.20
Path 3: Computes steps -> $16.20
Path 4: Computes steps -> $16.20
Path 5: Computes steps -> $16.20
Majority Vote: $16.20 (5/5 consensus)
Result: ✅ Correct, but complete OVERKILL.
Costs ~5x more tokens for the exact same answer already guaranteed by CoT.
```

---

## 3. 💡 Mental Model: Exam Approaches

```
+-------------------------------------------------------------------------------+
| Strategy           | Exam Student Analogy                                     |
+--------------------+---------------------------------------------------------+
| ⚡ Single-Shot     | Instant Recall: Writing the answer immediately without    |
|                    | showing rough work (fine for "Paris is capital of France"|
| 📝 Chain-of-Thought| Showing Your Work: Writing out each intermediate formula |
|                    | step-by-step; easy to verify and catches arithmetic slips|
| 🗳️ Self-Consistency| The Study Group / 5 Independent Solvers: 5 students      |
|                    | solve the problem separately; you pick the majority answer|
+--------------------+---------------------------------------------------------+
```

---

## 4. Rung 2: Chain-of-Thought (CoT) Deep Dive

### Why CoT Works Mechanically
Transformers generate text autoregressively token-by-token. In single-shot prompting, the model must jump from the question directly to the answer in a single forward pass through its layers. 

With CoT:
1. **Working Memory Expansion**: Intermediate tokens are written to the context window, physically expanding the working memory the model can attend to.
2. **Decomposition**: Complex mathematical and logical dependencies are decomposed into sequential, verifiable sub-steps.
3. **Debuggability**: If an error occurs, the exact corrupted step is exposed in the trace.

```
Direct Guess (High Error Rate):
[Prompt: Roger has 5 balls, buys 2 cans of 3] ───────────────────────> [Output: 17 ❌]

Chain-of-Thought (Verifiable Links):
[Prompt] ──> [Step 1: 2 cans of 3 = 6] ──> [Step 2: 5 + 6 = 11] ──> [Output: 11 ✅]
```

### Few-Shot vs. Zero-Shot CoT

| Technique | Implementation | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Few-Shot CoT** | Include 2–3 worked input-reasoning-output examples in prompt | Highest accuracy; anchors reasoning style | Uses ~500–1,000 prompt tokens per call |
| **Zero-Shot CoT** | Append *"Let's think step by step"* to prompt | Zero example overhead; +25–35% free boost | Slightly higher variance than few-shot |

### High-Impact Zero-Shot Triggers
- `"Let's think step by step."` (Kojima et al., 2022 — most benchmarked)
- `"Let's work through this carefully."`
- `"Break this down into logical steps."`
- `"Think about this sequentially before answering."`

---

## 5. Rung 3: Self-Consistency (Sample & Vote)

Introduced by Wang et al. (2022), **Self-Consistency** replaces greedy decoding with diverse stochastic sampling across multiple independent reasoning paths.

```
                           [Complex User Query]
                                    │
        ┌──────────────┬────────────┼────────────┬──────────────┐
        ▼              ▼            ▼            ▼              ▼
     [Path 1]       [Path 2]     [Path 3]     [Path 4]       [Path 5]
    (temp=0.7)     (temp=0.7)   (temp=0.7)   (temp=0.7)     (temp=0.7)
        │              │            │            │              │
        ▼              ▼            ▼            ▼              ▼
      $5.00          $20.00       $5.00        $5.00          $5.00
        │              │            │            │              │
        └──────────────┴────────────┼────────────┴──────────────┘
                                    ▼
                         [Majority Voting Engine]
                                    │
                                    ▼
                        Consensus: $5.00 (80% vote)
```

### Key Operating Rules for Self-Consistency
1. **Temperature Must Be Elevated ($\tau = 0.7 - 1.0$)**: If you run Self-Consistency at `temperature=0.0`, all paths are identical and voting is useless.
2. **Marginal Accuracy Gain**: Typically adds **+5% to +10%** on top of CoT.
3. **Cost Multiplier**: $5\times - 10\times$ token consumption.
4. **Weighted Voting (Advanced)**: Rather than unweighted majority count, evaluate each path's coherence or log-probabilities to weight juror votes.

---

## 6. Budget-Aware Strategy Selection & Routing

In production systems, never run all requests through Self-Consistency. Implement an automated **complexity router**:

```
+-------------------------------------------------------------------------------+
| Task Profile         | Strategy          | Avg Cost | Typical Scenario        |
+----------------------+-------------------+----------+-------------------------+
| Simple / Factual     | Single-Shot       | $0.001   | "What is the capital?"  |
| Medium / Logical     | Zero-Shot CoT     | $0.003   | "Summarize & extract KPI"|
| Complex / High-Risk  | Self-Consistency  | $0.015   | Medical, Legal, Pricing |
+----------------------+-------------------+----------+-------------------------+
```

### Economic Impact of Routing
Assuming a production distribution of **50% Simple**, **30% Medium**, and **20% High-Risk**:
$$\text{Average Blended Cost} = (0.50 \times \$0.001) + (0.30 \times \$0.003) + (0.20 \times \$0.015) = \mathbf{\$0.0044 / request}$$
* Compared to running pure Self-Consistency across all tasks (\$0.015), **intelligent routing saves ~71% of operational inference costs**.

---

## 7. When Reasoning Patterns Fail (And Mitigations)

```
+------------------------------------------------------------------------------------+
| Failure Mode        | Root Cause                       | Production Mitigation     |
+---------------------+----------------------------------+---------------------------+
| 1. Error            | An arithmetic slip at step 1     | Offload math to a         |
|    Propagation      | corrupts all downstream steps    | calculator or Python tool |
+---------------------+----------------------------------+---------------------------+
| 2. Overthinking     | Adding CoT to simple tasks       | Route simple tasks to     |
|                     | introduces confusion & latency   | direct single-shot prompts|
+---------------------+----------------------------------+---------------------------+
| 3. Correlated       | All 5 Self-Consistency paths     | Add external retrieval /  |
|    Training Bias    | share the same pre-training bias | RAG to ground the facts   |
+---------------------+----------------------------------+---------------------------+
| 4. Format           | Chain-of-thought paragraphs break| Use Structured Outputs    |
|    Degradation      | downstream JSON parsers          | (`response_format`)       |
+---------------------+----------------------------------+---------------------------+
```

### The Hybrid Solution: CoT + Tools
The most powerful architecture pairs linguistic reasoning with deterministic tool execution:
> *"The LLM reasons about what formulas to execute; an isolated sandbox or calculator computes the exact numbers."*

---

## 8. ⚡ 2-Minute Cheatsheet & Interview Guide

- **The Ladder**: Single-shot ($1\times$, factual) $\rightarrow$ CoT ($2-3\times$, math/logic) $\rightarrow$ Self-Consistency ($5-15\times$, high-stakes).
- **Zero-Shot CoT**: Appending `"Let's think step by step"` activates sequential intermediate reasoning learned during training with zero examples.
- **Self-Consistency Temperature**: Must be sampled at $\tau = 0.7 - 1.0$ so trajectories explore diverse reasoning paths.
- **Correlated Bias Trap**: If a model has a fundamental knowledge gap or bias, all 5 paths make the same mistake. Voting cannot fix shared pre-training errors.
- **Tool Offloading**: Never let an LLM do pure multi-digit arithmetic in thought steps; hand it a calculator tool.

### Top Interview Questions & Model Answers
1. **Why does Chain-of-Thought drastically improve math accuracy over direct prompting?**  
   *Because transformers generate text sequentially. CoT writes intermediate calculations into the context window, providing explicit tokens for subsequent attention heads to attend to, turning a single speculative guess into a verifiable chain.*
2. **What is Self-Consistency and when is it justified?**  
   *Self-Consistency samples multiple diverse CoT reasoning trajectories at elevated temperatures and picks the majority answer. It is justified only for high-stakes decisions (e.g. medical, legal, financial) where the cost of error exceeds a 5x–10x token overhead.*
3. **If 10 Self-Consistency paths all vote for the same wrong answer, what happened?**  
   *Correlated error. All paths share the same underlying model weights and pre-training biases. Sampling cannot overcome systematic lack of knowledge; only external tools or RAG can fix this.*

---

## 9. 💻 Code Implementations

### Complete Suite: Single-Shot vs. Zero-Shot CoT vs. Self-Consistency

```python
"""
reasoning_patterns_suite.py
Implementation of Single-Shot, Zero-Shot CoT, and Self-Consistency with majority voting.
"""

from openai import OpenAI
from collections import Counter
import re

client = OpenAI()

# 1. Rung 1: Single-Shot (Direct Answer)
def solve_single_shot(question: str) -> str:
    """Fast, 1x cost, best for factual recall."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

# 2. Rung 2: Zero-Shot Chain-of-Thought (CoT)
def solve_zero_shot_cot(question: str) -> str:
    """Forces intermediate steps via 'Let's think step by step'."""
    prompt = f"{question}\n\nLet's think step by step."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

# 3. Rung 3: Self-Consistency (Sample & Vote)
def extract_numeric_answer(text: str) -> str:
    """Extracts the final dollar or numeric amount from text for voting."""
    matches = re.findall(r"\$?\d+(?:\.\d+)?", text)
    return matches[-1] if matches else text.strip()

def solve_self_consistency(question: str, num_samples: int = 5) -> tuple[str, float]:
    """
    Samples multiple reasoning paths at temp=0.7 and takes the majority vote.
    Returns: (consensus_answer, confidence_percentage)
    """
    prompt = (
        f"{question}\n\n"
        "Let's think step by step. Conclude with 'Final Answer: <number>'."
    )

    extracted_answers = []

    for _ in range(num_samples):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7  # Non-zero temperature is required for diversity
        )
        output = response.choices[0].message.content
        answer = extract_numeric_answer(output)
        extracted_answers.append(answer)

    # Compute majority vote
    vote_counts = Counter(extracted_answers)
    most_common_answer, highest_count = vote_counts.most_common(1)[0]
    confidence = highest_count / num_samples

    return most_common_answer, confidence

if __name__ == "__main__":
    test_question = (
        "A cafe sells 3 coffees at $4 each and 2 muffins at $3 each. "
        "With a 10% discount on the total bill, what is the final price?"
    )

    print("--- [Rung 1: Single-Shot] ---")
    print(solve_single_shot(test_question))

    print("\n--- [Rung 2: Zero-Shot CoT] ---")
    print(solve_zero_shot_cot(test_question))

    print("\n--- [Rung 3: Self-Consistency (5 samples)] ---")
    consensus, conf = solve_self_consistency(test_question, num_samples=5)
    print(f"Consensus Answer: {consensus} (Confidence: {conf:.0%})")
```
