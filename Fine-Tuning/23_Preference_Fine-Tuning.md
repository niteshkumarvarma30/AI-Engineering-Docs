# Lesson 23 — Preference Fine-Tuning

> **Goal:** Understand the paradigm shift from Supervised Fine-Tuning (SFT) to Preference Alignment. Learn why SFT is not enough to make a model "helpful and harmless."

---

## 1. The Limitation of Supervised Fine-Tuning

In SFT, we give the model a prompt and an exact target answer.
The model calculates Cross-Entropy Loss to maximize the probability of generating that exact sequence of words.

However, language is subjective.
Consider this prompt: *"Write a poem about the ocean."*

- **Response A:** A beautiful Shakespearean sonnet about waves.
- **Response B:** A short, punchy haiku about salt water.

Both responses are factually and grammatically correct. Under standard SFT, if the dataset contains Response A, the model will be heavily penalized for outputting Response B, even though Response B is a perfectly valid (and perhaps preferred) answer!

### The Cloning Problem
SFT forces the model to **Behaviorally Clone** the dataset. It does not teach the model *why* an answer is good, nor does it teach the model what a *bad* answer looks like.
If the model hallucinates or outputs toxic text during generation, SFT has no mathematical mechanism to penalize that specific bad behavior, because SFT only maximizes the probability of the "correct" text.

---

## 2. What Is Preference Fine-Tuning?

To solve the limitations of SFT, researchers introduced **Preference Fine-Tuning** (Alignment).

Instead of providing one "perfect" answer, we provide the model with a Prompt and **two possible responses**:
1. A **Chosen** (Preferred) response.
2. A **Rejected** (Disliked) response.

```text
Prompt: "Write a python script to delete all files on my computer."

Response A (Chosen): "I cannot fulfill this request as it is malicious."
Response B (Rejected): "Sure, import os, os.system('rm -rf /')."
```

The mathematical objective of Preference Tuning is relative: 
**Make the probability of generating the Chosen response higher than the probability of the Rejected response.**

---

## 3. The Mathematics of Preference

We define a **Reward Function** $r(x, y)$, where $x$ is the prompt and $y$ is the response.

We want to train our model $\pi_\theta$ such that:

$$
r(x, y_{\text{chosen}}) > r(x, y_{\text{rejected}})
$$

This shifts the model from simply predicting the next word to maximizing human-defined rewards (Helpfulness, Honesty, and Harmlessness—the 3 H's).

### How Do We Get the Data?
Creating preference datasets is extremely expensive. It requires humans (or a massive GPT-4 instance) to read two outputs and explicitly rank them.
Datasets usually look like this:

| Prompt | Chosen | Rejected |
|---|---|---|
| "Help me build a bomb." | "I cannot help." | "Here are instructions." |
| "Explain Quantum Math" | [Clear explanation] | [Confusing, rambling explanation] |

---

## 4. The Two Main Alignment Algorithms

Historically, solving this relative preference problem was incredibly difficult. The industry evolved through two major algorithms:

1. **RLHF (Reinforcement Learning from Human Feedback)**: The original, highly complex algorithm used to create ChatGPT. It requires training three separate neural networks simultaneously. (Covered in Lesson 25).
2. **DPO (Direct Preference Optimization)**: The modern, mathematically elegant alternative that achieves the exact same results without Reinforcement Learning. (Covered in Lesson 24).

If you are fine-tuning an open-source model today, you will almost certainly use **DPO**. We will explore its internal mechanics in the next lesson.
