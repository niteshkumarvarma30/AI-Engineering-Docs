# Lesson 25 — Reinforcement Learning from Human Feedback (RLHF)

> **Goal:** Understand the classical RLHF pipeline used by OpenAI to create ChatGPT, including the three-step process: SFT, Reward Modeling, and PPO (Proximal Policy Optimization).

---

## 1. Why Study RLHF if DPO Exists?

In Lesson 24, we saw that DPO mathematically replaces RLHF by treating the Language Model as its own reward model. 
However, **RLHF is still widely used in frontier labs** (like OpenAI, Anthropic, and DeepMind) because explicitly training a separate Reward Model can sometimes capture complex human nuances better than the implicit probabilities of DPO.

To understand AI engineering at the highest level, you must understand the RLHF architecture.

---

## 2. Step 1: Supervised Fine-Tuning (SFT)

Before an AI can be "aligned," it must first be able to follow instructions.
We take a base model (trained on internet text) and fine-tune it on thousands of high-quality, human-written demonstrations.

```text
Prompt: "Write a polite email to decline an invitation."
Human Writer: "Dear John, thank you so much for the invitation. Unfortunately..."
```

After Step 1, the model is a competent assistant, but it might still hallucinate or produce toxic content because it is just behavioral cloning (as discussed in Lesson 23).

---

## 3. Step 2: Training the Reward Model (RM)

We want to train a "critic" model that can read an AI's response and output a scalar score (e.g., `+4.5` for excellent, `-2.0` for terrible).

1. We take our SFT model and give it a prompt: *"Explain quantum mechanics."*
2. We make the model generate **multiple different responses**.
3. **Human labelers** read the responses and rank them from best to worst.

| Prompt | Response A | Response B | Human Preference |
|---|---|---|---|
| "Explain math" | Clear, polite | Confusing, rude | A > B |

We train a **Reward Model** (usually initialized from the SFT model, but with its language generation head replaced by a single scalar regression head). 
The loss function forces the Reward Model to output a higher number for Response A than Response B.

---

## 4. Step 3: Proximal Policy Optimization (PPO)

Now we have two models:
1. Our **Policy Model** (the LLM we are trying to align).
2. Our **Reward Model** (the critic we just trained).

We freeze the Reward Model. We will use **Reinforcement Learning (PPO)** to train the Policy Model.

### The PPO Loop
1. We give the Policy Model a prompt: *"Write a joke."*
2. The Policy Model generates a response.
3. We pass this response to the frozen **Reward Model**, which gives it a score (e.g., `+3.2`).
4. We use this score as a **Reward Signal** to update the weights of the Policy Model using PPO.

```text
[Prompt] ---> (Policy Model) ---> [Response]
                                      |
                                      v
                                (Reward Model) ---> [Score: +3.2]
                                      |
       Update Weights via PPO <-------+
```

### The KL Divergence Penalty
If left unchecked, the Policy Model will realize it can "hack" the Reward Model. For example, if the Reward Model likes polite words, the Policy Model might just output: *"Please please please thank you thank you."* to get a massive score.

To prevent this, RLHF uses a **KL Divergence Penalty**. 
We keep a frozen copy of the original SFT model. During the PPO loop, we check how mathematically different the training model's output is from the frozen SFT model. If it deviates too far, we penalize the reward. 
This forces the model to stay coherent and grammatically correct while maximizing the reward.
