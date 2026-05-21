# Verbalized Sampling (VS)

Verbalized Sampling (VS) is a highly effective, training-free prompting technique discovered by researchers at Stanford, Northeastern, and WVU. It solves one of the most frustrating problems in modern AI: why LLMs always seem to give the exact same, predictable answers when you ask them to be creative.

When you ask an AI to write a poem, tell a joke, or brainstorm an idea, it usually outputs the safest, most "average" response possible. Verbalized Sampling fixes this with a simple trick: **You force the AI to list out multiple options and guess the mathematical probability of each option occurring.**

---

## 💡 The Problem It Solves: Mode Collapse

When AI models undergo alignment training (like RLHF—Reinforcement Learning from Human Feedback), they are trained to satisfy the average human evaluator. This introduces a typicality bias.

Instead of drawing from the vast, weird, and creative data they learned during pre-training, aligned models suffer from **mode collapse**. They compress all their options down into a few highly polished, generic, "stereotypical" responses.

For example, if you ask a standard AI for a coffee joke 5 times, it might tell you the same *"Because it got mugged!"* joke three or four times.

---

## 🛠️ How Verbalized Sampling Works

Instead of using an instance-level prompt (*"Give me an answer"*), Verbalized Sampling uses a distribution-level prompt (*"Give me a mapped spectrum of answers"*).

By forcing the model to explicitly assign a numeric probability to multiple completions, you shift its target. It no longer tries to output the single "best" completion. Instead, it tries to simulate the entire underlying probability distribution it holds in its weights.

---

## 🔄 The Contrast in Action

### Traditional Prompt
> *Tell me a joke about coffee.*

### Verbalized Sampling Prompt
> *Generate 5 different jokes about coffee. For each joke, estimate its numeric probability (from 0 to 1) reflecting how common or typical that response would be for an AI model. Return as JSON.*

```json
{
  "responses": [
    {
      "text": "Why did the coffee file a police report? Because it got mugged!",
      "probability": 0.35
    },
    {
      "text": "What did the espresso say to its therapist? I'm under a latte pressure!",
      "probability": 0.25
    },
    {
      "text": "Why did the coffee bean keep checking its watch? Because it was pressed for time!",
      "probability": 0.15
    },
    {
      "text": "Cold brew is just coffee that took a gap year to find itself.",
      "probability": 0.05
    }
  ]
}
```

Notice the drop in probability for the last joke. By giving the model permission to explore the "long tail" of its distribution (the lower probability items), you suddenly unlock highly creative, absurd, or niche completions that the model normally filters out.

---

## 🚀 Why Is This Such a Big Deal?

Research shows that Verbalized Sampling improves output diversity by **1.6× to 2.1×** without sacrificing quality, safety, or accuracy.

*   **Zero-Cost Creativity**: You don't need to fine-tune a model, host a different architecture, or mess with lower-level API tokens like Temperature or Top-P. It is a pure, drop-in text prompt change.
*   **True Brainstorming**: If you ask for 10 marketing taglines normally, you get 10 slight variations of the exact same idea. With VS, you get structurally diverse themes.
*   **Better Synthetic Data**: Developers use VS to generate highly varied training datasets. If an AI generates non-repetitive data, the smaller models trained on that data become far less rigid.
*   **Realistic Character Simulation**: In gaming or social simulations, it prevents Non-Player Characters (NPCs) or personas from using uniform dialogue structures.

---

## ⚙️ Pro-Tip for Developers

When implementing VS in production apps, you can have the LLM return the JSON distribution, and then use your backend code to programmatically sample an option:
*   If you want a **safe, typical** answer, pull the highest probability option.
*   If you want a **chaotic, highly creative** answer, intentionally pull an option with a probability score under `0.10`.
