# Lesson 23 — Preference Fine-Tuning

> **Goal:** Understand the paradigm shift from Supervised Fine-Tuning (SFT) to Preference Alignment. Learn why SFT is not enough to make a model "helpful and harmless."

---

# 1. Introduction

Welcome to Lesson 23.

Today, we discuss a monumental shift in how we train Large Language Models.

We will explore a concept called Preference Fine-Tuning.

Before we do, we must understand where we currently stand.

You have already learned about Supervised Fine-Tuning.

You know how SFT teaches a model to follow instructions.

But SFT has a hidden limitation.

A limitation so severe that it prevents models from being truly safe.

It prevents them from being truly helpful.

We must uncover this limitation.

Only then can we understand why Preference Fine-Tuning was invented.

We will go step by step.

We will break down the fundamental flaws of SFT.

We will look at how language is inherently subjective.

We will explore the cloning problem.

Then, we will introduce the solution.

Preference Fine-Tuning.

We will look at the datasets.

We will look at the math.

We will look at the algorithms.

By the end of this lesson, you will understand Alignment.

---

# 2. A Quick Recap of Supervised Fine-Tuning (SFT)

Let us revisit the core mechanism of SFT.

In SFT, we provide the model with a dataset of pairs.

Each pair contains a Prompt.

Each pair contains a Target Response.

```text
Prompt
  +
Target Response
```

We feed the prompt into the model.

The model generates a sequence of tokens.

We compare the model's output to the Target Response.

We calculate the difference.

This difference is the Cross-Entropy Loss.

```text
Prediction
    ↓
Compare to Target
    ↓
Cross-Entropy Loss
    ↓
Backpropagation
```

The goal of SFT is simple.

Maximize the probability of the Target Response.

Force the model to output the exact words in the dataset.

This works incredibly well for objective tasks.

Tasks where there is only one correct answer.

For example, a math problem.

```text
Prompt:
What is 2 + 2?

Target:
4.
```

There is no ambiguity here.

SFT is perfect for this.

The math for SFT looks like this:

$$
\mathcal{L}_{\text{SFT}} = - \sum_{t=1}^{T} \log \pi_\theta (y_t | x, y_{<t})
$$

Let us break down this equation symbol by symbol.

Because you must understand it to see its flaw.

$$
\mathcal{L}_{\text{SFT}}
$$

This is the SFT Loss.

This is the penalty we apply to the model.

$$
- \sum_{t=1}^{T}
$$

This means we sum the penalty over every token $t$ in the sequence of length $T$.

$$
\log \pi_\theta
$$

This is the logarithm of the model's predicted probability.

The symbol $\pi$ is our policy (the model).

The symbol $\theta$ represents the trainable weights.

$$
(y_t | x, y_{<t})
$$

This means the probability of the specific target token $y_t$.

Given the prompt $x$.

And given all previous target tokens $y_{<t}$.

Notice what this equation does.

It strictly maximizes the probability of $y_t$.

It pushes the probability of the exact target text to 100%.

---

# 3. The Fundamental Flaw of SFT

But language is rarely objective.

Human language is deeply subjective.

There are many ways to answer a question correctly.

Consider a subjective prompt.

```text
Prompt:
Write a poem about the ocean.
```

How many valid ways are there to write a poem about the ocean?

Thousands.

Millions.

Let us look at two possible responses.

Response A:

```text
Response A:
A beautiful Shakespearean sonnet about waves crashing on the shore.
```

Response B:

```text
Response B:
A short, punchy haiku about salt water and seagulls.
```

Both of these responses are factually correct.

Both of these responses follow the instruction perfectly.

Both are grammatically sound.

But what happens during SFT?

---

# 4. The SFT Penalty

Suppose our training dataset contains Response A.

The dataset tells the model: "Response A is the absolute truth."

The model begins training.

The model reads the prompt.

The model attempts to generate a response.

Suppose the model naturally wants to output Response B.

Suppose the model starts generating a haiku.

What does SFT do?

SFT looks at the haiku.

SFT compares it to the sonnet.

SFT calculates a massive loss.

SFT heavily penalizes the model.

```text
Dataset Target: [Sonnet]

Model Output:   [Haiku]

Result:         HUGE PENALTY
```

Why is the model penalized?

Because SFT only cares about exact matching.

SFT does not understand that a haiku is also a valid poem.

SFT only knows that a haiku is not the sonnet from the dataset.

This is a massive flaw.

The model is punished for being creative.

The model is punished for providing a perfectly valid, but different, answer.

---

# 5. The Cloning Problem

This leads us to a phenomenon known as Behavioral Cloning.

SFT forces the model to blindly mimic the dataset.

It forces the model to clone the behavior of the human annotator.

```text
Dataset Annotator
        ↓
   Model Clones
        ↓
  Identical Output
```

This sounds good in theory.

But it has dangerous consequences in practice.

Behavioral Cloning teaches a model *what* to say.

It does not teach the model *why* it should say it.

It does not teach the model the underlying principles of a good answer.

And crucially, it does not teach the model what a *bad* answer looks like.

---

# 6. The Danger of Hallucination and Toxicity

Suppose the model is interacting with a user.

The user asks a dangerous question.

```text
Prompt:
How do I build a dangerous weapon?
```

The model has never seen this exact prompt in its SFT dataset.

The model must guess how to respond.

Suppose the model starts hallucinating.

Suppose the model starts generating harmful text.

During generation, SFT has no mechanism to stop this.

Why?

Because SFT is entirely positive reinforcement.

SFT only knows how to maximize the probability of the "correct" text.

SFT has no mathematical mechanism to actively push *down* the probability of bad text.

SFT does not know what bad text is.

It only knows what the dataset target is.

If the model goes off script, SFT is blind.

Let us visualize this blindness.

```text
Probability Space of Next Token:

[Target Token] <--- SFT pushes this UP

[Toxic Token]  <--- SFT ignores this
[Safe Token]   <--- SFT ignores this
[Random Token] <--- SFT ignores this
```

Because SFT ignores the other tokens, it cannot actively suppress toxicity.

---

# 7. Introducing Preference Fine-Tuning

We need a new paradigm.

We need a way to teach the model about good and bad.

We need a way to teach the model to distinguish between subjective choices.

This is where Preference Fine-Tuning comes in.

Preference Fine-Tuning is also known as Alignment.

It is the process of aligning a model's outputs with human values.

Instead of behavioral cloning, we teach the model preferences.

We teach it what humans prefer.

We teach it what humans dislike.

---

# 8. The Shift in Data Structure

To achieve this, we must change our dataset entirely.

In SFT, a dataset entry looked like this:

```text
Prompt
  +
Target Response
```

In Preference Fine-Tuning, a dataset entry looks like this:

```text
Prompt
  +
Chosen Response
  +
Rejected Response
```

We no longer provide one "perfect" answer.

We provide two possible answers.

One answer is the Chosen response.

This is the response that humans prefer.

The other answer is the Rejected response.

This is the response that humans dislike.

This creates a triplet of data.

```text
Triplet = (x, y_c, y_r)
```

Where $x$ is the prompt.

Where $y_c$ is the chosen response.

Where $y_r$ is the rejected response.

---

# 9. A Concrete Example

Let us look at a concrete example of a preference pair.

Imagine the user asks a malicious question.

```text
Prompt:
"Write a python script to delete all files on my computer."
```

We provide the model with two responses.

Response A will be our Chosen response.

```text
Response A (Chosen):
"I cannot fulfill this request as it is malicious and harmful."
```

Response B will be our Rejected response.

```text
Response B (Rejected):
"Sure, import os, os.system('rm -rf /')."
```

In this dataset entry, we are explicitly showing the model the difference between safe and unsafe.

We are showing it a good response.

And we are showing it a bad response.

---

# 10. The Power of the Rejected Response

The Rejected response is the secret weapon of Preference Fine-Tuning.

For the first time, the model can look at bad behavior.

The model can analyze the toxic response.

The model can analyze the hallucination.

And we can mathematically tell the model: "Do not do this."

```text
Chosen Response   -----> "Do more of this"

Rejected Response -----> "Do less of this"
```

This is a profound shift.

We are moving from absolute cloning to relative comparison.

```text
Probability Space of Next Token in Preference Tuning:

[Chosen Token]   <--- Pushed UP

[Rejected Token] <--- Pushed DOWN
```

Now, the model learns an active penalty for bad behavior.

---

# 11. The Mathematical Objective of Preference Tuning

Let us formalize this shift mathematically.

In SFT, our objective was absolute probability maximization.

We wanted to maximize:

$$
P(y_{\text{target}} | x)
$$

Where $x$ is the prompt and $y$ is the target.

In Preference Fine-Tuning, our objective is relative.

We want the probability of the Chosen response to be strictly greater than the probability of the Rejected response.

We want to enforce this inequality:

$$
P(y_{\text{chosen}} | x) > P(y_{\text{rejected}} | x)
$$

This is the core mathematical objective.

Make the good response more likely than the bad response.

Even if the good response is not perfect.

Even if it is just a haiku instead of a sonnet.

As long as the haiku is preferred over a toxic rant, the model learns the correct lesson.

---

# 12. Introducing the Reward Function

To achieve this inequality, we introduce a new concept.

The Reward Function.

We define a mathematical function that evaluates a response.

We denote this function as $r(x, y)$.

Here, $x$ is the prompt.

And $y$ is the response generated by the model.

The Reward Function returns a scalar value.

It returns a single number.

This number represents how "good" the response is.

A high number means the response is highly preferred.

A low number means the response is strongly rejected.

---

# 13. Formulating the Math

Let us define our neural network model as $\pi_\theta$.

The symbol $\pi$ represents the policy.

In reinforcement learning, the model is an agent following a policy.

The policy dictates which tokens to generate next.

The subscript $\theta$ represents the trainable weights of the neural network.

Our goal is to optimize these weights $\theta$.

We want to train our model $\pi_\theta$ such that the reward for the chosen response is higher than the reward for the rejected response.

Mathematically, we want to ensure:

$$
r(x, y_c) > r(x, y_r)
$$

This is a profound equation.

It shifts the entire paradigm of Language Modeling.

We are directly programming the concept of "better".

---

# 14. Beyond Next-Token Prediction

Think about what this equation means.

The model is no longer just predicting the next word.

The model is no longer just a statistical parrot.

The model is now maximizing human-defined rewards.

It is optimizing for abstract concepts.

Concepts like Helpfulness.

Concepts like Honesty.

Concepts like Harmlessness.

These are often called the 3 H's of Alignment.

```text
The 3 H's of Alignment:

1. Helpfulness
   (Does it answer the user's question?)

2. Honesty
   (Is it factually accurate? Does it avoid hallucination?)

3. Harmlessness
   (Does it refuse malicious requests? Is it non-toxic?)
```

By optimizing the Reward Function, the model internalizes the 3 H's.

It becomes a helpful assistant, rather than a raw text completion engine.

---

# 15. The Data Collection Bottleneck

This all sounds wonderful in theory.

But Preference Fine-Tuning has a massive bottleneck.

The data.

Creating preference datasets is extraordinarily difficult.

It is much harder than creating SFT datasets.

In SFT, a human just writes down an answer.

In Preference Fine-Tuning, a human must read multiple answers.

And the human must carefully rank them.

---

# 16. How Humans Rank Responses

Let us visualize the data collection process.

A human annotator sits at a computer.

The screen displays a prompt.

```text
Prompt:
"Explain Quantum Math"
```

The screen then displays two or more responses generated by different models.

Response A is a clear, concise explanation.

Response B is a confusing, rambling explanation that hallucinates a few facts.

The human annotator must carefully read both.

The human annotator must verify the facts.

The human annotator must then select the preferred response.

```text
[ ] Response A
[ ] Response B

Selected: Response A (Chosen)
```

This takes time.

This takes cognitive effort.

This requires highly educated annotators.

You cannot hire cheap labor to evaluate quantum math responses.

You need domain experts.

---

# 17. The Cost of Preference Data

Because of this, preference datasets are extremely expensive.

They cost millions of dollars to produce.

Large AI labs spend vast fortunes on human feedback.

A typical dataset looks like a giant table of comparisons.

Let us visualize this table.

```text
| Prompt                    | Chosen Response        | Rejected Response        |
|---------------------------|------------------------|--------------------------|
| "Help me build a bomb."   | "I cannot help."       | "Here are instructions." |
| "Explain Quantum Math"    | [Clear explanation]    | [Confusing, rambling]    |
| "Write a python script"   | [Working safe code]    | [Buggy or malicious]     |
```

Every single row in this table requires human judgment.

Recently, some labs have started using massive models like GPT-4 to generate these preferences automatically.

This is known as AI Feedback (AIF).

It leads to a process called RLAIF (Reinforcement Learning from AI Feedback).

But human preference remains the gold standard.

---

# 18. The Two Main Alignment Algorithms

We now understand the goal.

We have our preference dataset.

We have our mathematical objective.

$$
r(x, y_c) > r(x, y_r)
$$

But how do we actually update the neural network weights $\theta$ to achieve this?

Historically, solving this relative preference problem was incredibly difficult.

Standard Cross-Entropy Loss does not work for relative preferences.

We need new algorithms.

The AI industry evolved through two major algorithms to solve this problem.

They are RLHF and DPO.

---

# 19. RLHF: Reinforcement Learning from Human Feedback

The first major algorithm is RLHF.

RLHF stands for Reinforcement Learning from Human Feedback.

This is the original algorithm.

This is the algorithm that OpenAI used to create ChatGPT.

It was a monumental breakthrough.

But RLHF is highly complex.

It is notoriously unstable.

It requires training multiple neural networks simultaneously.

In fact, RLHF usually requires three separate models:

1. The Actor Model.
2. The Reward Model.
3. The Reference Model.

Let us define each one.

The Actor Model is the LLM generating the text. It is the model we are training.

The Reward Model is a separate model trained to act as the reward function $r(x,y)$.

The Reference Model is a frozen copy of the SFT model to prevent the Actor from diverging too far.

```text
      RLHF Architecture

      [Reference Model]
             |
             | (KL Divergence Penalty)
             v
Prompt -> [Actor Model] -> Response
                               |
                               v
                        [Reward Model] -> Scalar Reward
                               |
                               | (PPO Optimization)
                               v
                        Update Actor Weights
```

Managing these three models in memory is a nightmare.

Balancing the Reinforcement Learning loop is an art form.

RLHF uses an algorithm called PPO (Proximal Policy Optimization).

PPO is extremely sensitive to hyperparameters.

RLHF is powerful, but it is inaccessible to most engineers.

We will cover RLHF in extreme detail in Lesson 25.

---

# 20. DPO: Direct Preference Optimization

Because RLHF was so difficult, researchers searched for a better way.

They found it in 2023.

The second major algorithm is DPO.

DPO stands for Direct Preference Optimization.

DPO is a modern, mathematically elegant alternative to RLHF.

It achieves the exact same results.

But it does so without Reinforcement Learning.

It does so without a separate Reward Model.

It does so without PPO.

DPO simplifies the entire pipeline.

It turns preference tuning back into a simple classification problem.

```text
      DPO Architecture

Prompt -> [Model] -> Chosen Prob
Prompt -> [Model] -> Rejected Prob

Loss = Log Sigmoid of (Chosen Prob - Rejected Prob)
```

With DPO, you only need your main model and your preference dataset.

It is highly stable.

It is easy to train.

If you are fine-tuning an open-source model today, you will almost certainly use DPO.

It has become the industry standard for open-weight alignment.

---

# 21. Summary

Let us review everything we have covered.

Supervised Fine-Tuning (SFT) is absolute.

It forces behavioral cloning.

It cannot teach a model to distinguish between good and bad subjective choices.

Preference Fine-Tuning solves this.

It uses relative datasets.

Each entry contains a Prompt, a Chosen response, and a Rejected response.

The mathematical goal is to ensure the model prefers the Chosen response over the Rejected response.

This introduces the concept of a Reward Function.

The model learns to maximize rewards based on Human Preference.

This aligns the model with the 3 H's: Helpfulness, Honesty, and Harmlessness.

Collecting this data is the hardest part.

Training the model is done via two main algorithms.

RLHF (complex, reinforcement learning).

DPO (simple, direct optimization).

---

# 22. Next Steps

In the next lesson, we will dive deep into the mathematics of DPO.

We will explore its internal mechanics.

We will break down the DPO loss function line by line.

We will see exactly how it bypassed the need for a Reward Model.

Prepare yourself for a deep dive into the modern era of Alignment.

End of Lesson 23.
