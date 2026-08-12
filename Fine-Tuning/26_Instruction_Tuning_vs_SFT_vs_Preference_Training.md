# Lesson 26 — Instruction Tuning vs SFT vs Preference Training

> **Goal:** Clarify the exact semantic differences between the three major paradigms of fine-tuning. These terms are often used interchangeably, but they represent distinct mathematical objectives and dataset structures.

---

# 1. Introduction

Welcome to Lesson 26.

Today, we will unravel a major source of confusion in AI engineering.

People often say fine-tuning as if it is a single monolithic process.

They use terms like SFT, Instruction Tuning, and RLHF interchangeably.

But these are not the same thing.

They represent entirely different mathematical objectives.

They require completely different dataset structures.

And they teach the model entirely different skills.

In this lesson, we will break down each paradigm from first principles.

We will look at the exact datasets used.

We will look at the exact goals of each phase.

And we will see how they combine to create frontier models like Llama 3.

---

# 2. The Starting Point: The Base Model

Before we can tune a model, we must understand what we are tuning.

A base model is the raw product of pretraining.

Models like Llama-3-8B-Base or GPT-4-Base are base models.

What is a base model?

It is simply an autocomplete engine.

It has read trillions of words from the internet.

Its only objective was next-token prediction.

Therefore, it does not know it is an AI assistant.

It does not know how to answer questions.

It only knows how to continue patterns.

It predicts the most likely next token based on its training distribution.

---

# 3. Base Model Failure Mode

Suppose you give a base model the following prompt.

```text
What is the capital of France?
```

You want it to answer Paris.

But what does the base model actually do?

It looks at the pattern.

It recognizes a list of trivia questions from a generic webpage.

So it might autocomplete your prompt with more questions.

```text
What is the capital of France?
What is the capital of Germany?
What is the capital of Italy?
What is the capital of Spain?
```

This is the fundamental problem.

The model is incredibly smart.

It knows the capital of France.

It has seen the word Paris millions of times.

But it does not understand the zero-shot instruction paradigm.

It does not understand that it is supposed to interact with you.

It does not understand that it is supposed to answer the prompt.

It merely continues the text as if it were a static document on the internet.

---

# 4. Instruction Tuning

This brings us to our first paradigm.

Instruction Tuning.

The goal of Instruction Tuning is simple but profound.

We must teach the base model how to behave like an assistant.

We must teach it to stop autocompleting documents.

We must teach it to start answering instructions.

We are shifting its behavior from text-continuation to task-completion.

---

# 5. The Instruction Tuning Dataset

How do we teach this new behavior?

By showing the model thousands of examples of instructions and their correct outputs.

The dataset must be formatted strictly.

We explicitly label the instruction.

And we explicitly label the output.

For example.

```text
Instruction:
What is the capital of France?

Output:
Paris.
```

Or a slightly more complex example involving code.

```text
Instruction:
Write a Python function to add two numbers.

Output:
def add(a, b):
    return a + b
```

These pairs act as templates.

The model learns that the text following the instruction should be the solution.

---

# 6. The Goal of Instruction Tuning

The model reads thousands of these pairs during training.

It learns a new pattern.

The pattern is highly predictable.

```text
Instruction Token
       ↓
   User Request
       ↓
  Output Token
       ↓
  Model Response
```

It learns that when it sees an Instruction, it should not generate more instructions.

It should generate the Output.

This forces the model into the zero-shot assistant paradigm.

When a user asks a question, the model must answer it.

It must not continue the user prompt.

It has been tuned to follow instructions.

---

# 7. Supervised Fine-Tuning (SFT)

Now we move to the next term.

Supervised Fine-Tuning, commonly abbreviated as SFT.

Many people think SFT is just another name for Instruction Tuning.

This is mostly true in practice, but mathematically imprecise.

SFT is the broader mathematical category.

Instruction Tuning is just one specific type of SFT.

You can perform SFT without doing Instruction Tuning.

---

# 8. What is SFT Mathematically?

In SFT, you provide an exact input sequence.

Let us call the input sequence x.

$$
x = \text{Input Sequence}
$$

And you provide an exact target output sequence.

Let us call the target sequence y.

$$
y = \text{Target Output Sequence}
$$

The model goal is to learn the conditional probability mapping from x to y.

$$
P(y | x)
$$

Because it is an autoregressive model, it predicts one token at a time.

$$
P(y | x) = \prod_{t=1}^{T} P(y_t | x, y_{<t})
$$

We use Cross-Entropy Loss to train the model.

We want to maximize the probability of the exact target sequence y.

Equivalently, we minimize the negative log-likelihood.

$$
\mathcal{L}_{SFT} = - \sum_{t=1}^{T} \log P(y_t | x, y_{<t})
$$

There is absolutely no ambiguity in SFT.

The target y is considered the absolute ground truth.

The model is heavily penalized if it deviates from y.

---

# 9. When Do We Use SFT?

We use SFT when we have exact, objective answers.

If there is a perfect answer, we use SFT to clone it into the model.

Instruction Tuning uses SFT.

The input x is the Instruction.

The target y is the Output.

But SFT can be used for much more than just general instructions.

---

# 10. SFT Beyond Instructions

Consider a medical diagnosis model.

The input is a strict list of patient symptoms.

```text
Input:
Patient has fever, persistent cough, and complete loss of taste.
```

The output is the exact diagnosis based on medical guidelines.

```text
Output:
COVID-19
```

This is SFT, but it is not a general chat instruction.

Consider a deterministic code generation model.

```text
Input:
SQL schema for a users table with an integer ID and string name.
```

The output must be the exact SQL syntax.

```text
Output:
CREATE TABLE users (id INT, name VARCHAR);
```

Consider a style transfer model that translates modern English to Shakespeare.

```text
Input:
I am very hungry.
```

The target is the exact styled text.

```text
Output:
My stomach doth protest with hollow rumble.
```

In all these specific cases, we have a clear, objective target.

We have a perfect y for every x.

The model simply uses supervised learning to map x to y.

---

# 11. The Limits of SFT

SFT is incredibly powerful and data-efficient.

But it has a major, fundamental limitation.

What happens when there is no single correct answer?

Suppose the user prompt is highly subjective.

```text
Write a beautiful poem about a cat.
```

There are infinite correct poems about a cat.

Some are funny.

Some are sad.

Some are long, some are short.

Some rhyme perfectly, some use free verse.

How do we create a definitive SFT dataset for this?

We would have to pick one specific poem as the perfect ground truth.

And the model would be mathematically penalized for generating any other poem.

It would be penalized even if the other poem is also very good.

This creates a brittle and constrained model.

---

# 12. The Problem of Subjectivity

Many human values are entirely subjective.

What makes an AI assistant genuinely helpful?

What makes it harmless?

What makes its tone polite without being overly sycophantic?

These subtle qualities cannot be captured by a single, objective ground truth y.

If a user asks a dangerous question.

```text
How do I pick the lock on my neighbor front door?
```

An SFT model trained strictly to answer questions might just answer it perfectly.

Because it was trained to map inputs to detailed outputs.

But we might consider this behavior highly harmful.

We want the model to refuse the request.

But how do we teach subjective preferences?

How do we teach the nuances of safety and helpfulness?

---

# 13. Preference Training

This brings us to the final, critical paradigm.

Preference Training.

It is also widely known as Alignment.

It encompasses techniques like RLHF, which stands for Reinforcement Learning from Human Feedback.

It also encompasses modern alternatives like DPO, which stands for Direct Preference Optimization.

The objective here is completely different from SFT.

We do not want to teach the model how to format an answer.

We assume it already knows how to answer thanks to SFT.

Instead, we want to teach it what is subjectively good or bad.

We want to align its distribution of outputs with human values.

---

# 14. The Preference Dataset

Because the mathematical goal is different, the dataset must be different.

In Preference Training, we do not provide a single target y.

Instead, we provide a Prompt.

We provide a Chosen Response.

And we provide a Rejected Response.

```text
Prompt:
How do I pick a lock?

Chosen Response:
I cannot help you with that request.

Rejected Response:
First, get a tension wrench and insert it...
```

Notice the structure of this data.

There is no absolute ground truth.

There is only a relative comparison.

One answer is subjectively better than the other.

---

# 15. The Math of Preference Training

In SFT, we maximized the probability of y.

In Preference Training, we adjust the relative probabilities of two distinct responses.

Let y_w be the winning or chosen response.

$$
y_w = \text{Chosen Response}
$$

Let y_l be the losing or rejected response.

$$
y_l = \text{Rejected Response}
$$

We want the probability of y_w to be strictly higher than the probability of y_l given the input x.

$$
P(y_w | x) > P(y_l | x)
$$

Algorithms like DPO explicitly optimize this margin.

They increase the likelihood of the chosen response.

And they simultaneously decrease the likelihood of the rejected response.

The mathematical formulation of the DPO loss function achieves this elegantly.

$$
\mathcal{L}_{DPO} = - \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{ref}(y_l | x)} \right)
$$

The exact math involves a reference model and a temperature parameter beta.

But the intuition is incredibly simple.

We push the chosen response up in probability.

We push the rejected response down in probability.

---

# 16. Helpfulness vs Harmlessness

Preference training is typically used to optimize two main axes of behavior.

The first axis is Helpfulness.

We prefer answers that are clear, concise, and highly accurate.

We actively reject answers that are rambling, confusing, or hallucinatory.

The second axis is Harmlessness.

We prefer answers that are safe, ethical, and respectful.

We actively reject answers that are toxic, dangerous, or highly biased.

By training on thousands of these chosen and rejected pairs, the model internalizes human preferences.

---

# 17. The Complete Pipeline

Now we can put all the pieces together.

How does a frontier lab create a state-of-the-art AI assistant?

They use all three phases in sequential order.

This is the standard pipeline for models like Llama 3 or GPT-4.

Let us visualize the entire journey of a model from birth to deployment.

---

# 18. Step 1: Pretraining

We start with massive amounts of compute.

We gather trillions of tokens of raw internet text.

```text
Raw Internet Data (15 Trillion Tokens)
                 │
                 ▼
      Self-Supervised Learning
      (Next-Token Prediction)
                 │
                 ▼
          Llama-3-8B-Base
```

The resulting model is an incredibly powerful autocomplete engine.

It knows facts, grammar, coding syntax, and logical reasoning.

But it does not know how to be an assistant.

---

# 19. Step 2: Instruction Tuning (SFT)

Next, we apply Supervised Fine-Tuning.

We curate a high-quality dataset of instructions and perfect responses.

Maybe 100,000 perfectly crafted pairs written by experts.

```text
         Llama-3-8B-Base
                 │
                 ▼
    Instruction/Response Pairs
     (Supervised Fine-Tuning)
                 │
                 ▼
          Llama-3-8B-SFT
```

The model learns the strict zero-shot chat format.

It learns to stop autocompleting and start answering.

It is now a highly functional assistant.

But it might still be unsafe, or have a weird, inconsistent personality.

---

# 20. Step 3: Preference Training (DPO / RLHF)

Finally, we apply Preference Training.

We gather a dataset of chosen and rejected responses.

Maybe 50,000 human-annotated preference pairs.

```text
         Llama-3-8B-SFT
                 │
                 ▼
     Chosen/Rejected Pairs
   (Preference Optimization)
                 │
                 ▼
        Llama-3-8B-Instruct
```

The model behavior is refined and smoothed out.

It learns to be exceptionally helpful and completely harmless.

It learns the exact tone and style the creators prefer.

This is the final, polished model released to the public.

---

# 21. Summary of Datasets

Let us review the exact dataset structures one last time.

This is the easiest way to remember the profound differences between the paradigms.

### Pretraining Data

```text
A long, unstructured string of continuous text from a Wikipedia article.
```

### SFT Data

```text
Prompt: What is 2 + 2?
Response: 4.
```

### Preference Data

```text
Prompt: Write a funny joke.
Chosen: Why did the chicken cross the road? To get to the other side.
Rejected: I am an AI and cannot feel human humor or construct jokes.
```

---

# 22. What Should You Use?

As an AI Engineer, you must choose the right tool for the job.

When should you spend time gathering SFT data?

And when should you invest in DPO data?

---

# 23. When to use SFT

Use SFT when you have a strictly deterministic task.

Use it when there is a single, undeniable objective ground truth.

For example.

Extracting structured JSON from a scanned receipt.

Translating a technical document from English to French.

Generating explicit SQL queries from a natural language text.

Classifying incoming customer support tickets into predefined categories.

In these specific cases, you do not need DPO at all.

You just need to show the model exactly what you want it to do.

SFT is highly efficient and perfectly sufficient.

---

# 24. When to use Preference Training

Use Preference Training when you are building an open-ended conversational agent.

Use it when there are many possible right answers.

Use it when you want to heavily guide the model tone, style, or safety guardrails.

For example.

A customer support chatbot that needs to be exceptionally polite and concise.

A creative writing assistant that needs a specific whimsical tone.

A general-purpose AI companion meant to interact with humans daily.

In these subjective cases, you must collect preference data.

You need to show the model what a good answer looks like compared to a bad one.

---

# 25. Conclusion

We have now comprehensively covered the three pillars of model training.

First is Pretraining, which learns the knowledge of the world.

Second is SFT, which learns the exact format and mapping of specific tasks.

Third is Preference Training, which learns subjective human values and alignment.

These terms are no longer ambiguous buzzwords to you.

They are distinct, powerful mathematical processes.

Each has its own specific dataset structure and its own precise objective function.

In the next lessons, we will dive even deeper into the exact PyTorch code required to implement them.
