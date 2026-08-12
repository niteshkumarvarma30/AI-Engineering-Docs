# Fine-Tuning — Lesson 2: Pretraining vs Fine-Tuning

> **Goal:** Understand what a model learns during pretraining, how next-token prediction works, what changes during fine-tuning, and why a pretrained model can be adapted to a specialized task.

---

# 1. The Big Picture

An LLM's development can be viewed as two major stages:

```text
                 PRETRAINING
                     ↓
             General-purpose
                 knowledge
                     ↓
              PRETRAINED LLM
                     ↓
                FINE-TUNING
                     ↓
             Task / Domain /
             Behavior Adaptation
                     ↓
             FINE-TUNED LLM
```

The two stages have different purposes.

### Pretraining

> Teach the model general patterns from a very large corpus.

### Fine-tuning

> Adapt the pretrained model to a particular task, domain, or behavior.

---

# 2. What Happens During Pretraining?

Before training, a model has parameters that can be represented as:

$$
\theta_0
$$

These parameters are initially random or otherwise untrained.

Suppose we give a language model:

```text
"The cat is sitting on the ___"
```

The model has to predict the next token.

Initially, its prediction may be poor:

```text
cat       → 0.02
house     → 0.01
mat       → 0.01
banana    → 0.03
...
```

The model makes a prediction.

We compare that prediction with the actual next token.

This produces a **loss**.

The process is:

```text
Text
 ↓
Model
 ↓
Prediction
 ↓
Compare with actual token
 ↓
Loss
 ↓
Backpropagation
 ↓
Update parameters
```

This happens over enormous amounts of text.

Eventually, the parameters become:

$$
\boxed{\theta_{\text{pretrained}}}
$$

---

# 3. Next-Token Prediction

For autoregressive language models, one fundamental pretraining objective is:

$$
\boxed{\text{Predict the next token}}
$$

Suppose the input is:

```text
The dog is
```

The model tries to predict a likely next token, such as:

```text
running
```

Conceptually:

$$
P(\text{running}\mid\text{The dog is})
$$

If the sequence becomes:

```text
The dog is running
```

the model can learn to predict another following token.

More generally:

$$
\boxed{
P(x_t\mid x_1,x_2,\ldots,x_{t-1})
}
$$

where:

- $x_1,\ldots,x_{t-1}$ = previous tokens
- $x_t$ = next token

---

# 4. Example of Next-Token Prediction

Suppose the training sentence is:

```text
The cat sat on the mat.
```

After tokenization, imagine:

```text
[The, cat, sat, on, the, mat]
```

Conceptually, the training relationships can be represented as:

| Input Context | Target |
|---|---|
| The | cat |
| The cat | sat |
| The cat sat | on |
| The cat sat on | the |
| The cat sat on the | mat |

The model learns to estimate:

$$
P(x_t|x_{<t})
$$

where $x_{<t}$ means the tokens before position $t$.

---

# 5. What Does the Model Learn During Pretraining?

The model is not given a simple database such as:

```text
cat = animal
Paris = capital of France
```

Instead, its parameters are optimized through enormous numbers of prediction tasks.

During this process, the model can learn useful statistical and representational patterns such as:

- Word relationships
- Sentence structure
- Grammar
- Syntax
- Semantic relationships
- Long-range dependencies
- Patterns in text
- Code patterns
- Factual associations

The exact capabilities depend on the model architecture, training data, objective, and training procedure.

The important idea is:

> **Pretraining produces parameters that encode useful representations and statistical patterns learned from the training corpus.**

---

# 6. Pretraining Loss

Suppose the model predicts a probability distribution over the vocabulary:

```text
Token        Probability

cat             0.05
dog             0.10
mat             0.70
car             0.02
...
```

If the correct next token is:

```text
mat
```

the model should assign a high probability to `mat`.

A common objective is **cross-entropy loss**.

For one target token:

$$
\boxed{
L=-\log P(x_t|x_{<t})
}
$$

If the model gives the correct token a high probability:

$$
P(x_t|x_{<t})\approx1
$$

then:

$$
-\log P(x_t|x_{<t})\approx0
$$

So the loss is small.

If the model assigns a very small probability to the correct token:

$$
P(x_t|x_{<t})\approx0
$$

the loss becomes large.

---

# 7. From One Token to an Entire Sequence

For a sequence:

$$
x_1,x_2,\ldots,x_T
$$

the model predicts each token based on previous tokens.

The total negative log-likelihood can be written as:

$$
\boxed{
L
=
-\sum_{t=1}^{T}
\log P(x_t|x_{<t})
}
$$

Often we use the average across tokens:

$$
\boxed{
L
=
-\frac{1}{T}
\sum_{t=1}^{T}
\log P(x_t|x_{<t})
}
$$

The model's parameters are optimized to reduce this loss.

---

# 8. The Training Loop During Pretraining

Conceptually:

```text
              Training Text
                    ↓
                Tokenizer
                    ↓
                Token IDs
                    ↓
              Transformer
                    ↓
          Probability Distribution
                    ↓
              Cross-Entropy
                    ↓
                  Loss
                    ↓
             Backpropagation
                    ↓
                Gradients
                    ↓
                Optimizer
                    ↓
           Updated Parameters
```

This process is repeated over very large training corpora.

---

# 9. What Comes Out of Pretraining?

After pretraining, we have:

$$
\boxed{\theta_{\text{pretrained}}}
$$

These parameters represent what the model learned during its training.

So:

```text
Massive Corpus
      ↓
Next-token prediction
      ↓
Loss
      ↓
Backpropagation
      ↓
Weight updates
      ↓
PRETRAINED PARAMETERS
```

These pretrained parameters become the starting point for fine-tuning.

---

# 10. What Happens During Fine-Tuning?

Suppose we take:

$$
\theta_{\text{pretrained}}
$$

and provide a new dataset.

For example:

```text
Question:
What is gradient descent?

Answer:
Gradient descent is an optimization algorithm...
```

Now the goal is no longer simply to learn general patterns from a massive corpus.

Instead, we want the model to perform a specific task or exhibit a desired behavior.

The process becomes:

```text
Pretrained Model
      ↓
Specialized Dataset
      ↓
Task-specific Loss
      ↓
Backpropagation
      ↓
Parameter Updates
      ↓
Adapted Model
```

---

# 11. The Critical Difference

The starting point is different.

## Pretraining

$$
\boxed{
\theta_{\text{random}}
\rightarrow
\theta_{\text{pretrained}}
}
$$

## Fine-Tuning

$$
\boxed{
\theta_{\text{pretrained}}
\rightarrow
\theta_{\text{fine-tuned}}
}
$$

The data and objective are also different.

### Pretraining

Usually involves:

```text
Huge, diverse corpus
        +
General learning objective
```

### Fine-Tuning

Usually involves:

```text
Smaller, task/domain-specific dataset
        +
Specific objective or desired behavior
```

---

# 12. Example: Sentiment Classification

Suppose we have a pretrained Transformer.

It already has useful language representations.

Now we have:

```text
"I loved the movie."      → Positive
"I hated the movie."      → Negative
"This was fantastic."     → Positive
"This was terrible."      → Negative
```

We fine-tune the model using this dataset.

Conceptually, the parameters move from:

$$
\theta_{\text{pretrained}}
$$

toward:

$$
\theta_{\text{sentiment}}
$$

The model becomes better suited to sentiment classification.

---

# 13. Fine-Tuning Does Not Mean "Teach Everything Again"

This is one of the most important ideas.

Suppose the pretrained model has already learned useful language representations.

You don't need to teach it basic language patterns again.

Instead, fine-tuning makes adjustments toward the desired objective.

```text
General Language Ability
          +
Task-Specific Training
          ↓
Task-Adapted Model
```

This is why starting from a pretrained model is powerful.

---

# 14. Why Can a Small Dataset Be Useful?

Imagine a pretrained model has already learned general language patterns from a huge corpus.

Now you have:

```text
10,000 high-quality examples
```

for a specialized task.

You can potentially use those examples to adapt the model because you are not asking it to learn language from zero.

You are asking it to adjust an already capable model toward a particular behavior.

This is one of the fundamental ideas behind **transfer learning**.

---

# 15. Transfer Learning

Fine-tuning is closely related to **transfer learning**.

The basic idea is:

```text
Knowledge learned on one problem
             ↓
       Transfer to
       another problem
```

For example:

```text
General language training
          ↓
      Pretrained
       language model
          ↓
     Transfer knowledge
          ↓
  Sentiment classification
```

So:

$$
\boxed{
\text{Pretraining}
\rightarrow
\text{Transfer Learning}
\rightarrow
\text{Fine-Tuning}
}
$$

More precisely, **fine-tuning is one important way of performing transfer learning**.

---

# 16. Pretraining vs Fine-Tuning — Comparison

| Property | Pretraining | Fine-Tuning |
|---|---|---|
| Starting point | Initial/random parameters | Pretrained parameters |
| Dataset | Usually massive | Usually smaller/task-specific |
| Objective | General learning objective | Task/domain/behavior objective |
| Goal | Build general capability | Adapt capability |
| Compute | Extremely large for modern LLMs | Usually much smaller |
| Output | Pretrained model | Specialized/adapted model |

---

# 17. Pretraining vs SFT

Later, you will hear the term:

> **SFT — Supervised Fine-Tuning**

Do not confuse it with pretraining.

## Pretraining

A model can learn from raw text:

```text
The cat is sitting on the mat.
```

The training signal can come from the text itself through next-token prediction.

## SFT

You explicitly provide desired examples:

```text
Instruction:
Explain gradient descent.

Response:
Gradient descent is...
```

The model learns to produce the desired response.

Conceptually:

```text
Pretraining
    ↓
General language model

SFT
    ↓
Instruction-following model
```

We will study SFT in detail later.

---

# 18. Knowledge vs Behavior

A useful mental model is:

### Pretraining

Primarily builds broad representations and capabilities from large-scale data.

### Fine-Tuning

Can strongly influence how the model behaves on particular:

- Tasks
- Formats
- Domains
- Instructions
- Behaviors

For example:

```text
Input:
Explain this concept.

Desired format:
1. Definition
2. Formula
3. Example
```

The goal is not necessarily to teach English again.

The goal is to adapt the model's behavior.

---

# 19. Why Fine-Tuning Can Cause Problems

Fine-tuning is not automatically beneficial.

If you use poor or very narrow data, the model can become worse at other capabilities.

Potential problems include:

- Overfitting
- Catastrophic forgetting
- Poor dataset quality
- Excessive training
- Distribution mismatch

For example:

```text
Pretrained Model
       ↓
Very narrow dataset
       ↓
Too much fine-tuning
       ↓
Highly specialized model
       ↓
May lose some general behavior
```

These problems become important during practical fine-tuning.

---

# 20. Complete Picture

Combining Lesson 1 and Lesson 2:

```text
                  PRETRAINING
                       │
                       ▼
              Massive Training Data
                       │
                       ▼
              General Objective
                       │
                       ▼
              Next-Token Prediction
                       │
                       ▼
                     Loss
                       │
                       ▼
                Backpropagation
                       │
                       ▼
               Updated Parameters
                       │
                       ▼
              PRETRAINED MODEL
          θ_pretrained
                       │
                       ▼
                  FINE-TUNING
                       │
                       ▼
            Task-Specific Dataset
                       │
                       ▼
              Task-Specific Objective
                       │
                       ▼
                     Loss
                       │
                       ▼
                Backpropagation
                       │
                       ▼
               Parameter Updates
                       │
                       ▼
             FINE-TUNED MODEL
          θ_fine-tuned
```

---

# 21. Important Equations

## Pretraining

For autoregressive language modeling:

$$
\boxed{
P(x_t|x_1,\ldots,x_{t-1})
}
$$

Loss:

$$
\boxed{
L
=
-\sum_t \log P(x_t|x_{<t})
}
$$

or the average token loss:

$$
\boxed{
L
=
-\frac{1}{T}
\sum_{t=1}^{T}
\log P(x_t|x_{<t})
}
$$

## Fine-Tuning

Starting point:

$$
\boxed{
\theta_{\text{pretrained}}
}
$$

Then:

$$
\boxed{
\theta_{\text{pretrained}}
\rightarrow
\theta_{\text{fine-tuned}}
}
$$

through optimization:

$$
\boxed{
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta\nabla_\theta L
}
$$

---

# 22. The Most Important Mental Model

Remember:

```text
PRETRAINING

"Learn general patterns"
        ↓
Huge dataset
        ↓
Pretrained model


FINE-TUNING

"Adapt those patterns"
        ↓
Task-specific dataset
        ↓
Fine-tuned model
```

Or:

$$
\boxed{
\text{General Learning}
\rightarrow
\text{Pretrained Model}
\rightarrow
\text{Specialized Learning}
\rightarrow
\text{Fine-Tuned Model}
}
$$

---

# 23. Lesson 2 Summary

The key points are:

1. **Pretraining** creates a general-purpose model through large-scale training.
2. Autoregressive LLMs can use **next-token prediction** as a fundamental pretraining objective.
3. The model learns by predicting tokens, calculating loss, and updating parameters through backpropagation.
4. A pretrained model contains learned parameters:
   $$\theta_{\text{pretrained}}$$
5. Fine-tuning starts from those pretrained parameters.
6. Fine-tuning usually uses a smaller, more specialized dataset.
7. Fine-tuning changes the model toward a specific task, domain, objective, or behavior.
8. Fine-tuning does not require relearning language from zero.
9. Fine-tuning is closely related to **transfer learning**.
10. **SFT (Supervised Fine-Tuning)** is a specific form of fine-tuning that we will study later.
11. Fine-tuning can cause problems such as overfitting and catastrophic forgetting.
12. Pretraining and fine-tuning use the same fundamental optimization machinery:
    $$\text{Forward} \rightarrow \text{Loss} \rightarrow \text{Backward} \rightarrow \text{Update}$$

---

# 24. Questions & Answers

## Q1. During autoregressive pretraining, what is the model fundamentally trying to predict?

### Answer

It is fundamentally trying to predict the **next token** given the previous tokens.

$$
\boxed{
P(x_t|x_1,x_2,\ldots,x_{t-1})
}
$$

For example:

```text
Input:
The cat is

Target:
sleeping
```

The model learns to assign a high probability to the correct next token.

---

## Q2. If the sequence is "The cat is sleeping", what does next-token prediction mean?

### Answer

Conceptually, the model learns relationships such as:

```text
The
 ↓
cat

The cat
 ↓
is

The cat is
 ↓
sleeping
```

So it learns to estimate:

$$
P(\text{cat}|\text{The})
$$

$$
P(\text{is}|\text{The cat})
$$

$$
P(\text{sleeping}|\text{The cat is})
$$

During actual autoregressive generation, the model predicts one token at a time and uses previous tokens as context.

---

## Q3. What is the difference between θ_random → θ_pretrained and θ_pretrained → θ_fine-tuned?

### Answer

The first represents **pretraining**:

$$
\boxed{
\theta_{\text{random}}
\rightarrow
\theta_{\text{pretrained}}
}
$$

The model starts with untrained/initial parameters and learns general patterns from a large corpus.

The second represents **fine-tuning**:

$$
\boxed{
\theta_{\text{pretrained}}
\rightarrow
\theta_{\text{fine-tuned}}
}
$$

The model starts with already learned parameters and adapts them to a new task, domain, objective, or behavior.

---

## Q4. Why can a relatively small task-specific dataset sometimes be enough to fine-tune a pretrained model?

### Answer

Because the pretrained model has already learned broad language representations and patterns.

The task-specific dataset does not necessarily need to teach the model language from zero.

Instead, it provides examples that guide the model toward the desired task or behavior.

Conceptually:

```text
Large-scale pretraining
        ↓
General capabilities
        +
Smaller specialized dataset
        ↓
Task adaptation
```

The quality, size, diversity, and suitability of the dataset still matter.

---

## Q5. What is the difference between the general objective of pretraining and the task-specific objective of fine-tuning?

### Answer

### Pretraining

The objective is generally to learn broad patterns from a large corpus.

For an autoregressive language model:

$$
\boxed{
\text{Predict the next token}
}
$$

### Fine-Tuning

The objective is adapted toward a particular task, domain, or behavior.

Examples:

```text
Sentiment classification
Translation
Summarization
Question answering
Instruction following
Specific response format
```

Therefore:

```text
Pretraining
→ General capability

Fine-tuning
→ Specialized capability / behavior
```

---

## Q6. What is transfer learning, and how is fine-tuning related to it?

### Answer

**Transfer learning** means transferring knowledge or learned representations from one learning setting to another.

For example:

```text
General language training
        ↓
Pretrained model
        ↓
Transfer learned representations
        ↓
New task
```

Fine-tuning is one important way to perform transfer learning because it starts with a pretrained model and adapts it to a new task or objective.

---

## Q7. Does the model learn facts by being explicitly given a database during pretraining?

### Answer

Not in the simple database-like sense.

The model is trained through prediction objectives over its training corpus.

Through this optimization, its parameters can encode statistical patterns, representations, and associations present in the training data.

So we should think of learned information as being represented in the model's parameters rather than as a simple database lookup table.

---

## Q8. Why does the correct token probability affect the loss?

### Answer

A common token-level loss is:

$$
L=-\log P(x_t|x_{<t})
$$

If the model gives the correct token a high probability:

$$
P(x_t|x_{<t})\rightarrow1
$$

then:

$$
L\rightarrow0
$$

If the model gives the correct token a very low probability, the loss becomes large.

Therefore, training encourages the model to increase the probability of correct target tokens.

---

## Q9. What happens after the model calculates the loss during pretraining or fine-tuning?

### Answer

The basic process is:

```text
Prediction
   ↓
Loss
   ↓
Backpropagation
   ↓
Gradients
   ↓
Optimizer
   ↓
Parameter update
```

Mathematically:

$$
\nabla_\theta L
$$

is calculated, and the optimizer can update the parameters using an update rule such as:

$$
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta\nabla_\theta L
$$

---

## Q10. Can fine-tuning make a model worse?

### Answer

Yes.

Poor or excessive fine-tuning can lead to problems such as:

- Overfitting
- Catastrophic forgetting
- Distribution mismatch
- Poor generalization
- Degradation of some previous capabilities

Therefore, fine-tuning requires appropriate data, training configuration, evaluation, and regularization/early stopping where appropriate.

---

# 25. Quick Self-Test

Before moving to Lesson 3, try answering these without looking at the notes:

1. What is pretraining?
2. What is next-token prediction?
3. What does:
   $$P(x_t|x_{<t})$$
   mean?
4. What is cross-entropy loss doing in language-model training?
5. What is the difference between:
   $$\theta_{\text{random}}\rightarrow\theta_{\text{pretrained}}$$
   and
   $$\theta_{\text{pretrained}}\rightarrow\theta_{\text{fine-tuned}}$$
6. Why can a smaller task-specific dataset be useful after pretraining?
7. What is transfer learning?
8. What is SFT?
9. Can fine-tuning cause catastrophic forgetting?
10. Explain pretraining and fine-tuning in your own words.

If you can explain these concepts without memorizing the exact wording, you are ready for **Lesson 3**.

---

# 26. Next Lesson

## Lesson 3 — From Dataset to Transformer

We will make the process practical:

```text
Raw Dataset
    ↓
Tokenizer
    ↓
Input IDs
    ↓
Attention Mask
    ↓
Pretrained Transformer
    ↓
Predictions / Logits
    ↓
Loss
    ↓
Backpropagation
    ↓
Parameter Update
```

We will learn what each object actually looks like in code:

```python
input_ids
attention_mask
labels
logits
loss
```

Then we will use a small pretrained Transformer and perform our **first real fine-tuning workflow**.
