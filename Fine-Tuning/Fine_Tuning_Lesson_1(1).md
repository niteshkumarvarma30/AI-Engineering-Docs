# Fine-Tuning — Lesson 1: What Is Fine-Tuning?

> **Goal:** Understand the fundamental idea of fine-tuning before learning Hugging Face, LoRA, QLoRA, SFT, or DPO.

---

## 1. What Is Training From Scratch?

Suppose we create a neural network with randomly initialized weights:

```text
Random weights
      ↓
Training data
      ↓
Training
      ↓
Learned weights
      ↓
Trained model
```

Initially, the model has no useful learned parameters.

For a large language model, training from scratch can require enormous amounts of data, GPU compute, training time, memory, and storage.

We can represent the model parameters as:

$$
\theta_{\text{random}}
$$

Training gradually changes them:

$$
\theta_0 \rightarrow \theta_1 \rightarrow \theta_2 \rightarrow \cdots \rightarrow \theta_{\text{trained}}
$$

---

## 2. What Is a Pretrained Model?

Instead of training a model from zero, someone may already have trained a large model on a massive dataset.

```text
Large Dataset
      ↓
Transformer / Neural Network
      ↓
Large-scale Training
      ↓
Pretrained Model
```

The resulting model contains learned parameters:

$$
\boxed{\theta_{\text{pretrained}}}
$$

A pretrained language model may already have learned useful patterns related to:

- Grammar
- Sentence structure
- Word relationships
- Semantic patterns
- Contextual representations
- General language patterns

Therefore, we don't need to start from random weights when adapting the model to a new task.

---

## 3. What Is Fine-Tuning?

Suppose we already have:

```text
Pretrained Model
```

and we have a new task-specific dataset.

For example:

| Text | Label |
|---|---|
| "I loved this movie!" | Positive |
| "This movie was amazing." | Positive |
| "I hated this movie." | Negative |
| "This was terrible." | Negative |

Instead of training a new model from scratch, we continue training the pretrained model on this new dataset.

```text
Pretrained Model
      +
Task-Specific Dataset
      ↓
Further Training
      ↓
Fine-Tuned Model
```

This process is called:

$$
\boxed{\text{Fine-Tuning}}
$$

### Definition

> **Fine-tuning is continuing the training of a pretrained model on a new dataset, domain, or objective so that the model adapts to the desired task or behavior.**

The word **"continuing"** is important.

We start with:

$$
\theta_{\text{pretrained}}
$$

rather than:

$$
\theta_{\text{random}}
$$

---

## 4. Why Don't We Randomly Initialize the Model Again?

There are two major reasons.

### Reason 1 — Preserve Learned Knowledge

The pretrained model has already learned useful representations and patterns.

If we randomly initialize everything again:

```text
Random weights
      ↓
Train from zero
      ↓
Learn general patterns
      ↓
Learn specific task
```

we throw away the knowledge contained in the pretrained model.

Instead:

```text
Pretrained weights
      ↓
Already contains useful knowledge
      ↓
Train on the new task
      ↓
Adapt existing knowledge
```

So fine-tuning allows us to **build on previously learned representations**.

### Reason 2 — Reduce Training Cost

Training a large model from scratch can require enormous:

- Computational resources
- GPU hours
- Training time
- Datasets

Fine-tuning starts from an already trained model, so the model does not have to relearn everything from zero.

Therefore:

$$
\boxed{\text{Fine-tuning preserves pretrained knowledge and avoids training from scratch}}
$$

---

## 5. Do Model Parameters Change During Fine-Tuning?

Yes.

During ordinary **full fine-tuning**, the model parameters are updated.

Suppose the pretrained parameters are:

$$
\theta_{\text{pretrained}}
$$

During fine-tuning:

$$
\theta_{\text{pretrained}}
\rightarrow
\theta_1
\rightarrow
\theta_2
\rightarrow
\theta_3
\rightarrow
\theta_{\text{fine-tuned}}
$$

The underlying mechanism is the same basic mechanism used when training neural networks.

```text
Input
  ↓
Forward Pass
  ↓
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
Updated Parameters
```

---

## 6. Mathematical View of Fine-Tuning

Let:

$$
\theta
$$

represent the model parameters.

During fine-tuning, we calculate a loss:

$$
L
$$

Then calculate the gradient:

$$
\nabla_\theta L
$$

The optimizer updates the parameters:

$$
\boxed{
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta\nabla_\theta L
}
$$

where:

- $\theta$ = model parameters
- $L$ = training loss
- $\nabla_\theta L$ = gradient of the loss with respect to the parameters
- $\eta$ = learning rate

Therefore, fine-tuning is not a completely different training mechanism.

It uses the same fundamental deep-learning process:

```text
Forward Pass
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

---

## 7. Connection to PyTorch

This connects directly to the PyTorch concepts we are learning.

During training:

```python
loss.backward()
```

calculates gradients.

Then:

```python
optimizer.step()
```

updates the trainable parameters.

Conceptually:

$$
\boxed{\texttt{loss.backward()} \rightarrow \text{calculate gradients}}
$$

$$
\boxed{\texttt{optimizer.step()} \rightarrow \text{update parameters}}
$$

Fine-tuning uses this same mechanism.

The major difference is that the model starts from:

$$
\theta_{\text{pretrained}}
$$

instead of random parameters.

---

## 8. Training From Scratch vs Fine-Tuning

### Training From Scratch

```text
Random Initialization
        ↓
Training Dataset
        ↓
Training
        ↓
Learn Parameters
        ↓
Final Model
```

Mathematically:

$$
\theta_{\text{random}}
\rightarrow
\theta_{\text{trained}}
$$

### Fine-Tuning

```text
Pretrained Model
        ↓
Task-Specific Dataset
        ↓
Further Training
        ↓
Adapted Model
```

Mathematically:

$$
\boxed{
\theta_{\text{pretrained}}
\rightarrow
\theta_{\text{fine-tuned}}
}
$$

### Core Difference

| Training from scratch | Fine-tuning |
|---|---|
| Starts from random/initial parameters | Starts from pretrained parameters |
| Learns general patterns and task patterns | Adapts existing knowledge |
| Usually requires much more computation | Usually much less than pretraining |
| Requires large-scale training for large models | Can work with much smaller task-specific datasets |

---

## 9. Example: Sentiment Classification

Suppose we have a pretrained Transformer.

It has already learned general language patterns.

Now we want sentiment classification.

```text
Pretrained Transformer
          ↓
Movie Review Dataset
          ↓
Fine-Tuning
          ↓
Sentiment Model
```

Example:

```text
"I absolutely loved this movie."
             ↓
        Fine-tuned model
             ↓
          Positive
```

Another example:

```text
"The movie was boring and terrible."
             ↓
        Fine-tuned model
             ↓
          Negative
```

The model's existing language representations provide the starting point, while fine-tuning adapts the model to the sentiment task.

---

## 10. Fine-Tuning vs RAG

This distinction is especially important when building LLM applications.

### RAG

Retrieval-Augmented Generation generally follows:

```text
User Question
      ↓
Retriever
      ↓
Relevant Documents
      ↓
LLM
      ↓
Answer
```

The retrieved information is supplied as context to the model.

The model's parameters generally remain unchanged during normal RAG inference.

Therefore:

$$
\boxed{\text{RAG provides external context to the model}}
$$

### Fine-Tuning

Fine-tuning follows:

```text
Training Dataset
      ↓
Pretrained Model
      ↓
Forward Pass
      ↓
Loss
      ↓
Backpropagation
      ↓
Parameter Updates
      ↓
Fine-Tuned Model
```

The model is trained further so that its trainable parameters adapt.

Therefore:

$$
\boxed{\text{Fine-tuning adapts the model parameters}}
$$

---

## 11. RAG vs Fine-Tuning — Core Difference

The simplest way to remember the distinction is:

$$
\boxed{\text{RAG} = \text{provide relevant external context}}
$$

$$
\boxed{\text{Fine-tuning} = \text{adapt trainable model parameters}}
$$

They can also be combined:

```text
                 User Question
                       ↓
                  RAG Retrieval
                       ↓
              Relevant Documents
                       ↓
                Fine-Tuned LLM
                       ↓
                     Answer
```

---

## 12. Is Fine-Tuning Only for One Task?

No.

Fine-tuning can adapt a pretrained model to many different objectives, domains, or behaviors.

Examples include:

- Sentiment classification
- Text classification
- Translation
- Summarization
- Question answering
- Coding
- Instruction following
- Domain-specific behavior
- Specific response formats
- Conversational behavior

So a better definition is:

$$
\boxed{
\text{Fine-tuning = adapting a pretrained model to a new objective, domain, task, or behavior}
}
$$

---

## 13. Important Exception: LoRA and QLoRA

At this stage, remember only that ordinary full fine-tuning and parameter-efficient fine-tuning are not exactly the same.

In **full fine-tuning**, many or all model parameters that are designated trainable can be updated.

Later we will learn **LoRA** and **QLoRA**, where the original model parameters can be frozen and additional trainable parameters/adapters are used.

Conceptually:

```text
Full Fine-Tuning

Base Model
    ↓
Update model parameters
    ↓
Fine-Tuned Model
```

versus:

```text
LoRA

Base Model
    ↓
Freeze base parameters
    +
Train small adapter parameters
    ↓
Fine-Tuned Adapter
```

We will study this in detail later.

---

## 14. Simple Analogy

Think of a pretrained model as a student who has already completed general education.

```text
Pretraining
     ↓
General education
```

Now the student receives specialized training:

```text
General knowledge
      +
Specialized training
      ↓
Specialized ability
```

Similarly:

```text
Pretrained Model
      +
Task / Domain Dataset
      ↓
Fine-Tuning
      ↓
Adapted Model
```

This analogy is useful for intuition, but remember that the actual mechanism is mathematical parameter optimization.

---

## 15. The Core Mental Model

Keep this diagram in mind:

```text
                 PRETRAINING
                     │
                     ▼
              Random / Initial
                 Parameters
                     │
                     ▼
              Massive Dataset
                     │
                     ▼
                  Training
                     │
                     ▼
            PRETRAINED MODEL
          θ_pretrained
                     │
                     ▼
              FINE-TUNING
                     │
             Task-specific
                  Dataset
                     │
                     ▼
                Forward Pass
                     │
                     ▼
                    Loss
                     │
                     ▼
              Backpropagation
                     │
                     ▼
                 Gradients
                     │
                     ▼
                 Optimizer
                     │
                     ▼
          UPDATED PARAMETERS
                     │
                     ▼
            FINE-TUNED MODEL
          θ_fine-tuned
```

---

## 16. Key Formulas

### Pretrained parameters

$$
\boxed{\theta_{\text{pretrained}}}
$$

### Fine-tuning loss

$$
\boxed{L(\theta)}
$$

### Gradient

$$
\boxed{\nabla_\theta L}
$$

### Parameter update

$$
\boxed{
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta\nabla_\theta L
}
$$

### Overall idea

$$
\boxed{
\text{Pretrained Model}
+
\text{Task-Specific Data}
+
\text{Further Training}
=
\text{Fine-Tuned Model}
}
$$

---

## 17. Lesson 1 Summary

Remember these points:

1. **Pretraining** creates a general-purpose model from large-scale training.
2. A **pretrained model already contains learned parameters**.
3. **Fine-tuning continues training from those pretrained parameters**.
4. In ordinary full fine-tuning, **trainable model parameters are updated**.
5. Fine-tuning uses the same fundamental mechanism as neural-network training:
   $$\text{Forward} \rightarrow \text{Loss} \rightarrow \text{Backward} \rightarrow \text{Update}$$
6. Starting from pretrained weights preserves useful learned knowledge.
7. Starting from pretrained weights also avoids the cost of training the whole model from scratch.
8. **RAG provides external context; fine-tuning adapts trainable parameters.**
9. Fine-tuning can be used for many tasks and behaviors, not only one task.
10. **LoRA/QLoRA** are parameter-efficient approaches that we will learn later.

---

## 18. What Comes Next?

### Lesson 1 — Completed

```text
[✓] What is training from scratch?
[✓] What is a pretrained model?
[✓] What is fine-tuning?
[✓] Why use pretrained weights?
[✓] Do parameters change?
[✓] Mathematical view
[✓] Connection to PyTorch
[✓] Fine-tuning vs RAG
[✓] Full fine-tuning vs adapters — basic idea
```

### Lesson 2 — Next

```text
Pretraining
     ↓
What does an LLM actually learn?
     ↓
Pretraining objectives
     ↓
Next-token prediction
     ↓
Language-model loss
     ↓
Why fine-tuning works
     ↓
Task-specific objective
```

This will give us the foundation needed before writing our first fine-tuning code.


---

# 19. Lesson 1 — Questions & Answers

These questions are based on the concepts covered in this lesson and are useful for checking your understanding before moving to Lesson 2.

## Q1. If a model already has pretrained weights, why don't we initialize the weights randomly again when fine-tuning?

### Answer

We do not randomly initialize the model again because that would throw away the useful knowledge learned during pretraining.

A pretrained model already contains learned representations and patterns:

```text
Pretrained weights
      ↓
Already learned useful patterns
      ↓
Fine-tune on new task
      ↓
Adapt existing knowledge
```

There is also a major practical benefit: training a large model from scratch requires enormous computational resources and time.

Therefore, the two main reasons are:

1. **Preserve the knowledge learned during pretraining.**
2. **Avoid the cost of training the entire model from scratch.**

The more fundamental reason is preserving the pretrained knowledge; the reduction in computation is the major practical advantage.

---

## Q2. During fine-tuning, do the model parameters change?

### Answer

**Yes**, during ordinary full fine-tuning, the trainable model parameters are updated.

If the pretrained parameters are:

$$
\theta_{\text{pretrained}}
$$

then fine-tuning produces updated parameters:

$$
\theta_{\text{pretrained}}
\rightarrow
\theta_1
\rightarrow
\theta_2
\rightarrow
\cdots
\rightarrow
\theta_{\text{fine-tuned}}
$$

The process is:

```text
Forward Pass
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

Mathematically:

$$
\boxed{
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta\nabla_\theta L
}
$$

> **Important:** Later, we will learn parameter-efficient methods such as LoRA and QLoRA. In those methods, the original model parameters can be frozen while separate adapter parameters are trained.

---

## Q3. What is the main difference between RAG and fine-tuning?

### Answer

The key difference is **where the adaptation/information happens**.

### RAG

RAG retrieves information from external sources and provides that information to the model as context.

```text
User Question
      ↓
Retriever
      ↓
External Documents
      ↓
Relevant Context
      ↓
LLM
      ↓
Answer
```

The model's parameters generally do not change during normal RAG inference.

Therefore:

$$
\boxed{
\text{RAG} = \text{provide external context}
}
$$

### Fine-Tuning

Fine-tuning continues training a pretrained model on a new dataset, domain, objective, or task.

```text
Pretrained Model
      +
Task-Specific Dataset
      ↓
Training
      ↓
Loss
      ↓
Backpropagation
      ↓
Parameter Updates
      ↓
Fine-Tuned Model
```

Therefore:

$$
\boxed{
\text{Fine-Tuning} = \text{adapt trainable model parameters}
}
$$

### Short Comparison

| RAG | Fine-Tuning |
|---|---|
| Retrieves external information | Trains the model further |
| Supplies information as context | Adapts trainable parameters |
| Normally does not change model parameters during inference | Changes trainable parameters during training |
| Useful for external/current/private information | Useful for task, domain, behavior, style, or format adaptation |

---

## Q4. Is fine-tuning only used to specialize a model for one task?

### Answer

**No.**

Fine-tuning can adapt a pretrained model to many different objectives, domains, or behaviors.

Examples include:

- Sentiment classification
- Text classification
- Translation
- Summarization
- Question answering
- Coding
- Instruction following
- Domain-specific behavior
- Specific response formats
- Conversational behavior

A better definition is:

$$
\boxed{
\text{Fine-tuning = adapting a pretrained model to a new objective, domain, task, or behavior}
}
$$

---

## Q5. Does fine-tuning use a completely different training mechanism from normal neural-network training?

### Answer

**No.**

The fundamental training mechanism is still:

```text
Forward Pass
      ↓
Loss
      ↓
Backpropagation
      ↓
Gradients
      ↓
Optimizer
      ↓
Parameter Update
```

The major difference is the **starting point**.

Training from scratch:

$$
\theta_{\text{random}}
\rightarrow
\theta_{\text{trained}}
$$

Fine-tuning:

$$
\theta_{\text{pretrained}}
\rightarrow
\theta_{\text{fine-tuned}}
$$

---

## Q6. How does fine-tuning connect to PyTorch?

### Answer

Fine-tuning uses the same core PyTorch mechanisms used in ordinary model training.

For example:

```python
loss.backward()
```

calculates gradients.

Then:

```python
optimizer.step()
```

updates the trainable parameters.

Conceptually:

$$
\boxed{
\texttt{loss.backward()}
\rightarrow
\text{calculate gradients}
}
$$

$$
\boxed{
\texttt{optimizer.step()}
\rightarrow
\text{update parameters}
}
$$

The important difference is that the model begins with pretrained weights rather than randomly initialized weights.

---

## Q7. What happens to the model's parameters during fine-tuning?

### Answer

Suppose the model starts with:

$$
\theta_{\text{pretrained}}
$$

A training example is passed through the model, producing a prediction.

The prediction is compared with the desired output to calculate a loss:

$$
L(\theta)
$$

Then the gradient is calculated:

$$
\nabla_\theta L
$$

The optimizer updates the parameters:

$$
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta\nabla_\theta L
$$

After many training steps, the model becomes:

$$
\theta_{\text{fine-tuned}}
$$

So the model has **adapted its parameters to the new training objective**.

---

## Q8. Why is starting from a pretrained model better than starting from random weights?

### Answer

Because the pretrained model has already learned useful representations.

Starting from random weights would require the model to learn general patterns again.

```text
Random Initialization
      ↓
Learn general patterns
      ↓
Learn task-specific patterns
```

With fine-tuning:

```text
Pretrained Knowledge
      ↓
Learn/adapt task-specific patterns
```

This makes fine-tuning much more efficient than retraining a large model from scratch for many practical tasks.

---

## Q9. Can RAG and fine-tuning be used together?

### Answer

**Yes.**

A system can use both:

```text
                 User Question
                       ↓
                  RAG Retrieval
                       ↓
              Relevant Documents
                       ↓
                Fine-Tuned LLM
                       ↓
                     Answer
```

In this setup:

- Fine-tuning adapts the model's behavior or capabilities.
- RAG provides relevant external information at inference time.

They solve different problems and can complement each other.

---

## Q10. What is the most important mental model for fine-tuning?

### Answer

Remember:

$$
\boxed{
\text{Pretrained Model}
+
\text{Task-Specific Data}
+
\text{Further Training}
=
\text{Fine-Tuned Model}
}
$$

Or as a process:

```text
Pretrained Model
      ↓
Task-Specific Dataset
      ↓
Forward Pass
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
      ↓
Fine-Tuned Model
```

---

## Quick Self-Test

Before moving to Lesson 2, you should be able to answer these without looking at the notes:

1. What is a pretrained model?
2. What is fine-tuning?
3. Why start from pretrained weights?
4. Do parameters change during full fine-tuning?
5. What does `loss.backward()` do?
6. What does `optimizer.step()` do?
7. What is the main difference between RAG and fine-tuning?
8. Can RAG and fine-tuning be combined?
9. What is the difference between training from scratch and fine-tuning?
10. Why are LoRA and QLoRA different from ordinary full fine-tuning?

If you can explain these in your own words, you are ready for **Lesson 2: Pretraining vs Fine-Tuning**.
