# Lesson 01 — What Are LLMs?

## Learning Objectives

By the end of this lesson, you should be able to explain:

- What a language model is
- What an LLM is
- What next-token prediction means
- How an LLM represents text
- What parameters are
- What happens during LLM training
- What happens during LLM inference
- What logits are at a high level
- Why modern LLMs are generally decoder-only Transformers
- The difference between a base/pretrained model and a chat/instruction model

---

# 1. What Is a Language Model?

A **language model (LM)** is a model that assigns probabilities to sequences of tokens.

The central question is:

> Given the tokens I have already seen, what is the probability of the next token?

For example:

```text
The cat is ___
```

A language model might estimate:

```text
sleeping  -> 0.45
sitting   -> 0.20
running   -> 0.12
eating    -> 0.08
blue      -> 0.01
...
```

The model does not normally use explicit rules such as:

```text
if subject == cat:
    choose sleeping
```

Instead, it learns statistical patterns and representations from its training data.

---

# 2. Mathematical Definition of a Language Model

Suppose a token sequence is:

```text
x_1, x_2, x_3, ..., x_n
```

A language model estimates:

```text
P(x_1, x_2, ..., x_n)
```

Using the chain rule of probability:

```text
P(x_1, ..., x_n)
=
P(x_1)
* P(x_2 | x_1)
* P(x_3 | x_1, x_2)
* ...
* P(x_n | x_1, ..., x_{n-1})
```

Therefore:

```text
P(x_1, ..., x_n)
=
product from t = 1 to n of P(x_t | x_{<t})
```

where:

```text
x_{<t} = x_1, ..., x_{t-1}
```

This equation is fundamental to understanding autoregressive LLMs.

---

# 3. Next-Token Prediction

Modern autoregressive LLMs are primarily trained around **next-token prediction**.

Suppose the input is:

```text
The cat is
```

The model predicts the next token:

```text
sleeping
```

Conceptually:

```text
The cat is
     |
     v
Transformer
     |
     v
Probability distribution
     |
     v
sleeping
```

After generating `sleeping`, the sequence becomes:

```text
The cat is sleeping
```

The model then predicts the next token again.

---

# 4. Autoregressive Generation

LLM text generation is an autoregressive process.

```text
Prompt
  |
  v
Tokenize
  |
  v
LLM
  |
  v
Predict next token
  |
  v
Append token
  |
  v
LLM
  |
  v
Predict next token
  |
  v
Append token
  |
  v
Repeat
```

Example:

```text
The
 |
 v
The cat
 |
 v
The cat is
 |
 v
The cat is sleeping
 |
 v
The cat is sleeping on
 |
 v
The cat is sleeping on the
 |
 v
The cat is sleeping on the mat
```

Eventually, the model produces an end-of-sequence or stop condition.

---

# 5. Why Is It Called "Large"?

The word **Large** primarily refers to the scale of the model and its training.

There are three important dimensions of scale.

## 5.1 Parameters

An LLM can contain millions, billions, or more trainable parameters.

Examples:

```text
1B parameters
7B parameters
13B parameters
70B parameters
...
```

A parameter is a learned numerical value.

For example:

```text
W =
[
    [ 0.21, -0.13 ],
    [ 0.72,  0.41 ]
]
```

During training, these values are adjusted.

---

## 5.2 Training Data

LLMs are trained on enormous datasets.

The simplified pipeline is:

```text
Documents
   |
   v
Cleaning / Filtering
   |
   v
Tokenization
   |
   v
Huge token dataset
```

The model ultimately trains on **tokens**, not directly on books or web pages.

---

## 5.3 Compute

Training requires enormous amounts of numerical computation.

Conceptually:

```text
Huge dataset
     |
     v
Token batches
     |
     v
GPU / TPU
     |
     v
Matrix operations
     |
     v
Forward pass
     |
     v
Loss
     |
     v
Backward pass
     |
     v
Weight update
```

---

# 6. What Is a Parameter?

An LLM is still a neural network.

For example, attention contains trainable matrices:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

Here:

```text
W_Q
W_K
W_V
```

are trainable parameters.

The complete Transformer contains many such matrices and other learned parameters.

Therefore:

> An LLM is not fundamentally different mathematics from a neural network. It is a very large Transformer-based neural network trained at enormous scale.

---

# 7. What Does an LLM Receive?

An LLM does not directly process a raw string.

The simplified pipeline is:

```text
Text
 |
 v
Tokenizer
 |
 v
Token IDs
 |
 v
Embedding
 |
 v
Vectors
 |
 v
Transformer
```

For example:

```text
"I love cats"
      |
      v
["I", "love", "cats"]
      |
      v
[42, 817, 2917]
      |
      v
Embedding vectors
      |
      v
Transformer
```

You will study tokenization in detail in **Lesson 03** and embeddings in **Lesson 04**.

---

# 8. What Does the Transformer Produce?

Suppose the input is:

```text
The cat is
```

The Transformer produces hidden representations.

Conceptually:

```text
The       -> hidden representation
cat       -> hidden representation
is        -> hidden representation
```

The relevant final hidden representation is passed to the language-model head.

```text
Hidden state
     |
     v
LM Head
     |
     v
Logits
     |
     v
Vocabulary scores
```

If the vocabulary contains 50,000 tokens, the model produces approximately 50,000 logits for a prediction position.

---

# 9. Logits Are Not Probabilities

Suppose the model produces:

```text
sleeping  -> 5.8
sitting   -> 4.1
running   -> 3.2
eating    -> 2.1
```

These are **logits**, not probabilities.

Softmax converts them into probabilities.

The softmax equation is:

```text
P_i = exp(z_i) / sum_j exp(z_j)
```

Conceptually:

```text
Logits
  |
  v
Softmax
  |
  v
Probabilities
```

For example:

```text
sleeping  -> 0.65
sitting   -> 0.18
running   -> 0.09
eating    -> 0.04
...
```

Logits, softmax, and temperature are covered in detail in **Lesson 10**.

---

# 10. What Does an LLM Learn?

It is an oversimplification to say:

> "The LLM stores a copy of the Internet."

It is also not simply a traditional database.

During training, its parameters are adjusted so that the neural network becomes increasingly good at modeling patterns in its training data.

It develops internal representations related to:

- Syntax
- Grammar
- Semantics
- Token relationships
- Concepts
- Patterns
- Programming structures
- Information present in training data
- Long-range dependencies
- Various reasoning-like behaviors

The exact nature of internal representations is an active research area.

---

# 11. How Is an LLM Trained?

Consider:

```text
The cat is sleeping
```

For causal language modeling, we can create next-token prediction relationships:

```text
Input                 Target

The                   cat
The cat               is
The cat is            sleeping
```

The model learns:

```text
P(cat | The)

P(is | The cat)

P(sleeping | The cat is)
```

The predictions are compared against the correct targets.

---

# 12. Why Can Transformer Training Be Parallel?

This connects directly to the Transformer training concepts you already learned.

Suppose:

```text
The cat is sleeping
```

During training, the model can process the complete sequence while applying a causal mask.

Conceptually:

```text
The              -> cat
The cat          -> is
The cat is       -> sleeping
```

These positions can be computed in parallel.

The causal mask prevents a position from using future tokens.

For example:

```text
             The   cat   is   sleeping

The           Y     N    N      N

cat           Y     Y    N      N

is            Y     Y    Y      N

sleeping      Y     Y    Y      Y
```

`Y` means the token is visible.

`N` means the token is masked.

This is one of the major reasons Transformers are highly parallelizable during training.

---

# 13. Training Objective

For a correct target token `y`:

```text
L = -log P(y | context)
```

If:

```text
P(y | context) = 0.8
```

then:

```text
L = -log(0.8)
```

which is relatively small.

If:

```text
P(y | context) = 0.01
```

then:

```text
L = -log(0.01)
```

which is much larger.

Therefore, training encourages the model to assign high probability to the correct next token.

---

# 14. Backpropagation

After calculating the loss:

```text
Prediction
    |
    v
Loss
    |
    v
Backpropagation
    |
    v
Gradients
    |
    v
Optimizer
    |
    v
Update parameters
```

A simplified gradient-descent update is:

```text
W_new = W_old - eta * (dL / dW)
```

where:

- `W` = model parameter
- `L` = loss
- `eta` = learning rate

In practice, modern LLM training commonly uses optimizers such as Adam or AdamW.

---

# 15. Complete LLM Training Loop

```text
                 TRAINING

         Training dataset
                 |
                 v
            Tokenization
                 |
                 v
              Batches
                 |
                 v
             Embeddings
                 |
                 v
         Transformer model
                 |
                 v
          Next-token logits
                 |
                 v
           Cross-entropy
                 |
                 v
                Loss
                 |
                 v
          Backpropagation
                 |
                 v
             Gradients
                 |
                 v
             Optimizer
                 |
                 v
          Update weights
                 |
                 +-------------------> Next batch
```

This repeats over very large numbers of tokens and training steps.

---

# 16. What Is Pretraining?

The initial large-scale training of a language model is generally called **pretraining**.

Conceptually:

```text
Massive general dataset
          |
          v
Next-token prediction
          |
          v
Transformer
          |
          v
Learned parameters
          |
          v
Base LLM
```

The resulting model is called a **base model** or **pretrained language model**.

---

# 17. Base Model vs Chat Model

A pretrained base model primarily learns to model token sequences.

A chat or instruction model has typically undergone additional training to make it better at following instructions and interacting conversationally.

Conceptually:

```text
Massive text dataset
        |
        v
    Pretraining
        |
        v
     Base LLM
        |
        v
Instruction / alignment training
        |
        v
Chat / Instruction model
```

The details of SFT, RLHF, and DPO are outside this current lesson.

---

# 18. Why Does an LLM Appear to "Know" Things?

Suppose you ask:

```text
What is the capital of France?
```

The model may generate:

```text
Paris
```

During training, it learned patterns that make the token sequence containing this answer highly probable.

Conceptually:

```text
Question
   |
   v
Internal representations
   |
   v
Logits
   |
   v
Probability distribution
   |
   v
Paris
```

However, a high probability does not guarantee factual correctness.

This distinction will become important when studying LLM limitations and behavior.

---

# 19. LLM vs Search Engine

A search engine generally works approximately like:

```text
Query
 |
 v
Retrieve documents
 |
 v
Rank documents
 |
 v
Return results
```

An autoregressive LLM generally works like:

```text
Prompt
 |
 v
Transformer
 |
 v
Probability distribution
 |
 v
Generate token
 |
 v
Generate next token
```

They are fundamentally different systems.

---

# 20. LLM vs Rule-Based System

A rule-based system might explicitly contain:

```text
IF temperature > 30:
    output "Hot"
```

An LLM normally does not contain an explicit rule for every possible sentence.

Instead, its behavior emerges from learned parameters and the computation performed by its neural network.

---

# 21. LLM vs Traditional Task-Specific ML

A traditional ML model might be trained for a relatively specific task:

```text
Features
   |
   v
Classifier
   |
   v
Spam / Not Spam
```

An LLM is a general-purpose sequence model:

```text
Tokens
   |
   v
Transformer
   |
   v
Next-token distribution
```

The same basic model can potentially generate:

- Text
- Code
- Explanations
- Summaries
- Translations
- Structured outputs

depending on its training and prompting.

---

# 22. Connection to the Transformer You Already Learned

You previously learned the original Encoder-Decoder Transformer:

```text
Encoder
   |
   v
Encoder representations
   |
   v
K, V
   |
   v
Decoder Cross-Attention
   |
   v
Output
```

Modern GPT-style LLMs generally use a different architecture:

```text
Input tokens
     |
     v
Decoder-only Transformer
     |
     v
Causal Self-Attention
     |
     v
FFN
     |
     v
Causal Self-Attention
     |
     v
FFN
     |
     v
...
     |
     v
Next-token prediction
```

There is generally:

```text
No separate encoder
No encoder-decoder cross-attention
```

Instead, the model uses stacked causal self-attention blocks.

This is the major architectural transition you will study in **Lesson 05 — LLM Architecture Internals**.

---

# 23. The Full LLM Lifecycle

At a high level:

```text
                         TEXT
                           |
                           v
                      TOKENIZER
                           |
                           v
                       TOKEN IDs
                           |
                           v
                      EMBEDDINGS
                           |
                           v
                  POSITION INFORMATION
                           |
                           v
             DECODER-ONLY TRANSFORMER
                           |
                  +--------+--------+
                  |                 |
            Self-Attention         FFN
                  |                 |
                  +--------+--------+
                           |
                       Repeat N
                           |
                           v
                        LM Head
                           |
                           v
                         Logits
                           |
                           v
                        Softmax
                           |
                           v
                     Probabilities
                           |
                           v
                       Sampling
                           |
                           v
                      Next Token
                           |
                           +----------> Repeat
```

The remaining lessons will explain each part of this pipeline.

---

# 24. Important Terms

| Term | Meaning |
|---|---|
| Language Model | Model that assigns probabilities to token sequences |
| LLM | Large-scale language model, typically Transformer-based |
| Token | Unit of text processed by the model |
| Token ID | Integer representing a vocabulary token |
| Parameter | Learned numerical value in the neural network |
| Hidden State | Internal vector representation produced by the model |
| Logit | Unnormalized score for a vocabulary token |
| Softmax | Converts logits into probabilities |
| Pretraining | Large-scale initial training of a model |
| Base Model | Model resulting primarily from pretraining |
| Autoregressive | Generates one token conditioned on previous tokens |
| Causal Mask | Prevents a position from attending to future tokens |
| LM Head | Projection from hidden states to vocabulary logits |

---

# 25. Key Equations

## Sequence Probability

```text
P(x_1, ..., x_n)
=
product from t = 1 to n of P(x_t | x_{<t})
```

## Next-Token Objective

```text
P(x_t | x_1, ..., x_{t-1})
```

## Cross-Entropy / Negative Log-Likelihood for One Target

```text
L = -log P(y | context)
```

## Softmax

```text
P_i = exp(z_i) / sum_j exp(z_j)
```

## Simplified Parameter Update

```text
W_new = W_old - eta * (dL / dW)
```

---

# 26. Training vs Inference

## Training

```text
Training text
     |
     v
Tokenization
     |
     v
Input tokens
     |
     v
Transformer
     |
     v
Next-token predictions
     |
     v
Loss
     |
     v
Backpropagation
     |
     v
Weight updates
```

The target tokens are already available, so the Transformer can compute many target positions in parallel using causal masking.

---

## Inference

```text
Prompt
  |
  v
Tokenization
  |
  v
Transformer
  |
  v
Logits
  |
  v
Sampling
  |
  v
Next token
  |
  v
Append token
  |
  v
Transformer
  |
  v
Next token
  |
  v
Repeat
```

The target is unknown, so generation is autoregressive.

---

# 27. The Most Important Concepts to Remember

If you remember only the following points, you have the foundation of Lesson 01.

### 1. LLM

> An LLM is a large neural language model that models token sequences.

### 2. Next-Token Prediction

The fundamental autoregressive objective is:

```text
P(x_t | x_1, ..., x_{t-1})
```

### 3. Tokens

The model operates on tokens:

```text
Text
 |
 v
Tokens
 |
 v
Token IDs
 |
 v
Embeddings
```

### 4. Hidden Representations

The Transformer converts embeddings into contextual hidden representations.

### 5. LM Head

The LM head converts hidden states into logits over the vocabulary.

### 6. Softmax

Softmax converts logits into probabilities.

### 7. Autoregressive Inference

Inference works approximately like:

```text
Prompt
 |
 v
Token
 |
 v
Append
 |
 v
Token
 |
 v
Append
 |
 v
Repeat
```

### 8. Training

Training uses next-token prediction:

```text
Prediction
 |
 v
Loss
 |
 v
Backpropagation
 |
 v
Weight update
```

### 9. Decoder-Only Architecture

Modern GPT-style LLMs are generally decoder-only Transformers.

---

# 28. Final Mental Model

You should now be able to understand this high-level pipeline:

```text
                     RAW TEXT
                        |
                        v
                   TOKENIZER
                        |
                        v
                    TOKEN IDs
                        |
                        v
                   EMBEDDINGS
                        |
                        v
               POSITION INFORMATION
                        |
                        v
           DECODER-ONLY TRANSFORMER
                        |
                 +------+------+
                 |             |
          Self-Attention       FFN
                 |             |
                 +------+------+
                        |
                    Repeat N
                        |
                        v
                    LM HEAD
                        |
                        v
                      LOGITS
                        |
                        v
                     SOFTMAX
                        |
                        v
                  PROBABILITIES
                        |
                        v
                    SAMPLING
                        |
                        v
                   NEXT TOKEN
                        |
                        +---------> Repeat
```

---

# 29. Self-Check Questions

Before moving to Lesson 02, you should be able to answer:

1. What is a language model?
2. What does next-token prediction mean?
3. What is the mathematical form of an autoregressive language model?
4. Why does an LLM need tokenization?
5. What is a token ID?
6. What is a parameter?
7. What is the difference between a logit and a probability?
8. What does the LM head do?
9. Why can Transformer training be parallelized?
10. Why is inference autoregressive?
11. What is pretraining?
12. What is a base model?
13. How is a chat model different from a base model at a high level?
14. Why are modern GPT-style LLMs called decoder-only?
15. How is a decoder-only LLM different from the Encoder-Decoder Transformer you already learned?

---

# 30. Roadmap Position

```text
PHASE 1 - LLM FOUNDATIONS

01. What Are LLMs?                  <- YOU ARE HERE
        |
        v
02. Generative AI: The Big Picture
        |
        v
03. Tokenization Deep Dive
        |
        v
04. Embeddings & Token Representations


PHASE 2 - MODERN LLM ARCHITECTURE

05. LLM Architecture Internals
        |
        v
06. Positional Encodings
        |
        v
07. Mixture of Experts
        |
        v
08. Context Window & Attention Patterns


PHASE 3 - LLM TRAINING & GENERATION

09. Pre-training Objectives
        |
        v
10. Logits, Softmax & Temperature
        |
        v
11. Sampling Strategies
        |
        v
12. Multi-turn Conversations & Memory
```

---

# Next Lesson

**Lesson 02 - Generative AI: The Big Picture**
