# Lesson 14 — Fine-Tuning LLMs With Custom Datasets

> **Goal:** Understand how to prepare custom data for LLM fine-tuning, from raw examples through formatting and tokenization to training-ready batches.

---

# 1. Why Do We Need a Custom Dataset?

A pretrained LLM already has general capabilities such as:

- Language understanding
- Programming
- Mathematics
- General knowledge
- Reasoning patterns

But we may want to specialize it for:

- Medical question answering
- Legal documents
- Customer support
- Company documentation
- Python tutoring
- Financial analysis

We need examples showing the model **what behavior we want**.

Example:

```text
User:
What is gradient descent?

Assistant:
Gradient descent is an optimization algorithm...
```

Many examples like this can be used for fine-tuning.

---

# 2. Basic Fine-Tuning Pipeline

The overall pipeline is:

```text
Raw Data
   ↓
Clean Data
   ↓
Format Data
   ↓
Instruction / Response Examples
   ↓
Tokenization
   ↓
Train / Validation Split
   ↓
Fine-Tuning
   ↓
Fine-Tuned Model
```

The model does not directly train on raw strings. The text eventually has to be converted into **tokens**, which become numerical IDs.

---

# 3. Example of Raw Data

Imagine a simple CSV:

```csv
question,answer
What is LoRA?,LoRA is a parameter-efficient fine-tuning method.
What is PyTorch?,PyTorch is a deep learning framework.
What is an LLM?,An LLM is a large language model.
```

This is human-readable data.

The fine-tuning pipeline transforms it into a format suitable for the model and tokenizer.

---

# 4. Instruction Dataset

A common structure is:

```json
{
  "instruction": "What is LoRA?",
  "response": "LoRA is a parameter-efficient fine-tuning method."
}
```

Another example:

```json
{
  "instruction": "Explain gradient descent.",
  "response": "Gradient descent is an optimization algorithm..."
}
```

Now we have:

```text
Instruction
     +
Expected Response
```

This is the foundation of **Instruction Fine-Tuning (IFT)**.

---

# 5. Input → Output

Think of the dataset as:

\[
\boxed{X\rightarrow Y}
\]

where:

\[
X=\text{input / instruction}
\]

and:

\[
Y=\text{desired response}
\]

Example:

```text
X:
Explain LoRA.

↓

Y:
LoRA is a parameter-efficient fine-tuning technique...
```

During supervised training, the model learns to produce \(Y\) given \(X\).

---

# 6. Another Dataset Format

You may also encounter:

```json
{
  "prompt": "Explain LoRA.",
  "completion": "LoRA is a parameter-efficient fine-tuning technique..."
}
```

Conceptually:

```text
prompt
  ↓
completion
```

This is similar to:

```text
instruction
  ↓
response
```

The exact field names are not universal.

The important semantic roles are:

```text
Input
+
Expected Output
```

---

# 7. Chat-Style Dataset

Modern conversational LLMs often use a message-based structure.

Example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is LoRA?"
    },
    {
      "role": "assistant",
      "content": "LoRA is a parameter-efficient fine-tuning method."
    }
  ]
}
```

This is closer to how chat models represent conversations.

---

# 8. Multiple Turns

A dataset can contain multiple turns:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is LoRA?"
    },
    {
      "role": "assistant",
      "content": "LoRA is a parameter-efficient fine-tuning method."
    },
    {
      "role": "user",
      "content": "Why is it efficient?"
    },
    {
      "role": "assistant",
      "content": "Because it trains a small number of low-rank parameters..."
    }
  ]
}
```

Now the model receives conversational context.

---

# 9. Why Dataset Format Matters

The training examples strongly influence the behavior the model learns.

For example:

### Dataset A

```text
Question → Short Answer
```

may encourage concise answers.

### Dataset B

```text
Question → Detailed Explanation
```

may encourage longer explanations.

### Dataset C

```text
User → Assistant conversation
```

may teach conversational behavior.

Therefore:

\[
\boxed{
\text{Dataset format + examples}
\rightarrow
\text{learned behavior}
}
\]

---

# 10. Dataset Quality Is Important

Suppose you have:

```text
1,000,000 poor examples
```

versus:

```text
100,000 high-quality examples
```

The smaller high-quality dataset may be more useful depending on the task.

Important dataset properties include:

- Correctness
- Relevance
- Diversity
- Consistency
- Formatting
- Appropriate difficulty
- Lack of unnecessary noise

Therefore:

> More examples do not automatically mean better fine-tuning.

---

# 11. Data Cleaning

Before training, datasets often need cleaning.

Typical operations:

```text
Raw Dataset
    ↓
Remove duplicates
    ↓
Remove corrupted examples
    ↓
Remove irrelevant examples
    ↓
Fix formatting
    ↓
Check answers
    ↓
Final Dataset
```

---

# 12. Deduplication

Suppose:

```text
Example 1:
What is Python? → Python is a programming language.

Example 2:
What is Python? → Python is a programming language.
```

These are duplicates.

Unnecessary duplicates can be removed before training.

---

# 13. Remove Bad Examples

Example of a poor training example:

```text
Question:
What is LoRA?

Answer:
I don't know.
```

If the goal is to teach the model to explain LoRA, this is a poor training signal.

Another example:

```text
Question:
What is LoRA?

Answer:
LoRA is a database management system.
```

This is incorrect information.

Training on incorrect data can teach incorrect behavior.

Therefore:

\[
\boxed{
\text{Bad data}\rightarrow\text{Bad training signal}
}
\]

---

# 14. Dataset Consistency

Suppose a dataset contains:

```text
Example 1:
User → Assistant

Example 2:
Question → Answer

Example 3:
Instruction → Response

Example 4:
Human → AI
```

It may be possible to use these, but inconsistent formatting makes preprocessing more complicated.

A common approach is to normalize them into one consistent structure.

For example:

```text
messages
  ↓
user
  ↓
assistant
```

---

# 15. Train / Validation / Test Split

Usually we do not train on every example.

For example:

```text
100,000 examples

        ↓

80,000 → Training
10,000 → Validation
10,000 → Test
```

The exact split is not universal.

---

# 16. Training Set

The training set is used to update model parameters.

```text
Training examples
       ↓
Model
       ↓
Loss
       ↓
Backpropagation
       ↓
Parameter updates
```

For LoRA fine-tuning:

```text
Base weights
↓
Frozen

LoRA parameters
↓
Updated
```

---

# 17. Validation Set

The validation set is used to monitor how well the model generalizes during training/development.

Validation examples should not be used as ordinary training examples if you want a clean measure of generalization.

---

# 18. Test Set

A separate test set can be used for final evaluation.

Conceptually:

```text
Training
   ↓
Used for learning

Validation
   ↓
Used during development

Test
   ↓
Final evaluation
```

---

# 19. Example Dataset Split

Suppose:

\[
N=10,000
\]

examples.

One possible split is:

\[
80\%=8,000
\]

training examples,

\[
10\%=1,000
\]

validation examples, and

\[
10\%=1,000
\]

test examples.

The exact split depends on the task and dataset.

---

# 20. Tokenization

The model does not directly process raw text such as:

```text
"Explain LoRA"
```

The tokenizer converts text into tokens.

For example, conceptually:

```text
"Explain LoRA"
       ↓
["Explain", " Lo", "RA"]
```

The exact tokens depend on the tokenizer.

Those tokens are mapped to IDs:

```text
["Explain", " Lo", "RA"]
          ↓
[1234, 5678, 9012]
```

The actual IDs depend on the tokenizer vocabulary.

---

# 21. Token IDs

The model receives numerical token IDs.

Conceptually:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embedding
 ↓
Transformer
```

This connects directly to the Transformer concepts learned earlier.

---

# 22. Tokenization Pipeline

The complete process is:

```text
"What is LoRA?"
       ↓
Tokenizer
       ↓
Token IDs
       ↓
Embedding vectors
       ↓
Transformer
       ↓
Predictions
```

---

# 23. What Is the Label?

During supervised fine-tuning, the model needs a target.

Example:

```text
Input:
What is LoRA?

Target:
LoRA is a parameter-efficient fine-tuning method.
```

The model generates predictions and compares them with the target tokens.

The difference contributes to the loss.

---

# 24. Language Modeling Objective

Suppose the target sequence is:

\[
y_1,y_2,\ldots,y_T
\]

The model predicts:

\[
P(y_t\mid y_{<t},x)
\]

A common training objective is negative log-likelihood / cross-entropy:

\[
\boxed{
\mathcal{L}
=
-\sum_{t=1}^{T}
\log P(y_t\mid y_{<t},x)
}
\]

This is the mathematical foundation of supervised next-token training.

---

# 25. Example of Next-Token Training

Suppose:

```text
Input:
What is LoRA?

Target:
LoRA is efficient.
```

Conceptually, the model learns:

```text
LoRA
 ↓
is
 ↓
efficient
```

At each position, the model predicts the next target token.

The loss measures how well those predictions match the target.

---

# 26. Teacher Forcing Connection

This connects to the Transformer decoder concepts learned earlier.

During training, the target sequence can be shifted:

```text
Input to decoder:
<BOS> LoRA is

Target:
LoRA is efficient
```

The model learns:

```text
<BOS> → LoRA
LoRA → is
LoRA is → efficient
```

This is the same autoregressive training principle discussed with **masked self-attention and teacher forcing**.

---

# 27. Why the Dataset Must Be Tokenized Correctly

Suppose your dataset is:

```text
User:
Explain LoRA.

Assistant:
LoRA is a parameter-efficient method.
```

The training pipeline must determine:

```text
Which tokens are input?
Which tokens are targets?
Which tokens should contribute to the loss?
```

This becomes especially important for chat models.

It is one reason **chat templates** and **SFT data collators** become important later.

---

# 28. Creating a Dataset With Hugging Face `datasets`

A common approach is:

```python
from datasets import Dataset

data = {
    "instruction": [
        "What is LoRA?",
        "What is PyTorch?",
        "What is an LLM?"
    ],
    "response": [
        "LoRA is a parameter-efficient fine-tuning method.",
        "PyTorch is a deep learning framework.",
        "An LLM is a large language model."
    ]
}

dataset = Dataset.from_dict(data)
```

Now:

```python
print(dataset)
```

produces a Hugging Face Dataset object.

---

# 29. Inspecting the Dataset

You can inspect an example:

```python
print(dataset[0])
```

Conceptually:

```text
{
    "instruction": "What is LoRA?",
    "response": "LoRA is a parameter-efficient fine-tuning method."
}
```

---

# 30. Splitting the Dataset

You can create a train/test split:

```python
dataset = dataset.train_test_split(
    test_size=0.1
)
```

Then:

```python
train_dataset = dataset["train"]
test_dataset = dataset["test"]
```

Conceptually:

```text
Dataset
   │
   ├── Train
   │
   └── Test
```

For real fine-tuning projects, you may also maintain a separate validation set.

---

# 31. Formatting Examples

Suppose:

```python
example = {
    "instruction": "Explain LoRA.",
    "response": "LoRA is a parameter-efficient fine-tuning method."
}
```

One possible formatting strategy is:

```python
text = (
    "### Instruction:\n"
    + example["instruction"]
    + "\n\n"
    + "### Response:\n"
    + example["response"]
)
```

Result:

```text
### Instruction:
Explain LoRA.

### Response:
LoRA is a parameter-efficient fine-tuning method.
```

---

# 32. Important: Formatting Is Model-Dependent

Do not assume every LLM should be formatted with:

```text
### Instruction:
...
### Response:
...
```

Modern chat models often have their own **chat template**.

For example, a model may expect:

```text
system
user
assistant
```

with special control tokens.

We will study this properly in the **Chat Templates** lesson.

---

# 33. Dataset → Tokenizer

Once the data is formatted appropriately, it can be tokenized.

Conceptually:

```python
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512
    )
```

Then:

```python
tokenized_dataset = dataset.map(
    tokenize
)
```

The dataset now contains token IDs rather than only raw text.

---

# 34. `max_length`

One important tokenizer parameter is:

```python
max_length=512
```

This determines the maximum sequence length used by the tokenizer in this example.

If a sequence is longer, depending on the configuration:

```text
Long sequence
      ↓
Truncation
      ↓
Maximum allowed length
```

Longer sequences generally require more memory and computation.

---

# 35. Padding

Sequences in a batch may have different lengths.

For example:

```text
Example 1 → 20 tokens
Example 2 → 35 tokens
Example 3 → 27 tokens
```

A batch often needs a common length.

They can be padded:

```text
Example 1 → 35
Example 2 → 35
Example 3 → 35
```

using a padding token.

Conceptually:

```text
Short sequence
      ↓
Padding tokens
      ↓
Same batch length
```

Attention masks are used so padding tokens do not incorrectly contribute to the model's attention computation.

---

# 36. Important Dataset Pipeline

The complete preparation pipeline is:

```text
Raw Dataset
     ↓
Clean
     ↓
Normalize Format
     ↓
Instruction / Response
     ↓
Chat Format (if required)
     ↓
Tokenizer
     ↓
Token IDs
     ↓
Padding / Truncation
     ↓
Training Batch
     ↓
Model
```

---

# 37. Dataset Quality Checklist

Before fine-tuning, check:

### 1. Correctness

Are the answers correct?

### 2. Relevance

Does each example relate to the target task?

### 3. Consistency

Are examples formatted consistently?

### 4. Diversity

Does the dataset cover different ways users might ask the same thing?

### 5. Duplicates

Have unnecessary duplicates been removed?

### 6. Length

Are examples within reasonable context limits?

### 7. Safety / Privacy

Does the dataset contain information that should not be used?

---

# 38. Example: Building a Small Instruction Dataset

```python
from datasets import Dataset

data = {
    "instruction": [
        "What is LoRA?",
        "What is QLoRA?",
        "What is fine-tuning?"
    ],

    "response": [
        "LoRA is a parameter-efficient fine-tuning method.",
        "QLoRA combines quantization with LoRA.",
        "Fine-tuning adapts a pretrained model to a specific task or domain."
    ]
}

dataset = Dataset.from_dict(data)

print(dataset)
print(dataset[0])
```

---

# 39. Formatting Function

```python
def format_example(example):

    return {
        "text": (
            "### Instruction:\n"
            + example["instruction"]
            + "\n\n"
            + "### Response:\n"
            + example["response"]
        )
    }
```

Apply it:

```python
formatted_dataset = dataset.map(
    format_example
)
```

Now each example contains a `text` field.

---

# 40. Tokenization

Conceptually:

```python
def tokenize(example):

    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512
    )
```

Then:

```python
tokenized_dataset = formatted_dataset.map(
    tokenize
)
```

Now:

```text
instruction
+
response
   ↓
formatted text
   ↓
tokenizer
   ↓
input_ids
   ↓
training
```

---

# 41. How This Connects to LoRA

The dataset pipeline and LoRA are different components.

```text
                    Fine-Tuning
                         │
           ┌─────────────┴─────────────┐
           ↓                           ↓
        MODEL                         DATA
           │                           │
         LoRA                    Custom Dataset
           │                           │
         QLoRA                    Formatting
           │                           │
           │                       Tokenization
           │                           │
           └─────────────┬─────────────┘
                         ↓
                       SFT
```

LoRA determines:

> **Which model parameters are adapted.**

The dataset determines:

> **What the model learns from.**

---

# 42. Very Important Distinction

Do not confuse:

### LoRA

A **parameter-efficient fine-tuning method**.

### SFT

A **supervised training approach / objective**.

### Dataset

The **training examples**.

So:

```text
Dataset
   ↓
What examples do we train on?

SFT
   ↓
How do we learn from supervised examples?

LoRA
   ↓
Which parameters do we update?
```

This distinction is extremely important.

---

# 43. LoRA + SFT

In practical LLM fine-tuning, we can combine them:

```text
Custom Instruction Dataset
            ↓
           SFT
            ↓
     Train using LoRA
            ↓
   Fine-Tuned LLM Adapter
```

So:

\[
\boxed{
\text{LoRA + SFT}
}
\]

is a common combination.

With quantization:

\[
\boxed{
\text{QLoRA + SFT}
}
\]

is also common.

---

# 44. Dataset → IFT → SFT

We are approaching another important distinction:

```text
Custom Dataset
      ↓
Instruction / Response Examples
      ↓
IFT
      ↓
SFT
```

Terminology can vary between sources, so the exact definitions should be checked against the source material when we study the next lesson.

At a high level:

```text
Instruction Fine-Tuning
        ↓
Teach the model to follow instructions

Supervised Fine-Tuning
        ↓
Train using labeled input/output examples
```

---

# 45. Questions & Answers

## Q1. Why do we need a custom dataset?

To provide examples that teach a pretrained model the desired task, domain, or behavior.

---

## Q2. What is an instruction dataset?

A dataset containing instructions/prompts and corresponding desired responses.

Example:

```text
Instruction:
Explain LoRA.

Response:
LoRA is a parameter-efficient fine-tuning method.
```

---

## Q3. What is the basic supervised training relationship?

\[
\boxed{X\rightarrow Y}
\]

where \(X\) is the input and \(Y\) is the desired output.

---

## Q4. Why do we tokenize the dataset?

Because the model operates on numerical representations, and the model receives token IDs rather than raw text.

---

## Q5. What is the purpose of a validation set?

To evaluate generalization during training/development without using those examples as ordinary training data.

---

## Q6. Why clean the dataset?

To remove incorrect, irrelevant, corrupted, duplicated, or inconsistent examples that could provide poor training signals.

---

## Q7. What is the difference between LoRA and a dataset?

LoRA determines **how model parameters are adapted**.

The dataset determines **what examples the model learns from**.

---

## Q8. Can LoRA and SFT be used together?

Yes.

Conceptually:

```text
SFT
+
LoRA
=
Parameter-efficient supervised fine-tuning
```

---

## Q9. What is a chat-style dataset?

A dataset represented as messages with roles such as:

```text
system
user
assistant
```

---

## Q10. Why are chat templates important?

Because different chat models expect conversations to be formatted using specific control tokens and structures.

---

# 46. Self-Test

Try answering these without looking back:

1. Why do we need a custom dataset for fine-tuning?
2. What is an instruction-response pair?
3. What is the difference between raw data and a training-ready dataset?
4. Why is dataset quality important?
5. What is deduplication?
6. What are training, validation, and test sets?
7. What does a tokenizer do?
8. What are token IDs?
9. Why do we need padding?
10. Why do we use truncation?
11. What is the purpose of `max_length`?
12. What is the relationship between input \(X\) and target \(Y\)?
13. What is the difference between LoRA and SFT?
14. Can LoRA and SFT be combined?
15. Why are chat templates important for modern LLMs?

---

# 47. Final Mental Model

Fine-tuning has two connected sides:

```text
                         LLM FINE-TUNING
                                │
                ┌───────────────┴───────────────┐
                ↓                               ↓
              MODEL                             DATA
                │                               │
              LoRA                         Custom Dataset
                │                               │
             QLoRA                         Cleaning
                │                               │
        LoRA Variants                       Formatting
                                                │
                                           Tokenization
                │                               │
                └───────────────┬───────────────┘
                                ↓
                               SFT
                                ↓
                         Fine-Tuned Model
```

The key idea:

\[
\boxed{
\text{LoRA tells us how to adapt the model}
}
\]

while:

\[
\boxed{
\text{The dataset tells us what behavior to learn}
}
\]

---

# 48. Lesson 14 Summary

You learned:

- Why custom datasets are needed
- Instruction-response datasets
- Chat-style datasets
- Multi-turn datasets
- Dataset cleaning
- Deduplication
- Training/validation/test splits
- Tokenization
- Token IDs
- Padding
- Truncation
- `max_length`
- Labels and targets
- Next-token training
- Teacher forcing connection
- Hugging Face `datasets`
- Dataset formatting
- Dataset → tokenizer pipeline
- How datasets connect to LoRA
- Difference between Dataset, LoRA, and SFT

The complete preparation/training pipeline is:

\[
\boxed{
\text{Raw Data}
\rightarrow
\text{Clean}
\rightarrow
\text{Format}
\rightarrow
\text{Tokenize}
\rightarrow
\text{Batch}
\rightarrow
\text{SFT}
\rightarrow
\text{LoRA/QLoRA}
}
\]

---

# 49. Next Lesson

## Lesson 15 — IFT: Instruction Fine-Tuning

Next we will focus specifically on **Instruction Fine-Tuning (IFT)**.

We will learn:

```text
Base LLM
   ↓
Instruction Dataset
   ↓
Instruction Formatting
   ↓
Instruction Tuning
   ↓
Instruction-Following Model
```

We will also distinguish:

```text
Pretraining
     ↓
IFT
     ↓
SFT
     ↓
Preference / Reward Training
     ↓
DPO / RFT / GRPO
```

This prepares us for:

```text
SFT
↓
RFT
↓
DPO
↓
GRPO
↓
Reasoning LLMs
```
