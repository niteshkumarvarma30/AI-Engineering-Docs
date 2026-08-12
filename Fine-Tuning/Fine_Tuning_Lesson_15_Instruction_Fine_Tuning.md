# Lesson 15 — Instruction Fine-Tuning (IFT)

> **Goal:** Understand Instruction Fine-Tuning (IFT), why it is used after pretraining, what instruction datasets look like, how IFT relates to SFT, LoRA, QLoRA, chat templates, and preference training.

---

# 1. Where Does IFT Fit?

A simplified evolution of an LLM is:

```text
                    Pretraining
                         ↓
                 Base Language Model
                         ↓
                       IFT
                         ↓
             Instruction-Following Model
                         ↓
                  Preference Training
                         ↓
                Aligned / Helpful Model
```

IFT happens **after pretraining**.

---

# 2. Pretraining vs IFT

This distinction is extremely important.

## Pretraining

During pretraining, the model learns general language patterns from a huge corpus.

Examples include:

```text
Books
Web pages
Articles
Code
Documents
```

The objective is primarily next-token prediction.

Conceptually:

```text
"The capital of France is"
                    ↓
                 "Paris"
```

The model learns:

- Language
- Syntax
- Semantics
- Facts and patterns
- General representations

---

# 3. What Does IFT Teach?

After pretraining, a model may know a lot but not necessarily behave like a useful assistant.

For example, given:

```text
Explain gradient descent.
```

a base model may simply continue the text rather than reliably produce a helpful answer.

IFT teaches the model:

> When a user gives an instruction, produce an appropriate response.

So the training pattern becomes:

```text
Instruction
     ↓
Desired Response
```

---

# 4. Example of IFT

Suppose the dataset contains:

```text
Instruction:
Explain gradient descent simply.

Response:
Gradient descent is an optimization algorithm that
iteratively adjusts model parameters to reduce the loss.
```

Another:

```text
Instruction:
Write a Python function that adds two numbers.

Response:
def add(a, b):
    return a + b
```

Another:

```text
Instruction:
Translate "Hello" into French.

Response:
Bonjour.
```

The model sees many such examples and learns the general behavior:

```text
User Instruction
       ↓
Understand request
       ↓
Generate appropriate response
```

---

# 5. IFT Dataset

A basic IFT dataset can look like:

```json
{
  "instruction": "Explain what LoRA is.",
  "response": "LoRA is a parameter-efficient fine-tuning technique..."
}
```

Or:

```json
{
  "instruction": "What is PyTorch?",
  "response": "PyTorch is an open-source deep learning framework..."
}
```

Or chat format:

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

---

# 6. What Actually Changes During IFT?

Suppose we start with a pretrained model:

```text
Pretrained LLM
      ↓
Weights W
```

During full fine-tuning:

```text
W
↓
Trainable
```

During LoRA fine-tuning:

```text
W
↓
Frozen

A + B
↓
Trainable
```

Therefore, IFT describes the **training behavior/data**, while LoRA describes **how the model parameters are adapted**.

---

# 7. IFT + LoRA

These concepts can be combined:

```text
                 Instruction Dataset
                         ↓
                        IFT
                         ↓
                 Supervised Training
                         ↓
                       LoRA
                         ↓
                Update A and B
                         ↓
                Adapted LLM
```

More precisely:

```text
Dataset
   ↓
Instruction/Response examples
   ↓
Tokenization
   ↓
SFT / Instruction Training
   ↓
LoRA adapter parameters updated
```

---

# 8. IFT Is Not a New Model Architecture

IFT does **not** mean that we create a new Transformer architecture.

The Transformer architecture can remain the same.

Instead:

```text
Same pretrained architecture
          +
Instruction-following examples
          ↓
Fine-tuned behavior
```

---

# 9. What Does the Model Learn?

Suppose the dataset contains:

```text
Instruction:
Explain recursion.

Response:
Recursion is a programming technique where...
```

The model does not ideally memorize only that one pair.

It learns a general pattern:

```text
"Explain X"
      ↓
Give an explanation of X
```

If trained on diverse examples:

```text
Explain LoRA
Explain CNN
Explain gradient descent
Explain attention
Explain backpropagation
```

the model can generalize to new instructions.

---

# 10. Instruction Diversity

Dataset diversity is important.

A dataset can contain different instruction types:

```text
Explain X
Summarize X
Translate X
Classify X
Write code for X
Compare X and Y
Extract information from X
Answer this question
Rewrite X
Generate X
```

The model gets exposure to different task formats.

---

# 11. Example of Diverse IFT Data

```text
Instruction:
Explain overfitting.

Response:
Overfitting occurs when a model learns the training data
too closely and performs poorly on unseen data.
```

```text
Instruction:
Compare CNN and RNN.

Response:
CNNs are commonly used for spatial patterns, while RNNs
were designed for sequential data.
```

```text
Instruction:
Write Python code to calculate the mean.

Response:
def mean(values):
    return sum(values) / len(values)
```

```text
Instruction:
Translate "Good morning" into Spanish.

Response:
Buenos días.
```

Different tasks can be trained into the same model.

---

# 12. Instruction Tuning vs Ordinary Fine-Tuning

The term **fine-tuning** is broad.

It can mean adapting a pretrained model to a particular dataset or task.

Instruction tuning specifically focuses on examples where the model is given an instruction and expected to produce an appropriate response.

Conceptually:

```text
Fine-Tuning
│
├── Task-specific fine-tuning
│
├── Domain adaptation
│
└── Instruction tuning
```

Terminology can vary between sources, so the exact meaning should be interpreted according to the context.

---

# 13. Instruction Following

The main behavioral objective is:

\[
\boxed{
\text{Instruction}
\rightarrow
\text{Appropriate Response}
}
\]

Example:

```text
User:
Explain attention in simple terms.

Model:
Attention allows the model to determine which
tokens are more relevant to each other...
```

The model is trained to follow the requested task.

---

# 14. Training Objective

Suppose the desired response contains:

\[
y_1,y_2,\ldots,y_T
\]

The autoregressive model estimates:

\[
P(y_t\mid y_{<t},x)
\]

where:

- \(x\) = instruction/input
- \(y_{<t}\) = previously generated response tokens
- \(y_t\) = current target token

A common training objective is:

\[
\boxed{
\mathcal{L}
=
-\sum_{t=1}^{T}
\log P(y_t\mid y_{<t},x)
}
\]

This is the next-token prediction principle used during supervised training.

---

# 15. Simple Example of the Objective

Suppose:

```text
Instruction:
What is LoRA?

Target:
LoRA is efficient.
```

The model learns:

```text
Instruction + previous context
          ↓
        LoRA

Instruction + LoRA
          ↓
         is

Instruction + LoRA is
          ↓
       efficient
```

At each position, the model predicts the correct next token.

---

# 16. Teacher Forcing Connection

This connects directly to the Transformer decoder concepts.

During training, the known target sequence can be shifted:

```text
Decoder Input:

<BOS> LoRA is

Target:

LoRA is efficient <EOS>
```

The model learns:

```text
<BOS>       → LoRA
LoRA        → is
LoRA is     → efficient
efficient   → <EOS>
```

Because of **causal masking**, the decoder cannot see future target tokens.

---

# 17. Why Causal Masking Still Matters

The decoder uses masked self-attention so it cannot look into the future.

For example:

```text
Tokens:

LoRA | is | efficient
  ↓     ↓       ↓
  ✓     ✓       ✓
```

When predicting `is`, the model can use:

```text
LoRA
```

but cannot use:

```text
efficient
```

This prevents future-token leakage.

---

# 18. IFT Does Not Remove Transformer Training Mechanics

The Transformer concepts you already learned still apply:

```text
Tokenization
     ↓
Embeddings
     ↓
Positional information
     ↓
Transformer
     ↓
Masked Self-Attention
     ↓
Feed Forward
     ↓
Logits
     ↓
Loss
     ↓
Backpropagation
     ↓
Parameter Update
```

IFT changes the **training data and desired behavior**, not the fundamental Transformer mechanism.

---

# 19. IFT and Chat Models

Modern chat models commonly use conversations:

```text
System:
You are a helpful AI tutor.

User:
Explain LoRA.

Assistant:
LoRA is...
```

The model is trained on many conversations like this.

It learns patterns associated with:

```text
system
user
assistant
```

This is why **chat templates** are important.

---

# 20. Why Chat Templates Matter

A model may expect a particular conversation format.

Conceptually:

```text
<|system|>
You are a helpful assistant.

<|user|>
Explain LoRA.

<|assistant|>
LoRA is...
```

Another model may use different special tokens.

Therefore, you should generally use the tokenizer/model's expected chat template rather than inventing an arbitrary format.

---

# 21. Chat Template Pipeline

```text
Messages
   ↓
Chat Template
   ↓
Formatted Token Sequence
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Model
```

Example:

```python
messages = [
    {
        "role": "user",
        "content": "What is LoRA?"
    },
    {
        "role": "assistant",
        "content": "LoRA is a parameter-efficient fine-tuning method."
    }
]
```

A tokenizer can apply the model's chat template:

```python
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False
)
```

The exact arguments and behavior depend on the model/tokenizer.

---

# 22. IFT and SFT

You will frequently encounter:

```text
IFT
SFT
Instruction Tuning
Supervised Fine-Tuning
```

These terms overlap considerably in practical LLM discussions.

A useful conceptual distinction is:

### IFT

Focuses on:

> Teaching the model to follow instructions.

### SFT

Focuses on:

> Training the model using supervised input/output examples.

Therefore:

```text
Instruction Dataset
        ↓
Supervised Fine-Tuning
        ↓
Instruction-Following Model
```

Instruction tuning is commonly implemented through supervised fine-tuning.

---

# 23. IFT Is Not the Same as Preference Training

This distinction becomes important later.

### IFT / SFT

You provide:

```text
Input
+
Desired Response
```

Example:

```text
Question:
Explain LoRA.

Answer:
LoRA is...
```

### Preference Training

You may provide:

```text
Input
+
Response A
+
Response B
+
Preference
```

For example:

```text
Question:
Explain LoRA.

Response A:
LoRA is a parameter-efficient method...

Response B:
LoRA is a database...

Preferred:
A
```

This is a different training setup.

---

# 24. Evolution of LLM Training

A high-level picture:

```text
                    Pretraining
                         ↓
                  Base Language Model
                         ↓
                  Instruction Tuning
                         ↓
                 Instruction-Following Model
                         ↓
                 Preference / Alignment
                         ↓
                  More Helpful Model
```

Later techniques include:

```text
DPO
RLHF
RFT
GRPO
```

These belong to later stages of the training/alignment pipeline.

---

# 25. IFT + LoRA + Custom Dataset

Suppose you have:

```text
100,000 instruction-response examples
```

You want to adapt a pretrained LLM.

A practical pipeline can be:

```text
Custom Dataset
       ↓
Clean Dataset
       ↓
Instruction/Response Format
       ↓
Chat Template
       ↓
Tokenizer
       ↓
SFT / Instruction Training
       ↓
LoRA
       ↓
Train A and B
       ↓
LoRA Adapter
```

The base model can remain frozen while LoRA parameters are trained.

---

# 26. What Is Actually Being Learned?

Suppose:

\[
W'=W+\frac{\alpha}{r}BA
\]

During LoRA-based IFT:

```text
W
↓
Frozen

A
↓
Updated

B
↓
Updated
```

The training objective encourages \(A\) and \(B\) to produce a useful adaptation for the instruction dataset.

So the model learns:

\[
\boxed{
\text{Instruction-following behavior}
}
\]

through the trainable adapter parameters.

---

# 27. Example: Domain-Specific Instruction Tuning

Suppose you want a Python tutor.

Dataset:

```text
Instruction:
Explain Python list comprehension.

Response:
A list comprehension provides a concise way to create
a list from an iterable...
```

Another:

```text
Instruction:
Write a Python function that checks whether a number is prime.

Response:
def is_prime(n):
    ...
```

Another:

```text
Instruction:
Explain the difference between a list and tuple.

Response:
A list is mutable while a tuple is immutable...
```

After sufficient training, the model becomes better at these kinds of requests.

---

# 28. Instruction Diversity vs Domain Specialization

These are different dimensions.

## Instruction Diversity

Different tasks:

```text
Explain
Compare
Summarize
Generate
Translate
Classify
Code
```

## Domain Specialization

Different subject matter:

```text
Python
Machine Learning
Medicine
Law
Finance
Engineering
```

A dataset can combine both:

```text
Explain → Python
Write code → Python
Compare → Python
Debug → Python
```

This teaches:

```text
Task behavior
+
Domain knowledge
```

---

# 29. What IFT Does NOT Guarantee

Instruction fine-tuning does not automatically guarantee:

- Perfect factual accuracy
- Perfect reasoning
- No hallucinations
- Complete knowledge of a domain
- Human preferences
- Safe behavior in every situation

Quality depends on:

```text
Base model
+
Dataset quality
+
Dataset diversity
+
Training setup
+
Evaluation
```

---

# 30. Evaluation After IFT

After training, evaluate the model.

For example:

```text
Test Instruction
      ↓
Fine-Tuned Model
      ↓
Generated Response
      ↓
Evaluation
```

Possible evaluation dimensions include:

- Task accuracy
- Instruction following
- Helpfulness
- Formatting
- Domain performance
- Factuality
- Safety

Evaluation should use held-out data or appropriate external benchmarks rather than simply checking training examples.

---

# 31. Common Mistakes

## Mistake 1: Thinking IFT creates knowledge from nothing

The pretrained model already contains substantial knowledge.

IFT primarily adapts its behavior and capabilities toward instruction following using the provided examples.

---

## Mistake 2: Thinking LoRA and IFT are the same thing

They are not.

```text
IFT
↓
Training/data objective and desired behavior

LoRA
↓
Parameter-efficient adaptation method
```

They can be combined.

---

## Mistake 3: Using poor instruction data

Bad examples can teach undesirable behavior.

---

## Mistake 4: Ignoring the model's chat template

A model may expect a specific conversation format.

---

## Mistake 5: Training and evaluating on the same examples

This can give a misleading impression of model quality.

---

# 32. Practical Mental Model

Remember:

```text
PRETRAINING
    ↓
Learn general language patterns
    ↓
BASE MODEL
    ↓
IFT / SFT
    ↓
Learn to follow instructions
    ↓
INSTRUCTION-FOLLOWING MODEL
    ↓
Preference / Alignment Training
    ↓
More aligned model
```

The parameter update method can be:

```text
Full Fine-Tuning
       OR
LoRA
       OR
QLoRA
       OR
other PEFT methods
```

These are different axes:

```text
             WHAT DATA / OBJECTIVE?
                       │
                 IFT / SFT
                       │
                       ↓
             HOW TO UPDATE MODEL?
                       │
            ┌──────────┼──────────┐
            ↓          ↓          ↓
          Full       LoRA       QLoRA
```

This distinction is one of the most important concepts in the fine-tuning roadmap.

---

# 33. Questions & Answers

## Q1. What is IFT?

Instruction Fine-Tuning teaches a pretrained model to better follow natural-language instructions using instruction/response examples.

---

## Q2. Does IFT create a new Transformer architecture?

No.

The underlying Transformer architecture can remain the same.

IFT changes the model's learned parameters/behavior through additional training.

---

## Q3. What type of dataset is commonly used?

Instruction-response or conversational datasets.

---

## Q4. What is the basic relationship?

\[
\boxed{
\text{Instruction}\rightarrow\text{Desired Response}
}
\]

---

## Q5. Is IFT the same as LoRA?

No.

IFT describes the training/data objective.

LoRA describes a parameter-efficient method for adapting model weights.

---

## Q6. Can IFT and LoRA be combined?

Yes.

A common setup is:

\[
\boxed{
\text{Instruction Dataset}
+
\text{SFT/IFT}
+
\text{LoRA}
}
\]

---

## Q7. What happens to the base model in LoRA-based IFT?

The pretrained base weights are typically frozen while the LoRA adapter parameters are trained.

---

## Q8. What does the model learn during IFT?

It learns patterns for responding appropriately to instructions and desired task formats.

---

## Q9. Does IFT guarantee factual correctness?

No.

The quality depends strongly on the pretrained model, dataset, training setup, and evaluation.

---

## Q10. Why are chat templates important?

They convert structured messages such as `system`, `user`, and `assistant` into the exact token sequence expected by a conversational model.

---

# 34. Self-Test

Try answering these without looking back:

1. What is IFT?
2. What is the difference between pretraining and IFT?
3. Why does a pretrained model need instruction tuning?
4. What type of data is used for IFT?
5. What does the model learn from instruction-response examples?
6. What is the difference between IFT and LoRA?
7. Can LoRA and IFT be combined?
8. What happens to \(W\), \(A\), and \(B\) during LoRA-based IFT?
9. Why is instruction diversity important?
10. What is the role of a chat template?
11. Why is causal masking still needed during autoregressive training?
12. What is the difference between instruction training and preference training?
13. Why should the test set not be used for ordinary training?
14. Does IFT guarantee that the model will never hallucinate?
15. What are the major stages from pretraining to an instruction-following model?

---

# 35. Final Mental Model

Connect the previous lessons:

```text
                 PRETRAINING
                     ↓
              Base Language Model
                     ↓
              Custom Dataset
                     ↓
          Instruction / Response
                     ↓
               Chat Template
                     ↓
                Tokenization
                     ↓
                 IFT / SFT
                     ↓
              Parameter Update
                     ↓
              ┌──────┴──────┐
              ↓             ↓
        Full Fine-Tuning   LoRA
                            ↓
                          QLoRA
                     ↓
          Instruction-Following Model
                     ↓
          Preference / Alignment
                     ↓
             DPO / RLHF / RFT
                     ↓
                    GRPO
```

The most important distinction is:

\[
\boxed{
\text{IFT/SFT}=\text{what supervised behavior we train}
}
\]

while:

\[
\boxed{
\text{LoRA/QLoRA}=\text{how we efficiently update the model}
}
\]

---

# 36. Lesson 15 Summary

You learned:

- What Instruction Fine-Tuning (IFT) is
- Where IFT fits after pretraining
- Pretraining vs instruction tuning
- Why a base model needs instruction tuning
- Instruction-response datasets
- Conversational datasets
- Instruction diversity
- Instruction-following behavior
- The autoregressive training objective
- Teacher forcing connection
- Causal masking connection
- Chat templates
- IFT vs SFT
- IFT vs preference training
- IFT + LoRA
- IFT + QLoRA
- Domain-specific instruction tuning
- Evaluation after IFT
- Common IFT mistakes

The central idea is:

\[
\boxed{
\text{Instruction}
\rightarrow
\text{Desired Response}
}
\]

and:

\[
\boxed{
\text{IFT/SFT}
=
\text{training the model to learn from supervised examples}
}
\]

while:

\[
\boxed{
\text{LoRA/QLoRA}
=
\text{parameter-efficient ways to update the model}
}
\]

---

# 37. Next Lesson

## Lesson 16 — Supervised Fine-Tuning (SFT)

Next we will go deeper into **Supervised Fine-Tuning** and connect the entire pipeline:

```text
Dataset
   ↓
Instruction / Response
   ↓
Chat Template
   ↓
Tokenization
   ↓
Labels
   ↓
Loss Masking
   ↓
Cross-Entropy
   ↓
Backpropagation
   ↓
LoRA / QLoRA
```

We will also implement a small **SFT pipeline with PyTorch / Hugging Face** so the concepts become practical.
