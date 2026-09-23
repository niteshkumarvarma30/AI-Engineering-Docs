# LLM Evaluation & Benchmarks

## Complete LLM Evaluation Notes

> **Core idea:** There is no single number that can tell you whether an LLM is "good." You must evaluate the model according to the capability and task that actually matter.

---

## 1. What Is LLM Evaluation?

LLM evaluation is the process of measuring how well a language model performs on a particular task.

An LLM can be evaluated for:

- Knowledge
- Reasoning
- Coding
- Factuality
- Instruction following
- Chat quality
- Safety
- Tool use
- Agentic task completion
- Retrieval-augmented generation (RAG)
- Confidence and calibration

These are different capabilities.

```text
                    LLM Quality
                         |
       +-----------------+------------------+
       |                 |                  |
   Knowledge         Reasoning           Coding
       |                 |                  |
    MMLU/GPQA        Math benchmarks    SWE-bench
       |                 |                  |
       +-----------------+------------------+
                         |
                  Other capabilities
                         |
       +---------+-------+-------+---------+
       |         |               |         |
   Factuality  Safety        Chat       Agents
```

Therefore:

> **No single benchmark can completely describe an LLM's capabilities.**

---

## 2. What Is a Benchmark?

A benchmark is a standardized test used to evaluate and compare models.

Think about a school examination:

```text
100 questions
      |
      v
   Student
      |
      v
80 correct answers
      |
      v
    80%
```

An LLM benchmark works similarly:

```text
Benchmark
    |
    v
Fixed evaluation questions
    |
    v
LLM
    |
    v
Compare with expected answers
    |
    v
Score
```

Examples:

| Benchmark | Main purpose |
|---|---|
| MMLU | Broad knowledge |
| GPQA | Difficult scientific reasoning |
| HumanEval | Function-level coding |
| SWE-bench | Real software engineering |
| GSM8K | Mathematical reasoning |
| SimpleQA | Factuality |
| Arena | Human preference |
| LiveBench | More contamination-resistant evaluation |

---

## 3. Why There Is No Single "Best" LLM

Different models can be strong at different tasks.

Consider athletes:

```text
Usain Bolt
    |
    +--> 100-meter sprint

Michael Phelps
    |
    +--> Swimming

Serena Williams
    |
    +--> Tennis
```

Asking:

> "Who is the best athlete?"

is incomplete.

You first need to ask:

> "Best at what?"

The same principle applies to LLMs.

For example:

```text
Model A
Knowledge: 92%
Coding:    70%

Model B
Knowledge: 88%
Coding:    90%
```

If you are building a coding assistant, the knowledge score alone is not enough to evaluate the model.

Therefore:

> **The appropriate model depends on the task and evaluation criteria.**

---

## 4. Benchmark Saturation

### What Is Saturation?

A benchmark becomes saturated when many advanced models achieve very high scores, making the benchmark less useful for distinguishing between them.

Imagine a benchmark initially produces:

```text
Model A -> 55%
Model B -> 65%
Model C -> 72%
Model D -> 80%
```

The benchmark clearly separates the models.

After several years of model improvements:

```text
Model A -> 93%
Model B -> 94%
Model C -> 96%
Model D -> 97%
```

Now the benchmark has much less discriminative power.

This is called:

> **Benchmark saturation**

Conceptually:

```text
Performance
100% |-------------------- Ceiling
 90% |        *  *  *  *
 80% |     *
 70% |   *
 60% | *
     +--------------------------> Time
```

Once most frontier models are close to the ceiling, a small score difference may not tell you much about real-world performance.

---

## 5. Why Saturation Matters

Suppose:

```text
Model A -> 96%
Model B -> 97%
```

It is tempting to conclude that Model B is significantly better.

But the benchmark may simply be too easy for both models.

```text
Benchmark capability
        |
        v
   Almost solved
        |
        v
Scores cluster near 100%
        |
        v
Poor ability to distinguish models
```

Therefore:

> **A high benchmark score is not automatically evidence that the benchmark is still useful for comparing frontier models.**

---

# 6. MMLU

MMLU stands for:

> **Massive Multitask Language Understanding**

It evaluates knowledge and reasoning across many academic subjects.

Examples include:

- Mathematics
- Physics
- Chemistry
- Biology
- History
- Economics
- Law
- Computer science

MMLU is useful for measuring broad academic knowledge, but a high MMLU score does not automatically mean that a model is good at every other task.

For example:

```text
MMLU
  |
  v
Broad knowledge
```

It does not directly measure:

```text
MMLU
  X
  |
  +--> Full repository software engineering
  +--> Long-running agent tasks
  +--> Your company's RAG workflow
```

---

# 7. HumanEval

HumanEval is a coding benchmark focused on function-level programming problems.

A simplified example:

```python
def reverse_string(text):
    # Complete the function
    pass
```

The model generates code.

Automated tests then determine whether the generated function works.

The basic pipeline is:

```text
Coding problem
      |
      v
     LLM
      |
      v
Generated function
      |
      v
Automated tests
      |
      v
Pass / Fail
```

HumanEval is useful for measuring isolated code-generation ability.

However, real software engineering is usually more complicated.

---

# 8. HumanEval vs Real Software Engineering

A HumanEval-style problem looks like:

```text
Write one function
        |
        v
Run tests
        |
        v
Pass / Fail
```

Real software engineering often looks like:

```text
GitHub repository
        |
        v
Understand project structure
        |
        v
Read many files
        |
        v
Find the relevant code
        |
        v
Understand the bug
        |
        v
Modify one or more files
        |
        v
Run tests
        |
        v
Debug failures
        |
        v
Submit a working patch
```

Therefore:

```text
HumanEval
    |
    +--> Isolated function-level coding

SWE-bench
    |
    +--> Repository-level software engineering
```

A model can perform extremely well on isolated coding problems while still struggling with large real-world repositories.

---

# 9. SWE-bench

SWE-bench evaluates software-engineering tasks using real GitHub issues.

The general workflow is:

```text
Real GitHub issue
        |
        v
Repository
        |
        v
AI coding agent
        |
        +--> Understand repository
        |
        +--> Locate relevant files
        |
        +--> Modify code
        |
        +--> Run tests
        |
        +--> Debug
        |
        v
Working patch
```

This makes SWE-bench much closer to real software engineering than a single-function benchmark.

---

# 10. pass@k

`pass@k` measures whether at least one correct solution is generated when the model is allowed to produce `k` attempts.

## pass@1

The model generates one solution:

```text
Problem
   |
   v
Attempt 1
   |
   v
PASS
```

This is:

```text
pass@1
```

## pass@5

The model generates five solutions:

```text
Attempt 1 -> FAIL
Attempt 2 -> FAIL
Attempt 3 -> PASS
Attempt 4 -> FAIL
Attempt 5 -> FAIL
```

At least one solution passed.

Therefore:

```text
pass@5 = success
```

Conceptually:

```text
k = 1
|
+--> One attempt

k = 5
|
+--> Five attempts
     |
     +--> If at least one passes -> pass@5
```

---

# 11. Why pass@1 Matters

For an interactive coding assistant, users generally want the first generated solution to work.

For example:

```text
User
  |
  v
"Fix this bug."
  |
  v
AI
  |
  v
Correct patch
```

If the model only succeeds after generating ten alternatives, that may be less useful in an interactive workflow.

Therefore:

> **pass@1 is especially important when first-attempt correctness matters.**

---

# 12. Perplexity

Perplexity is one of the fundamental metrics for language models.

An LLM predicts the probability of the next token.

Consider:

```text
The cat sat on the ___
```

Suppose the model predicts:

```text
mat     -> 0.80
floor   -> 0.10
chair   -> 0.05
road    -> 0.01
other   -> 0.04
```

If the actual next token is:

```text
mat
```

the model assigned a high probability to the correct token.

That is good.

If instead:

```text
P(mat) = 0.01
```

the model was very surprised by the actual token.

That is bad.

---

## Perplexity Formula

The standard formula is:

\[
PPL = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(x_i)\right)
\]

Where:

- `N` = number of tokens
- `x_i` = actual token at position `i`
- `P(x_i)` = probability assigned to the actual token

The intuition is:

```text
Lower perplexity
       |
       v
Model assigns higher probability
to the actual text
```

---

# 13. What Perplexity Does Not Tell You

Perplexity measures language-modeling quality.

It does not directly measure overall usefulness.

For example, a model might be very good at predicting common tokens:

```text
the the the the the...
```

but still be useless as a chatbot.

Therefore:

```text
Perplexity
    |
    +--> How well does the model predict text?
```

It does not mean:

```text
Perplexity
    |
    +--> How useful is the model for every application?
```

---

# 14. Arena and Elo

Sometimes we want to know:

> **Which model do humans prefer?**

A human-preference evaluation can compare two anonymous model responses.

Example:

```text
Prompt:
Explain quantum computing to a beginner.
```

Two models produce:

```text
Model A -> Answer A
Model B -> Answer B
```

The human evaluator does not know which model produced which response.

They choose:

```text
A
B
Tie
```

Repeated comparisons can be used to calculate an Elo-style ranking.

---

# 15. What Is Elo?

Elo is a rating system originally popularized for chess.

Suppose:

```text
Player A -> 1800
Player B -> 1600
```

If A defeats B:

```text
A rating increases
B rating decreases
```

If B defeats A:

```text
B rating increases
A rating decreases
```

The amount of rating change depends partly on the expected outcome.

An Arena-style model comparison works conceptually like this:

```text
Model A
   vs
Model B
   |
   v
Human vote
   |
   v
Elo update
```

---

# 16. Why Category-Specific Leaderboards Matter

Suppose a model is ranked highly overall.

That does not mean it is the best model for every task.

For example:

```text
Overall leaderboard
-------------------
Model A -> #1
Model B -> #2
Model C -> #3
```

But a coding leaderboard could look different:

```text
Coding leaderboard
------------------
Model B -> #1
Model A -> #4
Model C -> #5
```

And a creative-writing leaderboard could be different again.

Therefore:

> **Always evaluate the category that matches your use case.**

---

# 17. Contamination

Benchmark contamination occurs when evaluation data, or information very closely related to it, has appeared in the model's training data.

Think about a student taking an exam.

### Fair situation

```text
Student
   |
   v
Sees exam for first time
   |
   v
Solves questions
```

### Contaminated situation

```text
Student
   |
   v
Saw the exam before
   |
   v
Memorized answers
   |
   v
Takes exam
   |
   v
100%
```

A high score does not necessarily demonstrate the same level of genuine generalization.

The same concern applies to LLM benchmarks.

---

# 18. Rephrasing as a Contamination Check

One possible diagnostic is to rephrase benchmark questions while preserving their meaning.

Original:

```text
What is the capital of Australia?
```

Rephrased:

```text
Which city serves as Australia's national capital?
```

If performance changes dramatically, that can be evidence worth investigating.

However:

> **A performance drop after rephrasing is not automatic proof of contamination.**

The model may simply be sensitive to wording.

Therefore, contamination requires careful investigation rather than a single diagnostic.

---

# 19. Dynamic Benchmarks

A static benchmark can become less useful over time because:

- Models may encounter similar questions during training.
- Researchers may optimize systems against the benchmark.
- Models may eventually solve most of the questions.

Dynamic benchmarks attempt to reduce these problems by regularly introducing new evaluation data.

Conceptually:

```text
Static benchmark

Old questions
     |
     v
Model training
     |
     v
Memorization / saturation
     |
     v
Less useful benchmark
```

Compared with:

```text
Dynamic benchmark

New questions
     |
     v
Evaluation
     |
     v
New questions
     |
     v
Evaluation
     |
     v
More difficult to memorize the entire test
```

Examples include dynamically refreshed evaluation approaches such as LiveBench and LiveCodeBench.

---

# 20. Benchmark Gaming

Suppose a model developer knows exactly what benchmark is being used.

They can optimize heavily for the benchmark.

Then:

```text
Benchmark score
       |
       v
      High
```

but:

```text
Real-world performance
       |
       v
May not improve proportionally
```

This is often described as benchmark gaming or benchmark-specific optimization.

The important question is:

> **Does performance generalize beyond the benchmark?**

---

# 21. LLM-as-a-Judge

Evaluating thousands of LLM responses manually is expensive.

One solution is to use another LLM as an evaluator.

The architecture looks like:

```text
User question
      |
      v
Model being tested
      |
      v
Generated answer
      |
      v
Judge LLM
      |
      v
Score / preference / feedback
```

For example:

```text
Question:
Explain gradient descent.

Answer:
[Model response]

Judge:
Score from 1 to 5.
```

This is called:

> **LLM-as-a-judge**

It can make large-scale evaluation much cheaper.

---

# 22. Problems With LLM Judges

LLM judges can have systematic biases.

## 22.1 Position Bias

Suppose the judge compares:

```text
A -> Answer 1
B -> Answer 2
```

The judge selects A.

Now swap them:

```text
A -> Answer 2
B -> Answer 1
```

If the judge again selects A, the result may indicate position bias.

A useful strategy is:

```text
Test 1:
A vs B

Test 2:
B vs A
```

Then compare the outcomes.

---

## 22.2 Length Bias

A judge may sometimes prefer a longer answer simply because it contains more information.

But:

```text
Longer
  !=
Better
```

A concise and correct answer can be better than a very long answer.

Therefore evaluation rubrics should explicitly define what matters.

---

## 22.3 Self-Preference Bias

A judge model may potentially favor answers that resemble its own preferred style or outputs.

Using multiple diverse judge models can reduce dependence on one judge.

---

# 23. Calibrating an LLM Judge

Suppose you want to use an LLM judge in production.

First create examples evaluated by humans:

```text
100 examples
      |
      v
Human evaluation
      |
      v
Reference judgments
```

Then ask the LLM judge to evaluate the same examples:

```text
100 examples
      |
      v
LLM judge
      |
      v
Judge decisions
```

Compare the results.

For example:

```text
Human judgment
      vs
LLM judgment
```

This tells you whether your judge is sufficiently aligned with your human evaluation criteria.

---

# 24. Custom Evaluation

This is one of the most important concepts for an AI engineer.

Suppose you build a RAG chatbot.

You should not rely only on:

```text
MMLU = 90%
```

Instead, collect real examples from your intended application.

For example:

```text
100-500 real queries
```

For each query, define appropriate evaluation criteria.

Example:

```text
Question
Expected answer
Required information
Relevant documents
Citation requirements
Correctness criteria
```

Then run your model against this dataset.

---

# 25. Example: Evaluating a RAG System

Suppose your RAG architecture is:

```text
User
  |
  v
Retriever
  |
  v
Top-k documents
  |
  v
LLM
  |
  v
Final answer
```

Your evaluation can measure several components.

### Retrieval

Did the system retrieve the correct document?

```text
Retrieval accuracy
```

### Answer correctness

Did the model provide the correct answer?

```text
Answer correctness
```

### Faithfulness

Is the answer supported by the retrieved context?

```text
Retrieved context
       |
       v
Generated answer
       |
       v
Is every important claim supported?
```

### Citation correctness

Are the citations actually supporting the claims?

### Latency

How long does the system take?

### Cost

How much does each request cost?

Therefore, your evaluation becomes:

```text
             RAG Evaluation
                   |
       +-----------+-----------+
       |           |           |
  Retrieval    Generation   System
       |           |           |
  Accuracy     Correctness  Latency
  Recall       Faithfulness Cost
               Citations
```

---

# 26. Held-Out Evaluation Set

This concept comes directly from traditional machine learning.

Suppose you have 500 examples.

You could divide them into:

```text
500 examples
     |
     +--------------------+
     |                    |
     v                    v
Development set      Held-out test set
     |                    |
     v                    v
Tune system           Final evaluation
```

The held-out test set should remain unseen during tuning.

Why?

Because if you repeatedly tune your system against the final test set, you can overfit to it.

This is similar to overfitting in machine learning.

---

# 27. AI Evaluation as Regression Testing

Suppose your current model is:

```text
Model v1
    |
    v
Custom evaluation
    |
    v
85%
```

You replace it with Model v2.

```text
Model v2
    |
    v
Same custom evaluation
    |
    v
79%
```

Even if Model v2 has better public benchmark scores, it may be worse for your application.

Therefore:

```text
Model update
     |
     v
Run custom evaluation
     |
     v
Compare with previous version
     |
     v
Accept or reject update
```

This is essentially:

> **Regression testing for AI systems.**

---

# 28. Evaluation Metrics by Task

| Task | Useful evaluation |
|---|---|
| Language modeling | Perplexity |
| Broad knowledge | MMLU |
| Difficult scientific reasoning | GPQA |
| Isolated code generation | HumanEval |
| Real software engineering | SWE-bench |
| First-attempt coding | pass@1 |
| Multiple-attempt coding | pass@k |
| Human chatbot preference | Arena / Elo |
| Factuality | Factuality-specific benchmarks |
| Safety | Safety benchmarks |
| Your production application | Custom held-out evaluation |

---

# 29. The Evaluation Stack

A good evaluation process can be represented as:

```text
                    LLM Evaluation
                          |
                          v
                 1. Define the task
                          |
                          v
                 2. Choose the metric
                          |
             +------------+------------+
             |            |            |
             v            v            v
         Knowledge      Coding       Chat
          MMLU/GPQA   SWE-bench     Arena
             |            |            |
             +------------+------------+
                          |
                          v
                 3. Check limitations
                          |
              +-----------+-----------+
              |           |           |
              v           v           v
         Saturation  Contamination  Gaming
              |           |           |
              +-----------+-----------+
                          |
                          v
                 4. Build YOUR eval
                          |
                          v
                  Real examples
                     100-500+
                          |
                          v
                 5. Held-out testing
                          |
                          v
                 6. Repeat after
                    model updates
```

---

# 30. Accuracy vs Calibration

Calibration is an important concept when models return probabilities.

Suppose a model says:

```text
Spam probability = 90%
```

Calibration asks:

> When the model gives predictions around 90%, are those predictions actually correct about 90% of the time?

For example:

```text
100 predictions
Model confidence = 90%

Actual correct predictions:
90
```

This is approximately well calibrated.

But:

```text
Model confidence = 90%
Actual correct predictions = 60
```

means the model is overconfident.

Therefore:

```text
Accuracy
    |
    +--> How often is the model correct?

Calibration
    |
    +--> Does its confidence match reality?
```

These are different concepts.

---

# 31. Why Calibration Matters for AI Agents

Suppose an AI agent uses a confidence threshold:

```text
Confidence >= 0.90
        |
        v
Automatically act

Confidence < 0.90
        |
        v
Ask a human
```

This only works reliably if the confidence score is meaningful.

If:

```text
Model says 0.95
```

but its comparable predictions are correct only 60% of the time, the automation threshold is unsafe.

Therefore:

```text
Reliable automation
       |
       +--> Good accuracy
       |
       +--> Useful calibration
       |
       +--> Appropriate threshold
```

---

# 32. The Most Important Practical Lesson

Do not evaluate a model like this:

```text
Model A -> MMLU 92%
Model B -> MMLU 90%

Therefore:

Model A is better.
```

Instead:

```text
What am I building?
        |
        v
What capability matters?
        |
        v
Which metric measures that capability?
        |
        v
Is the benchmark saturated?
        |
        v
Could contamination affect the result?
        |
        v
Could benchmark-specific optimization affect it?
        |
        v
Build an evaluation set from real examples
        |
        v
Test the models
        |
        v
Choose based on the requirements of the application
```

---

# 33. Final Mental Model

Remember this four-step framework:

```text
             LLM Evaluation
                  |
                  v
        +--------------------+
        | 1. Match the task  |
        +--------------------+
                  |
                  v
        +--------------------+
        | 2. Match the metric|
        +--------------------+
                  |
                  v
        +--------------------+
        | 3. Distrust scores |
        | saturation          |
        | contamination       |
        | gaming              |
        | judge bias          |
        +--------------------+
                  |
                  v
        +--------------------+
        | 4. Build YOUR eval |
        | real examples      |
        | held-out set       |
        | repeat on updates  |
        +--------------------+
```

---

# 34. Key Takeaways

### 1. Benchmark

A standardized test for measuring a specific capability.

### 2. Saturation

A benchmark becomes less useful when most advanced models approach its maximum score.

### 3. Contamination

The model may have encountered benchmark data or very similar material during training.

### 4. Benchmark gaming

A system can be optimized specifically for a benchmark without necessarily improving general real-world performance.

### 5. Perplexity

Measures how well a language model predicts tokens.

```text
Lower perplexity
        |
        v
Better token prediction
```

It does **not** directly measure overall usefulness.

### 6. pass@k

Measures whether at least one of `k` generated solutions succeeds.

```text
pass@1 -> one attempt
pass@5 -> five attempts
pass@10 -> ten attempts
```

### 7. Arena Elo

Uses human comparisons to estimate relative model preference.

### 8. LLM-as-a-judge

Uses one LLM to evaluate another model's output.

It is useful but can have biases.

### 9. Custom evaluation

The most important evaluation for your application is a test set based on your actual task.

### 10. Held-out evaluation

Keep a final test set separate from tuning so that you do not overfit your system to the evaluation.

---

# 35. One-Sentence Summary

> **LLM evaluation is not about finding one model with the highest benchmark score; it is about matching the evaluation method to your task, checking whether the benchmark is trustworthy, and validating the model on representative real-world examples from your own application.**

---

## Quick Revision

```text
Benchmark
    -> Standardized test

Saturation
    -> Test becomes too easy

Contamination
    -> Model may have seen the test

Gaming
    -> System optimized for the benchmark

Perplexity
    -> How well the model predicts tokens

pass@k
    -> At least one success among k attempts

Arena Elo
    -> Human preference ranking

LLM-as-judge
    -> LLM evaluates another LLM

Calibration
    -> Confidence matches observed correctness

Custom eval
    -> Test on your actual use case

Held-out set
    -> Final unseen evaluation data

Best practice
    -> Match metric to task + build your own evaluation
```

---

# Conclusion

The most important shift in thinking is:

```text
Old mindset:

"What is the highest-scoring LLM?"
```

versus:

```text
AI engineering mindset:

"What capability do I need,
how should I measure it,
and does the model work on my real data?"
```

A public benchmark can tell you something about general model capability.

Your own carefully designed, held-out evaluation tells you whether the system is suitable for the specific problem you are building.
