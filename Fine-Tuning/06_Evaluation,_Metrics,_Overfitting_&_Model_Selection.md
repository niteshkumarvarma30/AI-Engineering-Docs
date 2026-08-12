# Fine-Tuning — Lesson 6: Evaluation, Metrics, Overfitting & Model Selection

> **Goal:** Learn how to determine whether a fine-tuned model is actually good, how to evaluate it on unseen data, how to detect overfitting, and how to select the best model.

---

# 1. Why Evaluation Is Necessary

Suppose we fine-tune a sentiment classifier.

After training:

```text
Training Accuracy = 98%
```

It is tempting to say:

> "The model is 98% accurate, so it is excellent."

But this conclusion may be wrong.

Suppose:

```text
Training Accuracy = 98%
Validation Accuracy = 68%
```

The model performs extremely well on training examples but poorly on unseen examples.

This is a classic sign of:

\[
\boxed{\text{Overfitting}}
\]

The real goal of machine learning is **generalization**, not memorization.

---

# 2. Training Set vs Validation Set vs Test Set

A dataset is commonly divided into:

```text
Complete Dataset
       ↓
 ┌─────┼─────┐
 ↓     ↓     ↓
Train Val   Test
```

For example:

```text
70% → Training
15% → Validation
15% → Test
```

The exact percentages depend on the problem and dataset size.

## Training

Used to update model parameters.

\[
\boxed{\text{Training data changes the model}}
\]

## Validation

Used during development for:

- Hyperparameter selection
- Model selection
- Overfitting detection
- Checkpoint selection

\[
\boxed{\text{Validation data guides model selection}}
\]

## Test

Kept aside for final evaluation.

\[
\boxed{\text{Test data estimates final generalization}}
\]

Easy memory:

```text
Training   → Learn
Validation → Choose
Test       → Report
```

---

# 3. Training Dataset

During training:

```text
Training example
      ↓
Model
      ↓
Prediction
      ↓
Loss
      ↓
Backpropagation
      ↓
Gradient
      ↓
Optimizer
      ↓
Updated parameters
```

Therefore:

\[
\boxed{\text{Training data updates parameters}}
\]

---

# 4. Validation Dataset

Validation data should not directly update the model parameters.

We use it to answer:

- Which model is better?
- Should we train longer?
- Is the model overfitting?
- Which hyperparameters are better?
- Which checkpoint should we keep?

Therefore:

\[
\boxed{\text{Validation = model development}}
\]

---

# 5. Test Dataset

After model and hyperparameters have been selected:

```text
Final Model
     ↓
Test Dataset
     ↓
Final Performance
```

The test set should ideally remain unseen during model selection.

---

# 6. What Is a Metric?

A metric is a numerical measure used to evaluate model performance.

For classification, common metrics are:

```text
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
```

---

# 7. Confusion Matrix

For binary classification:

```text
                    Actual
                 Positive  Negative
               ┌─────────┬─────────┐
Predicted Pos. │   TP    │   FP    │
               ├─────────┼─────────┤
Predicted Neg. │   FN    │   TN    │
               └─────────┴─────────┘
```

There are four possibilities.

---

# 8. True Positive — TP

The model predicts Positive and the actual label is Positive.

\[
\boxed{TP=\text{Correct Positive Prediction}}
\]

Example:

```text
Actual:    Positive
Predicted: Positive
```

---

# 9. True Negative — TN

The model predicts Negative and the actual label is Negative.

\[
\boxed{TN=\text{Correct Negative Prediction}}
\]

Example:

```text
Actual:    Negative
Predicted: Negative
```

---

# 10. False Positive — FP

The model predicts Positive but the actual label is Negative.

\[
\boxed{FP=\text{False Alarm}}
\]

Example:

```text
Actual:    Negative
Predicted: Positive
```

---

# 11. False Negative — FN

The model predicts Negative but the actual label is Positive.

\[
\boxed{FN=\text{Missed Positive}}
\]

Example:

```text
Actual:    Positive
Predicted: Negative
```

---

# 12. Memory Trick

```text
TRUE  = Prediction was correct
FALSE = Prediction was incorrect

POSITIVE = Model predicted Positive
NEGATIVE = Model predicted Negative
```

Therefore:

```text
TP → Correct Positive
TN → Correct Negative
FP → Incorrect Positive
FN → Incorrect Negative
```

---

# 13. Accuracy

Accuracy answers:

> **Out of all predictions, how many were correct?**

Formula:

\[
Accuracy =
\frac{TP+TN}
{TP+TN+FP+FN}
\]

Example:

```text
TP = 40
TN = 50
FP = 5
FN = 5
```

Then:

\[
Accuracy =
\frac{40+50}
{40+50+5+5}
\]

\[
Accuracy=90\%
\]

Therefore:

\[
\boxed{
Accuracy=\text{Fraction of all predictions that are correct}
}
\]

---

# 14. Why Accuracy Can Be Misleading

Suppose:

```text
1000 examples

990 → Negative
10  → Positive
```

A model that always predicts Negative gets:

```text
990 correct
10 incorrect
```

Therefore:

\[
Accuracy=\frac{990}{1000}=99\%
\]

The accuracy is 99%, but the model completely fails to identify the positive class.

This is the **class imbalance problem**.

Therefore:

\[
\boxed{\text{Accuracy alone can be misleading}}
\]

---

# 15. Precision

Precision answers:

> **Of everything the model predicted as positive, how many were actually positive?**

Formula:

\[
Precision=
\frac{TP}{TP+FP}
\]

Example:

```text
TP = 80
FP = 20
```

Then:

\[
Precision=
\frac{80}{80+20}
=80\%
\]

Memory:

\[
\boxed{
Precision=\text{Trustworthiness of Positive Predictions}
}
\]

High precision means fewer false positives.

---

# 16. Recall

Recall answers:

> **Of all actual positive examples, how many did the model successfully identify?**

Formula:

\[
Recall=
\frac{TP}{TP+FN}
\]

Example:

```text
TP = 80
FN = 20
```

Then:

\[
Recall=
\frac{80}{80+20}
=80\%
\]

Memory:

\[
\boxed{
Recall=\text{Ability to Find Positive Examples}
}
\]

High recall means fewer false negatives.

---

# 17. Precision vs Recall

## Precision

Focuses on:

```text
Predicted Positive
```

Question:

> When I say positive, am I correct?

\[
Precision=\frac{TP}{TP+FP}
\]

## Recall

Focuses on:

```text
Actual Positive
```

Question:

> Did I find the positives?

\[
Recall=\frac{TP}{TP+FN}
\]

Easy memory:

```text
Precision → Predicted Positive → Was it correct?
Recall    → Actual Positive    → Did we find it?
```

---

# 18. F1 Score

F1 is the harmonic mean of precision and recall.

\[
F1=
2
\frac{Precision\times Recall}
{Precision+Recall}
\]

Suppose:

```text
Precision = 0.8
Recall = 0.6
```

Then:

\[
F1=
2\frac{0.8\times0.6}{0.8+0.6}
\]

\[
F1\approx0.686
\]

F1 is useful because it penalizes situations where one of precision or recall is very low.

---

# 19. When Should We Use Which Metric?

## Accuracy

Useful when classes are reasonably balanced.

## Precision

Important when false positives are costly.

Example:

```text
Spam detection
```

## Recall

Important when false negatives are costly.

Example:

```text
Disease screening
```

## F1

Useful when you want a balance between precision and recall.

There is no universally best metric. The appropriate metric depends on the task and the cost of errors.

---

# 20. Training Loss

During training, the model calculates loss.

Example:

```text
Epoch 1 → Loss = 0.80
Epoch 2 → Loss = 0.50
Epoch 3 → Loss = 0.30
```

The model is learning to reduce training loss.

\[
\boxed{
\text{Lower training loss generally means better fit to training data}
}
\]

But lower training loss does **not** automatically mean better generalization.

---

# 21. Validation Loss

Example:

```text
Epoch       Train Loss       Validation Loss

1              0.80              0.82
2              0.50              0.55
3              0.30              0.40
4              0.20              0.45
5              0.10              0.60
```

Training loss keeps decreasing:

```text
0.80 → 0.50 → 0.30 → 0.20 → 0.10
```

But validation loss begins increasing after Epoch 3.

This is a classic overfitting pattern.

---

# 22. Overfitting

Overfitting occurs when the model learns the training data too specifically and performs worse on unseen data.

Typical pattern:

```text
Training performance ↑
Validation performance ↓
```

or:

```text
Training Loss ↓
Validation Loss ↑
```

Therefore:

\[
\boxed{
\text{Overfitting}
=
\text{Good training performance + poor generalization}
}
\]

---

# 23. Underfitting

Underfitting occurs when the model has not learned the underlying patterns sufficiently.

Example:

```text
Training accuracy = 60%
Validation accuracy = 58%
```

Both are poor.

Typical pattern:

```text
Training performance → poor
Validation performance → poor
```

Possible causes include:

- Insufficient training
- Model too simple
- Poor data
- Learning-rate issues
- Other optimization problems

---

# 24. Good Fit

A desirable situation is:

```text
Training performance → good
Validation performance → good
```

with a relatively small generalization gap.

Example:

```text
Training Accuracy = 92%
Validation Accuracy = 90%
```

This is generally more encouraging than:

```text
Training Accuracy = 99%
Validation Accuracy = 65%
```

---

# 25. Generalization

The central goal of machine learning is not:

> Memorize the training data.

The goal is:

> Learn patterns that generalize to unseen data.

Therefore:

\[
\boxed{
\text{Generalization}
=
\text{Good performance on unseen examples}
}
\]

This is one of the most important ideas in fine-tuning.

---

# 26. Training Accuracy vs Validation Accuracy

Example:

```text
Epoch     Train Acc     Validation Acc

1            75%            73%
2            82%            80%
3            88%            86%
4            93%            89%
5            97%            84%
```

At first, both improve.

Then:

```text
Training accuracy continues increasing
Validation accuracy starts decreasing
```

This suggests the model is beginning to overfit.

---

# 27. Early Stopping

Early stopping means:

> Stop training when validation performance stops improving.

Example:

```text
Epoch 1 → Validation F1 = 0.78
Epoch 2 → Validation F1 = 0.82
Epoch 3 → Validation F1 = 0.86
Epoch 4 → Validation F1 = 0.85
Epoch 5 → Validation F1 = 0.81
```

The best model was around Epoch 3.

We don't necessarily want the final Epoch 5 model.

---

# 28. Best Model Selection

Suppose:

```text
Epoch 1 → F1 = 0.78
Epoch 2 → F1 = 0.82
Epoch 3 → F1 = 0.86  ← Best
Epoch 4 → F1 = 0.85
Epoch 5 → F1 = 0.81
```

Then:

\[
\boxed{\text{Best Checkpoint = Epoch 3}}
\]

We can configure Hugging Face Trainer to load the best checkpoint at the end:

```python
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)
```

The metric used for "best" should also be configured according to the training setup.

---

# 29. Computing Metrics With Trainer

We can define a metric function:

```python
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary"
        )
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
```

Then:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics
)
```

---

# 30. What Are Logits?

The model produces logits:

```text
Example 1 → [2.3, -1.1]
Example 2 → [-0.4, 3.2]
```

These are raw class scores.

We select the class with the largest score:

```python
predictions = np.argmax(
    logits,
    axis=-1
)
```

For:

```text
[2.3, -1.1]
```

the predicted class is:

```text
0
```

For:

```text
[-0.4, 3.2]
```

the predicted class is:

```text
1
```

Therefore:

\[
\boxed{
\text{Prediction}=\arg\max(\text{logits})
}
\]

---

# 31. Why `argmax`?

Suppose:

```text
logits = [1.2, 4.8, 0.3]
```

The largest value is:

```text
4.8
```

at index:

```text
1
```

Therefore:

```text
Prediction = class 1
```

---

# 32. Complete Evaluation Flow

```text
Validation Dataset
       ↓
Tokenizer
       ↓
Model
       ↓
Logits
       ↓
argmax
       ↓
Predicted Labels
       ↓
Compare with True Labels
       ↓
┌────────────────────────┐
│ Accuracy               │
│ Precision              │
│ Recall                 │
│ F1                     │
└────────────────────────┘
```

---

# 33. Confusion Matrix Example

Suppose:

```text
TP = 80
TN = 90
FP = 10
FN = 20
```

Then:

```text
                    Actual
                 Pos       Neg
             ┌─────────┬─────────┐
Pred Pos     │   80    │   10    │
             ├─────────┼─────────┤
Pred Neg     │   20    │   90    │
             └─────────┴─────────┘
```

Accuracy:

\[
\frac{80+90}{80+90+10+20}
=
\frac{170}{200}
=
85\%
\]

Precision:

\[
\frac{80}{80+10}
=
88.9\%
\]

Recall:

\[
\frac{80}{80+20}
=
80\%
\]

F1:

\[
F1=
2\frac{0.889\times0.8}{0.889+0.8}
\approx84.4\%
\]

---

# 34. Why F1 Can Be Useful

Imagine:

```text
Model A:
Precision = 95%
Recall = 50%

Model B:
Precision = 80%
Recall = 80%
```

Model A has higher precision, but Model B has a better balance.

F1 helps summarize this balance.

---

# 35. Metric Selection Depends on the Task

Ask:

> What kind of error matters most?

### Spam Detection

False positives can be problematic.

You may care strongly about:

\[
Precision
\]

### Medical Screening

Missing a positive case can be very costly.

You may care strongly about:

\[
Recall
\]

### Balanced Classification

You may use:

\[
Accuracy
\]

### Imbalanced Classification

You may prefer:

\[
F1
\]

or another class-sensitive metric.

---

# 36. Accuracy Is Not Model Loss

Loss and accuracy are different.

### Loss

Used by the training process to optimize the model.

### Accuracy

Used to measure how many predictions are correct.

Therefore:

```text
Loss
 ↓
Optimization signal
```

while:

```text
Accuracy
 ↓
Evaluation metric
```

They are related but not identical.

---

# 37. Why Can Loss Decrease While Accuracy Doesn't Change?

Suppose:

```text
Before:
Class 0 = 0.55
Class 1 = 0.45
```

and after training:

```text
Class 0 = 0.90
Class 1 = 0.10
```

The predicted class remains Class 0.

So accuracy may remain unchanged even though the model's confidence changes and its loss improves.

Therefore:

\[
\boxed{
\text{Loss and accuracy measure different things}
}
\]

---

# 38. Validation Loss Is Extremely Useful

Consider:

```text
Epoch   Train Loss   Validation Loss

1       0.70         0.72
2       0.50         0.54
3       0.35         0.40
4       0.22         0.42
5       0.12         0.55
```

The best generalization may occur around Epoch 3.

After that:

```text
Training loss ↓
Validation loss ↑
```

This strongly suggests overfitting.

---

# 39. Generalization Gap

The difference between training and validation performance is often called the **generalization gap**.

Example:

```text
Training Accuracy = 95%
Validation Accuracy = 90%
```

Then:

\[
Generalization\ Gap=95\%-90\%=5\%
\]

A large gap can indicate overfitting.

---

# 40. Fine-Tuning and Overfitting

Fine-tuning starts from useful pretrained representations:

```text
Pretrained Transformer
       ↓
General language knowledge
       ↓
Fine-tuning
       ↓
Task-specific behavior
```

If the fine-tuning dataset is small, training for too long can cause the model to overfit that dataset.

Therefore, validation monitoring is especially important.

---

# 41. Hyperparameters

Evaluation is also used to select hyperparameters.

Examples:

```text
Learning rate
Batch size
Number of epochs
Weight decay
Warmup steps
```

Suppose:

```text
Learning rate = 1e-5 → F1 = 0.81
Learning rate = 3e-5 → F1 = 0.86
Learning rate = 5e-5 → F1 = 0.83
```

We may select:

\[
\boxed{3\times10^{-5}}
\]

based on validation performance.

---

# 42. Do Not Tune on the Test Set

Do not repeatedly use the test set for model decisions.

Bad workflow:

```text
Model A → Test
Model B → Test
Model C → Test
Model D → Test
```

This allows information about the test set to influence model selection.

Correct workflow:

```text
Train → Learn
Validation → Tune
Test → Final evaluation
```

---

# 43. Complete Fine-Tuning Evaluation Pipeline

```text
                   Dataset
                      ↓
          ┌───────────┴───────────┐
          ↓                       ↓
       Training                Validation
          ↓                       ↓
    Fine-Tuning               Evaluation
          ↓                       ↓
    Updated Model         Metrics
          │
          └───────────────┐
                          ↓
                    Best Model
                          ↓
                     Test Set
                          ↓
                  Final Metrics
```

---

# 44. Practical Evaluation Code

```python
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary"
        )
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
```

Then:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics
)
```

During evaluation, Trainer can report:

```text
eval_loss
eval_accuracy
eval_precision
eval_recall
eval_f1
```

---

# 45. Important Note About `average="binary"`

This:

```python
average="binary"
```

is appropriate for binary classification when the positive class is clearly defined.

For multiclass classification, you would normally use an averaging strategy such as:

```python
average="macro"
```

or:

```python
average="weighted"
```

depending on what you want to measure.

---

# 46. Macro vs Weighted F1

For multiclass classification, F1 can be calculated separately for each class.

## Macro F1

Calculate F1 for every class and give every class equal weight.

\[
F1_{macro}
=
\frac{F1_1+F1_2+\cdots+F1_n}{n}
\]

Every class matters equally.

## Weighted F1

Weights each class according to its number of examples.

Therefore:

```text
Macro F1
→ Every class matters equally

Weighted F1
→ Larger classes contribute more
```

---

# 47. Lesson 6 Summary

You should now understand:

1. Training data updates model parameters.
2. Validation data helps select and tune the model.
3. Test data is used for final evaluation.
4. Accuracy measures the fraction of correct predictions.
5. TP means correct positive prediction.
6. TN means correct negative prediction.
7. FP means incorrect positive prediction.
8. FN means incorrect negative prediction.
9. Precision measures how trustworthy positive predictions are.
10. Recall measures how many actual positives were found.
11. F1 balances precision and recall.
12. Accuracy can be misleading with imbalanced datasets.
13. Training loss measures fit to training data.
14. Validation loss helps evaluate generalization.
15. Overfitting occurs when training performance improves while unseen-data performance worsens.
16. Underfitting occurs when the model performs poorly even on the training data.
17. Early stopping can prevent excessive overfitting.
18. Best-checkpoint selection can preserve the strongest validation model.
19. Hyperparameters should be selected using validation data.
20. The test set should be kept for final evaluation.

---

# 48. Questions & Answers

## Q1. What is the purpose of the training dataset?

### Answer

The training dataset is used to update the model's parameters.

```text
Training data
 ↓
Loss
 ↓
Gradients
 ↓
Optimizer
 ↓
Updated parameters
```

---

## Q2. What is the purpose of the validation dataset?

### Answer

Validation data evaluates the model during development and helps with:

- Hyperparameter selection
- Model selection
- Overfitting detection
- Checkpoint selection

---

## Q3. What is the purpose of the test dataset?

### Answer

The test dataset is used for the final evaluation of the selected model.

It should ideally remain unseen during model selection.

---

## Q4. What is accuracy?

### Answer

\[
Accuracy=
\frac{TP+TN}
{TP+TN+FP+FN}
\]

It represents the fraction of all predictions that are correct.

---

## Q5. What is precision?

### Answer

Precision answers:

> When the model predicts positive, how often is it actually positive?

\[
Precision=
\frac{TP}{TP+FP}
\]

---

## Q6. What is recall?

### Answer

Recall answers:

> Of all actual positive examples, how many did the model identify?

\[
Recall=
\frac{TP}{TP+FN}
\]

---

## Q7. What is F1 score?

### Answer

F1 is the harmonic mean of precision and recall:

\[
F1=
2\frac{Precision\times Recall}
{Precision+Recall}
\]

---

## Q8. Why can accuracy be misleading?

### Answer

Because of class imbalance.

For example:

```text
990 negative
10 positive
```

A model predicting every example as negative gets 99% accuracy but completely fails to identify positive examples.

---

## Q9. What is overfitting?

### Answer

Overfitting occurs when the model learns the training data too specifically and generalizes poorly to unseen data.

Typical pattern:

```text
Training loss ↓
Validation loss ↑
```

---

## Q10. What is underfitting?

### Answer

Underfitting occurs when the model has not learned the underlying patterns sufficiently.

A common pattern:

```text
Training performance → poor
Validation performance → poor
```

---

## Q11. What is early stopping?

### Answer

Early stopping stops training when validation performance stops improving.

This can prevent excessive overfitting.

---

## Q12. Why shouldn't we tune hyperparameters on the test set?

### Answer

Because repeatedly using the test set for decisions causes information about the test set to influence model selection.

The test set should ideally remain an independent final evaluation.

---

## Q13. What is a confusion matrix?

### Answer

It summarizes classification predictions using:

```text
TP
TN
FP
FN
```

---

## Q14. What is the difference between precision and recall?

### Answer

Precision asks:

> Of predicted positives, how many were actually positive?

Recall asks:

> Of actual positives, how many did we find?

---

## Q15. Why can validation loss increase while training loss decreases?

### Answer

Because the model may be increasingly fitting training examples specifically rather than learning patterns that generalize.

This is a common sign of overfitting.

---

## Q16. Why aren't loss and accuracy the same thing?

### Answer

Loss is an optimization signal, while accuracy measures whether the predicted class is correct.

A model can become more confident in an already-correct prediction and reduce its loss without changing the predicted class.

---

# 49. Self-Test

Try answering these without looking back:

1. What is the difference between training, validation, and test data?
2. What are TP, TN, FP, and FN?
3. What does accuracy measure?
4. What does precision measure?
5. What does recall measure?
6. What does F1 measure?
7. Why can a 99% accurate model be useless?
8. What is overfitting?
9. What is underfitting?
10. What happens to training and validation loss during overfitting?
11. What is early stopping?
12. Why should we not tune hyperparameters using the test set?
13. Why can loss decrease without accuracy increasing?
14. When would recall be more important than precision?
15. When would precision be more important than recall?
16. Why is F1 useful?
17. What is the generalization gap?
18. Why is validation data important during fine-tuning?

---

# 50. Final Mental Model

```text
                 TRAINING DATA
                      ↓
                 Learn Parameters
                      ↓
                  Fine-Tuning
                      ↓
                 ┌───────────┐
                 │   Model   │
                 └─────┬─────┘
                       ↓
              VALIDATION DATA
                       ↓
          ┌────────────┼────────────┐
          ↓            ↓            ↓
      Accuracy      Precision     Recall
          ↓            ↓            ↓
                       F1
                       ↓
              Model Selection
                       ↓
                 BEST MODEL
                       ↓
                   TEST DATA
                       ↓
                FINAL METRICS
```

Remember:

\[
\boxed{\text{Training}=\text{Learn}}
\]

\[
\boxed{\text{Validation}=\text{Choose}}
\]

\[
\boxed{\text{Test}=\text{Report}}
\]

The ultimate objective of fine-tuning is **not to maximize training accuracy**.

It is to obtain a model that **generalizes well to unseen data**.

---

# 51. Next Lesson

## Lesson 7 — Learning Rate, Optimizers, Weight Decay & Learning-Rate Schedulers

Next we will go deeper into the actual parameter-update process during fine-tuning.

We will understand:

```text
Learning Rate
      ↓
Optimizer
      ↓
Adam / AdamW
      ↓
Weight Decay
      ↓
Learning Rate Scheduler
      ↓
Warmup
      ↓
Parameter Updates
```

We will answer:

> **Why is the learning rate during fine-tuning usually much smaller than the learning rate we might use when training a model from scratch?**

We will connect this directly to:

```text
Gradients
   ↓
loss.backward()
   ↓
optimizer.step()
   ↓
Updated Parameters
```

so that you understand exactly how the pretrained model changes during fine-tuning.
