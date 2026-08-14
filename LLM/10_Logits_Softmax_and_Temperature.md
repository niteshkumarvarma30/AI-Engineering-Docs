# Lesson 10 — Logits, Softmax & Temperature

## Complete LLM Generation Note

This lesson explains what happens after the Transformer produces its hidden states and how those hidden states become the next generated token.

```text
Hidden State
     ↓
LM Head
     ↓
Logits
     ↓
Temperature
     ↓
Softmax
     ↓
Probability Distribution
     ↓
Sampling / Selection
     ↓
Next Token
```

---

# 1. What Are Logits?

Suppose the vocabulary contains:

```text
["cat", "dog", "car", "food"]
```

After processing the current context, the Transformer produces a hidden state:

\[
h_t
\]

The LM head converts it into one score for every vocabulary token:

\[
z=W_oh_t+b
\]

If vocabulary size is \(V\):

\[
z\in\mathbb{R}^{V}
\]

Example:

```text
cat  → 2.1
dog  → 5.4
car  → 0.7
food → 1.2
```

These values are called:

\[
\boxed{\text{Logits}}
\]

---

# 2. Logits Are Not Probabilities

Logits are raw scores.

They do not need to:

- lie between 0 and 1
- sum to 1
- be positive

They can be:

```text
-10
-2
0
3
10
```

Probabilities, however, must satisfy:

\[
0\leq P_i\leq1
\]

and:

\[
\sum_iP_i=1
\]

Therefore:

\[
\boxed{\text{Logits} \neq \text{Probabilities}}
\]

---

# 3. Why Are Logits Useful?

The relative differences between logits determine the probability distribution.

For example:

```text
cat → 2
dog → 5
```

Dog has the larger logit, so after softmax dog gets the larger probability.

Softmax is shift-invariant:

\[
softmax(z+c)=softmax(z)
\]

Therefore adding the same constant to every logit does not change the resulting probabilities.

---

# 4. Softmax

Softmax converts logits into probabilities:

\[
\boxed{
P_i=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
}
\]

Conceptually:

```text
Logits
  ↓
Exponentiation
  ↓
Normalization
  ↓
Probabilities
```

---

# 5. Numerical Softmax Example

Suppose:

```text
cat → 1
dog → 2
car → 0
```

Then:

\[
P(cat)=
\frac{e^1}{e^1+e^2+e^0}
\]

Using:

\[
e^1\approx2.718
\]

\[
e^2\approx7.389
\]

\[
e^0=1
\]

Denominator:

\[
2.718+7.389+1=11.107
\]

Therefore:

\[
P(cat)\approx0.245
\]

\[
P(dog)\approx0.665
\]

\[
P(car)\approx0.090
\]

So:

```text
cat → 24.5%
dog → 66.5%
car → 9.0%
```

and:

\[
0.245+0.665+0.090=1
\]

---

# 6. Why Does the Largest Logit Get the Largest Probability?

Because the exponential function is increasing:

\[
z_i>z_j
\Rightarrow
e^{z_i}>e^{z_j}
\]

Therefore:

\[
\boxed{
z_i>z_j
\Rightarrow
P_i>P_j
}
\]

The larger logit gets the larger probability.

---

# 7. Stable Softmax

Large logits can cause numerical overflow.

For example:

\[
e^{1000}
\]

is too large for ordinary floating-point computation.

A numerically stable form is:

\[
\boxed{
P_i=
\frac{e^{z_i-\max(z)}}
{\sum_j e^{z_j-\max(z)}}
}
\]

Subtracting the same maximum from all logits does not change the probabilities.

Example:

```text
Original:
[1000, 1001, 999]

Subtract max = 1001:

[-1, 0, -2]
```

Now softmax can safely operate on the smaller values.

---

# 8. What Happens After Softmax?

Suppose:

```text
cat  → 0.10
dog  → 0.70
car  → 0.05
food → 0.15
```

The model now has a probability distribution.

It must choose the next token.

Two basic possibilities are:

```text
Greedy:
choose highest probability

Sampling:
randomly sample according to probabilities
```

More sophisticated strategies are covered in Lesson 11.

---

# 9. Greedy Selection

Given:

```text
cat  → 0.10
dog  → 0.70
car  → 0.05
food → 0.15
```

greedy decoding chooses:

```text
dog
```

Mathematically:

\[
\boxed{
token=\arg\max_i P_i
}
\]

Greedy decoding always chooses the currently highest-probability token.

---

# 10. Why Do We Need Temperature?

Suppose the model produces:

```text
dog  → high probability
food → medium probability
cat  → lower probability
car  → low probability
```

We may want to control how sharply the model favors high-probability tokens.

**Temperature** controls this sharpness.

The temperature-scaled softmax is:

\[
\boxed{
P_i=
\frac{e^{z_i/T}}
{\sum_j e^{z_j/T}}
}
\]

where:

\[
T>0
\]

---

# 11. Temperature = 1

If:

\[
T=1
\]

then:

\[
P_i=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
\]

which is ordinary softmax.

Therefore:

\[
\boxed{T=1=\text{normal softmax}}
\]

---

# 12. Temperature < 1

Suppose:

\[
T=0.5
\]

Then:

\[
z_i/T=2z_i
\]

The differences between logits become effectively larger.

The distribution becomes sharper.

Conceptually:

```text
Temperature ↓
      ↓
Distribution becomes sharper
      ↓
High-probability tokens dominate more
      ↓
Sampling becomes more concentrated
```

Example:

Before:

```text
dog  → 0.70
food → 0.15
cat  → 0.10
car  → 0.05
```

After lowering temperature, the distribution could become more like:

```text
dog  → 0.90
food → 0.06
cat  → 0.03
car  → 0.01
```

Exact values depend on the original logits.

---

# 13. Temperature > 1

Suppose:

\[
T=2
\]

Then:

\[
z_i/T
\]

reduces the differences between logits.

The distribution becomes flatter.

Conceptually:

```text
Temperature ↑
      ↓
Distribution becomes flatter
      ↓
Lower-probability tokens receive relatively more probability
      ↓
Sampling becomes more diverse
```

Example:

Before:

```text
dog  → 0.70
food → 0.15
cat  → 0.10
car  → 0.05
```

A higher temperature could produce something like:

```text
dog  → 0.45
food → 0.25
cat  → 0.18
car  → 0.12
```

Again, exact values depend on the original logits.

---

# 14. Numerical Temperature Example

Suppose the logits are:

\[
[1,2,0]
\]

## Temperature = 1

Softmax gives approximately:

```text
cat → 0.245
dog → 0.665
car → 0.090
```

## Temperature = 2

Divide logits by 2:

\[
[0.5,1,0]
\]

Softmax gives approximately:

```text
cat → 0.307
dog → 0.506
car → 0.186
```

The distribution is flatter.

## Temperature = 0.5

Divide logits by 0.5:

\[
[2,4,0]
\]

Softmax gives approximately:

```text
cat → 0.117
dog → 0.867
car → 0.016
```

The distribution is much sharper.

Therefore:

```text
T = 0.5 → sharper
T = 1.0 → normal
T = 2.0 → flatter
```

---

# 15. Temperature Does Not Change Model Weights

Temperature is a decoding-time operation.

Changing temperature does not retrain the model.

It does not change:

```text
Attention weights
FFN weights
Embedding weights
Transformer parameters
```

Instead:

```text
Model
  ↓
Logits
  ↓
Temperature scaling
  ↓
Softmax
  ↓
Sampling
```

---

# 16. Temperature Is a Decoding Parameter

Training:

```text
Tokens
  ↓
Transformer
  ↓
Logits
  ↓
Loss
  ↓
Backpropagation
  ↓
Parameter update
```

Inference:

```text
Tokens
  ↓
Transformer
  ↓
Logits
  ↓
Temperature
  ↓
Softmax
  ↓
Sampling
```

Temperature changes how the already-trained model's predictions are used.

---

# 17. Log Probabilities

A log probability is:

\[
\log P_i
\]

For a very small probability:

\[
P=0.000001
\]

we get:

\[
\log P\approx-13.816
\]

Log probabilities are useful because very small probabilities can be difficult to work with directly.

---

# 18. Log-Softmax

Instead of calculating:

```text
logits
 ↓
softmax
 ↓
probabilities
 ↓
log
```

we can calculate log-softmax directly:

\[
\boxed{
\log softmax(z_i)
=
z_i-\log\sum_j e^{z_j}
}
\]

Implementations generally use numerically stable log-sum-exp techniques.

---

# 19. Cross-Entropy and Log Probability

Recall the single-token loss:

\[
\boxed{
L=-\log P(\text{correct})
}
\]

Therefore cross-entropy is directly related to the negative log probability assigned to the correct token.

Example:

\[
P(\text{correct})=0.9
\]

gives:

\[
L=-\log(0.9)\approx0.105
\]

while:

\[
P(\text{correct})=0.01
\]

gives:

\[
L=-\log(0.01)\approx4.605
\]

---

# 20. Logits During Training

During training:

```text
Hidden States
     ↓
LM Head
     ↓
Logits
     ↓
Cross-Entropy
     ↓
Loss
     ↓
Backpropagation
```

Framework loss functions often combine numerically stable log-softmax and negative-log-likelihood calculations internally, so you usually do not manually calculate softmax before cross-entropy.

---

# 21. Logits During Inference

During generation:

```text
Hidden State
     ↓
LM Head
     ↓
Logits
     ↓
Temperature
     ↓
Softmax / equivalent probability calculation
     ↓
Probability Distribution
     ↓
Sampling / Selection
     ↓
Next Token
```

---

# 22. Complete Numerical Example

Suppose the model produces:

```text
Token    Logit

cat        1
dog        2
car        0
```

At:

\[
T=1
\]

we obtain:

```text
cat → 0.245
dog → 0.665
car → 0.090
```

### Greedy decoding

Choose:

```text
dog
```

### Sampling

Instead of always choosing dog, sampling uses approximately:

```text
dog → 66.5%
cat → 24.5%
car → 9.0%
```

Dog is most likely, but sampling may choose another token.

---

# 23. Temperature + Sampling

Temperature does not itself mean "randomly choose a token."

Instead:

```text
Logits
  ↓
Temperature
  ↓
Probability distribution
  ↓
Sampling
  ↓
Token
```

Temperature changes the distribution.

Sampling chooses from that distribution.

Therefore:

\[
\boxed{
\text{Temperature} \neq \text{Sampling}
}
\]

---

# 24. Temperature and Determinism

Lower temperature:

```text
Sharper distribution
       ↓
More concentration on high-probability tokens
       ↓
More deterministic behavior
```

Higher temperature:

```text
Flatter distribution
       ↓
More probability spread across alternatives
       ↓
More variation
```

However, temperature does not guarantee a particular quality level.

---

# 25. What Happens as Temperature Approaches Zero?

Conceptually:

\[
T\rightarrow0^+
\]

makes the distribution concentrate around the maximum-logit token.

So:

```text
T → 0
 ↓
Very sharp distribution
 ↓
Almost always highest-logit token
```

This approaches greedy behavior.

In real APIs, exact `temperature = 0` behavior can be implementation-specific.

---

# 26. What Happens at Very High Temperature?

As:

\[
T\rightarrow\infty
\]

the differences between logits become negligible.

The distribution approaches:

\[
P_i\approx\frac{1}{V}
\]

where \(V\) is vocabulary size.

So:

```text
Very high T
     ↓
Very flat distribution
     ↓
Nearly uniform probabilities
```

This generally produces highly random output.

---

# 27. Temperature Is Not Intelligence

Changing:

```text
T = 0.2
```

to:

```text
T = 1.2
```

does not make the model smarter.

It changes:

\[
\boxed{\text{Output distribution sharpness}}
\]

not:

```text
Model knowledge
Model architecture
Model weights
Model intelligence
```

---

# 28. Temperature Is Not a Hallucination Cure

Lower temperature can make output more deterministic, but it cannot guarantee factual correctness.

If the model's highest-probability answer is wrong, a low temperature may simply make it produce the same wrong answer more consistently.

Therefore:

\[
\boxed{
\text{Low temperature} \neq \text{guaranteed factuality}
}
\]

Grounding requires appropriate methods such as:

- RAG
- Tool use
- Verification
- Better training
- External sources

---

# 29. Training vs Inference

## Training

```text
Tokens
 ↓
Transformer
 ↓
Logits
 ↓
Cross-Entropy
 ↓
Loss
 ↓
Backpropagation
 ↓
Parameter Update
```

## Inference

```text
Tokens
 ↓
Transformer
 ↓
Logits
 ↓
Temperature
 ↓
Softmax
 ↓
Sampling / Selection
 ↓
Next Token
```

---

# 30. Complete LLM Generation Pipeline

Putting the previous lessons together:

```text
User Prompt
     ↓
Tokenizer
     ↓
Token IDs
     ↓
Embedding
     ↓
RoPE / Position Information
     ↓
Transformer Blocks
     │
     ├── Causal Self-Attention
     ├── MHA / GQA / MQA
     └── FFN / MoE
     ↓
Hidden State
     ↓
LM Head
     ↓
Logits
     ↓
Temperature
     ↓
Softmax
     ↓
Sampling Strategy
     ↓
Next Token
     ↓
Append Token
     ↓
Repeat
```

---

# 31. Important Formulas

### LM Head

\[
\boxed{
z=W_oh+b
}
\]

### Softmax

\[
\boxed{
P_i=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
}
\]

### Temperature-Scaled Softmax

\[
\boxed{
P_i=
\frac{e^{z_i/T}}
{\sum_j e^{z_j/T}}
}
\]

### Cross-Entropy

\[
\boxed{
L=-\log P(\text{correct token})
}
\]

### Stable Softmax

\[
\boxed{
P_i=
\frac{e^{z_i-\max(z)}}
{\sum_j e^{z_j-\max(z)}}
}
\]

---

# 32. What You Must Remember

### Logits

Raw scores produced by the LM head.

### Softmax

Converts logits into a probability distribution.

### Temperature

Controls how sharp or flat that distribution is.

### Sampling

Chooses a token from the distribution.

The complete inference chain is:

\[
\boxed{
Hidden\ State
\rightarrow
Logits
\rightarrow
Temperature
\rightarrow
Softmax
\rightarrow
Probability\ Distribution
\rightarrow
Sampling
\rightarrow
Next\ Token
}
\]

---

# 33. Quick Comparison

| Concept | Purpose |
|---|---|
| Logits | Raw token scores |
| Softmax | Converts logits to probabilities |
| Temperature | Controls distribution sharpness |
| Greedy decoding | Selects highest-probability token |
| Sampling | Samples from probability distribution |
| Cross-Entropy | Measures prediction error during training |
| LM Head | Converts hidden states to vocabulary logits |

---

# 34. Self-Check Questions

1. What is a logit?
2. Why are logits not probabilities?
3. Can logits be negative?
4. What does the LM head do?
5. What does softmax do?
6. Why do softmax probabilities sum to 1?
7. Why does the largest logit produce the largest probability?
8. Why is stable softmax useful?
9. What is temperature?
10. What happens when \(T<1\)?
11. What happens when \(T=1\)?
12. What happens when \(T>1\)?
13. Why does lower temperature make the distribution sharper?
14. Why does higher temperature make it flatter?
15. Does temperature change model parameters?
16. What is a log probability?
17. What is log-softmax?
18. What is greedy decoding?
19. What is sampling?
20. What is the difference between temperature and sampling?
21. What happens conceptually as \(T\rightarrow0^+\)?
22. What happens as \(T\rightarrow\infty\)?
23. Does lower temperature guarantee factual answers?
24. What is the complete hidden-state-to-next-token pipeline?

---

# 35. Next Lesson — Sampling Strategies

Next we study how the model chooses tokens from its probability distribution:

```text
Probability Distribution
        ↓
Greedy Decoding
        ↓
Random Sampling
        ↓
Top-k Sampling
        ↓
Top-p / Nucleus Sampling
        ↓
Min-p Sampling
        ↓
Beam Search
        ↓
Temperature + Sampling
        ↓
Generation quality vs diversity
```

The key question is:

> If the model gives probabilities to thousands of possible tokens, how do we decide which token to actually generate?
