# Lesson 10 - Logits, Softmax & Temperature

## Complete LLM Generation Note

> GitHub-safe version: formulas use plain ASCII notation. No LaTeX syntax or special mathematical symbols are used.

---

## Core Question

What happens after a Transformer produces its hidden state?

The simplified generation pipeline is:

```text
Hidden State
     |
     v
LM Head
     |
     v
Logits
     |
     v
Temperature
     |
     v
Softmax
     |
     v
Probability Distribution
     |
     v
Sampling / Selection
     |
     v
Next Token
```

This lesson explains how an LLM turns internal representations into the probability distribution used to generate the next token.

---

# 1. What Are Logits?

A Transformer produces a hidden representation for the current position.

Call it:

```text
h
```

The model must convert this hidden representation into scores for every token in its vocabulary.

The final projection is often called the:

```text
LM Head
```

A simplified equation is:

```text
z = W_o h + b
```

where:

```text
h = hidden state
W_o = output projection matrix
b = bias
z = logits
```

If the vocabulary contains 50,000 tokens:

```text
Hidden state
     |
     v
   LM Head
     |
     v
50,000 logits
```

There is approximately one logit for every vocabulary token.

---

# 2. Logits Are Not Probabilities

Suppose the vocabulary contains:

```text
["cat", "dog", "car", "food"]
```

The model might produce:

```text
cat  -> 2.1
dog  -> 5.4
car  -> 0.7
food -> 1.2
```

These values are logits.

They are not probabilities.

Logits can:

```text
be positive
be negative
be zero
be much larger or smaller than 1
```

They do not have to:

```text
sum to 1
```

A larger logit means that token receives a larger score relative to the other tokens.

To convert logits into probabilities, we normally use softmax.

---

# 3. Why Are Logits Useful?

The Transformer works in a high-dimensional vector space.

The final hidden state contains information about:

```text
Context
Semantics
Syntax
Previous tokens
Learned patterns
Task instructions
```

The LM head converts that representation into vocabulary-level scores.

Conceptually:

```text
Hidden representation
        |
        v
"What should come next?"
        |
        v
Score every vocabulary token
        |
        v
Logits
```

For example:

```text
Context:

"The capital of France is"

Possible logits:

Paris      -> 8.7
London     -> 2.1
Berlin     -> 1.5
banana     -> -3.2
```

The model strongly favors `Paris`.

---

# 4. Logits Can Be Negative

There is no requirement that logits be positive.

Example:

```text
cat  -> -1.2
dog  ->  2.4
car  -> -0.7
food -> -3.1
```

Softmax can still convert these into valid probabilities.

The important thing is the relative difference between logits.

For example:

```text
[2, 1, 0]
```

and:

```text
[102, 101, 100]
```

produce the same softmax probabilities because the difference between corresponding logits is identical.

This leads to an important property.

---

# 5. Softmax Is Shift-Invariant

Suppose:

```text
z = [2, 1, 0]
```

Now add 100 to every logit:

```text
z' = [102, 101, 100]
```

The probability distribution does not change.

Why?

Because:

```text
softmax(z + c) = softmax(z)
```

for any constant `c` added to every logit.

This property is useful in numerical implementations.

A common stable softmax implementation subtracts the maximum logit:

```text
z_stable = z - max(z)
```

Then:

```text
softmax(z)
=
softmax(z - max(z))
```

This helps avoid very large exponential values.

---

# 6. What Is Softmax?

Softmax converts a vector of logits into a probability distribution.

For logit `z_i`:

```text
P_i = exp(z_i) / sum_j exp(z_j)
```

The result satisfies:

```text
0 <= P_i <= 1
```

and:

```text
sum of all P_i = 1
```

Therefore:

```text
Logits
   |
   v
Softmax
   |
   v
Probabilities
```

---

# 7. Numerical Softmax Example

Suppose:

```text
Logits:

cat  -> 2
dog  -> 1
car  -> 0
```

First calculate exponentials:

```text
exp(2) ~= 7.389
exp(1) ~= 2.718
exp(0) = 1
```

Sum:

```text
7.389 + 2.718 + 1
= 11.107
```

Probabilities:

```text
P(cat) = 7.389 / 11.107 ~= 0.665

P(dog) = 2.718 / 11.107 ~= 0.245

P(car) = 1 / 11.107 ~= 0.090
```

So:

```text
cat -> 0.665
dog -> 0.245
car -> 0.090
```

Check:

```text
0.665 + 0.245 + 0.090
= 1.000
```

The highest logit produced the highest probability.

---

# 8. Why Does Softmax Use Exponential?

Softmax uses the exponential function because it:

```text
turns scores into positive values
```

and:

```text
amplifies differences between scores
```

Suppose:

```text
logits = [2, 1]
```

The difference is:

```text
2 - 1 = 1
```

After exponentiation:

```text
exp(2) ~= 7.389
exp(1) ~= 2.718
```

The larger score becomes much more dominant.

This produces a smooth probability distribution rather than simply selecting the largest logit.

---

# 9. Argmax vs Softmax

There are two different ideas.

## Argmax

Choose the token with the highest logit.

```text
logits:

cat -> 2
dog -> 5
car -> 1

argmax -> dog
```

This is deterministic.

## Softmax

Convert all logits into probabilities.

```text
cat -> 0.05
dog -> 0.90
car -> 0.05
```

Softmax does not itself select a token.

It produces the probability distribution that can then be used for:

```text
sampling
```

or:

```text
greedy selection
```

---

# 10. Greedy Decoding

Greedy decoding chooses:

```text
token = argmax(logits)
```

Example:

```text
Paris -> 0.72
London -> 0.12
Berlin -> 0.08
Rome -> 0.08
```

Greedy decoding selects:

```text
Paris
```

It always chooses the currently highest-probability token.

Advantages:

```text
Simple
Deterministic
Fast
```

Limitations:

```text
Can be repetitive
Can miss a better continuation
Can produce predictable text
```

---

# 11. Sampling

Instead of always choosing the highest-probability token, sampling treats probabilities as a distribution.

Suppose:

```text
Paris  -> 0.70
London -> 0.20
Berlin -> 0.08
Rome   -> 0.02
```

A sampling algorithm might select:

```text
Paris
```

most often, but:

```text
London
Berlin
Rome
```

can also be selected according to their probabilities.

This introduces controlled randomness.

---

# 12. Why Sampling Matters

Natural language can have multiple valid continuations.

Suppose:

```text
The weather today is
```

Possible continuations might include:

```text
sunny
pleasant
warm
cloudy
beautiful
```

If the model always selects only the highest-probability token, generation can become repetitive or overly predictable.

Sampling allows lower-probability but still plausible alternatives to appear.

---

# 13. Temperature

Temperature controls the sharpness of the probability distribution.

The temperature-adjusted softmax is:

```text
P_i
=
exp(z_i / T)
/
sum_j exp(z_j / T)
```

where:

```text
T = temperature
```

The most important cases are:

```text
T < 1
T = 1
T > 1
```

---

# 14. Temperature = 1

When:

```text
T = 1
```

the formula becomes normal softmax:

```text
P_i
=
exp(z_i)
/
sum_j exp(z_j)
```

So:

```text
Temperature = 1
```

means:

```text
No temperature scaling
```

relative to the original logits.

---

# 15. Temperature < 1

Suppose:

```text
T = 0.5
```

Then:

```text
z_i / 0.5 = 2 z_i
```

This makes differences between logits larger.

The probability distribution becomes sharper.

Example:

```text
Before:

A -> 0.60
B -> 0.30
C -> 0.10
```

After lowering temperature:

```text
A -> more dominant
B -> less likely
C -> much less likely
```

So:

```text
Lower temperature
      |
      v
Sharper distribution
      |
      v
More deterministic behavior
```

---

# 16. Temperature > 1

Suppose:

```text
T = 2
```

Then:

```text
z_i / 2
```

reduces the differences between logits.

The probability distribution becomes flatter.

Example:

```text
Before:

A -> 0.60
B -> 0.30
C -> 0.10
```

After increasing temperature:

```text
A -> less dominant
B -> more competitive
C -> more competitive
```

So:

```text
Higher temperature
      |
      v
Flatter distribution
      |
      v
More randomness
```

---

# 17. Temperature Mental Model

Remember:

```text
Low temperature
    |
    v
Sharper distribution
    |
    v
More predictable

Temperature = 1
    |
    v
Original distribution

High temperature
    |
    v
Flatter distribution
    |
    v
More random
```

A useful intuition:

```text
Temperature controls how strongly
the model's preferences are expressed.
```

It does not directly add knowledge or intelligence.

---

# 18. Numerical Temperature Example

Suppose logits are:

```text
A -> 2
B -> 1
C -> 0
```

At:

```text
T = 1
```

we already calculated approximately:

```text
A -> 0.665
B -> 0.245
C -> 0.090
```

Now use:

```text
T = 2
```

Scaled logits:

```text
A -> 1
B -> 0.5
C -> 0
```

Exponentials:

```text
exp(1)   ~= 2.718
exp(0.5) ~= 1.649
exp(0)   = 1
```

Sum:

```text
2.718 + 1.649 + 1
= 5.367
```

Probabilities:

```text
A -> 2.718 / 5.367 ~= 0.506
B -> 1.649 / 5.367 ~= 0.307
C -> 1 / 5.367      ~= 0.186
```

Compare:

```text
T = 1

A -> 0.665
B -> 0.245
C -> 0.090
```

with:

```text
T = 2

A -> 0.506
B -> 0.307
C -> 0.186
```

The distribution became flatter.

---

# 19. Temperature Below 1: Numerical Example

Using:

```text
A -> 2
B -> 1
C -> 0
```

and:

```text
T = 0.5
```

Scaled logits:

```text
A -> 4
B -> 2
C -> 0
```

Exponentials:

```text
exp(4) ~= 54.598
exp(2) ~= 7.389
exp(0) = 1
```

Sum:

```text
54.598 + 7.389 + 1
= 62.987
```

Probabilities:

```text
A -> 54.598 / 62.987 ~= 0.867
B -> 7.389 / 62.987  ~= 0.117
C -> 1 / 62.987       ~= 0.016
```

Compare:

```text
T = 0.5
A -> 0.867
B -> 0.117
C -> 0.016
```

The distribution is much sharper.

---

# 20. Why Temperature Changes Randomness

Suppose:

```text
A -> 0.90
B -> 0.08
C -> 0.02
```

Sampling already strongly prefers A.

With lower temperature:

```text
A becomes even more dominant.
```

With higher temperature:

```text
B and C become more competitive.
```

Therefore:

```text
Temperature
     |
     v
Changes probability distribution
     |
     v
Changes sampling behavior
```

Temperature does not directly modify the model's learned parameters.

---

# 21. Temperature Does Not Change the Ranking

Temperature scaling with positive `T` divides every logit by the same positive value.

Therefore:

```text
If z_A > z_B
```

then:

```text
z_A / T > z_B / T
```

for:

```text
T > 0
```

So temperature does not change the ordering of logits.

Example:

```text
Original:

A -> 5
B -> 3
C -> 1
```

At:

```text
T = 2
```

we get:

```text
A -> 2.5
B -> 1.5
C -> 0.5
```

The order remains:

```text
A > B > C
```

Temperature changes the gaps between scores, not their ranking.

---

# 22. Temperature and Entropy

Entropy measures the uncertainty of a probability distribution.

For a distribution `P`:

```text
H(P)
=
-sum_i P_i log(P_i)
```

A sharper distribution generally has lower entropy.

A flatter distribution generally has higher entropy.

Therefore, typically:

```text
Lower temperature
    |
    v
Lower entropy

Higher temperature
    |
    v
Higher entropy
```

The exact entropy change depends on the logits and distribution.

---

# 23. Why Softmax Is Usually Applied After Temperature

The standard sequence is:

```text
Logits
   |
   v
Divide by temperature
   |
   v
Softmax
   |
   v
Probabilities
```

Mathematically:

```text
P_i
=
softmax(z_i / T)
```

Temperature modifies the logits before softmax.

It is not normally applied to probabilities after softmax.

---

# 24. Logits -> Temperature -> Softmax

The complete numerical flow:

```text
Hidden state
     |
     v
LM Head
     |
     v
Logits
[2, 1, 0]
     |
     v
Temperature
T = 2
     |
     v
Scaled logits
[1, 0.5, 0]
     |
     v
Softmax
     |
     v
Probabilities
[0.506, 0.307, 0.186]
```

Then the decoder chooses a token using:

```text
argmax
```

or:

```text
sampling
```

depending on the decoding strategy.

---

# 25. Log Probabilities

Sometimes systems work with log probabilities.

Instead of:

```text
P(token)
```

we use:

```text
log P(token)
```

Example:

```text
P(token) = 0.8
```

then:

```text
log(0.8) ~= -0.223
```

Because probabilities are between 0 and 1:

```text
log(probability) <= 0
```

except:

```text
log(1) = 0
```

---

# 26. Why Use Log Probabilities?

There are several reasons.

## Numerical stability

Products of many probabilities can become extremely small.

For example:

```text
0.8 x 0.7 x 0.6 x 0.5
= 0.168
```

For thousands of tokens, the product can become extremely tiny.

Logs convert multiplication into addition:

```text
log(a x b)
=
log(a) + log(b)
```

Therefore sequence probabilities can be handled more conveniently.

## Training connection

Cross-entropy uses negative log probability:

```text
Loss = -log(P(correct))
```

So log probabilities are already central to language-model training.

---

# 27. Token Probability vs Sequence Probability

Suppose:

```text
P(token1) = 0.8
P(token2 | token1) = 0.6
P(token3 | token1, token2) = 0.5
```

The sequence probability is:

```text
P(token1, token2, token3)
=
0.8 x 0.6 x 0.5
=
0.24
```

In log space:

```text
log(0.24)
=
log(0.8) + log(0.6) + log(0.5)
```

This is important for:

```text
Language-model scoring
Beam search
Sequence evaluation
Perplexity
```

---

# 28. Perplexity

Perplexity is a common language-model evaluation metric.

For average negative log-likelihood:

```text
L = -(1/N) * sum log P(correct token)
```

perplexity is:

```text
Perplexity = exp(L)
```

Lower perplexity generally means the model assigned higher probability to the observed tokens on the evaluation data.

Example:

```text
Average loss = 1.0

Perplexity = exp(1)
           ~= 2.718
```

If:

```text
Average loss = 2.0
```

then:

```text
Perplexity = exp(2)
           ~= 7.389
```

Perplexity depends on the tokenizer, dataset, masking/evaluation setup, and other details, so values should be compared under compatible conditions.

---

# 29. What Happens After Softmax?

After softmax, we have:

```text
Vocabulary token
        +
Probability
```

For example:

```text
Paris  -> 0.72
London -> 0.12
Berlin -> 0.08
Rome   -> 0.08
```

The generation algorithm now decides how to select the next token.

Possible strategies include:

```text
Greedy decoding
Temperature sampling
Top-k sampling
Top-p sampling
Typical sampling
Beam search
```

These strategies will be studied in more detail in the next lesson.

---

# 30. Greedy vs Sampling

## Greedy

```text
Choose highest-probability token.
```

Example:

```text
A -> 0.70
B -> 0.20
C -> 0.10

Choose A.
```

## Sampling

```text
Sample according to the distribution.
```

Example:

```text
A -> 0.70
B -> 0.20
C -> 0.10
```

A random draw can produce:

```text
A
```

most often, but B or C can also appear.

---

# 31. Temperature vs Top-k vs Top-p

These are related but different.

## Temperature

Changes the shape of the probability distribution.

```text
Temperature
    |
    v
Rescale logits
```

## Top-k

Keeps only the:

```text
k highest-probability tokens
```

and removes the rest from consideration.

## Top-p

Keeps the smallest set of tokens whose cumulative probability reaches:

```text
p
```

Then samples from that restricted distribution.

Mental model:

```text
Temperature -> reshape probabilities

Top-k       -> keep k candidates

Top-p       -> keep a probability-mass nucleus
```

---

# 32. A Complete Generation Example

Suppose the model sees:

```text
"The capital of France is"
```

The Transformer produces:

```text
Hidden state
     |
     v
LM Head
     |
     v
Logits

Paris   -> 8.0
London  -> 4.0
Berlin  -> 3.5
Rome    -> 2.0
```

Suppose:

```text
Temperature = 1
```

Softmax produces a distribution where Paris has the highest probability.

Then:

```text
Greedy decoding
        |
        v
Paris
```

or:

```text
Sampling
        |
        v
Random draw from distribution
```

If temperature is increased, the probability gap between Paris and alternatives becomes smaller.

---

# 33. Complete LLM Generation Pipeline

```text
Prompt
  |
  v
Tokenizer
  |
  v
Token IDs
  |
  v
Embeddings
  |
  v
Transformer Blocks
  |
  +--> Attention
  |
  +--> FFN / MoE
  |
  v
Final Hidden State
  |
  v
LM Head
  |
  v
Logits
  |
  v
Temperature Scaling
  |
  v
Softmax
  |
  v
Probability Distribution
  |
  +--> Greedy
  |
  +--> Sampling
  |
  +--> Top-k
  |
  +--> Top-p
  |
  v
Next Token
  |
  v
Append Token
  |
  v
Repeat
```

This loop continues until:

```text
EOS token
```

or:

```text
maximum generation length
```

or another stopping condition is reached.

---

# 34. Important Numerical Stability Trick

Computing:

```text
exp(large_number)
```

can overflow.

For example:

```text
exp(1000)
```

is extremely large.

A stable softmax implementation uses:

```text
m = max(z)

P_i
=
exp(z_i - m)
/
sum_j exp(z_j - m)
```

Because subtracting the same value from every logit does not change the softmax distribution.

Example:

```text
Original logits:

[1000, 999, 998]
```

Subtract maximum:

```text
[0, -1, -2]
```

Then calculate:

```text
exp(0)
exp(-1)
exp(-2)
```

This is numerically much safer.

---

# 35. Why Does the Highest Logit Get the Highest Probability?

Suppose:

```text
z_A > z_B
```

Then:

```text
exp(z_A) > exp(z_B)
```

Therefore:

```text
P_A > P_B
```

So softmax preserves ranking.

Temperature with positive `T` also preserves ranking because:

```text
z_A > z_B
```

implies:

```text
z_A / T > z_B / T
```

for `T > 0`.

---

# 36. What Happens If Temperature Approaches Zero?

As:

```text
T -> 0+
```

the distribution becomes increasingly concentrated around the highest logit.

Conceptually:

```text
T very small
      |
      v
Highest-logit token dominates
      |
      v
Behavior approaches greedy selection
```

Exactly setting:

```text
T = 0
```

is not evaluated by the formula `z / T`; implementations normally treat zero-temperature decoding as a deterministic or greedy-style special case.

---

# 37. What Happens If Temperature Becomes Very Large?

As:

```text
T -> very large
```

the scaled logits:

```text
z_i / T
```

become closer to one another.

The resulting softmax distribution approaches a more uniform distribution.

Conceptually:

```text
Very high T
     |
     v
Logit differences become small
     |
     v
Probabilities become more similar
```

For a finite vocabulary with no other constraints, the limiting distribution approaches uniformity.

---

# 38. Temperature Is Not "Creativity"

A common oversimplification is:

```text
High temperature = creativity
Low temperature = intelligence
```

That is not technically correct.

Temperature is a decoding parameter that changes the probability distribution.

It does not:

```text
add knowledge
remove knowledge
change model parameters
retrain the model
```

It changes how the model's existing token preferences are converted into generation behavior.

---

# 39. Why Temperature Can Affect Output Quality

Suppose:

```text
Correct/plausible token -> 0.70
Alternative             -> 0.20
Weak alternative        -> 0.10
```

Low temperature may make the first option dominate strongly.

High temperature may increase the chance of alternatives.

Therefore temperature can affect:

```text
determinism
diversity
repetition
risk of unlikely tokens
style
```

The best setting depends on the task.

---

# 40. Logits, Softmax, Temperature - Mental Model

Remember:

```text
Logits
  |
  | raw scores
  v
Temperature
  |
  | reshape score differences
  v
Softmax
  |
  | convert scores to probabilities
  v
Probability distribution
  |
  | decoding strategy
  v
Next token
```

The three concepts have different jobs:

```text
Logits
= raw vocabulary scores

Temperature
= rescales logits before softmax

Softmax
= converts logits into probabilities
```

---

# 41. Connection to Lesson 09

Lesson 09 explained training:

```text
Hidden States
     |
     v
LM Head
     |
     v
Logits
     |
     v
Cross-Entropy Loss
     |
     v
Backpropagation
     |
     v
Parameter Updates
```

Lesson 10 focuses on generation:

```text
Hidden State
     |
     v
LM Head
     |
     v
Logits
     |
     v
Temperature
     |
     v
Softmax
     |
     v
Decoding
     |
     v
Next Token
```

The same logits have different roles:

```text
Training:
logits -> loss -> gradients

Inference:
logits -> probabilities -> token selection
```

---

# 42. Connection to Previous Lessons

The complete conceptual pipeline is now:

```text
Lesson 03
Tokenization
     |
     v
Token IDs

Lesson 04
Embeddings
     |
     v
Vector representations

Lesson 05
Transformer architecture
     |
     v
Q/K/V + attention + FFN

Lesson 06
RoPE
     |
     v
Position-aware Q/K

Lesson 07
MoE
     |
     v
Sparse FFN computation

Lesson 08
Context + efficient attention
     |
     v
KV Cache + MHA/MQA/GQA/MLA/Flash Attention

Lesson 09
Pretraining
     |
     v
Next-token loss + backpropagation

Lesson 10
Generation
     |
     v
Logits + temperature + softmax
```

---

# 43. Common Misconceptions

## Misconception 1

"Logits are probabilities."

False.

Logits are raw scores.

```text
Logits -> Softmax -> Probabilities
```

## Misconception 2

"Softmax selects the token."

Not exactly.

Softmax creates the probability distribution. A decoding strategy selects or samples the token.

## Misconception 3

"Higher temperature means the model knows more."

False.

Temperature changes decoding behavior, not model knowledge.

## Misconception 4

"Temperature changes the ranking of tokens."

For positive temperature, no.

It changes the gaps between logits but preserves their ordering.

## Misconception 5

"Temperature is applied after softmax."

Normally, temperature is applied to logits before softmax:

```text
logits / T -> softmax
```

## Misconception 6

"Temperature = 0 means divide logits by zero."

No.

Zero-temperature decoding is normally handled as a special deterministic/greedy case.

## Misconception 7

"Softmax always produces a very sharp distribution."

False.

The sharpness depends on the relative logits and temperature.

---

# 44. Interview Questions

## Q1. What is a logit?

A raw score produced by the LM head for a vocabulary token.

## Q2. Are logits probabilities?

No.

## Q3. What does softmax do?

It converts logits into a probability distribution whose values sum to 1.

## Q4. Why can logits be negative?

There is no requirement that logits be positive. They are unconstrained real-valued scores.

## Q5. What does temperature do?

It rescales logits before softmax:

```text
z_i / T
```

## Q6. What happens when temperature decreases?

The distribution becomes sharper and generation becomes more concentrated around high-probability tokens.

## Q7. What happens when temperature increases?

The distribution becomes flatter and lower-probability alternatives become more competitive.

## Q8. Does temperature change the token ranking?

Not for positive temperature.

## Q9. What is greedy decoding?

Selecting the highest-probability token, equivalently the highest-logit token.

## Q10. What is sampling?

Drawing a token according to the probability distribution.

## Q11. Why subtract the maximum logit before softmax?

For numerical stability. It prevents very large exponential values while preserving the same probability distribution.

## Q12. What is perplexity?

A metric derived from average negative log-likelihood:

```text
Perplexity = exp(average loss)
```

Lower perplexity generally indicates better probability assignment on compatible evaluation data.

## Q13. What is the difference between training and generation logits?

The logits are produced by the same general LM-head mechanism, but during training they are compared against target tokens to calculate loss, while during inference they are used for decoding.

---

# 45. Final Mental Model

Remember this:

```text
1. Transformer
   produces a hidden representation.

2. LM Head
   converts the hidden representation into vocabulary-sized logits.

3. Logits
   are raw scores, not probabilities.

4. Temperature
   rescales logits before softmax.

5. Softmax
   converts logits into probabilities.

6. Decoding
   uses those probabilities to choose or sample the next token.

7. The selected token
   is appended to the context.

8. The process repeats
   until a stopping condition is reached.
```

The complete generation loop:

```text
Prompt
  |
  v
Tokenizer
  |
  v
Transformer
  |
  v
Hidden State
  |
  v
LM Head
  |
  v
Logits
  |
  v
Temperature
  |
  v
Softmax
  |
  v
Probabilities
  |
  v
Decoding
  |
  v
Next Token
  |
  v
Append
  |
  +------------------+
                     |
                     v
                Transformer
```

---

# 46. Next Lesson - Sampling Strategies

Next, we will go deeper into how the probability distribution is converted into actual tokens.

Topics:

```text
Greedy decoding
      |
      v
Temperature sampling
      |
      v
Top-k sampling
      |
      v
Top-p / nucleus sampling
      |
      v
Typical sampling
      |
      v
Repetition penalty
      |
      v
Frequency penalty
      |
      v
Presence penalty
      |
      v
Beam search
      |
      v
Decoding trade-offs
```

The key question for Lesson 11 will be:

```text
Once the model gives us:

A -> 0.70
B -> 0.20
C -> 0.10

How should we choose the next token?
```

---

# End of Lesson 10
