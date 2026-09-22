# Lesson 11 - Sampling Strategies

## Complete LLM Generation Note

This lesson explains how an LLM chooses the next token after producing logits and probabilities.

The overall generation pipeline is:

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
Probability Distribution
   |
   v
Sampling / Selection
   |
   v
Next Token
```

The important idea is:

```text
The LLM does not directly output a word.

It produces scores for possible next tokens.
Those scores are converted into probabilities.
A decoding strategy then selects the next token.
```

---

# 1. Why Do We Need Sampling?

Suppose the model has generated:

```text
The cat is
```

The model may predict:

```text
sleeping -> 50%
hungry   -> 25%
running  -> 15%
cute     -> 7%
outside  -> 3%
```

There are several possible next tokens.

The model therefore needs a **decoding strategy** to decide which token to generate.

There are two broad approaches:

```text
Selection
    |
    +-- Greedy decoding
    |
    +-- Beam search
    |
    +-- Sampling
          |
          +-- Temperature
          +-- Top-k
          +-- Top-p
```

The word "sampling" is sometimes used broadly for the whole decoding stage, but technically:

```text
Greedy = deterministic selection
Sampling = probabilistic selection
```

---

# 2. Logits Come First

Before probabilities exist, the model produces logits.

Suppose the LM head produces:

```text
A -> 5
B -> 4
C -> 1
```

These are logits.

They are not probabilities.

They can be:

```text
positive
negative
zero
larger than 1
smaller than 1
```

The model then applies temperature and softmax.

```text
Logits
   |
   v
Temperature scaling
   |
   v
Softmax
   |
   v
Probabilities
```

---

# 3. Softmax Converts Logits to Probabilities

Softmax is:

```text
P(i) = exp(z_i) / sum_j exp(z_j)
```

Suppose:

```text
logits = [2, 1, 0]
```

Then approximately:

```text
A -> 0.665
B -> 0.245
C -> 0.090
```

So:

```text
A -> 66.5%
B -> 24.5%
C -> 9.0%
```

Now the model has a probability distribution.

The probabilities sum to approximately:

```text
0.665 + 0.245 + 0.090 = 1.000
```

---

# 4. The Model Now Has Choices

Imagine:

```text
A -> 66.5%
B -> 24.5%
C -> 9.0%
```

The model can now choose the next token.

There are multiple strategies.

```text
Strategy 1 -> Always choose A
Strategy 2 -> Randomly sample using probabilities
Strategy 3 -> Restrict candidates first, then sample
```

These correspond to different decoding methods.

---

# 5. Greedy Decoding

Greedy decoding chooses the token with the highest probability.

Example:

```text
A -> 66.5%
B -> 24.5%
C -> 9.0%
```

Greedy chooses:

```text
A
```

The rule is:

```text
next_token = argmax(probabilities)
```

Or directly from logits:

```text
next_token = argmax(logits)
```

because softmax preserves ranking.

---

## 5.1 Example

Suppose:

```text
The capital of France is
```

Model probabilities:

```text
Paris -> 95%
London -> 2%
Berlin -> 1%
Rome -> 1%
Other -> 1%
```

Greedy chooses:

```text
Paris
```

---

## 5.2 Advantage of Greedy Decoding

Greedy decoding is:

```text
simple
fast
deterministic
cheap
```

It is useful when the highest-probability continuation is usually what you want.

---

## 5.3 Limitation of Greedy Decoding

Greedy decoding only looks at the current step.

It does not consider whether a slightly lower-probability token could lead to a better sequence later.

Example:

```text
At step 1:

A -> 0.60
B -> 0.40
```

Greedy chooses:

```text
A
```

But suppose:

```text
If A:
A -> future probability = 0.10

If B:
B -> future probability = 0.90
```

The globally better sequence might start with B.

Greedy does not look ahead.

This is one reason beam search exists.

---

# 6. Random Sampling

Instead of always choosing the highest-probability token, sampling uses the probability distribution.

Suppose:

```text
A -> 0.60
B -> 0.25
C -> 0.10
D -> 0.05
```

Sampling means:

```text
A has a 60% chance
B has a 25% chance
C has a 10% chance
D has a 5% chance
```

A possible sequence of independent choices might be:

```text
Run 1 -> A
Run 2 -> B
Run 3 -> A
Run 4 -> A
Run 5 -> C
```

The model does not always choose A.

---

# 7. Why Use Random Sampling?

Because always selecting the highest-probability token can produce repetitive or overly predictable text.

Sampling can provide:

```text
diversity
variation
creativity
alternative phrasings
less deterministic output
```

For example, instead of always generating:

```text
The weather is nice today.
```

sampling may produce:

```text
The weather is beautiful today.
```

or:

```text
It is a pleasant day outside.
```

or:

```text
Today feels warm and pleasant.
```

The exact behavior depends on the model and decoding settings.

---

# 8. Important Difference: Probability Does Not Mean Certainty

If:

```text
A -> 70%
```

it does NOT mean:

```text
The model will definitely choose A.
```

It means that under probabilistic sampling, A has a 70% probability of being selected at that step, assuming the distribution is used directly and no additional filtering changes it.

Similarly:

```text
B -> 20%
```

means B still has a chance.

This is the fundamental difference between:

```text
Greedy selection
```

and:

```text
Probabilistic sampling
```

---

# 9. Temperature Sampling

Temperature changes the shape of the probability distribution before sampling.

The formula is:

```text
P(i) = exp(z_i / T) / sum_j exp(z_j / T)
```

where:

```text
T = temperature
```

Remember:

```text
Low T  -> sharper
High T -> flatter
```

---

## 9.1 Low Temperature

Suppose:

```text
logits = [5, 4, 1]
```

At:

```text
T = 0.5
```

scaled logits:

```text
[10, 8, 2]
```

The probability becomes approximately:

```text
A -> 88.1%
B -> 11.9%
C -> ~0%
```

A dominates.

So:

```text
Low T
-> concentrated distribution
-> less randomness
-> more predictable
```

---

## 9.2 High Temperature

At:

```text
T = 2
```

scaled logits:

```text
[2.5, 2, 0.5]
```

Probabilities become approximately:

```text
A -> 54.7%
B -> 33.1%
C -> 12.2%
```

The distribution is flatter.

So:

```text
High T
-> more spread-out distribution
-> more possible choices
-> more diversity
```

---

# 10. Temperature Does Not Change Ranking

For positive temperature:

```text
logits = [5, 4, 1]
```

Ranking:

```text
A > B > C
```

At:

```text
T = 0.5
```

```text
[10, 8, 2]
```

Still:

```text
A > B > C
```

At:

```text
T = 2
```

```text
[2.5, 2, 0.5]
```

Still:

```text
A > B > C
```

Therefore:

```text
Temperature changes probability concentration.

Temperature does not change ranking for T > 0.
```

---

# 11. Top-k Sampling

Top-k is different from temperature.

Temperature:

```text
changes probability shape
```

Top-k:

```text
limits the candidate set
```

Suppose:

```text
A -> 50%
B -> 25%
C -> 10%
D -> 8%
E -> 7%
```

If:

```text
k = 2
```

keep only:

```text
A
B
```

Discard:

```text
C
D
E
```

Then renormalize:

```text
A = 50 / (50 + 25)
  = 66.7%

B = 25 / (50 + 25)
  = 33.3%
```

Now sampling occurs only between A and B.

---

# 12. Effect of Different k Values

Starting distribution:

```text
A -> 50%
B -> 25%
C -> 10%
D -> 8%
E -> 7%
```

### k = 1

Keep:

```text
A
```

After renormalization:

```text
A -> 100%
```

This behaves like greedy selection.

### k = 2

Keep:

```text
A, B
```

Distribution:

```text
A -> 66.7%
B -> 33.3%
```

### k = 3

Keep:

```text
A, B, C
```

Distribution:

```text
A -> 58.8%
B -> 29.4%
C -> 11.8%
```

Therefore:

```text
Small k -> fewer choices -> more restrictive
Large k -> more choices -> more diversity
```

---

# 13. Top-p Sampling

Top-p is also called **nucleus sampling**.

Instead of selecting a fixed number of tokens, top-p selects the smallest set of high-probability tokens whose cumulative probability reaches the chosen threshold.

Suppose:

```text
A -> 0.50
B -> 0.25
C -> 0.10
D -> 0.08
E -> 0.07
```

Set:

```text
p = 0.85
```

Sort by probability:

```text
A -> 0.50
B -> 0.25
C -> 0.10
D -> 0.08
E -> 0.07
```

Cumulative probability:

```text
A                 = 0.50
A + B             = 0.75
A + B + C         = 0.85
```

Therefore keep:

```text
A, B, C
```

and remove:

```text
D, E
```

Then renormalize:

```text
A = 0.50 / 0.85 ~= 0.588
B = 0.25 / 0.85 ~= 0.294
C = 0.10 / 0.85 ~= 0.118
```

So:

```text
A -> 58.8%
B -> 29.4%
C -> 11.8%
```

---

# 14. Top-k vs Top-p

This is a very important distinction.

## Top-k

You decide:

```text
How many tokens?
```

Example:

```text
k = 3
```

Always keep three candidates.

## Top-p

You decide:

```text
How much cumulative probability?
```

Example:

```text
p = 0.85
```

Keep enough candidates to reach 85% cumulative probability.

Therefore:

```text
Top-k -> fixed number of candidates
Top-p -> variable number of candidates
```

---

# 15. Why Top-p Is Dynamic

Consider two different probability distributions.

### Distribution A

```text
A -> 0.80
B -> 0.10
C -> 0.05
D -> 0.03
E -> 0.02
```

With:

```text
top-p = 0.90
```

we need:

```text
A + B = 0.90
```

So only:

```text
A, B
```

are kept.

### Distribution B

```text
A -> 0.25
B -> 0.20
C -> 0.18
D -> 0.15
E -> 0.12
F -> 0.10
```

With:

```text
top-p = 0.90
```

we need several tokens:

```text
0.25
+ 0.20 = 0.45
+ 0.18 = 0.63
+ 0.15 = 0.78
+ 0.12 = 0.90
```

So:

```text
A, B, C, D, E
```

are kept.

This is why top-p is adaptive.

---

# 16. Temperature vs Top-k vs Top-p

| Method | What it does | Low value | High value |
|---|---|---|---|
| Temperature | Reshapes probabilities | Sharper | Flatter |
| Top-k | Keeps K candidates | Fewer candidates | More candidates |
| Top-p | Keeps candidates until cumulative probability reaches P | Smaller candidate set | Larger candidate set |

Easy memory trick:

```text
Temperature = SHAPE
Top-k       = COUNT
Top-p       = PROBABILITY MASS
```

---

# 17. Greedy vs Sampling

| Property | Greedy | Sampling |
|---|---|---|
| Selection | Highest probability | Random according to distribution |
| Deterministic | Yes, usually | No |
| Diversity | Low | Higher |
| Predictability | High | Lower |
| Main use | Factual / deterministic continuation | Creative / varied generation |
| Random seed | Not normally important | Can affect output |

Important:

```text
Sampling does not automatically mean bad or random nonsense.

The model still uses the learned probability distribution.
```

---

# 18. Beam Search

Beam search is different from ordinary sampling.

Instead of keeping only one candidate sequence, beam search keeps several promising sequences.

Suppose:

```text
Beam width = 3
```

At the first step:

```text
A -> 0.50
B -> 0.30
C -> 0.20
```

Keep:

```text
A
B
C
```

Then expand each sequence.

For example:

```text
A -> A1, A2, A3
B -> B1, B2, B3
C -> C1, C2, C3
```

Score the resulting sequences and keep the best few.

The key idea:

```text
Greedy:
keep 1 path

Beam search:
keep multiple paths
```

---

# 19. Why Beam Search Exists

Greedy decoding makes a decision based on the current token.

Beam search tries to preserve multiple promising paths.

Conceptually:

```text
Greedy

Start
  |
  +--> A
        |
        +--> A1
              |
              +--> A1X
```

Beam search:

```text
Start
  |
  +--> A ----> A1 ----> ...
  |
  +--> B ----> B1 ----> ...
  |
  +--> C ----> C1 ----> ...
```

This gives the decoder some look-ahead.

---

# 20. Beam Search Is Not the Same as Sampling

This distinction is important.

```text
Beam Search
-> keeps multiple high-scoring sequences
-> generally uses deterministic scoring
```

```text
Sampling
-> randomly selects from a probability distribution
-> introduces stochasticity
```

Beam search is common in some sequence-generation tasks such as translation and speech generation, while modern open-ended LLM chat often relies more on sampling-based decoding or greedy-like decoding.

---

# 21. Repetition Penalty

LLMs can sometimes repeat tokens or phrases.

A decoding system can modify token scores to discourage repetition.

Conceptually:

```text
Repeated token
     |
     v
Penalty
     |
     v
Lower probability / score
```

The exact implementation differs between systems.

The purpose is:

```text
reduce unwanted repetition
```

It is another decoding control, not a change to the underlying model weights.

---

# 22. Frequency and Presence Penalties

Some APIs expose penalties based on previous token usage.

### Frequency penalty

Penalizes tokens according to how often they have appeared.

Conceptually:

```text
More occurrences
-> larger penalty
```

### Presence penalty

Penalizes a token for appearing at all.

Conceptually:

```text
Already appeared
-> apply penalty
```

The exact mathematical implementation depends on the API or inference framework.

---

# 23. A Complete Example

Suppose the model predicts:

```text
A -> 0.40
B -> 0.30
C -> 0.15
D -> 0.10
E -> 0.05
```

## Greedy

Choose:

```text
A
```

No randomness.

---

## Sampling

Possible outputs:

```text
A
B
C
D
E
```

with their corresponding probabilities.

---

## Top-k with k = 2

Keep:

```text
A, B
```

Renormalize:

```text
A = 0.40 / 0.70 = 0.571
B = 0.30 / 0.70 = 0.429
```

Sample between A and B.

---

## Top-p with p = 0.80

Cumulative:

```text
A        = 0.40
A + B    = 0.70
A+B+C    = 0.85
```

So keep:

```text
A, B, C
```

Renormalize:

```text
A = 0.40 / 0.85 = 0.471
B = 0.30 / 0.85 = 0.353
C = 0.15 / 0.85 = 0.176
```

Then sample from those three.

---

# 24. Combining Temperature and Top-p

In practical generation systems, temperature and top-p may both be used.

Conceptually:

```text
Logits
   |
   v
Temperature scaling
   |
   v
Probability distribution
   |
   v
Top-p filtering
   |
   v
Renormalization
   |
   v
Sampling
```

Example:

```text
Temperature -> controls distribution shape
Top-p       -> removes low-probability tail
Sampling    -> chooses one token probabilistically
```

The exact ordering can differ between implementations, so always check the decoding implementation when reproducing results exactly.

---

# 25. Why Combining Them Can Be Useful

Imagine:

```text
Temperature = 0.8
Top-p = 0.9
```

Conceptually:

```text
Temperature 0.8
-> make the distribution somewhat sharper
```

Then:

```text
Top-p 0.9
-> remove the low-probability tail
```

Then:

```text
Sampling
-> randomly choose among the remaining candidates
```

This can produce a balance between:

```text
focus
+
diversity
```

But the best settings depend on the model and task.

---

# 26. Important Distinction: Selection vs Sampling

The complete decoding family can be viewed as:

```text
DECODING
|
+-- Deterministic
|    |
|    +-- Greedy
|    +-- Beam Search
|
+-- Stochastic
     |
     +-- Temperature Sampling
     +-- Top-k Sampling
     +-- Top-p Sampling
```

These categories can also be combined.

For example:

```text
Temperature
+
Top-p
+
Sampling
```

is a common style of stochastic decoding.

---

# 27. Why the Same Prompt Can Produce Different Answers

Suppose the model produces:

```text
A -> 0.50
B -> 0.30
C -> 0.20
```

With greedy decoding:

```text
A
```

will normally be selected.

With sampling:

```text
A, B, or C
```

can be selected.

If sampling is repeated:

```text
Run 1 -> A
Run 2 -> B
Run 3 -> A
Run 4 -> C
```

Therefore the same prompt can produce different outputs.

This is not necessarily because the model changed.

The decoding process introduced randomness.

---

# 28. Decoding Happens at Every Token

This is extremely important.

The decoder does not choose one token for the entire answer.

It chooses:

```text
one token
    |
    v
append token
    |
    v
run next generation step
    |
    v
choose next token
    |
    v
repeat
```

Example:

```text
Prompt:
"The sky is"
```

Step 1:

```text
blue -> selected
```

Sequence becomes:

```text
"The sky is blue"
```

Step 2:

```text
today -> selected
```

Sequence becomes:

```text
"The sky is blue today"
```

Step 3:

```text
. -> selected
```

And generation continues until a stop condition.

---

# 29. Connection With KV Cache

During autoregressive generation:

```text
Token 1
   |
Token 2
   |
Token 3
   |
Token 4
   |
...
```

The model generates one new token at a time.

The KV cache stores previous key/value states so the model does not need to recompute them from scratch at every decoding step.

Conceptually:

```text
Previous K/V
     |
     v
KV Cache
     |
     v
New Query
     |
     v
Attention
     |
     v
New hidden state
     |
     v
Logits
     |
     v
Decoding strategy
     |
     v
Next token
```

So:

```text
KV Cache = computational optimization

Sampling = token-selection strategy
```

They solve different problems.

---

# 30. Full LLM Generation Pipeline

Putting the previous lessons together:

```text
User Prompt
     |
     v
Tokenization
     |
     v
Token IDs
     |
     v
Token Embeddings
     |
     v
Position Information / RoPE
     |
     v
Transformer Layers
     |
     +---- Attention
     |
     +---- KV Cache during decoding
     |
     +---- MoE FFN if architecture uses MoE
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
Temperature
     |
     v
Softmax
     |
     v
Top-k / Top-p filtering
     |
     v
Sampling / Selection
     |
     v
Next Token
     |
     v
Append to Context
     |
     v
Repeat
```

---

# 31. Common Misconceptions

## Misconception 1

```text
Sampling means completely random text.
```

Incorrect.

Sampling uses the model's probability distribution.

High-probability tokens are generally much more likely to be selected.

---

## Misconception 2

```text
Higher temperature makes the model smarter.
```

Incorrect.

Temperature is a decoding parameter.

It changes probability concentration.

---

## Misconception 3

```text
Top-k = temperature.
```

Incorrect.

```text
Temperature -> changes probability shape
Top-k       -> limits candidate count
```

---

## Misconception 4

```text
Top-p always keeps the same number of tokens.
```

Incorrect.

The number of tokens can change depending on the probability distribution.

---

## Misconception 5

```text
Greedy always produces the globally best sequence.
```

Incorrect.

Greedy chooses the best current token according to the current distribution.

It does not perform global sequence optimization.

---

## Misconception 6

```text
KV cache chooses the next token.
```

Incorrect.

KV cache is an efficiency mechanism for autoregressive attention.

The decoding strategy chooses the next token.

---

# 32. Interview Questions

### Q1. What is greedy decoding?

Greedy decoding selects the highest-probability token at every generation step.

```text
next_token = argmax(probabilities)
```

### Q2. What is sampling?

Sampling selects the next token probabilistically according to the output distribution.

### Q3. What does temperature control?

Temperature controls the concentration of the probability distribution.

```text
Low T  -> sharper
High T -> flatter
```

### Q4. What does top-k do?

It keeps the `k` highest-probability candidate tokens and removes the rest before sampling.

### Q5. What does top-p do?

It keeps the smallest set of high-probability tokens whose cumulative probability reaches the chosen threshold `p`.

### Q6. What is the difference between top-k and top-p?

```text
Top-k -> fixed number of candidates
Top-p -> variable number of candidates
```

### Q7. Is beam search sampling?

Not in the usual sense.

Beam search maintains multiple high-scoring candidate sequences rather than randomly sampling one token according to the probability distribution.

### Q8. Does temperature change token ranking?

For positive temperature, no.

### Q9. Why can sampling produce different answers for the same prompt?

Because sampling introduces stochasticity into token selection.

### Q10. Does KV cache affect the decoding strategy?

No.

KV cache improves the efficiency of autoregressive attention; it does not define how the next token is selected.

---

# 33. Final Mental Model

Remember the complete story:

```text
Transformer
     |
     v
"What tokens are plausible?"
     |
     v
Logits
     |
     v
"What probabilities should they have?"
     |
     v
Temperature + Softmax
     |
     v
Probability Distribution
     |
     v
"Which candidates should remain?"
     |
     +---- Top-k
     |
     +---- Top-p
     |
     v
"Which token should we actually choose?"
     |
     +---- Greedy
     |
     +---- Sampling
     |
     +---- Beam Search
     |
     v
Next Token
```

The most important distinction is:

```text
MODEL
-> produces logits

TEMPERATURE
-> reshapes the distribution

TOP-K
-> keeps K candidates

TOP-P
-> keeps a probability-mass-based candidate set

GREEDY
-> chooses the highest-probability token

SAMPLING
-> randomly chooses according to probabilities

BEAM SEARCH
-> keeps multiple promising sequences
```

---

# 34. Connection to Previous Lessons

You have now connected the complete generation process:

```text
Lesson 03
Tokenization
     |
     v
Lesson 04
Embeddings + Contextual Representations
     |
     v
Lesson 06
RoPE
     |
     v
Lesson 07
MoE
     |
     v
Lesson 08
Context Window + Attention + KV Cache
     |
     v
Lesson 09
Pre-training + Next-token Prediction
     |
     v
Lesson 10
Logits + Softmax + Temperature
     |
     v
Lesson 11
Sampling + Decoding
```

The central idea of the entire pipeline is:

```text
The model predicts a probability distribution over possible next tokens.

Decoding determines how that distribution becomes an actual token.
```

---

# 35. One-Page Cheat Sheet

```text
LOGITS
Raw scores produced by the LM head.

SOFTMAX
Converts logits into probabilities.

TEMPERATURE
Controls probability concentration.

Low T
-> sharper
-> more predictable

High T
-> flatter
-> more diverse

GREEDY
Choose highest-probability token.

TOP-K
Keep exactly K highest-probability candidates.

TOP-P
Keep enough candidates to reach cumulative probability P.

SAMPLING
Randomly select according to the probability distribution.

BEAM SEARCH
Keep multiple promising sequences.

KV CACHE
Speeds up autoregressive generation.
It is not a sampling method.

REPETITION PENALTY
Discourages repeated tokens or phrases.

FREQUENCY PENALTY
Penalizes tokens based on their previous frequency.

PRESENCE PENALTY
Penalizes tokens that have already appeared.
```

---

# Final Takeaway

An LLM does not simply "choose the next word."

The actual process is closer to:

```text
Input
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
Temperature / Softmax
  |
  v
Probability Distribution
  |
  v
Decoding Strategy
  |
  +--> Greedy
  |
  +--> Sampling
  |      |
  |      +--> Top-k
  |      +--> Top-p
  |
  +--> Beam Search
  |
  v
Next Token
  |
  v
Repeat
```

The single most important distinction to remember:

```text
The MODEL produces probabilities.

The DECODER decides how those probabilities become a token.
```

That distinction is fundamental to understanding LLM inference.
