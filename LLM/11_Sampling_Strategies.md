# Lesson 11 — Sampling Strategies

## Complete LLM Generation Note

This lesson explains how an LLM chooses the next token after producing logits and probabilities.

The overall pipeline is:

```text
Prompt
   ↓
Tokenizer
   ↓
Embeddings
   ↓
Transformer
   ↓
Hidden State
   ↓
LM Head
   ↓
Logits
   ↓
Temperature
   ↓
Probability Distribution
   ↓
Sampling / Selection
   ↓
Next Token
```

---

# 1. Why Do We Need Sampling?

Suppose the model produces:

```text
Token       Probability

the           0.40
a             0.25
one           0.15
this          0.10
some          0.06
other         0.04
```

The model has multiple possible continuations.

We need a rule to select one.

Important decoding strategies include:

```text
Greedy
Random Sampling
Top-k
Top-p
Min-p
Beam Search
```

---

# 2. Greedy Decoding

The simplest strategy is:

\[
\boxed{
\text{Choose the token with maximum probability}
}
\]

For:

```text
the → 0.40
a   → 0.25
one → 0.15
```

greedy chooses:

```text
the
```

Mathematically:

\[
x_{t+1}=\arg\max_x P(x|x_{\leq t})
\]

---

# 3. Greedy Decoding Is Deterministic

If the model produces:

```text
the → 0.40
a   → 0.25
one → 0.15
```

greedy always chooses:

```text
the
```

There is no sampling randomness.

Therefore, under deterministic inference settings:

```text
Same prompt
   ↓
Same model
   ↓
Same highest-probability token
```

usually produces the same continuation.

---

# 4. Problem With Greedy Decoding

Suppose:

```text
The movie was
```

Probability distribution:

```text
great       → 0.35
excellent   → 0.32
amazing     → 0.20
good        → 0.08
terrible    → 0.05
```

Greedy chooses:

```text
great
```

But perhaps:

```text
excellent
```

would lead to a better overall continuation.

The problem is:

> Greedy only considers the current token. It does not explicitly explore alternative future sequences.

---

# 5. Random Sampling

Instead of always choosing the highest probability token, we can sample from the probability distribution.

Suppose:

```text
cat  → 0.60
dog  → 0.30
bird → 0.10
```

Sampling means:

```text
cat  → 60% chance
dog  → 30% chance
bird → 10% chance
```

Conceptually:

```text
Random number
     ↓
Probability distribution
     ↓
Selected token
```

---

# 6. Numerical Sampling Example

Suppose:

\[
r=0.72
\]

Cumulative probabilities:

```text
cat  → 0.60
dog  → 0.90
bird → 1.00
```

Intervals:

```text
0.00 ───── 0.60       → cat
0.60 ───── 0.90       → dog
0.90 ───── 1.00       → bird
```

Since:

\[
0.72\in[0.60,0.90)
\]

the selected token is:

```text
dog
```

---

# 7. Why Sampling Can Be Better

For:

```text
The sunset was
```

the model might produce:

```text
beautiful → 0.40
stunning  → 0.25
amazing   → 0.20
bright    → 0.10
strange   → 0.05
```

Greedy:

```text
always → beautiful
```

Sampling can produce:

```text
beautiful
stunning
amazing
...
```

according to their probabilities.

This creates diversity.

---

# 8. Problem With Full Sampling

A modern vocabulary can contain tens of thousands or more tokens.

Some tokens may have extremely tiny probabilities:

```text
beautiful → 0.40
stunning  → 0.25
amazing   → 0.20
bright    → 0.10
rare_token → 0.000001
```

We often do not want extremely unlikely tokens to participate in unrestricted sampling.

This motivates:

\[
\boxed{\text{Top-k}}
\]

and:

\[
\boxed{\text{Top-p / Nucleus Sampling}}
\]

as well as:

\[
\boxed{\text{Min-p}}
\]

---

# 9. Top-k Sampling

Top-k means:

> Keep only the \(k\) highest-probability tokens.

Suppose:

```text
Token       Probability

A             0.40
B             0.25
C             0.15
D             0.10
E             0.06
F             0.04
```

Set:

\[
k=3
\]

Keep:

```text
A
B
C
```

Discard:

```text
D
E
F
```

---

# 10. Top-k Requires Renormalization

After filtering:

```text
A → 0.40
B → 0.25
C → 0.15
```

Their total is:

\[
0.40+0.25+0.15=0.80
\]

The remaining probabilities must sum to 1.

Therefore:

\[
P'(A)=\frac{0.40}{0.80}=0.50
\]

\[
P'(B)=\frac{0.25}{0.80}=0.3125
\]

\[
P'(C)=\frac{0.15}{0.80}=0.1875
\]

Final distribution:

```text
A → 50.00%
B → 31.25%
C → 18.75%
```

---

# 11. Top-k Example

Suppose:

```text
A → 0.35
B → 0.25
C → 0.18
D → 0.10
E → 0.07
F → 0.05
```

Set:

\[
k=2
\]

Keep:

```text
A
B
```

Total:

\[
0.35+0.25=0.60
\]

After renormalization:

\[
P'(A)=\frac{0.35}{0.60}=0.5833
\]

\[
P'(B)=\frac{0.25}{0.60}=0.4167
\]

So:

```text
A → 58.33%
B → 41.67%
```

---

# 12. Problem With Fixed Top-k

Top-k always keeps a fixed number of tokens.

Suppose:

\[
k=10
\]

But probability distributions can have very different shapes.

### Sharp distribution

```text
A → 0.90
B → 0.03
C → 0.02
...
```

Only a few tokens are genuinely plausible.

### Flat distribution

```text
A → 0.15
B → 0.14
C → 0.13
D → 0.12
...
```

Many tokens may be plausible.

Using the same \(k\) for both cases is not always ideal.

This motivates:

\[
\boxed{\text{Top-p / Nucleus Sampling}}
\]

---

# 13. Top-p Sampling

Top-p dynamically selects the smallest set of tokens whose cumulative probability reaches at least \(p\).

Suppose:

```text
A → 0.40
B → 0.30
C → 0.15
D → 0.08
E → 0.04
F → 0.03
```

Set:

\[
p=0.90
\]

Cumulative probabilities:

```text
A       → 0.40
A+B     → 0.70
A+B+C   → 0.85
A+B+C+D → 0.93
```

The smallest set reaching 0.90 is:

```text
A
B
C
D
```

Then these candidates are renormalized and sampled.

---

# 14. Top-p Is Dynamic

The key difference:

### Top-k

```text
Always keep k tokens.
```

### Top-p

```text
Keep however many tokens are necessary
to reach cumulative probability p.
```

Therefore:

```text
Sharp distribution
      ↓
Few tokens may reach p

Flat distribution
      ↓
Many tokens may be required
```

---

# 15. Top-p Example With Sharp Distribution

Suppose:

```text
A → 0.80
B → 0.10
C → 0.05
D → 0.03
E → 0.02
```

For:

\[
p=0.90
\]

Cumulative:

```text
A     → 0.80
A+B   → 0.90
```

Keep:

```text
A
B
```

Only two tokens are needed.

---

# 16. Top-p Example With Flat Distribution

Suppose:

```text
A → 0.20
B → 0.18
C → 0.17
D → 0.15
E → 0.12
F → 0.10
G → 0.08
```

For:

\[
p=0.90
\]

many tokens may be required to reach the probability mass threshold.

Therefore top-p automatically adapts to the distribution.

---

# 17. Top-k vs Top-p

| Feature | Top-k | Top-p |
|---|---|---|
| Selection rule | Fixed number | Probability mass |
| Number of candidates | Fixed | Dynamic |
| Adapts to distribution | Less | More |
| Example | \(k=50\) | \(p=0.9\) |

Remember:

\[
\boxed{
Top-k=\text{fixed candidate count}
}
\]

\[
\boxed{
Top-p=\text{fixed probability mass}
}
\]

---

# 18. Min-p Sampling

Min-p uses the probability of the most likely token as a reference.

A simplified conceptual rule is:

\[
\boxed{
P_i\geq p_{\min}P_{\max}
}
\]

where:

- \(P_{\max}\) = highest token probability
- \(p_{\min}\) = relative probability threshold

---

# 19. Min-p Example

Suppose:

```text
A → 0.50
B → 0.25
C → 0.10
D → 0.05
E → 0.02
F → 0.01
```

Set:

\[
p_{\min}=0.2
\]

The maximum probability is:

\[
P_{\max}=0.50
\]

Threshold:

\[
0.2\times0.50=0.10
\]

Keep tokens satisfying:

\[
P_i\geq0.10
\]

Therefore:

```text
A → 0.50
B → 0.25
C → 0.10
```

The other tokens are removed, and the remaining probabilities are renormalized.

---

# 20. Why Min-p Is Useful

Min-p adapts to model confidence.

If one token dominates:

```text
A → 0.90
B → 0.03
C → 0.02
...
```

few alternatives may survive.

If the distribution is flatter:

```text
A → 0.20
B → 0.18
C → 0.15
D → 0.14
...
```

more alternatives may survive.

---

# 21. Temperature + Top-p

Sampling strategies are often combined.

Example:

```text
Temperature = 0.8
Top-p = 0.9
```

Conceptually:

```text
Logits
   ↓
Temperature scaling
   ↓
Probability distribution
   ↓
Top-p filtering
   ↓
Renormalization
   ↓
Sampling
   ↓
Next token
```

The exact implementation details can vary, but the core idea is:

> Temperature reshapes the distribution, while top-p restricts the candidates that can participate.

---

# 22. Temperature + Top-k

Similarly:

```text
Temperature = 0.7
Top-k = 50
```

Conceptually:

```text
Logits
   ↓
Temperature
   ↓
Probability distribution
   ↓
Keep top 50
   ↓
Renormalize
   ↓
Sample
```

---

# 23. Top-k = 1

If:

\[
k=1
\]

only the highest-probability token remains.

Therefore:

\[
\boxed{
Top-k,\ k=1\approx\text{Greedy}
}
\]

because the only remaining candidate is the maximum-probability token.

---

# 24. Beam Search

Beam search is different from ordinary stochastic sampling.

Instead of generating one sequence, it maintains multiple candidate sequences.

Suppose:

\[
beam\ size=2
\]

Starting token:

```text
The
```

Possible next tokens:

```text
cat  → 0.5
dog  → 0.3
bird → 0.2
```

Keep the top two:

```text
The cat
The dog
```

Then expand both candidates.

---

# 25. Beam Search Example

Suppose:

```text
The cat
```

can become:

```text
The cat sat → score 0.40
The cat ran → score 0.30
```

and:

```text
The dog
```

can become:

```text
The dog ran   → score 0.50
The dog slept → score 0.20
```

Beam search keeps the best candidate sequences according to its sequence-level scoring rule.

Conceptually:

```text
             The
          /       \
       cat         dog
      /  \        /   \
    sat  ran     ran  slept
```

With beam size 2, only the strongest partial sequences are retained at each step.

---

# 26. Why Beam Search?

Beam search attempts to find a high-scoring sequence rather than making an independent greedy choice at every position.

It has historically been useful for:

- Machine translation
- Speech recognition
- Structured generation

For open-ended modern chat generation, stochastic sampling is often preferred.

---

# 27. Beam Search vs Greedy

### Greedy

```text
Choose best token
      ↓
Continue
      ↓
Choose best token
```

### Beam Search

```text
Keep several candidate sequences
      ↓
Expand them
      ↓
Keep best candidates
      ↓
Repeat
```

Beam search therefore explores multiple paths.

---

# 28. Beam Search vs Sampling

### Sampling

Focuses on stochastic variation and diversity.

### Beam Search

Focuses on finding high-scoring candidate sequences.

They solve different generation objectives.

---

# 29. Why Beam Search Isn't Always Better

For open-ended generation, beam search can sometimes produce:

- repetitive text
- generic responses
- less diversity

Sampling is often better suited to:

- creative writing
- conversational responses
- open-ended generation

The best method still depends on the model and task.

---

# 30. Filtering vs Sampling

This distinction is essential.

Filtering asks:

> Which tokens are allowed to participate?

Sampling asks:

> Which allowed token is actually selected?

For example:

```text
Top-p
 ↓
A, B, C allowed
 ↓
Sampling
 ↓
B selected
```

Therefore:

\[
\boxed{
Filtering\neq Sampling
}
\]

---

# 31. Complete Sampling Pipeline

A modern conceptual generation pipeline is:

```text
Prompt
   ↓
Tokenizer
   ↓
Transformer
   ↓
Logits
   ↓
Temperature
   ↓
Softmax
   ↓
Top-k / Top-p / Min-p
   ↓
Renormalization
   ↓
Sampling
   ↓
Next Token
   ↓
Append Token
   ↓
KV Cache Updated
   ↓
Repeat
```

---

# 32. Numerical Comparison

Suppose:

```text
A → 0.50
B → 0.25
C → 0.12
D → 0.08
E → 0.05
```

## Greedy

Choose:

```text
A
```

## Top-k, k=2

Keep:

```text
A, B
```

Total:

\[
0.50+0.25=0.75
\]

Renormalize:

\[
A=\frac{0.50}{0.75}=0.6667
\]

\[
B=\frac{0.25}{0.75}=0.3333
\]

So:

```text
A → 66.67%
B → 33.33%
```

## Top-p, p=0.75

Cumulative:

```text
A     → 0.50
A+B   → 0.75
```

Keep:

```text
A, B
```

## Top-p, p=0.90

Cumulative:

```text
A       → 0.50
A+B     → 0.75
A+B+C   → 0.87
A+B+C+D → 0.95
```

Keep:

```text
A, B, C, D
```

This shows why top-p changes the candidate count dynamically.

---

# 33. Repetition

Suppose the model generates:

```text
The cat is nice.
The cat is nice.
The cat is nice.
```

Decoding strategy can influence repetition, but it does not completely solve the problem.

Other mechanisms include:

- repetition penalties
- frequency penalties
- presence penalties
- better prompting
- better training

These are additional generation controls.

---

# 34. Connection to KV Cache

During inference:

```text
New token
   ↓
New Q/K/V
   ↓
Use cached past K/V
   ↓
Logits
   ↓
Sampling
   ↓
Next token
```

Therefore:

```text
KV Cache + New Token
        ↓
Transformer
        ↓
Logits
        ↓
Temperature
        ↓
Top-k / Top-p / Min-p
        ↓
Sampling
        ↓
Next Token
        ↓
Update KV Cache
        ↓
Repeat
```

---

# 35. Training vs Inference

During training:

```text
Logits
  ↓
Cross-Entropy Loss
  ↓
Backpropagation
  ↓
Parameter Update
```

During inference:

```text
Logits
  ↓
Decoding Strategy
  ↓
Next Token
```

Therefore:

\[
\boxed{
\text{Training uses logits to calculate loss}
}
\]

while:

\[
\boxed{
\text{Inference uses logits to choose tokens}
}
\]

---

# 36. Essential Formulas

### Greedy

\[
\boxed{
x_{t+1}=\arg\max_x P(x|x_{\leq t})
}
\]

### Temperature

\[
\boxed{
P_i=
\frac{e^{z_i/T}}
{\sum_j e^{z_j/T}}
}
\]

### Top-k

Keep the \(k\) highest-probability tokens.

### Top-p

Choose the smallest candidate set \(S\) such that:

\[
\boxed{
\sum_{i\in S}P_i\geq p
}
\]

### Min-p

Conceptually keep tokens satisfying:

\[
\boxed{
P_i\geq p_{\min}P_{\max}
}
\]

---

# 37. Strategy Comparison

| Strategy | Main idea | Random? |
|---|---|---|
| Greedy | Highest probability | No |
| Random sampling | Sample from distribution | Yes |
| Top-k | Sample from k best tokens | Yes |
| Top-p | Sample from probability-mass nucleus | Yes |
| Min-p | Sample from tokens sufficiently probable relative to best | Yes |
| Beam search | Maintain multiple candidate sequences | Usually no |

---

# 38. Recommended Mental Model

Think of generation as a funnel:

```text
              All Vocabulary Tokens
                       │
                       ▼
                    Logits
                       │
                       ▼
                  Temperature
                       │
                       ▼
              Probability Distribution
                       │
             ┌─────────┼─────────┐
             │         │         │
           Top-k     Top-p     Min-p
             │         │         │
             └─────────┼─────────┘
                       ▼
                Allowed Tokens
                       │
                       ▼
                    Sampling
                       │
                       ▼
                  Next Token
```

---

# 39. Practical Example

Suppose:

```text
The restaurant served
```

Probability distribution:

```text
excellent → 0.40
delicious → 0.25
fresh     → 0.15
cold      → 0.08
amazing   → 0.07
strange   → 0.03
other     → 0.02
```

### Greedy

```text
excellent
```

### Top-k = 3

Candidates:

```text
excellent
delicious
fresh
```

### Top-p = 0.80

Cumulative:

```text
excellent                  0.40
excellent + delicious      0.65
+ fresh                    0.80
```

Candidates:

```text
excellent
delicious
fresh
```

### Min-p = 0.2

Maximum:

\[
0.40
\]

Threshold:

\[
0.2\times0.40=0.08
\]

Candidates:

```text
excellent
delicious
fresh
cold
```

---

# 40. Complete Inference Loop

Suppose the prompt is:

```text
The cat
```

Step 1:

```text
The cat
```

Transformer produces probabilities.

Step 2:

```text
sat
```

is selected.

Step 3:

```text
The cat sat
```

The new token is appended.

Step 4:

```text
on
```

is selected.

Step 5:

```text
The cat sat on
```

Continue until:

```text
<EOS>
```

or another stopping condition is reached.

So:

```text
Prompt
 ↓
Predict
 ↓
Append
 ↓
Predict
 ↓
Append
 ↓
Predict
 ↓
...
```

---

# 41. What You Must Remember

The generation process is:

\[
\boxed{
\text{Logits}
\rightarrow
\text{Temperature}
\rightarrow
\text{Filtering}
\rightarrow
\text{Sampling}
\rightarrow
\text{Next Token}
}
\]

### Greedy

Choose the highest-probability token.

### Random sampling

Sample from the distribution.

### Top-k

Keep the \(k\) highest-probability tokens.

### Top-p

Keep the smallest set whose cumulative probability reaches \(p\).

### Min-p

Keep tokens sufficiently probable relative to the highest-probability token.

### Beam search

Maintain multiple candidate sequences and expand them.

---

# 42. Self-Check Questions

Before moving to Lesson 12, make sure you can answer:

1. Why do we need a decoding strategy?
2. What is greedy decoding?
3. Why is greedy deterministic?
4. What is random sampling?
5. How does probability-based sampling work?
6. What is top-k sampling?
7. Why do we renormalize after top-k filtering?
8. What is the problem with a fixed \(k\)?
9. What is top-p sampling?
10. Why is top-p dynamic?
11. What is the difference between top-k and top-p?
12. What is min-p sampling?
13. How does min-p use the highest-probability token?
14. What is beam search?
15. How is beam search different from sampling?
16. Why can beam search be useful for translation?
17. Why is stochastic sampling often preferred for open-ended generation?
18. What does temperature control?
19. What is the difference between temperature and sampling?
20. What is the difference between filtering and sampling?
21. What is the complete inference loop with KV cache?
22. What happens to logits during training?
23. What happens to logits during inference?
24. Why does top-k = 1 approximately behave like greedy decoding?

---

# 43. Next Lesson — Multi-turn Conversations & Memory

Next we move from token generation to how real LLM applications maintain conversations.

We will cover:

```text
User message
      ↓
Conversation history
      ↓
Context construction
      ↓
Chat template
      ↓
LLM
      ↓
Response
      ↓
Store conversation
      ↓
Next turn
```

And importantly:

```text
Context Window
      ↓
Conversation History
      ↓
Short-term memory
      ↓
Long-term memory
      ↓
Memory retrieval
      ↓
LLM context
```

This connects directly to production memory-based chatbot architectures.
