# Lesson 11: Sampling Strategies and Decoding Methods

## 1. Introduction

In the realm of Large Language Models (LLMs), the neural network architecture (like the Transformer) is responsible for processing input text and producing a probability distribution over the vocabulary for the next possible token. However, generating text is an iterative process. Given a sequence of preceding tokens $x_{<t} = (x_1, x_2, \dots, x_{t-1})$, the model outputs:

$$
P(x_t \mid x_{<t}) = \text{Softmax}(\mathbf{z})
$$

where $\mathbf{z}$ represents the logits produced by the final linear layer of the model. 

The mechanism by which we *choose* the actual token $x_t$ from this probability distribution is known as **Decoding** or **Sampling**. The choice of decoding strategy fundamentally alters the perceived intelligence, creativity, and coherence of the language model. 

A poor sampling strategy can make a brilliant 100-billion parameter model sound like a broken record, constantly repeating itself, or hallucinating wildly. An optimal sampling strategy strikes a balance between coherence (staying on topic) and diversity (being creative and engaging).

This lesson explores the dominant sampling strategies, their mathematical foundations, their flaws, and why certain strategies are preferred for specific tasks like open-ended chat versus machine translation.

---

## 2. Greedy Decoding

Greedy decoding is the most straightforward, deterministic method of text generation. At each step $t$, the model simply selects the token with the highest probability.

### Mathematical Formulation

$$
x_t = \arg\max_{x \in V} P(x \mid x_{<t})
$$

where $V$ is the entire vocabulary.

### The Problem with Greedy Decoding

While intuitively simple, greedy decoding suffers from severe drawbacks in practice, particularly for open-ended generation:

1. **Repetition and Looping**: Greedy decoding often gets stuck in repetitive loops. Because it always picks the most likely token, if a sequence of tokens has a high internal transition probability (e.g., " I don't know what I don't know what I don't know..."), the model will infinitely loop.
2. **Local vs. Global Optima**: Greedy decoding is a local search heuristic. It makes the locally optimal choice at step $t$, but this might lead to a globally suboptimal sequence. A slightly less probable token at step $t$ might unlock a highly probable, coherent, and brilliant sequence of tokens at steps $t+1, t+2, \dots, T$. Greedy decoding cannot see ahead to realize this.
3. **Lack of Diversity**: In conversational AI, users expect varied and human-like responses. Greedy decoding will always produce the exact same response for a given prompt, leading to a robotic and deterministic user experience.

#### Example Scenario

Consider the probability tree for two possible sentences:
- Sentence A: "The dog barked loudly." ($0.4 \times 0.9 \times 0.9 = 0.324$)
- Sentence B: "The cat slept quietly." ($0.5 \times 0.3 \times 0.3 = 0.045$)
- Sentence C: "The cat is on the mat." ($0.5 \times 0.6 \times 0.8 = 0.24$)

If the model outputs:
Step 1: $P(\text{dog}) = 0.4$, $P(\text{cat}) = 0.5$  -> Greedy picks "cat".
Step 2: Given "cat", $P(\text{slept}) = 0.3$, $P(\text{is}) = 0.6$ -> Greedy picks "is".
Step 3: Given "is", $P(\text{on}) = 0.8$ -> Greedy picks "on".

The greedy approach yields a high probability path locally, but completely misses the globally highest probability path "The dog barked loudly" because $P(\text{dog}) < P(\text{cat})$ at step 1.

---

## 3. Temperature Scaling

Before diving into truncation strategies (Top-k, Top-p), we must understand **Temperature**, a hyperparameter used to modulate the probability distribution itself. Temperature scaling alters the logits $\mathbf{z}$ before they are passed through the softmax function.

### Mathematical Formulation

Given logits $z_i$ for each token $i \in V$, the adjusted probability $P(x_i)$ with temperature $T$ is:

$$
P(x_i) = \frac{\exp(z_i / T)}{\sum_{j \in V} \exp(z_j / T)}
$$

### Effects of Temperature

- **$T = 1.0$**: The standard softmax function. No changes to the probabilities.
- **$T < 1.0$ (e.g., $0.7$)**: The distribution becomes "sharper." High-probability tokens get even higher probabilities, and low-probability tokens get crushed to near zero. 
  - As $T \to 0$, sampling approaches Greedy Decoding.
- **$T > 1.0$ (e.g., $1.5$)**: The distribution becomes "flatter." The probabilities become more uniform, increasing the likelihood of selecting rare or unusual tokens.
  - As $T \to \infty$, the distribution approaches a uniform distribution over the entire vocabulary (completely random text).

Temperature is often used *in conjunction* with the sampling methods discussed below to fine-tune the creativity of the model.

---

## 4. Top-k Sampling

To inject randomness (and thus creativity) while avoiding completely absurd tokens, we introduce **Top-k Sampling**. Instead of sampling from the entire vocabulary, we restrict the sampling pool to the $k$ most likely tokens.

### Mathematical Formulation

1. Sort the vocabulary $V$ such that the probabilities are in descending order: $p_1 \ge p_2 \ge \dots \ge p_{|V|}$.
2. Define the subset $V^{(k)} = \{v_1, v_2, \dots, v_k\}$ containing the $k$ tokens with the highest probabilities.
3. Rescale the probabilities over this new subset to ensure they sum to 1:

$$
P'(x_i \mid x_{<t}) = 
\begin{cases} 
\frac{P(x_i \mid x_{<t})}{\sum_{x_j \in V^{(k)}} P(x_j \mid x_{<t})} & \text{if } x_i \in V^{(k)} \\
0 & \text{otherwise}
\end{cases}
$$

### ASCII Illustration of Top-k (k=4)

```text
Probability Distribution (Sorted)
---------------------------------
Token 1 (the)   | ############ 30%  <-- Included
Token 2 (a)     | ######## 20%      <-- Included
Token 3 (an)    | ###### 15%        <-- Included
Token 4 (this)  | #### 10%          <-- Included
--------------------------------- [CUT-OFF: k=4]
Token 5 (apple) | ## 5%             <-- Truncated to 0
Token 6 (car)   | # 2%              <-- Truncated to 0
...
Token N (xylop) | 0.001%            <-- Truncated to 0
```

### Flaws of Top-k

The primary flaw of Top-k sampling is that the value of $k$ is static, but the shape of the probability distribution is highly dynamic from step to step.

- **Flat Distributions**: If the model is unsure (e.g., a dozen valid synonyms are possible, each with ~5% probability), a small $k$ (like $k=5$) will unfairly truncate perfectly valid tokens.
- **Sharp Distributions**: If the model is absolutely certain (e.g., generating the second half of a common word, where the top token has 99% probability), a large $k$ (like $k=50$) will include 49 completely nonsensical tokens in the sampling pool, risking a hallucination or spelling error.

---

## 5. Top-p (Nucleus) Sampling

To address the limitations of a static $k$, Holtzman et al. introduced **Nucleus Sampling**, widely known as **Top-p Sampling**. Instead of selecting a fixed number of tokens, Top-p selects the smallest set of tokens whose cumulative probability exceeds a threshold $p$.

### Mathematical Formulation

1. Sort the vocabulary $V$ by probability in descending order: $p_1 \ge p_2 \ge \dots \ge p_{|V|}$.
2. Find the smallest index $k'$ such that the cumulative sum of probabilities is greater than or equal to $p$:

$$
\sum_{i=1}^{k'} P(x_i \mid x_{<t}) \ge p
$$

3. Define the subset $V^{(p)} = \{v_1, v_2, \dots, v_{k'}\}$.
4. Rescale the probabilities over $V^{(p)}$:

$$
P'(x_i \mid x_{<t}) = 
\begin{cases} 
\frac{P(x_i \mid x_{<t})}{\sum_{x_j \in V^{(p)}} P(x_j \mid x_{<t})} & \text{if } x_i \in V^{(p)} \\
0 & \text{otherwise}
\end{cases}
$$

### Why Top-p is Superior for Open-Ended Generation

Top-p dynamically adjusts the size of the sampling pool based on the model's confidence.

#### Case 1: Sharp Distribution (High Confidence)
If $p = 0.9$ and the top token has $P = 0.95$, the cumulative sum instantly exceeds $p$ on the very first token. The sampling pool $V^{(p)}$ will contain exactly 1 token. It falls back to greedy behavior when certainty is high.

#### Case 2: Flat Distribution (Low Confidence)
If the probabilities are spread out across 50 valid synonyms (each at 2%), the cumulative sum will slowly rise, and the algorithm will include 45 tokens in the pool to reach $p = 0.90$. It dynamically expands the pool when the model is unsure.

### ASCII Illustration of Top-p (p=0.8)

```text
Cumulative Probability Threshold: 0.8 (80%)

Token A | ################ 40% (Cum: 40%)  <-- Included
Token B | ########### 25%      (Cum: 65%)  <-- Included
Token C | ###### 15%           (Cum: 80%)  <-- Included
----------------------------------------- [CUT-OFF: threshold met]
Token D | #### 10%             (Cum: 90%)  <-- Truncated
Token E | ## 5%                (Cum: 95%)  <-- Truncated
```

---

## 6. Min-p Sampling

**Min-p Sampling** is a relatively modern innovation, popularized within the open-source language modeling community (e.g., `llama.cpp`, Text Generation WebUI). It addresses the fact that Top-p can sometimes include a "long tail" of highly improbable tokens if the threshold $p$ isn't hit quickly enough, which can lead to sudden derailments in logic.

### Mathematical Formulation

Min-p doesn't look at cumulative mass. Instead, it scales dynamically relative to the *most probable token*. We define a minimum probability ratio parameter, $p_{ratio} \in [0, 1]$.

1. Identify the maximum probability for the current step: $P_{max} = \max_{x \in V} P(x \mid x_{<t})$.
2. Calculate the threshold: $P_{threshold} = P_{max} \times p_{ratio}$.
3. Form the subset of tokens that meet this relative threshold:

$$
V^{(\text{min\_p})} = \{ x_i \in V \mid P(x_i \mid x_{<t}) \ge P_{threshold} \}
$$

4. Rescale the probabilities over $V^{(\text{min\_p})}$.

### Example Scenario
Assume $p_{ratio} = 0.1$.
- If the top token has $P_{max} = 0.8$, then the threshold is $0.8 \times 0.1 = 0.08$. Any token with less than $8\%$ probability is discarded.
- If the top token has $P_{max} = 0.2$ (the model is very unsure), the threshold is $0.2 \times 0.1 = 0.02$. Any token with at least $2\%$ probability is retained.

This acts as a dynamic Top-k, ensuring that the tokens considered are always "competitive" with the best possible token at that specific timestep.

---

## 7. Beam Search

While sampling methods (Top-k, Top-p, Min-p) inject stochasticity to produce varied text, **Beam Search** is a deterministic algorithm designed to find the globally optimal sequence (or an approximation of it) by exploring multiple paths simultaneously.

### How It Works

Instead of keeping only the single best sequence (greedy decoding), Beam Search maintains a fixed number of active candidate sequences, known as the **Beam Width ($B$)**.

At each time step $t$:
1. For each of the $B$ active sequences, the model predicts the probabilities for the next token.
2. This generates $B \times |V|$ possible new sequences.
3. The algorithm calculates the cumulative log-probability for all these sequences.
4. It sorts them and keeps only the top $B$ sequences, discarding the rest.
5. This process repeats until the sequences generate an End-of-Sequence (`<EOS>`) token or reach the maximum length.

### Mathematical Formulation

The objective is to maximize the joint probability of the sequence:

$$
x^* = \arg\max_{x_1, \dots, x_T} \prod_{t=1}^T P(x_t \mid x_{<t})
$$

To avoid numerical underflow (multiplying many small probabilities), this is computed in log-space:

$$
x^* = \arg\max_{x_1, \dots, x_T} \sum_{t=1}^T \log P(x_t \mid x_{<t})
$$

### Beam Search vs. Open-Ended Generation

Beam Search is highly effective for tasks with a clear, objective correct answer, such as:
- **Machine Translation (MT)**: There are limited ways to correctly translate a sentence; the goal is high fidelity.
- **Text Summarization**: The objective is to distill facts without hallucinating.

**Why is it less common for open-ended generation?**
In open-ended tasks (chatbots, story writing), Beam Search often produces incredibly safe, generic, and boring text. Studies have shown that human language does not sit at the absolute peak of probability distributions; humans frequently use unexpected words. Beam Search maximizes probability relentlessly, resulting in outputs like "I don't know," "I am a language model," or highly repetitive phrasing, because these are statistically the "safest" paths through the probability space. 

Furthermore, Beam Search is computationally expensive. Running $B=4$ requires evaluating the model $4$ times more than standard sampling for a given sequence length, demanding heavy memory bandwidth in production deployment.

---

## 8. Penalties: Repetition, Presence, and Frequency

While manipulating the probability distribution via Temperature, Top-p, and Top-k is effective, sometimes LLMs still exhibit pathological behaviors like repeating words or topics. To directly combat this, inference APIs offer penalty parameters that alter the logits *before* softmax is applied, based on the sequence generated so far.

### Repetition Penalty
Originally popularized in the CTRL paper, the repetition penalty scales the logit of previously generated tokens by a penalty scalar $c > 1.0$.
$$
z_i' = \begin{cases} 
z_i / c & \text{if } i \in \text{generated\_tokens, and } z_i > 0 \\
z_i \times c & \text{if } i \in \text{generated\_tokens, and } z_i \le 0
\end{cases}
$$
This statically penalizes any token that has appeared, lowering its chance of being selected again.

### Presence vs. Frequency Penalty (OpenAI Style)
Modern APIs (like OpenAI's) split this concept into two distinct additive parameters:
- **Presence Penalty ($\alpha$)**: A one-off penalty applied if the token has appeared *at all* in the generated text. It encourages the model to introduce new topics.
- **Frequency Penalty ($\beta$)**: A penalty that scales linearly with the *number of times* the token has appeared. It encourages the model to stop repeating the exact same words and phrases.

Let $c_i$ be the count of how many times token $i$ has been generated. The modified logit is:
$$
z_i' = z_i - (\alpha \times I(c_i > 0)) - (\beta \times c_i)
$$
where $I$ is an indicator function.

By carefully tuning these penalties alongside sampling strategies, engineers can force models to explore broader vocabulary spaces and maintain engaging, non-repetitive dialogue.

---

## 9. Typical Interview Questions

### Q1: Explain the difference between Top-k and Top-p sampling. Which one is generally preferred for conversational AI and why?
**Answer:** 
Top-k sampling truncates the probability distribution to the $k$ most likely tokens, regardless of their actual probability mass. Top-p (Nucleus) sampling truncates the distribution at the point where the cumulative probability of the sorted tokens exceeds $p$. 

Top-p is generally preferred for conversational AI because it is dynamic. In situations where the model is highly confident, Top-p will naturally restrict to 1 or 2 tokens, preventing hallucinations. In situations where the distribution is flat and there are many valid choices, Top-p will expand to include many tokens, preserving creativity. Top-k, being static, will either include garbage tokens when confident (if $k$ is too high) or truncate valid synonyms when unsure (if $k$ is too low).

### Q2: Why is Greedy Decoding prone to repetitive loops, and how do modern decoding strategies mitigate this?
**Answer:**
Greedy decoding always selects the $\arg\max$ token. If the model generates a sequence of tokens that has a high internal transition probability (e.g., an idiomatic phrase or a recursive pattern), greedy decoding will continually select the most probable next token, trapping itself in a local optimum that results in an infinite loop. 

Modern strategies mitigate this by injecting stochasticity (Top-p, Top-k, Temperature). By sampling from a distribution, the model can probabilistically "break out" of the loop. Additionally, inference engines employ Repetition, Frequency, and Presence Penalties, which artificially reduce the probability of previously generated tokens, forcing the model to select novel words.

### Q3: You are building a Machine Translation system. Should you use Top-p sampling with Temperature 0.8, or Beam Search? Justify your choice.
**Answer:**
Beam Search is the correct choice. Machine translation is a closed-ended task aiming for high fidelity and accuracy, not creativity. We want to find the sequence of tokens that maximizes the overall joint probability given the source text. Beam search approximates this global optimum by exploring multiple high-probability paths simultaneously. Top-p with Temperature 0.8 injects randomness, which in translation leads to incorrect terminology, dropped words, and lack of fidelity to the source text.

### Q4: How does Temperature scale the logits, and what happens mathematically as Temperature approaches zero?
**Answer:**
Temperature $T$ scales the logits $z_i$ by dividing them before applying the softmax function: $P(x_i) = \text{Softmax}(z_i / T)$. 

Mathematically, as $T \to 0$, the values of $z_i / T$ grow towards infinity. The exponentiation in the softmax function massively amplifies the largest logit relative to all others. The probability of the token with the maximum logit approaches $1.0$, and the probabilities of all other tokens approach $0.0$. Thus, as Temperature approaches zero, the sampling process becomes deterministic and identical to Greedy Decoding.

### Q5: What is Min-p sampling and what specific failure mode of Top-p does it address?
**Answer:**
Min-p sampling scales the threshold dynamically based on the probability of the most likely token (e.g., $P_{threshold} = p_{ratio} \times P_{max}$). 

It addresses the "long tail" failure mode of Top-p. In highly entropic (flat) distributions, Top-p will keep adding tokens to reach the threshold $p$, sometimes including a long tail of very low-probability (garbage) tokens. Min-p ensures that no token is ever considered unless its probability is at least some reasonable fraction (e.g., 10%) of the *best* token's probability at that specific step, preventing bizarre derailments while maintaining dynamic pool sizing.

---

*This concludes Lesson 11. Understanding these parameters is critical for AI Engineers, as they directly dictate the cost, latency, and quality of LLM applications in production.*
