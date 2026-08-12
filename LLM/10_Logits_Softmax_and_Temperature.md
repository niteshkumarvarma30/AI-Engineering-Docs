# Lesson 10: Logits, Softmax & Temperature

## 1. Introduction

In modern Large Language Models (LLMs), the generation of text is fundamentally a process of predicting the next token in a sequence. But how does a neural network, which essentially performs a series of matrix multiplications and non-linear transformations on dense vectors, ultimately decide which discrete word or subword to output?

This crucial transition—from high-dimensional continuous representations (embeddings/hidden states) to discrete probability distributions over a vocabulary—is orchestrated by three key components:
1.  **The Language Model Head (LM Head) / Logits**
2.  **The Softmax Function**
3.  **Temperature Scaling**

Understanding these components is absolutely essential for anyone looking to pass AI Engineering interviews, as they form the foundation of how we control and sample from generative models. In this lesson, we will explore the mathematical formulations, intuitions, and practical implications of logits, softmax, and temperature.

---

## 2. The Final Linear Layer and Raw Logits

### 2.1 The LM Head

After an input sequence passes through the multiple layers of a Transformer (e.g., self-attention, feed-forward networks, layer normalization), the model produces a final hidden state vector for the last token in the sequence. Let's denote this final hidden state vector as $h \in \mathbb{R}^d$, where $d$ is the hidden dimension of the model (e.g., 4096 for LLaMA-7B).

This dense vector $h$ encodes the contextualized representation of the entire input sequence up to that point. However, it does not directly tell us which token should come next. To map this continuous representation to our vocabulary, we use a final linear transformation called the **Language Model Head (LM Head)**.

The LM Head is simply a weight matrix $W \in \mathbb{R}^{V \times d}$, where $V$ is the vocabulary size (the total number of possible tokens the model can generate, e.g., 32,000 or 50,257).

The operation is a standard matrix-vector multiplication (often without a bias term in modern architectures):

$$ z = W \cdot h $$

### 2.2 What are Logits?

The output vector $z \in \mathbb{R}^V$ from the LM head is called the **Logits** vector. 

**Definition:** Logits are the raw, unnormalized scores output by the final linear layer of a neural network before any activation function (like Softmax) is applied.

Each element $z_i$ in the logits vector corresponds to a specific token $i$ in the vocabulary. 
- A higher positive value $z_i$ indicates that the model strongly believes token $i$ is the correct next token.
- A lower or negative value $z_i$ indicates a weak belief.

**Properties of Logits:**
1.  **Unbounded:** Logits can take any real value from $-\infty$ to $+\infty$.
2.  **Unnormalized:** The sum of all logits $\sum z_i$ does not equal 1. Therefore, they cannot be interpreted as probabilities.
3.  **Relative vs. Absolute:** The absolute magnitude of a single logit is less meaningful than its relative difference compared to other logits in the same vector.

#### ASCII Visualization: From Hidden State to Logits

```text
[Hidden State 'h'] (Dimension: d = 4)
      [ 1.2, -0.5, 0.8, 2.1 ]
                |
                v
 [LM Head Weight Matrix 'W'] (Dimension: V x d, e.g., 5 x 4)
      [ 0.1,  0.2, -0.1,  0.5 ]  --> Token 0 ("The")
      [-0.3,  0.8,  0.2, -0.2 ]  --> Token 1 ("A")
      [ 0.5,  0.1,  0.9,  0.1 ]  --> Token 2 ("Cat")
      [-0.1, -0.5, -0.3, -0.8 ]  --> Token 3 ("Dog")
      [ 0.8, -0.2,  0.5,  1.1 ]  --> Token 4 ("Apple")
                |
          (Matrix Multiply: z = Wh)
                v
      [ Logits 'z' ] (Dimension: V = 5)
      [  0.99 ]  --> "The"
      [ -0.98 ]  --> "A"
      [  1.48 ]  --> "Cat"
      [ -1.79 ]  --> "Dog"
      [  3.77 ]  --> "Apple"
```

In this simplified example, token 4 ("Apple") has the highest raw score (3.77), making it the most likely next token based purely on the logits.

---

## 3. The Softmax Function

While logits tell us which tokens are scored higher, we cannot use them to sample a token probabilistically because they are not valid probabilities. We need a function that maps the vector of real numbers $\mathbb{R}^V$ into a valid probability distribution over $V$ outcomes.

This is exactly what the **Softmax** function does.

### 3.1 Mathematical Definition

For a vector of logits $z = [z_1, z_2, ..., z_V]$, the softmax function $\sigma(z)$ outputs a vector $p = [p_1, p_2, ..., p_V]$, where each component $p_i$ is computed as:

$$ p_i = \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}} $$

Let's break down this formula:
1.  **Exponentiation ($e^{z_i}$):** By raising the mathematical constant $e$ (Euler's number, $\approx 2.718$) to the power of the logit, we achieve two things:
    *   **Positivity:** $e^x > 0$ for all real $x$. This ensures all scores become strictly positive, regardless of whether the original logit was negative or positive.
    *   **Exaggeration of Differences:** The exponential function grows very rapidly. This means that a slightly higher logit will result in a significantly larger exponentiated value. It "softly" pushes the maximum value to dominate, which is why it's called *soft*max (a differentiable, soft version of the `argmax` function).
2.  **Normalization ($\sum e^{z_j}$):** We divide each exponentiated logit by the sum of all exponentiated logits. This ensures that:
    *   $0 < p_i < 1$ for all $i$.
    *   $\sum_{i=1}^{V} p_i = 1$.

The resulting vector $p$ is a valid categorical probability distribution.

### 3.2 Numerical Example

Let's take a small subset of logits from our previous example: $z = [2.0, 1.0, 0.1]$. Let's say these correspond to tokens ["Apple", "Banana", "Cherry"].

**Step 1: Exponentiate**
*   $e^{2.0} \approx 7.389$
*   $e^{1.0} \approx 2.718$
*   $e^{0.1} \approx 1.105$

**Step 2: Sum**
*   Sum = $7.389 + 2.718 + 1.105 = 11.212$

**Step 3: Normalize (Divide by Sum)**
*   $p(\text{"Apple"}) = \frac{7.389}{11.212} \approx 0.659$ (or $65.9\%$)
*   $p(\text{"Banana"}) = \frac{2.718}{11.212} \approx 0.242$ (or $24.2\%$)
*   $p(\text{"Cherry"}) = \frac{1.105}{11.212} \approx 0.099$ (or $9.9\%$)

Notice how the logit 2.0 is only twice as large as the logit 1.0, but its resulting probability (65.9%) is almost three times larger than the probability for 1.0 (24.2%). This demonstrates the "winner-takes-all" tendency of the exponential function.

### 3.3 Translation Invariance of Softmax

A critical property of the Softmax function is **Translation Invariance**. If you add a constant $C$ to every logit in the vector, the resulting probability distribution remains entirely unchanged.

Mathematically:
$$ \text{Softmax}(z_i + C) = \frac{e^{z_i + C}}{\sum_{j} e^{z_j + C}} = \frac{e^{z_i} \cdot e^C}{\sum_{j} (e^{z_j} \cdot e^C)} = \frac{e^{z_i} \cdot e^C}{e^C \cdot \sum_{j} e^{z_j}} = \frac{e^{z_i}}{\sum_{j} e^{z_j}} = \text{Softmax}(z_i) $$

**Why is this important?**
In practice, exponentiating large numbers can lead to numerical overflow in computers (e.g., $e^{1000}$ is too large to represent in standard floating-point formats). To ensure numerical stability when computing softmax in PyTorch or TensorFlow, we subtract the maximum logit from all logits before exponentiating:

$$ z' = z - \max(z) $$

This makes the largest logit exactly 0 (since $e^0 = 1$), and all other logits negative. Exponentiating negative numbers results in values between 0 and 1, completely preventing overflow, while perfectly preserving the mathematical result of the probability distribution.

---

## 4. Temperature Scaling

Now that we have a probability distribution, we can sample a token from it. However, sometimes we want the model to be more deterministic (predictable, safe) and sometimes we want it to be more stochastic (creative, diverse).

This is where **Temperature ($T$)** comes in. Temperature is a hyperparameter used to manipulate the shape of the probability distribution *before* we sample from it.

### 4.1 Mathematical Definition

Temperature scaling modifies the softmax equation by dividing every logit $z_i$ by a scalar value $T > 0$ before exponentiation:

$$ p_i = \text{Softmax}\left(\frac{z_i}{T}\right) = \frac{e^{\frac{z_i}{T}}}{\sum_{j=1}^{V} e^{\frac{z_j}{T}}} $$

By changing the value of $T$, we change how stark the differences are between the logits.

### 4.2 How Temperature Affects the Distribution

Let's examine three regimes of Temperature: $T = 1$, $T < 1$, and $T > 1$.

#### Case 1: T = 1 (Standard Softmax)
When $T = 1$, the equation simplifies back to the standard softmax. The probabilities are exactly as the model naturally learned them.
$$ p_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}} $$

#### Case 2: T < 1 (Low Temperature: Sharpening)
When $T \to 0$ (e.g., $T = 0.1, 0.5$), we are dividing the logits by a fraction, which effectively *multiplies* their absolute values.
*   If $z = [2.0, 1.0]$, and $T = 0.1$, the scaled logits become $[20.0, 10.0]$.
*   The difference between the logits has exploded from 1.0 to 10.0.
*   When exponentiated, $e^{20}$ completely dwarfs $e^{10}$.

**Effect:** Low temperature "sharpens" or "peaks" the distribution. It takes probability mass away from the less likely tokens and gives it to the most likely tokens. As $T \to 0$, the softmax approaches the `argmax` function (one-hot vector), making the model highly deterministic and repetitive, but less prone to hallucinations.

*Use cases:* Coding tasks, extracting specific facts, JSON generation, where precision is more important than creativity.

#### Case 3: T > 1 (High Temperature: Flattening)
When $T \to \infty$ (e.g., $T = 1.5, 2.0$), we are dividing the logits by a number greater than 1, squashing them closer together.
*   If $z = [2.0, 1.0]$, and $T = 10$, the scaled logits become $[0.2, 0.1]$.
*   The difference is now only 0.1.
*   $e^{0.2} \approx 1.22$ and $e^{0.1} \approx 1.10$. They are much closer.

**Effect:** High temperature "flattens" or "smooths" the distribution. The probability distribution becomes closer to a uniform distribution as $T \to \infty$. It allows the model to sample less likely tokens more frequently, increasing randomness, creativity, and diversity, but also the likelihood of going off-topic or hallucinating.

*Use cases:* Creative writing, brainstorming, poetry generation.

### 4.3 Numerical Example of Temperature Scaling

Let's use our previous logits $z = [2.0, 1.0, 0.1]$ and see how temperature changes the probabilities.

| Token | Logit $z$ | $T=1.0$ (Prob %) | $T=0.5$ (Logits scaled: $z/0.5$) | $T=0.5$ (Prob %) | $T=2.0$ (Logits scaled: $z/2$) | $T=2.0$ (Prob %) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Apple | 2.0 | **65.9%** | 4.0 | **86.7%** | 1.0 | **45.5%** |
| Banana| 1.0 | **24.2%** | 2.0 | **11.7%** | 0.5 | **27.6%** |
| Cherry| 0.1 | **9.9%** | 0.2 | **1.6%** | 0.05 | **26.9%** |

*   **At $T=0.5$:** "Apple" jumped from 66% to 87%. The distribution became extremely confident in its top choice. "Cherry" is almost impossible to sample (1.6%).
*   **At $T=2.0$:** The probabilities flattened out significantly. "Apple" is still the most likely (45%), but "Banana" and "Cherry" are now much more competitive (~27% each). The model is much more likely to pick a "surprising" word.

### 4.4 ASCII Visualization: Temperature Effects

```text
Logits: [2.0, 1.0, 0.1]

T = 0.1 (Extremely Sharp - almost deterministic)
  |
1.0 |  *** (99.99%)
  |   *
0.5 |   *
  |   *
  |___*____________________
    Apple   Banana  Cherry


T = 1.0 (Standard)
  |
1.0 |
  |  *** (65.9%)
0.5 |   *
  |   *      *** (24.2%)
  |___*_______*_______*** (9.9%)__
    Apple   Banana  Cherry


T = 5.0 (Extremely Flat - approaches uniform)
  |
1.0 |
  |
0.5 |
  |  *** (39%) *** (32%) *** (29%)
  |___*_________*_________*_______
    Apple    Banana   Cherry
```

---

## 5. Implementation in Python/PyTorch

Understanding the math is crucial, but implementing it demonstrates engineering capability. Here is how you implement a numerically stable Softmax with Temperature in raw Python (using NumPy) and in PyTorch.

### 5.1 Python / NumPy Implementation

```python
import numpy as np

def softmax_with_temperature(logits, temperature=1.0):
    """
    Computes numerically stable softmax with temperature scaling.
    """
    if temperature == 0.0:
        # T=0 is mathematically undefined for division, but conceptually 
        # it means argmax (greedy decoding). We return a one-hot vector.
        probs = np.zeros_like(logits)
        probs[np.argmax(logits)] = 1.0
        return probs
        
    # 1. Apply Temperature Scaling
    scaled_logits = logits / temperature
    
    # 2. Numerical Stability Trick (Translation Invariance)
    # Subtract the max value to prevent np.exp() from overflowing
    max_logit = np.max(scaled_logits)
    stabilized_logits = scaled_logits - max_logit
    
    # 3. Exponentiate
    exp_logits = np.exp(stabilized_logits)
    
    # 4. Normalize
    probabilities = exp_logits / np.sum(exp_logits)
    
    return probabilities

# Example Usage
logits_arr = np.array([2.0, 1.0, 0.1])
print("T=1.0 :", softmax_with_temperature(logits_arr, temperature=1.0))
print("T=0.5 :", softmax_with_temperature(logits_arr, temperature=0.5))
print("T=2.0 :", softmax_with_temperature(logits_arr, temperature=2.0))
```

### 5.2 PyTorch Implementation

In PyTorch, we typically use built-in functions optimized for GPUs, but the logic remains identical.

```python
import torch
import torch.nn.functional as F

def sample_token(logits_tensor, temperature=1.0):
    """
    Applies temperature, calculates softmax, and samples a token.
    """
    if temperature == 0.0:
        return torch.argmax(logits_tensor, dim=-1)
        
    # Scale by temperature
    scaled_logits = logits_tensor / temperature
    
    # PyTorch's F.softmax handles the numerical stability trick internally!
    probabilities = F.softmax(scaled_logits, dim=-1)
    
    # Sample from the resulting multinomial distribution
    sampled_token_idx = torch.multinomial(probabilities, num_samples=1)
    
    return sampled_token_idx
```

---

## 6. Interactions with Other Decoding Strategies

Temperature is rarely used in isolation. In practice, it is combined with other decoding strategies to ensure high-quality text generation.

### 6.1 Temperature + Top-k Sampling
1.  Divide logits by Temperature $T$.
2.  Calculate Softmax probabilities.
3.  Sort probabilities in descending order.
4.  Keep only the top $k$ tokens (e.g., $k=50$).
5.  Set the probability of all other tokens to 0.
6.  Re-normalize the probabilities of the remaining $k$ tokens so they sum to 1.
7.  Sample.

Temperature changes the *shape* among the top candidates, while Top-k provides a hard cut-off to prevent the model from sampling completely nonsensical tail tokens when temperature is high.

### 6.2 Temperature + Top-p (Nucleus) Sampling
1.  Divide logits by Temperature $T$.
2.  Calculate Softmax probabilities.
3.  Sort probabilities in descending order.
4.  Compute the cumulative sum of the sorted probabilities.
5.  Find the cutoff index where the cumulative sum exceeds a threshold $p$ (e.g., $p=0.9$).
6.  Keep tokens up to this cutoff index (the "nucleus").
7.  Set remaining probabilities to 0 and re-normalize.
8.  Sample.

Top-p is dynamic; it might keep 50 tokens if the distribution is flat, or only 2 tokens if the distribution is very peaked. Temperature interacts strongly here: a high temperature flattens the distribution, meaning it will take *more* tokens to reach the cumulative sum $p$.

---

## 7. Typical Interview Questions

If you are interviewing for an AI Engineer, ML Engineer, or LLM Researcher role, you should be prepared to answer these questions based on this lesson:

### Q1: What is the difference between logits and probabilities?
**Answer:** Logits are the raw, unnormalized, real-valued outputs of the final linear layer of a neural network ($-\infty$ to $\infty$). Probabilities are the result of applying the Softmax function to logits, squashing them into a range between 0 and 1 such that the sum of all probabilities over the vocabulary equals 1. Logits represent absolute scores, whereas probabilities represent a normalized distribution.

### Q2: Why do we use $e$ (exponentiation) in the Softmax function?
**Answer:** Exponentiation serves two main purposes. First, it ensures that all values become strictly positive, which is a requirement for a probability distribution. Second, the exponential function $e^x$ grows non-linearly, which heavily penalizes smaller logits and significantly rewards larger ones. This exaggerates the differences between predictions, creating a "winner-takes-most" dynamic that helps the model make distinct choices rather than outputting a uniform distribution.

### Q3: How do you prevent numerical overflow when computing Softmax in PyTorch/NumPy?
**Answer:** We leverage the translation invariance property of Softmax. By subtracting the maximum logit value from all logits in the vector before exponentiating ($z' = z - \max(z)$), the largest logit becomes 0 ($e^0 = 1$) and all others become negative. Exponentiating negative numbers yields values between 0 and 1, preventing overflow (like $e^{1000}$) while mathematically yielding the exact same probability distribution.

### Q4: Explain the effect of Temperature on LLM generation. What happens when $T=0$, $T=1$, and $T \to \infty$?
**Answer:** 
*   Temperature scales the logits before the Softmax function is applied by dividing them by $T$. 
*   **$T=1$:** Standard softmax; probabilities reflect the model's natural confidence.
*   **$T < 1$ (approaching 0):** The logits are divided by a fraction, making their magnitudes larger. This sharpens the distribution. As $T \to 0$, generation becomes greedy (`argmax`), picking only the most likely token. This makes output deterministic and focused.
*   **$T > 1$ (approaching $\infty$):** The logits are divided by a large number, compressing them closer to zero. This flattens the distribution, making all tokens more equally likely. As $T \to \infty$, it approaches a uniform distribution, increasing randomness and creativity but risking nonsensical output.

### Q5: If I want an LLM to generate code, what temperature should I set and why?
**Answer:** For code generation, factual recall, or structured data output (like JSON), you want high precision, logic, and determinism. You do not want the model to be "creative" with syntax. Therefore, you should set a low temperature (e.g., $T \in [0.0, 0.3]$). $T=0$ (greedy decoding) is often the safest bet for ensuring syntactically correct code, as it always selects the highest-probability token.

### Q6: Can a temperature modification ever change the `argmax` (the most likely token)?
**Answer:** No. Temperature scaling divides all logits by a positive constant $T$. Because division by a positive constant is a monotonically increasing transformation, it preserves the relative ordering of the logits. The token with the highest logit will still have the highest scaled logit, and therefore the highest probability after Softmax, regardless of the temperature value.

---
*End of Lesson 10*
