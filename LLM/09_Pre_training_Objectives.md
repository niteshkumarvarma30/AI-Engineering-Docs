# Lesson 9: Pre-training Objectives

## 1. Introduction to Pre-training

In the lifecycle of a Large Language Model (LLM), **pre-training** is the first, most computationally expensive, and arguably the most crucial phase. During this phase, a model is trained on a massive corpus of unannotated text data (often trillions of tokens) to learn the statistical patterns of language, general world knowledge, reasoning abilities, and syntax.

The mechanism by which the model learns from this raw text is defined by its **pre-training objective** (also called the pre-training task or loss function). The choice of objective intrinsically shapes the model's architecture, what kind of context it can process, and its downstream capabilities.

In this lesson, we will deep-dive into the dominant pre-training objectives:
1. **Causal Language Modeling (CLM)**
2. **Masked Language Modeling (MLM)**
3. **Prefix Language Modeling (Prefix-LM)**
4. **Sequence-to-Sequence Modeling (Seq2Seq / Span Corruption)**

We will also thoroughly cover the mathematical underpinning of how these models are optimized using **Cross-Entropy Loss**.

---

## 2. Causal Language Modeling (CLM)

**Also known as:** Autoregressive Language Modeling, Next-Token Prediction  
**Prominent Models:** GPT (1/2/3/4), LLaMA, Chinchilla, PaLM, Mistral, Claude

### Overview
Causal Language Modeling is the standard objective for most modern generative LLMs. The task is straightforward: given a sequence of tokens, predict the very next token in the sequence. It is called "causal" because the prediction of the current token can only depend on past tokens (causes), not future ones.

### Mathematical Formulation
Given a sequence of tokens $X = (x_1, x_2, \dots, x_T)$, the joint probability of the sequence is factorized using the chain rule of probability:

$$ P(X) = \prod_{t=1}^{T} P(x_t \mid x_{<t}) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \dots, x_{t-1}) $$

The model's objective is to maximize the log-likelihood of the training corpus. For a single sequence, the log-likelihood is:

$$ \mathcal{L}_{CLM} = \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t}) $$

where $\theta$ represents the model parameters.

### Architectural Implication: Causal Masking
Because the model processes the entire sequence in parallel during training (using Transformers), we must ensure that the representation of token $x_t$ does not "cheat" by attending to $x_{t+1}$ or beyond. This is achieved via a **causal mask** (lower-triangular matrix) applied in the self-attention mechanism.

```text
Causal Mask Matrix (1 = allow attention, 0 = mask out):

      | t=1 | t=2 | t=3 | t=4 |
-------------------------------
t=1   |  1  |  0  |  0  |  0  |  <- token 1 only sees token 1
t=2   |  1  |  1  |  0  |  0  |  <- token 2 sees tokens 1, 2
t=3   |  1  |  1  |  1  |  0  |  <- token 3 sees tokens 1, 2, 3
t=4   |  1  |  1  |  1  |  1  |  <- token 4 sees tokens 1, 2, 3, 4
```

### Advantages and Disadvantages
*   **Advantages:** 
    *   Inherently suited for open-ended text generation.
    *   Zero-shot and few-shot learning capabilities emerge naturally (as demonstrated by GPT-3) since prompting is just predicting the next tokens after the prompt.
*   **Disadvantages:** 
    *   Representations are strictly unidirectional. The representation of $x_t$ cannot benefit from the rightward context, which can be limiting for tasks like classification or extractive question answering where the full context is available.

---

## 3. Masked Language Modeling (MLM)

**Also known as:** Bidirectional Modeling, Denoising Autoencoding  
**Prominent Models:** BERT, RoBERTa, ALBERT, DeBERTa

### Overview
Masked Language Modeling, popularized by BERT, takes a different approach. Instead of predicting the next token, a certain percentage of the tokens in the input sequence (typically 15%) are corrupted or masked. The model's objective is to reconstruct the original tokens based on the surrounding context—both left and right.

### The Masking Strategy
In BERT, the 15% chosen tokens are processed as follows to prevent the model from simply memorizing the `[MASK]` token:
- 80% of the time, replaced with the special `[MASK]` token.
- 10% of the time, replaced with a random token from the vocabulary.
- 10% of the time, kept as the original token (but still predicted to calculate loss).

### Mathematical Formulation
Let $\tilde{X}$ be the corrupted sequence and $M$ be the set of indices of the masked tokens. The model attempts to reconstruct the original tokens $X_M$ given the corrupted sequence. The objective is to maximize the log-likelihood of the masked tokens:

$$ \mathcal{L}_{MLM} = \sum_{m \in M} \log P_\theta(x_m \mid \tilde{X}) $$

Notice that unlike CLM, the probability of $x_m$ is conditioned on the entire sequence $\tilde{X}$, not just the preceding tokens.

### Architectural Implication: Bidirectional Attention
In MLM, there is no causal mask. The self-attention mechanism is fully unmasked, allowing every token to attend to every other token.

```text
Bidirectional Mask Matrix (1 = allow attention, 0 = mask out):

      | t=1 | t=2 | t=3 | t=4 |
-------------------------------
t=1   |  1  |  1  |  1  |  1  |
t=2   |  1  |  1  |  1  |  1  |
t=3   |  1  |  1  |  1  |  1  |
t=4   |  1  |  1  |  1  |  1  |
```

### Advantages and Disadvantages
*   **Advantages:**
    *   Produces incredibly rich, deep, bidirectional representations of language.
    *   Excels at Natural Language Understanding (NLU) tasks: text classification, named entity recognition, extractive QA.
*   **Disadvantages:**
    *   Not suitable for autoregressive generation. You cannot easily use BERT to write an essay.
    *   There is a pre-train/fine-tune discrepancy (the `[MASK]` token never appears in downstream tasks).

---

## 4. Prefix Language Modeling (Prefix-LM)

**Prominent Models:** GLM (General Language Model), UniLM

### Overview
Prefix-LM acts as a bridge between MLM and CLM. In many downstream tasks (like summarization or translation), we have a prompt (prefix) that is fully known, and we want to generate a continuation. Prefix-LM pre-trains a model to handle exactly this scenario.

The input sequence is split into two parts: a **prefix** and a **continuation**.
- Within the prefix, tokens have **bidirectional** attention (they can see each other fully, like BERT).
- Within the continuation, tokens have **causal** attention (they can only see the prefix and past tokens in the continuation, like GPT).

### Architectural Implication: Prefix Masking

```text
Let tokens 1, 2 be the prefix, and 3, 4 be the continuation.

      | t=1 | t=2 | t=3 | t=4 |
-------------------------------
t=1   |  1  |  1  |  0  |  0  |  <- Prefix sees prefix
t=2   |  1  |  1  |  0  |  0  |  <- Prefix sees prefix
t=3   |  1  |  1  |  1  |  0  |  <- Cont. 1 sees prefix + itself
t=4   |  1  |  1  |  1  |  1  |  <- Cont. 2 sees prefix + past cont.
```

### Advantages
Provides the bidirectional understanding of MLM for the prompt while maintaining the generative capabilities of CLM for the output.

---

## 5. Sequence-to-Sequence Modeling (Span Corruption)

**Also known as:** Encoder-Decoder Pre-training  
**Prominent Models:** T5 (Text-to-Text Transfer Transformer), BART

### Overview
Instead of using a decoder-only architecture (like GPT) or an encoder-only architecture (like BERT), Seq2Seq models use the full Transformer Encoder-Decoder architecture. 

The most famous objective for this is **Span Corruption**, introduced by T5.

### The Span Corruption Task
1.  Take an input text: `The quick brown fox jumps over the lazy dog.`
2.  Corrupt random spans (sequences of adjacent tokens) rather than individual tokens, replacing them with unique sentinel tokens.
    *   Input to Encoder: `The quick [X] jumps over the [Y] dog.`
3.  The target for the Decoder is to generate the masked spans sequentially, demarcated by the same sentinel tokens.
    *   Target for Decoder: `[X] brown fox [Y] lazy [Z]`

### Mathematical Formulation
Let $X_{enc}$ be the corrupted input and $Y_{dec}$ be the target sequence of spans. The model maximizes:

$$ \mathcal{L}_{Seq2Seq} = \sum_{t=1}^{|Y_{dec}|} \log P_\theta(y_t \mid Y_{<t, dec}, X_{enc}) $$

### Architectural Implication: Encoder-Decoder Attention
- The **Encoder** uses bidirectional attention (like BERT) over the corrupted input.
- The **Decoder** uses causal attention (like GPT) over the generated targets, and **cross-attention** to look back at the Encoder's hidden states.

### Advantages
*   Extremely versatile. T5 frames every NLP task (translation, classification, summarization) as a text-to-text problem.
*   Highly effective for tasks that naturally map to Seq2Seq, such as machine translation and summarization.

---

## 6. The Mathematics of Cross-Entropy Loss over a Vocabulary

Regardless of whether the objective is CLM, MLM, or Seq2Seq, the model must ultimately predict tokens from a fixed **vocabulary** of size $V$ (e.g., $V = 50,257$ for GPT-2). The loss function used to optimize these predictions is **Categorical Cross-Entropy Loss**.

### Step 1: Logits
At the final layer of the neural network, for a given position $t$, the model outputs a dense vector $h_t \in \mathbb{R}^d$ (where $d$ is the hidden dimension). 
This vector is multiplied by a language modeling head (a weight matrix $W \in \mathbb{R}^{V \times d}$) to produce **logits** $z \in \mathbb{R}^V$:

$$ z = W h_t + b $$

The vector $z$ contains an unnormalized score for every single word/token in the vocabulary.

### Step 2: Softmax Probabilities
To convert the raw logits $z$ into a valid probability distribution over the vocabulary, we apply the **softmax function**:

$$ \hat{y}_i = P(x_t = i \mid \text{context}) = \frac{\exp(z_i)}{\sum_{j=1}^{V} \exp(z_j)} $$

Where:
*   $\hat{y}_i$ is the predicted probability that the next token is token $i$.
*   $z_i$ is the logit score for token $i$.
*   The denominator ensures all probabilities sum to 1.

### Step 3: Cross-Entropy Loss
In training, we have a ground-truth target token for position $t$. We represent this true target as a **one-hot encoded vector** $y \in \mathbb{R}^V$, where $y_c = 1$ for the correct token index $c$, and $y_i = 0$ for all $i \neq c$.

The Categorical Cross-Entropy Loss between the true distribution $y$ and the predicted distribution $\hat{y}$ is:

$$ \mathcal{L} = - \sum_{i=1}^{V} y_i \log(\hat{y}_i) $$

Because $y$ is a one-hot vector (only $y_c = 1$), the sum collapses to just the negative log probability of the **correct** target token:

$$ \mathcal{L} = - \log(\hat{y}_c) = - \log \left( \frac{\exp(z_c)}{\sum_{j=1}^{V} \exp(z_j)} \right) $$

### Gradient Descent Intuition
*   To minimize $\mathcal{L}$, the model must **maximize** $\hat{y}_c$.
*   To maximize $\hat{y}_c$, the model must increase the logit $z_c$ (the score for the correct token) while simultaneously decreasing the logits $z_j$ for all incorrect tokens (because they appear in the denominator of the softmax).

### Practical Considerations: Perplexity
In language modeling, you will often see the metric **Perplexity (PPL)**. Mathematically, perplexity is simply the exponentiated average cross-entropy loss over a sequence:

$$ \text{Perplexity} = \exp\left( \frac{1}{T} \sum_{t=1}^{T} -\log P(x_t \mid x_{<t}) \right) $$

Intuitively, perplexity represents the effective branching factor. A perplexity of 10 means the model is as confused as if it had to guess uniformly among 10 possible next words. Lower is better.

---

## 7. Contrast: CLM vs. MLM vs. Seq2Seq

| Feature | Causal LM (GPT) | Masked LM (BERT) | Seq2Seq (T5) |
| :--- | :--- | :--- | :--- |
| **Architecture** | Decoder-only | Encoder-only | Encoder-Decoder |
| **Context** | Unidirectional (Left-to-Right) | Bidirectional (Left & Right) | Enc: Bidirectional, Dec: Unidirectional |
| **Training Task** | Predict next token | Predict masked tokens | Generate masked spans |
| **Primary Strength** | Generation, Few-shot prompting | NLU, Classification, Embeddings | Translation, Summarization |
| **Scaling Laws** | Scales predictably to trillions of parameters | Diminishing returns at massive scale | Scales well, but more computationally heavy |

---

## 8. Typical Interview Questions

### Q1: Why do modern LLMs (like GPT-4, LLaMA) overwhelmingly prefer Causal Language Modeling over Masked Language Modeling despite MLM having bidirectional context?
**Expert Answer:** 
While MLM yields superior representations for understanding tasks (NLU) because of bidirectional context, CLM won the architectural convergence for several reasons:
1. **Generative Utility:** The primary use case for modern LLMs is generation (chat, code, writing). CLM is natively aligned with generation, whereas MLM requires expensive Gibbs sampling or auxiliary decoders to generate text.
2. **Zero/Few-Shot Capabilities:** As OpenAI showed with GPT-3, framing tasks as "next-token prediction" naturally enables in-context learning. You can prompt a CLM to do classification, translation, or QA without updating weights. MLM requires fine-tuning specific task heads for each new task.
3. **Scaling Laws:** Empirical evidence shows that auto-regressive models scale more smoothly and predictably with compute, data, and parameters. The lack of bidirectional context is compensated for by sheer scale—a sufficiently large model learns to infer deep context even strictly left-to-right.

### Q2: How does the computational complexity of Cross-Entropy Loss scale with vocabulary size? How can this bottleneck be mitigated?
**Expert Answer:**
The computational complexity of the final projection and softmax scales linearly with vocabulary size $V$, specifically $\mathcal{O}(d \times V)$ where $d$ is the hidden dimension. As vocabularies grow (e.g., modern models using 100k+ tokens to support multilingualism), this final matrix multiplication and the softmax denominator calculation become a major bottleneck in both memory and compute.
*Mitigations include:*
1. **Vocabulary Truncation/Tying:** Sharing weights between the input embedding matrix and the output language modeling head (Weight Tying).
2. **Adaptive Softmax / Hierarchical Softmax:** Grouping tokens into clusters to avoid computing the full denominator.
3. **Optimized Kernels:** Using hardware-aware kernel implementations (like FlashAttention techniques applied to cross-entropy, e.g., Flash-Cross-Entropy) to fuse operations and avoid materializing the massive $B \times T \times V$ logit tensor in GPU memory.

### Q3: Explain how T5's Span Corruption differs from BERT's Masked Language Modeling.
**Expert Answer:**
BERT's MLM corrupts individual tokens (or occasionally whole words) and the model predicts the original token at the *exact same position* in the sequence using an Encoder-only architecture. 
T5's Span Corruption targets sequences of adjacent tokens (spans). It replaces a span of multiple tokens with a single sentinel token in the Encoder. The Decoder is then tasked with generating the missing span autoregressively. This forces the model to learn not just fill-in-the-blank, but actual generative sequencing, bridging the gap between NLU and generation tasks.

### Q4: If you were to pre-train a model specifically for code completion (like GitHub Copilot), what objective would you use and why?
**Expert Answer:**
For code completion, where we often have context both before the cursor and after the cursor, a standard CLM is suboptimal because it can't see the rightward context.
Instead, the **Fill-In-The-Middle (FIM)** objective is preferred. 
FIM takes a causal language model but restructures the training data. A document is split into Prefix, Middle, and Suffix. The data is reformatted as: `<PRE> Prefix <SUF> Suffix <MID> Middle`. 
The model is trained with standard causal next-token prediction on this rearranged sequence. This preserves the efficient, autoregressive generative capabilities of a decoder-only architecture while teaching the model to condition its generation (the Middle) on both the Prefix and the Suffix.

### Q5: Can you explain the gradient of the Cross Entropy Loss with respect to the pre-softmax logits?
**Expert Answer:**
Yes, it has a very elegant and intuitive form. 
Let $z$ be the vector of logits, $\hat{y} = \text{softmax}(z)$ be the predicted probabilities, and $y$ be the one-hot encoded true label.
The gradient of the cross-entropy loss $\mathcal{L}$ with respect to a specific logit $z_i$ is:
$$ \frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i $$
**Intuition:** 
* If $i$ is the correct class ($y_i = 1$), the gradient is $\hat{y}_i - 1$. Since $\hat{y}_i \in (0, 1)$, this gradient is always negative. Subtracting this negative gradient during gradient descent pushes $z_i$ higher.
* If $i$ is an incorrect class ($y_i = 0$), the gradient is $\hat{y}_i - 0 = \hat{y}_i$. This gradient is positive. Subtracting this positive gradient during gradient descent pushes $z_i$ lower.
* The magnitude of the gradient is exactly proportional to the error. If the model is very confident and wrong, the gradient is large. If it is confident and correct, the gradient approaches zero.
