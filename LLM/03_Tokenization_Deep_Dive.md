# Lesson 3: Tokenization Deep Dive

Welcome to Lesson 3 of the AI Engineering Interview Preparation series. In this lesson, we will explore one of the most critical foundational components of Large Language Models (LLMs) and modern Natural Language Processing (NLP): **Tokenization**.

Tokenization is the process of converting raw text into a sequence of discrete symbols (tokens) that a machine learning model can process. While it might seem like a simple pre-processing step, the choice of tokenization algorithm deeply impacts the vocabulary size, the model's ability to handle rare words, the context window utilization, and overall downstream performance.

---

## 1. The Out-Of-Vocabulary (OOV) Problem

Before diving into modern algorithms, we must understand the problem they were designed to solve: the **Out-Of-Vocabulary (OOV)** problem.

### 1.1 Word-Level Tokenization
Early NLP systems relied on word-level tokenization, splitting text by spaces and punctuation. 
For example: `"The quick brown fox jumps"` $\rightarrow$ `["The", "quick", "brown", "fox", "jumps"]`

**The Problem:** 
The number of possible words in any language is theoretically infinite due to morphological variations (e.g., "run", "running", "ran"), compound words, typos, and newly invented words. 
If we restrict our model's vocabulary to the top $V$ most frequent words (e.g., $V = 30,000$), any word not in this vocabulary is replaced by an `<UNK>` (Unknown) token.
If a user inputs `"I am neurodivergent"`, and `"neurodivergent"` is not in the vocabulary, the model sees `"I am <UNK>"`, losing critical semantic information. This is the OOV problem.

### 1.2 Character-Level Tokenization
To fix OOV, one could tokenize by individual characters.
`"Fox"` $\rightarrow$ `["F", "o", "x"]`

**The Problem:**
1. **Loss of Semantic Meaning:** Individual characters hold little to no semantic meaning on their own, making it much harder for the model to learn representations.
2. **Context Window Exhaustion:** A sentence of 20 words might turn into 100+ character tokens, rapidly consuming the LLM's limited context window and making self-attention computationally expensive ($O(N^2)$ complexity).

### 1.3 The Solution: Subword Tokenization
Subword tokenization strikes a balance between word-level and character-level tokenization. The core philosophy is:
**"Frequent words should be single tokens, while rare words should be broken down into meaningful subword units."**

For example, the rare word `"unhappiness"` might be tokenized into `["un", "happi", "ness"]`.
Since the subwords `"un"`, `"happi"`, and `"ness"` are frequent across the corpus, they are kept in the vocabulary. If a completely novel word is encountered, it can always be broken down into individual characters as a last resort, completely eliminating the `<UNK>` token problem.

---

## 2. Byte-Pair Encoding (BPE)

Byte-Pair Encoding (BPE) was originally a data compression algorithm introduced in 1994. In 2015, Sennrich et al. adapted it for neural machine translation, making it the de facto standard for models like GPT-2, GPT-3, GPT-4, and RoBERTa.

### 2.1 The BPE Algorithm

BPE is a frequency-based subword tokenization algorithm. It builds the vocabulary from the bottom up.

**Initialization:**
1. Pre-tokenize the training corpus into words (often using a simple space-based tokenizer).
2. Append a special end-of-word symbol (like `</w>`) to each word to preserve word boundaries.
3. Split all words into individual characters. This base set of characters forms the initial vocabulary $V$.

**Iteration:**
4. Count the frequency of all adjacent symbol pairs in the corpus.
5. Find the most frequent pair of symbols (e.g., `A` and `B`).
6. Merge this pair to create a new symbol `AB`.
7. Add `AB` to the vocabulary $V$.
8. Replace all occurrences of the pair `A B` in the corpus with the new symbol `AB`.
9. Repeat steps 4-8 for a pre-defined number of iterations (merge operations), which determines the final vocabulary size.

### 2.2 BPE Mathematical Representation & Example

Let our corpus consist of the following words and their frequencies:
- `"low</w>"` : 5
- `"lower</w>"` : 2
- `"newest</w>"` : 6
- `"widest</w>"` : 3

**Step 0: Character splits**
```text
Vocabulary: { l, o, w, e, r, n, s, t, i, d, </w> }

Corpus:
5: l o w </w>
2: l o w e r </w>
6: n e w e s t </w>
3: w i d e s t </w>
```

**Step 1: Count pairs and merge**
The most frequent pair is `e` and `s`. 
- In `"newest</w>"`: 6 times
- In `"widest</w>"`: 3 times
Total frequency of `e s` = 9.

We merge `e s` into `es`.
```text
Vocabulary: { ..., es }

Corpus:
5: l o w </w>
2: l o w e r </w>
6: n e w es t </w>
3: w i d es t </w>
```

**Step 2: Next most frequent pair**
The pair `es` and `t` appears 9 times (`6 + 3`).
Merge `es t` into `est`.
```text
Vocabulary: { ..., es, est }

Corpus:
...
6: n e w est </w>
3: w i d est </w>
```

**Step 3: Next most frequent pair**
The pair `est` and `</w>` appears 9 times.
Merge `est </w>` into `est</w>`.

This process continues until the desired vocabulary size is reached.

### 2.3 BPE ASCII Diagram

```text
[ INITIAL STATE ]
Corpus Words:   [ l, o, w, </w> ] (5) | [ n, e, w, e, s, t, </w> ] (6)
Freq Pairs:     (e, s): 9, (s, t): 9, (l, o): 7 ...

       |
       | Merge (e, s) -> es
       v

[ ITERATION 1 ]
Corpus Words:   [ l, o, w, </w> ] (5) | [ n, e, w, es, t, </w> ] (6)
Freq Pairs:     (es, t): 9, (l, o): 7 ...

       |
       | Merge (es, t) -> est
       v

[ ITERATION 2 ]
Corpus Words:   [ l, o, w, </w> ] (5) | [ n, e, w, est, </w> ] (6)
```

---

## 3. WordPiece

WordPiece was developed by Google and is famously used in BERT and Electra. While structurally similar to BPE, WordPiece differs in its merging criterion.

### 3.1 The WordPiece Algorithm

Instead of merging the most *frequently occurring* pair, WordPiece merges the pair that maximizes the likelihood of the language model's training data. In simpler terms, it evaluates pairs based on a **score** rather than raw frequency.

The score for merging symbol $A$ and symbol $B$ into $AB$ is calculated as:

$$ \text{Score}(A, B) = \frac{\text{Frequency}(A, B)}{\text{Frequency}(A) \times \text{Frequency}(B)} $$

Alternatively, expressed via probabilities:

$$ \text{Score}(A, B) = \frac{P(AB)}{P(A) P(B)} $$

**Why is this different from BPE?**
This scoring mechanism evaluates the Mutual Information between two symbols. 
If `A` and `B` appear together frequently, but `A` and `B` also appear very frequently on their own in other contexts, the denominator becomes large, lowering the score.
WordPiece prefers merging pairs where the individual parts are rare on their own but highly likely to appear together. 

### 3.2 WordPiece Prefixing (`##`)
Unlike BPE which appends `</w>` to denote word ends, WordPiece typically uses a prefix `##` to denote that a subword is part of a larger word and not the beginning of a word.

For example, the word `"unhappiness"` might be tokenized as:
`["un", "##happi", "##ness"]`
Here, `"un"` is a word starter, while `"##happi"` and `"##ness"` are continuations.

### 3.3 Algorithm Steps
1. Initialize the vocabulary with all single characters in the corpus.
2. Build a language model on the training data using the current vocabulary.
3. Evaluate all adjacent pairs and calculate the score: $\frac{P(AB)}{P(A) P(B)}$.
4. Merge the pair with the highest score.
5. Repeat until the target vocabulary size is reached.

---

## 4. SentencePiece

SentencePiece (developed by Google) addresses major engineering and linguistic limitations present in standard BPE and WordPiece implementations. Models like ALBERT, XLNet, T5, and LLaMA utilize SentencePiece (or variations of its concepts).

### 4.1 Limitations of BPE and WordPiece
1. **Pre-tokenization dependency:** BPE and WordPiece rely on an initial step to split text into words (usually by spaces).
2. **Language limitation:** Not all languages use spaces to separate words (e.g., Chinese, Japanese, Thai). Standard BPE struggles here because pre-tokenization is non-trivial.
3. **Reversibility:** In standard tokenization, converting tokens back to the original text (detokenization) can lose original formatting (like consecutive spaces).

### 4.2 The SentencePiece Approach

SentencePiece treats the input text as a raw stream of characters, **including spaces**. It does not require language-specific pre-tokenization.

1. **Space as a Character:** SentencePiece replaces spaces with a special meta-character, typically ` ` (U+2581). 
   For example: `"Hello World"` becomes `" Hello World"`.
2. **Subword Modeling:** Once the text is formatted with the meta-symbol, SentencePiece can apply either the BPE algorithm or the **Unigram Language Model** algorithm under the hood to generate the vocabulary.
3. **Lossless Tokenization:** Because spaces are preserved as a distinct character (` `), detokenization is perfectly lossless. You simply concatenate all tokens and replace ` ` with standard spaces.

### 4.3 Unigram Language Model (Often used with SentencePiece)
While SentencePiece can use BPE, it frequently utilizes the Unigram algorithm.
- **Top-Down Approach:** Unlike BPE (which starts small and merges up), Unigram starts with a massively oversized vocabulary (e.g., all words and large subwords in the corpus).
- **Pruning:** It iteratively removes (prunes) a percentage of the vocabulary (usually 20%) that causes the least increase in the overall loss of the language model.
- **Probabilistic Tokenization:** Unigram assigns multiple valid tokenizations to a single string with different probabilities. During training, this enables **Subword Regularization**—randomly choosing different tokenizations for the same text to make the model more robust.

---

## 5. Tokenization Comparison Matrix

| Feature | Byte-Pair Encoding (BPE) | WordPiece | SentencePiece |
| :--- | :--- | :--- | :--- |
| **Used By** | GPT-2, GPT-3, GPT-4, RoBERTa | BERT, Electra | T5, LLaMA, ALBERT, XLNet |
| **Direction** | Bottom-up (merging) | Bottom-up (merging) | Usually Top-down (Unigram) or BPE |
| **Merge Criterion** | Highest pair frequency | Highest likelihood $\frac{P(AB)}{P(A)P(B)}$ | Loss minimization (Unigram) |
| **Space Handling** | Pre-tokenization required | Pre-tokenization required | Treats space as character ` ` |
| **Language Agnostic**| No (relies on spaces) | No (relies on spaces) | Yes |

---

## 6. Typical Interview Questions

If you are interviewing for an AI Engineer or NLP Research Scientist role, expect deep-dive questions on tokenization. Here are common questions and how to answer them:

### Q1: Why do Large Language Models use subword tokenization instead of character or word-level tokenization?
**Answer:** Word-level tokenization suffers from the Out-Of-Vocabulary (OOV) problem, where rare or morphologically complex words are replaced by `<UNK>`, losing semantic meaning. Character-level tokenization solves OOV but creates excessively long token sequences, destroying semantic coherence and quadratically increasing the computational cost of the self-attention mechanism. Subword tokenization (like BPE) strikes the optimal balance: frequent words remain single tokens, while rare words are broken into manageable, frequent subwords, fully eliminating the `<UNK>` problem while maintaining sequence efficiency.

### Q2: What is the exact mathematical difference between how BPE and WordPiece decide which tokens to merge?
**Answer:** BPE merges based on strict **frequency**. It counts the occurrences of adjacent pairs $A$ and $B$, and merges the pair with the highest absolute count. WordPiece merges based on **likelihood score**, specifically calculating $\frac{P(AB)}{P(A)P(B)}$. WordPiece favors merging pairs that appear together frequently relative to how often they appear independently, effectively utilizing mutual information.

### Q3: Explain why SentencePiece is preferred for multilingual models.
**Answer:** BPE and WordPiece rely on a pre-tokenization step that typically splits text by spaces to define word boundaries. However, languages like Chinese, Japanese, and Thai do not use spaces to separate words. SentencePiece treats the input as a raw Unicode stream and replaces spaces with a special meta-symbol (like ` `). Because it does not rely on language-specific pre-tokenizers, it can natively process and tokenize text across any language, making it ideal for multilingual models like mT5 or LLaMA.

### Q4: What is Subword Regularization, and which tokenizer enables it?
**Answer:** Subword Regularization is a data augmentation technique where a single piece of text is tokenized in multiple different valid ways during training. This prevents the model from overfitting to one specific sequence of subwords. The **Unigram algorithm** (often implemented via SentencePiece) enables this because it is a probabilistic model. Given a string, Unigram can output a distribution of possible token sequences, allowing us to sample different tokenizations for the same input across different training epochs.

### Q5: If you train a BPE tokenizer on a corpus of medical documents, what happens if you apply it to a corpus of Shakespearean English?
**Answer:** The tokenizer will produce a highly fragmented token sequence. Because the medical BPE model merged characters based on frequencies of medical terminology (e.g., merging "steth", "os", "cope"), it will not possess the subwords common in Shakespearean text (e.g., "thou", "hath", "doth"). Consequently, Shakespearean words will be aggressively split into very small subwords or even individual characters, drastically increasing the token sequence length and potentially degrading the LLM's understanding of the text.

### Q6: Can a subword tokenizer output an `<UNK>` token?
**Answer:** Theoretically, it shouldn't, as long as the base vocabulary is initialized with every single character (or byte) that exists in the dataset. If a completely unseen word appears, it is simply split into its constituent characters. However, if a character appears during inference that was *never* seen during training (and thus isn't in the base character vocab) or if the model doesn't use fallback-to-byte (like Byte-level BPE does), an `<UNK>` token could technically be generated. Modern models use Byte-level BPE (BBPE) mapping characters to 256 bytes, guaranteeing that literally any input can be tokenized without `<UNK>`.

---
*End of Lesson 3*
