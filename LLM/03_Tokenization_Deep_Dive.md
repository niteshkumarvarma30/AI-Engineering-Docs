# Lesson 03 — Tokenization Deep Dive

## Learning Objectives

By the end of this lesson, you should be able to explain:

- Why tokenization is required for LLMs
- What a token is
- Character-level, word-level, subword, and byte-level tokenization
- Why modern LLMs generally use subword/byte-level approaches
- BPE (Byte Pair Encoding)
- WordPiece
- SentencePiece
- Byte-level tokenization
- Vocabulary and token IDs
- Special tokens such as BOS, EOS, PAD, UNK, and MASK
- Padding and truncation
- Attention masks and causal masks
- Tokenizer training vs LLM training
- Encoding and decoding
- `input_ids` and `attention_mask`
- Why token count affects LLM cost, latency, memory, and context length

---

# 1. Why Do We Need Tokenization?

Neural networks operate on numerical tensors, not raw text.

A computer can receive:

```text
"I love cats"
```

but a Transformer cannot directly perform matrix multiplication on the characters in the string.

So we convert:

```text
Text
 ↓
Tokens
 ↓
Numbers
```

For example:

```text
"I love cats"

      ↓

["I", "love", "cats"]

      ↓

[42, 817, 2917]
```

The numbers are called **token IDs**.

Later:

```text
Token IDs
    ↓
Embedding lookup
    ↓
Vectors
```

---

# 2. Complete LLM Input Pipeline

Keep this pipeline in mind:

```text
                    RAW TEXT
                       │
                       ▼
                   TOKENIZER
                       │
              ┌────────┴────────┐
              ▼                 ▼
            Tokens          Token IDs
                                │
                                ▼
                         Embedding Matrix
                                │
                                ▼
                         Embedding Vectors
                                │
                                ▼
                           Transformer
```

Tokenization happens **before embeddings**.

---

# 3. What Is a Token?

A token is a unit of text selected by a tokenizer.

A token does **not necessarily equal a word**.

For example:

```text
"cat"
```

could become:

```text
["cat"]
```

while:

```text
"unbelievable"
```

could become something conceptually like:

```text
["un", "believ", "able"]
```

The exact result depends on the tokenizer and vocabulary.

---

# 4. Tokenization Strategies

The major approaches are:

```text
Character-level
      ↓
Word-level
      ↓
Subword-level
      ↓
Byte-level
```

Modern LLMs generally use some form of **subword or byte-level tokenization**.

---

# 5. Character-Level Tokenization

The simplest approach is to treat each character as a token.

Example:

```text
"cat"
```

becomes:

```text
["c", "a", "t"]
```

Then:

```text
["c", "a", "t"]
       ↓
[12, 5, 19]
```

## Advantages

- Very small vocabulary
- Can represent arbitrary text when the character set is sufficiently broad

## Problems

Sequences become very long.

For:

```text
"machine learning"
```

we may need many tokens.

Longer sequences increase Transformer computation.

---

# 6. Word-Level Tokenization

Another approach is to make each word a token.

```text
"I love cats"
```

becomes:

```text
["I", "love", "cats"]
```

This looks attractive, but there is a major problem: vocabulary size and rare/unknown words.

---

# 7. The Vocabulary Problem

Imagine a vocabulary containing:

```text
cat
dog
house
computer
...
```

What happens when the tokenizer sees:

```text
electromagnetically
```

If the complete word is not in the vocabulary, we need another strategy.

A traditional solution is:

```text
[UNK]
```

which means:

> Unknown token.

This loses useful information.

---

# 8. The Rare-Word Problem

Consider:

```text
happy
happier
happiest
unhappy
unhappiness
unhappily
```

A word-level tokenizer would need many separate vocabulary entries.

There are enormous numbers of possible word forms.

This motivates **subword tokenization**.

---

# 9. Subword Tokenization

Instead of requiring every complete word to be in the vocabulary, we break words into reusable pieces.

For example:

```text
unhappiness
```

could conceptually become:

```text
["un", "happi", "ness"]
```

Now pieces can be reused across many words.

For example:

```text
unhappy
unusual
unknown
```

can share subword pieces.

---

# 10. Why Subword Tokenization Is Powerful

It provides a useful balance:

```text
Character-level
     │
     │ Small vocabulary
     │ Very long sequences
     ▼
Subword
     │
     │ Balanced
     ▼
Word-level
     │
     │ Large vocabulary
     │ Unknown-word problem
```

Modern language models generally use tokenization strategies based on this principle.

---

# 11. BPE — Byte Pair Encoding

One of the most important tokenization algorithms is:

> **BPE — Byte Pair Encoding**

The basic idea is:

> Start with small units and repeatedly merge frequently occurring pairs.

---

# 12. Simple BPE Intuition

Suppose training data contains:

```text
low
lower
lowest
```

Initially, we could represent the strings using small units:

```text
l o w
l o w e r
l o w e s t
```

Suppose:

```text
l + o
```

occurs frequently.

BPE can merge them:

```text
lo
```

Then:

```text
lo + w
```

could become:

```text
low
```

Eventually, frequently occurring patterns become useful tokens.

---

# 13. BPE Training Process

A simplified BPE training process is:

```text
Raw training text
       ↓
Initial small vocabulary
       ↓
Count adjacent pairs
       ↓
Find a frequent/useful pair
       ↓
Merge pair
       ↓
Update vocabulary
       ↓
Repeat
       ↓
Learned vocabulary + merge rules
```

For example:

```text
"t" + "h"
```

might become:

```text
"th"
```

and later a larger frequently occurring unit may be formed.

The exact merge sequence depends on the tokenizer training data and algorithm.

---

# 14. BPE Is Not Simply "Split Long Words"

A trained BPE tokenizer does not merely use a rule like:

```text
if word is long:
    split it
```

Instead, it learns a vocabulary and merge/ranking rules from training data.

Therefore tokenization is **data-dependent**.

---

# 15. BPE Example

Suppose the tokenizer learns reusable units such as:

```text
low
er
est
```

Then it might tokenize:

```text
lower
 ↓
low + er
```

and:

```text
lowest
 ↓
low + est
```

The exact tokens depend on the learned vocabulary.

---

# 16. WordPiece

Another important tokenization method is:

> **WordPiece**

It is strongly associated with BERT-style models.

Its goal is similar:

> Build a useful subword vocabulary that represents text efficiently.

Conceptually:

```text
Word
 ↓
Subword pieces
 ↓
Token IDs
```

A tokenizer could represent:

```text
playing
```

as something conceptually like:

```text
play + ##ing
```

The `##` convention is associated with common WordPiece implementations.

The exact representation depends on the tokenizer.

---

# 17. BPE vs WordPiece

They are related but not identical.

| | BPE | WordPiece |
|---|---|---|
| Basic idea | Merge token pairs | Learn useful subword units |
| Famous association | GPT-style/tokenizer families | BERT |
| Subword vocabulary | Yes | Yes |
| Handles rare words | Better than word-level | Better than word-level |

Do not reduce the difference to:

> BPE = GPT and WordPiece = BERT.

Modern tokenizer implementations are more varied.

---

# 18. SentencePiece

Another important technology is:

> **SentencePiece**

SentencePiece can tokenize text without requiring traditional whitespace-based word segmentation.

This is useful for languages where whitespace isn't a reliable word boundary.

For example:

```text
English:
I love cats
```

has spaces between words, while many other writing systems do not use spaces in the same way.

SentencePiece can operate directly on the input sequence.

---

# 19. SentencePiece Is a Framework

A subtle but important point:

**SentencePiece is a tokenization library/framework**, not one single algorithm.

It can support approaches such as:

```text
SentencePiece
   ├── BPE
   └── Unigram
```

Therefore:

```text
SentencePiece ≠ only BPE
```

---

# 20. Byte-Level Tokenization

Modern tokenizers can also operate at the byte level.

The basic idea is:

```text
Text
 ↓
UTF-8 bytes
 ↓
Tokenization
 ↓
Tokens
```

This gives the tokenizer a robust way to represent arbitrary text.

It can be useful for:

- Rare characters
- Unicode
- Misspellings
- Code
- Unusual strings

---

# 21. Why Bytes Matter

Consider:

```text
こんにちは
```

or:

```text
🚀
```

or:

```text
some_weird_identifier_123
```

A robust tokenizer needs a way to represent text even when an exact word or character wasn't seen during tokenizer training.

Byte-level approaches provide a strong fallback mechanism.

---

# 22. Vocabulary

A tokenizer has a **vocabulary**.

Suppose:

```text
Vocabulary size = 50,000
```

The vocabulary may conceptually look like:

```text
ID      Token
----------------
0       <pad>
1       <bos>
2       <eos>
3       the
4       a
...
918     cat
1273    dog
...
```

The exact IDs are tokenizer-specific.

---

# 23. Token ID

A **token ID** is an integer representing a token in the vocabulary.

For example:

```text
"cat"
 ↓
918
```

The number `918` itself has no inherent semantic meaning.

It means:

> Use vocabulary entry 918.

It can then be used to retrieve the corresponding embedding vector.

---

# 24. Token ID → Embedding

Suppose:

```text
Token ID = 918
```

The model has an embedding matrix:

\[
E \in \mathbb{R}^{V\times d}
\]

where:

- \(V\) = vocabulary size
- \(d\) = embedding dimension

Then:

\[
E[918]
\]

retrieves the vector associated with token 918.

Conceptually:

```text
"cat"
 ↓
918
 ↓
Embedding matrix
 ↓
[0.12, -0.51, 0.83, ...]
```

This leads directly into **Lesson 04 — Embeddings & Token Representations**.

---

# 25. Token IDs Are Not Embeddings

Do not confuse:

```text
Token ID
```

with:

```text
Token embedding
```

For example:

```text
"cat"
   ↓
918                  ← Token ID
   ↓
[0.12, -0.51, ...]  ← Embedding vector
```

The ID is an integer.

The embedding is a high-dimensional vector.

---

# 26. Special Tokens

Modern tokenizers often have special tokens.

Important examples:

| Token | Meaning |
|---|---|
| BOS | Beginning of sequence |
| EOS | End of sequence |
| PAD | Padding |
| UNK | Unknown token |
| MASK | Masked token |

Not every model uses every special token.

Always inspect the specific tokenizer configuration.

---

# 27. BOS — Beginning of Sequence

BOS indicates the beginning of a sequence.

Conceptually:

```text
<BOS> I love cats
```

Some modern decoder-only models do not necessarily use a separate BOS token in the same way.

Therefore do not assume every tokenizer uses one.

---

# 28. EOS — End of Sequence

EOS represents the end of a sequence.

During generation:

```text
The cat is sleeping <EOS>
```

The generation system can stop when EOS is produced.

EOS is one possible stopping mechanism.

---

# 29. PAD — Padding Token

Batches often need sequences of equal length.

Suppose:

```text
Sequence A:
I love cats

Sequence B:
I love machine learning
```

They have different lengths.

We can pad:

```text
I love cats <PAD> <PAD>

I love machine learning
```

Now both can fit into the same tensor shape.

---

# 30. Why Padding Is Needed

Suppose a batch contains:

```text
Sequence 1 → 4 tokens
Sequence 2 → 6 tokens
Sequence 3 → 5 tokens
```

We can pad them to length 6:

```text
Sequence 1 → token token token token PAD PAD
Sequence 2 → token token token token token token
Sequence 3 → token token token token token PAD
```

Then:

```text
Batch shape = [3, 6]
```

This makes efficient tensor processing possible.

---

# 31. Attention Mask and Padding

Padding tokens should not normally contribute to attention as real tokens.

So an attention mask can indicate which positions are valid.

Example:

```text
Tokens:
I love cats <PAD> <PAD>

Mask:
1  1    1     0     0
```

Typically:

```text
1 = valid token
0 = padding
```

The exact convention can vary by library/API.

---

# 32. Padding Mask vs Causal Mask

These are different concepts.

## Padding Mask

Prevents padding positions from being treated as normal content.

```text
Real tokens → valid
PAD          → ignored
```

## Causal Mask

Prevents a token from attending to future tokens.

```text
Token 1 → sees token 1
Token 2 → sees token 1,2
Token 3 → sees token 1,2,3
```

Therefore:

\[
\boxed{
Padding\ Mask \neq Causal\ Mask
}
\]

This distinction is extremely important.

---

# 33. Causal Attention Mask

For a decoder-only LLM:

```text
             Key positions
             1   2   3   4

Query 1      ✓   ✗   ✗   ✗
Query 2      ✓   ✓   ✗   ✗
Query 3      ✓   ✓   ✓   ✗
Query 4      ✓   ✓   ✓   ✓
```

A token cannot attend to future tokens.

This allows causal language modeling.

---

# 34. Truncation

Suppose a model supports:

```text
Context length = 2048 tokens
```

but your input contains:

```text
5000 tokens
```

You cannot simply provide all 5000 tokens if the model/configuration doesn't support that length.

You may need:

```text
5000 tokens
    ↓
Truncate
    ↓
2048 tokens
```

This is called **truncation**.

---

# 35. Padding vs Truncation

These are opposite operations.

## Padding

Makes sequences longer:

```text
100 tokens
 ↓
128 tokens
```

## Truncation

Makes sequences shorter:

```text
5000 tokens
 ↓
2048 tokens
```

---

# 36. Why Token Count Matters

This is extremely important for AI engineering.

Suppose two tokenizers represent the same prompt as:

```text
Tokenizer A → 1000 tokens
Tokenizer B → 1400 tokens
```

The second tokenizer may require more:

- Context capacity
- Attention computation
- KV-cache memory during generation
- API usage/cost
- Processing time

Therefore tokenization is not merely preprocessing.

It directly affects LLM efficiency.

---

# 37. Token Count Is Not Word Count

Suppose you have:

```text
1,000 words
```

That does not necessarily mean:

```text
1,000 tokens
```

Depending on the language and tokenizer:

```text
Text length ≠ Token length
```

This matters for:

- Context windows
- Memory
- Compute
- API cost
- Latency
- KV cache
- Prompt design

---

# 38. Tokenization Across Languages

Tokenization efficiency differs across languages.

For example, the same semantic amount of information may require different token counts depending on the tokenizer and language.

This matters for:

- Multilingual models
- Context usage
- Cost
- Latency
- Model performance

---

# 39. Tokenization of Numbers

Numbers are tokenized according to the tokenizer's vocabulary and rules.

For example:

```text
123456789
```

does not necessarily become one token.

It may be represented as several pieces.

This matters because the LLM does not inherently receive a number as a mathematical integer.

It receives tokens.

---

# 40. Tokenization of Code

Code is also tokenized.

For example:

```python
print("Hello")
```

may be represented using several tokens for pieces such as:

```text
print
(
"
Hello
"
)
```

The exact tokenization depends on the tokenizer.

Tokenization efficiency therefore matters for coding models too.

---

# 41. Tokenization and Multilingual Text

A tokenizer may need to represent:

```text
English
Hindi
Chinese
Japanese
Arabic
Emoji
Code
URLs
Numbers
Symbols
```

Modern tokenizers therefore need broad coverage.

Byte-level approaches can help ensure unusual inputs can still be represented.

---

# 42. Tokenizer Training vs Model Training

This distinction is important.

There are two different training processes.

## Tokenizer Training

Learns:

```text
Vocabulary
+
Merge rules / tokenization model
```

## LLM Training

Learns:

```text
Neural-network parameters
```

Conceptually:

```text
Tokenizer training
      ↓
Tokenizer
      ↓
Token IDs
      ↓
LLM training
      ↓
Model weights
```

The tokenizer and LLM are related but are not the same thing.

---

# 43. Tokenizer vs Embedding Matrix

Another important distinction:

```text
Tokenizer
   ↓
Token ID
```

Then:

```text
Token ID
   ↓
Embedding matrix
   ↓
Vector
```

Therefore:

\[
\boxed{
Tokenizer \neq Embedding
}
\]

The tokenizer maps text to IDs.

The embedding layer maps IDs to vectors.

---

# 44. Practical Hugging Face Tokenizer

A typical Hugging Face workflow looks like:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "model-name"
)

text = "I love machine learning"

tokens = tokenizer.tokenize(text)

token_ids = tokenizer.encode(text)

print(tokens)
print(token_ids)
```

Replace `"model-name"` with the tokenizer/model you want to inspect.

---

# 45. Encoding

Encoding converts text into a token representation.

```python
token_ids = tokenizer.encode(text)
```

Conceptually:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
```

---

# 46. Decoding

Decoding performs the reverse transformation.

```python
text = tokenizer.decode(token_ids)
```

Conceptually:

```text
Token IDs
    ↓
Tokenizer
    ↓
Text
```

Example:

```text
[42, 817, 2917]
        ↓
"I love cats"
```

The exact result depends on the tokenizer.

---

# 47. Encoding vs Decoding

Remember:

```text
Text
 ↓ encode
Token IDs
 ↓ decode
Text
```

Encoding and decoding are not the same as the neural network's forward and backward passes.

They are tokenizer operations.

---

# 48. Hugging Face `tokenizer()` Output

A tokenizer can return more than token IDs.

For example:

```python
inputs = tokenizer(
    "I love cats",
    return_tensors="pt"
)
```

Depending on the tokenizer/model, the output may contain:

```text
input_ids
attention_mask
```

and potentially other fields.

---

# 49. `input_ids`

Conceptually:

```text
input_ids =
[101, 1045, 2293, 8870, 102]
```

These are token IDs.

The actual values are tokenizer-specific.

---

# 50. `attention_mask`

Example:

```text
input_ids:
[101, 1045, 2293, 8870, 102, 0, 0]

attention_mask:
[ 1,    1,    1,    1,   1, 0, 0]
```

Typically:

```text
1 = valid token
0 = padding
```

Follow the specific model/library convention.

---

# 51. Complete Practical Tokenization Pipeline

```text
Raw text
   ↓
Tokenizer
   ↓
Tokens
   ↓
Token IDs
   ↓
Attention mask
   ↓
Padding / truncation if required
   ↓
PyTorch tensor
   ↓
Embedding layer
```

Then:

```text
Embedding vectors
   ↓
Position information
   ↓
Transformer
```

---

# 52. Important AI Engineer Insight

Tokenization affects system performance.

Suppose:

```text
Tokenizer A → 1000 tokens
Tokenizer B → 1400 tokens
```

For the same prompt, the second representation can increase:

- Context usage
- Attention computation
- KV-cache memory
- API cost
- Latency

Therefore tokenization is part of **LLM systems engineering**, not just NLP preprocessing.

---

# 53. Tokenization and Context Window

Suppose a model supports:

```text
128,000 tokens
```

This means **tokens**, not words.

It does not mean:

```text
128,000 words
```

The number of characters or words that fit depends on tokenization.

Therefore:

\[
\boxed{
Context\ Window = Token\ Count
}
\]

not raw character count.

---

# 54. Important Comparison

| Concept | Meaning |
|---|---|
| Text | Human-readable input |
| Token | Unit produced by tokenizer |
| Token ID | Integer representing a token |
| Vocabulary | Mapping between token pieces and IDs |
| Tokenizer | Converts text ↔ token representation |
| Embedding | Converts token ID into a dense vector |
| Input IDs | Tensor containing token IDs |
| Attention mask | Indicates which positions are valid/attendable |
| Padding | Adds tokens to equalize sequence lengths |
| Truncation | Removes tokens beyond allowed length |

---

# 55. Complete Mental Model

```text
                     RAW TEXT
                        │
                        ▼
                    TOKENIZER
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
          TOKENS               TOKEN IDs
                                   │
                                   ▼
                           EMBEDDING MATRIX
                                   │
                                   ▼
                            EMBEDDING VECTORS
                                   │
                                   ▼
                         POSITION INFORMATION
                                   │
                                   ▼
                             TRANSFORMER
```

The tokenizer is the bridge between:

```text
Human language
      ↓
Machine-readable token representation
```

---

# 56. What You Should Be Able to Explain

You should now be able to explain:

### Why tokenization exists

Because neural networks need numerical representations.

### Why not tokenize only by words

Vocabulary becomes enormous and rare/unknown words become problematic.

### Why subwords

They balance vocabulary size and sequence length.

### BPE

Learns useful token merges from training data.

### WordPiece

Learns subword units and is strongly associated with BERT-style tokenization.

### SentencePiece

A tokenizer framework that can work without relying on whitespace word boundaries and can support methods such as BPE and Unigram.

### Byte-level tokenization

Uses byte-level representations to provide broad text coverage.

### Token ID

An integer index representing a token.

### Embedding

A learned vector corresponding to a token ID.

### Padding

Makes sequences in a batch the same length.

### Truncation

Cuts sequences that exceed an allowed length.

### Attention mask

Controls which positions are valid/attendable. Padding masks and causal masks serve different purposes.

---

# 57. Self-Check Questions

Before moving to Lesson 04, you should be able to answer:

1. Why can't a Transformer directly process raw text?
2. What is a token?
3. Why isn't a token always a complete word?
4. What problem does word-level tokenization have?
5. What is subword tokenization?
6. What is BPE?
7. What does BPE learn?
8. What is WordPiece?
9. What is SentencePiece?
10. What is byte-level tokenization?
11. What is a vocabulary?
12. What is a token ID?
13. Why does token ID `918` have no inherent semantic meaning?
14. What is the difference between a token ID and an embedding?
15. What is BOS?
16. What is EOS?
17. What is PAD?
18. What is UNK?
19. What is MASK?
20. What is padding?
21. What is truncation?
22. What is an attention mask?
23. What is the difference between padding mask and causal mask?
24. Why does token count matter for LLM inference?
25. Why can the same text produce different token counts with different tokenizers?
26. What is the difference between tokenizer training and LLM training?
27. What does `tokenizer.encode()` do?
28. What does `tokenizer.decode()` do?
29. What are `input_ids`?
30. What is `attention_mask`?

---

# 58. Roadmap Progress

```text
PHASE 1 — LLM FOUNDATIONS

01. What Are LLMs?                    ✅
        ↓
02. Generative AI: The Big Picture    ✅
        ↓
03. Tokenization Deep Dive            ✅
        ↓
04. Embeddings & Token Representations

PHASE 2 — MODERN LLM ARCHITECTURE

05. LLM Architecture Internals
06. Positional Encodings
07. Mixture of Experts
08. Context Window & Attention Patterns

PHASE 3 — LLM TRAINING & GENERATION

09. Pre-training Objectives
10. Logits, Softmax & Temperature
11. Sampling Strategies
12. Multi-turn Conversations & Memory
```

## Next Lesson

**Lesson 04 — Embeddings & Token Representations**

We will go from:

```text
Token ID
   ↓
Embedding Matrix
   ↓
Embedding Vector
   ↓
Contextual Representation
   ↓
Transformer
```

Topics will include:

- Token embeddings
- Embedding matrices
- Embedding dimensions
- Embedding lookup
- `nn.Embedding` in PyTorch
- Token representation vs contextual representation
- Positional information
- Hidden states
- How representations change through Transformer layers
- Static vs contextual embeddings
