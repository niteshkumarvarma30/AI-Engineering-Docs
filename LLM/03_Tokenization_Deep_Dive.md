# Lesson 03 — Tokenization Deep Dive

## Learning Objectives

By the end of this lesson, you should be able to explain:

- What tokenization is
- Why LLMs need tokenization
- What a token is
- Why a token is not necessarily a word
- Character-level, word-level, and subword tokenization
- What vocabulary means
- What token IDs are
- What special tokens are
- What SentencePiece is
- How SentencePiece represents whitespace
- How SentencePiece Unigram tokenization works
- How SentencePiece BPE tokenization works
- What WordPiece is
- How WordPiece differs from BPE
- What byte-level tokenization is
- Why byte-level tokenization is useful
- How BPE, WordPiece, Unigram, and byte-level approaches differ
- How tokenization connects to embeddings and Transformers
- Why token count matters for LLM context windows and cost

---

# 1. What Is Tokenization?

**Tokenization** is the process of converting raw text into smaller units called **tokens**.

An LLM does not directly process a string such as:

```text
I love machine learning.
```

Instead, the text passes through a tokenizer:

```text
Raw text
   |
   v
Tokenizer
   |
   v
Tokens
   |
   v
Token IDs
   |
   v
Embedding vectors
   |
   v
Transformer
```

A simplified example:

```text
"I love cats"
      |
      v
["I", "love", "cats"]
      |
      v
[42, 817, 2917]
```

The exact tokens and IDs depend on the tokenizer and its vocabulary.

---

# 2. Why Do LLMs Need Tokenization?

Neural networks operate on numerical data.

Raw text contains:

```text
I love AI
```

but a neural network needs numbers.

Therefore:

```text
Text
  |
  v
Tokens
  |
  v
Token IDs
  |
  v
Vectors
  |
  v
Neural network
```

Tokenization is therefore the bridge between human-readable text and numerical computation.

---

# 3. What Is a Token?

A **token** is a unit of text recognized by a tokenizer.

A token can be:

- A complete word
- Part of a word
- A punctuation mark
- A whitespace-associated piece
- A special symbol
- A byte or byte-derived piece

For example, a tokenizer might represent:

```text
unhappiness
```

as:

```text
["un", "happiness"]
```

or:

```text
["un", "happi", "ness"]
```

The exact segmentation depends on the tokenizer.

Therefore:

> A token is not necessarily the same thing as a word.

---

# 4. Tokenization Example

Consider:

```text
I love machine learning.
```

A simple word tokenizer might produce:

```text
["I", "love", "machine", "learning", "."]
```

A subword tokenizer might produce something like:

```text
["I", "love", "machine", "learning", "."]
```

For a less common word:

```text
unbelievable
```

a subword tokenizer might produce:

```text
["un", "believ", "able"]
```

The exact result depends on the vocabulary.

---

# 5. Tokenization Is Model-Specific

Different LLMs can use different tokenizers.

Therefore, the same sentence can produce different token sequences in different models.

For example:

```text
Model A
"I love AI"
      |
      v
["I", "love", "AI"]
```

while another tokenizer might produce:

```text
Model B
"I love AI"
      |
      v
["▁I", "▁love", "▁AI"]
```

The token IDs will also be different.

So you should never assume:

```text
word -> one universal token ID
```

Token IDs are specific to a tokenizer vocabulary.

---

# 6. Character-Level Tokenization

The simplest approach is to treat each character as a token.

For example:

```text
hello
```

becomes:

```text
["h", "e", "l", "l", "o"]
```

## Advantages

- Very small vocabulary
- Can represent almost any text
- No unknown-word problem in the usual sense

## Disadvantages

- Very long sequences
- More computation for the same text
- The model must learn relationships across many small units

Conceptually:

```text
hello
 |
 +-- h
 +-- e
 +-- l
 +-- l
 +-- o
```

Character-level tokenization is simple, but it is usually inefficient for modern general-purpose LLMs.

---

# 7. Word-Level Tokenization

At the other extreme, we can treat each word as a token.

For example:

```text
I love machine learning
```

becomes:

```text
["I", "love", "machine", "learning"]
```

## Advantages

- Shorter sequences
- Tokens often correspond to meaningful words

## Problems

The vocabulary can become extremely large.

Consider:

```text
run
runs
running
runner
runners
rerunning
```

A word-level tokenizer may need separate entries for many forms.

Rare and previously unseen words can also cause problems.

---

# 8. Subword Tokenization

Modern LLM tokenizers commonly use some form of **subword tokenization**.

The idea is to find a balance between characters and complete words.

For example:

```text
unhappiness
```

could be represented as:

```text
["un", "happiness"]
```

or:

```text
["un", "happi", "ness"]
```

The exact segmentation depends on the tokenizer.

The major idea is:

> Frequently useful pieces can be reused across many different words.

---

# 9. Why Subword Tokenization Is Useful

Consider:

```text
happy
happiness
unhappy
unhappiness
```

A word-level tokenizer may need four separate vocabulary entries.

A subword tokenizer can potentially reuse pieces such as:

```text
happy
ness
un
```

Conceptually:

```text
happy
un + happy
happy + ness
un + happy + ness
```

This gives the tokenizer a useful balance:

```text
Character-level
    |
    | Small vocabulary
    | Long sequences
    v
Subword-level
    |
    | Balanced vocabulary and sequence length
    v
Word-level
    |
    | Large vocabulary
    | Shorter sequences
    v
```

---

# 10. Vocabulary

A tokenizer has a **vocabulary** containing the tokens it knows.

For example, a simplified vocabulary might look like:

```text
0  -> <PAD>
1  -> <UNK>
2  -> <BOS>
3  -> <EOS>
4  -> ▁the
5  -> ▁cat
6  -> ing
7  -> ▁machine
8  -> ▁learning
```

The actual vocabulary of an LLM can contain thousands, tens of thousands, or more token entries.

The vocabulary maps:

```text
Token <-> Token ID
```

For example:

```text
"▁cat" <-> 2451
```

The exact ID is tokenizer-specific.

---

# 11. Token IDs

The Transformer does not directly receive strings such as:

```text
"cat"
```

The tokenizer maps tokens to integer IDs.

For example:

```text
"I love cats"
       |
       v
["▁I", "▁love", "▁cats"]
       |
       v
[42, 817, 2917]
```

Then the embedding layer converts these IDs into vectors:

```text
42
 |
 v
Embedding vector

817
 |
 v
Embedding vector

2917
 |
 v
Embedding vector
```

So the pipeline is:

```text
Text
  |
  v
Tokens
  |
  v
Token IDs
  |
  v
Embedding lookup
  |
  v
Vectors
  |
  v
Transformer
```

---

# 12. Tokenization and Embeddings Are Different

Do not confuse these two concepts.

## Tokenization

Answers:

> How should the text be split into tokens?

Example:

```text
"machine learning"
        |
        v
["▁machine", "▁learning"]
```

## Embedding

Answers:

> How should each token ID be represented as a numerical vector?

Example:

```text
token ID
   |
   v
[0.12, -0.43, 0.87, ...]
```

Therefore:

```text
Text
 |
 v
Tokenizer
 |
 v
Token IDs
 |
 v
Embedding Layer
 |
 v
Vectors
```

---

# 13. What Is SentencePiece?

**SentencePiece** is a tokenization library/framework designed for subword tokenization.

It can train a tokenizer directly from raw text without requiring a separate whitespace-based word segmentation step.

A simplified pipeline is:

```text
Raw text
    |
    v
SentencePiece tokenizer
    |
    v
Subword tokens
    |
    v
Token IDs
```

SentencePiece supports multiple tokenization algorithms.

Two important ones are:

```text
SentencePiece
     |
     +-- Unigram
     |
     +-- BPE
```

This distinction is important:

> SentencePiece is not itself one single tokenization algorithm. It is a framework that can implement algorithms such as Unigram and BPE.

---

# 14. Why Is SentencePiece Useful?

Traditional tokenization often starts by splitting text on spaces.

For example:

```text
I love machine learning
```

could first become:

```text
I
love
machine
learning
```

SentencePiece can instead process the raw text directly and learn useful pieces.

Conceptually:

```text
Raw text
   |
   v
SentencePiece
   |
   v
Subword pieces
```

This is particularly useful for multilingual text and languages where whitespace is not a reliable word boundary.

---

# 15. The Special Whitespace Symbol in SentencePiece

SentencePiece commonly uses the symbol:

```text
▁
```

to represent a whitespace boundary.

For example:

```text
I love AI
```

might be represented as:

```text
["▁I", "▁love", "▁AI"]
```

The `▁` is not an ordinary space character.

It is a visible marker used to preserve information about whitespace boundaries.

For example:

```text
▁machine
```

roughly represents:

```text
 machine
```

The exact tokenization depends on the trained vocabulary.

---

# 16. SentencePiece Example

Suppose the input is:

```text
I love machine learning
```

A SentencePiece tokenizer might produce:

```text
["▁I", "▁love", "▁machine", "▁learning"]
```

Then the tokenizer maps them to IDs:

```text
[42, 817, 2917, 5031]
```

Then:

```text
[42, 817, 2917, 5031]
```

is passed to the embedding layer.

The actual tokens and IDs will differ between tokenizer models.

---

# 17. SentencePiece Unigram

One of the algorithms supported by SentencePiece is the **Unigram language model** approach.

The key idea is:

> Start with a large collection of candidate subword pieces and learn which pieces are useful for representing the training data.

Suppose the word is:

```text
unhappiness
```

Possible segmentations might include:

```text
▁un + happiness
```

or:

```text
▁un + happi + ness
```

or:

```text
▁un + hap + pi + ness
```

There can be multiple possible segmentations.

The Unigram model assigns probabilities to possible pieces and evaluates possible segmentations.

Conceptually:

```text
Input word
    |
    v
Possible segmentations
    |
    v
Probability model
    |
    v
Likely segmentation
```

The tokenizer chooses a segmentation according to the learned probabilistic model.

---

# 18. How Unigram Thinks About Tokenization

Imagine a vocabulary containing:

```text
▁un
happy
happi
ness
unh
...
```

The tokenizer tries to represent the input using pieces from this vocabulary.

For:

```text
unhappiness
```

it might find:

```text
▁un + happi + ness
```

The exact result depends on the learned vocabulary and probabilities.

The important idea is:

> Unigram treats tokenization as a probabilistic segmentation problem.

---

# 19. Why Is It Called "Unigram"?

A unigram model assigns probabilities to individual token pieces.

Conceptually:

```text
P(piece)
```

The probability of a complete segmentation can be modeled approximately as the product of the probabilities of its pieces.

For example:

```text
▁un + happi + ness
```

can be represented conceptually as:

```text
P(▁un) * P(happi) * P(ness)
```

The tokenizer compares possible segmentations using the learned model.

This is a simplified explanation; actual SentencePiece Unigram training and decoding involve additional details.

---

# 20. SentencePiece BPE

SentencePiece can also use **Byte Pair Encoding (BPE)**.

BPE works through repeated merging of frequently occurring symbol pairs.

The simplified idea is:

```text
Start with small units
        |
        v
Find frequent adjacent pair
        |
        v
Merge the pair
        |
        v
Repeat
        |
        v
Learn useful subword tokens
```

---

# 21. Simple BPE Example

Imagine the training data contains many occurrences of:

```text
l o v e
```

BPE may learn:

```text
l + o
```

and merge them into:

```text
lo
```

Then:

```text
lo + v
```

can become:

```text
lov
```

Then:

```text
lov + e
```

can become:

```text
love
```

This is a simplified conceptual example.

Real BPE tokenization operates over the tokenizer's chosen initial symbol representation and learned merge rules.

---

# 22. BPE Training

Conceptually:

```text
Training corpus
      |
      v
Initial small units
      |
      v
Count frequent adjacent pairs
      |
      v
Merge the most useful pair
      |
      v
Update vocabulary
      |
      v
Repeat
```

Eventually, the tokenizer learns a vocabulary containing reusable subword pieces.

---

# 23. BPE Tokenization at Inference

After training, the tokenizer has learned merge rules.

For new text:

```text
Input text
    |
    v
Initial pieces
    |
    v
Apply learned merge rules
    |
    v
Final tokens
    |
    v
Token IDs
```

The tokenizer does not retrain while processing your prompt.

It uses the vocabulary and rules it learned during tokenizer training.

---

# 24. What Is WordPiece?

**WordPiece** is another subword tokenization algorithm.

It was popularized by models such as **BERT**.

The main goal is similar to BPE:

> Represent text using reusable subword pieces instead of requiring every complete word to be present in the vocabulary.

For example, a word such as:

```text
unhappiness
```

might be segmented conceptually as:

```text
un + ##happi + ##ness
```

The `##` convention commonly means:

> This piece continues a word rather than starting a new word.

The exact segmentation depends on the WordPiece vocabulary.

---

# 25. Why Was WordPiece Needed?

Suppose the vocabulary contains:

```text
play
##ing
##ed
##er
```

Then the tokenizer can represent:

```text
playing
```

as:

```text
play + ##ing
```

and:

```text
played
```

as:

```text
play + ##ed
```

This allows the tokenizer to reuse learned pieces.

Instead of storing every possible word:

```text
play
playing
played
player
players
...
```

the vocabulary can contain reusable components.

---

# 26. WordPiece Word-Boundary Convention

A simplified example:

```text
playing is useful
```

might become:

```text
["playing", "is", "useful"]
```

if those words exist in the vocabulary.

A rare word might become:

```text
["play", "##ing"]
```

The `##` tells us that `ing` is attached to the previous piece rather than beginning a new word.

For example:

```text
unhappiness
```

could conceptually become:

```text
["un", "##happi", "##ness"]
```

Again, the exact result depends on the trained vocabulary.

---

# 27. How WordPiece Differs from BPE

WordPiece and BPE are closely related subword approaches, but they are not identical.

A useful high-level comparison is:

```text
BPE
 |
 +-- Learn frequent pair merges
 |
 +-- Apply merge rules
```

while:

```text
WordPiece
 |
 +-- Learn a vocabulary of useful subword units
 |
 +-- Select a segmentation using the learned vocabulary/model
```

The training objectives and exact algorithms differ.

Do not treat:

```text
BPE = WordPiece
```

as correct.

---

# 28. BPE vs WordPiece: Intuition

A useful mental model:

### BPE

> "Which adjacent pieces occur frequently enough that merging them is useful?"

```text
a + b
  |
  v
ab
```

### WordPiece

> "Which subword pieces are useful for building words while improving the model's likelihood/objective?"

```text
play + ##ing
```

Both produce reusable subword units, but they learn and apply them differently.

---

# 29. WordPiece vs SentencePiece

These terms are also easy to confuse.

**WordPiece** is a subword tokenization algorithm.

**SentencePiece** is a tokenizer framework/library that supports algorithms such as:

```text
SentencePiece
    |
    +-- Unigram
    |
    +-- BPE
```

So:

```text
WordPiece
```

and:

```text
SentencePiece
```

are not the same category of thing.

A simplified taxonomy is:

```text
Subword Tokenization
       |
       +-- BPE
       |
       +-- WordPiece
       |
       +-- Unigram
```

while:

```text
SentencePiece
       |
       +-- can implement BPE
       |
       +-- can implement Unigram
```

---

# 30. What Is Byte-Level Tokenization?

**Byte-level tokenization** starts from bytes rather than Unicode characters or complete words.

Text is ultimately stored and transmitted as bytes.

For example:

```text
Text
 |
 v
UTF-8 encoding
 |
 v
Bytes
 |
 v
Tokenizer
 |
 v
Tokens
```

A byte-level tokenizer can therefore represent arbitrary text using the underlying byte representation.

This makes it possible to handle unusual characters and many kinds of input without requiring a separate unknown-token mechanism for every possible Unicode character.

---

# 31. Why Bytes?

Unicode contains a very large number of possible characters.

A tokenizer that tried to treat every possible character as a separate vocabulary item could become inefficient.

UTF-8 represents Unicode text using bytes.

For example, ordinary ASCII characters use one byte in UTF-8, while many non-ASCII characters use multiple bytes.

Conceptually:

```text
Character
    |
    v
UTF-8
    |
    v
One or more bytes
```

A byte-level tokenizer can build tokens from these byte representations.

---

# 32. Byte-Level BPE

One important approach is **byte-level BPE**.

The simplified process is:

```text
Raw text
   |
   v
UTF-8 bytes
   |
   v
Initial byte-level units
   |
   v
BPE merge rules
   |
   v
Subword tokens
   |
   v
Token IDs
```

Instead of starting from characters or words, BPE starts from a byte-level representation.

This gives the tokenizer broad coverage of arbitrary text.

---

# 33. Why Byte-Level Tokenization Is Useful

Byte-level approaches have several useful properties:

- Can represent arbitrary byte sequences
- Reduce reliance on unknown tokens
- Handle unusual Unicode text
- Work naturally across many languages
- Can represent code and symbols
- Provide a consistent base representation

However, byte-level tokenization does not mean that every byte becomes one final token.

BPE-style merging can combine bytes into larger, more useful tokens.

---

# 34. Byte-Level Does Not Mean One Byte = One Final Token

This is important.

Suppose a string is represented initially as bytes:

```text
b1 b2 b3 b4 ...
```

A byte-level BPE tokenizer can learn merges such as:

```text
b1 + b2
   |
   v
token_A
```

and:

```text
b3 + b4
   |
   v
token_B
```

Eventually:

```text
Bytes
  |
  v
Learned merges
  |
  v
Subword-like tokens
```

So the final vocabulary can contain multi-byte and multi-character patterns.

---

# 35. Byte-Level vs Character-Level

These are not the same.

### Character-level

Starts from characters:

```text
hello
 |
 +-- h
 +-- e
 +-- l
 +-- l
 +-- o
```

### Byte-level

Starts from the byte representation:

```text
hello
 |
 v
UTF-8 bytes
 |
 v
byte units
```

For ASCII text, one character generally corresponds to one UTF-8 byte.

For many non-ASCII characters, one character can correspond to multiple UTF-8 bytes.

---

# 36. Byte-Level Tokenization and Unicode

Consider:

```text
café
```

The `é` character has a Unicode representation that uses multiple bytes in UTF-8.

Conceptually:

```text
é
 |
 v
UTF-8
 |
 v
multiple bytes
```

A byte-level tokenizer can operate on that underlying byte representation.

This provides a consistent way to handle text across different writing systems.

---

# 37. Byte-Level Tokenization and Code

Byte-level approaches can also be useful for source code.

For example:

```python
print("Hello, world!")
```

contains:

- Letters
- Punctuation
- Quotes
- Parentheses
- Spaces
- Symbols

A byte-level tokenizer can represent all of these through its byte-level base representation and learned merges.

This is useful for models that process both natural language and code.

---

# 38. Important Tokenization Families

At this point, you should know several important approaches:

```text
Tokenization
|
+-- Character-level
|
+-- Word-level
|
+-- Subword
|     |
|     +-- BPE
|     |
|     +-- WordPiece
|     |
|     +-- Unigram
|
+-- Byte-level
      |
      +-- Byte-level BPE
```

These approaches can also be combined with different vocabulary and normalization strategies.

---

# 39. BPE vs WordPiece vs Unigram vs Byte-Level BPE

| Method | Core idea | Typical representation |
|---|---|---|
| BPE | Repeatedly merge frequent pairs | `play`, `ing`, etc. |
| WordPiece | Learn useful subword vocabulary and segment words | `play`, `##ing` |
| Unigram | Probabilistic segmentation using learned token probabilities | `▁un`, `happi`, `ness` |
| Byte-level BPE | Start from bytes, then apply BPE merges | Byte-derived subword tokens |

These are conceptual summaries. Real tokenizer implementations can include additional normalization, pre-tokenization, special-token, and decoding rules.

---

# 40. SentencePiece vs WordPiece vs Byte-Level BPE

A common source of confusion is mixing the framework with the algorithm.

### SentencePiece

A tokenizer framework/library that supports algorithms such as:

```text
Unigram
BPE
```

### WordPiece

A subword tokenization algorithm associated strongly with BERT-style models.

### Byte-Level BPE

A BPE approach whose initial representation is based on bytes.

A simplified map:

```text
                     Tokenization
                          |
             +------------+------------+
             |            |            |
             v            v            v
            BPE       WordPiece     Unigram
             |                         |
             |                         |
             v                         v
      Can be implemented         Can be implemented
      by SentencePiece          by SentencePiece
```

Byte-level BPE is another BPE variant:

```text
Byte-level representation
          |
          v
         BPE
          |
          v
       Tokens
```

---

# 41. SentencePiece Unigram vs WordPiece

Both can produce pieces that look similar.

For example, a word may be split conceptually as:

```text
un + happi + ness
```

But the algorithms behind the segmentation differ.

### Unigram

Treats segmentation as a probabilistic problem:

```text
Possible segmentations
        |
        v
Learned probabilities
        |
        v
Selected segmentation
```

### WordPiece

Uses a learned subword vocabulary and a WordPiece-specific objective/segmentation procedure.

The important lesson is:

> Similar-looking token pieces do not mean the tokenizers use the same algorithm.

---

# 42. SentencePiece Whitespace vs WordPiece `##`

These symbols mean different things.

SentencePiece commonly uses:

```text
▁
```

to mark a whitespace boundary.

Example:

```text
["▁hello", "▁world"]
```

WordPiece commonly uses:

```text
##
```

to indicate that a piece continues the previous word.

Example:

```text
["play", "##ing"]
```

Therefore:

```text
▁
```

and:

```text
##
```

are tokenizer-specific conventions.

---

# 43. Special Tokens

Tokenizers often include special tokens used by the model.

Common examples include:

```text
<BOS>
```

Beginning of sequence.

```text
<EOS>
```

End of sequence.

```text
<PAD>
```

Padding token.

```text
<UNK>
```

Unknown token.

Other model-specific special tokens may also exist.

For example:

```text
<system>
<user>
<assistant>
```

may be represented using special tokens or special token sequences in some chat model formats.

The exact special-token vocabulary is model-specific.

---

# 44. What Is an Unknown Token?

An unknown token is used when the tokenizer cannot represent some input using its normal vocabulary.

A tokenizer may have a token such as:

```text
<UNK>
```

However, modern subword and byte-level tokenizers are often designed to minimize or avoid unknown-token problems for ordinary text.

Therefore, you should not assume that every tokenizer uses `<UNK>` for rare words.

---

# 45. Tokenization of Punctuation

Punctuation can also become tokens.

For example:

```text
Hello, world!
```

could conceptually become:

```text
["▁Hello", ",", "▁world", "!"]
```

or a different segmentation.

Again:

> The exact tokenization depends on the tokenizer.

---

# 46. Tokenization of Numbers

Numbers can be tokenized in many different ways.

For example:

```text
2026
```

might be represented as:

```text
["2026"]
```

or:

```text
["20", "26"]
```

or other pieces.

The tokenizer determines the segmentation.

This matters because numerical strings are not guaranteed to correspond to one token.

---

# 47. Tokenization of Code

Code is also tokenized.

For example:

```python
def add(a, b):
    return a + b
```

may be represented using tokens corresponding to:

```text
def
add
(
a
,
b
)
:
return
a
+
b
```

along with whitespace-associated or subword pieces.

The exact representation depends on the tokenizer.

This is one reason tokenization matters for coding LLMs as well.

---

# 48. Token Count Is Not Word Count

Consider:

```text
I am learning artificial intelligence.
```

You might count:

```text
6 words
```

But the tokenizer may produce a different number of tokens.

For example, conceptually:

```text
["▁I", "▁am", "▁learning", "▁artificial", "▁intelligence", "."]
```

That is:

```text
6 tokens
```

But another tokenizer could produce more or fewer tokens.

Therefore:

> Word count and token count are different quantities.

---

# 49. Why Token Count Matters

Token count affects:

- Context-window usage
- Memory requirements
- Inference computation
- API pricing for many commercial models
- Training cost
- Generation speed

For example, if a model has a context window of:

```text
128,000 tokens
```

that does not mean:

```text
128,000 words
```

It means approximately:

```text
128,000 tokenizer tokens
```

The exact relationship between tokens and words varies by language and tokenizer.

---

# 50. Tokenization and Context Windows

Suppose your prompt is:

```text
A very long document...
```

The tokenizer converts it into:

```text
[token_1, token_2, token_3, ..., token_n]
```

The model's context window limits how many tokens can be processed at once.

Conceptually:

```text
Document
   |
   v
Tokenizer
   |
   v
Token sequence
   |
   v
Context window
   |
   v
Transformer
```

Therefore, tokenization directly affects how much text can fit into a model's context.

---

# 51. Tokenization and Cost

Many LLM APIs calculate usage in tokens.

Conceptually:

```text
Input tokens
+
Output tokens
=
Total token usage
```

Therefore, a tokenizer determines how much text is counted as tokens.

For example:

```text
Prompt
   |
   v
Tokenizer
   |
   v
10,000 input tokens
```

The model processes those tokens rather than counting words directly.

---

# 52. Tokenization and Multilingual Text

Token efficiency can differ significantly across languages.

A tokenizer trained heavily on English may represent common English text efficiently but may require more tokens for some other languages.

Conceptually:

```text
Same semantic information
        |
        +---- English ----> fewer tokens
        |
        +---- Language B --> more tokens
```

The exact token count depends on the tokenizer and text.

This is one reason tokenizer vocabulary design matters for multilingual models.

---

# 53. Tokenization and Embeddings

Now connect tokenization to the next lesson.

The complete pipeline is:

```text
Raw text
    |
    v
Tokenizer
    |
    v
Token IDs
    |
    v
Embedding layer
    |
    v
Token vectors
    |
    v
Positional information
    |
    v
Transformer
```

For example:

```text
"I love AI"
```

might become:

```text
["▁I", "▁love", "▁AI"]
```

then:

```text
[42, 817, 2917]
```

then:

```text
[
  vector_42,
  vector_817,
  vector_2917
]
```

These vectors become inputs to the Transformer.

---

# 54. Tokenization Does Not Create Contextual Embeddings

This distinction is critical.

Tokenization gives you:

```text
Token IDs
```

The embedding layer gives you:

```text
Initial token vectors
```

The Transformer then creates contextual representations.

For example, the token:

```text
bank
```

could appear in:

```text
I deposited money in the bank.
```

or:

```text
We sat beside the river bank.
```

The tokenization may be similar, but the Transformer can produce different contextual representations based on surrounding tokens.

Conceptually:

```text
Tokenization
     |
     v
Token ID
     |
     v
Initial embedding
     |
     v
Self-attention
     |
     v
Contextual representation
```

This distinction becomes important in **Lesson 04 — Embeddings & Token Representations**.

---

# 55. Tokenization Pipeline in an LLM

The full process can be summarized as:

```text
                     RAW TEXT
                         |
                         v
                     TOKENIZER
                         |
                         v
                      TOKENS
                         |
                         v
                     TOKEN IDs
                         |
                         v
                  EMBEDDING LOOKUP
                         |
                         v
                  TOKEN VECTORS
                         |
                         v
              POSITION INFORMATION
                         |
                         v
             DECODER-ONLY TRANSFORMER
                         |
                         v
                       LOGITS
                         |
                         v
                     NEXT TOKEN
```

---

# 56. SentencePiece Pipeline

If the tokenizer uses SentencePiece:

```text
Raw text
    |
    v
SentencePiece
    |
    +--------------------+
    |                    |
    v                    v
 Unigram                 BPE
    |                    |
    +---------+----------+
              |
              v
         Token pieces
              |
              v
          Token IDs
              |
              v
          Embeddings
```

The tokenizer uses its trained vocabulary and algorithm to segment the input.

---

# 57. BPE vs Unigram: Intuitive Example

Suppose the word is:

```text
unhappiness
```

## BPE

BPE may progressively learn frequent combinations:

```text
u + n
  -> un

h + a
  -> ha

ha + p
  -> hap

hap + p
  -> happ

...
```

Eventually, the learned merge rules can produce a segmentation such as:

```text
▁un + happi + ness
```

The exact result depends on the vocabulary and merge rules.

## Unigram

Unigram starts with candidate pieces and evaluates possible segmentations probabilistically:

```text
▁un + happiness
```

versus:

```text
▁un + happi + ness
```

and potentially other segmentations.

It selects a segmentation according to the learned probabilities.

---

# 58. Important Correction: BPE Does Not "Understand Words"

BPE does not know that:

```text
happy
```

is a word or that:

```text
happiness
```

is semantically related to it.

It learns frequent symbol or subword combinations from training data.

For example:

```text
hap + py
```

may become a useful token because that combination occurs frequently.

Therefore:

> Tokenization is primarily a statistical/compression-oriented representation process, not semantic understanding.

The Transformer is responsible for learning contextual and semantic representations.

---

# 59. Important Correction: SentencePiece Does Not Mean "Sentence-Level Tokens"

The name **SentencePiece** can be misleading.

It does **not** mean:

```text
One sentence = one token
```

Instead, it is a tokenization framework whose name comes from its ability to train directly from sentences/raw text.

For example:

```text
I love machine learning.
```

can become:

```text
▁I
▁love
▁machine
▁learning
.
```

---

# 60. Why Tokenizers Are Trained Separately

A tokenizer generally has its own learned vocabulary or tokenization rules.

Conceptually:

```text
Large text corpus
       |
       v
Tokenizer training
       |
       v
Vocabulary + rules/model
       |
       v
Tokenizer
```

Then:

```text
New text
   |
   v
Trained tokenizer
   |
   v
Token IDs
```

The tokenizer is not learning new vocabulary every time you send a prompt.

---

# 61. Tokenizer Vocabulary and Model Vocabulary

The model's input/output vocabulary is closely tied to the tokenizer.

For a causal LLM:

```text
Text
  |
  v
Tokenizer
  |
  v
Token IDs
  |
  v
Embedding
  |
  v
Transformer
  |
  v
LM Head
  |
  v
Vocabulary logits
```

The LM head generally produces one score for each vocabulary token.

So if the tokenizer vocabulary has:

```text
50,000 tokens
```

the output layer generally produces approximately:

```text
50,000 logits
```

for each prediction position.

---

# 62. Tokenizer and LM Head Connection

This is an important architectural connection.

Input side:

```text
Token
  |
  v
Token ID
  |
  v
Embedding
```

Output side:

```text
Hidden state
  |
  v
LM Head
  |
  v
Logits for vocabulary tokens
```

Therefore:

```text
Tokenizer vocabulary
        |
        +----> Input token IDs
        |
        +----> Output vocabulary
```

The tokenizer and model vocabulary must be compatible.

---

# 63. Special Tokens in Chat Models

Chat models often need to represent conversation structure.

Conceptually:

```text
<system>
You are an assistant.

<user>
Explain tokenization.

<assistant>
Tokenization is...
```

The exact representation varies between model families.

Some models use special tokens, while others use structured templates that are converted into token sequences.

The important idea is:

> Chat formatting is eventually represented as tokens that the model processes.

---

# 64. Tokenization Is the First Step, Not the Intelligence

It is useful to separate the responsibilities:

```text
Tokenizer
    |
    +-- Splits/segments text
    +-- Maps pieces to IDs

Embedding
    |
    +-- Converts IDs to vectors

Transformer
    |
    +-- Builds contextual representations
    +-- Models relationships
    +-- Produces logits

Decoder / Sampling
    |
    +-- Selects the next token
```

This prevents a common misunderstanding:

> The tokenizer does not understand the text like the LLM does.

---

# 65. End-to-End Example

Suppose the user writes:

```text
Explain machine learning.
```

## Step 1: Raw Text

```text
Explain machine learning.
```

## Step 2: Tokenization

A tokenizer might produce:

```text
["▁Explain", "▁machine", "▁learning", "."]
```

## Step 3: Token IDs

For example:

```text
[1842, 927, 3615, 13]
```

The exact IDs are tokenizer-specific.

## Step 4: Embeddings

Each ID is mapped to a vector:

```text
1842 -> vector
927  -> vector
3615 -> vector
13   -> vector
```

## Step 5: Transformer

The vectors pass through Transformer layers.

```text
Embeddings
    |
    v
Self-Attention
    |
    v
FFN
    |
    v
Self-Attention
    |
    v
FFN
    |
    v
...
```

## Step 6: LM Head

The final hidden state is projected to vocabulary logits.

```text
Hidden state
    |
    v
LM Head
    |
    v
Vocabulary logits
```

## Step 7: Next Token

The model selects or samples the next token.

```text
Next token -> "Machine"
```

Then generation continues autoregressively.

---

# 66. Character vs Word vs Subword

| Method | Example for `unhappiness` | Vocabulary | Sequence length |
|---|---|---|---|
| Character | `u n h a p p i n e s s` | Small | Long |
| Word | `unhappiness` | Very large | Short |
| Subword | `un + happi + ness` | Moderate | Moderate |

Subword tokenization provides a practical compromise for modern language models.

---

# 67. BPE vs WordPiece vs Unigram vs Byte-Level BPE

| Method | Core idea | Typical clue |
|---|---|---|
| BPE | Repeatedly merge frequent pairs | Merge rules |
| WordPiece | Learn useful subword vocabulary and segment words | `##` continuation marker |
| Unigram | Probabilistic segmentation using learned token probabilities | SentencePiece Unigram |
| Byte-Level BPE | Start from bytes, then apply BPE merges | Byte-based initial representation |

These are conceptual summaries. Real tokenizer implementations can include additional normalization, pre-tokenization, special-token, and decoding rules.

---

# 68. Common Misconceptions

## Misconception 1

> One token equals one word.

False.

A token can be a word, part of a word, punctuation, whitespace-associated text, or another unit.

---

## Misconception 2

> SentencePiece means each sentence becomes one token.

False.

SentencePiece creates subword pieces.

---

## Misconception 3

> SentencePiece is the same thing as BPE.

Not exactly.

SentencePiece is a tokenization framework that supports algorithms such as BPE and Unigram.

---

## Misconception 4

> SentencePiece and WordPiece are the same.

False.

WordPiece is a subword tokenization algorithm. SentencePiece is a tokenizer framework that supports algorithms such as Unigram and BPE.

---

## Misconception 5

> Tokenization understands meaning.

Not in the semantic sense.

Tokenization primarily segments text and maps pieces to IDs.

---

## Misconception 6

> Token IDs are universal.

False.

Token IDs depend on the tokenizer vocabulary.

---

## Misconception 7

> Byte-level tokenization means every byte is one final token.

False.

Byte-level BPE can merge byte-level units into larger learned tokens.

---

## Misconception 8

> More words always means more tokens.

Not necessarily.

Different tokenizers and languages can produce different token counts.

---

# 69. Key Terms

| Term | Meaning |
|---|---|
| Token | A unit produced by a tokenizer |
| Tokenization | Conversion of text into tokens |
| Vocabulary | Collection of tokens known by the tokenizer |
| Token ID | Integer assigned to a token |
| Subword | A token representing part of a word or a word-like unit |
| SentencePiece | Tokenization framework/library for subword tokenization |
| Unigram | Probabilistic subword tokenization approach |
| BPE | Byte Pair Encoding, a merge-based subword algorithm |
| WordPiece | Subword tokenization algorithm used by models such as BERT |
| Byte-Level Tokenization | Tokenization based on byte representations |
| Byte-Level BPE | BPE performed starting from byte-level units |
| Special Token | Token used for structural/model-specific purposes |
| BOS | Beginning-of-sequence token |
| EOS | End-of-sequence token |
| PAD | Padding token |
| UNK | Unknown token |
| Embedding | Vector representation associated with a token ID |
| Context Window | Maximum token sequence the model can process in a context |

---

# 70. Most Important Mental Model

Remember this pipeline:

```text
                 HUMAN TEXT
                     |
                     v
                 TOKENIZER
                     |
                     v
                   TOKENS
                     |
                     v
                 TOKEN IDs
                     |
                     v
                 EMBEDDINGS
                     |
                     v
           CONTEXTUAL REPRESENTATIONS
                     |
                     v
                TRANSFORMER
                     |
                     v
                   LOGITS
                     |
                     v
              NEXT TOKEN
```

For the major tokenization families:

```text
                     TOKENIZATION
                          |
        +-----------------+-----------------+
        |                 |                 |
        v                 v                 v
      BPE             WordPiece          Unigram
        |                 |                 |
   Merge pairs       Subword vocab     Probabilistic
        |                 |             segmentation
        |                 |
        +--------+--------+
                 |
                 v
               Tokens
```

And for byte-level BPE:

```text
Raw text
   |
   v
UTF-8 bytes
   |
   v
Initial byte units
   |
   v
BPE merges
   |
   v
Final tokens
```

---

# 71. What You Should Remember for Interviews

### Question: What is tokenization?

Answer:

> Tokenization is the process of converting raw text into tokens that can be mapped to integer token IDs and then converted into vectors for neural-network processing.

### Question: Why use subword tokenization?

Answer:

> Subword tokenization provides a practical balance between vocabulary size and sequence length while allowing the model to represent rare or previously unseen words using reusable pieces.

### Question: What is SentencePiece?

Answer:

> SentencePiece is a tokenizer framework that can learn subword vocabularies directly from raw text. It supports algorithms such as Unigram and BPE.

### Question: What is BPE?

Answer:

> BPE is a merge-based subword tokenization algorithm that learns useful token pieces by repeatedly merging frequent adjacent units.

### Question: What is WordPiece?

Answer:

> WordPiece is a subword tokenization algorithm used by models such as BERT. It learns a vocabulary of useful subword units and commonly uses `##` to indicate continuation pieces.

### Question: What is Unigram tokenization?

Answer:

> Unigram tokenization treats tokenization as a probabilistic segmentation problem. It learns probabilities for candidate token pieces and selects a likely segmentation of the input.

### Question: What is byte-level tokenization?

Answer:

> Byte-level tokenization starts from a byte representation, commonly UTF-8, allowing broad coverage of arbitrary text. Byte-level BPE then learns merges over those byte-level units.

### Question: What is a token ID?

Answer:

> A token ID is an integer that identifies a token in a tokenizer's vocabulary. The model uses these IDs to perform embedding lookups.

---

# 72. Connection to the Next Lesson

You now have:

```text
Text
 |
 v
Tokenization
 |
 v
Token IDs
```

The next question is:

> What do these token IDs become inside the neural network?

The answer is:

```text
Token IDs
    |
    v
Embedding Layer
    |
    v
Dense Vectors
    |
    v
Contextual Representations
```

This leads directly to:

# Lesson 04 — Embeddings & Token Representations

You will learn:

- What embeddings are
- Token embeddings
- Positional representations
- Contextual embeddings
- Static vs contextual embeddings
- Embedding dimensions
- How token IDs become vectors
- How Transformer layers transform token representations
- Why the same token can have different contextual representations
- How embeddings are used in LLMs and RAG systems

---

# 73. Roadmap Position

```text
PHASE 1 - LLM FOUNDATIONS

01. What Are LLMs?
        |
        v
02. Generative AI: The Big Picture
        |
        v
03. Tokenization Deep Dive             <- YOU ARE HERE
        |
        v
04. Embeddings & Token Representations


PHASE 2 - MODERN LLM ARCHITECTURE

05. LLM Architecture Internals
        |
        v
06. Positional Encodings
        |
        v
07. Mixture of Experts
        |
        v
08. Context Window & Attention Patterns


PHASE 3 - LLM TRAINING & GENERATION

09. Pre-training Objectives
        |
        v
10. Logits, Softmax & Temperature
        |
        v
11. Sampling Strategies
        |
        v
12. Multi-turn Conversations & Memory
```

---

# Final Takeaway

The most important idea from this lesson is:

```text
Raw Text
   |
   v
Tokenization
   |
   v
Tokens
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
Contextual Representations
   |
   v
Logits
   |
   v
Next Token
```

The major tokenization approaches can be remembered as:

```text
BPE
 |
 +-- Merge frequent pairs

WordPiece
 |
 +-- Learn useful subword pieces
 +-- Often uses ## for continuation

Unigram
 |
 +-- Learn probabilities for pieces
 +-- Choose a likely segmentation

Byte-Level BPE
 |
 +-- Start from bytes
 +-- Apply BPE merges
```

And SentencePiece should be remembered as:

```text
SentencePiece
      |
      +-- Unigram
      |
      +-- BPE
```

Once you understand this pipeline, you have the foundation needed to understand how an LLM converts human language into the numerical representations processed by a Transformer.
