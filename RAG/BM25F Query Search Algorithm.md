# What is BM25?

Okapi BM25 is a ranking function used in search engines to score documents by relevance to a query. The "BM" is short for *best matching*, and 25 is the version of the function that worked best in testing. It improves on earlier methods like TF-IDF by accounting for document length and diminishing returns from repeated terms. It powers Apache Lucene-based search systems like Elasticsearch and Solr, and Tantivy-based search systems like ParadeDB.

Even though it's over 30 years old, BM25 remains the default ranking algorithm in most search engines because it’s simple, explainable, and performs well in practice.

---

## How BM25 Works

It's important to note that BM25 is what’s known as a **bag-of-words** retrieval function. It doesn’t look at term order, phrase structure, or proximity: only at which words appear and how frequently they occur.

BM25 starts by breaking the search query into individual terms. For each term, it scores how well every document matches based on three main signals:

1. **Term frequency (TF)**: documents that mention a query term more often score higher.
2. **Inverse document frequency (IDF)**: rare terms are weighted more than common ones.
3. **Document length normalization**: shorter documents are preferred to long ones that mention the term in passing.

Each term contributes its own score, and BM25 sums those scores to produce an overall relevance value for the document.

In simplified form, BM25 can be thought of as:

$$\text{Score} = \sum \text{IDF} \times \text{adjusted\_term\_frequency}$$

where the adjusted term frequency reduces the impact of very frequent terms and normalizes for document length.

The full BM25 formula is slightly more complex. It includes tunable parameters:
*   **$k_1$**: how quickly term frequency saturates.
*   **$b$**: how strongly document length is normalized.

These controls make BM25 adaptable across different datasets and document types.

---

## Why BM25 Works Well

BM25 has three key strengths which have allowed it to stay relevant for so long:
*   **Simplicity**: Easy to understand and implement.
*   **Efficiency**: Fast enough to run in real-time over large datasets.
*   **Explainability**: Each part of the score can be traced to a clear factor.

Because of these traits, BM25 remains the baseline for relevance in modern search systems.

---

## When to Use BM25

BM25 excels in retrieval workloads where specific keywords carry significant information. For example:
*   Searching for a brand name in a product catalog
*   Looking up a stock ticker in a trading app
*   Matching a merchant name in a credit card transaction ledger
*   Finding names in a legal document
*   Finding a diagnosis code in a medical report
*   Retrieving files to feed into an LLM in a RAG application

Because BM25 relies on simple term statistics, it is accurate, consistent, and extremely fast. That makes it a popular choice in applications where low query latency and keyword relevance are critical.

---

## Example: Scoring a Query with BM25

Imagine a user searches for *"inverted index"*. 

BM25 first breaks the query into two terms: **“inverted”** and **“index”**. Then, for each document, it scores how well those terms match:

| Document | Length | Term counts | TF–IDF signals | Relative Score |
| :--- | :--- | :--- | :--- | :--- |
| **Doc A** | 60 words | `inverted: 1`<br>`index: 1` | rare $\to$ high IDF<br>common $\to$ low IDF | **Higher** |
| **Doc B** | 200 words | `inverted: 1`<br>`index: 10` | rare $\to$ high IDF<br>common $\to$ low IDF<br>+ length penalty | **Lower** |

Even though **Doc B** repeats “index” more times, BM25 favors **Doc A** because:
1. **“inverted”** is a rare term with higher IDF.
2. Both documents mention it once (similar TF).
3. **Doc A** is shorter, so its matches are more concentrated.

BM25 rewards focused, relevant mentions of rare terms, not repetition in long text.

---

## Summary

BM25 remains the standard ranking algorithm for keyword search. It combines frequency, rarity, and length normalization into a single scoring model that quickly produces balanced and predictable results. It is the foundation that modern lexical hybrid search systems continue to build on.
