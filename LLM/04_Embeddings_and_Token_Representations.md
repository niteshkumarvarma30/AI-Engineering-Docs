# Lesson 4: Embeddings & Token Representations

## 1. Introduction to Representations in NLP

In natural language processing (NLP) and modern AI Engineering, the fundamental challenge is translation: how do we translate the chaotic, discrete, human-centric nature of language into the continuous, numerical format that machine learning models—specifically deep neural networks—require for computation?

Historically, AI practitioners utilized simplistic representation methods like Bag-of-Words (BoW) and One-Hot Encoding. While conceptually straightforward, these representations failed catastrophically at capturing the underlying semantic meaning and syntactic structure of human language.

Today, state-of-the-art AI relies entirely on **Embeddings**: dense, continuous vector representations of discrete tokens. Embeddings map tokens (words, subwords, or characters) into a highly structured geometric vector space where semantic and syntactic similarities are intuitively represented by spatial proximity.

This lesson provides a comprehensive deep dive into the mathematical and architectural foundations of embeddings, the evolution from static vectors to highly contextualized representations, the geometry of vector spaces, and how to conquer related questions in AI Engineering interviews.

---

## 2. The Flaws of Pre-Embedding NLP

To truly appreciate embeddings, one must understand the limitations of the techniques they replaced.

### 2.1 One-Hot Encoding and the Curse of Dimensionality

Assume a vocabulary $V$ containing $N = 50,000$ unique tokens. In one-hot encoding, a token is represented by a vector $x \in \mathbb{R}^N$. Exactly one element corresponding to the token's index is $1$, and all other $N-1$ elements are $0$.

```text
Vocabulary: ["apple", "banana", "car", "dog", ..., "zebra"]

apple:  [1, 0, 0, 0, ..., 0]
banana: [0, 1, 0, 0, ..., 0]
car:    [0, 0, 1, 0, ..., 0]
```

**Critical Drawbacks:**
1. **Extreme Sparsity:** $99.998\%$ of the memory allocated for these vectors is wasted on zeros. When performing matrix multiplications in early neural networks, this sparsity resulted in massive inefficiencies.
2. **Orthogonality and Lack of Semantics:** If we want to determine how similar two words are, we typically use the dot product. However, the dot product of any two distinct one-hot vectors is identically zero:
   $$ \text{apple} \cdot \text{banana} = (1 \times 0) + (0 \times 1) + ... = 0 $$
   $$ \text{apple} \cdot \text{car} = (1 \times 0) + (0 \times 0) + ... = 0 $$
   The model has no inherent way of knowing that an apple is more closely related to a banana than it is to a car. The semantic distance between any two distinct words is completely uniform.

### 2.2 TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF attempts to provide some weighting based on token importance across documents. While it improves upon raw word counts (Bag of Words) by penalizing extremely common words like "the" and "is", it still relies on a sparse, vocabulary-sized vector representation and fails to encode semantic meaning beyond word occurrence statistics.

---

## 3. The Embedding Matrix: From Discrete to Dense

To solve sparsity and orthogonality, neural networks utilize an **Embedding Layer**, essentially a mathematical lookup table. Instead of a $50,000$-dimensional sparse vector, each token ID is projected into a lower-dimensional, dense continuous space.

Let the embedding dimension be $d$. In modern models, $d$ ranges from $128$ (for small models) up to $4096$ or even $12288$ (for massive models like GPT-4 or LLaMA-3). 

### 3.1 The Mathematics of the Lookup

We define the embedding matrix as $E \in \mathbb{R}^{|V| \times d}$, where $|V|$ is the size of the vocabulary.
Let $x$ be the one-hot encoded vector of our input token, where $x \in \{0,1\}^{|V|}$.

The dense embedding vector $e \in \mathbb{R}^d$ for the token is obtained via the matrix multiplication:
$$ e = E^T x $$

Because $x$ is entirely zeros except for a single $1$ at index $i$, this matrix multiplication simplifies mathematically to simply extracting the $i$-th row of the matrix $E$.

```text
Input Token ID: 42 (e.g., the subword " play")

Embedding Matrix E (|V| x d):
Index 0:  [ 0.12, -0.34,  0.55, ... ,  0.81]
Index 1:  [-0.05,  0.99, -0.12, ... , -0.44]
...
Index 42: [ 0.91,  0.50,  0.33, ... , -0.25]  <-- O(1) Memory Lookup!
...
Index V-1:[-0.41,  0.72, -0.88, ... ,  0.11]

Resulting Vector e (d dimensions): [0.91, 0.50, 0.33, ..., -0.25]
```

In standard deep learning frameworks (like PyTorch's `nn.Embedding`), this is implemented as an $O(1)$ memory lookup array index, entirely bypassing the computationally expensive matrix multiplication.

### 3.2 What Do the Dimensions Represent?

In dense embeddings, each of the $d$ dimensions represents a latent, continuous feature of the word. Unlike manual feature engineering, these features are completely uninterpretable to humans. 
Dimension 1 might represent a combination of gender, verb tense, and sentiment. Dimension 2 might represent pluralization and geography. The neural network distributes semantic meaning across the continuous space through backpropagation during its pre-training phase, adjusting the vectors so that semantically similar words are pushed closer together.

---

## 4. Subword Tokenization and Embeddings

Embeddings do not exist in a vacuum; they map the discrete tokens generated by a tokenizer (like BPE, WordPiece, or SentencePiece). 

If a model uses Byte-Pair Encoding (BPE) with a vocabulary size of $50,257$ (like GPT-2), the Embedding matrix $E$ will have exactly $50,257$ rows. 

**Handling Out-Of-Vocabulary (OOV):**
Because modern tokenizers utilize subwords and eventually fall back to individual characters or bytes, they virtually eliminate the Out-Of-Vocabulary problem. Even if the model encounters a completely novel word, it will break it down into known subword tokens, look up their respective embeddings, and process them sequentially.

---

## 5. Static Embeddings (The First Revolution)

The concept of dense vectors exploded in popularity with the release of **Word2Vec** in 2013, followed closely by **GloVe** (2014) and **FastText** (2016). These are known as *Static Embeddings*.

**Static** means a strict 1-to-1 mapping. The token always maps to the exact same continuous vector, regardless of its surrounding context.

### 5.1 Word2Vec: Skip-Gram and CBOW

Word2Vec is a shallow two-layer neural network trained specifically to reconstruct linguistic contexts of words. It is built on the Distributional Hypothesis: *"You shall know a word by the company it keeps" - J.R. Firth.*

It has two primary architectures:
1. **Continuous Bag-of-Words (CBOW):** Predicts a target word given a window of surrounding context words.
2. **Skip-Gram:** Predicts the surrounding context words given a single target word.

```text
Architecture of Skip-Gram:

Sentence: "The quick brown fox jumps"
Target Word: "brown"
Context Window (size 2): ["The", "quick", "fox", "jumps"]

                /--> Output Layer (Softmax) -> "The"
               /
[Input "brown"] ---> Output Layer (Softmax) -> "quick"
               \
                \--> Output Layer (Softmax) -> "fox"
                 \
                  \-> Output Layer (Softmax) -> "jumps"
```

**The Math of Skip-Gram:**
The objective function is to maximize the log probability of context words given the target word:
$$ J(\theta) = \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \le j \le c, j \neq 0} \log P(w_{t+j} | w_t) $$
Where $c$ is the context window size. The probability $P(w_c | w_t)$ is defined using the softmax function:
$$ P(w_c | w_t) = \frac{\exp(u_{w_c}^T v_{w_t})}{\sum_{i=1}^{|V|} \exp(u_i^T v_{w_t})} $$
Because calculating the denominator over the entire vocabulary $|V|$ is computationally prohibitive, Word2Vec employs **Negative Sampling**. Instead of updating all weights, the model only updates the weights of the true context word and a small number ($k=5$ to $20$) of randomly selected "noise" words, transforming the problem into binary logistic regression.

### 5.2 GloVe (Global Vectors)

While Word2Vec relies on local sliding windows, GloVe is trained on global word-word co-occurrence statistics from a massive corpus. It applies matrix factorization to a co-occurrence matrix, attempting to ensure that the dot product of two word vectors equals the logarithm of their probability of co-occurrence.

### 5.3 FastText

Developed by Facebook AI, FastText improved upon Word2Vec by representing words as a bag of character n-grams. For the word `apple`, FastText considers `app`, `ppl`, `ple`, etc.
This allows the model to build embeddings for misspelled words or rare morphological variations dynamically.

**The Fatal Flaw of Static Embeddings:**
Polysemy. Consider the word "bank":
1. "I sat by the river **bank**."
2. "I deposited money in the **bank**."

In Word2Vec, the vector for "bank" is a single point in space. It is mathematically forced to represent the average of a "financial institution" and a "river edge," effectively becoming a muddy, inaccurate representation of both.

---

## 6. Contextual Embeddings (The Deep Learning Era)

To solve polysemy, researchers developed **Contextual Embeddings**. Modern LLMs (like BERT, GPT, LLaMA) do not just use static lookups; they dynamically alter the vector of a token based on the entire sequence.

### 6.1 The Contextualization Pipeline in Transformers

1. **Static Lookup:** The model still starts with a static lookup matrix (the Token Embedding layer). Both instances of the word "bank" begin with the exact same initial vector $e_{\text{bank}}$.
2. **Positional Encoding:** Because self-attention is permutation-invariant, a positional vector $p_i$ is added to $e_{\text{bank}}$ to encode its position in the sequence.
3. **Self-Attention Layers:** The Transformer blocks apply the Self-Attention mechanism. The vector for "bank" projects into Query, Key, and Value vectors.
   - In "river bank", the attention mechanism calculates a high attention score between "bank" and "river". The Value vector of "river" is heavily mixed into the representation of "bank".
   - In "money bank", "bank" attends heavily to "money" and "deposited".
4. **Final Contextual Vector:** By the time the vectors exit the final Transformer layer, the resulting vector $h_{\text{bank1}}$ is drastically different from $h_{\text{bank2}}$. 

```text
The Contextual Shift via Self-Attention:

"river" ---> [ e_river ] -----\
                               \ (Mixing via Attention) ---> [ h_bank (Geographic) ]
"bank"  ---> [ e_bank  ] ------/


"money" ---> [ e_money ] -----\
                               \ (Mixing via Attention) ---> [ h_bank (Financial) ]
"bank"  ---> [ e_bank  ] ------/
```

Through this mechanism, LLMs attain their deep understanding of context and nuance.

---

## 7. The Geometric Vector Space and Similarity Metrics

Once words (or sentences) are embedded in a high-dimensional space, we must define mathematical ways to measure their relationships.

### 7.1 Analogies and Vector Arithmetic

Word2Vec famously demonstrated that syntactic and semantic relationships map to linear translations in the vector space.

The canonical example:
$$ \vec{King} - \vec{Man} + \vec{Woman} \approx \vec{Queen} $$

**Why does this linear structure emerge?**
It stems from the objective function. If word A (King) and word B (Man) appear in very similar linguistic contexts except for pronouns (he vs she), their vectors differ primarily along a subspace that correlates with gender. By subtracting the "Man" vector, we move out of the male subspace, and adding "Woman" moves us into the female subspace, landing in the neighborhood of "Queen".

```text
       y-axis (Gender)
       ^
       |
     Queen 
      ^  \
      |   \ 
      |    \ <--- Semantic Translation Vector
     King   Woman
        \   ^
         \  | 
          \ | <--- Semantic Translation Vector
           Man
            ----------------> x-axis (Royalty vs Commoner)
```

### 7.2 Distance Metrics

To build semantic search engines or retrieval systems, we must calculate the mathematical distance between two vectors $A$ and $B$.

#### 7.2.1 Euclidean Distance (L2 Distance)

Euclidean distance measures the straight-line physical distance between the tips of two vectors.
$$ d(A,B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2} $$

**Drawback:** In NLP, the magnitude (length) of a word vector is often influenced by its frequency in the training corpus. Euclidean distance is extremely sensitive to vector magnitude. Two vectors could point in the exact same direction (same meaning), but if one is much longer than the other, their Euclidean distance will be large.

#### 7.2.2 Dot Product (Inner Product)

The dot product measures the unnormalized projection of one vector onto another.
$$ A \cdot B = \sum_{i=1}^{n} A_i B_i = ||A|| \cdot ||B|| \cos(\theta) $$
Because it scales with the magnitude $||A||$ and $||B||$, the dot product can be problematic if vectors are not normalized. 

#### 7.2.3 Cosine Similarity (The Industry Standard)

Cosine similarity measures the cosine of the angle $\theta$ between two vectors. By dividing the dot product by the product of their magnitudes, it effectively normalizes the vectors to a length of 1, completely ignoring magnitude and focusing purely on their directional alignment.

$$ \text{Cosine Similarity} = \cos(\theta) = \frac{A \cdot B}{||A|| \cdot ||B||} $$
$$ \text{Cosine Similarity} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}} $$

- **Value Range:** $[-1, 1]$
- **$\approx 1$:** Vectors point in exactly the same direction (Highly semantically similar).
- **$\approx 0$:** Vectors are orthogonal (Semantically unrelated).
- **$\approx -1$:** Vectors point in opposite directions (Antonyms, though rare in actual NLP embeddings).

*Note: If all vectors in a database are strictly L2-normalized (their magnitudes equal 1), then the Dot Product mathematically equals Cosine Similarity, and Euclidean Distance becomes monotonic with Cosine Similarity. Optimizing for normalized dot product is standard practice in Vector Databases for efficiency.*

---

## 8. Embeddings in Modern AI Engineering (RAG & Semantic Search)

In modern architecture, we rarely stop at token embeddings. We use models like **Sentence-BERT (SBERT)** or OpenAI's `text-embedding-ada-002` to compress entire sentences, paragraphs, or chunks of documents into a single, dense vector.

### 8.1 Retrieval-Augmented Generation (RAG)

In a RAG pipeline, enterprise data is processed as follows:
1. **Chunking:** Documents are split into 500-token chunks.
2. **Embedding:** An embedding model processes each chunk and outputs a dense vector (e.g., $d=1536$).
3. **Vector Database:** Vectors are stored in a specialized database like Pinecone, Milvus, or pgvector. These databases use algorithms like **HNSW (Hierarchical Navigable Small World)** to perform Approximate Nearest Neighbor (ANN) search in logarithmic time.
4. **Querying:** A user's query is embedded into the same vector space.
5. **Retrieval:** The database calculates Cosine Similarity between the query vector and all chunk vectors, returning the top-K chunks.
6. **Generation:** The chunks are appended to the LLM's prompt as context.

### 8.2 Bi-Encoders vs. Cross-Encoders

- **Bi-Encoders:** (e.g., standard embedding models). The query and the document are embedded separately into vectors. Their similarity is computed using Cosine Similarity. Very fast, heavily scalable, but slightly less accurate.
- **Cross-Encoders:** The query and document are passed *together* through a Transformer model, allowing cross-attention between the query and document tokens. Much more accurate, but extremely computationally expensive. They are often used as a re-ranking step *after* a Bi-Encoder retrieves the top 100 results.

---

## 9. Typical Interview Questions

**Q1: What is the primary difference between Word2Vec and BERT embeddings?**
*Answer:* Word2Vec generates static embeddings. A word like "bank" maps to a single, fixed vector, merging the meanings of a river bank and a financial bank. BERT generates contextual embeddings. While it starts with a static token lookup, the Self-Attention layers iteratively update the vector based on the surrounding tokens in the sentence. Thus, the final vector for "bank" dynamically shifts to represent geography or finance depending on the context, effectively solving the polysemy problem.

**Q2: Why is Cosine Similarity generally preferred over Euclidean Distance for measuring semantic similarity between text embeddings?**
*Answer:* In high-dimensional embedding spaces, the magnitude of a vector is often skewed by the frequency of the word in the training corpus or the length of the document. Euclidean distance calculates the physical distance between the vector endpoints, making it highly sensitive to these magnitude differences. Cosine similarity isolates the angle between the vectors, effectively ignoring their magnitude and comparing only their directional orientation. This provides a much more robust, normalized measure of semantic similarity.

**Q3: Explain how token IDs are mapped to dense vectors in a deep learning framework like PyTorch.**
*Answer:* Mathematically, it is represented as a matrix multiplication $E^T x$, where $E$ is the embedding matrix of shape $(|V|, d)$ and $x$ is a one-hot encoded vector of the token ID. However, because $x$ is one-hot, calculating the full matrix multiplication is a massive waste of FLOPs. In frameworks like PyTorch (`nn.Embedding`), this is implemented under the hood as a highly optimized $O(1)$ memory lookup, simply fetching the $i$-th row of the matrix $E$, where $i$ is the token ID.

**Q4: If two words are antonyms (e.g., "hot" and "cold"), what would you expect their Cosine Similarity to be in a Word2Vec embedding space?**
*Answer:* Intuitively, one might guess -1 (opposite direction). However, in Word2Vec (and most LLMs), antonyms actually have a *very high* positive cosine similarity (e.g., 0.7 to 0.9). This is because Word2Vec relies on the Distributional Hypothesis—words that appear in similar contexts have similar vectors. "Hot" and "cold" appear in almost identical grammatical and semantic contexts ("The coffee is too hot", "The coffee is too cold"). Therefore, their vectors are pushed very close together in the embedding space. Differentiating antonyms requires more advanced contextual models or specific contrastive fine-tuning.

**Q5: What is the purpose of HNSW in a Vector Database?**
*Answer:* HNSW (Hierarchical Navigable Small World) is an Approximate Nearest Neighbor (ANN) search algorithm. If a vector database has 1 billion document embeddings, performing an exact Cosine Similarity search (K-Nearest Neighbors or KNN) would require 1 billion dot products per query, which is unacceptably slow ($O(N)$). HNSW builds a multi-layered graph structure that allows the search to quickly traverse from long-range macro-clusters down to the micro-neighborhood of the query vector, finding the most similar vectors in $O(\log N)$ time, trading a tiny amount of accuracy for a massive boost in latency and scalability.

**Q6: In RAG architectures, why might you use a Bi-Encoder followed by a Cross-Encoder?**
*Answer:* Bi-Encoders compute the embedding for the query and the documents independently, allowing document embeddings to be pre-computed and stored in a vector database for ultra-fast ANN retrieval. However, they lack deep semantic matching because the query and document tokens never interact. A Cross-Encoder passes the query and document together through the Transformer, allowing rich cross-attention between them, producing highly accurate relevance scores. Because Cross-Encoders are too slow to run on millions of documents, the standard design pattern is a two-stage pipeline: use a Bi-Encoder to retrieve the top 100 candidate documents (fast), and then use a Cross-Encoder to re-rank those 100 documents (accurate).

**Q7: Explain the Out-Of-Vocabulary (OOV) problem and how modern architectures handle it compared to older models like Word2Vec.**
*Answer:* Word2Vec was trained on whole words. If it encountered a word during inference that wasn't in its training vocabulary (e.g., "neuroplasticity"), it couldn't generate an embedding and would throw an OOV error or use a generic `<UNK>` token vector, losing all meaning. FastText mitigated this by using character n-grams. Modern architectures (BERT, GPT) handle this elegantly by using subword tokenization (BPE, WordPiece). They break "neuroplasticity" down into known subwords (e.g., "neuro", "plastic", "ity"), look up the embeddings for those subwords, and allow the model's self-attention mechanism to compose their combined meaning dynamically. Thus, modern LLMs essentially never suffer from OOV issues.

**Q8: What is L2 Normalization, and how does it relate Cosine Similarity to the Dot Product?**
*Answer:* L2 Normalization is the process of scaling a vector so that its magnitude (length) becomes exactly 1. This is done by dividing every element in the vector by the vector's Euclidean norm ($||v||$). 
Since Cosine Similarity is defined as $\frac{A \cdot B}{||A|| \cdot ||B||}$, if both vectors $A$ and $B$ are already L2-normalized, then $||A|| = 1$ and $||B|| = 1$. The denominator becomes $1$, and the Cosine Similarity equation perfectly reduces to the simple Dot Product ($A \cdot B$). Vector databases heavily utilize this trick: they L2-normalize all vectors upon ingestion, allowing them to use the computationally cheaper Dot Product operation while mathematically achieving exact Cosine Similarity rankings.

**Q9: What is the curse of dimensionality, and how do dense embeddings solve it?**
*Answer:* In the context of NLP representations like One-Hot Encoding, the "curse of dimensionality" refers to the fact that as the vocabulary grows, the vector space dimensions grow linearly (e.g., a 100,000-word vocabulary requires 100,000-dimensional vectors). In this massive, hyper-sparse space, the distance between any two distinct points is identical, making it impossible for a machine learning model to detect patterns or proximity. Dense embeddings solve this by forcing the model to compress the 100,000 discrete tokens into a lower, fixed-dimensional continuous space (e.g., $d=768$). This compression forces the neural network to capture the latent semantic features of the tokens, clustering related concepts together geometrically so the space becomes dense and meaningful.

---
*End of Lesson 4. Next: Lesson 5 - The Self-Attention Mechanism & Transformer Block.*
