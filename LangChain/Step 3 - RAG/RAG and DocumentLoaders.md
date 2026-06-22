# LangChain RAG & Document Loaders

## Stage 1: Document Loading (DocumentLoaders)
Data lives in messy formats (PDFs, Confluence, SQL instances, AWS S3 buckets). LangChain abstracts these into `DocumentLoaders`. A loader takes a raw file/stream and serializes it into standard LangChain `Document` objects.

A `Document` object contains exactly two properties:

* **page_content**: A plain string containing the raw text.
* **metadata**: A Python dictionary containing key-value pairs (e.g., `{"source": "employee_handbook.pdf", "page": 43}`). Keeping track of metadata is essential for filtering down data later or providing citations to users.

---

## Stage 2: Document Chunking (TextSplitters)
You cannot pass a 500-page PDF straight into an LLM context window—it wastes tokens, degrades accuracy ("lost in the middle" phenomenon), and costs too much. You must break documents down into smaller, discrete parts called chunks.

As an engineer, your choice of text splitter directly dictates the semantic integrity of your data.

* **CharacterTextSplitter (Naive)**: Splits on a single hardcoded character (like `\n\n`). Highly brittle; it frequently slices sentences in half, destroying their context.
* **RecursiveCharacterTextSplitter (Standard Production)**: Splits by looking at an ordered array of separators (typically `["\n\n", "\n", " ", ""]`). It looks for paragraph breaks first. If the paragraph is still too large, it looks for sentence breaks (`\n`), then words (` `), down until it satisfies your two constraints:
  * `chunk_size`: The maximum number of characters/tokens allowed in a single chunk.
  * `chunk_overlap`: The number of characters shared between adjacent chunks. Overlap is mandatory to preserve continuous semantics across chunk boundaries.
* **MarkdownHeaderTextSplitter / HTMLHeaderTextSplitter (Structure-Aware)**: Splits documents based on structural elements (like `# H1`, `## H2`). This keeps nested lists and markdown tables completely intact.

---

## Stage 3: Embedding Models & Vector Stores
Once text is split into chunks, you cannot run standard regex or keyword searches on it. If a user asks "How do I request time off?", keyword matching will fail to find a document that reads "Submitting vacation requests on the portal." You must convert text into its semantic mathematical meaning.

* **Embeddings (Embeddings)**: LangChain coordinates calls to embedding models (like OpenAI's `text-embedding-3-small` or HuggingFace local models). These models take a text string and return an array of floating-point numbers (a vector, e.g., 1536 dimensions). Text pieces with similar real-world meanings sit closer to each other in this multidimensional mathematical space.
* **Vector Stores (VectorStore)**: Traditional databases like MySQL or MongoDB are not optimized for mathematical vector space queries. LangChain integrates natively with high-performance vector databases (Chroma, Pinecone, Milvus, pgvector). The vector store indices your embedding vectors so you can perform Cosine Similarity searches in milliseconds across millions of items.

---

## Stage 4: Advanced Retrieval Techniques (Retrievers)
A Vector Database is just storage. A Retriever is the algorithmic abstraction that executes queries against it and decides what comes back. In standard production systems, a basic similarity search returns too much noise. LangChain implements advanced retrieval patterns:

### A. Multi-Query Retriever
Users write bad search queries. If a user queries "db connectivity", they might miss a document titled "Configuring Postgres Network Connections." The `MultiQueryRetriever` takes the user's single question, uses a low-cost LLM to generate 3 or 4 variations of the question from different semantic angles, runs all of them against the vector database in parallel, and merges the unique results.

### B. Contextual Compression & Reranking (FlashRank / CohereRerank)
To ensure you aren't flooding your LLM with useless information, you can chain your retriever with a Reranker.
1. The vector store does a cheap, fast lookup and returns the top 20 most similar chunks.
2. A specialized Cross-Encoder Reranking model analyzes the actual semantics of those 20 chunks against the user's prompt and scores them strictly on relevance.
3. The compressor drops the bottom 15 items and sends only the top 5 highly relevant chunks to the LLM, dramatically cutting token usage and reducing hallucinations.

### C. Parent-Document Retriever
When chunking data, you are caught in a paradox: small chunks (e.g., 100 characters) capture precise embedding meanings, but they lose the overarching context of the document. Large chunks (e.g., 2000 characters) preserve context, but dilute the precise meaning of individual sentences.

The `ParentDocumentRetriever` resolves this:
1. It cuts documents into large Parent Chunks and maps them to tiny Child Chunks.
2. Only the tiny Child Chunks are converted to embeddings and indexed in the vector store.
3. When a user searches, the system matches against a Child Chunk, but instead of handing that tiny snippet to the LLM, it looks up its relational map and hands over the full, rich text of the Parent Chunk.

---

## Stage 5: Putting it Together in LCEL (The RAG Chain)
Here is a complete, production-grade implementation of a RAG pipeline utilizing LCEL:

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from operator import itemgetter

# 1. Initialize the vector database and expose it as a retriever interface
vectorstore = Chroma(persist_directory="./db", embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Fetch top 3 chunks

# 2. Build a RAG-specific context prompt
rag_prompt = ChatPromptTemplate.from_template("""
You are a secure internal enterprise assistant. Answer the user's question using ONLY the provided context below. 
If you do not know the answer based on the context, say "I cannot find that in the official documentation."

Context:
{context}

Question: {question}
""")

# 3. Build the LCEL RAG Graph
rag_chain = (
    {
        # Extract the user's question, pass it to the retriever to gather chunks, and format them as a single string
        "context": itemgetter("question") | retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": itemgetter("question")
    }
    | rag_prompt
    | ChatOpenAI(model="gpt-4o", temperature=0.0)
    | StrOutputParser()
)

# 4. Execute the pipeline
response = rag_chain.invoke({"question": "What is our company policy on remote work stipends?"})
```

---

## Summary of the RAG Layer
As a computer science student, look at RAG as a structured pipeline engineering problem. You load messy unstructured formats, split them using context-aware heuristics (`RecursiveCharacterTextSplitter`), map them into high-dimensional vector spaces using `Embeddings`, index them inside vector databases, and execute advanced semantic retrieval heuristics (Reranking, Parent-Document Mapping) to construct highly structured context injection payloads for your models.
