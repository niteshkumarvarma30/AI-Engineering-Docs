# Lesson 12 — Multi-turn Conversations & Memory

## Complete LLM Application Note

This lesson explains how an LLM application maintains context across multiple turns and how conversation memory works.

The key idea is:

> An LLM does not automatically remember every previous conversation. The application provides relevant previous information as context.

---

# 1. The Basic Problem

Suppose you tell an LLM:

```text
User:
My name is Alex.
```

The assistant responds:

```text
Assistant:
Nice to meet you, Alex!
```

Later you ask:

```text
User:
What is my name?
```

If the second request contains only:

```text
What is my name?
```

the model may not know that you previously said:

```text
My name is Alex.
```

The application therefore needs to preserve and provide relevant context.

```text
Previous conversation
        ↓
Application stores it
        ↓
Relevant history is added to new request
        ↓
LLM receives context
        ↓
LLM generates response
```

---

# 2. LLM Memory vs Application Memory

This distinction is extremely important.

## Model Parameters

During training, the model learns patterns in its parameters:

```text
Training
   ↓
Weights
   ↓
Learned knowledge / patterns
```

These weights normally do not change just because you have a conversation.

## Conversation Memory

The application can store information externally:

```text
SQL database
Vector database
Redis
Files
Application state
```

Then it can retrieve relevant information and put it back into the model's context.

Therefore:

\[
\boxed{
\text{Conversation memory} \neq \text{model weights}
}
\]

---

# 3. Simplest Chatbot Without Memory

Suppose the user sends:

```text
Message 1:
My name is Alex.
```

Application sends:

```text
My name is Alex.
```

LLM responds:

```text
Nice to meet you, Alex.
```

Then user sends:

```text
Message 2:
What is my name?
```

Application sends only:

```text
What is my name?
```

The previous message is not necessarily available to the model.

---

# 4. Chatbot With Conversation History

Instead, the application stores:

```text
User:
My name is Alex.

Assistant:
Nice to meet you, Alex.
```

When the user asks:

```text
What is my name?
```

the application constructs:

```text
User:
My name is Alex.

Assistant:
Nice to meet you, Alex.

User:
What is my name?
```

Then sends the relevant context to the model.

The model can answer:

```text
Your name is Alex.
```

---

# 5. Important Mental Model

Think of the LLM as:

```text
                    LLM
                     │
              ┌──────┴──────┐
              │             │
         Input Context   Parameters
              │             │
              └──────┬──────┘
                     ↓
                 Response
```

The application decides what previous information becomes part of the input context.

---

# 6. Multi-turn Conversation

Example:

```text
Turn 1

User:
I am learning Python.

Assistant:
Great! Python is useful for AI development.
```

Then:

```text
Turn 2

User:
What should I learn next?
```

The second question depends on Turn 1.

So the application can provide:

```text
User:
I am learning Python.

Assistant:
Great! Python is useful for AI development.

User:
What should I learn next?
```

Now the LLM can understand the question in the context of Python learning.

---

# 7. What Is a Turn?

A turn is generally one interaction between the user and assistant.

Example:

```text
Turn 1:
User → Assistant

Turn 2:
User → Assistant

Turn 3:
User → Assistant
```

A conversation might look like:

```text
Turn 1:
User: Hello
Assistant: Hi!

Turn 2:
User: Explain Transformers.
Assistant: ...

Turn 3:
User: Explain attention.
Assistant: ...

Turn 4:
User: Give me a numerical example.
Assistant: ...
```

Later messages may depend on previous turns.

---

# 8. Chat History

The application can store messages as structured records.

Example:

```text
id | role      | content
---|-----------|------------------------
1  | user      | My name is Alex.
2  | assistant | Nice to meet you, Alex.
3  | user      | What is my name?
```

Conceptually:

```text
Conversation
     │
     ├── User message
     ├── Assistant message
     ├── User message
     └── Assistant message
```

---

# 9. Chat History Becomes Context

When a new message arrives:

```text
User:
What is my name?
```

the application retrieves previous messages:

```text
My name is Alex.
Nice to meet you, Alex.
```

and constructs:

```text
System instructions

Conversation history:
User: My name is Alex.
Assistant: Nice to meet you, Alex.

Current user:
What is my name?
```

Then:

```text
Context
  ↓
LLM
  ↓
Response
```

---

# 10. Context Window

There is a limit to how much tokenized information a model can process in one context.

This is called the:

\[
\boxed{\text{Context Window}}
\]

For example, if a model supports a 128K-token context window, the request must stay within the model's applicable context limits, including the input and output according to the model/API.

---

# 11. Context Window Is Not Memory

This distinction is critical.

### Context Window

Information currently supplied to the model.

### Memory

Information stored by the application so it can potentially be retrieved later.

Think:

```text
Memory
  ↓
Stored outside the model
  ↓
Retrieve relevant information
  ↓
Context Window
  ↓
LLM
```

Therefore:

\[
\boxed{
\text{Memory} \rightarrow \text{Context}
}
\]

---

# 12. Why Can't We Send the Entire Conversation Forever?

Suppose a conversation grows:

```text
Turn 1
Turn 2
Turn 3
...
Turn 1000
```

Sending everything every time can become:

- expensive
- slower
- context-heavy
- eventually impossible because of context limits

Therefore applications need context management.

---

# 13. Simple Context Management

A basic strategy is:

> Keep only the most recent messages.

For example:

```text
Conversation:

Turn 1
Turn 2
Turn 3
Turn 4
Turn 5
Turn 6
Turn 7
Turn 8
```

Keep:

```text
Turn 6
Turn 7
Turn 8
```

Older turns are removed or handled separately.

This is commonly called a sliding-window approach.

---

# 14. Sliding Window

Conceptually:

```text
Conversation:

[1] [2] [3] [4] [5] [6] [7] [8]

Keep last 4:

             [5] [6] [7] [8]
```

When Turn 9 arrives:

```text
             [6] [7] [8] [9]
```

The window moves forward.

---

# 15. Problem With Sliding Window

Suppose the user said:

```text
Turn 1:
My project is called CognitRAG.
```

Then after many turns:

```text
Turn 101:
How should I improve my project?
```

If Turn 1 has been removed from the active context, the model may no longer know the project name.

This motivates long-term memory.

---

# 16. Short-term Memory

Short-term memory usually means information from the recent conversation.

Example:

```text
User:
I am building a chatbot.

Assistant:
Great.

User:
It uses Gemini.

Assistant:
That can work well.

User:
What database should I use?
```

Recent context provides:

```text
chatbot
Gemini
```

This is short-term conversational context.

---

# 17. Long-term Memory

Long-term memory stores information that may be useful much later.

Example:

```text
User:
My project uses Supabase for conversation storage.
```

The application can store this as a memory.

Much later:

```text
User:
How should I design the database?
```

The system can retrieve:

```text
User uses Supabase.
```

and include it in context.

---

# 18. Explicit vs Implicit Memory

## Explicit memory

The user explicitly says:

```text
Remember that my project uses Supabase.
```

The application can store it.

## Implicit memory

The system identifies potentially useful information from conversation and stores it according to its memory policy.

Production systems should be careful about what information is retained and why.

---

# 19. Memory Is Not Just a Vector Database

A common misconception is:

> LLM memory = vector database.

Not necessarily.

Memory can use:

```text
SQL database
NoSQL database
Redis
Vector database
Graph database
Files
Application state
```

A production system may use several of these together.

---

# 20. Structured Memory

Some information is better stored as structured data.

Example:

```text
user_id: 123
project: CognitRAG
database: Supabase
framework: FastAPI
```

Structured data is useful when exact retrieval or filtering is required.

---

# 21. Semantic Memory

Suppose the conversation contains:

```text
I am building a chatbot that remembers
previous conversations and retrieves relevant
past information.
```

This can be converted into an embedding:

```text
Text
 ↓
Embedding model
 ↓
Vector
 ↓
Vector database
```

For example:

```text
[0.12, -0.42, 0.77, ...]
```

The vector represents semantic information.

---

# 22. Why Embeddings Help Memory

Suppose the user later asks:

```text
How can I make my assistant remember previous chats?
```

The wording differs from:

```text
I am building a chatbot that remembers previous conversations.
```

Embedding similarity can identify that they are semantically related.

Conceptually:

```text
Current query
     ↓
Embedding
     ↓
Vector similarity search
     ↓
Relevant memories
     ↓
Context
     ↓
LLM
```

---

# 23. Memory Retrieval

A typical semantic-memory pipeline is:

```text
User Query
    ↓
Embedding Model
    ↓
Query Vector
    ↓
Vector Search
    ↓
Top-k Relevant Memories
    ↓
Optional Reranking
    ↓
Context Construction
    ↓
LLM
```

This is conceptually similar to RAG.

---

# 24. Memory vs RAG

They use similar mechanisms but solve different problems.

## RAG

Usually retrieves information from an external knowledge source.

```text
PDF
 ↓
Chunks
 ↓
Embeddings
 ↓
Vector DB
 ↓
Retrieve
 ↓
LLM
```

## Memory

Retrieves information about the ongoing user/application interaction.

```text
Past conversations
 ↓
Memory store
 ↓
Retrieve relevant memories
 ↓
LLM
```

Both can use embeddings and vector search.

---

# 25. A Memory-Based Chatbot

A more complete architecture:

```text
                  USER
                   │
                   ▼
              Current Query
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
   Recent History      Memory Retrieval
          │                 │
          │            Embedding Search
          │                 │
          └────────┬────────┘
                   ▼
             Context Builder
                   │
                   ▼
              LLM / Agent
                   │
                   ▼
                Response
                   │
          ┌────────┴────────┐
          ▼                 ▼
   Store Message       Store Memory
```

---

# 26. Chat Templates

Modern LLMs often use structured conversation formats.

Instead of simply concatenating:

```text
Hello
Hi
How are you?
```

the application may represent roles:

```text
system
user
assistant
user
assistant
```

For example:

```text
<system>
You are a helpful assistant.
</system>

<user>
Explain attention.
</user>

<assistant>
Attention allows...
</assistant>
```

The exact format depends on the model.

---

# 27. System, User and Assistant Roles

A conversation commonly contains:

```text
System:
Instructions for the model.

User:
The user's message.

Assistant:
The model's previous response.
```

Example:

```text
System:
You are a helpful AI tutor.

User:
Explain Transformers.

Assistant:
A Transformer uses attention...

User:
Explain attention numerically.
```

The model uses the supplied context to generate the next assistant response.

---

# 28. Why Chat Templates Matter

Different models may expect different special tokens and role formatting.

Conceptually:

```text
Conversation
     ↓
Chat Template
     ↓
Formatted Token Sequence
     ↓
Tokenizer
     ↓
LLM
```

The template tells the model:

- who is speaking
- where the assistant response begins
- how the conversation is structured

---

# 29. Multi-turn Request Flow

Suppose:

```text
Turn 1:
User: Explain Transformers.

Turn 2:
User: What is attention?

Turn 3:
User: Give me a numerical example.
```

The application can construct:

```text
System:
You are a helpful AI tutor.

User:
Explain Transformers.

Assistant:
...

User:
What is attention?

Assistant:
...

User:
Give me a numerical example.
```

Then the LLM generates Turn 3's response.

---

# 30. The LLM Doesn't "Replay" Conversation

Technically, the model is not normally reading some hidden conversation database.

Instead, the application provides tokens representing the relevant conversation:

```text
Conversation history
       ↓
Tokenizer
       ↓
Token sequence
       ↓
Transformer
```

The Transformer processes the supplied context.

---

# 31. Context Construction

A production application may construct:

```text
System Instructions
+
Recent Conversation
+
Retrieved Memories
+
Retrieved Documents
+
Current User Message
```

Then:

```text
Everything
    ↓
Context Builder
    ↓
LLM
```

This is a major AI-engineering concept.

---

# 32. Example Context Builder

Suppose:

### System

```text
You are an AI tutor.
```

### Recent history

```text
User: I am learning Transformers.
Assistant: Great.
```

### Memory

```text
User is studying LLM architecture.
```

### Current query

```text
Explain KV cache.
```

Final context:

```text
SYSTEM:
You are an AI tutor.

MEMORY:
User is studying LLM architecture.

HISTORY:
User: I am learning Transformers.
Assistant: Great.

CURRENT USER:
Explain KV cache.
```

Then:

```text
Context
 ↓
LLM
 ↓
Response
```

---

# 33. Context Budget

Suppose a model supports a large context window.

You still need to allocate space for:

```text
System prompt
+
Memory
+
Retrieved documents
+
Conversation history
+
Current query
+
Output
```

Therefore context management is a budgeting problem.

Conceptually:

\[
\boxed{
Context =
Instructions+
Memory+
History+
Retrieval+
Query
}
\]

while keeping the total within the applicable model context limits.

---

# 34. Summarization Memory

Instead of putting hundreds of old messages into the active context, the application can summarize them.

Original:

```text
Turn 1
Turn 2
Turn 3
...
Turn 50
```

Create:

```text
Conversation Summary:

The user is building a chatbot using FastAPI,
Supabase, Redis and Gemini. They want semantic
memory and contextual responses.
```

Then:

```text
Old conversation
       ↓
Summarizer
       ↓
Compact summary
       ↓
Future context
```

---

# 35. Hierarchical Memory

A sophisticated system may have:

```text
Recent messages
       ↓
Short-term memory

Conversation summaries
       ↓
Medium-term memory

Persistent user/project facts
       ↓
Long-term memory
```

Conceptually:

```text
             Memory System
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
    Recent      Summary     Persistent
    History     Memory       Facts
```

---

# 36. Memory Retrieval Example

Suppose stored memories are:

```text
M1:
User is learning Transformers.

M2:
User is building a chatbot.

M3:
User likes photography.

M4:
User uses Supabase for chat history.
```

Current query:

```text
How should I store conversation embeddings?
```

Semantic retrieval may rank:

```text
M4 → highly relevant
M2 → relevant
M1 → somewhat relevant
M3 → irrelevant
```

The application might retrieve only the relevant memories.

This prevents irrelevant information from consuming context.

---

# 37. Memory Retrieval Is a Ranking Problem

The system usually does not retrieve everything.

It attempts to find:

\[
\boxed{
\text{Most relevant memories for the current query}
}
\]

Common signals include:

```text
Vector similarity
Keyword search
Metadata filtering
Hybrid search
Reranking
Recency
Importance
```

---

# 38. Recency + Relevance

Suppose two memories are semantically similar:

```text
Memory A:
Created yesterday.

Memory B:
Created two years ago.
```

A memory system may consider:

```text
Semantic relevance
+
Recency
+
Importance
```

to determine which information should enter the context.

---

# 39. Memory Lifecycle

A complete memory system can be viewed as:

```text
Conversation
     ↓
Memory Extraction
     ↓
Memory Storage
     ↓
Memory Indexing
     ↓
Future Query
     ↓
Memory Retrieval
     ↓
Memory Ranking
     ↓
Context Injection
     ↓
LLM
```

---

# 40. Memory Extraction

Not every message should necessarily become long-term memory.

For example:

```text
User:
What is 2 + 2?
```

Probably no persistent memory is required.

But:

```text
User:
My project uses PostgreSQL and FastAPI.
```

may be useful later.

Therefore an application can use rules or an LLM-based extraction process:

```text
Conversation
    ↓
Should this be remembered?
    ↓
Yes / No
    ↓
If yes → store memory
```

---

# 41. Memory Storage Architecture

A possible production architecture:

```text
                    Application
                         │
                         ▼
                    FastAPI
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           Redis       SQL DB    Vector DB
              │          │          │
           Cache      Messages    Memories
```

For example:

```text
SQL:
conversation messages

Redis:
recent state / cache / events

Vector DB:
semantic memories
```

The exact architecture depends on the application.

---

# 42. Practical Memory-Based Chatbot Architecture

A memory-based chatbot could look like:

```text
                 User
                  │
                  ▼
             Chat UI
                  │
                  ▼
               FastAPI
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
      Redis    SQL DB    Vector DB
        │         │          │
      Cache    History    Memories
        │         │          │
        └─────────┼──────────┘
                  ▼
             Context Builder
                  │
                  ▼
                 LLM
                  │
                  ▼
              Response
```

---

# 43. Memory Does Not Change the LLM's Weights

Suppose:

```text
Before conversation:
Model weights = W
```

User says:

```text
My name is Alex.
```

The application stores:

```text
Alex
```

The model weights remain:

\[
W
\]

There is no training step.

Therefore:

```text
Conversation memory
      ↓
Does NOT normally update model weights
```

Instead:

```text
Memory
 ↓
Retrieved
 ↓
Added to context
```

---

# 44. Memory vs Fine-tuning

## Memory

```text
Store information
 ↓
Retrieve later
 ↓
Put into context
```

No model-weight update.

## Fine-tuning

```text
Training examples
 ↓
Loss
 ↓
Backpropagation
 ↓
Update model weights
```

These are fundamentally different mechanisms.

---

# 45. Memory vs RAG vs Fine-tuning

| Technique | Main purpose |
|---|---|
| Conversation memory | Remember relevant user/conversation information |
| RAG | Retrieve external knowledge |
| Fine-tuning | Change/adapt model behavior or capabilities |
| Prompting | Control behavior through context/instructions |

A production system can combine all four.

---

# 46. Complete Multi-turn Architecture

```text
                         USER
                           │
                           ▼
                    Current Message
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       Recent History            Memory Retrieval
              │                         │
              │                    Vector / SQL
              │                         │
              └────────────┬────────────┘
                           ▼
                     Context Builder
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
           System       Memories      History
              │            │            │
              └────────────┼────────────┘
                           ▼
                      Chat Template
                           │
                           ▼
                        Tokenizer
                           │
                           ▼
                       Transformer
                           │
                           ▼
                    Next-token logits
                           │
                           ▼
                     Decoding strategy
                           │
                           ▼
                       Response
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
            Store message       Extract memory
                 │                   │
                 └─────────┬─────────┘
                           ▼
                     Memory Store
```

---

# 47. Key Concept: Context Engineering

Context Engineering asks:

> What information should we put into the model's context, in what order, and within what token budget?

For example:

```text
System instructions
       ↓
Important memories
       ↓
Retrieved documents
       ↓
Recent conversation
       ↓
Current query
```

The quality of this context can strongly affect the model's response.

---

# 48. Context Management Strategies

Common strategies include:

### 1. Sliding window

Keep recent messages.

### 2. Summarization

Compress old conversation.

### 3. Retrieval

Retrieve relevant old messages.

### 4. Memory extraction

Store important facts separately.

### 5. Hybrid approach

Combine several strategies.

A production chatbot often uses a hybrid approach.

---

# 49. Most Important Mental Model

Remember:

```text
              LLM
               ▲
               │
          Context
               ▲
               │
      ┌────────┴────────┐
      │                 │
Conversation         Memory
 History           Retrieval
      │                 │
      └────────┬────────┘
               │
            Storage
```

The LLM does not need to permanently memorize every interaction.

The application can retrieve the right information and provide it at inference time.

---

# 50. Training vs Memory

### Training

```text
Dataset
 ↓
Model
 ↓
Loss
 ↓
Backpropagation
 ↓
Weights updated
```

### Memory

```text
Conversation
 ↓
Store
 ↓
Retrieve
 ↓
Context
 ↓
Model inference
```

Memory does not require retraining the model.

---

# 51. Self-Check Questions

Before moving to Lesson 13, make sure you can answer:

1. Does an LLM automatically remember every previous conversation?
2. What is conversation history?
3. What is a conversation turn?
4. What is the context window?
5. What is the difference between context and memory?
6. What is short-term conversational memory?
7. What is long-term memory?
8. What is structured memory?
9. What is semantic memory?
10. Why are embeddings useful for memory?
11. What is memory retrieval?
12. How is memory related to RAG?
13. How is memory different from fine-tuning?
14. What is a chat template?
15. Why are system/user/assistant roles important?
16. Why can't we always send the entire conversation?
17. What is a sliding-window strategy?
18. What is conversation summarization?
19. What is memory extraction?
20. What is memory ranking?
21. Why should irrelevant memories not be inserted into context?
22. What is context budgeting?
23. What is context engineering?
24. How can SQL, Redis and a vector database have different roles?
25. Does storing a memory normally update the model weights?
26. What is the complete multi-turn chatbot pipeline?

---

# 52. Next Lesson — Context Engineering

Next we go deeper into:

> Given a limited context window, what information should we actually send to the LLM?

Topics:

```text
Context Window
      ↓
Token Budget
      ↓
System Instructions
      ↓
Conversation History
      ↓
Memory
      ↓
RAG Documents
      ↓
Tool Results
      ↓
Context Ordering
      ↓
Compaction / Summarization
      ↓
Prompt Caching
      ↓
LLM
```

This is especially important for production-grade RAG and memory-based AI systems.
