# Chains in LangChain (LCEL)

As mentioned in Step 1, LangChain uses LCEL (LangChain Expression Language) to build pipelines using the `|` (pipe) operator.

Under the hood, LCEL is built entirely around a single interface called the `Runnable` protocol. Every component in LangChain (Models, Prompts, Parsers, Retrievers) extends `Runnable`. When you pipe them together, they form a `RunnableSequence`.

Because everything shares this protocol, you get powerful distributed computing features out of the box.

---

## 1. RunnablePassthrough (Preserving State)

Often, a step in your chain needs to generate new data, but a later step still needs the original user input. `RunnablePassthrough` allows you to pass data through a step without altering it.

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Tell a joke about {topic} in {language}")

# Imagine we have a function that detects the user's language based on their IP
def detect_language(topic_dict):
    return "Spanish" # Hardcoded for example

# We want to keep the 'topic' from the user, but ADD the 'language' dynamically
chain = (
    {"topic": RunnablePassthrough(), "language": detect_language} 
    | prompt 
    | llm 
    | parser
)

# We only pass 'topic'. The chain dynamically injects 'language'.
result = chain.invoke("computers") 
```

---

## 2. RunnableParallel (Concurrent Execution)

In a real app, latency is your biggest enemy. If you need to summarize a document and translate it, doing it sequentially takes twice as long. `RunnableParallel` executes multiple runnables at the exact same time on separate threads.

```python
from langchain_core.runnables import RunnableParallel

chain_a = prompt_summary | llm | parser
chain_b = prompt_translate | llm | parser

# Execute both chains concurrently
parallel_chain = RunnableParallel(
    summary=chain_a,
    translation=chain_b
)

# Total execution time is bounded by the slowest chain, NOT the sum of both.
result = parallel_chain.invoke({"text": "Massive document text..."})
print(result["summary"])
print(result["translation"])
```

---

## 3. Dynamic Routing (RunnableBranch)

Sometimes your chain needs "If/Else" logic. For example: If the user asks a math question, route to the Math LLM. If they ask a legal question, route to the Legal LLM.

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "math" in x["question"].lower(), math_chain),
    (lambda x: "law" in x["question"].lower(), law_chain),
    general_chain # Default fallback if no conditions are met
)

result = branch.invoke({"question": "What is 2+2?"}) # Routes to math_chain
```

---

## Part 2: Memory (State Management)

Because LCEL chains are stateless, you must explicitly inject history. In production, you don't store chat history in RAM (it wipes when the server restarts). You store it in a database like Redis or Postgres.

LangChain handles this via `RunnableWithMessageHistory`. It wraps your chain and automatically handles the database read/write operations before and after the LLM runs.

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory

# 1. Define how to fetch a specific user's chat session from the database
def get_redis_history(session_id: str):
    return RedisChatMessageHistory(session_id, url="redis://localhost:6379")

# 2. Wrap your existing chain
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_redis_history,
    input_messages_key="user_input",
    history_messages_key="chat_history" # Must match the MessagesPlaceholder in your prompt
)

# 3. Invoke with a specific Session ID
chain_with_memory.invoke(
    {"user_input": "Hi, my name is Alice."},
    config={"configurable": {"session_id": "user_123"}}
)
```

---

## Part 3: LangChain CLI & LangServe (Deployment)

Once you build your robust LCEL chain, how do you expose it to a frontend (like a React app)? You could write a FastAPI server from scratch, but writing endpoints that handle streaming Server-Sent Events (SSE) for LLMs is incredibly tedious.

LangServe is LangChain's deployment solution. It automatically wraps your LCEL chains in production-ready FastAPI endpoints. The LangChain CLI is the tool used to scaffold these projects.

### Step-by-Step Production Deployment

#### 1. Scaffold the App via CLI
Just like `npx create-react-app`, LangChain provides scaffolding. Open your terminal and run:

```bash
pip install langchain-cli
langchain app new my-ai-api
```
This generates a full folder structure with a `server.py` file.

#### 2. Add your Chain to LangServe
Inside `server.py`, you define your chain and use `add_routes`.

```python
from fastapi import FastAPI
from langserve import add_routes
from my_custom_chain import chain # The chain you built in Part 1

app = FastAPI(title="My LangChain Server")

# LangServe automatically generates the REST API
add_routes(
    app,
    chain,
    path="/my-chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

#### 3. What LangServe gives you for free:
Once that server is running, LangServe automatically generates:

* **POST `/my-chain/invoke`**: A standard endpoint that waits for the full response and returns JSON.
* **POST `/my-chain/stream`**: An endpoint configured for Server-Sent Events (SSE). It streams tokens one by one to your frontend.
* **POST `/my-chain/batch`**: Accepts an array of inputs and processes them concurrently.
* **A Built-in UI (`/my-chain/playground`)**: LangServe generates a beautiful web interface where you can test your chain, view the intermediate steps, and tweak parameters without writing frontend code.

By combining LCEL (for complex, parallelized data logic) with LangServe (for instant, streaming-ready REST APIs), you bypass hundreds of hours of boilerplate backend engineering.
