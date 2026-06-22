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
