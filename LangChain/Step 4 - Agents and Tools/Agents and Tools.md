# Agents & Tools (The Execution Engine)

In Step 2, we looked at Chains, where the execution path is hardcoded by you, the developer (e.g., Prompt $\rightarrow$ Model $\rightarrow$ Parser). While predictable, Chains fail when faced with complex, non-linear problems.

An Agent flips this paradigm. Instead of hardcoding a sequence, you give an LLM a list of Tools (Python functions) and ask it to solve a problem. The LLM acts as a continuous loop: it evaluates the user's goal, decides which tool to call, inspects the tool's output, and decides whether it has solved the problem or needs to call another tool.

---

## 1. The Core Abstraction: Tools (`@tool`)
In LangChain, a Tool is an interface that allows an LLM to interact with the outside world (databases, web APIs, local file systems). To turn any standard Python function into a LangChain tool, you use the `@tool` decorator.

```python
from langchain_core.tools import tool

@tool
def calculate_amortization(principal: float, interest_rate: float, years: int) -> str:
    """
    Calculates the monthly mortgage payment for a loan. 
    Use this tool whenever a user asks about monthly mortgage costs, house loans, or interest payments.
    """
    # Computer science student note: The docstring above is NOT just comments.
    # LangChain passes this exact text description to the LLM so the LLM knows *when* and *how* to use this tool.
    monthly_rate = (interest_rate / 100) / 12
    months = years * 12
    payment = principal * (monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
    return f"The monthly payment is ${payment:.2f}"
```

When an Agent runs, it reads the function name, its arguments, and the docstring description. It uses this schema to generate a Tool Call containing structured arguments matching your function's signature.

---

## 2. The ReAct Framework (Reasoning and Acting)
Most modern agents operate on the ReAct (Reason + Act) design pattern. This breaks the LLM's execution cycle down into a strict, programmatic loop:

$$\text{Thought} \longrightarrow \text{Action (Tool Selection)} \longrightarrow \text{Observation (Tool Output)} \longrightarrow \text{Thought (Repeat/Finish)}$$

* **Thought**: The model analyzes the current state and reasons about what to do next.
* **Action**: The model outputs a structural call requesting a tool (e.g., `calculate_amortization(principal=400000, interest_rate=6.5, years=30)`).
* **Observation**: LangChain catches this request, executes your Python function, fetches the return value, and appends it back into the model's chat log.
* **Thought**: The model reads the tool's output and determines if it can formulate a final response for the user.

Here is an interactive simulator showing exactly how an Agent steps through a complex multi-tool execution path dynamically based on a user's prompt.

---

## 3. Creating an Agent in Code
In modern LangChain, the standard way to build a reliable, production-ready agent is using the `create_tool_calling_agent` function combined with an `AgentExecutor`.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

# 1. Initialize our LLM and list of tools
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
tools = [calculate_amortization] # You can add multiple tools here

# 2. Build a prompt template that supports agent states
# Agents MANDATORILY require a 'messages_placeholder' named 'agent_scratchpad'
# This is where LangChain streams the intermediate thoughts and tool results
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial analysis agent. Use your tools to solve user queries."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad") 
])

# 3. Construct the logical Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# 4. Create the Runtime Executor (The loop mechanism)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, # Logs thoughts, actions, and observations directly to the terminal
    max_iterations=5 # Safety fallback to prevent infinite loops if the agent gets confused
)

# 5. Invoke the system
response = agent_executor.invoke({"input": "What would my monthly cost be on a $500,000 home loan at 6.5% interest over 30 years?"})
print(response["output"])
```

---

## 4. Advanced System Design: Moving from Agents to LangGraph
While `AgentExecutor` works well for straightforward, linear task-solving, it has massive architectural constraints in complex corporate environments:
* You cannot easily enforce control-flow rules (e.g., "The agent must ALWAYS validate the math with a tool before emailing").
* It struggles with state persistence across long periods.

To address this, LangChain has heavily moved its advanced Agent architecture over to a companion library called **LangGraph**.

LangGraph models applications as stateful multi-agent computational graphs (State Machines). Instead of letting a single LLM blindly loop through tools, you define structured Nodes (Python code or LLM calls) and Edges (conditional routing paths). This allows you to build multi-agent teams where one Agent acts as a Planner, another acts as a Researcher, and a third acts as a Code Reviewer, passing states deterministically between each other.

---

## Summary of the LangChain Ecosystem
You have now completed the step-by-step exploration of the entire foundational LangChain architecture:
* **Models, Prompts, & Parsers**: Standardizing LLM input interfaces, generating templates, and enforcing runtime structural data extraction (via Pydantic/Native JSON modes).
* **Chains & Memory**: Combining those pieces into sequential pipelines (LCEL), managing data passthroughs, parallel execution paths, and persisting history using database integration.
* **CLI & LangServe**: Scaffolding production microservices and automatically exposing endpoints for streaming data arrays.
* **Retrieval & RAG**: Augmenting the LLM context window by loading, chunking, embedding, and checking vector indices to search across custom data stores.
* **Agents & Tools**: Turning the LLM into a dynamic router that evaluates code-level tools inside a ReAct reasoning loop to solve complex tasks.

This foundational roadmap gives you the exact engineering knowledge needed to build, scale, and deploy robust LLM applications in the real world.
