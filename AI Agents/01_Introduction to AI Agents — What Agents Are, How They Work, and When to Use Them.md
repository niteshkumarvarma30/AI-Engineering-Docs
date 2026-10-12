# Introduction to AI Agents: What Agents Are, How They Work, and When to Use Them

> **Module**: Agentic AI & Multi-Agent Systems  
> **Topic**: Introduction to AI Agents  
> **Source**: [AIML Companion — Introduction to AI Agents](https://aimlcompanion.ai/module/aiAgents/agentsIntro#main-content)

---

## 🎯 In This Module

- **What distinguishes AI agents from chatbots**: Autonomy, tool use, and goal-oriented loops.
- **The core agent loop**: `Perceive → Reason → Act → Observe` and how it drives agent behavior.
- **Agent components**: LLM brain, tools, memory, and planning, and how they work together.
- **When to use agents vs. simple prompts**: The cost-benefit tradeoff.
- **Real-world agent architectures**: Used in coding assistants, research agents, and automation.

---

## 1. The Big Picture: How It Works, Start to Finish

An **AI agent** is a language model placed inside a loop. You give it a goal, and it repeats four steps until the goal is met:
1. **Perceive** the current state
2. **Reason** about the best next move
3. **Act** with a tool
4. **Observe** what happened, and go around again

The loop, plus tools and memory, is what separates an agent from a chatbot that answers once and stops.

```
       +-------------------------------------------------------------+
       |                       THE AGENT LOOP                        |
       |         Repeats until the goal is met or a limit stops it   |
       |                                                             |
       |                    01. PERCEIVE                             |
       |                   (read the state)                          |
       |              goal + last result + memory                    |
       |                           │                                 |
       |                           ▼                                 |
       |                      02. REASON                             |
       |                 (LLM picks next move)                       |
       |              search? run code? or done?                     |
       |                           │                                 |
       |                           ▼                                 |
       |                       03. ACT                               |
       |                    (call a tool)                            |
       |              search, code, API, database                    |
       |                           │                                 |
       |                           ▼                                 |
       |                      04. OBSERVE                            |
       |                   (check the result)                        |
       |                worked? goal met? else loop                  |
       |                           │                                 |
       |                           ▼                                 |
       |               ↻ not done? loop again                        |
       +-------------------------------------------------------------+
```

### The Whole Idea in One Line
$$\text{Perceive the state} \longrightarrow \text{Reason about next move} \longrightarrow \text{Act with a tool} \longrightarrow \text{Observe the result} \longrightarrow \text{Loop again}$$

---

## 2. End-to-End Walkthrough: One Agent Runs the Loop

**Goal**: *"Is Tokyo's metro area bigger than Delhi's, and by how much?"*

A chatbot would guess both numbers in one shot. An agent loops instead, with two tools: `search_web` and `finish`.

### Step 1: Set the goal, hand over the tools
- `search_web(query)`: Looks up current facts the model does not reliably know.
- `finish(answer)`: The agent calls this when it believes the goal is met.
- **Safety limit**: 10 loop turns, so it can never run forever.
- **Memory**: Starts empty. The agent knows neither population yet:
  ```python
  memory = {}  # (nothing observed so far)
  ```

### Step 2: Loop Turn 1 — Get Tokyo's number
- **Perceive**: The agent reads the goal and sees it knows neither number.
- **Reason**: It decides it needs Tokyo first.
- **Act**: It calls the search tool:
  ```text
  search_web('Tokyo metro area population') = ~37 million
  ```
- **Observe**: Saved to memory: `{ tokyo: "~37M" }`.

### Step 3: Loop Turn 2 — Get Delhi's number
- **Perceive**: Tokyo is now known, Delhi is not.
- **Reason**: Same move, different city.
- **Act**: Search again:
  ```text
  search_web('Delhi metro area population') = ~34 million
  ```
- **Observe**: Now memory holds both: `{ tokyo: "~37M", delhi: "~34M" }`.

### Step 4: Loop Turn 3 — It has enough, so it stops
- **Perceive**: Both numbers are in memory.
- **Reason**: No tool needed, it can answer now.
- **Act**: Call `finish()`:
  $$\text{Tokyo} = \sim 37\text{M}, \quad \text{Delhi} = \sim 34\text{M}$$
  $$37\text{M} - 34\text{M} = \sim 3\text{ million (computed from observations, not guessed)}$$
- **Answer**: *"Tokyo's metro is ~3M larger than Delhi's."*

> **Key Difference**: Three loop turns and two tool calls. The chatbot answers in one pass with no way to check itself; the agent gathered evidence first, then decided it was done.

---

## 3. 💡 Mental Model: An Agent Is an Intern With Tools

```
+----------------------------------------------------------------------------------+
| Component         | Intern Equivalent                                            |
+-------------------+--------------------------------------------------------------+
| 🧠 LLM Brain      | The intern's knowledge and reasoning ability                 |
| 🛠️ Tools          | Access to email, spreadsheets, databases, web browser        |
| 📝 Memory         | Notes and context from previous tasks                        |
| 🔄 Agent Loop     | Check task -> Think -> Take action -> Check result -> Repeat |
+----------------------------------------------------------------------------------+
```

* **Using a chatbot**: Like asking the intern a question in the hallway.
* **Using an agent**: Like handing the intern a task and letting them work out the steps, use tools to finish it, and check their own work.

---

## 4. Agents vs. Chatbots

Why an agent exists: **it adds the action loop a chatbot does not have**.

A chatbot generates text. An agent takes actions. The difference is the action loop:

```
Chatbot:  User ──> LLM ──> Response (Done)

Agent:    User ──> LLM ──> Tool Call ──> Result ──> LLM ──> Tool Call ──> ... ──> Response
```

| Feature | Chatbot | Agent |
| :--- | :--- | :--- |
| **Input / Output** | Single input $\rightarrow$ Single output | Iterative multi-step turns |
| **Autonomy** | None (Static answer) | Decides next steps without human guidance |
| **Tool Use** | No tool access | Calls APIs, runs code, queries databases |
| **External Systems** | Cannot affect external systems | Can mutate databases, fetch web pages, execute scripts |
| **Session Lifecycle**| Session ends after single response | Continues looping until task is complete |
| **Objective** | Conversational response | Works toward a defined objective (Goal-oriented) |
| **Cost & Latency** | Low: 1 LLM call (~$0.001, 0.5s) | Variable: 5–20+ LLM calls per task (~$0.021–$0.50+, 5s–30s+) |

---

## 5. The Agent Loop: Perceive → Reason → Act → Observe

Every agent runs the same core loop:
1. **Perceive**: Read the current state (user request, tool results, memory).
2. **Reason**: Decide what to do next (the LLM's thinking step).
3. **Act**: Execute an action (call a tool, generate output).
4. **Observe**: Process the result and update state.

### Universal Agent Loop (Pseudocode)

```python
messages = [system_prompt, user_task]

while not done:
    response = llm.generate(messages)       # Reason
    if response.has_tool_calls:
        results = execute_tools(response)    # Act
        messages.append(results)             # Observe
    else:
        return response.text                 # Done
```

### When Does the Loop Terminate?
- The LLM decides the task is complete (calls `finish()` or returns final text).
- Maximum iterations reached (safety limit).
- An unrecoverable error occurs or budget/timeout triggers.

> Every framework (**LangChain**, **CrewAI**, **OpenAI SDK**) implements this same loop. Once you know it, you can debug any of them.

---

## 6. Stop Rules & Failure Modes: What If the Agent Never Stops?

If a tool keeps failing (e.g. search errors, no results found), an agent might never decide it is done. It would loop forever, burning API credits.

```
Example Failure Scenario:
"The search tool keeps failing, so this agent never decides it is done.
Only the finish tool is on. Will the run stop? Press Run."

Without stopping rules:
Turns: 1 -> 2 -> ... -> 250 (stuck in loop, never calls finish)
Tokens: 0 -> 1,000 -> 10,000 -> 50,000+ ($$$ wasted)
```

Without proper stopping conditions, agents can:
1. **Loop forever**, burning API credits.
2. **Hallucinate tool calls** that don't exist.
3. **Repeat the same action** expecting different results.

### Production Stopping Strategies

| Stop Rule | How It Works | Example Setting |
| :--- | :--- | :--- |
| **1. Max Iterations** | Hard turn limit on the loop counter | Fires at Turn 10 or 15 |
| **2. Token Budget** | Stop when accumulated tokens/cost exceed threshold | Stops when cost exceeds ~$0.05 or 8,000 tokens |
| **3. Wall-Clock Timeout**| Terminates if execution exceeds duration | Stops at 60 seconds |
| **4. Explicit Finish Tool**| Agent explicitly calls `finish(answer)` | Only fires when agent decides it is done |
| **5. Confidence Check** | Periodic reflection / check: *"Are you done?"* | Runs every 3 iterations |

---

## 7. Agent Components: LLM Brain, Tools, Memory, Planning

```
                                  +-------------------+
                                  |    GUARDRAILS     |
                                  | 🔒 PII Filter     |
                                  | 💰 Cost Limiter   |
                                  +---------+---------+
                                            │
                                            ▼
+-----------------------------------------------------------------------------------+
| 1. 🧠 LLM BRAIN (Reasoning)                      2. 🛠️ TOOLS (Hands)              |
| The reasoning engine. Decides what to do,        Functions the agent can call:    |
| interprets results, generates responses.         Web search, Code execution, DB,  |
| Agent quality depends heavily on model.          APIs. Lets agent affect world.   |
|                                                                                   |
| 3. 📝 MEMORY (Context)                           4. 🗺️ PLANNING (Strategy)        |
| Short-term: Conversation history in session.     How agent breaks down goals.     |
| Long-term: Stored knowledge across sessions      Simple agents react;             |
| (Vector stores, databases).                      sophisticated agents plan ahead. |
+-----------------------------------------------------------------------------------+
```

### Component Diagnostics: Which Component Broke?

When an agent fails, diagnose which component broke first:
- **Bad tool selection** $\rightarrow$ Tool descriptions need work (tool descriptions are prompts).
- **Bad reasoning** $\rightarrow$ Call for a stronger model or better prompt instructions.
- **Lost context** $\rightarrow$ Memory-management problem (overflowing tokens, lack of summarization).
- **Infinite loop** $\rightarrow$ Missing or faulty stopping condition.

---

## 8. When to Use Agents (and When Not To)

Agents are not always the right choice. They add **cost, latency, and unpredictability**.

```
Simple prompt = 1 LLM call (~0.5s, $0.001)
Agent         = 5–20+ LLM calls per task (~10–30s, $0.02–$0.20+)
```

### Autonomy Spectrum: 4 Levels

```
Level 1: Prompt Only
[LLM]
Text in, text out. (e.g. Rewrite text, sentiment classification)

Level 2: One Tool Call
[LLM] ──> [Tool] ──> [LLM]
One lookup, then answer. (e.g. "What is the weather in Paris right now?")

Level 3: Fixed Chain
[LLM] ──> [Tool] ──> [Tool] ──> [Tool] ──> [LLM]
Steps set strictly in code. (e.g. Extract PDF receipt -> Validate -> Email)

Level 4: Agent Loop
[LLM] ──> [Tool] ──> [LLM] ──> [Tool] ──> ...
The model dynamically picks each next step based on observation. (e.g. Fix build, deep research)
```

### Task Decision Guide

| Task | Simplest Working Level | Why |
| :--- | :--- | :--- |
| **Rewrite an email** | **1. Prompt Only** | Pure text generation; no tools needed. |
| **Weather in Paris right now** | **2. One Tool Call** | Needs live lookup (`get_weather`), then answers. |
| **PDF receipt to email** | **3. Fixed Chain** | Known, linear steps (Extract $\rightarrow$ format $\rightarrow$ send). |
| **Fix build / Debug errors** | **4. Agent Loop** | Unknown path: read error $\rightarrow$ inspect code $\rightarrow$ edit $\rightarrow$ run tests $\rightarrow$ retry if failed. |

### When to Use vs. Avoid

* **USE AGENTS WHEN**:
  - Task requires multiple steps with decisions between them.
  - You need tool use across different systems (search + code + database).
  - The path to the answer is not known in advance.
  - A human would need to research, iterate, and adapt.

* **DON'T USE AGENTS WHEN**:
  - A single API call or prompt suffices (90% of tasks!).
  - Steps are fixed and known (use a pipeline/chain instead).
  - Low latency matters (agents make multiple sequential LLM calls).
  - Deterministic reliability is critical (agents can fail unpredictably).

---

## 9. ⚡ Cheat Sheet: Scan in Under 2 Min

- **What**: An AI agent is an LLM placed inside a loop. You give it a goal; it repeats `Perceive → Reason → Act → Observe` until the goal is met or a limit stops it.
- **Why it matters**: It is how software goes from answering to doing. The loop lets it check its own work and try again, which a chatbot, a script, and plain RAG cannot.
- **The one formula**: `while not done: P → R → A → O` (Observe feeds back to Perceive. That self-check edge is the whole difference).
- **Is it an agent? The Test**:
  - Chatbot $\rightarrow$ **No**: replies once.
  - RPA script $\rightarrow$ **No**: fixed rules.
  - Plain RAG $\rightarrow$ **No**: search then answer.
  - Agent $\rightarrow$ **YES**: loops + self-checks.
- **When to use (escalate only if simpler fails)**:
  $$\text{1 prompt} \longrightarrow \text{pipeline} \longrightarrow \text{function call} \longrightarrow \text{agent} \longrightarrow \text{multi-agent}$$
- **Watch out ⚠**:
  - Agents add cost, latency, unpredictability; most tasks just need a prompt.
  - Never ship without a stopping condition (it can loop forever).
  - Tool descriptions are prompts: vague ones cause wrong tool picks.
  - 1 prompt = 1 call; an agent = 5 to 20+ calls per task.
- **Make it work (your first agent)**:
  - Pick **ONE** task (e.g. research a topic).
  - Give it **2 to 3 tools**, no more.
  - Write a clear system prompt: role + when to stop.
  - Set `max_iterations = 10`, log every step.

### Interview Q & A
- **What makes it an agent, not a chatbot?**  
  *A loop with self-check: it observes its result and picks the next move. A chatbot replies once.*
- **Why are RPA scripts and RAG not agents?**  
  *Both run a fixed path with no "did that work? try again" decision.*
- **What stops an agent looping forever?**  
  *A stopping condition: max_iterations, token budget, timeout, or a finish tool.*
- **Agent keeps picking the wrong tool?**  
  *Fix the tool descriptions first; the LLM reads them like a prompt.*

---

## 10. 💻 Code Examples

### Example 1: Simple Agent Loop (From Scratch)
A minimal agent loop using OpenAI's Responses / Function Calling API:

```python
# ============================================================
# SETUP: Get your OpenAI API key (this example uses OpenAI Responses API)
# OpenAI (PAID): https://platform.openai.com/api-keys
#
# FREE alternatives for learning:
#   Gemini: https://aistudio.google.com/apikey (30 RPM, 1,500 RPD free)
#   Groq:   https://console.groq.com/keys    (30 RPM, 14,400 RPD free)
# ============================================================
from openai import OpenAI
import json

client = OpenAI()  # Set OPENAI_API_KEY in .env

# Define tools the agent can use
tools = [
    {
        "type": "function",
        "name": "search_web",
        "description": "Search the web for current information. Use when you need facts you don't know.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "type": "function",
        "name": "finish",
        "description": "Call this when you have the final answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The final answer"}
            },
            "required": ["answer"]
        }
    }
]

def execute_tool(name: str, args: dict) -> dict:
    """Execute a tool by name. Replace with real implementations."""
    if name == "search_web":
        return {"results": f"Search results for: {args.get('query', '')}"}
    return {"error": f"Unknown tool: {name}"}

def run_agent(task: str, max_iterations: int = 10):
    """Minimal agent loop using OpenAI Responses API."""
    input_messages = [
        {"role": "system", "content": "You are a helpful research agent. Use tools to find information. Call finish() when done."},
        {"role": "user", "content": task}
    ]

    for i in range(max_iterations):
        response = client.responses.create(
            model="gpt-5.6-terra",
            input=input_messages,
            tools=tools
        )

        # Keep the model's output items (reasoning + function calls): each
        # function_call_output must follow the call it answers
        input_messages += response.output

        # Check each output item
        for item in response.output:
            if item.type == "function_call":
                args = json.loads(item.arguments)
                if item.name == "finish":
                    return args["answer"]

                # Execute tool and add result
                result = execute_tool(item.name, args)
                input_messages.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result)
                })

    return "Max iterations reached"

# Run it
if __name__ == "__main__":
    answer = run_agent("What is the population of Tokyo?")
    print(answer)
```

---

### Example 2: Agent vs. Simple Prompt Comparison

```python
from openai import OpenAI

client = OpenAI()  # Set OPENAI_API_KEY in .env

# === 1. SIMPLE PROMPT (No agent needed) ===
# Task: Classify an email
response = client.responses.create(
    model="gpt-5.6-terra",
    input=[{
        "role": "user",
        "content": "Classify this email as: billing, technical, or general.\n\nEmail: My invoice shows the wrong amount."
    }]
)
print(f"Classification: {response.output_text}")
# Output: "billing" (one call, ~0.5s, done)

# === 2. AGENT (Needed for multi-step tasks) ===
# Task: Research a topic and write a summary with sources
# This requires: search -> read -> synthesize -> cite
# A single prompt cannot do this: it needs real-time web access & iterative verification

def research_agent(topic: str):
    """Agent that searches, reads, and synthesizes."""
    tools = [search_tool, read_page_tool, finish_tool]
    input_messages = [
        {"role": "system", "content": "Research the topic. Search for sources, read them, then write a summary with citations."},
        {"role": "user", "content": f"Research: {topic}"}
    ]

    for _ in range(15):  # Max 15 iterations
        response = client.responses.create(
            model="gpt-5.6-terra",
            input=input_messages,
            tools=tools
        )
        # ... (same loop pattern as above)

# Key insight: Use the simplest approach that works!
# Prompt -> Function Call -> Pipeline -> Agent -> Multi-Agent
```
