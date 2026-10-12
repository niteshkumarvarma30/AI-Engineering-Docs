# ReAct Pattern: Reasoning and Acting in Interleaved Loops

> **Module**: Agentic AI & Multi-Agent Systems  
> **Topic**: ReAct Pattern  
> **Source**: [AIML Companion — ReAct Pattern](https://aimlcompanion.ai/module/aiAgents/reactPattern)

---

## 🎯 In This Module

- **ReAct = Reasoning + Acting Interleaved**: Understanding the industry-standard agent architecture.
- **The Thought-Action-Observation loop**: How explicit reasoning before and after tool calls drives autonomous execution.
- **Self-correction patterns**: How helpful error observations let agents recover from typos and bad parameters without crashing.
- **Stopping conditions in depth**: Combining iteration limits, explicit finish tools, token budgets, and wall-clock timeouts.
- **When ReAct fails**: Detecting and mitigating infinite loops, hallucinated tools, over-thinking, and early stops.
- **When to use ReAct vs. simpler patterns**: Selecting between Prompts, Fixed Chains, ReAct, and Plan-and-Execute.

---

## 1. The Big Picture: How ReAct Works

**ReAct** (introduced by Yao et al., 2022) is the foundational pattern behind modern AI agents. It alternates between **verbalized internal reasoning** and **external environmental actions**.

```
+-------------------------------------------------------------------------------+
|                               THE ReAct LOOP                                  |
|                                                                               |
|        +-------------------> 01. THOUGHT <-------------------+                |
|        |           Reason out loud about next move           |                |
|        |                 "Think before acting"               |                |
|        |                           │                         |                |
|        |                           ▼                         |                |
|   03. OBSERVATION ───────────> 02. ACTION ───────────────────+                |
|   Read tool output             Call chosen tool                               |
|   Feeds the next thought       Search, API, code, DB                          |
|        │                                                                      |
|        ▼ (When goal is met)                                                   |
|   [Final Answer / finish()]                                                   |
+-------------------------------------------------------------------------------+
```

### The ReAct Formula
$$\mathbf{\text{Thought } \longrightarrow \text{Action } \longrightarrow \text{Observation } \longrightarrow \text{loop back to Thought}}$$

* **Thought**: The agent reflects on its current progress, determines what information is missing, and plans its immediate next move.
* **Action**: The agent selects a specific tool and serializes structured arguments.
* **Observation**: The agent ingests the tool's output into its context window, updating its working memory.

---

## 2. Walkthrough: One ReAct Trace (Typo Recovery)

**Task**: *"What's the weather in Paris?"*

Watch how the agent recovers autonomously from a human or generated typo without failing the session:

```
Turn 1: A Reasonable Try That Stumbles
--------------------------------------
[Thought]: "I need the current weather in Paris. I will call the weather tool."
[Action]:  get_weather(city="Pariss")  <-- Typo!
[Observe]: Error: "City 'Pariss' not found. Did you mean 'Paris'?"

Turn 2: Read the Observation & Self-Correct
-------------------------------------------
[Thought]: "That was a typo, and the error observation suggested 'Paris'.
            Let me retry with the corrected spelling."
[Action]:  get_weather(city="Paris")
[Observe]: {"temp": 18, "condition": "sunny", "humidity": 65}

Turn 3: Conclude & Answer
-------------------------
[Thought]: "I have the verified weather data. No more tool calls are needed."
[Action]:  finish(answer="Paris is currently 18°C and sunny.")

Result: "Paris is currently 18°C and sunny."
```

> **Why This Worked**: An **Act-only** agent without reasoning traces would have repeated the broken call or crashed. Because ReAct **re-reads the observation and thinks**, it spotted the error message's suggestion and pivoted.

---

## 3. 💡 Mental Model: The Detective Solving a Case

```
+-------------------------------------------------------------------------------+
| Component           | Detective Analogy                                       |
+---------------------+---------------------------------------------------------+
| 💭 Thought          | Detective's internal notebook: "The suspect was seen     |
|                     | near the bank at 2:00 PM; I should pull traffic cameras"|
| ⚡ Action           | Fieldwork: Requesting traffic footage, interviewing clerk|
| 👁️ Observation     | Evidence: Camera footage confirms suspect at 2:15 PM     |
| 🏁 Conclusion       | Case Closed: Writing the final arrest warrant           |
+---------------------+---------------------------------------------------------+
```

A ReAct agent records its reasoning trail before taking each action. When an agent produces an unexpected answer, **you can read the trace** to locate the exact thought or observation that sent it off course.

---

## 4. ReAct vs. Chain-of-Thought vs. Act-Only

```
+-------------------------------------------------------------------------------+
| Pattern             | Internal Reasoning? | External Tools? | Failure Mode    |
+---------------------+---------------------+-----------------+-----------------+
| Chain-of-Thought    | ✅ Yes ("Step by step")| ❌ No (Weights) | Hallucination on|
| (CoT)               |                     |                 | live/fresh facts|
+---------------------+---------------------+-----------------+-----------------+
| Act-Only            | ❌ No (Tool calls   | ✅ Yes (APIs)   | Loops blindly on|
| (Function Calling)  | without thoughts)   |                 | errors; no pivot|
+---------------------+---------------------+-----------------+-----------------+
| ReAct               | ✅ Interleaved      | ✅ Dynamic      | High accuracy;  |
| (Reason + Act)      | Thought + Action    | Tool Calling    | fully debuggable|
+---------------------+---------------------+-----------------+-----------------+
```

---

## 5. Stopping Conditions: Defense in Depth

Without explicit stopping conditions, ReAct agents can loop forever in production, exhausting token budgets and server resources.

```
+-------------------------------------------------------------------------------+
| Stop Strategy        | Implementation              | Purpose                  |
+----------------------+-----------------------------+--------------------------+
| 1. Max Iterations    | `for i in range(10):`       | Hard loop ceiling        |
| 2. Finish Tool       | `if item.name == "finish":` | Explicit goal completion |
| 3. No Tool Calls     | `if not msg.tool_calls:`    | Model answers in text    |
| 4. Cost/Token Budget | `if total_cost > $0.50:`    | Financial circuit-breaker|
| 5. Wall-Clock Timeout| `if time() - start > 60s:`  | Prevents thread freezing |
+----------------------+-----------------------------+--------------------------+
```

### Production Stopping Guard Pattern
```python
start_time = time.time()
for iteration in range(1, MAX_ITERATIONS + 1):
    if total_tokens > TOKEN_BUDGET:
        return "🛑 Token budget exceeded."
    if time.time() - start_time > TIMEOUT_SECONDS:
        return "🛑 Wall-clock timeout exceeded."
    
    # ... execute ReAct step ...
```

---

## 6. Self-Correction & Error Recovery Patterns

Self-correction is the primary engineering justification for the ReAct pattern. It requires three conditions:
1. **Helpful Tool Errors**: Tools must return descriptive error dictionaries containing `error`, `suggestion`, and `available_options`, not just a raw `500` or generic string.
2. **System Prompt Alignment**: System prompt must instruct the model to analyze tool errors and adjust parameters.
3. **Iteration Budget Headroom**: The task must have enough remaining turns (e.g. at least 2–3 turns left) to retry.

```
Error Feedback Loop:
[Failing Tool Call] ──> [Descriptive Error + Suggestion] ──> [LLM Interprets Error]
                                                                     │
                                                                     ▼
[Successful Output] <── [Tool Executes with Fix] <── [Emits Corrected Parameters]
```

> **Recovery Metric**: Adding structured error feedback to ReAct increases recovery rates from **~20% to over 85%**, costing only 1 additional LLM turn (~$0.003).

---

## 7. When ReAct Fails (And How to Fix It)

```
+------------------------------------------------------------------------------------+
| Failure Mode        | Symptom                          | Production Fix            |
+---------------------+----------------------------------+---------------------------+
| 🔄 Infinite Loop    | Calls same tool repeatedly with  | Track call history; inject|
|                     | identical parameters             | repetition alert on repeat|
+---------------------+----------------------------------+---------------------------+
| 👻 Ghost /          | Calls functions that do not      | Validate tool names before|
|    Hallucinated Tool| exist on the menu                | execution; return choices |
+---------------------+----------------------------------+---------------------------+
| 🤔 Over-Thinking    | Emits long multi-paragraph       | Shorten system prompt;    |
|                     | thoughts without calling tools   | enforce concise thoughts  |
+---------------------+----------------------------------+---------------------------+
| 🏃 Under-Thinking   | Calls tools prematurely without  | Explicit prompt directive:|
|                     | formulating a plan               | "Think before acting"     |
+---------------------+----------------------------------+---------------------------+
| 🛑 Early Stopping   | Concludes task before verifying  | Require confirmation step |
|                     | all sub-goals                    | in prompt instructions    |
+---------------------+----------------------------------+---------------------------+
```

### Repetition Detection Algorithm
```python
action_history = []

def is_repetitive(tool_name: str, args: dict) -> bool:
    signature = f"{tool_name}:{sorted(args.items())}"
    if action_history.count(signature) >= 2:
        return True
    action_history.append(signature)
    return False
```
If `is_repetitive()` returns true, inject a synthetic observation:
> *"System Warning: You have invoked this tool with the identical arguments twice. Change your strategy or summarize your current progress."*

---

## 8. When to Use ReAct vs. Other Patterns

```
                          [Task Complexity Assessment]
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
        [1 to 2 Steps]                                [3 to 10 Steps]
        - Simple query? ──> Single Prompt             - Needs dynamic tool decisions?
        - Fixed order?  ──> Pipeline / Chain          - Path unknown in advance?
                                                              │
                                                              ▼
                                                     [Use ReAct Pattern]
                                                              │
                                                              ▼
                                                      [> 10 Steps / Multi-Day]
                                                      - Complex decomposition?
                                                      ──> [Plan-and-Execute /
                                                           Multi-Agent Hierarchies]
```

---

## 9. ⚡ 2-Minute Cheatsheet & Interview Guide

- **What**: ReAct interleaves **Thought $\rightarrow$ Action $\rightarrow$ Observation** in a loop until the agent decides it is done.
- **Why It Matters**: Reasoning before tool calls improves parameter selection; re-reading results allows self-correction; the trace makes errors debuggable.
- **How It Ends**: The agent replies with no tool call (direct text), calls `finish()`, or a stopping guard triggers.
- **Repetition Guard**: Always detect consecutive duplicate tool calls to prevent infinite billing loops.
- **When NOT to Use ReAct**: Simple questions (use prompt), fixed deterministic workflows (use chains), or massive projects (use plan-and-solve).

### Top Interview Questions & Model Answers
1. **What does ReAct stand for and why is it superior to Act-Only architectures?**  
   *ReAct stands for **Reasoning + Acting**. Act-Only models emit tool calls without intermediate reasoning, meaning they cannot plan multi-step actions or analyze error observations to self-correct.*
2. **How do you break an agent out of an infinite loop in ReAct?**  
   *By tracking action signatures (`tool_name + args`). When a duplicate action is detected, intercept the loop and inject a prompt reminding the model that this step has already been attempted.*
3. **What is the primary trade-off of using ReAct over a standard prompt?**  
   *Latency and cost. A single prompt takes ~0.5s and 1 LLM call (~$0.001), while a ReAct loop takes 3 to 10+ sequential LLM turns (~10–30s, ~$0.03–$0.20).*

---

## 10. 💻 Code Implementation: Production ReAct Agent from Scratch

```python
"""
react_agent_from_scratch.py
A complete, robust ReAct agent featuring Thought tracking,
repetition detection, stopping conditions, and error recovery.
"""

from openai import OpenAI
import json
import time

client = OpenAI()

# 1. Define Tool Schemas
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Use when asked about temperature or weather conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call when the user's request has been completely answered.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "Final response text"}
                },
                "required": ["answer"]
            }
        }
    }
]

# 2. Mock Tool Executor with Typo Correction
def execute_tool(name: str, args: dict) -> dict:
    if name == "get_weather":
        city = args.get("city", "").strip()
        if city.lower() in ["pariss", "pariis"]:
            return {
                "status": "error",
                "error": f"City '{city}' not found.",
                "suggestion": "Did you mean 'Paris'?"
            }
        elif city.lower() == "paris":
            return {"status": "success", "temp": 18, "condition": "sunny", "humidity": 65}
        else:
            return {"status": "error", "error": f"City '{city}' not recognized."}

    return {"status": "error", "error": f"Unknown tool: '{name}'"}

# 3. The Core ReAct Controller
def run_react_agent(
    task: str,
    max_iters: int = 10,
    cost_budget: float = 0.50,
    timeout_sec: float = 60.0
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an autonomous ReAct problem solver. For each turn:\n"
                "1. THOUGHT: Explain your reasoning and what you need next.\n"
                "2. ACTION: Call an appropriate tool with arguments.\n"
                "3. When you have sufficient data, call finish(answer=...) to conclude."
            )
        },
        {"role": "user", "content": task}
    ]

    action_history = []
    start_time = time.time()
    total_tokens = 0

    print(f"🚀 Launching ReAct Agent for task: '{task}'\n")

    for turn in range(1, max_iters + 1):
        # Guard 1: Wall-Clock Timeout
        if time.time() - start_time > timeout_sec:
            return "🛑 Loop aborted: Timeout limit reached."

        print(f"--- [Turn {turn}/{max_iters}] ---")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        msg = response.choices[0].message
        messages.append(msg)

        if response.usage:
            total_tokens += response.usage.total_tokens

        # If model outputs direct text without tools, task is complete
        if not msg.tool_calls:
            print("ℹ️ Model concluded with plain text response.")
            return msg.content or ""

        # Process tool calls
        for call in msg.tool_calls:
            fn_name = call.function.name
            fn_args = json.loads(call.function.arguments)

            print(f"⚡ [ACTION]: {fn_name}({fn_args})")

            # Check explicit termination tool
            if fn_name == "finish":
                final_answer = fn_args.get("answer", "")
                print(f"🎯 [FINISH]: Final answer delivered at turn {turn}.")
                return final_answer

            # Guard 2: Repetition Detection
            action_sig = f"{fn_name}:{sorted(fn_args.items())}"
            if action_history.count(action_sig) >= 2:
                print("⚠️ Repetition detected! Forcing strategy adjustment.")
                observation = {
                    "status": "error",
                    "error": "Duplicate action detected. You have called this exact action twice without progress.",
                    "suggestion": "Pivot to a different tool, check parameters, or deliver best-effort response."
                }
            else:
                action_history.append(action_sig)
                observation = execute_tool(fn_name, fn_args)

            print(f"👁️ [OBSERVATION]: {observation}")

            # Append observation into conversation context
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(observation)
            })

    return "⚠️ Loop ended: Max iterations reached without a resolution."

if __name__ == "__main__":
    result = run_react_agent("What is the current weather in Paris?")
    print(f"\nFinal Result:\n{result}")
```
