# Function Calling: How LLMs Call Tools via OpenAI and Anthropic APIs

> **Module**: Agentic AI & Multi-Agent Systems  
> **Topic**: Function Calling  
> **Source**: [AIML Companion — Function Calling](https://aimlcompanion.ai/module/aiAgents/functionCalling)

---

## 🎯 In This Module

- **How function calling works**: The LLM generates structured arguments; your code executes the function.
- **OpenAI Responses API vs. Anthropic `tool_use` format**: Side-by-side comparison and translation.
- **Writing tool schemas that guide LLMs**: Why descriptions are prompts that steer tool selection and parameter constraints.
- **Handling parallel tool calls and multi-turn conversations**: Running concurrent tools and looping until completion.
- **Error handling & input validation patterns**: Returning errors so the LLM self-corrects instead of crashing.

---

## 1. The Big Picture: How Function Calling Works

**Function calling** is how an LLM uses external tools without ever executing code directly. 

```
+-------------------------------------------------------------------------------+
|                           THE FUNCTION-CALLING FLOW                           |
|                                                                               |
|   01. DEFINE TOOLS ─────────> 02. LLM DECIDES ─────────> 03. YOU EXECUTE      |
|   (Name + JSON Schema)        (Picks tool + args)         (run(name, args))   |
|   The menu of actions         Emits JSON, runs nothing    Your code, not LLM  |
|            ▲                                                      │           |
|            │                                                      ▼           |
|            +───────────────── 04. RETURN RESULT <─────────────────+           |
|                                (result -> LLM)                                |
|                                Answer or loop again                           |
+-------------------------------------------------------------------------------+
```

### The Safety Split: Why Function Calling Is Safe
The core safety mechanism is the strict architectural boundary between the LLM and execution:
* **The LLM's Job**: Reads the prompt and tool descriptions, selects a function, and serializes arguments as structured JSON. **The LLM never runs code and never touches external APIs directly.**
* **Your Code's Job**: Validates the arguments, manages secrets/API keys, runs the real function, and hands the resulting data back to the LLM.

```
[User Request] ──> [LLM Brain] ──(Emits JSON args)──> [Your Secure Executor] ──(Calls API)──> [External World]
                          ▲                                       │
                          └───────(Returns Result Data)───────────┘
```

---

## 2. Walkthrough: One Tool Call in 4 Steps

**User asks**: *"What's the weather in Paris right now?"*

The LLM cannot know live real-time conditions from static weights, so you offer it one tool: `get_weather`.

```
Step 1: Define the Tool (The Menu)
----------------------------------
name: get_weather
description: "Get current weather for a city. Use when user asks about weather or temperature."
parameters:
  - city: string (required)
  - units: enum ["celsius", "fahrenheit"] (default: "celsius")

Step 2: The LLM Decides
-----------------------
The LLM matches "weather in Paris" against the tool description and emits structured JSON:
get_weather(city='Paris', units='celsius')
(A request to call, not a result; no code has executed yet)

Step 3: You Execute
-------------------
Your application code receives the tool call, validates the arguments, and calls the live weather API:
get_weather('Paris') -> { "temp": "18°C", "condition": "sunny" }
(Real data fetched securely by your code)

Step 4: Return Result, LLM Answers
----------------------------------
You pass the observation dictionary back into the LLM context:
LLM(question + tool result) -> "It's 18°C and sunny in Paris right now."
(Answer is grounded in real tool observation, not hallucinated)
```

---

## 3. 💡 Mental Model: Ordering at a Restaurant

```
+-------------------------------------------------------------------------------+
| Component           | Restaurant Analogy                                      |
+---------------------+---------------------------------------------------------+
| 📜 Menu (Schema)    | Lists available dishes with ingredients and options     |
| 📝 Order (Tool Call)| Customer picks a dish and specifies options (rare, no onions)
| 👨‍🍳 Kitchen (Exec)  | Chef prepares exactly what was ordered safely in kitchen|
| 🍽️ Delivery (Result)| Waiter brings the prepared dish back to the customer     |
+---------------------+---------------------------------------------------------+
```

The LLM reads the menu (tool descriptions), places an order (generates JSON arguments), and your application runs the kitchen (validates and executes the function).

---

## 4. The 3 Code Checks: Validation Before Execution

Before your code executes any tool emitted by the LLM, it must pass three defensive validation gates:

```
[LLM emits: get_weather(city='Paris', units='celsius')]
                          │
                          ▼
        [1. Is the function name on the menu?]
            ├── NO  ──> Return error: "Unknown tool: {name}"
            └── YES ──┐
                      ▼
        [2. Do the arguments match JSON Schema?]
            ├── NO  ──> Return error: "Invalid argument types/missing required fields"
            └── YES ──┐
                      ▼
        [3. Your code executes the function safely]
            ├── Exception (timeout/500) ──> Catch and return error dictionary to LLM
            └── Success ──────────────────> Return result data to LLM
```

> **Security Rule**: Treat all LLM-generated arguments like untrusted user input. Never pass raw arguments into `eval()`, SQL queries, or shell subprocesses without parameterization and validation.

---

## 5. Tool Schemas: Why Descriptions Matter More Than Names

The tool description is a **prompt**. The model uses semantic similarity between the user's intent and your description to decide:
1. When to invoke the tool.
2. What arguments to pass.
3. What structure to anticipate in return.

```
❌ Poor Schema:
name: "get_weather"
description: "Get weather"

✓ Production-Grade Schema:
name: "get_weather"
description: "Get current weather and temperature for a specified city. Use whenever the user asks about weather, forecasts, or temperature conditions. Returns temperature in Celsius and conditions (sunny, rainy, cloudy)."
```

### High-Impact Schema Design Rules
- **Include Trigger Keywords**: State explicitly *"Use when the user asks about X, Y, or Z."*
- **State the Return Shape**: Tell the model what data it will receive back.
- **Use `enum` Constraints**: Constrain strings to valid literal values (e.g. `["celsius", "fahrenheit"]`).
- **Provide Sensible Defaults**: Prevents tool failures when optional parameters are omitted.
- **Specify `required` Fields**: Explicitly list all required properties.

---

## 6. Provider Formats Side-by-Side: OpenAI vs. Anthropic

While both providers implement the exact same 4-step concept, their schema keys and message roles differ slightly:

```
+-------------------------------------------------------------------------------+
| Feature              | OpenAI (Responses / Chat)      | Anthropic (Messages)  |
+----------------------+--------------------------------+-----------------------+
| Schema Definition    | "parameters": { ... }          | "input_schema": { ... }
| Tool Declaration     | "type": "function", "function" | direct list of tools  |
| Model Output Type    | item.type == "function_call"   | block.type == "tool_use"
| Argument Storage     | JSON string (`item.arguments`) | Python dict (`block.input`)
| Tool Call Identifier | item.call_id                   | block.id              |
| Returning Result     | role: "tool" (or call output)  | type: "tool_result"   |
| Matching Identifier  | tool_call_id / call_id         | tool_use_id           |
+----------------------+--------------------------------+-----------------------+
```

### 1. OpenAI Responses API Format
```python
tools = [{
    "type": "function",
    "name": "search_products",
    "description": "Search product catalog. Use when user asks about products or pricing.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keyword"},
            "category": {"type": "string", "enum": ["electronics", "clothing", "all"]},
            "max_results": {"type": "integer", "default": 5}
        },
        "required": ["query"]
    }
}]
```

### 2. Anthropic Messages API Format
```python
tools = [{
    "name": "search_products",
    "description": "Search product catalog. Use when user asks about products or pricing.",
    "input_schema": {  # Note: input_schema instead of parameters
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keyword"},
            "category": {"type": "string", "enum": ["electronics", "clothing", "all"]},
            "max_results": {"type": "integer", "default": 5}
        },
        "required": ["query"]
    }
}]
```

---

## 7. Parallel Tool Calls & Multi-Turn Conversations

Modern LLMs can emit multiple tool calls in a single turn.

### Scenario: *"What's the weather in Paris, and find me a pizza place in Rome?"*
The LLM realizes these are **two independent queries** and emits two tool calls simultaneously:
1. `get_weather(city="Paris")`
2. `search_restaurants(city="Rome", cuisine="pizza")`

```
Sequential Execution (Slow):
[Paris: 180ms] ───> [Rome: 220ms] ───> Total: 400ms

Parallel Execution (Fast - Recommended):
├── [Paris: 180ms] ──┐
└── [Rome:  220ms] ──┴──> Total: 220ms (60% faster, matches slowest tool)
```

### The Parallel Execution Rule
When an LLM emits multiple tool calls in one response, you must execute all of them and return **all tool results before invoking the LLM for the next turn**. Omitting any tool response will cause API rejections.

---

## 8. Error Handling: Recover Instead of Crashing

Tools inevitably fail due to network timeouts, rate limits, or invalid arguments. **Never crash your application—return the error dictionary to the LLM context.**

```python
def execute_tool_safely(name: str, args: dict, handlers: dict) -> dict:
    if name not in handlers:
        return {"status": "error", "message": f"Tool '{name}' is not supported."}
    
    try:
        data = handlers[name](**args)
        return {"status": "success", "data": data}
    except TypeError as e:
        return {"status": "error", "message": f"Argument error: {str(e)}. Please correct arguments."}
    except Exception as e:
        return {"status": "error", "message": f"Execution failed: {str(e)}. Try an alternative tool or query."}
```

> **Recovery Rate**: LLMs recover from tool execution errors in **~60% of cases** when provided with descriptive error feedback, either by re-formulating arguments or choosing a fallback tool.

---

## 9. Controlling Tool Selection with `tool_choice`

| Setting | Behavior | Best Used When |
| :--- | :--- | :--- |
| `"auto"` | LLM decides whether to respond in text or call tools (default). | General conversational agent turns. |
| `"required"` | LLM must invoke at least one tool before responding. | First turn of a dedicated action pipeline. |
| `{"type": "function", "name": "..."}` | Forces the model to call one specific tool. | Extracting structured JSON schema. |
| `"none"` | Disables all tools for the current turn. | Final response generation; prevents loop overrun. |

---

## 10. ⚡ 2-Minute Cheatsheet & Interview Guide

- **The Split**: The LLM writes the call (`{name, args}`). Your code runs it. The LLM never touches code or API keys.
- **The 4 Steps**: Define $\rightarrow$ Decide $\rightarrow$ Execute $\rightarrow$ Return.
- **Most Critical Field**: `description`. It is a prompt instructing the LLM when to call the tool and what to expect in return.
- **Error Handling**: Always return exceptions and error codes as structured tool outputs so the agent can self-correct.
- **Parallel Rule**: Return all matching tool outputs before triggering the next LLM step.

### Top Interview Questions & Model Answers
1. **Who actually executes the function in function calling?**  
   *Your code executes the function. The LLM is purely a reasoning and serialization engine that emits structured JSON.*
2. **What is the difference between OpenAI and Anthropic tool calling?**  
   *OpenAI uses `parameters` and emits `tool_calls`; Anthropic uses `input_schema` and emits `tool_use` blocks. The underlying JSON schema concept is identical.*
3. **What should your application do if an API call fails with a timeout?**  
   *Catch the exception and return a structured JSON error (`{"error": "timeout", "suggestion": "retry or use fallback"}`) back to the model rather than crashing.*

---

## 11. 💻 Code Implementations

### Example 1: Complete OpenAI Tool Loop with Parallel Calls

```python
"""
openai_tool_loop.py
Demonstration of OpenAI Responses API with tool calling and parallel handling.
"""

from openai import OpenAI
import json

client = OpenAI()

# 1. Define tools
tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a city. Use when user asks about weather or temperature.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name (e.g. 'Paris')"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
            },
            "required": ["city"]
        }
    },
    {
        "type": "function",
        "name": "search_restaurants",
        "description": "Search restaurants in a city. Use when user asks about dining or food.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "cuisine": {"type": "string", "enum": ["italian", "japanese", "mexican", "any"]}
            },
            "required": ["city"]
        }
    }
]

# 2. Tool handler registry
def execute_tool(name: str, args: dict) -> dict:
    handlers = {
        "get_weather": lambda city, units="celsius": {"temp": "18°C", "condition": "sunny", "city": city},
        "search_restaurants": lambda city, cuisine="any": {"results": [f"Top {cuisine} restaurant in {city}"], "count": 3}
    }
    if name not in handlers:
        return {"error": f"Unknown tool: {name}"}
    try:
        return handlers[name](**args)
    except Exception as e:
        return {"error": str(e)}

# 3. Execution loop
def run():
    input_messages = [
        {"role": "user", "content": "What's the weather in Paris and find me Italian restaurants there?"}
    ]

    response = client.responses.create(
        model="gpt-4o-mini",
        input=input_messages,
        tools=tools
    )

    input_messages += response.output

    # Process all tool calls emitted
    for item in response.output:
        if item.type == "function_call":
            args = json.loads(item.arguments)
            result = execute_tool(item.name, args)
            input_messages.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(result)
            })

    # Final synthesis call
    final = client.responses.create(
        model="gpt-4o-mini",
        input=input_messages,
        tools=tools
    )
    print("Final Response:\n", final.output_text)

if __name__ == "__main__":
    run()
```

---

### Example 2: Anthropic Claude Tool Use Loop

```python
"""
anthropic_tool_loop.py
Demonstration of Anthropic Messages API with tool_use and tool_result.
"""

import anthropic
import json

client = anthropic.Anthropic()

# 1. Define Anthropic tool schema (input_schema)
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["city"]
    }
}]

def get_weather(city: str, units="celsius") -> dict:
    return {"temp": "22°C", "condition": "partly cloudy", "city": city}

# 2. Execution flow
def run_claude():
    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

    # Step 1: Claude decides to invoke tool
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    # Step 2: Handle tool_use blocks
    for block in response.content:
        if block.type == "tool_use":
            tool_result = get_weather(block.input["city"])

            # Append assistant message and tool_result response
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(tool_result)
                }]
            })

    # Step 3: Claude synthesizes the tool result
    final = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    for b in final.content:
        if b.type == "text":
            print("Claude Response:\n", b.text)

if __name__ == "__main__":
    run_claude()
```
