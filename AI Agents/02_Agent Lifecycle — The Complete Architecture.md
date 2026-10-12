# Agent Lifecycle: The Complete Architecture
## From Goal Definition to Self-Correction: Every Phase of How Agents Work

> **Module**: Agentic AI & Multi-Agent Systems  
> **Topic**: Agent Lifecycle: The Complete Architecture  
> **Source**: [AIML Companion — Agent Lifecycle](https://aimlcompanion.ai/module/aiAgents/agentLifecycle)

---

## 🎯 In This Module

- **The 6-phase agent lifecycle**: `Goal → Plan → Perceive → Reason → Act → Evaluate`, and how each phase works internally.
- **How agents perceive the world**: Multimodal inputs, context assembly, and RAG retrieval within token limits.
- **Four reasoning strategies**: Chain-of-Thought (CoT), ReAct, Reflection, and Tree-of-Thought (ToT), and when to use each.
- **Memory architecture**: Short-term context window (STM) vs. long-term persistent storage (LTM) vs. RAG retrieval bridge.
- **Evaluation and feedback loops**: Self-critique, human-in-the-loop (HITL), environmental signals, and why agents self-correct.
- **Production Guardrails**: Input, output, action, and loop guards that keep autonomous agents safe.

---

## 1. The Big Picture: How It Works, Start to Finish

An agent sets a measurable **Goal** and **Plans** it into ordered sub-tasks **once**, then for each sub-task repeats **Perceive, Reason, Act, and Evaluate** until that sub-task passes.

```
+-----------------------------------------------------------------------------+
|                          THE 6-PHASE AGENT LIFECYCLE                         |
|                                                                             |
|      01. GOAL (Once)                  04. PLAN (Once)                       |
|   What success looks like       Split into ordered sub-tasks                |
|          │                                   │                              |
|          +─────────────────┬─────────────────+                              |
|                            │                                                |
|                            ▼                                                |
|            +------------------------------------+                           |
|            |      THE PER-SUBTASK EXECUTION LOOP|                           |
|            |                                    |                           |
|            |        02. PERCEIVE                |                           |
|            |      (gather context)              |                           |
|            |    inputs + memory + RAG           |                           |
|            |             │                      |                           |
|            |             ▼                      |                           |
|            |         03. REASON                 |                           |
|            |    (LLM decides next move)         |                           |
|            |             │                      |                           |
|            |             ▼                      |                           |
|            |          05. ACT                   |                           |
|            |       (run a tool)                 |                           |
|            |     search, code, API              |                           |
|            |             │                      |                           |
|            |             ▼                      |                           |
|            |        06. EVALUATE                |                           |
|            |   (check & self-correct)           |                           |
|            |   passed? else re-plan             |                           |
|            |             │                      |                           |
|            |             ▼                      |                           |
|            |     ↻ not done? loop again         |                           |
|            +------------------------------------+                           |
+-----------------------------------------------------------------------------+
```

### The Universal Execution Formula
$$\text{Goal} + \text{Plan} \longrightarrow \left( \text{Perceive} \longrightarrow \text{Reason} \longrightarrow \text{Act} \longrightarrow \text{Evaluate} \right) \times N$$

> **The Evaluate Edge**: **Evaluate** is what separates an agent from a script. A script executes blindly; a failed check in an agent triggers a **re-plan** instead of shipping a wrong result.

---

## 2. Walkthrough: One Support Ticket Through the Whole Lifecycle

**Customer Email**: *"My order #4471 never arrived and I want a refund."*

### Step 1: Goal + Plan (Once, Up Front)
- **Goal**: Resolve ticket #4471: refund or reship, then close it (measurable finish line).
- **Plan**: Decompose into three ordered sub-tasks:
  1. Look up the order status
  2. Apply the refund policy
  3. Reply and close
- **Tools Available (Act Phase)**: `lookup_order`, `issue_refund`, `send_email` (plus a max safety limit of 15 loop turns).

---

### Step 2: Sub-task 1 Loop — Find the Order
- **Perceive**: Read the ticket and memory.
- **Reason**: The agent needs the order's status, so it picks the database lookup tool.
- **Act**: Call tool:
  ```text
  lookup_order(4471) = 'lost in transit'
  ```
- **Evaluate**: Valid record returned; sub-task 1 passed.

---

### Step 3: Sub-task 2 Loop — Apply the Policy
- **Perceive**: Status is `'lost in transit'`.
- **Reason**: Policy states a lost order qualifies for a full refund without requiring human approval under the \$100 limit.
- **Act**: Call tool:
  ```text
  issue_refund(4471) = HTTP 200 OK
  ```
- **Evaluate**: Environment signal confirms HTTP 200 success; sub-task 2 passed.

---

### Step 4: Sub-task 3 Loop — Reply and Close
- **Perceive**: Status is resolved; refund issued.
- **Reason**: Draft a clear, polite customer email confirming the refund.
- **Act**: Call tool:
  ```text
  send_email(customer, refund_confirmation) = sent
  ```
- **Evaluate**: Email delivery accepted; all sub-tasks passed.
- **Result**: Ticket closed, refund issued, no human intervention needed.

---

### What if `issue_refund` had returned HTTP 500 instead of 200?
**Evaluate** catches the failure. An HTTP 500 status is **environmental feedback**—the most objective kind of signal. Instead of telling the customer their refund was processed when it failed, the agent:
1. Stores the error in memory: `{"refund_error": "HTTP 500"}`
2. **Re-plans**: Retries with exponential backoff, or escalates to a human agent.
3. Loops again to handle the failure cleanly.

---

## 3. 💡 Mental Model: The Agent as an ER Doctor

```
+---------------------------------------------------------------------------------+
| Phase        | ER Doctor Analogy                                                |
+--------------+------------------------------------------------------------------+
| 🎯 Goal      | Save the patient: stabilize, diagnose, treat                     |
| 👁️ Perceive   | Check vitals, inspect chart, ask patient about symptoms          |
| 🧠 Reason    | Symptoms + medical training = potential diagnosis                |
| 🗺️ Plan      | Treatment roadmap: labs first, then medication, then monitor      |
| 💉 Act       | Order blood tests, administer medication, call specialist        |
| ✅ Evaluate  | Did vitals improve? Adjust treatment dosage or continue?        |
+---------------------------------------------------------------------------------+
```

An ER doctor continually assesses, administers treatments, checks vitals, and adjusts until the patient is stable. An agent executes the exact same adaptive feedback loop until the goal is satisfied.

---

## 4. Phase Breakdown & Failure Diagnostics

Every agent, from ChatGPT to Claude Code to autonomous coding assistants, follows these 6 phases. When an agent fails, the lifecycle tells you **where** it failed:

```
+------------------------------------------------------------------------------------+
| Phase       | Common Failure         | Symptom              | Fix                  |
+-------------+------------------------+----------------------+----------------------+
| 🎯 Goal     | Ambiguous goal         | Agent does wrong task| Clearer prompt & KPIs|
| 👁️ Perceive | Missing context        | Agent lacks key info | Add RAG / more tools |
| 🧠 Reason   | Hallucination          | Confident but wrong  | Stronger model / CoT |
| 🗺️ Plan     | Over-decomposition     | Too many steps, slow | Fewer, bigger steps  |
| ⚡ Act      | Wrong tool selection   | Wrong tool called    | Better tool docstring|
| ✅ Evaluate | No stopping condition  | Infinite loop        | max_iter + finish()  |
+------------------------------------------------------------------------------------+
```

> **Production Debug Order**: Always check **Evaluate first** (Is the agent stuck?), then **Perceive** (Does it have the right info?), then **Reason** (Is it thinking correctly?).

---

## 5. Phase 2 — Perception: How Agents Gather Context

The Perception layer assembles everything the agent "knows" into a single prompt that fits the LLM's context window.

```
                         THE PERCEPTION FUNNEL
                         
     [📝 User Input]     [🌐 API Data]     [📁 Documents]     [🧠 Memory]
          ~200 tokens       ~1,500 tokens       ~5,000 tokens    ~2,000 tokens
                │                 │                  │                │
                └─────────────────┼──────────────────┴────────────────┘
                                  ▼
                    [Prompt Context Assembly]
                          (~8,700 tokens)
                                  │
                                  ▼
                          [LLM Reasoning]
```

### Context Budgeting in Frontier Models (e.g. 200K Window)
Even with large context windows, you cannot stuff everything in without degrading performance:
- Total context available: **200,000 tokens**
- System prompt + tool schemas: **~5,000 tokens**
- Conversation history: **~10,000 tokens**
- Retrieved knowledge (RAG): **~15,000 tokens**
- Current tool outputs: **~5,000 tokens**
- Available headroom for reasoning: **~165,000 tokens**

### RAG: The Perception Upgrade
Instead of memorizing everything, the agent retrieves relevant context on demand:
1. Embed the current query as a dense vector.
2. Search the vector database for similar content.
3. Inject the top-$K$ chunks into the prompt context.
4. The LLM reasons over fresh, grounded information.

> **Risk — The "Lost in the Middle" Effect**: LLMs attend heavily to the start and end of long prompts, but miss facts buried in the middle. Place critical instructions and retrieved facts at the very top and bottom of the prompt.

---

## 6. Phase 3 — Reasoning: 4 Strategies Compared

```
+--------------------------------------------------------------------------------------+
| Strategy           | Pattern                   | Calls | Best For                    |
+--------------------+---------------------------+-------+-----------------------------+
| 🔗 Chain-of-Thought| "Think step by step"      | 1     | Math, logic, no tools       |
| ⚡ ReAct           | "Thought -> Act -> Observe| N     | Any task needing tools      |
| 🔄 Reflection      | Draft -> Critique -> Fix  | ~3x   | Code generation, writing    |
| 🌳 Tree-of-Thought | Branch paths -> Vote      | 3–5x  | Creative, open-ended tasks  |
+--------------------+---------------------------+-------+-----------------------------+
```

### Scenario: *"Find the cheapest flight from Delhi to Mumbai for March 15"*
- **CoT**: Thinks through constraints and answers from weights (1 call, fast, no live prices).
- **ReAct**: Search live flights $\rightarrow$ observe 47 results $\rightarrow$ filter lowest fare $\rightarrow$ book (4 calls, accurate).
- **Reflection**: Searches $\rightarrow$ drafts itinerary $\rightarrow$ self-critiques baggage fees $\rightarrow$ improves selection (6 calls, high quality).
- **Tree-of-Thought**: Explores 3 airline APIs in parallel $\rightarrow$ branches train alternatives $\rightarrow$ votes on optimal trade-off (9 calls, thorough).

### Strategy Decision Framework

| Question | If Yes | If No |
| :--- | :--- | :--- |
| **Does it need tools?** | ReAct or Reflection | Chain-of-Thought (CoT) |
| **Is quality critical?** | Reflection | ReAct |
| **Are there many valid approaches?** | Tree-of-Thought (ToT) | ReAct |
| **Is cost a major concern?** | CoT or ReAct | Reflection or ToT |

### Production Hybrid Patterns
- **ReAct + Reflection**: The agent uses ReAct for tool calling, then Reflection to review and polish the final synthesized response.
- **CoT inside ReAct**: Each intermediate "Thought" step inside ReAct uses chain-of-thought prompting internally.
- **ToT for Planning, ReAct for Execution**: Explore multiple plan options with Tree-of-Thought, then execute the winning plan using ReAct.

---

## 7. Phase 6 — Evaluation & Guardrails: Self-Correction

The **Evaluation phase** checks the agent's work and triggers self-correction before shipping bad outputs.

### The 3 Feedback Channels

```
1. 🧠 Self-Evaluation (Internal)
   - LLM reviews its own output ("Confidence: 72%. Missing Q3 data. Recommend re-query.")
   - Speed: Fast | Cost: Cheap (~1 LLM call) | Reliability: Medium

2. 👤 Human-in-the-Loop (HITL - External)
   - Critical decisions require human sign-off ("Approve refund of $450? [Yes/No]")
   - Speed: Slow (blocks execution) | Cost: Free | Reliability: Very High

3. 🌍 Environmental Feedback (Automatic)
   - Objective ground truth from systems: HTTP 200/404/500, test passes, exit codes.
   - Speed: Fast | Cost: Free (automated) | Reliability: Very High
```

---

### The 4 Layers of Guardrails

Guardrails wrap the entire lifecycle to guarantee safety:

```
[User Input] 
     │
     ▼
[🚫 1. INPUT GUARDS]      -> Prompt injection check, length limits, topic scope
     │
     ▼
[🧠 LLM REASONING]
     │
     ▼
[⛔ 3. ACTION GUARDS]     -> Human approval for payments/deletions, rate limits
     │
     ▼
[⚡ TOOL EXECUTION]
     │
     ▼
[🔒 2. OUTPUT GUARDS]     -> PII filtering (SSN, credit cards), JSON schema check
     │
     ▼
[🔄 4. LOOP GUARDS]       -> Max 15 iterations, $0.50 budget cap, 60s timeout
```

### Production Guardrail Checklist
- **Before the LLM (Input)**:
  - [ ] Prompt injection detection (reject "ignore previous instructions").
  - [ ] Input length limits (prevent context stuffing).
  - [ ] Topic scope filtering (restrict agent to designated domain).
- **After the LLM (Output)**:
  - [ ] PII and secret detection (strip API keys, credit cards).
  - [ ] JSON schema validation.
  - [ ] Factual grounding (verify claims match retrieved documents).
- **Before Tool Execution (Action)**:
  - [ ] Human approval gates for destructive calls (`delete_database()`, `send_money()`).
  - [ ] Rate limits on API calls.
  - [ ] Permission scoping (least-privilege access tokens).
- **On the Loop (Runtime)**:
  - [ ] `max_iterations` limit (10–25 turns).
  - [ ] Token budget ceiling (\$0.50–\$5.00).
  - [ ] Wall-clock timeout (30–120 seconds).

---

## 8. Memory Architecture: The 3 Layers

Without memory, every conversation starts from zero. Production agents combine three distinct tiers:

```
+----------------------------------------------------------------------------------+
| Layer                  | Analogy | Capacity     | Persistence     | Storage      |
+------------------------+---------+--------------+-----------------+--------------+
| ⚡ Short-Term (STM)    | RAM     | 4K–128K tok  | Current session | Prompt window|
| 💾 Long-Term (LTM)     | Hard Disk| Unlimited   | Cross-session   | Database     |
| 🔗 RAG (Bridge)        | Library | Top-K Chunks | On-demand       | Vector DB    |
+------------------------+---------+--------------+-----------------+--------------+
```

### Sub-types of Long-Term Memory (LTM)
1. **Episodic Memory**: Past experiences (*"Last time user asked about order #123, they preferred credit over refund"*).
2. **Semantic Memory**: Factual knowledge (*"Customer is a Tier-1 VIP member with 5-day return window"*).
3. **Procedural Memory**: How-to operational rules (*"To process a return: verify receipt, create RMA, notify warehouse"*).

### Production Memory Patterns

| Pattern | How It Works | Typical Use Case |
| :--- | :--- | :--- |
| **Sliding Window** | Retains only the last $N$ messages in STM | Standard customer support chat |
| **Summary Buffer** | Summarizes older turns and appends recent turns | Extended conversations (ChatGPT memory) |
| **Entity Memory** | Extracts and stores key facts keyed by user/org | CRM and account management agents |
| **Episodic Memory** | Stores complete interaction histories; searches by vector similarity | Deep research & coding assistants |
| **Hybrid** | Combines sliding window + summary + vector RAG | Enterprise production agents |

> **Cost Lever**: Every token in context costs money on every call. A 50,000-token history costs \$0.20 per call at \$4/M tokens. Aggressive context pruning and summarization is your primary cost-reduction tool.

---

## 9. ⚡ Cheat Sheet: Scan in Under 2 Min

- **Lifecycle Shape**: `Goal + Plan (once) -> (Perceive -> Reason -> Act -> Evaluate) x N`
- **Why Evaluate Matters**: Catches errors early and triggers re-planning. A script fails; an agent re-plans.
- **The Debug Map**:
  - Wrong task $\rightarrow$ **Goal** (clearer prompt & success criteria)
  - Stale input / missing data $\rightarrow$ **Perceive** (add RAG / tool)
  - Hallucination $\rightarrow$ **Reason** (better model / CoT)
  - Too many steps $\rightarrow$ **Plan** (fewer, coarser steps)
  - Wrong tool called $\rightarrow$ **Act** (improve tool docstring)
  - Infinite loop $\rightarrow$ **Evaluate** (add `max_iterations` + finish tool)
- **4 Reasoning Models**:
  - CoT: 1 call (step-by-step logic)
  - ReAct: $N$ calls (default tool-use pattern)
  - Reflection: ~3 calls (draft, critique, improve)
  - Tree-of-Thought: 3–5 calls (branching search & voting)
- **3 Feedback Channels**: Self (soft, internal), Human (HITL, slow but safe), Environment (HTTP codes, tests, ground truth).

---

## 10. 💻 Code Examples

### 1. Complete Agent Lifecycle Implementation

```python
"""
complete_agent_lifecycle.py
Full lifecycle agent with Goal, Perception, Reasoning, Action, Evaluation, and Memory.
"""

from openai import OpenAI
import json

client = OpenAI()

# === 1. MEMORY LAYER ===
class AgentMemory:
    """Three-layer memory: STM (conversation) + LTM (facts) + Working Memory"""
    def __init__(self):
        self.conversation = []    # STM: Current session context
        self.facts = {}           # LTM: Persisted facts across steps
        self.plan = []            # Working memory: Current plan steps

    def add_message(self, role: str, content: str):
        self.conversation.append({"role": role, "content": content})

    def store_fact(self, key: str, value: str):
        self.facts[key] = value

    def get_context(self) -> str:
        """Assembles perception context: system + known facts"""
        if not self.facts:
            return ""
        summary = "\n".join(f"- {k}: {v}" for k, v in self.facts.items())
        return f"\nKnown facts from memory:\n{summary}\n"

# === 2. TOOLS (ACT PHASE) ===
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_orders",
            "description": "Search customer orders by customer ID or order number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_return",
            "description": "Initiates a refund or return for an order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["order_id", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call when the customer request is completely resolved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {"type": "string", "description": "Final customer reply"}
                },
                "required": ["response"]
            }
        }
    }
]

def execute_tool(name: str, args: dict) -> dict:
    """Act phase: Execute external action and return observation."""
    if name == "search_orders":
        return {
            "order_id": args.get("order_id"),
            "item": "Laptop",
            "status": "lost in transit",
            "customer_type": "VIP"
        }
    elif name == "create_return":
        return {
            "return_id": "RET-456",
            "status": "initiated",
            "http_status": 200
        }
    return {"error": f"Unknown tool: {name}"}

# === 3. EVALUATION (EVALUATE PHASE) ===
def evaluate_result(result: dict, tool_name: str) -> tuple[bool, str]:
    """Evaluate phase: Check environmental signals and outputs."""
    if "error" in result:
        return False, f"Environmental error in {tool_name}: {result['error']}"
    if result.get("http_status") and result["http_status"] >= 400:
        return False, f"Server returned error {result['http_status']}"
    return True, f"{tool_name} completed successfully."

# === 4. MAIN LIFECYCLE CONTROLLER ===
def run_lifecycle_agent(goal: str, max_iterations: int = 15, token_budget: int = 50000) -> str:
    memory = AgentMemory()
    tokens_used = 0

    # GOAL Phase
    system_prompt = (
        f"You are an autonomous customer support agent.\n"
        f"GOAL: {goal}\n"
        f"Use tools to investigate and resolve issues. Call finish() when complete."
    )
    memory.add_message("system", system_prompt)
    memory.add_message("user", goal)

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- [Cycle {iteration}] ---")
        
        # PERCEIVE Phase: Assemble conversation and facts
        messages = memory.conversation.copy()

        # REASON Phase: LLM decides next action
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        
        if response.usage:
            tokens_used += response.usage.total_tokens

        # LOOP GUARD: Token budget check
        if tokens_used > token_budget:
            return "🛑 Token budget exceeded: Aborting task for cost safety."

        msg = response.choices[0].message
        memory.conversation.append(msg)

        if not msg.tool_calls:
            print("ℹ️ Model answered with direct text.")
            return msg.content or ""

        for call in msg.tool_calls:
            fn_name = call.function.name
            fn_args = json.loads(call.function.arguments)

            # Check explicit finish tool
            if fn_name == "finish":
                print("🎯 [FINISH] Goal confirmed complete.")
                return fn_args.get("response", "")

            # ACT Phase: Execute tool
            print(f"👉 [ACT] Calling: {fn_name}({fn_args})")
            tool_result = execute_tool(fn_name, fn_args)

            # EVALUATE Phase: Check environmental outcome
            success, eval_msg = evaluate_result(tool_result, fn_name)
            print(f"✅ [EVALUATE] Success: {success} -> {eval_msg}")

            # Update STM context
            memory.conversation.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(tool_result)
            })

            # Update LTM memory
            status_key = f"step_{iteration}_{fn_name}"
            memory.store_fact(status_key, eval_msg)

    return "⚠️ Max iterations reached without completing goal."

if __name__ == "__main__":
    ticket = "My order #ORD-4471 never arrived and I want a refund."
    final_output = run_lifecycle_agent(ticket)
    print(f"\nResult:\n{final_output}")
```

---

### 2. Reasoning Strategies in Code: CoT vs. ReAct vs. Reflection

```python
"""
reasoning_strategies.py
Demonstration of Chain-of-Thought, ReAct, and Reflection.
"""

from openai import OpenAI
client = OpenAI()

# === 1. CHAIN-OF-THOUGHT (CoT) ===
def solve_with_cot(problem: str) -> str:
    """Single call with explicit step-by-step reasoning (Zero-shot CoT)."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a logical problem solver. Think step by step before answering."},
            {"role": "user", "content": problem}
        ]
    )
    return response.choices[0].message.content or ""
    # Cost: 1 LLM call. Best for: math, closed-world logic.

# === 2. REFLECTION ===
def solve_with_reflection(problem: str) -> str:
    """Generate draft -> Self-critique -> Improve output."""
    # Step 1: Draft initial solution
    draft = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Write a comprehensive solution for: {problem}"}]
    ).choices[0].message.content or ""

    # Step 2: Self-critique
    critique = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a rigorous code reviewer and auditor."},
            {"role": "user", "content": f"Review this draft for logical flaws, edge cases, and missing details:\n\n{draft}"}
        ]
    ).choices[0].message.content or ""

    # Step 3: Revised improvement
    final = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Original Draft:\n{draft}\n\nCritique:\n{critique}\n\nWrite a revised, perfected final version."}
        ]
    ).choices[0].message.content or ""

    return final
    # Cost: 3 LLM calls. Best for: code review, high-stakes reports.
```
