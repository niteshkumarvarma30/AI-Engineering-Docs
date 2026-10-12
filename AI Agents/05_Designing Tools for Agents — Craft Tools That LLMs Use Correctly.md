# Designing Tools for Agents: Craft Tools That LLMs Use Correctly

> **Module**: Agentic AI & Multi-Agent Systems  
> **Topic**: Designing Tools for Agents  
> **Source**: [AIML Companion — Designing Tools](https://aimlcompanion.ai/module/aiAgents/designingTools)

---

## 🎯 In This Module

- **Tool description engineering**: Writing descriptions that guide the LLM to correct tool selection.
- **Parameter design**: Minimal required parameters, sensible defaults, and strict enum constraints.
- **Tool composability**: Why small, single-purpose tools vastly outperform complex "mega-tools".
- **Error reporting patterns**: Returning actionable error dictionaries that enable LLMs to self-correct and recover.
- **Idempotency & safety**: Designing safe-to-retry operations and guarding against duplicate side-effects.
- **Security boundaries**: Sanitizing arguments, preventing SQL/command injection, and treating all LLM inputs as untrusted.

---

## 1. The Big Picture: How It Works, Start to Finish

The tools you provide an agent set the absolute ceiling on its production reliability. A tool the LLM misinterprets gets called at the wrong moment or with hallucinated arguments.

To make an agent use tools correctly on the first attempt, design each tool in a strict 4-step sequence:

```
+-------------------------------------------------------------------------------+
|                        THE 4-STEP TOOL DESIGN SEQUENCE                        |
|                                                                               |
|   01. DESCRIBE IT ──────> 02. SHAPE INPUTS ────> 03. KEEP IT SMALL ──> 04. SAFE TO FAIL |
|   when + what +           1 required param,      One tool, one job;    Validate first;  |
|   returns + example       enums, defaults        split mega-tools      return error +   |
|   (Description = Prompt)  (Fewer ways to slip)   (Higher precision)    fix suggestion   |
+-------------------------------------------------------------------------------+
```

### Two Guiding Principles
1. **Descriptions & Schemas are Prompts**: The LLM reads the description and parameter docs as system instructions. Ambiguity here causes 90% of tool-selection failures.
2. **Treat LLM Output as Untrusted User Input**: An LLM can be manipulated or hallucinate. Never pass raw tool arguments into `eval()`, shell processes, or unsanitized SQL queries.

---

## 2. Walkthrough: Redesigning a Misused Tool

Consider an agent that constantly fails when calling an e-commerce product search tool—passing invalid filters, hallucinating fields, or picking the wrong tool entirely.

```
+-------------------------------------------------------------------------------+
|                       BEFORE vs. AFTER TOOL REDESIGN                          |
+----------------------+-----------------------------+--------------------------+
| Dimension            | ❌ Fragile / Misused Tool    | ✓ Production Redesign    |
+----------------------+-----------------------------+--------------------------+
| Description          | "Search database"           | Comprehensive prompt     |
|                      | (Vague, LLM guesses)        | with WHEN + WHAT + RETURN|
| Parameters           | 8 required parameters       | 1 required parameter     |
|                      | (8 opportunities to fail)   | (`query`), enums, default|
| Architecture         | 1 Mega-tool: `manage_shop()`| 4 Single-purpose tools   |
|                      | (Multi-mode confusion)      | (`search`, `get`, etc.)  |
| Error Handling       | Throws Python Exception     | Returns actionable JSON: |
|                      | (Crashes entire run)        | `{error, suggestion, ...}`|
+----------------------+-----------------------------+--------------------------+
```

### Step 1: Describe It (The Description is a Prompt)
* **Purpose**: *"Search the product catalogue by keyword."*
* **When to use**: *"Use when the user asks about products, prices, or inventory availability."*
* **Returns**: *"List of `{ name, price, in_stock, category }` objects."*
* **Example**: `search_products(query='wireless headphones', category='electronics')`

### Step 2: Shape the Inputs
* **Required parameters cut from 8 to 1**: Only `query` is mandatory.
* **Enums & Defaults**: `category` defaults to `"all"` with allowed enum choices `["electronics", "clothing", "home", "all"]`. Invalid categories are rendered impossible.

### Step 3: Keep It Small (One Tool, One Job)
* **Mega-tool Anti-Pattern**: `manage_products(action="search|update|delete", data={...})` forces the model to juggle conflicting schemas.
* **Composed Suite**: Split into distinct tools: `search_products()`, `get_product_details()`, `update_product()`, `delete_product()`.

### Step 4: Make It Safe to Fail
* Validate inputs *before* database execution.
* If the user asks for a blank search, return:
  ```json
  {
    "error": "Empty query",
    "suggestion": "Please supply a product title or keyword",
    "available_categories": ["electronics", "clothing", "home"]
  }
  ```
  The LLM reads this structured feedback, self-corrects its arguments, and retries successfully.

---

## 3. 💡 Mental Model: Building a Precision Toolkit

```
+-------------------------------------------------------------------------------+
| Component           | Toolkit Analogy                                         |
+---------------------+---------------------------------------------------------+
| 🏷️ Description       | Clear physical label explaining exactly when to use it   |
| 🎛️ Parameters       | Clearly marked adjustment knobs with minimum play        |
| ⚠️ Error Messages   | Safety auto-shutoff with instructions on how to unjam it |
| 🧰 Composability     | Small, specialized tools that fit together into a set   |
+---------------------+---------------------------------------------------------+
```

An agent selects the right tool with far higher consistency from a set of small, clearly labeled tools than from an all-in-one swiss army knife that tries to do everything.

---

## 4. Description Engineering: The 5-Part Formula

When drafting tool documentation, always use this 5-part template:

```
1. Purpose: What does this tool do?
2. When to Use: What user intents or conditions trigger this tool?
3. What It Returns: What schema or data structure comes back?
4. Concrete Example: A sample invocation with realistic arguments.
5. Limitations: What can this tool NOT do? (prevents misuse).
```

### Good vs. Bad Descriptions

```python
# ❌ BAD: Vague (LLM Comprehension: ~20%)
description = "Search database"

# ✓ GOOD: Complete Prompt (LLM Comprehension: ~95%)
description = """Search the product database by keyword or category.
Use when: The user asks to find, browse, or check prices for retail products.
Returns: A list of product objects: {id, name, price, in_stock}.
Example: search_products(query='mechanical keyboard', category='electronics')
Limitations: Does not place orders or show user purchase history."""
```

> **Why do agents pick the wrong tool despite good descriptions?**  
> **Overlapping tools**: If two tools have descriptions that cover adjacent or identical capabilities (e.g., `lookup_user` and `search_customer`), the LLM cannot confidently choose. Sharpen the boundaries and eliminate redundant tools.

---

## 5. Parameter Design: Enums, Defaults, and Anti-Patterns

```
             PARAMETER DESIGN BEST PRACTICES
             
[Minimal Required] ──> Only 1 or 2 fields strictly required
[Use Enums]        ──> Constrain options: ["asc", "desc"] instead of open text
[Smart Defaults]   ──> Supply defaults (e.g. limit=10) so the model doesn't guess
[Rich Descriptions]──> Explain what each parameter controls and expected formats
```

### Parameter Anti-Patterns to Avoid
- **Too many required parameters**: Every required argument increases the probability of hallucination by compounding error rates.
- **Free-text strings where enums fit**: Passing strings like `"cheap"` or `"expensive"` instead of `sort_order: ["price_asc", "price_desc"]`.
- **Missing parameter descriptions**: Omitting parameter docstrings forces the LLM to infer the meaning from the parameter name alone.
- **Deeply nested dictionaries**: Difficult for models to reliably construct in a single pass. Prefer flat, explicit parameters.

---

## 6. Tool Composability: One Tool, One Job

```
❌ MEGA-TOOL (High Failure Rate)
manage_user(action="create|update|delete", user_id="...", payload={...})
↳ LLM frequently picks the wrong action mode or supplies mismatched payload fields.

✓ COMPOSABLE SUITE (High Precision)
├── create_user(name, email)
├── get_user(user_id)
├── update_user(user_id, fields)
└── delete_user(user_id)
↳ Each tool has an unambiguous purpose, a minimal schema, and dedicated validation.
```

### Granularity Guidelines
* **Too Broad**: `manage_everything(action, target, data)` $\rightarrow$ confusing modes, brittle schemas.
* **Too Granular**: `set_user_first_name()`, `set_user_last_name()`, `set_user_phone()` $\rightarrow$ explodes the tool menu, exhausting context windows.
* **Just Right**: `create_user()`, `get_user(id)`, `update_user(id, fields)`.

---

## 7. Error Reporting for Self-Correction & Idempotency

When a tool fails, **never crash the Python runtime**. Returning structured, instructive errors allows the LLM to self-correct in the next turn (**85% recovery rate**).

```python
def search_products(query: str, category: str = "all"):
    # 1. Validate inputs BEFORE execution
    if not isinstance(query, str) or not query.strip():
        return {
            "error": "Empty search query.",
            "suggestion": "Provide a non-empty keyword string.",
            "available_categories": ["electronics", "clothing", "home", "all"]
        }

    # 2. Execute and trap runtime exceptions
    try:
        results = db.query(query, category)
        return {"success": True, "count": len(results), "data": results}
    except ConnectionError as e:
        return {
            "error": f"Database temporarily unavailable: {str(e)}",
            "suggestion": "Retry in 2 seconds or inform user of temporary outage."
        }
```

### Idempotent Tool Design

$$\text{Idempotent: Calling the tool twice produces the identical result without side effects.}$$

```
+-------------------------------------------------------------------------------+
| Type                 | Examples              | Safety & Retry Strategy        |
+----------------------+-----------------------+--------------------------------+
| Idempotent (Safe)    | `get_user(id)`        | Safe to retry automatically    |
|                      | `search_products(q)`  | on network failure.            |
+----------------------+-----------------------+--------------------------------+
| Non-Idempotent       | `create_order(items)` | NOT safe to retry blindly      |
| (Side-effects)       | `charge_credit_card()`| (risk of double charges/dups). |
+----------------------+-----------------------+--------------------------------+
```

#### Best Practices for Non-Idempotent Tools:
1. **Require Idempotency Keys**: Accept a unique client request ID (`idempotency_key="req_123"`) to prevent duplicate transactions.
2. **Add Confirmation Gates**: Introduce a dry-run confirmation step (`confirm=True`) before executing financial or destructive actions.
3. **Deduplication Middleware**: Check whether the exact action was executed in a recent turn within the same session.

---

## 8. Security Boundaries: Guarding the Tool Layer

Tools represent the physical bridge between generative models and real-world systems. An agent should never trust an argument simply because an LLM generated it.

```
[Prompt Injection / Attacker] ──> [LLM] ──(Emits malicious args)──> [🛡️ SECURITY GUARD] ──> [Database / Shell]
                                                                     ↳ Blocks dangerous queries,
                                                                       validates types, checks auth
```

### Tool Security Checklist
- [ ] **No Raw Shell/Exec**: Never pass arguments directly to `os.system()`, `subprocess.call()`, or `eval()`.
- [ ] **Parameterized SQL Queries**: Never interpolate LLM strings into raw SQL (`f"SELECT * WHERE id={id}"`). Always use parameterized placeholders (`db.execute("... WHERE id=?", [id])`).
- [ ] **Strict Directory Allowlists**: File operations must check that target paths are within an approved sandbox (prevent path traversal `../../etc/passwd`).
- [ ] **Authentication & Permissions**: Verify the current user's role before executing privileged operations.
- [ ] **Rate Limiting & Logging**: Track tool call frequency per session and log all invocations for auditability.

---

## 9. ⚡ 2-Minute Cheatsheet & Interview Guide

- **The 4 Steps**: **Describe** (when + what + returns + example) $\rightarrow$ **Shape** (1 required, enums, defaults) $\rightarrow$ **Shrink** (one tool, one job) $\rightarrow$ **Guard** (validate args, return error + suggestion).
- **The Core Equation**: `Description + Schema = Prompts`. Clear schemas eliminate 90% of failures upfront.
- **Why Small Tools Win**: Splitting mega-tools into single-purpose functions gives each tool an unambiguous schema and prevents mode confusion.
- **Error Recovery**: Return `{error, suggestion, available_options}` so the agent can self-correct instead of dying on an unhandled exception.
- **Security Rule**: Treat all LLM-generated arguments like untrusted user input. Use parameterization, allowlists, and confirm steps.

### Top Interview Questions & Model Answers
1. **If an agent keeps picking the wrong tool, what is the first thing you should fix?**  
   *The tool description. Descriptions act as targeted prompts. Clarify the "When to use" clause and include a concrete invocation example.*
2. **Why do we prefer several small tools over one multi-purpose mega-tool?**  
   *Mega-tools force the LLM to choose an internal mode and construct complex conditional arguments. Small tools have simpler, dedicated schemas that LLMs select with significantly higher accuracy.*
3. **How do you handle tools that execute non-idempotent actions (like charging a credit card)?**  
   *Use idempotency keys, deduplication caches, and human confirmation gates to prevent accidental re-runs on network retries.*

---

## 10. 💻 Code Examples

### Example 1: Production-Grade Tool Suite with Error Handling

```python
"""
production_tool_suite.py
A robust product catalog tool suite with explicit descriptions,
defaults, validation, and actionable error handling.
"""

from openai import OpenAI
import json

client = OpenAI()

# 1. Production Tool Definitions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Search the product catalog by keyword or category. "
                "Use when: User asks about products, prices, or inventory. "
                "Returns: List of {id, name, price, in_stock, category} items. "
                "Example: search_products(query='wireless headphones', category='electronics')"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords to match against product titles and descriptions."
                    },
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "home", "all"],
                        "default": "all",
                        "description": "Product category filter."
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Upper price limit in USD."
                    },
                    "in_stock_only": {
                        "type": "boolean",
                        "default": True,
                        "description": "If true, only returns items currently in stock."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": (
                "Get full specifications and review summary for a single product by ID. "
                "Use when: User references a specific product found via previous search results. "
                "Returns: {id, name, price, specs, reviews_summary, in_stock}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The exact product identifier (e.g., 'PROD-101')."
                    }
                },
                "required": ["product_id"]
            }
        }
    }
]

# 2. Defensive Tool Execution Layer
def execute_tool_safely(name: str, args: dict) -> dict:
    """Validates parameters before execution and returns structured errors."""
    if name == "search_products":
        query = args.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return {
                "status": "error",
                "error": "Query cannot be blank.",
                "suggestion": "Supply a specific product name or keyword."
            }

        # Mock successful database query
        return {
            "status": "success",
            "count": 1,
            "results": [
                {"id": "PROD-101", "name": "AeroPress Go", "price": 29.99, "in_stock": True, "category": "home"}
            ]
        }

    elif name == "get_product_details":
        product_id = args.get("product_id", "")
        if not product_id.startswith("PROD-"):
            return {
                "status": "error",
                "error": f"Invalid product ID format: '{product_id}'",
                "suggestion": "Product IDs must match the pattern 'PROD-XXX' found in search results."
            }

        return {
            "status": "success",
            "product": {
                "id": product_id,
                "name": "AeroPress Go",
                "specs": "Travel coffee press, 8 oz capacity",
                "reviews_summary": "4.8/5 based on 1,200 reviews"
            }
        }

    return {"status": "error", "error": f"Unknown tool: '{name}'"}
```

---

### Example 2: Automated Tool Selection Unit Testing

```python
"""
test_tool_selection.py
Testing that the LLM reliably selects the expected tool for varied user prompts.
"""

from openai import OpenAI

client = OpenAI()

def test_tool_selection(user_prompt: str, expected_tool: str, tool_schemas: list) -> bool:
    """Verifies that an LLM selects the correct tool on the first turn."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_prompt}],
        tools=tool_schemas,
        tool_choice="auto"
    )

    msg = response.choices[0].message
    if not msg.tool_calls:
        print(f"❌ FAIL: No tool called for: '{user_prompt}'")
        return False

    selected_tool = msg.tool_calls[0].function.name
    if selected_tool == expected_tool:
        print(f"✅ PASS: '{user_prompt}' -> {selected_tool}")
        return True
    else:
        print(f"❌ FAIL: Expected '{expected_tool}', got '{selected_tool}'")
        return False

if __name__ == "__main__":
    test_suite = [
        ("Find me a portable coffee maker under $40", "search_products"),
        ("Tell me more specifications about PROD-101", "get_product_details"),
        ("What home appliances do you have in stock?", "search_products")
    ]

    print("Running Tool Selection Test Suite...")
    for prompt, expected in test_suite:
        test_tool_selection(prompt, expected, TOOLS)
```
