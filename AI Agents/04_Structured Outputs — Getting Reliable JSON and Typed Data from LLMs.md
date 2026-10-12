# Structured Outputs: Getting Reliable JSON and Typed Data from LLMs

> **Module**: Agentic AI & Multi-Agent Systems  
> **Topic**: Structured Outputs  
> **Source**: [AIML Companion — Structured Outputs](https://aimlcompanion.ai/module/aiAgents/structuredOutputs)

---

## 🎯 In This Module

- **The output modes spectrum**: The fundamental differences between **Free Text**, **JSON Mode**, and **Structured Outputs**.
- **Pydantic schema definition**: How to define and validate LLM output schemas with Python types and field constraints.
- **Constrained decoding**: How OpenAI and Anthropic guarantee exact schema compliance at generation time.
- **OpenAI vs. Anthropic approaches**: Native `response_format` / `text_format` vs. forced `tool_choice` extraction patterns.
- **The Instructor library**: Building resilient structured data workflows with automatic validation and retries.

---

## 1. The Big Picture: How It Works, Start to Finish

When you ask an LLM for structured data without enforcement, it typically hands back conversational paragraphs or messy Markdown blocks. Your code attempts to parse it with regular expressions or string trimming, and downstream pipelines break.

**Structured Outputs** solve this by attaching a schema directly to the request:
1. You define the exact shape you want (usually a **Pydantic model**).
2. The LLM is **constrained at token generation time** to emit only tokens that keep the output valid against that schema.
3. You parse the raw output directly into a **typed, validated object**.

```
+-------------------------------------------------------------------------------+
|                        STRUCTURED OUTPUT GENERATION FLOW                      |
|                                                                               |
|   01. DEFINE SHAPE ──────> 02. ATTACH SCHEMA ────> 03. CONSTRAINED ──> 04. VALIDATE   |
|   (Pydantic Model)         (Prompt + Schema)        DECODING           (Typed Object) |
|   Python types become       Model receives prompt   Only schema-valid   Parse into    |
|   JSON Schema               with format constraint  tokens generated    checked data  |
+-------------------------------------------------------------------------------+
```

### The Core Principle: Structural Guarantee vs. Semantic Validity
$$\mathbf{\text{Structural Guarantee } \checkmark \quad \text{vs.} \quad \text{Semantic Guarantee } \times}$$

* **What is guaranteed**: Field names, data types, required keys, and enum constraints are **correct by construction**. No missing fields, no unexpected keys, and no malformed JSON.
* **What is NOT guaranteed**: The LLM can still place an inaccurate or hallucinated value into a valid slot. You still need domain checks (`Field(ge=0)`, enums, and `@field_validator`).

---

## 2. Core Concepts & Terminology: Demystifying the Terms

Before diving deeper, it is critical to understand the distinction between these four closely related terms:

```
[Developer writes] ──> Pydantic Schema (Python class with type hints)
                              │
                              ▼ (Auto-converted)
[API receives]     ──> JSON Schema (Universal JSON specification)
                              │
                              ▼ (Constrains generation)
[LLM executes]     ──> Structured Output (Constrained token decoding)
                              │
                              ▼ (Parsed back into)
[Code consumes]    ──> Typed Pydantic Object (Validated Python instance)
```

### 1. What is a Schema? (The Blueprint)
A **Schema** is a generic concept: it is an **abstract blueprint or contract** that defines how data must be organized.
* It defines what field names/keys must exist.
* It defines the allowed data types (e.g., string, integer, boolean).
* It defines constraints and rules (e.g., minimum values, regex patterns).
* 💡 *Analogy*: An empty physical job application form. It requires you to fill in your name (letters only), age (number only), and email.

---

### 2. What is a JSON Schema? (The Universal Standard)
A **JSON Schema** is a formal, standardized specification written in JSON that defines the structure and validation rules for JSON documents.
* It is provider-agnostic and language-agnostic.
* Every major AI provider (OpenAI, Anthropic, Google Gemini) uses **JSON Schema** as the universal format to define tool parameters and structured output contracts.
* *Example*:
  ```json
  {
    "type": "object",
    "properties": {
      "name": { "type": "string", "description": "User's full name" },
      "age": { "type": "integer", "minimum": 18 },
      "is_active": { "type": "boolean" }
    },
    "required": ["name", "age"],
    "additionalProperties": false
  }
  ```

---

### 3. What is a Pydantic Schema? (The Python Developer's Tool)
A **Pydantic Schema** is a Python class inheriting from `pydantic.BaseModel` that defines your data structure using native Python type hints.
* Instead of writing verbose JSON Schemas manually, developers write clean Python code.
* Pydantic automatically converts your Python model into a JSON Schema via `.model_json_schema()`.
* When the LLM responds, Pydantic validates the JSON and instantiates a typed Python object (`Product.model_validate_json(...)`).
* *Example*:
  ```python
  from pydantic import BaseModel, Field

  class User(BaseModel):
    name: str = Field(description="User's full name")
    age: int = Field(ge=18, description="Age, must be at least 18")
    is_active: bool = True
  ```

---

### 4. What is Structured Output? (The LLM Feature)
**Structured Output** is a capability provided by LLM inference engines where the model is **mechanically constrained at token-generation time** to emit text that 100% conforms to a given JSON Schema.
* **Prompt pleading (Old way)**: *"Please reply in valid JSON with keys 'name' and 'age'"* $\rightarrow$ The model often adds markdown backticks, hallucinates keys, or returns invalid syntax.
* **Structured Output (Modern way)**: The API engine uses **grammar-based constrained decoding**. The token sampler masks out any tokens that would violate the schema grammar. The model physically cannot emit invalid JSON or the wrong keys.

---

### Summary Table

| Term | What It Is | Who Uses It | Key Purpose |
| :--- | :--- | :--- | :--- |
| **Schema** | Abstract concept of a data contract | Computer Science & DBs | Defines structure and rules |
| **JSON Schema** | Standardized JSON document defining a contract | APIs, LLM Engines | Universal machine-readable format |
| **Pydantic Schema** | Python class (`BaseModel`) with type annotations | Python Developers | Clean authoring + runtime validation |
| **Structured Output** | LLM inference feature enforcing exact schema match | LLM Provider APIs | Guarantees 100% schema compliance |

---

## 3. Walkthrough: One Messy Blurb into Typed Data

**Input Blurb**:
> *"The AeroPress Go is just \$29.99 and yes, it's in stock!"*

**Desired Output**: Typed data your code can trust: `{ name: str, price: float, in_stock: bool }`.

```
Step 1: Define the Shape (Pydantic Model)
------------------------------------------
class Product(BaseModel):
    name: str
    price: float = Field(ge=0)  # Numeric float, never a '$29.99' string
    in_stock: bool              # Boolean true/false, never the word 'yes'

schema = Product.model_json_schema()

Step 2: Send the Blurb with Schema Attached
-------------------------------------------
You pass the blurb plus the schema to the API. The decoding engine constrains
the model so it can only emit tokens that satisfy { name, price, in_stock }.

Step 3 & 4: Constrained Decode & Parse Straight to Object
---------------------------------------------------------
Because types are enforced during token generation, the '$' symbol and the word
'yes' are masked out and cannot be emitted for numeric or boolean slots:

Raw Output: {"name": "AeroPress Go", "price": 29.99, "in_stock": true}
Parsed:     Product(name='AeroPress Go', price=29.99, in_stock=True)

Result: Typed, validated data with zero regex or string parsing hacks!
```

---

## 3. 💡 Mental Model: Filling Out a Form

```
+-------------------------------------------------------------------------------+
| Mode               | Form Analogy                                             |
+--------------------+---------------------------------------------------------+
| 📄 Free Text       | Writing an open letter: Any format, any arbitrary content|
| 📋 JSON Mode       | A form with blank boxes: Structured syntax, but flexible|
|                    | keys and missing fields                                 |
| 🔒 Structured Output| A standardized digital form with dropdowns and enforced |
|                    | checkboxes: Exact schema compliance guaranteed           |
+--------------------+---------------------------------------------------------+
```

JSON mode only guarantees **valid JSON syntax** `{ ... }`. Structured outputs guarantee **your exact schema**—every required key, correct type, and valid enum.

---

## 4. The Output Modes Spectrum: Pick Your Level

```
1. FREE TEXT
   - Configuration: Default (`response_format=None`)
   - Output: Unconstrained text, Markdown, prose
   - Reliability: ~30% for automated pipelines
   - Best for: Creative writing, summaries, user-facing conversation

2. JSON MODE
   - Configuration: `response_format={"type": "json_object"}`
   - Output: Valid JSON syntax, but keys and types may deviate or hallucinate
   - Reliability: ~70%
   - Best for: Flexible, schema-less key-value pairs

3. STRUCTURED OUTPUTS
   - Configuration: `response_format={"type": "json_schema", ...}` / `responses.parse`
   - Output: Strict JSON Schema compliance (constrained decoding)
   - Reliability: ~98%+
   - Best for: Data extraction, tool calling, classification, production pipelines
```

### Reliability Comparison

| Feature | Free Text | JSON Mode | Structured Outputs |
| :--- | :---: | :---: | :---: |
| **Valid JSON Guaranteed?** | ❌ No | ✅ Yes | ✅ Yes |
| **Key Names Enforced?** | ❌ No | ❌ No | ✅ Yes |
| **Field Types Enforced?** | ❌ No | ❌ No | ✅ Yes |
| **Required Fields Checked?** | ❌ No | ❌ No | ✅ Yes |
| **Enum Constraints Enforced?** | ❌ No | ❌ No | ✅ Yes |
| **Parsing Effort** | High (Regex, fallback) | Medium (Key checks) | Zero (`model_validate_json`) |

---

## 5. Pydantic Models for Schema Definition

Pydantic is the industry standard for defining data contracts in Python. When you define a `BaseModel`, you get:
1. Automatic JSON Schema conversion via `.model_json_schema()`.
2. Runtime data validation and type coercion.
3. Informative docstrings that guide LLM token generation.

```python
from pydantic import BaseModel, Field
from typing import Literal

class MovieReview(BaseModel):
    title: str = Field(description="Official title of the movie")
    year: int = Field(ge=1900, le=2030, description="Release year")
    rating: float = Field(ge=0.0, le=10.0, description="Rating score between 0 and 10")
    genre: Literal["action", "drama", "comedy", "sci-fi", "horror", "other"]
    pros: list[str] = Field(min_length=1, max_length=5, description="Top positive points")
    cons: list[str] = Field(min_length=1, max_length=5, description="Key criticisms")
    recommendation: bool = Field(description="True if recommended, False otherwise")
```

### Why Strong Schemas Prevent Hallucinations
* `Literal[...]` / `Enum`: Limits choices to a fixed set, eliminating made-up category strings.
* `ge` / `le` constraints: Bounds numeric outputs (e.g. `rating: 1-10`), stopping out-of-range figures.
* `Field(description=...)`: Acts as a targeted sub-prompt explaining what belongs in each field.

---

## 6. Provider Implementations: OpenAI vs. Anthropic

```
+-------------------------------------------------------------------------------+
| Provider   | Primary Method                     | Underlying Mechanism        |
+------------+------------------------------------+-----------------------------+
| OpenAI     | `responses.parse` / `json_schema`  | Native constrained decoding |
| Anthropic  | Native `output_config.format` or   | Grammar-constrained decode  |
|            | Forced `tool_choice` extraction    | or single-tool enforcement  |
+-------------------------------------------------------------------------------+
```

### 1. OpenAI Native Structured Output
OpenAI enforces schema compliance at the sampling layer using **constrained decoding** (masking invalid tokens during generation):

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class ExtractionResult(BaseModel):
    company: str
    valuation_usd: float
    is_public: bool

# Using the Responses API with strict parsing
response = client.responses.parse(
    model="gpt-4o-mini",
    input=[{"role": "user", "content": "Stripe raised at a $65 billion valuation as a private company."}],
    text_format=ExtractionResult
)

# Parsed result is an instantiated, type-safe ExtractionResult object
data = response.output_parsed
print(f"{data.company}: ${data.valuation_usd:,.0f} (Public: {data.is_public})")
```

---

### 2. Anthropic Forced Tool Use Pattern
On models that use the Messages API, you can guarantee structured outputs by offering a single tool and **forcing** Claude to call it:

```python
import anthropic

client = anthropic.Anthropic()

extraction_tool = {
    "name": "record_review",
    "description": "Save structured movie review data.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "rating": {"type": "number"},
            "verdict": {"type": "string", "enum": ["watch", "skip"]}
        },
        "required": ["title", "rating", "verdict"]
    }
}

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=[extraction_tool],
    tool_choice={"type": "tool", "name": "record_review"},  # Forces Claude to emit this tool call
    messages=[{"role": "user", "content": "The Matrix is a classic 9.5/10 masterpiece. Must watch!"}]
)

# Extract structured arguments directly
data = response.content[0].input
print(data)  # {'title': 'The Matrix', 'rating': 9.5, 'verdict': 'watch'}
```

---

## 7. The Instructor Library: Clean, Provider-Agnostic Workflows

The [`instructor`](https://python.useinstructor.com) library wraps both OpenAI and Anthropic clients to provide:
1. Native integration with Pydantic `response_model`.
2. Automatic re-prompting and retries when Pydantic validation fails (`max_retries=2`).
3. Support for custom `@field_validator` functions.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

# Patch client with Instructor
client = instructor.from_openai(OpenAI())

class UserProfile(BaseModel):
    username: str
    age: int = Field(ge=18, le=120)
    email: str

    @field_validator("email")
    @classmethod
    def must_contain_at(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Email must contain '@'")
        return v

# If the LLM generates an invalid email, Instructor automatically feeds the
# validation error back to the model and retries!
profile = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=UserProfile,
    max_retries=2,
    messages=[{"role": "user", "content": "Create a profile for Alex, 28 years old, alex[at]example.com"}]
)

print(profile.username, profile.email)
```

---

## 8. Where to Use Structured Outputs in Agent Systems

Structured outputs are not just for final user responses; they are essential across an agent's internal architecture:

```
[User Input]
     │
     ▼
[1. Routing / Intent Classification] ──> { intent: "refund", confidence: 0.94 }
     │
     ▼
[2. Entity Extraction]               ──> { order_id: "ORD-4471", item: "Laptop" }
     │
     ▼
[3. Action Selection]                ──> { tool: "lookup_order", args: { ... } }
     │
     ▼
[4. Final Synthesis]                 ──> Typed Response Object
```

---

## 9. ⚡ 2-Minute Cheatsheet & Interview Guide

- **What**: Enforcing exact schema compliance on LLM generation so outputs parse directly into typed, validated Python objects.
- **The 3 Levels**: Free Text (unpredictable) $\rightarrow$ JSON Mode (valid JSON syntax only) $\rightarrow$ Structured Outputs (exact JSON Schema guaranteed).
- **How It Works**: Constrained decoding. Token logits are filtered so the model can only sample tokens that maintain schema validity.
- **The Golden Rule**: The guarantee is **structural**, not semantic. The shape is always right; values must still be validated with bounds, enums, and validators.
- **Instructor**: Use `instructor` to wrap OpenAI, Anthropic, Gemini, or Groq for automatic validation retries.

### Top Interview Questions & Model Answers
1. **What is the difference between JSON Mode and Structured Outputs?**  
   *JSON Mode guarantees that the string is syntactically valid JSON, but the LLM may invent unexpected keys or omit required fields. Structured Outputs guarantees 100% adherence to your specific JSON Schema (field names, types, enums, and required properties).*
2. **How do providers enforce Structured Outputs?**  
   *Through constrained decoding at inference time. The generation engine converts the JSON schema into a grammar (CFG) or finite-state machine that masks out invalid tokens before sampling.*
3. **Can an LLM still hallucinate when Structured Outputs are active?**  
   *Yes. Structured outputs enforce the data container (e.g. `price` will be a `float`), but the numerical value itself could still be factually incorrect. Always pair structural constraints with semantic validators.*

---

## 10. 💻 Code Examples

### Example 1: Entity Extraction with Pydantic & OpenAI

```python
"""
pydantic_openai_extraction.py
Extracting named entities with strict schema compliance.
"""

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal

client = OpenAI()

class Entity(BaseModel):
    name: str = Field(description="Entity name as it appears in text")
    type: Literal["person", "organization", "location", "date"]
    confidence: float = Field(ge=0, le=1, description="Confidence score from 0 to 1")

class ExtractionResult(BaseModel):
    entities: list[Entity]
    summary: str = Field(description="One-sentence summary of the passage")
    language: str = Field(description="Detected language")

def extract():
    text = "Apple CEO Tim Cook announced a new product launch in Cupertino on March 15, 2026."
    
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": text}],
        text_format=ExtractionResult
    )

    result = response.output_parsed
    print(f"Summary: {result.summary} ({result.language})\n")
    for entity in result.entities:
        print(f"- {entity.name} [{entity.type}] ({entity.confidence:.0%})")

if __name__ == "__main__":
    extract()
```

---

### Example 2: Instructor with Custom Validation & Retries

```python
"""
instructor_validation_example.py
Demonstration of Instructor with automatic retries and field validators.
"""

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from typing import Literal

# Initialize instructor client
client = instructor.from_openai(OpenAI())

class MovieReview(BaseModel):
    title: str
    year: int = Field(ge=1900, le=2030)
    rating: float = Field(ge=0, le=10)
    genre: Literal["action", "drama", "comedy", "sci-fi", "horror", "other"]
    pros: list[str] = Field(min_length=1, max_length=5)
    cons: list[str] = Field(min_length=1, max_length=5)
    verdict: Literal["watch", "skip", "maybe"]

    @field_validator("pros", "cons")
    @classmethod
    def not_empty_strings(cls, items: list[str]) -> list[str]:
        cleaned = [item.strip() for item in items if item.strip()]
        if not cleaned:
            raise ValueError("Items list cannot be empty or contain blank strings.")
        return cleaned

def review_movie():
    review = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=MovieReview,
        max_retries=2,
        messages=[{"role": "user", "content": "Write a critical review of Inception (2010)"}]
    )

    print(f"Title: {review.title} ({review.year})")
    print(f"Rating: {review.rating}/10 | Verdict: {review.verdict.upper()}")
    print(f"Genre: {review.genre}")
    print(f"Pros: {', '.join(review.pros)}")
    print(f"Cons: {', '.join(review.cons)}")

if __name__ == "__main__":
    review_movie()
```
