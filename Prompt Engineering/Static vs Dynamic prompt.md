# Static vs Dynamic Prompts

In AI engineering and application development, the difference between a **Static Prompt** and a **Dynamic Prompt** comes down to whether the text sent to the LLM is permanent and unchanging, or assembled on the fly using real-time data.

Here is a breakdown of how both work, how they are implemented, and where they are used.

---

## 1. What is a Static Prompt?

A **Static Prompt** is a fixed, hardcoded string of text that never changes, regardless of who is using the application or when it is run. 

You write it once during development, paste it directly into your code or configuration file, and it remains identical for every single API call.

### Code Example (Python)
```python
# The prompt is hardcoded and fixed
static_prompt = "Translate the following English sentence into French: Hello, how are you today?"

response = client.generate(prompt=static_prompt)
```

### When to use it:
*   **One-off exploratory tasks**: Typing directly into a web interface like ChatGPT, Claude, or Gemini.
*   **Highly predictable automation**: System instructions that set an unchanging persona, such as: *"You are a helpful customer service agent. Always respond politely and keep answers under 3 sentences."*

---

## 2. What is a Dynamic Prompt?

A **Dynamic Prompt** acts as a programming template. It contains variables, placeholders, or slots that are programmatically injected with live data right at the exact moment a user triggers an action.

Dynamic prompts are the backbone of all production-grade AI applications, including RAG pipelines, chatbots, and AI agents.

### Code Example (Python using F-Strings)
```python
# The prompt uses placeholders ({user_name}, {user_query}, {retrieved_context})
dynamic_prompt_template = """
You are an AI assistant helping our customer, {user_name}.

Use ONLY the following context to answer their question:
---
{retrieved_context}
---

Customer Question: {user_query}
Answer:"""

# Data is injected dynamically at runtime
final_prompt = dynamic_prompt_template.format(
    user_name="Alice",
    user_query="What is my current shipping status?",
    retrieved_context="Order #1042: Shipped on May 21, 2026 via FedEx. Status: Out for delivery."
)

response = client.generate(prompt=final_prompt)
```

### When to use it:
*   **Personalized User Experiences**: Pulling in a user's name, subscription tier, or past preferences.
*   **Retrieval-Augmented Generation (RAG)**: Squeezing relevant document chunks matching a user's search query into the prompt template.
*   **Time-Aware/Location-Aware Applications**: Automatically injecting variables like `current_date` or `user_location` to ground the model's logic.

---

## ⚖️ Summary of Key Differences

| Feature | Static Prompt | Dynamic Prompt |
| :--- | :--- | :--- |
| **Flexibility** | Unchanging. Every execution runs the exact same text. | Fluid. Adapts instantly based on inputs, database queries, or user state. |
| **Where it Lives** | Directly in the code strings or configuration files. | Handled via template engines (like Jinja2 or LangChain prompt templates). |
| **Primary Use Case** | Setting permanent rules, constraints, or tones (System Prompts). | Injecting operational runtime data and handling dynamic user inputs. |
| **Maintenance** | Easy to update manually, but doesn't scale for complex user interactions. | Requires engineering logic to ensure data injection doesn't break formatting. |

---

## 🔄 The Hybrid Approach

In modern AI systems, developers typically blend both approaches: they use a **Static Prompt** to dictate the core rules, persona, and safety boundaries of the AI system (e.g., System Prompt), and nest a **Dynamic Prompt** canvas right underneath it to feed the live query and retrieved data (e.g., User Prompt).
