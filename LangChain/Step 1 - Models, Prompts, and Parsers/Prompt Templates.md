# Prompt Templates in LangChain

To build robust applications, you cannot simply concatenate strings. If user input contains weird characters, or if you need to dynamically inject 50 pages of retrieved documents, standard Python string manipulation becomes a source of critical bugs and security vulnerabilities (like Prompt Injection).

LangChain’s Prompt Templates layer solves this by treating prompts as strongly-typed, compiled functions. Here is the comprehensive computer science breakdown of all concepts and tools available in this layer.

---

## 1. The Core Abstractions

LangChain divides templates into two main classes to match the two types of models (Legacy LLMs vs Modern Chat Models).

* **PromptTemplate (String Prompts)**: Used for older models. It outputs a single monolithic string.
* **ChatPromptTemplate (Message Prompts)**: The standard for modern apps. It outputs a strictly ordered list of `BaseMessage` objects (System, Human, AI). When you invoke a `ChatPromptTemplate`, it doesn't return a string; it returns a `ChatPromptValue` object, which the Model execution engine knows how to serialize into the specific API format for OpenAI, Google, or Anthropic.

---

## 2. Message Placeholders (Dynamic Message Arrays)

Standard variables like `{user_name}` inject a string into a message. But what if you need to inject an entire list of previous chat messages? You cannot shove a list of objects into a single string variable.

To solve this, LangChain uses `MessagesPlaceholder`. It reserves a slot in the prompt structure where an array of `BaseMessage` objects will be injected at runtime.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant."),
    # This placeholder will be replaced by the actual history list
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{current_input}")
])

# Injecting the history at runtime
formatted_prompt = prompt.invoke({
    "chat_history": [
        HumanMessage(content="Hi, I'm Bob."),
        AIMessage(content="Hello Bob!")
    ],
    "current_input": "What is my name?"
})
```

---

## 3. Template Formatting Engines

LangChain supports three different templating languages to parse your variables.

* **f-string (Default)**: Python's native format. Best for flat data. Fails if you try to pass nested JSON (e.g., `{user.name}`) or if you need logic.
* **mustache (Advanced)**: A logic-less templating engine (syntax: `{{variable}}`). You must use Mustache if you are passing nested objects or if you need to loop through an array directly inside the prompt text (using `{{#items}}...{{/items}}`).
* **jinja2 (Legacy/Dangerous)**: Allows arbitrary Python code execution inside the template. LangChain sandboxes it by default now, but its use is strongly discouraged in production due to security risks.

---

## 4. Composition and Partialing (DRY Principle)

In enterprise apps, you don't write one massive prompt. You build modular prompt components and combine them, adhering to the DRY (Don't Repeat Yourself) principle.

### A. Partial Variables (`.partial()`)
Sometimes you have a template with two variables (e.g., `{date}` and `{user_input}`), but you only know the date at startup, while the `user_input` comes later from the web request. You can "partial" the template to pre-fill known variables, creating a new template that only requires the remaining ones.

```python
from datetime import datetime
from langchain_core.prompts import PromptTemplate

base_prompt = PromptTemplate.from_template("Today is {date}. User says: {input}")

# Bind the date variable immediately
partial_prompt = base_prompt.partial(date=datetime.now().strftime("%Y-%m-%d"))

# Later in the API route, you only need to pass 'input'
final_prompt = partial_prompt.invoke({"input": "Hello!"})
```
*Note: You can also pass a function to `.partial()`. LangChain will execute the function at runtime to grab the value (excellent for fetching the live system time).*

### B. Pipeline Prompts (Composition)
You can compose multiple templates together. For example, you might have one template for "Formatting rules", one for "Tone of voice", and one for "The task", and merge them into a single `PipelinePromptTemplate`. You can also simply concatenate string templates using the `+` operator.

---

## 5. Few-Shot Prompting & Example Selectors

LLMs perform drastically better when you show them examples of the desired input and output ("Few-Shotting"). LangChain provides specialized classes to manage these examples.

* **FewShotPromptTemplate / FewShotChatMessagePromptTemplate**: These classes take a list of dictionary examples and automatically format them into System/Human/AI message blocks before appending the user's actual question.

### Dynamic ExampleSelector
If you have 1,000 examples of how to generate SQL queries, you cannot pass all 1,000 into the prompt (it exceeds the token limit and costs too much).

LangChain uses an `ExampleSelector` (specifically `SemanticSimilarityExampleSelector`). It stores your 1,000 examples in a Vector Database. When a user asks a question, the selector runs a mathematical similarity search, grabs only the 3 examples most relevant to the user's specific question, and injects only those 3 into the prompt dynamically.

```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# 1. Define how a SINGLE example should look
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

# 2. Provide the data (In a real app, this is fetched via an ExampleSelector)
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"}
]

# 3. Create the Few-Shot template
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

# 4. Assemble the final prompt
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an antonym generator."),
    few_shot_prompt, # Injects the examples here
    ("human", "{user_word}")
])
```

---

## Summary of the Prompts Layer

As a developer, you should view this layer as the compiler for the LLM's context window. You use `ChatPromptTemplate` to structure the layout, `MessagesPlaceholder` to manage dynamic memory state, `.partial()` to inject environment variables, and `ExampleSelector` to algorithmically choose which few-shot examples will yield the most accurate model execution.
