# Introduction to FastAPI & Environment Setup

FastAPI is a modern, high-performance web framework for building APIs with Python. It is designed to be developer-friendly, fast to write, and ready for production serving.

---

## ⚡ 1. Key Advantages of FastAPI

*   **Speed**: It is one of the fastest Python frameworks available. Running on top of **Starlette** (for web handling) and **Pydantic** (for data validation), its raw performance is on par with NodeJS and Go.
*   **Fast Development & Fewer Bugs**: It leverages Python's native type hinting to speed up development and significantly reduce human-induced coding errors (up to 40% reduction in developer bugs).
*   **Standards-Based**: It is fully compatible with open standards, including **OpenAPI** (formerly Swagger) and **JSON Schema**.

### Installation
To install FastAPI along with its core dependencies, run Python's package manager:

```bash
pip3 install fastapi
```

> [!NOTE]
> Installing `fastapi` automatically downloads its required underlying libraries, including Starlette (the async web microframework) and Pydantic (the data validation library).

---

## ⚙️ 2. Understanding the ASGI Server (Uvicorn)

Unlike traditional Python web frameworks (such as Flask), FastAPI does not come with a built-in development server. To run a FastAPI application, you need an **ASGI** (Asynchronous Server Gateway Interface) server.

### Why Uvicorn?
*   **Asynchronous Handling**: Older WSGI servers (like Gunicorn or uWSGI) are synchronous by design and cannot natively handle async (`asyncio`) workloads. ASGI servers allow FastAPI to handle high-concurrency connections efficiently.
*   **Advanced Protocols**: Uvicorn natively supports modern web protocols, including WebSockets and HTTP/2.

### Installation & Launching
You can install Uvicorn with standard, cython-based speed boosts and development tools (like watchfiles for hot-reloading) using the `[standard]` flag:

```bash
pip3 install "uvicorn[standard]"
```

---

## 🚀 3. Creating Your First "Hello World" App

Building a basic endpoint involves initializing the application object and binding an async handler function to a specific path using a route decorator.

### Code Example (`main.py`)
```python
from fastapi import FastAPI

# Step 1: Initialize the FastAPI application object
app = FastAPI()

# Step 2: Define a path operation decorator and a view function
@app.get("/")
async def index():
    return {"message": "Hello World"}
```

### Key Terms Explained

| Term | Explanation |
| :--- | :--- |
| **Application Object (`app`)** | The primary interaction point between your application and the client web server. The Uvicorn server listens for requests and routes them through this instance. |
| **Path / Route** | The trailing portion of the URL after the domain name. For example, in `http://localhost:8000/hello`, the path is `/hello`. |
| **Operation** | The HTTP request method/verb used by the client (e.g., `GET`, `POST`, `PUT`, `DELETE`). |
| **Path Operation Decorator** | The decorator line directly preceding the view function (like `@app.get("/")`) mapping both the path and the HTTP method to the code. |
| **Path Operation Function** | The Python function (e.g., `index()`) that executes and returns a response (which is automatically serialized into JSON) when a client visits the mapped route. |
| **`async` Keyword** | Informs FastAPI that the function can run asynchronously without blocking the underlying execution threads. *(Note: You can write standard synchronous functions without the `async` prefix if preferred).* |

### Starting the Server

#### A. Via the Command Line
```bash
uvicorn main:app --reload
```
*   `main`: Refers to your Python script name (`main.py`).
*   `app`: Refers to the variable holding your `FastAPI()` instance.
*   `--reload`: Enables automatic code reloading. The server will restart itself whenever it detects code modifications.

#### B. Programmatically (Inside the Script)
Alternatively, you can run the server directly inside your script:
```python
import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def index():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
```

---

## 📖 4. Automatic Documentation (OpenAPI & ReDoc)

One of FastAPI's most powerful native features is automatic schema generation. It reads your code and types, instantly providing two standard interactive API documentation interfaces:

1.  **Swagger UI** (`http://127.0.0.1:8000/docs`): An interactive portal where you can inspect available routes, review expected parameters, and execute live API requests directly from the browser using the **"Try it out"** feature.
2.  **ReDoc** (`http://127.0.0.1:8000/redoc`): A clean, professional, three-panel alternative design for viewing API specifications.

> [!TIP]
> Under the hood, FastAPI generates a complete, standardized API schema following the OpenAPI specification. This raw JSON schema is accessible directly at `http://127.0.0.1:8000/openapi.json`.

---

## 🛡️ 5. Python Type Hints & Pydantic Validation

FastAPI leverages standard Python **Type Hints** (introduced in Python 3.5+) to enforce robust runtime data validation, parsing, and type coercion.

### Type Hints Recap
Vanilla Python is dynamically typed. Type annotations (e.g., `variable: type`) inform linters, IDEs, and frameworks of the expected data categories:
*   **Basic Types**: Primitives like `int`, `float`, `str`, and `bool`.
*   **Complex Collections**: Structured collections imported from the built-in `typing` library, such as `List`, `Dict`, and `Tuple` (e.g., `subjects: List[str]`).

### Integrating Pydantic
Pydantic is the validation engine under the hood. You define data structures by creating schemas that inherit from Pydantic's `BaseModel`.

```python
from typing import List
from pydantic import BaseModel, Field

class Student(BaseModel):
    id: int
    name: str = Field(None, title="Name of student", max_length=10)
    subjects: List[str] = []
```

*   **Data Coercion**: Pydantic is smart enough to handle safe conversions. If a client submits a string `"123"` to an `int` field, Pydantic automatically converts (coerces) it into the actual integer `123`.
*   **Validation Failure**: If a value cannot be safely coerced (e.g., passing alphabetic characters to an `int` field), Pydantic blocks the request and returns a detailed `ValidationError` response with `422 Unprocessable Entity` status.
*   **Advanced Restrictions**: Using Pydantic's `Field` class allows you to append custom metadata or enforce structural constraints (e.g., `max_length=10`).

---

## 🔗 6. Parameters: Path vs. Query vs. Request Body

When receiving parameters from a client, FastAPI distinguishes them based on how they are defined in your route path and signature:

```mermaid
graph TD
    Client[Client Request] -->|"GET /hello/Ravi/20"| Path["Path Parameter<br/>/hello/{name}/{age}"]
    Client -->|"GET /hello/Ravi?age=20"| Query["Query Parameter<br/>?age=20"]
    Client -->|"POST /students/ {JSON}"| Body["Request Body<br/>Student (Pydantic Model)"]
    
    classDef default fill:#1e293b,stroke:#475569,stroke-width:1px,color:#f8fafc;
    classDef path fill:#1e1b4b,stroke:#4f46e5,stroke-width:2px,color:#e0e7ff;
    classDef query fill:#064e3b,stroke:#059669,stroke-width:2px,color:#ecfdf5;
    classDef body fill:#7c2d12,stroke:#ea580c,stroke-width:2px,color:#fff7ed;
    
    class Path path;
    class Query query;
    class Body body;
```

### A. Path Parameters
These are dynamic variables embedded directly inside the URL path string, defined using curly brackets `{}` in the route decorator:

```python
@app.get("/hello/{name}/{age}")
async def hello(name: str, age: int):
    return {"name": name, "age": age}
```
*   **Example URL**: `http://localhost:8000/hello/Ravi/20`
*   **Validation**: If a user attempts to request `/hello/20/Ravi`, FastAPI returns a validation error because `"Ravi"` cannot be parsed into an `age: int`.

### B. Query Parameters
When you add function parameters that are *not* defined in the literal URL decorator path, FastAPI automatically reads them as a query string (key-value pairs trailing after the `?` character):

```python
@app.get("/hello/{name}")
async def hello(name: str, age: int):
    return {"name": name, "age": age}
```
*   **Example URL**: `http://localhost:8000/hello/Ravi?age=20`
*   *Here, `name` is parsed from the path, and `age` is extracted from the query parameters.*

### C. Request Body (POST Requests)
To accept complex, nested, or bulk data structures, send them inside the HTTP request body. You declare this in FastAPI simply by using your Pydantic schema as a parameter type hint:

```python
@app.post("/students/")
async def student_data(s1: Student):
    return s1
```
*   FastAPI intercepts the payload, parses the JSON body to verify compatibility with the `Student` model, and instantiates the Pydantic object to be used inside your function.

---

## 🛠️ 7. Parameter Validation Using Operators

You can enforce numeric and string constraints directly on path and query parameters by importing `Path` and `Query` from `fastapi`.

### Numeric Validation Comparison Operators
Use these short conditional abbreviations to restrict parameter bounds:
*   `gt`: Greater than
*   `ge`: Greater than or equal to
*   `lt`: Less than
*   `le`: Less than or equal to

### Code Combining Path, Query, and Request Body Restrictions
```python
from fastapi import FastAPI, Path, Query, Body

app = FastAPI()

@app.get("/hello/{name}/{age}")
async def hello(
    *, 
    name: str = Path(..., min_length=3, max_length=10), # Path parameter string length validation
    age: int = Path(..., ge=1, le=100),                # Path parameter numeric boundaries
    percent: float = Query(..., ge=0, le=100)          # Query parameter numeric boundaries
):
    return {"name": name, "age": age, "percent": percent}
```

> [!IMPORTANT]
> The ellipsis (`...`) used inside the validator (e.g. `Path(...)` or `Query(...)`) indicates that the parameter is **required** and cannot be omitted by the client. The prepended asterisk (`*`) in the function arguments tells Python that all subsequent arguments must be passed as keyword arguments.
