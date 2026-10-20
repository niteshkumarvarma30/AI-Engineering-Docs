# REST API Fundamentals: A Complete Beginner's Guide

> A GitHub-ready lesson covering REST, HTTP, endpoints, JSON, query parameters, HTTP methods, REST constraints, status codes, FastAPI, and practical Python examples.

## Table of Contents

1. [What Is an API?](#1-what-is-an-api)
2. [What Is REST?](#2-what-is-rest)
3. [REST API vs. RESTful API](#3-rest-api-vs-restful-api)
4. [HTTP Request–Response Cycle](#4-http-requestresponse-cycle)
5. [Parts of an HTTP Request and Response](#5-parts-of-an-http-request-and-response)
6. [URLs, Paths, Endpoints, and Resources](#6-urls-paths-endpoints-and-resources)
7. [Query Parameters](#7-query-parameters)
8. [The iTunes Search API Example](#8-the-itunes-search-api-example)
9. [HTTP Methods and CRUD](#9-http-methods-and-crud)
10. [The Six REST Architectural Constraints](#10-the-six-rest-architectural-constraints)
11. [HTTP Status Codes](#11-http-status-codes)
12. [What Is FastAPI?](#12-what-is-fastapi)
13. [REST API vs. FastAPI](#13-rest-api-vs-fastapi)
14. [Building a REST-Style API with FastAPI](#14-building-a-rest-style-api-with-fastapi)
15. [Calling an API from Python](#15-calling-an-api-from-python)
16. [Connecting REST APIs to an AI Chatbot](#16-connecting-rest-apis-to-an-ai-chatbot)
17. [Common Misconceptions](#17-common-misconceptions)
18. [Knowledge Check](#18-knowledge-check)
19. [Summary](#19-summary)

---

## 1. What Is an API?

**API** stands for **Application Programming Interface**. It defines how one software component can interact with another.

An API may let a program:

- Retrieve a user's profile.
- Create a chat session.
- Send a message to an AI model.
- Search for products.
- Update an order.
- Delete a record.

For example, a frontend application may need to retrieve a user's profile from a backend. Rather than accessing the backend's database directly, the frontend calls an API that exposes the required operation.

An API is an interface or contract. It is not necessarily a web API; APIs can also exist within libraries, operating systems, and other software.

## 2. What Is REST?

**REST** stands for **Representational State Transfer**. It is an architectural style for designing networked software systems. Roy Fielding described REST in his doctoral dissertation.

REST provides architectural constraints that help systems communicate through a consistent interface.

In everyday development, the term **REST API** usually refers to an HTTP-based API that exposes resources through URLs and commonly exchanges JSON data.

A typical interaction looks like this:

1. A client sends an HTTP request to a URL.
2. The request uses an HTTP method such as `GET`, `POST`, `PUT`, `PATCH`, or `DELETE`.
3. The server processes the request.
4. The server returns an HTTP response with a status code and, often, a response body.

The response body may contain JSON, text, or another representation. JSON is common, but REST does not require JSON or even HTTP as its communication protocol.

### Example

A client requests a product:

```http
GET /products/15 HTTP/1.1
Host: api.example.com
Accept: application/json
```

The server might respond:

```http
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{
  "id": 15,
  "name": "Keyboard",
  "price": 1499
}
```

The client can use this response without knowing how the server stores or retrieves the product internally.

## 3. REST API vs. RESTful API

These terms are often used interchangeably, but there is a technical distinction.

| Term | Meaning |
|---|---|
| API | An interface through which software components interact. |
| REST | An architectural style. |
| REST API | An API designed around REST principles; the term is often used loosely. |
| RESTful API | An API that satisfies REST's architectural constraints. |

Using HTTP, JSON, and familiar HTTP methods does not automatically make an API fully RESTful. In particular, REST's uniform-interface constraint includes hypermedia-driven interaction, which many APIs described as RESTful do not fully implement.

## 4. HTTP Request–Response Cycle

**HTTP** stands for **Hypertext Transfer Protocol**. It is the protocol commonly used by web APIs to exchange requests and responses.

The basic cycle is:

1. **Client sends a request.** For example, `GET /users/42`.
2. **Server processes the request.** It may validate inputs, apply business logic, and query a database.
3. **Server returns a response.** For example, status `200 OK` and a JSON representation of the user.
4. **Client handles the response.** It displays the data or handles an error.

### Example architecture

```text
Frontend (React or Streamlit)
          |
          | HTTP request
          v
Backend API (FastAPI, Flask, etc.)
          |
          | Application logic
          v
Database or AI model
          |
          | Result
          v
Backend API
          |
          | HTTP response
          v
Frontend
```

The frontend and backend can be written in different programming languages. They communicate through the API contract.

## 5. Parts of an HTTP Request and Response

### 5.1 HTTP request

Example:

```http
POST /users HTTP/1.1
Host: api.example.com
Content-Type: application/json
Accept: application/json

{
  "name": "Nitesh",
  "email": "nitesh@example.com"
}
```

Important parts:

1. **Method:** `POST` communicates the intended operation.
2. **Path:** `/users` identifies the target resource or collection.
3. **Headers:** Metadata about the request. `Content-Type` identifies the format of the request body.
4. **Body:** The data sent to the server, in this example a JSON object.

A request does not always have a body. A typical `GET` request, for example, does not require one.

### 5.2 HTTP response

Example:

```http
HTTP/1.1 201 Created
Content-Type: application/json

{
  "id": 42,
  "name": "Nitesh",
  "email": "nitesh@example.com"
}
```

Important parts:

- **Status code:** `201 Created` indicates that a resource was created.
- **Headers:** Metadata about the response.
- **Body:** The returned representation, here JSON.

A response body can be JSON, text, HTML, binary data, or empty. Some successful responses, such as `204 No Content`, have no response body.

### 5.3 What is JSON?

**JSON** stands for **JavaScript Object Notation**. It is a text format used to represent structured data.

Example:

```json
{
  "name": "Nitesh",
  "age": 20,
  "skills": ["Python", "FastAPI", "Machine Learning"],
  "is_student": true
}
```

JSON supports objects, arrays, strings, numbers, booleans, and `null`.

A JSON object is not exactly the same thing as a Python dictionary, although the two have a convenient correspondence. A client library typically parses JSON into language-specific data structures.

## 6. URLs, Paths, Endpoints, and Resources

Consider this URL:

```text
https://api.example.com/users/42
```

Its components are:

- **Scheme:** `https` — indicates secure HTTP transport.
- **Host:** `api.example.com` — the server's hostname.
- **Path:** `/users/42` — identifies the target within the API.
- **Resource:** The user represented by ID `42`.

An **endpoint** is commonly described as an API URL/path and HTTP method combination through which an operation is available.

For example:

```text
GET    /users/42
PATCH  /users/42
DELETE /users/42
```

These use the same path but express different intended operations.

### Resource-oriented naming

REST-style APIs commonly use nouns to represent resources:

```text
/users
/users/42
/products
/products/15
/orders/900
```

Prefer resource-oriented paths such as:

```text
GET /users/42
```

over action-oriented paths such as:

```text
GET /getUserById/42
```

The HTTP method normally communicates the operation, while the path identifies the resource.

This is a useful convention, not a rule that every real-world API follows perfectly.

## 7. Query Parameters

Query parameters pass additional information through a URL. They are commonly used for search, filtering, sorting, and pagination.

Example:

```text
/products?category=phones&limit=10
```

In this URL:

- `?` starts the query string.
- `category=phones` is one parameter.
- `&` separates parameters.
- `limit=10` is another parameter.

Another example:

```text
/search?term=radiohead&media=music&limit=3
```

This asks a search API to search for `radiohead`, restrict the media type to music, and return at most three results.

Query parameters should be URL-encoded when necessary. For example, a space may be represented as `%20` or `+`, depending on the encoding context.

### Path parameters vs. query parameters

| Type | Example | Typical purpose |
|---|---|---|
| Path parameter | `/users/42` | Identifies a specific resource. |
| Query parameter | `/users?role=admin` | Filters or modifies a collection request. |

The distinction is a design convention; API documentation defines the actual meaning of each parameter.

## 8. The iTunes Search API Example

The source lesson uses Apple's iTunes Search API to search for Radiohead songs.

Command:

```bash
curl -i 'https://itunes.apple.com/search?term=radiohead&media=music&limit=3'
```

### What does the command mean?

- `curl` is a command-line tool that can send HTTP requests.
- `-i` tells `curl` to include the HTTP response headers in its output.
- `/search` is the API path.
- `term=radiohead` is the search term.
- `media=music` specifies the media type.
- `limit=3` requests at most three results.

The request is a `GET` request because `curl` uses `GET` by default when no other method is specified and no request body is supplied.

### Example response structure

A shortened illustrative response could look like this:

```json
{
  "resultCount": 3,
  "results": [
    {
      "artistName": "Radiohead",
      "collectionName": "In Rainbows",
      "trackName": "15 Step",
      "trackPrice": 1.29
    },
    {
      "artistName": "Radiohead",
      "collectionName": "In Rainbows",
      "trackName": "Bodysnatchers",
      "trackPrice": 1.29
    },
    {
      "artistName": "Radiohead",
      "collectionName": "In Rainbows",
      "trackName": "Weird Fishes / Arpeggi",
      "trackPrice": 1.29
    }
  ]
}
```

This is an abbreviated example, not a guarantee of the current live response.

Notice the structure:

- `resultCount` is a number.
- `results` is an array.
- Each item in `results` is an object.
- Each object contains fields such as `artistName` and `trackName`.

A client can parse the response and display the song names without knowing how the service performs the search internally.

### What is `curl` useful for?

`curl` is useful for testing APIs without building a frontend. It can send requests, attach headers, submit JSON bodies, and display responses.

For example:

```bash
curl -i https://api.example.com/users/42
```

To submit JSON:

```bash
curl -X POST https://api.example.com/users \
  -H "Content-Type: application/json" \
  -d '{"name":"Nitesh","email":"nitesh@example.com"}'
```

The second command sends a `POST` request with a JSON body.

## 9. HTTP Methods and CRUD

HTTP methods communicate the intended operation. The most common methods in REST-style APIs are `GET`, `POST`, `PUT`, `PATCH`, and `DELETE`.

### 9.1 GET — retrieve a resource

```http
GET /users/42
```

Typically retrieves a representation of user `42`. A `GET` request is intended to be safe: it should not request a change to the server's application state.

### 9.2 POST — submit data or create a resource

```http
POST /users
Content-Type: application/json

{
  "name": "Nitesh",
  "email": "nitesh@example.com"
}
```

The server might create a user and return `201 Created`.

`POST` is not limited to creation. It can also submit data for processing, initiate an operation, or perform other actions defined by the API.

### 9.3 PUT — replace a resource

```http
PUT /users/42
Content-Type: application/json

{
  "name": "Nitesh Kumar",
  "email": "nitesh@example.com"
}
```

`PUT` generally requests creation or replacement of the target resource's state, according to the API's semantics. It is idempotent by definition.

### 9.4 PATCH — partially modify a resource

```http
PATCH /users/42
Content-Type: application/json

{
  "name": "Nitesh Kumar"
}
```

`PATCH` applies a partial modification. The exact format and semantics of the patch document depend on the API. Unlike `PUT`, `PATCH` is not guaranteed to be idempotent in every implementation.

### 9.5 DELETE — remove a resource

```http
DELETE /users/42
```

Requests removal of the target resource. The response might be `204 No Content` if the operation succeeds and no body is returned.

### 9.6 CRUD mapping

CRUD means **Create, Read, Update, Delete**.

| CRUD operation | Common HTTP method | Example endpoint |
|---|---|---|
| Create | POST | `POST /products` |
| Read collection | GET | `GET /products` |
| Read one resource | GET | `GET /products/15` |
| Replace | PUT | `PUT /products/15` |
| Partially update | PATCH | `PATCH /products/15` |
| Delete | DELETE | `DELETE /products/15` |

This mapping is common, but the precise behavior should always be defined in the API contract.

## 10. The Six REST Architectural Constraints

Roy Fielding's REST architectural style is defined by six constraints. A system must satisfy the required constraints to be strictly RESTful; **Code on Demand is optional**.

### 10.1 Client–Server

The client interface and server-side data or business logic are separated.

Example:

- React or Streamlit displays the user interface.
- FastAPI handles HTTP requests and application logic.
- A database stores persistent data.

This separation allows the client and server to evolve independently as long as their interface remains compatible.

### 10.2 Stateless

Each request must contain the information necessary for the server to understand and process it. The server must not rely on stored conversational session context from a previous request to interpret the current request.

For example, a request might include:

```http
Authorization: Bearer <access-token>
```

and a body containing a session identifier.

**Stateless does not mean that the application cannot use a database.** The server can store users, messages, orders, and other persistent data. The constraint concerns how requests are understood and processed, not a ban on persistent storage.

### 10.3 Cacheable

Responses should indicate whether they can be cached and under what conditions.

For example, a public product catalog might be cacheable for a short period. Sensitive user information generally requires stricter caching rules.

HTTP caching uses headers such as:

```http
Cache-Control: public, max-age=60
```

This example allows shared caching for up to 60 seconds, subject to applicable HTTP caching rules.

### 10.4 Uniform Interface

The interface follows consistent rules for identifying resources and exchanging representations.

Important aspects include:

- Resources are identified through resource identifiers.
- Clients manipulate resources through representations.
- Messages are self-descriptive.
- Hypermedia controls guide the client toward available transitions.

Many APIs use consistent resource paths and JSON representations but do not implement the full hypermedia aspect of REST's uniform interface.

### 10.5 Layered System

A client does not need to know every intermediary component involved in processing its request.

A request may pass through:

```text
Client
  |
  v
Load Balancer
  |
  v
API Gateway
  |
  v
Application Server
  |
  v
Database
```

The client communicates through the exposed interface rather than depending on the internal arrangement of all layers.

### 10.6 Code on Demand (Optional)

A server may send executable code to a client to extend its functionality.

For example, a server might deliver JavaScript that a browser executes.

This constraint is optional. A RESTful system does not have to use Code on Demand.

### Why the constraints matter

Together, these constraints encourage separation of concerns, consistent interfaces, scalability, cacheability, and independent evolution of client and server.

A JSON API that uses HTTP methods may follow several REST-inspired conventions without satisfying every REST constraint. It is important to distinguish common industry usage from the strict architectural definition.

## 11. HTTP Status Codes

The HTTP status code communicates the result of a request.

| Status code | Meaning | Typical example |
|---|---|---|
| `200 OK` | Request succeeded | Retrieving a user |
| `201 Created` | Resource created | Creating a product |
| `202 Accepted` | Request accepted for processing | Starting an asynchronous job |
| `204 No Content` | Request succeeded with no response body | Deleting a resource |
| `400 Bad Request` | Request is invalid or malformed | Invalid request syntax |
| `401 Unauthorized` | Authentication is missing or invalid | Missing or invalid token |
| `403 Forbidden` | Server refuses access | Insufficient permission |
| `404 Not Found` | Target resource not found | Unknown user ID |
| `409 Conflict` | Request conflicts with current resource state | Duplicate or conflicting record |
| `422 Unprocessable Content` | Request content cannot be processed as submitted | Validation failure in some frameworks |
| `429 Too Many Requests` | Rate limit exceeded | Too many requests in a short period |
| `500 Internal Server Error` | Unexpected server failure | Unhandled backend exception |
| `503 Service Unavailable` | Service temporarily unavailable | Overloaded or under-maintenance service |

The exact status code depends on the API's semantics and implementation.

### Example: a not-found response

Request:

```http
GET /users/9999
```

Response:

```http
HTTP/1.1 404 Not Found
Content-Type: application/json
```

```json
{
  "detail": "User not found"
}
```

The status code communicates the broad outcome, while the response body provides additional details.

Do not return `200 OK` for every request regardless of its outcome. Meaningful status codes help clients handle success and failure correctly.

## 12. What Is FastAPI?

**FastAPI is a Python web framework for building APIs.** It helps developers implement HTTP endpoints, validate request data, serialize responses, and generate interactive API documentation.

FastAPI commonly uses:

- Python type hints for parameter declarations.
- Pydantic models for request and response validation.
- ASGI servers such as Uvicorn to run applications.
- OpenAPI to describe API operations.
- Swagger UI or ReDoc to provide interactive documentation.

FastAPI is a tool for implementing an API. REST is an architectural style. FastAPI can be used to build a REST-style API, but using FastAPI does not automatically make an application fully RESTful.

## 13. REST API vs. FastAPI

| Aspect | REST API | FastAPI |
|---|---|---|
| What it is | An API designed using REST principles | A Python web framework |
| Category | Architectural style / API design | Software framework |
| Purpose | Defines how clients interact with resources | Helps implement HTTP endpoints |
| Programming language | Not tied to one language | Python |
| HTTP required? | REST itself does not mandate HTTP, though HTTP is common | Commonly used to build HTTP APIs |
| JSON required? | No | Frequently used, but not required for every response |
| Example | `GET /products/10` | A Python function registered with `@app.get(...)` |

### Other frameworks can implement APIs too

| Framework | Language | Common use |
|---|---|---|
| FastAPI | Python | APIs, AI backends, typed request validation |
| Flask | Python | Lightweight web applications and APIs |
| Django REST Framework | Python | APIs within Django projects |
| Express | JavaScript / TypeScript | Node.js APIs |
| Spring Boot | Java | Enterprise services and APIs |
| ASP.NET Core | C# | Web APIs and enterprise services |

These frameworks offer different capabilities and defaults. Whether an API is RESTful depends on its design and behavior, not only on the framework.

### The same endpoint in FastAPI and Flask

**FastAPI:**

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {
        "id": user_id,
        "name": "Nitesh"
    }
```

**Flask:**

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.get("/users/<int:user_id>")
def get_user(user_id):
    return jsonify({
        "id": user_id,
        "name": "Nitesh"
    })
```

Both expose an HTTP `GET` endpoint that returns JSON. The route syntax differs, and the frameworks provide different validation and documentation features.

## 14. Building a REST-Style API with FastAPI

This practical example implements a small product API with create, read, update, and delete operations.

### 14.1 Install dependencies

```bash
pip install fastapi uvicorn
```

### 14.2 Create `main.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Product API")

class Product(BaseModel):
    name: str
    price: float

# Temporary in-memory storage.
# Data disappears when the process restarts.
products = {}
next_id = 1


@app.get("/")
def home():
    return {"message": "Welcome to the Product API"}


@app.post("/products", status_code=201)
def create_product(product: Product):
    global next_id

    product_id = next_id
    next_id += 1

    saved_product = {
        "id": product_id,
        "name": product.name,
        "price": product.price,
    }

    products[product_id] = saved_product
    return saved_product


@app.get("/products")
def get_products():
    return list(products.values())


@app.get("/products/{product_id}")
def get_product(product_id: int):
    if product_id not in products:
        raise HTTPException(
            status_code=404,
            detail="Product not found",
        )

    return products[product_id]


@app.patch("/products/{product_id}")
def update_product(product_id: int, product: Product):
    if product_id not in products:
        raise HTTPException(
            status_code=404,
            detail="Product not found",
        )

    # Simplified example: replaces both fields.
    # A production PATCH endpoint would usually accept
    # only the fields that need to change.
    products[product_id]["name"] = product.name
    products[product_id]["price"] = product.price

    return products[product_id]


@app.delete("/products/{product_id}", status_code=204)
def delete_product(product_id: int):
    if product_id not in products:
        raise HTTPException(
            status_code=404,
            detail="Product not found",
        )

    del products[product_id]
    return None
```

### 14.3 Run the server

From the directory containing `main.py`, execute:

```bash
uvicorn main:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

The `--reload` option restarts the development server when source files change. It is intended for development, not production deployment.

### 14.4 Open the interactive documentation

With the server running, open:

```text
http://127.0.0.1:8000/docs
```

FastAPI generates interactive Swagger UI documentation from the application's routes and schemas. You can inspect endpoints, provide request data, and send test requests from the browser.

### 14.5 Test the API with `curl`

Create a product:

```bash
curl -X POST http://127.0.0.1:8000/products \
  -H "Content-Type: application/json" \
  -d '{"name":"Keyboard","price":1499}'
```

Example response:

```json
{
  "id": 1,
  "name": "Keyboard",
  "price": 1499.0
}
```

Retrieve all products:

```bash
curl http://127.0.0.1:8000/products
```

Retrieve product `1`:

```bash
curl http://127.0.0.1:8000/products/1
```

Update product `1` using the simplified `PATCH` endpoint:

```bash
curl -X PATCH http://127.0.0.1:8000/products/1 \
  -H "Content-Type: application/json" \
  -d '{"name":"Mechanical Keyboard","price":2499}'
```

Delete product `1`:

```bash
curl -i -X DELETE http://127.0.0.1:8000/products/1
```

The delete operation returns `204 No Content` when it succeeds.

**Limitations of this example:**

- Data is stored in a Python dictionary, not a database.
- Data is lost when the process restarts.
- The `PATCH` operation requires both fields rather than supporting a true partial update.
- Authentication, authorization, logging, and production-grade error handling are not included.

These simplifications keep the first implementation focused on HTTP endpoints and request-response behavior.

### 14.6 What happens inside FastAPI?

For the create-product request:

1. `curl` sends an HTTP `POST` request.
2. FastAPI routes the request to `create_product`.
3. Pydantic validates the incoming JSON against the `Product` model.
4. The function stores the product in memory.
5. The function returns a Python dictionary.
6. FastAPI serializes the dictionary into JSON and sends the HTTP response.

This is the request-response cycle implemented in Python.

## 15. Calling an API from Python

A server exposes endpoints; a client sends requests to them. Python's `requests` library is a common way to make HTTP calls.

Install it:

```bash
pip install requests
```

### 15.1 Send a POST request

Keep the FastAPI server running, then execute this code in another terminal or Python script:

```python
import requests

url = "http://127.0.0.1:8000/products"

response = requests.post(
    url,
    json={
        "name": "Keyboard",
        "price": 1499,
    },
    timeout=10,
)

print(response.status_code)
print(response.json())
```

Explanation:

- `requests.post()` sends an HTTP `POST` request.
- `json={...}` serializes the dictionary as JSON and sets the appropriate content-type header.
- `response.status_code` reads the HTTP status code.
- `response.json()` parses the JSON response into Python data structures.
- `timeout=10` limits how long the client waits for the request.

### 15.2 Send a GET request

```python
import requests

url = "http://127.0.0.1:8000/products"

response = requests.get(url, timeout=10)

if response.ok:
    products = response.json()
    print(products)
else:
    print("Request failed:", response.status_code)
    print(response.text)
```

A client should handle failures rather than assuming every request succeeds.

### Client and server are separate roles

The client and server can be implemented in different languages. A browser, Python script, mobile app, or another backend service can call the same API as long as it follows the API contract.

## 16. Connecting REST APIs to an AI Chatbot

Consider a chatbot with a Streamlit frontend, a FastAPI backend, an LLM, and a database.

A possible request:

```http
POST /sessions/123/messages
Content-Type: application/json
```

```json
{
  "message": "Explain transformers"
}
```

A possible response:

```json
{
  "session_id": "123",
  "reply": "Transformers are neural network architectures...",
  "status": "completed"
}
```

A backend might perform these steps:

1. Validate the request and authenticate the user.
2. Retrieve the relevant conversation history from a database.
3. Construct the prompt or model input.
4. Call the LLM.
5. Save the user message and generated response.
6. Return the result to the client.

The API does not dictate whether the backend uses Gemini, OpenAI, a queue, Redis, Supabase, or another service. Those are implementation choices behind the interface.

### Synchronous vs. asynchronous processing

AI requests may take several seconds or longer.

**Synchronous request-response:**

```text
Client -> POST /chat -> Backend calls LLM
Client <- JSON answer <- Backend
```

The client waits until the answer is ready or the request times out.

**Asynchronous job pattern:**

```text
Client -> POST /jobs -> Backend starts work
Client <- 202 Accepted + job ID
Client -> GET /jobs/abc123
Client <- Job status or completed result
```

This pattern is useful when processing takes a long time. Other options include server-sent events or WebSockets when streaming or bidirectional communication is required.

### REST and persistent chat history

A REST API can store persistent conversation history in a database. Statelessness does not prohibit saving data.

For example, a client can send a `session_id` with each request. The backend retrieves the relevant history from the database and processes the request using that information. The server does not have to rely on hidden conversational state held only in memory from the previous HTTP request.

## 17. Common Misconceptions

### Misconception 1: REST API and FastAPI are the same thing

**Correction:** REST is an architectural style; FastAPI is a Python framework for building APIs.

### Misconception 2: Every API must use JSON

**Correction:** JSON is common, but HTTP APIs can exchange other representations. REST itself does not require JSON.

### Misconception 3: REST requires HTTP

**Correction:** REST's architectural constraints do not specify HTTP as the required protocol. HTTP is the dominant protocol for APIs commonly called REST APIs.

### Misconception 4: Stateless means the server cannot store data

**Correction:** Persistent database storage is compatible with stateless request processing. The server must not rely on stored conversational session context to understand a request.

### Misconception 5: Every API using HTTP methods is fully RESTful

**Correction:** A strict REST implementation must satisfy all required constraints, including the uniform interface. Many practical APIs are REST-inspired without implementing every aspect.

### Misconception 6: GET, POST, PUT, and PATCH are interchangeable

**Correction:** HTTP methods have different semantics, including differences in safety and idempotence. Their correct use matters for predictable clients, caching, and retries.

### Misconception 7: A successful HTTP response always has a JSON body

**Correction:** A response can have no body, as with `204 No Content`, or use another representation format.

## 18. Knowledge Check

Try answering these questions before checking the answers.

1. What does REST stand for?
2. What is the difference between an API and a REST API?
3. What is the difference between a path parameter and a query parameter?
4. Which HTTP method is normally used to retrieve a resource?
5. What is the difference between `PUT` and `PATCH`?
6. What does HTTP status code `404` mean?
7. Does REST statelessness prohibit storing chat history in a database?
8. What is FastAPI, and how does it relate to REST APIs?
9. What does `requests.post(..., json=payload)` do in Python?
10. Why does using HTTP and JSON alone not prove that an API is fully RESTful?

<details>
<summary><strong>Show answers</strong></summary>

1. **Representational State Transfer.**
2. An API is an interface between software components; a REST API is designed around REST principles.
3. A path parameter typically identifies a resource, such as `/users/42`. A query parameter commonly filters or modifies a request, such as `/users?role=admin`.
4. `GET`.
5. `PUT` generally requests creation or replacement of the target resource state; `PATCH` applies a partial modification.
6. `404 Not Found`.
7. No. Persistent application data can be stored; statelessness concerns dependence on stored conversational session state when processing requests.
8. FastAPI is a Python framework used to implement web APIs, including REST-style APIs.
9. It sends an HTTP POST request and serializes the supplied Python object as a JSON request body.
10. Strict REST also requires the architectural constraints, including the uniform interface and its hypermedia-related requirements.

</details>

## 19. Summary

Remember these core concepts:

- **API:** An interface through which software components interact.
- **REST:** An architectural style for networked systems.
- **HTTP:** A common protocol for REST-style web APIs.
- **Endpoint:** An API entry point, commonly described by its HTTP method and path.
- **Resource:** The entity or collection represented by an API.
- **JSON:** A common data-interchange format.
- **HTTP methods:** Communicate intended operations such as retrieval, creation, modification, and deletion.
- **Status codes:** Communicate request outcomes.
- **FastAPI:** A Python framework for implementing APIs.
- **Client and server:** Communicate through a defined interface without needing to share the same programming language.

### Recommended learning order

1. Learn HTTP requests, responses, headers, methods, status codes, and JSON.
2. Study REST constraints and resource-oriented API design.
3. Learn safety, idempotence, authentication, and error handling.
4. Implement CRUD endpoints using FastAPI.
5. Connect the API to a database.
6. Add automated tests, security controls, logging, and deployment.

The central distinction is:

> **REST describes an architectural approach to API design. FastAPI is one tool you can use to implement an API following that approach.**
