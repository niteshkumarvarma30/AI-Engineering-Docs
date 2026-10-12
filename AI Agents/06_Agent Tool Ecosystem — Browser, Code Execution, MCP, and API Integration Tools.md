# Agent Tool Ecosystem: Browser, Code Execution, MCP, and API Integration Tools

> **Module**: Agentic AI & Multi-Agent Systems  
> **Topic**: Agent Tool Ecosystem  
> **Source**: [AIML Companion — Agent Tool Ecosystem](https://aimlcompanion.ai/module/aiAgents/agentTools)

---

## 🎯 In This Module

- **The 5 core tool categories**: Browser automation, code execution, file I/O, API integration, and database retrieval.
- **Connecting tools via MCP (Model Context Protocol)**: Using the universal adapter standard instead of custom glue code.
- **Sandboxed code execution (E2B & Docker)**: Isolating untrusted LLM-generated code from your host environment.
- **Browser automation with Playwright MCP**: Navigating JavaScript-rendered pages and trimming 50K-token HTML down to ~2K tokens.
- **Composing tool chains**: Building reliable multi-step pipelines where small tools exchange consistent JSON.
- **Secure API integration**: Auto-generating focused tools from OpenAPI specs and injecting secrets at the tool layer.

---

## 1. The Big Picture: How It Works, Start to Finish

A language model can only generate text; **tools are what enable an agent to take real-world action**.

Equipping an agent involves four sequential steps:
1. **Pick Categories**: Match the necessary tool categories to the agent's job.
2. **Connect via MCP**: Plug tools in using the Model Context Protocol as the universal adapter.
3. **Sandbox the Risky**: Run untrusted, agent-written code in isolated environments (E2B or Docker), never on your host machine.
4. **Compose Chains**: Ensure each tool does one focused job with consistent JSON inputs/outputs so the agent can orchestrate them cleanly.

```
+-------------------------------------------------------------------------------+
|                       EQUIPPING AN AGENT WITH TOOLS                           |
|                                                                               |
|   01. PICK CATEGORIES ──> 02. CONNECT VIA MCP ──> 03. SANDBOX RISKY ──> 04. COMPOSE   |
|   Browser / Code /        Universal adapter:      E2B or Docker;        Small tools,  |
|   File / API / DB         One standard for        Isolate untrusted     JSON in/out,  |
|   (Match agent's job)     every tool              agent code            orchestrate   |
+-------------------------------------------------------------------------------+
```

### Two Foundational Rules
* **MCP is the Universal Adapter**: The same open standard connects Playwright (browser), E2B (code execution), and custom API wrappers without writing bespoke glue code.
* **Many Small Tools Beat One Mega-Tool**: LLMs select correctly among 50 focused, well-described tools far more reliably than trying to drive a single complex tool with a `"method"` parameter.

---

## 2. Walkthrough: Equipping a Full Coding Agent

Consider equipping an autonomous coding assistant (like GitHub Copilot's coding agent or Claude Code) that must write features, run test suites, and visually verify that an application works.

```
Step 1: Pick Tool Categories for the Job
----------------------------------------
A coding agent needs 3 of the 5 categories:
- File I/O: To inspect the repository, read source files, and write code patches.
- Code Execution: To run test suites, linters, and build commands.
- Browser Automation: To navigate the running local app and verify UI components.

Step 2: Connect via MCP & Sandbox the Risky One
-----------------------------------------------
Each tool connects via Model Context Protocol (MCP).
Because the agent writes arbitrary Python and Shell scripts, direct host execution is unacceptable:
Agent Code ──> E2B Cloud Sandbox (~150ms boot, isolated CPU/RAM/Network)
Result: Even if the agent runs `rm -rf /` or malicious scripts, your host remains 100% untouched.

Step 3: Compose into a Working Chain
------------------------------------
Chain: Write Patch ──> Run Tests in Sandbox ──> Verify in Browser
1. File I/O modifies `app.py`.
2. E2B Sandbox executes `pytest test_app.py` -> All pass.
3. Playwright MCP visits `http://localhost:3000` -> Takes screenshot & verifies button renders.

Result: An autonomous agent that writes, executes, and verifies its own code!
```

---

## 3. 💡 Mental Model: The Craftsperson's Kit

```
+-------------------------------------------------------------------------------+
| Category            | Toolkit Analogy                                         |
+---------------------+---------------------------------------------------------+
| 🌐 Browser Tools     | Eyes: Seeing and navigating the web; reading live pages |
| 🤲 Code Execution    | Hands: Building, computing, and transforming data       |
| 📱 APIs              | Phone: Talking to external services, triggering actions|
| 📁 File I/O          | Filing Cabinet: Reading, modifying, and saving documents|
| 🗄️ Databases         | Reference Library: Querying structured & vector records |
+---------------------+---------------------------------------------------------+
| 🔌 MCP Standard      | Universal Power Strip: Plugs every tool into the agent  |
+---------------------+---------------------------------------------------------+
```

Just as a craftsperson needs specialized tools for different materials, an agent combines distinct capabilities to interact with external environments.

---

## 4. The 5 Tool Categories & Combinations by Agent Type

Most production agents combine **2 to 3 categories** tailored to their operational domain:

```
+-------------------------------------------------------------------------------+
| Agent Type          | Recommended Tool Set            | Primary Objective     |
+---------------------+---------------------------------+-----------------------+
| 🔬 Research Agent   | Browser + API + File I/O        | Web crawl, summarize  |
| 💻 Coding Agent     | Code Execution + File I/O + DB  | Edit repo, run tests  |
| 📊 Data Analyst     | Database + Code Execution + File| Query SQL, plot charts|
| 🎧 Support Agent    | API + Database                  | Look up orders, refund|
+---------------------+---------------------------------+-----------------------+
```

---

## 5. Code Execution Sandboxes: E2B vs. Docker

Never execute agent-generated code directly via `eval()` or host shell commands. Sandboxes provide strict multi-tenant boundary isolation.

```
+-------------------------------------------------------------------------------+
| Feature             | E2B Cloud Sandboxes            | Docker Local Sandboxes |
+---------------------+--------------------------------+------------------------+
| Boot Latency        | ~150ms – 500ms                 | 1s – 3s                |
| Setup Overhead      | Zero (Managed Cloud)           | High (Maintain daemon) |
| Pre-installed Tools | NumPy, Pandas, Matplotlib, ML  | Custom Dockerfile      |
| Rich Outputs        | Native charts, PNGs, logs      | Requires volume mount  |
| Best For            | Hosted SaaS, serverless agents | On-prem, private VPCs  |
| Pricing             | Cloud usage billing            | Self-hosted / Free     |
+---------------------+--------------------------------+------------------------+
```

> **Security Rule**: Sandboxes isolate CPU, memory, filesystem, and networking. Even if a prompt-injected LLM generates malicious payloads, the sandbox boundary stops unauthorized host access.

---

## 6. Browser Automation with Playwright MCP

Traditional HTTP scrapers fail on modern Single Page Applications (React, Vue) because they cannot evaluate JavaScript. **Playwright MCP** launches a headless Chromium instance to click, type, and read live web apps.

```
[Agent] ──(MCP Protocol)──> [Playwright MCP Server] ──> [Headless Chromium]
                                                              │
                                                              ▼
                                                   [JavaScript Rendering]
                                                   [Click / Type / Wait]
```

### The 50K-Token Problem & Output Optimization
A raw web page's HTML can easily exceed **50,000+ tokens**, overflowing the context window.
* **Optimization 1 (Text & Links)**: Extract page inner text and link dictionaries (~2,000 tokens).
* **Optimization 2 (Accessibility Tree)**: Query the browser's accessibility tree (`AXTree`) to preserve structural hierarchy (buttons, headings, inputs) without HTML boilerplate.
* **Optimization 3 (Visual Grounding)**: Capture viewport screenshots and feed them into vision-capable models (e.g. GPT-4o).

---

## 7. API Integration via MCP (OpenAPI to Tools)

The **Model Context Protocol (MCP)** provides an open, standardized JSON-RPC protocol for connecting LLMs to external tools, data sources, and system prompts.

```
[Frontier LLM / Agent]
         │ (Standard MCP Protocol)
         ▼
[MCP Client Host]
    ├── [Playwright MCP Server]   (Browser automation)
    ├── [E2B MCP Server]          (Code execution)
    ├── [PostgreSQL MCP Server]   (Database queries)
    └── [Custom Stripe/API MCP]   (Auto-generated from OpenAPI spec)
```

### Exposing 50 Endpoints: Small Tools vs. Mega-Tools
* **Anti-Pattern (One Mega-Tool)**: `call_api(method="GET|POST", endpoint="...", body={...})` $\rightarrow$ LLM constantly fails on HTTP methods and query parameters.
* **Best Practice (Auto-Generated Focused Tools)**: Parse your OpenAPI/Swagger specification to automatically generate 50 distinct, strongly typed tools (`get_user()`, `refund_charge()`, etc.). The LLM selects the exact matching endpoint with **>95% accuracy**.

### Credential Handling: Never Expose API Keys
Inject API keys at the tool server level using environment variables (`os.environ["STRIPE_API_KEY"]`). **Never paste API keys or bearer tokens into system prompts or tool descriptions.**

---

## 8. Composing Tool Chains: The Unix Philosophy for Agents

Each tool should adhere to the **Unix philosophy**: do one thing well with predictable inputs and outputs.

```
[Browser Tool]      ──(JSON Output)──> [Code Exec Tool]      ──(JSON Output)──> [File I/O Tool]
Scrapes stock table                    Computes 30-day moving                   Saves report
                                       averages with Pandas                     to disk
```

### Production Composition Best Practices
1. **Consistent JSON Contracts**: Use structured dictionaries for both inputs and outputs rather than raw text strings (**98% reliability**).
2. **Built-in Retries at Tool Level**: Network retries and exponential backoff belong inside the tool wrapper, not in the LLM's reasoning loop.
3. **Structured Error Dictionaries**: Always return `{ "error": "...", "suggestion": "..." }` so the agent can pivot.
4. **Audit Logging**: Log every input, output, and latency metric for debugging and tracing.

---

## 9. ⚡ 2-Minute Cheatsheet & Interview Guide

- **The 5 Categories**: Browser (Playwright), Code (E2B/Docker), File I/O, API (MCP), Database (SQL/Vectors).
- **The 4 Steps**: Pick categories $\rightarrow$ Connect via MCP $\rightarrow$ Sandbox risky code $\rightarrow$ Compose small tools.
- **The Sandbox Rule**: Non-negotiable. Agent-generated code is untrusted; always isolate CPU, RAM, and filesystem in E2B or Docker.
- **HTML Optimization**: Raw HTML is 50K+ tokens; trim down to clean text, links, and forms (~2K tokens).
- **MCP Standard**: Replaces bespoke glue code with a universal adapter for tools, prompts, and resources.

### Top Interview Questions & Model Answers
1. **If a website requires JavaScript execution and authentication, which tool category do you use?**  
   *Browser automation with Playwright MCP. Standard HTTP client libraries (`requests`, `urllib`) cannot evaluate client-side JavaScript or execute interactive UI actions.*
2. **What protects your infrastructure if an agent generates dangerous code like `rm -rf /`?**  
   *An isolated sandbox (such as E2B or ephemeral Docker containers). The sandbox provides hardware-level virtualization that prevents agent actions from escaping to the host machine.*
3. **How should you expose a backend with 50 REST API endpoints to an agent?**  
   *Auto-generate 50 small, focused, well-documented tools from the OpenAPI specification rather than exposing one complex generic tool with dynamic parameters.*
4. **Where should third-party API credentials be managed?**  
   *Injected securely at the tool executor level via environment variables. Credentials should never be passed into prompt contexts or tool parameter schemas.*

---

## 10. 💻 Code Examples

### Example 1: Browser Automation Tool with Playwright

```python
"""
browser_automation_tool.py
Playwright tool for navigating web pages, extracting text, links, and screenshots.
"""

from playwright.async_api import async_playwright
import asyncio
import json

async def browse_webpage(url: str, action: str = "extract") -> str:
    """
    Browse a webpage, execute JavaScript, and extract clean content.
    Args:
        url: The web URL to navigate to.
        action: 'extract' for text, 'links' for anchor tags, 'screenshot' for visual save.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.goto(url, timeout=15000)
            await page.wait_for_load_state("networkidle")

            if action == "extract":
                title = await page.title()
                # Extract clean text, slicing to 3,000 chars to avoid token blowup
                text = await page.inner_text("body")
                return f"Title: {title}\n\nContent:\n{text[:3000]}"

            elif action == "links":
                # Extract all valid links
                links = await page.eval_on_selector_all(
                    "a[href]",
                    "els => els.map(e => ({text: e.innerText.trim(), href: e.href})).filter(l => l.text)"
                )
                return json.dumps(links[:20], indent=2)

            elif action == "screenshot":
                await page.screenshot(path="page_preview.png")
                return "Screenshot captured and saved to 'page_preview.png'."

            return f"Error: Unknown action '{action}'. Use 'extract', 'links', or 'screenshot'."

        except Exception as e:
            return f"Browser error navigating to '{url}': {str(e)}"
        finally:
            await browser.close()

if __name__ == "__main__":
    result = asyncio.run(browse_webpage("https://example.com", action="extract"))
    print(result)
```

---

### Example 2: Sandboxed Code Execution with E2B

```python
"""
e2b_sandbox_execution.py
Safely executing agent-generated Python code in an isolated E2B cloud sandbox.
Requires: pip install e2b-code-interpreter
"""

from e2b_code_interpreter import Sandbox

def execute_python_sandboxed(code: str, timeout: int = 30) -> str:
    """
    Executes arbitrary Python code in a secure cloud sandbox.
    Boots in ~150ms and isolates the host system completely.
    """
    # Initialize ephemeral cloud sandbox
    sandbox = Sandbox()
    try:
        execution = sandbox.run_code(code, timeout=timeout)

        output_parts = []
        if execution.text:
            output_parts.append(f"Standard Output:\n{execution.text}")

        if execution.error:
            output_parts.append(f"Runtime Error:\n{execution.error.name}: {execution.error.value}")

        # Handle rich media (charts, graphs, images)
        for result in execution.results:
            if hasattr(result, "png") and result.png:
                output_parts.append("[Generated Matplotlib/Seaborn Chart: PNG]")
            elif hasattr(result, "text") and result.text:
                output_parts.append(result.text)

        return "\n\n".join(output_parts) if output_parts else "Code executed successfully (no stdout)."

    except Exception as e:
        return f"Sandbox execution failure: {str(e)}"
    finally:
        # Guarantee sandbox teardown to free resources
        sandbox.kill()

if __name__ == "__main__":
    sample_script = """
import pandas as pd
data = {'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'], 'Revenue': [15000, 22000, 18500, 31000]}
df = pd.DataFrame(data)
print(f"Total Annual Revenue: ${df['Revenue'].sum():,}")
print(f"Best Performing Quarter: {df.loc[df['Revenue'].idxmax(), 'Quarter']}")
"""
    result = execute_python_sandboxed(sample_script)
    print(result)
```
