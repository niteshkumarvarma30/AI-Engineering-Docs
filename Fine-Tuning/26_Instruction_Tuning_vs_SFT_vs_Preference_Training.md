# Lesson 26 — Instruction Tuning vs SFT vs Preference Training

> **Goal:** Clarify the exact semantic differences between the three major paradigms of fine-tuning. These terms are often used interchangeably, but they represent distinct mathematical objectives and dataset structures.

---

## 1. Instruction Tuning

**Objective:** Teach a base model *how* to behave like an assistant.
Base models (like Llama 3 Base) are autocomplete engines. If you give them the prompt:
*"What is the capital of France?"*
They might autocomplete it with:
*"What is the capital of Germany? What is the capital of Italy?"*

**Instruction Tuning** uses datasets formatted strictly as Instructions and Outputs.
```text
Instruction: What is the capital of France?
Output: Paris.
```
This forces the model to understand the zero-shot paradigm: when a user asks a question, the model must answer it, not continue it.

---

## 2. Supervised Fine-Tuning (SFT)

**Objective:** Teach a model a specific domain, style, or format using exact ground truth.

SFT is the broader mathematical category that *includes* Instruction Tuning.
In SFT, you provide an exact `$x$` (input) and `$y$` (target), and use Cross-Entropy Loss to maximize the probability of `$y$`.

You use SFT when you have **exact, objective ground truth**.
- **Medical Fine-Tuning:** (Input: Patient symptoms -> Output: Exact diagnosis).
- **Code Generation:** (Input: Python spec -> Output: Exact Python function).
- **Style Transfer:** (Input: Normal text -> Output: Text rewritten in Shakespearean style).

In SFT, there is no ambiguity. The dataset contains the "perfect" answer, and the model must clone it.

---

## 3. Preference Training (Alignment / RLHF / DPO)

**Objective:** Teach a model what is *subjectively* good or bad based on human values.

You use Preference Training when there is **no single objective correct answer**, but there are definitely *wrong* or *undesirable* answers.
- **Helpfulness:** Answering clearly rather than rambling.
- **Harmlessness:** Refusing to generate hate speech or dangerous instructions.

In Preference Training, the dataset provides a Prompt, a Chosen Response, and a Rejected Response. The math (DPO or PPO) forces the model to increase the probability of the Chosen response *relative* to the Rejected response.

---

## 4. The Complete Pipeline

When a frontier AI lab releases a new model (like Llama-3-8B-Instruct), the pipeline looks like this:

1. **Pretraining**: 15 Trillion tokens of raw internet text $\rightarrow$ `Llama-3-8B-Base`
2. **Instruction Tuning (SFT)**: 100,000 high-quality prompt/response pairs $\rightarrow$ `Llama-3-8B-SFT`
3. **Preference Training (DPO)**: 50,000 Chosen/Rejected pairs $\rightarrow$ `Llama-3-8B-Instruct`

As an AI Engineer, if you are fine-tuning a model for a highly specific, deterministic task (like extracting JSON from a receipt), you only need **SFT**.
If you are building a generalized Chatbot for your company, you need both **SFT** and **DPO**.
