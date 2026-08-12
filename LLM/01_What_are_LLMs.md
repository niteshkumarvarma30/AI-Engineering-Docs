# 1. What are LLMs?

> **Goal:** Understand the fundamental definition, capabilities, and underlying mathematical objective of Large Language Models to prepare for AI Engineering interviews.

---

## 1. The Core Definition

At their absolute core, **Large Language Models (LLMs)** are massive statistical pattern-matching engines. 
They are neural networks—specifically based on the **Transformer** architecture—that are trained on vast amounts of text data.

From a mathematical perspective, an LLM is simply an **autoregressive next-token predictor**. 

Given a sequence of input tokens $x_1, x_2, ..., x_t$, the model's objective is to compute the probability distribution over the entire vocabulary for the next token $x_{t+1}$:

$$ P(x_{t+1} | x_1, x_2, ..., x_t) $$

---

## 2. What makes them "Large"?

The "Large" in LLM refers to three interacting pillars of scale, often referred to by the **Chinchilla Scaling Laws**:

1. **Parameters:** The number of trainable weights in the neural network (e.g., 8 Billion, 70 Billion, or 1 Trillion parameters). These parameters store the "knowledge" and reasoning pathways learned during training.
2. **Data:** The volume of text used during pre-training. Modern LLMs are trained on trillions of tokens (essentially the entire public internet, books, code repositories, and research papers).
3. **Compute:** The massive amount of GPU processing power (measured in FLOPs) required to update the parameters during the training phase.

---

## 3. The Capability Shift (Emergence)

Why do we care about LLMs now, when language models have existed for decades?

As models scale up in parameters and data, they stop just "predicting the next word" and begin exhibiting **emergent abilities**—skills they were not explicitly trained to do.

*   **Zero-Shot Learning:** The ability to perform a task (e.g., translating English to French) without seeing any examples in the prompt, simply because the model learned the underlying structure of language during pre-training.
*   **Few-Shot Learning:** The ability to adapt to a new task given only 2 or 3 examples in the context window.
*   **In-Context Learning:** The model's ability to "learn" temporarily during inference based on the prompt, without updating its actual parameters.

---

## 4. The Engineering Perspective

As an AI Engineer, you do not just chat with LLMs; you build systems around them. You interact with LLMs through three main paradigms:

1.  **Prompt Engineering / RAG (Retrieval-Augmented Generation):** Injecting context into the input prompt to ground the model's predictions in facts (low cost, high flexibility).
2.  **Fine-Tuning:** Modifying the model's weights using a smaller, task-specific dataset to change its behavior or format (e.g., Supervised Fine-Tuning or LoRA).
3.  **Inference Optimization:** Serving the model efficiently in production (managing GPU memory, batching requests, using quantization).

---

## 5. Typical Interview Questions

If you are interviewing for an AI Engineering role, expect questions like:

**Q: What is the primary objective function used to train a standard LLM?**
*Answer:* The primary objective is causal language modeling (next-token prediction) using a Cross-Entropy Loss function to maximize the likelihood of the correct next token in a sequence.

**Q: Explain the difference between updating a model's knowledge via Fine-Tuning vs RAG.**
*Answer:* Fine-tuning permanently updates the model's internal weights (parameters), which is good for teaching the model a specific format or tone. RAG (Retrieval-Augmented Generation) retrieves external data and injects it into the prompt at inference time, which is better for injecting rapidly changing factual knowledge without retraining.
