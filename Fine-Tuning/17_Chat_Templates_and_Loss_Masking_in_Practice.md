# Lesson 17 — Chat Templates & Loss Masking in Practice

> **Goal:** Understand the exact mechanism of formatting multi-turn conversational data for LLMs, and how Loss Masking prevents the model from being penalized for the user's instructions.

---

## 1. The Multi-Turn Conversation Problem

Base LLMs are trained to predict the next token on raw, unstructured text. However, when we perform **Supervised Fine-Tuning (SFT)** for a chat model (like Llama 3-Instruct or ChatGPT), the model must understand the difference between:
1. The System Prompt (Rules)
2. The User's Message (Instruction)
3. The AI's Response (Generation)

If we simply concatenate them:
```text
System: You are helpful.
User: What is 2+2?
AI: 4.
```
The model sees a single flat string. It doesn't inherently understand where the user stops speaking and the AI starts.

---

## 2. What Is a Chat Template?

A **Chat Template** is a specific string formatting standard embedded in the model's tokenizer. It uses special control tokens to explicitly demarcate conversational roles.

For example, **Llama 3** uses tokens like `<|start_header_id|>` and `<|eot_id|>` (End of Turn ID).

A raw conversation array:
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is LoRA?"},
    {"role": "assistant", "content": "LoRA is Low-Rank Adaptation."}
]
```

When passed through a **Chat Template**, it compiles down to exactly this string:
```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is LoRA?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

LoRA is Low-Rank Adaptation.<|eot_id|>
```

### Applying Chat Templates in Code
Hugging Face tokenizers have this built-in:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

formatted_string = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
```

---

## 3. What Is Loss Masking?

During SFT, the model takes the entire templated string as input and predicts the next token at every position.

Recall the Cross-Entropy loss from Lesson 16:

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t|x, y_{<t})
$$

### The Flaw Without Masking
If we calculate loss across all $T$ tokens in the sequence, the model is penalized for failing to predict the **System Prompt** and the **User's Instruction**.

But we *don't care* if the AI can predict the user's question! We only want the AI to learn how to predict the **Assistant's Response**.

### The Solution: The `-100` Ignore Index
In PyTorch, the `CrossEntropyLoss` function has a parameter called `ignore_index` (which defaults to `-100`).

We create a **Labels Array** that is identical to the **Input IDs**, but we replace every token belonging to the System and User with `-100`.

```text
Input IDs:  [<bos>, User:, What, is, LoRA?, <eos>, AI:, It, is, PEFT]
Labels:     [ -100,  -100, -100, -100, -100,  -100, -100, It, is, PEFT]
```

When calculating loss:

$$
\text{Loss}_{t} = 
\begin{cases} 
-\log P(y_t) & \text{if } y_t \neq -100 \\\\
0 & \text{if } y_t = -100 
\end{cases}
$$

### Why This Is Crucial
By masking out the user's prompt:
1. The loss drops significantly (as the model isn't struggling to predict unpredictable user inputs).
2. The gradients computed during backpropagation are derived **exclusively** from the AI's response generation quality.
3. The model avoids "hallucinating" user instructions during generation.

---

## 4. Implementing Loss Masking (DataCollatorForCompletionOnlyLM)

Manually finding the indices of the AI's response to apply `-100` is tedious. 
The **TRL** (Transformer Reinforcement Learning) library by Hugging Face provides a specialized tool for this.

```python
from trl import DataCollatorForCompletionOnlyLM

# Tell the collator exactly what string precedes the AI's response
response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer
)
```

During training, this collator intercepts your batches, scans for the `response_template`, and automatically applies the `-100` mask to everything before it.
