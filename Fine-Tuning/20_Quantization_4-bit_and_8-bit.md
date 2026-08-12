# Lesson 20 — Quantization: 4-bit and 8-bit

> **Goal:** Understand the math and mechanics of reducing a neural network's precision. Learn why Quantization is the ultimate memory-saving technique and how `LLM.int8()` solves the outlier problem.

---

## 1. The Memory Bottleneck

A neural network is essentially a massive matrix of numbers (weights). 

By default, PyTorch initializes weights in **FP32** (32-bit floating point).
- 1 Parameter = 32 bits = 4 bytes.
- A 7 Billion parameter model = $7,000,000,000 \times 4$ bytes = **28 GB of VRAM**.

If we want to load a 7B model, we need an expensive 32GB GPU (like a V100) just to hold the weights—not even counting the memory needed for gradients and optimizer states during fine-tuning!

### The Solution: Lower Precision
If we can store the weights using fewer bits, we save memory.
- **FP16 / BF16 (16-bit)**: 2 bytes per parameter (14 GB for a 7B model).
- **INT8 (8-bit)**: 1 byte per parameter (7 GB for a 7B model).
- **INT4 (4-bit)**: 0.5 bytes per parameter (3.5 GB for a 7B model).

---

## 2. The Math of Quantization (Min-Max Scaling)

How do we convert a highly precise float (like `3.14159`) into an 8-bit integer (which can only hold values from `-128` to `127`)?

We use a technique called **Linear Quantization**.

### Step 1: Find the Scale Factor
Suppose our weights matrix $W$ has a maximum value of $3.0$ and a minimum of $-1.0$.

The range of an INT8 integer is $[-128, 127]$. 
We map the floating-point range to the integer range using a scaling factor $S$:

$$
S = \frac{\max(W) - \min(W)}{127 - (-128)}
$$

### Step 2: Calculate the Zero Point
The Zero Point ($Z$) ensures that the float `0.0` maps perfectly to an integer, which is crucial for operations like zero-padding.

$$
Z = \text{round}\left( 127 - \frac{\max(W)}{S} \right)
$$

### Step 3: Quantize the Weights
To convert a float weight $w$ to an INT8 weight $q$:

$$
q = \text{round}\left( \frac{w}{S} + Z \right)
$$

### Step 4: Dequantization (During Inference)
The GPU can't do matrix multiplication on integers as accurately, so during a forward pass, the model quickly converts $q$ back to a float:

$$
w_{\text{approx}} = (q - Z) \times S
$$

Because of the rounding step, $w_{\text{approx}} \neq w$. This difference is called **Quantization Error**.

---

## 3. The LLM Outlier Problem

Quantization works beautifully for small models (like BERT). However, as LLMs cross the 6B parameter mark, a strange mathematical phenomenon occurs: **Emergent Outliers**.

Certain feature activations in the model become massive (e.g., instead of floating between -1 and 1, a few values jump to 50 or 60).

### Why Outliers Destroy Quantization
If your weights are mostly between $[-1, 1]$, but one outlier is `60`, your $\max(W)$ becomes `60`. 
Your scale factor $S$ becomes huge. When you divide the normal weights by this huge $S$, they all get crushed to `0`. The model loses all its detail and outputs gibberish.

---

## 4. The Solution: LLM.int8()

To solve the outlier problem, researchers published **LLM.int8()** (which powers the popular Hugging Face `bitsandbytes` library).

Here is the exact architecture of `LLM.int8()`:

1. **Vector-wise Quantization**: Instead of finding a single scale factor $S$ for the entire matrix, it calculates a separate $S$ for every single row/column.
2. **Mixed Precision Decomposition**: 
   - It scans the matrix for outliers (any value $> 6.0$).
   - It separates the outliers into a small **FP16 matrix**.
   - It puts the remaining 99% of normal weights into an **INT8 matrix**.
   - It multiplies them separately and adds the results together.

```text
Weights Matrix (W)
      ↓
Split into two streams:
      ↓
[99% Normal Weights] ---> Quantized to INT8 ---> MatMul
      +
[ 1% Outlier Weights] --> Kept in FP16 ----> MatMul
      ↓
   Combine Outputs
```

This allows us to run 8-bit quantization on massive LLMs with **zero performance degradation**.
In the next lesson, we will push this even further to 4-bit (QLoRA).
