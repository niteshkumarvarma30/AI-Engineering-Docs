# Lesson 20 — Quantization: 4-bit and 8-bit

> **Goal:** Understand the math and mechanics of reducing a neural network's precision. Learn why Quantization is the ultimate memory-saving technique and how `LLM.int8()` solves the outlier problem.

---

# 1. What Is Quantization?

**Quantization = Reducing numerical precision.**

The word **precision** is extremely important.

In computer science, numbers are stored in physical memory.

Memory is finite.

We cannot store infinite decimal places.

So we are taking a high-precision number.

And converting it into a lower-precision number.

For example, consider the mathematical constant Pi.

Suppose we have a highly precise representation.

```text
3.14159265359
```

This number takes up a lot of memory space.

We can approximate it.

```text
3.14
```

We lost some precision.

We lost the exactness of the original number.

But we saved a significant amount of space.

This is the core, fundamental idea of quantization.

We trade a tiny bit of accuracy.

For a massive amount of memory savings.

---

# 2. How Computers Store Numbers

Before we apply this to AI.

We must understand how computers store numbers.

At the lowest level, everything in a computer is a bit.

A bit is a binary digit.

It can be a `0`.

Or it can be a `1`.

```text
Bit
 ↓
0 or 1
```

Bits are grouped together to form larger structures.

A group of 8 bits is called a byte.

```text
1 Byte = 8 Bits
```

Visually, a byte looks like this in memory:

```text
[ 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 ]
```

When we train a neural network, we are learning parameters.

These parameters are usually called **weights**.

A weight is just a number.

By default, PyTorch initializes weights in a format called **FP32**.

---

# 3. What is FP32?

**FP32** stands for 32-bit floating point.

It is also known as single-precision floating point.

As the name suggests, it uses exactly 32 bits of memory.

```text
32 bits
```

Since 8 bits make a single byte:

$$
32 \text{ bits} / 8 = 4 \text{ bytes}
$$

This means every single parameter in a standard neural network requires 4 bytes of memory.

$$
\text{1 Parameter = 4 bytes}
$$

This is the default precision for deep learning.

It is highly accurate.

It can represent very large numbers.

And it can represent very small fractions.

But it comes at a massive cost.

Memory.

---

# 4. The Memory Bottleneck

Let's look at a modern Large Language Model (LLM).

Consider a relatively small LLM today.

A 7 Billion parameter model.

Like Llama-2-7B or Mistral-7B.

How much memory does this model need just to sit on a hard drive?

We must multiply the total number of parameters.

By the size of each parameter in memory.

$$
7,000,000,000 \times 4 \text{ bytes}
$$

Let's do the exact math.

This equals 28,000,000,000 bytes.

Which is exactly 28 Gigabytes (GB).

```text
  7,000,000,000 Parameters
x             4 Bytes
--------------------------
 28,000,000,000 Bytes
             28 GB
```

You would need 28 GB of VRAM (Video RAM).

Just to load the raw weights onto the GPU.

This does not include the memory needed for the forward pass.

It does not include the KV cache for text generation.

It does not include the optimizer states during fine-tuning.

You would need an expensive 32GB or 40GB GPU (like an A100).

Just to run a basic 7B model in FP32.

This is a massive bottleneck.

This prevents everyday developers from running LLMs locally.

---

# 5. The Solution: Lower Precision Formats

If we can store the weights using fewer bits.

We will save memory.

Let's look at the alternatives to FP32.

First, we have **FP16**.

And its cousin **BF16** (Bfloat16).

These are 16-bit formats.

They use exactly half the bits of FP32.

$$
16 \text{ bits} / 8 = 2 \text{ bytes}
$$

So, 1 parameter equals 2 bytes.

Let's calculate the memory for a 7B model in FP16.

$$
7,000,000,000 \times 2 \text{ bytes} = 14 \text{ GB}
$$

We just cut the memory requirements exactly in half.

Instead of 28 GB, we only need 14 GB.

This is why most LLMs are distributed in FP16 or BF16 by default.

But we can go further.

Much further.

---

# 6. Moving to Integers: INT8

Next, we have **INT8**.

This is an 8-bit integer format.

It uses exactly half the bits of FP16.

And one quarter the bits of FP32.

$$
8 \text{ bits} / 8 = 1 \text{ byte}
$$

So, 1 parameter equals exactly 1 byte.

Let's calculate the memory for a 7B model in INT8.

$$
7,000,000,000 \times 1 \text{ byte} = 7 \text{ GB}
$$

We cut the memory in half again.

A 7B model now fits in just 7 GB of VRAM.

This fits comfortably on a standard consumer gaming GPU.

---

# 7. Pushing the Limits: INT4

Finally, we have **INT4**.

This is a 4-bit integer format.

It uses exactly half the bits of INT8.

$$
4 \text{ bits} / 8 = 0.5 \text{ bytes}
$$

So, 1 parameter equals 0.5 bytes.

Or, to put it another way, we can fit 2 entire parameters into just 1 byte.

Let's calculate the memory for a 7B model in INT4.

$$
7,000,000,000 \times 0.5 \text{ bytes} = 3.5 \text{ GB}
$$

A 7B model now takes only 3.5 GB of VRAM.

You can run this on a normal laptop.

Or even a high-end smartphone.

This is the true power of quantization.

---

# 8. The Mathematical Challenge

But how do we actually perform this conversion?

It is not as simple as just chopping off decimals.

We need to convert a highly precise floating-point number.

Like `3.14159`.

Into an 8-bit integer.

An 8-bit integer is very constrained mathematically.

It only has 8 bits of space.

Which means it has exactly $2^8$ possible values.

$$
2^8 = 256 \text{ total values}
$$

In a signed integer format (which includes negative numbers).

These 256 values are perfectly split around zero.

So the minimum possible value is:

$$
-128
$$

And the maximum possible value is:

$$
127
$$

```text
INT8 Range
    ↓
[-128, 127]
```

How do we compress a massive matrix of continuous floats into this tiny, discrete box?

We use a mathematical technique called **Linear Quantization**.

It is also commonly known as **Min-Max Scaling**.

---

# 9. Step 1: Find the Scale Factor

Let's look at the math step by step.

Suppose we have a weights matrix.

Let's call this matrix $W$.

First, we must scan the entire matrix $W$.

We need to find the absolute largest value in the matrix.

This is $\max(W)$.

We also need to find the absolute smallest value in the matrix.

This is $\min(W)$.

Suppose we scan our matrix and find:

$$
\max(W) = 3.0
$$

And:

$$
\min(W) = -1.0
$$

Our floating-point range is $[-1.0, 3.0]$.

Our target integer range is $[-128, 127]$.

We need to mathematically map the float range to the integer range.

We do this using a scaling factor.

Let's call the scale factor $S$.

The formula for the scale factor is:

$$
S = \frac{\max(W) - \min(W)}{q_{\max} - q_{\min}}
$$

Where $q_{\max}$ is the maximum integer value in our format.

And $q_{\min}$ is the minimum integer value in our format.

So for INT8:

$$
q_{\max} = 127
$$

$$
q_{\min} = -128
$$

Let's plug our numbers into the formula:

$$
S = \frac{3.0 - (-1.0)}{127 - (-128)}
$$

Simplify the numerator:

$$
S = \frac{4.0}{127 - (-128)}
$$

Simplify the denominator:

$$
S = \frac{4.0}{255}
$$

Calculate the final decimal value:

$$
S \approx 0.01568
$$

This scale factor $S$ is incredibly important.

It tells us exactly how much one integer step represents in float space.

Every time we increase the integer by $1$, the floating-point value increases by $0.01568$.

---

# 10. Step 2: Calculate the Zero Point

Next, we need to calculate the Zero Point.

Let's call the Zero Point $Z$.

What exactly is the Zero Point?

It is the integer value that perfectly represents the float value `0.0`.

Why is this important?

In neural networks, the number zero is extremely common.

We use it for padding tokens.

We use it for ReLU activations.

We use it for dropout.

If zero is not represented accurately, errors will cascade through the entire network.

The formula for the Zero Point is:

$$
Z = \text{round}\left( q_{\max} - \frac{\max(W)}{S} \right)
$$

This formula mathematically shifts the range so that float `0.0` lands exactly on an integer.

Let's calculate it for our running example.

$$
Z = \text{round}\left( 127 - \frac{3.0}{0.01568} \right)
$$

Divide the maximum by the scale:

$$
Z = \text{round}(127 - 191.32)
$$

Subtract from the max integer:

$$
Z = \text{round}(-64.32)
$$

Round to the nearest whole integer:

$$
Z = -64
$$

So, in our quantized INT8 matrix, the integer `-64` perfectly represents the float `0.0`.

---

# 11. Step 3: Quantize the Weights

Now we have our Scale Factor ($S$).

And we have our Zero Point ($Z$).

We are finally ready to quantize.

We want to convert a floating-point weight $w$.

Into an INT8 integer weight $q$.

The formula applies the scale and the zero point:

$$
q = \text{round}\left( \frac{w}{S} + Z \right)
$$

Let's trace this operation visually.

```text
Step 1: Take Float Weight (w)
             ↓
Step 2: Divide by Scale (S)
             ↓
Step 3: Add Zero Point (Z)
             ↓
Step 4: Round to nearest whole number
             ↓
Step 5: Output INT8 Weight (q)
```

Let's do a concrete mathematical example.

Suppose we have a specific weight $w = 1.5$.

Let's plug it into the formula:

$$
q = \text{round}\left( \frac{1.5}{0.01568} - 64 \right)
$$

Perform the division:

$$
q = \text{round}(95.66 - 64)
$$

Perform the subtraction:

$$
q = \text{round}(31.66)
$$

Round to the nearest integer:

$$
q = 32
$$

So the float `1.5` is stored as the integer `32` in memory.

We repeat this exact calculation for every single weight in the matrix.

A 7B model has 7 billion weights.

This rounding operation runs 7 billion times during the quantization process.

Once it is completely finished, we can delete the FP32 weights.

And we are left with a tiny INT8 model.

---

# 12. Step 4: Dequantization During Inference

Our weights are now stored as 8-bit integers.

We have successfully saved massive amounts of physical memory.

But there is a catch.

Graphics Processing Units (GPUs) are highly optimized for floating-point matrix multiplication.

They are not as efficient or accurate at large-scale integer matrix multiplication.

So, what happens during the actual forward pass?

When the model receives an input token and needs to do math.

It must temporarily convert the integers back to floats.

This process is called **Dequantization**.

It happens on the fly, right before the matrix multiplication in the GPU cores.

The formula is the exact algebraic reverse of the quantization formula:

$$
w_{\text{approx}} = (q - Z) \times S
$$

Let's trace this operation visually.

```text
Step 1: Take INT8 Weight (q)
             ↓
Step 2: Subtract Zero Point (Z)
             ↓
Step 3: Multiply by Scale (S)
             ↓
Step 4: Output Approximate Float (w_approx)
```

Let's use our previous example.

We stored the integer $q = 32$.

Let's dequantize it back into a float.

$$
w_{\text{approx}} = (32 - (-64)) \times 0.01568
$$

Resolve the double negative:

$$
w_{\text{approx}} = (32 + 64) \times 0.01568
$$

Add the numbers:

$$
w_{\text{approx}} = 96 \times 0.01568
$$

Multiply by the scale:

$$
w_{\text{approx}} = 1.50528
$$

---

# 13. Quantization Error

Look closely at the final result we just calculated.

The original float weight before quantization was:

$$
w = 1.5
$$

The dequantized float weight after quantization is:

$$
w_{\text{approx}} = 1.50528
$$

They are not exactly the same.

$$
1.5 \neq 1.50528
$$

Why did this happen?

It happened specifically because of the `round()` function in Step 3.

When we rounded to the nearest integer, we threw away fractional information.

That fractional information is gone forever.

It cannot be algebraically recovered.

This mathematical difference between the original weight and the dequantized weight has a specific name.

It is called **Quantization Error**.

$$
\text{Error} = |w - w_{\text{approx}}|
$$

In our example:

$$
\text{Error} = |1.5 - 1.50528| = 0.00528
$$

If the quantization error is very small across the entire matrix.

The model will perform almost exactly the same as the original.

The user will never notice the difference in text quality.

But if the quantization error is extremely large.

The errors will compound layer after layer during the forward pass.

The model's internal logic will completely collapse.

And it will generate completely nonsensical text.

---

# 14. The LLM Outlier Problem

For many years, linear quantization worked perfectly.

Small models, like BERT (340M parameters) or ResNet.

Could be quantized to 8-bit with almost zero performance degradation.

The quantization error was negligible.

But then, Large Language Models (LLMs) arrived.

Researchers tried to apply the exact same 8-bit quantization techniques to these massive LLMs.

And the models completely broke.

They output pure garbage.

Why?

As LLMs get bigger.

Specifically, as they cross the 6 Billion parameter threshold.

A strange mathematical phenomenon emerges in the network structure.

This phenomenon is officially known as **Emergent Outliers**.

---

# 15. Understanding Emergent Outliers

To understand outliers, we must look at how weights are distributed in a neural network.

In a normal, well-behaved model.

The weights and activations form a neat bell curve around zero.

This is a standard Gaussian distribution.

Most values are very small.

They might range beautifully between $-1.0$ and $1.0$.

There are almost no extreme values.

But in massive LLMs, the network learns to rely heavily on a few specific feature activations.

These specific activations become massive in magnitude.

Instead of staying between $-1$ and $1$.

A few individual values jump to $50$, $60$, or even $100$.

These extreme, anomalous values are called **outliers**.

```text
Normal Distribution of Weights:
-0.5, 0.2, -0.8, 0.9, 0.1, -0.3, 0.4

Emergent Outlier:
65.0
```

---

# 16. Why Outliers Destroy Quantization

Why does a single outlier destroy the entire matrix?

Let's revisit our scale factor formula one more time.

$$
S = \frac{\max(W) - \min(W)}{255}
$$

Suppose 99.9% of our weights are perfectly bounded between $-1$ and $1$.

If we had absolutely no outliers.

$\max(W)$ would be exactly $1$.

$\min(W)$ would be exactly $-1$.

And our scale factor $S$ would be very small.

A small $S$ means we have very high precision.

The 256 integer values are packed tightly over a small, dense range.

But what if we have a single outlier of $65$?

Now, $\max(W)$ becomes $65$.

Our scale factor $S$ mathematically explodes.

$$
S = \frac{65 - (-1)}{255}
$$

$$
S = \frac{66}{255}
$$

$$
S = 0.258
$$

Now, watch what happens when we quantize a normal weight.

Let's take a normal, healthy weight like $w = 0.2$.

Using our new, massive scale factor:

$$
q = \text{round}\left( \frac{0.2}{0.258} \right)
$$

$$
q = \text{round}(0.77)
$$

$$
q = 1
$$

The normal weight got completely crushed to `1`.

In fact, almost all normal weights in the matrix will now quantize to just `0`, `1`, or `-1`.

All the nuance, detail, and fine-grained information is gone.

The entire 8-bit range of 256 values is wasted.

It is stretched entirely to accommodate that single outlier at $65$.

Let's visualize this destruction.

```text
Normal Range: [-1, 1]
Outlier: 65

Integer Mapping:
-128 ----------------------- 0 ----------------------- 127
 |                           |                          |
-65                        Normal                      +65
                           weights
                           crushed
                           here
```

When we dequantize this broken matrix during inference.

The model has lost all its intelligence.

It is functionally brain-dead.

---

# 17. The Solution: LLM.int8()

How do we solve this catastrophic mathematical failure?

We cannot just manually delete the outliers.

Researchers found that these outliers are critical to the LLM's reasoning performance.

If you clip them or delete them, the model severely degrades.

We need a way to quantize the model, without crushing the normal weights, while still preserving the outliers.

In 2022, researchers published a landmark paper called **LLM.int8()**.

This algorithm is the foundation of the popular Hugging Face `bitsandbytes` library.

It completely solves the outlier problem.

It introduces two major architectural breakthroughs.

1. Vector-wise Quantization.
2. Mixed Precision Decomposition.

Let's explore both of these in deep detail.

---

# 18. Breakthrough 1: Vector-wise Quantization

Previously, we found a single $\max(W)$ and $\min(W)$ for the entire matrix.

This is called **Tensor-wise Quantization**.

And it is exactly why one outlier ruins the whole tensor.

`LLM.int8()` completely changes this approach.

It does not find a single scale factor for the whole matrix.

Instead, it calculates a completely separate scale factor for every single row.

And every single column.

This is called **Vector-wise Quantization**.

Why is this brilliant?

If row 5 contains a massive outlier.

Only row 5's specific scale factor is affected.

Row 5 loses its precision.

But row 1, row 2, row 3, and row 4 are perfectly safe.

They keep their tight, highly precise scale factors.

```text
Matrix W

Row 1: [0.1, 0.2, 0.1] -> Normal Scale
Row 2: [0.5, 65., 0.2] -> Huge Scale (Outlier Isolated!)
Row 3: [0.2, 0.1, 0.3] -> Normal Scale
```

The mathematical damage is contained entirely to one row.

But this alone is not enough to completely fix the LLM.

Because row 2 still completely lost its precision.

---

# 19. Breakthrough 2: Mixed Precision Decomposition

The true genius of `LLM.int8()` is the second breakthrough.

**Mixed Precision Decomposition**.

The algorithm actively scans the matrix before doing any math.

It looks at every single value in the matrix.

It checks if the value is greater than a specific mathematical threshold.

The threshold used in the paper is usually `6.0`.

Any single value greater than `6.0` is officially flagged as an outlier.

```text
Is weight > 6.0?
       ↓
Yes -> Outlier
No  -> Normal
```

Once the scan is complete, it physically separates the matrix into two pieces.

It decomposes the matrix.

---

# 20. The 99% and the 1%

The algorithm takes the 99% of normal weights.

The ones that are safely below the `6.0` threshold.

And it puts them into an entirely new **INT8 matrix**.

Because there are absolutely no outliers in this new matrix.

They quantize perfectly.

There is zero scale factor collapse.

Then, the algorithm takes the remaining 1% of outlier weights.

And it puts them into a separate **FP16 matrix**.

It does not quantize them at all.

It leaves them in highly precise 16-bit floating point.

Because there are so few outliers (usually less than 1% of the total matrix).

Keeping them in FP16 barely takes up any extra memory footprint.

---

# 21. The Split Matrix Multiplication

Now, how does the model actually do matrix multiplication with two different matrices in two different formats?

It does it in two parallel hardware streams.

Stream 1 handles the normal weights.

The normal weights are multiplied using fast, 8-bit integer tensor cores.

Stream 2 handles the outliers.

The outlier weights are multiplied using highly precise, 16-bit floating point tensor cores.

Let's visualize the exact architecture of `LLM.int8()`:

```text
                Input Vector (X)
                       |
         ┌─────────────┴─────────────┐
         ↓                           ↓
      Stream 1                    Stream 2
 [99% Normal Weights]        [1% Outlier Weights]
         ↓                           ↓
  Quantize to INT8              Keep in FP16
         ↓                           ↓
   INT8 MatMul                  FP16 MatMul
         ↓                           ↓
    INT8 Result                 FP16 Result
         |                           |
         └─────────────┬─────────────┘
                       ↓
              Convert to FP16
                       ↓
               Add together
                       ↓
                 Final Output
```

After both parallel multiplications are finished.

The system converts the INT8 result back into FP16.

And simply adds the two results together.

---

# 22. A Complete Numerical Walkthrough

Let's solidify this entire process with a complete, step-by-step numerical example.

We will quantize a tiny 2x2 matrix.

From FP32 down to INT8.

Here is our starting FP32 matrix:

```text
W = [
  [ 2.5, -1.0],
  [ 0.0,  1.5]
]
```

Step 1 is to find the maximum and minimum values in the tensor.

We scan the matrix.

The largest value is `2.5`.

So, $\max(W) = 2.5$.

The smallest value is `-1.0`.

So, $\min(W) = -1.0$.

Step 2 is to calculate the Scale Factor ($S$).

The formula is:

$$
S = \frac{\max(W) - \min(W)}{127 - (-128)}
$$

We plug in our values.

$$
S = \frac{2.5 - (-1.0)}{255}
$$

$$
S = \frac{3.5}{255}
$$

$$
S \approx 0.0137
$$

Step 3 is to calculate the Zero Point ($Z$).

The formula is:

$$
Z = \text{round}\left( 127 - \frac{\max(W)}{S} \right)
$$

We plug in our values.

$$
Z = \text{round}\left( 127 - \frac{2.5}{0.0137} \right)
$$

$$
Z = \text{round}(127 - 182.48)
$$

$$
Z = \text{round}(-55.48)
$$

$$
Z = -55
$$

Step 4 is to quantize every single weight individually.

The formula is:

$$
q = \text{round}\left( \frac{w}{S} + Z \right)
$$

Let's quantize the first weight: `2.5`.

$$
q_{11} = \text{round}\left( \frac{2.5}{0.0137} - 55 \right)
$$

$$
q_{11} = \text{round}(182.48 - 55)
$$

$$
q_{11} = \text{round}(127.48)
$$

$$
q_{11} = 127
$$

Notice how the maximum float value mapped exactly to the maximum integer value.

Let's quantize the second weight: `-1.0`.

$$
q_{12} = \text{round}\left( \frac{-1.0}{0.0137} - 55 \right)
$$

$$
q_{12} = \text{round}(-72.99 - 55)
$$

$$
q_{12} = \text{round}(-127.99)
$$

$$
q_{12} = -128
$$

Notice how the minimum float value mapped exactly to the minimum integer value.

Let's quantize the third weight: `0.0`.

$$
q_{21} = \text{round}\left( \frac{0.0}{0.0137} - 55 \right)
$$

$$
q_{21} = \text{round}(0 - 55)
$$

$$
q_{21} = -55
$$

Notice how the float zero mapped exactly to our calculated Zero Point.

Let's quantize the final weight: `1.5`.

$$
q_{22} = \text{round}\left( \frac{1.5}{0.0137} - 55 \right)
$$

$$
q_{22} = \text{round}(109.48 - 55)
$$

$$
q_{22} = \text{round}(54.48)
$$

$$
q_{22} = 54
$$

We have finished quantizing.

Here is our final INT8 matrix in memory:

```text
W_int8 = [
  [ 127, -128],
  [ -55,   54]
]
```

We successfully compressed the matrix.

---

# 23. A Complete Dequantization Walkthrough

Now let's simulate the forward pass during inference.

The GPU needs to dequantize our INT8 matrix back to floats to perform math.

The formula is:

$$
w_{\text{approx}} = (q - Z) \times S
$$

Let's dequantize the first weight: `127`.

$$
w_{11} = (127 - (-55)) \times 0.0137
$$

$$
w_{11} = (127 + 55) \times 0.0137
$$

$$
w_{11} = 182 \times 0.0137
$$

$$
w_{11} = 2.4934
$$

Let's dequantize the second weight: `-128`.

$$
w_{12} = (-128 - (-55)) \times 0.0137
$$

$$
w_{12} = (-128 + 55) \times 0.0137
$$

$$
w_{12} = -73 \times 0.0137
$$

$$
w_{12} = -1.0001
$$

Let's dequantize the third weight: `-55`.

$$
w_{21} = (-55 - (-55)) \times 0.0137
$$

$$
w_{21} = 0 \times 0.0137
$$

$$
w_{21} = 0.0
$$

Let's dequantize the final weight: `54`.

$$
w_{22} = (54 - (-55)) \times 0.0137
$$

$$
w_{22} = (54 + 55) \times 0.0137
$$

$$
w_{22} = 109 \times 0.0137
$$

$$
w_{22} = 1.4933
$$

Here is our final, dequantized approximate matrix:

```text
W_approx = [
  [ 2.4934, -1.0001],
  [ 0.0000,  1.4933]
]
```

Let's compare it to the original FP32 matrix.

```text
Original:
[ 2.5, -1.0 ]
[ 0.0,  1.5 ]

Approximate:
[ 2.4934, -1.0001 ]
[ 0.0000,  1.4933 ]
```

The values are extremely close.

The quantization error is tiny.

This is exactly how linear quantization works under the hood.

And this is exactly how LLMs save memory.

---

# 24. The Cost of Quantization

We have seen how much memory quantization saves.

It is a massive, transformative reduction.

But it is not entirely free.

There is no free lunch in computer science.

What is the hidden cost?

The cost is computation time.

Or latency.

When we store weights in INT8.

We must dequantize them to FP16 during every single forward pass.

This continuous conversion process takes time.

The GPU must pause to do the arithmetic.

```text
Memory is Saved
      ↓
But Compute is Increased
      ↓
Slightly Slower Generation
```

This means a quantized model will often generate text slightly slower than an unquantized model.

Because it has to constantly unpack its weights before using them.

However, the trade-off is almost always worth it.

Because without quantization, the model would not even fit on the GPU in the first place.

---

# 25. Summary

Let's review everything we have learned in this lesson.

Quantization reduces numerical precision.

It maps large FP32 numbers into small INT8 boxes.

We use Min-Max scaling to mathematically find a Scale Factor.

We use the Zero Point to handle zero-padding safely.

This saves massive amounts of physical memory.

But it inherently causes Quantization Error.

In large LLMs, Emergent Outliers completely destroy this process.

A single outlier crushes the scale factor for the entire matrix.

`LLM.int8()` solves this elegantly with two breakthroughs.

Vector-wise quantization isolates scale factors to individual rows.

Mixed Precision Decomposition physically separates outliers into FP16.

This gives us the memory footprint of 8-bit and the accuracy of 16-bit.

You now understand the fundamental math of LLM memory compression.

In the next lesson, we will push this concept even further.

We will learn how to compress models down to 4-bit.

And we will explore the revolutionary technique known as QLoRA.
