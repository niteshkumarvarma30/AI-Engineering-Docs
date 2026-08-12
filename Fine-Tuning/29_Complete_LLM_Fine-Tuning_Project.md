# Lesson 29 — Complete LLM Fine-Tuning Project

> **Goal:** Synthesize all 28 previous lessons into a single, production-ready, highly optimized AI engineering script.

---

# 1. The Architecture

We have spent the last 28 lessons exploring deep theoretical concepts.

Now, we are going to build a complete, end-to-end Python script.

This script will fine-tune a massive large language model from scratch.

Specifically, we will fine-tune the `Llama-3-8B-Instruct` model.

We will train it entirely on a custom conversational dataset.

We will utilize four major components to make this operation possible.

First, we will use **Unsloth** (which we comprehensively covered in Lesson 22).

Unsloth provides highly optimized Triton GPU kernels.

These kernels make the training process up to two times faster.

Second, we will use **QLoRA** (which we covered in Lessons 20 and 21).

QLoRA uses 4-bit NormalFloat quantization.

This technique is mathematically brilliant and highly efficient.

It allows us to fit an 8 billion parameter model into just 8GB of consumer VRAM.

Third, we will use **Chat Templates and Loss Masking** (from Lesson 17).

This ensures perfect conversational formatting for the model.

It also prevents the model from accidentally learning to predict the user's prompt.

Fourth, we will use the **SFTTrainer** (from Lesson 18).

The SFTTrainer will cleanly orchestrate the entire training pipeline.

Let us break down the script, literally line by line.

We will carefully examine every single concept, token, and line of code.

We will leave absolutely no stone unturned in this analysis.

---

# 2. The Imports

We begin our script by importing the necessary libraries.

```python
import torch
```

First, we import the `torch` module.

PyTorch is the fundamental tensor mathematics library.

It underlies all of our neural network and deep learning operations.

Without PyTorch, none of this AI engineering is possible.

```python
from datasets import load_dataset
```

Next, we import the `load_dataset` function.

This function comes directly from the Hugging Face `datasets` library.

It allows us to easily and quickly download datasets from the Hugging Face Hub.

It also intelligently handles aggressive local caching on your hard drive.

This means you only ever have to download the training data once.

```python
from unsloth import FastLanguageModel
```

Next, we import `FastLanguageModel` from the `unsloth` library.

This is a highly optimized, custom wrapper class.

It wraps around standard Hugging Face transformer models to supercharge them.

It rewrites the standard attention mechanisms at the hardware level.

It also rewrites the standard feed-forward mechanisms.

It replaces them with highly memory-efficient Triton kernels.

This is exactly where the massive speedups come from during training.

Conceptually, the wrapper replaces standard operations like this:

```text
Standard PyTorch:
[Memory] -> [Attention] -> [Memory] -> [MLP] -> [Memory]

Unsloth Triton Kernel:
[Memory] -> [Fused Attention + MLP] -> [Memory]
```

This reduces slow memory read/write operations.

```python
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
```

Next, we import `SFTTrainer` from the `trl` library.

The acronym `trl` stands for Transformer Reinforcement Learning.

This library is designed specifically for fine-tuning and aligning large language models.

We also import a class called `DataCollatorForCompletionOnlyLM`.

This is admittedly a very long and confusing name.

However, it is a very important tool for our pipeline.

It will help us perform a critical operation called loss masking.

Loss masking ensures we only ever calculate gradients for the assistant's response.

We will discuss this mathematical process in extreme detail later.

```python
from transformers import TrainingArguments
```

Finally, we import `TrainingArguments` from the `transformers` library.

This class acts as a central configuration container.

It holds all of our mathematical hyperparameters in one place.

---

# 3. Model Configuration

We need to define the basic shape of our training pipeline.

```python
# ==========================================
# 1. Configuration & Unsloth Initialization
# ==========================================
max_seq_length = 2048 
```

We explicitly set `max_seq_length` to exactly 2048.

This variable represents the maximum context window of the model during training.

It defines exactly how many tokens the model can "see" at one single time.

If a conversation in our dataset is longer than 2048 tokens, it will be forcefully truncated.

Truncation simply means the end of the sequence is chopped off and discarded.

A sequence length of 2048 is a standard, healthy default for most instruction tuning tasks.

It balances context size with memory consumption.

```python
model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
```

Next, we define the `model_name` as a string.

Notice the specific suffix at the very end of the string.

The suffix is `-bnb-4bit`.

This acronym stands for `bitsandbytes` 4-bit quantization.

We are downloading a model that has *already* been quantized by the community.

The weights on the Hugging Face Hub are already stored in this compressed 4-bit format.

This saves massive amounts of download time.

It also saves significant internet bandwidth.

---

# 4. Unsloth Initialization

Now we must physically initialize the model and the tokenizer.

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
```

We call the `from_pretrained` method on the `FastLanguageModel` class.

This method does all of the heavy lifting for us.

It downloads the billions of weights.

It downloads the tokenizer vocabulary mapping.

It instantiates the PyTorch modules in memory.

We pass in our `model_name` string.

We pass in our `max_seq_length` integer.

```python
    dtype = None,           # Auto-detects BF16
```

We set the `dtype` argument to `None`.

When `dtype` is set to `None`, Unsloth performs intelligent hardware auto-detection.

It queries your specific GPU architecture using CUDA.

If you have a modern Ampere GPU (like an RTX 3090, 4090, or A100), it will automatically use Bfloat16.

Bfloat16 (BF16) is a specialized 16-bit floating-point format.

It was designed originally by Google specifically for deep learning workloads.

It has the exact same dynamic range exponent as standard 32-bit floats.

This explicitly prevents gradients from underflowing (becoming absolute zero) during training.

If your GPU is older, it will safely fall back to standard Float16 (FP16).

```python
    load_in_4bit = True,    # Uses QLoRA (NF4 + Double Quantization)
)
```

Finally, we set `load_in_4bit` to `True`.

This is the master toggle switch that activates the QLoRA pipeline.

### How QLoRA Works Internally

When we set `load_in_4bit = True`, several magical things happen under the hood.

First, it loads the model weights directly into the 4-bit NormalFloat (NF4) data type.

NF4 is an information-theoretically optimal data type.

It is specifically designed for neural weights that follow a zero-centered normal distribution.

Empirically, most neural network weights follow this exact statistical distribution.

Second, it applies a technique called Double Quantization.

Double Quantization takes the quantization constants from the first quantization step.

It then actively quantizes those constants again.

This mathematical trick saves an additional 0.37 bits per parameter.

Conceptually, the memory footprint transforms dramatically in VRAM.

Consider the original, unquantized state of the model:

```text
32-bit Float Weights in VRAM:

[ 0.1234, -0.9876, 1.2345, 0.4455, -0.1111, 0.9999 ... ]

(This requires roughly 32 Gigabytes of VRAM for 8 Billion parameters)
```

Now consider the post-QLoRA state:

```text
4-bit NF4 Weights in VRAM:

[ 3, -7, 5, 1, -2, 7 ... ]

(This requires roughly 5 Gigabytes of VRAM for 8 Billion parameters)
```

This massive mathematical reduction is the only reason we can train this model.

Without QLoRA, this script would crash immediately with a CUDA Out Of Memory (OOM) error on consumer hardware.

---

# 5. Injecting LoRA Adapters

The 4-bit weights we just loaded into VRAM are completely frozen.

We absolutely cannot update them during the backpropagation step.

They are stored in a mathematically un-trainable format.

Therefore, we must inject Low-Rank Adapters (LoRA) into the transformer architecture.

```python
# ==========================================
# 2. Inject LoRA Adapters
# ==========================================
model = FastLanguageModel.get_peft_model(
    model,
```

We call the `get_peft_model` function provided by Unsloth.

The acronym PEFT stands for Parameter-Efficient Fine-Tuning.

This function wraps our massive base model in a new class.

It dynamically injects tiny, trainable matrices into specific linear layers.

Let us examine each argument of this function carefully.

### The Rank Parameter (r)

```python
    r = 16, 
```

We set `r = 16`.

This is the rank of the LoRA mathematical matrices.

Instead of updating a massive weight matrix `W`, we freeze `W` entirely.

We then instantiate and train two much smaller matrices, called `A` and `B`.

The forward pass update is computed algebraically as:

$$
\Delta W = A \times B
$$

If the original matrix `W` has dimensions `4096 \times 4096`, it contains exactly 16,777,216 parameters.

If we set `r = 16`, matrix `A` has dimensions `4096 \times 16`.

Matrix `B` has dimensions `16 \times 4096`.

Together, matrices `A` and `B` contain only `131,072` trainable parameters.

This represents a 99.2% reduction in trainable parameters for that specific layer.

This is the entire essence of LoRA.

Visually, the matrix multiplication looks like this:

```text
    Matrix A         Matrix B           Delta W
  (4096 x 16)      (16 x 4096)       (4096 x 4096)
 
 [ . . . . ]      [ . . . . . ]     [ . . . . . . ]
 [ . . . . ]  X   [ . . . . . ]  =  [ . . . . . . ]
 [ . . . . ]      [ . . . . . ]     [ . . . . . . ]
 [ . . . . ]                        [ . . . . . . ]
```

The tiny matrices `A` and `B` multiply to form a sparse update matrix of the original size.

### Target Modules

Next, we must specify the `target_modules`.

```python
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
```

These are the specific linear layers in the transformer architecture where we will attach our adapters.

The `q_proj`, `k_proj`, `v_proj`, and `o_proj` modules are the core of the Self-Attention mechanism.

They create the Query, Key, Value, and Output representation vectors for every token.

The `gate_proj`, `up_proj`, and `down_proj` modules are the core of the Feed-Forward Multi-Layer Perceptron (MLP).

In early LoRA research, scientists only targeted the attention modules to save time.

However, modern empirical research shows that targeting all linear layers produces vastly superior results.

By targeting all of these modules, we give the model maximum mathematical flexibility to learn new concepts.

### LoRA Alpha

```python
    lora_alpha = 16,
```

We set `lora_alpha = 16`.

Alpha is a scaling factor used during the forward pass.

It controls how strongly the LoRA adapter output influences the original base model output.

The final scaling factor applied to the adapter activations is computed as:

$$
\text{Scaling Factor} = \frac{\alpha}{r}
$$

Because our `r = 16` and our `lora_alpha = 16`, our scaling factor evaluates to exactly `1.0`.

This means the adapter output is added to the base model output at a 1-to-1 ratio.

This is a very safe and very standard default value for most projects.

### Dropout and Bias

```python
    lora_dropout = 0, 
    bias = "none",    
```

We set `lora_dropout = 0`.

Dropout is a regularization technique that randomly zeros out activations to prevent neural overfitting.

However, Unsloth is designed and optimized to run fastest when dropout is completely disabled (set to zero).

We also set `bias = "none"`.

This means we do not train any bias vectors in the entire model.

We only train the LoRA weight matrices `A` and `B`.

This saves memory and compute time without noticeably degrading performance.

### Gradient Checkpointing

```python
    use_gradient_checkpointing = "unsloth", # 50% VRAM saving
)
```

Finally, we set `use_gradient_checkpointing = "unsloth"`.

This is arguably the most important memory optimization in this entire block of code.

During standard backpropagation, PyTorch stores all intermediate forward-pass activations in VRAM.

It desperately needs these activations later to compute the gradients via the chain rule.

For a large language model, storing these activations requires massive amounts of VRAM.

Gradient checkpointing deletes most of these activations to save physical space.

When they are needed for the backward pass, it simply re-computes them on the fly.

Conceptually, standard training memory usage looks like this:

```text
Without Checkpointing (High VRAM Usage):

Forward Pass:   Layer 1 -> Layer 2 -> Layer 3 -> Layer 4 -> Loss
Memory Stores: [Act 1]    [Act 2]    [Act 3]    [Act 4]
```

Gradient checkpointing memory usage looks like this:

```text
With Checkpointing (Low VRAM Usage):

Forward Pass:   Layer 1 -> (recompute) -> Layer 3 -> (recompute) -> Loss
Memory Stores: [Act 1]                   [Act 3]
```

Unsloth provides a custom, highly optimized version of this exact algorithm.

It saves up to 50% of your total VRAM.

And because it is written in highly optimized Triton, it does not sacrifice training speed.

---

# 6. Data Preparation

Our model architecture is now fully initialized.

It is heavily optimized and physically ready to learn.

We must now prepare the training data.

```python
# ==========================================
# 3. Data Preparation & Formatting
# ==========================================
dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train")
```

We load the `dolly-15k-oai-style` dataset.

We request specifically the `train` split of the data.

This dataset contains 15,000 high-quality instructions and responses.

It is formatted in the highly popular OpenAI (oai) style.

The OpenAI style represents conversations as a standard list of JSON dictionaries.

Each dictionary has a `role` key and a `content` key.

For example, a single training row in this dataset looks exactly like this:

```text
[
  {
    "role": "user", 
    "content": "Explain the theory of relativity."
  },
  {
    "role": "assistant", 
    "content": "The theory of relativity, developed by Albert Einstein..."
  }
]
```

However, Llama-3 does not understand this JSON format natively.

A language model does not understand JSON brackets, indentation, or dictionary keys.

Llama-3 only understands integer token IDs.

We must convert this abstract list of dictionaries into a single, continuous text string.

This string must aggressively use Llama-3's specific, highly structured control tokens.

We achieve this conversion using a Chat Template.

---

# 7. The Chat Template

We define a Python mapping function to handle the conversion.

```python
def format_chat_template(example):
    # Applies Llama 3's exact chat template `<|start_header_id|>`
    example["text"] = tokenizer.apply_chat_template(
        example["messages"], 
        tokenize=False
    )
    return example
```

We call the `tokenizer.apply_chat_template` method inside the function.

This method takes the abstract list of JSON messages.

It strictly injects the exact tokens the Llama-3 model was trained on during its alignment phase.

For Llama-3, these special control tokens look exactly like this:

```text
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Explain the theory of relativity.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The theory of relativity, developed by Albert Einstein...<|eot_id|>
```

The `<|start_header_id|>` and `<|end_header_id|>` tokens are completely unique to the Llama-3 architecture.

They explicitly tell the neural network exactly which persona is currently speaking.

The `<|eot_id|>` token represents the "End Of Turn" signal.

By setting `tokenize=False`, we command the method to return a raw Python string.

We do not want it to return the integer IDs just yet.

Returning a raw string makes manual debugging significantly easier.

It also integrates cleanly with the `SFTTrainer` class later on in the script.

Next, we apply this mapping function to our entire dataset.

```python
formatted_dataset = dataset.map(format_chat_template)
```

The `.map()` function is highly efficient and parallelized.

It runs our formatter over all 15,000 examples in the dataset sequentially.

It creates a brand new column in our dataset called `"text"`.

This column contains the final, perfectly formatted conversational strings.

---

# 8. Loss Masking

Now we encounter one of the most critical conceptual leaps in instruction tuning.

We must explicitly define our Data Collator.

```python
# ==========================================
# 4. Loss Masking (Data Collator)
# ==========================================
# We only want to train the model on its own responses, not the user prompts.
response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
```

First, we define a static string variable called `response_template`.

This string is the exact sequence of characters that immediately precedes the assistant's reply.

```python
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template, 
    tokenizer=tokenizer
)
```

We instantiate the `DataCollatorForCompletionOnlyLM` object.

What does a Data Collator actually do during the training loop?

A Data Collator takes a batch of raw examples and prepares them as padded PyTorch tensors for the model.

In a standard, base-model language modeling task, the model tries to mathematically predict every single token in the sequence.

If we blindly did that here, the model would actively learn to predict the user's prompt.

We absolutely do not want the model to generate user prompts.

If it learns to predict user prompts, it might start talking to itself during inference.

We only want it to learn how to generate high-quality assistant responses.

Therefore, we must mathematically mask the loss for all of the user tokens.

This is exactly what the `DataCollatorForCompletionOnlyLM` does for us automatically.

It takes the `response_template` we provided.

It searches through the tokenized input sequence for this exact pattern of tokens.

Everything that appears *before* this string is aggressively assigned a label of `-100`.

In PyTorch, a label of `-100` is a highly specific special constant.

It explicitly commands the Cross-Entropy loss function to completely ignore that token.

Conceptually, the labels array ends up looking exactly like this:

```text
Tokens in Sequence:
[<|user|>] [What] [is] [gravity?] [<|assistant|>] [Gravity] [is] [a] [force.]

Corresponding Labels Array:
[  -100  ] [-100] [-100] [ -100 ] [    -100     ] [Gravity] [is] [a] [force.]
```

For the first five tokens in this example, the loss evaluates to zero.

No gradients are computed for those positions.

No weights are updated based on those positions.

The model is only mathematically penalized if it predicts the assistant's response incorrectly.

This guarantees that gradients are exclusively computed for the assistant's behavior.

This is the secret sauce to creating high-quality, obedient, and conversational agents.

---

# 9. Initializing the SFTTrainer

We are finally ready to assemble the core PyTorch training pipeline.

```python
# ==========================================
# 5. SFTTrainer Execution
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
```

We instantiate the `SFTTrainer` class.

We pass in our Unsloth-optimized `model` object.

We pass in our `tokenizer` object.

```python
    train_dataset = formatted_dataset,
    dataset_text_field = "text",
```

We pass in our `formatted_dataset`.

We must explicitly tell the trainer which dataset column contains our actual text data.

We set `dataset_text_field` to the string `"text"`.

This exactly matches the column we created earlier during the `.map()` operation.

```python
    max_seq_length = max_seq_length,
    data_collator = collator,
```

We pass in our `max_seq_length` integer of 2048.

We pass in our custom loss-masking `collator` object.

```python
    dataset_num_proc = 2,
    packing = False, 
```

We set `dataset_num_proc = 2`.

This tells the trainer to use two separate CPU cores for background data tokenization.

This speeds up the data loading bottleneck significantly.

We set `packing = False`.

Packing is an advanced technique that tightly combines multiple short conversations into a single 2048-token sequence.

It separates them internally with an End-Of-Sequence token.

While packing dramatically increases raw training speed, it can sometimes degrade chat performance.

It can cause the model's attention mechanism to accidentally mix contexts between different conversations.

For maximum conversational quality and safety, we leave packing firmly disabled.

---

# 10. Training Arguments

Inside the trainer initialization, we must define the `TrainingArguments`.

These are the strict mathematical hyperparameters that control the optimization process.

```python
    args = TrainingArguments(
        output_dir = "unsloth-llama-3-finetuned",
```

We set the `output_dir`.

This is the folder path on your hard drive where checkpoint files will be periodically saved.

```python
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Effective batch size = 8
```

We set `per_device_train_batch_size = 2`.

This explicitly means the GPU will only process 2 conversations simultaneously in a single forward pass.

Why is this number so disappointingly small?

Because an 8-Billion parameter model requires a vast amount of physical memory, even when compressed to 4-bit.

If we blindly tried a batch size of 8 or 16, we would immediately run out of VRAM.

A batch size of 2 safely and reliably prevents Out-Of-Memory (OOM) errors.

However, from a mathematical perspective, a batch size of 2 is terrible for convergence.

The gradient vectors generated by 2 examples will be extremely noisy and erratic.

The model will struggle to learn a stable, smooth path down the loss landscape.

To fix this mathematical problem, we use `gradient_accumulation_steps = 4`.

This setting tells PyTorch to run the forward and backward pass 4 separate times.

It accumulates (adds up) the resulting gradients from each pass in memory.

It only takes a single optimizer step after 4 passes have been fully completed.

Mathematically, this simulates a much larger batch size:

$$
\text{Effective Batch Size} = \text{Batch Size} \times \text{Accumulation Steps}
$$

$$
\text{Effective Batch Size} = 2 \times 4 = 8
$$

This simulates a batch size of 8, which stabilizes the training mathematics perfectly.

Visually, the accumulation process looks like this:

```text
Step 1: Forward Pass (Batch 2) -> Backward Pass -> Accumulate Gradients (Sum = G1)
Step 2: Forward Pass (Batch 2) -> Backward Pass -> Accumulate Gradients (Sum = G1 + G2)
Step 3: Forward Pass (Batch 2) -> Backward Pass -> Accumulate Gradients (Sum = G1 + G2 + G3)
Step 4: Forward Pass (Batch 2) -> Backward Pass -> Accumulate Gradients (Sum = G1 + G2 + G3 + G4)

Optimizer Step: Update Weights using Total Sum (G1 + G2 + G3 + G4)
Zero Gradients: Reset memory and start again.
```

Next, we configure the learning rate dynamics.

```python
        learning_rate = 2e-4,
        lr_scheduler_type = "linear",
```

We set `learning_rate = 2e-4`.

This translates to exactly 0.0002.

This is a highly standard and empirically reliable learning rate for QLoRA fine-tuning.

Because we are only training a tiny adapter matrix, we can afford to use a higher learning rate than we would for full fine-tuning.

We set `lr_scheduler_type = "linear"`.

This means the learning rate will slowly and linearly decrease over time as the training progresses.

This helps the model settle gracefully into a precise local minimum in the complex loss landscape.

```python
        warmup_steps = 5,
        max_steps = 60,
```

We set `warmup_steps = 5`.

For the very first 5 steps of training, the learning rate will gradually rise from 0 up to 2e-4.

This is mathematically crucial.

It prevents massive, chaotic gradient spikes at the very beginning of training when the adapters are completely randomly initialized.

We set `max_steps = 60`.

For a full production run on 15,000 examples, you would normally use full epochs.

You would set `num_train_epochs = 1` or `num_train_epochs = 2`.

For the sake of this specific script and rapid testing, we hardcode it to stop quickly after 60 steps.

Next, we handle numerical precision logic.

```python
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
```

This elegant boolean logic automatically selects the optimal mixed-precision format for your specific hardware.

It dynamically queries the PyTorch CUDA backend.

If your GPU successfully supports Bfloat16, it sets `bf16 = True` and `fp16 = False`.

If your GPU is older, it sets `fp16 = True` and `bf16 = False`.

Mixed precision dynamically casts certain mathematical operations to 16-bit while keeping others safely in 32-bit.

This dramatically accelerates matrix multiplications.

Next, we configure the optimizer itself.

```python
        logging_steps = 1,
        optim = "adamw_8bit",            # 8-bit optimizer to save VRAM
```

We set `logging_steps = 1`.

This ensures the trainer prints the current loss value to the console on every single optimization step.

This allows us to watch the convergence happen in real-time.

We set `optim = "adamw_8bit"`.

This is yet another massive, crucial memory optimization.

The standard AdamW optimizer is a heavily stateful optimizer.

It stores two separate state variables (momentum and variance) for every single parameter in the neural network.

In standard 32-bit float format, this requires exactly 8 bytes of memory per parameter.

By using the specialized 8-bit version of AdamW (provided by the `bitsandbytes` library), we actively quantize the optimizer states.

We mathematically reduce the memory requirement from 8 bytes down to just 2 bytes per parameter.

This saves multiple gigabytes of precious VRAM.

```python
        weight_decay = 0.01,
        seed = 3407,
    ),
)
```

We set `weight_decay = 0.01`.

Weight decay is a standard regularization technique.

It gently penalizes excessively large weights in the LoRA adapters by pushing them towards zero.

This helps explicitly prevent the model from overfitting to the training data.

Finally, we set a random `seed = 3407`.

This ensures that our dataset shuffling and initial weight initialization are completely reproducible.

If you run this script twice, you will get the exact same mathematical results.

---

# 11. Execution

Everything is now perfectly and completely configured.

```python
print("Starting Highly Optimized Training...")
trainer.train()
```

We call `trainer.train()`.

At this exact moment, the GPU will spin up to maximum utilization.

The dataset will rapidly flow through the tokenizer.

The loss masking collator will dynamically apply the `-100` labels to the tensors.

The forward pass will mathematically compute the logits for the sequence.

The Cross-Entropy loss will be calculated exclusively on the assistant's specific tokens.

The backward pass will compute the gradients based on that loss.

It will use gradient checkpointing to aggressively save memory.

The gradients will flow back down into the tiny LoRA adapters.

The 8-bit AdamW optimizer will read those gradients.

It will carefully update the small `A` and `B` weight matrices.

This entire loop will repeat exactly 60 times.

You will watch the loss number steadily and predictably decrease on your screen.

---

# 12. Saving the Model

Once the training loop is completely finished, we must save our hard work to the hard drive.

```python
# ==========================================
# 6. Save Final Model
# ==========================================
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("Training Complete. Model Saved.")
```

We call `model.save_pretrained("lora_model")`.

This creates a brand new folder called `lora_model`.

It is absolutely crucial to understand what this method does *not* do.

Because we exclusively used LoRA, this does **not** save an 8-Billion parameter model to your drive.

It does not save gigabytes upon gigabytes of data.

It only saves the tiny, highly concentrated LoRA adapter weights.

These output files will usually be incredibly small, between 50 and 150 Megabytes.

These adapter weights act precisely like a small software patch.

They can be loaded later at any time.

They will be mathematically injected back into the frozen base Llama-3 model at inference time.

Finally, we also save the tokenizer.

```python
tokenizer.save_pretrained("lora_model")
```

We must always save the tokenizer alongside the newly trained model.

This ensures that during future inference, our input text is mapped to the exact same token IDs that were used during training.

If you use a different tokenizer by mistake, the model will output complete, incoherent gibberish.

---

# 13. Summary of Optimizations

Let us review the massive, incredible stack of optimizations we just successfully deployed.

1.  **4-bit NormalFloat (NF4)**: We compressed the base model from 32GB down to just 5GB of VRAM.

2.  **Double Quantization**: We aggressively saved an additional 0.37 bits per parameter.

3.  **LoRA**: We reduced the number of mathematically trainable parameters by a staggering 99.2%.

4.  **Unsloth Triton Kernels**: We accelerated both the forward and backward passes by 2x.

5.  **Gradient Checkpointing**: We reduced the activation memory overhead by 50%.

6.  **Gradient Accumulation**: We achieved a mathematically stable batch size on severely limited hardware.

7.  **Mixed Precision (BF16)**: We sped up matrix multiplications without losing numerical stability.

8.  **8-bit AdamW**: We reduced the optimizer state memory overhead by a massive 75%.

9.  **Loss Masking**: We forced the model to only learn conversational responses, not user prompts.

10. **Chat Templates**: We seamlessly handled complex, model-specific prompt formatting.

---

# 14. Conclusion

You have now reached the absolute cutting edge of AI Engineering.

You deeply understand the foundational theory of large language models.

You meticulously understand how to gather and format instruction data.

You know exactly how to build complex, multi-stage training architectures.

You have overcome the severe physical memory limits of standard consumer hardware.

You understand the complex mathematics of Quantization and Fully Sharded Data Parallel (FSDP).

You know how to perfectly align models to human values using Direct Preference Optimization (DPO).

And you can orchestrate all of this seamlessly using high-performance custom kernels.

You are no longer a beginner.

You are now fully ready to build production AI systems.

This formally concludes the course.
