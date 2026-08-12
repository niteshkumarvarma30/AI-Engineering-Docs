# Lesson 28 — Distributed Training (FSDP & DeepSpeed)

> **Goal:** Understand Model Parallelism. Learn how to train models that are so large (e.g., 70B parameters) they cannot mathematically fit on a single GPU, using FSDP and DeepSpeed Zero.

---

# 1. The Limit of Data Parallelism (DDP)

In Lesson 27, we learned about Distributed Data Parallel (DDP).

DDP is a fantastic tool for training smaller models.

It is the industry standard for distributed training.

It is built directly into PyTorch.

It works by placing a full copy of the model on every GPU.

Every GPU receives a completely different mini-batch of data.

Every GPU performs a forward pass independently.

Every GPU computes its own loss.

Every GPU computes its own gradients during the backward pass.

Then, the GPUs communicate over the network.

They average their gradients together.

This network process is called an All-Reduce operation.

After the gradients are mathematically averaged, every GPU updates its weights.

Because they all started with the exact same model weights.

And because they all apply the exact same averaged gradient updates.

They remain perfectly synchronized at all times.

This is simple.

This is elegant.

This is highly effective.

But there is a major, fundamental problem.

DDP has a hard mathematical limit.

DDP requires a **full copy of the entire model** to exist in the VRAM of every single GPU.

What happens if the model is larger than the total VRAM of a single GPU?

What happens if you want to train a massive frontier model?

The answer is brutally simple.

The script crashes instantly.

You get the dreaded Out of Memory (OOM) error.

Your training run dies before it even computes a single step.

---

# 2. VRAM Math for LLMs

Let us do some precise math to truly understand the scale of the problem.

Suppose you want to fine-tune the Llama 3 70B model.

The "70B" in the name means the model has 70 billion parameters.

A parameter is a single weight or bias in the neural network.

It is a single floating-point number.

During training, we usually use mixed precision to save memory and increase speed.

The most common formats are FP16 or BF16 (Bfloat16).

Both of these formats use exactly 16 bits to store a single parameter.

As we know from basic computer science, 8 bits make up 1 byte.

Therefore, 16 bits is exactly equal to 2 bytes.

Let us calculate the VRAM required just to hold the model weights in memory.

$$

\text{Parameters} = 70,000,000,000

$$

$$

\text{Bytes per Parameter} = 2

$$

$$

\text{Total Bytes} = 70,000,000,000 \times 2 = 140,000,000,000

$$

We can convert bytes to Gigabytes (GB).

$$

\text{Total GB} \approx 140 \text{ GB}

$$

So, the raw weights of the model alone require 140 Gigabytes of VRAM.

This is just to load the model into memory.

It does not include any training overhead.

This is a massive amount of memory.

---

# 3. Hardware Limitations

Let us look at the reality of modern GPU hardware.

The Nvidia A100 GPU is a standard workhorse in the AI industry.

It comes in two main variants.

There is a 40 GB variant.

And there is an 80 GB variant.

The Nvidia H100 GPU is the current flagship Hopper architecture.

It also comes primarily with 80 GB of VRAM.

Do you see the fundamental problem here?

Our Llama 3 70B weights require 140 GB of VRAM.

Our largest, most expensive single GPU only has 80 GB of VRAM.

It is physically impossible to load the model onto a single GPU.

Therefore, standard Distributed Data Parallel (DDP) is mathematically impossible for this model.

It does not matter how many GPUs you have.

If you have a cluster of 1,000 H100 GPUs, DDP will still fail.

Because DDP requires the full model to fit on *each* individual GPU.

---

# 4. Training Memory Overhead

The problem is actually much, much worse than 140 GB.

The weights are only one small part of the memory required for a training run.

During training, we do not just store weights.

We must also store three other massive components:

1. Gradients

2. Optimizer States

3. Activations

Let us break down each of these components mathematically.

---

# 5. Gradients Memory

During the backward pass (backpropagation), we calculate a gradient for every single parameter.

The gradient tells the optimizer how to update the parameter to minimize the loss function.

Because there is one gradient per parameter, there are exactly 70 billion gradients.

Gradients are usually stored in 16-bit precision during mixed-precision training.

This matches the BF16 precision of the model weights.

So, gradients require another 2 bytes per parameter.

$$

\text{Gradients Memory} = 70,000,000,000 \times 2 \text{ bytes}

$$

$$

\text{Gradients Memory} \approx 140 \text{ GB}

$$

Our total VRAM requirement has now doubled from 140 GB to 280 GB.

But we are not done yet.

---

# 6. Optimizer States Memory

This is where the memory requirement truly explodes out of control.

Modern optimizers like Adam or AdamW are stateful.

They do not just multiply the gradient by the learning rate.

They maintain internal states for every single parameter over time.

Specifically, Adam keeps track of two distinct values for every parameter:

1. The momentum (the first moment estimate).

2. The variance (the second moment estimate).

These historical states are absolutely crucial for stable and fast training.

However, there is a catch.

Adam requires these states to be stored in full 32-bit precision (FP32).

If you store them in 16-bit precision, the numerical instability will destroy your training run.

32 bits is exactly equal to 4 bytes.

So, Adam stores two 4-byte values for every single parameter.

That is 8 bytes per parameter just for the momentum and variance.

Furthermore, Adam usually requires a 32-bit master copy of the weights to perform the update accurately.

This 32-bit master weight copy adds another 4 bytes per parameter.

Let us sum up the optimizer memory per parameter.

```text
Momentum (FP32):           4 bytes

Variance (FP32):           4 bytes

Master Weight (FP32):      4 bytes
----------------------------------
Total Optimizer Memory:   12 bytes
```

Now let us multiply this by 70 billion parameters.

$$

\text{Optimizer Memory} = 70,000,000,000 \times 12 \text{ bytes}

$$

$$

\text{Optimizer Memory} \approx 840 \text{ GB}

$$

This is a staggering, terrifying amount of memory.

---

# 7. Total VRAM Requirement

Let us sum up the absolute minimum static memory requirements for fine-tuning Llama 3 70B.

```text
Weights (BF16):               140 GB

Gradients (BF16):             140 GB

Optimizer States (FP32):      840 GB
--------------------------------------
Total Static Memory:        1,120 GB
```

You need 1.12 Terabytes of VRAM.

And remember, this does not even include activations.

Activations are the intermediate tensor values computed during the forward pass.

They must be saved in memory to compute the backward pass.

Activations depend heavily on the batch size and the sequence length.

If you are training on large documents (e.g., 8,000 tokens), activations will consume hundreds of additional Gigabytes.

But already, our static memory is 1,120 GB.

An 80 GB H100 GPU costs roughly $30,000.

Even if you buy an 8-GPU node for $250,000.

That node only has 640 GB of total VRAM.

DDP requires the full 1,120 GB to exist on every single GPU.

Clearly, we need a fundamentally different mathematical approach to distributed training.

---

# 8. Model Parallelism (Sharding)

If the model is too big for one GPU, we cannot duplicate it.

We must accept reality.

Instead of duplicating the model, we must cut the model into pieces.

These pieces are called **shards**.

We then distribute the pieces across multiple GPUs.

This overarching concept is known in the industry as **Model Parallelism**.

There are several distinct ways to implement Model Parallelism.

We will explore the historical approaches first.

Then we will explore the modern state-of-the-art approach.

---

# 9. Pipeline Parallelism

The earliest and most conceptually intuitive form of Model Parallelism is Pipeline Parallelism.

Think about the architecture of a transformer neural network.

It is a sequential stack of layers.

Suppose our massive model has 80 transformer layers.

And suppose we have a cluster of 4 GPUs.

We can simply slice the model vertically, layer by layer.

```text
GPU 0: Layers 1 to 20

GPU 1: Layers 21 to 40

GPU 2: Layers 41 to 60

GPU 3: Layers 61 to 80
```

This seems like a perfect, elegant solution.

Each GPU only holds 25% of the model layers.

Therefore, the memory requirement per GPU is reduced by a factor of 4.

The model now fits in VRAM.

---

# 10. The Pipeline Bubble

However, Pipeline Parallelism has a severe, fatal flaw regarding compute efficiency.

Let us trace a forward pass step-by-step through this multi-GPU system.

When a fresh batch of training data arrives, GPU 0 starts computing Layer 1.

While GPU 0 is furiously working on Layers 1 through 20, what are the other GPUs doing?

The answer is nothing.

They are doing absolutely nothing.

They are sitting completely idle at 0% utilization.

They must wait for GPU 0 to finish its 20 layers.

Then GPU 0 must pass the intermediate activations over the network to GPU 1.

Only then can GPU 1 start computing Layer 21.

While GPU 1 is working, GPU 0 is now idle.

GPU 2 and GPU 3 are still idle.

This massive period of idle time is called the **Pipeline Bubble**.

Let us visualize this inefficiency with an ASCII timeline.

```text
Time ->

GPU 0: [ Compute L1-L20 ] [ Idle          ] [ Idle          ] [ Idle          ]

GPU 1: [ Idle          ] [ Compute L21-L40] [ Idle          ] [ Idle          ]

GPU 2: [ Idle          ] [ Idle          ] [ Compute L41-L60] [ Idle          ]

GPU 3: [ Idle          ] [ Idle          ] [ Idle          ] [ Compute L61-L80]
```

This is incredibly inefficient.

AI accelerators are extremely expensive hardware.

Having them sit at 0% utilization for 75% of the time wastes millions of dollars of compute budget.

Researchers have invented complex micro-batching scheduling schemes to minimize this bubble.

They chop the batch into tiny pieces and interleave them.

But the bubble can never be completely eliminated in pure Pipeline Parallelism.

You always lose hardware efficiency.

---

# 11. Tensor Parallelism

Another historical approach is Tensor Parallelism (TP).

Instead of slicing the model sequentially by layers, TP slices the individual math operations inside each layer.

Inside a transformer, the heaviest operations are massive matrix multiplications.

TP takes a single large matrix multiplication and splits it into smaller matrix multiplications.

Each GPU computes a piece of the matrix multiplication simultaneously.

Then, they rapidly synchronize their partial results over the network to get the final answer.

This is highly efficient regarding compute.

It effectively eliminates the pipeline bubble.

Every GPU is working on a piece of the math all the time.

However, Tensor Parallelism has a different fatal flaw.

It requires truly massive network bandwidth.

The GPUs must synchronize their partial results multiple times per single layer.

It only works efficiently if the GPUs are connected by ultra-fast, dedicated hardware links like Nvidia NVLink.

NVLink is extremely fast, but it only exists within a single server chassis (usually up to 8 GPUs).

If you try to run Tensor Parallelism across different server nodes using standard Ethernet, the network becomes a massive bottleneck.

The GPUs will spend more time waiting for data over the network than they spend computing.

So Tensor Parallelism does not scale well beyond a single 8-GPU node.

---

# 12. The Breakthrough: ZeRO

In 2020, researchers at Microsoft published a groundbreaking, paradigm-shifting paper.

They introduced the **Zero Redundancy Optimizer**.

It is almost exclusively referred to by its acronym: **ZeRO**.

ZeRO observed something profound and obvious about standard DDP.

DDP duplicates all weights, all gradients, and all optimizer states identically across all GPUs.

This massive redundancy is completely mathematically unnecessary.

Why on earth should we store 8 identical copies of the Adam momentum state in VRAM when we only need 1?

ZeRO proposed to shard the data, just like Model Parallelism.

But it proposed to do it while keeping the simplicity and full utilization of the DDP training loop.

---

# 13. Fully Sharded Data Parallel (FSDP)

Inspired by the success of ZeRO, engineers at Meta (Facebook) implemented a very similar algorithm.

They built it natively directly into PyTorch.

They called it **FSDP** (Fully Sharded Data Parallel).

ZeRO and FSDP are fundamentally based on the exact same brilliant mathematical concept.

They use a technique that can be summarized as:

**Weight Sharding with Just-In-Time Gathering**.

They allow you to train infinitely massive models.

As long as the total combined VRAM of your entire server cluster is larger than the model requirements, you can train it.

Let us break down exactly how this magic works, step-by-step.

---

# 14. How FSDP Works

Remember how Pipeline Parallelism gives Layer 1 to GPU 0, and Layer 2 to GPU 1?

FSDP does not do that.

FSDP looks at the weights of a *single* layer.

It then mathematically slices those weights into equal, smaller chunks across all available GPUs in the cluster.

Let us trace the precise lifecycle of a single layer under FSDP.

Assume we are running a cluster of 4 GPUs.

---

# 15. Step 1: Sharding at Rest

At rest, when the model is loaded into VRAM, no single GPU holds the full weights for Layer 1.

GPU 0 holds only the first 25% chunk of Layer 1.

GPU 1 holds only the next 25% chunk.

GPU 2 holds only the third 25% chunk.

GPU 3 holds only the final 25% chunk.

Let us visualize this sharded memory state.

```text
       FSDP Memory Allocation at Rest (4 GPUs)
       
[ Full Layer 1 Weights ] =  [ 25% ] + [ 25% ] + [ 25% ] + [ 25% ]
                               |         |         |         |
                            (GPU 0)   (GPU 1)   (GPU 2)   (GPU 3)
```

This sharding state persists at all times while the GPUs are idle, or while they are computing other layers.

Because the weights are physically sharded, the memory usage per GPU is reduced by 75%.

The model has been successfully distributed.

But how do we compute a forward pass if no GPU has the full layer?

---

# 16. Step 2: The All-Gather Operation

Now, the forward pass begins and reaches Layer 1.

In order to compute the matrix multiplication for Layer 1, every GPU absolutely needs the full weights.

FSDP dynamically triggers a highly optimized network communication operation.

This operation is called an **All-Gather**.

All GPUs rapidly blast their 25% shards to each other over the network.

```text
GPU 0 sends its 25% chunk to GPUs 1, 2, and 3.

GPU 1 sends its 25% chunk to GPUs 0, 2, and 3.

GPU 2 sends its 25% chunk to GPUs 0, 1, and 3.

GPU 3 sends its 25% chunk to GPUs 0, 1, and 2.
```

This happens incredibly fast.

For a brief fraction of a second, the full Layer 1 weights are assembled and exist entirely in the VRAM on all 4 GPUs.

They have successfully "gathered" the weights.

---

# 17. Step 3: Compute the Forward Pass

Now that every GPU temporarily has the full Layer 1 weights, they can do the math.

Just like in standard DDP, each GPU has its own unique mini-batch of training data.

They all independently compute the forward pass matrix multiplication for Layer 1.

They produce the intermediate activations for the next layer.

This happens in parallel across all GPUs simultaneously.

Therefore, every GPU is fully utilized.

There is absolutely no pipeline bubble.

Compute efficiency is maintained at near 100%.

---

# 18. Step 4: Discard and Free VRAM

As soon as the computation for Layer 1 is mathematically finished, the true magic of FSDP happens.

The GPUs immediately delete the gathered weights from their VRAM.

GPU 0 ruthlessly throws away the 75% of weights it just received from the other GPUs.

It only keeps its original assigned 25% shard.

The other GPUs do exactly the same thing.

The VRAM that was temporarily consumed by the full layer is instantly freed up.

Then, the exact same process repeats for Layer 2.

```text
1. All-Gather Layer 2 weights over the network.

2. Compute Layer 2 forward pass.

3. Discard Layer 2 gathered weights.
```

And then for Layer 3.

```text
1. All-Gather Layer 3 weights over the network.

2. Compute Layer 3 forward pass.

3. Discard Layer 3 gathered weights.
```

This brilliant technique is Just-In-Time Gathering.

It ensures that at any given moment, only one full layer exists in VRAM.

The rest of the model remains safely sharded.

---

# 19. Backpropagation with FSDP

The exact same Just-In-Time process happens in reverse during the backward pass.

When it is time to compute gradients for Layer 80, the GPUs All-Gather the weights again.

They compute the local gradients.

Then they instantly discard the weights again.

Furthermore, under full FSDP, the gradients themselves are also sharded.

When GPU 0 computes its gradient, it only cares about the gradient for its specific 25% of the weights.

A network operation called Reduce-Scatter averages the gradients and ensures GPU 0 only holds its 25% gradient shard.

Finally, the optimizer updates the weights.

Because the optimizer states are also sharded exactly like the weights.

Each GPU only updates its own 25% slice of the model.

No GPU ever holds the full optimizer state.

No GPU ever holds the full gradient tensor.

---

# 20. FSDP Memory Reduction

Let us revisit the memory savings for our massive Llama 3 70B model.

Recall our calculated static memory requirement was 1,120 GB.

If we rent a cluster of 16 H100 GPUs (with 80 GB of VRAM each).

Our total cluster VRAM capacity is:

$$

16 \times 80 = 1,280 \text{ GB}

$$

Under FSDP, the massive 1,120 GB requirement is distributed evenly across all 16 GPUs.

$$

\text{Memory per GPU} = \frac{1,120 \text{ GB}}{16} = 70 \text{ GB}

$$

70 GB fits perfectly and comfortably within the 80 GB hard limit of a single H100 GPU.

We have plenty of room left over for activations and context windows.

We can now successfully train the 70B model.

We have solved the OOM error without wasting compute on pipeline bubbles.

We have achieved the holy grail of distributed training.

---

# 21. DeepSpeed ZeRO Stages

DeepSpeed formalizes this sharding concept into three distinct levels of aggression.

These are known globally as ZeRO (Zero Redundancy Optimizer) stages.

You as the AI Engineer can choose how aggressively you want to shard based on your model size and hardware.

Each stage saves more memory, but requires more network communication overhead.

Let us review each stage.

---

# 22. ZeRO Stage 1

In ZeRO Stage 1, we only shard the Optimizer States.

Recall our math from earlier.

The Adam optimizer states take up the vast majority of the memory overhead (12 bytes per parameter).

In Stage 1, we leave the Model Weights fully replicated across all GPUs (just like DDP).

We leave the Gradients fully replicated across all GPUs (just like DDP).

We only slice the massive Adam momentum and variance arrays into shards.

At the end of the backward pass, each GPU is responsible for updating a small portion of the weights.

Then, they share the updated weights with everyone else over the network.

This simple change saves roughly 4x total VRAM.

It introduces very little network overhead.

It is highly efficient for moderately sized models.

---

# 23. ZeRO Stage 2

In ZeRO Stage 2, we get more aggressive.

We shard the Optimizer States.

And we also shard the Gradients.

We leave only the Model Weights fully replicated across all GPUs.

This saves roughly 8x total VRAM compared to raw DDP.

This is the standard, go-to configuration for fine-tuning medium-sized open-source models.

If you are fine-tuning a 7B parameter or 13B parameter model, ZeRO Stage 2 is usually the optimal choice.

It provides massive memory savings while keeping communication overhead relatively low.

---

# 24. ZeRO Stage 3

In ZeRO Stage 3, we shard absolutely everything.

We shard the Optimizer States.

We shard the Gradients.

We shard the Model Weights.

This is exactly functionally equivalent to the Fully Sharded Data Parallel (FSDP) algorithm we described above.

Nothing is fully replicated.

Weights are gathered just-in-time over the network.

Weights are discarded immediately after computation.

This is strictly required for massive frontier models (greater than 30B parameters).

It allows you to train infinitely large models.

If your model is too big, you simply add more GPUs to the cluster to distribute the shards further.

---

# 25. ZeRO-Offload

DeepSpeed also introduced another incredible, paradigm-shifting feature called ZeRO-Offload.

What if your model is so incredibly big that even ZeRO Stage 3 cannot fit it on your GPUs?

Or what if you are a hobbyist and you only have a single GPU or two GPUs?

ZeRO-Offload allows you to offload the massive Optimizer States out of GPU VRAM entirely.

It offloads them to the CPU memory (System RAM).

CPU RAM is significantly cheaper and far more plentiful than GPU VRAM.

A standard cloud server might have 1 or 2 Terabytes of standard DDR4 or DDR5 CPU RAM.

During the backward pass, DeepSpeed will dynamically stream the computed gradients over the PCIe bus to the CPU.

The CPU itself computes the Adam optimizer update using system RAM.

Then, the CPU streams the freshly updated weights back over the PCIe bus to the GPU.

This fundamentally slows down training.

The CPU is much slower at matrix math than a GPU.

And the PCIe bus is much slower than VRAM bandwidth.

But it allows you to train massive models on severely limited hardware that would otherwise be impossible.

---

# 26. Using DeepSpeed in Hugging Face

You might reasonably think that implementing ZeRO requires rewriting all your PyTorch training code from scratch.

Thankfully, it does not.

Hugging Face `transformers` and `accelerate` have deep, native DeepSpeed integration.

You do not need to change a single line of your actual `SFTTrainer` python script.

Your training loop remains identical.

You simply need to provide a configuration file to tell the system how to shard.

---

# 27. The DeepSpeed Configuration File

You create a standard JSON file, typically named `deepspeed_config.json`.

This file tells the DeepSpeed engine exactly how to behave.

Here is a standard, robust configuration for ZeRO Stage 3:

```json
{
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "fp16": {
        "enabled": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

This JSON explicitly specifies `"stage": 3`.

It also uses the `"auto"` keyword extensively.

This tells Hugging Face to automatically fill in the correct batch sizes from your `TrainingArguments` so you do not have to hardcode them.

---

# 28. Adding CPU Offload

If you want to use the CPU offload feature, you simply modify the JSON config.

You add the `offload_optimizer` block inside the `zero_optimization` settings.

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "fp16": {
        "enabled": true
    }
}
```

The `"pin_memory": true` setting is critical.

It allocates page-locked memory on the CPU, which drastically speeds up the data transfer over the PCIe bus between the CPU and GPU.

---

# 29. Launching the Script

Normally, you launch a Python training script like this:

```bash
python my_training_script.py
```

To use DeepSpeed, you cannot do this.

You must use the Hugging Face `accelerate` CLI tool.

You tell `accelerate` to use DeepSpeed and you point it to your JSON configuration file.

```bash
accelerate launch \
  --use_deepspeed \
  --deepspeed_config_file deepspeed_config.json \
  my_training_script.py
```

The Accelerate library intercepts the script execution.

It spawns multiple independent Python processes (exactly one per GPU).

It hijacks the standard PyTorch model initialization and wraps the model tightly in the DeepSpeed engine.

The `SFTTrainer` runs exactly as it did before.

But underneath the hood, all the complex sharding, gathering over the network, and discarding of memory happens automatically and invisibly.

---

# 30. FSDP vs DeepSpeed

As an AI Engineer, you might wonder whether you should use PyTorch native FSDP or Microsoft's DeepSpeed.

Both accomplish the exact same fundamental goal (which is ZeRO Stage 3).

DeepSpeed is an external third-party library maintained by Microsoft.

FSDP is built natively directly into the core of PyTorch by Meta.

Currently, DeepSpeed has slightly more advanced edge-case features (like CPU offload and NVMe storage offload).

However, FSDP is catching up incredibly fast.

FSDP is often considered more stable and less prone to weird library conflicts because it is fully native to PyTorch.

Hugging Face `accelerate` fully supports both options identically.

You can launch native FSDP simply by running the configuration wizard:

```bash
accelerate config
```

And answering the interactive terminal prompts to enable FSDP.

Then you simply run your script:

```bash
accelerate launch my_training_script.py
```

It is ultimately a matter of preference and hardware constraints.

---

# 31. Summary and Conclusion

Let us recap the immense amount of theory we have learned in this lesson.

DDP duplicates the entire model on every GPU, which mathematically causes OOM for large models.

Model Parallelism is strictly required for frontier models like Llama 3 70B.

Pipeline Parallelism slices the model by layers, but creates highly inefficient pipeline bubbles where GPUs sit idle.

ZeRO and FSDP solve this brilliantly by sharding the weights, gradients, and optimizer states across all GPUs.

They dynamically All-Gather the weights Just-In-Time for matrix computation.

They ruthlessly discard the weights immediately after computation to free VRAM.

This elegant mathematical trick reduces memory linearly with the number of GPUs you add to your cluster.

DeepSpeed ZeRO offers Stage 1, Stage 2, and Stage 3 depending on exactly how much VRAM you need to save.

You can easily enable all of this advanced engineering in Hugging Face using `accelerate` and a simple JSON config file.

You now possess the foundational knowledge required to train the largest language models on earth.

