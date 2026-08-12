# Lesson 27 — Multi-GPU Fine-Tuning

> **Goal:** Understand how to move beyond a single GPU, the mechanics of Data Parallelism (DP) and Distributed Data Parallelism (DDP), the Ring-AllReduce algorithm, and the mathematics of global batch sizes.

---

# 1. The Single GPU Bottleneck

We have learned how to fine-tune models on a single GPU.

We have used techniques like QLoRA to reduce memory usage.

We have used techniques like Unsloth to speed up training.

These techniques save massive amounts of memory.

They allow us to fit large models onto consumer hardware.

For example, you can fit a 7B parameter model on an RTX 4090.

The RTX 4090 has 24GB of VRAM.

This is a tremendous achievement for open-source AI.

However, memory is only one part of the training equation.

The other fundamental part is compute.

Compute refers to the sheer number of mathematical operations required.

Specifically, the GPU must perform trillions of matrix multiplications.

It must perform these multiplications for every single batch of data.

It must perform them during the forward pass.

It must perform them during the backward pass.

Even if a model perfectly fits inside your GPU memory, processing a large dataset takes time.

Consider a dataset with one million conversational examples.

A single RTX 4090 is incredibly fast.

But it can only process a few examples per second.

If you process 5 examples per second, one million examples will take a very long time.

It might take days.

It might take weeks.

If you have a strict deadline, a three-week training run is unacceptable.

You cannot iterate on your ideas if each experiment takes three weeks.

You cannot test new hyperparameters quickly.

Therefore, we must find a way to train faster.

We cannot make the single GPU magically run ten times faster.

The hardware has physical limitations.

To reduce training time from weeks to hours, we must distribute the workload.

We must use multiple GPUs simultaneously.

---

# 2. Vertical vs Horizontal Scaling

In traditional software engineering, there are two ways to scale a system.

The first way is vertical scaling.

Vertical scaling means buying a bigger, faster machine.

In the context of AI, this means upgrading your GPU.

You might upgrade from an RTX 3090 to an RTX 4090.

You might upgrade from an RTX 4090 to an H100.

This provides a noticeable speedup.

But there is a hard ceiling on vertical scaling.

The H100 is currently one of the fastest GPUs in the world.

Once you have an H100, you cannot buy a "super H100" that is ten times faster.

Physical silicon limits prevent infinite vertical scaling.

The second way to scale is horizontal scaling.

Horizontal scaling means adding more machines.

Instead of buying one infinitely fast GPU, you buy eight fast GPUs.

Or you rent a server with 8x H100 GPUs.

Or you rent a cluster with 1,024 GPUs.

Horizontal scaling theoretically allows infinite compute power.

However, horizontal scaling introduces a massive new problem.

The problem is coordination.

How do you get eight independent GPUs to work together on the exact same model?

How do they share their findings?

How do they update the model weights without overwriting each other?

This is the domain of distributed training.

---

# 3. Data Parallelism (DP)

There are several ways to distribute training across multiple GPUs.

The most intuitive method is called Data Parallelism.

Data Parallelism is often abbreviated as DP.

In Data Parallelism, the model is duplicated.

The data is split.

This is the core philosophy of DP.

Let us explore exactly how this works step by step.

Assume we have 4 GPUs.

Assume we have a batch size of 32 examples.

```text
Dataset Batch:
[ Example 1, Example 2, ..., Example 32 ]

GPUs Available:
[ GPU 0, GPU 1, GPU 2, GPU 3 ]
```

In the traditional DP approach, one GPU is designated as the "Master."

Usually, GPU 0 is the Master GPU.

The Master GPU holds the main copy of the neural network.

Step 1: The Master GPU copies its neural network to the other GPUs.

```text
Master (GPU 0) Neural Network
       │
       ├─► Copy to GPU 1
       ├─► Copy to GPU 2
       └─► Copy to GPU 3
```

Now, all four GPUs have the exact same model weights.

Step 2: The Master GPU takes the batch of 32 examples.

It divides this batch into four smaller mini-batches.

Each mini-batch contains 8 examples.

```text
Batch (32)
   │
   ├─► 8 examples ─► GPU 0
   ├─► 8 examples ─► GPU 1
   ├─► 8 examples ─► GPU 2
   └─► 8 examples ─► GPU 3
```

Step 3: Each GPU performs a forward pass independently.

GPU 0 processes its 8 examples and calculates a loss.

GPU 1 processes its 8 examples and calculates a loss.

GPU 2 processes its 8 examples and calculates a loss.

GPU 3 processes its 8 examples and calculates a loss.

Step 4: Each GPU performs a backward pass independently.

This calculates the gradients for its specific 8 examples.

Remember, gradients represent the direction and magnitude to update the weights.

Step 5: The worker GPUs send their gradients back to the Master GPU.

```text
GPU 1 Gradients ─► Master (GPU 0)
GPU 2 Gradients ─► Master (GPU 0)
GPU 3 Gradients ─► Master (GPU 0)
```

Step 6: The Master GPU receives all the gradients.

It averages the gradients from all four GPUs.

$$
\text{Average Gradient} = \frac{G_0 + G_1 + G_2 + G_3}{4}
$$

Step 7: The Master GPU uses this average gradient to update its model weights.

The Master GPU now has an updated, improved model.

The worker GPUs still have the old model.

Step 8: The Master GPU copies the new, updated model back to the worker GPUs.

The cycle then repeats for the next batch of data.

---

# 4. The Master GPU Bottleneck

Data Parallelism (DP) is conceptually very simple.

It is very easy to understand.

However, it has a fatal flaw.

The flaw is the Master GPU.

Look closely at the steps we just outlined.

The Master GPU has to do significantly more work than the other GPUs.

It must hold the data.

It must split the data.

It must scatter the data to the workers.

It must wait for all workers to finish.

It must gather all the gradients from the workers.

It must average those gradients.

It must update the model weights.

It must broadcast the new model back to the workers.

This creates a massive bottleneck.

```text
          Worker 1
             ▲
             │ (Data/Model/Gradients)
             ▼
Worker 2 ◄── MASTER ──► Worker 3
```

The communication happens over the PCIe bus.

The PCIe bus is the physical connection between the GPU and the motherboard.

The PCIe bus has limited bandwidth.

Copying a 7-billion parameter model back and forth over the PCIe bus takes time.

Sending gigabytes of gradients takes time.

Because the Master GPU has to communicate with every single worker, the network becomes congested.

Furthermore, the Master GPU requires more VRAM.

It needs to hold the gathered gradients from all workers simultaneously.

If you monitor GPU usage during DP training, you will see something alarming.

GPU 0 (the Master) might be at 100% memory and 100% utilization.

GPU 1, 2, and 3 might be at 50% memory and 30% utilization.

The worker GPUs spend most of their time waiting.

They wait to receive data.

They wait to receive the updated model.

Because GPU 0 runs out of memory first, your maximum batch size is severely limited.

DP is highly inefficient.

It does not scale well beyond 2 or 4 GPUs.

If you try to use DP on 8 GPUs, the overhead of the Master GPU destroys any performance gains.

We need a better way.

---

# 5. Distributed Data Parallelism (DDP)

The solution to the Master GPU bottleneck is Distributed Data Parallelism.

This is universally abbreviated as DDP.

DDP is the modern standard for training large language models.

Every major framework uses DDP.

PyTorch natively supports DDP.

Hugging Face uses DDP under the hood.

The fundamental principle of DDP is equality.

There is no Master GPU.

There is no single bottleneck.

Every GPU is treated exactly the same.

Every GPU is a peer in a distributed network.

Let us explore how DDP works step by step.

It solves every problem that DP created.

---

# 6. Identical Replicas in DDP

In DDP, we start by launching a separate Python process for every GPU.

If you have 4 GPUs, DDP launches 4 independent Python processes.

```text
Process 0 ─► Controls GPU 0
Process 1 ─► Controls GPU 1
Process 2 ─► Controls GPU 2
Process 3 ─► Controls GPU 3
```

These processes do not share memory.

They run completely independently.

At the very beginning of the training script, each process loads a copy of the model.

Process 0 loads the model onto GPU 0.

Process 1 loads the model onto GPU 1.

Process 2 loads the model onto GPU 2.

Process 3 loads the model onto GPU 3.

```text
GPU 0: Complete Model Replica
GPU 1: Complete Model Replica
GPU 2: Complete Model Replica
GPU 3: Complete Model Replica
```

Crucially, all of these models start with the exact same initialized weights.

They are identical twins.

There is no need for a Master GPU to copy the model to the workers.

Every process handles its own initialization.

This completely eliminates the initial broadcast overhead.

---

# 7. The Distributed Sampler

Since every GPU has the same model, we must ensure they process different data.

If every GPU processed the exact same data, they would calculate the exact same gradients.

That would be a waste of compute.

We need to chunk the dataset.

In DDP, this is handled by a component called the `DistributedSampler`.

The `DistributedSampler` looks at the total dataset.

It also looks at the total number of GPUs (often called the `world_size`).

It mathematically divides the dataset into non-overlapping chunks.

```text
Total Dataset: 1,000,000 examples
World Size: 4 GPUs

Chunk 0: 250,000 examples
Chunk 1: 250,000 examples
Chunk 2: 250,000 examples
Chunk 3: 250,000 examples
```

Process 0 is assigned Chunk 0.

Process 1 is assigned Chunk 1.

Process 2 is assigned Chunk 2.

Process 3 is assigned Chunk 3.

```text
Distributed Sampler
       │
       ├─► Chunk 0 ─► Process 0 (GPU 0)
       ├─► Chunk 1 ─► Process 1 (GPU 1)
       ├─► Chunk 2 ─► Process 2 (GPU 2)
       └─► Chunk 3 ─► Process 3 (GPU 3)
```

No data is sent over the PCIe bus between GPUs.

Each Python process loads its own chunk directly from the hard drive or RAM.

This completely eliminates the Master GPU data-scattering bottleneck.

Every GPU is now responsible for its own data loading.

---

# 8. Independent Forward and Backward Passes

Now the training loop begins.

Process 0 takes a batch of 8 examples from Chunk 0.

Process 1 takes a batch of 8 examples from Chunk 1.

Process 2 takes a batch of 8 examples from Chunk 2.

Process 3 takes a batch of 8 examples from Chunk 3.

Each GPU performs the forward pass independently.

```text
GPU 0: Forward Pass (Batch A) ─► Loss A
GPU 1: Forward Pass (Batch B) ─► Loss B
GPU 2: Forward Pass (Batch C) ─► Loss C
GPU 3: Forward Pass (Batch D) ─► Loss D
```

There is no communication required during the forward pass.

Next, each GPU performs the backward pass independently.

```text
GPU 0: Loss A.backward() ─► Gradients A
GPU 1: Loss B.backward() ─► Gradients B
GPU 2: Loss C.backward() ─► Gradients C
GPU 3: Loss D.backward() ─► Gradients D
```

At this exact moment, every GPU holds a different set of gradients.

GPU 0 knows how the weights should change to minimize Loss A.

GPU 1 knows how the weights should change to minimize Loss B.

If each GPU updated its weights right now, the models would diverge.

GPU 0 would become a different model than GPU 1.

They would no longer be identical twins.

We cannot let this happen.

We must synchronize the models.

We must ensure that every GPU applies the exact same weight update.

To do this, we must average the gradients across all GPUs.

But remember, we cannot use a Master GPU.

We must average the gradients without a central bottleneck.

This is where the magic of DDP happens.

---

# 9. The Communication Problem

How do you average data across 4 independent nodes without a master node?

A naive approach would be for every GPU to send its gradients to every other GPU.

```text
GPU 0 sends to GPU 1, GPU 2, GPU 3
GPU 1 sends to GPU 0, GPU 2, GPU 3
GPU 2 sends to GPU 0, GPU 1, GPU 3
GPU 3 sends to GPU 0, GPU 1, GPU 2
```

This is called an all-to-all communication pattern.

It is extremely inefficient.

The number of connections grows exponentially with the number of GPUs.

The network becomes completely overwhelmed.

We need a mathematically elegant solution.

We need an algorithm that minimizes the amount of data transferred.

We need an algorithm that maximizes the use of available bandwidth.

The solution is an algorithm called Ring-AllReduce.

---

# 10. The Ring-AllReduce Algorithm

Ring-AllReduce is the beating heart of modern distributed AI.

It is a communication protocol.

It allows multiple nodes to share and average their data efficiently.

In Ring-AllReduce, the GPUs are conceptually arranged in a logical ring.

```text
    GPU 0 ────────► GPU 1
      ▲               │
      │               │
      │               ▼
    GPU 3 ◄──────── GPU 2
```

GPU 0 only sends data to GPU 1.

GPU 1 only sends data to GPU 2.

GPU 2 only sends data to GPU 3.

GPU 3 only sends data to GPU 0.

There is no central master.

There is no all-to-all chaos.

The data flows in a continuous circle.

To understand how it computes the average, we must break down the mathematics.

---

# 11. The Mathematics of Gradient Synchronization

Let us simplify the scenario.

Assume our entire model consists of only 4 parameters.

We have 4 GPUs.

Each GPU has computed a gradient for these 4 parameters.

Let us represent the gradients as arrays.

```text
GPU 0 Gradients: [ A0, B0, C0, D0 ]
GPU 1 Gradients: [ A1, B1, C1, D1 ]
GPU 2 Gradients: [ A2, B2, C2, D2 ]
GPU 3 Gradients: [ A3, B3, C3, D3 ]
```

Our goal is for every GPU to end up with the exact same averaged array:

$$
\text{Goal} = [ \sum A, \sum B, \sum C, \sum D ]
$$

Where:

$$
\sum A = A0 + A1 + A2 + A3
$$

The Ring-AllReduce algorithm works in two phases.

The first phase is called the **Scatter-Reduce Phase**.

The second phase is called the **All-Gather Phase**.

### Phase 1: Scatter-Reduce

In this phase, we divide the gradient array into chunks.

Since we have 4 GPUs, we divide the array into 4 chunks.

Conveniently, our array has 4 elements, so each chunk is 1 element.

Each GPU is assigned a specific chunk to be responsible for.

GPU 0 is responsible for chunk A.

GPU 1 is responsible for chunk B.

GPU 2 is responsible for chunk C.

GPU 3 is responsible for chunk D.

In step 1, each GPU sends a different chunk to its right neighbor.

GPU 0 sends chunk B0 to GPU 1.

GPU 1 sends chunk C1 to GPU 2.

GPU 2 sends chunk D2 to GPU 3.

GPU 3 sends chunk A3 to GPU 0.

When a GPU receives a chunk, it adds it to its own chunk.

```text
GPU 1 receives B0, computes: B0 + B1
GPU 2 receives C1, computes: C1 + C2
GPU 3 receives D2, computes: D2 + D3
GPU 0 receives A3, computes: A3 + A0
```

In step 2, the GPUs pass these new sums to the right.

GPU 1 sends (B0 + B1) to GPU 2.

GPU 2 receives this and computes: (B0 + B1) + B2.

At the same time:

GPU 2 sends (C1 + C2) to GPU 3.

GPU 3 receives this and computes: (C1 + C2) + C3.

This continues around the ring.

After $N-1$ steps (where N is the number of GPUs), a magical thing happens.

Each GPU now holds the complete sum for its assigned chunk.

```text
GPU 0 holds the complete sum of A: (A0 + A1 + A2 + A3)
GPU 1 holds the complete sum of B: (B0 + B1 + B2 + B3)
GPU 2 holds the complete sum of C: (C0 + C1 + C2 + C3)
GPU 3 holds the complete sum of D: (D0 + D1 + D2 + D3)
```

At this point, the total sum is scattered across the GPUs.

No single GPU has the full answer.

But collectively, the cluster has calculated all the sums.

### Phase 2: All-Gather

Now we must distribute these final sums to everyone.

We use the exact same ring structure.

GPU 0 sends the complete sum of A to GPU 1.

GPU 1 receives it, and stores it.

GPU 1 sends the complete sum of B to GPU 2.

GPU 2 receives it, and stores it.

In the next step, they pass these completed chunks forward again.

GPU 1 passes the complete sum of A to GPU 2.

GPU 2 passes the complete sum of B to GPU 3.

After $N-1$ steps of the All-Gather phase, every GPU has received every completed chunk.

Every GPU now holds the exact same array:

```text
[ (A0+A1+A2+A3), (B0+B1+B2+B3), (C0+C1+C2+C3), (D0+D1+D2+D3) ]
```

Finally, each GPU independently divides the array by $N$ (which is 4) to compute the average.

The synchronization is complete.

---

# 12. Updating the Weights

Through the elegance of Ring-AllReduce, every GPU now has the exact same averaged gradients.

They achieved this without a master node.

They achieved this using a perfectly balanced communication network.

Bandwidth was utilized uniformly.

Now, every GPU looks at these averaged gradients.

Every GPU independently feeds these gradients into its optimizer (e.g., AdamW).

Every GPU independently updates its model weights.

Because the models started identical, and the gradients are identical, the weight updates are identical.

Therefore, the models remain identical twins.

```text
GPU 0 Model ────────► Updated Model
GPU 1 Model ────────► Updated Model
GPU 2 Model ────────► Updated Model
GPU 3 Model ────────► Updated Model
```

The loop is now complete.

The distributed sampler grabs the next batch of data.

The cycle repeats.

This is the beauty of Distributed Data Parallelism.

It scales near-linearly.

If you use 8 GPUs, it is almost exactly 8 times faster than 1 GPU.

---

# 13. High-Speed Interconnects

We mentioned that the GPUs communicate in a ring.

If they communicate over the standard PCIe bus, it is still somewhat slow.

To maximize DDP performance, servers use specialized hardware.

NVIDIA created a technology called NVLink.

NVLink is a dedicated, ultra-high-bandwidth bridge between GPUs.

It completely bypasses the motherboard and the CPU.

```text
GPU 0 ◄══ NVLink ══► GPU 1
```

NVLink is orders of magnitude faster than PCIe.

When Ring-AllReduce operates over NVLink, gradient synchronization happens in milliseconds.

This is why professional AI training servers (like an 8x H100 node) are so expensive.

You are not just paying for 8 GPUs.

You are paying for the NVLink switches that connect them into a massive unified brain.

---

# 14. Implementing DDP in Hugging Face

Historically, writing the PyTorch code for DDP was difficult.

You had to manually spawn processes.

You had to initialize process groups.

You had to configure network ports and IP addresses.

You had to wrap your model in a `DistributedDataParallel` class.

You had to manually write the `DistributedSampler` logic.

It was very easy to make a mistake and cause a deadlock.

A deadlock is when GPU 0 waits for GPU 1, but GPU 1 is waiting for GPU 0.

The entire cluster freezes indefinitely.

Fortunately, Hugging Face abstracts all of this away.

Hugging Face created a library called `accelerate`.

The `accelerate` library handles all the complex distributed logic for you.

When you use the Hugging Face `Trainer` or `SFTTrainer`, it uses `accelerate` under the hood automatically.

You do not need to change your Python code.

Your standard single-GPU training script will work on multiple GPUs without modification.

---

# 15. The Accelerate Library

To use DDP with Hugging Face, you use the `accelerate` CLI.

First, you must configure the environment.

You open your terminal and type:

```bash
accelerate config
```

This launches an interactive questionnaire.

It asks you questions about your hardware.

```text
In which compute environment are you running?
➔ This machine

Which type of machine are you using?
➔ multi-GPU

How many different machines will you use?
➔ 1

Do you wish to optimize your script with torch dynamo?
➔ No

How many GPU(s) should be used for distributed training?
➔ 4
```

Accelerate saves your answers to a configuration file.

This configuration file tells Hugging Face exactly how to set up the DDP process group.

Next, instead of running your script with `python`, you use `accelerate launch`.

```bash
accelerate launch my_sft_script.py
```

When you run this command, `accelerate` reads the config file.

It sees that you specified 4 GPUs.

It automatically spawns 4 separate Python processes.

It assigns the correct environment variables to each process.

It sets up the Ring-AllReduce communication channels.

Inside `my_sft_script.py`, the `SFTTrainer` detects that it was launched via `accelerate`.

It automatically wraps your model in PyTorch's `DistributedDataParallel`.

It automatically switches your dataloader to use a `DistributedSampler`.

It handles everything.

It is a seamless, magical experience.

---

# 16. The Global Batch Size Formula

While the code is handled for you, the mathematics of your hyperparameters are not.

When you move from a single GPU to multiple GPUs via DDP, your batch size changes fundamentally.

This is the most common mistake beginners make in distributed training.

You must understand the Global Batch Size formula.

Let us define our variables.

**Per-Device Batch:** The batch size you specify in your script (e.g., `per_device_train_batch_size = 4`).

**Gradient Accumulation:** The number of steps you accumulate gradients before updating (e.g., `gradient_accumulation_steps = 4`).

**Number of GPUs:** The number of GPUs in your DDP cluster (e.g., `world_size = 8`).

The formula for the true, effective batch size is:

$$
\text{Global Batch Size} = (\text{Per-Device Batch}) \times (\text{Gradient Accumulation}) \times (\text{Number of GPUs})
$$

Let us calculate this step by step.

If your script says `per_device_train_batch_size = 4`.

On a single GPU with no accumulation, your batch size is 4.

If you add `gradient_accumulation_steps = 4`.

Your batch size is now $4 \times 4 = 16$.

The single GPU processes 16 examples before updating its weights.

Now, you run this exact same script on an 8-GPU server using `accelerate launch`.

What happens?

Each of the 8 GPUs is processing a batch of 4.

Each of the 8 GPUs is accumulating for 4 steps.

So each GPU processes 16 examples independently.

Then, DDP synchronizes the gradients across all 8 GPUs.

The Ring-AllReduce averages the gradients computed from $16 \times 8$ examples.

Therefore, the global batch size is:

$$
4 \times 4 \times 8 = 128
$$

Your model is now updating its weights based on 128 examples at a time.

---

# 17. The Impact on Learning Rate

Why does the Global Batch Size matter?

It matters because it drastically alters the landscape of your loss function.

When your batch size is small (e.g., 4), the gradients are very noisy.

They point in erratic directions because they only represent a tiny sample of the data.

To compensate for this noise, you typically use a smaller learning rate.

You take small, cautious steps.

When your batch size is large (e.g., 128), the gradients are very stable.

They represent a much better approximation of the true gradient of the entire dataset.

Because the gradient is stable and reliable, you can take larger steps.

In fact, you MUST take larger steps.

If you use the same tiny learning rate with a massive batch size, your model will barely learn anything.

It will take forever to converge.

There is a mathematical rule of thumb in deep learning called the Linear Scaling Rule.

The Linear Scaling Rule states:

When you multiply your batch size by a factor of $K$, you should also multiply your learning rate by $K$.

For example, if you move from 1 GPU to 8 GPUs.

Your global batch size increased by a factor of 8.

Therefore, you should increase your learning rate by a factor of 8.

If your original learning rate was $2e-5$.

Your new learning rate should be $1.6e-4$.

This rule is not absolute, and modern optimizers like AdamW have complex dynamics that sometimes defy strict linear scaling.

However, the core principle remains intact.

If you expand to multiple GPUs, your global batch size increases.

If your global batch size increases, you must increase your learning rate.

If you forget this, your multi-GPU fine-tuning run will fail to converge.

You will waste thousands of dollars of compute time.

Always calculate your Global Batch Size before launching a distributed training run.

---

# 18. Summary

We have covered the fundamentals of distributed training.

You now understand why a single GPU is insufficient for massive datasets.

You understand the flaws of the Master-Worker architecture in Data Parallelism (DP).

You understand why Distributed Data Parallelism (DDP) is the modern standard.

You learned how DDP places identical replicas on every GPU.

You learned how the Distributed Sampler slices the data.

You explored the mathematical beauty of the Ring-AllReduce algorithm.

You saw how Hugging Face Accelerate makes launching DDP effortless.

And most importantly, you learned the Global Batch Size formula.

You now know how to scale your learning rate to ensure successful convergence on clusters of any size.

In the next lesson, we will explore even more advanced distributed techniques.

We will look at Fully Sharded Data Parallelism (FSDP).

FSDP allows us to break the model itself into pieces.

This will allow us to train models that are far larger than the memory of a single GPU.

But for now, mastering DDP is your key to unlocking the true power of multi-GPU training.
