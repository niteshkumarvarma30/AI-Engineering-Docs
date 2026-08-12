# Lesson 27 — Multi-GPU Fine-Tuning

> **Goal:** Understand how to move beyond a single GPU. Learn the mechanics of Data Parallelism (DP) and Distributed Data Parallelism (DDP) for faster training.

---

## 1. The Need for Multiple GPUs

Even with memory-saving techniques like QLoRA and Unsloth, a single GPU can only process so much data per second. 
If you have a dataset of 1 million conversations, fine-tuning on a single RTX 4090 might take 3 weeks.

To reduce training time from weeks to hours, we must distribute the workload across multiple GPUs (e.g., an 8x H100 server).

---

## 2. Data Parallelism (DP) vs. Distributed Data Parallelism (DDP)

There are two fundamental ways to split a dataset across GPUs.

### A. Data Parallelism (DP - The Old Way)
In DP, one GPU acts as the "Master."
1. The Master GPU holds the main model.
2. It copies the model to GPU 1, 2, and 3.
3. It splits the batch of data (e.g., 8 rows) and sends 2 rows to each GPU.
4. Each GPU calculates gradients and sends them *back* to the Master GPU.
5. The Master GPU averages the gradients, updates the weights, and copies the new model back to the workers.

**The Flaw:** The Master GPU becomes a massive bottleneck. It runs out of memory quickly, and the constant copying across the PCIe bus destroys performance.

### B. Distributed Data Parallelism (DDP - The Modern Standard)
In DDP, there is no "Master" bottleneck.
1. A complete, identical copy of the model is placed on every GPU.
2. The dataset is chunked (using a `DistributedSampler`). GPU 0 gets Chunk 0, GPU 1 gets Chunk 1.
3. Every GPU computes its gradients independently.
4. **The Magic (Ring-AllReduce):** The GPUs use high-speed interconnects (NVLink) to mathematically synchronize their gradients in a circle. They average the gradients simultaneously without a master node.
5. Every GPU updates its own identical copy of the model weights.

---

## 3. Implementing DDP in Hugging Face

Hugging Face's `Trainer` and `SFTTrainer` support DDP out of the box using the `accelerate` library.

Instead of writing complex PyTorch multiprocessing code, you simply configure Accelerate via the CLI.

```bash
# 1. Configure the environment (tells the system how many GPUs you have)
accelerate config

# 2. Launch your standard Python script
accelerate launch my_sft_script.py
```

Inside your `my_sft_script.py`, you don't need to change much. Hugging Face automatically detects that Accelerate launched the script and switches the underlying PyTorch backend to use `DistributedDataParallel`.

---

## 4. The Global Batch Size Formula

When using DDP, your effective batch size changes. It is a critical math formula you must understand to prevent destroying your learning rate:

$$
\text{Global Batch Size} = (\text{Per-Device Batch}) \times (\text{Gradient Accumulation}) \times (\text{Number of GPUs})
$$

If you use a batch size of `4`, gradient accumulation of `4`, and `8` GPUs, your model is actually updating its weights based on `128` examples at a time. 
You must scale your Learning Rate accordingly, or the model will fail to converge!
