# Lesson 28 — Distributed Training (FSDP & DeepSpeed)

> **Goal:** Understand Model Parallelism. Learn how to train models that are so large (e.g., 70B parameters) they cannot mathematically fit on a single GPU, using FSDP and DeepSpeed Zero.

---

## 1. The Limit of Data Parallelism (DDP)

In Lesson 27, we learned DDP. DDP requires a **full copy of the model** to exist on every GPU.
What happens if you want to fine-tune Llama 3 70B?
- 70B parameters in BF16 = **140 GB of VRAM**.
- The largest standard GPU (H100) only has **80 GB of VRAM**.

It is physically impossible to load the model onto a single GPU. DDP crashes instantly.

---

## 2. Model Parallelism (Sharding)

If the model is too big for one GPU, we must cut the model into pieces (Shards) and distribute the pieces across multiple GPUs.

### The Problem with Naive Pipeline Parallelism
Early attempts at Model Parallelism put Layer 1-10 on GPU 0, and Layer 11-20 on GPU 1.
The problem? GPU 1 sits completely idle (0% utilization) while waiting for GPU 0 to finish Layer 10. This is called the "Pipeline Bubble," and it wastes millions of dollars of compute.

---

## 3. The Breakthrough: ZeRO and FSDP

To solve this, Microsoft created **DeepSpeed ZeRO** (Zero Redundancy Optimizer), and Meta created **FSDP** (Fully Sharded Data Parallel). Both algorithms use the same brilliant mathematical concept: **Weight Sharding with Just-In-Time Gathering**.

### How FSDP Works
Instead of giving Layer 1 to GPU 0 and Layer 2 to GPU 1, FSDP takes the weights of a *single* layer and slices them across all GPUs.

1. **Sharding:** GPU 0 holds 25% of Layer 1, GPU 1 holds 25%, etc.
2. **Forward Pass (All-Gather):** When the forward pass reaches Layer 1, all GPUs rapidly share their 25% with each other over NVLink. For a fraction of a second, the full Layer 1 exists in VRAM.
3. **Compute:** The GPUs compute the matrix multiplication for Layer 1.
4. **Discard:** Immediately after computation, the GPUs delete the other 75% of the weights from VRAM to free up space.

```text
       FSDP Memory Allocation (4 GPUs)
       
[ Layer 1 Weights ] =  [ 25% ] + [ 25% ] + [ 25% ] + [ 25% ]
                         |         |         |         |
                      (GPU 0)   (GPU 1)   (GPU 2)   (GPU 3)
```

By sharding the Weights, Gradients, and Optimizer States, FSDP allows you to train massive models as long as the *total combined VRAM* of your cluster is larger than the model.

---

## 4. DeepSpeed ZeRO Stages

DeepSpeed offers three levels of sharding, known as ZeRO (Zero Redundancy Optimizer) stages.

- **ZeRO Stage 1:** Shards only the Optimizer States (Adam momentum/variance). Saves ~4x memory.
- **ZeRO Stage 2:** Shards Optimizer States + Gradients. Saves ~8x memory. (Standard for most fine-tuning).
- **ZeRO Stage 3:** Shards Optimizer States + Gradients + Model Weights. (Required for models > 30B parameters).

### Using DeepSpeed in Hugging Face

You don't need to rewrite your PyTorch code. You simply write a `deepspeed_config.json` file:

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

And launch your `SFTTrainer` script using the Accelerate / DeepSpeed CLI:
```bash
accelerate launch --use_deepspeed --deepspeed_config_file ds_config.json my_script.py
```
