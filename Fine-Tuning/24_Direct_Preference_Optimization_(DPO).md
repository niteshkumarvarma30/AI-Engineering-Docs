# Lesson 24 — Direct Preference Optimization (DPO)

> **Goal:** Understand how Direct Preference Optimization (DPO) bypasses complex Reinforcement Learning and mathematically simplifies preference alignment into a single loss function.

---

# 1. Introduction to Preference Alignment

Why do we need preference alignment?

Models trained with Supervised Fine-Tuning (SFT) just mimic data.

SFT models do not understand what is inherently "good" or "bad".

They only know what is "likely" based on the patterns in their training data.

If the training data contains toxic, biased, or unhelpful responses, the SFT model might generate those exact types of responses.

We need a way to tell the model our preferences.

We need a mechanism to align the model's outputs with human values.

This critical process is known in the industry as Preference Alignment.

Preference Alignment is what transforms a raw text predictor into a helpful, harmless, and honest AI assistant.

---

# 2. The Old Way: RLHF

Before the year 2023, the industry standard method for preference alignment was RLHF.

RLHF stands for Reinforcement Learning from Human Feedback.

RLHF is incredibly complex and notorious to implement correctly.

It is not a single training step.

It requires multiple separate stages of training.

It requires maintaining multiple different neural networks simultaneously.

Let us look at the traditional RLHF pipeline.

```text
=========================================
      THE TRADITIONAL RLHF PIPELINE
=========================================

Step 1: Supervised Fine-Tuning (SFT)
  (Train a base model to follow instructions)
                 ↓
Step 2: Reward Model Training (RM)
  (Train a separate model to score responses)
                 ↓
Step 3: Proximal Policy Optimization (PPO)
  (Use RL to optimize the SFT model using RM)
```

Each of these steps is a massive undertaking on its own.

Each step introduces new hyperparameters, new points of failure, and new infrastructure requirements.

---

# 3. Breaking Down RLHF Step-by-Step

First, you train the base model using Supervised Fine-Tuning (SFT).

This gives you an instruction-tuned model.

The model can now follow instructions.

But it is not yet aligned to nuanced human preferences.

Second, you must train a Reward Model.

The Reward Model is a completely separate neural network.

It is usually the exact same size as the SFT model.

Its job is to look at a prompt and a response.

And then it outputs a single scalar score indicating the quality of the response.

A high score means the response is highly desired by humans.

A low score means the response is poor, unsafe, or incorrect.

```text
-----------------------------------------
           REWARD MODEL FLOW
-----------------------------------------

Prompt: "How do I make a bomb?"
Response: "I cannot help with that."
                 ↓
           Reward Model
                 ↓
         Score: +5.8 (Good!)

-----------------------------------------
```

Third, you use an algorithm called PPO.

PPO stands for Proximal Policy Optimization.

PPO is a complex Reinforcement Learning algorithm.

PPO uses the Reward Model to actively score the SFT model's outputs during training.

PPO then updates the SFT model's weights to maximize that expected score.

---

# 4. The Problem with RLHF

Make no mistake, RLHF works.

It is the exact technology that created the original version of ChatGPT.

But it is a complete nightmare to engineer and scale.

Why is it such a nightmare?

First, you need to load multiple massive models into memory at the exact same time.

You need the Policy Model, which is the one you are actively training.

You need the Reward Model, which is providing the scores.

You need the Reference Model, which is a frozen copy of the Policy Model to prevent it from drifting too far.

You also need the Value Model, which is used internally by the PPO algorithm to estimate future rewards.

This requires a truly massive amount of VRAM.

```text
=========================================
      VRAM REQUIREMENTS FOR RLHF
=========================================

1. Policy Model    [████████] Trainable
2. Value Model     [████████] Trainable
3. Reward Model    [████████] Frozen
4. Reference Model [████████] Frozen

Total VRAM = 4x the base model size!
=========================================
```

If you are training a 7 Billion parameter model, you need enough VRAM to hold four 7 Billion parameter models in memory!

Second, the PPO algorithm is highly unstable.

The hyperparameters are extremely sensitive to minor changes.

If your learning rate is slightly off, the entire model collapses into generating gibberish.

If your Reward Model is not absolutely perfect, the Policy Model learns to mathematically exploit it.

This phenomenon is called Reward Hacking.

In Reward Hacking, the model starts outputting nonsense that somehow triggers a high score from the imperfect Reward Model.

The AI community desperately needed a better way.

---

# 5. The Mathematical Breakthrough of DPO

In the year 2023, researchers at Stanford University published a groundbreaking paper.

The paper introduced a concept called Direct Preference Optimization.

Direct Preference Optimization is abbreviated as DPO.

DPO entirely eliminates the need for a separate Reward Model.

DPO entirely eliminates the need for the complex PPO algorithm.

DPO accomplishes preference alignment in a single, elegant training step.

This was a massive paradigm shift in AI engineering.

How did the researchers accomplish this?

They made a brilliant mathematical observation.

They mathematically proved that the Language Model itself inherently acts as a reward model.

---

# 6. The Implicit Reward Model

What does it mean that the Language Model is its own reward model?

Think about how an autoregressive Large Language Model fundamentally works.

An LLM assigns explicit probabilities to sequences of tokens.

If an LLM thinks a sentence is highly likely, it assigns that sequence a high probability.

If an LLM thinks a sentence is unlikely or unnatural, it assigns it a low probability.

The Stanford researchers realized something profound.

You can mathematically map the concept of a "Reward Model" directly to the Language Model's implicit token probabilities.

You do not need a separate, explicitly trained neural network to output a scalar score.

The raw probability that the LLM assigns to a response *is* its implicit score!

By reformulating the math, this completely eliminates the Reward Model from the pipeline.

---

# 7. The Two Models of DPO

To implement DPO, you only need two models loaded into memory.

You do not need four models like you do in traditional RLHF.

First, you need the Policy Model.

Second, you need the Reference Model.

```text
=========================================
      VRAM REQUIREMENTS FOR DPO
=========================================

1. Policy Model    [████████] Trainable
2. Reference Model [████████] Frozen

Total VRAM = 2x the base model size!
(And even less with LoRA!)
=========================================
```

The Policy Model is the active model you are training and updating.

We denote the Policy Model mathematically as:

$$
\pi_\theta
$$

Here, $\pi$ stands for policy.

And $\theta$ stands for the trainable weights or parameters of the neural network.

The Reference Model is a strictly frozen copy of the model.

It is an exact clone of the model *before* DPO training begins.

We denote the Reference Model mathematically as:

$$
\pi_{\text{ref}}
$$

Here, $\text{ref}$ stands for reference.

It has absolutely zero trainable parameters.

Its weights are frozen in place and will never change during the training loop.

---

# 8. Why Do We Need a Reference Model?

Why can we not just train the Policy Model completely by itself?

Because if we only tell the Policy Model to maximize the probability of good responses without any anchor...

...it will eventually destroy its own internal representation of grammar and language.

It will try to maximize the mathematical probability so aggressively that it forgets how to speak English.

It might find a single "good" token and just output it repeatedly forever.

The Reference Model acts as an essential mathematical anchor.

It prevents the Policy Model from deviating too radically from its original linguistic behavior.

We want the Policy Model to output better, more aligned responses.

But we want it to still sound like the original, coherent SFT model.

---

# 9. The DPO Dataset Format

Before we can dive into the exact math, we must look at the data format.

DPO requires a very specific, strict dataset structure.

It requires pairwise human preferences.

For every single training prompt, you must provide exactly two responses.

One response is the "Chosen" response.

The Chosen response is the good, helpful, or safe response.

The other response is the "Rejected" response.

The Rejected response is the bad, unhelpful, or toxic response.

```text
=========================================
           DPO DATASET ROW
=========================================

[Prompt]
"Write a python loop to print 1 to 5."

[Chosen]
"Here is the loop:
for i in range(1, 6):
    print(i)"

[Rejected]
"for i in range(1,6): print(i)"

=========================================
```

We want the model to strongly prefer the Chosen response.

We want the model to actively avoid the Rejected response.

In our mathematical formulas, we denote the prompt as:

$$
x
$$

We denote the Chosen response as:

$$
y_w
$$

Here, the subscript $w$ stands for "win" or "winner".

We denote the Rejected response as:

$$
y_l
$$

Here, the subscript $l$ stands for "lose" or "loser".

---

# 10. The Core Objective of DPO

What exactly are we trying to achieve mathematically during training?

When we feed the prompt $x$ and the Chosen response $y_w$ into the models...

We want our active Policy Model to assign a HIGHER probability to it than the frozen Reference Model does.

When we feed the prompt $x$ and the Rejected response $y_l$ into the models...

We want our active Policy Model to assign a LOWER probability to it than the frozen Reference Model does.

We are actively comparing probabilities across the two models.

We are measuring how much the Policy Model's beliefs have shifted away from the Reference Model.

---

# 11. The Probability Ratio

To mathematically compare the two models, we formulate a ratio.

Let us look at the ratio for the Chosen response:

$$
\frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)}
$$

The numerator is the Policy Model's predicted probability for the Chosen response given the prompt.

The denominator is the Reference Model's predicted probability for the Chosen response given the prompt.

If the Policy Model likes the Chosen response MORE than the Reference Model likes it...

...then the numerator is larger, and the ratio will be greater than 1.

Our goal during training is to force this ratio to go UP.

Now let us look at the exact same ratio, but for the Rejected response:

$$
\frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}
$$

The numerator is the Policy Model's predicted probability for the Rejected response.

The denominator is the Reference Model's predicted probability for the Rejected response.

If the Policy Model likes the Rejected response LESS than the Reference Model likes it...

...then the numerator is smaller, and the ratio will be less than 1.

Our goal during training is to force this ratio to go DOWN.

---

# 12. Taking the Logarithm

In machine learning, we rarely work with raw probability values.

Raw probabilities in language models are astronomically small numbers.

Multiplying many extremely small numbers together quickly leads to numerical underflow in floating-point arithmetic.

To solve this, we take the natural logarithm of the values.

We take the log of our ratios.

For the Chosen response ratio:

$$
\log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)}
$$

For the Rejected response ratio:

$$
\log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}
$$

Taking the logarithm does not change our fundamental goal.

Logarithms are monotonically increasing functions.

We still aggressively want the Chosen log-ratio to go UP.

We still aggressively want the Rejected log-ratio to go DOWN.

---

# 13. The Beta Parameter

Next, we must introduce a crucial hyperparameter called Beta.

Beta is denoted mathematically as:

$$
\beta
$$

Beta acts as a temperature parameter in the DPO formula.

It explicitly controls how much the Policy Model is mathematically allowed to deviate from the Reference Model.

If Beta is set very high, the Policy Model is heavily penalized for changing, and it stays very close to the Reference Model.

If Beta is set very low, the Policy Model is given free rein to change drastically.

We multiply both of our log-ratios by this Beta parameter.

For the Chosen response term:

$$
\beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)}
$$

For the Rejected response term:

$$
\beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}
$$

In modern DPO setups, a very typical and stable value for Beta is 0.1.

---

# 14. Combining the Terms into a Reward Margin

Now we must combine the Chosen term and the Rejected term into a single metric.

We want the Chosen term to be significantly larger than the Rejected term.

To measure this, we simply subtract the Rejected term from the Chosen term.

$$
\left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right)
$$

This combined value represents our implicit "Reward Margin".

If this value is positive and very large, it means the Policy Model is doing a fantastic job.

It means the Policy Model prefers the Chosen response vastly more than it prefers the Rejected response.

If this value is negative, it means the Policy Model is failing and actually prefers the bad response.

---

# 15. The Sigmoid Function

We now have a combined numerical value representing the margin.

But loss functions in machine learning almost always need to operate on probabilities bounded between 0 and 1.

To convert our unbounded margin value into a strict probability, we use the Sigmoid function.

The Sigmoid function is denoted by the Greek letter sigma:

$$
\sigma
$$

The Sigmoid function takes any real number and elegantly squashes it into a curve between 0 and 1.

```text
=========================================
           SIGMOID FUNCTION
=========================================

       1.0 |            .-------
           |          .´
       0.5 |--------.´----------
           |      .´
       0.0 |____.´______________
           
           Negative      Positive
            Margin        Margin
=========================================
```

We wrap our entire mathematical expression inside the Sigmoid function.

$$
\sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right)
$$

Now we finally have a clean probability.

A high probability (close to 1.0) means the model is correctly assigning a much higher likelihood to the Chosen response.

A low probability (close to 0.0) means the model is incorrectly favoring the Rejected response.

---

# 16. The Final DPO Loss Function

In machine learning frameworks like PyTorch, optimizers are designed to minimize loss.

We want the probability of our model being correct to be as high as possible.

Therefore, we want the negative logarithm of that probability to be as low as possible.

This is the standard cross-entropy logic used across all of deep learning.

So, we take the negative logarithm of our entire Sigmoid expression.

This gives us the final, complete DPO Loss Function!

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right)
$$

This is the exact, unsimplified formula published in the Stanford paper.

Let us review the entire formula one more time, piece by piece.

$\mathcal{L}_{\text{DPO}}$ is the final loss scalar that we are trying to minimize.

$-\log$ turns our probability of being correct into a minimizable loss value.

$\sigma$ turns our raw mathematical difference into a bounded probability.

$\beta$ controls how much the model is penalized for drifting from the reference.

The first inner term calculates how much more the model likes the Chosen response compared to the Reference.

The second inner term calculates how much more the model likes the Rejected response compared to the Reference.

By minimizing this exact loss, PyTorch calculates precise gradients.

Those gradients flow backwards through the network to update the weights of the Policy Model.

The Policy Model becomes perfectly aligned, without ever needing PPO!

---

# 17. Visualizing the DPO Training Loop

Let us visualize exactly what happens inside the GPU during a single step of DPO training.

```text
=========================================
          THE DPO TRAINING STEP
=========================================

[Step 1: Get Batch]
Prompt: "Hello"
Chosen: "Hi there!"
Rejected: "What."

[Step 2: Forward Pass (Reference Model)]
Input: Prompt + Chosen
Output: Ref Chosen Logits
Input: Prompt + Rejected
Output: Ref Rejected Logits
* (No gradients calculated. Extremely fast.)

[Step 3: Forward Pass (Policy Model)]
Input: Prompt + Chosen
Output: Policy Chosen Logits
Input: Prompt + Rejected
Output: Policy Rejected Logits
* (Gradients are actively tracked here.)

[Step 4: Calculate DPO Loss]
Plug all 4 outputs into the math formula.
Multiply by Beta.
Apply Sigmoid.
Take negative log.
Result: Loss Scalar (e.g., 0.34)

[Step 5: Backward Pass]
Loss.backward()
Optimizer.step()
Update Policy Model weights!

=========================================
```

This training loop is incredibly stable and predictable.

There is no PPO algorithm guessing value functions.

There is no separate Reward Model hallucinating scores.

It is just standard, robust backpropagation operating over a brilliantly specialized loss function.

---

# 18. DPO in Practice with Hugging Face

How do we actually write the Python code to execute DPO?

We do not have to write the complex math from scratch.

Hugging Face provides an incredible library called TRL.

TRL stands for Transformer Reinforcement Learning.

Even though DPO is not traditional Reinforcement Learning, it is bundled inside TRL.

TRL provides a highly optimized class called `DPOTrainer`.

It is functionally very similar to the `SFTTrainer` class you have used previously.

---

# 19. Preparing the DPO Dataset

First, we must format our dataset exactly how TRL expects it.

As we discussed in the theory section, it needs three specific columns.

The columns must be named exactly: `prompt`, `chosen`, and `rejected`.

```python
dataset = [
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris.",
        "rejected": "Paris."
    },
    {
        "prompt": "Write a python loop.",
        "chosen": "Here is a loop: \n```python\nfor i in range(10):\n    print(i)\n```",
        "rejected": "for i in range(10): print(i)"
    }
]
```

Notice that both the chosen and rejected fields contain the complete, raw strings.

You do not need to tokenize them manually.

The `DPOTrainer` will automatically handle tokenizing and formatting them using your model's chat template.

---

# 20. Loading the Models into Memory

Next, we need to load our models into GPU VRAM.

Remember the cardinal rule of DPO: we need TWO distinct models loaded.

We need the Policy Model.

And we need the Reference Model.

We use the standard `AutoModelForCausalLM` class from the `transformers` library to load them.

```python
import torch
from transformers import AutoModelForCausalLM

# 1. Load the Policy Model (the one we will actively train)
model = AutoModelForCausalLM.from_pretrained(
    "my-sft-model", 
    device_map="auto"
)
```

Now we load the Reference Model right next to it.

We load it from the exact same file path as the Policy Model.

It must start its life as an identical, perfect copy of the Policy Model.

```python
# 2. Load the Reference Model (the frozen anchor copy)
ref_model = AutoModelForCausalLM.from_pretrained(
    "my-sft-model", 
    device_map="auto"
)
```

You do not need to write code to freeze the Reference Model's gradients.

The `DPOTrainer` is smart enough to automatically call `ref_model.eval()` and disable gradient tracking on it internally.

---

# 21. Setting up LoRA (Low-Rank Adaptation)

In the real world, we almost never do full-parameter fine-tuning with DPO.

Full fine-tuning requires too much VRAM and is highly prone to catastrophic forgetting.

We almost exclusively use LoRA.

LoRA drastically saves VRAM and acts as a natural regularizer to prevent overfitting.

We define our LoRA configuration using the `peft` library.

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
```

This configuration tells the PEFT library to inject small trainable LoRA adapters into the attention layers.

The `DPOTrainer` will automatically apply this exact configuration to the Policy Model.

The Reference Model will remain completely untouched by PEFT, preserving its frozen state.

---

# 22. Defining the Training Arguments

We need to configure our standard training hyperparameters.

We use the `TrainingArguments` class from the core `transformers` library.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./dpo_results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    # === CRITICAL SETTING FOR DPO BELOW ===
    remove_unused_columns=False
)
```

Why is the `remove_unused_columns=False` setting absolutely crucial?

By default, the Hugging Face standard Trainer aggressively deletes any columns in your dataset that are not named `input_ids`, `labels`, or `attention_mask`.

It does this to save memory during normal training.

But the custom `DPOTrainer` specifically needs the raw string columns named `prompt`, `chosen`, and `rejected` to do its dynamic formatting.

If you let the Trainer delete them, the DPO process will immediately crash with a KeyError.

You must explicitly set `remove_unused_columns=False` to prevent this disaster.

---

# 23. Initializing the DPOTrainer Class

Now we bring all the individual pieces together.

We initialize the powerful `DPOTrainer` class from TRL.

```python
from trl import DPOTrainer

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    beta=0.1,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config
)
```

Let us meticulously review every single argument being passed in here.

`model=model`: This is our active Policy Model that will receive the LoRA adapters and get its weights updated.

`ref_model=ref_model`: This is our strictly frozen Reference Model that acts as the mathematical anchor.

`args=training_args`: These are our learning rates, batch sizes, and directory paths.

`beta=0.1`: This is the exact $\beta$ from our mathematical formula, controlling the deviation penalty!

`train_dataset=dpo_dataset`: This is our pairwise dataset containing the prompt, chosen, and rejected columns.

`tokenizer=tokenizer`: The tokenizer that will convert our raw strings into integer IDs based on the chat template.

`peft_config=peft_config`: This instructs the trainer to automatically wrap the Policy Model in LoRA before training begins.

---

# 24. Executing the Training Run

Finally, we start the actual optimization process.

```python
dpo_trainer.train()
```

When you execute this command, your GPU will spin up and training will commence.

In your console logs, you will see the loss steadily decreasing over time.

You will also see a highly important secondary metric in the logs: **Reward Margin**.

The Reward Margin is the numerical difference between the implicit reward of the Chosen response and the Rejected response.

We desperately want the Reward Margin to continually increase over time.

An increasing Reward Margin definitively proves that the Policy Model is successfully learning to prefer the Chosen response over the Rejected response.

---

# 25. Conclusion and Impact

Direct Preference Optimization has completely revolutionized the field of AI alignment.

It has almost entirely replaced RLHF in the open-source community, and increasingly in enterprise labs.

It provides absolute mathematical stability by reducing a multi-stage reinforcement learning problem into a single binary cross-entropy loss function.

It requires significantly less VRAM because it eliminates the explicit Reward Model and Value Model.

It requires vastly less hyperparameter tuning because it bypasses PPO entirely.

By brilliantly realizing that the Language Model is mathematically identical to its own Reward Model, DPO elegantly solves the preference alignment problem.

You have now thoroughly learned both the underlying calculus and the practical implementation behind Direct Preference Optimization.
