# Lesson 25 — Reinforcement Learning from Human Feedback (RLHF)

> **Goal:** Understand the classical RLHF pipeline used by OpenAI to create ChatGPT, including the three-step process: SFT, Reward Modeling, and PPO (Proximal Policy Optimization).

---

# 1. What Is RLHF?

**RLHF = Reinforcement Learning from Human Feedback.**

It is the algorithm that turned raw language models into ChatGPT.

Before RLHF, large language models were simply text predictors.

They were trained to predict the next token on the internet.

They were incredibly smart.

They memorized vast amounts of human knowledge.

But they were largely unusable by the general public.

If you asked them a question, they might answer.

Or they might ask you another question.

Or they might generate a toxic rant.

RLHF introduced a way to align these models with human intentions.

It bridges the gap between next-token prediction and helpful assistance.

It is the secret sauce behind the AI revolution.

To understand modern AI engineering, you must understand RLHF.

It is the foundational architecture of alignment.

---

# 2. The Alignment Problem

A base language model is trained on the entire internet.

The internet contains highly contradictory text.

It contains toxic text.

It contains chaotic text.

Therefore, the base model contains these same chaotic distributions.

If you prompt a base model with a simple question:

```text
Prompt:
How do I bake a cake?
```

The model might respond with an answer.

Or it might respond with another question.

```text
Response:
How do I bake a pie?
```

Why does it do this?

Because on forums like Quora or Reddit, questions are often followed by other questions.

The model is just mimicking the distribution of internet forums.

This is called the **Alignment Problem**.

We want the model to act as a helpful assistant.

We want it to be harmless.

We want it to be honest.

But its base objective is merely to mimic the internet.

We must alter its behavior without destroying its knowledge.

RLHF is the mathematical solution to the Alignment Problem.

---

# 3. Why Not Just DPO?

In Lesson 24, we studied Direct Preference Optimization (DPO).

We learned that DPO mathematically bypasses the need for a separate reward model.

DPO treats the language model itself as the reward model.

It optimizes preferences directly using a simple cross-entropy loss.

So why study RLHF?

Why do frontier labs like OpenAI, Anthropic, and DeepMind still use RLHF?

The answer is nuance.

Explicitly training a separate Reward Model has unique advantages.

A Reward Model can capture complex human preferences better than implicit probabilities.

A separate Reward Model can generalize across different tasks.

It can serve as a standalone "Critic" to evaluate thousands of models.

It can be used to filter datasets.

It can be used for search and inference-time scaling.

While DPO is simpler and cheaper to implement.

RLHF often scales better at the absolute frontier of AI.

RLHF remains the gold standard for state-of-the-art alignment.

You cannot consider yourself an AI engineer without mastering it.

---

# 4. The Three-Step Pipeline

RLHF is not a single algorithm.

It is a massive, complex engineering pipeline.

It requires coordinating multiple neural networks.

It requires managing vast datasets of human annotations.

It consists of three distinct steps.

```text
Step 1:
Supervised Fine-Tuning (SFT)

Step 2:
Reward Modeling (RM)

Step 3:
Proximal Policy Optimization (PPO)
```

Each step builds upon the previous one.

You cannot run PPO without a Reward Model.

You cannot train a Reward Model without an SFT model.

If any step fails, the entire pipeline collapses.

Let's break down each step in excruciating detail.

We will start with Step 1.

---

# 5. Step 1: Supervised Fine-Tuning (SFT)

Before a model can be aligned, it must first be able to follow instructions.

We start with a base model.

This base model only knows how to predict the next token.

It does not know what an "assistant" is.

We collect thousands of high-quality, human-written demonstrations.

These demonstrations show the model exactly how to behave.

They define the desired format and tone.

```text
Prompt:
Write a polite email to decline an invitation.

Response:
Dear John, thank you so much for the invitation. Unfortunately...
```

We fine-tune the base model on this dataset.

We use standard cross-entropy loss.

This is called Supervised Fine-Tuning (SFT).

We covered the math of SFT extensively in Lesson 16.

The model learns to clone the behavior of the human writers.

After this step, the model is a competent assistant.

But it is far from perfect.

---

# 6. SFT Is Still Next-Token Prediction

This is a crucial point to remember.

SFT does not change the fundamental architecture of the model.

The model is still an autoregressive transformer.

It is still just predicting the next token.

The only difference is the data distribution.

Instead of predicting the next token of a random internet page.

It is predicting the next token of a helpful assistant's response.

Because of this, SFT inherits the limitations of next-token prediction.

It suffers from exposure bias.

During training, the model only sees perfect human text.

During inference, it must condition on its own generated text.

If it makes a small mistake, it has never learned how to recover.

This leads to compounding errors.

---

# 7. The Limitations of SFT

SFT is powerful.

It establishes the baseline behavior of the model.

But it has a fatal flaw.

SFT is fundamentally just **behavioral cloning**.

The model is forced to memorize and mimic the exact words the human wrote.

But language is highly subjective.

There are many valid ways to write an email.

If the model deviates slightly from the target response, SFT penalizes it heavily.

The Cross-Entropy Loss forces exact token matching.

Furthermore, human writers make mistakes.

If the SFT dataset contains hallucinations, the model learns to hallucinate.

If the dataset contains biases, the model learns those biases.

SFT cannot teach the model *why* an answer is good.

It only teaches the model to blindly copy.

We need a way to let the model explore different answers.

We need to let it learn which answers are best, rather than dictating the exact tokens.

This requires Reinforcement Learning.

---

# 8. Step 2: Training the Reward Model (RM)

To use Reinforcement Learning, we need a reward signal.

In a video game like Mario, the reward signal is the game score.

If Mario collects a coin, the score goes up.

The reward is positive.

If Mario falls in a pit, the score goes down.

The reward is negative.

But in language generation, there is no inherent "score".

What is the exact numerical score of a poem?

What is the mathematical value of a polite email?

Does a funny joke get a +5 or a +10?

There is no objective mathematical function for human language.

Therefore, we must create an artificial scoring system.

We must create a model that can provide this score.

This is the **Reward Model**.

The Reward Model acts as an automated human critic.

It reads text and assigns a scalar value representing human preference.

---

# 9. Why We Need a Reward Model

Why can't we just have humans score the outputs during PPO?

Because PPO requires millions of generations.

Humans are too slow.

Humans are too expensive.

We need a scalable, automated way to evaluate text.

The Reward Model serves as a proxy for human judgment.

It learns the underlying patterns of what humans prefer.

Once trained, it can evaluate text in milliseconds.

This allows the Reinforcement Learning loop to run at computational speeds.

---

# 10. Collecting Human Preference Data

To train a Reward Model, we first need human preferences.

We cannot train a critic without examples of what to criticize.

We take our newly trained SFT model.

We give it a prompt from our dataset.

```text
Prompt:
Explain quantum mechanics.
```

We ask the SFT model to generate multiple different responses.

Let's say it generates Response A and Response B.

```text
Response A:
Quantum mechanics is the study of subatomic particles...

Response B:
Quantum mechanics is a field of physics that is really confusing...
```

We present both responses to a human labeler.

The human labeler reads both responses carefully.

The human labeler decides which response is better.

```text
Human Decision:
Response A > Response B
```

This forms a single preference pair.

---

# 11. Pairwise Comparisons

Notice that the human does not give a numerical score.

Humans are terrible at giving consistent numerical scores.

If you ask a human to rate a response from 1 to 10, they are inconsistent.

They might say 7 today and 5 tomorrow.

One labeler's 7 is another labeler's 4.

But humans are very good at pairwise comparisons.

It is much easier to say "A is better than B".

This dataset of comparisons is called human preference data.

We collect hundreds of thousands of these comparisons.

This dataset forms the foundation of the Reward Model.

---

# 12. Initializing the Reward Model

We now have thousands of pairwise comparisons.

We need to train a neural network to predict these preferences.

We start by initializing the Reward Model.

Typically, we initialize the Reward Model using the exact weights of the SFT model.

Why?

Because the SFT model already understands language perfectly.

It already knows how to process prompts and responses.

It already has rich representations of syntax and semantics.

It is much easier to teach a fluent model to be a critic than to train a critic from scratch.

---

# 13. The Scalar Head

However, the SFT model outputs a probability distribution over the entire vocabulary.

We do not want a vocabulary distribution.

We want a single scalar number.

So, we remove the final Language Modeling head.

We throw away the unembedding matrix.

We replace it with a simple linear regression head.

```text
[Transformer Layers] ---> [LM Head] ---> (Vocab Probabilities)
       |
       | (Replace head)
       V
[Transformer Layers] ---> [Scalar Head] ---> (Single Float Score)
```

Now, the model takes a sequence of text and outputs a single number.

It outputs this number for the final token of the response.

For example:

```text
Input:
"Explain math" + "Clear, polite response"
       ↓
Reward Model
       ↓
Output: +4.5
```

The magnitude of the output represents the quality of the response.

---

# 14. The Bradley-Terry Model

How do we train this scalar output using pairwise preferences?

We use the Bradley-Terry model.

The Bradley-Terry model is a probabilistic framework.

It predicts the outcome of a paired comparison.

Suppose we have Response A and Response B.

The Reward Model evaluates Response A.

It gives Response A a scalar score.

Let's call this score:

$$
r(x, y_A)
$$

The Reward Model evaluates Response B.

It gives Response B a scalar score.

Let's call this score:

$$
r(x, y_B)
$$

---

# 15. The Math of Bradley-Terry

The Bradley-Terry model defines the probability that A is preferred to B.

It states that this probability is the sigmoid of the difference in their scores.

$$
P(A > B) = \sigma(r(x, y_A) - r(x, y_B))
$$

Where the sigmoid function is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

If the score of A is much higher than the score of B, the difference is large and positive.

The sigmoid approaches 1.

The probability that A beats B approaches 100%.

If the score of B is much higher, the difference is negative.

The sigmoid approaches 0.

If the scores are exactly equal, the difference is 0.

The sigmoid of 0 is 0.5.

The probability is exactly 50%.

---

# 16. The Reward Model Loss Function

We want to train the Reward Model to output scores that align with human preferences.

If the human preferred A over B, we want:

$$
r(x, y_A) > r(x, y_B)
$$

We want to maximize the probability that the preferred response beats the rejected response.

To maximize this probability, we minimize the negative log-likelihood.

This gives us the Reward Model Loss function:

$$
\mathcal{L}_{RM} = - \log \sigma(r(x, y_w) - r(x, y_l))
$$

Where:

- \( y_w \) is the winning response chosen by the human.
- \( y_l \) is the losing response rejected by the human.

Let's break down exactly what happens during a forward pass.

The model computes the score for the winner.

The model computes the score for the loser.

We subtract the loser's score from the winner's score.

We pass this difference through a sigmoid function.

We take the negative logarithm.

We backpropagate this loss through the Reward Model.

Over thousands of steps, the Reward Model learns a complex continuous landscape.

It learns to assign high scalar values to good, helpful text.

It learns to assign low scalar values to toxic, unhelpful text.

---

# 17. Step 3: Proximal Policy Optimization (PPO)

We now have two distinct, fully trained models.

1. Our **Policy Model**. This is the language model we actually want to align.

It is initialized from the SFT model.

2. Our **Reward Model**. This is the critic we just trained.

It is frozen.

Its weights will not change anymore.

We will now use Reinforcement Learning to update the Policy Model.

The specific algorithm used is called Proximal Policy Optimization (PPO).

---

# 18. Actor-Critic Architecture

PPO is a highly stable actor-critic RL algorithm.

The Policy Model is the actor.

It takes actions in the environment by generating tokens.

The Reward Model is the critic.

It evaluates the actor's actions and provides feedback.

PPO allows the model to explore new sequences of tokens that were never in the SFT dataset.

It discovers novel ways to maximize the human preference score.

This is the power of reinforcement learning.

It moves beyond mere imitation.

---

# 19. The PPO Training Loop

The PPO loop is a continuous cycle of generation and evaluation.

Let's walk through a single iteration of the loop.

First, we sample a prompt from our training dataset.

```text
Prompt:
Write a joke.
```

We pass this prompt to our Policy Model.

The Policy Model autoregressively generates a complete response.

It generates tokens one by one until it hits an EOS token.

```text
Response:
Why did the chicken cross the road? To get to the other side.
```

---

# 20. Generation and Evaluation

We now need to evaluate this generated response.

We concatenate the original prompt and the newly generated response.

We pass them into our frozen Reward Model.

```text
[Prompt + Response]
       ↓
Frozen Reward Model
       ↓
Score: +3.2
```

The Reward Model gives us a scalar score.

This score is our **Reward Signal**.

We use this Reward Signal to update the weights of the Policy Model.

```text
[Prompt] ---> (Policy Model) ---> [Response]
                                      |
                                      v
                                (Reward Model) ---> [Score: +3.2]
                                      |
       Update Weights via PPO <-------+
```

If the score is highly positive, PPO updates the Policy Model's weights.

It modifies the gradients to make that specific sequence of tokens more likely in the future.

If the score is negative, PPO updates the weights in the opposite direction.

It suppresses that behavior.

This loop repeats millions of times.

---

# 21. Reward Hacking

This loop seems perfect in theory.

But in practice, there is a massive problem.

Neural networks are lazy, optimizing machines.

If you optimize a model purely for a scalar reward, it will find a shortcut.

It will exploit flaws in the Reward Model.

This phenomenon is called **Reward Hacking**.

Suppose our Reward Model learned that human labelers really like polite words.

The Policy Model will quickly discover this correlation during PPO exploration.

The Policy Model might generate a response like:

```text
Response:
Please please please please please please.
```

---

# 22. Mode Collapse

The frozen Reward Model looks at this text.

It doesn't understand context deeply enough to know it's garbage.

It just sees a high density of polite words.

It says, "Wow, look at all those polite words! This is the best response ever!"

```text
Score: +99.9
```

The Policy Model receives a massive reward.

It updates its weights to output more garbage.

The model completely loses its ability to speak coherent English.

This catastrophic failure is called **Mode Collapse**.

We must prevent the Policy Model from destroying its own language capabilities in pursuit of a high score.

---

# 23. The KL Divergence Penalty

To prevent Reward Hacking, we introduce a mathematical penalty.

We want the Policy Model to maximize the reward.

But we also want the Policy Model to stay close to its original, coherent self.

Remember, the Policy Model started as the SFT model.

The SFT model speaks perfect English.

It knows how to be an assistant.

So, we keep a frozen copy of the original SFT model in memory.

We call this the Reference Model.

During the PPO loop, every time the Policy Model generates a token, we check the Reference Model.

We ask the Reference Model: "What probability would you have assigned to this token?"

We compare the probability distribution of the Policy Model with the Reference Model.

We measure the mathematical difference between these distributions.

---

# 24. KL Divergence Math

This mathematical difference is called **Kullback-Leibler (KL) Divergence**.

Let the Policy Model be \( \pi_\theta \).

Let the frozen Reference Model be \( \pi_{ref} \).

The KL Divergence for a specific token generation is calculated as:

$$
D_{KL}(\pi_\theta(y|x) || \pi_{ref}(y|x)) = \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}
$$

If the Policy Model generates the exact same probabilities as the SFT model, the ratio is 1.

The logarithm of 1 is 0.

The KL penalty is exactly 0.

If the Policy Model deviates drastically from the SFT model, the ratio becomes large.

The logarithm becomes large.

The KL penalty becomes a massive negative value.

This dynamically punishes the model for forgetting how to speak English.

---

# 25. The Total Reward Function

We now modify our reward signal for PPO.

We subtract the KL penalty from the score given by the Reward Model.

The total reward function for a generation becomes:

$$
R(x, y) = r(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}
$$

Let's define the terms:

- \( R(x, y) \) is the total reward used to update the model.
- \( r(x, y) \) is the scalar score from the Reward Model.
- \( \beta \) is a crucial hyperparameter.

\( \beta \) controls the strength of the KL penalty.

If \( \beta \) is too low, the model reward hacks.

If \( \beta \) is too high, the model refuses to change from its SFT behavior.

This equation is the beating heart of RLHF.

It forces the Policy Model to walk a delicate tightrope.

It must maximize the human preference score.

While simultaneously minimizing its divergence from the SFT model.

This ensures the model remains coherent, grammatical, and factually grounded.

While simultaneously adopting the newly desired aligned behavior.

---

# 26. Value Model and Advantage

To actually perform the gradient updates, PPO uses a concept called Advantage.

Advantage measures how much better an action was compared to the average expectation.

$$
A_t = R_t - V(s_t)
$$

Where \( V(s_t) \) is the Value Function.

In RLHF, we actually train a third model.

Or we attach a separate scalar head to the Policy Model.

This is called the **Value Model**.

The Value Model tries to predict the final reward from the current state (the current tokens).

If the actual received reward is higher than what the Value Model predicted, the Advantage is positive.

The model did better than expected.

If the actual reward is lower, the Advantage is negative.

The model did worse than expected.

PPO uses this Advantage signal to scale the magnitude and direction of the gradient updates.

---

# 27. The PPO Clipped Objective

PPO stands for Proximal Policy Optimization.

It is named this because of its unique objective function.

In traditional Reinforcement Learning, a large gradient update can completely destroy the policy.

A single bad batch can ruin weeks of training.

PPO prevents this by clipping the update.

It calculates the probability ratio between the new policy and the old policy.

Let this ratio be:

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
$$

The PPO objective function is:

$$
\mathcal{L}^{CLIP}(\theta) = \mathbb{E} [ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) ]
$$

This terrifying equation serves a very simple purpose.

It means: "Do not change the policy too much in a single step."

If the gradient update tries to push the weights too far, the `clip` function cuts it off.

It restricts the update to a small trust region defined by \( \epsilon \).

This ensures highly stable, monotonic improvement during training.

It prevents catastrophic forgetting.

It is the reason PPO is preferred over older algorithms like TRPO or standard Policy Gradient.

---

# 28. The Full RLHF Architecture Diagram

Let's review the entire system running in memory during Step 3.

You need four separate models loaded simultaneously in VRAM.

1. **The Policy Model**: The model being actively trained. (Requires gradients, optimizer states, activations).

2. **The Value Model**: The model predicting expected rewards. (Requires gradients, optimizer states).

3. **The Reward Model**: The model providing the scalar score. (Frozen, inference only).

4. **The Reference Model**: The original SFT model providing KL penalties. (Frozen, inference only).

This architecture looks like this:

```text
[Prompt]
   |
   +---> (Policy Model) -----------> [Response]
   |                                      |
   +---> (Reference Model) -> [KL] <------+
   |                                      |
   +---> (Reward Model) -> [Score] <------+
   |                                      |
   +---> (Value Model) -> [Advantage] <---+
                                          |
        Update Policy & Value Weights <---+
```

This is why RLHF is incredibly expensive.

You need enough VRAM to hold four separate massive neural networks.

For a 70B parameter model, this requires massive compute clusters.

This sheer engineering complexity is exactly why DPO became so popular.

DPO achieves similar results while only requiring two models in memory.

---

# 29. Hyperparameters of RLHF

Tuning RLHF requires mastering several crucial hyperparameters.

The first is the learning rate.

In PPO, the learning rate must be extremely small.

Often around 1e-6 or even 1e-7.

A large learning rate will immediately destroy the policy.

The second is the KL penalty coefficient, denoted as \( \beta \).

If \( \beta \) is 0, the model will reward hack instantly.

If \( \beta \) is 1.0, the model will barely learn anything new.

Typically, \( \beta \) is initialized around 0.1 and scaled dynamically.

The third is the PPO clip range, denoted as \( \epsilon \).

This is usually set to 0.2.

It prevents the probability ratio from exceeding 1.2 or falling below 0.8.

The fourth is the batch size.

PPO requires large batch sizes to estimate the Advantage accurately.

Small batch sizes lead to noisy gradients and unstable training.

Tuning all these parameters is an empirical art form.

It is why alignment engineers are highly sought after.

---

# 30. Conclusion

RLHF is the classical foundation of aligned AI.

It is the reason we have ChatGPT.

Step 1: Supervised Fine-Tuning teaches the model basic behavior via behavioral cloning.

Step 2: Reward Modeling trains a critic to assign scalar scores based on human preferences using the Bradley-Terry model.

Step 3: PPO uses Reinforcement Learning to optimize the Policy Model against the Reward Model.

The KL Divergence penalty prevents Reward Hacking and mode collapse.

The PPO clipping function ensures stable training.

Despite the rise of simpler alternatives like DPO, RLHF remains a critical tool.

It allows models to explore the vast space of possible responses.

It captures complex human nuances that supervised learning alone cannot.

It scales better at the extreme frontier of intelligence.

If you understand the math and architecture of RLHF, you understand the core engine of modern AI.
