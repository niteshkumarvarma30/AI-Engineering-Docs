# Lesson 02 — Generative AI: The Big Picture

## Learning Objectives

By the end of this lesson, you should be able to explain:

- What Generative AI is
- The difference between generative and discriminative models
- Where LLMs fit within Generative AI
- What GANs are and how they work
- What VAEs are and how they work
- What diffusion models are and how they work
- The difference between autoregressive and diffusion generation
- What Transformers are
- The relationship between Transformers and LLMs
- Encoder-only, Encoder-Decoder, and Decoder-only Transformer architectures
- Why decoder-only Transformers are important for modern LLMs
- How your existing attention/Transformer knowledge fits into the broader Generative AI ecosystem

---

# 1. What Is Generative AI?

**Generative AI** refers to models that learn patterns from data and can generate new content.

The generated content can include:

- Text
- Images
- Audio
- Video
- Code
- 3D content
- Multimodal content

A simplified view:

```text
                    Generative AI
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
       Text            Images           Audio
        |                |                |
        v                v                v
      LLMs           Diffusion         Generative
   Transformers       Models             Models
```

Generative AI is a broad field.

LLMs are one important part of it.

---

# 2. Discriminative vs Generative Models

Before understanding Generative AI, distinguish two broad types of machine learning models.

## 2.1 Discriminative Model

A discriminative model learns to predict a label or output from an input.

For example:

```text
Image
  |
  v
Classifier
  |
  v
Cat
```

Mathematically, it focuses on something like:

```text
P(y | x)
```

For example:

```text
P(cat | image)
```

The model is primarily concerned with deciding which output or label is appropriate for a given input.

---

## 2.2 Generative Model

A generative model learns patterns in a data distribution so that it can generate new samples.

Conceptually:

```text
Learn data distribution
        |
        v
Generate new sample
```

For example:

```text
Random noise / latent representation
        |
        v
Generative model
        |
        v
New image
```

This distinction is useful as a first mental model, although real-world models can have more complicated objectives.

---

# 3. The Generative AI Family

Important generative architectures include:

```text
Generative AI
|
+-- Autoregressive Models
|      |
|      +-- Transformers
|             |
|             +-- LLMs
|
+-- GANs
|
+-- VAEs
|
+-- Diffusion Models
```

There are also hybrid and newer architectures.

For this roadmap, the goal is to understand the major families and then go deeply into Transformers and LLMs.

---

# 4. Autoregressive Models

You already encountered this in Lesson 01.

An autoregressive model generates one element at a time based on previous elements.

For language:

```text
The
 |
 v
The cat
 |
 v
The cat is
 |
 v
The cat is sleeping
```

Mathematically:

```text
P(x_1, x_2, ..., x_n)
=
product from t = 1 to n of P(x_t | x_{<t})
```

This is the fundamental generation mechanism behind GPT-style LLMs.

---

# 5. Transformers

Transformers are neural-network architectures designed to model relationships between elements in sequences using attention mechanisms.

You already studied the original Transformer:

```text
             Transformer
                 |
        +--------+--------+
        |                 |
        v                 v
     Encoder            Decoder
        |                 |
 Self-Attention     Masked Self-Attention
                          |
                   Cross-Attention
```

Modern LLMs commonly use a decoder-only Transformer:

```text
             Decoder-Only Transformer
                       |
                       v
                Causal Attention
                       |
                       v
                      FFN
                       |
                       v
                   Repeat
                       |
                       v
                 Next Token
```

---

# 6. Why Transformers Work Well for Language

Language contains relationships between tokens that may be far apart.

For example:

```text
The scientist who studied the disease for many years
finally published her research.
```

The model needs to represent relationships between tokens such as:

```text
scientist
    |
    v
her
```

and many other relationships.

Self-attention allows each token representation to incorporate information from other relevant tokens.

You already know the core equation:

```text
Attention(Q, K, V)
=
softmax(QK^T / sqrt(d_k)) V
```

This mechanism is one of the foundations of modern LLMs.

---

# 7. GANs

**GAN = Generative Adversarial Network**

A GAN contains two neural networks:

```text
             GAN
              |
       +------+------+
       |             |
       v             v
   Generator     Discriminator
```

They have competing objectives.

---

# 8. Generator

The generator creates synthetic samples.

For example:

```text
Random noise
     |
     v
Generator
     |
     v
Generated image
```

The goal is to generate samples that look similar to real data.

---

# 9. Discriminator

The discriminator tries to distinguish between real and generated samples.

```text
Real image --------+
                   |
                   v
              Discriminator
                   |
Generated image ---+
                   |
                   v
              Real / Fake
```

Conceptually:

```text
Generator:
"Create something realistic."

Discriminator:
"Determine whether it is real."
```

---

# 10. GAN Training

The generator tries to fool the discriminator.

The discriminator tries not to be fooled.

Conceptually:

```text
Generator
    |
    v
Fake samples
    |
    v
Discriminator
    |
    v
Feedback
    |
    v
Generator improves
    |
    v
Better fake samples
    |
    v
Discriminator improves
    |
    v
...
```

This is called **adversarial training**.

---

# 11. Why GANs Were Important

GANs demonstrated that neural networks could generate highly realistic synthetic data.

They became particularly important for:

- Image generation
- Image manipulation
- Super-resolution
- Style transfer
- Synthetic data

However, GAN training can be difficult.

Common issues include:

- Mode collapse
- Training instability
- Generator/discriminator imbalance
- Difficult optimization

These limitations contributed to the rise of other generative approaches.

---

# 12. Variational Autoencoders

**VAE = Variational Autoencoder**

A VAE uses an encoder-decoder architecture and learns a probabilistic latent representation.

```text
             VAE
              |
        +-----+-----+
        |           |
        v           v
     Encoder      Decoder
        |           ^
        v           |
      Latent -------+
      Space
```

The key idea is:

> Learn a useful probabilistic latent representation of the data.

---

# 13. VAE Encoder

Suppose we have an image:

```text
Image
 |
 v
Encoder
 |
 v
Latent representation
```

Instead of simply mapping the image to one deterministic point, a VAE learns parameters describing a probability distribution.

Typically:

```text
mu
```

and

```text
sigma
```

describe the latent distribution.

---

# 14. Latent Space

The model represents data in a lower-dimensional latent space.

Conceptually:

```text
Images
   |
   v
Encoder
   |
   v
Latent Space
```

You can imagine different samples occupying different regions:

```text
      Cat *

                 * Dog


  * Car

          * Bird
```

The VAE attempts to learn a structured and smooth latent representation.

---

# 15. VAE Decoder

The decoder takes a latent representation and reconstructs or generates data.

```text
Latent vector
      |
      v
   Decoder
      |
      v
Generated image
```

The overall idea is:

```text
Image
 |
 v
Encoder
 |
 v
Latent representation
 |
 v
Decoder
 |
 v
Reconstructed image
```

---

# 16. Why Is It Called "Variational"?

A VAE learns a probability distribution over latent representations rather than simply learning a deterministic encoding.

Its training objective has two major components.

## Reconstruction Loss

Encourages the output to resemble the original input.

## KL Divergence

Regularizes the latent distribution toward a prior, commonly:

```text
N(0, I)
```

A simplified objective is:

```text
L = L_reconstruction + beta * L_KL
```

This helps create a useful latent space from which new samples can be generated.

---

# 17. GAN vs VAE

| | GAN | VAE |
|---|---|---|
| Main components | Generator + Discriminator | Encoder + Decoder |
| Latent representation | Yes | Yes |
| Training | Adversarial | Reconstruction + KL |
| Training stability | Can be difficult | Generally more stable |
| Image sharpness | Historically strong | Can be smoother/blurrer |
| Latent regularization | Less explicit | Explicit |

These are broad conceptual comparisons rather than universal rules for every implementation.

---

# 18. Diffusion Models

Diffusion models are another major family of generative models.

The central idea is:

> Start with clean data, progressively add noise, and train a model to reverse the process.

For example:

```text
Clean Image
     |
     v
Add noise
     |
     v
Noisier image
     |
     v
Add more noise
     |
     v
More noise
     |
     v
...
     |
     v
Almost pure noise
```

---

# 19. Forward Diffusion

The forward process gradually corrupts the original data.

Conceptually:

```text
x_0
 |
 v
x_1
 |
 v
x_2
 |
 v
x_3
 |
 v
...
 |
 v
x_t approximately equal to noise
```

The standard forward noise process is generally fixed rather than learned.

---

# 20. Reverse Diffusion

The model learns to reverse the corruption process.

```text
Random noise
     |
     v
Denoising step
     |
     v
Less noise
     |
     v
Denoising step
     |
     v
Less noise
     |
     v
...
     |
     v
Generated image
```

So:

```text
Noise
 |
 v
Denoise
 |
 v
Denoise
 |
 v
Denoise
 |
 v
Image
```

---

# 21. What Does a Diffusion Model Learn?

A common diffusion formulation trains a neural network to predict the noise added to a sample.

Conceptually:

```text
Clean image
     |
     v
Add known noise
     |
     v
Noisy image
     |
     v
Neural network
     |
     v
Predict noise
     |
     v
Compare predicted vs actual noise
     |
     v
Loss
```

The network learns how to estimate the corruption so that generation can move in the opposite direction.

---

# 22. Why Does Denoising Generate an Image?

During generation, we begin with random noise.

The learned model repeatedly transforms the noisy sample toward the learned data distribution.

```text
Random noise
     |
     v
Remove noise
     |
     v
Remove noise
     |
     v
Remove noise
     |
     v
...
     |
     v
Image-like sample
```

The result is a newly generated sample.

---

# 23. Text Conditioning in Diffusion Models

Modern text-to-image systems can condition generation on a text prompt.

Conceptually:

```text
Text prompt
    |
    v
Text representation
    |
    v
Conditioning information
    |
    v
Diffusion model
    ^
    |
Random noise
    |
    v
Generated image
```

The exact architecture differs between models.

Attention and cross-attention can be used to connect text conditioning with the generative process.

You already understand cross-attention:

```text
Q = one representation
K, V = another representation
```

---

# 24. Autoregressive vs Diffusion Generation

This is one of the most important comparisons.

## Autoregressive Generation

Used by LLMs:

```text
Token
 |
 v
Next token
 |
 v
Next token
 |
 v
Next token
```

The model generates a sequence progressively.

---

## Diffusion Generation

Used in many image, audio, and video generation systems:

```text
Noise
 |
 v
Denoising
 |
 v
Denoising
 |
 v
Denoising
 |
 v
Generated sample
```

The model starts from noise and iteratively refines it.

---

# 25. LLM vs Diffusion Model

For a language model:

```text
Prompt
 |
 v
Token
 |
 v
Token
 |
 v
Token
 |
 v
...
```

For a diffusion image model:

```text
Noise
 |
 v
Denoising
 |
 v
Denoising
 |
 v
Denoising
 |
 v
Image
```

The generation mechanism is fundamentally different.

---

# 26. Transformers Are Not the Same as LLMs

This distinction is important.

A **Transformer** is an architecture.

An **LLM** is a large language model, commonly implemented using a Transformer architecture.

Think of:

```text
Transformer
    |
    +-- Encoder-only
    |
    +-- Encoder-Decoder
    |
    +-- Decoder-only
```

Different Transformer architectures can be used for different tasks.

---

# 27. Encoder-Only Transformers

Structure:

```text
Input
 |
 v
Encoder
 |
 v
Contextual representation
```

Examples include BERT-style models.

Typical uses:

- Classification
- Token classification
- Representation learning
- Embeddings
- Understanding tasks

---

# 28. Encoder-Decoder Transformers

Structure:

```text
Source
 |
 v
Encoder
 |
 v
Encoder representations
 |
 v
Decoder
 |
 v
Output
```

The decoder uses cross-attention to access encoder information.

Conceptually:

```text
Q = Decoder
K, V = Encoder
```

This is the architecture you already studied.

Typical uses:

- Translation
- Summarization
- Sequence-to-sequence tasks

---

# 29. Decoder-Only Transformers

Structure:

```text
Prompt
 |
 v
Causal Transformer
 |
 v
Next token
 |
 v
Next token
 |
 v
Next token
```

This is the architecture used by many modern general-purpose LLMs.

Examples of model families using this general paradigm include:

- GPT-style models
- Llama-style models
- Qwen-style models
- Many other causal language models

---

# 30. Why Decoder-Only Transformers Matter for LLMs

A decoder-only model can perform autoregressive language generation naturally.

For example:

```text
The capital of France is
                       |
                       v
                     Paris
```

Then:

```text
The capital of France is Paris
                              |
                              v
                            <EOS>
```

The model repeatedly predicts the next token.

This is why causal self-attention is central to modern LLMs.

---

# 31. The Transformer Family

You should now have this conceptual map:

```text
                         Transformer
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
         Encoder-only   Encoder-Decoder   Decoder-only
              |               |               |
            BERT              T5          GPT-style
                                              |
                                              v
                                             LLM
```

This is a simplified map; actual model families and architectures can be more complex.

---

# 32. The Larger Generative AI Family Tree

You should now have this mental model:

```text
                         GENERATIVE AI
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
       GANs                   VAEs              Diffusion
        |                     |                     |
        +---------------------+---------------------+
                              |
                    Autoregressive Models
                              |
                              v
                         Transformers
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
         Encoder-only   Encoder-Decoder   Decoder-only
              |               |               |
            BERT              T5          GPT / Llama
                                              |
                                              v
                                             LLMs
```

This is a conceptual taxonomy. Modern systems can combine multiple ideas.

---

# 33. Where Does Multimodal AI Fit?

Modern generative systems increasingly combine multiple modalities.

```text
                    Multimodal AI
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
       Text            Image            Audio
        |                |                |
        +----------------+----------------+
                         |
                         v
                  Multimodal Model
```

Examples of multimodal generation include:

```text
Text
 |
 v
Image generation
```

```text
Text
 |
 v
Video generation
```

```text
Image + Text
 |
 v
Multimodal LLM
 |
 v
Text response
```

Different systems use different combinations of encoders, Transformers, diffusion components, and other architectures.

---

# 34. How Your Existing Knowledge Fits

You have already learned:

```text
RNN
 |
 v
LSTM
 |
 v
GRU
 |
 v
Encoder-Decoder
 |
 v
Bahdanau Attention
 |
 v
Luong Attention
 |
 v
Self-Attention
 |
 v
Q / K / V
 |
 v
Multi-Head Attention
 |
 v
Masked Attention
 |
 v
Cross-Attention
 |
 v
Transformer
```

Now place that knowledge into the larger Generative AI picture:

```text
Generative AI
      |
      v
Sequence Generation
      |
      v
Transformers
      |
      v
Decoder-only Transformers
      |
      v
LLMs
```

Your previous attention lessons are therefore the architectural foundation for understanding modern LLMs.

---

# 35. High-Level Comparison

| Architecture | Main generation mechanism | Common applications |
|---|---|---|
| GAN | Generator vs discriminator | Image synthesis, manipulation |
| VAE | Latent sampling + decoder | Representation learning, generation |
| Diffusion | Iterative denoising | Image, audio, video generation |
| Autoregressive Transformer | Sequential prediction | Text/code generation |
| Decoder-only Transformer | Next-token prediction | Modern LLMs |
| Encoder-Decoder Transformer | Sequence-to-sequence transformation | Translation, summarization |

---

# 36. What You Need to Learn Deeply

Because this roadmap is specifically about LLMs, do not spend equal time on every Generative AI architecture.

## Deep Understanding

You need:

```text
Transformers
     |
     v
Decoder-only architecture
     |
     v
LLMs
```

## Conceptual Understanding

You need:

```text
GANs
VAEs
Diffusion
```

For each, understand:

- What problem it solves
- Basic architecture
- Training principle
- Generation mechanism
- Strengths and limitations
- Difference from autoregressive LLMs

You do not need to derive every GAN, VAE, or diffusion equation for this roadmap.

---

# 37. Important AI Engineer Distinction

Do not confuse these terms.

## Generative AI

A broad field involving models capable of generating new content.

## Transformer

A neural-network architecture based heavily on attention mechanisms.

## LLM

A large language model, commonly implemented using a Transformer.

A useful conceptual picture is:

```text
Generative AI
      |
      +-- Diffusion
      |
      +-- GAN
      |
      +-- VAE
      |
      +-- Autoregressive models
                |
                +-- Transformers
                        |
                        +-- LLMs
```

---

# 38. Key Concepts to Remember

### Generative AI

Models that generate new content from learned patterns or distributions.

### GAN

```text
Generator <-> Discriminator
```

### VAE

```text
Data -> Encoder -> Latent -> Decoder
```

### Diffusion

```text
Data -> Noise
Noise -> Iterative denoising -> Generated data
```

### Transformer

Attention-based architecture for sequence modeling.

### LLM

A large language model, commonly based on a Transformer.

### Autoregressive Generation

```text
Previous tokens
      |
      v
Next token
      |
      v
Append
      |
      v
Next token
```

---

# 39. Final Mental Model

```text
                    GENERATIVE AI
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
         GAN             VAE        Diffusion
                                         |
                                         |
          +------------------------------+
          |
          v
   Autoregressive Generation
          |
          v
      Transformer
          |
     +----+----+
     |         |
     v         v
 Encoder    Decoder
             |
             v
       Decoder-only
             |
             v
            LLM
             |
             v
      Next-token prediction
```

The most important connection is:

```text
Generative AI
      |
      v
Autoregressive Generation
      |
      v
Transformer
      |
      v
Decoder-only Transformer
      |
      v
LLM
      |
      v
Next-token prediction
```

At the same time, not all Generative AI uses Transformers or autoregressive generation:

```text
Generative AI
 |
 +-- GANs
 |
 +-- VAEs
 |
 +-- Diffusion Models
 |
 +-- Autoregressive Models
        |
        +-- Transformers
              |
              +-- LLMs
```

This distinction will help you understand where LLMs fit within the much larger Generative AI ecosystem.

---

# 40. Self-Check Questions

Before moving to Lesson 03, you should be able to answer:

1. What is Generative AI?
2. What is the difference between a discriminative model and a generative model?
3. What is an autoregressive model?
4. Why are Transformers important for modern LLMs?
5. What is a GAN?
6. What are the roles of the generator and discriminator?
7. What is mode collapse?
8. What is a VAE?
9. What is latent space?
10. Why does a VAE use KL divergence?
11. What is a diffusion model?
12. What happens during the forward diffusion process?
13. What happens during reverse diffusion?
14. What is the difference between autoregressive and diffusion generation?
15. What is a Transformer?
16. What is the difference between a Transformer and an LLM?
17. What is an encoder-only Transformer?
18. What is an encoder-decoder Transformer?
19. What is a decoder-only Transformer?
20. Why are decoder-only Transformers important for modern LLMs?
21. How does your previous attention knowledge connect to LLMs?
22. Are all Generative AI systems LLMs?
23. Are all Generative AI systems Transformers?

---

# 41. Roadmap Position

```text
PHASE 1 - LLM FOUNDATIONS

01. What Are LLMs?
        |
        v
02. Generative AI: The Big Picture    <- YOU ARE HERE
        |
        v
03. Tokenization Deep Dive
        |
        v
04. Embeddings & Token Representations


PHASE 2 - MODERN LLM ARCHITECTURE

05. LLM Architecture Internals
        |
        v
06. Positional Encodings
        |
        v
07. Mixture of Experts
        |
        v
08. Context Window & Attention Patterns


PHASE 3 - LLM TRAINING & GENERATION

09. Pre-training Objectives
        |
        v
10. Logits, Softmax & Temperature
        |
        v
11. Sampling Strategies
        |
        v
12. Multi-turn Conversations & Memory
```

---

# Next Lesson

**Lesson 03 - Tokenization Deep Dive**
