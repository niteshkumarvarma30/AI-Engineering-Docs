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
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
       Text            Images           Audio
        │                │                │
        ▼                ▼                ▼
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
  ↓
Classifier
  ↓
Cat
```

Mathematically, it focuses on something like:

\[
P(y|x)
\]

For example:

\[
P(\text{cat}|\text{image})
\]

The model is primarily concerned with deciding which output/label is appropriate for a given input.

---

## 2.2 Generative Model

A generative model learns patterns in a data distribution so that it can generate new samples.

Conceptually:

```text
Learn data distribution
        ↓
Generate new sample
```

For example:

```text
Random noise / latent representation
        ↓
Generative model
        ↓
New image
```

This distinction is useful as a first mental model, although real-world models can have more complicated objectives.

---

# 3. The Generative AI Family

Important generative architectures include:

```text
Generative AI
│
├── Autoregressive Models
│      │
│      └── Transformers
│             │
│             └── LLMs
│
├── GANs
│
├── VAEs
│
└── Diffusion Models
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
 ↓
The cat
 ↓
The cat is
 ↓
The cat is sleeping
```

Mathematically:

\[
P(x_1,x_2,\ldots,x_n)
=
\prod_{t=1}^{n}
P(x_t|x_{<t})
\]

This is the fundamental generation mechanism behind GPT-style LLMs.

---

# 5. Transformers

Transformers are neural-network architectures designed to model relationships between elements in sequences using attention mechanisms.

You already studied the original Transformer:

```text
             Transformer
                 │
        ┌────────┴────────┐
        ▼                 ▼
     Encoder            Decoder
        │                 │
 Self-Attention     Masked Self-Attention
                          │
                   Cross-Attention
```

Modern LLMs commonly use a decoder-only Transformer:

```text
             Decoder-Only Transformer
                       │
                       ▼
                Causal Attention
                       │
                       ▼
                      FFN
                       │
                       ▼
                   Repeat
                       │
                       ▼
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
    ↕
her
```

and many other relationships.

Self-attention allows each token representation to incorporate information from other relevant tokens.

You already know the core equation:

\[
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

This mechanism is one of the foundations of modern LLMs.

---

# 7. GANs

**GAN = Generative Adversarial Network**

A GAN contains two neural networks:

```text
             GAN
              │
       ┌──────┴──────┐
       ▼             ▼
   Generator     Discriminator
```

They have competing objectives.

---

# 8. Generator

The generator creates synthetic samples.

For example:

```text
Random noise
     ↓
Generator
     ↓
Generated image
```

The goal is to generate samples that look similar to real data.

---

# 9. Discriminator

The discriminator tries to distinguish between real and generated samples.

```text
Real image ───────┐
                  ▼
             Discriminator
                  │
Generated image ──┘
                  ↓
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
    ↓
Fake samples
    ↓
Discriminator
    ↓
Feedback
    ↓
Generator improves
    ↓
Better fake samples
    ↓
Discriminator improves
    ↓
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
              │
        ┌─────┴─────┐
        ▼           ▼
     Encoder      Decoder
        │           ▲
        ▼           │
      Latent ───────┘
      Space
```

The key idea is:

> Learn a useful probabilistic latent representation of the data.

---

# 13. VAE Encoder

Suppose we have an image:

```text
Image
 ↓
Encoder
 ↓
Latent representation
```

Instead of simply mapping the image to one deterministic point, a VAE learns parameters describing a probability distribution.

Typically:

\[
\mu
\]

and

\[
\sigma
\]

describe the latent distribution.

---

# 14. Latent Space

The model represents data in a lower-dimensional latent space.

Conceptually:

```text
Images
   ↓
Encoder
   ↓
Latent Space
```

You can imagine different samples occupying different regions:

```text
      Cat ●

                 ● Dog


  ● Car

          ● Bird
```

The VAE attempts to learn a structured and smooth latent representation.

---

# 15. VAE Decoder

The decoder takes a latent representation and reconstructs or generates data.

```text
Latent vector
      ↓
   Decoder
      ↓
Generated image
```

The overall idea is:

```text
Image
 ↓
Encoder
 ↓
Latent representation
 ↓
Decoder
 ↓
Reconstructed image
```

---

# 16. Why Is It Called "Variational"?

A VAE learns a probability distribution over latent representations rather than simply learning a deterministic encoding.

Its training objective has two major components.

## Reconstruction Loss

Encourage the output to resemble the original input.

## KL Divergence

Regularize the latent distribution toward a prior, commonly:

\[
N(0,I)
\]

A simplified objective is:

\[
\boxed{
L =
L_{reconstruction}
+
\beta L_{KL}
}
\]

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
     ↓
Add noise
     ↓
Noisier image
     ↓
Add more noise
     ↓
More noise
     ↓
...
     ↓
Almost pure noise
```

---

# 19. Forward Diffusion

The forward process gradually corrupts the original data.

Conceptually:

```text
x₀
 ↓
x₁
 ↓
x₂
 ↓
x₃
 ↓
...
 ↓
xₜ ≈ noise
```

The standard forward noise process is generally fixed rather than learned.

---

# 20. Reverse Diffusion

The model learns to reverse the corruption process.

```text
Random noise
     ↓
Denoising step
     ↓
Less noise
     ↓
Denoising step
     ↓
Less noise
     ↓
...
     ↓
Generated image
```

So:

```text
Noise
 ↓
Denoise
 ↓
Denoise
 ↓
Denoise
 ↓
Image
```

---

# 21. What Does a Diffusion Model Learn?

A common diffusion formulation trains a neural network to predict the noise added to a sample.

Conceptually:

```text
Clean image
     ↓
Add known noise
     ↓
Noisy image
     ↓
Neural network
     ↓
Predict noise
     ↓
Compare predicted vs actual noise
     ↓
Loss
```

The network learns how to estimate the corruption so that generation can move in the opposite direction.

---

# 22. Why Does Denoising Generate an Image?

During generation, we begin with random noise.

The learned model repeatedly transforms the noisy sample toward the learned data distribution.

```text
Random noise
     ↓
Remove noise
     ↓
Remove noise
     ↓
Remove noise
     ↓
...
     ↓
Image-like sample
```

The result is a newly generated sample.

---

# 23. Text Conditioning in Diffusion Models

Modern text-to-image systems can condition generation on a text prompt.

Conceptually:

```text
Text prompt
    ↓
Text representation
    ↓
Conditioning information
    ↓
Diffusion model
    ↑
    │
Random noise
    ↓
Generated image
```

The exact architecture differs between models.

Attention and cross-attention can be used to connect text conditioning with the generative process.

You already understand cross-attention:

```text
Q = one representation
K,V = another representation
```

---

# 24. Autoregressive vs Diffusion Generation

This is one of the most important comparisons.

## Autoregressive Generation

Used by LLMs:

```text
Token
 ↓
Next token
 ↓
Next token
 ↓
Next token
```

The model generates a sequence progressively.

---

## Diffusion Generation

Used in many image/audio/video generation systems:

```text
Noise
 ↓
Denoising
 ↓
Denoising
 ↓
Denoising
 ↓
Generated sample
```

The model starts from noise and iteratively refines it.

---

# 25. LLM vs Diffusion Model

For a language model:

```text
Prompt
 ↓
Token
 ↓
Token
 ↓
Token
 ↓
...
```

For a diffusion image model:

```text
Noise
 ↓
Denoising
 ↓
Denoising
 ↓
Denoising
 ↓
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
    │
    ├── Encoder-only
    │
    ├── Encoder-Decoder
    │
    └── Decoder-only
```

Different Transformer architectures can be used for different tasks.

---

# 27. Encoder-Only Transformers

Structure:

```text
Input
 ↓
Encoder
 ↓
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
 ↓
Encoder
 ↓
Encoder representations
 ↓
Decoder
 ↓
Output
```

The decoder uses cross-attention to access encoder information.

Conceptually:

```text
Q = Decoder
K,V = Encoder
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
 ↓
Causal Transformer
 ↓
Next token
 ↓
Next token
 ↓
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
                       ↓
                    Paris
```

Then:

```text
The capital of France is Paris
                              ↓
                         <EOS>
```

The model repeatedly predicts the next token.

This is why causal self-attention is central to modern LLMs.

---

# 31. The Transformer Family

You should now have this conceptual map:

```text
                         Transformer
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
         Encoder-only   Encoder-Decoder   Decoder-only
              │               │               │
            BERT             T5            GPT-style
                                              │
                                              ▼
                                             LLM
```

This is a simplified map; actual model families and architectures can be more complex.

---

# 32. The Larger Generative AI Family Tree

You should now have this mental model:

```text
                         GENERATIVE AI
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
       GANs                   VAEs              Diffusion
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    Autoregressive Models
                              │
                              ▼
                         Transformers
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
         Encoder-only   Encoder-Decoder   Decoder-only
              │               │               │
            BERT             T5            GPT/Llama
                                              │
                                              ▼
                                             LLMs
```

This is a conceptual taxonomy. Modern systems can combine multiple ideas.

---

# 33. Where Does Multimodal AI Fit?

Modern generative systems increasingly combine multiple modalities.

```text
                    Multimodal AI
                         │
        ┌────────────────┼────────────────┐
        │                │                │
       Text            Image            Audio
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
                  Multimodal Model
```

Examples of multimodal generation include:

```text
Text
 ↓
Image generation
```

```text
Text
 ↓
Video generation
```

```text
Image + Text
 ↓
Multimodal LLM
 ↓
Text response
```

Different systems use different combinations of encoders, Transformers, diffusion components, and other architectures.

---

# 34. How Your Existing Knowledge Fits

You have already learned:

```text
RNN
 ↓
LSTM
 ↓
GRU
 ↓
Encoder-Decoder
 ↓
Bahdanau Attention
 ↓
Luong Attention
 ↓
Self-Attention
 ↓
Q/K/V
 ↓
Multi-Head Attention
 ↓
Masked Attention
 ↓
Cross-Attention
 ↓
Transformer
```

Now place that knowledge into the larger Generative AI picture:

```text
Generative AI
      │
      ▼
Sequence Generation
      │
      ▼
Transformers
      │
      ▼
Decoder-only Transformers
      │
      ▼
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
     ↓
Decoder-only architecture
     ↓
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
- Strengths/limitations
- Difference from autoregressive LLMs

You do not need to derive every GAN/VAE/diffusion equation for this roadmap.

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
      │
      ├── Diffusion
      │
      ├── GAN
      │
      ├── VAE
      │
      └── Autoregressive models
                │
                └── Transformers
                        │
                        └── LLMs
```

---

# 38. Key Concepts to Remember

### Generative AI

Models that generate new content from learned patterns/distributions.

### GAN

```text
Generator ↔ Discriminator
```

### VAE

```text
Data → Encoder → Latent → Decoder
```

### Diffusion

```text
Data → Noise
Noise → Iterative denoising → Generated data
```

### Transformer

Attention-based architecture for sequence modeling.

### LLM

A large language model, commonly based on a Transformer.

### Autoregressive generation

```text
Previous tokens
      ↓
Next token
      ↓
Append
      ↓
Next token
```

---

# 39. Final Mental Model

```text
                    GENERATIVE AI
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
         GAN             VAE        Diffusion
                                         │
                                         │
          ┌──────────────────────────────┘
          │
          ▼
   Autoregressive Generation
          │
          ▼
      Transformer
          │
     ┌────┴────┐
     │         │
 Encoder    Decoder
             │
             ▼
       Decoder-only
             │
             ▼
            LLM
             │
             ▼
     Next-token prediction
             │
             ▼
        Text generation
```

---

# 40. Self-Check Questions

Before moving to Lesson 03, you should be able to answer:

1. What is Generative AI?
2. What is the difference between generative and discriminative models?
3. What is a GAN?
4. What are the generator and discriminator?
5. What is a VAE?
6. What is latent space?
7. What is the basic idea behind diffusion?
8. What is the difference between autoregressive generation and diffusion generation?
9. Is a Transformer the same thing as an LLM?
10. What is the difference between encoder-only, encoder-decoder, and decoder-only Transformers?
11. Why are decoder-only Transformers important for modern LLMs?
12. Where do LLMs fit inside the larger Generative AI ecosystem?

---

# 41. Roadmap Progress

```text
PHASE 1 — LLM FOUNDATIONS

01. What Are LLMs?                   ✅
        ↓
02. Generative AI: The Big Picture   ✅
        ↓
03. Tokenization Deep Dive
        ↓
04. Embeddings & Token Representations

PHASE 2 — MODERN LLM ARCHITECTURE

05. LLM Architecture Internals
06. Positional Encodings
07. Mixture of Experts
08. Context Window & Attention Patterns

PHASE 3 — LLM TRAINING & GENERATION

09. Pre-training Objectives
10. Logits, Softmax & Temperature
11. Sampling Strategies
12. Multi-turn Conversations & Memory
```

## Next Lesson

**Lesson 03 — Tokenization Deep Dive**

Topics:

- Character vs word vs subword tokenization
- BPE
- WordPiece
- SentencePiece
- Byte-level tokenization
- Vocabulary
- Token IDs
- Special tokens
- BOS/EOS/PAD/UNK/MASK
- Padding
- Truncation
- Attention masks
- Hugging Face tokenizers
