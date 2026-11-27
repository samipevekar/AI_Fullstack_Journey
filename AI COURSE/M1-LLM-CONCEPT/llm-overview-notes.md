# Understanding Large Language Models (LLMs)

My learning notes on Large Language Models (LLMs), covering how they work, how they are trained, and how they generate text.

---

## Table of Contents

1. [What is an LLM?](#1-what-is-an-llm)
2. [LLM Parameters](#2-llm-parameters)
3. [Neural Networks](#3-neural-networks)
4. [Transformer Architecture](#4-transformer-architecture)
5. [Tokenization](#5-tokenization)
6. [Vector Embeddings](#6-vector-embeddings)
7. [Positional Encoding](#7-positional-encoding)
8. [Self-Attention](#8-self-attention)
9. [Multi-Head Attention](#9-multi-head-attention)
10. [Transformer Block](#10-transformer-block)
11. [Next Token Prediction](#11-next-token-prediction)
12. [How an LLM Generates Text](#12-how-an-llm-generates-text)
13. [LLM Training Process](#13-llm-training-process)
14. [Important Differences](#14-important-differences)
15. [Context Window](#15-context-windows)
16. [Maximum Output Tokens](#16-maximum-output-tokens)
17. [Temperature](#17-temperature)
18. [Top-K Sampling](#18-Top-K-sampling)
19. [Top-P Sampling (Nucleus Sampling)](#19-Top-P-samplings)
20. [Difference Between Temperature, Top-K, and Top-P](#20-Temperature)
21. [Hallucination in LLMs](#21-hallucination-in-LLMs)

---

# 1. What is an LLM?

LLM stands for **Large Language Model**.

An LLM is an AI model trained on a large amount of text data to understand language patterns and generate text.

It learns to predict the next token based on the tokens that came before it.

For example:

**Input:**
```text
The cat drinks
```

**Possible prediction:**
```text
milk
```

The model does not simply store sentences from its training data. Instead, it learns patterns and relationships through training.

### Examples of LLMs

| Model family | Company |
|---|---|
| GPT | OpenAI |
| Llama | Meta |
| Claude | Anthropic |
| Gemini | Google |

A model family can have multiple versions and sizes.

For example:
- Llama 3.1 8B
- Llama 3.1 70B
- Llama 3.3 70B

The number represents the approximate number of parameters.

---

# 2. LLM Parameters

Parameters are the numerical values (weights and biases) that a neural network learns during training.

They determine how information is processed inside the model.

For example:
- Embedding matrix values
- Attention weights
- Feed-forward network weights
- Output projection weights

Parameters are not words, sentences, or stored answers.

### What does 8B mean?

8B means approximately 8 billion parameters.

Similarly:
- 1B = 1 billion parameters
- 8B = 8 billion parameters
- 70B = 70 billion parameters

Generally, larger models require more memory and computational resources.

However, more parameters do not automatically mean better performance.

---

# 3. Neural Networks

A neural network is a computational model inspired by the way biological neurons are connected.

It consists of interconnected mathematical operations organized into layers.

During training, the network adjusts its parameters to improve its predictions.

A Transformer is one type of neural network architecture.

---

# 4. Transformer Architecture

The Transformer architecture was introduced in the 2017 research paper:

**Attention Is All You Need**

Transformers are the foundation of many modern LLMs.

There are two common Transformer architecture types:

1. Encoder-only
2. Decoder-only

Many modern generative LLMs, such as GPT and Llama, use a **decoder-only Transformer architecture**.

A simplified decoder-only Transformer looks like this:

```text
Input Tokens
     |
     v
Token Embeddings
     |
     v
Positional Information
     |
     v
Transformer Block 1
     |
     v
Transformer Block 2
     |
     v
Transformer Block 3
     |
     .
     .
     v
Final Transformer Block
     |
     v
Output Prediction
```

A Transformer consists of multiple repeated blocks.

Each block processes the token representations and adds contextual information.

---

# 5. Tokenization

Tokenization is the process of converting text into smaller units called **tokens**.

A token can represent:
- A complete word
- A part of a word
- Punctuation
- Special characters

For example:

```text
Input:
I love programming!

Possible tokens:
["I", " love", " program", "ming", "!"]
```

The actual tokenization depends on the tokenizer.

### Who performs tokenization?

A tokenizer is software associated with a particular model.

The model developers create or select a tokenizer and its vocabulary.

### Tokenizer Vocabulary

A tokenizer vocabulary contains all the token units that the tokenizer recognizes.

Each token has a unique numerical ID.

Example:

| Token | Token ID |
|---|---:|
| I | 101 |
| love | 102 |
| program | 103 |
| ming | 104 |
| ! | 105 |

The text is converted into token IDs:

```text
"I love programming!"

       |
       v

[101, 102, 103, 104, 105]
```

### Do all LLMs use the same vocabulary?

No.

Different models can use different tokenizers and vocabularies.

For example, Llama 2 and Llama 3 use different tokenization systems.

The tokenizer must be compatible with the model because token IDs are used to look up embedding vectors.

### Natural Language Vocabulary vs Tokenizer Vocabulary

These are different concepts.

**Natural language vocabulary:**
- Contains complete words.
- Example: playing, running, programming.

**Tokenizer vocabulary:**
- Contains token units.
- Example: play, ing, run, ning, program.

A complete word can be represented by one token or multiple tokens.

The model's output vocabulary generally corresponds to the tokenizer's vocabulary.

---

# 6. Vector Embeddings

After tokenization, token IDs are converted into numerical vectors using an embedding layer.

An embedding is an array of floating-point numbers representing a token.

For example:

```text
Token: Cat
Token ID: 101

Embedding:
[0.25, 0.87, -0.34, 0.12]
```

Another token:

```text
Token: Dog
Token ID: 102

Embedding:
[0.31, 0.79, -0.28, 0.18]
```

These numbers are illustrative, not actual model values.

### Embedding Matrix

The embedding layer contains a matrix of learned numerical values.

Suppose:

- Vocabulary size = 5,000
- Embedding dimension = 4

Then the embedding matrix contains:

```text
5000 × 4 = 20,000 numbers
```

Each token ID is used to look up its corresponding four-dimensional vector.

### How are embeddings created?

Initially, embedding values are generally randomly initialized.

During training, the optimizer updates these values through backpropagation.

Over time, the model learns useful patterns and relationships between tokens.

### Important points

- An embedding is a vector, not a word.
- A token ID is an integer used to look up its vector.
- Embedding vectors are learned parameters.
- Similarity between embeddings can be measured using cosine similarity.
- Individual dimensions do not necessarily have human-readable meanings.

### Contextual Representations

The initial embedding of a token is not necessarily the same as its representation after Transformer processing.

For example:

```text
I deposited money in the bank.

I sat beside the river bank.
```

The word `bank` starts with the same token embedding, but Transformer layers produce different contextual representations based on the surrounding words.

---

# 7. Positional Encoding

Token embeddings represent tokens, but the model also needs information about their positions.

Consider:

```text
Dog bites man.
Man bites dog.
```

The words are similar, but their order changes the meaning.

Positional information helps the Transformer understand token order.

In the original Transformer architecture, positional encoding is added to token embeddings.

Conceptually:

```text
Token Embedding + Positional Information
                 |
                 v
          Input Representation
```

For example:

```text
Dog   -> Position 0
bites -> Position 1
man   -> Position 2
```

### Types of positional information

- Sinusoidal positional encoding
- Learned positional embeddings
- Rotary Positional Embeddings (RoPE)

Modern models use different approaches.

For example, Llama uses RoPE, which incorporates positional information into attention calculations.

---

# 8. Self-Attention

Self-attention allows each token to consider information from other tokens in the same input sequence.

It helps the model build contextual representations.

Consider:

```text
The animal was tired because it ran all day.
```

The model can learn that `it` is related to `animal`.

### Query, Key, and Value (Q, K, V)

Self-attention uses three vectors:

1. Query (Q)
2. Key (K)
3. Value (V)

These are calculated using learned linear projections of the input token representations.

**Query:**

Represents what information a token is looking for.

**Key:**

Represents information that a token offers for matching against queries.

**Value:**

Contains the information that can be passed to other tokens.

### How Self-Attention Works

Consider:

```text
Dog drinks milk
```

Each token gets its own Query, Key, and Value vectors.

For the token `drinks`:

**Step 1: Calculate attention scores**

Compare its Query with the Keys of all accessible tokens.

```text
Query(drinks) · Key(Dog)
Query(drinks) · Key(drinks)
Query(drinks) · Key(milk)
```

These dot products produce attention scores.

**Step 2: Apply Softmax**

Softmax converts the scores into attention weights.

Illustrative example:

| Token | Attention weight |
|---|---:|
| Dog | 0.60 |
| drinks | 0.25 |
| milk | 0.15 |

**Step 3: Weighted sum of Values**

Each Value vector is multiplied by its attention weight.

```text
Output =
0.60 × Value(Dog)
+
0.25 × Value(drinks)
+
0.15 × Value(milk)
```

The result is a new contextual representation for `drinks`.

This process happens for each token.

### Important

The original embedding is not simply overwritten.

Attention produces new contextual representations by combining information from other tokens.

In decoder-only LLMs, causal masking prevents a token from attending to future tokens.

---

# 9. Multi-Head Attention

Self-attention can be implemented using one attention head or multiple attention heads.

### Single-Head Attention

Uses one set of Query, Key, and Value projections.

```text
Input
  |
  v
Q, K, V
  |
  v
Attention Calculation
  |
  v
Output
```

### Multi-Head Attention

Uses multiple independent attention heads operating in parallel.

```text
                  Input
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
      Head 1      Head 2      Head 3
        |           |           |
        v           v           v
      Output 1    Output 2    Output 3
        |           |           |
        +-----------+-----------+
                    |
                    v
              Concatenation
                    |
                    v
             Output Projection
                    |
                    v
                  Output
```

Different heads can learn different attention patterns.

For example, one head might learn a pattern involving nearby words, while another might learn a pattern involving relationships between more distant words.

These are illustrative possibilities; attention heads do not necessarily have fixed human-readable roles.

---

# 10. Transformer Block

A Transformer block contains attention and feed-forward processing.

A simplified original post-norm Transformer block looks like this:

```text
Input
  |
  v
Multi-Head Self-Attention
  |
  v
Add & Norm
  |
  v
Feed-Forward Network (FFN)
  |
  v
Add & Norm
  |
  v
Output
```

This block is repeated multiple times.

### 10.1 Add & Norm

Add & Norm combines two operations:

**Residual Connection:**

Adds the original input vector to the sublayer output.

Example:

```text
Original Input:
[1, 2, 3]

Attention Output:
[0.5, 1, 0.5]

Addition:
[1.5, 3, 3.5]
```

**Layer Normalization:**

Normalizes values within the vector.

Conceptually:

```text
Normalized Value =
(Value - Mean) / Standard Deviation
```

Learned scale and shift parameters are also applied.

Normalization helps stabilize model training.

### 10.2 Feed-Forward Network (FFN)

The Feed-Forward Network is also called an MLP (Multi-Layer Perceptron).

It processes each token's contextual representation independently, using shared weights across token positions.

A simplified FFN:

```text
Input Vector
     |
     v
Linear Layer
     |
     v
Activation Function
     |
     v
Linear Layer
     |
     v
Output Vector
```

The first linear layer commonly expands the hidden dimension.

The second projects it back to the original hidden dimension.

Modern architectures may use activations such as GELU or SwiGLU.

### Important distinction

Attention combines contextual information across tokens.

The FFN transforms each token's representation.

### Modern Transformer architectures

Not every LLM follows the exact original post-norm diagram.

For example, Llama uses a Pre-Norm design with RMSNorm and residual connections.

---

# 11. Next Token Prediction

After all Transformer blocks have processed the input, the model produces contextual representations.

For next-token prediction, the model uses the representation at the last input position.

For example:

```text
Input:
Dog drinks milk

       |
       v

Final Transformer Representations

       |
       v

Select Last Token Representation
       |
       v

LM Head / Output Projection
```

### Language Model Head (LM Head)

The LM Head is a learned linear projection.

It converts the last token's contextual vector into a score for every token in the tokenizer vocabulary.

Suppose:

```text
Hidden Dimension = 4
Vocabulary Size = 5000
```

The output projection maps:

```text
4-dimensional vector
        |
        v
5000 logits
```

Each logit corresponds to a token ID.

### Logits

Logits are raw numerical scores.

Example:

| Token ID | Token | Logit |
|---|---|---:|
| 101 | Dog | 1.2 |
| 102 | drinks | 0.8 |
| 103 | milk | 5.1 |
| 104 | water | 2.4 |
| 105 | food | 3.2 |

These values are illustrative.

### Softmax

Softmax converts logits into probabilities.

Formula:

```text
P(i) = exp(zᵢ) / Σ exp(zⱼ)
```

The resulting probabilities sum to 1.

Example:

| Token | Probability |
|---|---:|
| milk | 65% |
| food | 20% |
| water | 10% |
| Dog | 3% |
| drinks | 2% |

The actual probabilities depend on the model and input.

**Important:** The model predicts token IDs, not embedding vectors or necessarily complete words.

The tokenizer maps the selected token ID back to its token text.

---

# 12. How an LLM Generates Text

An LLM generates text one token at a time.

Consider:

```text
Input:
Dog drinks milk
```

The model predicts the next token.

```text
Dog drinks milk
        |
        v
Next token: every
```

The selected token is appended to the input.

```text
Dog drinks milk every
```

The model predicts again:

```text
Next token: day
```

The sequence becomes:

```text
Dog drinks milk every day.
```

This process repeats until the model generates a stopping token or reaches the generation limit.

### Does the model process the whole input again?

Conceptually, the updated sequence is used for the next prediction.

However, efficient implementations use a technique called **KV Cache**.

It stores previously calculated attention Keys and Values, avoiding the need to recompute all previous tokens at every generation step.

---

# 13. LLM Training Process

Training is the process through which a model learns its parameters from data.

The main objective of pretraining a typical autoregressive LLM is next-token prediction.

## Training Flow Diagram

```text
          Training Data
                |
                v
        Data Cleaning
                |
                v
           Tokenization
                |
                v
           Token IDs
                |
                v
      Initialize Parameters
                |
                v
        +------------------+
        |   Forward Pass   |
        |                  |
        | Embeddings       |
        |       |          |
        | Transformer      |
        | Blocks           |
        |       |          |
        | LM Head          |
        |       |          |
        | Predicted Logits |
        +------------------+
                |
                v
          Calculate Loss
                |
                v
         Backpropagation
                |
                v
       Calculate Gradients
                |
                v
       Optimizer Updates
          Parameters
                |
                v
        Repeat Training
                |
                v
       Save Trained Weights
                |
                v
        Fine-tuning (Optional)
```

### 13.1 Collect Training Data

The model is trained on large amounts of data, such as:

- Books
- Websites
- Articles
- Code
- Other text sources

### 13.2 Clean and Prepare Data

Unwanted content, duplicates, and errors can be removed or filtered.

### 13.3 Tokenization

Text is converted into token IDs using the model's tokenizer.

### 13.4 Initialize Parameters

The model's trainable parameters are initialized, generally with random values.

These include embedding weights, attention weights, FFN weights, and other parameters.

### 13.5 Forward Pass

The input token IDs pass through:

1. Embedding layer
2. Positional information mechanism
3. Transformer blocks
4. Final normalization
5. Language Model Head

The model produces logits for the next-token prediction.

### 13.6 Calculate Loss

The model's prediction is compared with the actual next token from the training data.

Loss measures how far the prediction is from the correct target.

For next-token prediction, cross-entropy loss is commonly used.

### 13.7 Backpropagation

Backpropagation calculates gradients.

Gradients indicate how changing model parameters would affect the loss.

### 13.8 Optimizer Updates Parameters

The optimizer uses gradients to update the model's parameters.

For example, AdamW is a commonly used optimizer.

The objective is to reduce the loss over training iterations.

### 13.9 Repeat

The process repeats over many batches of training data.

Over time, the model learns language patterns and improves its predictions.

### 13.10 Save Trained Model

The learned parameters are saved in checkpoint files.

Common formats include:

- `.safetensors`
- `.pth`

A model package may also include:

- Configuration files
- Tokenizer files
- Model weight files
- Weight index files

The weights contain learned numerical values, not a compressed copy of all training text.

### Fine-tuning

After pretraining, a model can undergo additional training to improve its ability to follow instructions and respond conversationally.

Examples include:

- Supervised Fine-Tuning (SFT)
- Preference-based training
- Reinforcement Learning from Human Feedback (RLHF)

---

# 14. Important Differences

| Concept | Meaning |
|---|---|
| Token | A unit of text |
| Token ID | An integer representing a token |
| Tokenizer | Software that converts text into tokens and IDs |
| Tokenizer Vocabulary | The collection of tokens recognized by a tokenizer |
| Embedding | A numerical vector representing a token |
| Embedding Matrix | A collection of token embedding vectors |
| Positional Encoding | A mechanism for representing token positions |
| Self-Attention | Allows tokens to gather information from other tokens |
| Query | Represents what a token is looking for |
| Key | Represents information used for matching |
| Value | Represents information passed through attention |
| Multi-Head Attention | Multiple attention heads operating in parallel |
| FFN | Transforms token representations |
| Residual Connection | Adds a sublayer's input to its output |
| Layer Normalization | Normalizes a vector |
| Transformer Block | A repeated unit containing attention and FFN processing |
| LM Head | Projects a contextual vector to vocabulary logits |
| Logits | Raw scores for vocabulary token IDs |
| Softmax | Converts logits into probabilities |
| Decoding | Selects the next token from the probability distribution |
| Backpropagation | Calculates gradients for parameter updates |
| Optimizer | Updates model parameters using gradients |
| KV Cache | Stores previous attention Keys and Values during generation |

---

## Complete LLM Overview

```text
                  LLM
                   |
          +--------+--------+
          |                 |
          v                 v
       Training          Inference
          |                 |
          v                 v
     Training Data       User Input
          |                 |
          v                 v
     Tokenization       Tokenization
          |                 |
          v                 v
     Token IDs          Token IDs
          |                 |
          v                 v
     Forward Pass       Transformer
          |                 |
          v                 v
     Calculate Loss     LM Head
          |                 |
          v                 v
     Backpropagation    Softmax
          |                 |
          v                 v
     Update Weights     Decode Token
          |                 |
          v                 v
     Save Weights       Generated Text
     
```


## 15. Context Window

The **context window** is the maximum number of tokens an LLM can consider at one time while processing a conversation.

- It includes the input prompt, conversation history, system instructions, and generated responses.
- The context window is measured in tokens, not words.
- When the context window becomes full, the model may no longer have access to older conversation details.
- Depending on the implementation, older messages may be removed, summarized, or the request may be rejected.
- A larger context window allows the model to process more information at once, but does not automatically make it more intelligent.

**Example:**

If a model has a context window of 128,000 tokens, it can process up to that token capacity across its available context and generation.

## 16. Maximum Output Tokens

**Maximum output tokens** define the maximum number of tokens an LLM is allowed to generate in a single response.

- It is a generation limit, not the same as the context window.
- If the model reaches this limit, its response may stop even if the answer is incomplete.
- The limit can be configured through an API parameter such as `max_tokens` or `max_output_tokens`, depending on the provider.
- A model can have a large context window but a smaller maximum output token limit.

**Example:**

If `max_output_tokens = 500`, the model can generate at most 500 output tokens for that response.

## 17. Temperature

**Temperature** controls the randomness of token selection during text generation.

It is applied to the logits before the softmax function calculates probabilities.

Formula:

\[
P(i) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
\]

Where:
- `z` = logits produced by the model.
- `T` = temperature.
- `P(i)` = probability of selecting a particular token.

### How temperature affects generation

- **Temperature = 1:** Uses the original probability distribution.
- **Temperature < 1:** Makes the distribution sharper, increasing the preference for high-probability tokens.
- **Temperature > 1:** Makes the distribution flatter, giving lower-probability tokens a better chance of being selected.
- **Temperature close to 0:** Makes generation more deterministic, usually favoring the highest-probability token.

### How do we actually use temperature?

Temperature is generally configured during inference (when we generate responses), not during model training.

For example, when using an LLM API:

```javascript
const response = await openai.chat.completions.create({
  model: "your-model",
  messages: [
    { role: "user", content: "Write a story about a robot." }
  ],
  temperature: 0.7
});
```

- `0.2`: More predictable and consistent responses.
- `0.7`: A balance between predictability and variety.
- `1.2`: More varied and creative responses.

**Important:** Temperature does not change the model's learned parameters or knowledge. It only changes the probability distribution used during generation.

## 18. Top-K Sampling

**Top-K sampling** restricts token selection to the K tokens with the highest probabilities.

Instead of sampling from the entire vocabulary, the model:

1. Calculates probabilities for all vocabulary tokens.
2. Selects the K tokens with the highest probabilities.
3. Removes all other tokens from consideration.
4. Renormalizes the remaining probabilities.
5. Randomly selects a token from the remaining candidates.

**Example:**

Suppose the model produces these probabilities:

| Token | Probability |
|---|---:|
| Milk | 40% |
| Water | 25% |
| Juice | 15% |
| Coffee | 10% |
| Tea | 6% |
| Soda | 4% |

If `K = 3`, only Milk, Water, and Juice remain.

Their probabilities are renormalized:

| Token | New Probability |
|---|---:|
| Milk | 50% |
| Water | 31.25% |
| Juice | 18.75% |

The model then samples from these three candidates.

## 19. Top-P Sampling (Nucleus Sampling)

**Top-P sampling** selects the smallest group of tokens whose cumulative probability reaches or exceeds a specified threshold P.

Unlike Top-K, the number of candidate tokens is not fixed.

The process:

1. Sort tokens by probability, from highest to lowest.
2. Add their probabilities until the cumulative probability reaches or exceeds P.
3. Remove the remaining tokens.
4. Renormalize the selected probabilities.
5. Randomly select a token from the remaining candidates.

**Example:**

Using the previous probabilities:

| Token | Probability | Cumulative Probability |
|---|---:|---:|
| Milk | 40% | 40% |
| Water | 25% | 65% |
| Juice | 15% | 80% |
| Coffee | 10% | 90% |
| Tea | 6% | 96% |
| Soda | 4% | 100% |

If `Top-P = 0.8`, the model keeps Milk, Water, and Juice because their cumulative probability reaches 80%.

**Difference between Top-K and Top-P:**

- Top-K keeps a fixed number of candidate tokens.
- Top-P keeps a variable number of candidate tokens based on cumulative probability.

## 20. Difference Between Temperature, Top-K, and Top-P

All three are decoding controls that influence how an LLM selects its next token.

| Technique | What it controls |
|---|---|
| Temperature | How sharp or flat the probability distribution becomes. |
| Top-K | How many of the highest-probability tokens can be considered. |
| Top-P | How much cumulative probability the candidate tokens must cover. |

They can be used together, depending on the model and generation implementation.

A simplified generation pipeline:

```text
Input Context
     |
     v
Transformer Model
     |
     v
Logits
     |
     v
Temperature
     |
     v
Softmax (Probabilities)
     |
     v
Top-K / Top-P Filtering
     |
     v
Renormalization
     |
     v
Sampling
     |
     v
Selected Token ID
     |
     v
Tokenizer Decodes Token
     |
     v
Generated Text
```

Note: The exact order of temperature and sampling filters can vary between implementations.

## 21. Hallucination in LLMs

**Hallucination** occurs when an LLM generates information that sounds convincing but is factually incorrect, fabricated, or unsupported.

LLMs are trained primarily to predict the next token, not to independently verify every factual statement.

### How does hallucination happen?

1. **Next-token prediction:** The model predicts likely tokens rather than independently checking whether every statement is true.
2. **Outdated knowledge:** The model may rely on information learned during training that is no longer current.
3. **Missing information:** When necessary information is unavailable, the model may generate a plausible guess.
4. **Poor training data:** Incorrect or low-quality training data can influence the model to produce incorrect answers.
5. **Conflicting information:** Contradictory information in training data or the provided context can lead to incorrect responses.
6. **Ambiguous prompts:** The model may misunderstand the user's intention and generate an irrelevant or incorrect answer.
7. **Fabricated information:** The model may generate fake references, names, statistics, URLs, or citations.
8. **Context loss:** When relevant older conversation details are no longer available, the model may produce an incorrect answer based on incomplete context.
9. **High temperature:** Higher randomness can increase the chance of unusual or incorrect responses, although temperature is not the root cause of hallucination.

### How can we reduce hallucinations?

- Use high-quality and reliable training data.
- Provide clear and specific prompts.
- Use RAG (Retrieval-Augmented Generation) to provide relevant external information.
- Use web search to access up-to-date information when needed.
- Verify important facts using reliable sources.
- Encourage the model to acknowledge uncertainty instead of guessing.

**Important:** These techniques can reduce hallucinations, but they cannot guarantee that an LLM will always produce correct information.

### What is RAG (Retrieval-Augmented Generation)?

RAG is a technique that retrieves relevant information from external sources, such as documents or databases, and provides it to an LLM as additional context before generating a response.

**Basic RAG workflow:**

```text
User Question
      |
      v
Retrieve Relevant Information
      |
      v
Provide Information to LLM
      |
      v
LLM Generates Response
      |
      v
Final Answer
```

RAG helps the model answer using relevant external information instead of relying only on its learned parameters. However, incorrect or incomplete retrieved information can still lead to incorrect answers.


**My key takeaway:**

An LLM learns by repeatedly predicting the next token and updating its parameters based on prediction errors.

During inference, it uses those learned parameters to process input tokens, calculate probabilities for possible next tokens, and generate text one token at a time.

---

## Learning Resources

- [Attention Is All You Need (Original Transformer Paper)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

*These are my personal learning notes. I will continue updating this README as I learn more about LLMs.*