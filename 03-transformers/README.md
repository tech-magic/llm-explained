# 🤖🧠🌍 GPT and Transformers (Explained For Laymen)

Transformers are a type of machine learning model that revolutionized natural language processing. Imagine trying to understand a story not just word-by-word but by understanding how each word relates to every other word in the sentence. Transformers make this possible by looking at all words in a sentence at once and figuring out how much attention each word should pay to every other word. This is known as **self-attention**.

**GPT (Generative Pre-trained Transformer)** models, like GPT-3, are built on top of this idea. They are essentially stacks of transformer layers trained on massive amounts of text to predict the next word in a sentence. By using many layers and multiple "heads" in each layer, the model can learn complex relationships between words at various levels of abstraction.

- **Multiple Heads**: Each head can focus on a different type of relationship between words (e.g., grammar, meaning, topic).
- **Multiple Layers**: Stacking layers allows the model to refine its understanding step by step, like building deeper levels of comprehension.

### 🎛️ GPT-3 Parameter Count Breakdown

Below is a simplified breakdown of how GPT-3 (175 billion parameters) might reach such a large parameter count:

| Component               | Formula                                           | Count (Approx)         |
|------------------------|----------------------------------------------------|------------------------|
| Embedding Layer        | vocab_size × d_model                               | 50,000 × 12,288        |
| Attention QKV          | 3 × d_model × d_model                              | 3 × 12,288²             |
| Attention Output       | d_model × d_model                                  | 12,288²                |
| Feedforward Layer      | 4 × d_model × d_model                              | 4 × 12,288²            |
| Unembedding Layer      | d_model × vocab_size                               | 12,288 × 50,000        |
| **Per Layer Total**    | QKV + Out + FF                                     | ~1.2B per layer        |
| **Total Layers (96)**  | 96 × per-layer params                              | ~115B                  |
| **Total Parameters**   | Embeddings + Layers + Unembedding                 | ~175B                  |

> Note: These numbers are rough estimates and depend on implementation details, weight sharing, and compression tricks.

---

# 🤖🧠🌍 Transformer Simulation Walkthrough (Step-by-Step)

This document explains the 9 key steps involved in developing a basic Transformer model, as demonstrated in the given Python program (`gpt_demo.py`). The aim is to make each step understandable through simple hand-calculations and analogies.

---

## 🔧 Configuration Summary

We simulate a Transformer model with:

- **9 tokens**: `["the", "cat", "sat", "over", "mat", "because", "it", "was", "tired"]`
- **Embedding dimension**: 16
- **Model dimension (Q/K/V)**: 5
- **Number of heads**: 2
- **Number of layers**: 2

---

## ⚙️ Step 1: Initialize Embeddings

Each token is converted into a fixed-size vector (`embedding_dim = 16`), consisting of:

- First 8 values: normalized **character encoding** (`a = 0.0`, ..., `z = 1.0`)
- Last 8 values: **position encoding**, where the `position`-th index is set to `1`.

### 🧮 Example (token = "cat", position = 1):

Characters:
- `'c' → 2 / 25 = 0.08`
- `'a' → 0 / 25 = 0.0`
- `'t' → 19 / 25 ≈ 0.76`

Result (first part of vector): `[0.08, 0.0, 0.76, 0.0, ..., 0.0]`

Position encoding:
- Set index `8 + 1 = 9` to `1`

Final vector: `[0.08, 0.0, 0.76, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ..., 0.0]`

---

## 🧲 Step 2: Initialize Q/K/V Weights

For each head, we randomly initialize weight matrices to map each token’s embedding into three representations:

- **Q (Query)**, **K (Key)**, **V (Value)** matrices:  
  Each is a `5 x 16` matrix → transforms a 16-dim embedding into a 5-dim vector.

These are like asking:
- Query: "What am I looking for?"
- Key: "What do I contain?"
- Value: "What can I offer?"

---

## 🧮 Step 3: Compute Q, K, V Vectors

For each token, we compute:

```python
Q[i] = W_Q @ embedding[i]
K[i] = W_K @ embedding[i]
V[i] = W_V @ embedding[i]
```

Each token gets its own 5D vector for Q, K, and V.

---

## 🔍 Step 4: Attention Scores & Weighted Sum

For each token `i`:
1. Dot product `q_i @ k_j` for every other token `j`
2. Scale by `√d_model = √5 ≈ 2.24`
3. Apply **softmax** to get attention weights
4. Multiply each `v_j` by corresponding weight and **sum**

### 🧠 Intuition:
The token "asks" others for help based on how well their key matches its query.

### 🧮 Mini Example:

If:
- `q_i = [1, 0, 0, 0, 0]`
- `k_j = [[1, 0, 0, 0, 0], [0.5, 0, 0, 0, 0]]`

Then:
- Dot products: `[1, 0.5]`
- Scaled: `[1/√5, 0.5/√5] ≈ [0.45, 0.22]`
- Softmax: `[0.57, 0.43]`

Result:
- `attention_output = 0.57*v_0 + 0.43*v_1`

---

## 🧠 Step 5: Concatenate Multi-Head Outputs

Each head gives a 5D vector per token → combine results:

```python
concat = head1_output + head2_output  # → 10D vector per token
```

---

## 🧪 Step 6: Linear Projection

Use a matrix `W_proj` of shape `(5, 10)` to **project** the 10D concatenated vector back to a `d_model = 5` dimensional vector.

```python
projected = W_proj @ concat
```

---

## 🔁 Step 7: Residual Connection + LayerNorm + Feedforward

1. **Residual Connection**: Add original embedding to `projected`
   > If original = `[1, 2, 3, 4, 5]`, projected = `[5, 4, 3, 2, 1]` → sum = `[6, 6, 6, 6, 6]`

2. **LayerNorm** (simplified):  
   Normalize vector by subtracting mean and dividing by std deviation

3. **Feedforward network**:
   - Two linear layers with ReLU in between
   - Boosts model's nonlinear capacity

---

## 🔄 Step 8: Set New Embeddings for Next Layer

Replace each token's embedding with the result from Step 7.

This enables deeper layers to process **refined representations**.

---

## 🎯 Step 9: Final Output

After completing all layers:

```python
print(f"{token}: {final_vector}")
```

You now have **contextualized embeddings** for each token, meaning:
- `"tired"` knows it was influenced by `"because"` and `"it"`  
- `"sat"` may pay attention to `"cat"` and `"mat"`

---

## ✅ Summary of Steps

| Step | Description |
|------|-------------|
| 1    | Embed tokens (char + position) |
| 2    | Initialize weights (Q, K, V) |
| 3    | Compute Q, K, V for each token |
| 4    | Attention score + softmax + weighted sum |
| 5    | Concatenate head outputs |
| 6    | Project to model dimension |
| 7    | Residual + LayerNorm + Feedforward |
| 8    | Update token embeddings |
| 9    | Output final representations |

---

🧾 **Tip for Learning**:  
To simulate with real numbers, pick small tokens (e.g., `"at"`, `"on"`) and set embedding size to 4. Manually calculate 1 head, 1 layer attention — it’s a fun way to understand it from scratch!

## 🐍 Python Code

```python
import numpy as np

# -------------------------------
# Configuration
# -------------------------------
tokens = ["the", "cat", "sat", "over", "mat", "because", "it", "was", "tired"]
embedding_dim = 16   # Initial token embedding (char + position)
d_model = 5          # Dim for Q/K/V vectors
n_heads = 2
n_layers = 2

np.random.seed(42)
char_vocab = {chr(i + ord('a')): i for i in range(26)}

# -------------------------------
# Token Embedding Function
# -------------------------------
def word_to_embedding(word, position):
    vec = np.zeros(embedding_dim)
    for i, c in enumerate(word.lower()):
        if i >= embedding_dim // 2:
            break
        if c in char_vocab:
            vec[i] = char_vocab[c] / 25.0
    pos_offset = embedding_dim // 2
    if position < embedding_dim // 2:
        vec[pos_offset + position] = 1.0
    return vec

# -------------------------------
# Softmax Function
# -------------------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# -------------------------------
# Feedforward Network
# -------------------------------
def feedforward(x, hidden_dim=32):
    W1 = np.random.rand(hidden_dim, len(x))
    b1 = np.random.rand(hidden_dim)
    W2 = np.random.rand(len(x), hidden_dim)
    b2 = np.random.rand(len(x))
    x = np.maximum(0, W1 @ x + b1)  # ReLU
    return W2 @ x + b2

# -------------------------------
# Transformer Simulation
# -------------------------------
# Step 1: Initialize embeddings
token_embeddings = [word_to_embedding(token, idx) for idx, token in enumerate(tokens)]

for layer_num in range(n_layers):
    print(f"\n====== Layer {layer_num} ======")
    
    head_outputs = []

    for head_num in range(n_heads):
        print(f"\n-- Head {head_num} --")

        # Step 2: Initialize weights
        W_Q = np.random.rand(d_model, embedding_dim)
        W_K = np.random.rand(d_model, embedding_dim)
        W_V = np.random.rand(d_model, embedding_dim)

        # Step 3: Compute Q/K/V for each token
        Q = [W_Q @ emb for emb in token_embeddings]
        K = [W_K @ emb for emb in token_embeddings]
        V = [W_V @ emb for emb in token_embeddings]

        # Step 4: Compute Attention Scores & Output
        head_result = []
        for i in range(len(tokens)):
            q_i = Q[i]
            scores = np.array([q_i @ k_j for k_j in K]) / np.sqrt(d_model)
            weights = softmax(scores)
            weighted_sum = np.sum([w * v for w, v in zip(weights, V)], axis=0)
            head_result.append(weighted_sum)

        head_outputs.append(head_result)

    # Step 5: Concatenate multi-head outputs
    combined_outputs = []
    for i in range(len(tokens)):
        concat = np.concatenate([head_outputs[h][i] for h in range(n_heads)])
        combined_outputs.append(concat)

    # Step 6: Linear projection to d_model
    projection_W = np.random.rand(d_model, d_model * n_heads)
    projected_outputs = [projection_W @ out for out in combined_outputs]

    # Step 7: Residual connection + LayerNorm + Feedforward
    new_embeddings = []
    for orig, proj in zip(token_embeddings, projected_outputs):
        # Project original embedding to match proj dim if needed
        if len(orig) != len(proj):
            residual = np.pad(orig, (0, len(proj) - len(orig)))
        else:
            residual = orig

        # Add & Normalize (LayerNorm simplified)
        summed = proj + residual[:len(proj)]
        normed = (summed - np.mean(summed)) / (np.std(summed) + 1e-6)

        # Feedforward transformation
        ff_output = feedforward(normed)

        new_embeddings.append(ff_output)

    # Step 8: Set new input for next layer
    token_embeddings = new_embeddings

# Final token representations:
print("\n\n✅ Final token representations after last layer:\n")
for token, vec in zip(tokens, token_embeddings):
    print(f"{token:>8}: {np.round(vec, 3)}")
```
