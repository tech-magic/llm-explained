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
