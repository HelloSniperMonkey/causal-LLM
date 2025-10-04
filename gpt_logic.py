import numpy as np
from AttentionHead import MultiHeadAttention
import nltk
from FeedForwardNetwork import FeedForwardNetwork
from transformers import AutoTokenizer

E = embedding_matrix

def embed(tokens, embedding_matrix):
    return np.dot(tokens, embedding_matrix)

def LayerNorm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def forward_pass(tokens):
    multi_head_attention = MultiHeadAttention(embed_size=512, num_heads=8)
    H = tokens + multi_head_attention.forward(tokens)
    normalized_H = LayerNorm(H)
    ffn = FeedForwardNetwork(embed_size=512, hidden_size=2048)
    O = normalized_H + ffn.forward(normalized_H)
    normalized_O = LayerNorm(O)
    return normalized_O

def multi_layer_forward(tokens, num_layers=16):
    for _ in range(num_layers):
        tokens = forward_pass(tokens)
    return tokens

def de_embed(tokens, embedding_matrix):
    return np.dot(tokens, embedding_matrix.T)

def logits_to_probs(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

def top_p_sampling(probs, p=0.9):
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff_index = np.searchsorted(cumulative_probs, p) + 1
    top_indices = sorted_indices[:cutoff_index]
    top_probs = sorted_probs[:cutoff_index]
    top_probs /= top_probs.sum()
    return np.random.choice(top_indices, p=top_probs)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_text = "Your input text here"
    tokens = tokenizer(input_text, return_tensors="np")["input_ids"]
    embedded_tokens = embed(tokens, E)
    processed_tokens = multi_layer_forward(embedded_tokens)
    logits = de_embed(processed_tokens, E)
    probs = logits_to_probs(logits[0, -1])
    next_token = top_p_sampling(probs, p=0.9)
    generated_text = tokenizer.decode([next_token])
    print("Generated text:", generated_text)

else:
    print("This script is being imported as a module.")