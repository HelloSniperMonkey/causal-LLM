import numpy as np
import torch 
from transformers import AutoTokenizer
import pickle
import torch.nn.functional as F

# ----------------
# Hyperparameters
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
EMBED_SIZE = 512
NUM_HEADS = 8
HIDDEN_SIZE = 2048
NUM_LAYERS = 8
VOCAB_SIZE = 50257
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
TOP_P = 0.9
EPS = 1e-6
# ----------------

class SingleHeadAttention:
    def __init__(self, embed_size):
        self.embed_size = embed_size
        self.head_size = embed_size // NUM_HEADS
        self.W_q = torch.randn(embed_size, self.head_size) * 0.01
        self.W_k = torch.randn(embed_size, self.head_size) * 0.01
        self.W_v = torch.randn(embed_size, self.head_size) * 0.01

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_size)
        Q = torch.matmul(x, self.W_q)  # (batch_size, seq_length, head_size)
        K = torch.matmul(x, self.W_k)  # (batch_size, seq_length, head_size)
        V = torch.matmul(x, self.W_v)  # (batch_size, seq_length, head_size)

        scores = torch.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.head_size)
        # masking future tokens
        for i in range(len(scores)):
            for j in range(len(scores[i])):
                if j > i:
                    scores[i][j] = -1e9  # Masking future tokens
        weights = self.softmax(scores)
        out = torch.matmul(weights, V)  # (batch_size, seq_length, head_size)

        # out = torch.matmul(out, self.W_o)  # (batch_size, seq_length, embed_size)
        return out

    def softmax(self, x):
        e_x = torch.exp(x - torch.max(x, axis=-1, keepdims=True).values)
        return e_x / e_x.sum(axis=-1, keepdims=True)

class MultiHeadAttention:
    def __init__(self, embed_size, num_heads):
        self.heads = [SingleHeadAttention(embed_size) for _ in range(num_heads)]
        self.W_o = torch.randn(embed_size, embed_size) * 0.01

    def forward(self, x):
        prefinal_token = torch.cat([head.forward(x) for head in self.heads], dim=-1)
        return torch.matmul(prefinal_token, self.W_o)

class FeedForwardNetwork:
    def __init__(self, embed_size, hidden_size):
        self.W1 = torch.randn(embed_size, hidden_size) * 0.01
        self.b1 = torch.zeros((1, hidden_size))
        self.W2 = torch.randn(hidden_size, embed_size) * 0.01
        self.b2 = torch.zeros((1, embed_size))

    def forward(self, x):
        z1 = torch.matmul(x, self.W1) + self.b1
        a1 = torch.maximum(torch.tensor(0.0), z1)  # ReLU activation
        z2 = torch.matmul(a1, self.W2) + self.b2
        return z2
    

def embed(tokens, embedding_matrix):
    return embedding_matrix[tokens]

def LayerNorm(x, eps=EPS):
    mean = torch.mean(x, axis=-1, keepdims=True)
    std = torch.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def forward_pass(tokens):
    multi_head_attention = MultiHeadAttention(embed_size=EMBED_SIZE, num_heads=NUM_HEADS)
    H = tokens + multi_head_attention.forward(tokens)
    normalized_H = LayerNorm(H)
    ffn = FeedForwardNetwork(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE)
    O = normalized_H + ffn.forward(normalized_H)
    normalized_O = LayerNorm(O)
    return normalized_O

def multi_layer_forward(tokens, num_layers=NUM_LAYERS):
    for _ in range(num_layers):
        tokens = forward_pass(tokens)
    return tokens

def de_embed(tokens, embedding_matrix):
    return torch.matmul(tokens, embedding_matrix.T)

def logits_to_probs(logits):
    exp_logits = torch.exp(logits - torch.max(logits, axis=-1, keepdims=True).values)
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

def top_p_sampling(probs, p=TOP_P):
    sorted_indices = torch.argsort(probs, descending=True)
    sorted_probs = probs[sorted_indices]
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff_index = torch.searchsorted(cumulative_probs, p) + 1
    top_indices = sorted_indices[:cutoff_index]
    top_probs = sorted_probs[:cutoff_index]
    top_probs /= top_probs.sum()
    return torch.multinomial(top_probs, num_samples=1).item()


if __name__ == "__main__":
    try:
        with open("embedding_matrix.pkl", "rb") as f:
            E = torch.tensor(pickle.load(f))
    except FileNotFoundError:
        E = torch.randn(VOCAB_SIZE, EMBED_SIZE) * 0.01  # Example embedding matrix
        
    with open("input.txt", "r") as f:
        input_text = f.read().strip()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = tokenizer(input_text, return_tensors="pt")["input_ids"]
    embedded_tokens = embed(tokens, E)

    data = torch.tensor(embedded_tokens, dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    # --- Corrected Training Logic ---

    # For language modeling, the input is all but the last token
    x = train_data[:, :-1]
    # The target is all but the first token (shifted by one)
    y = train_data[:, 1:]

    # Get model predictions
    processed_tokens = multi_layer_forward(x)
    logits = de_embed(processed_tokens, E)

    # Reshape for loss calculation
    # (batch, seq_length, vocab_size) -> (batch * seq_length, vocab_size)
    b, t, c = logits.shape
    logits_for_loss = logits.view(b * t, c)
    # (batch, seq_length) -> (batch * seq_length)
    targets = y.view(b * t)

    # Calculate cross-entropy loss
    loss = F.cross_entropy(logits_for_loss, targets)
    
    print(f"Loss: {loss.item()}")

    # --- Generation (Inference) Logic ---
    # To generate text, you would typically start with a context 
    # and generate one token at a time in a loop.
    # This part is simplified for demonstration.

    # Use the logits from the last time step to generate the next token
    probs = logits_to_probs(logits[:, -1, :]) # Get probs for the last token
    next_token_generated = top_p_sampling(probs, p=TOP_P)
    generated_text = tokenizer.decode(next_token_generated.tolist())
    print("Generated text:", generated_text)

else:
    print("This script is being imported as a module.")