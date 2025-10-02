import numpy as np
import torch 
import torch.nn as nn
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

class SingleHeadAttention(nn.Module):
    def __init__(self, embed_size, head_size):
        super().__init__()
        self.head_size = head_size
        self.W_q = nn.Linear(embed_size, head_size, bias=False)
        self.W_k = nn.Linear(embed_size, head_size, bias=False)
        self.W_v = nn.Linear(embed_size, head_size, bias=False)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size ** 0.5)
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        head_size = embed_size // num_heads
        self.heads = nn.ModuleList([SingleHeadAttention(embed_size, head_size) for _ in range(num_heads)])
        self.W_o = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.W_o(out)

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_size)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_size):
        super().__init__()
        self.attn = MultiHeadAttention(embed_size, num_heads)
        self.ffn = FeedForwardNetwork(embed_size, hidden_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_size, num_layers, max_seq_len=1000):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_size)
        self.blocks = nn.ModuleList([Block(embed_size, num_heads, hidden_size) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)
        # tie weights
        self.lm_head.weight = self.embedding.weight

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    

def get_batch(split, batch_size, block_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)


if __name__ == "__main__":
    with open("corpus.txt", "r") as f:
        input_text = f.read().strip()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = tokenizer(input_text)["input_ids"]
    data = torch.tensor(tokens, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Hyperparameters for training
    block_size = 256
    num_epochs = 10
    steps_per_epoch = 100

    model = GPT(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, HIDDEN_SIZE, NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            x, y = get_batch('train', BATCH_SIZE, block_size)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Save weights
    weights = {
        'embedding_matrix': model.embedding.weight.cpu().detach(),
        'multi_head_attention': [block.attn.state_dict() for block in model.blocks],
        'feed_forward': [block.ffn.state_dict() for block in model.blocks]
    }
    with open('model_weights.pkl', 'wb') as f:
        pickle.dump(weights, f)
    print("Weights saved to model_weights.pkl")

else:
    print("This script is being imported as a module.")