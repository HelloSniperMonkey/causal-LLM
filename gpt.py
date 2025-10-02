import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import pickle
import os

# ----------------
# Hyperparameters
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
EMBED_SIZE = 512
NUM_HEADS = 8
HIDDEN_SIZE = 2048
NUM_LAYERS = 8
VOCAB_SIZE = 50257
TOP_P = 0.9
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=0.9):
        """
        Generate new tokens given a context.
        
        Args:
            idx: (B, T) tensor of token indices
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (higher = more random)
            top_p: nucleus sampling parameter
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context if it exceeds max_seq_len
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            # Get predictions
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            
            # Apply top-p (nucleus) sampling
            probs = F.softmax(logits, dim=-1)
            
            # Sort probabilities
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Set probabilities to 0 for removed indices
            sorted_probs[sorted_indices_to_remove] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            
            # Sample from the filtered distribution
            next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_token_idx)
            
            # Append to sequence
            idx = torch.cat([idx, next_token], dim=1)
        
        return idx


def load_model(weights_path='model_weights.pkl'):
    """Load the trained model weights."""
    model = GPT(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, HIDDEN_SIZE, NUM_LAYERS).to(device)
    
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        with open(weights_path, 'rb') as f:
            weights = pickle.load(f)
        
        # Load embedding matrix
        if 'embedding_matrix' in weights:
            model.embedding.weight.data = weights['embedding_matrix'].to(device)
            print("✓ Loaded embedding matrix")
        
        # Load multi-head attention weights
        if 'multi_head_attention' in weights:
            for i, block in enumerate(model.blocks):
                if i < len(weights['multi_head_attention']):
                    block.attn.load_state_dict(weights['multi_head_attention'][i])
            print(f"✓ Loaded attention weights for {len(weights['multi_head_attention'])} layers")
        
        # Load feed-forward weights
        if 'feed_forward' in weights:
            for i, block in enumerate(model.blocks):
                if i < len(weights['feed_forward']):
                    block.ffn.load_state_dict(weights['feed_forward'][i])
            print(f"✓ Loaded feed-forward weights for {len(weights['feed_forward'])} layers")
        
        print("Model loaded successfully!")
    else:
        print(f"Warning: No weights file found at {weights_path}. Using randomly initialized model.")
    
    model.eval()
    return model


def chat(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P):
    """Generate a response to the given prompt."""
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    
    # Decode and return
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    print("=" * 60)
    print("GPT Chat Interface")
    print("=" * 60)
    print(f"Device: {device}")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
    print()
    
    # Load model
    model = load_model('model_weights.pkl')
    print()
    
    print("=" * 60)
    print("Chat started! Type 'quit', 'exit', or 'q' to end the conversation.")
    print("=" * 60)
    print()
    
    # Interactive chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            print("GPT: ", end="", flush=True)
            response = chat(model, tokenizer, user_input)
            
            # Extract only the generated part (remove the prompt)
            if response.startswith(user_input):
                generated = response[len(user_input):].strip()
            else:
                generated = response
            
            print(generated)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")
