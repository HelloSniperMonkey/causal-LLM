import numpy as np

class SingleHeadAttention:
    def __init__(self, embed_size, head_size):
        self.embed_size = embed_size
        self.head_size = head_size
        self.W_q = np.random.randn(embed_size, head_size) * 0.01
        self.W_k = np.random.randn(embed_size, head_size) * 0.01
        self.W_v = np.random.randn(embed_size, head_size) * 0.01

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_size)
        Q = np.dot(x, self.W_q)  # (batch_size, seq_length, head_size)
        K = np.dot(x, self.W_k)  # (batch_size, seq_length, head_size)
        V = np.dot(x, self.W_v)  # (batch_size, seq_length, head_size)

        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.head_size)
        # masking future tokens
        for i in range(len(scores)):
            for j in range(len(scores[i])):
                if j > i:
                    scores[i][j] = -1e9  # Masking future tokens
        weights = self.softmax(scores)
        out = np.matmul(weights, V)  # (batch_size, seq_length, head_size)

        out = np.dot(out, self.W_o)  # (batch_size, seq_length, embed_size)
        return out

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

class MultiHeadAttention:
    def __init__(self, embed_size, num_heads):
        self.heads = [SingleHeadAttention(embed_size, embed_size) for _ in range(num_heads)]
        self.W_o = np.random.randn(embed_size*num_heads, embed_size) * 0.01

    def forward(self, x):
        prefinal_token = np.concatenate([head.forward(x) for head in self.heads], axis=-1)
        return np.dot(prefinal_token, self.W_o)
