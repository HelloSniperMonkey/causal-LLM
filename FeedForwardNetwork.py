import numpy as np

class FeedForwardNetwork:
    def __init__(self, embed_size, hidden_size):
        self.W1 = np.random.randn(embed_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, embed_size) * 0.01
        self.b2 = np.zeros((1, embed_size))

    def forward(self, x):
        z1 = np.dot(x, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU activation
        z2 = np.dot(a1, self.W2) + self.b2
        return z2