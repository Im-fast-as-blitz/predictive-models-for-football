import torch
import torch.nn as nn

from src.model import MultiHeadAttention


class Layer(nn.Module):
    def __init__(self,
                 hidden_size: int = 128, 
                 n_heads: int = 8, 
                 dropout: float = 0.1
                 ):
        super(Layer, self).__init__()
        
        self.attention = MultiHeadAttention(hidden_size=hidden_size, n_heads=n_heads, dropout=dropout)

        # self.ln1 = nn.LayerNorm(...)

        # self.ffn = ...

        # self.ln2 = nn.LayerNorm(...)

    def forward(self, x):
        pass
        raise Exception("Not implement yet")
        
