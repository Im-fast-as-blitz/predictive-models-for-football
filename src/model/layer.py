import torch
import torch.nn as nn

from src.model.attention import MultiHeadAttention
from src.model.ffn import FeedForward
from src.model.norm import TemporalLayerNorm

class Layer(nn.Module):
    def __init__(self,
                 args,
                 hidden_size: int = 128, 
                 dropout: float = 0.1
                 ):
        super(Layer, self).__init__()
        
        self.attention = MultiHeadAttention(args["attention"], hidden_size=hidden_size, dropout=dropout)

        self.ln1 = TemporalLayerNorm(hidden_size)

        self.ffn = FeedForward(args["ffn"], hidden_size, dropout)

        self.ln2 = TemporalLayerNorm(hidden_size)

    def forward(self, x):
        x = self.attention(x) + x
        x = self.ln1(x)
        x = self.ffn(x) + x
        x = self.ln2(x)
        return x
        
