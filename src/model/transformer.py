import torch
import torch.nn as nn

from src.model import Layer


class Transformer(nn.Module):
    def __init__(self, args, in_features):
        super(Transformer, self).__init__()

        self.hidden_size = args["hidden_size"]
        self.n_heads = args["n_heads"]
        self.dropout = args["dropout"]
        self.out_features = args["out_features"]


        # self.embeder = ...

        self.layer = Layer(hidden_size = self.hidden_size, n_heads = self.n_heads, dropout=self.dropout)

        # self.projector = ...

        self.tmp = nn.Sequential(
            nn.Linear(in_features, self.hidden_size),
            nn.Linear(self.hidden_size, self.out_features)
        )

    def forward(self, x):
        x = self.tmp(x)
        return x
        
