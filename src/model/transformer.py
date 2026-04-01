import torch
import torch.nn as nn

from src.model import Layer


class Transformer(nn.Module):
    def __init__(self, args, in_features):
        super(Transformer, self).__init__()
        
        # self.embeder = ...

        # self.layer = Layer()

        # self.projector = ...

        self.hidden_size = args["hidden_size"]
        self.out_features = args["out_features"]

        self.tmp = nn.Sequential(
            nn.Linear(in_features, self.hidden_size),
            nn.Linear(self.hidden_size, self.out_features)
        )

    def forward(self, x):
        x = self.tmp(x)
        return x
        
