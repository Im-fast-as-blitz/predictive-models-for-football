import torch
import torch.nn as nn


class InvertedDataEmbedding(nn.Module):
    def __init__(self, t: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(t, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.value_embedding(x)
        x = self.activation(x)
        return self.dropout(x)
