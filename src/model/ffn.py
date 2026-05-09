from typing import Optional

import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return self.dropout2(x)
