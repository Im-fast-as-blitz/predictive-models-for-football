import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
        self,
        args,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        d_ff = args["d_ff"]

        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
