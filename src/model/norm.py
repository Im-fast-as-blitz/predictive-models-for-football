import torch
import torch.nn as nn

class TemporalLayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]

        mean = x.mean(dim=1, keepdim=True)   # [B, 1, F]
        std  = x.std(dim=1, keepdim=True)    # [B, 1, F]

        x_norm = (x - mean) / (std + self.eps)

        return self.gamma * x_norm + self.beta   # [B, T, F]
