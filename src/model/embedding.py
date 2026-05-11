import torch
import torch.nn as nn


class InvertedDataEmbedding(nn.Module):
    """
    Вход (три варианта):
      mode: ts_full - [B, N, T, F] — N variates, T шагов, F признаков на шаг;
      mode: ts - [B, N, T] — добавляется ось F=1;
      mode: stats - [B, N, F] — добавляется ось T=1.
    Выход: [B, N, D]
    """

    def __init__(
        self,
        args,
        feat_dim: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.t = args["t"]
        self.feat_dim = feat_dim
        self.mode = args["mode"]

        self.value_embedding = nn.Linear(self.t * self.feat_dim, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            if self.mode == "ts":
                if self.feat_dim != 1:
                    raise ValueError(
                        f"Вход [B, N, T] только при feat_dim=1; "
                        f"сейчас feat_dim={self.feat_dim}, передай [B, N, T, {self.feat_dim}]."
                    )
                x = x.unsqueeze(-1)
            if self.mode == "stats":
                if self.t != 1:
                    raise ValueError(
                        f"Вход [B, N, F] только при t=1; "
                        f"сейчас t={self.feat_dim}, передай [B, N, T, {self.feat_dim}]."
                    )
                x = x.unsqueeze(-2)
            
        if self.mode == "ts_full" and x.dim() != 4:
            raise ValueError(
                f"Ожидается [B, N, T, F], получили {tuple(x.shape)}"
            )

        B, N, T, F = x.shape
        if T != self.t or F != self.feat_dim:
            raise ValueError(
                f"Ожидалось T={self.t}, F={self.feat_dim}, получили T={T}, F={F}"
            )

        x = x.reshape(B, N, T * F)
        x = self.value_embedding(x)
        x = self.activation(x)
        return self.dropout(x)
