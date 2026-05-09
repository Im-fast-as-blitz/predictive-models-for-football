import torch
import torch.nn as nn


class InvertedDataEmbedding(nn.Module):
    """
    Вход (два варианта):
      - [B, N, T, F] — N variates, T шагов, F признаков на шаг;
      - [B, N, T] — добавляется ось F=1.
    Выход: [B, N, D]
    """

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.value_embedding = nn.Linear(seq_len * feat_dim, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            if self.feat_dim != 1:
                raise ValueError(
                    f"Вход [B, N, T] только при feat_dim=1; "
                    f"сейчас feat_dim={self.feat_dim}, передай [B, N, T, {self.feat_dim}]."
                )
            x = x.unsqueeze(-1)
        elif x.dim() != 4:
            raise ValueError(
                f"Ожидается [B, N, T] или [B, N, T, F], получили {tuple(x.shape)}"
            )

        B, N, T, F = x.shape
        if T != self.seq_len or F != self.feat_dim:
            raise ValueError(
                f"Ожидалось T={self.seq_len}, F={self.feat_dim}, получили T={T}, F={F}"
            )

        x = x.reshape(B, N, T * F)
        x = self.value_embedding(x)
        x = self.activation(x)
        return self.dropout(x)
