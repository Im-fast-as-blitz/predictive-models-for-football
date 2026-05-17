from __future__ import annotations

import torch
import torch.nn as nn


class InvertedDataEmbedding(nn.Module):
    """
    Вход (три варианта):
      mode: ts_full - [B, N, T, F] — N variates, T шагов, F признаков на шаг;
      mode: ts - [B, N, T] — добавляется ось F=1;
      mode: stats - [B, N, F] — добавляется ось T=1.

      Категориальные признаки x_cat [B, N, T, C]
    Выход: [B, N, D]
    """
    def __init__(
        self,
        args,
        feat_dim: int,
        hidden_size: int,
        cat_cardinalities: list[int] = (),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.t = args["t"]
        self.feat_dim = feat_dim
        self.mode = args["mode"]
        cat_emb_dim = int(args["cat_emb_dim"])

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_classes, cat_emb_dim)
            for num_classes in cat_cardinalities
        ])

        total_feat_dim = feat_dim + len(cat_cardinalities) * cat_emb_dim

        self.value_embedding = nn.Linear(self.t * total_feat_dim, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_cat: torch.Tensor, full_history: torch.Tensor) -> torch.Tensor:
        if self.mode == "ts_full":
            if full_history.dim() != 4:
                raise ValueError(
                    f"Ожидается [B, N, T, F], получили {tuple(full_history.shape)}"
                )
            x = full_history
        else:
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
                            f"сейчас t={self.t}, передай [B, N, T, {self.feat_dim}]."
                        )
                    x = x.unsqueeze(-2)

        B, N, _, _ = x.shape

        cat_parts = [emb(x_cat[..., i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_embedded = torch.cat(cat_parts, dim=-1)
        x = torch.cat([x, cat_embedded], dim=-1)

        x = x.reshape(B, N, -1)
        x = self.value_embedding(x)
        x = self.activation(x)
        return self.dropout(x)
