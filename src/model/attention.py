import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(nn.Module):
    def __init__(self, args, hidden_size: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = args["n_heads"]
        self.max_tours = args["depth"]

        assert hidden_size % self.n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = hidden_size
        self.d_k = hidden_size // self.n_heads

        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

        # 0=не играли, 1=поражение, 2=ничья, 3=победа
        self.result_embed = nn.Embedding(4, hidden_size)

        # Позиционный энкодинг по турам
        self.tour_pe = nn.Embedding(self.max_tours - 1, hidden_size)

        # Свертка истории пары → один вектор
        self.history_proj = nn.Linear(hidden_size, hidden_size)

        # Проекция R в пространство головы для каждой головы
        # [n_heads, d_k] чтобы считать relation score на каждой голове
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)

    def _build_relation(self, hist: torch.Tensor) -> torch.Tensor:
        """
        hist: (B, m, N, N) — история туров
              0 индекс = текущий/ближний тур
             -1 индекс = самый дальний тур

        return R: (B, n_heads, N, N, d_k)
        """
        B, m, N, _ = hist.shape
        device = hist.device

        # (B, m, N, N) → (B, m, N, N, D)
        R = self.result_embed(hist)

        # тур 0 - текущий (не имеем тк нет результата), 1 = ближний, тур m-1 = дальний
        tour_ids = torch.arange(m, device=device)
        pe = self.tour_pe(tour_ids)
        pe = pe.view(1, m, 1, 1, self.d_model)

        R = R + pe

        R = R.sum(dim=1)
        R = self.history_proj(R)                       # (B, N, N, D)

        R = self.W_r(R).view(B, N, N, self.n_heads, self.d_k)

        R = R.permute(0, 3, 1, 2, 4)

        return R

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        hist: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, N, D)    — batch, num_teams, d_model
            adj:  (N, N)       — маска графа
            hist: (B, m, N, N) — история туров (int: 0,1,2,3)
                                 hist[:,0,:,:] = текущий/ближний тур
                                 hist[:,-1,:,:] = самый дальний тур
        Returns:
            out: (B, N, D)
        """
        B, N, D = x.shape

        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, n_heads, N, N)

        if hist is not None:
            R = self._build_relation(hist)  #  (B, n_heads, N, N, d_k)

            relation_scores = torch.einsum(
                'bhid, bhijd -> bhij', Q, R
            ) / self.scale

            scores = scores + relation_scores

        if adj is not None:
            if adj.dim() == 3:
                adj = adj.unsqueeze(1)        # (B, 1, N, N)
            scores = scores.masked_fill(adj == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)             # (B, n_heads, N, d_k)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.W_o(out)

        return out
