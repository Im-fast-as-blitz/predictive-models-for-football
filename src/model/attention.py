import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = hidden_size
        self.n_heads = n_heads
        self.d_k = hidden_size // n_heads
        
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) - batch, num_variates, d_model
        Returns:
            out: (B, N, D)
            attn_weights: (B, n_heads, N, N)
        """
        B, N, D = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        # Q, K, V: (B, n_heads, N, d_k)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, n_heads, N, N)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted aggregation
        out = torch.matmul(attn_weights, V)  # (B, n_heads, N, d_k)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.W_o(out)
        
        return out
