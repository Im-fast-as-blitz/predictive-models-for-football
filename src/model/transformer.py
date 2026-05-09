import torch.nn as nn

from src.model import InvertedDataEmbedding
from src.model.ffn import FeedForward


class ITransformer(nn.Module):
    """
    Пока только inverted embedding и FFN.
    Остальное — в TODO ниже.
    """

    def __init__(self, args, seq_len: int, dropout: float = 0.1):
        super(ITransformer, self).__init__()

        self.hidden_size = args["hidden_size"]
        self.out_features = args["out_features"]
        self.seq_len = seq_len
        self.d_ff = args.get("dim_feedforward", 4 * self.hidden_size)

        self.embeder = InvertedDataEmbedding(
            t=seq_len,
            hidden_size=self.hidden_size,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(self.hidden_size)
        self.ffn = FeedForward(
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            dropout=dropout,
        )

    def forward(self, x):
        # [B, T, N] -> [B, N, D]
        x = self.embeder(x)
        x = x + self.ffn(self.norm(x))
        return x
