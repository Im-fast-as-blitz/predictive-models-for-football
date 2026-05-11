import torch.nn as nn

from src.model.embedding import InvertedDataEmbedding
from src.model.layer import Layer


class ITransformer(nn.Module):
    def __init__(self, args, feat_dim: int):
        super(ITransformer, self).__init__()

        self.dropout = args["dropout"]
        self.hidden_size = args["hidden_size"]
        self.out_features = args["out_features"]

        self.feat_dim = feat_dim
        self.embeder = InvertedDataEmbedding(
            args=args["embedder"],
            feat_dim=feat_dim,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        )

        layers = []
        for _ in range(args["L"]):
            layers.append(Layer(args["layer"], hidden_size=self.hidden_size, dropout=self.dropout))
        self.base = nn.Sequential(*layers)
        
        self.projector = nn.Linear(self.hidden_size, self.out_features)

    def forward(self, x):
        # [B, N, T, F] или [B, N, T] при feat_dim=1 -> [B, N, D]
        x = self.embeder(x) # B, N, D

        x = self.base(x)

        x = self.projector(x)
        
        return x[:, 0, :]
