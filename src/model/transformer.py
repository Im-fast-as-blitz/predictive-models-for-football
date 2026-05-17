import torch.nn as nn

from src.model.embedding import InvertedDataEmbedding
from src.model.layer import Layer


class ITransformer(nn.Module):
    def __init__(self, args, feat_dim: int, cat_cardinalities=()):
        super(ITransformer, self).__init__()

        self.dropout = args["dropout"]
        self.hidden_size = args["hidden_size"]
        self.out_features = args["out_features"]
        self.pooler = args["pooler"]

        self.feat_dim = feat_dim
        emb_cfg = args["embedder"]
        self.embeder = InvertedDataEmbedding(
            args=emb_cfg,
            feat_dim=feat_dim,
            hidden_size=self.hidden_size,
            cat_cardinalities=list(cat_cardinalities),
            dropout=self.dropout,
        )

        layers = []
        for _ in range(args["L"]):
            layers.append(Layer(args["layer"], hidden_size=self.hidden_size, dropout=self.dropout))
        self.base = nn.Sequential(*layers)
        
        self.projector = nn.Linear(self.hidden_size, self.out_features)

    def forward(self, x, adj, hist, full_history, x_cat):
        # x: [B, N, T, F] или [B, N, T] или [B, N, F]
        #                                             -> [B, N, D]
        # x_cat: [B, N, T, C]

        x = self.embeder(x, x_cat, full_history) # B, N, D

        for layer in self.base:
            x = layer(x, adj, hist)

        if self.pooler == "avg":
            pooled = x.mean(dim=1)     # [B, N, D] → [B, D]
        elif self.pooler == "one_token":
            pooled = x[:, 0, :]
        else:
            raise Exception("Unknown pooler type")
        out = self.projector(pooled)
        
        return out
