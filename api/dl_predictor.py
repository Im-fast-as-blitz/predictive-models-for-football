import os
from typing import Optional

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.dataset import PandasDataset
from src.model import ITransformer


API_LABELS = {0: "home win", 1: "away win", 2: "draw"}


class DLPredictor:
    def __init__(
        self,
        config_name: Optional[str] = None,
        config_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        config_name = config_name or os.getenv("DL_CONFIG", "full_ts_approach")
        config_path = config_path or os.getenv(
            "DL_CONFIG_PATH", f"src/config/{config_name}.yaml"
        )
        weights_path = weights_path or os.getenv(
            "DL_WEIGHTS_PATH", "saved/best_model.pth"
        )

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Конфиг не найден: {config_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Веса не найдены: {weights_path}. "
                f"Скачайте их: python scripts/download_artifacts.py"
            )

        self.cfg = OmegaConf.load(config_path)
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        kfold_steps = self.cfg["train"]["kfold_steps"]
        self.dataset = PandasDataset(
            args=self.cfg["dataset"],
            kfold_step=1,
            kfold_steps=kfold_steps + 2,
            train="train",
            cat_encoder=None,
        )

        self.model = ITransformer(
            self.cfg["model"],
            len(self.dataset.feature_cols),
            cat_cardinalities=self.dataset.cat_cardinalities,
        ).to(self.device)

        checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def teams(self) -> list[str]:
        return sorted({str(t) for t in self.dataset.teams})

    def seasons(self) -> list[str]:
        return sorted({str(s) for s in self.dataset.seasons})

    @torch.no_grad()
    def predict(self, home_team: str, away_team: str, season: str) -> Optional[dict]:
        real_idx = self.dataset.find_match_index(home_team, away_team, season)
        if real_idx is None:
            return None

        node_features, adj, hist, _, full_history, x_cat, _ = self.dataset.build_sample(
            real_idx
        )

        out = self.model(
            node_features.unsqueeze(0).to(self.device),
            adj.unsqueeze(0).to(self.device),
            hist.unsqueeze(0).to(self.device),
            full_history.unsqueeze(0).to(self.device),
            x_cat.unsqueeze(0).to(self.device),
        )
        probs = F.softmax(out, dim=1)[0].cpu()

        home_lost, draw, home_won = (
            float(probs[0]),
            float(probs[1]),
            float(probs[2]),
        )
        api_probs = {
            "home_win": home_won,
            "away_win": home_lost,
            "draw": draw,
        }

        ordered = [api_probs["home_win"], api_probs["away_win"], api_probs["draw"]]
        pred_value = int(max(range(3), key=lambda i: ordered[i]))

        return {
            "prediction": pred_value,
            "prediction_label": API_LABELS[pred_value],
            "probabilities": api_probs,
        }
