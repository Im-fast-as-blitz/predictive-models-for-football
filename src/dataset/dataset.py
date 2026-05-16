import numpy as np
import pandas as pd
import torch
from collections import deque
from torch.utils.data import Dataset


class PandasDataset(Dataset):
    def __init__(self, args, kfold_step, kfold_steps, train):
        dataset_name = args["name"]
        none_fill = args["none_fill"]

        season_col = args["data"]["season_column"]
        team_col = args["data"].get("team_column", "team")
        enemy_col = args["data"].get("enemy_column", "enemy_team")

        df_full = pd.read_csv(f"data/{dataset_name}.csv")
        df_full = df_full[df_full[season_col] <= args["data"]["val_season"]].reset_index(drop=True)

        val_df = df_full[df_full[season_col] == args["data"]["val_season"]]
        fold_size = val_df.shape[0] // kfold_steps
        train_date_thr = ""
        for date in sorted(val_df["date"].unique()):
            if train_date_thr != "" and val_df[val_df["date"] <= date].shape[0] > fold_size * kfold_step:
                break
            train_date_thr = date

        if train == "train":
            split_mask = ~(
                (df_full[season_col] == args["data"]["val_season"])
                & (df_full["date"] >= train_date_thr)
            )
        elif train == "val":
            split_mask = (
                (df_full[season_col] == args["data"]["val_season"])
                & (df_full["date"] == train_date_thr)
            )
        elif train == "test":
            split_mask = (
                (df_full[season_col] == args["data"]["val_season"])
                & (df_full["date"] > train_date_thr)
            )
        else:
            raise Exception("Unknown Split data strategy")

        # Индексы строк сплита внутри полного df_full
        self.split_indices = list(df_full[split_mask].index)

        self.depth = args["data"]["depth"]
        self.t = args["data"]["t"]

        # Полные массивы — BFS обращается к self.enemies[prev_idx] по индексу в df_full
        self.teams = df_full[team_col].values.copy()
        self.enemies = df_full[enemy_col].values.copy()

        self.df = df_full.drop(columns=args["data"]["igonre_columns"])

        # TODO нормально сделать кат фичи
        self.cat_columns = args["data"]["cat_columns"]
        self.df = self.df.drop(columns=self.cat_columns)  # пока просто удаляем но это плохо!

        self.target_col = args["data"]["target_column"]
        self.feature_cols = [col for col in self.df.columns if col != self.target_col]

        if none_fill == "zero":
            self.df[self.feature_cols] = self.df[self.feature_cols].fillna(0)
        else:
            raise Exception("Unknown strategy for fill none")

        self.dates = pd.to_datetime(df_full["date"]).map(lambda d: d.toordinal()).values.astype(np.float32)

        # Индексы строятся по всему df — история полностью доступна для val/test
        self.team_to_indices = {}
        for i in range(len(self.df)):
            self.team_to_indices.setdefault(self.teams[i], []).append(i)

        self.pair_to_indices = {}
        for i in range(len(self.df)):
            key = (self.teams[i], self.enemies[i])
            self.pair_to_indices.setdefault(key, []).append(i)

    def __len__(self) -> int:
        return len(self.split_indices)

    def _get_team_features(self, team: str, up_to_idx: int) -> torch.Tensor:
        indices = self.team_to_indices.get(team, [])
        selected = [i for i in indices if i <= up_to_idx][-self.t:]
        rows = self.df.iloc[selected][self.feature_cols].values.astype(np.float32)
        if len(rows) < self.t:
            pad = np.zeros((self.t - len(rows), len(self.feature_cols)), dtype=np.float32)
            rows = np.vstack([pad, rows])
        return torch.tensor(rows, dtype=torch.float32)

    def _get_graph_features(
        self, t1: str, idx: int, t2: str, t2_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
        """
        BFS tree expansion of depth-1 levels starting from the match t1 vs t2.

        Returns:
            adj:            [max_n, max_n]          — 1 where teams played each other
            hist:           [depth-1, max_n, max_n] — encoded result per BFS level
                            (1=row-team lost, 2=draw, 3=row-team won, 0=no match)
            team_to_node:   dict team_name -> node index
            team_to_feat_idx: dict team_name -> df index for _get_team_features

        max_n = 2**depth (upper bound on unique teams in the tree).
        """
        num_levels = self.depth - 1
        max_n = 2 ** self.depth

        team_to_node: dict[str, int] = {t1: 0, t2: 1}
        team_to_feat_idx: dict[str, int] = {t1: idx, t2: t2_idx}
        node_count = 2

        adj = np.zeros((max_n, max_n), dtype=np.float32)
        hist = np.zeros((num_levels, max_n, max_n), dtype=np.float32)

        if num_levels == 0:
            return (
                torch.tensor(adj, dtype=torch.float32),
                torch.tensor(hist, dtype=torch.float32),
                team_to_node,
                team_to_feat_idx,
            )

        # Queue items: (team_a, team_b, df_idx_a, df_idx_b, bfs_level)
        queue: deque = deque([(t1, t2, idx, t2_idx, 0)])

        while queue:
            team_a, team_b, idx_a, idx_b, level = queue.popleft()

            if level >= num_levels:
                continue

            for team, idx_self in [(team_a, idx_a), (team_b, idx_b)]:
                node = team_to_node[team]

                prev = [i for i in self.team_to_indices.get(team, []) if i < idx_self]
                if not prev:
                    continue

                prev_idx = prev[-1]
                opp = self.enemies[prev_idx]
                encoded = int(self.df.iloc[prev_idx][self.target_col]) + 1  # 1=lost 2=draw 3=won

                if opp not in team_to_node:
                    team_to_node[opp] = node_count
                    node_count += 1

                node_opp = team_to_node[opp]

                adj[node, node_opp] = 1.0
                adj[node_opp, node] = 1.0
                hist[level, node, node_opp] = float(encoded)
                hist[level, node_opp, node] = float(4 - encoded)

                pair = self.pair_to_indices.get((team, opp), [])
                pair_mirror = self.pair_to_indices.get((opp, team), [])
                if prev_idx in pair and pair_mirror:
                    pos = pair.index(prev_idx)
                    if pos < len(pair_mirror):
                        opp_mirror_idx = pair_mirror[pos]
                        if opp not in team_to_feat_idx:
                            team_to_feat_idx[opp] = opp_mirror_idx
                        queue.append((team, opp, prev_idx, opp_mirror_idx, level + 1))

        return (
            torch.tensor(adj, dtype=torch.float32),
            torch.tensor(hist, dtype=torch.float32),
            team_to_node,
            team_to_feat_idx,
        )

    def _get_all_team_features(
        self, team_to_node: dict, team_to_feat_idx: dict
    ) -> torch.Tensor:
        """Returns [max_n, F] — feature vector for every team in the BFS tree.
        Teams with no entry in team_to_feat_idx get a zero row."""
        max_n = 2 ** self.depth
        n_features = len(self.feature_cols)
        out = np.zeros((max_n, n_features), dtype=np.float32)
        for team, node in team_to_node.items():
            if team in team_to_feat_idx:
                feat_idx = team_to_feat_idx[team]
                rows = self._get_team_features(team, feat_idx).numpy()  # (depth, F)
                out[node] = rows[-1]
        return torch.tensor(out, dtype=torch.float32)

    def _get_all_team_timestamps(
        self, team_to_node: dict, team_to_feat_idx: dict
    ) -> torch.Tensor:
        """Returns [max_n, depth] — ordinal dates of last `depth` matches per team.
        If team has fewer than depth matches, pads left by repeating the earliest date.
        Teams with no matches at all get a zero row."""
        max_n = 2 ** self.depth
        out = np.zeros((max_n, self.t), dtype=np.float32)
        for team, node in team_to_node.items():
            if team not in team_to_feat_idx:
                continue
            feat_idx = team_to_feat_idx[team]
            indices = self.team_to_indices.get(team, [])
            selected = [i for i in indices if i <= feat_idx][-self.t:]
            if not selected:
                continue
            dates = self.dates[selected]  # (n_actual,)
            if len(dates) < self.t:
                pad = np.full(self.t - len(dates), dates[0], dtype=np.float32)
                dates = np.concatenate([pad, dates])
            out[node] = dates
        return torch.tensor(out, dtype=torch.float32)

    def _get_all_team_full_history(
        self, team_to_node: dict, team_to_feat_idx: dict
    ) -> torch.Tensor:
        """Returns [max_n, depth, F] — full match history per team.
        Teams with fewer than depth matches are zero-padded on the left."""
        max_n = 2 ** self.depth
        n_features = len(self.feature_cols)
        out = np.zeros((max_n, self.t, n_features), dtype=np.float32)
        for team, node in team_to_node.items():
            if team not in team_to_feat_idx:
                continue
            feat_idx = team_to_feat_idx[team]
            rows = self._get_team_features(team, feat_idx).numpy()  # [T, F]
            out[node] = rows
        return torch.tensor(out, dtype=torch.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        real_idx = self.split_indices[idx]
        t1 = self.teams[real_idx]
        t2 = self.enemies[real_idx]
        target = torch.tensor(self.df.iloc[real_idx][self.target_col], dtype=torch.long)

        t1_pair = self.pair_to_indices[(t1, t2)]
        match_pos = t1_pair.index(real_idx)
        t2_idx = self.pair_to_indices[(t2, t1)][match_pos]

        adj, hist, team_to_node, team_to_feat_idx = self._get_graph_features(t1, real_idx, t2, t2_idx)
        node_features = self._get_all_team_features(team_to_node, team_to_feat_idx)      # [max_n, F]
        timestamps = self._get_all_team_timestamps(team_to_node, team_to_feat_idx)       # [max_n, T]
        full_history = self._get_all_team_full_history(team_to_node, team_to_feat_idx)   # [max_n, T, F]

        max_n = 2 ** self.depth
        codes = torch.from_numpy(self._cat_codes[real_idx]).to(dtype=torch.long)
        x_cat = codes.view(1, 1, -1).expand(max_n, 1, -1).contiguous()                  # [max_n, 1, n_cat]

        return node_features, adj, hist, timestamps, full_history, x_cat, target
