"""
Проверяем что DataLoader корректно возвращает батчи с правильными формами.
Запуск: python check_dataloader.py
"""
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from dataset.dataset import PandasDataset

IGNORE_COLS = [
    "ftg", "enemy_ftg", "s", "st", "f", "c", "y", "r",
    "enemy_s", "enemy_st", "enemy_f", "enemy_c", "enemy_y", "enemy_r",
    "ht_w", "ht_d", "ht_l", "date", "season", "referee",
    "team", "enemy_team", "league_code",
]

CSV = "notebooks/time siries catboost/selected_leagues_one_line.csv"
DEPTH = 3
BATCH_SIZE = 4


def build_ds(csv_path: str, depth: int) -> PandasDataset:
    df = pd.read_csv(csv_path)
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    ds = object.__new__(PandasDataset)
    ds.depth = depth
    ds.target_col = "match_result"
    ds.teams = df["team"].values
    ds.enemies = df["enemy_team"].values

    drop_cols = [c for c in IGNORE_COLS if c in df.columns]
    ds.df = df.drop(columns=drop_cols).fillna(0)
    ds.feature_cols = [c for c in ds.df.columns if c != ds.target_col]

    ds.team_to_indices = {}
    for i, t in enumerate(ds.teams):
        ds.team_to_indices.setdefault(t, []).append(i)

    ds.pair_to_indices = {}
    for i in range(len(df)):
        key = (ds.teams[i], ds.enemies[i])
        ds.pair_to_indices.setdefault(key, []).append(i)

    ds.cat_columns = []
    ds.cat_encoder = {}
    ds.cat_cardinalities = []
    ds._cat_codes = np.zeros((len(df), 0), dtype=np.int64)

    return ds


if __name__ == "__main__":
    print(f"Загружаем датасет (depth={DEPTH})...")
    ds = build_ds(CSV, depth=DEPTH)

    max_n   = 2 ** DEPTH
    n_feats = len(ds.feature_cols)
    print(f"Строк в датасете : {len(ds)}")
    print(f"Фичей на команду : {n_feats}")
    print(f"max_n            : {max_n}  (2^{DEPTH})")
    print(f"Ожидаемые формы батча [{BATCH_SIZE}]:")
    print(f"  node_features : [{BATCH_SIZE}, {max_n}, {n_feats}]")
    print(f"  adj           : [{BATCH_SIZE}, {max_n}, {max_n}]")
    print(f"  hist          : [{BATCH_SIZE}, {DEPTH-1}, {max_n}, {max_n}]")
    print(f"  x_cat         : [{BATCH_SIZE}, {max_n}, 1, 0]  (без категориальных колонок)")
    print(f"  target        : [{BATCH_SIZE}]")

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    print("\nПроходим первые 3 батча...")
    for batch_idx, (node_features, adj, hist, x_cat, target) in enumerate(loader):
        print(f"\n  Батч {batch_idx}:")
        print(f"    node_features : {tuple(node_features.shape)}")
        print(f"    adj           : {tuple(adj.shape)}")
        print(f"    hist          : {tuple(hist.shape)}")
        print(f"    x_cat         : {tuple(x_cat.shape)}")
        print(f"    target        : {tuple(target.shape)}  значения={target.tolist()}")

        # adj симметрична
        assert torch.allclose(adj, adj.transpose(1, 2)), "adj не симметричная!"

        # где adj=0 там hist=0
        adj_exp = adj.unsqueeze(1).expand_as(hist)
        assert (hist[adj_exp == 0] == 0).all(), "hist ненулевой там где нет ребра!"

        # hist только из {0,1,2,3}
        unique_vals = hist.unique().tolist()
        assert all(v in [0.0, 1.0, 2.0, 3.0] for v in unique_vals), \
            f"Некорректные значения в hist: {unique_vals}"

        print(f"    [OK] adj симметрична, hist значения корректны")

        if batch_idx >= 2:
            break

    print("\n=== ВСЕ ПРОВЕРКИ ПРОШЛИ ===")

    # Показываем фичи для конкретного матча с историей: Arsenal vs Chelsea
    raw = pd.read_csv(CSV)
    raw = raw.sort_values(["team", "date"]).reset_index(drop=True)
    mask = (raw["team"] == "Arsenal") & (raw["enemy_team"] == "Chelsea")
    candidates = raw[mask]
    # берём матч из середины — точно есть история с обеих сторон
    sample_idx = candidates.index[len(candidates) // 2]

    node_features_s, adj_s, hist_s, x_cat_s, target_s = ds[sample_idx]
    date = raw.iloc[sample_idx]["date"]
    print(f"\nМатч Arsenal vs Chelsea [{date}], target={target_s.item()}")
    print(f"(0=проиграл, 1=ничья, 2=выиграл — с точки зрения Arsenal)")

    # восстанавливаем team_to_node через _get_graph_features
    t1_pair = ds.pair_to_indices[("Arsenal", "Chelsea")]
    match_pos = t1_pair.index(sample_idx)
    t2_idx = ds.pair_to_indices[("Chelsea", "Arsenal")][match_pos]
    _, _, team_to_node, _ = ds._get_graph_features("Arsenal", sample_idx, "Chelsea", t2_idx)
    node_to_team = {v: k for k, v in team_to_node.items()}

    # Показываем ключевые фичи для каждого узла
    key_feats = [
        "match_result_lag_1", "match_result_lag_2", "match_result_lag_3",
        "ftg_lag_1", "ftg_lag_2",
        "enemy_ftg_lag_1", "enemy_ftg_lag_2",
        "match_result_roll_mean_5", "ftg_roll_mean_5",
    ]
    key_indices = [ds.feature_cols.index(f) for f in key_feats if f in ds.feature_cols]
    key_names   = [f for f in key_feats if f in ds.feature_cols]

    col_w = 22
    header = f"  {'команда':<20} | " + " | ".join(f"{n[:col_w]:>{col_w}}" for n in key_names)
    print(f"\nКлючевые фичи по узлам:")
    print(header)
    print("  " + "-" * len(header))
    for node_idx in range(max_n):
        feat = node_features_s[node_idx]
        team_name = node_to_team.get(node_idx, "—")
        vals = " | ".join(f"{feat[i].item():>{col_w}.2f}" for i in key_indices)
        print(f"  {team_name:<20} | {vals}")