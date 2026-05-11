"""
Боевой тест на реальном датасете, depth=8.
Запуск: python test_real.py
"""
import sys
import time
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, "src")
from dataset.dataset import PandasDataset

CSV   = "notebooks/time siries catboost/selected_leagues_one_line.csv"
DEPTH = 8

with open("src/config/base.yaml") as f:
    CONFIG = yaml.safe_load(f)
IGNORE_COLS = CONFIG["dataset"]["data"]["igonre_columns"]


def build_ds(csv_path: str, depth: int) -> PandasDataset:
    df = pd.read_csv(csv_path)
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    ds = object.__new__(PandasDataset)
    ds.depth = depth
    ds.target_col = "match_result"
    ds.teams = df["team"].values
    ds.enemies = df["enemy_team"].values

    drop_cols = [c for c in IGNORE_COLS + ["league_code"] if c in df.columns]
    ds.df = df.drop(columns=drop_cols).fillna(0)
    ds.feature_cols = [c for c in ds.df.columns if c != ds.target_col]
    ds._raw_df = df

    ds.team_to_indices = {}
    for i, t in enumerate(ds.teams):
        ds.team_to_indices.setdefault(t, []).append(i)

    ds.pair_to_indices = {}
    for i in range(len(df)):
        key = (ds.teams[i], ds.enemies[i])
        ds.pair_to_indices.setdefault(key, []).append(i)

    return ds


def run(ds: PandasDataset, idx: int):
    raw  = ds._raw_df
    t1   = ds.teams[idx]
    t2   = ds.enemies[idx]
    date = raw.iloc[idx]["date"]

    max_n      = 2 ** ds.depth
    num_levels = ds.depth - 1
    n_feats    = len(ds.feature_cols)

    print(f"\n{'='*65}")
    print(f"  Матч : {t1} vs {t2}  [{date}]")
    print(f"  depth={ds.depth}  max_n={max_n}  num_levels={num_levels}")
    print(f"  Фичей на команду: {n_feats}")
    print(f"{'='*65}")

    t0 = time.time()
    node_features, adj, hist, target = ds[idx]
    elapsed = time.time() - t0

    result_label = {0: "проиграл", 1: "ничья", 2: "победил"}
    print(f"\n  Время __getitem__ : {elapsed*1000:.1f} мс")
    print(f"  target            : {target.item()}  ({t1} {result_label[target.item()]})")

    print(f"\n  Формы тензоров:")
    print(f"    node_features : {tuple(node_features.shape)}")
    print(f"    adj           : {tuple(adj.shape)}")
    print(f"    hist          : {tuple(hist.shape)}")

    # восстанавливаем team_to_node
    t1_pair   = ds.pair_to_indices[(t1, t2)]
    match_pos = t1_pair.index(idx)
    t2_idx    = ds.pair_to_indices[(t2, t1)][match_pos]
    _, _, team_to_node, _ = ds._get_graph_features(t1, idx, t2, t2_idx)
    node_to_team = {v: k for k, v in team_to_node.items()}

    n_unique = len(team_to_node)
    print(f"\n  Уникальных команд в дереве : {n_unique} / {max_n}")
    print(f"  Паддинг-узлов (нет истории): {max_n - n_unique}")
    print(f"  Рёбер в adj               : {int(adj.sum().item()) // 2}")

    print(f"\n  Матчей по уровням hist:")
    for lv in range(num_levels):
        n_matches = int((hist[lv] > 0).sum().item()) // 2
        print(f"    level {lv}: {n_matches} матчей")

    # фичи по узлам
    key_feats = ["match_result_lag_1", "match_result_lag_2",
                 "ftg_lag_1", "match_result_roll_mean_5", "ftg_roll_mean_5"]
    key_idx   = [ds.feature_cols.index(f) for f in key_feats if f in ds.feature_cols]
    key_names = [f for f in key_feats if f in ds.feature_cols]

    print(f"\n  Фичи узлов:")
    header = f"    {'n':>3}  {'команда':<22}  {'ненул':>5}  " + \
             "  ".join(f"{n[:18]:>18}" for n in key_names)
    print(header)
    print("    " + "-" * (len(header) - 4))
    for node in range(max_n):
        feat    = node_features[node]
        nonzero = (feat != 0).sum().item()
        name    = node_to_team.get(node, "—")
        vals    = "  ".join(f"{feat[i].item():>18.3f}" for i in key_idx)
        marker  = "  ← цель" if node < 2 else ""
        print(f"    {node:>3}  {name:<22}  {nonzero:>5}  {vals}{marker}")

    # проверки
    print(f"\n  Проверки:")
    checks = [
        ("adj симметрична",          torch.allclose(adj, adj.T)),
        ("где adj=0 там hist=0",     (hist[adj.unsqueeze(0).expand_as(hist) == 0] == 0).all().item()),
        ("hist только из {0,1,2,3}", all(v in [0.,1.,2.,3.] for v in hist.unique().tolist())),
        (f"node 0 ({t1}) ненулевой", (node_features[0] != 0).any().item()),
        (f"node 1 ({t2}) ненулевой", (node_features[1] != 0).any().item()),
    ]
    all_ok = True
    for name, ok in checks:
        print(f"    {'PASS' if ok else 'FAIL'}: {name}")
        all_ok &= ok

    print(f"\n  {'ВСЕ ПРОВЕРКИ ПРОШЛИ ✓' if all_ok else 'ЕСТЬ ОШИБКИ ✗'}")


if __name__ == "__main__":
    print("Загружаем датасет...")
    ds = build_ds(CSV, depth=DEPTH)
    print(f"Строк: {len(ds)}, фичей: {len(ds.feature_cols)}")

    raw  = ds._raw_df
    mask = (raw["team"] == "Arsenal") & (raw["enemy_team"] == "Chelsea")
    idx  = raw[mask].index[-3]

    run(ds, idx)
