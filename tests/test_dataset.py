"""
Тесты PandasDataset.__getitem__ → (node_features, adj, hist, x_cat, target)

Синтетический датасет:
  idx  team  enemy  result  feat_a  feat_b
  0    A     C      2       1.0     2.0     A победил C
  1    C     A      0       3.0     4.0     зеркало
  2    B     D      0       5.0     6.0     B проиграл D
  3    D     B      2       7.0     8.0     зеркало
  4    A     B      1       9.0     10.0    ничья  ← предсказываем
  5    B     A      1       11.0    12.0    зеркало

BFS дерево для idx=4 (A vs B), depth=3:
  Level 0 (корень):      A(node=0)  vs  B(node=1)
  hist[0], expansion 0:  A→C(node=2), B→D(node=3)
  hist[1], expansion 1:  нет истории → nodes 4-7 пустые

Ожидаемые node_features (rows[-1] из _get_team_features):
  node 0 (A): feat_idx=4  → [9.0,  10.0]
  node 1 (B): feat_idx=5  → [11.0, 12.0]
  node 2 (C): feat_idx=1  → [3.0,   4.0]
  node 3 (D): feat_idx=3  → [7.0,   8.0]
  nodes 4-7:  нет истории → [0.0,   0.0]
"""

import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
from dataset.dataset import PandasDataset


def _build_cat_state(ds: PandasDataset, df: pd.DataFrame) -> None:
    ds.cat_columns = ["league_code"]
    ds.cat_encoder = {
        "league_code": {v: i + 1 for i, v in enumerate(pd.unique(df["league_code"].dropna()))},
    }
    ds.cat_cardinalities = [
        max(ds.cat_encoder[col].values(), default=0) + 1 for col in ds.cat_columns
    ]
    code_mat = np.zeros((len(df), len(ds.cat_columns)), dtype=np.int64)
    for j, col in enumerate(ds.cat_columns):
        mapped = df[col].map(ds.cat_encoder[col])
        code_mat[:, j] = mapped.fillna(0).astype(np.int64).to_numpy()
    ds._cat_codes = code_mat


def make_mock_dataset(depth: int, with_categories: bool = False) -> PandasDataset:
    rows = [
        {"team": "A", "enemy_team": "C", "match_result": 2, "feat_a": 1.0,  "feat_b": 2.0, "league_code": "E1"},
        {"team": "C", "enemy_team": "A", "match_result": 0, "feat_a": 3.0,  "feat_b": 4.0, "league_code": "E1"},
        {"team": "B", "enemy_team": "D", "match_result": 0, "feat_a": 5.0,  "feat_b": 6.0, "league_code": "E2"},
        {"team": "D", "enemy_team": "B", "match_result": 2, "feat_a": 7.0,  "feat_b": 8.0, "league_code": "E2"},
        {"team": "A", "enemy_team": "B", "match_result": 1, "feat_a": 9.0,  "feat_b": 10.0, "league_code": "E1"},
        {"team": "B", "enemy_team": "A", "match_result": 1, "feat_a": 11.0, "feat_b": 12.0, "league_code": "E1"},
    ]
    df = pd.DataFrame(rows)

    ds = object.__new__(PandasDataset)
    ds.depth = depth
    ds.target_col = "match_result"
    ds.teams = df["team"].values
    ds.enemies = df["enemy_team"].values

    if with_categories:
        _build_cat_state(ds, df)
        ds.df = df.drop(columns=["team", "enemy_team", "league_code"])
    else:
        ds.cat_columns = []
        ds.cat_encoder = {}
        ds.cat_cardinalities = []
        ds._cat_codes = np.zeros((len(df), 0), dtype=np.int64)
        ds.df = df.drop(columns=["team", "enemy_team"])

    ds.feature_cols = ["feat_a", "feat_b"]

    ds.team_to_indices = {}
    for i, t in enumerate(ds.teams):
        ds.team_to_indices.setdefault(t, []).append(i)

    ds.pair_to_indices = {}
    for i in range(len(df)):
        key = (ds.teams[i], ds.enemies[i])
        ds.pair_to_indices.setdefault(key, []).append(i)

    return ds


def check(name: bool, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    msg = f"{status}: {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    assert condition, msg


# ─────────────────────────────────────────────
# depth=3
# ─────────────────────────────────────────────
def test_depth3():
    print("\n=== depth=3, idx=4 (A vs B) ===")
    ds = make_mock_dataset(depth=3)
    node_features, adj, hist, x_cat, target = ds[4]

    max_n = 2 ** 3  # 8
    F = 2

    # формы
    check("node_features shape", node_features.shape == (max_n, F),
          str(tuple(node_features.shape)))
    check("adj shape",           adj.shape == (max_n, max_n))
    check("hist shape",          hist.shape == (2, max_n, max_n))
    check("x_cat shape", x_cat.shape == (max_n, 1, 0), str(tuple(x_cat.shape)))
    check("target value",        target.item() == 1)
    check("adj symmetric", torch.allclose(adj, adj.T))

    # ожидаемые рёбра
    t2n = {"A": 0, "B": 1, "C": 2, "D": 3}
    check("adj A-C = 1", adj[t2n["A"], t2n["C"]].item() == 1.0)
    check("adj B-D = 1", adj[t2n["B"], t2n["D"]].item() == 1.0)
    check("adj A-B = 0", adj[t2n["A"], t2n["B"]].item() == 0.0)

    # hist[0]: A победил C → encoded=3, C проиграл A → encoded=1
    check("hist[0] A→C = 3", hist[0, t2n["A"], t2n["C"]].item() == 3.0)
    check("hist[0] C→A = 1", hist[0, t2n["C"], t2n["A"]].item() == 1.0)
    # hist[0]: B проиграл D → encoded=1, D выиграл → encoded=3
    check("hist[0] B→D = 1", hist[0, t2n["B"], t2n["D"]].item() == 1.0)
    check("hist[0] D→B = 3", hist[0, t2n["D"], t2n["B"]].item() == 3.0)
    # hist[1] всё нули (нет истории глубже)
    check("hist[1] all zeros", hist[1].sum().item() == 0.0)

    # node_features
    check("node 0 (A) feat_a=9",  node_features[0, 0].item() == 9.0)
    check("node 0 (A) feat_b=10", node_features[0, 1].item() == 10.0)
    check("node 1 (B) feat_a=11", node_features[1, 0].item() == 11.0)
    check("node 1 (B) feat_b=12", node_features[1, 1].item() == 12.0)
    check("node 2 (C) feat_a=3",  node_features[2, 0].item() == 3.0)
    check("node 2 (C) feat_b=4",  node_features[2, 1].item() == 4.0)
    check("node 3 (D) feat_a=7",  node_features[3, 0].item() == 7.0)
    check("node 3 (D) feat_b=8",  node_features[3, 1].item() == 8.0)
    # nodes 4-7 — нет истории → нули
    for n in range(4, 8):
        check(f"node {n} all zeros", node_features[n].sum().item() == 0.0)


# ─────────────────────────────────────────────
# depth=2
# ─────────────────────────────────────────────
def test_depth2():
    print("\n=== depth=2, idx=4 (A vs B) ===")
    ds = make_mock_dataset(depth=2)
    node_features, adj, hist, x_cat, target = ds[4]

    max_n = 2 ** 2  # 4
    check("node_features shape", node_features.shape == (max_n, 2))
    check("adj shape",           adj.shape == (max_n, max_n))
    check("hist shape",          hist.shape == (1, max_n, max_n))

    t2n = {"A": 0, "B": 1, "C": 2, "D": 3}
    check("hist[0] A→C = 3", hist[0, t2n["A"], t2n["C"]].item() == 3.0)
    check("hist[0] B→D = 1", hist[0, t2n["B"], t2n["D"]].item() == 1.0)
    check("node 2 (C) feat_a=3", node_features[2, 0].item() == 3.0)
    check("node 3 (D) feat_a=7", node_features[3, 0].item() == 7.0)


# ─────────────────────────────────────────────
# depth=1 — только T1 и T2, граф пустой
# ─────────────────────────────────────────────
def test_depth1():
    print("\n=== depth=1, idx=4 (A vs B) ===")
    ds = make_mock_dataset(depth=1)
    node_features, adj, hist, x_cat, target = ds[4]

    max_n = 2 ** 1  # 2
    check("node_features shape", node_features.shape == (max_n, 2))
    check("adj shape",           adj.shape == (max_n, max_n))
    check("hist shape (0 levels)", hist.shape == (0, max_n, max_n))
    check("adj all zeros (no expansion)", adj.sum().item() == 0.0)
    check("node 0 (A) feat_a=9",  node_features[0, 0].item() == 9.0)
    check("node 1 (B) feat_a=11", node_features[1, 0].item() == 11.0)


# ─────────────────────────────────────────────
# категориальные признаки (league_code)
# ─────────────────────────────────────────────
def test_categorical():
    print("\n=== categorical, depth=1, idx=4 (A vs B, league E1) ===")
    ds = make_mock_dataset(depth=1, with_categories=True)

    check("cat_encoder E1", ds.cat_encoder["league_code"]["E1"] == 1)
    check("cat_encoder E2", ds.cat_encoder["league_code"]["E2"] == 2)
    check("cat_cardinalities", ds.cat_cardinalities == [3])  # 0 + E1 + E2
    check("_cat_codes row 4", ds._cat_codes[4, 0] == 1)

    node_features, adj, hist, x_cat, target = ds[4]
    max_n = 2

    check("x_cat shape", x_cat.shape == (max_n, 1, 1), str(tuple(x_cat.shape)))
    check("x_cat dtype", x_cat.dtype == torch.long)
    check("x_cat all nodes same league", (x_cat == 1).all().item())
    check("league_code not in df", "league_code" not in ds.df.columns)
    check("node_features unchanged", node_features[0, 0].item() == 9.0)


def test_categorical_reuse_encoder():
    print("\n=== categorical reuse train encoder ===")
    train_ds = make_mock_dataset(depth=1, with_categories=True)
    encoder = train_ds.cat_encoder

    mapped = pd.Series(["E9"]).map(encoder["league_code"]).fillna(0).astype(np.int64).to_numpy()
    check("unknown league -> 0", mapped[0] == 0)


if __name__ == "__main__":
    test_depth3()
    test_depth2()
    test_depth1()
    test_categorical()
    test_categorical_reuse_encoder()
    print("\n=== ALL TESTS PASSED ===")
