"""
Наглядный дебаг BFS-графа на реальных данных.
Запуск: python debug_graph.py
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from dataset.dataset import PandasDataset


def build_ds_from_csv(csv_path: str, depth: int) -> PandasDataset:
    df = pd.read_csv(csv_path)
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    ds = object.__new__(PandasDataset)
    ds.depth = depth
    ds.target_col = "match_result"
    ds.feature_cols = []
    ds.teams = df["team"].values
    ds.enemies = df["enemy_team"].values

    # убираем ненужные колонки чтобы df.iloc[i][target_col] работал
    ds.df = df[["match_result"]].copy()

    ds.team_to_indices = {}
    for i, t in enumerate(ds.teams):
        ds.team_to_indices.setdefault(t, []).append(i)

    ds.pair_to_indices = {}
    for i in range(len(df)):
        key = (ds.teams[i], ds.enemies[i])
        ds.pair_to_indices.setdefault(key, []).append(i)

    ds._raw_df = df  # для отображения дат и команд
    return ds


def print_bfs_tree(ds: PandasDataset, idx: int):
    t1 = ds.teams[idx]
    t2 = ds.enemies[idx]

    t1_pair = ds.pair_to_indices[(t1, t2)]
    match_pos = t1_pair.index(idx)
    t2_idx = ds.pair_to_indices[(t2, t1)][match_pos]

    raw = ds._raw_df
    date = raw.iloc[idx]["date"]
    result_names = {0: "проиграл", 1: "ничья", 2: "выиграл"}
    encoded_names = {1: "проиграл(1)", 2: "ничья(2)", 3: "выиграл(3)"}

    print(f"\n{'='*60}")
    print(f"ПРЕДСКАЗЫВАЕМЫЙ МАТЧ: {t1} vs {t2}  [{date}]")
    print(f"{'='*60}")
    print(f"depth={ds.depth}  num_levels={ds.depth-1}  max_n={2**ds.depth}")

    from collections import deque
    num_levels = ds.depth - 1
    team_to_node: dict[str, int] = {t1: 0, t2: 1}
    node_count = 2

    queue = deque([(t1, t2, idx, t2_idx, 0)])

    print(f"\nLevel 0 (корень): {t1}(node=0)  vs  {t2}(node=1)")

    while queue:
        team_a, team_b, idx_a, idx_b, level = queue.popleft()

        if level >= num_levels:
            continue

        print(f"\n  --- Expansion level={level} ---")

        for team, idx_self in [(team_a, idx_a), (team_b, idx_b)]:
            node = team_to_node[team]
            prev = [i for i in ds.team_to_indices.get(team, []) if i < idx_self]

            if not prev:
                print(f"  {team}(node={node}): нет истории до {idx_self}")
                continue

            prev_idx = prev[-1]
            opp = ds.enemies[prev_idx]
            result = int(ds.df.iloc[prev_idx][ds.target_col])
            encoded = result + 1
            prev_date = raw.iloc[prev_idx]["date"]

            if opp not in team_to_node:
                team_to_node[opp] = node_count
                node_count += 1
            node_opp = team_to_node[opp]

            print(f"  {team}(node={node}) — последний матч [{prev_date}]: "
                  f"vs {opp}(node={node_opp}) → {team} {result_names[result]}  "
                  f"[encoded={encoded_names[encoded]}]")
            print(f"    adj[{node},{node_opp}] = adj[{node_opp},{node}] = 1")
            print(f"    hist[{level},{node},{node_opp}] = {encoded}  "
                  f"hist[{level},{node_opp},{node}] = {4 - encoded}")

            pair = ds.pair_to_indices.get((team, opp), [])
            pair_mirror = ds.pair_to_indices.get((opp, team), [])
            if prev_idx in pair and pair_mirror:
                pos = pair.index(prev_idx)
                if pos < len(pair_mirror):
                    queue.append((team, opp, prev_idx, pair_mirror[pos], level + 1))

    print(f"\nИТОГ team_to_node: {team_to_node}")

    # Получаем реальные тензоры
    adj, hist, t2n = ds._get_graph_features(t1, idx, t2, t2_idx)

    print(f"\nadj [{tuple(adj.shape)}]:")
    nodes_list = sorted(t2n.items(), key=lambda x: x[1])
    labels = [name for name, _ in nodes_list]
    max_label = max(len(l) for l in labels)
    header = " " * (max_label + 2) + "  ".join(f"{l:>{max_label}}" for l in labels)
    print("  " + header)
    for i, (name_i, ni) in enumerate(nodes_list):
        row_vals = "  ".join(f"{adj[ni, nj].item():>{max_label}.0f}" for _, nj in nodes_list)
        print(f"  {name_i:>{max_label}} | {row_vals}")

    if hist.shape[0] > 0:
        for lv in range(hist.shape[0]):
            print(f"\nhist[{lv}] (expansion level={lv}):")
            print("  " + header)
            for name_i, ni in nodes_list:
                row_vals = "  ".join(f"{hist[lv, ni, nj].item():>{max_label}.0f}" for _, nj in nodes_list)
                print(f"  {name_i:>{max_label}} | {row_vals}")


if __name__ == "__main__":
    CSV = "notebooks/time siries catboost/selected_leagues_one_line.csv"
    DEPTH = 3

    ds = build_ds_from_csv(CSV, depth=DEPTH)

    # Найдём матч Arsenal vs Chelsea подальше от начала (чтобы была история)
    raw = ds._raw_df
    mask = (raw["team"] == "Arsenal") & (raw["enemy_team"] == "Chelsea")
    candidates = raw[mask]
    print(f"Матчи Arsenal vs Chelsea в датасете:\n{candidates[['date','team','enemy_team','match_result']]}\n")

    if len(candidates) == 0:
        # fallback: берём просто какой-нибудь матч подальше от начала
        idx = len(ds.teams) // 2
    else:
        # берём третий от конца (чтобы точно была история)
        idx = candidates.index[-3] if len(candidates) >= 3 else candidates.index[-1]

    print_bfs_tree(ds, idx)