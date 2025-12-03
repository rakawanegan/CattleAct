import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

# ----- トラック対応付け関連の関数 ----- #


def align_and_fill(df, new_index):
    """
    Align the track DataFrame to a new time index and fill missing values via interpolation.

    Parameters:
        df (pd.DataFrame): DataFrame with timestamp as index.
        new_index (pd.DatetimeIndex): Target time index.

    Returns:
        pd.DataFrame: Aligned and filled DataFrame.
    """
    df_aligned = df.reindex(new_index, method="nearest")
    df_aligned = df_aligned.interpolate(method="time").ffill().bfill()
    return df_aligned


def interpolate_gpsdf(df: pd.DataFrame) -> pd.DataFrame:
    need_cols = ["timestamp", "cattle_id", "x", "y"]
    cattle_id = df["cattle_id"].unique()[0]
    assert all(
        [col in df.columns for col in need_cols]
    ), f"必要なカラムが不足しています: {need_cols}"
    assert df["cattle_id"].nunique() == 1, "複数のcattle_idが含まれています"
    # check timestamp type
    assert pd.api.types.is_datetime64_any_dtype(
        df["timestamp"]
    ), "timestampの型が不正です"
    df = df[need_cols].copy()
    time_min, time_max = df["timestamp"].min(), df["timestamp"].max()
    new_index = pd.date_range(start=time_min, end=time_max, freq="s")
    df = df.set_index("timestamp").reindex(new_index)
    df = df.interpolate(method="time")
    df = df.reset_index()
    df["cattle_id"] = cattle_id
    df = df.rename(columns={"index": "timestamp"})
    return df


def interpolate_trackdf(df: pd.DataFrame) -> pd.DataFrame:
    need_cols = ["frame_id", "track_id", "x1", "y1", "x2", "y2"]
    assert all(
        [col in df.columns for col in need_cols]
    ), f"必要なカラムが不足しています: {need_cols}"
    df = df.copy()
    new_dfs = list()
    for _, group in df.groupby("track_id"):
        new_index = list(range(group["frame_id"].min(), group["frame_id"].max() + 1))
        group = group.set_index("frame_id")
        group = group.reindex(new_index)
        group = group.interpolate(method="linear")
        group = group.dropna()
        group = group.reset_index()
        new_dfs.append(group)
    df = pd.concat(new_dfs)
    df["frame_id"] = df["frame_id"].map(int)
    df["track_id"] = df["track_id"].map(int)
    return df


def calc_assignment(match_result_df, axis):
    # 損失行列を作成（1 - 類似度）
    if axis == "max":
        cost_matrix = 1 - match_result_df.copy().values
    else:
        cost_matrix = match_result_df.copy().values

    # ハンガリアン法の適用（最小コスト割当）
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 割当結果を DataFrame として出力
    assignment = pd.DataFrame(
        {
            "track_id": match_result_df.index[row_ind],
            "gps_id": match_result_df.columns[col_ind],
            "similarity": match_result_df.values[row_ind, col_ind],
        }
    )

    assignment = assignment.sort_values("track_id").reset_index(drop=True)
    return assignment


def calc_assignment_multi(match_result_df, axis):
    """
    各 track（行）ごとに最適な gps（列）を選択し、多対一マッチングを行う。

    Parameters
    ----------
    match_result_df : DataFrame
        index に track_id、columns に gps_id、各セルに類似度
    axis : {'max', 'min'}
        'max' のとき similarity を最大化／'min' のとき cost を最小化

    Returns
    -------
    assignment : DataFrame
        track_id, gps_id, similarity の３列を持つ DataFrame
    """
    M = match_result_df.values
    # 各行の argmax（'min' なら argmin）
    if axis == "max":
        col_idx = M.argmax(axis=1)
    elif axis == "min":
        col_idx = M.argmin(axis=1)
    else:
        raise NotImplementedError

    rows = np.arange(M.shape[0])
    assignment = pd.DataFrame(
        {
            "track_id": match_result_df.index,
            "gps_id": match_result_df.columns[col_idx],
            "similarity": M[rows, col_idx],
        }
    )

    return assignment.sort_values("track_id").reset_index(drop=True)


def make_disallow_multi_assign_pairs(trackdf):
    trackdf = trackdf.copy()
    track_frames_dict = {
        track_id: track_subdf["frame_id"].unique().tolist()
        for track_id, track_subdf in trackdf.groupby("track_id")
    }

    disallow_multi_assign_pairs = list()
    for track_id1, frame_ids1 in track_frames_dict.items():
        for track_id2, frame_ids2 in track_frames_dict.items():
            if track_id1 == track_id2:
                continue
            if set(frame_ids1) & set(frame_ids2):
                disallow_multi_assign_pairs.append((track_id1, track_id2))

    return disallow_multi_assign_pairs


def calc_assignment_multi_lp(
    match_result_df: pd.DataFrame,
    axis: str,
    disallow_multi_assign_pairs=None,
    timeLimit=120,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    match_result_df : DataFrame
        行 = track_id, 列 = gps_id, 要素 = 類似度（大きいほど良い/小さいほど良い）
    axis : {"max", "min"}
        最大化か最小化か
    disallow_multi_assign_pairs : Iterable[Tuple[Hashable, Hashable]] or None
        (track_id1, track_id2) のタプル集合。
        同じ gps_id へ同時に割当てることを禁止する組だけ列挙する。

    Returns
    -------
    DataFrame
        columns = ["track_id", "gps_id", "similarity"]、track_id 順に整列
    """
    if disallow_multi_assign_pairs is None:
        disallow_multi_assign_pairs = []

    # ---------- 前処理 ----------
    cost = match_result_df.values
    if axis == "min":
        cost = -cost  # 最小化→最大化へ変換
    elif axis != "max":
        raise ValueError("axis must be 'max' or 'min'")

    n_track, n_gps = cost.shape
    t_idx = {t: i for i, t in enumerate(match_result_df.index)}
    g_idx = {g: j for j, g in enumerate(match_result_df.columns)}

    # ---------- モデル ----------
    prob = pulp.LpProblem("partial_multi_assignment", pulp.LpMaximize)

    # 変数 x_{ij} ∈ {0,1}
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
        for i in range(n_track)
        for j in range(n_gps)
    }

    # 各 track はちょうど 1 個の gps に割当て
    for i in range(n_track):
        prob += pulp.lpSum(x[i, j] for j in range(n_gps)) == 1

    # 禁止ペア制約
    for t1, t2 in disallow_multi_assign_pairs:
        if len(set([t1, t2]) & set(t_idx.keys())) < 2:
            continue
        i1, i2 = t_idx[t1], t_idx[t2]
        for j in range(n_gps):
            prob += x[i1, j] + x[i2, j] <= 1

    # 目的関数
    prob += pulp.lpSum(
        cost[i, j] * x[i, j] for i in range(n_track) for j in range(n_gps)
    )

    # ---------- 求解 ----------
    _ = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=timeLimit))
    if prob.status != pulp.LpStatusOptimal:
        print(
            f"Warning: LP solver did not find an optimal solution. Status: {prob.status}"
        )

    # ---------- 結果復元 ----------
    assignment = []
    for i in range(n_track):
        for j in range(n_gps):
            if pulp.value(x[i, j]) > 0.5:
                assignment.append(
                    {
                        "track_id": match_result_df.index[i],
                        "gps_id": match_result_df.columns[j],
                        "similarity": match_result_df.values[i, j],
                    }
                )
                break

    return pd.DataFrame(assignment).sort_values("track_id").reset_index(drop=True)
