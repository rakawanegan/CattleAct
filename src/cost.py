import math
from collections import Counter

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def calc_feature(df, xy_col=["world_x", "world_y"]):
    g = df.copy()
    avg_ax = g[xy_col].mean().tolist()
    g = g.sort_values(by=["timestamp"])
    dx_dy = (g.iloc[-1][xy_col] - g.iloc[0][xy_col]).values
    distance = np.linalg.norm(dx_dy)
    ang = np.arctan2(dx_dy[1], dx_dy[0]) * 180 / np.pi
    ang = ang + 360 if ang < 0 else ang
    ang = int(ang)
    return ang, distance, avg_ax


def wrap_to_pi(diff: float) -> float:
    """
    差分を [-π, +π] の範囲に折り返す。
    """
    return (diff + np.pi) % (2 * np.pi) - np.pi


def angle_similarity_cos(a: float, b: float) -> float:
    """
    コサイン類似度を用いた角度類似度。

    類似度 = cos(Δθ)
    戻り値の範囲: [-1, +1]
        +1 → 完全一致
        0  → 直交（90°ずれ）
        -1 → 反対方向（180°ずれ）
    """
    Δ = wrap_to_pi(a - b)
    return np.cos(Δ)


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calc_diff_norm_from_df(df):
    df = df.copy()
    df["diff_x"] = df["world_x"].diff()
    df["diff_y"] = df["world_y"].diff()
    df["diff_x"] = df["diff_x"].fillna(0)
    df["diff_y"] = df["diff_y"].fillna(0)
    df["diff_norm"] = np.sqrt(df["diff_x"] ** 2 + df["diff_y"] ** 2)
    return df["diff_norm"]


def get_event_df(sourcedf, timestamp, duration):
    event_start_ts = timestamp - pd.Timedelta(seconds=duration // 2)
    event_end_ts = timestamp + pd.Timedelta(seconds=duration // 2)
    eventdf = sourcedf[
        (sourcedf["timestamp"] >= event_start_ts)
        & (sourcedf["timestamp"] <= event_end_ts)
    ]
    return eventdf


def get_event_timestamps_from_peak(df):
    df = df.copy()
    peaks, _ = find_peaks(calc_diff_norm_from_df(df), height=0.5)
    peak_timestamps = df.iloc[peaks]["timestamp"].values
    return peak_timestamps


def get_counter_dict(l):
    return dict(Counter(l))


def get_most_frequent_number(l):
    counter = Counter(l)
    count_ser = pd.Series(counter)
    count_ser /= sum(count_ser)
    return int(count_ser.idxmax()), count_ser.max()


def calc_euclidean_match_score_df(trackdf, gpsdf):
    def calc_euclidean_distance(cam_df, sens_df):
        diff = cam_df[["world_x", "world_y"]].values - sens_df[["x", "y"]].values
        return np.average(np.abs(diff))

    track_index = sorted(trackdf["track_id"].unique())
    gps_index = sorted(gpsdf["cattle_id"].unique())
    cost_matrix = np.zeros((len(track_index), len(gps_index)))
    match_result_df = pd.DataFrame(
        data=cost_matrix, index=track_index, columns=gps_index
    )
    for track_id, cam_df in trackdf.groupby("track_id"):
        for gps_id, sens_df in gpsdf.groupby("cattle_id"):
            match_result_df.loc[track_id, gps_id] = calc_euclidean_distance(
                cam_df, sens_df
            )
    return match_result_df


def calc_euclidean_match_score_only_same_parts_df(trackdf, gpsdf):
    def calc_euclidean_distance(cam_df, sens_df):
        same_timestamps = list(set(cam_df["timestamp"]) & set(sens_df["timestamp"]))
        cam_df = cam_df[cam_df["timestamp"].isin(same_timestamps)]
        sens_df = sens_df[sens_df["timestamp"].isin(same_timestamps)]
        diff = cam_df[["world_x", "world_y"]].values - sens_df[["x", "y"]].values
        return np.sqrt(np.sum(np.square(diff)))

    track_index = sorted(trackdf["track_id"].unique())
    gps_index = sorted(gpsdf["cattle_id"].unique())
    cost_matrix = np.zeros((len(track_index), len(gps_index)))
    match_result_df = pd.DataFrame(
        data=cost_matrix, index=track_index, columns=gps_index
    )
    for track_id, cam_df in trackdf.groupby("track_id"):
        for gps_id, sens_df in gpsdf.groupby("cattle_id"):
            match_result_df.loc[track_id, gps_id] = calc_euclidean_distance(
                cam_df, sens_df
            )
    return match_result_df


def calc_event_match_score_df(
    trackdf,
    gpsdf,
    use_abs_sim=False,
    event_time_duration=10,  # seconds
    drop_rotation_match_distance=1,  # m 向き変化
    drop_idiot_match_distance=20,  # m GPS-Track 距離誤差
):
    # TODO: 向き変化による中心座標の変化が考慮されていない。→BBOXの面積で正規化
    # TODO: 射影変換の誤差により縦移動の値変化が大きくなりやすい。

    track_id_index = list()
    gps_id_unq = gpsdf["cattle_id"].unique()
    match_result_df = list()

    for target_track_id, target_sub_trackdf in trackdf.groupby("track_id"):
        target_sub_trackdf = target_sub_trackdf.copy()

        peak_timestamps = get_event_timestamps_from_peak(target_sub_trackdf)

        matched_gps_list = list()
        for peak_timestamp in peak_timestamps:
            match_df = list()
            peak_timestamp = pd.to_datetime(peak_timestamp, format="%Y-%m-%d %H:%M:%S")
            event_track_df = get_event_df(
                target_sub_trackdf, peak_timestamp, event_time_duration
            )
            if not len(event_track_df):
                continue
            track_ang, _, track_avg_xy = calc_feature(
                event_track_df, xy_col=["world_x", "world_y"]
            )

            for gps_id, group in gpsdf.groupby("cattle_id"):
                event_gps_df = get_event_df(group, peak_timestamp, event_time_duration)
                if len(event_gps_df) == 0:
                    continue

                gps_ang, gps_distance, gps_avg_xy = calc_feature(
                    event_gps_df, xy_col=["x", "y"]
                )

                if (
                    gps_distance < drop_rotation_match_distance
                ):  # 向き変化による中心座標の変化を緩和
                    continue

                ang_sim = angle_similarity_cos(track_ang, gps_ang)
                euc_dist = euclidean_distance(track_avg_xy, gps_avg_xy)

                if (
                    euc_dist > drop_idiot_match_distance
                ):  # drop_idiot_match_distance [m] 以上離れた距離誤差は発生しない
                    continue

                match_df.append(
                    {
                        "gps_id": gps_id,
                        "ang_sim": ang_sim,
                        "euc_dist": euc_dist,
                    }
                )

            if not len(match_df):
                continue

            match_df = pd.DataFrame(match_df)

            if len(match_df) == 1:
                matched_gps_list.append(int(match_df["gps_id"].iloc[0]))
                continue

            if use_abs_sim:
                match_df["abs_sim"] = 1 - (
                    match_df["euc_dist"] - match_df["euc_dist"].min()
                ) / (match_df["euc_dist"].max() - match_df["euc_dist"].min())
                match_df["sim"] = match_df["ang_sim"] + match_df["abs_sim"]
            else:
                match_df["sim"] = match_df["ang_sim"].copy()

            matched_gps_list.append(
                int(match_df.loc[match_df["sim"].idxmax(), "gps_id"])
            )

        if not len(matched_gps_list):
            continue

        base_dict = {gps_id: None for gps_id in gps_id_unq}
        matched_dict = get_counter_dict(matched_gps_list)
        match_row = {key: matched_dict.get(key, base_dict[key]) for key in base_dict}
        track_id_index.append(target_track_id)
        match_result_df.append(match_row)

    match_result_df = pd.DataFrame(match_result_df)
    match_result_df.index = track_id_index
    match_result_df = match_result_df.astype(float).fillna(0)
    max_match_count = match_result_df.values.max()
    min_match_count = match_result_df.values.min()
    match_result_df = match_result_df.map(
        lambda x: (x - min_match_count) / (max_match_count - min_match_count)
    )
    return match_result_df
