import os
import json

import numpy as np
import pandas as pd
import hydra

import sys;sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cost import calc_euclidean_match_score_only_same_parts_df
from src.matching import (
    align_and_fill,
    calc_assignment_multi_lp,
    make_disallow_multi_assign_pairs,
)
from src.track import drop_impossible_track_gps_df
from src.utils import frameid_to_timestamp, get_start_end_datetime, load_matching_gt, plot_matching_result
from src.metric import matching_accuracy, matching_precision


def align_and_fill_limited(df, new_index):
    """
    Align the track DataFrame to a new time index and fill missing values via interpolation.

    Parameters:
        df (pd.DataFrame): DataFrame with timestamp as index.
        new_index (pd.DatetimeIndex): Target time index.

    Returns:
        pd.DataFrame: Aligned and filled DataFrame.
    """
    df_aligned = df.reindex(new_index, method="nearest")
    df_aligned = df_aligned.interpolate(method="time", limit=3)
    return df_aligned

@hydra.main(config_path="conf", config_name="demo", version_base=None)
def main(cfg):
    use_col = [
        "timestamp",
        "track_id",
        "frame_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "world_x",
        "world_y",
    ]
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.data.data_dir), exist_ok=True)

    data_dir = os.path.join(cfg.data_dir, cfg.data.data_dir)
    p_trackdf = os.path.join(data_dir, cfg.data.p_trackdf)
    p_gt_match = os.path.join(data_dir, cfg.data.p_gt_match)
    p_movie = os.path.join(data_dir, cfg.data.p_movie)
    p_gpsdf = os.path.join(data_dir, cfg.data.p_gpsdf)

    mat_drone_to_world = np.load(cfg.p_mat_drone_to_world)
    mat_suzume_to_drone = np.load(cfg.p_mat_suzume_to_drone)
    mat_suzume_to_world = mat_drone_to_world @ mat_suzume_to_drone
    trackdf = pd.read_csv(p_trackdf)
    match_gt, match_gt_difficult = load_matching_gt(p_gt_match)
    gpsdf = pd.read_csv(p_gpsdf)

    _trackdf, _gpsdf = drop_impossible_track_gps_df(
        trackdf, gpsdf, cfg.drop_point, mat_suzume_to_world, cfg.gps_buff
    )
    trackdf = trackdf.loc[trackdf["track_id"].isin(_trackdf["track_id"])]
    gpsdf = gpsdf.loc[gpsdf["cattle_id"].isin(_gpsdf["cattle_id"])]

    gpsdf["timestamp"] = pd.to_datetime(gpsdf["timestamp"], format="%Y-%m-%d %H:%M:%S")

    start_ts, end_ts = get_start_end_datetime(p_trackdf)
    duration = (end_ts - start_ts).total_seconds()
    num_frames = trackdf["frame_id"].max()
    fps = num_frames / duration

    trackdf["timestamp"] = trackdf["frame_id"].map(
        lambda fidx: frameid_to_timestamp(fidx, fps, start_ts)
    )
    trackdf = trackdf[use_col]

    disallow_multi_assign_pairs = make_disallow_multi_assign_pairs(trackdf)

    new_index = pd.date_range(start=start_ts, end=end_ts, freq="1s")

    new_dfs = list()
    for _, group in trackdf.groupby("track_id"):
        track_df = group.copy().set_index("timestamp")
        track_df = track_df.resample("1s").mean()
        # track_df = align_and_fill_limited(track_df, new_index)
        new_dfs.append(track_df)
    trackdf = pd.concat(new_dfs)
    trackdf = trackdf.reset_index()
    trackdf = trackdf.rename(columns={'index': "timestamp"})
    trackdf = trackdf.dropna(subset=["world_x", "world_y"])
    trackdf[["track_id", "frame_id"]] = trackdf[["track_id", "frame_id"]].map(int)
    trackdf["time_id"] = (trackdf["timestamp"] - start_ts).dt.total_seconds().map(int)

    new_dfs = list()
    for _, group in gpsdf.groupby("cattle_id"):
        gpsdf = group.copy().set_index("timestamp")
        gpsdf = gpsdf.resample("1s").mean()
        gpsdf = align_and_fill_limited(gpsdf, new_index)
        new_dfs.append(gpsdf)
    gpsdf = pd.concat(new_dfs)
    gpsdf = gpsdf.reset_index()
    gpsdf = gpsdf.rename(columns={'index': "timestamp"})
    gpsdf = gpsdf.dropna(subset=["x", "y"])
    gpsdf["timestamp"] = pd.to_datetime(gpsdf["timestamp"], format="%Y-%m-%d %H:%M:%S")
    gpsdf["time_id"] = (gpsdf["timestamp"] - start_ts).dt.total_seconds()
    gpsdf[["cattle_id", "time_id"]] = gpsdf[["cattle_id", "time_id"]].map(int)

    match_euclidean_result_df = calc_euclidean_match_score_only_same_parts_df(trackdf, gpsdf)
    match_euclidean_result_min = match_euclidean_result_df.values.min()
    match_euclidean_result_max = match_euclidean_result_df.values.max()
    match_euclidean_result_df = match_euclidean_result_df.map(
        lambda x: (x - match_euclidean_result_min) / (match_euclidean_result_max - match_euclidean_result_min)
    )
    match_euclidean_result_df.to_csv(os.path.join(cfg.output_dir, cfg.data.data_dir, 'cost_euclidean_only_same_parts.csv'))

    euclidean_lp_assignment = calc_assignment_multi_lp(
        match_euclidean_result_df,
        axis='min',
        disallow_multi_assign_pairs=disallow_multi_assign_pairs
    )
    euclidean_lp_assignment_result_dict = dict(zip(euclidean_lp_assignment['track_id'], euclidean_lp_assignment['gps_id']))
    euclidean_lp_assignment_accuracy = matching_accuracy(match_gt, euclidean_lp_assignment_result_dict)
    print(f'{euclidean_lp_assignment_accuracy=:.4f}')

    save_path = os.path.join(cfg.output_dir, cfg.data.data_dir, 'assignment_euclidean_lp_only_same_parts.png')

    with open(save_path.replace('.png', '.json'), 'w') as f:
        json.dump(euclidean_lp_assignment_result_dict, f, indent=4)

    plot_matching_result(
        match_gt,
        euclidean_lp_assignment_result_dict,
        save_path=save_path
    )


if __name__ == "__main__":
    main()