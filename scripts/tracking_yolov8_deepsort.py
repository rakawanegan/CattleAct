import os
from datetime import datetime

import cv2
import hydra
import numpy as np
import pandas as pd
import swifter  # noqa: F401
from ultralytics import YOLO

import sys;sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gps import (
    load_drone_to_world_homography_matrix,
    load_gps_data,
    load_suzume_to_drone_homography_matrix,
)
from src.track import (
    apply_homography,
    convert_avi_to_mp4,
    init_deepsort,
    process_with_deepsort,
)
from src.utils import (
    generate_gps_path,
    parse_video_filename_to_gps_params,
)

@hydra.main(config_path="conf", config_name="demo", version_base=None)
def main(cfg):
    is_debug = cfg.is_debug
    print("Start processing...")
    print("Debug mode:", is_debug)

    p_movie = cfg.p_movie
    sensor_data_base_dir = cfg.sensor_data_base_dir
    output_dir = cfg.output_dir
    fps = (
        cv2.VideoCapture(p_movie).get(cv2.CAP_PROP_FPS) if cfg.fps is None else cfg.fps
    )

    if is_debug:
        fps = 1

    os.makedirs(output_dir, exist_ok=True)
    gps_dir, start_time, end_time = parse_video_filename_to_gps_params(
        p_movie, sensor_data_base_dir
    )

    if is_debug:
        end_time = str(
            datetime.strptime(start_time, "%H:%M:%S") + pd.Timedelta(seconds=30)
        ).split(" ")[1]
        print(f"Start time: {start_time}, End time: {end_time}")

    # GPSデータの読み込み・前処理
    p_gps = generate_gps_path(gps_dir, start_time, end_time, output_dir)

    if is_debug:
        print(f"GPSファイル: {p_gps}")

    sensor_df = load_gps_data(p_gps, gps_dir, start_time, end_time)

    # YOLOとDeepSORTによる物体検出・追跡
    model = YOLO(cfg.yolo_model_path)

    # AVIをMP4に変換
    output_video_path = os.path.join(
        output_dir, os.path.basename(p_movie).replace(".avi", ".mp4").replace(" ", "_")
    )
    if os.path.exists(output_video_path):
        os.remove(output_video_path)
    convert_avi_to_mp4(p_movie, output_video_path, fps, is_debug=is_debug)
    p_movie = output_video_path

    results = model.predict(
        source=p_movie,
        project=output_dir,
        imgsz=cfg.imgsz,
        iou=cfg.iou,
        conf=cfg.conf,
        device=cfg.device if cfg.device else 0,
        save_txt=False,
        save=False,
        stream=True,
    )

    deepsort = init_deepsort(cfg.deep_sort_config_path, cfg.deep_sort_ckpt_path)
    camera_df = process_with_deepsort(results, deepsort, model.names)

    camera_df["center_x"] = (camera_df["x1"] + camera_df["x2"]) / 2
    camera_df["center_y"] = (camera_df["y1"] + camera_df["y2"]) / 2

    # カメラ座標からドローン・世界座標へ変換
    suzume_to_drone_H = load_suzume_to_drone_homography_matrix(
        cfg.suzume_to_drone_matrix_path
    )
    drone_to_world_H = load_drone_to_world_homography_matrix(
        cfg.drone_to_world_matrix_path
    )

    camera_df[["drone_center_x", "drone_center_y"]] = camera_df[
        ["center_x", "center_y"]
    ].swifter.apply(lambda x: pd.Series(apply_homography(x, suzume_to_drone_H)), axis=1)
    camera_df[["world_x", "world_y"]] = camera_df[
        ["drone_center_x", "drone_center_y"]
    ].swifter.apply(lambda x: pd.Series(apply_homography(x, drone_to_world_H)), axis=1)
    p_tracked = p_gps.replace("gps", "tracked")
    camera_df.to_csv(p_tracked, index=False)


if __name__ == "__main__":
    main()
