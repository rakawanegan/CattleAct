import os
import subprocess

import cv2
import numpy as np
import pandas as pd
import torch

# ----- YOLO/DeepSORT関連の関数 ----- #


def extract_bboxes(results):
    """
    Extract bounding boxes from detection results.

    Parameters:
        results: YOLO detection results object.

    Returns:
        list: List of bounding boxes in [x1, y1, x2, y2] format.
    """
    bbox_list = list()
    if hasattr(results, "boxes"):
        bboxes = results.boxes.xyxy
        if hasattr(bboxes, "tolist"):
            bbox_list = bboxes.tolist()
    return bbox_list


def calculate_bbox_centers(bbox_list):
    """
    Calculate centers of bounding boxes.

    Parameters:
        bbox_list (list): List of bounding boxes in [x1, y1, x2, y2] format.

    Returns:
        list: List of center coordinates [center_x, center_y].
    """
    centers = list()
    for bbox in bbox_list:
        x1, y1, x2, y2 = bbox[:4]
        centers.append([int((x1 + x2) / 2), int((y1 + y2) / 2)])
    return centers


def convert_avi_to_mp4(input_path, output_path, fps, is_debug=False):
    """
    AVI形式の動画ファイルをffmpegを用いてMP4形式に変換します。

    Parameters:
        input_path (str): 変換元のAVIファイルのパス。
        output_path (str): 変換後のMP4ファイルのパス。
        fps (int or float): 出力動画のフレームレート。
        is_debug (bool): Trueの場合は、先頭30秒のみ変換します。
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if os.path.exists(output_path):
        return

    # fpsパラメータを"-r"オプションで指定
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-r",
        str(fps),
        "-vcodec",
        "libx264",
        "-an",
    ]

    if is_debug:
        command.extend(["-t", "00:00:30"])

    command.append(output_path)
    subprocess.run(command)


def init_deepsort(
    config_path="deep_sort_pytorch/configs/deep_sort.yaml",
    ckpt_path="checkpoints/ckpt.t7",
):
    """
    Initialize DeepSort tracker with the given configuration and checkpoint.

    Parameters:
        config_path (str): Path to DeepSort configuration file.
        ckpt_path (str): Path to DeepSort checkpoint file.

    Returns:
        DeepSort: Initialized DeepSort object.
    """
    from deep_sort_pytorch.deep_sort import DeepSort
    from deep_sort_pytorch.utils.parser import get_config

    cfg = get_config()
    cfg.merge_from_file(config_path)
    deepsort = DeepSort(
        ckpt_path,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=torch.cuda.is_available(),
    )
    return deepsort


def compute_color_for_id(track_id):
    """
    Compute a color tuple for a given track ID.

    Parameters:
        track_id (int): Identifier of the track.

    Returns:
        tuple: Color in (R, G, B).
    """
    palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
    return tuple(int((p * (track_id**2 - track_id + 1)) % 255) for p in palette)


def process_with_deepsort(results, deepsort, class_names):
    """
    Process YOLO detection results with DeepSort tracker.

    Parameters:
        results: YOLO detection results stream.
        deepsort: Initialized DeepSort object.
        class_names (dict): Mapping from class indices to names.

    Returns:
        pd.DataFrame: DataFrame containing tracking results with columns
                        ['frame_id', 'track_id', 'class', 'conf', 'x1', 'y1', 'x2', 'y2'].
    """
    camera_df = pd.DataFrame(
        columns=["frame_id", "track_id", "class", "conf", "x1", "y1", "x2", "y2"]
    )
    for frame_id, result in enumerate(results):
        print(f"Processing frame {frame_id}")
        frame = result.orig_img.copy()
        if result.boxes is None or len(result.boxes) == 0:
            deepsort.increment_ages()
            continue

        detections = list()
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == 19 and conf > 0.2:
                bbox_xywh = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
                detections.append((bbox_xywh, conf, cls))

        if detections:
            xywhs = torch.Tensor([d[0] for d in detections])
            confs = torch.Tensor([d[1] for d in detections])
            clss = torch.Tensor([d[2] for d in detections])
            outputs = deepsort.update(xywhs, confs, clss, frame)

            for output, conf in zip(outputs, confs):
                x1, y1, x2, y2, track_id, cls = output
                label = f"{track_id} {class_names[int(cls)]} {conf:.2f}"
                color = compute_color_for_id(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                cv2.putText(
                    frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
                )
                new_row = pd.DataFrame(
                    [
                        [
                            frame_id,
                            track_id,
                            class_names[int(cls)],
                            int(conf),
                            x1,
                            y1,
                            x2,
                            y2,
                        ]
                    ],
                    columns=camera_df.columns,
                )
                camera_df = pd.concat([camera_df, new_row], ignore_index=True)
    return camera_df


def apply_homography(p, H):
    """
    Apply a perspective transformation to a point using the homography matrix.

    Parameters:
        p (list or tuple): Point coordinates [x, y].
        H (np.ndarray): Homography matrix.

    Returns:
        np.ndarray: Transformed point coordinates.
    """
    p_array = np.array([[[p[0], p[1]]]], dtype=np.float32)
    return cv2.perspectiveTransform(p_array, H)[0][0]


def is_non_overlapping_with_masks(bbox, drop_condition):
    """
    bbox: (x_min, y_min, x_max, y_max)
    frame_shape: (height, width)
    定数 X1,Y1,X2,Y2 によって定義される２つのマスク領域と
    入力 bbox の重なりを評価する。
    - いずれかのマスク領域と重なれば False
    - 両方とも重なっていなければ True
    """
    X1, Y1 = map(int, drop_condition)  # 木の領域の右下
    region_tree = (0, 0, X1, Y1)

    def overlaps(b1, b2):
        x1_min, y1_min, x1_max, y1_max = b1
        x2_min, y2_min, x2_max, y2_max = b2
        # 重なりなしの条件を否定して重なりありと判定
        return not (
            x1_max <= x2_min  # b1 が b2 の左側
            or x2_max <= x1_min  # b2 が b1 の左側
            or y1_max <= y2_min  # b1 が b2 の上側
            or y2_max <= y1_min  # b2 が b1 の上側
        )

    return not overlaps(bbox, region_tree)


def drop_impossible_track_gps_df(
    trackdf, gpsdf, drop_point, mat_suzume_to_world, gps_buff=5
):
    trackdf = trackdf.copy()
    gpsdf = gpsdf.copy()

    world_x_limit, world_y_limit = apply_homography(drop_point, mat_suzume_to_world)
    world_x_limit -= gps_buff
    world_y_limit -= gps_buff

    gpsdf = gpsdf.loc[~((gpsdf["x"] < world_x_limit) & (gpsdf["y"] < world_y_limit))]
    gpsdf = gpsdf.loc[gpsdf["y"] > 20]
    trackdf = trackdf.loc[
        trackdf.apply(
            lambda row: is_non_overlapping_with_masks(
                (row["x1"], row["y1"], row["x2"], row["y2"]), drop_point
            ),
            axis=1,
        )
    ]
    return trackdf, gpsdf
