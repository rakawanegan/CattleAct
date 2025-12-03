import os
import random
import json
from datetime import datetime
from collections import defaultdict

import cv2
import hydra
import numpy as np
import pandas as pd
import swifter
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import sys

# Add the parent directory to the system path to allow for package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 必要な関数とクラスをインポート
from src.cost import calc_euclidean_match_score_only_same_parts_df
from src.gps import (
    load_drone_to_world_homography_matrix,
    load_gps_data,
    load_suzume_to_drone_homography_matrix,
)
from src.matching import (
    align_and_fill,
    calc_assignment_multi_lp,
    make_disallow_multi_assign_pairs,
)
from src.metric import matching_accuracy
from src.track import (
    apply_homography,
    convert_avi_to_mp4,
    drop_impossible_track_gps_df,
    init_deepsort,
    process_with_deepsort,
)
from src.utils import (
    frameid_to_timestamp,
    generate_gps_path,
    get_start_end_datetime,
    load_matching_gt,
    parse_video_filename_to_gps_params,
    plot_matching_result,
)

# ---【追加】--- 距離学習モデルのクラスをインポート ---
# (注意: 実際のファイルパスに合わせて修正してください)
from metric.action_with_image import LitVisionTransformer
from metric.interaction_cls_image_metric import LitHybridStreamFusionForMetricLearning

# ---【修正】--- 行動推定用のヘルパー関数群 (距離学習ベース) ---

def load_action_class_names():
    """行動クラス名とIDの辞書を返す。"""
    return {0: 'grazing', 1: 'standing', 2: 'lying', 3: 'riding'}

def load_action_model_from_checkpoint(model_path, device):
    """距離学習で学習した行動推定モデル(LitVisionTransformer)をロードする。"""
    model = LitVisionTransformer.load_from_checkpoint(model_path, map_location=device)
    model = model.to(device)
    model.eval()
    return model

def run_action_inference_knn(model, bbox_image, train_embeddings, train_labels, class_names, device, k=5):
    """BBOX画像からk-NN法で行動を推定する。"""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(bbox_image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        query_embedding = nn.functional.normalize(model(input_batch), p=2, dim=1)
    
    distances = torch.cdist(query_embedding, train_embeddings.to(device)).squeeze(0)
    top_k_indices = torch.topk(distances, k, largest=False).indices
    # ---【修正】--- インデックスをCPUに移動 ---
    top_k_labels = train_labels[top_k_indices.cpu()].numpy()
    
    predicted_label_id = np.bincount(top_k_labels).argmax()
    predicted_label = class_names.get(predicted_label_id, "Unknown")
    confidence = np.count_nonzero(top_k_labels == predicted_label_id) / k
    return predicted_label, confidence

# ---【修正】--- インタラクション検出用のヘルパー関数群 (距離学習ベース) ---

def load_interaction_class_names():
    """インタラクションクラス名とIDの辞書を返す。"""
    return {0: 'no_interaction', 1: 'interest', 2: 'conflict', 3: 'mount'}

def load_interaction_model_from_checkpoint(model_path, device):
    """距離学習で学習したインタラクション推定モデルをロードする。"""
    model = LitHybridStreamFusionForMetricLearning.load_from_checkpoint(
        model_path,
        map_location=device,
        embedding_size=256,
        triplet_margin=0.5,
        learning_rate=1e-4,
        freeze_vit=True,
        fusion_type="attention",
        num_classes=4,
        map_label=load_interaction_class_names(),
        vit_ckpt_path="",
    )
    model = model.to(device)
    model.eval()
    return model

def run_interaction_inference_knn(model, image1, image2, image_context, train_embeddings, train_labels, class_names, device, k=5):
    """3つの画像からk-NN法でインタラクションを推定する。"""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    t1 = preprocess(image1).unsqueeze(0).to(device)
    t2 = preprocess(image2).unsqueeze(0).to(device)
    t_context = preprocess(image_context).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_embedding = model(t1, t2, t_context)
        query_embedding = nn.functional.normalize(raw_embedding, p=2, dim=1)

    distances = torch.cdist(query_embedding, train_embeddings.to(device)).squeeze(0)
    top_k_indices = torch.topk(distances, k, largest=False).indices
    # ---【修正】--- インデックスをCPUに移動 ---
    top_k_labels = train_labels[top_k_indices.cpu()].numpy()

    predicted_label_id = np.bincount(top_k_labels).argmax()
    predicted_label = class_names.get(predicted_label_id, "Unknown")
    confidence = np.count_nonzero(top_k_labels == predicted_label_id) / k
    return predicted_label, confidence

# --- 共通および元からあるヘルパー関数群 ---

def crop_image_from_frame(frame, bbox):
    """OpenCVのフレームから指定BBOXを切り抜き、PIL Imageを返す。"""
    x1, y1, x2, y2 = map(int, bbox)
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x1 >= x2 or y1 >= y2:
        return Image.new('RGB', (0, 0))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    cropped_image = pil_image.crop((x1, y1, x2, y2))
    return cropped_image

def calculate_iou(box1, box2):
    """2つのBBOXのIOUを計算する。"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def find_interacting_groups(frame_data):
    """IOUが0より大きいBBOXのグループを見つける。"""
    tracks = list(frame_data.itertuples(index=False))
    if len(tracks) < 2:
        return [], set()
    adj = defaultdict(list)
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            bbox1 = (tracks[i].x1, tracks[i].y1, tracks[i].x2, tracks[i].y2)
            bbox2 = (tracks[j].x1, tracks[j].y1, tracks[j].x2, tracks[j].y2)
            if calculate_iou(bbox1, bbox2) > 0:
                adj[i].append(j)
                adj[j].append(i)
    visited = set()
    interacting_groups = []
    processed_track_ids = set()
    for i in range(len(tracks)):
        if i not in visited:
            current_group_indices = []
            q = [i]
            visited.add(i)
            while q:
                u = q.pop(0)
                current_group_indices.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            if len(current_group_indices) > 1:
                group_tracks = [tracks[idx] for idx in current_group_indices]
                interacting_groups.append(group_tracks)
                for track in group_tracks:
                    processed_track_ids.add(track.track_id)
    return interacting_groups, processed_track_ids

def get_union_bbox(group):
    """BBOXのリストを包含する統合BBOXを計算する。"""
    x1s = [track.x1 for track in group]
    y1s = [track.y1 for track in group]
    x2s = [track.x2 for track in group]
    y2s = [track.y2 for track in group]
    return min(x1s), min(y1s), max(x2s), max(y2s)

# --- メイン処理関数群 (変更なし) ---

def run_detection_and_tracking(cfg, output_dir):
    """Part 1: 検出、追跡、世界座標系への変換。"""
    # (この関数の内容は変更ありません)
    print("--- Part 1: Starting Detection and Tracking ---")
    p_movie = os.path.join(cfg.data.data_dir, cfg.data.p_movie)
    sensor_data_base_dir = cfg.sensor_data_base_dir
    is_debug = cfg.is_debug
    fps = (cv2.VideoCapture(p_movie).get(cv2.CAP_PROP_FPS) if cfg.fps is None else cfg.fps)
    if is_debug: fps = 1
    _, start_time, end_time = parse_video_filename_to_gps_params(p_movie, sensor_data_base_dir)
    if is_debug:
        end_time = str(datetime.strptime(start_time, "%H:%M:%S") + pd.Timedelta(seconds=30)).split(" ")[1]
        print(f"Debug Mode: Start time: {start_time}, End time: {end_time}")
    p_gps = generate_gps_path(sensor_data_base_dir, start_time, end_time, output_dir)
    gpsdf = load_gps_data(p_gps, sensor_data_base_dir, start_time, end_time)
    model = YOLO(cfg.yolo_model_path)
    output_video_path = os.path.join(output_dir, os.path.basename(p_movie).replace(".avi", ".mp4").replace(" ", "_"))
    if os.path.exists(output_video_path): os.remove(output_video_path)
    convert_avi_to_mp4(p_movie, output_video_path, fps, is_debug=is_debug)
    p_movie_mp4 = output_video_path
    results = model.predict(source=p_movie_mp4, project=output_dir, imgsz=cfg.imgsz, iou=cfg.iou, conf=cfg.conf, device=cfg.device if cfg.device else 0, save_txt=False, save=False, stream=True)
    deepsort = init_deepsort(cfg.deep_sort_config_path, cfg.deep_sort_ckpt_path)
    trackdf = process_with_deepsort(results, deepsort, model.names)
    trackdf["center_x"] = (trackdf["x1"] + trackdf["x2"]) / 2
    trackdf["center_y"] = (trackdf["y1"] + trackdf["y2"]) / 2
    suzume_to_drone_H = load_suzume_to_drone_homography_matrix(cfg.p_mat_suzume_to_drone)
    drone_to_world_H = load_drone_to_world_homography_matrix(cfg.p_mat_drone_to_world)
    trackdf[["drone_center_x", "drone_center_y"]] = trackdf[["center_x", "center_y"]].swifter.apply(lambda x: pd.Series(apply_homography(x, suzume_to_drone_H)), axis=1)
    trackdf[["world_x", "world_y"]] = trackdf[["drone_center_x", "drone_center_y"]].swifter.apply(lambda x: pd.Series(apply_homography(x, drone_to_world_H)), axis=1)
    p_tracked = p_gps.replace("gps", "tracked")
    trackdf.to_csv(p_tracked, index=False)
    print(f"Tracking data saved to: {p_tracked}")
    print("--- Part 1: Finished ---")
    return trackdf, gpsdf, p_movie_mp4, suzume_to_drone_H, drone_to_world_H, p_tracked

def run_matching(cfg, trackdf_raw, gpsdf_raw, suzume_to_drone_H, drone_to_world_H, p_tracked, matching_output_dir):
    """Part 2: 追跡データとGPSデータのマッチング。"""
    # (この関数の内容は変更ありません)
    print("\n--- Part 2: Starting Matching ---")
    use_col = ["timestamp", "track_id", "frame_id", "x1", "y1", "x2", "y2", "world_x", "world_y"]
    mat_suzume_to_world = drone_to_world_H @ suzume_to_drone_H
    _trackdf, _gpsdf = drop_impossible_track_gps_df(trackdf_raw.copy(), gpsdf_raw.copy(), cfg.drop_point, mat_suzume_to_world, cfg.gps_buff)
    trackdf = trackdf_raw[trackdf_raw["track_id"].isin(_trackdf["track_id"])]
    gpsdf = gpsdf_raw[gpsdf_raw["cattle_id"].isin(_gpsdf["cattle_id"])]
    gpsdf["timestamp"] = pd.to_datetime(gpsdf["timestamp"], format="%Y-%m-%d %H:%M:%S")
    start_ts, end_ts = get_start_end_datetime(p_tracked)
    duration = (end_ts - start_ts).total_seconds()
    num_frames = trackdf["frame_id"].max()
    fps_calc = num_frames / duration if duration > 0 else 30
    trackdf = trackdf.copy()
    trackdf["timestamp"] = trackdf["frame_id"].map(lambda fidx: frameid_to_timestamp(fidx, fps_calc, start_ts))
    trackdf = trackdf[use_col]
    disallow_multi_assign_pairs = make_disallow_multi_assign_pairs(trackdf)
    new_index = pd.date_range(start=start_ts, end=end_ts, freq="1s")
    new_dfs = []
    for _, group in trackdf.groupby("track_id"):
        track_df = group.copy().set_index("timestamp").resample("1s").mean()
        new_dfs.append(track_df)
    trackdf = pd.concat(new_dfs).reset_index().rename(columns={'index': "timestamp"})
    trackdf = trackdf.dropna(subset=["world_x", "world_y"])
    trackdf[["track_id", "frame_id"]] = trackdf[["track_id", "frame_id"]].astype(int)
    trackdf["time_id"] = (trackdf["timestamp"] - start_ts).dt.total_seconds().astype(int)
    new_dfs = []
    for _, group in gpsdf.groupby("cattle_id"):
        gps_df = group.copy().set_index("timestamp").resample("1s").mean()
        gps_df = align_and_fill(gps_df, new_index)
        new_dfs.append(gps_df)
    gpsdf = pd.concat(new_dfs).reset_index().rename(columns={'index': "timestamp"})
    gpsdf = gpsdf.dropna(subset=["x", "y"])
    gpsdf["timestamp"] = pd.to_datetime(gpsdf["timestamp"], format="%Y-%m-%d %H:%M:%S")
    gpsdf["time_id"] = (gpsdf["timestamp"] - start_ts).dt.total_seconds()
    gpsdf[["cattle_id", "time_id"]] = gpsdf[["cattle_id", "time_id"]].astype(int)
    cost_df = calc_euclidean_match_score_only_same_parts_df(trackdf, gpsdf)
    cost_min = cost_df.values.min()
    cost_max = cost_df.values.max()
    normalized_cost_df = (cost_df - cost_min) / (cost_max - cost_min)
    normalized_cost_df.to_csv(os.path.join(matching_output_dir, 'cost.csv'))
    assignment_df = calc_assignment_multi_lp(normalized_cost_df, axis='min', disallow_multi_assign_pairs=disallow_multi_assign_pairs)
    assignment_dict = dict(zip(assignment_df['track_id'], assignment_df['gps_id']))
    save_path = os.path.join(matching_output_dir, 'assignment.png')
    with open(save_path.replace('.png', '.json'), 'w') as f:
        json.dump({k: int(v) for k, v in assignment_dict.items()}, f, indent=4)
    print("--- Part 2: Finished ---")
    return assignment_dict


# ---【修正】--- run_visualization 関数 ---

def run_visualization(cfg, p_movie_mp4, trackdf_raw, assignment_dict, matching_output_dir):
    """Part 3: マッチング結果、行動推定、インタラクション推定結果を動画に描画する。"""
    print("\n--- Part 3: Starting Individual Identification and Action/Interaction Estimation Video Generation ---")
    assignment_df = pd.DataFrame(list(assignment_dict.items()), columns=['track_id', 'cattle_id'])
    assignment_df['cattle_id'] = assignment_df['cattle_id'].astype(int)
    visualize_df = pd.merge(trackdf_raw, assignment_df, on='track_id', how='left')
    visualize_df['cattle_id'] = visualize_df['cattle_id'].fillna(-1).astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- モデルと埋め込みデータのロード ---
    # 行動推定モデル
    action_class_names = load_action_class_names()
    action_model = load_action_model_from_checkpoint(cfg.action_model.path, device)
    action_embedding_dir = cfg.embedding_dir
    action_train_embeddings = torch.load(os.path.join(action_embedding_dir, 'action', 'train_embeddings.pt'))
    action_train_labels = torch.load(os.path.join(action_embedding_dir, 'action', 'train_labels.pt'))

    # インタラクション推定モデル
    interaction_class_names = load_interaction_class_names()
    interaction_model = load_interaction_model_from_checkpoint(cfg.interaction_model.path, device)
    interaction_embedding_dir = cfg.embedding_dir
    interaction_train_embeddings = torch.load(os.path.join(interaction_embedding_dir, 'interaction', 'train_embeddings.pt'))
    interaction_train_labels = torch.load(os.path.join(interaction_embedding_dir, 'interaction', 'train_labels.pt'))

    # --- 動画処理の準備 ---
    cap = cv2.VideoCapture(p_movie_mp4)
    if not cap.isOpened():
        print(f"Error: Could not open video {p_movie_mp4}")
        return
    output_video_path = os.path.join(matching_output_dir, 'tracking_with_ids_actions_and_interactions.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (video_width, video_height))
    data_by_frame = {frame_id: group for frame_id, group in visualize_df.groupby('frame_id')}

    # --- フレームごとの処理ループ ---
    frame_idx = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx in data_by_frame:
            current_frame_data = data_by_frame[frame_idx]
            interacting_groups, _ = find_interacting_groups(current_frame_data)
            processed_track_ids = set()

            # 1. インタラクションの推定と描画
            for group in interacting_groups:
                if len(group) < 2: continue
                track1, track2 = group[0], group[1] # 2頭のインタラクションを想定
                bbox1 = (track1.x1, track1.y1, track1.x2, track1.y2)
                bbox2 = (track2.x1, track2.y1, track2.x2, track2.y2)
                union_bbox = get_union_bbox(group)

                image1 = crop_image_from_frame(frame, bbox1)
                image2 = crop_image_from_frame(frame, bbox2)
                image_context = crop_image_from_frame(frame, union_bbox)
                
                if all(img.size[0] > 0 and img.size[1] > 0 for img in [image1, image2, image_context]):
                    interaction, conf = run_interaction_inference_knn(
                        interaction_model, image1, image2, image_context,
                        interaction_train_embeddings, interaction_train_labels,
                        interaction_class_names, device
                    )
                    
                    # ---【修正】--- "no_interaction" でない場合のみ描画し、処理済みIDに追加 ---
                    if interaction != 'no_interaction':
                        x1_u, y1_u, x2_u, y2_u = map(int, union_bbox)
                        cv2.rectangle(frame, (x1_u, y1_u), (x2_u, y2_u), (255, 0, 255), 4)

                        label = f"Interaction: {interaction} ({conf:.2f})"
                        cv2.putText(frame, label, (x1_u, y1_u - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                        
                        # このグループの個体を処理済みとしてIDを追加
                        for track in group:
                            processed_track_ids.add(track.track_id)

            # 2. 個別の個体（行動）の推定と描画
            for _, row in current_frame_data.iterrows():
                # ---【修正】--- インタラクション中と判定された個体はスキップ ---
                if row['track_id'] in processed_track_ids: continue
                
                x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                cattle_id = row['cattle_id']
                color = (0, 255, 0) if cattle_id != -1 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

                if cattle_id != -1:
                    label = f"ID: {cattle_id}"
                    font_scale, font_thickness = 1.2, 3
                    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    text_pos = (x1, y1 - 15)
                    cv2.rectangle(frame, (text_pos[0], text_pos[1] - text_h - baseline), (text_pos[0] + text_w, text_pos[1] + baseline), (0, 0, 0), -1)
                    cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

                    bbox_image = crop_image_from_frame(frame, (x1, y1, x2, y2))
                    if bbox_image.size[0] > 0 and bbox_image.size[1] > 0:
                        action, conf = run_action_inference_knn(
                            action_model, bbox_image, action_train_embeddings,
                            action_train_labels, action_class_names, device
                        )
                        action_label = f"Action: {action} ({conf:.2f})"
                        cv2.putText(frame, action_label, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Individual identified tracking video with actions and interactions saved to: {output_video_path}")
    print("--- Part 3: Finished ---")


@hydra.main(config_path="conf", config_name="demo", version_base=None)
def main(cfg):
    """検出、追跡、マッチング、可視化のパイプラインを実行するメイン関数。"""
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    matching_output_dir = os.path.join(output_dir, cfg.data.data_dir)
    os.makedirs(matching_output_dir, exist_ok=True)

    trackdf_raw, gpsdf_raw, p_movie_mp4, mat_s2d, mat_d2w, p_tracked = run_detection_and_tracking(cfg, output_dir)

    assignment_dict = run_matching(
        cfg, trackdf_raw, gpsdf_raw, mat_s2d, mat_d2w, p_tracked, matching_output_dir
    )

    run_visualization(cfg, p_movie_mp4, trackdf_raw, assignment_dict, matching_output_dir)


if __name__ == "__main__":
    main()