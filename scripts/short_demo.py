import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from ultralytics import YOLO

import sys
sys.path.append('.')  # カスタムモジュールのパスを追加
from train.action_with_image import LitVisionTransformer
from train.interaction_with_image import LitHybridStreamFusion

# ==============================================================================
# 1. ヘルパー関数群 (モデルロード、推論、画像処理)
# ==============================================================================

def load_action_classification_model(model_path, device):
    """行動「分類」モデルをチェックポイントからロードする。"""
    model = LitVisionTransformer.load_from_checkpoint(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def load_interaction_model(model_path, device):
    """インタラクション「分類」モデルをチェックポイントからロードする。"""
    model = LitHybridStreamFusion.load_from_checkpoint(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def run_classification_inference(model, image, class_names, device):
    """単一画像のクラス分類推論を実行する。"""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted_id = torch.max(probabilities, 1)
        predicted_label = class_names.get(predicted_id.item(), "Unknown")
    return predicted_label, confidence.item()

def run_interaction_inference(model, image1, image2, image_context, class_names, device):
    """3つの画像のインタラクション分類推論を実行する。"""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    t1 = preprocess(image1).unsqueeze(0).to(device)
    t2 = preprocess(image2).unsqueeze(0).to(device)
    t_context = preprocess(image_context).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(t1, t2, t_context)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted_id = torch.max(probabilities, 1)
        predicted_label = class_names.get(predicted_id.item(), "Unknown")
    return predicted_label, confidence.item()

def crop_image_from_frame(frame, bbox):
    """フレームから指定BBoxを切り抜き、PIL Imageを返す。"""
    x1, y1, x2, y2 = map(int, bbox)
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x1 >= x2 or y1 >= y2:
        return Image.new('RGB', (0, 0))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb).crop((x1, y1, x2, y2))

def calculate_iou(box1, box2):
    """2つのBBoxのIoUを計算する。"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

# ==============================================================================
# 2. 単一フレーム処理関数
# ==============================================================================

def process_single_frame(frame, yolo_model, action_model, interaction_model, device, action_class_names, interaction_class_names):
    """単一フレームに対し、検出、行動・インタラクション推定、描画を行う。"""
    detection_results = yolo_model.predict(source=frame, verbose=False, iou=0.5)[0]
    annotated_frame = frame.copy()
    
    action_color = (255, 0, 0)      # Blue for action
    interaction_color = (255, 0, 255) # Magenta for interaction

    all_boxes_info = list(detection_results.boxes)
    if not all_boxes_info:
        return annotated_frame

    # BBoxのグループ化
    all_coords = [list(map(int, box.xyxy[0])) for box in all_boxes_info]
    num_boxes = len(all_coords)
    visited = [False] * num_boxes
    merged_groups = []
    for i in range(num_boxes):
        if visited[i]: continue
        q = [i]
        visited[i] = True
        min_x1, min_y1, max_x2, max_y2 = all_coords[i]
        head = 0
        while head < len(q):
            current_idx, head = q[head], head + 1
            for j in range(num_boxes):
                if not visited[j] and calculate_iou(all_coords[current_idx], all_coords[j]) > 0.0:
                    visited[j] = True
                    q.append(j)
                    other_coords = all_coords[j]
                    min_x1, min_y1 = min(min_x1, other_coords[0]), min(min_y1, other_coords[1])
                    max_x2, max_y2 = max(max_x2, other_coords[2]), max(max_y2, other_coords[3])
        if len(q) > 1:
            merged_groups.append(([min_x1, min_y1, max_x2, max_y2], [all_boxes_info[idx] for idx in q]))

    # 全個体の個別行動を推定・描画
    for box in all_boxes_info:
        bbox_image = crop_image_from_frame(frame, box.xyxy[0])
        if bbox_image.width > 0 and bbox_image.height > 0:
            action_label, _ = run_classification_inference(action_model, bbox_image, action_class_names, device)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), action_color, 2)
            (w, h), _ = cv2.getTextSize(action_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - h - 10), (x1 + w, y1), action_color, -1)
            cv2.putText(annotated_frame, action_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # インタラクションを推定・描画
    for merged_coords, original_boxes in merged_groups:
        if len(original_boxes) >= 2:
            sorted_boxes = sorted(original_boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]), reverse=True)
            image1 = crop_image_from_frame(frame, sorted_boxes[0].xyxy[0])
            image2 = crop_image_from_frame(frame, sorted_boxes[1].xyxy[0])
            image_context = crop_image_from_frame(frame, merged_coords)
            if all(img.width > 0 and img.height > 0 for img in [image1, image2, image_context]):
                interaction_label, _ = run_interaction_inference(interaction_model, image1, image2, image_context, interaction_class_names, device)
                x1, y1, x2, y2 = merged_coords
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), interaction_color, 3)
                label = f"Interaction: {interaction_label}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                text_x, text_y = x2 + 10, y1
                cv2.rectangle(annotated_frame, (text_x - 5, text_y - 5), (text_x + w + 5, text_y + h + 5), interaction_color, -1)
                cv2.putText(annotated_frame, label, (text_x, text_y + h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.line(annotated_frame, (text_x, text_y + h // 2), (x2, y1 + (y2 - y1) // 2), interaction_color, 2)

    return annotated_frame

# ==============================================================================
# 3. 動画生成メイン関数
# ==============================================================================

def create_processed_video(input_video_path, output_video_path, center_frame_num, 
                           yolo_model_path, action_model_path, interaction_model_path,
                           duration_seconds=4, output_fps=5):
    """動画を読み込み、指定区間を処理して新しい動画として保存する。"""
    
    # --- 1. デバイスとモデルの準備 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    yolo_model = YOLO(yolo_model_path)
    action_class_names = {0: 'grazing', 1: 'standing', 2: 'lying', 3: 'riding'}
    interaction_class_names = {0: 'no_interaction', 1: 'interest', 2: 'conflict', 3: 'mount'}
    
    action_model = load_action_classification_model(action_model_path, DEVICE)
    interaction_model = load_interaction_model(interaction_model_path, DEVICE)
    print("All models loaded successfully.")

    # --- 2. 動画I/Oの準備 ---
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_video_path}'")
        return
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (frame_width, frame_height))
    
    # --- 3. 処理対象フレームの計算 ---
    frame_step = int(original_fps / output_fps)
    num_frames_to_process_half = int(duration_seconds / 2 * output_fps)
    
    start_frame = max(0, center_frame_num - num_frames_to_process_half * frame_step)
    end_frame = min(total_frames, center_frame_num + num_frames_to_process_half * frame_step)

    print(f"Processing video from frame {start_frame} to {end_frame} (step: {frame_step}).")
    
    # --- 4. メインループ ---
    for frame_idx in range(start_frame, end_frame, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}.")
            continue
        
        print(f"  - Processing frame: {frame_idx}")
        processed_frame = process_single_frame(
            frame, yolo_model, action_model, interaction_model, DEVICE, 
            action_class_names, interaction_class_names
        )
        writer.write(processed_frame)

    # --- 5. リソースの解放 ---
    cap.release()
    writer.release()
    print(f"\nSuccessfully created video: {output_video_path}")

# ==============================================================================
# 4. 実行
# ==============================================================================

if __name__ == '__main__':
    # --- パラメータ設定 ---
    target_action = 'mount'
    # conflict ---
    if target_action == 'conflict':
        INPUT_VIDEO_PATH = '/mnt/nfs/CameraData/hiiku2/2025-03-05/17/2025-03-05 17-10-00~17-20-00.avi'
        CENTER_FRAME_NUM = 11400
        OUTPUT_VIDEO_PATH = 'outputs/output_clip_conflict.mp4'
    # mount ---
    elif target_action == 'mount':
        INPUT_VIDEO_PATH = '/mnt/nfs/CameraData/hiiku2/2025-03-05/16/2025-03-05 16-40-00~16-50-00.avi'
        CENTER_FRAME_NUM = 9510
        OUTPUT_VIDEO_PATH = 'outputs/output_clip_mount.mp4'
    else:
        raise ValueError("Unsupported target_action. Choose 'conflict' or 'mount'.")
    
    # モデルのチェックポイントファイルへのパス
    YOLO_MODEL_PATH = './checkpoints/best.pt'
    ACTION_MODEL_PATH = './checkpoints/action_image_aug_pose.ckpt' # 行動「分類」モデルのパスを想定
    INTERACTION_MODEL_PATH = './checkpoints/interaction_finetuned_postedwacv.ckpt'

    # --- 実行 ---
    # action_classification.ckptが存在することを確認してください
    if not os.path.exists(ACTION_MODEL_PATH):
        print(f"Error: Action classification model not found at '{ACTION_MODEL_PATH}'")
        print("Please provide the correct path to the classification model checkpoint.")
    else:
        create_processed_video(
            input_video_path=INPUT_VIDEO_PATH,
            output_video_path=OUTPUT_VIDEO_PATH,
            center_frame_num=CENTER_FRAME_NUM,
            yolo_model_path=YOLO_MODEL_PATH,
            action_model_path=ACTION_MODEL_PATH,
            interaction_model_path=INTERACTION_MODEL_PATH,
            duration_seconds=8,
            output_fps=5
        )