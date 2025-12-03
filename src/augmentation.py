import math
import os
import random
import shutil

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, UnidentifiedImageError
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, F1Score
from torchvision import transforms as T

import wandb


class ImageMaskingFromSkeleton:
    """
    骨格情報に基づき、元の画像にラベル不変のマスキングを適用するTransformである。
    Cutout領域が保護対象のキーポイントと重複する場合、再試行を行う。
    TODO: 判別不可能なマスキング処理を行った場合には[0, 0, 0]とし、Unknownラベルに対応
    """

    def __init__(
        self,
        joint_map,
        cutout_prob=0.5,
        n_holes=1,
        scale=(0.02, 0.2),
        ratio=(0.3, 3.3),
        max_trials=10,
        skip_label=True,
        unuse_low_conf_skel=True,
        margin=10,
    ):
        """
        Args:
            joint_map (dict): 部位名と骨格点インデックスのマッピングである。
            cutout_prob (float): CutOutを適用する確率である。
            n_holes (int): マスキングする領域の数である。
            scale (tuple): 画像面積に対するマスク面積の比率の範囲である。
            ratio (tuple): マスクのアスペクト比の範囲である。
            max_trials (int): 保護キーポイントを避けるための最大試行回数である。
            skip_label (bool): 保護対象がないラベルの場合、マスキングをスキップするかどうか。
            unuse_low_conf_skel (bool): 骨格の信頼度が低い場合、マスキングをスキップするかどうか。
        """
        self.joint_map = joint_map
        self.cutout_prob = cutout_prob
        self.max_n_holes = n_holes
        self.scale = scale
        self.ratio = ratio
        self.max_trials = max_trials
        self.skip_label = skip_label
        self.unuse_low_conf_skel = unuse_low_conf_skel
        self.margin = margin

        # マスキング「可能」部位の定義
        label_to_maskable_parts = {
            0: ["left_hind_leg", "right_hind_leg"],  # grazing
            1: ["left_hind_leg", "right_hind_leg"],  # standing
            2: [],  # lying は骨格推定結果が信頼できないため、マスキングしない
            3: [],  # riding は骨格推定結果が信頼できないため、マスキングしない
        }

        self.pass_labels = [
            label for label, parts in label_to_maskable_parts.items() if not parts
        ]

        # 上記を反転させ、ラベル毎の「保護対象」部位を定義する
        all_parts = set(self.joint_map.keys())
        self.label_to_protected_parts = {
            label: list(all_parts - set(maskable))
            for label, maskable in label_to_maskable_parts.items()
        }

    def __call__(self, image, skeleton, label):
        """
        入力画像に対し、保護キーポイントを避けながらランダムなCutoutを適用する。
        """
        # if skeleton mean confidence under 0.4, return original image
        if self.unuse_low_conf_skel and skeleton[:, 2].mean() < 0.4:
            return image

        # if pass label, return original image
        if self.skip_label and label.item() in self.pass_labels:
            return image

        # determine n_holes with random at 0 to max_n_holes(prob: cutout_prob)
        n_holes = 0
        if random.random() < self.cutout_prob:
            n_holes = random.randint(1, self.max_n_holes)

        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)
        img_w, img_h = image.size
        img_area = img_w * img_h

        label_item = label.item() if hasattr(label, "item") else label
        protected_parts = self.label_to_protected_parts.get(label_item)
        if not protected_parts:
            # 保護対象がない場合は通常のCutOutと同様の処理でよいが、
            # ここでは何もしない実装とする
            return image

        # 保護対象のキーポイント座標群を取得
        protected_indices = {
            idx for part in protected_parts for idx in self.joint_map.get(part, [])
        }
        skeleton_np = skeleton.cpu().numpy() if hasattr(skeleton, "cpu") else skeleton
        protected_kpts = skeleton_np[list(protected_indices), :2]
        valid_protected_kpts = protected_kpts[
            (protected_kpts[:, 0] > 1) & (protected_kpts[:, 1] > 1)
        ]

        for _ in range(n_holes):
            for _ in range(self.max_trials):
                # Cutout領域の面積とアスペクト比をランダムに決定
                hole_area = img_area * random.uniform(self.scale[0], self.scale[1])
                aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

                h = int(round(math.sqrt(hole_area / aspect_ratio)))
                w = int(round(math.sqrt(hole_area * aspect_ratio)))

                if h >= img_h or w >= img_w:
                    continue

                # Cutout領域の左上の座標をランダムに決定
                x1 = random.randint(0, img_w - w)
                y1 = random.randint(0, img_h - h)
                x2, y2 = x1 + w, y1 + h

                # 保護キーポイントとの衝突チェック
                is_colliding = np.any(
                    (valid_protected_kpts[:, 0] >= x1 - self.margin) &
                    (valid_protected_kpts[:, 0] < x2 + self.margin) &
                    (valid_protected_kpts[:, 1] >= y1 - self.margin) &
                    (valid_protected_kpts[:, 1] < y2 + self.margin)
                )

                if not is_colliding:
                    draw.rectangle([x1, y1, x2, y2], fill="black")
                    break  # 衝突がなければ次のholeの生成へ

        return image_copy


class ImageMaskingFromSkeletonForInteraction:
    """
    2頭の牛の骨格情報に基づき、元の画像にラベル不変のマスキングを適用するTransformである。
    """

    def __init__(
        self,
        joint_map,
        cutout_prob=0.5,
        n_holes=1,
        scale=(0.02, 0.2),
        ratio=(0.3, 3.3),
        max_trials=10,
        margin=10,
    ):
        self.joint_map = joint_map
        self.cutout_prob = cutout_prob
        self.max_n_holes = n_holes
        self.scale = scale
        self.ratio = ratio
        self.max_trials = max_trials
        self.margin = margin

        # ラベル毎の「保護対象」部位を詳細化
        self.label_to_protected_parts = {
            0: [],  # no_interaction
            1: ["head", "neck"],  # interest
            2: ["head", "neck", "torso"],  # conflict
            3: ["head", "neck", "torso"],  # mount
        }

    def __call__(self, image, skeleton1, skeleton2, label):
        # determine n_holes with random at 0 to max_n_holes(prob: cutout_prob)
        n_holes = 0
        if random.random() < self.cutout_prob:
            n_holes = random.randint(1, self.max_n_holes)

        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)
        img_w, img_h = image.size
        img_area = img_w * img_h

        label_item = label.item() if hasattr(label, "item") else label
        protected_parts = self.label_to_protected_parts.get(label_item)
        if protected_parts is None:
            return image

        protected_indices = {
            idx for part in protected_parts for idx in self.joint_map.get(part, [])
        }

        all_protected_kpts = []
        for skeleton in [skeleton1, skeleton2]:
            skeleton_np = (
                skeleton.cpu().numpy()
                if hasattr(skeleton, "cpu")
                else np.array(skeleton)
            )
            if skeleton_np.size == 0:
                continue

            valid_indices = [idx for idx in protected_indices if idx < len(skeleton_np)]
            if not valid_indices:
                continue

            kpts = skeleton_np[valid_indices, :2]
            valid_kpts = kpts[(kpts[:, 0] > 1) & (kpts[:, 1] > 1)]
            if valid_kpts.shape[0] > 0:
                all_protected_kpts.append(valid_kpts)

        if not all_protected_kpts:
            return image

        valid_protected_kpts = np.vstack(all_protected_kpts)

        for _ in range(n_holes):
            for _ in range(self.max_trials):
                hole_area = img_area * random.uniform(self.scale[0], self.scale[1])
                aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
                h = int(round(math.sqrt(hole_area / aspect_ratio)))
                w = int(round(math.sqrt(hole_area * aspect_ratio)))

                if h >= img_h or w >= img_w:
                    continue

                x1 = random.randint(0, img_w - w)
                y1 = random.randint(0, img_h - h)
                x2, y2 = x1 + w, y1 + h

                is_colliding = np.any(
                    (valid_protected_kpts[:, 0] >= x1 - self.margin) &
                    (valid_protected_kpts[:, 0] < x2 + self.margin) &
                    (valid_protected_kpts[:, 1] >= y1 - self.margin) &
                    (valid_protected_kpts[:, 1] < y2 + self.margin)
                )

                if not is_colliding:
                    draw.rectangle([x1, y1, x2, y2], fill="black")
                    break
        return image_copy


class StandardCutout:
    """
    画像上のランダムな位置にCutoutを適用するTransform。
    骨格情報による保護は行わない。
    """
    def __init__(self, cutout_prob=0.5, n_holes=1,
                 scale=(0.02, 0.2), ratio=(0.3, 3.3)):
        """
        Args:
            cutout_prob (float): CutOutを適用する確率。
            n_holes (int): マスキングする領域の数。
            scale (tuple): 画像面積に対するマスク面積の比率の範囲。
            ratio (tuple): マスクのアスペクト比の範囲。
        """
        self.cutout_prob = cutout_prob
        self.n_holes = n_holes
        self.scale = scale
        self.ratio = ratio

    def __call__(self, image, skeleton, label):
        """
        入力画像に対し、ランダムなCutoutを適用する。
        """
        if random.random() > self.cutout_prob:
            return image

        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)
        img_w, img_h = image.size
        img_area = img_w * img_h

        for _ in range(self.n_holes):
            # Cutout領域の面積とアスペクト比をランダムに決定
            hole_area = img_area * random.uniform(self.scale[0], self.scale[1])
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            h = int(round(math.sqrt(hole_area / aspect_ratio)))
            w = int(round(math.sqrt(hole_area * aspect_ratio)))

            if h >= img_h or w >= img_w:
                continue

            # Cutout領域の左上の座標をランダムに決定
            x1 = random.randint(0, img_w - w)
            y1 = random.randint(0, img_h - h)
            x2, y2 = x1 + w, y1 + h

            draw.rectangle([x1, y1, x2, y2], fill='black')
            
        return image_copy
    

class PoseAugmentor:
    """
    骨格データに対するデータ拡張を実行するクラスである。
    コンストラクタでjoint_mapや各種拡張パラメータを設定する。
    """

    AUGMENTATION_LEVELS = {
        "Weak": {"rot": 5, "shear": 0.05, "parts": 1, "joints": 1, "noise": 1.0},
        "Mild": {"rot": 10, "shear": 0.1, "parts": 1, "joints": 1, "noise": 2.0},
        "Medium": {"rot": 15, "shear": 0.15, "parts": 2, "joints": 2, "noise": 3.0},
        "Strong": {"rot": 25, "shear": 0.3, "parts": 2, "joints": 2, "noise": 4.0},
    }

    def __init__(
        self,
        joint_map: dict,
        level: str = "Mild",
        augmentations_to_apply: list = None,
        rotation_range_deg: tuple = None,
        shear_range: tuple = None,
        max_parts_to_mask: int = None,
        max_joints_to_mask: int = None,
        noise_stddev: float = None,
    ):
        """
        コンストラクタ。

        Args:
            joint_map (dict): 関節の部位情報を定義した辞書。
            level (str, optional): 拡張の強度レベル。'Weak', 'Mild', 'Medium', 'Strong'から選択。デフォルトは'Mild'。
            augmentations_to_apply (list, optional): 適用する拡張手法のリスト。Noneの場合は全て適用。
            rotation_range_deg (tuple, optional): 回転角度の範囲（度）。level設定を上書きする。
            shear_range (tuple, optional): せん断の範囲。level設定を上書きする。
            max_parts_to_mask (int, optional): マスキングする最大部位数。level設定を上書きする。
            max_joints_to_mask (int, optional): マスキングする最大関節数。level設定を上書きする。
            noise_stddev (float, optional): 付加するノイズの標準偏差。level設定を上書きする。
        """
        self.joint_map = joint_map
        self.augmentations_to_apply = (
            augmentations_to_apply
            if augmentations_to_apply is not None
            else ["rotation_shear", "masking", "noise"]
        )

        if level not in self.AUGMENTATION_LEVELS:
            raise ValueError(
                f"Invalid level: {level}. Choose from {list(self.AUGMENTATION_LEVELS.keys())}"
            )

        params = self.AUGMENTATION_LEVELS[level]

        self.rotation_range_deg = (
            rotation_range_deg
            if rotation_range_deg is not None
            else (-params["rot"], params["rot"])
        )
        self.shear_range = (
            shear_range
            if shear_range is not None
            else (-params["shear"], params["shear"])
        )
        self.max_parts_to_mask = (
            max_parts_to_mask if max_parts_to_mask is not None else params["parts"]
        )
        self.max_joints_to_mask = (
            max_joints_to_mask if max_joints_to_mask is not None else params["joints"]
        )
        self.noise_stddev = (
            noise_stddev if noise_stddev is not None else params["noise"]
        )

    def __call__(self, skeleton: np.ndarray) -> np.ndarray:
        """
        インスタンスが関数として呼び出された際に、骨格データに一連の拡張を適用する。
        """
        augmented_skeleton = skeleton.copy()

        if "rotation_shear" in self.augmentations_to_apply:
            rotation_angle = random.uniform(*self.rotation_range_deg)
            shear_x = random.uniform(*self.shear_range)
            shear_y = random.uniform(*self.shear_range)
            augmented_skeleton = self._apply_rotation_shear(
                augmented_skeleton, rotation_angle, shear_x, shear_y
            )

        if "masking" in self.augmentations_to_apply:
            num_parts = random.randint(0, self.max_parts_to_mask)
            num_joints = random.randint(0, self.max_joints_to_mask)
            augmented_skeleton = self._apply_masking(
                augmented_skeleton, num_parts, num_joints
            )

        if "noise" in self.augmentations_to_apply and self.noise_stddev > 0:
            augmented_skeleton = self._add_noise(augmented_skeleton, self.noise_stddev)

        return augmented_skeleton

    def _apply_rotation_shear(
        self,
        skeleton: np.ndarray,
        rotation_angle_deg: float,
        shear_x: float,
        shear_y: float,
    ) -> np.ndarray:
        skeleton_coords = skeleton[:, :2]
        valid_mask = ~np.isnan(skeleton_coords).any(axis=1)
        if not np.any(valid_mask):
            return skeleton
        center = np.mean(skeleton_coords[valid_mask], axis=0)
        rotation_angle_rad = np.deg2rad(rotation_angle_deg)
        cos_val, sin_val = np.cos(rotation_angle_rad), np.sin(rotation_angle_rad)
        rotation_matrix = np.array([[cos_val, -sin_val], [sin_val, cos_val]])
        shear_matrix = np.array([[1, shear_x], [shear_y, 1]])
        transform_matrix = shear_matrix @ rotation_matrix
        transformed_skeleton = skeleton.copy()
        original_coords = skeleton_coords[valid_mask]
        transformed_coords = (original_coords - center) @ transform_matrix.T + center
        transformed_skeleton[valid_mask, :2] = transformed_coords
        return transformed_skeleton

    def _apply_masking(
        self, skeleton: np.ndarray, num_parts_to_mask: int, num_joints_to_mask: int
    ) -> np.ndarray:
        augmented_skeleton = skeleton.copy()
        num_dims = skeleton.shape[1]
        if num_parts_to_mask > 0:
            parts_to_mask = random.sample(
                list(self.joint_map.keys()),
                k=min(num_parts_to_mask, len(self.joint_map)),
            )
            for part in parts_to_mask:
                for joint_index in self.joint_map[part]:
                    augmented_skeleton[joint_index] = [0] * num_dims  # nanを0に変更
        if num_joints_to_mask > 0:
            available_indices = [
                i
                for i, joint in enumerate(augmented_skeleton)
                if not np.all(joint == 0)
            ]  # 0でない関節を対象にする
            if available_indices:
                joints_to_mask = random.sample(
                    available_indices, k=min(num_joints_to_mask, len(available_indices))
                )
                for joint_index in joints_to_mask:
                    augmented_skeleton[joint_index] = [0] * num_dims  # nanを0に変更
        return augmented_skeleton

    def _add_noise(self, skeleton: np.ndarray, noise_stddev: float) -> np.ndarray:
        noisy_skeleton = skeleton.copy()
        valid_data_mask = ~np.isnan(skeleton[:, :2])
        noise = np.random.normal(0, noise_stddev, skeleton.shape)
        noise[:, 2] = 0
        noisy_skeleton[:, :2][valid_data_mask] += noise[:, :2][valid_data_mask]
        return noisy_skeleton
