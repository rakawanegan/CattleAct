import os
import sys
import hydra
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, BatchSampler, Dataset
from torchmetrics import Accuracy, F1Score
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import umap
import wandb
import io
from PIL import Image, ImageDraw, UnidentifiedImageError
import random
import math


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# 以下のカスタムモジュールは、指定されたパスに存在する必要があります
from src.model import ShallowCNNforContext
from src.dataset import (
    CattleCroppedInteractionDataset,
    get_all_interaction_annotations_entries,
    split_interaction_dataset_entries,
)
from src.augmentation import ImageMaskingFromSkeletonForInteraction, StandardCutout
from src.loss_utils import InfoNCE


class LitHybridStreamFusionForMetricLearning(pl.LightningModule):
    """
    ハイブリッド・ストリーム構成をMetric Learningで学習するためのLightningModuleである。
    """

    def __init__(
        self,
        num_classes: int,
        map_label: dict,
        learning_rate: float,
        embedding_size: int,
        triplet_margin: float,
        fusion_type: str,
        vit_ckpt_path: str,
        freeze_vit: bool,
        pre_fusion_loss_weight: float = 0.5,
        pre_fusion_loss_type: str = "none",
    ):
        super().__init__()
        if fusion_type not in ["attention", "mlp"]:
            raise ValueError("fusion_type must be 'attention' or 'mlp'")
        if pre_fusion_loss_type not in ["triplet", "infonce", "none"]:
            raise ValueError("pre_fusion_loss_type must be 'triplet', 'infonce', or 'none'")
        self.save_hyperparameters()
        self.map_label = map_label
        self.backbone_vit = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0
        )
        if vit_ckpt_path and os.path.exists(vit_ckpt_path):
            self.load_vit_backbone_from_checkpoint(vit_ckpt_path)
        else:
            print(
                f"Warning: ViT checkpoint not found at {vit_ckpt_path}. Using random weights."
            )
        if self.hparams.freeze_vit:
            print("Freezing ViT backbone weights.")
            for param in self.backbone_vit.parameters():
                param.requires_grad = False
        num_ftrs_vit = self.backbone_vit.num_features
        self.backbone_cnn = ShallowCNNforContext(num_features_out=num_ftrs_vit)
        if self.hparams.fusion_type == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=num_ftrs_vit, num_heads=8, batch_first=True
            )
            self.embedding_generator = nn.Linear(
                num_ftrs_vit * 3, self.hparams.embedding_size
            )
        else:
            fused_feature_dim = num_ftrs_vit * 3
            self.embedding_generator = nn.Sequential(
                nn.Linear(fused_feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, self.hparams.embedding_size),
            )
        self.criterion = nn.TripletMarginLoss(margin=self.hparams.triplet_margin, p=2)
        self.infonce = InfoNCE(negative_mode='unpaired')
        metrics_kwargs = {"task": "multiclass", "num_classes": num_classes}
        self.val_knn_accuracy = Accuracy(**metrics_kwargs)
        self.val_f1score = F1Score(average="macro", **metrics_kwargs)
        self.test_knn_accuracy = Accuracy(**metrics_kwargs)
        self.test_f1score = F1Score(average="macro", **metrics_kwargs)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.val_data_for_test_umap = None
        self.train_embeddings = None
        self.train_labels = None

    def load_vit_backbone_from_checkpoint(self, ckpt_path: str):
        print(f"Loading ViT backbone weights from checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=self.device)["state_dict"]
        prefix = "model."
        backbone_state_dict = {
            k[len(prefix) :]: v
            for k, v in state_dict.items()
            if k.startswith(prefix) and not k.startswith(prefix + "head")
        }
        if not backbone_state_dict:
            raise ValueError(
                f"No backbone weights with prefix '{prefix}' found in checkpoint."
            )
        self.backbone_vit.load_state_dict(backbone_state_dict, strict=False)

    def forward(self, images1, images2, images_context):
        # 1. 特徴抽出
        features1 = self.backbone_vit(images1)
        features2 = self.backbone_vit(images2)
        features_context = self.backbone_cnn(images_context)

        # 2. 特徴融合
        if self.hparams.fusion_type == "attention":
            stacked_features = torch.stack(
                (features1, features2, features_context), dim=1
            )
            attn_output, _ = self.attention(
                stacked_features, stacked_features, stacked_features
            )
            combined_features = attn_output.flatten(start_dim=1)
        else:
            combined_features = torch.cat(
                (features1, features2, features_context), dim=1
            )
        
        # 3. 埋め込みベクトル生成
        embedding = self.embedding_generator(combined_features)

        # training時のみ、融合前の特徴量も返す
        if self.training:
            return embedding, (features1, features2, features_context)
        else:
            return embedding

    def _get_all_valid_triplets(self, embeddings, labels):
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        n = labels.size(0)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        identity = torch.eye(n, dtype=torch.bool, device=self.device)
        mask_positive = labels_eq & ~identity
        mask_negative = ~labels_eq
        if not torch.any(mask_positive) or not torch.any(mask_negative):
            return None, None, None
        anchor_positive_dist = dist_matrix.unsqueeze(2)
        anchor_negative_dist = dist_matrix.unsqueeze(1)
        triplet_loss = (
            anchor_positive_dist - anchor_negative_dist + self.hparams.triplet_margin
        )
        mask_valid_triplets = mask_positive.unsqueeze(2) & mask_negative.unsqueeze(1)
        mask_loss = (triplet_loss > 0) & mask_valid_triplets
        if not mask_loss.any():
            return None, None, None
        anchor_idx, positive_idx, negative_idx = torch.where(mask_loss)
        return (
            embeddings[anchor_idx],
            embeddings[positive_idx],
            embeddings[negative_idx],
        )

    def training_step(self, batch, batch_idx):
        images1, images2, images_context, labels, _ = batch
        
        # 融合後の埋め込みと融合前の特徴量を受け取る
        raw_embeddings, pre_fusion_features = self(images1, images2, images_context)
        features1, features2, features_context = pre_fusion_features

        # ===== 1. 既存の損失計算 (融合後の埋め込みベクトルを使用) =====
        embeddings = F.normalize(raw_embeddings, p=2, dim=1)
        
        # メモリ節約のためCPUに移動して保存
        self.training_step_outputs.append(
            {"embeddings": embeddings.cpu(), "labels": labels.cpu()}
        )
        
        anchor, positive, negative = self._get_all_valid_triplets(embeddings, labels)
        
        if anchor is None:
            main_loss = torch.tensor(0.0, device=self.device)
        else:
            main_loss = self.criterion(anchor, positive, negative)
        
        self.log("train_loss", main_loss, on_step=False, on_epoch=True, sync_dist=True)

        # ===== 2. 追加の損失 (融合前の特徴量を使用) =====
        pre_fusion_loss = torch.tensor(0.0, device=self.device)
        
        if self.hparams.pre_fusion_loss_type != "none":
            interaction_mask = labels != 0
            no_interaction_mask = labels == 0

            if torch.any(interaction_mask) and torch.any(no_interaction_mask):
                interaction_indices = torch.where(interaction_mask)[0]
                no_interaction_indices = torch.where(no_interaction_mask)[0]

                anchors, positives, negatives = [], [], []
                
                for idx in interaction_indices:
                    anchor_feat = features_context[idx]
                    positive_feat1 = features1[idx]
                    positive_feat2 = features2[idx]
                    neg_idx = random.choice(no_interaction_indices)
                    negative_feat = features_context[neg_idx]

                    anchors.extend([anchor_feat, anchor_feat])
                    positives.extend([positive_feat1, positive_feat2])
                    negatives.extend([negative_feat, negative_feat])
                
                if anchors:
                    anchor_embs = torch.stack(anchors)
                    positive_embs = torch.stack(positives)
                    negative_embs = torch.stack(negatives)
                    if self.hparams.pre_fusion_loss_type == "triplet":
                        pre_fusion_loss = self.criterion(anchor_embs, positive_embs, negative_embs)
                    elif self.hparams.pre_fusion_loss_type == "infonce":
                        pre_fusion_loss = self.infonce(anchor_embs, positive_embs, negative_embs)
        
        self.log("train_pre_fusion_loss", pre_fusion_loss, on_step=False, on_epoch=True, sync_dist=True)

        # ===== 3. 最終的な損失の計算 =====
        total_loss = main_loss + self.hparams.pre_fusion_loss_weight * pre_fusion_loss
        
        self.log(
            "train_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return total_loss

    def on_train_epoch_end(self):
        """学習エポック終了時に、そのエポックで収集したすべての学習データの埋め込みとラベルを結合して保存する"""
        if not self.training_step_outputs:
            return
        
        # 全てのバッチの埋め込みとラベルを結合
        embeddings = torch.cat(
            [x["embeddings"] for x in self.training_step_outputs], dim=0
        )
        labels = torch.cat([x["labels"] for x in self.training_step_outputs], dim=0)
        
        # モデルの属性として保存
        self.train_embeddings = embeddings.to(self.device) # 評価時にGPUで計算するため転送
        self.train_labels = labels.to(self.device)
        
        # メモリを解放
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        images1, images2, images_context, labels, _ = batch
        raw_embeddings = self(images1, images2, images_context)
        embeddings = F.normalize(raw_embeddings, p=2, dim=1)
        self.validation_step_outputs.append(
            {"embeddings": embeddings.cpu(), "labels": labels.cpu()}
        )

    def on_validation_epoch_start(self):
        """
        検証エポックの開始時に、監視対象のメトリックを初期値でログに記録する。
        これにより、古いバージョンのPytorch Lightningで発生するModelCheckpointのエラーを回避する。
        """
        self.log("val_f1score", 0.0, sync_dist=True)

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs or self.train_embeddings is None:
            return
            
        val_embeddings = torch.cat(
            [x["embeddings"] for x in self.validation_step_outputs], dim=0
        ).to(self.device)
        val_labels = torch.cat([x["labels"] for x in self.validation_step_outputs], dim=0).to(self.device)
        
        # 検証データの埋め込みと学習データ全体の埋め込みとの距離を計算
        dist_matrix = torch.cdist(val_embeddings, self.train_embeddings, p=2)
        
        # 各検証サンプルについて、最も距離が近い学習サンプルのインデックスを取得
        closest_indices = torch.argmin(dist_matrix, dim=1)
        
        # 最も近い学習サンプルのラベルを予測ラベルとする
        preds = self.train_labels[closest_indices]
        
        # 精度とF1スコアを計算
        self.val_knn_accuracy.update(preds, val_labels)
        self.val_f1score.update(preds, val_labels)
        
        self.log(
            "val_knn_acc", self.val_knn_accuracy.compute(), on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            "val_f1score", self.val_f1score.compute(), on_epoch=True, prog_bar=True, sync_dist=True
        )
        
        if self.current_epoch > 0 and self.current_epoch % 10 == 0:
            self._plot_umap(stage="val")
            
        self.val_data_for_test_umap = {"embeddings": val_embeddings.cpu(), "labels": val_labels.cpu()}
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        images1, images2, images_context, labels, _ = batch
        raw_embeddings = self(images1, images2, images_context)
        embeddings = F.normalize(raw_embeddings, p=2, dim=1)
        self.test_step_outputs.append(
            {"embeddings": embeddings.cpu(), "labels": labels.cpu()}
        )

    def on_test_epoch_end(self):
        if not self.test_step_outputs or self.train_embeddings is None:
            return
            
        test_embeddings = torch.cat(
            [x["embeddings"] for x in self.test_step_outputs], dim=0
        ).to(self.device)
        test_labels = torch.cat([x["labels"] for x in self.test_step_outputs], dim=0).to(self.device)
        
        # テストデータの埋め込みと学習データ全体の埋め込みとの距離を計算
        dist_matrix = torch.cdist(test_embeddings, self.train_embeddings, p=2)
        
        # 各テストサンプルについて、最も距離が近い学習サンプルのインデックスを取得
        closest_indices = torch.argmin(dist_matrix, dim=1)
        
        # 最も近い学習サンプルのラベルを予測ラベルとする
        preds = self.train_labels[closest_indices]

        # 精度とF1スコアを計算
        self.test_knn_accuracy.update(preds, test_labels)
        self.test_f1score.update(preds, test_labels)
        
        self.log(
            "test_knn_acc", self.test_knn_accuracy.compute(), on_epoch=True, sync_dist=True
        )
        self.log(
            "test_f1score", self.test_f1score.compute(), on_epoch=True, sync_dist=True
        )
        
        if hasattr(self, "val_data_for_test_umap"):
            self._plot_umap(stage="test", val_data=self.val_data_for_test_umap)
        else:
            self._plot_umap(stage="test")
            
        self.test_step_outputs.clear()

    def _plot_umap(self, stage: str, val_data: dict = None):
        outputs = (
            self.validation_step_outputs if stage == "val" else self.test_step_outputs
        )
        if not outputs and not (stage == 'test' and val_data is not None):
            print(f"Skipping UMAP for stage '{stage}' due to no outputs collected.")
            return
        
        if stage == 'test' and val_data is not None:
            test_embeddings = torch.cat([x["embeddings"] for x in self.test_step_outputs], dim=0).numpy()
            test_labels = torch.cat([x["labels"] for x in self.test_step_outputs], dim=0).numpy()
            val_embeddings = val_data["embeddings"].numpy()
            val_labels = val_data["labels"].numpy()
            combined_embeddings = np.vstack([val_embeddings, test_embeddings])
            combined_labels = np.concatenate([val_labels, test_labels])
            num_val_samples = len(val_embeddings)
        else:
            combined_embeddings = torch.cat([x["embeddings"] for x in self.validation_step_outputs], dim=0).numpy()
            combined_labels = torch.cat([x["labels"] for x in self.validation_step_outputs], dim=0).numpy()
            num_val_samples = 0

        if len(combined_embeddings) < 2:
            print(f"Skipping UMAP for stage '{stage}' due to insufficient samples ({len(combined_embeddings)}).")
            return
        
        n_neighbors = min(15, len(combined_embeddings) - 1)
        if n_neighbors <= 0:
            print(f"Skipping UMAP for stage '{stage}' due to n_neighbors <= 0.")
            return

        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42
        )
        umap_results = reducer.fit_transform(combined_embeddings)
        
        fig, ax = plt.subplots(figsize=(15, 15))
        inv_map_label = {v: k for k, v in self.map_label.items()}
        unique_labels = sorted(np.unique(combined_labels))
        cmap = plt.get_cmap("viridis", len(unique_labels))
        colors = {label: cmap(i) for i, label in enumerate(unique_labels)}
        
        handles = []
        labels_for_legend = []

        for label_val in unique_labels:
            label_name = inv_map_label.get(label_val, f"Label {label_val}")
            
            if stage == 'test' and val_data is not None:
                mask_val = (combined_labels == label_val) & (np.arange(len(combined_labels)) < num_val_samples)
                if np.any(mask_val):
                    ax.scatter(
                        umap_results[mask_val, 0], umap_results[mask_val, 1],
                        color=colors[label_val],
                        alpha=0.4, s=50, marker='o'
                    )
                
                mask_test = (combined_labels == label_val) & (np.arange(len(combined_labels)) >= num_val_samples)
                if np.any(mask_test):
                    ax.scatter(
                        umap_results[mask_test, 0], umap_results[mask_test, 1],
                        color=colors[label_val],
                        alpha=0.9, s=80, marker='X'
                    )
                
                if np.any(mask_val) and f"{label_name} (Val)" not in labels_for_legend:
                    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label_val], markersize=10, alpha=0.6))
                    labels_for_legend.append(f"{label_name} (Val)")
                if np.any(mask_test) and f"{label_name} (Test)" not in labels_for_legend:
                    handles.append(plt.Line2D([0], [0], marker='X', color='w', markerfacecolor=colors[label_val], markersize=10, alpha=0.9))
                    labels_for_legend.append(f"{label_name} (Test)")

            else:
                mask = combined_labels == label_val
                if np.any(mask):
                    ax.scatter(
                        umap_results[mask, 0], umap_results[mask, 1],
                        color=colors[label_val],
                        alpha=0.7, s=50
                    )
                    if label_name not in labels_for_legend:
                        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label_val], markersize=10, alpha=0.7))
                        labels_for_legend.append(label_name)

        ax.legend(handles, labels_for_legend, title="Interaction", fontsize=12, title_fontsize=14)
            
        title = f"UMAP Visualization of Embeddings ({stage.capitalize()})"
        if stage == 'test' and val_data is not None:
            title += " with Validation Data"
        ax.set_title(title, fontsize=18)
        fig.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        im = Image.open(buf)
        
        log_key = f"{stage}_umap_combined" if stage == 'test' and val_data is not None else (f"{stage}_umap_epoch_{self.current_epoch}" if stage=='val' else f"{stage}_umap")

        if isinstance(self.logger, WandbLogger) and self.logger.experiment:
            self.logger.experiment.log({log_key: wandb.Image(im)})
        plt.close(fig)
        buf.close()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1score"},
        }


class BalancedBatchSampler(BatchSampler):
    """
    クラスごとに均等な数のサンプルを抽出するバッチサンプラーである。
    """
    def __init__(self, labels, n_classes, n_samples, sampler=None):
        if isinstance(labels, BalancedBatchSampler):
            original_sampler = labels
            self.labels = original_sampler.labels
            self.n_classes = original_sampler.n_classes
            self.n_samples = original_sampler.n_samples
        elif sampler is not None and isinstance(sampler, BalancedBatchSampler):
            original_sampler = sampler
            self.labels = original_sampler.labels
            self.n_classes = original_sampler.n_classes
            self.n_samples = original_sampler.n_samples
        else:
            self.labels = np.array(labels)
            self.n_classes = n_classes
            self.n_samples = n_samples
        if self.labels is None or self.n_classes is None or self.n_samples is None:
            raise ValueError("Invalid arguments for BalancedBatchSampler.")
        super().__init__(
            sampler=None, batch_size=(self.n_classes * self.n_samples), drop_last=False
        )
        self.n_batches = len(self.labels) // (self.n_classes * self.n_samples)
        if self.n_batches == 0:
            raise ValueError("Not enough samples for one batch.")
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[int(label)].append(idx)
        for label, indices in self.label_to_indices.items():
            if len(indices) < self.n_samples:
                print(f"Warning: Class {label} has {len(indices)} samples, need {self.n_samples}. Repeating.")

    def __iter__(self):
        for _ in range(self.n_batches):
            available_classes = list(self.label_to_indices.keys())
            if len(available_classes) < self.n_classes:
                raise ValueError(f"Need {self.n_classes} classes, but only {len(available_classes)} available.")
            batch_classes = np.random.choice(
                available_classes, self.n_classes, replace=False
            )
            batch_indices = []
            for class_label in batch_classes:
                indices_for_class = self.label_to_indices[class_label]
                replace = len(indices_for_class) < self.n_samples
                class_indices = np.random.choice(
                    indices_for_class, self.n_samples, replace=replace
                )
                batch_indices.extend(class_indices)
            yield batch_indices

    def __len__(self):
        return self.n_batches


class CattleInteractionDataModule(pl.LightningDataModule):
    """相互作用データセット用のDataModuleである。BalancedBatchSamplerとカスタム拡張を使用する。"""

    def __init__(self, data_cfg: DictConfig, aug_cfg: DictConfig):
        super().__init__()
        self.data_cfg = data_cfg
        self.aug_cfg = aug_cfg
        image_size = self.data_cfg.get("image_size", 224)
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.aug_cfg.masking_from_skeleton.use:
            joint_map = {
                "head": [0, 1, 2],
                "neck": [3],
                "torso": [4],
                "left_front_leg": [5, 6, 7],
                "right_front_leg": [8, 9, 10],
                "left_hind_leg": [11, 12, 13],
                "right_hind_leg": [14, 15, 16],
            }
            self.skeleton_aware_transform = ImageMaskingFromSkeletonForInteraction(
                joint_map=joint_map,
                cutout_prob=self.aug_cfg.masking_from_skeleton.cutout_prob,
                n_holes=self.aug_cfg.masking_from_skeleton.n_holes,
                scale=tuple(self.aug_cfg.masking_from_skeleton.scale),
                ratio=tuple(self.aug_cfg.masking_from_skeleton.ratio),
            )
        else:
            self.skeleton_aware_transform = StandardCutout(
                cutout_prob=self.aug_cfg.masking_from_skeleton.cutout_prob,
                n_holes=self.aug_cfg.masking_from_skeleton.n_holes,
                scale=tuple(self.aug_cfg.masking_from_skeleton.scale),
                ratio=tuple(self.aug_cfg.masking_from_skeleton.ratio),
            )

        self.transform_train = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.RandAugment(num_ops=2, magnitude=12),
                T.ToTensor(),
                normalize,
            ]
        )
        self.transform_val = T.Compose(
            [T.Resize((image_size, image_size)), T.ToTensor(), normalize]
        )
        self.train_batch_sampler = None

    def setup(self, stage=None):
        entries_args = {
            "root_dir": self.data_cfg.root_dir,
            "map_label": dict(self.data_cfg.map_label),
            "delete_base_dirs": list(self.data_cfg.get("delete_base_dirs", [])),
            "use_more_than_three_cattles": self.data_cfg.get(
                "use_more_than_three_cattles", False
            ),
        }
        full_dataset_entries = get_all_interaction_annotations_entries(**entries_args)
        train_entries, val_entries, test_entries = split_interaction_dataset_entries(
            full_dataset_entries
        )

        self.train_dataset = CattleCroppedInteractionDataset(
            entries=train_entries,
            transform=self.transform_train,
            skeleton_aware_transform=self.skeleton_aware_transform,
        )
        self.val_dataset = CattleCroppedInteractionDataset(
            entries=val_entries, transform=self.transform_val
        )
        self.test_dataset = CattleCroppedInteractionDataset(
            entries=test_entries, transform=self.transform_val
        )

        if stage in ("fit", None):
            if hasattr(self.train_dataset, "entries") and self.train_dataset.entries:
                labels = [entry["label"] for entry in self.train_dataset.entries]
                self.train_batch_sampler = BalancedBatchSampler(
                    labels=labels,
                    n_classes=self.data_cfg.classes_per_batch,
                    n_samples=self.data_cfg.samples_per_class,
                )

    def train_dataloader(self):
        if self.train_batch_sampler is None:
            raise ValueError(
                "Train dataset is empty or sampler not initialized."
            )
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        batch_size = self.data_cfg.classes_per_batch * self.data_cfg.samples_per_class
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        batch_size = self.data_cfg.classes_per_batch * self.data_cfg.samples_per_class
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
        )


@hydra.main(version_base=None, config_path="conf", config_name="interaction_train")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    data_module = CattleInteractionDataModule(data_cfg=cfg.data, aug_cfg=cfg.augmentation)
    num_classes = len(cfg.data.map_label)
    model = LitHybridStreamFusionForMetricLearning(
        num_classes=num_classes,
        map_label=dict(cfg.data.map_label),
        learning_rate=cfg.training.learning_rate,
        embedding_size=cfg.model.embedding_size,
        triplet_margin=cfg.model.triplet_margin,
        fusion_type=cfg.model.fusion_type,
        vit_ckpt_path="checkpoints/action_metric_learning.ckpt",
        freeze_vit=cfg.training.freeze_backbone,
        pre_fusion_loss_weight=cfg.model.pre_fusion_loss_weight,
        pre_fusion_loss_type=cfg.model.get("pre_fusion_loss_type", "none"),
    )

    suffix = ""
    use_skeleton_aug = cfg.augmentation.masking_from_skeleton.use
    if use_skeleton_aug:
        suffix += "_aug_pose"

    pre_fusion_loss_type = cfg.model.get("pre_fusion_loss_type", "none")
    if pre_fusion_loss_type and pre_fusion_loss_type != "none":
        suffix += f"_pfl_{pre_fusion_loss_type}"

    if cfg.training.max_epochs > 2:
        logger = WandbLogger(
            name=cfg.wandb.name,
            project="wacv_cattle_activity_recognition_interaction",
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            group="interaction_metric_learning" + suffix,
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_f1score",
            mode="max",
            save_top_k=1,
            dirpath="checkpoints_dev",
            filename="metric-interaction-{epoch:02d}" + suffix,
        )
        callbacks = [checkpoint_callback]
    else:
        # DEBUG MODE
        print(f"--- Running in debug mode (max_epochs={cfg.training.max_epochs}). Disabling logger and checkpoints. ---")
        logger = False
        callbacks = None
        print(f"[DEBUG] Generated suffix: '{suffix}'")

    devices = (
        OmegaConf.to_container(cfg.training.devices, resolve=True)
        if cfg.training.devices
        else "auto"
    )
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=devices,
        logger=logger,
        callbacks=callbacks,
        use_distributed_sampler=False,
    )

    print("--- Starting Metric Learning for Interaction Classification ---")
    trainer.fit(model, datamodule=data_module)
    print("--- Testing the best model ---")
    trainer.test(datamodule=data_module, ckpt_path="best")
    print("--- Metric Learning finished. ---")


if __name__ == "__main__":
    main()