import os
import sys
from typing import Optional

import hydra
import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import Accuracy, F1Score, MetricCollection

# 親ディレクトリをシステムパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import (
    CattleCroppedInteractionDataset,
    get_all_interaction_annotations_entries,
    split_interaction_dataset_entries,
)
from src.augmentation import ImageMaskingFromSkeletonForInteraction, StandardCutout
from src.loss_utils import InfoNCE, LDAMLoss, FocalLoss


class ShallowCNNforContext(nn.Module):
    """
    コンテキスト画像用の浅いCNN特徴量抽出器である。
    """

    def __init__(self, in_channels: int = 3, num_features_out: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=32, stride=4, padding=14),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, num_features_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features_out),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.num_features = num_features_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x


class LitHybridStreamFusion(pl.LightningModule):
    """ハイブリッド・ストリーム構成で相互作用分類を学習するLightningModule。"""

    def __init__(
        self,
        num_classes: int,
        fusion_type: str,
        learning_rate: float,
        vit_ckpt_path: Optional[str],
        freeze_vit: bool,
        cls_num_list: list,
        main_loss_cfg: DictConfig,
        pre_fusion_loss_cfg: DictConfig,
        pooling_type: str,
        pre_fusion_loss_weight: float,
    ):
        super().__init__()

        valid_fusion_types = {"attention", "mlp"}
        if fusion_type not in valid_fusion_types:
            raise ValueError("fusion_type must be 'attention' or 'mlp'")

        pre_fusion_loss_name = (
            pre_fusion_loss_cfg.name if pre_fusion_loss_cfg is not None else "none"
        )
        valid_pre_fusion_losses = {"triplet", "infonce", "none"}
        if pre_fusion_loss_name not in valid_pre_fusion_losses:
            raise ValueError(
                "pre_fusion_loss.name must be 'triplet', 'infonce', or 'none'"
            )

        valid_pooling_types = {"flatten", "gmp", "gap", "gap_gmp"}
        if pooling_type not in valid_pooling_types:
            raise ValueError(
                "pooling_type must be one of 'flatten', 'gmp', 'gap', or 'gap_gmp'"
            )

        if pre_fusion_loss_name == "none" and pre_fusion_loss_weight != 0.0:
            raise ValueError(
                "pre_fusion_loss_weight must be 0.0 when pre_fusion_loss.name is 'none'"
            )
        if pre_fusion_loss_name != "none" and pre_fusion_loss_weight <= 0.0:
            raise ValueError(
                "pre_fusion_loss_weight must be positive when pre_fusion_loss.name is not 'none'"
            )

        self.pre_fusion_loss_name = pre_fusion_loss_name
        self.pre_fusion_loss_weight = float(pre_fusion_loss_weight)

        self.save_hyperparameters(ignore=["main_loss_cfg", "pre_fusion_loss_cfg"])

        self.backbone_vit = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0
        )

        is_load_pretrained = bool(vit_ckpt_path) and os.path.exists(vit_ckpt_path)
        if is_load_pretrained:
            self.load_vit_backbone_from_checkpoint(vit_ckpt_path)
        elif vit_ckpt_path:
            print(
                f"Warning: ViT checkpoint path provided but not found at {vit_ckpt_path}."
            )
        else:
            print("Info: No ViT checkpoint path provided. Using random weights.")

        if self.hparams.freeze_vit and is_load_pretrained:
            print("Freezing ViT backbone weights.")
            for param in self.backbone_vit.parameters():
                param.requires_grad = False

        num_ftrs_vit = self.backbone_vit.num_features
        projection_dim = num_ftrs_vit

        self.vit_feature_projector = nn.Sequential(
            nn.Linear(num_ftrs_vit, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.cnn_feature_projector = nn.Sequential(
            nn.Linear(num_ftrs_vit, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.context_infonce_projector = nn.Sequential(
            nn.Linear(num_ftrs_vit, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.backbone_cnn = ShallowCNNforContext(num_features_out=num_ftrs_vit)

        if self.hparams.fusion_type == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=projection_dim, num_heads=8, batch_first=True
            )

            if self.hparams.pooling_type == "flatten":
                fusion_out_dim = projection_dim * 3
            elif self.hparams.pooling_type == "gap_gmp":
                fusion_out_dim = projection_dim * 2
            else:
                fusion_out_dim = projection_dim

            self.classifier = nn.Linear(fusion_out_dim, num_classes)
        else:
            fused_feature_dim = projection_dim * 3
            self.classifier = nn.Sequential(
                nn.Linear(fused_feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )

        if main_loss_cfg.name == "focal":
            if "gamma" not in main_loss_cfg:
                raise ValueError("main_loss.focal.gamma must be defined in the configuration.")
            gamma = float(main_loss_cfg.gamma)
            alpha = main_loss_cfg.alpha if "alpha" in main_loss_cfg else None
            self.classification_criterion = FocalLoss(
                gamma=gamma,
                alpha=alpha,
            )
        elif main_loss_cfg.name == "ldam":
            if not self.hparams.cls_num_list:
                raise ValueError("cls_num_list must be provided when using LDAMLoss.")
            if "max_m" not in main_loss_cfg or "s" not in main_loss_cfg:
                raise ValueError("main_loss.ldam must define both max_m and s in the configuration.")
            self.classification_criterion = LDAMLoss(
                cls_num_list=self.hparams.cls_num_list,
                max_m=float(main_loss_cfg.max_m),
                s=float(main_loss_cfg.s),
            )
        elif main_loss_cfg.name == "cross_entropy":
            self.classification_criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported main_loss: {main_loss_cfg.name}")

        if self.pre_fusion_loss_name == "triplet":
            if "margin" not in pre_fusion_loss_cfg:
                raise ValueError("pre_fusion_loss.triplet.margin must be defined in the configuration.")
            margin = float(pre_fusion_loss_cfg.margin)
            self.pre_fusion_criterion = nn.TripletMarginLoss(margin=margin)
        elif self.pre_fusion_loss_name == "infonce":
            if "temperature" not in pre_fusion_loss_cfg:
                raise ValueError("pre_fusion_loss.infonce.temperature must be defined in the configuration.")
            temperature = float(pre_fusion_loss_cfg.temperature)
            self.pre_fusion_criterion = InfoNCE(
                negative_mode="unpaired", temperature=temperature
            )
        else:
            self.pre_fusion_criterion = None

        metrics = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", num_classes=num_classes),
                "f1score": F1Score(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def load_vit_backbone_from_checkpoint(self, ckpt_path: str) -> None:
        print(f"Loading ViT backbone weights from checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        prefix = "model."
        backbone_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix) and not k.startswith(prefix + "head"):
                new_key = k[len(prefix) :]
                backbone_state_dict[new_key] = v
        if not backbone_state_dict:
            raise ValueError(
                f"No backbone weights with prefix '{prefix}' found in checkpoint."
            )
        self.backbone_vit.load_state_dict(backbone_state_dict, strict=False)

    def forward(self, images1, images2, images_context):
        features1 = self.backbone_vit(images1)
        features2 = self.backbone_vit(images2)
        raw_context = self.backbone_cnn(images_context)

        features1 = self.vit_feature_projector(features1)
        features2 = self.vit_feature_projector(features2)
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)

        features_context_fusion = self.cnn_feature_projector(raw_context)
        features_context_fusion = F.normalize(features_context_fusion, p=2, dim=1)

        features_context_infonce = self.context_infonce_projector(raw_context)
        features_context_infonce = F.normalize(features_context_infonce, p=2, dim=1)

        if self.hparams.fusion_type == "attention":
            stacked_features = torch.stack(
                (features1, features2, features_context_fusion), dim=1
            )
            attn_output, _ = self.attention(
                stacked_features, stacked_features, stacked_features
            )

            if self.hparams.pooling_type == "flatten":
                combined_features = attn_output.flatten(start_dim=1)
            elif self.hparams.pooling_type == "gmp":
                combined_features, _ = torch.max(attn_output, dim=1)
            elif self.hparams.pooling_type == "gap":
                combined_features = torch.mean(attn_output, dim=1)
            else:  # gap_gmp
                avg_pool = torch.mean(attn_output, dim=1)
                max_pool, _ = torch.max(attn_output, dim=1)
                combined_features = torch.cat((avg_pool, max_pool), dim=1)
        else:
            combined_features = torch.cat(
                (features1, features2, features_context_fusion), dim=1
            )

        logits = self.classifier(combined_features)
        return logits, (features1, features2, features_context_fusion), features_context_infonce

    def _compute_pre_fusion_loss(
        self,
        labels: torch.Tensor,
        features1: torch.Tensor,
        features2: torch.Tensor,
        features_context_infonce: torch.Tensor,
    ) -> torch.Tensor:
        if self.pre_fusion_loss_name == "none" or self.pre_fusion_criterion is None:
            return features1.new_zeros(())

        interaction_mask = labels != 0
        no_interaction_mask = labels == 0

        if not torch.any(interaction_mask) or not torch.any(no_interaction_mask):
            return features1.new_zeros(())

        interaction_indices = torch.where(interaction_mask)[0]
        no_interaction_indices = torch.where(no_interaction_mask)[0]

        anchors = []
        positives = []
        negatives = []

        for idx_tensor in interaction_indices:
            idx = idx_tensor.item()
            anchor_feat = features_context_infonce[idx]
            positive_feat1 = features1[idx]
            positive_feat2 = features2[idx]

            rand_idx = torch.randint(
                low=0,
                high=len(no_interaction_indices),
                size=(1,),
                device=labels.device,
            ).item()
            negative_index = no_interaction_indices[rand_idx]
            negative_feat = features_context_infonce[negative_index]

            anchors.extend([anchor_feat, anchor_feat])
            positives.extend([positive_feat1, positive_feat2])
            negatives.extend([negative_feat, negative_feat])

        if not anchors:
            return features1.new_zeros(())

        anchor_embs = torch.stack(anchors)
        positive_embs = torch.stack(positives)
        negative_embs = torch.stack(negatives)
        return self.pre_fusion_criterion(anchor_embs, positive_embs, negative_embs)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        images1, images2, images_context, labels, _ = batch
        logits, pre_fusion_features, features_context_infonce = self(
            images1, images2, images_context
        )
        features1, features2, _ = pre_fusion_features

        main_loss = self.classification_criterion(logits, labels)
        pre_fusion_loss = self._compute_pre_fusion_loss(
            labels, features1, features2, features_context_infonce
        )

        weight = self.pre_fusion_loss_weight if self.pre_fusion_loss_name != "none" else 0.0
        total_loss = main_loss + weight * pre_fusion_loss

        metrics = getattr(self, f"{stage}_metrics", None)
        if metrics is not None:
            metrics.update(logits, labels)

        on_step = stage == "train"
        prog_bar = stage in {"train", "val"}
        total_loss_name = "train_total_loss" if stage == "train" else f"{stage}_loss"

        self.log(
            total_loss_name,
            total_loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=prog_bar,
            sync_dist=True,
        )
        self.log(
            f"{stage}_main_loss",
            main_loss,
            on_step=on_step,
            on_epoch=True,
            sync_dist=True,
        )
        if self.pre_fusion_loss_name != "none":
            self.log(
                f"{stage}_pre_fusion_loss",
                pre_fusion_loss,
                on_step=on_step,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_pre_fusion_weight",
                torch.tensor(weight, device=self.device),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1score",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class CattleInteractionDataModule(pl.LightningDataModule):
    """相互作用データセット用のDataModuleである。"""

    def __init__(self, data_cfg: DictConfig, aug_cfg: DictConfig):
        super().__init__()
        self.data_cfg = data_cfg
        self.aug_cfg = aug_cfg
        self.cls_num_list = None
        self.train_sampler_weights = None
        image_size = self.data_cfg.image_size
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
                margin=self.aug_cfg.masking_from_skeleton.margin,
            )
            print("Using skeleton-aware augmentation.")
        else:
            self.skeleton_aware_transform = StandardCutout(
                cutout_prob=self.aug_cfg.masking_from_skeleton.cutout_prob,
                n_holes=self.aug_cfg.masking_from_skeleton.n_holes,
                scale=tuple(self.aug_cfg.masking_from_skeleton.scale),
                ratio=tuple(self.aug_cfg.masking_from_skeleton.ratio),
            )
            print("Using standard cutout augmentation.")

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

    def setup(self, stage=None):
        entries_args = {
            "root_dir": self.data_cfg.root_dir,
            "map_label": dict(self.data_cfg.map_label),
            "delete_base_dirs": list(self.data_cfg.delete_base_dirs),
            "use_more_than_three_cattles": self.data_cfg.use_more_than_three_cattles,
        }
        full_dataset_entries = get_all_interaction_annotations_entries(**entries_args)
        train_entries, val_entries, test_entries = split_interaction_dataset_entries(
            full_dataset_entries
        )

        self.train_dataset = CattleCroppedInteractionDataset(
            entries=train_entries,
            transform=self.transform_train,
            skeleton_aware_transform=self.skeleton_aware_transform,
            is_aware_skeleton=self.aug_cfg.masking_from_skeleton.use,
        )
        self.val_dataset = CattleCroppedInteractionDataset(
            entries=val_entries, transform=self.transform_val
        )
        self.test_dataset = CattleCroppedInteractionDataset(
            entries=test_entries, transform=self.transform_val
        )

        if self.cls_num_list is None:
            num_classes = len(self.data_cfg.map_label)
            cls_counts = torch.zeros(num_classes, dtype=torch.float)
            train_labels = [e["label"] for e in train_entries]
            for label in train_labels:
                cls_counts[label] += 1
            
            class_weights = 1.0 / torch.where(cls_counts > 0, cls_counts, torch.tensor(float('inf')))
            self.train_sampler_weights = torch.tensor([class_weights[label] for label in train_labels])
            self.cls_num_list = [int(count) for count in cls_counts]
            print(f"Calculated class counts for LDAMLoss: {self.cls_num_list}")

    def train_dataloader(self):
        sampler = WeightedRandomSampler(
            weights=self.train_sampler_weights,
            num_samples=len(self.train_sampler_weights),
            replacement=True
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.data_cfg.batch_size,
            sampler=sampler,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_cfg.batch_size,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_cfg.batch_size,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
        )


@hydra.main(version_base=None, config_path="conf", config_name="interaction_train")
def main(cfg: DictConfig) -> None:
    """学習のメインプロセスを実行する。"""
    pl.seed_everything(cfg.seed, workers=True)

    data_module = CattleInteractionDataModule(
        data_cfg=cfg.data, aug_cfg=cfg.augmentation
    )
    data_module.setup(stage="fit")

    num_classes = len(cfg.data.map_label)
    if "pooling_type" not in cfg.model:
        raise ValueError("cfg.model.pooling_type must be defined in the configuration.")
    pooling_type = str(cfg.model.pooling_type)

    if "weight" not in cfg.pre_fusion_loss:
        raise ValueError("cfg.pre_fusion_loss.weight must be defined in the configuration.")
    pre_fusion_loss_weight = float(cfg.pre_fusion_loss.weight)
    
    suffix = ""
    if cfg.training.pretrained_backbone:
        suffix += "_pretrained"
    suffix += f"_{cfg.model.fusion_type}"
    if cfg.augmentation.masking_from_skeleton.use:
        suffix += "_aug_pose"
    if cfg.pre_fusion_loss.name != "none":
        suffix += f"_pfl_{cfg.pre_fusion_loss.name}"

    model = LitHybridStreamFusion(
        num_classes=num_classes,
        learning_rate=cfg.training.learning_rate,
        vit_ckpt_path="checkpoints/action_metric_learning.ckpt" if cfg.training.pretrained_backbone else None,
        freeze_vit=cfg.training.freeze_backbone,
        fusion_type=cfg.model.fusion_type,
        cls_num_list=data_module.cls_num_list,
        main_loss_cfg=cfg.main_loss,
        pre_fusion_loss_cfg=cfg.pre_fusion_loss,
        pooling_type=pooling_type,
        pre_fusion_loss_weight=pre_fusion_loss_weight,
    )

    logger = WandbLogger(
        name=cfg.wandb.name,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        group="finetune_interaction" + suffix,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1score",
        mode="max",
        save_top_k=1,
        dirpath="checkpoints_dev",
        filename=f"best-f1score-finetune-interaction-{{epoch:02d}}{suffix}",
        auto_insert_metric_name=False,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_f1score",
        mode="max",
        patience=10,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    print("--- Starting Fine-tuning for Interaction Classification ---")
    trainer.fit(model, datamodule=data_module)

    print("--- Testing the model with the best val_f1score ---")
    trainer.test(datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)

    print("--- Fine-tuning and testing finished. ---")


if __name__ == "__main__":
    main()