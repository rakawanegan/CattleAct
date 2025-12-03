import math
import os
from glob import glob
import random
import shutil
import sys

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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, F1Score
from torchvision import transforms as T

import wandb

# Add the parent directory to the system path to allow for package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.augmentation import ImageMaskingFromSkeleton
from src.dataset import CattleActionDataset, get_all_action_annotations_entries, split_action_dataset_entries


class LitVisionTransformer(pl.LightningModule):
    """
    Vision Transformer (ViT) モデルをファインチューニングするためのPyTorch Lightningラッパーである。
    モデルの初期化、学習・検証・テストの各ステップ、およびオプティマイザの設定を管理する。
    """

    def __init__(self, num_classes, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.map_label = {"grazing": 0, "standing": 1, "lying": 2, "unknown": -1}

        self.learning_rate = learning_rate

        # 事前学習済みのVision Transformerモデルをロードする
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

        # # 全ての層のパラメータを訓練不可に設定する
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # 最終層をデータセットのクラス数に合わせて入れ替える
        # この新しい層のパラメータはデフォルトで訓練可能(requires_grad=True)となる
        num_ftrs = self.model.heads[0].in_features
        self.model.heads[0] = nn.Linear(num_ftrs, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1score = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_f1score = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        self.all_test_preds = []
        self.all_test_labels = []

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch):
        """学習、検証、テストステップで共通の処理をまとめる。"""
        imgs, _, labels, _ = batch
        labels = labels.view(-1)
        logits = self.forward(imgs)
        loss = self.criterion(logits, labels)
        return logits, loss, labels

    def training_step(self, batch, batch_idx):
        logits, loss, labels = self._common_step(batch)

        loss = self.criterion(logits, labels)
        self.train_accuracy(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss, labels = self._common_step(batch)

        self.val_accuracy(logits, labels)
        self.val_f1score(logits, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("val_f1score", self.val_f1score, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        logits, loss, labels = self._common_step(batch)

        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # Confusion Matrix用に予測とラベルを蓄積する
        self.all_test_preds.append(preds.cpu())
        self.all_test_labels.append(labels.cpu())
        self.test_accuracy.update(preds, labels)
        self.test_f1score.update(preds, labels)

        self.log("test_loss", loss, on_epoch=True, sync_dist=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True, sync_dist=True)
        self.log("test_f1score", self.test_f1score, on_epoch=True, sync_dist=True)

        return loss

    def on_test_epoch_end(self):
        # 蓄積された予測とラベルを結合
        preds = torch.cat(self.all_test_preds).numpy()
        labels = torch.cat(self.all_test_labels).numpy()

        # wandbにConfusion Matrixを記録
        # self.logger.experimentがwandbのrunオブジェクトを指していることを想定
        if isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
            self.logger.experiment.log(
                {
                    "test_confusion_matrix": wandb.plot.confusion_matrix(
                        preds=preds,
                        y_true=labels,
                        class_names=list(self.map_label.keys()),
                    )
                }
            )

    def configure_optimizers(self):
        # requires_grad=Trueのパラメータのみが更新対象となる
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class CattleActionDataModule(pl.LightningDataModule):
    """
    CattleActionDatasetのためのPyTorch Lightningデータモジュールである。
    データセットの定義とデータ拡張の設定を分離して受け取り、管理する。
    """

    def __init__(self, data_cfg: DictConfig, aug_cfg: DictConfig):
        super().__init__()
        # データ設定と拡張設定を個別の名前空間でハイパーパラメータとして保存する
        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_sampler = None
        self.mask_augment = self._build_mask_augment()
        self._build_transforms()

    def _build_mask_augment(self) -> ImageMaskingFromSkeleton :
        """設定に基づき、骨格情報を用いたマスキング拡張を構築する。"""
        cfg = self.hparams.aug_cfg.masking_from_skeleton
        if not cfg.use:
            return None

        joint_map = {
            "head": [0, 1, 2], "neck": [3], "torso": [4],
            "left_front_leg": [5, 6, 7], "right_front_leg": [8, 9, 10],
            "left_hind_leg": [11, 12, 13], "right_hind_leg": [14, 15, 16],
        }
        return ImageMaskingFromSkeleton(
            joint_map=joint_map,
            cutout_prob=cfg.cutout_prob,
            n_holes=cfg.n_holes,
            scale=tuple(cfg.scale),
            ratio=tuple(cfg.ratio),
            skip_label=cfg.skip_label,
            unuse_low_conf_skel=cfg.unuse_low_conf_skel,
        )

    def _build_transforms(self):
        """設定に基づき、学習用および検証・テスト用の画像変換パイプラインを構築する。"""
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_size = self.hparams.data_cfg.image_size

        if self.hparams.aug_cfg.use:
            rand_aug_cfg = self.hparams.aug_cfg.randaugment
            self.transform_train = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandAugment(
                    num_ops=rand_aug_cfg.num_ops,
                    magnitude=rand_aug_cfg.magnitude,
                ),
                T.ToTensor(), normalize,
            ])
        else:
            self.transform_train = T.Compose([
                T.Resize(image_size), T.CenterCrop(image_size),
                T.ToTensor(), normalize,
            ])

        self.transform_val = T.Compose([
            T.Resize(image_size), T.CenterCrop(image_size),
            T.ToTensor(), normalize,
        ])

    def setup(self, stage: str  = None):
        if self.train_dataset:
            return

        map_label = {"grazing": 0, "standing": 1, "lying": 2, "unknown": -1}
        self.label_map = map_label

        full_dataset_entries = get_all_action_annotations_entries(
            root_dir=self.hparams.data_cfg.root_dir,
            map_label=map_label,
            delete_base_dirs=self.hparams.data_cfg.delete_base_dirs,
            drop_unknown_label=self.hparams.data_cfg.drop_unknown_label,
        )

        train_entries, val_entries, test_entries = split_action_dataset_entries(
            full_dataset_entries, split_type=self.hparams.data_cfg.split_type,
        )

        self.train_dataset = CattleActionDataset(
            entries=train_entries, label_map=self.label_map,
            image_transform=self.transform_train, custom_image_transform=self.mask_augment,
        )
        self.val_dataset = CattleActionDataset(
            entries=val_entries, label_map=self.label_map,
            image_transform=self.transform_val, custom_image_transform=None,
        )
        self.test_dataset = CattleActionDataset(
            entries=test_entries, label_map=self.label_map,
            image_transform=self.transform_val, custom_image_transform=None,
        )

        if stage == "fit" or stage is None:
            if not self.train_dataset.entries:
                return
            labels = [entry["label"] for entry in self.train_dataset.entries]
            class_counts = torch.bincount(torch.tensor(labels))
            class_weights = 1.0 / (class_counts.float() + 1e-6)
            sample_weights = torch.tensor([class_weights[label] for label in labels])
            self.train_sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights, num_samples=len(sample_weights), replacement=True,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.data_cfg.batch_size,
            shuffle=False, num_workers=self.hparams.data_cfg.num_workers,
            pin_memory=True, sampler=self.train_sampler,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.data_cfg.batch_size,
            num_workers=self.hparams.data_cfg.num_workers, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.hparams.data_cfg.batch_size,
            num_workers=self.hparams.data_cfg.num_workers, pin_memory=True,
        )


@hydra.main(config_path="conf", config_name="action_train", version_base=None)
def main(cfg: DictConfig) -> float:
    """
    Hydraによって制御される学習・評価のメインプロセスである。
    Optunaによるハイパーパラメータ探索のために、検証F1スコアの最大値を返す。
    """
    pl.seed_everything(cfg.seed, workers=True)

    data_module = CattleActionDataModule(
        data_cfg=cfg.data,
        aug_cfg=cfg.augmentation
    )

    n_classes = 3 if cfg.data.drop_unknown_label else 4

    lit_model = LitVisionTransformer(
        num_classes=n_classes, learning_rate=cfg.training.learning_rate
    )

    run_name = f"{cfg.wandb.name}_{hydra.core.hydra_config.HydraConfig.get().job.num}"
    wandb_logger = WandbLogger(
        name=run_name,
        project=str(cfg.wandb.project) + '_optuna',
        entity=cfg.wandb.entity,
        group="action_image_aug_pose_optuna",
        log_model=False,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    wandb_logger.watch(lit_model, log="all", log_freq=100)
    
    # ModelCheckpointコールバックで最良のモデルを保存する
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1score",  # 最適化の目的指標
        mode="max",
    )

    # EarlyStoppingコールバックで不要な学習を早期に打ち切る
    early_stopping_callback = EarlyStopping(
        monitor="val_f1score",
        patience=5,
        mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        default_root_dir=cfg.work_dir,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(lit_model, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path="best")

    best_score = checkpoint_callback.best_model_score
    objective_value = best_score.item() if best_score is not None else 0.0

    wandb.finish()

    best_model_path = checkpoint_callback.best_model_path
    
    if best_model_path:
        # ディレクトリ内のすべての.ckptファイルを検索
        all_ckpts = glob(os.path.join(os.path.dirname(best_model_path), "*.ckpt"))
        for ckpt in all_ckpts:
            if ckpt != best_model_path:
                try:
                    os.remove(ckpt)
                    print(f"Removed old checkpoint: {ckpt}")
                except OSError as e:
                    print(f"Error removing file {ckpt}: {e.strerror}")

    return objective_value


if __name__ == "__main__":
    main()
