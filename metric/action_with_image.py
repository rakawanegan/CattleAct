import math
import os
import random
import shutil
import sys
from collections import defaultdict
import io

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
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler
from torchmetrics import Accuracy, F1Score
from torchvision import transforms as T
import umap

import wandb

# Add the parent directory to the system path to allow for package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import (
    CattleActionDataset,
    get_all_action_annotations_entries,
    split_action_dataset_entries,
)
from src.augmentation import ImageMaskingFromSkeleton, StandardCutout


class LitVisionTransformer(pl.LightningModule):
    """
    Vision Transformer (ViT) モデルをMetric Learning（Triplet Loss）でファインチューニングするための
    PyTorch Lightningラッパーである。
    モデルの初期化、学習・検証・テストの各ステップ、およびオプティマイザの設定を管理する。
    UMAPによる埋め込みベクトルの可視化機能と、埋め込みベクトルに対する正則化項を追加している。
    """

    # ## 変更: __init__に正則化関連の引数を追加 ##
    def __init__(
        self,
        embedding_size,
        learning_rate=1e-4,
        margin=0.2,
        reg_loss_weight=0.1,
        sigma_s_squared=1.0,
    ):
        super().__init__()
        # save_hyperparameters()は追加された引数も自動的に保存する
        self.save_hyperparameters()
        self.map_label = {
            "grazing": 0,
            "standing": 1,
            "lying": 2,
            "riding": 3,
            "unknown": -1,
        }

        self.learning_rate = learning_rate
        self.margin = margin

        # 事前学習済みのVision Transformerモデルをロードする
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

        # 最終層をデータセットのクラス数に合わせた分類器から、
        # 指定された次元数の埋め込みベクトルを出力する層に入れ替える
        num_ftrs = self.model.heads[0].in_features
        self.model.heads[0] = nn.Linear(num_ftrs, embedding_size)

        self.criterion = nn.TripletMarginLoss(margin=self.margin, p=2)

        # 評価用にAccuracyとF1Scoreメトリックを初期化
        num_classes = len(self.map_label) - 1  # 'unknown'ラベルを除外
        self.val_knn_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_knn_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1score = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_f1score = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        # テスト時の予測とラベルを保存するリスト
        self.validation_step_outputs = []
        self.test_step_outputs = []

    # ## 変更: forwardからL2正規化を削除 ##
    def forward(self, x):
        """モデルのフォワードパス。埋め込みベクトルを返す。"""
        embedding = self.model(x)
        return embedding

    # ## 変更: training_stepに正則化損失の計算を追加 ##
    def training_step(self, batch, batch_idx):
        imgs, _, labels, _ = batch
        labels = labels.view(-1)

        # 正規化前の生の埋め込みベクトルを取得
        raw_embeddings = self.forward(imgs)

        # --- 正則化損失 L_lat の計算 ---
        l2_norm_squared = torch.sum(raw_embeddings.pow(2))
        reg_loss = l2_norm_squared / (2 * self.hparams.sigma_s_squared)

        # Triplet Loss計算のために埋め込みベクトルをL2正規化
        embeddings = F.normalize(raw_embeddings, p=2, dim=1)

        # バッチ内から全ての有効なトリプレットをマイニングする
        anchor, positive, negative = self._get_all_valid_triplets(embeddings, labels)

        if anchor is None:
            # 有効なトリプレットが見つからない場合はTriplet Lossを0とする
            triplet_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            triplet_loss = self.criterion(anchor, positive, negative)

        # --- 合計損失の計算 ---
        total_loss = triplet_loss + self.hparams.reg_loss_weight * reg_loss

        self.log("train_triplet_loss", triplet_loss, on_step=True, on_epoch=True)
        self.log("train_reg_loss", reg_loss, on_step=True, on_epoch=True)
        self.log(
            "train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True
        )

        return total_loss

    def _get_all_valid_triplets(self, embeddings, labels):
        """
        バッチ内の全ての有効なトリプレットを抽出する（Batch All）。
        ブロードキャスティングを用いて効率的に計算する。
        """
        # (この関数の内部実装は変更なし)
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        n = labels.size(0)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        identity = torch.eye(n, dtype=torch.bool, device=self.device)
        mask_positive = labels_eq & ~identity
        mask_negative = ~labels_eq
        anchor_positive_dist = dist_matrix.unsqueeze(2)
        anchor_negative_dist = dist_matrix.unsqueeze(1)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        mask_valid_triplets = mask_positive.unsqueeze(2) & mask_negative.unsqueeze(1)
        mask_loss = (triplet_loss > 0) & mask_valid_triplets

        if not mask_loss.any():
            return None, None, None

        anchor_idx, positive_idx, negative_idx = torch.where(mask_loss)
        anchor = embeddings[anchor_idx]
        positive = embeddings[positive_idx]
        negative = embeddings[negative_idx]

        return anchor, positive, negative

    # ## 変更: validation_stepでL2正規化を追加 ##
    def validation_step(self, batch, batch_idx):
        imgs, _, labels, _ = batch
        labels = labels.view(-1)
        raw_embeddings = self.forward(imgs)
        # 評価（k-NN）のために埋め込みベクトルをL2正規化
        embeddings = F.normalize(raw_embeddings, p=2, dim=1)
        self.validation_step_outputs.append(
            {"embeddings": embeddings.cpu(), "labels": labels.cpu()}
        )

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        embeddings = torch.cat(
            [x["embeddings"] for x in self.validation_step_outputs], dim=0
        )
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs], dim=0)
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        dist_matrix.fill_diagonal_(float("inf"))
        preds = labels[torch.argmin(dist_matrix, dim=1)]

        # ## 変更: preds と labels をモデルと同じデバイスに移動 ##
        preds_on_device = preds.to(self.device)
        labels_on_device = labels.to(self.device)

        self.val_knn_accuracy.update(preds_on_device, labels_on_device)
        self.val_f1score.update(preds_on_device, labels_on_device)
        self.log(
            "val_knn_acc", self.val_knn_accuracy.compute(), on_epoch=True, prog_bar=True
        )
        self.log(
            "val_f1score", self.val_f1score.compute(), on_epoch=True, prog_bar=True
        )

        if self.current_epoch % 10 == 0:
            self._plot_umap(stage="val")

        self.validation_step_outputs.clear()

    # ## 変更: test_stepでL2正規化を追加 ##
    def test_step(self, batch, batch_idx):
        imgs, _, labels, _ = batch
        labels = labels.view(-1)
        raw_embeddings = self.forward(imgs)
        # 評価（k-NN）のために埋め込みベクトルをL2正規化
        embeddings = F.normalize(raw_embeddings, p=2, dim=1)
        self.test_step_outputs.append(
            {"embeddings": embeddings.cpu(), "labels": labels.cpu()}
        )

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return

        embeddings = torch.cat([x["embeddings"] for x in self.test_step_outputs], dim=0)
        labels = torch.cat([x["labels"] for x in self.test_step_outputs], dim=0)
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        dist_matrix.fill_diagonal_(float("inf"))
        preds = labels[torch.argmin(dist_matrix, dim=1)]

        # ## 変更: preds と labels をモデルと同じデバイスに移動 ##
        preds_on_device = preds.to(self.device)
        labels_on_device = labels.to(self.device)

        self.test_knn_accuracy.update(preds_on_device, labels_on_device)
        self.test_f1score.update(preds_on_device, labels_on_device)
        self.log("test_knn_acc", self.test_knn_accuracy.compute(), on_epoch=True)
        self.log("test_f1score", self.test_f1score.compute(), on_epoch=True)

        if isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
            class_names = [k for k, v in self.map_label.items() if v != -1]
            self.logger.experiment.log(
                {
                    "test_confusion_matrix": wandb.plot.confusion_matrix(
                        preds=preds.numpy(),
                        y_true=labels.numpy(),
                        class_names=class_names,
                    )
                }
            )

        self._plot_umap(stage="test")
        self.test_step_outputs.clear()

    def _plot_umap(self, stage: str):
        outputs = (
            self.validation_step_outputs if stage == "val" else self.test_step_outputs
        )
        if not outputs:
            print(f"Skipping UMAP for stage '{stage}' due to no outputs collected.")
            return

        embeddings = torch.cat([x["embeddings"] for x in outputs], dim=0).numpy()
        labels = torch.cat([x["labels"] for x in outputs], dim=0).numpy()

        if len(embeddings) < 2:
            print(
                f"Skipping UMAP for stage '{stage}' due to insufficient samples ({len(embeddings)})."
            )
            return

        n_neighbors = min(15, len(embeddings) - 1)
        if n_neighbors <= 0:
            print(f"Skipping UMAP for stage '{stage}' due to n_neighbors <= 0.")
            return

        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42
        )
        umap_results = reducer.fit_transform(embeddings)
        fig, ax = plt.subplots(figsize=(15, 15))
        inv_map_label = {v: k for k, v in self.map_label.items() if v != -1}
        unique_labels = sorted(np.unique(labels))
        cmap = plt.get_cmap("viridis", len(unique_labels))
        colors = {label: cmap(i) for i, label in enumerate(unique_labels)}

        for label_val in unique_labels:
            mask = labels == label_val
            if np.any(mask):
                ax.scatter(
                    umap_results[mask, 0],
                    umap_results[mask, 1],
                    color=colors[label_val],
                    label=inv_map_label.get(label_val, f"Label {label_val}"),
                    alpha=0.7,
                    s=50,
                )

        ax.legend(title="Action", fontsize=12, title_fontsize=14)
        ax.set_title(
            f"UMAP Visualization of Embeddings ({stage.capitalize()})", fontsize=18
        )
        fig.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        im = Image.open(buf)

        log_key = f"{stage}_umap"
        if stage == "val":
            log_key = f"{stage}_umap_epoch_{self.current_epoch}"

        if isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
            self.logger.experiment.log({log_key: wandb.Image(im)})

        plt.close(fig)
        buf.close()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class BalancedBatchSampler(BatchSampler):
    """
    各バッチに指定された数のクラスから、各クラス指定された数のサンプルを含むようにサンプリングする。
    torch.utils.data.BatchSamplerを継承し、PyTorch Lightningの分散学習に対応する。
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = np.array(labels)
        self.n_classes = n_classes
        self.n_samples = n_samples

        self.n_batches = len(self.labels) // (self.n_classes * self.n_samples)
        if self.n_batches == 0:
            raise ValueError(
                f"Not enough samples to form even one batch. "
                f"Total samples: {len(self.labels)}, "
                f"Required per batch: {self.n_classes * self.n_samples}"
            )

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        for label, indices in self.label_to_indices.items():
            if len(indices) < self.n_samples:
                print(
                    f"Warning: Class {label} has only {len(indices)} samples, but {self.n_samples} are required per batch. "
                    "Samples will be repeated."
                )

    def __iter__(self):
        for _ in range(self.n_batches):
            available_classes = [
                lbl
                for lbl, indices in self.label_to_indices.items()
                if len(indices) > 0
            ]
            if len(available_classes) < self.n_classes:
                raise ValueError(
                    f"Not enough classes with samples to form a batch. "
                    f"Required classes: {self.n_classes}, Available: {len(available_classes)}"
                )
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


class CattleActionDataModule(pl.LightningDataModule):
    """
    CattleActionDatasetのためのPyTorch Lightningデータモジュールである。
    データセットの定義とデータ拡張の設定を分離して受け取り、管理する。
    """

    def __init__(self, data_cfg: DictConfig, aug_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_sampler = None
        self.mask_augment = None
        self._build_transforms()

    def _build_transforms(self):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_size = self.hparams.data_cfg.image_size

        if self.hparams.aug_cfg.use:
            rand_aug_cfg = self.hparams.aug_cfg.randaugment
            self.transform_train = T.Compose(
                [
                    T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.RandAugment(
                        num_ops=rand_aug_cfg.num_ops,
                        magnitude=rand_aug_cfg.magnitude,
                    ),
                    T.ToTensor(),
                    normalize,
                ]
            )
        else:
            self.transform_train = T.Compose(
                [
                    T.Resize(image_size),
                    T.CenterCrop(image_size),
                    T.ToTensor(),
                    normalize,
                ]
            )

        self.transform_val = T.Compose(
            [
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
                normalize,
            ]
        )
        cfg = self.hparams.aug_cfg.masking_from_skeleton
        if not cfg.use:
            self.mask_augment = StandardCutout(
                cutout_prob=cfg.cutout_prob,
                n_holes=cfg.n_holes,
                scale=tuple(cfg.scale),
                ratio=tuple(cfg.ratio),
            )
        else:
            joint_map = {
                "head": [0, 1, 2],
                "neck": [3],
                "torso": [4],
                "left_front_leg": [5, 6, 7],
                "right_front_leg": [8, 9, 10],
                "left_hind_leg": [11, 12, 13],
                "right_hind_leg": [14, 15, 16],
            }
            self.mask_augment = ImageMaskingFromSkeleton(
                joint_map=joint_map,
                cutout_prob=cfg.cutout_prob,
                n_holes=cfg.n_holes,
                scale=tuple(cfg.scale),
                ratio=tuple(cfg.ratio),
                skip_label=cfg.skip_label,
                unuse_low_conf_skel=cfg.unuse_low_conf_skel,
            )

    def setup(self, stage: str = None):
        map_label = {
            "grazing": 0,
            "standing": 1,
            "lying": 2,
            "riding": 3,
            "unknown": -1,
        }
        self.label_map = map_label

        full_dataset_entries = get_all_action_annotations_entries(
            root_dir=self.hparams.data_cfg.root_dir,
            map_label=map_label,
            delete_base_dirs=self.hparams.data_cfg.delete_base_dirs,
            drop_unknown_label=self.hparams.data_cfg.drop_unknown_label,
        )

        train_entries, val_entries, test_entries = split_action_dataset_entries(
            full_dataset_entries,
            split_type=self.hparams.data_cfg.split_type,
        )

        self.train_dataset = CattleActionDataset(
            entries=train_entries,
            label_map=self.label_map,
            image_transform=self.transform_train,
            custom_image_transform=self.mask_augment,
        )
        self.val_dataset = CattleActionDataset(
            entries=val_entries,
            label_map=self.label_map,
            image_transform=self.transform_val,
            custom_image_transform=None,
        )
        self.test_dataset = CattleActionDataset(
            entries=test_entries,
            label_map=self.label_map,
            image_transform=self.transform_val,
            custom_image_transform=None,
        )

        if stage == "fit" or stage is None:
            if not self.train_dataset.entries:
                return
            labels = [entry["label"] for entry in self.train_dataset.entries]
            self.train_batch_sampler = BalancedBatchSampler(
                labels,
                n_classes=self.hparams.data_cfg.classes_per_batch,
                n_samples=self.hparams.data_cfg.samples_per_class,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.hparams.data_cfg.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        batch_size = (
            self.hparams.data_cfg.classes_per_batch
            * self.hparams.data_cfg.samples_per_class
        )
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            num_workers=self.hparams.data_cfg.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        batch_size = (
            self.hparams.data_cfg.classes_per_batch
            * self.hparams.data_cfg.samples_per_class
        )
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=self.hparams.data_cfg.num_workers,
            pin_memory=True,
        )


@hydra.main(config_path="conf", config_name="action_train", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Hydraによって制御される学習・評価のメインプロセスである。
    """
    pl.seed_everything(cfg.seed, workers=True)

    data_module = CattleActionDataModule(data_cfg=cfg.data, aug_cfg=cfg.augmentation)

    lit_model = LitVisionTransformer(
        embedding_size=cfg.model.embedding_size,
        learning_rate=cfg.training.learning_rate,
        margin=cfg.model.margin,
        reg_loss_weight=cfg.model.reg_loss_weight,
        sigma_s_squared=cfg.model.sigma_s_squared,
    )

    suffix = ""
    use_skeleton_aug = cfg.augmentation.masking_from_skeleton.use
    if use_skeleton_aug:
        suffix += "_aug_pose"

    run_id = wandb.util.generate_id()

    wandb_logger = WandbLogger(
        id=run_id,
        resume="allow",
        name=cfg.wandb.name,
        project="wacv_cattle_activity_recognition_action",
        entity=cfg.wandb.entity,
        group="action_image_metric_learning" + suffix,
        log_model=False,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    wandb_logger.watch(lit_model, log="all", log_freq=100)


    checkpoint_callback = ModelCheckpoint(
        monitor="val_knn_acc",
        mode="max",
        save_top_k=1,
        dirpath="checkpoints_dev",
        filename="action_metric_learning" + suffix,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        default_root_dir=cfg.work_dir,
        logger=wandb_logger,
        use_distributed_sampler=False,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(lit_model, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()
