import os
import sys

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, F1Score

import wandb

# 親ディレクトリをシステムパスに追加し、パッケージのインポートを可能にする
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loss_utils import LDAMLoss, FocalLoss
from train.interaction_with_image import (
    CattleInteractionDataModule
)


class LitInteractionViT(pl.LightningModule):
    """
    Vision Transformer (ViT) モデルをファインチューニングするためのPyTorch Lightningラッパーである。
    モデルの初期化、学習・検証・テストの各ステップ、およびオプティマイザの設定を管理する。
    """

    def __init__(
        self,
        num_classes,
        cls_num_list: list,
        main_loss_cfg: DictConfig,
        learning_rate=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 事前学習済みのVision Transformerモデルをロードする
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

        # 最終層をデータセットのクラス数に合わせて入れ替える
        num_ftrs = self.model.heads[0].in_features
        self.model.heads[0] = nn.Linear(num_ftrs, num_classes)

        if main_loss_cfg.name == "focal":
            self.criterion = FocalLoss(
                gamma=main_loss_cfg.gamma,
                alpha=main_loss_cfg.alpha,
            )
        elif main_loss_cfg.name == "ldam":
            self.criterion = LDAMLoss(
                cls_num_list=self.hparams.cls_num_list,
                max_m=self.hparams.main_loss_cfg.max_m,
                s=self.hparams.main_loss_cfg.s,
            )
        elif main_loss_cfg.name == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported main_loss: {main_loss_cfg.name}")

        # --- Metrics ---
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1score = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_f1score = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        """学習、検証、テストステップで共通の処理を行う内部メソッドである。"""
        _, _, images, labels, _ = batch
        labels = labels.view(-1)
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._shared_step(batch)
        self.train_accuracy(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._shared_step(batch)
        self.val_accuracy(logits, labels)
        self.val_f1score(logits, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("val_f1score", self.val_f1score, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self._shared_step(batch)
        self.test_accuracy(logits, labels)
        self.test_f1score(logits, labels)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True)
        self.log("test_f1score", self.test_f1score, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


@hydra.main(config_path="conf", config_name="interaction_train", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Hydraによって制御される学習・評価のメインプロセスである。
    """
    pl.seed_everything(cfg.seed, workers=True)

    map_label = dict(cfg.data.map_label)

    data_module = CattleInteractionDataModule(
        data_cfg=cfg.data, aug_cfg=cfg.augmentation
    )
    data_module.setup(stage="fit")

    model = LitInteractionViT(
        num_classes=len(map_label),
        learning_rate=cfg.training.learning_rate,
        cls_num_list=data_module.cls_num_list,
        main_loss_cfg=cfg.main_loss,
    )

    run_id = wandb.util.generate_id()

    wandb_logger = WandbLogger(
        id=run_id,
        resume="allow",
        name=cfg.wandb.name,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        log_model=False,
        group="interaction_image_vit",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    wandb_logger.watch(model, log="all", log_freq=100)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1score",
        mode="max",
        save_top_k=1,
        dirpath="checkpoints_develop",
        filename="best-f1-model-{epoch}-{val_f1score:.2f}",
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
        default_root_dir=cfg.work_dir,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(model, datamodule=data_module)

    print("--- Testing the model with the best val_f1score ---")
    trainer.test(datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)

    print("--- Fine-tuning and testing finished. ---")


if __name__ == "__main__":
    main()
