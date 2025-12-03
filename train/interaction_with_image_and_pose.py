import os
import sys

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score

import wandb

# 親ディレクトリをシステムパスに追加し、パッケージのインポートを可能にする
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 必要なデータセットクラスとモデルクラスをインポート
from src.vit_gcn import Model, ViT_STGCN_Fusion

# データモジュールはご提示のものをそのまま使用します
from train.interaction_with_image_vit import CattleInteractionDataModule


class LitInteractionMultiModal(pl.LightningModule):
    """
    ViT_STGCN_Fusionモデルをインタラクションデータセットでファインチューニングするための
    PyTorch Lightningラッパーである。2個体分のポーズデータを扱う。
    """

    def __init__(self, num_classes, model_kwargs, learning_rate=1e-4, image_size=224):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.image_size = image_size

        # --- モデル初期化 ---
        # Hydra設定からGCNモデルとFusionモデルの引数を取得
        use_local = model_kwargs.get("use_local", True)
        self.in_channels = model_kwargs.get("in_channels", 3)

        # ST-GCNモデルのインスタンスを作成
        gcn_model = Model(**model_kwargs)

        # ViT_STGCN_Fusionモデルのインスタンスを作成
        self.model = ViT_STGCN_Fusion(
            num_classes=num_classes, gcn=gcn_model, use_local=use_local
        )

        self.criterion = nn.CrossEntropyLoss()

        # --- メトリクス ---
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1score = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1score = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        
        # テスト結果収集用
        self.all_test_preds = []
        self.all_test_labels = []

    def forward(self, img, skel, joints_xy):
        """モデルのフォワードパスを定義する。"""
        return self.model(img, skel, joints_xy)

    def _prepare_inputs(self, poses):
        """
        データローダーからの 'poses' テンソルをモデル入力用の 'skel' と 'joints_xy' に変換する。
        poses の形状は (B, C, T, V, M=2) である。
        - skel: GCN用の入力。posesをそのまま利用。座標はクロップ画像基準の相対座標。
        - joints_xy: ViTアライメント用。絶対ピクセル座標に変換し、2個体分を連結。
        """
        # GCN用のスケルトンデータはそのまま利用
        skel = poses.clone()

        # ViTアライメント用の絶対座標を計算
        poses_abs = poses.clone()
        # データセット側で既にクロップ画像基準になっているため、モデル入力サイズを掛けるだけで絶対座標になる
        poses_abs[:, 0, :, :, :] *= self.image_size # width
        poses_abs[:, 1, :, :, :] *= self.image_size # height

        # (B, C, T, V, M) -> (B, 2, V, M) -> (B, V, M, 2) -> (B, V*M, 2)
        B, _, _, V, M = poses_abs.shape
        joints_xy = poses_abs[:, :2, 0, :, :].permute(0, 2, 3, 1).reshape(B, V * M, 2)
        
        return skel, joints_xy

    def _common_step(self, batch):
        """学習、検証、テストステップで共通の処理をまとめる。"""
        _, _, imgs, labels, supplemental_info = batch
        poses = supplemental_info['pose']

        # 入力チャネルをself.in_channelsに基づいて調整
        poses = poses[:, : self.in_channels, :, :, :]
        labels = labels.view(-int(1))
        
        # 2個体分のポーズデータをモデル入力形式に変換
        skel, joints_xy = self._prepare_inputs(poses)
        
        logits = self.forward(imgs, skel, joints_xy)
        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch)
        self.train_accuracy(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch)
        self.val_accuracy(logits, labels)
        self.val_f1score(logits, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("val_f1score", self.val_f1score, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch)
        preds = torch.argmax(logits, dim=1)
        
        self.test_accuracy(preds, labels)
        self.test_f1score(preds, labels)
        self.all_test_preds.append(preds.cpu())
        self.all_test_labels.append(labels.cpu())
        
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True)
        self.log("test_f1score", self.test_f1score, on_epoch=True)
        
    def on_test_epoch_end(self):
        """テストエポック終了時に混同行列をロギングする。"""
        preds = torch.cat(self.all_test_preds).numpy()
        labels = torch.cat(self.all_test_labels).numpy()


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
    """Hydraによって制御される学習・評価のメインプロセスである。"""
    pl.seed_everything(cfg.seed, workers=True)

    map_label = dict(cfg.data.map_label)
    n_classes = len(map_label)

    data_module = CattleInteractionDataModule(
        data_cfg=cfg.data, aug_cfg=cfg.augmentation
    )
    data_module.setup(stage="fit")

    # --- モデル初期化 ---
    # Hydra設定ファイルからモデル引数を取得
    model_kwargs = OmegaConf.to_container(cfg.model.model_kwargs, resolve=True)
    # GCNの引数にクラス数を設定
    model_kwargs["num_class"] = n_classes

    lit_model = LitInteractionMultiModal(
        num_classes=n_classes,
        model_kwargs=model_kwargs,
        learning_rate=cfg.training.learning_rate,
        image_size=cfg.data.image_size,
    )

    run_id = wandb.util.generate_id()
    wandb_logger = WandbLogger(
        id=run_id,
        resume="allow",
        name=cfg.wandb.name,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        log_model=False,
        group="interaction_image_and_pose",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    wandb_logger.watch(lit_model, log="all", log_freq=100)

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
        strategy="ddp_find_unused_parameters_true",
    )

    trainer.fit(lit_model, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()