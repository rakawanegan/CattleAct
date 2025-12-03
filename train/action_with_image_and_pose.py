import os
import sys

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score

import wandb

# Add the parent directory to the system path to allow for package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import CattleActionDataset
from src.vit_gcn import Model, ViT_STGCN_Fusion
from train.action_with_image_and_pose_v1 import CattleActionDataModule


class LitMultiModal(pl.LightningModule):
    """
    ViT_STGCN_FusionモデルをファインチューニングするためのPyTorch Lightningラッパーである。
    モデルの初期化、学習・検証・テストの各ステップ、およびオプティマイザの設定を管理する。
    """

    def __init__(self, num_classes, model_kwargs, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.map_label = {"grazing": 0, "standing": 1, "lying": 2, "unknown": -1}
        self.learning_rate = learning_rate

        # ViT_STGCN_Fusionモデルを初期化する
        gcn_model_args = model_kwargs.get("gcn_model_args", {})
        use_local = model_kwargs.get("use_local", True)
        self.in_channels = model_kwargs["gcn_model_args"].get("in_channels", 3)

        # ST-GCNモデルのインスタンスを作成する
        gcn_model = Model(**gcn_model_args)

        # ViT_STGCN_Fusionモデルのインスタンスを作成する
        self.model = ViT_STGCN_Fusion(
            num_classes=num_classes, gcn=gcn_model, use_local=use_local
        )

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

    def forward(self, img, skel, joints_xy):
        """モデルのフォワードパスを定義する。"""
        return self.model(img, skel, joints_xy)

    def _prepare_inputs(self, poses, image_size):
        """
        データローダーからの 'poses' テンソルを画像サイズをもとにモデル入力用の 'skel' と 'joints_xy' に変換する。
        poses の形状は (N, C, T, V, M) であると仮定する。
        - skel: poses 自身
        - joints_xy: poses から抽出した (x, y) 座標。形状は (N, V, 2)で絶対座標
        """
        skel = poses.clone()
        # to 絶対座標
        widths = image_size[0].view(-1, 1, 1, 1)
        heights = image_size[1].view(-1, 1, 1, 1)
        skel[:, 0, :, :, :] *= widths
        skel[:, 1, :, :, :] *= heights

        # Cの最初の2次元がx, y座標、T>=1, M>=1と仮定し、最初のフレーム・人物の座標を抽出
        if not (
            skel.dim() == 5
            and skel.shape[1] >= 2
            and skel.shape[2] >= 1
            and skel.shape[4] >= 1
        ):
            raise ValueError(
                f"Unexpected pose shape: {skel.shape}. Expected (N, C, T, V, M) with C>=2, T>=1, M>=1."
            )

        # (N, C, T, V, M) -> (N, 2, V) -> (N, V, 2)
        joints_xy = skel[:, :2, 0, :, 0].permute(0, 2, 1)
        return poses, joints_xy

    def _common_step(self, batch):
        """学習、検証、テストステップで共通の処理をまとめる。"""
        imgs, poses, labels, suppmental_info = batch
        poses = poses[
            :, : self.in_channels, :, :, :
        ]  # 入力チャネルをself.in_channelsに基づいて調整
        labels = labels.view(-1)
        skel, joints_xy = self._prepare_inputs(poses, suppmental_info["image_size"])
        logits = self.forward(imgs, skel, joints_xy)
        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch)
        self.train_accuracy(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True
        )
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
        self.test_accuracy.update(preds, labels)
        self.test_f1score.update(preds, labels)

        self.log("test_loss", loss, on_epoch=True, sync_dist=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True, sync_dist=True)
        self.log("test_f1score", self.test_f1score, on_epoch=True, sync_dist=True)
        return loss

    def on_test_epoch_end(self):
        """テストエポック終了時に混同行列をロギングする。"""
        preds = torch.cat(self.all_test_preds).numpy()
        labels = torch.cat(self.all_test_labels).numpy()


    def configure_optimizers(self):
        """オプティマイザを設定する。"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


@hydra.main(config_path="conf", config_name="action_train", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Hydraによって制御される学習・評価のメインプロセスである。
    """
    pl.seed_everything(cfg.seed, workers=True)

    data_module = CattleActionDataModule(
        root_dir=cfg.data.root_dir,
        delete_base_dirs=cfg.data.delete_base_dirs,
        drop_unknown_label=cfg.data.drop_unknown_label,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
    )

    n_classes = 3 if cfg.data.drop_unknown_label else 4

    # --- モデル初期化の変更箇所 ---
    # Hydra設定ファイルからモデル引数を取得する
    model_kwargs = OmegaConf.to_container(cfg.model.model_kwargs, resolve=True)
    # GCNの引数にクラス数を設定する
    if "gcn_model_args" not in model_kwargs:
        model_kwargs["gcn_model_args"] = {}
    model_kwargs["gcn_model_args"]["num_class"] = n_classes
    model_kwargs["gcn_model_args"]["in_channels"] = 3
    model_kwargs["gcn_model_args"]["graph_args"] = {}
    model_kwargs["gcn_model_args"]["edge_importance_weighting"] = True
    model_kwargs["gcn_model_args"]["graph_args"] = {
        "layout": "cattle",
        "strategy": "spatial",
    }

    lit_model = LitMultiModal(
        num_classes=n_classes,
        model_kwargs=model_kwargs,
        learning_rate=cfg.training.learning_rate,
    )

    run_id = wandb.util.generate_id()

    wandb_logger = WandbLogger(
        id=run_id,
        resume="allow",
        name=cfg.wandb.name,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        log_model=False,
        group="action_image_and_pose_v2",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    wandb_logger.watch(lit_model, log="all", log_freq=100)

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        default_root_dir=cfg.work_dir,
        logger=wandb_logger,
        strategy="ddp_find_unused_parameters_true",
    )

    trainer.fit(lit_model, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()
