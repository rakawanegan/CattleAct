import os
import sys

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, WeightedRandomSampler

# 親ディレクトリをシステムパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import (
    CattleCroppedInteractionDataset,
    get_all_interaction_annotations_entries,
    split_interaction_dataset_entries,
)
from train.interaction_with_image import (
    LitHybridStreamFusion,
)


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

        self.skeleton_aware_transform = None

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
            
            # 0で割ることを防ぐため、サンプル数が0のクラスは重みを0にする
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

    data_module = CattleInteractionDataModule(data_cfg=cfg.data, aug_cfg=cfg.augmentation)
    data_module.setup(stage="fit")

    num_classes = len(cfg.data.map_label)
    
    suffix = ""
    if cfg.training.pretrained_backbone:
        suffix += "_pretrained"
    suffix += f"_{cfg.model.fusion_type}"
    suffix += "_aug_no_aug"
    if cfg.pre_fusion_loss.name != "none":
        suffix += f"_pfl_{cfg.pre_fusion_loss.name}"

    if "pooling_type" not in cfg.model:
        raise ValueError("cfg.model.pooling_type must be defined in the configuration.")
    pooling_type = str(cfg.model.pooling_type)

    if "weight" not in cfg.pre_fusion_loss:
        raise ValueError("cfg.pre_fusion_loss.weight must be defined in the configuration.")
    pre_fusion_loss_weight = float(cfg.pre_fusion_loss.weight)

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
        project="interaction_ablation_study_augmentation",
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        group="finetune_interaction" + suffix,
    )
    checkpoint_callback_f1 = ModelCheckpoint(
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
        callbacks=[checkpoint_callback_f1, early_stopping_callback],
    )

    print("--- Starting Fine-tuning for Interaction Classification ---")
    trainer.fit(model, datamodule=data_module)

    print("--- Testing the model with the best val_f1score ---")
    trainer.test(datamodule=data_module, ckpt_path=checkpoint_callback_f1.best_model_path)

    print("--- Fine-tuning and testing finished. ---")


if __name__ == "__main__":
    main()