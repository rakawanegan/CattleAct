import os
import sys
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm
from typing import Union

# 親ディレクトリをシステムパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 行動推定モデルとデータモジュールをインポート
from metric.action_with_image import LitVisionTransformer, CattleActionDataModule

# インタラクション推定モデルとデータモジュールをインポート
from metric.interaction_cls_image_metric import (
    LitHybridStreamFusionForMetricLearning,
    CattleInteractionDataModule,
)

# 型エイリアスの定義
DataModuleType = Union[CattleActionDataModule, CattleInteractionDataModule]


def _generate_and_save_embeddings(
    model: nn.Module,
    data_module: DataModuleType,
    device: torch.device,
    embedding_dir: str,
    model_type: str,
    num_augmentations: int,
):
    """
    指定されたデータセット（学習/検証）に対して、データ拡張を用いた埋め込みを生成し保存する汎用関数。
    """
    if model_type not in ["action", "interaction"]:
        raise ValueError("model_type must be 'action' or 'interaction'")

    # 学習データと検証データの両方に対して処理を実行
    for stage in ["train", "val"]:
        print(f"\n--- Processing '{stage}' dataset for {model_type} model ---")

        if stage == "train":
            dataset = data_module.train_dataset
            # データ拡張ありの transform を取得
            aug_transform = data_module.transform_train
        else:
            dataset = data_module.val_dataset
            # 検証データセットには元々augがないため、学習用のtransformを流用
            aug_transform = data_module.transform_train

        # データ拡張なしの transform を取得
        base_transform = data_module.transform_val
        batch_size = data_module.val_dataloader().batch_size
        num_workers = data_module.val_dataloader().num_workers

        all_embeddings_list = []
        all_labels_list = []

        # --- 1. ベース埋め込みの計算 (拡張なし) ---
        dataset.transform = base_transform
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        with torch.no_grad():
            print(f"Calculating base embeddings for '{stage}' dataset...")
            for batch in tqdm(loader, desc=f"Base Embeddings ({stage})"):
                if model_type == "action":
                    imgs, _, labels, _ = batch
                    inputs = (imgs.to(device),)
                else: # interaction
                    images1, images2, images_context, labels, _ = batch
                    inputs = (images1.to(device), images2.to(device), images_context.to(device))
                
                raw_embeddings = model(*inputs)
                embeddings = nn.functional.normalize(raw_embeddings, p=2, dim=1)
                all_embeddings_list.append(embeddings.cpu())
                all_labels_list.append(labels.cpu().view(-1))
        
        # --- 2. 拡張埋め込みの計算 (TTA) ---
        if num_augmentations > 0:
            dataset.transform = aug_transform
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
            )
            for i in range(num_augmentations):
                print(f"Calculating augmented embeddings for '{stage}' dataset (round {i + 1}/{num_augmentations})...")
                with torch.no_grad():
                    for batch in tqdm(loader, desc=f"Augmentation {i + 1} ({stage})"):
                        if model_type == "action":
                            imgs, _, labels, _ = batch
                            inputs = (imgs.to(device),)
                        else: # interaction
                            images1, images2, images_context, labels, _ = batch
                            inputs = (images1.to(device), images2.to(device), images_context.to(device))

                        raw_embeddings = model(*inputs)
                        embeddings = nn.functional.normalize(raw_embeddings, p=2, dim=1)
                        all_embeddings_list.append(embeddings.cpu())
                        all_labels_list.append(labels.cpu().view(-1))

        # --- 3. 結合と保存 ---
        final_embeddings = torch.cat(all_embeddings_list, dim=0)
        final_labels = torch.cat(all_labels_list, dim=0)
        
        save_path_embeddings = os.path.join(embedding_dir, f"{stage}_embeddings.pt")
        save_path_labels = os.path.join(embedding_dir, f"{stage}_labels.pt")
        
        torch.save(final_embeddings, save_path_embeddings)
        torch.save(final_labels, save_path_labels)
        print(f"Saved {stage} embeddings ({len(final_embeddings)} samples) and labels for {model_type} model.")
        print(f"  - Embeddings: {save_path_embeddings}")
        print(f"  - Labels: {save_path_labels}")

        # データセットの transform を元に戻しておく
        if stage == "train":
            dataset.transform = aug_transform
        else:
            dataset.transform = base_transform


def precompute_action_embeddings(cfg: DictConfig, num_augmentations: int = 4):
    """
    行動推定モデルの学習・検証データセット全体の埋め込みを、TTAを適用して計算・保存する。
    """
    print("--- Starting precomputation for Action model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./checkpoints/action_metric_learning.ckpt"
    embedding_dir = "./embeddings/action"
    os.makedirs(embedding_dir, exist_ok=True)

    print(f"Loading action model from: {model_path}")
    model = LitVisionTransformer.load_from_checkpoint(model_path, map_location=device)
    model.to(device)
    model.eval()

    data_module = CattleActionDataModule(data_cfg=cfg.data, aug_cfg=cfg.augmentation)
    data_module.setup("fit")

    _generate_and_save_embeddings(
        model=model,
        data_module=data_module,
        device=device,
        embedding_dir=embedding_dir,
        model_type="action",
        num_augmentations=num_augmentations,
    )


def precompute_interaction_embeddings(cfg: DictConfig, num_augmentations: int = 4):
    """
    インタラクション推定モデルの学習・検証データセット全体の埋め込みを、TTAを適用して計算・保存する。
    """
    print("--- Starting precomputation for Interaction model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_path = "./checkpoints/interaction_metric_learning.ckpt"
    model_path = "./checkpoints_dev/metric-interaction-bin-best-epoch=1.ckpt"
    embedding_dir = "./embeddings/interaction"
    os.makedirs(embedding_dir, exist_ok=True)
    
    print(f"Loading interaction model from: {model_path}")
    model = LitHybridStreamFusionForMetricLearning.load_from_checkpoint(
        model_path, map_location=device
    )
    model.to(device)
    model.eval()

    data_module = CattleInteractionDataModule(data_cfg=cfg.data)
    data_module.setup("fit")

    _generate_and_save_embeddings(
        model=model,
        data_module=data_module,
        device=device,
        embedding_dir=embedding_dir,
        model_type="interaction",
        num_augmentations=num_augmentations,
    )


def main():
    """
    HydraのCompose APIを使い、各設定ファイルを個別に読み込んで事前計算を実行する。
    """
    # データ拡張を適用する回数。0に設定すると拡張なしのベース埋め込みのみが生成される。
    NUM_AUGMENTATIONS = 1

    # 1. 行動推定の計算
    with hydra.initialize(config_path="../metric/conf", version_base=None):
        cfg_action = hydra.compose(config_name="action_train")
        precompute_action_embeddings(cfg_action, num_augmentations=NUM_AUGMENTATIONS)

    # 2. インタラクション推定の計算
    with hydra.initialize(config_path="../metric/conf", version_base=None):
        cfg_interaction = hydra.compose(config_name="interaction_train")
        precompute_interaction_embeddings(cfg_interaction, num_augmentations=NUM_AUGMENTATIONS)

    print("\n--- All precomputations finished. ---")


if __name__ == "__main__":
    main()