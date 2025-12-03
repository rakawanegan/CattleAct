import sys
import os
import torch
import hydra
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import torch.nn as nn
import pytorch_lightning as pl

# --- 必要なモジュールをインポート ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train.action_with_image import LitVisionTransformer, CattleActionDataModule
from pretrain.interaction_bin_image_metric import CattleInteractionDataModule as CattleInteractionDataModuleForInteraction


def extract_features(model: nn.Module, dataloader: DataLoader, device: str) -> tuple:
    """
    データローダーからデータを取得し、モデルを用いて特徴量とラベル情報を抽出する。
    Interactionデータセットから渡されるsupplemental_labelも安全に抽出する。
    """
    model.to(device)
    model.eval()

    if isinstance(model, LitVisionTransformer):
        original_head = model.model.heads[0]
        model.model.heads[0] = nn.Identity()
    else:
        original_head = None

    features_list = []
    labels_list = []
    supplemental_labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch[0].to(device)
            labels = batch[2].cpu()

            # --- エラー修正箇所 ---
            # supplemental_labelの存在を安全にチェックする
            if len(batch) > 3 and isinstance(batch[3], dict) and 'supplemental_label' in batch[3]:
                # Interaction Dataloaderの場合
                supp_info = batch[3]
                supplemental_labels = supp_info['supplemental_label'].cpu()
            else:
                # Action Dataloaderなど、supplemental_labelがない場合
                supplemental_labels = torch.full_like(labels, -1)

            if isinstance(model, LitVisionTransformer):
                 features = model.model(imgs)
            else:
                 features = model(imgs)

            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            supplemental_labels_list.append(supplemental_labels.numpy())

    if isinstance(model, LitVisionTransformer) and original_head is not None:
        model.model.heads[0] = original_head

    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    all_supplemental_labels = np.concatenate(supplemental_labels_list, axis=0)

    return all_features, all_labels, all_supplemental_labels


@hydra.main(config_path="conf", config_name="multitask_train", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Actionの特徴量空間にInteractionのデータをプロットし、UMAPによる可視化を実行するメイン関数である。
    """
    pl.seed_everything(cfg.seed, workers=True)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    data_loader_mode = 'train'
    data_loader_mode = 'val'
    data_loader_mode = 'test'
    # data_loader_mode = 'all'

    action_model = LitVisionTransformer.load_from_checkpoint(
        checkpoint_path='checkpoints/action_image.ckpt', map_location=torch.device(device)
    )

    # Actionデータモジュールを準備し、特徴量を抽出する
    common_dict = OmegaConf.to_container(cfg.data.common, resolve=True)
    action_dict = OmegaConf.to_container(cfg.data.action, resolve=True)
    action_data_dict = {**common_dict, **action_dict}
    action_data_dict['split_type'] = 'date'
    action_data_cfg = OmegaConf.create(action_data_dict)

    action_data_module = CattleActionDataModule(data_cfg=action_data_cfg, aug_cfg=cfg.augmentation)
    action_data_module.setup()
    action_data_loader = action_data_module.test_dataloader()
    
    print("Actionテストデータから特徴量を抽出しています...")
    action_features, action_labels, _ = extract_features(action_model, action_data_loader, device)
    print(f"Action特徴量 shape: {action_features.shape}, ラベル shape: {action_labels.shape}")

    # Interactionデータモジュールを準備し、特徴量を抽出する
    interaction_dict = OmegaConf.to_container(cfg.data.interaction, resolve=True)
    interaction_data_dict = {**common_dict, **interaction_dict}

    interaction_data_module = CattleInteractionDataModuleForInteraction(
        root_dir=interaction_data_dict['root_dir'],
        delete_base_dirs=interaction_data_dict['delete_base_dirs'],
        batch_size=interaction_data_dict['batch_size'],
        num_workers=interaction_data_dict['num_workers'],
        image_size=interaction_data_dict['image_size'],
        use_more_than_three_cattles=interaction_data_dict['use_more_than_three_cattles'],
        bin_map_label=interaction_data_dict.get('bin_map_label', {}),
        map_label=interaction_data_dict['map_label']
    )
    interaction_data_module.setup()
    if data_loader_mode == 'train':
        interaction_data_loader = interaction_data_module.train_dataloader()
    elif data_loader_mode == 'val':
        interaction_data_loader = interaction_data_module.val_dataloader()
    elif data_loader_mode == 'test':
        interaction_data_loader = interaction_data_module.test_dataloader()
    elif data_loader_mode == 'all':
        interaction_data_loaders = [
            interaction_data_module.train_dataloader(),
            interaction_data_module.val_dataloader(),
            interaction_data_module.test_dataloader()
        ]
    else:
        raise ValueError(f"Invalid data_loader_mode: {data_loader_mode}")

    print("Interactionテストデータから特徴量を抽出しています...")
    if data_loader_mode == 'all':
        all_interaction_features = []
        all_interaction_supplemental_labels = []
        for loader in interaction_data_loaders:
            features, _, supplemental_labels = extract_features(action_model, loader, device)
            all_interaction_features.append(features)
            all_interaction_supplemental_labels.append(supplemental_labels)
        interaction_features = np.concatenate(all_interaction_features, axis=0)
        interaction_supplemental_labels = np.concatenate(all_interaction_supplemental_labels, axis=0)
    else:
        interaction_features, _, interaction_supplemental_labels = extract_features(action_model, interaction_data_loader, device)
    print(f"Interaction特徴量 shape: {interaction_features.shape}, supplemental_label shape: {interaction_supplemental_labels.shape}")

    print("UMAPによる次元削減を実行しています...")
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, random_state=cfg.seed
    )
    reducer.fit(action_features)
    action_embedding = reducer.transform(action_features)
    interaction_embedding = reducer.transform(interaction_features)
    print(f"Action埋め込みベクトル shape: {action_embedding.shape}")
    print(f"Interaction埋め込みベクトル shape: {interaction_embedding.shape}")

    # --- DataFrameの作成 ---
    action_label_map_inv = {v: k for k, v in action_data_module.label_map.items()}
    action_label_names = [action_label_map_inv.get(l, "Unknown") for l in action_labels]
    df_action = pd.DataFrame({
        "UMAP-1": action_embedding[:, 0],
        "UMAP-2": action_embedding[:, 1],
        "label": action_label_names,
        "source": "Action"
    })

    interaction_label_map_inv = {v: k for k, v in interaction_data_module.map_label.items()}
    interaction_label_names = []
    for l in interaction_supplemental_labels:
        if l == -1:
            interaction_label_names.append("no_interaction (Interaction)")
        else:
            label_name = interaction_label_map_inv.get(l, f"Unknown_{l}")
            interaction_label_names.append(f"{label_name} (Interaction)")
            
    df_interaction = pd.DataFrame({
        "UMAP-1": interaction_embedding[:, 0],
        "UMAP-2": interaction_embedding[:, 1],
        "label": interaction_label_names,
        "source": "Interaction"
    })

    df_combined = pd.concat([df_action, df_interaction], ignore_index=True)

    # --- プロット処理 ---
    print("結果をプロットしています...")
    plt.figure(figsize=(16, 14))
    
    marker_map = {"Action": "o", "Interaction": "v"}
    
    color_map = {}
    all_unique_labels = sorted(df_combined["label"].unique())
    
    no_interaction_label = "no_interaction (Interaction)"
    other_labels = [label for label in all_unique_labels if label != no_interaction_label]
    
    if no_interaction_label in all_unique_labels:
        color_map[no_interaction_label] = 'gray'
    
    palette = sns.color_palette("husl", n_colors=len(other_labels))
    for label, color in zip(other_labels, palette):
        color_map[label] = color
    
    ax = sns.scatterplot(
        x="UMAP-1",
        y="UMAP-2",
        hue="label",
        style="source",
        markers=marker_map,
        s=100,
        palette=color_map,
        data=df_combined,
        legend="full",
        alpha=0.85,
    )
    
    plt.title(f"UMAP Projection of Action and Interaction Datasets in Action Space ({data_loader_mode} data)", fontsize=16)
    plt.grid(True)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=12)

    output_filename = f"outputs/combined_action_interaction_umap_plot_{data_loader_mode}.png"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"UMAPのプロットを '{output_filename}' として保存しました。")
    plt.show()


if __name__ == "__main__":
    main()