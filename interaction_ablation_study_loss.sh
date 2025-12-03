#!/bin/bash

# ==========================================
# 開発理念: 安全性・確実性・保守性
# エラー発生時は即座に停止し、問題の連鎖を防ぐ
# ==========================================
set -e

# --- スクリプト引数の処理 ---
# 第1引数: GPUデバイスID (デフォルト: 0 に修正済み)
# メモ: 元のコメントは「デフォルト: 1」となっていましたが、コードの実態に合わせています
readonly DEVICE_ID=0

# --- 設定項目 ---

# 実行対象のPythonスクリプト
readonly FILE="train/interaction_with_image.py"

# 実験対象のパラメータ: loss_weight (0, 0.05, 0.1, 0.2)
readonly LOSS_WEIGHTS=("0" "0.05" "0.1" "0.2")

# 実行するシードの最大数 (1から5まで実行)
readonly MAX_SEED=5

# Wandb設定
readonly PROJECT_NAME="wacv_cattle_activity_recognition_interaction_loss_ablation"
# Group tagがスクリプト内で定義されていなかったため、必要に応じて追加してください
# readonly GROUP_TAG="interaction_loss_ablation" 


# --- 実験の実行 ---

for seed in $(seq 1 ${MAX_SEED}); do
  
  echo "######################################################################"
  echo "###   STARTING ALL EXPERIMENTS FOR SEED = ${seed}   ###"
  echo "###   Running on GPU Device ID: ${DEVICE_ID}                          ###"
  echo "######################################################################"

  for weight in "${LOSS_WEIGHTS[@]}"; do
    
    echo "--------------------------------------------------"
    echo "EXECUTING: ${FILE}"
    echo "  - seed: ${seed}"
    echo "  - pre_fusion_loss_weight: ${weight}"
    # echo "  - wandb.group: ${GROUP_TAG}"
    echo "  - training.devices: [${DEVICE_ID}]"
    echo "--------------------------------------------------"

    # Hydra引数指定
    # 修正: training.devices をリスト形式 [0] で渡すように変更
    python3 "${FILE}" \
      "seed=${seed}" \
      "model.pre_fusion_loss_weight=${weight}" \
      "training.devices=[${DEVICE_ID}]" \
      "wandb.project=${PROJECT_NAME}" \
      
  done
done

echo "All experiments successfully completed."