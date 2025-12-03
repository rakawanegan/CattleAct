#!/bin/bash

# コマンドの実行に失敗した場合、スクリプトを即座に終了させる
set -e

# --- スクリプト引数の処理 ---
# スクリプトの第1引数をGPUデバイスIDとして使用する。
# 引数が指定されていない場合は、デフォルト値として '1' を設定する。
# 使い方: ./your_script_name.sh [デバイスID]
# 例: ./your_script_name.sh 0  (GPU 0 を使用)
#     ./your_script_name.sh    (GPU 1 を使用)
readonly DEVICE_ID=${1:-1}


# --- 設定項目 ---

# 実行対象のPythonスクリプト
readonly FILE="train/interaction_with_image.py"

# training.pretrained_backbone の有無のリスト
readonly USE_PRETRAINED_OPTIONS=(true false)

# 各実験で実行するシードの最大数
readonly MAX_SEED=5


# --- 実験: training.pretrained_backbone の有無による比較 ---

echo "######################################################################"
echo "###   STARTING EXPERIMENT: PRETRAINED BACKBONE ABLATION   ###"
echo "###   Running on GPU Device ID: ${DEVICE_ID}                          ###"
echo "######################################################################"

for seed in $(seq 1 ${MAX_SEED}); do
  for use_pretrained in "${USE_PRETRAINED_OPTIONS[@]}"; do
    echo "--------------------------------------------------"
    echo "EXECUTING (Pretrained Backbone Comparison):"
    echo "  - seed: ${seed}"
    echo "  - training.pretrained_backbone: ${use_pretrained}"
    echo "  - training.devices: [${DEVICE_ID}]"
    echo "--------------------------------------------------"
    python3 "${FILE}" \
      "seed=${seed}" \
      "training.pretrained_backbone=${use_pretrained}" \
      "training.devices=[${DEVICE_ID}]" \
      "wandb.project=interaction_ablation_study_pretrain"
  done
done


echo "All experiments successfully completed."