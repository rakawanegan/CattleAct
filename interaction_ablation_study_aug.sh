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

# augmentation.masking_from_skeleton.use の有無のリスト
readonly USE_MASKING_OPTIONS=(true false)

# グリッドサーチ用のハイパーパラメータリスト
readonly CUTOUT_PROB_OPTIONS=(0.5 1)
readonly N_HOLES_OPTIONS=(5 7)
readonly MARGIN_OPTIONS=(10 20)

# 各実験で実行するシードの最大数
readonly MAX_SEED=5


# --- 実験1: augmentation.masking_from_skeleton.use の有無および詳細パラメータによる比較 ---

echo "######################################################################"
echo "###   STARTING EXPERIMENT SET 1: MASKING ABLATION & GRID SEARCH   ###"
echo "###   Running on GPU Device ID: ${DEVICE_ID}                      ###"
echo "######################################################################"

# シード値のループ
for seed in $(seq 1 ${MAX_SEED}); do
  echo "--------------------------------------------------"
  echo "EXECUTING (Baseline):"
  echo "  - seed: ${seed}"
  echo "  - only basic augmentation (no masking)"
  echo "  - training.devices: [${DEVICE_ID}]"
  echo "--------------------------------------------------"
  python3 "train/interaction_with_image_no_aug.py" \
    "seed=${seed}" \
    "training.devices=[${DEVICE_ID}]"

  # マスキングの有無のループ
  for use_masking in "${USE_MASKING_OPTIONS[@]}"; do
    if [ "${use_masking}" = "true" ]; then
      # use_masking=true の場合、ハイパーパラメータのグリッドサーチを実行
      for n_holes in "${N_HOLES_OPTIONS[@]}"; do
        for margin in "${MARGIN_OPTIONS[@]}"; do
          echo "--------------------------------------------------"
          echo "EXECUTING (Masking Grid Search):"
          echo "  - seed: ${seed}"
          echo "  - use_masking: true"
          echo "  - n_holes: ${n_holes}"
          echo "  - margin: ${margin}"
          echo "  - training.devices: [${DEVICE_ID}]"
          echo "--------------------------------------------------"

          python3 "${FILE}" \
            "seed=${seed}" \
            "augmentation.masking_from_skeleton.use=true" \
            "augmentation.masking_from_skeleton.n_holes=${n_holes}" \
            "augmentation.masking_from_skeleton.margin=${margin}" \
            "training.devices=[${DEVICE_ID}]" \
            "wandb.project=interaction_ablation_study_augmentation"
        done
      done
    else
      # use_masking=false の場合、従来通りの単一実行
      echo "--------------------------------------------------"
      echo "EXECUTING (Masking Comparison):"
      echo "  - seed: ${seed}"
      echo "  - augmentation.masking_from_skeleton.use: false"
      echo "  - training.devices: [${DEVICE_ID}]"
      echo "--------------------------------------------------"

      python3 "${FILE}" \
        "seed=${seed}" \
        "augmentation.masking_from_skeleton.use=false" \
        "training.devices=[${DEVICE_ID}]" \
        "wandb.project=interaction_ablation_study_augmentation"
    fi
  done
done

echo "All experiments successfully completed."