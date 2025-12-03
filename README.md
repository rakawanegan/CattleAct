# CattleAct: Interaction Understanding for Smart Pasture Management

## Authors

* **rakawanegan** - *Me*
* **Yrainy0615** - *Co-developer*

See also the list of [contributors](https://github.com/rakawanegan/CattleAct/contributors) who participated in this project.

## Overview

This repository hosts **CattleAct**, a data-efficient framework for detecting
behavioral interactions among grazing cattle. The system decomposes rare
interaction events into combinations of individual actions, allowing it to
leverage a large-scale action dataset and fine-tune a unified latent space that
captures both actions and interactions. By combining video analytics with GPS
tracking, CattleAct supports practical smart-livestock workflows such as estrus
detection on large commercial pastures.

Key ideas:
- learn an action latent space using abundant single-cattle clips;
- fine-tune the latent space with contrastive objectives on scarce interaction
	samples;
- fuse dual cattle crops and a contextual crop through a hybrid ViT/CNN stream;
- integrate detections, tracking, GPS alignment, and downstream visualization in
	a single pipeline.

## Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended)
- PyTorch Lightning, Hydra, timm, torchmetrics, torchvision, ultralytics, etc.

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Training Pipeline

Interaction fine-tuning is orchestrated by Hydra configs under `train/conf/` and
executed via `train/interaction_with_image.py`.

### Dataset Preparation

1. Organize cropped cattle interaction samples and metadata as expected by
	 `src.dataset.CattleCroppedInteractionDataset`.
2. Set the root directory and label mapping in `train/conf/data/interaction.yaml`.
3. Verify augmentation options in `train/conf/augmentation/` if you plan to use
	 skeleton-aware masking.

### Launch Training

```bash
cd wrap_monitor_system
python train/interaction_with_image.py \
	training.accelerator=gpu training.devices=[0] \
	wandb.entity=<your_entity> wandb.project=<your_project>
```

Important hyper-parameters (no silent defaults):
- `model.fusion_type`: `attention` or `mlp`.
- `model.pooling_type`: `flatten`, `gap`, `gmp`, or `gap_gmp`.
- `pre_fusion_loss`: specify `name`, `weight`, and the relevant margin or
	temperature.
- `main_loss`: choose `cross_entropy`, `focal`, or `ldam` with all required
	scalars defined.

The script logs metrics to Weights & Biases, saves checkpoints to
`checkpoints_dev/`, and automatically runs evaluation on the best
`val_f1score` checkpoint.

### Variants

- `train/interaction_with_image_no_aug.py` mirrors the same Lightning module
	but disables skeleton-aware augmentation. Override configs as needed via
	Hydra command-line options.

## Demo Applications

Two demo entry points illustrate how the trained models integrate with the full
video+GPS pipeline.

### Full Pipeline Demo

Run `scripts/demo.py` to execute the end-to-end workflow:

1. **Detection & Tracking**: Ultralytics YOLO + DeepSORT generate cattle tracks.
2. **GPS Alignment**: Homography-based projection aligns tracks with pasture
	 GPS traces for individual identification.
3. **Visualization**: Action/action embeddings and interaction predictions are
	 overlaid on video clips, producing annotated MP4 outputs.

Adjust Hydra config `scripts/conf/demo.yaml` (not shown here) to point to video
sources, GPS files, checkpoints, and embedding caches. Launch with:

```bash
python scripts/demo.py output_dir=outputs/demo_run
```

### Short Video Clip Demo

Use `scripts/short_demo.py` to process a focused time window around a target
event:

```bash
python scripts/short_demo.py \
	-- # edit paths inside the script before running
```

It loads the classification-style checkpoints directly, samples frames around a
center timestamp, and writes a short annotated clip for quick inspection.

## Repository Structure (excerpt)

- `train/`: Lightning modules, Hydra configs, training scripts.
- `metric/`: Metric-learning counterparts for embedding analysis.
- `scripts/`: Demo utilities for production-style inference.
- `src/`: Dataset loaders, augmentations, losses, tracking/matching helpers.
- `checkpoints/`: Pretrained backbones and fine-tuned weights (not tracked in
	git).

## Citation

If this work contributes to your research, please cite the CattleAct paper:

```
@inproceedings{CattleAct2025,
	title={CattleAct: Data-Efficient Interaction Detection for Grazing Cattle},
	author={...},
	booktitle={...},
	year={2025}
}
```

## License

See `LICENSE` for details. Contact the authors for commercial usage inquiries.