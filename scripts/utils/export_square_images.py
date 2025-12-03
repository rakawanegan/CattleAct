#!/usr/bin/env python3
"""Export action/interaction images into square label folders with Uniform Slicing & White Padding."""
import argparse
import collections
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import yaml
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (  # noqa: E402
    get_all_action_annotations_entries,
    get_all_interaction_annotations_entries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize images with white padding (keep aspect ratio) and export by label using uniform slicing."
    )
    parser.add_argument(
        "--datasets",
        choices=["action", "interaction", "both"],
        default="both",
        help="Dataset(s) to export.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "train/conf/data/multi_task.yaml",
        help="YAML config that contains 'action' and 'interaction' sections.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/sample_image",
        help="Directory where processed images are dumped by label name.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Target square side length in pixels.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of images per label to sample uniformly. Default is 10.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential logging.",
    )
    return parser.parse_args()


def load_dataset_config(config_path: Path, dataset_name: str) -> dict:
    with config_path.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    if dataset_name not in cfg:
        raise KeyError(f"'{dataset_name}' section is missing in {config_path}")

    dataset_cfg = {}
    dataset_cfg.update(cfg.get("common", {}))
    dataset_cfg.update(cfg[dataset_name])
    return dataset_cfg


def resolve_path(path_like: Union[str, Path]) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def ensure_label_lookup(map_label: Dict[str, int]) -> Dict[int, str]:
    lookup = {}
    for label_name, label_value in map_label.items():
        try:
            idx = int(label_value)
        except (TypeError, ValueError):
            continue
        if idx < 0:
            continue
        lookup[idx] = label_name
    if not lookup:
        raise ValueError("Label map does not contain any non-negative class indices.")
    return lookup


def load_entries(
    dataset: str, cfg: Dict[str, Union[str, int, bool, Dict, List]]
) -> List[Dict]:
    root_dir = resolve_path(cfg["root_dir"])
    delete_base_dirs = cfg.get("delete_base_dirs", []) or []

    if dataset == "action":
        entries = get_all_action_annotations_entries(
            root_dir=str(root_dir),
            map_label=cfg["map_label"],
            delete_base_dirs=delete_base_dirs,
            drop_unknown_label=cfg.get("drop_unknown_label", True),
        )
    else:
        entries = get_all_interaction_annotations_entries(
            root_dir=str(root_dir),
            map_label=cfg["map_label"],
            delete_base_dirs=delete_base_dirs,
            use_more_than_three_cattles=cfg.get("use_more_than_three_cattles", False),
        )
    return entries


def save_padded_image(src_path: Path, dst_path: Path, size: int) -> None:
    """
    Resizes image to fit within (size, size) maintaining aspect ratio,
    then pads with WHITE to make it a square.
    """
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        old_size = img.size  # (width, height)

        # アスペクト比を維持するための比率計算
        ratio = float(size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # リサイズ (LANCZOSフィルタ推奨)
        img_resized = img.resize(new_size, Image.LANCZOS)

        # 新しい正方形画像を作成 (背景白: 255, 255, 255)
        new_img = Image.new("RGB", (size, size), (255, 255, 255))

        # 中央寄せのための座標計算
        paste_x = (size - new_size[0]) // 2
        paste_y = (size - new_size[1]) // 2
        
        # 貼り付け
        new_img.paste(img_resized, (paste_x, paste_y))
        
        new_img.save(dst_path)


def export_dataset(
    dataset: str,
    cfg: Dict[str, Union[str, int, bool, Dict, List]],
    output_dir: Path,
    image_size: int,
    overwrite: bool,
    quiet: bool,
    limit: int,
) -> Tuple[collections.Counter, List[str]]:
    label_lookup = ensure_label_lookup(cfg["map_label"])
    raw_entries = load_entries(dataset, cfg)

    # 1. Group entries by label
    entries_by_label: Dict[str, List[dict]] = collections.defaultdict(list)
    for entry in raw_entries:
        label_idx = int(entry["label"])
        label_name = label_lookup.get(label_idx)
        if label_name:
            entries_by_label[label_name].append(entry)

    # 2. Uniform Slicing per label
    target_entries: List[dict] = []
    
    for label_name, entries in entries_by_label.items():
        total_items = len(entries)
        
        if total_items <= limit:
            selected_entries = entries
        else:
            step = total_items // limit
            step = max(1, step)
            selected_entries = entries[::step][:limit]
            
        target_entries.extend(selected_entries)

    # 3. Process the selected subset
    counts: collections.Counter = collections.Counter()
    errors: List[str] = []
    processed_paths: Set[Path] = set()

    for entry in target_entries:
        image_path = resolve_path(entry["image_path"])
        
        label_idx = int(entry["label"])
        label_name = label_lookup[label_idx]

        if image_path in processed_paths:
            continue
        processed_paths.add(image_path)

        if not image_path.exists():
            errors.append(f"missing: {image_path}")
            continue

        target_dir = output_dir / label_name
        target_dir.mkdir(parents=True, exist_ok=True)

        suffix = image_path.suffix.lower() or ".jpg"
        dst_name = f"{dataset}_{image_path.stem}{suffix}"
        dst_path = target_dir / dst_name

        if dst_path.exists() and not overwrite:
            counts[label_name] += 1
            continue

        try:
            save_padded_image(image_path, dst_path, image_size)
        except (UnidentifiedImageError, OSError) as err:
            errors.append(f"failed ({image_path}): {err}")
            continue

        counts[label_name] += 1
        if not quiet:
            print(f"[{dataset}][{label_name}] {counts[label_name]}/{limit}: {image_path} -> {dst_path}")

    return counts, errors


def main() -> None:
    args = parse_args()

    datasets = ["action", "interaction"] if args.datasets == "both" else [args.datasets]
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, collections.Counter] = {}
    aggregated_errors: list[str] = []

    for dataset in datasets:
        dataset_cfg = load_dataset_config(args.config, dataset)
        if "map_label" not in dataset_cfg:
            raise KeyError(f"'map_label' is required in the {dataset} config section.")

        counts, errors = export_dataset(
            dataset=dataset,
            cfg=dataset_cfg,
            output_dir=output_dir,
            image_size=args.image_size,
            overwrite=args.overwrite,
            quiet=args.quiet,
            limit=args.limit,
        )
        summary[dataset] = counts
        aggregated_errors.extend(errors)

    print("\nExport summary (Uniform Slice, White Padding):")
    for dataset, counter in summary.items():
        total = sum(counter.values())
        print(f"  {dataset}: {total} images processed")
        for label, count in sorted(counter.items()):
            print(f"    - {label}: {count}")

    if aggregated_errors:
        print("\nWarnings:")
        for message in aggregated_errors:
            print(f"  {message}")


if __name__ == "__main__":
    main()