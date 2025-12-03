import collections
import os
import re
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold

DEBUG = False  # デバッグモードのフラグ


def get_all_action_annotations_entries(
    root_dir, map_label, delete_base_dirs=None, drop_unknown_label=True
):
    full_dataset_entries = list()

    annotation_file = os.path.join(root_dir, "master.csv")

    with open(annotation_file, "r") as f:
        header = f.readline().strip().split(",")
        for line in f:
            fields = line.strip().split(",")
            if len(fields) != len(header):
                continue
            info = dict(zip(header, fields))

            label = map_label.get(info["Label"], -1)
            if drop_unknown_label and label == -1:
                continue
            info["label"] = label

            for delete_base_dir in delete_base_dirs:
                info["image_path"] = info["image_path"].replace(delete_base_dir, "")
                info["pose_path"] = info["pose_path"].replace(delete_base_dir, "")
            info["image_path"] = os.path.join(root_dir, info["image_path"])
            info["pose_path"] = os.path.join(root_dir, info["pose_path"])

            full_dataset_entries.append(info)

    use_additive_lying_dataset = True
    if use_additive_lying_dataset:
        annotation_file = os.path.join(root_dir, "additive_lying_dataset.csv")

        with open(annotation_file, "r") as f:
            header = f.readline().strip().split(",")
            for line in f:
                fields = line.strip().split(",")
                if len(fields) != len(header):
                    continue
                info = dict(zip(header, fields))

                label = map_label.get(info["Label"], -1)
                if drop_unknown_label and label == -1:
                    continue
                info["label"] = label

                for delete_base_dir in delete_base_dirs:
                    info["image_path"] = info["image_path"].replace(delete_base_dir, "")
                    info["pose_path"] = info["pose_path"].replace(delete_base_dir, "")
                info["image_path"] = os.path.join(root_dir, info["image_path"])
                info["pose_path"] = os.path.join(root_dir, info["pose_path"])

            full_dataset_entries.append(info)

    use_additive_riding_dataset = True
    if use_additive_riding_dataset:
        annotation_file = os.path.join(root_dir, "additive_riding_dataset.csv")

        with open(annotation_file, "r") as f:
            header = f.readline().strip().split(",")
            for line in f:
                fields = line.strip().split(",")
                if len(fields) != len(header):
                    continue
                info = dict(zip(header, fields))

                label = map_label.get(info["Label"], -1)
                if drop_unknown_label and label == -1:
                    continue
                info["label"] = label

                for delete_base_dir in delete_base_dirs:
                    info["image_path"] = info["image_path"].replace(delete_base_dir, "")
                    info["pose_path"] = info["pose_path"].replace(delete_base_dir, "")
                info["image_path"] = os.path.join(root_dir, info["image_path"])
                info["pose_path"] = os.path.join(root_dir, info["pose_path"])

                full_dataset_entries.append(info)

    return full_dataset_entries


def get_all_interaction_annotations_entries(
    root_dir,
    map_label,
    delete_base_dirs=None,
    use_more_than_three_cattles=False,
    supplemental_map_label=None,
):
    full_dataset_entries = list()
    supplemental_map_label = {
        "no_interaction": 0,
        "interest": 1,
        "conflict": 2,
        "mount": 3,
    }
    annotation_file = os.path.join(root_dir, "master_v3.csv")

    with open(annotation_file, "r") as f:
        header = f.readline().strip().split(",")
        for i, line in enumerate(f):
            fields = line.strip().split(",")
            if len(fields) != len(header):
                continue
            info = dict(zip(header, fields))

            if (
                not use_more_than_three_cattles
                and "more_than_three" in info["label_v1"]
            ):
                continue

            info["label"] = map_label.get(info["label_v1"], -1)
            if info["label"] == -1 and info["label_v1"] == "interaction":
                info["label"] = map_label.get(info["label_v2"], -1)
            elif len(map_label) == 2 and info["label"] == 1:
                info["supplemental_label"] = supplemental_map_label.get(
                    info["label_v2"], -1
                )

            if info["label"] == -1:
                continue

            # BBox情報を追加
            info["bbox1_xyxy"] = info.get("bbox1_xyxy", "[0 0 0 0]")
            info["bbox2_xyxy"] = info.get("bbox2_xyxy", "[0 0 0 0]")
            info["merged_bbox_xyxy"] = info.get("merged_bbox_xyxy", "[0 0 0 0]")

            for delete_base_dir in delete_base_dirs:
                info["image_path"] = info["image_path"].replace(delete_base_dir, "")
                info["pose_path_1"] = info["pose_path_1"].replace(delete_base_dir, "")
                info["pose_path_2"] = info["pose_path_2"].replace(delete_base_dir, "")

            info["image_path"] = os.path.join(root_dir, info["image_path"])
            info["pose_path_1"] = os.path.join(root_dir, info["pose_path_1"])
            info["pose_path_2"] = os.path.join(root_dir, info["pose_path_2"])

            if not os.path.exists(info["image_path"]):
                print(f"[WARN] Image path does not exist: {info['image_path']}")
                continue

            full_dataset_entries.append(info)

    return full_dataset_entries


def split_action_dataset_entries(full_dataset_entries, split_type="filename"):
    train_entries, val_entries, test_entries = [], [], []
    if split_type == "date":
        for entry in full_dataset_entries:
            if "train" in entry["image_path"]:
                train_entries.append(entry)
            elif "val" in entry["image_path"]:
                val_entries.append(entry)
            elif "test" in entry["image_path"]:
                test_entries.append(entry)
    elif split_type == "stratified":
        train_val_entries = []
        for entry in full_dataset_entries:
            if "test" in entry["image_path"]:
                test_entries.append(entry)
            else:
                train_val_entries.append(entry)
        # ラベルごとに分割
        train_entries, val_entries = train_test_split(
            train_val_entries,
            test_size=0.2,
            stratify=[e["label"] for e in train_val_entries],
            random_state=42,
        )
    else:
        raise NotImplementedError(f"Unknown split_type: {split_type}")

    return train_entries, val_entries, test_entries


def split_interaction_dataset_entries(full_dataset_entries, split_type="handle"):
    if split_type == "video_ids":
        for entry in full_dataset_entries:
            # 2. 動画クリップIDごとにエントリをグループ化
            entries_by_clip = collections.defaultdict(list)
            for entry in full_dataset_entries:
                filename = os.path.basename(entry["image_path"])
                clip_id = "_".join(filename.split("_")[:3])
                entries_by_clip[clip_id].append(entry)

            # 3. 動画クリップIDのリストを時系列順に学習・検証・テスト用に分割
            clip_ids = sorted(list(entries_by_clip.keys()))  # 時系列順にするためソート
            train_val_clip_ids, test_clip_ids = train_test_split(
                clip_ids, test_size=0.2, shuffle=False
            )
            train_clip_ids, val_clip_ids = train_test_split(
                train_val_clip_ids, test_size=(1 / 8), shuffle=False
            )

            # 4. 分割されたIDに基づいて、エントリリストを再構築
            train_entries = [
                entry
                for clip_id in train_clip_ids
                for entry in entries_by_clip[clip_id]
            ]
            val_entries = [
                entry for clip_id in val_clip_ids for entry in entries_by_clip[clip_id]
            ]
            test_entries = [
                entry for clip_id in test_clip_ids for entry in entries_by_clip[clip_id]
            ]
    elif split_type == "stratified":
        full_dataset_df = pd.DataFrame(full_dataset_entries)
        full_dataset_df["entries"] = full_dataset_df["source_video"].str.replace(
            ".avi", ""
        )

        test_1_entries = full_dataset_df.sort_values(
            by=["entries", "frame_number"], ascending=False
        ).iloc[:149]["entries"]
        conflict_df = full_dataset_df.loc[full_dataset_df["label"] == 2]
        test_conflict_entries = sorted(conflict_df["entries"].unique())[:2]
        test_entries = sorted(set(test_1_entries) | set(test_conflict_entries))
        test_df = full_dataset_df[full_dataset_df["entries"].isin(test_entries)]
        train_val_df = full_dataset_df.drop(test_df.index)

        X = train_val_df.index  # 特徴量としてはインデックスを利用
        y = train_val_df["label"]
        groups = train_val_df["entries"]

        # --- 第一段階：TrainとTemp(Val+Test)への分割 (80% / 20%) ---
        # n_splits=5とすることで、1/5 (20%)をテスト用に分割する
        n_splits_train_test = 5
        sgkf_train_test = StratifiedGroupKFold(
            n_splits=n_splits_train_test, shuffle=True, random_state=42
        )

        # イテレータから最初の分割を取得
        train_idx, val_idx = next(sgkf_train_test.split(X, y, groups))

        # DataFrameを分割
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

        train_entries = train_df.to_dict("records")
        val_entries = val_df.to_dict("records")
        test_entries = test_df.to_dict("records")
    elif split_type == "handle":
        test_videos = [
            # version 1
            "2025-03-05 17-10-00~17-20-00.avi",
            "2019-03-23 16-50-00~17-00-00.avi",
            # version 2
            '2025-03-05 14-10-00~14-19-59.avi',
            '2025-03-05 16-40-00~16-50-00.avi',
        ]

        val_videos = [
            # version 1
            "2025-03-05 17-30-00~17-39-59.avi",
            "2021-09-24 08-10-00~08-20-00.avi",
            # version 2
            '2025-03-05 17-00-00~17-10-00.avi',
        ]

        full_df = pd.DataFrame(full_dataset_entries)

        # DataFrameを分割する。
        test_df = full_df[full_df["source_video"].isin(test_videos)].copy()
        val_df = full_df[full_df["source_video"].isin(val_videos)].copy()
        train_df = full_df[~full_df["source_video"].isin(test_videos + val_videos)].copy()

        train_entries = train_df.to_dict("records")
        val_entries = val_df.to_dict("records")
        test_entries = test_df.to_dict("records")
    else:
        raise NotImplementedError(f"Unknown split_type: {split_type}")

    return train_entries, val_entries, test_entries


class CattleInteractionDataset(Dataset):
    """
    牛のペアのインタラクションを認識するためのデータセットクラスである。
    DataModuleから渡されたデータエントリのリストを受け取り、画像と2頭分の姿勢データを返す。
    クロップされた画像に合わせて骨格座標を補正し、カスタム変換と標準の画像変換を適用する。
    A dataset class for recognizing interactions between pairs of cattle.
    It takes a list of data entries, adjusts skeleton coordinates to match cropped images,
    and applies custom and standard image transforms.
    """

    def __init__(self, entries, transform=None, custom_transform=None, use_pose=True):
        self.entries = entries
        self.transform = transform
        self.custom_transform = custom_transform
        self.use_pose = use_pose
        self.num_instance = 2

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # 画像が破損している場合は、このサンプルは使用不可とする
        try:
            img = Image.open(entry["image_path"]).convert("RGB")
        except (UnidentifiedImageError, OSError, FileNotFoundError) as err:
            if DEBUG:
                print(
                    f"[WARN] Skipping corrupted image at index {idx} ({entry.get('image_path', 'N/A')}). Error: {err}"
                )
            # 破損画像の場合、ダミーデータを返すか、Noneを返してcollate_fnで処理する
            # ここでは、次のインデックスのデータを試みる再帰的な呼び出しを行う（簡易的な代替策）
            return self.__getitem__((idx + 1) % len(self.entries))

        skeleton1_orig, skeleton2_orig = None, None
        if self.use_pose:
            try:
                skeleton1_orig = np.load(entry["pose_path_1"], allow_pickle=True)
            except (IOError, ValueError, pickle.UnpicklingError, EOFError):
                if DEBUG:
                    print(f"[WARN] Failed to load pose_path_1: {entry['pose_path_1']}")
                skeleton1_orig = np.array([])  # 失敗した場合は空の配列を代入

            try:
                skeleton2_orig = np.load(entry["pose_path_2"], allow_pickle=True)
            except (IOError, ValueError, pickle.UnpicklingError, EOFError):
                if DEBUG:
                    print(f"[WARN] Failed to load pose_path_2: {entry['pose_path_2']}")
                skeleton2_orig = np.array([])  # 失敗した場合は空の配列を代入

        # 両方のポーズが有効であるかを判定
        are_poses_valid = (
            skeleton1_orig is not None
            and skeleton1_orig.size > 0
            and skeleton2_orig is not None
            and skeleton2_orig.size > 0
        )

        supplemental_info = dict()
        supplemental_info["image_path"] = entry["image_path"]
        supplemental_info["pose_path_1"] = entry["pose_path_1"]
        supplemental_info["pose_path_2"] = entry["pose_path_2"]
        supplemental_info["merged_bbox_xyxy"] = entry["merged_bbox_xyxy"]
        supplemental_info["bbox1_xyxy"] = entry["bbox1_xyxy"]
        supplemental_info["bbox2_xyxy"] = entry["bbox2_xyxy"]
        supplemental_info["is_pose_valid"] = are_poses_valid
        supplemental_info["supplemental_label"] = entry.get("supplemental_label", -1)

        label = torch.tensor(entry["label"], dtype=torch.long)

        if are_poses_valid:
            try:
                bbox_str = entry["merged_bbox_xyxy"]
                merged_bbox = [int(n) for n in bbox_str.strip("[]").split()]
                crop_x_min, crop_y_min = merged_bbox[0], merged_bbox[1]
            except (ValueError, KeyError, TypeError, AttributeError):
                crop_x_min, crop_y_min = 0, 0

            skeleton1_adj = skeleton1_orig.copy()
            if skeleton1_adj.size > 0:
                skeleton1_adj[:, 0] -= crop_x_min
                skeleton1_adj[:, 1] -= crop_y_min

            skeleton2_adj = skeleton2_orig.copy()
            if skeleton2_adj.size > 0:
                skeleton2_adj[:, 0] -= crop_x_min
                skeleton2_adj[:, 1] -= crop_y_min

            if self.custom_transform:
                img = self.custom_transform(
                    image=img,
                    skeleton1=skeleton1_adj,
                    skeleton2=skeleton2_adj,
                    label=label,
                )

        if self.transform:
            img = self.transform(img)

        if are_poses_valid:
            pose1 = (
                torch.tensor(skeleton1_adj, dtype=torch.float32)
                .permute(1, 0)
                .unsqueeze(1)
                .unsqueeze(-1)
            )
            pose2 = (
                torch.tensor(skeleton2_adj, dtype=torch.float32)
                .permute(1, 0)
                .unsqueeze(1)
                .unsqueeze(-1)
            )
            pose = torch.cat((pose1, pose2), dim=-1)
        else:
            # 骨格情報が無効な場合、後段の処理でエラーを起こさないためのダミーテンソルを生成
            num_coords = 3  # x, y, conf
            num_joints = 17
            pose = torch.zeros(
                (num_coords, 1, num_joints, self.num_instance), dtype=torch.float32
            )

        return img, pose, label, supplemental_info


class CattleCroppedInteractionDataset(Dataset):
    """
    牛のペアのインタラクションを認識するためのデータセットクラスである。
    __getitem__で2頭の牛を個別にクロップし、リサイズと変換を適用した2枚の画像テンソルを返す。
    """

    # --- 変更点1: __init__メソッドに skeleton_aware_transform を追加 ---
    def __init__(self, entries, transform=None, use_pose=True, skeleton_aware_transform=None, is_aware_skeleton=True):
        self.entries = entries
        self.transform = transform
        self.use_pose = use_pose
        self.skeleton_aware_transform = skeleton_aware_transform
        self.is_aware_skeleton = is_aware_skeleton
        self.num_instance = 2

    def __len__(self):
        return len(self.entries)

    def _parse_bbox(self, bbox_str: str) -> list:
        """
        様々な形式のBBox文字列を頑健にパースし、整数のリストとして返却する。
        """
        cleaned_str = (
            bbox_str.replace("[", " ")
            .replace("]", " ")
            .replace(",", " ")
            .replace('"', "")
            .replace("'", "")
        )
        return [int(num) for num in cleaned_str.split() if num]

    def __getitem__(self, idx):
        entry = self.entries[idx]

        try:
            # 元画像を読み込む
            cropped_img = Image.open(entry["image_path"]).convert("RGB")
        except (UnidentifiedImageError, OSError, FileNotFoundError) as err:
            if DEBUG:
                print(
                    f"[WARN] Skipping corrupted image: {entry['image_path']}. Error: {err}"
                )
            return self.__getitem__((idx + 1) % len(self.entries))

        skeleton1_orig, skeleton2_orig = None, None
        if self.use_pose:
            try:
                skeleton1_orig = np.load(entry["pose_path_1"], allow_pickle=True)
            except (IOError, ValueError, pickle.UnpicklingError, EOFError):
                if DEBUG:
                    print(f"[WARN] Failed to load pose_path_1: {entry['pose_path_1']}")
                skeleton1_orig = np.array([])  # 失敗した場合は空の配列を代入

            try:
                skeleton2_orig = np.load(entry["pose_path_2"], allow_pickle=True)
            except (IOError, ValueError, pickle.UnpicklingError, EOFError):
                if DEBUG:
                    print(f"[WARN] Failed to load pose_path_2: {entry['pose_path_2']}")
                skeleton2_orig = np.array([])  # 失敗した場合は空の配列を代入

        # 両方のポーズが有効であるかを判定
        are_poses_valid = (
            skeleton1_orig is not None
            and skeleton1_orig.size > 0
            and skeleton2_orig is not None
            and skeleton2_orig.size > 0
        )

        # --- 変更点1: 骨格座標の調整処理をこの位置に移動 ---
        skeleton1_adj, skeleton2_adj = None, None
        if are_poses_valid:
            try:
                bbox_str = entry["merged_bbox_xyxy"]
                merged_bbox = [int(n) for n in bbox_str.strip("[]").split()]
                crop_x_min, crop_y_min = merged_bbox[0], merged_bbox[1]
            except (ValueError, KeyError, TypeError, AttributeError):
                crop_x_min, crop_y_min = 0, 0

            # バウンディングボックスに合わせて骨格座標を調整する
            skeleton1_adj = skeleton1_orig.copy()
            skeleton1_adj[:, 0] -= crop_x_min
            skeleton1_adj[:, 1] -= crop_y_min

            skeleton2_adj = skeleton2_orig.copy()
            skeleton2_adj[:, 0] -= crop_x_min
            skeleton2_adj[:, 1] -= crop_y_min

        # --- 変更点2: 画像読み込み直後に骨格情報に基づくマスキングを適用 ---
        if self.skeleton_aware_transform is not None:
            if self.is_aware_skeleton:
                # 有効な骨格情報が存在する場合のみ変換処理を実行する
                if are_poses_valid:
                    label = entry.get('label')
                    if label is not None:
                        # 変換関数には調整済みの座標を渡す
                        output_img = self.skeleton_aware_transform(
                            cropped_img, skeleton1_adj, skeleton2_adj, label
                        )
                        cropped_img = output_img
            else:
                # 骨格情報を考慮しない変換を適用する
                cropped_img = self.skeleton_aware_transform(cropped_img, None, None)

        # --- 以下、既存のBBoxパースおよびクロップ処理 ---
        try:
            merged_bbox = self._parse_bbox(entry["merged_bbox_xyxy"])
            bbox1 = self._parse_bbox(entry["bbox1_xyxy"])
            bbox2 = self._parse_bbox(entry["bbox2_xyxy"])
        except (ValueError, AttributeError) as e:
            if DEBUG:
                print(
                    f"[WARN] Failed to parse bbox for {entry['image_path']}. Error: {e}"
                )
            return self.__getitem__((idx + 1) % len(self.entries))

        if len(merged_bbox) != 4 or len(bbox1) != 4 or len(bbox2) != 4:
            if DEBUG:
                print(
                    f"[WARN] Invalid bbox format for {entry['image_path']}. Skipping."
                )
            return self.__getitem__((idx + 1) % len(self.entries))

        offset_x, offset_y = merged_bbox[0], merged_bbox[1]
        rel_bbox1 = (
            bbox1[0] - offset_x,
            bbox1[1] - offset_y,
            bbox1[2] - offset_x,
            bbox1[3] - offset_y,
        )
        rel_bbox2 = (
            bbox2[0] - offset_x,
            bbox2[1] - offset_y,
            bbox2[2] - offset_x,
            bbox2[3] - offset_y,
        )

        img1 = cropped_img.crop(rel_bbox1)
        img2 = cropped_img.crop(rel_bbox2)

        if self.transform:
            cropped_img = self.transform(cropped_img)
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor(entry["label"], dtype=torch.long)
        supplemental_info = {k: v for k, v in entry.items()}

        if are_poses_valid:
            # このブロックでは skeleton1_adj と skeleton2_adj は常に定義されている
            pose1 = (
                torch.tensor(skeleton1_adj, dtype=torch.float32)
                .permute(1, 0)
                .unsqueeze(1)
                .unsqueeze(-1)
            )
            pose2 = (
                torch.tensor(skeleton2_adj, dtype=torch.float32)
                .permute(1, 0)
                .unsqueeze(1)
                .unsqueeze(-1)
            )
            pose = torch.cat((pose1, pose2), dim=-1)
        else:
            num_coords = 3
            num_joints = 17
            pose = torch.zeros(
                (num_coords, 1, num_joints, self.num_instance), dtype=torch.float32
            )
        supplemental_info['pose'] = pose

        return img1, img2, cropped_img, label, supplemental_info


class CattleActionDataset(Dataset):
    """牛の行動認識のためのデータセットクラスである。"""

    def __init__(
        self,
        entries,
        label_map,
        image_transform=None,
        custom_image_transform=None,
        pose_transform=None,
    ):
        """
        Args:
            entries (list): データエントリのリストである。
            label_map (dict): ラベル名とインデックスのマッピング辞書である。
            image_transform (callable, optional): 標準の画像変換である。
            custom_image_transform (callable, optional): 骨格情報も利用するカスタム変換である。
        """
        self.entries = entries
        self.label_map = label_map
        self.image_transform = image_transform
        self.custom_image_transform = custom_image_transform
        self.pose_transform = pose_transform
        self.edge_index = torch.tensor(
            [
                (0, 1),
                (0, 2),
                (1, 2),
                (2, 3),
                (3, 4),
                (3, 5),
                (5, 6),
                (6, 7),
                (3, 8),
                (8, 9),
                (9, 10),
                (4, 11),
                (11, 12),
                (12, 13),
                (4, 14),
                (14, 15),
                (15, 16),
            ]
        )
        self.num_instance = 1

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        for _ in range(len(self.entries)):
            entry = self.entries[idx]
            img_path = entry["image_path"]
            try:
                img = Image.open(img_path).convert("RGB")
                break
            except (UnidentifiedImageError, OSError) as err:
                print(f"[WARN] Unrecognized image, skipping: {img_path}, error: {err}")
                idx = (idx + 1) % len(self.entries)
        else:
            raise RuntimeError("Failed to load any valid image in the dataset.")

        supplemental_info = dict()
        supplemental_info["image_path"] = img_path
        supplemental_info["pose_path"] = entry["pose_path"]

        # 骨格情報をロードし、失敗した場合は空配列とする
        try:
            pose_array = np.load(entry["pose_path"], allow_pickle=True)
        except (
            UnidentifiedImageError,
            OSError,
            FileNotFoundError,
            pickle.UnpicklingError,
            ValueError,
            EOFError,
        ) as err:
            pose_array = np.array([])

        is_pose_valid = (
            pose_array.size > 0
            and not np.isnan(pose_array).any()
            and not np.isinf(pose_array).any()
        )
        supplemental_info["is_pose_valid"] = is_pose_valid

        if not is_pose_valid:
            pose_array = np.array([])

        label = torch.tensor(entry["label"], dtype=torch.long)

        # 骨格情報が有効な場合に限り、カスタムのデータ拡張を適用する
        if self.custom_image_transform and is_pose_valid:
            # この変換で用いるため、一時的にテンソル化する
            skeleton_for_aug = torch.tensor(pose_array, dtype=torch.float32)
            img = self.custom_image_transform(
                image=img, skeleton=skeleton_for_aug, label=label
            )

        # 骨格情報の変換を適用する
        if self.pose_transform and is_pose_valid:
            pose_array = self.pose_transform(pose_array)

        if is_pose_valid:
            # calc image size from image_path
            img_size = Image.open(entry["image_path"]).size
            # normalize pose coodinate with image size 0-1-2 is x-y-conf
            pose_array[:, 0] = pose_array[:, 0] / img_size[0]  # x
            pose_array[:, 1] = pose_array[:, 1] / img_size[1]  # y

            supplemental_info["image_size"] = img_size
        else:
            supplemental_info["image_size"] = (0, 0)

        # 標準の画像変換を適用する
        if self.image_transform:
            img = self.image_transform(img)

        # poseテンソルを生成する
        if is_pose_valid:
            skeleton = torch.tensor(pose_array, dtype=torch.float32)
            # 元のコードに基づき、(C, 1, V, M) 形式のテンソルを生成する
            # skeleton: (V, C) -> permute -> (C, V) -> unsqueeze -> (C, 1, V, 1) -> repeat -> (C, 1, V, M)
            pose = skeleton.permute(1, 0).unsqueeze(1).unsqueeze(-1)
            pose = pose.repeat(1, 1, 1, self.num_instance)
        else:
            # 骨格情報が無効な場合、後段の処理でエラーを起こさないためのダミーテンソルを生成する
            num_coords = 3  # x, y, confidence座標
            num_joints = 17  # データセットの関節数
            # (C, 1, V, M) の形状に合わせたゼロテンソル
            pose = torch.zeros(
                (num_coords, 1, num_joints, self.num_instance), dtype=torch.float32
            )

        return img, pose, label, supplemental_info


import os
import re
import pandas as pd

# 【ユーザー保存情報: 開発の基本理念】に基づき、
# エラーを握りつぶさず、特定パターンを優先処理することで品質と正確性を担保します。

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Generate Table 1: Categories and sample counts for behaviors (Fixed)
    # -------------------------------------------------------------------------

    action_root_dir = "/mnt/nfs/processed/action_data"
    interaction_root_dir = "/mnt/nfs/processed/interaction"
    
    delete_base_dirs = []

    print(f"Loading Action data from: {action_root_dir}")
    print(f"Loading Interaction data from: {interaction_root_dir}")

    # --- 共通の集計ロジック関数 (パターンマッチング強化版) ---
    def calculate_counts_with_grouping(entries):
        if not entries:
            return pd.Series(dtype=int), pd.Series(dtype=int)

        df = pd.DataFrame(entries)

        # 1. 画像パスから 'source_video' と 'frame_number' を生成
        if 'source_video' not in df.columns or 'frame_number' not in df.columns:
            def extract_info(path):
                filename = os.path.basename(path)
                name, ext = os.path.splitext(filename)
                
                # --- Strategy A: Explicit "frame" keyword (優先) ---
                # 例: "2019..._frame_00000090_pair_01" -> Video="2019...", Frame=90
                # "frame_" の後ろの数字をキャプチャし、それより前を動画名とする
                match_frame = re.search(r'(.*)_frame_(\d+)', name)
                if match_frame:
                    video_name = match_frame.group(1)
                    frame_num = int(match_frame.group(2))
                    return pd.Series([video_name, frame_num, False])

                # --- Strategy B: Generic suffix number (フォールバック) ---
                # 例: "CowVideo_123" -> Video="CowVideo", Frame=123
                # 従来のロジック。末尾の数字をフレーム番号とする
                match_suffix = re.search(r'^(.*)[_-](\d+)$', name)
                if match_suffix:
                    video_name = match_suffix.group(1)
                    frame_num = int(match_suffix.group(2))
                    return pd.Series([video_name, frame_num, False])

                # --- Strategy C: Parse Failed ---
                return pd.Series([name, 0, True]) 

            # パスから情報を抽出
            extracted = df['image_path'].apply(extract_info)
            extracted.columns = ['source_video', 'frame_number', 'parse_error']
            
            # エラーチェック
            error_count = extracted['parse_error'].sum()
            if error_count > 0:
                print(f"  [Warning] Failed to parse frame number for {error_count} images. Treated as single-frame actions.")

            # 既存カラムを優先しつつ結合
            if 'source_video' not in df.columns:
                df['source_video'] = extracted['source_video']
            if 'frame_number' not in df.columns:
                df['frame_number'] = extracted['frame_number']

        # 2. 型変換
        df['label'] = df['label'].astype(int)
        df['frame_number'] = df['frame_number'].astype(int)

        # 3. ソート (厳密に適用)
        df_sorted = df.sort_values(['source_video', 'label', 'frame_number']).reset_index(drop=True)

        # 4. グループ化の条件判定
        #    - 動画が変わった
        #    - ラベルが変わった
        #    - フレーム間隔が100を超えた (30fps換算で約3.3秒)
        is_new_group_start = (df_sorted['source_video'] != df_sorted['source_video'].shift(1)) | \
                             (df_sorted['label'] != df_sorted['label'].shift(1)) | \
                             (df_sorted['frame_number'].diff() > 100)

        # 5. グループID付与
        df_sorted['group'] = is_new_group_start.cumsum()

        # 6. 集計
        action_counts_series = df_sorted.groupby('label')['group'].nunique()
        image_counts_series = df_sorted['label'].value_counts()

        return image_counts_series, action_counts_series


    # --- 1. Individual Behaviors Counting ---
    action_map = {
        "grazing": 0,
        "standing": 1,
        "lying": 2,
        "riding": 3
    }
    
    # ※関数定義は外部にある前提
    action_entries = get_all_action_annotations_entries(
        action_root_dir, action_map, delete_base_dirs, drop_unknown_label=True
    )

    print("Processing Individual Behaviors...")
    indiv_img_counts, indiv_act_counts = calculate_counts_with_grouping(action_entries)


    # --- 2. Interactions Counting ---
    interaction_map = {
        "no_interaction": 0,
        "interest": 1,
        "conflict": 2,
        "mount": 3,
    }
    
    # ※関数定義は外部にある前提
    interaction_entries = get_all_interaction_annotations_entries(
        interaction_root_dir, 
        interaction_map, 
        delete_base_dirs, 
        use_more_than_three_cattles=False
    )

    print("Processing Interactions...")
    inter_img_counts, inter_act_counts = calculate_counts_with_grouping(interaction_entries)


    # --- 3. Print Formatting (Table 1) ---
    print("\n")
    print(f"Table 1. Categories and sample counts for behaviors.")
    print("=" * 72)
    # Header
    print(f"{'Category':<12} {'Images':>8} {'Actions':>8} | {'Category':<15} {'Images':>8} {'Actions':>8}")
    print("-" * 72)
    print(f"{'Individual Behaviors':<30} | {'Interactions':<33}")
    
    left_order = ["grazing", "standing", "lying", "riding"]
    right_order = ["no_interaction", "interest", "conflict", "mount"]
    
    subtotal_left_img = 0
    subtotal_left_act = 0
    subtotal_right_img = 0
    subtotal_right_act = 0

    for i in range(4):
        # Left Side (Individual)
        l_name = left_order[i]
        l_idx = action_map[l_name]
        
        l_img_count = indiv_img_counts.get(l_idx, 0)
        l_act_count = indiv_act_counts.get(l_idx, 0)
        
        subtotal_left_img += l_img_count
        subtotal_left_act += l_act_count

        # Right Side (Interaction)
        r_name = right_order[i]
        r_idx = interaction_map[r_name]
        
        r_img_count = inter_img_counts.get(r_idx, 0)
        r_act_count = inter_act_counts.get(r_idx, 0)

        subtotal_right_img += r_img_count
        subtotal_right_act += r_act_count

        print(
            f"  {l_name:<10} {l_img_count:>8} {l_act_count:>8} | "
            f"  {r_name:<13} {r_img_count:>8} {r_act_count:>8}"
        )

    print("-" * 72)
    print(
        f"  {'Subtotal':<10} {subtotal_left_img:>8} {subtotal_left_act:>8} | "
        f"  {'Subtotal':<13} {subtotal_right_img:>8} {subtotal_right_act:>8}"
    )
    print("=" * 72)