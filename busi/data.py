import csv
import json
import os
import random
from typing import Dict, List, Sequence

import numpy as np


LABEL_MAP = {"benign": 0, "malignant": 1, "normal": 2}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _is_mask_file(filename: str) -> bool:
    return "_mask" in filename.lower()


def _base_name_from_mask(mask_name: str) -> str:
    stem = os.path.splitext(mask_name)[0]
    stem = stem.replace("_mask", "")
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        stem = parts[0]
    return stem


def scan_busi_samples(dataset_root: str) -> List[Dict]:
    samples = []
    for class_name in ("benign", "malignant", "normal"):
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = {}
        masks = {}
        for fname in os.listdir(class_dir):
            path = os.path.join(class_dir, fname)
            if not os.path.isfile(path):
                continue
            if not fname.lower().endswith(".png"):
                continue
            if _is_mask_file(fname):
                key = _base_name_from_mask(fname)
                masks.setdefault(key, []).append(path)
            else:
                key = os.path.splitext(fname)[0]
                images[key] = path

        for key, image_path in images.items():
            sample_id = f"{class_name}/{key}"
            sample_masks = sorted(masks.get(key, []))
            samples.append(
                {
                    "id": sample_id,
                    "image_path": image_path,
                    "label_name": class_name,
                    "label": LABEL_MAP[class_name],
                    "mask_paths": sample_masks,
                }
            )
    return samples


def stratified_split(
    samples: Sequence[Dict], train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> Dict[str, List[Dict]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    set_seed(seed)
    by_label: Dict[int, List[Dict]] = {}
    for s in samples:
        by_label.setdefault(int(s["label"]), []).append(s)

    splits = {"train": [], "val": [], "test": []}
    for _, group in by_label.items():
        group = list(group)
        random.shuffle(group)
        n = len(group)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train

        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val : n_train + n_val + n_test])

    for split_name in splits:
        random.shuffle(splits[split_name])
    return splits


def write_split_manifests(splits: Dict[str, List[Dict]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for split_name, rows in splits.items():
        json_path = os.path.join(output_dir, f"{split_name}.json")
        csv_path = os.path.join(output_dir, f"{split_name}.csv")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["id", "image_path", "label_name", "label", "mask_paths"]
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "id": row["id"],
                        "image_path": row["image_path"],
                        "label_name": row["label_name"],
                        "label": row["label"],
                        "mask_paths": ";".join(row["mask_paths"]),
                    }
                )


def summarize_splits(splits: Dict[str, List[Dict]]) -> Dict[str, Dict[str, int]]:
    inv_label = {v: k for k, v in LABEL_MAP.items()}
    summary = {}
    for split_name, rows in splits.items():
        cls_count = {k: 0 for k in LABEL_MAP}
        with_mask = 0
        for row in rows:
            cls_count[inv_label[int(row["label"])]] += 1
            if len(row["mask_paths"]) > 0:
                with_mask += 1
        summary[split_name] = {
            "total": len(rows),
            "with_mask": with_mask,
            "benign": cls_count["benign"],
            "malignant": cls_count["malignant"],
            "normal": cls_count["normal"],
        }
    return summary


def load_manifest(manifest_path: str) -> List[Dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Manifest must be a list: {manifest_path}")
    return data

