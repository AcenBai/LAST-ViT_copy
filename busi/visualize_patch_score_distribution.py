#!/usr/bin/env python
"""
Visualize BUSI foreground/background patch-score distributions for ViT or DenseViT.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader
from typing import Any

try:
    from omegaconf.base import ContainerMetadata
    from omegaconf.dictconfig import DictConfig
    from omegaconf.listconfig import ListConfig

    torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata])
except Exception:
    pass

_CUR_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_CUR_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from busi.pipeline import BUSIManifestDataset, build_busi_vit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize BUSI patch-score distributions and compute PIM/PIB."
    )
    parser.add_argument("--split-manifest", type=str, default=None)
    parser.add_argument("--split-manifests", type=str, nargs="+", default=None)
    parser.add_argument("--manifest-dir", type=str, default="./datasets/busi_splits")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["vit", "dense", "dense_inv", "resnet50"],
        required=True,
    )
    parser.add_argument(
        "--resnet-feature-layer",
        type=str,
        choices=["layer3", "layer4"],
        default="layer4",
    )
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--majority-ratio", type=float, default=0.5)
    parser.add_argument("--resnet-fg-threshold", type=float, default=0.05)
    parser.add_argument("--normalize-per-image", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-plot", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    return parser.parse_args()


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        normalized[new_key] = value
    return normalized


def load_checkpoint(path: str) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint)
        if isinstance(state_dict, dict):
            return normalize_state_dict_keys(state_dict)
    raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")


def per_image_minmax(scores: torch.Tensor) -> torch.Tensor:
    min_values = scores.min(dim=1, keepdim=True).values
    max_values = scores.max(dim=1, keepdim=True).values
    denom = (max_values - min_values).clamp_min(1e-8)
    return (scores - min_values) / denom


def mask_to_patch_indices(mask_2d: torch.Tensor, patches_per_side: int, majority_ratio: float) -> np.ndarray:
    if mask_2d.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={tuple(mask_2d.shape)}")
    h, w = mask_2d.shape
    if h % patches_per_side != 0 or w % patches_per_side != 0:
        raise ValueError(f"Mask shape {h}x{w} is not divisible by patches_per_side {patches_per_side}")

    patch_h = h // patches_per_side
    patch_w = w // patches_per_side
    grid_h = patches_per_side
    grid_w = patches_per_side
    patches = mask_2d.view(grid_h, patch_h, grid_w, patch_w)
    covered = patches.sum(dim=(1, 3)).float() / float(patch_h * patch_w)
    active = covered > majority_ratio
    indices = torch.nonzero(active, as_tuple=False)
    if indices.numel() == 0:
        return np.empty((0,), dtype=np.int64)
    return (indices[:, 0] * grid_w + indices[:, 1]).cpu().numpy().astype(np.int64)


def top_patch_center(top_idx: int, image_size: int, patches_per_side: int) -> tuple[int, int]:
    patch_size = image_size // patches_per_side
    y_idx = top_idx // patches_per_side
    x_idx = top_idx % patches_per_side
    cx = int(x_idx * patch_size + (patch_size // 2))
    cy = int(y_idx * patch_size + (patch_size // 2))
    return cx, cy


def infer_grid_hw(num_patches: int) -> tuple[int, int]:
    side = int(round(math.sqrt(num_patches)))
    if side * side != num_patches:
        raise ValueError(f"Expected square patch grid, got num_patches={num_patches}")
    return side, side


def downsample_mask_ratio(mask_2d: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    mask = mask_2d.float().unsqueeze(0).unsqueeze(0)
    ratio = F.interpolate(mask, size=(grid_h, grid_w), mode="area")
    return ratio.squeeze(0).squeeze(0)


def fg_bbox_from_bool_mask(fg_mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    ys, xs = torch.where(fg_mask)
    if ys.numel() == 0:
        return None
    x0 = int(xs.min().item())
    y0 = int(ys.min().item())
    x1 = int(xs.max().item())
    y1 = int(ys.max().item())
    return x0, y0, x1, y1


def smooth_density(hist: np.ndarray) -> np.ndarray:
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel = kernel / kernel.sum()
    return np.convolve(hist, kernel, mode="same")


def resolve_manifest_paths(args: argparse.Namespace) -> list[str]:
    if args.split_manifests:
        paths = [str(Path(p)) for p in args.split_manifests]
    elif args.split_manifest:
        paths = [str(Path(args.split_manifest))]
    else:
        root = Path(args.manifest_dir)
        paths = [str(root / split_name) for split_name in ("train.json", "val.json", "test.json")]

    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Manifest file(s) not found: {joined}")
    return paths


def summarize_scores(fg_scores: np.ndarray, bg_scores: np.ndarray, bins: int = 120) -> dict[str, Any]:
    hist_fg, edges = np.histogram(fg_scores, bins=bins, range=(0.0, 1.0), density=True)
    hist_bg, _ = np.histogram(bg_scores, bins=bins, range=(0.0, 1.0), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "num_foreground_scores": int(fg_scores.size),
        "num_background_scores": int(bg_scores.size),
        "foreground_mean": float(fg_scores.mean()) if fg_scores.size else None,
        "background_mean": float(bg_scores.mean()) if bg_scores.size else None,
        "foreground_q50": float(np.quantile(fg_scores, 0.5)) if fg_scores.size else None,
        "background_q50": float(np.quantile(bg_scores, 0.5)) if bg_scores.size else None,
        "foreground_q90": float(np.quantile(fg_scores, 0.9)) if fg_scores.size else None,
        "background_q90": float(np.quantile(bg_scores, 0.9)) if bg_scores.size else None,
        "hist_bin_centers": centers.tolist(),
        "foreground_density": smooth_density(hist_fg).tolist(),
        "background_density": smooth_density(hist_bg).tolist(),
    }


def make_plot(summary: dict[str, Any], output_plot: Path, title: str) -> None:
    centers = np.array(summary["hist_bin_centers"], dtype=np.float64)
    foreground = np.array(summary["foreground_density"], dtype=np.float64)
    background = np.array(summary["background_density"], dtype=np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.4))
    ax.plot(centers, foreground, color="#1b7f5a", linewidth=2.2, label="Foreground")
    ax.plot(centers, background, color="#c75b39", linewidth=2.2, label="Background")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Normalized Patch Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=240)
    plt.close(fig)


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    manifest_paths = resolve_manifest_paths(args)
    datasets = [
        BUSIManifestDataset(
            manifest_path=manifest_path,
            image_size=args.image_size,
        )
        for manifest_path in manifest_paths
    ]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_busi_vit(
        num_classes=args.num_classes,
        pretrained=False,
        use_dense=args.backbone in {"dense", "dense_inv"},
        image_size=args.image_size,
        backbone_kind=args.backbone,
        resnet_feature_layer=args.resnet_feature_layer,
    )
    state_dict = load_checkpoint(args.checkpoint)
    load_result = model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    fg_scores_all: list[np.ndarray] = []
    bg_scores_all: list[np.ndarray] = []
    pim_hits: list[float] = []
    pib_hits: list[float] = []
    used_images = 0
    skipped_images = 0
    normal_skipped = 0
    lesion_images = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"]
        bboxes = batch["bbox"]

        _, patch_scores = model(images)
        if args.normalize_per_image:
            patch_scores_for_dist = per_image_minmax(patch_scores).cpu().numpy()
        else:
            patch_scores_for_dist = patch_scores.cpu().numpy()

        top_idx = patch_scores.argmax(dim=1).cpu().numpy()
        num_patches = int(patch_scores.shape[1])
        grid_h, grid_w = infer_grid_hw(num_patches)

        for i in range(images.shape[0]):
            label_i = int(batch["label"][i].item())
            if label_i == 2:
                normal_skipped += 1
                skipped_images += 1
                continue

            lesion_images += 1
            mask_2d = masks[i]
            if args.backbone == "resnet50":
                mask_ratio = downsample_mask_ratio(mask_2d=mask_2d, grid_h=grid_h, grid_w=grid_w)
                fg_bool = mask_ratio > float(args.resnet_fg_threshold)
                bg_bool = ~fg_bool
                fg_indices = torch.nonzero(fg_bool.flatten(), as_tuple=False).squeeze(1).cpu().numpy()
                bg_indices = torch.nonzero(bg_bool.flatten(), as_tuple=False).squeeze(1).cpu().numpy()
            else:
                fg_indices = mask_to_patch_indices(
                    mask_2d=mask_2d,
                    patches_per_side=grid_h,
                    majority_ratio=args.majority_ratio,
                )
                all_indices = np.arange(patch_scores_for_dist.shape[1], dtype=np.int64)
                bg_mask = np.ones_like(all_indices, dtype=bool)
                bg_mask[fg_indices] = False
                bg_indices = all_indices[bg_mask]

            if fg_indices.size == 0:
                skipped_images += 1
                continue

            if bg_indices.size == 0:
                skipped_images += 1
                continue

            scores_i = patch_scores_for_dist[i]
            fg_scores_all.append(scores_i[fg_indices])
            bg_scores_all.append(scores_i[bg_indices])
            used_images += 1

            cx, cy = top_patch_center(
                top_idx=int(top_idx[i]),
                image_size=args.image_size,
                patches_per_side=grid_h,
            )
            if args.backbone == "resnet50":
                top_flat = int(top_idx[i])
                top_y = top_flat // grid_w
                top_x = top_flat % grid_w
                mask_ratio = downsample_mask_ratio(mask_2d=mask_2d, grid_h=grid_h, grid_w=grid_w)
                fg_bool = mask_ratio > float(args.resnet_fg_threshold)
                in_mask = float(fg_bool[top_y, top_x].item())
                pim_hits.append(in_mask)

                bbox_ds = fg_bbox_from_bool_mask(fg_bool)
                if bbox_ds is not None:
                    x0, y0, x1, y1 = bbox_ds
                    in_bbox = float((top_x >= x0) and (top_x <= x1) and (top_y >= y0) and (top_y <= y1))
                    pib_hits.append(in_bbox)
            else:
                cy = int(np.clip(cy, 0, args.image_size - 1))
                cx = int(np.clip(cx, 0, args.image_size - 1))
                in_mask = float(mask_2d[cy, cx] > 0)
                pim_hits.append(in_mask)

                x0, y0, x1, y1 = [int(v.item()) for v in bboxes[i]]
                in_bbox = float((cx >= x0) and (cx <= x1) and (cy >= y0) and (cy <= y1))
                pib_hits.append(in_bbox)

    if not fg_scores_all or not bg_scores_all:
        raise RuntimeError("No valid foreground/background patch scores were collected.")

    fg_scores = np.concatenate(fg_scores_all).astype(np.float32)
    bg_scores = np.concatenate(bg_scores_all).astype(np.float32)
    summary = summarize_scores(fg_scores=fg_scores, bg_scores=bg_scores, bins=120)

    pim = float(np.mean(pim_hits)) if pim_hits else float("nan")
    pib = float(np.mean(pib_hits)) if pib_hits else float("nan")

    split_scope = "all-splits" if len(manifest_paths) > 1 else Path(manifest_paths[0]).stem
    title = f"BUSI Patch Score Distribution ({args.backbone}, {split_scope})"
    output_plot = Path(args.output_plot)
    output_json = Path(args.output_json)
    make_plot(summary=summary, output_plot=output_plot, title=title)

    payload = {
        "split_manifest": manifest_paths[0] if len(manifest_paths) == 1 else None,
        "split_manifests": manifest_paths,
        "checkpoint": args.checkpoint,
        "backbone": args.backbone,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "resnet_feature_layer": args.resnet_feature_layer,
        "majority_ratio": args.majority_ratio,
        "resnet_fg_threshold": args.resnet_fg_threshold,
        "normalize_per_image": args.normalize_per_image,
        "model_load": {
            "missing_keys": len(load_result.missing_keys),
            "unexpected_keys": len(load_result.unexpected_keys),
        },
        "diagnostics": {
            "num_input_splits": len(manifest_paths),
            "num_input_images": int(len(dataset)),
            "lesion_images": lesion_images,
            "normal_skipped": normal_skipped,
            "used_images": used_images,
            "skipped_images": skipped_images,
            "pim_count": len(pim_hits),
        },
        "metrics": {
            "pim": pim,
            "pib": pib,
        },
        "summary": summary,
        "plot_path": str(output_plot),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

