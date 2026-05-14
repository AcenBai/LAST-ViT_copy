#!/usr/bin/env python
import argparse
import os
import shlex
import subprocess
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_cmd(cmd):
    print("Running:\n", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def build_base_cmd(num_gpus, config_file):
    return [
        sys.executable,
        os.path.join(PROJECT_ROOT, "cls_pretrain", "lazy_train.py"),
        "--config-file",
        config_file,
        "--num-gpus",
        str(num_gpus),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="BUSI train/eval wrapper (ViT, Dense-ViT, DenseInv-ViT, or ResNet50)."
    )
    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "eval-val", "eval-test", "eval-both"],
        default="train",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--backbone",
        choices=["vit", "dense", "dense_inv", "resnet50"],
        default="vit",
        help="Choose ViT, Dense-ViT, inverse-stability Dense-ViT, or ResNet50.",
    )
    parser.add_argument(
        "--resnet-feature-layer",
        choices=["layer3", "layer4"],
        default="layer4",
        help="Patch-score feature layer for resnet50 backbone.",
    )
    parser.add_argument(
        "--resnet-fg-threshold",
        type=float,
        default=0.05,
        help="Foreground threshold on area-downsampled mask ratio for ResNet feature cells.",
    )
    parser.add_argument(
        "--config-file",
        default="busi/config.py",
        help="LazyConfig path.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint path for evaluation modes.",
    )
    parser.add_argument(
        "--dataset-root",
        default="/share/baihexiang/datasets/BUSI/Dataset_BUSI_with_GT",
        help="BUSI root with benign/malignant/normal folders.",
    )
    parser.add_argument(
        "--split-dir",
        default="./datasets/busi_splits",
        help="Directory containing train/val/test manifest JSON files.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir-base",
        default="output/busi_vit_b16",
        help="Base output directory before appending backbone suffix.",
    )
    args, extra_opts = parser.parse_known_args()

    os.makedirs(args.split_dir, exist_ok=True)
    output_dir = f"{args.output_dir_base}_{args.backbone}"

    if args.mode == "prepare":
        prep_cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "busi", "prepare_splits.py"),
            "--dataset-root",
            args.dataset_root,
            "--output-dir",
            args.split_dir,
            "--train-ratio",
            str(args.train_ratio),
            "--val-ratio",
            str(args.val_ratio),
            "--test-ratio",
            str(args.test_ratio),
            "--seed",
            str(args.seed),
        ]
        run_cmd(prep_cmd)
        return

    common_overrides = [
        f"dataloader.train.dataset.manifest_path={args.split_dir}/train.json",
        f"model.model.use_dense={'True' if args.backbone in ('dense', 'dense_inv') else 'False'}",
        f"model.model.backbone_kind={args.backbone}",
        f"model.model.resnet_feature_layer={args.resnet_feature_layer}",
        f"dataloader.evaluator.metric_backbone_kind={args.backbone}",
        f"dataloader.evaluator.resnet_fg_threshold={args.resnet_fg_threshold}",
        f"train.output_dir={output_dir}",
    ] + extra_opts

    if args.mode == "train":
        cmd = build_base_cmd(args.num_gpus, args.config_file) + common_overrides
        run_cmd(cmd)
        return

    if not args.checkpoint:
        raise ValueError("--checkpoint is required for evaluation modes.")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    def eval_split(split):
        cmd = build_base_cmd(args.num_gpus, args.config_file) + [
            "--eval-only",
            f"train.init_checkpoint={args.checkpoint}",
            f"dataloader.test.dataset.manifest_path={args.split_dir}/{split}.json",
        ] + common_overrides
        run_cmd(cmd)

    if args.mode == "eval-val":
        eval_split("val")
    elif args.mode == "eval-test":
        eval_split("test")
    else:
        eval_split("val")
        eval_split("test")


if __name__ == "__main__":
    main()

