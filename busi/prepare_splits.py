#!/usr/bin/env python
import argparse
import json
import os

from busi.data import scan_busi_samples, stratified_split, summarize_splits, write_split_manifests


def main():
    parser = argparse.ArgumentParser(description="Prepare BUSI train/val/test split manifests.")
    parser.add_argument(
        "--dataset-root",
        default="/share/baihexiang/datasets/BUSI/Dataset_BUSI_with_GT",
        help="BUSI root path containing benign/malignant/normal folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="./datasets/busi_splits",
        help="Output directory for split manifests.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    samples = scan_busi_samples(args.dataset_root)
    if len(samples) == 0:
        raise RuntimeError("No BUSI samples found under dataset root.")

    splits = stratified_split(
        samples=samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    write_split_manifests(splits, args.output_dir)

    summary = summarize_splits(splits)
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print("BUSI split manifests generated.")
    print(f"Output dir: {args.output_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

