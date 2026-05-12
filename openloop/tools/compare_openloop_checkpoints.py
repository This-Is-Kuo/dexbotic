#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pathlib
import subprocess
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openloop.tools.openloop_debug_utils import load_array, per_dim_metrics, save_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--exp_file", default="playground/benchmarks/custom/post_data_01_dm0.py")
    parser.add_argument("--dataset_name", default="post_data_01_default")
    parser.add_argument("--norm_stats", default=None)
    parser.add_argument("--chunk_merge", default="mean", choices=["chunk", "first", "mean", "exp"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_batches", type=int, default=100000)
    parser.add_argument("--binary_dims", default="6,13")
    parser.add_argument("--binary_threshold", type=float, default=0.5)
    parser.add_argument("--single_gpu_id", type=int, default=None)
    return parser.parse_args()


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _gripper_accuracy(pred: np.ndarray, gt: np.ndarray, dim: int, threshold: float) -> float:
    if dim >= pred.shape[1] or dim >= gt.shape[1]:
        return float("nan")
    pred_dim = pred[:, dim]
    gt_dim = gt[:, dim]
    mask = np.isfinite(pred_dim) & np.isfinite(gt_dim)
    if not np.any(mask):
        return float("nan")
    pred_bin = pred_dim[mask] >= threshold
    gt_bin = gt_dim[mask] >= threshold
    return float(np.mean(pred_bin == gt_bin))


def main() -> None:
    args = parse_args()
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    binary_dims = _parse_csv_ints(args.binary_dims)

    summary_rows = []
    for checkpoint in args.checkpoints:
        checkpoint_path = pathlib.Path(checkpoint)
        label = checkpoint_path.name
        ckpt_dir = save_dir / label
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "openloop/eval_openloop.py",
            "--checkpoint",
            str(checkpoint_path),
            "--exp-file",
            args.exp_file,
            "--dataset-name",
            args.dataset_name,
            "--episode-index",
            str(args.episode_id),
            "--chunk_merge",
            args.chunk_merge,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--num-batches",
            str(args.num_batches),
            "--save-arrays",
            "true",
            "--metrics-path",
            str(ckpt_dir / "openloop_metrics.json"),
            "--plot-path",
            str(ckpt_dir / "openloop_raw.png"),
            "--normalized-plot-path",
            str(ckpt_dir / "openloop_norm.png"),
            "--array-dir",
            str(ckpt_dir),
        ]
        if args.single_gpu_id is not None:
            cmd.extend(["--single-gpu-id", str(args.single_gpu_id)])
        if args.norm_stats:
            cmd.extend(["--norm-stats", args.norm_stats])
        subprocess.run(cmd, check=True)

        pred = load_array(ckpt_dir / "pred_norm.npy")
        gt = load_array(ckpt_dir / "gt_norm.npy")
        rows = per_dim_metrics(pred, gt)
        save_csv(rows, ckpt_dir / "per_dim_metrics.csv")

        raw_pred = load_array(ckpt_dir / "pred_raw.npy")
        raw_gt = load_array(ckpt_dir / "gt_raw.npy")
        mean_mae = float(np.nanmean([row["mae"] for row in rows]))
        mean_rmse = float(np.nanmean([row["rmse"] for row in rows]))
        mean_pearson = float(np.nanmean([row["pearson"] for row in rows]))
        summary = {
            "checkpoint": str(checkpoint_path),
            "mean_mae": mean_mae,
            "mean_rmse": mean_rmse,
            "mean_pearson": mean_pearson,
        }
        for dim in binary_dims:
            summary[f"gripper_acc_dim{dim}"] = _gripper_accuracy(raw_pred, raw_gt, dim, args.binary_threshold)
        summary_rows.append(summary)

    summary_path = save_dir / "summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
