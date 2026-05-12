#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-file", default="playground/post_data_01_dm0_deltafix.py")
    parser.add_argument(
        "--output-dir",
        default="/dexbotic/user_checkpoints/dexbotic/custom_dm0/post_data_01_deltafix-0511",
    )
    parser.add_argument("--train-targets", default="150,200,300,400")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--dataset-name", default="post_data_01_default")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--save-dir", default="/dexbotic/openloop/artifacts/resume_eval")
    parser.add_argument("--chunk-merge", default="mean", choices=["chunk", "first", "mean", "exp"])
    parser.add_argument("--single-gpu-id", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--eval-num-batches", type=int, default=100000)
    parser.add_argument("--binary-dims", default="6,13")
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--train-num-workers", type=int, default=None)
    parser.add_argument("--norm-num-workers", type=int, default=None)
    parser.add_argument("--norm-batch-size", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--deepspeed-config", default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _latest_checkpoint(output_dir: pathlib.Path) -> pathlib.Path | None:
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
    return checkpoints[-1] if checkpoints else None


def _checkpoint_step(checkpoint_dir: pathlib.Path) -> int:
    return int(checkpoint_dir.name.split("-")[-1])


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _train_to_step(
    args: argparse.Namespace,
    base_checkpoint: pathlib.Path,
    base_step: int,
    target_step: int,
    env: dict[str, str],
) -> pathlib.Path:
    if target_step <= base_step:
        raise ValueError(f"target_step={target_step} must be greater than base_step={base_step}")
    stage_steps = target_step - base_step
    stage_output_dir = pathlib.Path(args.output_dir) / f"warm_restart_step_{target_step}"
    train_env = env.copy()
    train_env["DEXBOTIC_BASE_MODEL"] = str(base_checkpoint)
    train_env["DEXBOTIC_OUTPUT_DIR"] = str(stage_output_dir)
    train_env["DEXBOTIC_NUM_TRAIN_STEPS"] = str(stage_steps)
    train_env["DEXBOTIC_SAVE_STEPS"] = str(min(50, stage_steps))
    if args.train_batch_size is not None:
        train_env["DEXBOTIC_TRAIN_BATCH_SIZE"] = str(args.train_batch_size)
    if args.grad_accum is not None:
        train_env["DEXBOTIC_GRAD_ACCUM"] = str(args.grad_accum)
    if args.train_num_workers is not None:
        train_env["DEXBOTIC_TRAIN_NUM_WORKERS"] = str(args.train_num_workers)
    if args.norm_num_workers is not None:
        train_env["DEXBOTIC_NORM_NUM_WORKERS"] = str(args.norm_num_workers)
    if args.norm_batch_size is not None:
        train_env["DEXBOTIC_NORM_BATCH_SIZE"] = str(args.norm_batch_size)
    if args.warmup_steps is not None:
        train_env["DEXBOTIC_WARMUP_STEPS"] = str(args.warmup_steps)
    if args.deepspeed_config:
        train_env["DEXBOTIC_DEEPSPEED_CONFIG"] = args.deepspeed_config
    if args.disable_wandb:
        train_env["WANDB_DISABLED"] = "true"
        train_env["DEXBOTIC_WANDB_PROJECT"] = "none"

    cmd = [
        "torchrun",
        f"--nproc_per_node={args.nproc_per_node}",
        args.benchmark_file,
    ]
    _run(cmd, train_env)
    latest = _latest_checkpoint(stage_output_dir)
    if latest is None:
        raise RuntimeError(f"No checkpoint found under {stage_output_dir} after training to {target_step}.")
    return latest


def _evaluate_checkpoint(
    args: argparse.Namespace,
    checkpoint_dir: pathlib.Path,
    target_step: int,
    env: dict[str, str],
) -> pathlib.Path:
    eval_dir = pathlib.Path(args.save_dir) / f"checkpoint-{target_step}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "openloop/eval_openloop.py",
        "--checkpoint",
        str(checkpoint_dir),
        "--exp-file",
        args.benchmark_file,
        "--dataset-name",
        args.dataset_name,
        "--episode-index",
        str(args.episode_index),
        "--chunk_merge",
        args.chunk_merge,
        "--single-gpu-id",
        str(args.single_gpu_id),
        "--batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.eval_num_workers),
        "--num-batches",
        str(args.eval_num_batches),
        "--save-arrays",
        "true",
        "--binary-dims",
        args.binary_dims,
        "--array-dir",
        str(eval_dir),
        "--metrics-path",
        str(eval_dir / "openloop_metrics.json"),
        "--plot-path",
        str(eval_dir / "openloop_raw.png"),
        "--normalized-plot-path",
        str(eval_dir / "openloop_norm.png"),
    ]
    _run(cmd, env)

    _run(
        [
            sys.executable,
            "openloop/tools/debug_openloop_metrics.py",
            "--pred",
            str(eval_dir / "pred_norm.npy"),
            "--gt",
            str(eval_dir / "gt_norm.npy"),
            "--save_csv",
            str(eval_dir / "per_dim_metrics_norm.csv"),
        ],
        env,
    )
    _run(
        [
            sys.executable,
            "openloop/tools/lag_correlation_check.py",
            "--pred",
            str(eval_dir / "pred_norm.npy"),
            "--gt",
            str(eval_dir / "gt_norm.npy"),
            "--max_lag",
            "20",
            "--save_csv",
            str(eval_dir / "lag_metrics.csv"),
        ],
        env,
    )
    return eval_dir


def _summarize(save_dir: pathlib.Path) -> None:
    rows = []
    for ckpt_dir in sorted(save_dir.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1])):
        metrics_path = ckpt_dir / "openloop_metrics.json"
        per_dim_path = ckpt_dir / "per_dim_metrics_norm.csv"
        if not metrics_path.exists() or not per_dim_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        with per_dim_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            per_dim = list(reader)
        mean_pearson = sum(float(row["pearson"]) for row in per_dim) / len(per_dim)
        rows.append(
            {
                "checkpoint": ckpt_dir.name,
                "raw_action_mae": metrics["raw_action_mae"],
                "normalized_action_mae": metrics["normalized_action_mae"],
                "mean_pearson": mean_pearson,
            }
        )
    if not rows:
        return
    summary_path = save_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary to {summary_path}")


def main() -> None:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    targets = _parse_csv_ints(args.train_targets)
    env = os.environ.copy()
    base_checkpoint = _latest_checkpoint(output_dir)
    if base_checkpoint is None:
        raise RuntimeError(f"No checkpoint found under {output_dir}.")
    base_step = _checkpoint_step(base_checkpoint)

    for target in targets:
        eval_dir = save_dir / f"checkpoint-{target}"
        if eval_dir.exists() and (eval_dir / "openloop_metrics.json").exists():
            print(f"Evaluation for checkpoint-{target} already exists; skipping.")
            base_step = target
            continue
        if target <= base_step:
            print(f"Target step {target} already covered by current base checkpoint-{base_step}; skipping training.")
            _evaluate_checkpoint(args, base_checkpoint, target, env)
            base_step = target
            continue
        base_checkpoint = _train_to_step(args, base_checkpoint, base_step, target, env)
        base_step = target
        _evaluate_checkpoint(args, base_checkpoint, target, env)

    _summarize(save_dir)


if __name__ == "__main__":
    main()
