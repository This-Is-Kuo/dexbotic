#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import megfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-file", default="playground/benchmarks/custom/post_data_01_dm0.py")
    parser.add_argument("--jsonl-dir", default="/dexbotic/data/post_data_01_dexdata/jsonl")
    parser.add_argument("--video-dir", default="/dexbotic/data/post_data_01_dexdata/video")
    parser.add_argument("--episode-indices", default="0")
    parser.add_argument("--dataset-name", default="post_data_01_overfit_debug")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-train-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-model", default="/dexbotic/checkpoints/DM0-base")
    parser.add_argument("--norm-stats", default=None)
    parser.add_argument("--run-openloop", action="store_true")
    parser.add_argument("--openloop-chunk-merge", default="mean", choices=["chunk", "first", "mean", "exp"])
    return parser.parse_args()


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _load_module_from_file(file_path: str):
    path = pathlib.Path(file_path).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wrap_norm_stats_if_needed(path: str, dst_dir: pathlib.Path) -> str:
    with megfile.smart_open(path, "r") as f:
        data = json.load(f)
    if "norm_stats" in data:
        return path
    wrapped = dst_dir / "wrapped_norm_stats.json"
    with wrapped.open("w") as f:
        json.dump({"norm_stats": data}, f, indent=2)
    return str(wrapped)


def _create_subset_dataset(
    jsonl_dir: pathlib.Path,
    video_dir: pathlib.Path,
    dataset_name: str,
    episode_indices: list[int],
    work_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    files = sorted(jsonl_dir.glob("episode_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No episode_*.jsonl found under {jsonl_dir}")
    unknown = [idx for idx in episode_indices if idx < 0 or idx >= len(files)]
    if unknown:
        raise ValueError(f"Episode indices out of range: {unknown}")

    subset_jsonl_dir = work_dir / "jsonl_subset"
    subset_jsonl_dir.mkdir(parents=True, exist_ok=True)
    for idx in episode_indices:
        src = files[idx]
        dst = subset_jsonl_dir / src.name
        if not dst.exists():
            os.symlink(src, dst)

    register_dir = work_dir / "data_source"
    register_dir.mkdir(parents=True, exist_ok=True)
    register_path = register_dir / "overfit_subset.py"
    register_contents = "\n".join(
        [
            "from dexbotic.data.data_source.register import register_dataset",
            "",
            f"DATASET = {{'default': {{'data_path_prefix': '{video_dir}', 'annotations': '{subset_jsonl_dir}', 'frequency': 1}}}}",
            "meta_data = {'non_delta_mask': [6, 13], 'periodic_mask': None, 'periodic_range': None}",
            f"register_dataset(DATASET, meta_data=meta_data, prefix='{dataset_name}')",
            "",
        ]
    )
    with register_path.open("w") as f:
        f.write(register_contents)
    return subset_jsonl_dir, register_path


def _import_registration(register_path: pathlib.Path) -> None:
    spec = importlib.util.spec_from_file_location(register_path.stem, register_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import registration file {register_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _latest_checkpoint(output_dir: pathlib.Path) -> pathlib.Path | None:
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
    return checkpoints[-1] if checkpoints else None


def main() -> None:
    args = parse_args()
    benchmark_module = _load_module_from_file(args.benchmark_file)
    exp = benchmark_module.DM0Exp()

    work_dir = pathlib.Path(tempfile.mkdtemp(prefix="dm0_overfit_"))
    jsonl_dir = pathlib.Path(args.jsonl_dir)
    video_dir = pathlib.Path(args.video_dir)
    episode_indices = _parse_csv_ints(args.episode_indices)
    _, register_path = _create_subset_dataset(jsonl_dir, video_dir, args.dataset_name, episode_indices, work_dir)
    _import_registration(register_path)
    os.environ["DEXBOTIC_DATA_PATH"] = str(register_path.parent)

    exp.model_config.model_name_or_path = args.base_model
    exp.data_config.dataset_name = f"{args.dataset_name}_default"
    exp.trainer_config.output_dir = args.output_dir
    exp.trainer_config.num_train_steps = args.num_train_steps
    exp.trainer_config.save_steps = args.save_steps
    exp.trainer_config.save_total_limit = 20
    exp.trainer_config.per_device_train_batch_size = args.batch_size
    exp.trainer_config.gradient_accumulation_steps = args.grad_accum
    exp.trainer_config.dataloader_num_workers = args.num_workers
    exp.trainer_config.logging_steps = 1
    exp.inference_config.model_name_or_path = args.output_dir
    exp.inference_config.non_delta_mask = [6, 13]
    exp.inference_config.action_dim = 14

    if args.norm_stats:
        wrapped = _wrap_norm_stats_if_needed(args.norm_stats, work_dir)
        exp.data_config.auto_norm = False
        exp.data_config.action_config.statistic_mapping = wrapped
    else:
        exp.data_config.auto_norm = True
        exp.data_config.action_config.statistic_mapping = None

    print(f"Running one-episode overfit training on episodes {episode_indices}")
    print(f"Temporary registration file: {register_path}")
    print(f"Output directory: {args.output_dir}")
    exp.train()

    if args.run_openloop:
        latest_checkpoint = _latest_checkpoint(pathlib.Path(args.output_dir))
        checkpoint = latest_checkpoint or pathlib.Path(args.output_dir)
        cmd = [
            sys.executable,
            "openloop/eval_openloop.py",
            "--checkpoint",
            str(checkpoint),
            "--exp-file",
            args.benchmark_file,
            "--dataset-name",
            exp.data_config.dataset_name,
            "--episode-index",
            "0",
            "--chunk_merge",
            args.openloop_chunk_merge,
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--num-batches",
            "100000",
            "--save-arrays",
            "true",
        ]
        if exp.data_config.action_config.statistic_mapping:
            cmd.extend(["--norm-stats", exp.data_config.action_config.statistic_mapping])
        print("Running open-loop evaluation:")
        print(" ".join(cmd))
        env = os.environ.copy()
        env["DEXBOTIC_DATA_PATH"] = str(register_path.parent)
        subprocess.run(cmd, check=True, env=env)

    print("\nIf this setup still cannot fit one episode, prioritize checking:")
    print("- action dimension order")
    print("- observation/action alignment")
    print("- norm stats path and format")
    print("- action target transform (especially DeltaAction on already-delta actions)")
    print("- loss mask and output dimension")
    print("- open-loop chunk merge mode")


if __name__ == "__main__":
    main()
