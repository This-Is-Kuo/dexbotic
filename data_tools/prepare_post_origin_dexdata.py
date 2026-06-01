#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a LeRobot jsonl-meta post_origin dataset for DM0: "
            "convert train/test DexData, register datasets, and run preflight checks."
        )
    )
    parser.add_argument("--raw_root", required=True)
    parser.add_argument(
        "--dataset_prefix",
        default=None,
        help="Dataset registry prefix. Defaults to the raw_root directory name.",
    )
    parser.add_argument(
        "--output_base",
        default="/dexbotic/data",
        help="Directory where *_dexdata_train and *_dexdata_test are created.",
    )
    parser.add_argument(
        "--register_dir",
        default=None,
        help="Directory for generated data_source registration file.",
    )
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--test_episodes", type=int, default=None)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--state_mode",
        choices=["minimal", "stateful"],
        default="minimal",
    )
    parser.add_argument(
        "--schema",
        choices=["auto", "post_data_01", "origin_102"],
        default="auto",
    )
    parser.add_argument(
        "--video_mode",
        choices=["symlink", "copy", "skip"],
        default="symlink",
    )
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_convert", action="store_true")
    parser.add_argument("--skip_register", action="store_true")
    parser.add_argument("--skip_validate", action="store_true")
    parser.add_argument("--num_check_episodes", type=int, default=5)
    parser.add_argument("--num_check_samples", type=int, default=5)
    return parser.parse_args()


def sanitize_prefix(value: str) -> str:
    prefix = re.sub(r"[^0-9a-zA-Z_]+", "_", value.strip())
    prefix = re.sub(r"_+", "_", prefix).strip("_")
    if not prefix:
        raise ValueError(f"Invalid dataset prefix derived from {value!r}")
    if prefix[0].isdigit():
        prefix = f"data_{prefix}"
    return prefix


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_raw_root(raw_root: Path) -> None:
    required = [
        raw_root / "meta" / "tasks.jsonl",
        raw_root / "meta" / "episodes.jsonl",
        raw_root / "data",
        raw_root / "videos",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("raw_root is missing required paths:\n" + "\n".join(missing))


def dexdata_roots(output_base: Path, dataset_prefix: str, state_mode: str) -> tuple[Path, Path]:
    suffix = "_stateful" if state_mode == "stateful" else ""
    train_root = output_base / f"{dataset_prefix}_dexdata{suffix}_train"
    test_root = output_base / f"{dataset_prefix}_dexdata{suffix}_test"
    return train_root, test_root


def convert_dataset(args: argparse.Namespace, dataset_prefix: str, train_root: Path, test_root: Path) -> None:
    cmd = [
        sys.executable,
        str(repo_root() / "data_tools" / "convert_lerobot_v2_post_origin_to_dexdata.py"),
        "--raw_root",
        args.raw_root,
        "--output_root",
        str(train_root),
        "--test_output_root",
        str(test_root),
        "--test_ratio",
        str(args.test_ratio),
        "--split_seed",
        str(args.split_seed),
        "--video_mode",
        args.video_mode,
        "--state_mode",
        args.state_mode,
        "--schema",
        args.schema,
    ]
    if args.test_episodes is not None:
        cmd += ["--test_episodes", str(args.test_episodes)]
    if args.max_episodes is not None:
        cmd += ["--max_episodes", str(args.max_episodes)]
    if args.overwrite:
        cmd.append("--overwrite")
    run(cmd)


def to_container_path(path: Path) -> str:
    root = repo_root()
    try:
        return "/dexbotic/" + path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def register_file_text(dataset_prefix: str, train_root: Path, test_root: Path) -> str:
    constant_name = sanitize_prefix(dataset_prefix).upper()
    train_container = to_container_path(train_root)
    test_container = to_container_path(test_root)
    return f'''from dexbotic.data.data_source.register import register_dataset


{constant_name}_DATASET = {{
    "train": {{
        "data_path_prefix": "{train_container}/video",
        "annotations": "{train_container}/jsonl",
        "frequency": 1,
    }},
    "test": {{
        "data_path_prefix": "{test_container}/video",
        "annotations": "{test_container}/jsonl",
        "frequency": 1,
    }},
}}

meta_data = {{
    "non_delta_mask": [6, 13],
    "periodic_mask": None,
    "periodic_range": None,
}}

register_dataset(
    {constant_name}_DATASET,
    meta_data=meta_data,
    prefix="{dataset_prefix}",
)
'''


def write_registration(args: argparse.Namespace, dataset_prefix: str, train_root: Path, test_root: Path) -> Path:
    register_dir = (
        Path(args.register_dir)
        if args.register_dir
        else repo_root() / "dexbotic" / "data" / "data_source"
    )
    register_dir.mkdir(parents=True, exist_ok=True)
    register_path = register_dir / f"{dataset_prefix}.py"
    register_path.write_text(register_file_text(dataset_prefix, train_root, test_root), encoding="utf-8")
    ast.parse(register_path.read_text(encoding="utf-8"), filename=str(register_path))
    print(f"[OK] Wrote registration: {register_path}")
    print(f"[OK] Train dataset name: {dataset_prefix}_train")
    print(f"[OK] Test dataset name : {dataset_prefix}_test")
    return register_path


def read_jsonl_first(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"Empty jsonl file: {path}")


def basic_validate(root: Path, expected_state_dim: int) -> None:
    jsonl_dir = root / "jsonl"
    video_dir = root / "video"
    jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No jsonl files under {jsonl_dir}")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Missing video dir: {video_dir}")
    sample = read_jsonl_first(jsonl_files[0])
    state = sample.get("state")
    action = sample.get("action")
    if not isinstance(state, list) or len(state) != expected_state_dim:
        raise ValueError(f"{jsonl_files[0]} state dim {len(state) if isinstance(state, list) else None}, expected {expected_state_dim}")
    if not isinstance(action, list) or len(action) != 14:
        raise ValueError(f"{jsonl_files[0]} action dim {len(action) if isinstance(action, list) else None}, expected 14")
    for key in ["images_1", "images_2", "images_3"]:
        url = sample[key]["url"]
        video_path = video_dir / url
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video referenced by {jsonl_files[0]}: {video_path}")
    print(f"[OK] Basic validation: {root} episodes={len(jsonl_files)} state_dim={expected_state_dim} action_dim=14")


def run_preflight(args: argparse.Namespace, root: Path, split_name: str) -> None:
    output_dir = repo_root() / "tmp" / "dm0_data_checks" / root.name
    cmd = [
        sys.executable,
        str(repo_root() / "scripts" / "check_dm0_data_alignment.py"),
        "--dexdata_root",
        str(root),
        "--jsonl_dir",
        str(root / "jsonl"),
        "--video_dir",
        str(root / "video"),
        "--num_episodes",
        str(args.num_check_episodes),
        "--num_samples_per_episode",
        str(args.num_check_samples),
        "--output_dir",
        str(output_dir),
    ]
    print(f"[CHECK] {split_name}")
    run(cmd)


def print_next_steps(dataset_prefix: str) -> None:
    print()
    print("[NEXT] Compute norm stats:")
    print(f"  export DEXBOTIC_DATASET_NAME={dataset_prefix}_train")
    print("  python3 playground/post_data_01_dm0_deltafix.py --task compute_norm_stats")
    print()
    print("[NEXT] Train:")
    print(f"  export DEXBOTIC_DATASET_NAME={dataset_prefix}_train")
    print("  torchrun --nproc_per_node=4 playground/post_data_01_dm0_deltafix.py")
    print()
    print("[NEXT] Open-loop eval dataset:")
    print(f"  --dataset-name {dataset_prefix}_test")


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    ensure_raw_root(raw_root)

    dataset_prefix = sanitize_prefix(args.dataset_prefix or raw_root.name)
    output_base = Path(args.output_base)
    train_root, test_root = dexdata_roots(output_base, dataset_prefix, args.state_mode)
    expected_state_dim = 28 if args.state_mode == "stateful" else 16

    if not args.skip_convert:
        convert_dataset(args, dataset_prefix, train_root, test_root)

    if not args.skip_register:
        write_registration(args, dataset_prefix, train_root, test_root)

    if not args.skip_validate:
        basic_validate(train_root, expected_state_dim)
        basic_validate(test_root, expected_state_dim)
        run_preflight(args, train_root, "train")
        run_preflight(args, test_root, "test")

    print_next_steps(dataset_prefix)


if __name__ == "__main__":
    main()
