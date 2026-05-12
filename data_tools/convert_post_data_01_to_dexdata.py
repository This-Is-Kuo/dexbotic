#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert a custom LeRobot-style dataset (your post_data_01 layout)
into Dexdata format for Dexbotic / DM0 training.

Input structure (observed):
raw_root/
├── data/chunk-000/file-001.parquet
├── meta/tasks.parquet
├── meta/episodes/chunk-000/file-001.parquet
└── videos/
    ├── observation.images.chest/chunk-000/file-001.mp4
    ├── observation.images.left/chunk-000/file-001.mp4
    └── observation.images.right/chunk-000/file-001.mp4

Output structure:
output_root/
├── jsonl/
│   ├── episode_000001.jsonl
│   ├── episode_000002.jsonl
│   └── ...
└── video/
    ├── observation.images.chest/chunk-000/file-001.mp4
    ├── observation.images.left/chunk-000/file-001.mp4
    └── observation.images.right/chunk-000/file-001.mp4

Recommended first use:
1) Run with --max_episodes 1
2) Inspect output jsonl
3) Then remove --max_episodes and convert all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_root",
        type=str,
        required=True,
        help="Path to raw dataset root, e.g. /dexbotic/data/post_data_01",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to output Dexdata root, e.g. /dexbotic/data/post_data_01_dexdata",
    )
    parser.add_argument(
        "--video_mode",
        type=str,
        choices=["symlink", "copy", "skip"],
        default="symlink",
        help="How to place videos under output_root/video",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Only convert the first N episodes for smoke testing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing jsonl files",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def chunk_name(idx: int) -> str:
    return f"chunk-{int(idx):03d}"


def file_name(idx: int, suffix: str) -> str:
    return f"file-{int(idx):03d}.{suffix}"


def safe_float(x: Any) -> float:
    if pd.isna(x):
        return float("nan")
    return float(x)


def to_float_list(x: Any) -> List[float]:
    if x is None:
        return []
    if isinstance(x, list):
        return [float(v) for v in x]
    if hasattr(x, "tolist"):
        y = x.tolist()
        if isinstance(y, list):
            return [float(v) for v in y]
        return [float(y)]
    return [float(x)]


def load_task_map(raw_root: Path) -> Dict[int, str]:
    task_file = raw_root / "meta" / "tasks.parquet"
    if not task_file.is_file():
        raise FileNotFoundError(f"Missing task file: {task_file}")

    df = pd.read_parquet(task_file)
    required_cols = {"task_index", "task"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"tasks.parquet missing columns: {missing}")

    task_map = {}
    for _, row in df.iterrows():
        task_map[int(row["task_index"])] = str(row["task"])
    if not task_map:
        raise ValueError("Empty task map loaded from tasks.parquet")
    return task_map


def collect_episode_meta_files(raw_root: Path) -> List[Path]:
    ep_root = raw_root / "meta" / "episodes"
    if not ep_root.is_dir():
        raise FileNotFoundError(f"Missing episode meta dir: {ep_root}")

    files = sorted(ep_root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode meta parquet found under: {ep_root}")
    return files


def locate_raw_data_file(raw_root: Path, row: pd.Series) -> Path:
    cidx = int(row["data/chunk_index"])
    fidx = int(row["data/file_index"])
    path = raw_root / "data" / chunk_name(cidx) / file_name(fidx, "parquet")
    if not path.is_file():
        raise FileNotFoundError(f"Raw data parquet not found: {path}")
    return path


def locate_raw_video_file(raw_root: Path, row: pd.Series, view_key: str) -> Path:
    chunk_col = f"videos/{view_key}/chunk_index"
    file_col = f"videos/{view_key}/file_index"
    if chunk_col not in row.index or file_col not in row.index:
        raise KeyError(f"Missing video columns for {view_key}: {chunk_col}, {file_col}")

    cidx = int(row[chunk_col])
    fidx = int(row[file_col])

    path = raw_root / "videos" / view_key / chunk_name(cidx) / file_name(fidx, "mp4")
    if not path.is_file():
        raise FileNotFoundError(f"Raw video not found: {path}")
    return path


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)

    if dst.exists() or dst.is_symlink():
        return

    if mode == "skip":
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return

    raise ValueError(f"Unknown video_mode: {mode}")


def build_state(row: pd.Series) -> List[float]:
    # Minimal, non-duplicated state:
    # left_tcp(7) + right_tcp(7) + left_pinch(1) + right_pinch(1) = 16 dims
    left_tcp = to_float_list(row["observation.state.left_tcp"])
    right_tcp = to_float_list(row["observation.state.right_tcp"])
    left_pinch = [safe_float(row["observation.state.left_pinch"])]
    right_pinch = [safe_float(row["observation.state.right_pinch"])]

    state = left_tcp + right_tcp + left_pinch + right_pinch

    if len(left_tcp) != 7 or len(right_tcp) != 7:
        raise ValueError(
            f"Unexpected tcp dims: left={len(left_tcp)} right={len(right_tcp)}; expected 7 and 7"
        )
    if len(state) != 16:
        raise ValueError(f"Unexpected state dim: {len(state)}; expected 16")

    return state


def build_action(row: pd.Series) -> List[float]:
    # left_delta_tcp(6) + left_pinch(1) + right_delta_tcp(6) + right_pinch(1) = 14 dims
    left_delta = to_float_list(row["action.left_delta_tcp"])
    right_delta = to_float_list(row["action.right_delta_tcp"])
    left_pinch = [safe_float(row["action.left_pinch"])]
    right_pinch = [safe_float(row["action.right_pinch"])]

    action = left_delta + left_pinch + right_delta + right_pinch

    if len(left_delta) != 6 or len(right_delta) != 6:
        raise ValueError(
            f"Unexpected delta dims: left={len(left_delta)} right={len(right_delta)}; expected 6 and 6"
        )
    if len(action) != 14:
        raise ValueError(f"Unexpected action dim: {len(action)}; expected 14")

    return action


def build_record(
    row: pd.Series,
    prompt: str,
    rel_chest_video: str,
    rel_left_video: str,
    rel_right_video: str,
) -> Dict[str, Any]:
    frame_idx = int(row["frame_index"])

    record = {
        # main view first, then left and right hand views
        "images_1": {
            "type": "video",
            "url": rel_chest_video,
            "frame_idx": frame_idx,
        },
        "images_2": {
            "type": "video",
            "url": rel_left_video,
            "frame_idx": frame_idx,
        },
        "images_3": {
            "type": "video",
            "url": rel_right_video,
            "frame_idx": frame_idx,
        },
        "state": build_state(row),
        "prompt": prompt,
        "is_robot": True,
        "action": build_action(row),
        # Leave answer absent; Dexbotic can derive textualized actions if desired.
        "extra": {
            "timestamp": safe_float(row["timestamp"]),
            "frame_index": int(row["frame_index"]),
            "episode_index": int(row["episode_index"]),
            "task_index": int(row["task_index"]),
            "teleoperated": bool(row["teleoperated"]),
            "progress": {
                "chest": safe_float(row["task.progress.chest"]),
                "left": safe_float(row["task.progress.left"]),
                "right": safe_float(row["task.progress.right"]),
            },
            "source_format": "lerobot_custom_post_data_01",
        },
    }
    return record


def convert_one_episode(
    raw_root: Path,
    output_root: Path,
    task_map: Dict[int, str],
    episode_meta_row: pd.Series,
    video_mode: str,
    overwrite: bool,
) -> Path:
    episode_index = int(episode_meta_row["episode_index"])

    raw_data_file = locate_raw_data_file(raw_root, episode_meta_row)
    raw_chest_video = locate_raw_video_file(raw_root, episode_meta_row, "observation.images.chest")
    raw_left_video = locate_raw_video_file(raw_root, episode_meta_row, "observation.images.left")
    raw_right_video = locate_raw_video_file(raw_root, episode_meta_row, "observation.images.right")

    df = pd.read_parquet(raw_data_file)
    if df.empty:
        raise ValueError(f"Empty episode data file: {raw_data_file}")

    if "task_index" not in df.columns:
        raise KeyError(f"Missing task_index in {raw_data_file}")

    task_indices = sorted(set(int(v) for v in df["task_index"].dropna().tolist()))
    if len(task_indices) != 1:
        raise ValueError(
            f"Expected one task_index per episode, got {task_indices} in {raw_data_file}"
        )

    task_index = task_indices[0]
    if task_index not in task_map:
        raise KeyError(f"task_index={task_index} not found in task map")

    prompt = task_map[task_index]

    # Keep the same relative layout under output_root/video
    chest_rel = os.path.join(
        "observation.images.chest",
        raw_chest_video.parent.name,
        raw_chest_video.name,
    )
    left_rel = os.path.join(
        "observation.images.left",
        raw_left_video.parent.name,
        raw_left_video.name,
    )
    right_rel = os.path.join(
        "observation.images.right",
        raw_right_video.parent.name,
        raw_right_video.name,
    )

    chest_out = output_root / "video" / chest_rel
    left_out = output_root / "video" / left_rel
    right_out = output_root / "video" / right_rel

    link_or_copy(raw_chest_video, chest_out, video_mode)
    link_or_copy(raw_left_video, left_out, video_mode)
    link_or_copy(raw_right_video, right_out, video_mode)

    jsonl_dir = output_root / "jsonl"
    ensure_dir(jsonl_dir)
    jsonl_path = jsonl_dir / f"episode_{episode_index:06d}.jsonl"

    if jsonl_path.exists() and not overwrite:
        raise FileExistsError(f"{jsonl_path} exists. Use --overwrite to replace it.")

    num_rows = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = build_record(
                row=row,
                prompt=prompt,
                rel_chest_video=chest_rel,
                rel_left_video=left_rel,
                rel_right_video=right_rel,
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_rows += 1

    print(
        f"[OK] episode={episode_index:06d} "
        f"frames={num_rows} "
        f"task={task_index} "
        f"jsonl={jsonl_path.name}"
    )
    return jsonl_path


def main() -> None:
    args = parse_args()

    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")

    ensure_dir(output_root)
    ensure_dir(output_root / "jsonl")
    ensure_dir(output_root / "video")

    task_map = load_task_map(raw_root)
    meta_files = collect_episode_meta_files(raw_root)

    converted = 0
    for meta_file in meta_files:
        df_meta = pd.read_parquet(meta_file)
        if df_meta.empty:
            continue

        for _, ep_row in df_meta.iterrows():
            convert_one_episode(
                raw_root=raw_root,
                output_root=output_root,
                task_map=task_map,
                episode_meta_row=ep_row,
                video_mode=args.video_mode,
                overwrite=args.overwrite,
            )
            converted += 1
            if args.max_episodes is not None and converted >= args.max_episodes:
                print(f"[DONE] Converted {converted} episodes (limited by --max_episodes)")
                return

    print(f"[DONE] Converted {converted} episodes in total.")
    print(f"Output jsonl dir : {output_root / 'jsonl'}")
    print(f"Output video dir : {output_root / 'video'}")
    print()
    print("Suggested next step:")
    print("1) Inspect one jsonl file")
    print("2) Register dataset in dexbotic/data/data_source/")
    print("3) Set data_path_prefix to output_root/video and annotations to output_root/jsonl")


if __name__ == "__main__":
    main()