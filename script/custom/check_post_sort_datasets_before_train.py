#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any


DATASETS = [
    Path("data/post_data_merged_dm0_dexdata_nobframes_train"),
    Path("data/post_data_merged_dm0_dexdata_nobframes_test"),
    Path("data/post_origin_data_0423_dm0_dexdata_nobframes_train"),
    Path("data/post_origin_data_0423_dm0_dexdata_nobframes_test"),
]
VIEWS = [
    "observation.images.chest",
    "observation.images.left",
    "observation.images.right",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-dataset", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_ffprobe(args: list[str]) -> str | None:
    if shutil.which("ffprobe") is None:
        return None
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(os.getenv("DEXBOTIC_FFPROBE_TIMEOUT", "60")),
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def video_frame_count(path: Path) -> int | None:
    output = run_ffprobe([
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ])
    if not output:
        return None
    try:
        return int(output.splitlines()[0])
    except (ValueError, IndexError):
        return None


def video_start_time(path: Path) -> float | None:
    output = run_ffprobe([
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=start_time",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ])
    if not output or output == "N/A":
        return None
    try:
        return float(output.splitlines()[0])
    except (ValueError, IndexError):
        return None


def video_has_b_frames(path: Path) -> bool | None:
    output = run_ffprobe([
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pict_type",
        "-of",
        "csv=p=0",
        str(path),
    ])
    if output is None:
        return None
    return any(line.strip() == "B" for line in output.splitlines())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as f:
        for _ in f:
            count += 1
    return count


def is_finite_vector(values: Any) -> bool:
    if not isinstance(values, list):
        return False
    for value in values:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return False
    return True


def image_entry(row: dict[str, Any], view: str) -> dict[str, Any] | None:
    for value in row.values():
        if isinstance(value, dict) and value.get("type") == "video" and view in str(value.get("url", "")):
            return value
    return None


def check_dataset(root: Path, samples_per_dataset: int, rng: random.Random) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    jsonl_dir = root / "jsonl"
    video_dir = root / "video"

    if not root.exists():
        errors.append(f"missing root: {root}")
        return {"dataset": str(root), "num_errors": len(errors), "num_warnings": len(warnings), "errors": errors}
    if not jsonl_dir.is_dir():
        errors.append(f"missing jsonl dir: {jsonl_dir}")
    if not video_dir.is_dir():
        errors.append(f"missing video dir: {video_dir}")
    for view in VIEWS:
        if not (video_dir / view).is_dir():
            errors.append(f"missing video view dir: {video_dir / view}")

    episode_files = sorted(jsonl_dir.glob("*.jsonl")) if jsonl_dir.is_dir() else []
    row_counts = [count_lines(path) for path in episode_files]
    sample_files = rng.sample(episode_files, min(samples_per_dataset, len(episode_files))) if episode_files else []

    state_dims = set()
    action_dims = set()
    any_b_frames = False
    b_frame_unknown = False

    for jsonl_path in sample_files:
        rows = read_jsonl(jsonl_path)
        if not rows:
            errors.append(f"{jsonl_path}: empty episode")
            continue
        frame_indices = [row.get("extra", {}).get("frame_index", row.get("frame_idx")) for row in rows]
        if frame_indices[0] != 0:
            errors.append(f"{jsonl_path}: first frame_idx is {frame_indices[0]}, expected 0")
        if frame_indices != list(range(len(rows))):
            warnings.append(f"{jsonl_path}: frame_idx sequence is not contiguous 0..N-1")

        first = rows[0]
        state = first.get("state")
        action = first.get("action")
        if isinstance(state, list):
            state_dims.add(len(state))
        else:
            errors.append(f"{jsonl_path}: missing/list-invalid state")
        if isinstance(action, list):
            action_dims.add(len(action))
        else:
            errors.append(f"{jsonl_path}: missing/list-invalid action")

        for idx, row in enumerate(rows):
            if not is_finite_vector(row.get("state")):
                errors.append(f"{jsonl_path}: non-finite or invalid state at row {idx}")
                break
        for idx, row in enumerate(rows):
            if not is_finite_vector(row.get("action")):
                errors.append(f"{jsonl_path}: non-finite or invalid action at row {idx}")
                break

        timestamp = first.get("extra", {}).get("timestamp")
        if timestamp is not None and abs(float(timestamp)) > 1e-6:
            warnings.append(f"{jsonl_path}: first timestamp is {timestamp}, expected near 0")

        for view in VIEWS:
            entry = image_entry(first, view)
            if entry is None:
                errors.append(f"{jsonl_path}: missing image entry for {view}")
                continue
            video_path = video_dir / entry["url"]
            if not video_path.exists():
                errors.append(f"{jsonl_path}: missing video file {video_path}")
                continue
            frame_count = video_frame_count(video_path)
            if frame_count is not None and frame_count < len(rows):
                errors.append(f"{video_path}: frame_count={frame_count} < jsonl_rows={len(rows)}")
            start_time = video_start_time(video_path)
            if start_time is not None and abs(start_time) > 1e-3:
                warnings.append(f"{video_path}: start_time={start_time}, expected near 0")
            has_b = video_has_b_frames(video_path)
            if has_b is None:
                b_frame_unknown = True
            elif has_b:
                any_b_frames = True
                warnings.append(f"{video_path}: contains B-frames")

    num_rows = sum(row_counts)
    summary = {
        "dataset": str(root),
        "num_episodes": len(episode_files),
        "num_rows": num_rows,
        "avg_rows_per_episode": num_rows / len(row_counts) if row_counts else 0,
        "min_rows_per_episode": min(row_counts) if row_counts else 0,
        "max_rows_per_episode": max(row_counts) if row_counts else 0,
        "state_dim": sorted(state_dims),
        "action_dim": sorted(action_dims),
        "video_has_b_frames": any_b_frames if not b_frame_unknown else "unknown_or_" + str(any_b_frames),
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "sampled_episodes": [str(path) for path in sample_files],
        "errors": errors[:50],
        "warnings": warnings[:50],
    }
    return summary


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    all_summaries = [check_dataset(root, args.samples_per_dataset, rng) for root in DATASETS]
    print(json.dumps(all_summaries, indent=2))
    total_errors = sum(summary.get("num_errors", 0) for summary in all_summaries)
    total_warnings = sum(summary.get("num_warnings", 0) for summary in all_summaries)
    print(f"\nSUMMARY num_errors={total_errors} num_warnings={total_warnings}")
    if total_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
