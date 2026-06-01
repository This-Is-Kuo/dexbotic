#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - keeps --help usable in minimal envs
    pd = None


VIEW_KEY_CANDIDATES = {
    "chest": [
        "observation.images.chest",
        "observation.images.chest_rgb",
    ],
    "left": [
        "observation.images.left",
        "observation.images.left_wrist_rgb",
    ],
    "right": [
        "observation.images.right",
        "observation.images.right_wrist_rgb",
    ],
}

LEFT_TCP_NAMES = [
    "end_position_l_x",
    "end_position_l_y",
    "end_position_l_z",
    "end_quaternion_l_x",
    "end_quaternion_l_y",
    "end_quaternion_l_z",
    "end_quaternion_l_w",
]
RIGHT_TCP_NAMES = [
    "end_position_r_x",
    "end_position_r_y",
    "end_position_r_z",
    "end_quaternion_r_x",
    "end_quaternion_r_y",
    "end_quaternion_r_z",
    "end_quaternion_r_w",
]
LEFT_HAND_NAMES = [
    "THUMB_MP_LEFT",
    "THUMB_CMC_LEFT",
    "INDEX_MCP_LEFT",
    "MIDDLE_MCP_LEFT",
    "RING_MCP_LEFT",
    "LITTLE_MCP_LEFT",
]
RIGHT_HAND_NAMES = [
    "THUMB_MP_RIGHT",
    "THUMB_CMC_RIGHT",
    "INDEX_MCP_RIGHT",
    "MIDDLE_MCP_RIGHT",
    "RING_MCP_RIGHT",
    "LITTLE_MCP_RIGHT",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--test_output_root",
        default=None,
        help="Optional DexData output root for held-out open-loop evaluation episodes.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of episodes to hold out when --test_output_root is set.",
    )
    parser.add_argument(
        "--test_episodes",
        type=int,
        default=None,
        help="Exact number of held-out episodes. Overrides --test_ratio.",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Seed for deterministic train/test episode split.",
    )
    parser.add_argument("--video_mode", choices=["symlink", "copy", "skip"], default="symlink")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--state_mode",
        choices=["minimal", "stateful"],
        default="minimal",
        help="minimal=16D state, stateful=28D state with observation delta_tcp appended",
    )
    parser.add_argument(
        "--schema",
        choices=["auto", "post_data_01", "origin_102"],
        default="auto",
        help="Input parquet schema. auto selects split columns or 102D vector columns.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x: Any, default: float = float("nan")) -> float:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except ValueError:
        pass
    return float(x)


def to_float_list(x: Any) -> list[float]:
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing jsonl: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_task_map(raw_root: Path) -> dict[int, str]:
    tasks = load_jsonl(raw_root / "meta" / "tasks.jsonl")
    task_map = {}
    for row in tasks:
        task_map[int(row["task_index"])] = str(row["task"])
    if not task_map:
        raise ValueError("Empty task map")
    return task_map


def load_feature_names(raw_root: Path) -> dict[str, list[str]]:
    info_file = raw_root / "meta" / "info.json"
    if not info_file.is_file():
        return {}
    with open(info_file, "r", encoding="utf-8") as f:
        info = json.load(f)
    features = info.get("features") or {}
    result = {}
    for key, value in features.items():
        names = value.get("names")
        if isinstance(names, list):
            result[key] = [str(name) for name in names]
    return result


def name_indices(names: list[str], wanted: list[str]) -> list[int]:
    index_by_name = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in wanted if name not in index_by_name]
    if missing:
        raise KeyError(f"Missing feature names {missing}; available names include {names[:10]}")
    return [index_by_name[name] for name in wanted]


def vector_by_names(row: pd.Series, column: str, names: list[str], wanted: list[str]) -> list[float]:
    values = to_float_list(row[column])
    indices = name_indices(names, wanted)
    return [float(values[idx]) for idx in indices]


def mean_by_names(row: pd.Series, column: str, names: list[str], wanted: list[str]) -> float:
    values = vector_by_names(row, column, names, wanted)
    return float(sum(values) / len(values))


def tcp6_delta_with_quat_alignment(
    state_tcp7: list[float],
    action_tcp7: list[float],
) -> list[float]:
    pos_delta = [a - s for a, s in zip(action_tcp7[:3], state_tcp7[:3])]
    state_quat = state_tcp7[3:7]
    action_quat = action_tcp7[3:7]
    dot = sum(s * a for s, a in zip(state_quat, action_quat))
    if dot < 0.0:
        action_quat = [-v for v in action_quat]
    quat_xyz_delta = [a - s for a, s in zip(action_quat[:3], state_quat[:3])]
    return pos_delta + quat_xyz_delta


def split_episodes(
    episodes: list[dict[str, Any]],
    test_ratio: float,
    test_episodes: int | None,
    split_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 <= test_ratio < 1:
        raise ValueError(f"--test_ratio must be in [0, 1), got {test_ratio}")

    episode_indices = list(range(len(episodes)))
    rng = random.Random(split_seed)
    rng.shuffle(episode_indices)

    if test_episodes is None:
        num_test = round(len(episodes) * test_ratio)
    else:
        num_test = test_episodes
    num_test = max(0, min(num_test, len(episodes) - 1))

    test_index_set = set(episode_indices[:num_test])
    train = [ep for idx, ep in enumerate(episodes) if idx not in test_index_set]
    test = [ep for idx, ep in enumerate(episodes) if idx in test_index_set]
    return train, test


def write_split_manifest(
    output_root: Path,
    split_name: str,
    episodes: list[dict[str, Any]],
    raw_root: Path,
    state_mode: str,
    split_seed: int | None,
) -> None:
    ensure_dir(output_root)
    payload = {
        "split": split_name,
        "raw_root": str(raw_root),
        "state_mode": state_mode,
        "split_seed": split_seed,
        "num_episodes": len(episodes),
        "episode_indices": [int(ep["episode_index"]) for ep in episodes],
    }
    with open(output_root / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def select_video(episode: dict[str, Any], logical_view: str) -> tuple[str, Path]:
    videos = episode.get("videos") or {}
    for key in VIEW_KEY_CANDIDATES[logical_view]:
        if key in videos:
            return key, Path(videos[key])
    raise KeyError(
        f"Episode {episode.get('episode_index')} missing {logical_view} video. "
        f"Available keys: {sorted(videos.keys())}"
    )


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


def detect_schema(row: pd.Series, requested_schema: str) -> str:
    if requested_schema != "auto":
        return requested_schema
    if "observation.state.left_tcp" in row.index:
        return "post_data_01"
    if "observation.state" in row.index and ("actions" in row.index or "action" in row.index):
        return "origin_102"
    raise KeyError(
        "Could not auto-detect parquet schema. Expected either split post_data_01 "
        "columns or vector columns observation.state/actions."
    )


def build_state(
    row: pd.Series,
    state_mode: str,
    schema: str,
    feature_names: dict[str, list[str]],
) -> list[float]:
    if schema == "origin_102":
        names = feature_names.get("observation.state")
        if not names:
            raise KeyError("origin_102 schema requires observation.state names in meta/info.json")
        left_tcp = vector_by_names(row, "observation.state", names, LEFT_TCP_NAMES)
        right_tcp = vector_by_names(row, "observation.state", names, RIGHT_TCP_NAMES)
        left_pinch = [mean_by_names(row, "observation.state", names, LEFT_HAND_NAMES)]
        right_pinch = [mean_by_names(row, "observation.state", names, RIGHT_HAND_NAMES)]
        state = left_tcp + right_tcp + left_pinch + right_pinch
        if len(state) != 16:
            raise ValueError(f"Unexpected origin_102 state dim: {len(state)}")

        if state_mode == "stateful":
            action_col = "actions" if "actions" in row.index else "action"
            action_names = feature_names.get(action_col) or feature_names.get("actions")
            if not action_names:
                raise KeyError("origin_102 stateful mode requires actions names in meta/info.json")
            left_action = vector_by_names(row, action_col, action_names, LEFT_TCP_NAMES)
            right_action = vector_by_names(row, action_col, action_names, RIGHT_TCP_NAMES)
            state += tcp6_delta_with_quat_alignment(left_tcp, left_action)
            state += tcp6_delta_with_quat_alignment(right_tcp, right_action)
            if len(state) != 28:
                raise ValueError(f"Unexpected origin_102 stateful dim: {len(state)}")
        return state

    left_tcp = to_float_list(row["observation.state.left_tcp"])
    right_tcp = to_float_list(row["observation.state.right_tcp"])
    left_pinch = [safe_float(row["observation.state.left_pinch"])]
    right_pinch = [safe_float(row["observation.state.right_pinch"])]
    state = left_tcp + right_tcp + left_pinch + right_pinch

    if len(left_tcp) != 7 or len(right_tcp) != 7 or len(state) != 16:
        raise ValueError(
            f"Unexpected minimal state dims: left_tcp={len(left_tcp)} "
            f"right_tcp={len(right_tcp)} state={len(state)}"
        )

    if state_mode == "stateful":
        left_delta = to_float_list(row["observation.state.left_delta_tcp"])
        right_delta = to_float_list(row["observation.state.right_delta_tcp"])
        state += left_delta + right_delta
        if len(left_delta) != 6 or len(right_delta) != 6 or len(state) != 28:
            raise ValueError(
                f"Unexpected stateful dims: left_delta={len(left_delta)} "
                f"right_delta={len(right_delta)} state={len(state)}"
            )

    return state


def build_action(
    row: pd.Series,
    schema: str,
    feature_names: dict[str, list[str]],
) -> list[float]:
    if schema == "origin_102":
        state_names = feature_names.get("observation.state")
        action_col = "actions" if "actions" in row.index else "action"
        action_names = feature_names.get(action_col) or feature_names.get("actions")
        if not state_names or not action_names:
            raise KeyError("origin_102 schema requires observation.state/actions names in meta/info.json")
        left_state = vector_by_names(row, "observation.state", state_names, LEFT_TCP_NAMES)
        right_state = vector_by_names(row, "observation.state", state_names, RIGHT_TCP_NAMES)
        left_action_target = vector_by_names(row, action_col, action_names, LEFT_TCP_NAMES)
        right_action_target = vector_by_names(row, action_col, action_names, RIGHT_TCP_NAMES)
        left_delta = tcp6_delta_with_quat_alignment(left_state, left_action_target)
        right_delta = tcp6_delta_with_quat_alignment(right_state, right_action_target)
        left_pinch = [mean_by_names(row, action_col, action_names, LEFT_HAND_NAMES)]
        right_pinch = [mean_by_names(row, action_col, action_names, RIGHT_HAND_NAMES)]
        action = left_delta + left_pinch + right_delta + right_pinch
        if len(action) != 14:
            raise ValueError(f"Unexpected origin_102 action dim: {len(action)}")
        return action

    left_delta = to_float_list(row["action.left_delta_tcp"])
    right_delta = to_float_list(row["action.right_delta_tcp"])
    left_pinch = [safe_float(row["action.left_pinch"])]
    right_pinch = [safe_float(row["action.right_pinch"])]
    action = left_delta + left_pinch + right_delta + right_pinch
    if len(left_delta) != 6 or len(right_delta) != 6 or len(action) != 14:
        raise ValueError(
            f"Unexpected action dims: left_delta={len(left_delta)} "
            f"right_delta={len(right_delta)} action={len(action)}"
        )
    return action


def data_file_for_episode(raw_root: Path, episode_index: int) -> Path:
    chunk = f"chunk-{episode_index // 1000:03d}"
    path = raw_root / "data" / chunk / f"episode_{episode_index:06d}.parquet"
    if not path.is_file():
        matches = sorted((raw_root / "data").rglob(f"episode_{episode_index:06d}.parquet"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Missing episode parquet: {path}")
    return path


def output_video_rel(logical_view: str, src_rel: Path) -> Path:
    return Path(f"observation.images.{logical_view}") / src_rel.parent.name / src_rel.name


def convert_episode(
    raw_root: Path,
    output_root: Path,
    task_map: dict[int, str],
    episode: dict[str, Any],
    video_mode: str,
    overwrite: bool,
    state_mode: str,
    schema: str,
    feature_names: dict[str, list[str]],
) -> None:
    episode_index = int(episode["episode_index"])
    df = pd.read_parquet(data_file_for_episode(raw_root, episode_index))
    if df.empty:
        raise ValueError(f"Empty episode: {episode_index}")
    episode_schema = detect_schema(df.iloc[0], schema)

    if "task_index" in df.columns:
        task_index = int(df["task_index"].dropna().iloc[0])
        prompt = task_map.get(task_index, str(episode.get("tasks", "")))
    else:
        task_index = None
        prompt = str(episode.get("tasks", ""))

    video_rels = {}
    for logical_view in ["chest", "left", "right"]:
        _, src_rel = select_video(episode, logical_view)
        src = raw_root / src_rel
        if not src.is_file():
            raise FileNotFoundError(f"Missing video: {src}")
        dst_rel = output_video_rel(logical_view, src_rel)
        link_or_copy(src, output_root / "video" / dst_rel, video_mode)
        video_rels[logical_view] = dst_rel.as_posix()

    jsonl_path = output_root / "jsonl" / f"episode_{episode_index:06d}.jsonl"
    ensure_dir(jsonl_path.parent)
    if jsonl_path.exists() and not overwrite:
        raise FileExistsError(f"{jsonl_path} exists. Use --overwrite.")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row_idx, row in df.iterrows():
            frame_idx = int(row["frame_index"]) if "frame_index" in row.index else int(row_idx)
            record = {
                "images_1": {"type": "video", "url": video_rels["chest"], "frame_idx": frame_idx},
                "images_2": {"type": "video", "url": video_rels["left"], "frame_idx": frame_idx},
                "images_3": {"type": "video", "url": video_rels["right"], "frame_idx": frame_idx},
                "state": build_state(row, state_mode, episode_schema, feature_names),
                "prompt": prompt,
                "is_robot": True,
                "action": build_action(row, episode_schema, feature_names),
                "extra": {
                    "timestamp": safe_float(row["timestamp"]) if "timestamp" in row.index else float("nan"),
                    "frame_index": frame_idx,
                    "episode_index": episode_index,
                    "task_index": task_index,
                    "source_format": "lerobot_v2_jsonl_post_origin",
                    "state_mode": state_mode,
                    "schema": episode_schema,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] episode={episode_index:06d} frames={len(df)} task={prompt} jsonl={jsonl_path.name}")


def main() -> None:
    args = parse_args()
    if pd is None:
        raise ModuleNotFoundError(
            "pandas is required to convert parquet data. Install the project data dependencies "
            "or run this inside the Dexbotic training environment."
        )
    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()
    test_output_root = Path(args.test_output_root).resolve() if args.test_output_root else None
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")

    ensure_dir(output_root / "jsonl")
    ensure_dir(output_root / "video")
    if test_output_root is not None:
        ensure_dir(test_output_root / "jsonl")
        ensure_dir(test_output_root / "video")

    task_map = load_task_map(raw_root)
    feature_names = load_feature_names(raw_root)
    episodes = load_jsonl(raw_root / "meta" / "episodes.jsonl")
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    if test_output_root is None:
        split_jobs = [("train", output_root, episodes)]
        write_split_manifest(output_root, "all", episodes, raw_root, args.state_mode, None)
    else:
        train_episodes, test_episodes = split_episodes(
            episodes=episodes,
            test_ratio=args.test_ratio,
            test_episodes=args.test_episodes,
            split_seed=args.split_seed,
        )
        split_jobs = [
            ("train", output_root, train_episodes),
            ("test", test_output_root, test_episodes),
        ]
        write_split_manifest(output_root, "train", train_episodes, raw_root, args.state_mode, args.split_seed)
        write_split_manifest(test_output_root, "test", test_episodes, raw_root, args.state_mode, args.split_seed)
        print(
            f"[SPLIT] total={len(episodes)} train={len(train_episodes)} "
            f"test={len(test_episodes)} seed={args.split_seed}"
        )

    converted = 0
    for split_name, split_output_root, split_episodes_for_job in split_jobs:
        for episode in split_episodes_for_job:
            convert_episode(
                raw_root=raw_root,
                output_root=split_output_root,
                task_map=task_map,
                episode=episode,
                video_mode=args.video_mode,
                overwrite=args.overwrite,
                state_mode=args.state_mode,
                schema=args.schema,
                feature_names=feature_names,
            )
            converted += 1
        print(f"[DONE] Split {split_name}: {len(split_episodes_for_job)} episodes.")

    print(f"[DONE] Converted {converted} episodes.")
    print(f"Output jsonl dir : {output_root / 'jsonl'}")
    print(f"Output video dir : {output_root / 'video'}")
    if test_output_root is not None:
        print(f"Test jsonl dir   : {test_output_root / 'jsonl'}")
        print(f"Test video dir   : {test_output_root / 'video'}")


if __name__ == "__main__":
    main()
