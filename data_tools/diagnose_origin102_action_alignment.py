#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_tools.convert_lerobot_v2_post_origin_to_dexdata import (
    LEFT_HAND_NAMES,
    LEFT_TCP_NAMES,
    RIGHT_HAND_NAMES,
    RIGHT_TCP_NAMES,
    load_feature_names,
    load_jsonl,
    name_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether origin_102 actions are same-frame/next-frame targets and whether quaternion sign flips create spikes."
    )
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_lag", type=int, default=5)
    return parser.parse_args()


def data_file_for_episode(raw_root: Path, episode_index: int) -> Path:
    chunk = f"chunk-{episode_index // 1000:03d}"
    path = raw_root / "data" / chunk / f"episode_{episode_index:06d}.parquet"
    if path.is_file():
        return path
    matches = sorted((raw_root / "data").rglob(f"episode_{episode_index:06d}.parquet"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing episode parquet for {episode_index}: {path}")


def vector(df: pd.DataFrame, column: str, names: list[str], wanted: list[str]) -> np.ndarray:
    indices = name_indices(names, wanted)
    values = np.stack(df[column].map(lambda x: np.asarray(x, dtype=np.float64)).to_numpy())
    return values[:, indices]


def hand_mean(df: pd.DataFrame, column: str, names: list[str], wanted: list[str]) -> np.ndarray:
    return vector(df, column, names, wanted).mean(axis=1, keepdims=True)


def pose6_and_pinch(df: pd.DataFrame, column: str, names: list[str]) -> dict[str, np.ndarray]:
    left_tcp7 = vector(df, column, names, LEFT_TCP_NAMES)
    right_tcp7 = vector(df, column, names, RIGHT_TCP_NAMES)
    return {
        "left_pose6": left_tcp7[:, :6],
        "right_pose6": right_tcp7[:, :6],
        "left_quat": left_tcp7[:, 3:7],
        "right_quat": right_tcp7[:, 3:7],
        "left_pinch": hand_mean(df, column, names, LEFT_HAND_NAMES),
        "right_pinch": hand_mean(df, column, names, RIGHT_HAND_NAMES),
    }


def mae(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x))) if x.size else math.nan


def bias(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else math.nan


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.reshape(-1)
    y = y.reshape(-1)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def quat_flip_rate(state_quat: np.ndarray, action_quat: np.ndarray) -> float:
    dots = np.sum(state_quat * action_quat, axis=1)
    return float(np.mean(dots < 0.0)) if dots.size else math.nan


def align_quat_xyz_delta(state_quat: np.ndarray, action_quat: np.ndarray) -> np.ndarray:
    dots = np.sum(state_quat * action_quat, axis=1, keepdims=True)
    aligned = np.where(dots < 0.0, -action_quat, action_quat)
    return aligned[:, :3] - state_quat[:, :3]


def update_metric(store: dict[str, list[float]], key: str, value: float) -> None:
    if not math.isnan(value):
        store.setdefault(key, []).append(value)


def summarize(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    feature_names = load_feature_names(raw_root)
    state_names = feature_names["observation.state"]
    action_col = "actions" if "actions" in feature_names else "action"
    action_names = feature_names[action_col]

    episodes = load_jsonl(raw_root / "meta" / "episodes.jsonl")
    rng = random.Random(args.seed)
    selected = episodes[:]
    rng.shuffle(selected)
    selected = selected[: args.num_episodes]

    target_lag_metrics: dict[int, dict[str, list[float]]] = {
        lag: {} for lag in range(0, args.max_lag + 1)
    }
    delta_lag_metrics: dict[int, dict[str, list[float]]] = {
        lag: {} for lag in range(1, args.max_lag + 1)
    }
    quat_metrics: dict[str, list[float]] = {}

    for episode in selected:
        ep_idx = int(episode["episode_index"])
        df = pd.read_parquet(data_file_for_episode(raw_root, ep_idx))
        if len(df) <= args.max_lag + 1:
            continue

        state = pose6_and_pinch(df, "observation.state", state_names)
        action = pose6_and_pinch(df, action_col, action_names)

        for side in ["left", "right"]:
            update_metric(
                quat_metrics,
                f"{side}_quat_flip_rate_same_frame",
                quat_flip_rate(state[f"{side}_quat"], action[f"{side}_quat"]),
            )
            raw_delta = action[f"{side}_pose6"][:, 3:6] - state[f"{side}_pose6"][:, 3:6]
            aligned_delta = align_quat_xyz_delta(state[f"{side}_quat"], action[f"{side}_quat"])
            update_metric(quat_metrics, f"{side}_quat_xyz_raw_delta_abs_max", float(np.max(np.abs(raw_delta))))
            update_metric(quat_metrics, f"{side}_quat_xyz_aligned_delta_abs_max", float(np.max(np.abs(aligned_delta))))
            update_metric(quat_metrics, f"{side}_quat_xyz_raw_delta_mae", mae(raw_delta))
            update_metric(quat_metrics, f"{side}_quat_xyz_aligned_delta_mae", mae(aligned_delta))

        for lag in range(0, args.max_lag + 1):
            if lag == 0:
                curr = slice(None)
                fut = slice(None)
            else:
                curr = slice(None, -lag)
                fut = slice(lag, None)
            for side in ["left", "right"]:
                action_pose = action[f"{side}_pose6"][curr]
                state_pose_future = state[f"{side}_pose6"][fut]
                action_pinch = action[f"{side}_pinch"][curr]
                state_pinch_future = state[f"{side}_pinch"][fut]
                err_pose = action_pose - state_pose_future
                err_pinch = action_pinch - state_pinch_future
                update_metric(target_lag_metrics[lag], f"{side}_pose6_target_mae", mae(err_pose))
                update_metric(target_lag_metrics[lag], f"{side}_pose6_target_bias", bias(err_pose))
                update_metric(target_lag_metrics[lag], f"{side}_pinch_target_mae", mae(err_pinch))
                update_metric(target_lag_metrics[lag], f"{side}_pinch_target_bias", bias(err_pinch))

        for lag in range(1, args.max_lag + 1):
            curr = slice(None, -lag)
            fut = slice(lag, None)
            for side in ["left", "right"]:
                converted_delta = action[f"{side}_pose6"][curr] - state[f"{side}_pose6"][curr]
                future_delta = state[f"{side}_pose6"][fut] - state[f"{side}_pose6"][curr]
                update_metric(
                    delta_lag_metrics[lag],
                    f"{side}_converted_vs_state_delta_mae",
                    mae(converted_delta - future_delta),
                )
                update_metric(
                    delta_lag_metrics[lag],
                    f"{side}_converted_vs_state_delta_corr",
                    corr(converted_delta, future_delta),
                )

    print("\nTarget semantics: lower MAE means actions[t] is closer to state[t+lag].")
    print("lag | left_pose_mae | right_pose_mae | left_pinch_mae | right_pinch_mae")
    print("----+---------------+----------------+----------------+-----------------")
    for lag in range(0, args.max_lag + 1):
        m = target_lag_metrics[lag]
        print(
            f"{lag:>3} | "
            f"{summarize(m.get('left_pose6_target_mae', [])):.6f}      | "
            f"{summarize(m.get('right_pose6_target_mae', [])):.6f}       | "
            f"{summarize(m.get('left_pinch_target_mae', [])):.6f}       | "
            f"{summarize(m.get('right_pinch_target_mae', [])):.6f}"
        )

    print("\nConverted delta check: lower MAE/higher corr means action_abs[t]-state[t] matches state[t+lag]-state[t].")
    print("lag | left_delta_mae | left_corr | right_delta_mae | right_corr")
    print("----+----------------+-----------+-----------------+-----------")
    for lag in range(1, args.max_lag + 1):
        m = delta_lag_metrics[lag]
        print(
            f"{lag:>3} | "
            f"{summarize(m.get('left_converted_vs_state_delta_mae', [])):.6f}       | "
            f"{summarize(m.get('left_converted_vs_state_delta_corr', [])):.6f}  | "
            f"{summarize(m.get('right_converted_vs_state_delta_mae', [])):.6f}        | "
            f"{summarize(m.get('right_converted_vs_state_delta_corr', [])):.6f}"
        )

    print("\nQuaternion sign diagnostics: high flip rate or raw>>aligned means direct quaternion subtraction creates spikes.")
    for key in sorted(quat_metrics):
        print(f"{key}: {summarize(quat_metrics[key]):.6f}")

    print("\nReadout:")
    print("- If target lag 0 is best, actions are same-frame absolute targets; current conversion creates near-zero pose deltas.")
    print("- If target lag 1 is best, actions are next-frame absolute targets; current conversion is one-step delta-like.")
    print("- If quaternion raw delta max/MAE is much larger than aligned, add quaternion sign alignment before subtracting.")


if __name__ == "__main__":
    main()
