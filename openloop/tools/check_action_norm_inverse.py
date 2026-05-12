#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dexbotic.data.dataset.dex_dataset import load_jsonl
from dexbotic.data.dataset.transform.action import ActionNorm, AddTrajectory, DeltaAction, PadAction, PadState
from dexbotic.data.dataset.transform.common import ToNumpy
from dexbotic.data.dataset.transform.output import AbsoluteAction, ActionDenorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-dir", default="/dexbotic/data/post_data_01_dexdata/jsonl")
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--trajectory-length", type=int, default=50)
    parser.add_argument("--non-delta-mask", default="6,13")
    parser.add_argument(
        "--action-is-delta",
        type=lambda v: str(v).lower() in {"1", "true", "yes", "y", "on"},
        default=True,
        help="Whether the jsonl action is already a delta/next-target action. post_data_01 should use true.",
    )
    return parser.parse_args()


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _load_norm_stats(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    if "norm_stats" in data:
        data = data["norm_stats"]
    return ToNumpy()(data)


def _sample_episode_frames(
    jsonl_dir: pathlib.Path,
    num_samples: int,
    seed: int,
) -> list[tuple[pathlib.Path, int]]:
    files = sorted(jsonl_dir.glob("episode_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No episode_*.jsonl found under {jsonl_dir}")

    rng = random.Random(seed)
    samples: list[tuple[pathlib.Path, int]] = []
    while len(samples) < num_samples:
        file_path = rng.choice(files)
        episode = load_jsonl(str(file_path), parse=True)
        if not episode:
            continue
        frame_idx = rng.randrange(len(episode))
        samples.append((file_path, frame_idx))
    return samples


def _episode_to_arrays(episode: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray([step["state"] for step in episode], dtype=np.float32)
    actions = np.asarray([step["action"] for step in episode], dtype=np.float32)
    return states, actions


def main() -> None:
    args = parse_args()
    jsonl_dir = pathlib.Path(args.jsonl_dir)
    norm_stats = _load_norm_stats(args.norm_stats)
    non_delta_mask = _parse_csv_ints(args.non_delta_mask)

    pad_state = PadState(ndim=args.state_dim, axis=-1)
    pad_action = PadAction(ndim=args.action_dim, axis=-1)
    add_trajectory = AddTrajectory(
        trajectory_length=args.trajectory_length,
        flatten=False,
        padding_mode="last",
    )
    delta_action = DeltaAction(enable=not args.action_is_delta)
    action_norm = ActionNorm(statistic_mapping=norm_stats, strict=False, use_quantiles=True)
    action_denorm = ActionDenorm(statistic_mapping=norm_stats, strict=False, use_quantiles=True)
    absolute_action = AbsoluteAction()

    raw_chunks: list[np.ndarray] = []
    recovered_chunks: list[np.ndarray] = []
    samples = _sample_episode_frames(jsonl_dir, args.num_samples, args.seed)

    for file_path, frame_idx in samples:
        episode = load_jsonl(str(file_path), parse=True)
        states, actions = _episode_to_arrays(episode)
        raw_action_dim = actions.shape[-1]
        payload = {
            "state": states,
            "action": actions,
            "meta_data": {
                "jsonl_file": str(file_path),
                "non_delta_mask": non_delta_mask,
                "periodic_mask": None,
                "periodic_range": None,
            },
        }

        payload = pad_state(payload)
        payload = pad_action(payload)
        payload = add_trajectory(payload)
        raw_chunk = np.asarray(payload["action"][frame_idx, :, :raw_action_dim], dtype=np.float32)
        payload = delta_action(payload)
        payload = action_norm(payload)
        payload = action_denorm(payload)
        payload = absolute_action(payload)
        recovered_chunk = np.asarray(
            payload["action"][frame_idx, :, :raw_action_dim],
            dtype=np.float32,
        )

        raw_chunks.append(raw_chunk)
        recovered_chunks.append(recovered_chunk)

    raw_np = np.stack(raw_chunks, axis=0)
    recovered_np = np.stack(recovered_chunks, axis=0)
    error = np.abs(recovered_np - raw_np)
    print(f"max_error: {float(np.max(error)):.10f}")
    print(f"mean_error: {float(np.mean(error)):.10f}")
    print(f"per_dim_error: {np.mean(error, axis=(0, 1)).tolist()}")


if __name__ == "__main__":
    main()
