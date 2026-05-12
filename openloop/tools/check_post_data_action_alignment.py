#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-jsonl",
        default="/dexbotic/data/post_data_01_dexdata/jsonl/episode_000000.jsonl",
        help="Path to one converted post_data_01 episode jsonl.",
    )
    return parser.parse_args()


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in pathlib.Path(args.episode_jsonl).read_text().splitlines() if line.strip()]
    states = np.asarray([row["state"] for row in rows], dtype=np.float64)
    actions = np.asarray([row["action"] for row in rows], dtype=np.float64)

    if states.shape[1] not in {16, 28} or actions.shape[1] != 14:
        raise ValueError(
            f"Expected post_data_01 schema state_dim in {{16, 28}} and action_dim=14, got {states.shape[1]} and {actions.shape[1]}"
        )

    next_states = states[1:]
    curr_states = states[:-1]
    curr_actions = actions[:-1]

    left_pos_delta = next_states[:, 0:3] - curr_states[:, 0:3]
    right_pos_delta = next_states[:, 7:10] - curr_states[:, 7:10]

    print("Action-to-next-state alignment")
    print("dim | target                     | corr     | mae")
    print("----+----------------------------+----------+----------")
    for dim in range(3):
        print(
            f"{dim:<3} | left_tcp_xyz_delta[{dim}]      | {_corr(curr_actions[:, dim], left_pos_delta[:, dim]):.6f} | "
            f"{np.mean(np.abs(curr_actions[:, dim] - left_pos_delta[:, dim])):.6f}"
        )
    for offset, dim in enumerate(range(7, 10)):
        print(
            f"{dim:<3} | right_tcp_xyz_delta[{offset}]   | {_corr(curr_actions[:, dim], right_pos_delta[:, offset]):.6f} | "
            f"{np.mean(np.abs(curr_actions[:, dim] - right_pos_delta[:, offset])):.6f}"
        )
    print(
        f"6   | left_pinch_next              | {_corr(curr_actions[:, 6], next_states[:, 14]):.6f} | "
        f"{np.mean(np.abs(curr_actions[:, 6] - next_states[:, 14])):.6f}"
    )
    print(
        f"13  | right_pinch_next             | {_corr(curr_actions[:, 13], next_states[:, 15]):.6f} | "
        f"{np.mean(np.abs(curr_actions[:, 13] - next_states[:, 15])):.6f}"
    )

    if states.shape[1] >= 28:
        print("\nAction-to-current-state-delta copies inside state")
        print("dim | target                     | corr     | mae")
        print("----+----------------------------+----------+----------")
        for offset, dim in enumerate(range(0, 6)):
            state_dim = 16 + offset
            print(
                f"{dim:<3} | state_left_delta_tcp[{offset}]  | {_corr(curr_actions[:, dim], curr_states[:, state_dim]):.6f} | "
                f"{np.mean(np.abs(curr_actions[:, dim] - curr_states[:, state_dim])):.6f}"
            )
        for offset, dim in enumerate(range(7, 13)):
            state_dim = 22 + offset
            print(
                f"{dim:<3} | state_right_delta_tcp[{offset}] | {_corr(curr_actions[:, dim], curr_states[:, state_dim]):.6f} | "
                f"{np.mean(np.abs(curr_actions[:, dim] - curr_states[:, state_dim])):.6f}"
            )

    pad_to = max(states.shape[1], actions.shape[1], 32)
    padded_states = np.pad(states, ((0, 0), (0, pad_to - states.shape[1])))
    padded_actions = np.pad(actions, ((0, 0), (0, pad_to - actions.shape[1])))
    delta_action_output = padded_actions - padded_states
    delta_action_output[:, 6] = padded_actions[:, 6]
    delta_action_output[:, 13] = padded_actions[:, 13]

    print("\nEffect of current DeltaAction transform")
    print("dim | mean_action | mean_after_deltaaction | mean_state_ref")
    print("----+-------------+------------------------+---------------")
    for dim in range(14):
        print(
            f"{dim:<3} | {np.mean(padded_actions[:, dim]):.6f} | {np.mean(delta_action_output[:, dim]):.6f} | "
            f"{np.mean(padded_states[:, dim]):.6f}"
        )

    print(
        "\nInterpretation: if the alignment block shows corr≈1 and mae≈0 against next-state deltas/next pinch state, "
        "the converted action is already a delta/next-target action and training should not apply DeltaAction again."
    )


if __name__ == "__main__":
    main()
