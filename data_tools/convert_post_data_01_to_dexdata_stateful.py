#!/usr/bin/env python3
from __future__ import annotations

from data_tools import convert_post_data_01_to_dexdata as base


def build_state(row):
    # Preserve the first 16 dims used by the existing DM0 absolute/delta pipeline:
    #   0..6   : left_tcp
    #   7..13  : right_tcp
    #   14     : left_pinch
    #   15     : right_pinch
    #
    # Append the delta_tcp controller state so the model can observe the same
    # delta signal it is asked to predict in action space:
    #   16..21 : left_delta_tcp
    #   22..27 : right_delta_tcp
    left_tcp = base.to_float_list(row["observation.state.left_tcp"])
    right_tcp = base.to_float_list(row["observation.state.right_tcp"])
    left_pinch = [base.safe_float(row["observation.state.left_pinch"])]
    right_pinch = [base.safe_float(row["observation.state.right_pinch"])]
    left_delta = base.to_float_list(row["observation.state.left_delta_tcp"])
    right_delta = base.to_float_list(row["observation.state.right_delta_tcp"])

    state = left_tcp + right_tcp + left_pinch + right_pinch + left_delta + right_delta

    if len(left_tcp) != 7 or len(right_tcp) != 7:
        raise ValueError(
            f"Unexpected tcp dims: left={len(left_tcp)} right={len(right_tcp)}; expected 7 and 7"
        )
    if len(left_delta) != 6 or len(right_delta) != 6:
        raise ValueError(
            f"Unexpected delta dims in state: left={len(left_delta)} right={len(right_delta)}; expected 6 and 6"
        )
    if len(state) != 28:
        raise ValueError(f"Unexpected state dim: {len(state)}; expected 28")
    return state


def main() -> None:
    base.build_state = build_state
    base.main()


if __name__ == "__main__":
    main()
