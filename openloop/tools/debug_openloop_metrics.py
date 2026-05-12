#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openloop.tools.openloop_debug_utils import format_metric_table, load_array, per_dim_metrics, save_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--save_csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred = load_array(args.pred)
    gt = load_array(args.gt)
    rows = per_dim_metrics(pred, gt)
    print(format_metric_table(rows))
    if args.save_csv:
        save_csv(rows, args.save_csv)
        print(f"\nSaved CSV to {args.save_csv}")


if __name__ == "__main__":
    main()
