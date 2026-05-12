#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openloop.tools.openloop_debug_utils import (
    ensure_2d,
    format_metric_table,
    load_array,
    pearson_1d,
    save_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--max_lag", type=int, default=20)
    parser.add_argument("--save_csv", default=None)
    return parser.parse_args()


def _pearson_at_lag(pred: np.ndarray, gt: np.ndarray, lag: int) -> float:
    if lag > 0:
        pred_slice = pred[lag:]
        gt_slice = gt[:-lag]
    elif lag < 0:
        pred_slice = pred[:lag]
        gt_slice = gt[-lag:]
    else:
        pred_slice = pred
        gt_slice = gt
    return pearson_1d(pred_slice, gt_slice)


def main() -> None:
    args = parse_args()
    pred = ensure_2d(load_array(args.pred))
    gt = ensure_2d(load_array(args.gt))
    if pred.shape != gt.shape:
        raise ValueError(f"pred shape {pred.shape} must match gt shape {gt.shape}")

    rows = []
    for dim in range(pred.shape[1]):
        pred_dim = pred[:, dim]
        gt_dim = gt[:, dim]
        corr_at_lag0 = _pearson_at_lag(pred_dim, gt_dim, 0)
        best_lag = 0
        best_corr = corr_at_lag0
        best_score = -math.inf if math.isnan(corr_at_lag0) else corr_at_lag0
        for lag in range(-args.max_lag, args.max_lag + 1):
            corr = _pearson_at_lag(pred_dim, gt_dim, lag)
            if math.isnan(corr):
                continue
            if corr > best_score:
                best_score = corr
                best_corr = corr
                best_lag = lag
        improvement = best_corr - corr_at_lag0 if not (math.isnan(best_corr) or math.isnan(corr_at_lag0)) else math.nan
        rows.append(
            {
                "dim": dim,
                "best_lag": best_lag,
                "best_corr": best_corr,
                "corr_at_lag0": corr_at_lag0,
                "improvement": improvement,
            }
        )

    print(format_metric_table(rows, headers=["dim", "best_lag", "best_corr", "corr_at_lag0", "improvement"]))
    if args.save_csv:
        save_csv(rows, args.save_csv)
        print(f"\nSaved CSV to {args.save_csv}")


if __name__ == "__main__":
    main()
