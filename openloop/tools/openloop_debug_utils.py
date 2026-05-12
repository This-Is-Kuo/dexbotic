from __future__ import annotations

import csv
import math
import pathlib
from typing import Any

import numpy as np


def load_array(path: str | pathlib.Path) -> np.ndarray:
    return np.asarray(np.load(path))


def ensure_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return array.reshape(-1, array.shape[-1])
    raise ValueError(f"Expected a 2D or 3D array, got shape {array.shape}.")


def nanmean_safe(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def nanstd_safe(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanstd(values))


def pearson_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std < 1e-12 or y_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def align_valid_1d(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(gt)
    return pred[mask], gt[mask]


def per_dim_metrics(pred: np.ndarray, gt: np.ndarray) -> list[dict[str, Any]]:
    pred = ensure_2d(pred)
    gt = ensure_2d(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred shape {pred.shape} must match gt shape {gt.shape}")

    rows: list[dict[str, Any]] = []
    for dim in range(pred.shape[1]):
        pred_dim, gt_dim = align_valid_1d(pred[:, dim], gt[:, dim])
        if len(pred_dim) == 0:
            rows.append(
                {
                    "dim": dim,
                    "mae": math.nan,
                    "rmse": math.nan,
                    "pearson": math.nan,
                    "bias": math.nan,
                    "std_gt": math.nan,
                    "std_pred": math.nan,
                    "std_ratio": math.nan,
                    "max_abs_error": math.nan,
                }
            )
            continue
        error = pred_dim - gt_dim
        std_gt = float(np.std(gt_dim))
        std_pred = float(np.std(pred_dim))
        rows.append(
            {
                "dim": dim,
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "pearson": pearson_1d(pred_dim, gt_dim),
                "bias": float(np.mean(error)),
                "std_gt": std_gt,
                "std_pred": std_pred,
                "std_ratio": float(std_pred / std_gt) if std_gt > 1e-12 else math.nan,
                "max_abs_error": float(np.max(np.abs(error))),
            }
        )
    return rows


def save_csv(rows: list[dict[str, Any]], path: str | pathlib.Path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_metric_table(rows: list[dict[str, Any]], headers: list[str] | None = None) -> str:
    if not rows:
        return ""
    if headers is None:
        headers = list(rows[0].keys())
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                text = "nan" if math.isnan(value) else f"{value:.6f}"
            else:
                text = str(value)
            widths[header] = max(widths[header], len(text))

    lines = []
    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    sep_line = "-+-".join("-" * widths[header] for header in headers)
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        parts = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                text = "nan" if math.isnan(value) else f"{value:.6f}"
            else:
                text = str(value)
            parts.append(text.ljust(widths[header]))
        lines.append(" | ".join(parts))
    return "\n".join(lines)
