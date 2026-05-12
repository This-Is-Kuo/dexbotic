#!/usr/bin/env python3
"""Preflight checks for DM0 DexData/jsonl/video alignment.

This script intentionally keeps dependencies light. It prefers ffprobe/ffmpeg
for video metadata and sampled-frame extraction, and falls back to OpenCV only
when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None


IMAGE_KEYS = ("images_1", "images_2", "images_3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit DM0 DexData jsonl/video/action/state alignment before training."
    )
    parser.add_argument("--dexdata_root", type=str, required=True)
    parser.add_argument("--jsonl_dir", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--num_samples_per_episode", type=int, default=10)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--path_alias",
        action="append",
        default=[],
        help="Optional path rewrite OLD=NEW, useful for container paths such as /dexbotic=/home/user/dexbotic.",
    )
    parser.add_argument(
        "--strict_paths",
        action="store_true",
        help="Do not auto-remap /dexbotic to the current repository root.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def path_aliases(args: argparse.Namespace) -> list[tuple[str, str]]:
    aliases: list[tuple[str, str]] = []
    for item in args.path_alias:
        if "=" not in item:
            raise ValueError(f"Invalid --path_alias {item!r}; expected OLD=NEW")
        old, new = item.split("=", 1)
        aliases.append((old.rstrip("/"), new.rstrip("/")))
    if not args.strict_paths:
        aliases.append(("/dexbotic", str(repo_root())))
    return aliases


def rewrite_path(path: str | Path, aliases: list[tuple[str, str]]) -> Path:
    text = str(path)
    for old, new in aliases:
        if text == old or text.startswith(old + "/"):
            return Path(new + text[len(old) :])
    return Path(text)


def existing_path(path: str | Path, aliases: list[tuple[str, str]]) -> tuple[Path, bool]:
    original = Path(path)
    if original.exists():
        return original, False
    rewritten = rewrite_path(original, aliases)
    if rewritten.exists():
        return rewritten, True
    return original, False


def resolve_video_path(video_dir: Path, rel_url: str, aliases: list[tuple[str, str]]) -> tuple[Path, bool, bool]:
    path = video_dir / rel_url
    if path.exists():
        return path, False, False

    rewritten = rewrite_path(path, aliases)
    if rewritten.exists():
        return rewritten, True, False

    if path.is_symlink():
        target = os.readlink(path)
        target_path = Path(target)
        if not target_path.is_absolute():
            target_path = path.parent / target_path
        rewritten_target = rewrite_path(target_path, aliases)
        if rewritten_target.exists():
            return rewritten_target, True, True
        return target_path, False, True

    return path, False, False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def as_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return value
    return value


def finite_vector(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            result.append(float("nan"))
    return result


class RunningVectorStats:
    def __init__(self) -> None:
        self.count = 0
        self.dim: int | None = None
        self.sum: list[float] = []
        self.sum_sq: list[float] = []
        self.min: list[float] = []
        self.max: list[float] = []
        self.nan_count = 0
        self.inf_count = 0

    def update(self, values: list[float]) -> None:
        if self.dim is None:
            self.dim = len(values)
            self.sum = [0.0] * self.dim
            self.sum_sq = [0.0] * self.dim
            self.min = [float("inf")] * self.dim
            self.max = [float("-inf")] * self.dim
        if len(values) != self.dim:
            return
        self.count += 1
        for i, value in enumerate(values):
            if math.isnan(value):
                self.nan_count += 1
                continue
            if math.isinf(value):
                self.inf_count += 1
                continue
            self.sum[i] += value
            self.sum_sq[i] += value * value
            self.min[i] = min(self.min[i], value)
            self.max[i] = max(self.max[i], value)

    def summary(self) -> dict[str, Any]:
        if not self.count or self.dim is None:
            return {"count": self.count, "dim": self.dim}
        mean = [x / self.count for x in self.sum]
        std = []
        for i, mu in enumerate(mean):
            var = max(0.0, self.sum_sq[i] / self.count - mu * mu)
            std.append(math.sqrt(var))
        return {
            "count": self.count,
            "dim": self.dim,
            "min": self.min,
            "max": self.max,
            "mean": mean,
            "std": std,
            "near_zero_std_indices": [i for i, value in enumerate(std) if value < 1e-6],
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
        }


def run_json(cmd: list[str]) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def parse_ratio(text: str | None) -> float | None:
    if not text or text == "0/0":
        return None
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else None
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def probe_video(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "video_path": str(path),
        "exists": path.exists(),
        "is_symlink": path.is_symlink(),
        "probe_backend": None,
        "fps": None,
        "frame_count": None,
        "duration": None,
        "start_time": None,
        "has_b_frames": None,
        "width": None,
        "height": None,
    }
    if not path.exists():
        return report

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        data = run_json(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate,nb_frames,start_time,duration,has_b_frames,width,height",
                "-show_entries",
                "format=start_time,duration",
                "-of",
                "json",
                str(path),
            ]
        )
        if data and data.get("streams"):
            stream = data["streams"][0]
            fmt = data.get("format", {})
            report["probe_backend"] = "ffprobe"
            report["fps"] = parse_ratio(stream.get("avg_frame_rate")) or parse_ratio(stream.get("r_frame_rate"))
            report["frame_count"] = int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None
            report["duration"] = as_float(stream.get("duration")) or as_float(fmt.get("duration"))
            report["start_time"] = as_float(stream.get("start_time"))
            if report["start_time"] is None:
                report["start_time"] = as_float(fmt.get("start_time"))
            report["has_b_frames"] = stream.get("has_b_frames")
            report["width"] = stream.get("width")
            report["height"] = stream.get("height")
            return report

    try:
        import cv2  # type: ignore
    except Exception:
        return report

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return report
    report["probe_backend"] = "opencv"
    report["fps"] = cap.get(cv2.CAP_PROP_FPS) or None
    report["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
    report["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    report["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
    if report["fps"] and report["frame_count"]:
        report["duration"] = report["frame_count"] / report["fps"]
    report["start_time"] = 0.0
    cap.release()
    return report


def extract_frame(video_path: Path, frame_idx: int, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        vf = f"select=eq(n\\,{frame_idx})"
        cmd = [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            vf,
            "-frames:v",
            "1",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return output_path.exists() and output_path.stat().st_size > 0
        except subprocess.CalledProcessError:
            return False

    try:
        import cv2  # type: ignore
    except Exception:
        return False
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    return bool(cv2.imwrite(str(output_path), frame))


def annotate_image(path: Path, lines: list[str]) -> None:
    if Image is None or ImageDraw is None or not path.exists():
        return
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    line_height = 16
    box_height = line_height * len(lines) + 8
    draw.rectangle((0, 0, img.width, box_height), fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((6, 4 + i * line_height), line, fill=(255, 255, 255))
    img.save(path)


def list_jsonl_files(jsonl_dir: Path) -> list[Path]:
    return sorted(path for path in jsonl_dir.rglob("*.jsonl") if path.is_file())


def row_timestamp(row: dict[str, Any]) -> float | None:
    extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
    return as_float(extra.get("timestamp"))


def row_frame_index(row: dict[str, Any], image_key: str = "images_1") -> int | None:
    extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
    value = extra.get("frame_index")
    if value is None and isinstance(row.get(image_key), dict):
        value = row[image_key].get("frame_idx")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    aliases = path_aliases(args)
    random.seed(args.seed)

    dexdata_root, dex_remapped = existing_path(args.dexdata_root, aliases)
    jsonl_dir, jsonl_remapped = existing_path(args.jsonl_dir, aliases)
    video_dir, video_remapped = existing_path(args.video_dir, aliases)
    output_dir = rewrite_path(args.output_dir, aliases)
    output_dir.mkdir(parents=True, exist_ok=True)

    suspicious: list[dict[str, Any]] = []
    episode_reports: list[dict[str, Any]] = []
    video_reports_by_url: dict[str, dict[str, Any]] = {}
    action_stats = RunningVectorStats()
    state_stats = RunningVectorStats()
    action_dims: Counter[int] = Counter()
    state_dims: Counter[int] = Counter()
    prompt_empty = 0
    total_rows = 0
    errors = 0
    warnings = 0

    if dex_remapped or jsonl_remapped or video_remapped:
        print(
            "WARNING path remap applied: "
            f"dexdata_root={dexdata_root}, jsonl_dir={jsonl_dir}, video_dir={video_dir}"
        )
        warnings += 1

    if not dexdata_root.exists():
        print(f"ERROR dexdata_root does not exist: {dexdata_root}")
        errors += 1
    if not jsonl_dir.exists():
        print(f"ERROR jsonl_dir does not exist: {jsonl_dir}")
        errors += 1
    if not video_dir.exists():
        print(f"ERROR video_dir does not exist: {video_dir}")
        errors += 1

    jsonl_files = list_jsonl_files(jsonl_dir) if jsonl_dir.exists() else []
    selected_jsonl_files = jsonl_files[: args.num_episodes] if args.num_episodes > 0 else jsonl_files

    for jsonl_path in selected_jsonl_files:
        rows = load_jsonl(jsonl_path)
        if not rows:
            suspicious.append({"severity": "ERROR", "episode": jsonl_path.name, "reason": "empty_jsonl"})
            errors += 1
            continue

        episode_id = rows[0].get("extra", {}).get("episode_index", jsonl_path.stem)
        frame_indices = [row_frame_index(row) for row in rows]
        timestamps = [row_timestamp(row) for row in rows]
        valid_frames = [x for x in frame_indices if x is not None]
        valid_timestamps = [x for x in timestamps if x is not None and not math.isnan(x)]
        first_ts = valid_timestamps[0] if valid_timestamps else None

        step_continuous = valid_frames == list(range(valid_frames[0], valid_frames[0] + len(valid_frames))) if valid_frames else False
        timestamp_monotonic = all(
            valid_timestamps[i] <= valid_timestamps[i + 1] for i in range(len(valid_timestamps) - 1)
        )
        frame_starts_zero = bool(valid_frames and valid_frames[0] == 0)
        timestamp_starts_zero = first_ts is not None and abs(first_ts) < 1e-6

        if not frame_starts_zero:
            suspicious.append({"severity": "WARNING", "episode": jsonl_path.name, "reason": "frame_index_not_start_zero", "value": valid_frames[0] if valid_frames else None})
            warnings += 1
        if first_ts is not None and not timestamp_starts_zero:
            suspicious.append({"severity": "WARNING", "episode": jsonl_path.name, "reason": "timestamp_not_start_zero", "value": first_ts})
            warnings += 1
        if not step_continuous:
            suspicious.append({"severity": "ERROR", "episode": jsonl_path.name, "reason": "frame_index_not_continuous"})
            errors += 1
        if not timestamp_monotonic:
            suspicious.append({"severity": "ERROR", "episode": jsonl_path.name, "reason": "timestamp_not_monotonic"})
            errors += 1

        max_mapping_error = 0.0
        mean_mapping_error = 0.0
        mapping_errors: list[float] = []
        camera_urls: dict[str, str] = {}
        for row_idx, row in enumerate(rows):
            total_rows += 1
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                prompt_empty += 1
            action = finite_vector(row.get("action"))
            state = finite_vector(row.get("state"))
            action_dims[len(action)] += 1
            state_dims[len(state)] += 1
            action_stats.update(action)
            state_stats.update(state)

            for image_key in IMAGE_KEYS:
                image_info = row.get(image_key)
                if not isinstance(image_info, dict):
                    continue
                url = image_info.get("url")
                if isinstance(url, str):
                    camera_urls[image_key] = url
                    if url not in video_reports_by_url:
                        path, remapped, broken_symlink = resolve_video_path(video_dir, url, aliases)
                        video_report = probe_video(path)
                        video_report.update(
                            {
                                "camera_key": image_key,
                                "url": url,
                                "path_remapped": remapped,
                                "broken_symlink_recovered": broken_symlink and path.exists(),
                            }
                        )
                        video_reports_by_url[url] = video_report

                    report = video_reports_by_url[url]
                    frame_idx = row_frame_index(row, image_key)
                    if not report.get("exists"):
                        suspicious.append({"severity": "ERROR", "episode": jsonl_path.name, "row": row_idx, "camera": image_key, "reason": "video_missing", "url": url, "path": report.get("video_path")})
                        errors += 1
                    elif frame_idx is not None and report.get("frame_count") is not None and frame_idx >= int(report["frame_count"]):
                        suspicious.append({"severity": "ERROR", "episode": jsonl_path.name, "row": row_idx, "camera": image_key, "reason": "frame_index_out_of_range", "frame_idx": frame_idx, "frame_count": report["frame_count"], "url": url})
                        errors += 1

            if first_ts is not None and timestamps[row_idx] is not None:
                fps_values = [
                    video_reports_by_url[url].get("fps")
                    for url in camera_urls.values()
                    if url in video_reports_by_url and video_reports_by_url[url].get("fps")
                ]
                if fps_values and frame_indices[row_idx] is not None:
                    fps = float(fps_values[0])
                    expected = round((float(timestamps[row_idx]) - first_ts) * fps)
                    error = abs(expected - int(frame_indices[row_idx]))
                    mapping_errors.append(error)
                    if error > 1:
                        suspicious.append({"severity": "WARNING", "episode": jsonl_path.name, "row": row_idx, "reason": "timestamp_frame_mapping_error", "timestamp": timestamps[row_idx], "frame_idx": frame_indices[row_idx], "expected_frame": expected, "fps": fps, "error": error})
                        warnings += 1

        if mapping_errors:
            max_mapping_error = max(mapping_errors)
            mean_mapping_error = statistics.fmean(mapping_errors)

        camera_frame_counts = [
            video_reports_by_url[url].get("frame_count")
            for url in camera_urls.values()
            if url in video_reports_by_url and video_reports_by_url[url].get("frame_count") is not None
        ]
        camera_start_times = [
            video_reports_by_url[url].get("start_time")
            for url in camera_urls.values()
            if url in video_reports_by_url and video_reports_by_url[url].get("start_time") is not None
        ]
        camera_lengths_consistent = len(set(camera_frame_counts)) <= 1
        camera_start_consistent = (
            max(camera_start_times) - min(camera_start_times) < 1e-6 if len(camera_start_times) > 1 else True
        )
        if not camera_lengths_consistent:
            suspicious.append({"severity": "WARNING", "episode": jsonl_path.name, "reason": "camera_frame_count_mismatch", "frame_counts": camera_frame_counts})
            warnings += 1
        if not camera_start_consistent:
            suspicious.append({"severity": "WARNING", "episode": jsonl_path.name, "reason": "camera_start_time_mismatch", "start_times": camera_start_times})
            warnings += 1

        episode_reports.append(
            {
                "episode_file": jsonl_path.name,
                "episode_id": episode_id,
                "num_steps": len(rows),
                "frame_start": valid_frames[0] if valid_frames else None,
                "frame_end": valid_frames[-1] if valid_frames else None,
                "frame_continuous": step_continuous,
                "timestamp_start": first_ts,
                "timestamp_end": valid_timestamps[-1] if valid_timestamps else None,
                "timestamp_monotonic": timestamp_monotonic,
                "prompt_empty_count": sum(1 for row in rows if not str(row.get("prompt", "")).strip()),
                "action_dims": dict(Counter(len(finite_vector(row.get("action"))) for row in rows)),
                "state_dims": dict(Counter(len(finite_vector(row.get("state"))) for row in rows)),
                "max_timestamp_frame_error": max_mapping_error,
                "mean_timestamp_frame_error": mean_mapping_error,
                "camera_frame_counts": camera_frame_counts,
                "camera_start_times": camera_start_times,
                "camera_lengths_consistent": camera_lengths_consistent,
                "camera_start_times_consistent": camera_start_consistent,
            }
        )

        sample_indices = list(range(len(rows)))
        if len(sample_indices) > args.num_samples_per_episode:
            sample_indices = sorted(random.sample(sample_indices, args.num_samples_per_episode))
        for row_idx in sample_indices:
            row = rows[row_idx]
            for image_key in IMAGE_KEYS:
                image_info = row.get(image_key)
                if not isinstance(image_info, dict) or image_info.get("type") != "video":
                    continue
                url = image_info.get("url")
                frame_idx = int(image_info.get("frame_idx", row_frame_index(row, image_key) or 0))
                path, _, _ = resolve_video_path(video_dir, str(url), aliases)
                out_path = output_dir / "sample_frames" / jsonl_path.stem / f"step_{row_idx:06d}_{image_key}_frame_{frame_idx:06d}.jpg"
                if extract_frame(path, frame_idx, out_path):
                    action = finite_vector(row.get("action"))[:6]
                    state = finite_vector(row.get("state"))[:6]
                    annotate_image(
                        out_path,
                        [
                            f"episode={episode_id} step={row_idx} frame={frame_idx}",
                            f"timestamp={row_timestamp(row)} camera={image_key}",
                            "action[:6]=" + ",".join(f"{x:.4g}" for x in action),
                            "state[:6]=" + ",".join(f"{x:.4g}" for x in state),
                        ],
                    )
                else:
                    suspicious.append({"severity": "WARNING", "episode": jsonl_path.name, "row": row_idx, "camera": image_key, "reason": "sample_frame_extract_failed", "url": url, "frame_idx": frame_idx, "path": str(path)})
                    warnings += 1

    video_reports = list(video_reports_by_url.values())
    for report in video_reports:
        start_time = report.get("start_time")
        if start_time is not None and abs(float(start_time)) > 1e-6:
            suspicious.append({"severity": "WARNING", "reason": "video_start_time_not_zero", "url": report.get("url"), "start_time": start_time, "path": report.get("video_path")})
            warnings += 1
        if report.get("has_b_frames") not in (None, 0, "0"):
            suspicious.append({"severity": "WARNING", "reason": "video_has_b_frames", "url": report.get("url"), "has_b_frames": report.get("has_b_frames"), "path": report.get("video_path")})
            warnings += 1

    summary = {
        "inputs": {
            "dexdata_root": str(dexdata_root),
            "jsonl_dir": str(jsonl_dir),
            "video_dir": str(video_dir),
            "output_dir": str(output_dir),
            "path_aliases": aliases,
        },
        "num_jsonl_files_total": len(jsonl_files),
        "num_jsonl_files_checked": len(selected_jsonl_files),
        "num_rows_checked": total_rows,
        "prompt_empty_count": prompt_empty,
        "action_dims": dict(action_dims),
        "state_dims": dict(state_dims),
        "action_stats": action_stats.summary(),
        "state_stats": state_stats.summary(),
        "num_video_files_referenced": len(video_reports),
        "num_missing_videos": sum(1 for report in video_reports if not report.get("exists")),
        "num_videos_start_time_nonzero": sum(
            1 for report in video_reports if report.get("start_time") is not None and abs(float(report["start_time"])) > 1e-6
        ),
        "num_videos_with_b_frames": sum(1 for report in video_reports if report.get("has_b_frames") not in (None, 0, "0")),
        "num_errors": errors,
        "num_warnings": warnings,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(
        output_dir / "per_episode_report.csv",
        episode_reports,
        [
            "episode_file",
            "episode_id",
            "num_steps",
            "frame_start",
            "frame_end",
            "frame_continuous",
            "timestamp_start",
            "timestamp_end",
            "timestamp_monotonic",
            "prompt_empty_count",
            "action_dims",
            "state_dims",
            "max_timestamp_frame_error",
            "mean_timestamp_frame_error",
            "camera_frame_counts",
            "camera_start_times",
            "camera_lengths_consistent",
            "camera_start_times_consistent",
        ],
    )
    write_csv(
        output_dir / "per_camera_video_report.csv",
        video_reports,
        [
            "camera_key",
            "url",
            "video_path",
            "exists",
            "is_symlink",
            "path_remapped",
            "broken_symlink_recovered",
            "probe_backend",
            "fps",
            "frame_count",
            "duration",
            "start_time",
            "has_b_frames",
            "width",
            "height",
        ],
    )
    suspicious_fields = sorted({key for row in suspicious for key in row.keys()})
    write_csv(output_dir / "suspicious_samples.csv", suspicious, suspicious_fields or ["severity", "reason"])

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        print(f"ERROR found {errors} blocking data issues. See {output_dir / 'suspicious_samples.csv'}")
        return 2
    if warnings:
        print(f"WARNING found {warnings} suspicious items. See {output_dir / 'suspicious_samples.csv'}")
        return 1
    print("OK data alignment preflight passed for checked episodes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
