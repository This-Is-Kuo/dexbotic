#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a DexData dataset and transcode videos to all-intra H.264 without B-frames."
    )
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def ffprobe_stream(path: Path) -> dict[str, Any]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,has_b_frames,nb_frames,r_frame_rate,avg_frame_rate,start_time,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"No video stream found: {path}")
    return streams[0]


def copy_metadata(input_root: Path, output_root: Path) -> None:
    for name in ["jsonl", "merge_manifest.json", "split_manifest.json"]:
        src = input_root / name
        if not src.exists():
            continue
        dst = output_root / name
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def transcode_one(src: Path, dst: Path, input_root: Path, output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    rel = src.relative_to(input_root / "video")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp.mp4")
    if dst.exists() and not args.overwrite:
        stream = ffprobe_stream(dst)
        return {
            "url": rel.as_posix(),
            "status": "skipped",
            "has_b_frames": int(stream.get("has_b_frames", -1)),
            "nb_frames": stream.get("nb_frames"),
        }

    if tmp.exists():
        tmp.unlink()

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-bf",
        "0",
        "-g",
        "1",
        "-keyint_min",
        "1",
        "-sc_threshold",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(tmp),
    ]
    run(cmd)
    tmp.replace(dst)

    src_stream = ffprobe_stream(src)
    dst_stream = ffprobe_stream(dst)
    src_frames = src_stream.get("nb_frames")
    dst_frames = dst_stream.get("nb_frames")
    if src_frames not in (None, "N/A") and dst_frames not in (None, "N/A") and str(src_frames) != str(dst_frames):
        raise ValueError(f"Frame count mismatch for {rel}: source={src_frames} output={dst_frames}")
    if int(dst_stream.get("has_b_frames", -1)) != 0:
        raise ValueError(f"Output still has B-frames: {dst}")

    return {
        "url": rel.as_posix(),
        "status": "transcoded",
        "source_bytes": src.stat().st_size,
        "output_bytes": dst.stat().st_size,
        "source_nb_frames": src_frames,
        "output_nb_frames": dst_frames,
        "has_b_frames": int(dst_stream.get("has_b_frames", -1)),
        "start_time": dst_stream.get("start_time"),
        "duration": dst_stream.get("duration"),
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not (input_root / "jsonl").is_dir() or not (input_root / "video").is_dir():
        raise FileNotFoundError(f"Expected DexData jsonl/video dirs under {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    copy_metadata(input_root, output_root)

    videos = sorted((input_root / "video").rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No mp4 videos found under {input_root / 'video'}")

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_src = {
            executor.submit(
                transcode_one,
                src,
                output_root / "video" / src.relative_to(input_root / "video"),
                input_root,
                output_root,
                args,
            ): src
            for src in videos
        }
        for index, future in enumerate(as_completed(future_to_src), start=1):
            src = future_to_src[future]
            try:
                result = future.result()
                results.append(result)
                if index == 1 or index % 25 == 0 or index == len(videos):
                    print(f"[{index}/{len(videos)}] {result['status']} {result['url']}")
            except Exception as exc:
                failures.append(f"{src}: {exc}")
                print(f"[ERROR] {src}: {exc}")

    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "video_count": len(videos),
        "num_results": len(results),
        "num_failures": len(failures),
        "encoding": {
            "codec": "libx264",
            "crf": args.crf,
            "preset": args.preset,
            "bf": 0,
            "g": 1,
            "keyint_min": 1,
            "pix_fmt": "yuv420p",
        },
        "results": sorted(results, key=lambda item: item["url"]),
        "failures": failures,
    }
    with open(output_root / "no_bframes_transcode_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if failures:
        raise SystemExit(f"Failed to transcode {len(failures)} videos. See manifest for details.")
    print(f"[DONE] Transcoded {len(results)} videos to {output_root / 'video'}")
    print(f"Manifest: {output_root / 'no_bframes_transcode_manifest.json'}")


if __name__ == "__main__":
    main()
