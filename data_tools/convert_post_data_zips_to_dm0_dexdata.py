#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

LOCAL_DEPS = Path(__file__).resolve().parents[1] / ".codex_deps"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

import pandas as pd

from data_tools.convert_lerobot_v2_post_origin_to_dexdata import (
    build_action,
    build_state,
    detect_schema,
    ensure_dir,
    output_video_rel,
    select_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert and merge post_data*.zip LeRobot v2 archives into DM0 DexData."
    )
    parser.add_argument(
        "--zip_dir",
        default="/mnt/datadisk/guoyaokun/downloads",
        help="Directory containing post_data*.zip archives.",
    )
    parser.add_argument(
        "--output_root",
        default="data/post_data_merged_dm0_dexdata",
        help="Merged DexData output root.",
    )
    parser.add_argument(
        "--archives",
        nargs="*",
        default=None,
        help="Optional explicit archive paths or names. Defaults to post_data*.zip in natural order.",
    )
    parser.add_argument(
        "--video_mode",
        choices=["extract", "skip"],
        default="extract",
        help="extract writes videos under output_root/video; skip only writes jsonl.",
    )
    parser.add_argument("--state_mode", choices=["minimal", "stateful"], default="minimal")
    parser.add_argument("--schema", choices=["auto", "post_data_01", "origin_102"], default="auto")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def archive_sort_key(path: Path) -> tuple[int, str]:
    if path.name == "post_data.zip":
        return (0, path.name)
    stem = path.stem
    suffix = stem.removeprefix("post_data")
    if suffix.isdigit():
        return (int(suffix), path.name)
    return (9999, path.name)


def resolve_archives(zip_dir: Path, names: list[str] | None) -> list[Path]:
    if names:
        paths = [Path(name) if Path(name).is_absolute() else zip_dir / name for name in names]
    else:
        paths = sorted(zip_dir.glob("post_data*.zip"), key=archive_sort_key)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing archives: {missing}")
    if not paths:
        raise FileNotFoundError(f"No post_data*.zip archives found under {zip_dir}")
    return paths


def load_jsonl_from_zip(zf: zipfile.ZipFile, name: str) -> list[dict[str, Any]]:
    rows = []
    with zf.open(name) as f:
        for raw_line in f:
            line = raw_line.decode("utf-8").strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json_from_zip(zf: zipfile.ZipFile, name: str) -> dict[str, Any]:
    with zf.open(name) as f:
        return json.load(f)


def load_task_map(zf: zipfile.ZipFile) -> dict[int, str]:
    return {int(row["task_index"]): str(row["task"]) for row in load_jsonl_from_zip(zf, "meta/tasks.jsonl")}


def load_feature_names(zf: zipfile.ZipFile) -> dict[str, list[str]]:
    info = load_json_from_zip(zf, "meta/info.json")
    result = {}
    for key, value in (info.get("features") or {}).items():
        names = value.get("names")
        if isinstance(names, list):
            result[key] = [str(name) for name in names]
    return result


def read_episode_dataframe(zf: zipfile.ZipFile, episode_index: int) -> pd.DataFrame:
    preferred = f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"
    name = preferred
    if preferred not in zf.namelist():
        suffix = f"/episode_{episode_index:06d}.parquet"
        matches = sorted(path for path in zf.namelist() if path.startswith("data/") and path.endswith(suffix))
        if not matches:
            raise FileNotFoundError(f"Missing episode parquet in archive: {preferred}")
        name = matches[0]
    return pd.read_parquet(io.BytesIO(zf.read(name)))


def extract_video(
    zf: zipfile.ZipFile,
    src_rel: Path,
    dst: Path,
    overwrite: bool,
) -> None:
    ensure_dir(dst.parent)
    if dst.exists() and not overwrite:
        return
    with zf.open(src_rel.as_posix()) as src, open(dst, "wb") as out:
        shutil.copyfileobj(src, out, length=1024 * 1024)


def resolve_clean_video_rel(
    zf: zipfile.ZipFile,
    episode: dict[str, Any],
    logical_view: str,
) -> tuple[str, Path]:
    camera_key, meta_rel = select_video(episode, logical_view)
    episode_index = int(episode["episode_index"])
    clean_rel = Path("videos") / "chunk-000" / camera_key / f"episode_{episode_index:06d}.mp4"
    if clean_rel.as_posix() in zf.namelist():
        return camera_key, clean_rel
    if meta_rel.as_posix() in zf.namelist():
        return camera_key, meta_rel
    raise FileNotFoundError(
        f"Missing {logical_view} video for episode {episode_index}: "
        f"tried {clean_rel.as_posix()} and {meta_rel.as_posix()}"
    )


def write_manifest(output_root: Path, archives: list[Path], rows: list[dict[str, Any]]) -> None:
    payload = {
        "format": "dm0_dexdata",
        "state_mode": rows[0]["state_mode"] if rows else None,
        "num_archives": len(archives),
        "archives": [str(path) for path in archives],
        "num_episodes": len(rows),
        "episodes": rows,
    }
    with open(output_root / "merge_manifest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def convert_archive(
    archive: Path,
    archive_order: int,
    output_root: Path,
    global_start_index: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    converted = []
    archive_label = archive.stem
    with zipfile.ZipFile(archive) as zf:
        task_map = load_task_map(zf)
        feature_names = load_feature_names(zf)
        episodes = load_jsonl_from_zip(zf, "meta/episodes.jsonl")
        for episode in episodes:
            if args.max_episodes is not None and global_start_index + len(converted) >= args.max_episodes:
                break

            source_episode_index = int(episode["episode_index"])
            global_episode_index = global_start_index + len(converted)
            df = read_episode_dataframe(zf, source_episode_index)
            if df.empty:
                raise ValueError(f"Empty episode {source_episode_index} in {archive}")
            episode_schema = detect_schema(df.iloc[0], args.schema)

            task_index = int(df["task_index"].dropna().iloc[0]) if "task_index" in df.columns else None
            prompt = task_map.get(task_index, str(episode.get("tasks", "")))

            video_rels: dict[str, str] = {}
            for logical_view in ["chest", "left", "right"]:
                _, src_rel = resolve_clean_video_rel(zf, episode, logical_view)
                dst_rel = (
                    Path(archive_label)
                    / output_video_rel(logical_view, src_rel).parent
                    / f"episode_{global_episode_index:06d}.mp4"
                )
                if args.video_mode == "extract":
                    extract_video(zf, src_rel, output_root / "video" / dst_rel, args.overwrite)
                video_rels[logical_view] = dst_rel.as_posix()

            jsonl_path = output_root / "jsonl" / f"episode_{global_episode_index:06d}.jsonl"
            ensure_dir(jsonl_path.parent)
            if jsonl_path.exists() and not args.overwrite:
                raise FileExistsError(f"{jsonl_path} exists. Use --overwrite.")

            with open(jsonl_path, "w", encoding="utf-8") as f:
                for row_idx, row in df.iterrows():
                    frame_idx = int(row["frame_index"]) if "frame_index" in row.index else int(row_idx)
                    record = {
                        "images_1": {"type": "video", "url": video_rels["chest"], "frame_idx": frame_idx},
                        "images_2": {"type": "video", "url": video_rels["left"], "frame_idx": frame_idx},
                        "images_3": {"type": "video", "url": video_rels["right"], "frame_idx": frame_idx},
                        "state": build_state(row, args.state_mode, episode_schema, feature_names),
                        "prompt": prompt,
                        "is_robot": True,
                        "action": build_action(row, episode_schema, feature_names),
                        "extra": {
                            "timestamp": float(row["timestamp"]) if "timestamp" in row.index else float("nan"),
                            "frame_index": frame_idx,
                            "episode_index": global_episode_index,
                            "source_episode_index": source_episode_index,
                            "source_archive": archive.name,
                            "source_archive_order": archive_order,
                            "task_index": task_index,
                            "source_format": "lerobot_v2_jsonl_post_data_zip",
                            "state_mode": args.state_mode,
                            "schema": episode_schema,
                        },
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            converted.append(
                {
                    "episode_index": global_episode_index,
                    "source_archive": archive.name,
                    "source_episode_index": source_episode_index,
                    "frames": len(df),
                    "prompt": prompt,
                    "state_mode": args.state_mode,
                    "schema": episode_schema,
                }
            )
            print(
                f"[OK] global={global_episode_index:06d} "
                f"source={archive.name}:{source_episode_index:06d} frames={len(df)}"
            )
    return converted


def main() -> None:
    args = parse_args()
    zip_dir = Path(args.zip_dir).resolve()
    output_root = Path(args.output_root).resolve()
    archives = resolve_archives(zip_dir, args.archives)

    ensure_dir(output_root / "jsonl")
    ensure_dir(output_root / "video")

    all_rows: list[dict[str, Any]] = []
    for archive_order, archive in enumerate(archives):
        rows = convert_archive(
            archive=archive,
            archive_order=archive_order,
            output_root=output_root,
            global_start_index=len(all_rows),
            args=args,
        )
        all_rows.extend(rows)
        if args.max_episodes is not None and len(all_rows) >= args.max_episodes:
            break

    write_manifest(output_root, archives, all_rows)
    print(f"[DONE] Converted {len(all_rows)} episodes from {len(archives)} archives.")
    print(f"Output jsonl dir: {output_root / 'jsonl'}")
    print(f"Output video dir: {output_root / 'video'}")
    print(f"Manifest: {output_root / 'merge_manifest.json'}")


if __name__ == "__main__":
    main()
