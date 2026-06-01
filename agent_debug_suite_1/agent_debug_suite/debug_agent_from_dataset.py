#!/usr/bin/env python3
"""Run a duck-typed agent against frames from a LeRobot dataset.

This script is intentionally standalone: it does not import project-specific
robotics framework modules. The agent under test can be loaded from a Python
file or an installed module and only needs to provide an act(obs, task) method.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


INTERNAL_KEYS = {"frame_index", "episode_index", "index", "timestamp", "task_index"}


@dataclass(frozen=True)
class BufferProxy:
    """Small read-only dataset facade for standalone debug agents."""

    repo_id: str
    root: Path
    features: dict[str, dict[str, Any]]
    fps: int | None
    dataset: Any

    @property
    def sample_available(self) -> bool:
        return len(self.dataset) > 0

    def __len__(self) -> int:
        return len(self.dataset)

    def sample(self, temporal: int, batch_size: int) -> dict[str, torch.Tensor]:
        """Return a simple random temporal batch from the loaded dataset.

        This helper is deliberately small and read-only. It is meant for agents
        that expect a buffer-like object during construction, not for training
        throughput.
        """
        if temporal <= 0:
            raise ValueError(f"temporal must be positive, got {temporal}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if len(self.dataset) < temporal:
            raise RuntimeError(
                f"Dataset has {len(self.dataset)} frames, shorter than temporal={temporal}"
            )

        max_start = len(self.dataset) - temporal
        starts = np.random.randint(0, max_start + 1, size=batch_size)
        result: dict[str, list[torch.Tensor]] = {}
        for start in starts:
            frames = [_frame_to_numpy(self.dataset[int(start + offset)]) for offset in range(temporal)]
            for key, info in self.features.items():
                if key in INTERNAL_KEYS or key == "task":
                    continue
                if key not in frames[0]:
                    continue
                values = [_numpy_to_tensor(_convert_feature_value(frame[key], info)) for frame in frames]
                result.setdefault(key, []).append(torch.stack(values, dim=0))

        return {key: torch.stack(value, dim=0) for key, value in result.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug a standalone agent by replaying observations from a LeRobot dataset."
    )
    parser.add_argument("--root", type=Path, required=True, help="Path to the local LeRobot dataset root.")
    parser.add_argument(
        "--dataset-format",
        choices=("auto", "lerobot", "jsonl"),
        default="auto",
        help="Dataset format. auto uses LeRobot when meta/info.json exists, otherwise JSONL when root/jsonl exists.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="LeRobot repo_id. Defaults to the dataset root directory name.",
    )

    agent_source = parser.add_mutually_exclusive_group(required=True)
    agent_source.add_argument("--agent-file", type=Path, help="Path to a Python file containing the agent class.")
    agent_source.add_argument("--agent-module", help="Importable module path containing the agent class.")
    parser.add_argument("--agent-class", required=True, help="Name of the agent class to instantiate.")
    parser.add_argument(
        "--agent-kwargs-json",
        default="{}",
        help="JSON object passed as keyword arguments to the agent constructor.",
    )

    parser.add_argument("--episode", type=int, default=None, help="Optional episode index to load.")
    parser.add_argument("--start", type=int, default=0, help="First frame index within the loaded selection.")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to run.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--task", default=None, help="Optional task string override.")
    parser.add_argument(
        "--video-backend",
        default="pyav",
        help="LeRobot video backend. Defaults to pyav.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="For JSONL datasets, do not decode referenced video frames.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help=(
            "For JSONL datasets, optional directory containing local videos. "
            "Supports roots like root/video or LeRobot-style videos/chunk-000."
        ),
    )
    parser.add_argument("--sleep-s", type=float, default=0.0, help="Delay between act() calls.")
    parser.add_argument("--print-actions", action="store_true", help="Print full action arrays.")
    parser.add_argument("--stop-on-error", action="store_true", help="Exit after the first frame error.")
    return parser.parse_args()


def _load_module_from_file(path: Path) -> ModuleType:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Agent file does not exist: {resolved}")

    module_name = f"standalone_debug_agent_{resolved.stem}_{abs(hash(resolved))}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {resolved}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_agent_class(args: argparse.Namespace) -> type:
    if args.agent_file is not None:
        module = _load_module_from_file(args.agent_file)
    else:
        module = importlib.import_module(args.agent_module)

    try:
        cls = getattr(module, args.agent_class)
    except AttributeError as exc:
        raise AttributeError(f"{module.__name__} has no class named {args.agent_class!r}") from exc
    if not isinstance(cls, type):
        raise TypeError(f"{args.agent_class!r} is not a class")
    return cls


def _parse_agent_kwargs(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--agent-kwargs-json is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("--agent-kwargs-json must decode to a JSON object")
    return value


def _instantiate_agent(cls: type, spec: dict[str, dict[str, Any]], buffer: BufferProxy, kwargs: dict[str, Any]) -> Any:
    try:
        return cls(spec, buffer, **kwargs)
    except TypeError as first_error:
        try:
            return cls(**kwargs)
        except TypeError as second_error:
            raise TypeError(
                "Could not instantiate agent either as "
                f"{cls.__name__}(spec, buffer, **kwargs) or {cls.__name__}(**kwargs).\n"
                f"First error: {first_error}\nSecond error: {second_error}"
            ) from second_error


class SimpleMeta:
    def __init__(self, features: dict[str, dict[str, Any]], fps: int | None = None) -> None:
        self.features = features
        self.fps = fps


class JsonlDataset:
    """Small reader for the post_origin_data JSONL debug format."""

    def __init__(
        self,
        root: Path,
        episode: int | None = None,
        skip_images: bool = False,
        video_root: Path | None = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.jsonl_root = self.root / "jsonl"
        self.skip_images = skip_images
        self.video_root = video_root.expanduser().resolve() if video_root is not None else None
        self._warned_missing_images: set[str] = set()

        files = sorted(self.jsonl_root.glob("episode_*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No episode_*.jsonl files found under {self.jsonl_root}")

        if episode is not None:
            exact = self.jsonl_root / f"episode_{episode:06d}.jsonl"
            if exact.exists():
                files = [exact]
            elif 0 <= episode < len(files):
                files = [files[episode]]
            else:
                raise FileNotFoundError(
                    f"--episode {episode} did not match {exact.name} and is outside 0..{len(files) - 1}"
                )

        self.rows: list[dict[str, Any]] = []
        for path in files:
            with path.open("r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                    row["_jsonl_path"] = str(path)
                    self.rows.append(row)

        if not self.rows:
            raise ValueError(f"No frames loaded from {self.jsonl_root}")

        self.meta = SimpleMeta(self._infer_features(self.rows[0]), fps=30)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        frame: dict[str, Any] = {
            "observation.state": np.asarray(row.get("state", []), dtype=np.float32),
            "action": np.asarray(row.get("action", []), dtype=np.float32),
            "task": row.get("prompt", ""),
            "teleoperated": np.array([0], dtype=np.int64),
        }

        extra = row.get("extra")
        if isinstance(extra, dict):
            for key in ("frame_index", "episode_index", "timestamp", "task_index"):
                if key in extra:
                    frame[key] = extra[key]

        if not self.skip_images:
            for value in row.values():
                if not self._is_video_ref(value):
                    continue
                feature_name = self._image_feature_name(value)
                if feature_name is None:
                    continue
                video_path = self._resolve_video_path(value["url"])
                if not video_path.exists():
                    self._warn_missing_image(video_path)
                    continue
                frame[feature_name] = self._read_video_frame(video_path, int(value.get("frame_idx", 0)))

        return frame

    def _infer_features(self, row: dict[str, Any]) -> dict[str, dict[str, Any]]:
        features: dict[str, dict[str, Any]] = {
            "observation.state": {
                "dtype": "float32",
                "shape": tuple(np.asarray(row.get("state", []), dtype=np.float32).shape),
            },
            "action": {
                "dtype": "float32",
                "shape": tuple(np.asarray(row.get("action", []), dtype=np.float32).shape),
            },
            "task": {"dtype": "string", "shape": (1,)},
            "teleoperated": {"dtype": "int64", "shape": (1,)},
        }

        if self.skip_images:
            return features

        for value in row.values():
            if not self._is_video_ref(value):
                continue
            feature_name = self._image_feature_name(value)
            if feature_name is None:
                continue
            video_path = self._resolve_video_path(value["url"])
            shape = self._video_feature_shape(video_path)
            if shape is not None:
                features[feature_name] = {"dtype": "video", "shape": shape}
        return features

    @staticmethod
    def _is_video_ref(value: Any) -> bool:
        return isinstance(value, dict) and value.get("type") == "video" and isinstance(value.get("url"), str)

    @staticmethod
    def _image_feature_name(value: dict[str, Any]) -> str | None:
        parts = Path(value["url"]).parts
        if len(parts) >= 2:
            return parts[-2]
        return None

    def _resolve_video_path(self, url: str) -> Path:
        rel = Path(url)
        candidates: list[Path] = []
        if self.video_root is not None:
            candidates.extend(self._video_root_candidates(self.video_root, rel))
        candidates.extend(self._video_root_candidates(self.root / "video", rel))
        candidates.extend(self._video_root_candidates(self.root / "videos", rel))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @staticmethod
    def _video_root_candidates(video_root: Path, rel: Path) -> list[Path]:
        candidates = [video_root / rel]
        parts = rel.parts
        if len(parts) >= 2:
            camera_dir = parts[-2]
            filename = parts[-1]
            candidates.append(video_root / camera_dir / filename)
            candidates.append(video_root / "chunk-000" / camera_dir / filename)
        return candidates

    def _video_feature_shape(self, path: Path) -> tuple[int, int, int] | None:
        if not path.exists():
            self._warn_missing_image(path)
            return None
        try:
            import av

            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                return (3, int(stream.height), int(stream.width))
        except Exception as exc:
            print(f"WARNING: could not inspect video {path}: {exc}", file=sys.stderr)
            return None

    @staticmethod
    def _read_video_frame(path: Path, frame_idx: int) -> np.ndarray:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for idx, frame in enumerate(container.decode(stream)):
                if idx == frame_idx:
                    arr = frame.to_ndarray(format="rgb24")
                    return np.ascontiguousarray(arr.transpose(2, 0, 1))
        raise IndexError(f"{path} does not contain frame_idx={frame_idx}")

    def _warn_missing_image(self, path: Path) -> None:
        key = str(path)
        if key in self._warned_missing_images:
            return
        self._warned_missing_images.add(key)
        suffix = ""
        if path.is_symlink():
            try:
                suffix = f" -> {path.readlink()}"
            except OSError:
                suffix = " (broken symlink)"
        print(f"WARNING: skipping missing JSONL video {path}{suffix}", file=sys.stderr)


def _is_jsonl_dataset(root: Path) -> bool:
    return (root / "jsonl").is_dir() and any((root / "jsonl").glob("episode_*.jsonl"))


def _select_dataset_format(args: argparse.Namespace) -> str:
    if args.dataset_format != "auto":
        return args.dataset_format
    if (args.root / "meta" / "info.json").exists():
        return "lerobot"
    if _is_jsonl_dataset(args.root):
        return "jsonl"
    return "lerobot"


def _build_spec(dataset: Any) -> dict[str, dict[str, Any]]:
    spec: dict[str, dict[str, Any]] = {}
    for key, info in dict(dataset.meta.features).items():
        if key in INTERNAL_KEYS:
            continue
        entry = dict(info)
        if "shape" in entry:
            entry["shape"] = tuple(entry["shape"])
        spec[key] = entry
    return spec


def _frame_to_numpy(frame: Any) -> dict[str, Any]:
    return {key: _feature_to_numpy(value) for key, value in dict(frame).items()}


def _feature_to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (str, bytes)):
        return value
    try:
        return np.asarray(value)
    except Exception:
        return value


def _numpy_to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _convert_feature_value(value: Any, feature_info: dict[str, Any] | None) -> Any:
    arr_or_value = _feature_to_numpy(value)
    if not isinstance(arr_or_value, np.ndarray):
        return arr_or_value

    arr = arr_or_value
    if feature_info and feature_info.get("dtype") == "video":
        arr = np.asarray(arr)
        if arr.dtype.kind == "f":
            arr = np.clip(arr, 0.0, 1.0)
            return _reshape_to_feature_shape((arr * 255.0).round().astype(np.uint8), feature_info)
        return _reshape_to_feature_shape(arr.astype(np.uint8, copy=False), feature_info)

    dtype_name = feature_info.get("dtype") if feature_info else None
    if isinstance(dtype_name, str):
        try:
            arr = arr.astype(np.dtype(dtype_name), copy=False)
        except TypeError:
            pass
    return _reshape_to_feature_shape(arr, feature_info)


def _reshape_to_feature_shape(arr: np.ndarray, feature_info: dict[str, Any] | None) -> np.ndarray:
    if not feature_info or "shape" not in feature_info:
        return arr
    shape = tuple(feature_info.get("shape", ()))
    if arr.shape == shape:
        return arr
    if shape and arr.size == int(np.prod(shape)):
        return arr.reshape(shape)
    return arr


def _build_observation(frame: dict[str, Any], spec: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    obs: dict[str, np.ndarray] = {}
    for key, value in frame.items():
        if not key.startswith("observation."):
            continue
        converted = _convert_feature_value(value, spec.get(key))
        if isinstance(converted, np.ndarray):
            obs[key] = converted
        else:
            obs[key] = np.asarray(converted)

    if "teleoperated" in frame:
        converted = _convert_feature_value(frame["teleoperated"], spec.get("teleoperated"))
        obs["teleoperated"] = np.asarray(converted)
    else:
        obs["teleoperated"] = np.array([0], dtype=np.int64)
    return obs


def _task_from_frame(frame: dict[str, Any], override: str | None) -> str:
    if override is not None:
        return override
    value = frame.get("task", "")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.reshape(-1)[0]
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return ""
        value = value.reshape(-1)[0].detach().cpu().item()
    return str(value)


def _expected_action_shapes(spec: dict[str, dict[str, Any]]) -> dict[str, tuple[int, ...]]:
    return {
        key: tuple(info.get("shape", ()))
        for key, info in spec.items()
        if key == "action" or key.startswith("action.")
    }


def _validate_action(action: Any, expected_shapes: dict[str, tuple[int, ...]]) -> tuple[dict[str, np.ndarray], list[str]]:
    errors: list[str] = []
    converted: dict[str, np.ndarray] = {}

    if not isinstance(action, dict):
        return converted, [f"action return is {type(action).__name__}, expected dict"]

    expected_keys = set(expected_shapes)
    actual_keys = set(action)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing:
        errors.append(f"missing action keys: {missing}")
    if extra:
        errors.append(f"unexpected action keys: {extra}")

    for key in sorted(expected_keys & actual_keys):
        try:
            arr = np.asarray(_feature_to_numpy(action[key]))
        except Exception as exc:
            errors.append(f"{key}: could not convert to ndarray: {exc}")
            continue

        expected_shape = expected_shapes[key]
        if arr.shape != expected_shape:
            errors.append(f"{key}: shape {arr.shape}, expected {expected_shape}")
        if arr.dtype.kind not in {"b", "i", "u", "f"}:
            errors.append(f"{key}: dtype {arr.dtype} is not numeric")
        elif not np.isfinite(arr.astype(np.float64, copy=False)).all():
            errors.append(f"{key}: contains non-finite values")
        converted[key] = arr

    return converted, errors


def _shape_summary(values: dict[str, Any]) -> str:
    parts = []
    for key in sorted(values):
        value = values[key]
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is None:
            parts.append(f"{key}=<{type(value).__name__}>")
        else:
            parts.append(f"{key}={tuple(shape)}:{dtype}")
    return ", ".join(parts)


def _format_action(action: dict[str, np.ndarray], print_actions: bool) -> str:
    if print_actions:
        return str({key: value.tolist() for key, value in sorted(action.items())})
    return _shape_summary(action)


def _iter_indices(dataset_len: int, start: int, num_frames: int, stride: int) -> list[int]:
    if start < 0:
        raise ValueError(f"--start must be non-negative, got {start}")
    if num_frames < 0:
        raise ValueError(f"--num-frames must be non-negative, got {num_frames}")
    if stride <= 0:
        raise ValueError(f"--stride must be positive, got {stride}")
    indices = [start + i * stride for i in range(num_frames)]
    return [idx for idx in indices if idx < dataset_len]


def main() -> int:
    args = _parse_args()
    repo_id = args.repo_id or args.root.name
    agent_kwargs = _parse_agent_kwargs(args.agent_kwargs_json)

    dataset_format = _select_dataset_format(args)
    if dataset_format == "jsonl":
        dataset = JsonlDataset(
            args.root,
            episode=args.episode,
            skip_images=args.skip_images,
            video_root=args.video_root,
        )
    else:
        episodes = [args.episode] if args.episode is not None else None
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=args.root,
            episodes=episodes,
            video_backend=args.video_backend,
        )
    spec = _build_spec(dataset)
    buffer = BufferProxy(
        repo_id=repo_id,
        root=args.root,
        features=spec,
        fps=getattr(dataset.meta, "fps", None),
        dataset=dataset,
    )
    expected_shapes = _expected_action_shapes(spec)

    agent_cls = _load_agent_class(args)
    agent = _instantiate_agent(agent_cls, spec, buffer, agent_kwargs)

    indices = _iter_indices(len(dataset), args.start, args.num_frames, args.stride)
    print(
        f"Dataset: format={dataset_format!r} repo_id={repo_id!r} "
        f"root={str(args.root)!r} frames_loaded={len(dataset)}"
    )
    print(f"Agent: {agent_cls.__module__}.{agent_cls.__name__}")
    print(f"Expected actions: {expected_shapes}")
    print(f"Running {len(indices)} frame(s): {indices}")

    total_errors = 0
    try:
        for frame_number, dataset_index in enumerate(indices):
            frame = _frame_to_numpy(dataset[dataset_index])
            obs = _build_observation(frame, spec)
            task = _task_from_frame(frame, args.task)

            start_s = time.perf_counter()
            try:
                raw_action = agent.act(obs, task)
                elapsed_ms = (time.perf_counter() - start_s) * 1000.0
                action, errors = _validate_action(raw_action, expected_shapes)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start_s) * 1000.0
                action = {}
                errors = [f"agent.act raised {type(exc).__name__}: {exc}"]

            status = "FAIL" if errors else "PASS"
            print(
                f"[{status}] frame={frame_number} dataset_index={dataset_index} "
                f"task={task!r} act_ms={elapsed_ms:.1f}"
            )
            print(f"  obs: {_shape_summary(obs)}")
            print(f"  action: {_format_action(action, args.print_actions)}")

            if errors:
                total_errors += len(errors)
                for error in errors:
                    print(f"  error: {error}")
                if args.stop_on_error:
                    break

            if args.sleep_s > 0:
                time.sleep(args.sleep_s)
    finally:
        teardown = getattr(agent, "teardown", None)
        if callable(teardown):
            teardown()

    if total_errors:
        print(f"FAILED: {total_errors} validation error(s)")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
