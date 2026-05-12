"""Offline open-loop evaluation for Dexbotic checkpoints.

This script is intentionally aligned with the current Dexbotic training stack:
- loads a Dexbotic checkpoint with its native `Exp` class
- rebuilds the evaluation dataset through `DataConfig`
- runs `model.inference_action(...)` in batch mode
- reports normalized/raw action errors and saves comparison plots

Typical usage:

  python openloop/eval_openloop.py \
    --checkpoint ./user_checkpoints/dexbotic/custom_dm0/post_data_01-0416/checkpoint-400 \
    --exp-file playground/benchmarks/custom/post_data_01_dm0.py \
    --dataset-name post_data_01_default \
    --chunk_merge mean \
    --save-arrays true

If `--exp-file` is omitted, the script will try to infer a built-in Dexbotic
experiment from the checkpoint `config.json`.

Current scope:
- tested design target: PI0 / PI0.5 / DM0 style action-chunk models
- not intended for old text-to-action inference paths such as OFT/CogACT
"""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import inspect
import json
import logging
import pathlib
import random
from collections import defaultdict
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import dexbotic.data.data_source  # noqa: F401
from dexbotic.exp.base_exp import BaseExp


DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_BATCHES = 50
DEFAULT_NUM_WORKERS = 4
DEFAULT_DIFFUSION_STEPS = 10
DEFAULT_CHUNK_MERGE_EXP_DECAY = 0.7


@dataclass(frozen=True)
class SampleRecord:
    dataset_index: int
    file_index: int
    frame_index: int
    file_path: str
    episode_index: int


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Could not parse boolean value from {value!r}.")


def _parse_csv_ints(value: str | None) -> list[int]:
    if value is None or value == "":
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory.")
    parser.add_argument(
        "--exp-file",
        default=None,
        help="Path to the training experiment python file, e.g. playground/benchmarks/custom/post_data_01_dm0.py",
    )
    parser.add_argument(
        "--exp-module",
        default=None,
        help="Import path for the experiment module, e.g. dexbotic.exp.dm0_exp",
    )
    parser.add_argument(
        "--exp-class",
        default=None,
        help="Experiment class name inside the module/file. Auto-detected when omitted.",
    )
    parser.add_argument("--dataset-name", default=None, help="Override dataset_name for evaluation.")
    parser.add_argument(
        "--norm-stats",
        default=None,
        help="Optional norm_stats.json path. Defaults to <checkpoint>/norm_stats.json when available.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-batches", type=int, default=DEFAULT_NUM_BATCHES)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--diffusion-steps", type=int, default=DEFAULT_DIFFUSION_STEPS)
    parser.add_argument(
        "--single-gpu-id",
        type=int,
        default=None,
        help="Pin inference to a single CUDA device id and avoid model auto-sharding across multiple GPUs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle eval dataloader.")
    parser.add_argument(
        "--episode-index",
        default=None,
        help="Optional comma-separated episode indices (based on sorted jsonl filenames) to evaluate.",
    )
    parser.add_argument(
        "--chunk_merge",
        choices=["chunk", "first", "mean", "exp"],
        default="chunk",
        help="How to merge action chunks into a time series. `chunk` preserves the legacy chunked-timestep view.",
    )
    parser.add_argument(
        "--chunk_merge_exp_decay",
        type=float,
        default=DEFAULT_CHUNK_MERGE_EXP_DECAY,
        help="Decay used by `--chunk_merge exp`. Higher weight is given to newer chunk predictions.",
    )
    parser.add_argument(
        "--plot-path",
        default=None,
        help="Output raw plot path. Defaults to <checkpoint>/openloop_action_plot.png",
    )
    parser.add_argument(
        "--normalized-plot-path",
        default=None,
        help="Output normalized plot path. Defaults to <checkpoint>/openloop_action_plot_normalized.png",
    )
    parser.add_argument(
        "--metrics-path",
        default=None,
        help="Output json path. Defaults to <checkpoint>/openloop_metrics.json",
    )
    parser.add_argument(
        "--array-dir",
        default=None,
        help="Directory for saved npy arrays. Defaults to the checkpoint directory.",
    )
    parser.add_argument(
        "--save-arrays",
        type=_parse_bool,
        default=True,
        help="Whether to save pred/gt arrays to npy files.",
    )
    parser.add_argument(
        "--plot-max-samples",
        type=int,
        default=400,
        help="Only plot the first N evaluated chunks or merged steps to keep figures readable.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=None,
        help="Optional physical action dimension to plot/export. Defaults to inference_config.action_dim when present.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Optional moving-average window for postprocessed visualization in raw action space.",
    )
    parser.add_argument(
        "--binary-dims",
        default="",
        help="Comma-separated binary/switch dimensions for hysteresis postprocessing, e.g. 6,13.",
    )
    parser.add_argument("--binary-threshold-high", type=float, default=0.6)
    parser.add_argument("--binary-threshold-low", type=float, default=0.4)
    parser.add_argument(
        "--plot-pred-source",
        choices=["pred", "post"],
        default="pred",
        help="Whether the raw plot should visualize the original prediction or the postprocessed prediction.",
    )
    return parser.parse_args()


def _load_module_from_file(file_path: str) -> ModuleType:
    path = pathlib.Path(file_path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _infer_builtin_exp_module(checkpoint_dir: pathlib.Path) -> tuple[str, str]:
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find {config_path}. Pass --exp-file/--exp-module explicitly."
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    model_type = config.get("model_type")
    mapping = {
        "dexbotic_pi0": ("dexbotic.exp.pi0_exp", "Pi0Exp"),
        "dexbotic_pi05": ("dexbotic.exp.pi05_exp", "Pi05Exp"),
        "dexbotic_dm0": ("dexbotic.exp.dm0_exp", "DM0Exp"),
        "dexbotic_dm0_prog": ("dexbotic.exp.dm0_exp", "DM0Exp"),
    }
    if model_type not in mapping:
        raise ValueError(
            f"Unsupported or unknown checkpoint model_type={model_type!r}. "
            "This script currently targets PI0 / PI0.5 / DM0-style action-chunk checkpoints. "
            "If you need another model family, extend this script with that model's inference path explicitly."
        )
    return mapping[model_type]


def _find_exp_class(module: ModuleType, class_name: str | None) -> type[BaseExp]:
    if class_name is not None:
        exp_cls = getattr(module, class_name)
        if not issubclass(exp_cls, BaseExp):
            raise TypeError(f"{class_name} is not a BaseExp subclass.")
        return exp_cls

    candidates: list[type[BaseExp]] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseExp) and obj is not BaseExp and obj.__module__ == module.__name__:
            candidates.append(obj)

    if len(candidates) == 1:
        return candidates[0]

    preferred = [cls for cls in candidates if cls.__name__.endswith("Exp")]
    if len(preferred) == 1:
        return preferred[0]

    candidate_names = ", ".join(sorted(cls.__name__ for cls in candidates)) or "<none>"
    raise ValueError(
        f"Could not uniquely determine the experiment class from module {module.__name__}. "
        f"Candidates: {candidate_names}. Pass --exp-class explicitly."
    )


def _load_exp_class(args: argparse.Namespace, checkpoint_dir: pathlib.Path) -> type[BaseExp]:
    if args.exp_file:
        module = _load_module_from_file(args.exp_file)
        return _find_exp_class(module, args.exp_class)

    module_name = args.exp_module
    class_name = args.exp_class
    if module_name is None:
        module_name, inferred_class_name = _infer_builtin_exp_module(checkpoint_dir)
        if class_name is None:
            class_name = inferred_class_name

    module = importlib.import_module(module_name)
    return _find_exp_class(module, class_name)


def _resolve_norm_stats_path(args: argparse.Namespace, checkpoint_dir: pathlib.Path) -> str | None:
    if args.norm_stats is not None:
        return str(pathlib.Path(args.norm_stats).expanduser().resolve())

    default_path = checkpoint_dir / "norm_stats.json"
    if default_path.exists():
        return str(default_path)
    return None


def _disable_eval_augmentations(exp: BaseExp) -> None:
    aug_policy = exp.data_config.aug_policy
    if isinstance(aug_policy, list):
        exp.data_config.aug_policy = ["identity"] * len(aug_policy)
    else:
        exp.data_config.aug_policy = "identity"


def _prepare_exp(args: argparse.Namespace, checkpoint_dir: pathlib.Path) -> BaseExp:
    exp_cls = _load_exp_class(args, checkpoint_dir)
    exp = exp_cls()

    exp.model_config.model_name_or_path = str(checkpoint_dir)
    if hasattr(exp.inference_config, "model_name_or_path"):
        exp.inference_config.model_name_or_path = str(checkpoint_dir)

    if args.dataset_name is not None:
        exp.data_config.dataset_name = args.dataset_name

    if args.single_gpu_id is not None:
        target_device_map = {"": f"cuda:{args.single_gpu_id}"}
        exp.inference_config.cuda_device = args.single_gpu_id
        exp.inference_config.device_map = target_device_map
        _patch_model_from_pretrained_device_map(target_device_map)

    norm_stats_path = _resolve_norm_stats_path(args, checkpoint_dir)
    if norm_stats_path is not None:
        exp.data_config.action_config.statistic_mapping = norm_stats_path
        if hasattr(exp.inference_config, "read_normalization_stats"):
            exp.inference_config.norm_stats = exp.inference_config.read_normalization_stats(norm_stats_path)
        else:
            exp.inference_config.norm_stats = norm_stats_path
    elif getattr(exp.data_config.action_config, "statistic_mapping", None) is None:
        raise FileNotFoundError(
            f"Could not find norm_stats.json under {checkpoint_dir}. "
            "Pass --norm-stats explicitly."
        )

    exp.data_config.auto_norm = False
    _disable_eval_augmentations(exp)
    return exp


def _patch_model_from_pretrained_device_map(target_device_map: dict[str, str]) -> None:
    def _patch_class_from_pretrained(cls: type) -> None:
        original = cls.from_pretrained
        if getattr(original, "_eval_openloop_single_gpu_patched", False):
            return

        def _wrapped(inner_cls, pretrained_model_name_or_path, *args, **kwargs):
            kwargs = dict(kwargs)
            current_device_map = kwargs.get("device_map")
            if current_device_map is None or current_device_map == "auto":
                kwargs["device_map"] = target_device_map
            return original(pretrained_model_name_or_path, *args, **kwargs)

        _wrapped._eval_openloop_single_gpu_patched = True
        cls.from_pretrained = classmethod(_wrapped)

    candidate_specs = [
        ("dexbotic.model.dm0.dm0_arch", "DM0ForCausalLM"),
        ("dexbotic.model.pi0.pi0_arch", "Pi0ForCausalLM"),
        ("dexbotic.model.pi05.pi05_arch", "Pi05ForCausalLM"),
        ("dexbotic.model.pi05.hybrid_pi05_arch", "HybridPi05ForCausalLM"),
    ]
    for module_name, class_name in candidate_specs:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name, None)
            if cls is not None:
                _patch_class_from_pretrained(cls)
        except Exception:
            continue


def _enumerate_episode_paths(dataset: Any) -> list[str]:
    return sorted(dataset.file_name_map.values())


def _build_sample_records(dataset: Any) -> list[SampleRecord]:
    episode_paths = _enumerate_episode_paths(dataset)
    episode_map = {path: idx for idx, path in enumerate(episode_paths)}
    records: list[SampleRecord] = []
    for dataset_index, file_index, frame_index in dataset.global_index:
        file_path = dataset.file_name_map[file_index]
        records.append(
            SampleRecord(
                dataset_index=dataset_index,
                file_index=file_index,
                frame_index=frame_index,
                file_path=file_path,
                episode_index=episode_map[file_path],
            )
        )
    return records


def _select_subset_indices(dataset: Any, episode_indices: list[int]) -> tuple[list[int], list[SampleRecord], list[str]]:
    all_records = _build_sample_records(dataset)
    episode_paths = _enumerate_episode_paths(dataset)
    if not episode_indices:
        indices = list(range(len(all_records)))
        return indices, all_records, episode_paths

    unknown = [idx for idx in episode_indices if idx < 0 or idx >= len(episode_paths)]
    if unknown:
        raise ValueError(
            f"Requested episode indices {unknown} are out of range. "
            f"Available episode indices: 0..{len(episode_paths) - 1}"
        )

    selected = set(episode_indices)
    indices = [idx for idx, record in enumerate(all_records) if record.episode_index in selected]
    records = [all_records[idx] for idx in indices]
    return indices, records, episode_paths


def _build_eval_components(
    exp: BaseExp,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    episode_indices: list[int],
) -> tuple[torch.nn.Module, DataLoader, Any, Any, list[SampleRecord], list[str]]:
    inference_cfg = exp.inference_config
    inference_cfg._initialize_inference()
    model = inference_cfg.model
    model.eval()

    tokenizer = inference_cfg.tokenizer
    image_processor = model.model.mm_vision_module.image_processor
    dataset, collator = exp.data_config.build_data(tokenizer, model.config.chat_template, image_processor)
    subset_indices, sample_records, episode_paths = _select_subset_indices(dataset, episode_indices)
    subset = Subset(dataset, subset_indices)
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )
    return model, dataloader, inference_cfg, dataset, sample_records, episode_paths


def _postprocess_actions(inference_cfg: Any, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    payload = {
        "state": states.copy(),
        "action": actions.copy(),
        "meta_data": {
            "non_delta_mask": np.array(getattr(inference_cfg, "non_delta_mask", [-1])),
        },
    }
    outputs = inference_cfg.output_transform(copy.deepcopy(payload))
    action_dim = getattr(inference_cfg, "action_dim", outputs["action"].shape[-1])
    return np.asarray(outputs["action"][..., :action_dim], dtype=np.float32)


def _mean_over_non_batch_dims(x: np.ndarray) -> np.ndarray:
    if x.ndim <= 1:
        return x
    return np.mean(x, axis=tuple(range(1, x.ndim)))


def _per_horizon_mse(pred_actions: np.ndarray, gt_actions: np.ndarray) -> np.ndarray:
    squared_error = np.square(pred_actions - gt_actions)
    if squared_error.ndim == 2:
        return np.mean(squared_error, axis=0)
    return np.mean(squared_error, axis=(0, 2))


def _chunk_series_with_gaps(actions: np.ndarray) -> np.ndarray:
    if actions.ndim != 3:
        raise ValueError(f"Expected [batch, horizon, action_dim], got {actions.shape}")
    batch_size, horizon, action_dim = actions.shape
    series = np.full((batch_size, horizon + 1, action_dim), np.nan, dtype=np.float32)
    series[:, :horizon, :] = actions.astype(np.float32)
    return series.reshape(-1, action_dim)


def _concat_episode_series(series_list: list[np.ndarray], action_dim: int) -> np.ndarray:
    if not series_list:
        return np.empty((0, action_dim), dtype=np.float32)
    if len(series_list) == 1:
        return series_list[0]
    gap = np.full((1, action_dim), np.nan, dtype=np.float32)
    combined: list[np.ndarray] = []
    for idx, series in enumerate(series_list):
        if idx > 0:
            combined.append(gap.copy())
        combined.append(series)
    return np.concatenate(combined, axis=0)


def _merge_chunks_for_episode(
    chunks: np.ndarray,
    states: np.ndarray,
    frame_indices: np.ndarray,
    mode: str,
    exp_decay: float,
) -> tuple[np.ndarray, np.ndarray]:
    if chunks.ndim != 3:
        raise ValueError(f"Expected chunk array [N, H, D], got {chunks.shape}")
    if len(chunks) == 0:
        return np.empty((0, chunks.shape[-1]), dtype=np.float32), np.empty((0, states.shape[-1]), dtype=np.float32)

    horizon = chunks.shape[1]
    action_dim = chunks.shape[2]
    total_steps = int(np.max(frame_indices)) + 1

    state_series = np.full((total_steps, states.shape[-1]), np.nan, dtype=np.float32)
    for state, frame_index in zip(states, frame_indices, strict=True):
        if 0 <= frame_index < total_steps:
            state_series[frame_index] = np.asarray(state, dtype=np.float32)

    if mode == "first":
        series = np.full((total_steps, action_dim), np.nan, dtype=np.float32)
        for chunk, frame_index in zip(chunks, frame_indices, strict=True):
            if 0 <= frame_index < total_steps:
                series[frame_index] = chunk[0]
        return series, state_series

    values: list[list[np.ndarray]] = [[] for _ in range(total_steps)]
    weights: list[list[float]] = [[] for _ in range(total_steps)]
    for chunk, frame_index in zip(chunks, frame_indices, strict=True):
        for offset in range(horizon):
            timestep = frame_index + offset
            if timestep < 0 or timestep >= total_steps:
                continue
            values[timestep].append(chunk[offset])
            if mode == "exp":
                weights[timestep].append(exp_decay**offset)
            else:
                weights[timestep].append(1.0)

    series = np.full((total_steps, action_dim), np.nan, dtype=np.float32)
    for timestep in range(total_steps):
        if not values[timestep]:
            continue
        value_stack = np.stack(values[timestep], axis=0).astype(np.float32)
        weight_array = np.asarray(weights[timestep], dtype=np.float32)
        weight_array /= np.sum(weight_array)
        series[timestep] = np.sum(value_stack * weight_array[:, None], axis=0)

    return series, state_series


def _merge_or_flatten_chunks(
    chunks: np.ndarray,
    states: np.ndarray,
    sample_records: list[SampleRecord],
    mode: str,
    exp_decay: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(sample_records) != len(chunks):
        raise ValueError("sample_records length must match the number of chunks.")
    if len(chunks) == 0:
        return np.empty((0, chunks.shape[-1]), dtype=np.float32), np.empty((0, states.shape[-1]), dtype=np.float32)

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(sample_records):
        grouped_indices[record.file_path].append(idx)

    merged_series_parts: list[np.ndarray] = []
    merged_state_parts: list[np.ndarray] = []
    for file_path in sorted(grouped_indices):
        indices = grouped_indices[file_path]
        episode_chunks = chunks[indices]
        episode_states = states[indices]
        frame_indices = np.asarray([sample_records[idx].frame_index for idx in indices], dtype=np.int64)

        if mode == "chunk":
            merged_series_parts.append(_chunk_series_with_gaps(episode_chunks))
            state_repeated = np.repeat(episode_states[:, None, :], episode_chunks.shape[1] + 1, axis=1)
            state_repeated[:, -1, :] = np.nan
            merged_state_parts.append(state_repeated.reshape(-1, episode_states.shape[-1]).astype(np.float32))
        else:
            episode_series, episode_state_series = _merge_chunks_for_episode(
                episode_chunks,
                episode_states,
                frame_indices,
                mode,
                exp_decay,
            )
            merged_series_parts.append(episode_series.astype(np.float32))
            merged_state_parts.append(episode_state_series.astype(np.float32))

    action_dim = chunks.shape[-1]
    state_dim = states.shape[-1]
    merged_series = _concat_episode_series(merged_series_parts, action_dim)
    merged_state_series = _concat_episode_series(merged_state_parts, state_dim)
    return merged_series, merged_state_series


def _expand_action_labels(action_dim: int) -> list[str]:
    return [f"action_{idx}" for idx in range(action_dim)]


def _plot_action_sequence_comparison(
    *,
    output_path: pathlib.Path,
    gt_series: np.ndarray,
    pred_series: np.ndarray,
    action_labels: list[str],
    checkpoint_dir: pathlib.Path,
    plot_space: str,
) -> None:
    if gt_series.shape != pred_series.shape:
        raise ValueError(f"gt_series shape {gt_series.shape} must match pred_series shape {pred_series.shape}")

    num_steps, action_dim = gt_series.shape
    x = np.arange(num_steps)
    fig_height = max(4.0, 3.2 * action_dim)
    fig, axes = plt.subplots(action_dim, 1, figsize=(10, fig_height), sharex=True)
    if action_dim == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        ax.plot(x, gt_series[:, idx], label="gt", linewidth=1.0, alpha=0.9)
        ax.plot(x, pred_series[:, idx], label="pred", linewidth=1.0, alpha=0.9)
        ax.set_title(action_labels[idx], fontsize=10)
        ax.set_ylabel(plot_space)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"Dexbotic open-loop comparison\n{checkpoint_dir.name}", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _slice_for_plot(series: np.ndarray, max_samples: int | None) -> np.ndarray:
    if max_samples is None or max_samples <= 0 or len(series) <= max_samples:
        return series
    return series[:max_samples]


def _moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values.astype(np.float32)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _apply_moving_average(series: np.ndarray, window: int, binary_dims: set[int]) -> np.ndarray:
    if window <= 1:
        return series.astype(np.float32)
    result = np.array(series, copy=True, dtype=np.float32)
    valid_rows = ~np.isnan(series).all(axis=1)
    start = None
    continuous_dims = [dim for dim in range(series.shape[1]) if dim not in binary_dims]
    for idx, is_valid in enumerate(valid_rows):
        if is_valid and start is None:
            start = idx
        if (not is_valid or idx == len(valid_rows) - 1) and start is not None:
            end = idx if not is_valid else idx + 1
            segment = result[start:end]
            for dim in continuous_dims:
                segment[:, dim] = _moving_average_1d(segment[:, dim], window)
            result[start:end] = segment
            start = None
    return result


def _apply_binary_hysteresis(
    series: np.ndarray,
    binary_dims: list[int],
    low: float,
    high: float,
) -> np.ndarray:
    if not binary_dims:
        return series.astype(np.float32)
    result = np.array(series, copy=True, dtype=np.float32)
    valid_rows = ~np.isnan(series).all(axis=1)
    for dim in binary_dims:
        start = None
        for idx, is_valid in enumerate(valid_rows):
            if is_valid and start is None:
                start = idx
            if (not is_valid or idx == len(valid_rows) - 1) and start is not None:
                end = idx if not is_valid else idx + 1
                segment = result[start:end, dim]
                if len(segment) == 0:
                    start = None
                    continue
                current = 1.0 if segment[0] > high else 0.0 if segment[0] < low else float(segment[0] >= 0.5)
                for seg_idx, value in enumerate(segment):
                    if value > high:
                        current = 1.0
                    elif value < low:
                        current = 0.0
                    segment[seg_idx] = current
                result[start:end, dim] = segment
                start = None
    return result


def _build_postprocessed_prediction(
    pred_raw_series: np.ndarray,
    smooth_window: int,
    binary_dims: list[int],
    low: float,
    high: float,
) -> np.ndarray:
    result = np.array(pred_raw_series, copy=True, dtype=np.float32)
    binary_dim_set = set(binary_dims)
    result = _apply_moving_average(result, smooth_window, binary_dim_set)
    result = _apply_binary_hysteresis(result, binary_dims, low, high)
    return result


def _save_arrays(
    array_dir: pathlib.Path,
    *,
    pred_raw: np.ndarray,
    gt_raw: np.ndarray,
    pred_norm: np.ndarray,
    gt_norm: np.ndarray,
    pred_post: np.ndarray | None,
    state_raw: np.ndarray,
    pred_raw_chunks: np.ndarray,
    gt_raw_chunks: np.ndarray,
    pred_norm_chunks: np.ndarray,
    gt_norm_chunks: np.ndarray,
) -> dict[str, str]:
    array_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pred_raw": str(array_dir / "pred_raw.npy"),
        "gt_raw": str(array_dir / "gt_raw.npy"),
        "pred_norm": str(array_dir / "pred_norm.npy"),
        "gt_norm": str(array_dir / "gt_norm.npy"),
        "state_raw": str(array_dir / "state_raw.npy"),
        "pred_raw_chunks": str(array_dir / "pred_raw_chunks.npy"),
        "gt_raw_chunks": str(array_dir / "gt_raw_chunks.npy"),
        "pred_norm_chunks": str(array_dir / "pred_norm_chunks.npy"),
        "gt_norm_chunks": str(array_dir / "gt_norm_chunks.npy"),
    }
    np.save(paths["pred_raw"], pred_raw)
    np.save(paths["gt_raw"], gt_raw)
    np.save(paths["pred_norm"], pred_norm)
    np.save(paths["gt_norm"], gt_norm)
    np.save(paths["state_raw"], state_raw)
    np.save(paths["pred_raw_chunks"], pred_raw_chunks)
    np.save(paths["gt_raw_chunks"], gt_raw_chunks)
    np.save(paths["pred_norm_chunks"], pred_norm_chunks)
    np.save(paths["gt_norm_chunks"], gt_norm_chunks)
    if pred_post is not None:
        paths["pred_post"] = str(array_dir / "pred_post.npy")
        np.save(paths["pred_post"], pred_post)
    return paths


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_dir = pathlib.Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_batches <= 0:
        raise ValueError("--num-batches must be positive.")
    if args.diffusion_steps <= 0:
        raise ValueError("--diffusion-steps must be positive.")
    if args.smooth_window <= 0:
        raise ValueError("--smooth-window must be positive.")
    if not (0.0 <= args.chunk_merge_exp_decay <= 1.0):
        raise ValueError("--chunk_merge_exp_decay must be in [0, 1].")
    if args.binary_threshold_low > args.binary_threshold_high:
        raise ValueError("--binary-threshold-low must be <= --binary-threshold-high.")

    selected_episode_indices = _parse_csv_ints(args.episode_index)
    binary_dims = _parse_csv_ints(args.binary_dims)

    exp = _prepare_exp(args, checkpoint_dir)
    logging.info("Using experiment class: %s", type(exp).__name__)
    logging.info("Evaluation dataset: %s", exp.data_config.dataset_name)

    model, dataloader, inference_cfg, dataset, sample_records, episode_paths = _build_eval_components(
        exp,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        episode_indices=selected_episode_indices,
    )

    if selected_episode_indices:
        selected_paths = [episode_paths[idx] for idx in selected_episode_indices]
        logging.info("Selected episodes: %s", selected_paths)

    total_examples = 0
    norm_mse_sum = 0.0
    norm_mae_sum = 0.0
    raw_mse_sum = 0.0
    raw_mae_sum = 0.0
    norm_per_horizon_sum = None
    raw_per_horizon_sum = None

    norm_gt_chunks_all: list[np.ndarray] = []
    norm_pred_chunks_all: list[np.ndarray] = []
    raw_gt_chunks_all: list[np.ndarray] = []
    raw_pred_chunks_all: list[np.ndarray] = []
    state_all: list[np.ndarray] = []
    used_sample_records: list[SampleRecord] = []

    device = inference_cfg.device
    sample_cursor = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        if batch_idx > args.num_batches:
            break

        inputs = dict(batch)
        gt_actions = inputs.pop("actions")
        batch_size = gt_actions.shape[0]
        batch_records = sample_records[sample_cursor : sample_cursor + batch_size]
        sample_cursor += batch_size
        used_sample_records.extend(batch_records)

        tensor_inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            pred_actions = model.inference_action(
                **tensor_inputs,
                diffusion_steps=args.diffusion_steps,
            )

        gt_actions_np = gt_actions.detach().cpu().numpy().astype(np.float32)
        pred_actions_np = pred_actions.detach().cpu().numpy().astype(np.float32)
        states_np = batch["states"].detach().cpu().numpy().astype(np.float32)
        states_for_export_np = states_np.copy()

        pred_raw_np = _postprocess_actions(inference_cfg, states_np, pred_actions_np)
        gt_raw_np = _postprocess_actions(inference_cfg, states_np, gt_actions_np)

        plot_action_dim = args.action_dim
        if plot_action_dim is None:
            plot_action_dim = int(getattr(inference_cfg, "action_dim", gt_raw_np.shape[-1]))
        plot_action_dim = min(
            plot_action_dim,
            gt_raw_np.shape[-1],
            pred_raw_np.shape[-1],
            gt_actions_np.shape[-1],
            pred_actions_np.shape[-1],
        )

        pred_actions_np = pred_actions_np[..., :plot_action_dim]
        gt_actions_np = gt_actions_np[..., :plot_action_dim]
        pred_raw_np = pred_raw_np[..., :plot_action_dim]
        gt_raw_np = gt_raw_np[..., :plot_action_dim]
        norm_error = pred_actions_np - gt_actions_np
        raw_error = pred_raw_np - gt_raw_np
        norm_mse_per_example = _mean_over_non_batch_dims(np.square(norm_error))
        norm_mae_per_example = _mean_over_non_batch_dims(np.abs(norm_error))
        raw_mse_per_example = _mean_over_non_batch_dims(np.square(raw_error))
        raw_mae_per_example = _mean_over_non_batch_dims(np.abs(raw_error))

        total_examples += batch_size
        norm_mse_sum += float(np.sum(norm_mse_per_example))
        norm_mae_sum += float(np.sum(norm_mae_per_example))
        raw_mse_sum += float(np.sum(raw_mse_per_example))
        raw_mae_sum += float(np.sum(raw_mae_per_example))

        batch_norm_per_horizon = _per_horizon_mse(pred_actions_np, gt_actions_np) * batch_size
        batch_raw_per_horizon = _per_horizon_mse(pred_raw_np, gt_raw_np) * batch_size
        if norm_per_horizon_sum is None:
            norm_per_horizon_sum = np.zeros_like(batch_norm_per_horizon, dtype=np.float64)
            raw_per_horizon_sum = np.zeros_like(batch_raw_per_horizon, dtype=np.float64)
        norm_per_horizon_sum += batch_norm_per_horizon
        raw_per_horizon_sum += batch_raw_per_horizon

        norm_gt_chunks_all.append(gt_actions_np)
        norm_pred_chunks_all.append(pred_actions_np)
        raw_gt_chunks_all.append(gt_raw_np)
        raw_pred_chunks_all.append(pred_raw_np)
        state_all.append(states_for_export_np)

        logging.info("Processed batch %d/%d", batch_idx, args.num_batches)

    if total_examples == 0:
        raise RuntimeError("No evaluation samples were processed.")

    norm_gt_chunks = np.concatenate(norm_gt_chunks_all, axis=0)
    norm_pred_chunks = np.concatenate(norm_pred_chunks_all, axis=0)
    raw_gt_chunks = np.concatenate(raw_gt_chunks_all, axis=0)
    raw_pred_chunks = np.concatenate(raw_pred_chunks_all, axis=0)
    state_chunks = np.concatenate(state_all, axis=0)

    raw_gt_series, raw_state_series = _merge_or_flatten_chunks(
        raw_gt_chunks,
        state_chunks,
        used_sample_records,
        args.chunk_merge,
        args.chunk_merge_exp_decay,
    )
    raw_pred_series, _ = _merge_or_flatten_chunks(
        raw_pred_chunks,
        state_chunks,
        used_sample_records,
        args.chunk_merge,
        args.chunk_merge_exp_decay,
    )
    norm_gt_series, _ = _merge_or_flatten_chunks(
        norm_gt_chunks,
        state_chunks,
        used_sample_records,
        args.chunk_merge,
        args.chunk_merge_exp_decay,
    )
    norm_pred_series, _ = _merge_or_flatten_chunks(
        norm_pred_chunks,
        state_chunks,
        used_sample_records,
        args.chunk_merge,
        args.chunk_merge_exp_decay,
    )

    pred_post_series = None
    if args.smooth_window > 1 or binary_dims:
        pred_post_series = _build_postprocessed_prediction(
            raw_pred_series,
            smooth_window=args.smooth_window,
            binary_dims=binary_dims,
            low=args.binary_threshold_low,
            high=args.binary_threshold_high,
        )

    raw_plot_series = raw_pred_series
    if args.plot_pred_source == "post" and pred_post_series is not None:
        raw_plot_series = pred_post_series

    raw_gt_plot_series = _slice_for_plot(raw_gt_series, args.plot_max_samples)
    raw_pred_plot_series = _slice_for_plot(raw_plot_series, args.plot_max_samples)
    norm_gt_plot_series = _slice_for_plot(norm_gt_series, args.plot_max_samples)
    norm_pred_plot_series = _slice_for_plot(norm_pred_series, args.plot_max_samples)

    plot_path = pathlib.Path(args.plot_path) if args.plot_path else checkpoint_dir / "openloop_action_plot.png"
    normalized_plot_path = (
        pathlib.Path(args.normalized_plot_path)
        if args.normalized_plot_path
        else checkpoint_dir / "openloop_action_plot_normalized.png"
    )
    metrics_path = pathlib.Path(args.metrics_path) if args.metrics_path else checkpoint_dir / "openloop_metrics.json"
    array_dir = pathlib.Path(args.array_dir) if args.array_dir else checkpoint_dir

    action_labels = _expand_action_labels(raw_gt_chunks.shape[-1])
    _plot_action_sequence_comparison(
        output_path=plot_path,
        gt_series=raw_gt_plot_series,
        pred_series=raw_pred_plot_series,
        action_labels=action_labels,
        checkpoint_dir=checkpoint_dir,
        plot_space="raw",
    )
    _plot_action_sequence_comparison(
        output_path=normalized_plot_path,
        gt_series=norm_gt_plot_series,
        pred_series=norm_pred_plot_series,
        action_labels=action_labels,
        checkpoint_dir=checkpoint_dir,
        plot_space="normalized",
    )

    array_paths: dict[str, str] = {}
    if args.save_arrays:
        array_paths = _save_arrays(
            array_dir,
            pred_raw=raw_pred_series,
            gt_raw=raw_gt_series,
            pred_norm=norm_pred_series,
            gt_norm=norm_gt_series,
            pred_post=pred_post_series,
            state_raw=raw_state_series,
            pred_raw_chunks=raw_pred_chunks,
            gt_raw_chunks=raw_gt_chunks,
            pred_norm_chunks=norm_pred_chunks,
            gt_norm_chunks=norm_gt_chunks,
        )

    results = {
        "checkpoint_dir": str(checkpoint_dir),
        "exp_class": type(exp).__name__,
        "dataset_name": exp.data_config.dataset_name,
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "num_examples": total_examples,
        "diffusion_steps": args.diffusion_steps,
        "single_gpu_id": args.single_gpu_id,
        "chunk_merge": args.chunk_merge,
        "chunk_merge_exp_decay": args.chunk_merge_exp_decay,
        "smooth_window": args.smooth_window,
        "binary_dims": binary_dims,
        "binary_threshold_high": args.binary_threshold_high,
        "binary_threshold_low": args.binary_threshold_low,
        "plot_pred_source": args.plot_pred_source,
        "plot_action_dim": plot_action_dim,
        "model_chunk_size": int(getattr(model.model.config, "chunk_size", norm_gt_chunks.shape[1])),
        "model_action_dim": int(getattr(model.config, "action_dim", norm_gt_chunks.shape[-1])),
        "selected_episode_indices": selected_episode_indices,
        "selected_episode_paths": [episode_paths[idx] for idx in selected_episode_indices] if selected_episode_indices else [],
        "normalized_action_mse": norm_mse_sum / total_examples,
        "normalized_action_mae": norm_mae_sum / total_examples,
        "raw_action_mse": raw_mse_sum / total_examples,
        "raw_action_mae": raw_mae_sum / total_examples,
        "normalized_per_horizon_mse": (norm_per_horizon_sum / total_examples).tolist(),
        "raw_per_horizon_mse": (raw_per_horizon_sum / total_examples).tolist(),
        "plot_path": str(plot_path),
        "normalized_plot_path": str(normalized_plot_path),
        "array_paths": array_paths,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    logging.info("Saved raw plot to %s", plot_path)
    logging.info("Saved normalized plot to %s", normalized_plot_path)
    logging.info("Saved metrics to %s", metrics_path)
    logging.info("raw_action_mse=%.6f raw_action_mae=%.6f", results["raw_action_mse"], results["raw_action_mae"])
    logging.info(
        "normalized_action_mse=%.6f normalized_action_mae=%.6f",
        results["normalized_action_mse"],
        results["normalized_action_mae"],
    )


if __name__ == "__main__":
    main()
