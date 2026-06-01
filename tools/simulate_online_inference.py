#!/usr/bin/env python3
"""Replay a Dexdata dataset as simulated online inference.

The script mimics the real robot client loop:
- read one observation frame from a Dexdata jsonl episode
- when the local action queue is empty, call the HTTP inference service
- consume one action per simulated control step
- optionally compare consumed actions with dataset actions

Example:
  python tools/simulate_online_inference.py \
    --jsonl-dir data/post_data_merged_dm0_dexdata_nobframes_test/jsonl \
    --video-root data/post_data_merged_dm0_dexdata_nobframes_test/video \
    --base-url http://localhost:7891 \
    --max-episodes 1 --max-steps 200 --hz 10
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import statistics
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
from requests import HTTPError, RequestException
from decord import VideoReader
from PIL import Image

try:
    import megfile
except ImportError:  # Local-only fallback for lightweight runtime environments.
    megfile = None


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_IMAGE_KEYS = ("images_1", "images_2", "images_3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay Dexdata frames through a /process_frame service."
    )
    dataset = parser.add_argument_group("dataset")
    dataset.add_argument(
        "--dataset-name",
        default=None,
        help="Registered Dexbotic dataset name. Overrides --jsonl-dir/--video-root.",
    )
    dataset.add_argument(
        "--jsonl-dir",
        default="data/post_data_merged_dm0_dexdata_nobframes_test/jsonl",
        help="Directory containing episode *.jsonl files.",
    )
    dataset.add_argument(
        "--video-root",
        default="data/post_data_merged_dm0_dexdata_nobframes_test/video",
        help="Root used to resolve relative image/video URLs.",
    )
    dataset.add_argument(
        "--episode-glob",
        default="*.jsonl",
        help="Glob under --jsonl-dir used to select episodes.",
    )
    dataset.add_argument("--max-episodes", type=int, default=1)
    dataset.add_argument("--max-steps", type=int, default=0, help="0 means no limit.")
    dataset.add_argument("--start-frame", type=int, default=0)
    dataset.add_argument("--stride", type=int, default=1)
    dataset.add_argument(
        "--image-keys",
        default=",".join(DEFAULT_IMAGE_KEYS),
        help="Comma-separated image keys, in upload order.",
    )

    service = parser.add_argument_group("service")
    service.add_argument("--base-url", default="http://localhost:7891")
    service.add_argument("--timeout", type=float, default=120.0)
    service.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and simulate timing without calling the inference service.",
    )
    service.add_argument(
        "--send-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send frame['state'] as the states form field.",
    )
    service.add_argument(
        "--dry-run-action-dim",
        type=int,
        default=14,
        help="Zero-action dimension used by --dry-run.",
    )

    loop = parser.add_argument_group("loop")
    loop.add_argument("--hz", type=float, default=0.0, help="0 disables sleeping.")
    loop.add_argument(
        "--async-prefetch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable background inference prefetch. Default keeps synchronous baseline.",
    )
    loop.add_argument(
        "--queue-mode",
        choices=["chunk", "latest"],
        default="chunk",
        help="chunk consumes the returned action queue; latest keeps only response[0].",
    )
    loop.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="Call the service this many times before metrics are recorded.",
    )
    loop.add_argument(
        "--replan-every-steps",
        type=int,
        default=0,
        help=(
            "Force a fresh inference request every N control steps and discard any "
            "remaining queued actions in synchronous mode. In async mode, trigger a "
            "background request every N steps. 0 disables it in sync mode and uses "
            "20 in async mode."
        ),
    )
    loop.add_argument(
        "--empty-queue-policy",
        choices=["request", "repeat-last", "hold", "zero", "error"],
        default="request",
        help=(
            "Behavior when the action queue runs out before the next forced replan. "
            "Async mode maps request/error to repeat-last for real-time safety."
        ),
    )
    loop.add_argument(
        "--prefetch-threshold",
        type=int,
        default=20,
        help="Async mode submits a background request when queue length is <= this value.",
    )
    loop.add_argument(
        "--queue-update-policy",
        choices=["append", "replace", "soft-replace"],
        default="soft-replace",
        help="How async responses update the action queue.",
    )
    loop.add_argument(
        "--keep-old-actions",
        type=int,
        default=3,
        help="soft-replace keeps this many queued old actions before appending the new chunk.",
    )
    loop.add_argument(
        "--blend-steps",
        type=int,
        default=5,
        help="soft-replace linearly blends the first N new actions from last_action. Use 0 to disable.",
    )
    loop.add_argument(
        "--max-inflight-requests",
        type=int,
        default=1,
        help="Maximum concurrent async inference requests. 1 is recommended.",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--log-jsonl",
        default=None,
        help="Optional path to save per-step replay records.",
    )
    output.add_argument(
        "--compare-action",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compare predictions against frame['action'] when available. Enable only "
            "when the service output and dataset action use the same action semantics."
        ),
    )
    output.add_argument("--print-every", type=int, default=20)
    return parser.parse_args()


def resolve_dataset(args: argparse.Namespace) -> tuple[pathlib.Path, pathlib.Path]:
    if args.dataset_name is None:
        return pathlib.Path(args.jsonl_dir), pathlib.Path(args.video_root)

    import dexbotic.data.data_source  # noqa: F401
    from dexbotic.data.data_source.register import CONVERSATION_DATA

    if args.dataset_name not in CONVERSATION_DATA:
        choices = ", ".join(sorted(CONVERSATION_DATA))
        raise KeyError(f"Unknown dataset {args.dataset_name!r}. Available: {choices}")
    info = CONVERSATION_DATA[args.dataset_name]
    return pathlib.Path(info["annotations"]), pathlib.Path(info.get("data_path_prefix", ""))


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    opener = megfile.smart_open if megfile is not None else open
    with opener(str(path), "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def list_episode_files(
    jsonl_dir: pathlib.Path, pattern: str, max_episodes: int
) -> list[pathlib.Path]:
    files = sorted(jsonl_dir.glob(pattern))
    if max_episodes > 0:
        files = files[:max_episodes]
    if not files:
        raise FileNotFoundError(f"No jsonl files matched {jsonl_dir / pattern}")
    return files


class FrameLoader:
    def __init__(self, video_root: pathlib.Path):
        self.video_root = video_root
        self._video_cache: dict[str, VideoReader] = {}

    def load_images(
        self, frame: dict[str, Any], image_keys: list[str]
    ) -> list[Image.Image]:
        images = []
        for key in image_keys:
            if key not in frame:
                continue
            images.append(self._load_one(frame[key]))
        return images

    def _resolve(self, url: str) -> str:
        path = pathlib.Path(url)
        if path.is_absolute():
            return str(path)
        return str(self.video_root / path)

    def _load_one(self, item: dict[str, Any]) -> Image.Image:
        kind = item.get("type")
        url = self._resolve(item["url"])
        if kind == "image":
            opener = megfile.smart_open if megfile is not None else open
            with opener(url, "rb") as f:
                return Image.open(io.BytesIO(f.read())).convert("RGB")
        if kind == "video":
            frame_idx = int(item["frame_idx"])
            if url not in self._video_cache:
                self._video_cache[url] = VideoReader(url, num_threads=1)
            arr = self._video_cache[url][frame_idx].asnumpy()
            return Image.fromarray(arr).convert("RGB")
        raise ValueError(f"Unsupported image item type {kind!r}: {item}")


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass
class RequestStats:
    count: int = 0
    latencies: list[float] | None = None
    response_lengths: list[int] | None = None
    queue_empty_count: int = 0
    repeat_last_count: int = 0
    hold_count: int = 0
    zero_count: int = 0
    async_requests_submitted: int = 0
    async_requests_applied: int = 0
    async_requests_failed: int = 0
    stale_responses_discarded: int = 0
    last_latency: float | None = None
    last_update_policy: str = "-"

    def __post_init__(self) -> None:
        if self.latencies is None:
            self.latencies = []
        if self.response_lengths is None:
            self.response_lengths = []

    def add(self, latency: float, response_length: int) -> None:
        self.count += 1
        self.latencies.append(latency)
        self.response_lengths.append(response_length)
        self.last_latency = latency

    def snapshot(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "latencies": list(self.latencies or []),
            "response_lengths": list(self.response_lengths or []),
            "queue_empty_count": self.queue_empty_count,
            "repeat_last_count": self.repeat_last_count,
            "hold_count": self.hold_count,
            "zero_count": self.zero_count,
            "async_requests_submitted": self.async_requests_submitted,
            "async_requests_applied": self.async_requests_applied,
            "async_requests_failed": self.async_requests_failed,
            "stale_responses_discarded": self.stale_responses_discarded,
            "last_latency": self.last_latency,
            "last_update_policy": self.last_update_policy,
        }


def normalize_action_response(actions: Any) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.size == 0:
        return actions
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim == 1:
        actions = actions[None, :]
    return actions


class OnlineActionClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        dry_run: bool,
        queue_mode: str,
        dry_run_action_dim: int,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.dry_run = dry_run
        self.queue_mode = queue_mode
        self.dry_run_action_dim = dry_run_action_dim
        self.queue: deque[np.ndarray] = deque()
        self.stats = RequestStats()
        self.last_action: np.ndarray | None = None

    def act(
        self,
        prompt: str,
        images: list[Image.Image],
        state: Any | None,
        force_replan: bool = False,
        empty_queue_policy: str = "request",
    ) -> np.ndarray:
        if force_replan:
            self.queue.clear()
            self._request_actions(prompt, images, state)
        elif not self.queue and empty_queue_policy == "request":
            self._request_actions(prompt, images, state)
        elif not self.queue and empty_queue_policy == "repeat-last":
            if self.last_action is None:
                self._request_actions(prompt, images, state)
            else:
                return np.copy(self.last_action)
        elif not self.queue and empty_queue_policy == "zero":
            action_dim = (
                len(self.last_action)
                if self.last_action is not None
                else self.dry_run_action_dim
            )
            self.last_action = np.zeros(action_dim, dtype=np.float32)
            return np.copy(self.last_action)
        elif not self.queue and empty_queue_policy == "error":
            raise RuntimeError("Action queue is empty before the next replan step.")

        action = self.queue.popleft()
        self.last_action = np.copy(action)
        return action

    def _request_actions(
        self, prompt: str, images: list[Image.Image], state: Any | None
    ) -> None:
        if self.dry_run:
            self.queue.append(np.zeros(self.dry_run_action_dim, dtype=np.float32))
            return

        files = [
            ("image", ("image.png", image_to_png_bytes(image), "image/png"))
            for image in images
        ]
        data: dict[str, str] = {"text": prompt}
        if state is not None:
            data["states"] = json.dumps(state)

        t0 = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/process_frame",
            data=data,
            files=files,
            timeout=self.timeout,
        )
        latency = time.perf_counter() - t0
        try:
            response.raise_for_status()
        except HTTPError as exc:
            body = response.text[:2000].replace("\n", "\\n")
            raise RuntimeError(
                f"Inference service returned HTTP {response.status_code}: {body}"
            ) from exc
        payload = response.json()
        actions = payload.get("response")
        if not actions:
            raise RuntimeError(f"Service returned no actions: {payload}")

        actions = normalize_action_response(actions)
        self.stats.add(latency, len(actions))
        if self.queue_mode == "latest":
            actions = actions[:1]
        for action in actions:
            self.queue.append(np.asarray(action, dtype=np.float32))


class AsyncOnlineActionClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        dry_run: bool,
        queue_mode: str,
        dry_run_action_dim: int,
        prefetch_threshold: int,
        queue_update_policy: str,
        keep_old_actions: int,
        blend_steps: int,
        max_inflight_requests: int,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.dry_run = dry_run
        self.queue_mode = queue_mode
        self.dry_run_action_dim = dry_run_action_dim
        self.prefetch_threshold = prefetch_threshold
        self.queue_update_policy = queue_update_policy
        self.keep_old_actions = max(0, keep_old_actions)
        self.blend_steps = max(0, blend_steps)
        self.max_inflight_requests = max(1, max_inflight_requests)

        self.lock = threading.Lock()
        self.action_queue: deque[np.ndarray] = deque()
        self.last_action: np.ndarray | None = None
        self.inflight_count = 0
        self.next_request_id = 0
        self.latest_applied_request_id = 0
        self.stats = RequestStats()

    def prime(self, prompt: str, images: list[Image.Image], state: Any | None) -> bool:
        try:
            actions, latency = self._request_actions(prompt, images, state)
        except Exception as exc:
            with self.lock:
                self.stats.async_requests_failed += 1
            print(f"[async-prime-failed] {exc}", file=sys.stderr, flush=True)
            return False
        with self.lock:
            self.stats.add(latency, len(actions))
            self._apply_actions_locked(actions, request_id=0)
        return True

    def maybe_submit(
        self,
        step_idx: int,
        prompt: str,
        images: list[Image.Image],
        state: Any | None,
        force_replan: bool,
    ) -> bool:
        with self.lock:
            queue_len = len(self.action_queue)
            if not force_replan and queue_len > self.prefetch_threshold:
                return False
            if self.inflight_count >= self.max_inflight_requests:
                return False
            self.next_request_id += 1
            request_id = self.next_request_id
            self.inflight_count += 1
            self.stats.async_requests_submitted += 1

        image_snapshot = [image.copy() for image in images]
        state_snapshot = json.loads(json.dumps(state)) if state is not None else None
        thread = threading.Thread(
            target=self._request_worker,
            args=(request_id, step_idx, prompt, image_snapshot, state_snapshot),
            daemon=True,
        )
        thread.start()
        print(
            f"[async-submit] request_id={request_id} step={step_idx} "
            f"queue={queue_len} force={force_replan}",
            flush=True,
        )
        return True

    def act(self, empty_queue_policy: str) -> np.ndarray:
        with self.lock:
            if self.action_queue:
                action = self.action_queue.popleft()
                self.last_action = np.copy(action)
                return np.copy(action)

            self.stats.queue_empty_count += 1
            policy = empty_queue_policy
            if policy in {"request", "error"}:
                policy = "repeat-last"

            if policy == "repeat-last" and self.last_action is not None:
                self.stats.repeat_last_count += 1
                return np.copy(self.last_action)
            if policy == "hold":
                self.stats.hold_count += 1
                action_dim = len(self.last_action) if self.last_action is not None else self.dry_run_action_dim
                return np.zeros(action_dim, dtype=np.float32)

            self.stats.zero_count += 1
            action_dim = len(self.last_action) if self.last_action is not None else self.dry_run_action_dim
            return np.zeros(action_dim, dtype=np.float32)

    def queue_len(self) -> int:
        with self.lock:
            return len(self.action_queue)

    def inflight(self) -> int:
        with self.lock:
            return self.inflight_count

    def stats_snapshot(self) -> dict[str, Any]:
        with self.lock:
            snapshot = self.stats.snapshot()
            snapshot["inflight_count"] = self.inflight_count
            snapshot["queue_len"] = len(self.action_queue)
            return snapshot

    def needs_prime(self) -> bool:
        with self.lock:
            return not self.action_queue and self.last_action is None

    def _request_worker(
        self,
        request_id: int,
        step_idx: int,
        prompt: str,
        images: list[Image.Image],
        state: Any | None,
    ) -> None:
        try:
            actions, latency = self._request_actions(prompt, images, state)
        except Exception as exc:
            with self.lock:
                self.inflight_count -= 1
                self.stats.async_requests_failed += 1
            print(
                f"[async-failed] request_id={request_id} step={step_idx}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return

        with self.lock:
            self.inflight_count -= 1
            self.stats.add(latency, len(actions))
            if request_id < self.latest_applied_request_id:
                self.stats.stale_responses_discarded += 1
                print(
                    f"[async-stale] request_id={request_id} "
                    f"latest_applied={self.latest_applied_request_id}",
                    flush=True,
                )
                return
            self._apply_actions_locked(actions, request_id=request_id)
            self.stats.async_requests_applied += 1

        print(
            f"[async-applied] request_id={request_id} step={step_idx} "
            f"latency={latency:.3f}s actions={len(actions)}",
            flush=True,
        )

    def _request_actions(
        self, prompt: str, images: list[Image.Image], state: Any | None
    ) -> tuple[np.ndarray, float]:
        if self.dry_run:
            return np.zeros((50, self.dry_run_action_dim), dtype=np.float32), 0.0

        files = [
            ("image", ("image.png", image_to_png_bytes(image), "image/png"))
            for image in images
        ]
        data: dict[str, str] = {"text": prompt}
        if state is not None:
            data["states"] = json.dumps(state)

        t0 = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/process_frame",
            data=data,
            files=files,
            timeout=self.timeout,
        )
        latency = time.perf_counter() - t0
        try:
            response.raise_for_status()
        except HTTPError as exc:
            body = response.text[:2000].replace("\n", "\\n")
            raise RuntimeError(
                f"Inference service returned HTTP {response.status_code}: {body}"
            ) from exc
        actions = normalize_action_response(response.json().get("response", []))
        if actions.size == 0:
            raise RuntimeError("Service returned no actions.")
        if self.queue_mode == "latest":
            actions = actions[:1]
        return actions, latency

    def _apply_actions_locked(self, actions: np.ndarray, request_id: int) -> None:
        actions = np.asarray(actions, dtype=np.float32)
        if self.blend_steps > 0 and self.last_action is not None and len(actions) > 0:
            blend_count = min(self.blend_steps, len(actions))
            blended = np.copy(actions)
            last = np.asarray(self.last_action, dtype=np.float32)
            for idx in range(blend_count):
                alpha = float(idx + 1) / float(blend_count + 1)
                blended[idx] = (1.0 - alpha) * last + alpha * blended[idx]
            actions = blended

        if self.queue_update_policy == "append":
            self.action_queue.extend(actions)
        elif self.queue_update_policy == "replace":
            self.action_queue = deque(actions)
        else:
            kept = list(self.action_queue)[: self.keep_old_actions]
            self.action_queue = deque(kept + [np.asarray(a, dtype=np.float32) for a in actions])

        self.latest_applied_request_id = max(self.latest_applied_request_id, request_id)
        self.stats.last_update_policy = self.queue_update_policy


def maybe_sleep(step_started: float, hz: float) -> None:
    if hz <= 0:
        return
    period = 1.0 / hz
    remaining = period - (time.perf_counter() - step_started)
    if remaining > 0:
        time.sleep(remaining)


def action_error(pred: np.ndarray, gt: Any) -> dict[str, float] | None:
    if gt is None:
        return None
    gt_arr = np.asarray(gt, dtype=np.float32).reshape(-1)
    pred_arr = pred.reshape(-1)
    dim = min(len(pred_arr), len(gt_arr))
    if dim == 0:
        return None
    diff = pred_arr[:dim] - gt_arr[:dim]
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def write_log_record(log_f, record: dict[str, Any]) -> None:
    if log_f is None:
        return
    log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log_f.flush()


def run(args: argparse.Namespace) -> None:
    jsonl_dir, video_root = resolve_dataset(args)
    image_keys = [part.strip() for part in args.image_keys.split(",") if part.strip()]
    episodes = list_episode_files(jsonl_dir, args.episode_glob, args.max_episodes)
    frame_loader = FrameLoader(video_root)
    effective_replan_every_steps = args.replan_every_steps
    if args.async_prefetch and effective_replan_every_steps == 0:
        effective_replan_every_steps = 20
    if args.async_prefetch:
        client = AsyncOnlineActionClient(
            base_url=args.base_url,
            timeout=args.timeout,
            dry_run=args.dry_run,
            queue_mode=args.queue_mode,
            dry_run_action_dim=args.dry_run_action_dim,
            prefetch_threshold=args.prefetch_threshold,
            queue_update_policy=args.queue_update_policy,
            keep_old_actions=args.keep_old_actions,
            blend_steps=args.blend_steps,
            max_inflight_requests=args.max_inflight_requests,
        )
    else:
        client = OnlineActionClient(
            base_url=args.base_url,
            timeout=args.timeout,
            dry_run=args.dry_run,
            queue_mode=args.queue_mode,
            dry_run_action_dim=args.dry_run_action_dim,
        )

    if args.warmup_requests > 0:
        first_episode = load_jsonl(episodes[0])
        first_frame = first_episode[args.start_frame]
        first_images = frame_loader.load_images(first_frame, image_keys)
        for _ in range(args.warmup_requests):
            if args.async_prefetch:
                client.prime(
                    first_frame.get("prompt", ""),
                    first_images,
                    first_frame.get("state") if args.send_state else None,
                )
                with client.lock:
                    client.action_queue.clear()
            else:
                client._request_actions(
                    first_frame.get("prompt", ""),
                    first_images,
                    first_frame.get("state") if args.send_state else None,
                )
                client.queue.clear()

    total_steps = 0
    errors: list[dict[str, float]] = []
    started = time.perf_counter()
    log_f = open(args.log_jsonl, "w", encoding="utf-8") if args.log_jsonl else None
    interrupted = False
    try:
        for episode_i, episode_path in enumerate(episodes):
            frames = load_jsonl(episode_path)
            frame_indices = range(args.start_frame, len(frames), max(args.stride, 1))
            for frame_i in frame_indices:
                if args.max_steps > 0 and total_steps >= args.max_steps:
                    break
                step_started = time.perf_counter()
                frame = frames[frame_i]
                images = frame_loader.load_images(frame, image_keys)
                force_replan = effective_replan_every_steps > 0 and (
                    total_steps % effective_replan_every_steps == 0
                )
                if args.async_prefetch:
                    if client.needs_prime():
                        client.prime(
                            frame.get("prompt", ""),
                            images,
                            frame.get("state") if args.send_state else None,
                        )
                    elif total_steps > 0:
                        client.maybe_submit(
                            step_idx=total_steps,
                            prompt=frame.get("prompt", ""),
                            images=images,
                            state=frame.get("state") if args.send_state else None,
                            force_replan=force_replan,
                        )
                    action = client.act(args.empty_queue_policy)
                    queue_remaining = client.queue_len()
                    stats_snapshot = client.stats_snapshot()
                    service_requests = stats_snapshot["count"]
                    inflight_count = stats_snapshot["inflight_count"]
                else:
                    action = client.act(
                        prompt=frame.get("prompt", ""),
                        images=images,
                        state=frame.get("state") if args.send_state else None,
                        force_replan=force_replan,
                        empty_queue_policy=args.empty_queue_policy,
                    )
                    queue_remaining = len(client.queue)
                    stats_snapshot = client.stats.snapshot()
                    inflight_count = 0
                    service_requests = client.stats.count
                err = action_error(action, frame.get("action")) if args.compare_action else None
                if err is not None:
                    errors.append(err)

                total_steps += 1
                record = {
                    "episode": str(episode_path),
                    "episode_index": episode_i,
                    "frame_index": frame_i,
                    "prompt": frame.get("prompt", ""),
                    "queue_remaining": queue_remaining,
                    "service_requests": service_requests,
                    "inflight_count": inflight_count,
                    "forced_replan": force_replan,
                    "action": action.tolist(),
                    "error": err,
                }
                write_log_record(log_f, record)

                if args.print_every > 0 and total_steps % args.print_every == 0:
                    suffix = ""
                    if errors:
                        suffix = f", mae={statistics.fmean(e['mae'] for e in errors):.6f}"
                    if args.async_prefetch:
                        suffix += (
                            f", inflight={stats_snapshot['inflight_count']}"
                            f", last_latency={stats_snapshot['last_latency']}"
                            f", empty={stats_snapshot['queue_empty_count']}"
                            f", repeat={stats_snapshot['repeat_last_count']}"
                            f", submitted={stats_snapshot['async_requests_submitted']}"
                            f", applied={stats_snapshot['async_requests_applied']}"
                        )
                    print(
                        f"[step {total_steps}] episode={episode_path.name} "
                        f"frame={frame_i} queue={queue_remaining}{suffix}",
                        flush=True,
                    )
                maybe_sleep(step_started, args.hz)
            if args.max_steps > 0 and total_steps >= args.max_steps:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user; keeping partial results.", flush=True)
    except RequestException as exc:
        print(f"\nRequest failed: {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        if log_f is not None:
            log_f.close()

    elapsed = time.perf_counter() - started
    print("\nReplay interrupted" if interrupted else "\nReplay finished")
    print(f"  episodes: {len(episodes)}")
    print(f"  steps: {total_steps}")
    print(f"  wall_time_sec: {elapsed:.3f}")
    if elapsed > 0:
        print(f"  effective_hz: {total_steps / elapsed:.3f}")
    final_stats = client.stats_snapshot() if args.async_prefetch else client.stats.snapshot()
    print(f"  service_requests: {final_stats['count']}")
    if final_stats["latencies"]:
        print(f"  latency_mean_sec: {statistics.fmean(final_stats['latencies']):.3f}")
        print(f"  latency_max_sec: {max(final_stats['latencies']):.3f}")
    if final_stats["response_lengths"]:
        print(
            f"  response_actions_mean: "
            f"{statistics.fmean(final_stats['response_lengths']):.2f}"
        )
        print(f"  response_actions_max: {max(final_stats['response_lengths'])}")
    print(f"  queue_empty_count: {final_stats['queue_empty_count']}")
    print(f"  repeat_last_count: {final_stats['repeat_last_count']}")
    print(f"  hold_count: {final_stats['hold_count']}")
    print(f"  zero_count: {final_stats['zero_count']}")
    print(f"  async_requests_submitted: {final_stats['async_requests_submitted']}")
    print(f"  async_requests_applied: {final_stats['async_requests_applied']}")
    print(f"  async_requests_failed: {final_stats['async_requests_failed']}")
    print(f"  stale_responses_discarded: {final_stats['stale_responses_discarded']}")
    print(f"  queue_update_policy: {args.queue_update_policy}")
    print(f"  replan_every_steps: {effective_replan_every_steps}")
    print(f"  prefetch_threshold: {args.prefetch_threshold}")
    if errors:
        print(f"  action_mae: {statistics.fmean(e['mae'] for e in errors):.6f}")
        print(f"  action_rmse: {statistics.fmean(e['rmse'] for e in errors):.6f}")
        print(f"  action_max_abs: {max(e['max_abs'] for e in errors):.6f}")
    if args.log_jsonl:
        print(f"  log_jsonl: {args.log_jsonl}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
