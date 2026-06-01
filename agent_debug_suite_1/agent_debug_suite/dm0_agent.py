#!/usr/bin/env python3
"""Standalone DM0 agent implementation used by sample_agent.SampleAgent."""

from __future__ import annotations

import io
import json
import logging
import threading
from collections import deque
from typing import Any

import numpy as np
import requests
from PIL import Image


logger = logging.getLogger(__name__)


DEFAULT_IMAGE_KEY_GROUPS = (
    (
        "images_1",
        "image.chest",
        "image.head",
        "image.front",
        "observation.image",
        "observation.images.chest",
        "observation.images.chest_rgb",
        "observation.images.head",
        "observation.images.front",
    ),
    (
        "images_2",
        "image.left",
        "image.wrist_left",
        "observation.images.left",
        "observation.images.left_wrist",
        "observation.images.left_wrist_rgb",
        "observation.images.wrist_left",
    ),
    (
        "images_3",
        "image.right",
        "image.wrist_right",
        "observation.images.right",
        "observation.images.right_wrist",
        "observation.images.right_wrist_rgb",
        "observation.images.wrist_right",
    ),
)

SPLIT_ACTION_SLICES = {
    "action.left_delta_tcp": slice(0, 6),
    "action.left_pinch": slice(6, 7),
    "action.right_delta_tcp": slice(7, 13),
    "action.right_pinch": slice(13, 14),
}


class DM0Agent:
    """DM0 /process_frame client.

    This mirrors ``hardware/xlerobot/custom_dm0_async_bridge.py``: upload the
    current image set and state to the Flask policy server, cache the returned
    action chunk, and return one action per ``act`` call.
    """

    def __init__(
        self,
        spec: dict[str, dict[str, Any]] | None = None,
        buffer: Any | None = None,
        action_value: float = 0.0,
        base_url: str | None = None,
        endpoint: str | None = None,
        vla_url: str | None = None,
        timeout_s: float = 30.0,
        image_keys: str | list[str] | None = None,
        state_key: str | None = None,
        action_dim: int = 14,
        queue_mode: str = "chunk",
        async_prefetch: bool = True,
        prefetch_threshold: int = 30,
        replan_every_steps: int = 20,
        queue_update_policy: str = "soft-replace",
        keep_old_actions: int = 3,
        blend_steps: int = 5,
    ) -> None:
        self.spec = spec or {}
        self.buffer = buffer
        self.action_value = float(action_value)
        self.endpoint = self._resolve_endpoint(endpoint=endpoint, vla_url=vla_url, base_url=base_url)
        self.timeout_s = float(timeout_s)
        self.image_keys = self._parse_image_keys(image_keys)
        self.state_key = state_key
        self.action_dim = int(action_dim)
        self.queue_mode = queue_mode
        self.async_prefetch = bool(async_prefetch)
        self.prefetch_threshold = int(prefetch_threshold)
        self.replan_every_steps = int(replan_every_steps)
        self.queue_update_policy = queue_update_policy
        self.keep_old_actions = max(0, int(keep_old_actions))
        self.blend_steps = max(0, int(blend_steps))
        self.action_queue: deque[np.ndarray] = deque()
        self.lock = threading.Lock()
        self.last_action: np.ndarray | None = None
        self.last_task: str | None = None
        self.step_counter = 0
        self.inflight = False
        self.request_generation = 0
        self.last_async_error: str | None = None
        self.async_requests_submitted = 0
        self.async_requests_applied = 0
        self.async_requests_failed = 0
        self.queue_empty_count = 0
        self.action_specs = {
            key: info
            for key, info in self.spec.items()
            if key == "action" or key.startswith("action.")
        }

    def act(self, obs: dict[str, np.ndarray], task: str) -> dict[str, np.ndarray]:
        """Return one action dict for the current observation and task."""
        if self.endpoint:
            if task != self.last_task:
                self._reset_for_task(task)
            if self._needs_prime():
                self._apply_action_chunk(self._fetch_action_chunk(obs, task))
            if self.async_prefetch:
                self._maybe_submit_prefetch(obs, task)
            action = self._pop_action()
            return self._format_action_for_spec(action)

        return {
            key: self._predict_action(key, info, obs, task)
            for key, info in self.action_specs.items()
        }

    @staticmethod
    def _resolve_endpoint(
        endpoint: str | None,
        vla_url: str | None,
        base_url: str | None,
    ) -> str | None:
        url = endpoint or vla_url or base_url
        if not url:
            return None
        url = url.rstrip("/")
        if url.endswith("/process_frame"):
            return url
        return f"{url}/process_frame"

    @staticmethod
    def _parse_image_keys(image_keys: str | list[str] | None) -> list[str] | None:
        if image_keys is None:
            return None
        if isinstance(image_keys, str):
            return [item.strip() for item in image_keys.split(",") if item.strip()]
        return list(image_keys)

    def _fetch_action_chunk(self, obs: dict[str, np.ndarray], task: str) -> np.ndarray:
        images = self._select_images(obs)
        files = [
            ("image", (f"image_{idx}.png", self._encode_png_rgb(image), "image/png"))
            for idx, image in enumerate(images)
        ]
        data: dict[str, str] = {"text": task, "batch_size": "1"}
        state = self._select_state(obs)
        if state is not None:
            data["states"] = json.dumps(np.asarray(state, dtype=np.float32).reshape(-1).tolist())

        response = requests.post(self.endpoint, data=data, files=files, timeout=self.timeout_s)
        if response.status_code != 200:
            body = response.text[:2000].replace("\n", "\\n")
            raise RuntimeError(f"DM0 inference service returned HTTP {response.status_code}: {body}")

        payload = response.json()
        chunk = self._normalize_action_response(payload.get("response"))
        if chunk.size == 0:
            raise RuntimeError(f"DM0 inference service returned no actions: {payload}")
        if chunk.ndim != 2:
            raise ValueError(f"Expected DM0 action chunk [T, D], got shape {chunk.shape}")
        if self.queue_mode == "latest":
            chunk = chunk[:1]
        return np.asarray(chunk[:, : self.action_dim], dtype=np.float32)

    def _reset_for_task(self, task: str) -> None:
        with self.lock:
            self.request_generation += 1
            self.action_queue.clear()
            self.last_action = None
            self.last_task = task
            self.step_counter = 0
            self.inflight = False
            self.last_async_error = None

    def _needs_prime(self) -> bool:
        with self.lock:
            return not self.action_queue and self.last_action is None

    def _maybe_submit_prefetch(self, obs: dict[str, np.ndarray], task: str) -> None:
        with self.lock:
            force_replan = (
                self.replan_every_steps > 0
                and self.step_counter > 0
                and self.step_counter % self.replan_every_steps == 0
            )
            if not force_replan and len(self.action_queue) > self.prefetch_threshold:
                return
            if self.inflight:
                return
            self.inflight = True
            generation = self.request_generation
            self.async_requests_submitted += 1

        obs_snapshot = {key: np.copy(value) for key, value in obs.items()}
        threading.Thread(
            target=self._prefetch_worker,
            args=(generation, obs_snapshot, task),
            daemon=True,
        ).start()

    def _prefetch_worker(
        self,
        generation: int,
        obs: dict[str, np.ndarray],
        task: str,
    ) -> None:
        try:
            chunk = self._fetch_action_chunk(obs, task)
        except Exception as exc:
            with self.lock:
                if generation == self.request_generation:
                    self.inflight = False
                    self.last_async_error = f"{type(exc).__name__}: {exc}"
                    self.async_requests_failed += 1
                    logger.warning("DM0 async prefetch failed: %s", self.last_async_error)
            return

        with self.lock:
            if generation != self.request_generation or task != self.last_task:
                return
            self.inflight = False
            self.last_async_error = None
            self._apply_action_chunk_locked(chunk)
            self.async_requests_applied += 1

    def _apply_action_chunk(self, chunk: np.ndarray) -> None:
        with self.lock:
            self._apply_action_chunk_locked(chunk)

    def _apply_action_chunk_locked(self, chunk: np.ndarray) -> None:
        actions = np.asarray(chunk, dtype=np.float32)
        if self.blend_steps > 0 and self.last_action is not None and len(actions) > 0:
            blend_count = min(self.blend_steps, len(actions))
            actions = np.copy(actions)
            for idx in range(blend_count):
                alpha = float(idx + 1) / float(blend_count + 1)
                actions[idx] = (1.0 - alpha) * self.last_action + alpha * actions[idx]

        if self.queue_update_policy == "append":
            self.action_queue.extend(actions)
        elif self.queue_update_policy == "replace":
            self.action_queue = deque(actions)
        elif self.queue_update_policy == "soft-replace":
            kept = list(self.action_queue)[: self.keep_old_actions]
            self.action_queue = deque(kept + [np.asarray(action, dtype=np.float32) for action in actions])
        else:
            raise ValueError(f"Unsupported queue_update_policy: {self.queue_update_policy!r}")

    def _pop_action(self) -> np.ndarray:
        with self.lock:
            self.step_counter += 1
            if self.action_queue:
                action = np.asarray(self.action_queue.popleft(), dtype=np.float32)
                self.last_action = np.copy(action)
                return action
            self.queue_empty_count += 1
            if self.queue_empty_count == 1 or self.queue_empty_count % 10 == 0:
                logger.warning(
                    "DM0 action queue empty %d time(s); repeating the last action while prefetch is running",
                    self.queue_empty_count,
                )
            if self.last_action is not None:
                return np.copy(self.last_action)
            return np.full((self.action_dim,), self.action_value, dtype=np.float32)

    def _select_images(self, obs: dict[str, np.ndarray]) -> list[np.ndarray]:
        if self.image_keys:
            missing = [key for key in self.image_keys if key not in obs]
            if missing:
                raise KeyError(f"Missing configured image observation keys: {missing}")
            return [self._prepare_rgb_image(obs[key]) for key in self.image_keys]

        images: list[np.ndarray] = []
        for key_group in DEFAULT_IMAGE_KEY_GROUPS:
            for key in key_group:
                if key in obs:
                    images.append(self._prepare_rgb_image(obs[key]))
                    break

        if images:
            return images

        image_keys = self._auto_image_keys(obs)
        if not image_keys:
            raise KeyError(
                "No image observations found. Expected keys like images_1/images_2/images_3 "
                "or observation.images.{chest,left,right}."
            )
        return [self._prepare_rgb_image(obs[key]) for key in image_keys]

    @staticmethod
    def _auto_image_keys(obs: dict[str, np.ndarray]) -> list[str]:
        keys = []
        for key, value in obs.items():
            arr = np.asarray(value)
            if arr.ndim == 3 and (arr.shape[0] in {1, 3} or arr.shape[-1] in {1, 3}):
                keys.append(key)

        def score(key: str) -> tuple[int, str]:
            name = key.lower()
            if any(part in name for part in ("chest", "head", "front")):
                return (0, key)
            if "left" in name:
                return (1, key)
            if "right" in name:
                return (2, key)
            return (3, key)

        return sorted(keys, key=score)[:3]

    def _select_state(self, obs: dict[str, np.ndarray]) -> np.ndarray | None:
        candidates = []
        if self.state_key:
            candidates.append(self.state_key)
        candidates.extend(("state", "observation.state"))
        for key in candidates:
            if key in obs:
                return np.asarray(obs[key], dtype=np.float32)
        return None

    def _format_action_for_spec(self, action: np.ndarray) -> dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if not self.action_specs:
            return {"action": action}
        if "action" in self.action_specs:
            return {"action": self._coerce_shape(action, self.action_specs["action"])}

        result: dict[str, np.ndarray] = {}
        for key, info in self.action_specs.items():
            if key in SPLIT_ACTION_SLICES:
                result[key] = self._coerce_shape(action[SPLIT_ACTION_SLICES[key]], info)
            elif key.startswith("action."):
                raise KeyError(f"Unsupported split action spec key: {key}")
        return result or {"action": action}

    @staticmethod
    def _coerce_shape(action: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        shape = tuple(info.get("shape", ()))
        if not shape:
            return action.astype(np.float32, copy=False)
        size = int(np.prod(shape))
        flat = action.reshape(-1)
        if flat.size < size:
            flat = np.pad(flat, (0, size - flat.size))
        elif flat.size > size:
            flat = flat[:size]
        return flat.astype(np.float32, copy=False).reshape(shape)

    @staticmethod
    def _normalize_action_response(actions: Any) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.size == 0:
            return actions
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim == 1:
            actions = actions[None, :]
        return actions

    @staticmethod
    def _prepare_rgb_image(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError(f"Expected image [H, W, C] or [C, H, W], got {arr.shape}")
        if arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] != 3:
            raise ValueError(f"Expected RGB image with 3 channels, got {arr.shape}")
        if arr.dtype != np.uint8:
            if arr.dtype.kind == "f" and np.nanmax(arr) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).round().astype(np.uint8)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _encode_png_rgb(image: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        Image.fromarray(DM0Agent._prepare_rgb_image(image), mode="RGB").save(buffer, format="PNG")
        return buffer.getvalue()

    def _predict_action(
        self,
        key: str,
        info: dict[str, Any],
        obs: dict[str, np.ndarray],
        task: str,
    ) -> np.ndarray:
        """Fallback zero/constant action used when no inference endpoint is set."""
        del key, obs, task
        shape = tuple(info.get("shape", ()))
        return np.full(shape, self.action_value, dtype=np.float32)

    def teardown(self) -> None:
        with self.lock:
            self.request_generation += 1
            self.action_queue.clear()
            self.last_action = None
            self.inflight = False
