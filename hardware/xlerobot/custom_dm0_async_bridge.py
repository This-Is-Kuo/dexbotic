#!/usr/bin/env python3
# Custom DM0 async bridge.
# This file is intentionally separated from hardware/xlerobot/bridge.py
# to avoid modifying upstream/original dexbotic code.

from __future__ import annotations

import argparse
import json
import logging
import pickle
import statistics
import threading
import time
from collections import deque
from concurrent import futures
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Iterator, List

import cv2
import grpc
import numpy as np
import requests
import torch
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import receive_bytes_in_chunks


logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(asctime)s %(name)s: %(message)s",
)
logger = logging.getLogger("CustomDM0AsyncBridge")
logger.setLevel(logging.INFO)

ACTION_DIM = 16
CAMERA_NAMES = ("head", "wrist_left", "wrist_right")

STATE_FIELD_NAMES = (
    "left_arm_shoulder_pan.pos",
    "left_arm_shoulder_lift.pos",
    "left_arm_elbow_flex.pos",
    "left_arm_wrist_flex.pos",
    "left_arm_wrist_roll.pos",
    "left_arm_gripper.pos",
    "right_arm_shoulder_pan.pos",
    "right_arm_shoulder_lift.pos",
    "right_arm_elbow_flex.pos",
    "right_arm_wrist_flex.pos",
    "right_arm_wrist_roll.pos",
    "right_arm_gripper.pos",
    "head_motor_1.pos",
    "head_motor_2.pos",
    "x.vel",
    "theta.vel",
)


@dataclass
class TimedAction:
    timestamp: float
    timestep: int
    action: torch.Tensor


@dataclass
class BridgeStats:
    steps: int = 0
    service_requests: int = 0
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

    def add_request(self, latency: float, response_length: int) -> None:
        self.service_requests += 1
        self.latencies.append(latency)
        self.response_lengths.append(response_length)
        self.last_latency = latency


def normalize_action_response(actions: Any) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.size == 0:
        return actions
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim == 1:
        actions = actions[None, :]
    return actions


class InferenceClient:
    def __init__(self, vla_url: str, prompt: str, timeout: float = 10.0):
        self.vla_url = vla_url.rstrip("/")
        self.prompt = prompt
        self.timeout = timeout

    def request_actions(
        self, images: List[np.ndarray], states: List[float] | None
    ) -> tuple[np.ndarray, float]:
        files = []
        for i, image in enumerate(images):
            success, encoded_image = cv2.imencode(".png", image)
            if not success:
                raise RuntimeError(f"Failed to encode image from camera {CAMERA_NAMES[i]}")
            files.append(
                (
                    "image",
                    (f"{CAMERA_NAMES[i]}.png", encoded_image.tobytes(), "image/png"),
                )
            )

        data = {"text": self.prompt}
        if states is not None:
            data["states"] = json.dumps(states)

        t0 = time.perf_counter()
        response = requests.post(self.vla_url, data=data, files=files, timeout=self.timeout)
        latency = time.perf_counter() - t0
        response.raise_for_status()
        actions = normalize_action_response(response.json().get("response", []))
        if actions.size == 0:
            raise RuntimeError("Received empty actions from VLA")
        return actions, latency


class AsyncActionBuffer:
    def __init__(
        self,
        inference_client: InferenceClient,
        *,
        async_prefetch: bool,
        prefetch_threshold: int,
        queue_update_policy: str,
        keep_old_actions: int,
        blend_steps: int,
        max_inflight_requests: int,
        empty_queue_policy: str,
    ):
        self.inference_client = inference_client
        self.async_prefetch = async_prefetch
        self.prefetch_threshold = prefetch_threshold
        self.queue_update_policy = queue_update_policy
        self.keep_old_actions = max(0, keep_old_actions)
        self.blend_steps = max(0, blend_steps)
        self.max_inflight_requests = max(1, max_inflight_requests)
        self.empty_queue_policy = empty_queue_policy

        self.lock = threading.Lock()
        self.action_queue: deque[np.ndarray] = deque()
        self.last_action: np.ndarray | None = None
        self.inflight_count = 0
        self.next_request_id = 0
        self.latest_applied_request_id = 0
        self.stats = BridgeStats()

    def reset(self) -> None:
        with self.lock:
            self.action_queue.clear()
            self.last_action = None
            self.inflight_count = 0
            self.next_request_id = 0
            self.latest_applied_request_id = 0
            self.stats = BridgeStats()

    def prime(self, images: list[np.ndarray], state: list[float]) -> bool:
        try:
            actions, latency = self.inference_client.request_actions(images, state)
        except Exception as exc:
            with self.lock:
                self.stats.async_requests_failed += 1
            logger.warning("Prime request failed: %s", exc)
            return False
        with self.lock:
            self.stats.add_request(latency, len(actions))
            self._apply_actions_locked(actions, request_id=0)
        logger.info("Primed queue with %d actions in %.3fs", len(actions), latency)
        return True

    def maybe_submit(
        self,
        step_idx: int,
        images: list[np.ndarray],
        state: list[float],
        force_replan: bool,
    ) -> None:
        with self.lock:
            queue_len = len(self.action_queue)
            if not force_replan and queue_len > self.prefetch_threshold:
                return
            if self.inflight_count >= self.max_inflight_requests:
                return
            self.next_request_id += 1
            request_id = self.next_request_id
            self.inflight_count += 1
            self.stats.async_requests_submitted += 1

        image_snapshot = [np.copy(image) for image in images]
        state_snapshot = list(state)
        thread = threading.Thread(
            target=self._request_worker,
            args=(request_id, step_idx, image_snapshot, state_snapshot),
            daemon=True,
        )
        thread.start()
        logger.info(
            "Submitted async request_id=%s step=%s queue=%s force=%s",
            request_id,
            step_idx,
            queue_len,
            force_replan,
        )

    def pop_action(self) -> np.ndarray:
        with self.lock:
            self.stats.steps += 1
            if self.action_queue:
                action = self.action_queue.popleft()
                self.last_action = np.copy(action)
                return np.copy(action)

            self.stats.queue_empty_count += 1
            if self.empty_queue_policy == "repeat-last" and self.last_action is not None:
                self.stats.repeat_last_count += 1
                return np.copy(self.last_action)

            action = np.zeros(ACTION_DIM, dtype=np.float32)
            if self.empty_queue_policy == "hold":
                self.stats.hold_count += 1
            else:
                self.stats.zero_count += 1
            return action

    def needs_prime(self) -> bool:
        with self.lock:
            return not self.action_queue and self.last_action is None

    def queue_len(self) -> int:
        with self.lock:
            return len(self.action_queue)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "steps": self.stats.steps,
                "service_requests": self.stats.service_requests,
                "latencies": list(self.stats.latencies or []),
                "response_lengths": list(self.stats.response_lengths or []),
                "queue_empty_count": self.stats.queue_empty_count,
                "repeat_last_count": self.stats.repeat_last_count,
                "hold_count": self.stats.hold_count,
                "zero_count": self.stats.zero_count,
                "async_requests_submitted": self.stats.async_requests_submitted,
                "async_requests_applied": self.stats.async_requests_applied,
                "async_requests_failed": self.stats.async_requests_failed,
                "stale_responses_discarded": self.stats.stale_responses_discarded,
                "last_latency": self.stats.last_latency,
                "last_update_policy": self.stats.last_update_policy,
                "inflight_count": self.inflight_count,
                "queue_len": len(self.action_queue),
            }

    def _request_worker(
        self,
        request_id: int,
        step_idx: int,
        images: list[np.ndarray],
        state: list[float],
    ) -> None:
        try:
            actions, latency = self.inference_client.request_actions(images, state)
        except Exception as exc:
            with self.lock:
                self.inflight_count -= 1
                self.stats.async_requests_failed += 1
            logger.warning("Async request_id=%s failed: %s", request_id, exc)
            return

        with self.lock:
            self.inflight_count -= 1
            self.stats.add_request(latency, len(actions))
            if request_id < self.latest_applied_request_id:
                self.stats.stale_responses_discarded += 1
                logger.info("Stale response discarded request_id=%s", request_id)
                return
            self._apply_actions_locked(actions, request_id)
            self.stats.async_requests_applied += 1
        logger.info(
            "Applied async request_id=%s step=%s latency=%.3fs actions=%d",
            request_id,
            step_idx,
            latency,
            len(actions),
        )

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


class BridgeService(services_pb2_grpc.AsyncInferenceServicer):
    def __init__(
        self,
        action_buffer: AsyncActionBuffer,
        *,
        hz: float,
        replan_every_steps: int,
        print_every: int,
        show_images: bool = False,
    ):
        self.action_buffer = action_buffer
        self.environment_dt = 1.0 / hz if hz > 0 else 0.1
        self.replan_every_steps = replan_every_steps
        self.print_every = print_every
        self.show_images = show_images
        self.observation_queue = Queue(maxsize=1)
        self.vis_queue = Queue(maxsize=1)
        self.shutdown_event = threading.Event()
        self.started_at = time.perf_counter()
        self.step_counter = 0

    def Ready(self, request: services_pb2.Empty, context: grpc.ServicerContext) -> services_pb2.Empty:
        self.observation_queue = Queue(maxsize=1)
        self._drain_vis_queue()
        self.action_buffer.reset()
        self.shutdown_event.clear()
        self.started_at = time.perf_counter()
        self.step_counter = 0
        logger.info("Robot Connected and Ready.")
        return services_pb2.Empty()

    def SendPolicyInstructions(
        self, request: services_pb2.PolicySetup, context: grpc.ServicerContext
    ) -> services_pb2.Empty:
        return services_pb2.Empty()

    def SendObservations(
        self,
        request_iterator: Iterator[services_pb2.Observation],
        context: grpc.ServicerContext,
    ) -> services_pb2.Empty:
        data = receive_bytes_in_chunks(request_iterator, None, self.shutdown_event)
        if data is None:
            return services_pb2.Empty()

        obs = pickle.loads(data)
        for cam in CAMERA_NAMES:
            if cam in obs.observation:
                obs.observation[cam] = cv2.cvtColor(obs.observation[cam], cv2.COLOR_RGB2BGR)

        if self.observation_queue.full():
            try:
                self.observation_queue.get_nowait()
            except Empty:
                pass
        self.observation_queue.put(obs)
        return services_pb2.Empty()

    def GetActions(
        self, request: services_pb2.Empty, context: grpc.ServicerContext
    ) -> services_pb2.Actions:
        try:
            timed_obs = self.observation_queue.get(timeout=2.0)
            raw_obs = timed_obs.observation
            images = [raw_obs[cam] for cam in CAMERA_NAMES]
            state = self._extract_state(raw_obs)

            if self.action_buffer.needs_prime():
                self.action_buffer.prime(images, state)
            elif self.action_buffer.async_prefetch:
                force_replan = (
                    self.replan_every_steps > 0
                    and self.step_counter > 0
                    and self.step_counter % self.replan_every_steps == 0
                )
                self.action_buffer.maybe_submit(
                    self.step_counter,
                    images,
                    state,
                    force_replan=force_replan,
                )

            if self.show_images:
                self._push_vis(np.hstack(images))

            action = self.action_buffer.pop_action()
            self.step_counter += 1
            self._maybe_log_status()
            timed_action = TimedAction(
                timestamp=timed_obs.timestamp,
                timestep=timed_obs.timestep + 1,
                action=torch.from_numpy(action).float(),
            )
            return services_pb2.Actions(data=pickle.dumps([timed_action]))
        except Empty:
            return services_pb2.Empty()
        except Exception as exc:
            logger.error("Error in GetActions: %s", exc, exc_info=True)
            return services_pb2.Empty()

    def print_summary(self) -> None:
        stats = self.action_buffer.snapshot()
        elapsed = time.perf_counter() - self.started_at
        logger.info("Replay/bridge summary")
        logger.info("  steps: %s", stats["steps"])
        logger.info("  wall_time_sec: %.3f", elapsed)
        if elapsed > 0:
            logger.info("  effective_hz: %.3f", stats["steps"] / elapsed)
        logger.info("  service_requests: %s", stats["service_requests"])
        if stats["latencies"]:
            logger.info("  latency_mean_sec: %.3f", statistics.fmean(stats["latencies"]))
            logger.info("  latency_max_sec: %.3f", max(stats["latencies"]))
        if stats["response_lengths"]:
            logger.info(
                "  response_actions_mean: %.2f",
                statistics.fmean(stats["response_lengths"]),
            )
        for key in (
            "queue_empty_count",
            "repeat_last_count",
            "hold_count",
            "zero_count",
            "async_requests_submitted",
            "async_requests_applied",
            "async_requests_failed",
            "stale_responses_discarded",
        ):
            logger.info("  %s: %s", key, stats[key])
        logger.info("  queue_update_policy: %s", self.action_buffer.queue_update_policy)
        logger.info("  replan_every_steps: %s", self.replan_every_steps)
        logger.info("  prefetch_threshold: %s", self.action_buffer.prefetch_threshold)

    @staticmethod
    def _extract_state(raw_obs: dict[str, Any]) -> list[float]:
        state = []
        for key in STATE_FIELD_NAMES:
            val = raw_obs.get(key, 0.0)
            if hasattr(val, "item"):
                val = val.item()
            state.append(float(val))
        return state

    def _drain_vis_queue(self) -> None:
        while not self.vis_queue.empty():
            try:
                self.vis_queue.get_nowait()
            except Empty:
                break

    def _push_vis(self, canvas: np.ndarray) -> None:
        if self.vis_queue.full():
            try:
                self.vis_queue.get_nowait()
            except Empty:
                pass
        self.vis_queue.put(canvas)

    def _maybe_log_status(self) -> None:
        if self.print_every <= 0 or self.step_counter % self.print_every != 0:
            return
        stats = self.action_buffer.snapshot()
        logger.info(
            "step=%s queue=%s inflight=%s last_latency=%s empty=%s repeat=%s "
            "submitted=%s applied=%s",
            self.step_counter,
            stats["queue_len"],
            stats["inflight_count"],
            stats["last_latency"],
            stats["queue_empty_count"],
            stats["repeat_last_count"],
            stats["async_requests_submitted"],
            stats["async_requests_applied"],
        )


def serve(args: argparse.Namespace) -> None:
    inference_client = InferenceClient(args.vla_url, args.prompt, timeout=args.timeout)
    action_buffer = AsyncActionBuffer(
        inference_client,
        async_prefetch=args.async_prefetch,
        prefetch_threshold=args.prefetch_threshold,
        queue_update_policy=args.queue_update_policy,
        keep_old_actions=args.keep_old_actions,
        blend_steps=args.blend_steps,
        max_inflight_requests=args.max_inflight_requests,
        empty_queue_policy=args.empty_queue_policy,
    )
    bridge = BridgeService(
        action_buffer,
        hz=args.hz,
        replan_every_steps=args.replan_every_steps,
        print_every=args.print_every,
        show_images=args.show_images,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(bridge, server)
    server.add_insecure_port(f"[::]:{args.port}")
    logger.info("Custom DM0 async bridge started on [::]:%s", args.port)
    logger.info("async_prefetch=%s prompt=%s", args.async_prefetch, args.prompt)
    server.start()

    try:
        if args.show_images:
            cv2.namedWindow("Custom DM0 Async Bridge", cv2.WINDOW_AUTOSIZE)
            while not bridge.shutdown_event.is_set():
                try:
                    canvas = bridge.vis_queue.get(timeout=0.05)
                    cv2.imshow("Custom DM0 Async Bridge", canvas)
                except Empty:
                    pass
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    bridge.shutdown_event.set()
                    break
            cv2.destroyAllWindows()
        else:
            server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
    finally:
        bridge.shutdown_event.set()
        bridge.print_summary()
        server.stop(0)
        logger.info("Bridge stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom DM0 Async XLeRobot Bridge")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--vla_url", type=str, default="http://localhost:7891/process_frame")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Pick up scattered longans from the table and place them into the box",
    )
    parser.add_argument("--show_images", action="store_true")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--async-prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replan-every-steps", type=int, default=20)
    parser.add_argument("--prefetch-threshold", type=int, default=20)
    parser.add_argument(
        "--empty-queue-policy",
        choices=["repeat-last", "hold", "zero"],
        default="repeat-last",
    )
    parser.add_argument(
        "--queue-update-policy",
        choices=["append", "replace", "soft-replace"],
        default="soft-replace",
    )
    parser.add_argument("--keep-old-actions", type=int, default=3)
    parser.add_argument("--blend-steps", type=int, default=5)
    parser.add_argument("--max-inflight-requests", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    serve(parse_args())
