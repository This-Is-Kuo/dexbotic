# DM0 Inference Service UI Integration Notes

This document describes the data format, model input/output contract, and the
recommended adapter behavior for connecting the current DM0 inference service to
an external UI agent interface.

The target UI interface is:

```python
class SampleAgent:
    def __init__(self, spec=None, buffer=None) -> None:
        self.spec = spec or {}

    def act(self, obs: dict[str, np.ndarray], task: str) -> dict[str, np.ndarray]:
        ...

    def teardown(self) -> None:
        pass
```

## Current Model Variant

The current custom DM0 entrypoint is:

```text
playground/post_data_01_dm0_deltafix.py
```

The relevant runtime defaults are:

```text
num_images: 3
raw_state_dim: 16
raw_action_dim: 14
model_action_dim: 32
chunk_size: 50
inference_service_port: 7891
```

The training dataset currently inspected locally is:

```text
data/post_origin_data_0423_dexdata_train/jsonl
```

Dataset summary:

```text
episode files: 493
frames: 281610
image views per frame: 3
state dim: 16
action dim: 14
```

Every frame has these fields:

```text
images_1
images_2
images_3
state
prompt
is_robot
action
extra
```

## Raw Dataset Fields

Each jsonl row represents one timestep.

Example structure. The `state` and `action` arrays are abbreviated here; the
actual lengths are 16 and 14.

```json
{
  "images_1": {
    "type": "video",
    "url": "observation.images.chest/observation.images.chest_rgb/episode_000000.mp4",
    "frame_idx": 0
  },
  "images_2": {
    "type": "video",
    "url": "observation.images.left/observation.images.left_wrist_rgb/episode_000000.mp4",
    "frame_idx": 0
  },
  "images_3": {
    "type": "video",
    "url": "observation.images.right/observation.images.right_wrist_rgb/episode_000000.mp4",
    "frame_idx": 0
  },
  "state": [0.0, 0.0, 0.0],
  "prompt": "express_pick",
  "is_robot": true,
  "action": [0.0, 0.0, 0.0],
  "extra": {
    "timestamp": 0.0,
    "frame_index": 0,
    "episode_index": 0,
    "task_index": 0,
    "source_format": "lerobot_v2_jsonl_post_origin",
    "state_mode": "minimal",
    "schema": "origin_102"
  }
}
```

The three image streams are:

| Dataset key | Logical camera | Original video resolution |
|---|---|---:|
| `images_1` | chest / global camera | `1280x720` |
| `images_2` | left wrist camera | `1280x720` |
| `images_3` | right wrist camera | `1280x720` |

At inference, images are converted to RGB PIL images and preprocessed by the
model's image processor. For the current `DM0-base` vision tower
`pe_lang_l14_728`, each image becomes:

```text
[3, 728, 728]
```

With three cameras, model input image tensor shape is:

```text
[batch, 3, 3, 728, 728]
```

## State Format

Raw state is 16D:

```text
left_tcp(7) + right_tcp(7) + left_pinch(1) + right_pinch(1)
```

Index layout:

| Index range | Meaning | Dim |
|---:|---|---:|
| `0:7` | left TCP absolute pose | 7 |
| `7:14` | right TCP absolute pose | 7 |
| `14` | left pinch state | 1 |
| `15` | right pinch state | 1 |

TCP pose is expected to be:

```text
x, y, z, qx, qy, qz, qw
```

During training and inference, state is padded to the model action dimension:

```text
raw state: [16]
model state: [32]
```

The UI side should provide the raw 16D state when available. The inference
service will pad it to 32D internally. If no state is provided, the current
service fills a zero state of shape `[batch, 32]`; this is supported by code but
is not recommended for this model because the policy was trained with state.

## Action Format

Raw training action is 14D:

```text
left_delta_tcp(6) + left_pinch_target(1) + right_delta_tcp(6) + right_pinch_target(1)
```

Index layout:

| Index range | Meaning | Dim |
|---:|---|---:|
| `0:6` | left TCP delta command | 6 |
| `6` | left pinch target | 1 |
| `7:13` | right TCP delta command | 6 |
| `13` | right pinch target | 1 |

Each 6D TCP delta is expected to be:

```text
dx, dy, dz, droll, dpitch, dyaw
```

The gripper/pinch dimensions are non-delta dimensions:

```text
non_delta_mask = [6, 13]
```

During training, action is padded and chunked:

```text
raw action per step: [14]
padded action per step: [32]
training target chunk: [50, 32]
```

The model predicts a 50-step action chunk. The inference service then slices the
first 14 dimensions before returning to clients:

```text
service response shape: [50, 14]
```

Important: in `playground/post_data_01_dm0_deltafix.py`, inference applies:

```text
ActionDenorm -> AbsoluteAction -> slice first 14 dims
```

So the current HTTP service response is the postprocessed command after adding
the current state on dimensions where `AbsoluteAction` applies. The pinch
dimensions `6` and `13` remain direct predicted targets because they are listed
in `non_delta_mask`.

If the UI/robot executor expects pure raw delta actions exactly matching the
training jsonl `action`, either:

1. Update the inference output transform to remove `AbsoluteAction`, or
2. Convert the returned command back to the executor's expected representation.

Do not apply an additional generic delta-to-absolute conversion unless the UI
executor explicitly expects absolute commands. Double-applying this conversion
will produce incorrect commands.

## HTTP Inference Service Contract

Endpoint:

```text
POST /process_frame
```

Default base URL:

```text
http://localhost:7891
```

Form fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | yes | Task prompt, same semantic role as dataset `prompt`. |
| `image` | repeated multipart file | yes | One or more RGB images. Current model expects 3 images. |
| `states` | JSON string | recommended | State vector `[16]` or batch state `[[16], ...]`. |
| `batch_size` | int string | optional | Defaults to `1`. |

Image ordering must match training:

```text
image[0] = chest/global camera
image[1] = left wrist camera
image[2] = right wrist camera
```

If fewer than 3 images are sent, the service pads missing images with zeros and
sets image masks accordingly. This is supported but not recommended unless the
model was validated with missing views.

Response:

```json
{
  "response": [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  ]
}
```

For a single request, `response` is normally a list of 50 actions:

```text
response: [50, 14]
```

Recommended client behavior:

1. Call `/process_frame` when the local action queue is empty.
2. Cache the returned 50-step action chunk.
3. Return one 14D action per `Agent.act(...)` call.
4. Refresh the chunk when the queue is empty or when the task/context changes.

## Recommended UI Agent Mapping

The UI gives:

```python
obs: dict[str, np.ndarray]
task: str
```

Recommended expected observation keys:

| UI obs key | Shape | Dtype | Required | Mapping |
|---|---:|---|---|---|
| `images_1` or `image.chest` | `[H, W, 3]` | `uint8` | yes | chest image |
| `images_2` or `image.left` | `[H, W, 3]` | `uint8` | yes | left wrist image |
| `images_3` or `image.right` | `[H, W, 3]` | `uint8` | yes | right wrist image |
| `state` | `[16]` | float | recommended | raw 16D state |

If the UI already uses dataset-style names, prefer:

```text
obs["images_1"]
obs["images_2"]
obs["images_3"]
obs["state"]
```

If the UI uses camera names, use:

```text
obs["image.chest"]
obs["image.left"]
obs["image.right"]
obs["state"]
```

The adapter should return a dict matching the UI `spec`. For this model, the
main action key should be a 14D vector:

```python
{
    "action": np.ndarray(shape=(14,), dtype=np.float32)
}
```

If the UI spec uses split action keys, map the 14D vector as:

| Output key | Slice |
|---|---|
| `action.left_delta_tcp` | `action[0:6]` |
| `action.left_pinch` | `action[6:7]` |
| `action.right_delta_tcp` | `action[7:13]` |
| `action.right_pinch` | `action[13:14]` |

## Adapter Skeleton

```python
from __future__ import annotations

from collections import deque
from typing import Any
import io
import json

import numpy as np
import requests
from PIL import Image


class DexboticDM0Agent:
    def __init__(
        self,
        spec: dict[str, dict[str, Any]] | None = None,
        buffer: Any | None = None,
        base_url: str = "http://localhost:7891",
    ) -> None:
        self.spec = spec or {}
        self.base_url = base_url.rstrip("/")
        self.queue: deque[np.ndarray] = deque()
        self.last_task: str | None = None

    def act(self, obs: dict[str, np.ndarray], task: str) -> dict[str, np.ndarray]:
        if task != self.last_task:
            self.queue.clear()
            self.last_task = task

        if not self.queue:
            self._request_action_chunk(obs, task)

        action = self.queue.popleft().astype(np.float32)
        return self._format_action_for_spec(action)

    def _request_action_chunk(self, obs: dict[str, np.ndarray], task: str) -> None:
        images = [
            self._get_image(obs, "images_1", "image.chest"),
            self._get_image(obs, "images_2", "image.left"),
            self._get_image(obs, "images_3", "image.right"),
        ]

        files = []
        for image in images:
            files.append(("image", self._encode_png_rgb(image)))

        data = {"text": task, "batch_size": "1"}
        if "state" in obs:
            data["states"] = json.dumps(np.asarray(obs["state"], dtype=np.float32).tolist())

        response = requests.post(
            f"{self.base_url}/process_frame",
            data=data,
            files=files,
            timeout=30,
        )
        response.raise_for_status()
        chunk = np.asarray(response.json()["response"], dtype=np.float32)

        if chunk.ndim != 2 or chunk.shape[1] != 14:
            raise ValueError(f"Expected action chunk [T, 14], got {chunk.shape}")

        for item in chunk:
            self.queue.append(item)

    def _format_action_for_spec(self, action: np.ndarray) -> dict[str, np.ndarray]:
        if not self.spec or "action" in self.spec:
            return {"action": action}

        result: dict[str, np.ndarray] = {}
        for key in self.spec:
            if key == "action.left_delta_tcp":
                result[key] = action[0:6]
            elif key == "action.left_pinch":
                result[key] = action[6:7]
            elif key == "action.right_delta_tcp":
                result[key] = action[7:13]
            elif key == "action.right_pinch":
                result[key] = action[13:14]
            elif key.startswith("action."):
                raise KeyError(f"Unsupported action spec key: {key}")

        if not result:
            result["action"] = action
        return result

    @staticmethod
    def _get_image(obs: dict[str, np.ndarray], primary: str, fallback: str) -> np.ndarray:
        if primary in obs:
            return obs[primary]
        if fallback in obs:
            return obs[fallback]
        raise KeyError(f"Missing image observation: expected {primary!r} or {fallback!r}")

    @staticmethod
    def _encode_png_rgb(image: np.ndarray) -> tuple[str, bytes, str]:
        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected RGB image [H, W, 3], got {image.shape}")

        buffer = io.BytesIO()
        Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
        return ("image.png", buffer.getvalue(), "image/png")

    def teardown(self) -> None:
        self.queue.clear()
```

## Validation Checklist

Before connecting to the UI, verify:

1. The inference service is running on the expected port.
2. The UI sends images in the order `chest, left wrist, right wrist`.
3. Images are RGB arrays with shape `[H, W, 3]`.
4. State is the raw 16D vector if available.
5. The UI spec accepts either `action` shape `[14]` or the four split action keys.
6. The executor's expected action semantics match the service output semantics.
7. No extra delta-to-absolute conversion is applied unless explicitly required by the executor.

## Minimal Curl Example

```bash
curl -X POST http://localhost:7891/process_frame \
  -F 'text=express_pick' \
  -F 'states=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]' \
  -F 'image=@chest.png' \
  -F 'image=@left.png' \
  -F 'image=@right.png'
```

Expected response:

```json
{
  "response": [[... 14 floats ...], "... up to 50 actions ..."]
}
```
