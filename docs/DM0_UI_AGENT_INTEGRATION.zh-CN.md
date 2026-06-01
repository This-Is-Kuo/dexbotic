# DM0 推理服务 UI 对接说明

本文档说明当前 DM0 推理服务的数据格式、模型输入输出约定，以及如何对接外部 UI 端的 Agent 接口。

目标 UI 端接口如下：

```python
class SampleAgent:
    def __init__(self, spec=None, buffer=None) -> None:
        self.spec = spec or {}

    def act(self, obs: dict[str, np.ndarray], task: str) -> dict[str, np.ndarray]:
        ...

    def teardown(self) -> None:
        pass
```

## 当前模型版本

当前自定义 DM0 入口是：

```text
playground/post_data_01_dm0_deltafix.py
```

相关运行时默认配置：

```text
图像数量: 3
原始 state 维度: 16
原始 action 维度: 14
模型内部 action 维度: 32
action chunk 长度: 50
推理服务默认端口: 7891
```

当前本地检查过的训练数据路径是：

```text
data/post_origin_data_0423_dexdata_train/jsonl
```

数据统计：

```text
episode 文件数: 493
总帧数: 281610
每帧图像视角数: 3
state 维度: 16
action 维度: 14
```

每一帧都有以下字段：

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

## 原始数据字段

每一行 jsonl 对应一个时间步。

示例结构如下。这里的 `state` 和 `action` 数组做了省略，实际长度分别是 16 和 14。

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

三路图像含义：

| 数据字段 | 相机含义 | 原始视频分辨率 |
|---|---|---:|
| `images_1` | chest / 全局相机 | `1280x720` |
| `images_2` | left wrist / 左腕相机 | `1280x720` |
| `images_3` | right wrist / 右腕相机 | `1280x720` |

推理时，图片会被转换为 RGB PIL 图像，然后经过模型自带 image processor。当前 `DM0-base` 使用的视觉塔是 `pe_lang_l14_728`，单张图像预处理后的 shape 是：

```text
[3, 728, 728]
```

三路图像组成的模型输入 shape 是：

```text
[batch, 3, 3, 728, 728]
```

含义分别是：

```text
[batch, num_images, channels, height, width]
```

## State 格式

原始 `state` 是 16 维：

```text
left_tcp(7) + right_tcp(7) + left_pinch(1) + right_pinch(1)
```

索引布局：

| 索引范围 | 含义 | 维度 |
|---:|---|---:|
| `0:7` | 左手 TCP 绝对位姿 | 7 |
| `7:14` | 右手 TCP 绝对位姿 | 7 |
| `14` | 左手 pinch 当前状态 | 1 |
| `15` | 右手 pinch 当前状态 | 1 |

TCP 位姿格式预期是：

```text
x, y, z, qx, qy, qz, qw
```

也就是位置 3 维 + 四元数姿态 4 维。

训练和推理时，`state` 会被 pad 到模型内部 action 维度：

```text
原始 state: [16]
模型输入 state: [32]
```

UI 端如果能拿到机器人当前状态，应传原始 16D `state`。推理服务内部会自动 pad 到 32D。

如果不传 `state`，当前服务会填一个 `[batch, 32]` 的零向量。代码支持这种情况，但不建议用于当前模型，因为训练时模型是带 state 条件训练的。

## Action 格式

原始训练 `action` 是 14 维：

```text
left_delta_tcp(6) + left_pinch_target(1) + right_delta_tcp(6) + right_pinch_target(1)
```

索引布局：

| 索引范围 | 含义 | 维度 |
|---:|---|---:|
| `0:6` | 左手 TCP delta 控制量 | 6 |
| `6` | 左手 pinch 目标值 | 1 |
| `7:13` | 右手 TCP delta 控制量 | 6 |
| `13` | 右手 pinch 目标值 | 1 |

每个 6D TCP delta 预期格式是：

```text
dx, dy, dz, droll, dpitch, dyaw
```

其中 pinch 维度不是 delta，而是直接目标值：

```text
non_delta_mask = [6, 13]
```

训练时，action 会先 pad 到 32 维，再构造成 50 步 chunk：

```text
原始单步 action: [14]
pad 后单步 action: [32]
训练目标 action chunk: [50, 32]
```

模型推理时一次预测 50 步 action chunk。推理服务最终只返回前 14 维：

```text
服务返回 action shape: [50, 14]
```

需要注意：在 `playground/post_data_01_dm0_deltafix.py` 当前推理路径中，输出会经过：

```text
ActionDenorm -> AbsoluteAction -> 截取前 14 维
```

因此，当前 HTTP 服务返回的不是最原始训练 jsonl 中的纯 delta action，而是经过反归一化和 `AbsoluteAction` 后处理后的 command。`6` 和 `13` 两个 pinch 维度由于在 `non_delta_mask` 中，会保持为模型预测的直接目标值。

如果 UI 端或机器人执行器期望的是训练数据里的纯 14D delta action，需要二选一：

1. 修改推理输出 transform，去掉 `AbsoluteAction`；
2. 在 UI adapter 里把当前服务返回值转换成执行器需要的语义。

不要在 UI 端无条件再做一次 delta-to-absolute 转换。否则可能会重复转换，导致动作错误。

## HTTP 推理服务接口

接口地址：

```text
POST /process_frame
```

默认 base URL：

```text
http://localhost:7891
```

表单字段：

| 字段 | 类型 | 是否必需 | 说明 |
|---|---|---|---|
| `text` | string | 是 | 任务描述，对应数据里的 `prompt` 语义。 |
| `image` | repeated multipart file | 是 | 一张或多张 RGB 图片。当前模型期望 3 张。 |
| `states` | JSON string | 推荐 | 状态向量 `[16]`，或 batch 状态 `[[16], ...]`。 |
| `batch_size` | int string | 可选 | 默认是 `1`。 |

图片上传顺序必须和训练一致：

```text
image[0] = chest / 全局相机
image[1] = left wrist / 左腕相机
image[2] = right wrist / 右腕相机
```

如果传入图片少于 3 张，服务会自动用零图像补齐，并设置对应 image mask。代码支持这个逻辑，但除非已经验证过缺视角效果，否则不建议这么用。

返回格式：

```json
{
  "response": [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  ]
}
```

单个请求正常会返回 50 个 14D action：

```text
response: [50, 14]
```

推荐客户端行为：

1. 本地 action 队列为空时，请求一次 `/process_frame`。
2. 缓存返回的 50 步 action chunk。
3. 每次 `Agent.act(...)` 返回其中一帧 14D action。
4. 队列为空或任务切换时，重新请求服务。

## UI Agent 映射建议

UI 端调用：

```python
obs: dict[str, np.ndarray]
task: str
```

推荐观测字段：

| UI obs key | Shape | Dtype | 是否必需 | 映射 |
|---|---:|---|---|---|
| `images_1` 或 `image.chest` | `[H, W, 3]` | `uint8` | 是 | chest 图像 |
| `images_2` 或 `image.left` | `[H, W, 3]` | `uint8` | 是 | left wrist 图像 |
| `images_3` 或 `image.right` | `[H, W, 3]` | `uint8` | 是 | right wrist 图像 |
| `state` | `[16]` | float | 推荐 | 原始 16D state |

如果 UI 端能使用数据集风格命名，推荐：

```text
obs["images_1"]
obs["images_2"]
obs["images_3"]
obs["state"]
```

如果 UI 端使用相机名，推荐：

```text
obs["image.chest"]
obs["image.left"]
obs["image.right"]
obs["state"]
```

adapter 应返回符合 UI `spec` 的 dict。对当前模型，主 action key 推荐是一个 14D 向量：

```python
{
    "action": np.ndarray(shape=(14,), dtype=np.float32)
}
```

如果 UI 端的 `spec` 使用拆分 action key，可以按下面方式映射：

| 输出 key | 对应切片 |
|---|---|
| `action.left_delta_tcp` | `action[0:6]` |
| `action.left_pinch` | `action[6:7]` |
| `action.right_delta_tcp` | `action[7:13]` |
| `action.right_pinch` | `action[13:14]` |

## Adapter 示例

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

## 对接前检查项

对接 UI 前建议确认：

1. 推理服务已在预期端口启动。
2. UI 端按 `chest, left wrist, right wrist` 顺序传图。
3. 图片是 RGB 数组，shape 为 `[H, W, 3]`。
4. 如果可用，UI 端传入原始 16D `state`。
5. UI `spec` 支持 `action` shape `[14]`，或支持四个拆分 action key。
6. 执行器期望的 action 语义和当前服务返回语义一致。
7. 除非执行器明确需要，否则 UI 端不要额外做 delta-to-absolute 转换。

## 最小 curl 示例

```bash
curl -X POST http://localhost:7891/process_frame \
  -F 'text=express_pick' \
  -F 'states=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]' \
  -F 'image=@chest.png' \
  -F 'image=@left.png' \
  -F 'image=@right.png'
```

预期返回：

```json
{
  "response": [[... 14 floats ...], "... up to 50 actions ..."]
}
```
