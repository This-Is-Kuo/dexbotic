# Standalone Agent Dataset Debug Suite

This folder contains a framework-free harness for replaying LeRobot dataset
frames into a duck-typed agent. The runner imports public dependencies and the
agent file or module you provide; it does not depend on this repository's robot
control packages.

## Requirements

Install the standalone runner dependencies with:

```bash
python -m pip install -r agent_debug_suite/requirements.txt
```

The default video backend is `pyav`, so `av` is included.

## Agent Protocol

Your agent only needs:

```python
class MyAgent:
    def __init__(self, spec, buffer):
        ...

    def act(self, obs, task):
        return {"action.some_key": action_array}

    def teardown(self):
        ...
```

`teardown()` is optional. If your constructor does not accept `spec, buffer`,
the runner retries with only the keyword arguments from `--agent-kwargs-json`.

`obs` contains runtime-like values:

- `observation.*` entries from the dataset
- `teleoperated`, from the dataset or `np.array([0], np.int64)` when missing
- video observations converted to channel-first `uint8`

## Usage

Run the included zero-action sample:

```bash
python agent_debug_suite/debug_agent_from_dataset.py \
  --root datasets/jean_base_alignment_locked \
  --agent-file agent_debug_suite/sample_agent.py \
  --agent-class SampleAgent \
  --episode 0 \
  --num-frames 5
```

Run the same adapter against a real HTTP inference server:

```bash
python agent_debug_suite/debug_agent_from_dataset.py \
  --root datasets/jean_base_alignment_locked \
  --agent-file agent_debug_suite/sample_agent.py \
  --agent-class SampleAgent \
  --episode 0 \
  --num-frames 5 \
  --agent-kwargs-json '{"base_url":"http://127.0.0.1:7891","timeout_s":30}'
```

You can also pass the full route, matching the custom async bridge:

```bash
--agent-kwargs-json '{"vla_url":"http://127.0.0.1:7891/process_frame"}'
```

Load an installed or otherwise importable module:

```bash
python agent_debug_suite/debug_agent_from_dataset.py \
  --root datasets/jean_base_alignment_locked \
  --agent-module my_agent_package.policy \
  --agent-class MyAgent \
  --agent-kwargs-json '{"device": "cpu"}'
```

Useful flags:

- `--task "custom instruction"` overrides dataset task strings.
- `--print-actions` prints full action arrays instead of shape summaries.
- `--stop-on-error` exits after the first failed frame.
- `--video-backend pyav` selects a LeRobot video decoder backend. `pyav` is the default.

## HTTP Inference Protocol

The real DM0 service is the Flask route in `dexbotic/exp/dm0_exp.py`, and the
client behavior mirrors `hardware/xlerobot/custom_dm0_async_bridge.py`.

Endpoint:

```text
POST /process_frame
```

Multipart form fields:

- `text`: task prompt.
- `image`: repeated PNG files, in training camera order.
- `states`: optional JSON string for the raw robot state, usually 16D.
- `batch_size`: optional string, defaults to `1`.

The agent caches the returned action chunk locally. A request usually returns
`response` with shape `[50, 14]`; each `act()` call pops one action.

For online control, asynchronous prefetch is enabled by default. The agent
starts a background request when the queue reaches 30 actions or every 20
control steps, then applies the new chunk with `soft-replace`: keep 3 queued
actions and blend the first 5 new actions from the last command. These values
can be overridden through `--agent-kwargs-json`.

Response shape:

```json
{
  "response": [
    [0.0, 0.0, 0.0, 0.0]
  ]
}
```

The adapter auto-detects common image keys such as:

```text
images_1, images_2, images_3
image.chest, image.left, image.right
observation.images.chest_rgb, observation.images.left_wrist_rgb, observation.images.right_wrist_rgb
```

For Docker, use the address that is reachable from where this debug runner is
executed. If the policy server runs inside Docker with `-p 7891:7891`, use
`http://127.0.0.1:7891` from the host. If the debug runner runs in another
container on the same Docker network, use the policy container name, for example
`http://dm0-policy:7891`.

## Validation

For each selected frame, the runner checks that `act()` returns exactly the
dataset's `action.*` keys, with matching shapes, numeric values, and no NaNs or
infinities. It exits with a nonzero status if any frame fails.
