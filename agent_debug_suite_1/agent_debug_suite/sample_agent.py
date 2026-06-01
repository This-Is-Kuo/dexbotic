#!/usr/bin/env python3
"""DM0 compatibility entrypoint for debug_agent_from_dataset.py.

The runner/client can keep loading ``SampleAgent`` from this file, while the
real DM0 agent implementation lives in ``dm0_agent.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from dm0_agent import DM0Agent


class SampleAgent:
    """Thin adapter that keeps the original SampleAgent import path stable."""

    def __init__(
        self,
        spec: dict[str, dict[str, Any]] | None = None,
        buffer: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.agent = DM0Agent(spec=spec, buffer=buffer, **kwargs)

    def act(self, obs: dict[str, np.ndarray], task: str) -> dict[str, np.ndarray]:
        return self.agent.act(obs, task)

    def teardown(self) -> None:
        teardown = getattr(self.agent, "teardown", None)
        if callable(teardown):
            teardown()
