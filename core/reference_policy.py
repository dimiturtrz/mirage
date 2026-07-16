"""A trivial reference policy — proves the `policy.py` boundary is real and runnable before any sim.

`ConstantActionPolicy` emits a fixed action for every observation; `train` is a no-op that returns a
stateless state. It is the control-leg counterpart of a stub `Method` closure: the cheapest object that
still *satisfies* `ControlPolicy`, so the rollout boundary can be exercised (and contract-tested) with
no environment, no data, and no training. `closure()` returns the same behaviour in `Policy` (train, act)
closure form, so both the class and the closure demonstrably satisfy the one protocol.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from jaxtyping import Shaped

from core.policy import Policy


class ConstantActionPolicy:
    """Emits `action` for any observation; the smallest object satisfying `ControlPolicy`. Being constant,
    it discards both the state and the observation (the `_`-prefixed args mark that intentional non-use)."""

    def __init__(self, action: Shaped[np.ndarray, "*act"]):
        self._action = np.asarray(action)

    def train(self, _task: str) -> Any:
        return None

    def act(self, _state: Any, _obs: Shaped[np.ndarray, "*obs"]) -> Shaped[np.ndarray, "*act"]:
        return self._action

    def closure(self) -> Policy:
        return Policy(train=self.train, act=self.act)
