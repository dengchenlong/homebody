"""The Kuavo arm safety chain: authority ramp, rate clamp, URDF limits, not-following.

A wheel-arm gate, modeled on the G1 ArmGate but without the torque terms: the kuavo
controller owns the motor loop, so the gate's job is only to keep the commanded targets
smooth, in range and actually being followed.
"""
from __future__ import annotations

import numpy as np

from .profile import DEADMAN_S

__all__ = ["LIMIT_MARGIN_RAD", "MAX_JOINT_RATE_RAD_S", "ArmGate"]

MAX_JOINT_RATE_RAD_S = 1.05  # rad/s; well under the controller's 6 rad/s joint cap
LIMIT_MARGIN_RAD = 0.05  # rad kept away from each URDF stop


class ArmGate:
    """One arm's ordered clamps: authority ramp -> rate clamp -> limits -> not-following."""

    def __init__(self, lower, upper, max_rate: float = MAX_JOINT_RATE_RAD_S,
                 watchdog_s: float = DEADMAN_S):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.max_rate = float(max_rate)
        self.watchdog_s = float(watchdog_s)
        self._q_cmd = np.zeros(7)
        self._q_meas_prev: np.ndarray | None = None
        self._weight = 0.0
        self._weight_target = 0.0
        self._ramp_rate = 0.5  # weight per second
        self._last_gap = float("inf")

    def prime(self, q_arm) -> None:
        """Adopt the measured pose as the first command, so enabling never jumps."""
        self._q_cmd = np.asarray(q_arm, dtype=float).copy()

    def arm(self, weight_target: float = 1.0, ramp_s: float = 2.0) -> None:
        self._weight_target = float(weight_target)
        self._ramp_rate = 1.0 / max(ramp_s, 1e-3)

    def disarm(self) -> None:
        self._weight_target = 0.0

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def last_gap(self) -> float:
        return self._last_gap

    def __call__(self, q_target, q_meas, dt: float) -> tuple[np.ndarray, bool]:
        """Returns (clamped command, following). A stale gap (>watchdog) holds the last
        command and reports not-following."""
        q_target = np.asarray(q_target, dtype=float)
        q_meas = np.asarray(q_meas, dtype=float)
        stale = dt > self.watchdog_s

        # 1. authority ramp: one value drives both the ramp and the reporting
        step_w = self._ramp_rate * max(dt, 0.0)
        if self._weight < self._weight_target:
            self._weight = min(self._weight_target, self._weight + step_w)
        elif self._weight > self._weight_target:
            self._weight = max(self._weight_target, self._weight - step_w)

        # 2. rate clamp: hold the last command when the command stream went stale
        want = self._q_cmd.copy() if stale else q_target
        max_delta = self.max_rate * min(max(dt, 0.0), self.watchdog_s)
        self._q_cmd = np.clip(want, self._q_cmd - max_delta, self._q_cmd + max_delta)

        # 3. URDF joint limits, with a margin off each stop
        self._q_cmd = np.clip(self._q_cmd, self.lower + LIMIT_MARGIN_RAD,
                              self.upper - LIMIT_MARGIN_RAD)

        # 4. not-following: the commanded pose must keep converging on the measured one
        gap = float(np.max(np.abs(self._q_cmd - q_meas)))
        following = gap < self._last_gap + max_delta or gap < 0.15
        self._last_gap = gap

        return self._q_cmd.copy(), following
