#!/usr/bin/env python3
"""M2's software-only acceptance check: the joint-limit gate refuses over-limit targets.

The robot never moves — nothing is published on /kuavo_arm_traj. Safe to run any time
the controller is up. Verifies the M2 acceptance item "deliberately over-limit → clip
refusal takes effect" without touching hardware.

Run:  python3 tools/m2_limit_clip_test.py
"""
from __future__ import annotations

import sys

import numpy as np

from homebody.backends.kuavo.backend import KuavoBackend
from homebody.backends.kuavo.limits import arm_limits
from homebody.backends.kuavo.profile import urdf_path
from homebody.primitives.actions import ActionRejected


def main() -> int:
    limits = arm_limits(urdf_path())
    backend = KuavoBackend(urdf=urdf_path())
    epoch = backend.new_epoch()

    passed, failed = 0, 0
    for side in ("left", "right"):
        lower, upper = limits[side]
        maxes = np.maximum(np.abs(lower), np.abs(upper))
        for index, limit in enumerate(maxes):
            target = np.zeros(7)
            target[index] = limit + 1.0  # one radian past the URDF stop
            try:
                backend.arm_target(epoch, side, target)
                print(f"FAIL {side} joint {index + 1}: over-limit target was accepted")
                failed += 1
            except ActionRejected as exc:
                print(f"OK   {side} joint {index + 1}: refused ({exc})")
                passed += 1

    # The gate must also refuse stale-epoch commands after a stop.
    old = epoch
    backend.stop(epoch)
    backend.new_epoch()
    try:
        backend.arm_target(old, "left", np.zeros(7))
        print("FAIL stale epoch accepted after new_epoch")
        failed += 1
    except ActionRejected as exc:
        print(f"OK   stale epoch refused ({exc})")
        passed += 1

    backend.close()
    print(f"\n{passed} passed, {failed} failed; nothing was published to the robot")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
