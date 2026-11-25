#!/usr/bin/env python3
"""M2 on-site demo, half speed: the M1 five steps against the real robot.

Read m2_checklist.md first. Ctrl+C at any moment is the emergency stop (zeroes the
base immediately; the controller's deadman does the rest). DRIVE_SPEED is halved to
0.1 m/s and the arm preset is applied one arm at a time.

Run:  python3 tools/m2_demo_speed_half.py
"""
from __future__ import annotations

import signal
import sys
import time

import numpy as np

from homebody.backends.kuavo.backend import KuavoBackend
from homebody.backends.kuavo.profile import urdf_path

# --- on-site dials: slower than the M1 sim demo in every dimension
DRIVE_SPEED_MPS = 0.1     # M1 sim used 0.2; halved for the first real motion
DRIVE_SECONDS = 1.0       # M1 sim used 2; one metre of wall time, stop and reassess
ARM_STEP_HZ = 50          # M1 sim ran the arm stream at 100; gentler on the wire

# A mild two-sided preset, radians, well inside every URDF stop.
PRESET = np.array([0.0, 0.15, 0.0, -0.3, 0.0, -0.15, 0.0])

stopping = False


def halt(signum, frame):
    global stopping
    stopping = True
    print("\n!! emergency stop: zeroing the base", flush=True)


def banner(step: str) -> None:
    print(f"\n=== {step} ===", flush=True)


def main() -> int:
    backend = KuavoBackend(urdf=urdf_path())
    signal.signal(signal.SIGINT, halt)
    epoch = backend.new_epoch()
    try:
        banner("① joint states (read-only)")
        deadline = time.monotonic() + 5.0
        while backend._port.sensors() is None and time.monotonic() < deadline and not stopping:
            time.sleep(0.05)
        sensors = backend._port.sensors()
        if sensors is None:
            print("no /sensors_data_raw within 5 s; is the controller up?", file=sys.stderr)
            return 1
        left = np.round(sensors["joint_q"][4:11], 3).tolist()
        right = np.round(sensors["joint_q"][11:18], 3).tolist()
        print(f"20 joints ok; left arm = {left}\n             right arm = {right}")
        banner("② BASE_ARM + external control")
        backend._port.watch_reach(tuple(f"zarm_{s}{i}_joint" for s in ("l", "r") for i in range(1, 8)))
        backend.ensure_arm_control()
        print("modes switched (verify with: rostopic echo /mobile_manipulator/lb_mpc_control_mode)")

        for side in ("left", "right"):
            banner(f"③{side[0]} {side} arm to the preset (one arm at a time)")
            lo, hi = (4, 11) if side == "left" else (11, 18)
            backend._gates[side].prime(np.asarray(sensors["joint_q"][lo:hi]))
            backend._gates[side].arm(ramp_s=3.0)
            for _ in range(4 * ARM_STEP_HZ):
                if stopping:
                    break
                backend.arm_target(epoch, side, PRESET)
                time.sleep(1.0 / ARM_STEP_HZ)
            if stopping:
                break
            # the controller's own plan duration when it reports one, else a fixed settle
            settle = backend._port.arm_settle_time()
            time.sleep(settle if settle and settle > 0.0 else 3.0)
            now = backend._measured_arm(side)
            print(f"{side} now = {np.round(now, 3).tolist()}")
        if stopping:
            return 130

        banner(f"④ drive forward {DRIVE_SPEED_MPS} m/s for {DRIVE_SECONDS} s — WALK ALONGSIDE, hands on the stop")
        end = time.monotonic() + DRIVE_SECONDS
        while time.monotonic() < end and not stopping:
            backend.base_velocity(epoch, DRIVE_SPEED_MPS, 0.0, 0.0)
            time.sleep(0.1)
        backend.stop(epoch)
        print("stopped")

        banner("⑤ done — stop published; Ctrl+C was live throughout")
        return 0
    finally:
        backend.stop(epoch)
        backend.close()
        print("\nbase zeroed, gates disarmed, port closed")


if __name__ == "__main__":
    sys.exit(main())
