"""The M1 smoke demo, run inside the kuavo controller container against the MuJoCo sim:

    roslaunch humanoid_controllers load_kuavo_mujoco_sim_wheel.launch   # terminal 1
    python -m homebody.backends.kuavo.demo                              # terminal 2

Five steps, Ctrl+C any time: read 20 joint states, switch BASE_ARM + external control,
move both arms to a preset pose and wait by reach time, drive 0.2 m/s for 2 s, stop.
"""
from __future__ import annotations

import signal
import sys
import time

import numpy as np

from .backend import KuavoBackend
from .joints import from_sensor_radians
from .profile import (
    CONTROL_HZ,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
)

# A reachable two-sided preset, radians, inside every URDF stop with margin.
PRESET_LEFT = np.array([0.0, 0.3, 0.0, -0.6, 0.0, -0.3, 0.0])
PRESET_RIGHT = np.array([0.0, 0.3, 0.0, -0.6, 0.0, -0.3, 0.0])
DRIVE_SPEED_MPS = 0.2
DRIVE_SECONDS = 2.0


def banner(step: str) -> None:
    print(f"\n=== {step} ===", flush=True)


def wait_reach(backend, joint_names, timeout_s: float = 8.0) -> None:
    """Sleep the Ruckig plan's longest reach time, once the port reports it."""
    port = backend._port
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        times = [port.arm_reach_time(name) for name in joint_names]
        if all(t is not None and t > 0.0 for t in times):
            time.sleep(max(times))
            return
        time.sleep(0.05)
    time.sleep(3.0)  # reach topics missing: fall back to a fixed settle


def main() -> int:
    backend = KuavoBackend()
    stopping = False

    def halt(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, halt)
    epoch = backend.new_epoch()
    try:
        banner("① joint states")
        deadline = time.monotonic() + 5.0
        while backend._port.sensors() is None and time.monotonic() < deadline and not stopping:
            time.sleep(0.05)
        sensors = backend._port.sensors()
        if sensors is None:
            print("no /sensors_data_raw within 5 s; is the controller up?", file=sys.stderr)
            return 1
        arms = from_sensor_radians(sensors["joint_q"])
        print(f"20 joints ok; left arm = {np.round(arms['left'], 3).tolist()}")
        print(f"               right arm = {np.round(arms['right'], 3).tolist()}")

        banner("② BASE_ARM + external control")
        backend._port.watch_reach(LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS)
        backend.ensure_arm_control()
        print("modes switched")

        banner("③ both arms to the preset")
        for side, gate in backend._gates.items():
            gate.prime(np.asarray(arms[side]))
            gate.arm(ramp_s=2.0)
        for tick in range(4 * CONTROL_HZ):
            if stopping:
                break
            backend.arm_target(epoch, "left", PRESET_LEFT)
            backend.arm_target(epoch, "right", PRESET_RIGHT)
            time.sleep(1.0 / CONTROL_HZ)
        wait_reach(backend, LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS)
        arms = from_sensor_radians(backend._port.sensors()["joint_q"])
        print(f"left now = {np.round(arms['left'], 3).tolist()}")
        print(f"right now = {np.round(arms['right'], 3).tolist()}")

        banner("④ drive forward 0.2 m/s for 2 s")
        end = time.monotonic() + DRIVE_SECONDS
        while time.monotonic() < end and not stopping:
            backend.base_velocity(epoch, DRIVE_SPEED_MPS, 0.0, 0.0)
            time.sleep(0.1)
        backend.stop(epoch)
        print("stopped")

        banner("⑤ done — Ctrl+C was the emergency stop throughout")
        return 0
    finally:
        backend.stop(epoch)
        backend.close()
        print("\nbase zeroed, gates disarmed, port closed")


if __name__ == "__main__":
    sys.exit(main())
