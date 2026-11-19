#!/usr/bin/env bash
# The laptop's one-shot environment check for the M2 real-robot smoke.
# Everything green before any robot script runs. Never starts robot services.
set -u
FAIL=0

ok()   { printf '  OK      %s\n' "$1"; }
bad()  { printf '  MISSING %s\n' "$1"; FAIL=1; }
warn() { printf '  WARN    %s\n' "$1"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== python =="
if command -v python3 >/dev/null; then
    PYVER=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    ok "python3 $PYVER"
    case "$PYVER" in 3.1[0-9]|3.[89]) ;; *) bad "python 3.8+ required, found $PYVER";; esac
else
    bad "python3"; echo "  install: sudo apt install python3 python3-venv python3-pip"; exit 1
fi
python3 -c 'import numpy' 2>/dev/null && ok "numpy" || bad "numpy (pip install numpy)"
python3 -c 'import scipy' 2>/dev/null && ok "scipy" || bad "scipy (pip install scipy)"

echo "== ros =="
if [ -n "${ROS_DISTRO:-}" ]; then
    ok "ROS_DISTRO=$ROS_DISTRO"
else
    if [ -f /opt/ros/noetic/setup.bash ]; then
        # shellcheck disable=SC1091
        source /opt/ros/noetic/setup.bash
        ok "sourced /opt/ros/noetic"
    else
        bad "no ROS environment (sudo apt install ros-noetic-ros-base, then source /opt/ros/noetic/setup.bash)"
    fi
fi

echo "== kuavo_msgs (from the robot over the wire, or a local devel copy) =="
KUAVO_PKGS="${KUAVO_DEVEL:-$HOME/robot_ws_devel/lib/python3/dist-packages}"
if python3 -c "import kuavo_msgs" 2>/dev/null; then
    ok "kuavo_msgs importable"
else
    if [ -d "$KUAVO_PKGS" ]; then
        export PYTHONPATH="$KUAVO_PKGS:$PYTHONPATH"
        python3 -c "import kuavo_msgs" 2>/dev/null \
            && ok "kuavo_msgs via $KUAVO_PKGS" \
            || bad "kuavo_msgs not importable from $KUAVO_PKGS"
    else
        bad "kuavo_msgs: mount the robot's devel tree first"
        echo "          sshfs robot@<ROBOT_IP>:<path-to-kuavo_ws>/devel ~/robot_ws_devel"
        echo "          or copy it:  scp -r robot@<ROBOT_IP>:<path-to-kuavo_ws>/devel/lib/python3/dist-packages ~/robot_ws_devel/lib/python3/dist-packages"
    fi
fi

echo "== the wire (robot must be powered and standing; these only read) =="
MASTER="${ROS_MASTER_URI:-http://localhost:11311}"
ROBOT_IP="$(printf '%s' "$MASTER" | sed -n 's|http://\([0-9.]*\):.*|\1|p')"
if [ -n "$ROBOT_IP" ] && ping -c 1 -W 2 "$ROBOT_IP" >/dev/null 2>&1; then
    ok "ping $ROBOT_IP"
    RTT=$(ping -c 3 -W 2 "$ROBOT_IP" 2>/dev/null | tail -1 | grep -oE '[0-9.]+/[0-9.]+/[0-9.]+' | cut -d/ -f2)
    warn "rtt avg ${RTT:-?} ms (wired should be < 1 ms; > 20 ms means run on the robot PC, not from here)"
else
    warn "cannot ping $ROBOT_IP — set ROS_MASTER_URI=http://<ROBOT_IP>:11311 once the robot is up"
fi
if command -v rostopic >/dev/null && timeout 6 rostopic list >/dev/null 2>&1; then
    ok "ros master reachable"
    for topic in /sensors_data_raw /odom /humanoid/mpc/arm_control_mode; do
        if timeout 6 rostopic list | grep -qx "$topic"; then
            ok "topic $topic"
        else
            bad "topic $topic absent — is the controller up?"
        fi
    done
    HZ=$(timeout 6 rostopic hz /sensors_data_raw 2>/dev/null | grep -oE 'average rate: [0-9.]+' | grep -oE '[0-9.]+')
    if [ -n "$HZ" ]; then
        ok "/sensors_data_raw at ${HZ} Hz"
        python3 -c "import sys; sys.exit(0 if float('$HZ') > 50 else 1)" \
            || bad "sensor rate ${HZ} Hz is too low over this link (< 50 Hz): move the run to the robot PC"
    else
        bad "no /sensors_data_raw rate within 6 s"
    fi
else
    warn "ros master not up yet — re-run this script after the controller starts"
fi

echo
if [ "$FAIL" -eq 0 ]; then
    echo "ALL GREEN — safe to run:  python3 tools/m2_limit_clip_test.py   (software-only, robot will not move)"
    echo "                then:      python3 tools/m2_demo_speed_half.py    (motion; read m2_checklist.md first)"
else
    echo "NOT READY — fix the MISSING lines above, re-run."
fi
exit "$FAIL"
