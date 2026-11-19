#!/usr/bin/env bash
# Pack the M2 on-site bundle: everything the laptop needs for the real-robot smoke.
# Run on the dev server:  bash tools/kuavo_robot_pack.sh [output.tar.gz]
# Then move the tarball to the laptop (scp/U盘) and untar it anywhere.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/kuavo_m2_bundle.tar.gz}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# The release source, minus the heavy simulator bits a robot-side run never imports.
rsync -a --exclude '.venv' --exclude '__pycache__' --exclude 'runs' --exclude '.git' \
      --exclude '.codegraph' --exclude '.pytest_cache' --exclude '.ruff_cache' \
      --exclude 'temp' --exclude 'evaluation' --exclude '*.tar.gz' \
      --exclude 'hardware' --exclude 'assets/real2sim' --exclude 'assets/models' \
      --exclude 'assets/robot/g1_grasp' --exclude 'assets/robot/g1_viz' \
      --exclude 'assets/robot/g1_*.urdf' --exclude 'assets/robot/g1_body*' \
      --exclude 'vendor' --exclude 'docs' --exclude '.github' \
      --exclude 'tools/m2_bundle' --exclude '.vscode' --exclude '.agents' \
      --exclude 'src/homebody.egg-info' \
      "$ROOT/" "$STAGE/homebody-main/"

# The on-site scripts live under tools/m2_bundle/ in the bundle's homebody copy.
cp "$ROOT/tools/m2_bundle/m2_setup.sh" "$STAGE/homebody-main/tools/"
cp "$ROOT/tools/m2_bundle/m2_demo_speed_half.py" "$STAGE/homebody-main/tools/"
cp "$ROOT/tools/m2_bundle/m2_limit_clip_test.py" "$STAGE/homebody-main/tools/"
cp "$ROOT/tools/m2_bundle/m2_checklist.md" "$STAGE/homebody-main/tools/"

tar -czf "$OUT" -C "$STAGE" homebody-main
echo "packed: $OUT ($(du -h "$OUT" | cut -f1))"
echo "next: move it to the laptop, untar, then:  bash homebody-main/tools/m2_setup.sh"
