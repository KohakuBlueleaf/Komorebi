#!/usr/bin/env bash
# Builds the two wheels into DIST: the tt-metal tree at the pin with the X280 runtime
# (updated in seconds when it is already built), the komorebi-runtime wheel, and the
# komorebi-tt wheel with the prebuilt native host module, both from that one tree.
# usage: build_wheels.sh TREE DIST   (env: PYTHON, default python3.13; PATCHELF, default
# patchelf; KMR_JOBS and KMR_CCACHE for build_tt_metal.sh; KMR_SKIP_TREE=1 to leave the
# tree as it is)
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
tree=$(realpath -m "${1:?usage: build_wheels.sh TREE DIST}")
dist=$(realpath -m "${2:?usage: build_wheels.sh TREE DIST}")
python=${PYTHON:-python3.13}
patchelf=${PATCHELF:-patchelf}

mkdir -p "$dist"
if [ "${KMR_SKIP_TREE:-0}" != 1 ]; then
    bash "$repo/packaging/runtime/build_tt_metal.sh" "$tree"
fi
"$python" "$repo/packaging/runtime/build_wheel.py" --tt-metal "$tree" --out "$dist" \
    --patchelf "$patchelf"
"$python" "$repo/packaging/build_wheel.py" --tt-metal "$tree" --out "$dist" \
    --host-module --isolated --patchelf "$patchelf"
ls -l "$dist"
