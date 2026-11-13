#!/usr/bin/env bash
# The wheel in a fresh environment: installed alone, `kmr doctor`, the headers and the
# prebuilt image found inside the install; with KMR_CARD=1 also examples/01_mapcollect.
# usage: smoke.sh WHEEL VENV   (env: PYTHON, default python3.13; KMR_SITE, directories
# put on the environment's path: the tree's ttnn, the tree, its tools, and torch's
# site-packages; PIP_ARGS, extra pip flags; TT_VISIBLE_DEVICES, the card)
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
wheel=$(realpath "${1:?wheel}")
venv=$(realpath -m "${2:?venv directory}")

"${PYTHON:-python3.13}" -m venv --clear "$venv"
site=$("$venv/bin/python" -c "import sysconfig; print(sysconfig.get_path('purelib'))")
if [ -n "${KMR_SITE:-}" ]; then
    tr ':' '\n' <<< "$KMR_SITE" > "$site/komorebi_smoke_site.pth"
fi
# shellcheck disable=SC2086
"$venv/bin/pip" install -q --no-cache-dir ${PIP_ARGS:-} "$wheel"

# Outside the repository, so nothing is imported from the checkout.
cd "$(mktemp -d)"
"$venv/bin/kmr" doctor
"$venv/bin/python" - "$site" << 'EOF'
import sys
from pathlib import Path

from komorebi import _env, abi, native

site = Path(sys.argv[1]).resolve()
assert _env.package_dir().is_relative_to(site), _env.package_dir()
assert abi.headers().abi.is_relative_to(site), abi.headers().abi
image = native.image_path()
assert image == _env.package_dir() / native.IMAGE, image
print(f"headers and image from the install: {image} ({image.stat().st_size} bytes)")
EOF
if [ "${KMR_CARD:-0}" = 1 ]; then
    cd "$repo/examples/01_mapcollect"
    "$venv/bin/python" mapcollect.py --jobs 20 --count 8 --out-tensors 2
fi
