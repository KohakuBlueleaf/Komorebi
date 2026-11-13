#!/usr/bin/env bash
# The wheel in a fresh environment: installed alone, `kmr doctor`, the headers and the
# prebuilt image found inside the install; with KMR_CARD=1 also examples/01_mapcollect.
# usage: smoke.sh WHEEL VENV   (env: PYTHON, default python3.13; KMR_SITE, directories
# put on the environment's path; PIP_ARGS, extra pip flags; TT_VISIBLE_DEVICES, the card)
# With KMR_RUNTIME_WHEEL (the komorebi-runtime wheel) it is the run from the two wheels:
# both installed with the requirements the wheels name (numpy), TT_METAL_HOME unset and
# fresh caches (KMR_CACHE, TT_METAL_CACHE kept if set), the runtime's compiler fetched,
# `doctor` of both packages, and with KMR_CARD=1 the example, run from the install.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
wheel=$(realpath "${1:?wheel}")
venv=$(realpath -m "${2:?venv directory}")
runtime=${KMR_RUNTIME_WHEEL:+$(realpath "$KMR_RUNTIME_WHEEL")}
python=${PYTHON:-python3.13}
step() { echo "smoke: $* ($(date +%T))"; }

step "environment"
"$python" -m venv --clear "$venv"
site=$("$venv/bin/python" -c "import sysconfig; print(sysconfig.get_path('purelib'))")
if [ -n "${KMR_SITE:-}" ]; then
    tr ':' '\n' <<< "$KMR_SITE" > "$site/komorebi_smoke_site.pth"
fi
step "install"
if [ -n "$runtime" ]; then
    # The two wheels alone, then what they require from elsewhere. The requirement of
    # the komorebi-tt wheel on the runtime wheel is checked by wheel_info.py.
    info="$repo/packaging/ci/wheel_info.py"
    mapfile -t others < <(
        "$python" "$info" requires "$wheel"
        "$python" "$info" requires "$runtime"
    )
    # shellcheck disable=SC2086
    "$venv/bin/pip" install -q --disable-pip-version-check --no-cache-dir ${PIP_ARGS:-} --no-deps "$runtime" "$wheel"
    # shellcheck disable=SC2086
    "$venv/bin/pip" install -q --disable-pip-version-check --no-cache-dir ${PIP_ARGS:-} "${others[@]}"
else
    # shellcheck disable=SC2086
    "$venv/bin/pip" install -q --disable-pip-version-check --no-cache-dir ${PIP_ARGS:-} "$wheel"
fi

# Outside the repository, so nothing is imported from the checkout.
work=$(mktemp -d)
cd "$work"
if [ -n "$runtime" ]; then
    unset TT_METAL_HOME TT_METAL_RUNTIME_ROOT PYTHONPATH
    export KMR_CACHE=${KMR_CACHE:-$work/kmr-cache}
    export TT_METAL_CACHE=${TT_METAL_CACHE:-$work/tt-cache}
    step "the runtime's compiler and doctor"
    "$venv/bin/python" -m komorebi_runtime sfpi
    "$venv/bin/python" -m komorebi_runtime doctor
fi
step "kmr doctor and the install check"
"$venv/bin/kmr" doctor | tee "$work/kmr-doctor.txt"
if [ -n "$runtime" ]; then
    grep -Eq '^from +komorebi-runtime ' "$work/kmr-doctor.txt"
fi
"$venv/bin/python" - "$site" "${runtime:+runtime}" << 'EOF'
import sys
from pathlib import Path

from komorebi import _env, abi, native

site = Path(sys.argv[1]).resolve()
assert _env.package_dir().is_relative_to(site), _env.package_dir()
assert abi.headers().abi.is_relative_to(site), abi.headers().abi
image = native.image_path()
assert image == _env.package_dir() / native.IMAGE, image
print(f"headers and image from the install: {image} ({image.stat().st_size} bytes)")
if sys.argv[2]:
    import komorebi_runtime

    assert Path(komorebi_runtime.__file__).resolve().is_relative_to(site)
    module = _env.package_dir() / native.MODULE_FILE
    assert module.is_file(), module
    print(f"host module from the install: {module} ({module.stat().st_size} bytes)")
EOF
if [ "${KMR_CARD:-0}" = 1 ]; then
    step "examples/01_mapcollect"
    cd "$repo/examples/01_mapcollect"
    "$venv/bin/python" mapcollect.py --jobs 20 --count 8 --out-tensors 2
fi
step "done"
