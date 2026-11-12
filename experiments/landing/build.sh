#!/usr/bin/env bash
# build.sh: builds the X280 image and the host module into build/ (out of tree).
#   TT_METAL_TREE  a configured tt-metal source tree with the X280 runtime built
#                  (and the build targets that tt-x280-dispatch adds, see README)
#   TTX_DISPATCH   a tt-x280-dispatch checkout (for the host header)
# Needs `kmr` (pip install -e . in the repository root).
set -euo pipefail
: "${TT_METAL_TREE:?set TT_METAL_TREE}" "${TTX_DISPATCH:?set TTX_DISPATCH}"
here=$(cd "$(dirname "$0")" && pwd)
kmr build image "$here/build/landing-seq.bin" "$here/x280/landing_seq.c" \
    --tree "$TT_METAL_TREE"
kmr build pymodule "$here/build/_landing_x280.abi3.so" "$here/host/landing_x280.cpp" \
    --tree "$TT_METAL_TREE" -I "$TTX_DISPATCH/src/dispatcher/host"
