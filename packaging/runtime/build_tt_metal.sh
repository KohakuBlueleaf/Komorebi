#!/usr/bin/env bash
# Clones the pinned tt-metal (packaging/tt-metal.pin) into DIR, applies patches/*.patch of
# this folder, configures it with the X280 runtime and without distributed, and builds what
# the komorebi-runtime wheel carries: the libraries, ttnn, the kernel toolchain, the X280
# runtime objects and idle image, the X280 priming program, and the X280 host runtime that the
# native host module links.
# usage: build_tt_metal.sh DIR   (an existing build in DIR is updated; KMR_JOBS=N limits
# ninja's jobs; KMR_CCACHE=1 compiles through ccache, with CCACHE_DIR from the caller)
# No system packages are installed: the host needs tt-metal's build dependencies already
# (its install_dependencies.sh lists them).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dir=$(realpath -m "${1:?usage: build_tt_metal.sh DIR}")
pin=$(tr -d '[:space:]' < "$here/../tt-metal.pin")

# Cloned next to DIR, then moved: an interrupted clone leaves no DIR behind.
if [ ! -d "$dir/.git" ]; then
    if [ -e "$dir.clone" ]; then
        echo "$dir.clone exists (an interrupted clone?): remove it first"
        exit 1
    fi
    git clone --depth 1 --branch "$pin" --recurse-submodules --shallow-submodules \
        https://github.com/tenstorrent/tt-metal.git "$dir.clone"
    mv "$dir.clone" "$dir"
fi
have=$(git -C "$dir" describe --tags --always)
[ "$have" = "$pin" ] || { echo "$dir holds tt-metal $have, the pin is $pin"; exit 1; }

# Each patch once: skipped when it is already applied (it then applies in reverse).
for patch in "$here"/patches/*.patch; do
    [ -e "$patch" ] || continue
    if git -C "$dir" apply --check "$patch" 2> /dev/null; then
        git -C "$dir" apply "$patch"
        echo "applied $(basename "$patch")"
    elif ! git -C "$dir" apply --reverse --check "$patch" 2> /dev/null; then
        echo "$patch applies to $dir neither forward nor in reverse"
        exit 1
    fi
done

# Without multi-host support: OpenMPI cannot be carried inside a wheel. The X280 runtime
# needs the metal tests configured (its CMake rejects otherwise); none of them is built.
if [ ! -f "$dir/build/build.ninja" ]; then
    options=(--release --without-distributed --build-metal-tests --configure-only)
    if [ "${KMR_CCACHE:-0}" = 1 ]; then
        options+=(--enable-ccache)
    fi
    (cd "$dir" && ./build_metal.sh "${options[@]}")
fi
# Paths under DIR are written relative to it (__FILE__, std::source_location, debug
# information), so the libraries carry no path of this machine.
map="-ffile-prefix-map=$dir=."
cmake -B "$dir/build" -S "$dir" -DTT_METAL_BUILD_X280=ON \
    -DCMAKE_C_FLAGS="$map" -DCMAKE_CXX_FLAGS="$map"
grep -q '^TT_METAL_BUILD_X280:BOOL=ON' "$dir/build/CMakeCache.txt" ||
    { echo "$dir/build is configured without the X280 runtime"; exit 1; }

# Only what the wheel carries, plus the static X280 host runtime the host module links.
targets=(ttnn hw_toolchain x280/obj/entry_active.o x280/obj/x280_active_start.o)
targets+=(x280/firmware-lim-idle.bin x280_device x280_state x280_prime_test)
ninja -C "$dir/build" ${KMR_JOBS:+-j "$KMR_JOBS"} "${targets[@]}"
ls -la "$dir/build/tt_metal/libtt_metal.so" "$dir/build/ttnn/_ttnncpp.so" \
    "$dir/build/ttnn/_ttnn.so" "$dir/build/x280/obj/entry_active.o" \
    "$dir/build/test/tt_metal/x280/x280_prime_test"
