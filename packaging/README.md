# Packaging

Nothing is published yet: no package on PyPI, no GitHub release, no tags. The release
workflow builds and tests both wheels, and publishes them only on a `v*` tag push or
when it is started by hand with `publish: true` ([the release workflow](#the-release-workflow)).

| File | What it does |
|---|---|
| `tt-metal.pin` | A link to `runtime/tt-metal.pin`, the tt-metal release Komorebi is developed and built against. |
| `build_wheel.py` | Builds the `komorebi-tt` wheel (the package it carries is imported as `komorebi`) from a commit (`--ref`, default `HEAD`), exported with `git archive` so nothing untracked comes along. It builds the dispatch kernel image against a tt-metal tree with the X280 runtime, puts it beside the package, and refuses a wheel that misses a header or a source. |
| `smoke.sh` | Installs a wheel alone into a fresh environment, runs `kmr doctor` outside the checkout, and checks that the headers and the prebuilt image are found inside the install; with `KMR_CARD=1` it also runs `examples/01_mapcollect`. |
| `ci/` | What the release workflow runs: `build_wheels.sh` (tt-metal, then both wheels), `wheel_info.py` (names, versions and requirements read from the wheels, the version checks), `check_paths.py` (no binary names a path of the build machine). |
| `version.py` | Reads, stamps and bumps the version in `pyproject.toml` and `komorebi.__version__`. |

The tt-metal build itself is a second package of this repository, `komorebi-runtime`
([runtime](../runtime/README.md)): a platform wheel with the pinned tt-metal libraries,
the kernel runtime tree, the X280 runtime pieces and a locator.

## What the wheel carries

- the `komorebi` package and the `kmr` command;
- `komorebi/km-dispatch.bin`, the dispatch kernel image for X280 cluster 0, prebuilt;
- `komorebi_native/`: the ABI and device headers (`include/`), the dispatch kernel's
  sources and its host interface `km_dispatch.h` (`x280/kernel/`), the native host
  module's source (`csrc/host/`), its CMake file (`cmake/km_host.cmake`), and every
  file under `kernels/` (empty so far);
- with `--host-module`: `komorebi/_km_host.abi3.so`, the native host module prebuilt
  for `komorebi-runtime`.

Without `--host-module` the wheel is pure Python (`py3-none-any`): the image is a
RISC-V binary, and the native host module is built against the user's own tree on
first use. With it, the module is built in the tt-metal build `komorebi-runtime` is
made from (`--tt-metal` names that tree), by `cmake/km_host.cmake` with tt-metal's own
targets: the tree is reconfigured with the file as `CMAKE_PROJECT_Metalium_INCLUDE`,
`ninja _km_host` runs, and the configuration is restored. The module's rpath is
`$ORIGIN/../komorebi_runtime/lib`, so it loads the runtime package's libraries, the
same files ttnn's module loads, when both wheels are installed in one environment. The
wheel is then tagged `cp312-abi3-manylinux_<floor>_x86_64`, the floor being the newest
`GLIBC_` symbol version the module requires (2.32). The prebuilt module is used only
when the tree is the runtime package's: with `TT_METAL_HOME` set, the module is built
against that tree as before.

## What a run needs

The wheel installs and `kmr doctor` runs with nothing else. A wheel with the module and
`komorebi-runtime` run a program on a card with no tree
([a run from the two wheels](#a-run-from-the-two-wheels)). Otherwise a run needs, as
for a clone ([install](../docs/install.md#running-on-a-card)):

1. a tt-metal tree at the pinned version, built with the X280 runtime
   (`-DTT_METAL_BUILD_X280=ON`), with the tt-x280-dispatch targets hooked in, named by
   `TT_METAL_HOME`. The tt-metal inside the `kohakutt` wheel does not do: it is built
   without the X280 runtime;
2. `ttnn` from that tree and `torch` in the environment the wheel is installed in;
3. `clang++-20`: the native host module is built once into `KMR_CACHE` (default
   `~/.cache/komorebi`) the first time a session loads, about ten seconds;
4. the card's X280 memory prepared once.

The prebuilt image is linked with the X280 runtime objects of the tree it was built
against, so it matches a tree at the same pin.

## Building and checking a wheel

```
python3.13 packaging/build_wheel.py --tt-metal "$TT_METAL_HOME" --out dist
bash packaging/smoke.sh dist/komorebi_tt-*.whl /tmp/komorebi-smoke
# the wheel for komorebi-runtime, in the tree runtime/build_tt_metal.sh built
python3.13 packaging/build_wheel.py --tt-metal /path/to/tt-metal --out dist --host-module
```

`build_wheel.py` builds with `--no-build-isolation` by default (the running Python
needs setuptools 77 or later); `--isolated` lets pip fetch the build backend instead.
`smoke.sh` takes `PYTHON` (default `python3.13`), `KMR_SITE` (directories added to the
environment's path) and `PIP_ARGS` (extra pip flags). `ttnn` from a source tree imports
only with three of the tree's directories on the path, `ttnn`, the tree itself and
`tools` (which holds `tracy`); `torch` and the packages `ttnn` imports come from an
environment that has them, such as the one the tree was built with:

```
export KMR_SITE=$TT_METAL_HOME/ttnn:$TT_METAL_HOME:$TT_METAL_HOME/tools:<site-packages with torch>
```

`KMR_CARD=1` runs the example on a card; the card is the first of `TT_VISIBLE_DEVICES`
(default 0). The script makes its working directory with `mktemp -d`, under `TMPDIR`.

### A run on a card

What was run, on a Blackhole p150a, from a fresh clone, with a fresh environment and
empty caches (`KMR_CACHE`, `TT_METAL_CACHE`), twice in a row with the same result:

```
python3.13 packaging/build_wheel.py --tt-metal "$TT_METAL_HOME" --out dist
KMR_CARD=1 TT_VISIBLE_DEVICES=1 TT_METAL_SLOW_DISPATCH_MODE=1 \
    bash packaging/smoke.sh dist/komorebi_tt-0.1.0-py3-none-any.whl smoke-venv
```

| Step | Time |
|---|---|
| the wheel (234 kB, 69 files, the image 22,864 bytes) | 3.9 s |
| `smoke.sh`: environment, install, `kmr doctor`, the install check, `examples/01_mapcollect` | 29 s and 30 s |
| of which the example (native module built, device opened, kernels compiled cold, 20 jobs, outputs read back) | 20.9 s and 24.3 s |

The example finished with 20 jobs checked, none wrong, 120 cores ended. Then, with the
same environment, a copy of `examples/` outside any checkout:

| Run | Result | Wall |
|---|---|---|
| `python work_cluster.py --paths thread` in `07_work_cluster` | 204 rounds, sum 1,004,917, same as the numpy replay, the thread 10.2 ms on the card | 6.9 s |
| `python graph.py` in `06_graph` | both paths ended after 5 and 7 rounds, state exact, paths agree | 7.9 s |

### A run from the two wheels

On a Blackhole p150a, 2026-11-13, card 0: `komorebi-runtime` 0.81.1.1 (50.5 MB) and the
`komorebi-tt` wheel with the module (632 kB, 77 files, `_km_host` 974,097 bytes), built
from one tt-metal build, installed with `torch` 2.15.0 (CPU) into a fresh environment;
`TT_METAL_HOME` unset, fresh `KMR_CACHE` and `TT_METAL_CACHE`, slow dispatch:

```
python3.13 runtime/build_wheel.py --tt-metal TREE --out dist
python3.13 packaging/build_wheel.py --tt-metal TREE --out dist --host-module
uv venv -p 3.13 venv && uv pip install -p venv/bin/python \
    "dist/komorebi_runtime-0.81.1.1-py3-none-manylinux_2_38_x86_64.whl[ttnn]" \
    dist/komorebi_tt-0.1.0-cp312-abi3-manylinux_2_32_x86_64.whl torch==2.15.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu
venv/bin/python -m komorebi_runtime sfpi
eval "$(venv/bin/python -m komorebi_runtime env)"   # the examples import ttnn first
cd examples/01_mapcollect && ../../venv/bin/python mapcollect.py --jobs 20 --count 8 --out-tensors 2
cd ../07_work_cluster && ../../venv/bin/python work_cluster.py
```

| Run | Result | Wall |
|---|---|---|
| `mapcollect.py` | 20 jobs checked (10 exact), none wrong, 120 cores ended, kernels compiled cold from the runtime's tree | 23.0 s |
| `work_cluster.py` | the CPU, host-fed and X280-thread paths: 204 rounds, sum 1,004,917, all three the same as the numpy replay; the thread 10.82 ms on the card | 6.0 s |

The process maps one copy of `libtt_metal.so`, `libtt-umd.so.0` and `_ttnncpp.so`, from
`komorebi_runtime/lib`, for both ttnn's module and `_km_host`.

## The release workflow

`release.yml` builds the two wheels, `komorebi-tt` and `komorebi-runtime`, tests them and,
on the owner's trigger, publishes them. Every step of the `build` job is a script of
this folder, so each runs by hand the same way.

What the build runner needs:

- tt-metal's build dependencies (its `install_dependencies.sh --no-distributed`
  list: clang 20, CMake, Ninja and the system libraries), installed beforehand: the
  build step installs nothing;
- `python3.13` with setuptools 77 or later, `git`, and `patchelf`;
- network access for the tt-metal clone, its CMake package fetches and the sfpi it
  downloads at configure time, and for the smoke test's `pip` (the requirements the
  wheels name, `torch`) and the runtime's compiler;
- a persistent directory for the tt-metal tree (about 6 GB), named by `KMR_RUNTIME_TREE`
  in the runner's environment, and optionally ccache (`KMR_CCACHE=1`, `CCACHE_DIR`) and
  `KMR_JOBS`;
- for the smoke run on a card: a Blackhole card whose X280 memory is prepared, with
  `KMR_CARD=1`, `TT_VISIBLE_DEVICES` and `TT_METAL_SLOW_DISPATCH_MODE=1` in the
  runner's environment.

The `build` job:

1. Checks out the commit; with the `version` input it stamps that version; for a tag it
   stops when the tag does not name the version.
2. `ci/build_wheels.sh "$KMR_RUNTIME_TREE" dist`: `runtime/build_tt_metal.sh` (it
   applies `runtime/patches/` and configures with `-ffile-prefix-map`; the tree is
   rebuilt only when `runtime/tt-metal.pin` or a patch changes, otherwise it is updated in
   seconds), `runtime/build_wheel.py` and `build_wheel.py --host-module --isolated`.
   A new runtime wheel is only worth publishing when the pin, a patch or `runtime/BUILD`
   changed.
3. `ci/wheel_info.py identify dist` reads both names, versions and file names from the
   wheels (the workflow writes no distribution name); `versions` checks that a tag names
   the `komorebi-tt` version and that its `runtime` extra admits the runtime wheel.
4. `ci/check_paths.py dist/*.whl`: every ELF file, object, archive and image of both
   wheels is searched for the home directory, the runner's workspace, work and temporary
   directories and `KMR_RUNTIME_TREE`; one hit fails the job.
5. `smoke.sh` with `KMR_RUNTIME_WHEEL`: a fresh environment, both wheels installed
   with `--no-deps` and then what they require from elsewhere (`wheel_info.py requires`:
   `numpy`, the runtime's `ttnn` extra, `torch`), `TT_METAL_HOME` unset and fresh
   caches. It fetches the compiler (`komorebi_runtime sfpi`), runs
   `komorebi_runtime doctor` and `kmr doctor` (which must say `from komorebi-runtime`),
   checks that the headers, the image and the host module are found inside the install
   and, with `KMR_CARD=1`, applies the `komorebi_runtime env` lines and runs
   `examples/01_mapcollect`. The install does not ask pip for the project's own
   distributions: the two wheels are the files just built.
6. Both wheels are uploaded as one artifact.

Publishing, in three jobs that run only on a `v*` tag push or a dispatch with
`publish: true` (default false): `publish-pypi` uploads the `komorebi-tt` wheel and
`publish-pypi-runtime` the runtime wheel, each through trusted publishing (no token, no
secret) with its own environment (`pypi`, `pypi-runtime`), because PyPI does not accept
one workflow and environment for two projects; the runtime job skips a file whose
version is already on the index. `publish-github` then creates or updates the release and
attaches both wheels. Each PyPI project needs this workflow set up as its trusted
publisher before the first release.

`nightly.yml` and `auto-patch-release.yml` start by hand only. No workflow here has a
schedule, and `tests/core/test_release.py` checks that, the gate of the publish jobs and
that they name no secret.

### What was run

The workflow itself has not run: the repository is not on GitHub yet. Its scripts were
run in the order of the `build` job, on 2026-11-13, against the tree
`runtime/build_tt_metal.sh` had built (the incremental path: no change to the pin or the
patches), with the runner's variables set by hand, and `smoke.sh` on card 1 of a
Blackhole p150a with slow dispatch, through the card watcher:

| Step | Result | Time |
|---|---|---|
| `ci/build_wheels.sh TREE dist` | `komorebi_runtime-0.81.1.1-py3-none-manylinux_2_38_x86_64.whl` (50.5 MB) and `komorebi_tt-0.1.0-cp312-abi3-manylinux_2_32_x86_64.whl` (642 kB, 78 files, `_km_host` 974,097 bytes) | 45 s |
| `ci/wheel_info.py identify dist` | `komorebi-tt` 0.1.0, `komorebi-runtime` 0.81.1.1, requirement `>=0.81.1.1,<0.81.2`, not a prerelease | under 1 s |
| `ci/wheel_info.py versions dist --tag v0.1.0` and `--tag v0.1.1` | passes; fails with `tag v0.1.1 does not name version 0.1.0` | under 1 s |
| `ci/check_paths.py dist/*.whl` (home, workspace, work, temporary and tree directories) | 26 and 2 binaries, none names one; with `/usr` forbidden it finds 5 places in `libtt_metal.so` | 1 s |
| `smoke.sh` without a card | environment, install, both `doctor`s, the install check | 58 s |
| `smoke.sh` with `KMR_CARD=1` | the same, then `examples/01_mapcollect`: 20 jobs, no error | 85 s |

## Workflows

| Workflow | Runs on | What it does |
|---|---|---|
| `ci.yml` | every push and pull request | format, lint, CPU tests, the version check |
| `release.yml` | a `v*` tag or by hand | on the build machine: builds both wheels, checks them, runs the smoke test from both; its three publish jobs run only on a tag or with `publish: true` |
| `nightly.yml` | by hand | reports whether main has moved since the last release and the dev version; the step that would start `release.yml` is off |
| `auto-patch-release.yml` | by hand | reports whether a patch release is due and its version; tagging and pushing are off |
