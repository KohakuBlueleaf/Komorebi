# Packaging

Nothing is published: no package on PyPI, no GitHub release, no tags. The release
workflows build and test a wheel; their publish steps are off.

| File | What it does |
|---|---|
| `tt-metal.pin` | The tt-metal release Komorebi is developed and built against. |
| `build_wheel.py` | Builds the wheel from a commit (`--ref`, default `HEAD`), exported with `git archive` so nothing untracked comes along. It builds the dispatch kernel image against a tt-metal tree with the X280 runtime, puts it beside the package, and refuses a wheel that misses a header or a source. |
| `smoke.sh` | Installs a wheel alone into a fresh environment, runs `kmr doctor` outside the checkout, and checks that the headers and the prebuilt image are found inside the install; with `KMR_CARD=1` it also runs `examples/01_mapcollect`. |
| `version.py` | Reads, stamps and bumps the version in `pyproject.toml` and `komorebi.__version__`. |

## What the wheel carries

- the `komorebi` package and the `kmr` command;
- `komorebi/km-dispatch.bin`, the dispatch kernel image for X280 cluster 0, prebuilt;
- `komorebi_native/`: the ABI and device headers (`include/`), the dispatch kernel's
  sources and its host interface `km_dispatch.h` (`x280/kernel/`), the native host
  module's source (`csrc/host/`), and every file under `kernels/` and `cmake/` (both
  empty so far).

The wheel is pure Python (`py3-none-any`): the image is a RISC-V binary, and the native
host module links the tt-metal libraries of the user's own tree, so it is built there on
first use, not shipped.

## What a run needs

The wheel installs and `kmr doctor` runs with nothing else. Running a program on a card
needs, as for a clone ([install](../docs/install.md#running-on-a-card)):

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
bash packaging/smoke.sh dist/komorebi-*.whl /tmp/komorebi-smoke
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

## Workflows

| Workflow | Runs on | What it does |
|---|---|---|
| `ci.yml` | every push and pull request | format, lint, CPU tests, the version check |
| `release.yml` | a `v*` tag or by hand | on a build machine holding a tree with the X280 runtime (`KMR_TT_METAL`): builds the wheel and runs `smoke.sh`; its PyPI and GitHub release jobs are off |
| `nightly.yml` | by hand (its schedule is added at release time) | reports whether main has moved since the last release and the dev version; the step that would start `release.yml` is off |
| `auto-patch-release.yml` | by hand (its schedule is added at release time) | reports whether a patch release is due and its version; tagging and pushing are off |
