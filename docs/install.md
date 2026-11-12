---
title: Install
summary: Installing Komorebi from a clone or from a wheel built locally, against a tt-metal source tree, and checking the host with kmr doctor. Nothing is published yet.
tags:
  - overview
  - install
---

# Install

Nothing is published: there is no package on PyPI. Komorebi is used from a clone of
this repository, or from a wheel built from one ([from a wheel](#from-a-wheel)). What
works is the Python package
(`kmr doctor`, `kmr build`, the layout helpers), the host API that runs a program on a
card under the dispatch kernel ([host API](reference/host.md)), the example
`examples/01_mapcollect`, and the tests.

## Requirements

- Linux on x86-64 and Python 3.13.
- To check out and test the code: `git`, and either [uv](https://docs.astral.sh/uv/) or
  plain `python3.13` with `venv`.
- To build for a card: a tt-metal source tree at the version in
  `packaging/tt-metal.pin`, built in `<tree>/build`, and `clang++-20` for C++ hosts.
- For X280 images: that tree built with its X280 runtime, which also provides the
  RISC-V compiler under `<tree>/runtime/sfpi`.
- A Blackhole card with its kernel driver and firmware, to run anything on hardware.

None of this is needed to import the package or run the CPU tests.

## From a clone

```
git clone <this repository> komorebi && cd komorebi

# with uv
uv venv -p 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"

# or with plain Python 3.13
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

export TT_METAL_HOME=/path/to/tt-metal      # optional; kmr doctor and kmr build read it
kmr doctor
```

Run the later commands in the activated environment (`source .venv/bin/activate` in
each new shell).

`kmr doctor` prints the Python in use, the package and include directories, the
tt-metal tree and build directory it found (or that it found none), whether that build
has the X280 runtime, and the X280 compiler if it can locate one. It opens no device,
and exits with status 0 whether or not a tree is set.

## From a wheel

`packaging/build_wheel.py` builds a wheel from a commit of a clone. It carries the
package, the headers, the dispatch kernel's sources and its image, prebuilt against the
tree it is given (which needs the X280 runtime):

```
python3.13 packaging/build_wheel.py --tt-metal "$TT_METAL_HOME" --out dist
python3.13 -m venv .venv && source .venv/bin/activate
pip install dist/komorebi-*.whl
kmr doctor
```

The wheel needs nothing else to install and to run `kmr doctor`. To run on a card it
needs what a clone needs ([running on a card](#running-on-a-card)): the tt-metal tree
with the X280 runtime and the tt-x280-dispatch targets in `TT_METAL_HOME`, `ttnn` from
that tree and `torch` in the same environment, and `clang++-20`. The native host module
is built from the sources in the wheel on first use, into `KMR_CACHE` (default
`~/.cache/komorebi`). The examples are not in the wheel; run them from a clone with the
wheel's environment active. The tt-metal inside the `kohakutt` wheel cannot serve: it
is built without the X280 runtime. [Packaging](../packaging/README.md) lists what the
wheel carries.

## Building against a tt-metal tree

`kmr build` compiles outside the tree; nothing is written inside it.

```
kmr build image out/hello.bin hello.c                 # an X280 image (needs the X280 runtime)
kmr build host out/prog main.cpp -I include -D NAME=1 # a C++ host
kmr build pymodule out/mod.so module.cpp              # a Python module
```

`--tree` names the tree; the default is `TT_METAL_HOME`. The `host` and `pymodule`
builds copy their compile and link flags from two targets that the tt-x280-dispatch
project adds to a tt-metal build, so they need a tree built with them; see the
limitation in [development](development.md#known-limitations). Building an X280 image does not.

## Running on a card

Running the host API on a card needs a tt-metal tree with the X280 runtime, the
[tt-x280-dispatch](https://github.com/KohakuBlueleaf/tt-x280-dispatch) project hooked
into it, and a card whose X280 memory is prepared once. Step by step:

1. **tt-metal v0.81.1 with the X280 runtime** (the version in `packaging/tt-metal.pin`):

   ```
   git clone --branch v0.81.1 https://github.com/tenstorrent/tt-metal && cd tt-metal
   ./build_metal.sh --configure-only && cmake -B build -DTT_METAL_BUILD_X280=ON
   ```

   The X280 runtime is the experimental one under
   `tt_metal/fabric/experimental/x280` (its README explains it).

2. **The companion project's targets.** Its `install.sh` adds a three-line CMake hook to
   the tree and builds, among others, the `ttx_bench` and `_ttx280` targets whose flags
   `kmr build host` and `kmr build pymodule` copy:

   ```
   git clone https://github.com/KohakuBlueleaf/tt-x280-dispatch
   tt-x280-dispatch/install.sh $PWD
   ```

3. **Prepare the card once.** Prime the X280 LIM ECC of the card, then reset it; the
   priming survives later resets, so this is once per card, not per run:

   ```
   X280_ALLOW_LIM_ECC_PRIME=1 TT_VISIBLE_DEVICES=0 build/test/tt_metal/x280/x280_prime_test
   env -u TT_VISIBLE_DEVICES tt-smi -r 0
   ```

4. **The Python environment.** `ttnn` from that tree and `torch` in the same environment
   as the package, and `TT_METAL_HOME` naming the tree:

   ```
   export TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD
   ```

5. **Build ahead (optional).** On first use the session builds the dispatch kernel image
   and the package's native module from the tree into `build/komorebi/` of the clone (or
   `KMR_CACHE`), about ten seconds:

   ```
   python -m komorebi.native        # prints the image and the module it built or found
   ```

6. **Run.** The session opens the card in slow dispatch:

   ```
   cd examples/01_mapcollect
   python mapcollect.py --jobs 20 --count 8
   ```

   The card tests (`tests/hardware`) run only with `KMR_CARD=1`:

   ```
   KMR_CARD=1 python -m pytest tests/hardware
   ```

What does not work without the companion project: the package's native module is
built with the flags of its `_ttx280` target, so without step 2 `python -m
komorebi.native` builds the dispatch kernel image but stops at the module, and no
session can run. `kmr build image` and everything on the CPU work without it. This
dependency ends when Komorebi has its own build files
([known limitations](development.md#known-limitations)).

## Checks

```
python -m pytest tests/core
```

See [development](development.md#checks) for the full list.

## Watching power

`kmr power` prints the board power of every card once a second, then a summary of the
run, so you can watch a benchmark from a second shell. It reads sensors only: it opens
no card and takes no lock, and needs no root.

```
kmr power                          # every card, 10 seconds
kmr power --card 1 --seconds 60    # one card (the N of /dev/tenstorrent/N)
kmr power --gpu 0 --cpu            # a GPU through nvidia-smi; the CPU package
```

A card's power comes from its `blackhole` hwmon device, found through the card's PCI
device link. The CPU package comes from the powercap energy counter when this user may
read it; otherwise, like a missing `nvidia-smi`, it is reported as not available.

The same sampler wraps a section of a program:

```python
from komorebi.power import PowerSampler

with PowerSampler(1) as sampler:  # card index 1; one second of idle samples first
    sampler.mark("fill")
    ...
    sampler.mark("compute")
    ...
print(sampler.result.format())
print(sampler.result.phases["compute"].mean_w)
```

The result has seconds, samples, mean, peak and energy (trapezoid), the mean
temperature, the card's power limit and the idle mean from before the block. The
sensor of a card updates every 0.1 s, so the default period is 0.05 s, and a section
shorter than five updates (0.5 s) says so and reports no mean. A sensor that disappears
ends the sampling; the result covers the time before it and says it was lost.
