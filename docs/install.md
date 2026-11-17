---
title: Install
summary: Installing Komorebi from a clone against a tt-metal source tree, or from its two wheels on PyPI (komorebi-tt 0.2.0 and komorebi-runtime 0.81.1.4), and checking the host with kmr doctor.
tags:
  - overview
  - install
---

# Install

Komorebi 0.2.0 is on PyPI as `komorebi-tt` and `komorebi-runtime`, for one card model, the
Blackhole P150A: one card, or two joined by their Ethernet links ([two cards](#two-cards)).
Komorebi runs in two ways:

- **from a clone** against a tt-metal source tree you build ([from a clone](#from-a-clone),
  [running on a card](#running-on-a-card)): the way to develop Komorebi itself;
- **from the two wheels**, `komorebi-tt` and `komorebi-runtime`, with no tt-metal tree:
  installed from PyPI ([from 0.1.0: the two wheels](#from-010-the-two-wheels), which is
  also how 0.2.0 is installed) or built locally ([the runtime package](#the-runtime-package)).

What works is the Python package
(`kmr doctor`, `kmr build`, the layout helpers), the host API that runs a program on a
card under the dispatcher ([host API](reference/host.md)), the X280 threads, the
examples and the programs of `projects/`, and the tests.

## Requirements

- Linux on x86-64 and Python 3.13.
- To check out and test the code: `git`, and either [uv](https://docs.astral.sh/uv/) or
  plain `python3.13` with `venv`.
- To build for a card: a tt-metal source tree at the version in
  `packaging/tt-metal.pin`, built in `<tree>/build`, and `clang++-20` for C++ hosts.
- For X280 images: that tree built with its X280 runtime, which also provides the
  RISC-V compiler under `<tree>/runtime/sfpi`; or, without a tree, the
  `komorebi-runtime` wheel ([the runtime package](#the-runtime-package)).
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
package, the headers, the dispatcher's sources and its image, prebuilt against the
tree it is given (which needs the X280 runtime):

```
python3.13 packaging/build_wheel.py --tt-metal "$TT_METAL_HOME" --out dist
python3.13 -m venv .venv && source .venv/bin/activate
pip install dist/komorebi_tt-*.whl
kmr doctor
```

The wheel needs nothing else to install and to run `kmr doctor`. To run on a card it
needs what a clone needs ([running on a card](#running-on-a-card)): the tt-metal tree
with the X280 runtime in `TT_METAL_HOME`, and `clang++-20`; neither `ttnn` nor
`torch`. The native host module is built from the sources in the wheel on first use,
into `KMR_CACHE` (default `~/.cache/komorebi`). The examples are not in the wheel; a copy of `examples/`
runs from anywhere with the wheel's environment active, and finds the headers, the X280
library and the image in the install ([a run on a card](../packaging/README.md#a-run-on-a-card)).
The tt-metal inside the `kohakutt` wheel cannot serve: it is built without the X280
runtime. [Packaging](../packaging/README.md) lists what the wheel carries.

## The runtime package

`komorebi-runtime` (`packaging/runtime/` and `src/komorebi_runtime/` in this
repository) is the pinned tt-metal build as a platform wheel: the libraries, the kernel runtime tree, the X280 runtime objects,
linker script, headers and idle image, and a locator. With it and a `komorebi-tt` wheel
that carries the prebuilt native module, a program runs on a card with no tt-metal tree.
Both are built locally ([runtime](../packaging/runtime/README.md),
[packaging](../packaging/README.md#a-run-from-the-two-wheels)):

```
packaging/runtime/build_tt_metal.sh /path/to/tt-metal
python3.13 packaging/runtime/build_wheel.py --tt-metal /path/to/tt-metal --out dist
python3.13 packaging/build_wheel.py --tt-metal /path/to/tt-metal --out dist --host-module
pip install dist/komorebi_runtime-0.81.1.4-py3-none-manylinux_2_38_x86_64.whl \
    dist/komorebi_tt-0.2.0-cp312-abi3-manylinux_2_32_x86_64.whl
kmr setup                                      # the RISC-V compiler, then the card
kmr doctor                                     # "from komorebi-runtime 0.81.1.4"
```

The two wheels need `numpy` and nothing else: no `ttnn`, no `torch`. From PyPI the
install is `pip install "komorebi-tt[runtime]"` ([below](#from-010-the-two-wheels)).

When `TT_METAL_HOME` is unset and the package is installed, Komorebi takes its tree from
it: `kmr doctor`, `kmr build image`, the dispatcher's image and thread images use the
runtime's X280 pieces and compiler, a session activates the runtime
(`TT_METAL_RUNTIME_ROOT`) and loads the prebuilt native module.
`TT_METAL_HOME` set always means that tree, and the module is built against it. The
card is prepared once with `kmr setup` ([below](#setting-up-once)).

## From 0.1.0: the two wheels

From the 0.1.0 release on, both wheels are installed from PyPI, 0.2.0 among them. Wheels built as in
[the runtime package](#the-runtime-package) take the same steps, with the two files in
place of the `pip install` line. You need Linux on x86-64, Python 3.13 and a Blackhole
P150A; no tt-metal tree, no compiler of your own, neither `ttnn` nor `torch`.

```
python3.13 -m venv .venv && source .venv/bin/activate
pip install "komorebi-tt[runtime]"             # komorebi-tt, komorebi-runtime and numpy
kmr setup                                      # the RISC-V compiler, then the card
kmr doctor                                     # "from komorebi-runtime 0.81.1.4"
```

`pip install komorebi-tt` without `[runtime]` installs the framework alone, which
needs a tt-metal tree as from a clone. Leave `TT_METAL_HOME` unset: when it is set,
Komorebi uses that tree instead of the runtime package. The other extras are `dev`
(the checks of [development](development.md)) and `scenes` (pillow, only to decode the
public scenes' textures).

### What 0.2.0 asks for

- **The runtime wheel `komorebi-runtime` 0.81.1.4.** The extra `runtime` names
  `komorebi-runtime>=0.81.1.4,<0.81.2`, and `pip install -U "komorebi-tt[runtime]"` takes
  the newest build. Build 4 is the first with 1250, 1350 and 1500 MHz in the X280 clock
  table, and the session's clock is 1350 MHz unless it is given one
  ([host API](reference/host.md#session)). With build 3 (0.81.1.3, the runtime of 0.1.0)
  the session finds that 1350 is not offered, says so in its log, and runs the X280 at
  1000 MHz. `kmr doctor` says `from komorebi-runtime 0.81.1.4`.
- **The card's preparation is as above:** `kmr setup` once for each card.

### Two cards

A session over two cards, `km.Session(cards=[0, 1])` ([islands](reference/islands.md)),
needs two Blackhole P150A cards in one machine, joined by their Ethernet links (four, on
the pair it has run on), and `kmr setup --card N` for each card. The session sets
`TT_VISIBLE_DEVICES` to the cards it is given, and raises `ValueError` if the variable
already names others. Two cards have been run from a clone against a tt-metal tree
([from a clone](#from-a-clone), [running on a card](#running-on-a-card)); the wheels have
been run on one card.

### Setting up once

`kmr setup [--card N]` does the two one-time steps, each also a command of its own:

- `kmr sfpi` fetches the RISC-V compiler the runtime names into `KMR_CACHE` (default
  `~/.cache/komorebi`), which also holds the runtime root. A new cache directory needs
  it again; `kmr doctor` says `x280 compiler  not found` until then.
- `kmr prime` prepares the card's X280 memory (it primes the ECC of the X280 tiles'
  local memory) with the program the runtime package carries, then asks for one card
  reset (`tt-smi -r N`). The priming survives resets, so this is once per card. On a
  card prepared before, it changes nothing and says so. It opens the card, so nothing
  else may use the card while it runs.

The card is the N of `/dev/tenstorrent/N`: `--card N`, else the first of
`TT_VISIBLE_DEVICES`, else 0. Both commands need `komorebi-runtime` 0.81.1.3 or later;
without it they say so and stop.

### The examples and the tutorial's programs

The wheels carry the framework, not the programs. `examples/` and `tutorial/` come from
the repository at the release's tag, as a clone or as the source archive of the GitHub
release:

```
git clone --depth 1 --branch v0.2.0 <this repository> komorebi-src
cp -r komorebi-src/examples komorebi-src/tutorial .
cd tutorial/04_hello_core && python hello.py        # the command is done
```

A lesson's folder runs on its own from anywhere, with the wheels' environment active:
the kernels sit beside the program, and the headers, the dispatcher's image, the native
module and the runtime come from the install. So do the examples, except three programs
written for a clone, which import the card tests' feeder from `tests/hardware/feeder`
and put the clone's `src/` first on the path: `02_merge_tree/merge_fed.py`,
`03_broadcast` and `05_forward`; they run from a clone set up as in
[from a clone](#from-a-clone). Use the tag that matches
the installed `komorebi-tt` version (`kmr doctor` prints it): a program of a later
commit may need a later package.

## Building against a tt-metal tree

`kmr build` compiles outside the tree; nothing is written inside it.

```
kmr build image out/hello.bin hello.c                 # an X280 image (needs the X280 runtime)
kmr build host out/prog main.cpp -I include -D NAME=1 # a C++ host
kmr build pymodule out/mod.so module.cpp              # a Python module
```

`--tree` names the tree; the default is `TT_METAL_HOME`, else the runtime package. The
`host` and `pymodule` builds read their flags from tt-metal's own statements in the
tree's `build/build.ninja`, so they need a built tree (the runtime package carries none):
the compile flags (include directories, defines, language and warning flags) of the X280
runtime's library `x280_device`, which carry tt-metal's, less the warning tt-metal adds
for its own sources (`-Wunused-parameter`); the link libraries of the X280
runtime's priming test `x280_prime_test` (the runtime's static libraries, then
`libtt_metal.so` and the libraries it needs); for a module, nanobind's include
directories and archive from ttnn's `nanobind-static-abi3`. Both are tt-metal's targets
in a tree configured with `-DTT_METAL_BUILD_X280=ON`; on a tree without them the error
names that option. Building an X280 image needs no build statement.

## Running on a card

Running the host API on a card needs a tt-metal tree with the X280 runtime and a card
whose X280 memory is prepared once. Nothing outside tt-metal is needed. Step by step:

1. **tt-metal v0.81.1 with the X280 runtime** (the version in `packaging/tt-metal.pin`).
   `packaging/runtime/build_tt_metal.sh` clones the pinned release, applies this
   repository's patches (among them the X280 clock table the session's 1350 MHz needs),
   configures it with the X280 runtime and builds what Komorebi uses:

   ```
   packaging/runtime/build_tt_metal.sh $HOME/tt-metal   # KMR_JOBS=N limits the jobs
   cd $HOME/tt-metal
   ```

   By hand, the same without the patches (the session then falls back to 1000 MHz):

   ```
   git clone --branch v0.81.1 --recurse-submodules https://github.com/tenstorrent/tt-metal
   cd tt-metal
   ./build_metal.sh --build-metal-tests --configure-only
   cmake -B build -DTT_METAL_BUILD_X280=ON
   ninja -C build ttnn hw_toolchain x280/obj/entry_active.o x280/obj/x280_active_start.o \
       x280/firmware-lim-idle.bin x280_device x280_state x280_prime_test
   ```

   tt-metal accepts `TT_METAL_BUILD_X280` only with its tests configured
   (`--build-metal-tests`); none of them is built. The X280 runtime is the experimental
   one under `tt_metal/fabric/experimental/x280` (its README explains it).

2. **Prepare the card once.** Prime the X280 LIM ECC of the card, then reset it; the
   priming survives later resets, so this is once per card, not per run. With the
   tree's own program (`ninja -C build x280_prime_test` builds it):

   ```
   X280_ALLOW_LIM_ECC_PRIME=1 TT_VISIBLE_DEVICES=0 build/test/tt_metal/x280/x280_prime_test
   env -u TT_VISIBLE_DEVICES tt-smi -r 0
   ```

   or, with `komorebi-runtime` installed in the environment, `kmr prime --card 0`
   ([setting up once](#setting-up-once)), then the same reset.

3. **The Python environment.** `numpy` beside the package (neither `ttnn` nor `torch`
   is needed), and `TT_METAL_HOME` naming the tree:

   ```
   export TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD
   ```

4. **Build ahead (optional).** On first use the session builds the dispatcher image
   and the package's native module from the tree into `build/komorebi/` of the clone (or
   `KMR_CACHE`), about ten seconds:

   ```
   python -m komorebi.device.native        # prints the image and the module it built or found
   ```

5. **Run.** The session opens the card in slow dispatch:

   ```
   cd examples/01_mapcollect
   python mapcollect.py --jobs 20 --count 8
   ```

   The card tests (`tests/hardware`) run only with `KMR_CARD=1`:

   ```
   KMR_CARD=1 python -m pytest tests/hardware
   ```

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
from komorebi.bench.power import PowerSampler

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
`power.summarize(samples, start, stop, update_s, limit_w=None, lost_at=None)` gives the
same summary of samples `(time, watts, temperature)` a script took itself.

**The clock beside a time.** `power.aiclk_mhz(card)` reads the Tensix clock (AICLK)
from the driver (`/sys/class/tenstorrent/tenstorrent!N/tt_aiclk`); `card` None is
`power.visible_card()`, the first card `TT_VISIBLE_DEVICES` names (0 when unset). The clock is
800 MHz while no process holds the card. tt-metal's UMD raises it to the busy clock
(1,350 MHz on a p150a) when a process opens the card: `Cluster::start_device` sets the
busy state. It keeps that clock until the process exits, even after the session
closes. A reading taken before a session therefore says nothing about that session's
times. Every example's and project's driver puts the clock beside its times in its
JSON line as `aiclk_mhz`, read in the process that ran the session:

```python
from komorebi.bench import power

print(json.dumps(power.clocked(result)))  # adds "aiclk_mhz" unless already there
```
