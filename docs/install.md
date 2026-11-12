---
title: Install
summary: Building Komorebi from a clone against a tt-metal source tree, and checking the host with kmr doctor. Nothing is released yet.
tags:
  - overview
  - install
---

# Install

Nothing is released. There is no package on PyPI, no wheel and no prebuilt X280 image
yet: Komorebi can only be used from a clone of this repository, and most of it is not
written. What works today is the Python package with `kmr doctor`, `kmr build`, the
layout helpers and the tests.

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

## Checks

```
python -m pytest tests/core
```

See [development](development.md#checks) for the full list.
