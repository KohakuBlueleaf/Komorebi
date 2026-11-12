# Komorebi

A programming model and runtime for the Tensix grid of Tenstorrent Blackhole cards.

A program is written as straight-line threads. A thread has a program counter and a
place: it stays on its core until it needs state held on another core, then it moves
there or sends a value there. A resident worker on every Tensix core runs the threads;
a small kernel on the card's X280 cores decides what to run.

## Status

The repository is being set up. Nothing here is released, and no part of the runtime is
implemented yet. What exists:

- the Python package `komorebi` (version 0.1.0): `kmr doctor` to inspect a host,
  `kmr build` to build a host program, a Python module or an X280 image against a
  tt-metal tree, pure layout functions for the core grid, its panels and teams, and
  statistics helpers;
- the layout of the native layer, as empty directories with a README;
- the design, as three concept pages: [the programming model](docs/concepts/model.md),
  [the dispatch kernel](docs/concepts/dispatch.md) and
  [the Tensix runtime](docs/concepts/runtime.md).

## Map

```text
src/komorebi/            the Python package (kmr, layout, build, stats, tools)
src/komorebi_native/     the native layer: device and X280 headers, kernels, host module
examples/  projects/     runnable examples; larger programs built on Komorebi
experiments/  ref/       standalone proofs; CPU references in numpy
tests/core/              tests that need no card
docs/                    install, development and the concept pages
packaging/               the tt-metal version Komorebi is developed against
```

Start with [docs/README.md](docs/README.md). To build from a clone:
[docs/install.md](docs/install.md).

## Checks

Python 3.13:

```
uv venv -p 3.13 && source .venv/bin/activate && uv pip install -e ".[dev]"
# or, without uv:
python3.13 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

python -m pytest tests/core
ruff check . && black --check .
python -m komorebi.tools.cxx_format --check
```

The formatter tests need clang-format (the `dev` extra installs it) and are skipped
without it.

`kmr doctor` reports what the host offers (Python, the tt-metal tree from
`TT_METAL_HOME`, its build and X280 runtime) without opening a device.

## Licence

Apache-2.0; see [LICENSE](LICENSE).
