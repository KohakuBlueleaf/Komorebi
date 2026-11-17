---
title: Development
summary: Repository layout, the checks to run before a change, what the tests cover, the code conventions and the known limitations.
tags:
  - development
---

# Development

Building from a clone: [install](install.md#from-a-clone).

## Repository layout

```text
src/komorebi/                  the package you import
  _env.py                      finds the tt-metal tree, its build and the X280 toolchain
  cli.py                       the `kmr` command: doctor, setup, sfpi, prime, build, power
  layout.py                    the grid, its panels and core groups (pure functions)
  errors.py  formats.py        errors; bfloat16 and tiles on the host
  abi.py  merge.py             constants read from the C headers; programs merged into one
  program/                     a program:
    description.py             Program
    several.py  overlay.py     several programs in one run; overlays: the builder and the slots
    movement.py                both data-movement CPUs on one stream
    thread.py                  X280 threads: declaration, arguments, regions, results
    addon/                     code added to a live run: added handlers, an added class's program
    checks/                    the instruction check of a kernel; the page calls
  session/                     the session on one card:
    session.py                 Session (loads a program under the dispatcher)
    run.py  state.py           Run (submit, wait, stop); result, state and trace decoding
    page_bounds.py             a class's command checked at submit against its tensors' pages
  device/                      the device side on the host:
    channel.py                 the host channel
    tensor.py  dram.py         device tensors; interleaved DRAM buffers, bank by bank
    native.py                  the image and native module
    extents.py                 the page bound on a card with no tensor table
    kernel_buffer.py           the kernel buffer that follows the built kernels
    remembered.py              that buffer's size, remembered in the build cache
  chain/                       a chain's calls and the records they become
  islands/                     islands:
    graph.py                   the graph, ranks, the island table, placement (host values)
    session.py                 the session over several cards (or one card split into islands)
    link_area.py  trees.py     the link line and area; collectives over the islands' trees
  builder/                     out-of-tree builds (flags.py: host program, Python module, X280
                               image) and the prebuild (prebuild.py)
  bench/                       power.py: power sampling (cards, a GPU, the CPU package);
                               stats.py: timing statistics and error against a reference
  engines/                     micro-op streams and graphs: builders, upload encoding, checks,
                               simulators, the host-fed executors
  tools/cxx_format.py          clang-format with the rules it cannot express
  manifest/                    reserved: a program's commands and handlers declared (README inside, no code yet)
src/komorebi_native/           the native layer (README inside)
src/komorebi_runtime/          komorebi-runtime's locator: where the runtime's files are
examples/  projects/           runnable examples; larger programs (README inside)
experiments/  reference/       standalone proofs; reference implementations (README inside)
tests/core/                    tests that need no card
tests/projects/                CPU tests of the programs under projects/
tests/ref/                     reference renderers compared (README inside)
docs/                          concepts, install, this page
packaging/                     the komorebi-tt wheel: build, smoke test, version tools; the pin
packaging/runtime/             komorebi-runtime: its patches and build (README inside)
```

`import komorebi` needs no tt-metal tree and imports no tt-metal or TT-NN module; the
package boundary test enforces it. Nothing runs at import and no device is opened
except by code that is asked to.

## Checks

```
python -m pytest tests/core                     # CPU tests
python -m pytest tests/projects                 # the programs' CPU tests
KMR_SIZES=1 python -m pytest tests/projects/sizes/test_kernel_sizes.py   # kernel sizes
python -m komorebi.tools.cxx_format --check
python -m komorebi.tools.comment_lint --check src tests examples experiments packaging tutorial projects
python -m komorebi.tools.module_checks --check     # statements a file, cycles, imports in functions
ruff check . && black --check . && blacken-docs -t py313 -l 88 $(git ls-files '*.md')
NO_MKDOCS_2_WARNING=1 mkdocs build --strict -d /tmp/komorebi-site   # the documentation site, tutorial included
```

The site is the reading surface of the [tutorial](tutorial/index.md): MkDocs with the Material
theme, configured in `mkdocs.yml`; `mkdocs serve` shows it on localhost. Its code listings are
included from the programs in `tutorial/` when it is built, so a listing cannot differ
from the program that ran, and `.github/workflows/docs.yml` deploys it to GitHub Pages
when started by hand.

The comment check leaves out `reference/`, whose code keeps its longer docstrings. The
`dev` extra installs the tools, including clang-format 22 or later (the style file
`.clang-format` uses options older versions reject). Run the commands from the
repository, in the environment of [install](install.md#from-a-clone).

`tests/core` covers the layout functions (panels, groups, partitions), statistics, the
package boundary, `kmr doctor` with and without a tree, the runtime package's locator
on a fake layout and the order in which the tree is chosen, the parsing and command
assembly of `builder/flags.py` against a small fake `build.ninja`, the power sampler against
fake sensors and a fake sysfs tree, and the formatter. Tests
marked `card` need a Tenstorrent card and are skipped unless `KMR_CARD=1` (they live in
`tests/hardware`). The host package's CPU tests run `Run` against a fake dispatcher
(`tests/core/fake_port.py`). The formatter tests are skipped when clang-format is not installed.

The reference tests in `tests/core/ref` (`test_ref_*.py`) cover the scene generator and
the numpy rasterizer and ray tracer with the box tree. The comparisons of the C++ and
CUDA versions with numpy, and a slow rasterizer case, are in `tests/ref`, run by hand
([README](../tests/ref/README.md)); the CUDA ones only with `KMR_GPU=1`.

The CPU tests of a program under `projects/` live in
`tests/projects/test_<project>_*.py` (a folder of their own for a program with several,
as `raster/` and `raytrace/`), not in `tests/core`, which is the framework's
suite. They run as a step of their own, with a budget of 5 s for all programs
together (about 1 s each).

`tests/projects/sizes/test_kernel_sizes.py` checks that every program's kernels fit the
session's kernel buffer with 1 KiB to spare, without a card: it builds the kernels on
a mock device (tt-metal's Blackhole P150 cluster descriptor) with the runtime's
compiler, so it needs `TT_METAL_HOME` set to a built tree and `clang++-20`, builds the
native module and compiles each program (a few minutes the first time), and runs only
with `KMR_SIZES=1`. The continuous-integration workflow has no tree and skips it.

The continuous-integration workflow runs ruff, black, the CPU tests and the
programs' CPU tests. It counts their skips and fails when the count is not the one
kept in the workflow (`SKIPS` of each step), so a test that stops running is seen and
a planned skip is a one-line change.

Some tests in `tests/core` need a tt-metal tree and its X280 compiler (the class
threads' C, the added program's native module, the RISC-V instruction check and the
optimization levels: the files listed in the `tree-tests` job of
`.github/workflows/ci.yml`); without a tree they are skipped. Run them locally with
`TT_METAL_HOME` set to a built tree, for example `TT_METAL_HOME=~/tt-metal python -m
pytest tests/core/dsl/test_class_thread_chain.py`. They open no device. CI runs them on
the build machine on every push to main (never for a pull request), and there none of
them may skip; a new test of that kind is added to that job's list.

## The runtime on the CPU

[`experiments/checks/runtime_harness`](../experiments/checks/runtime_harness/README.md) compiles the
Tensix runtime's device headers and the map/collect kernels unchanged against a stub of
the tt-metal calls they use, runs them as host threads on a simulated grid, and forces
failure paths on it: stop flags, give-ups, lost or late command chunks, mailbox races,
items past a buffer's end. It is a tool run by hand (`python
experiments/checks/runtime_harness/run.py`), not part of the test suite.

The rule: a new wait in the runtime comes with a scenario there that makes it give up,
by the stop flag and by its timeout, and shows that every kernel of the tile still
returns; when a defect is fixed, one test pins it.

## Conventions

Black and ruff at 88 columns; C and C++ through `python -m komorebi.tools.cxx_format`
(clang-format with `.clang-format`, plus the rules it cannot express); Python blocks in
Markdown through blacken-docs; no imports inside functions; comment blocks of at most
two lines and docstrings of at most four; comments that say what the code does;
numerics reported as statistics, with exactness checked on exactly representable
inputs; every check able to fail on a broken implementation.

A module of `src/komorebi` is counted in statements: every statement node of its tree
at any depth, imports and docstrings left out, so comments and layout cost nothing.
Over 1000 is a warning (keep it only if it is one large coherent unit), over 2000 an
error. No import cycle of any kind: the graph counts every import, at the top of a file
and inside a function, `importlib.import_module` and `__import__` of a literal name
included (any other argument is an error). An import inside a function is allowed only
of an optional third-party package (pyproject.toml's optional dependencies, or the
config's `[optional]` list with its reason), never of a komorebi module. When a module
needs one above it, the modules are in the wrong order: move the shared piece down into
a module both import, or pass it in as an argument. `komorebi/__init__.py`'s lazy names
(a module `__getattr__`) are the one exemption. `python -m komorebi.tools.module_checks
--check` enforces this in CI against `src/komorebi/tools/module_checks.toml`: the files
over 2000 today and the import edges left to the work that removes them, each named;
both lists only shrink (a listed file may not grow, an entry no longer needed leaves).
A folder of `src/` or `tests/` with more than 12 Python files directly in it is a
warning (group them by concern), unless the config's `[flat_ok]` names it with its reason.

Files that must stay as they are (copies of another project's files) are listed in
`.clang-format-ignore`, one path, directory or pattern per line; the formatter skips them
unless they are named on its command line.

Experiments come first; a regression test comes after behaviour settles (see
`experiments/README.md`).

Result records are not committed. A run's output (a bench, probe, proof or gate record,
JSON or any other dump) stays in the script's `results/` folder or the path its `--json`
names, both ignored by git; a number that matters goes into a README or docs table with
the command that reproduces it. The JSON files git keeps are inputs no run produces
(`reference/public_scenes.json`, the reference streams and graphs under
`experiments/checks/`), each named in `.gitignore`.

## Known limitations

- The wheels on PyPI (0.2.0) are built for one card model, the Blackhole P150A, and
  run with no tree ([the two wheels](install.md#from-010-the-two-wheels)); a
  `komorebi-tt` wheel built locally without the native module builds it from a tt-metal
  tree on first use ([from a wheel](install.md#from-a-wheel)).
