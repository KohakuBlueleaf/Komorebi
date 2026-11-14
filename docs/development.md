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
  stats.py                     timing statistics and error against a reference
  program.py  session.py       Program; Session (loads a program under the dispatcher)
  run.py  errors.py  state.py  Run (submit, wait, stop); errors; result, state and trace decoding
  programs.py  overlay.py      several programs in one run; overlays: the builder and the slots
  movement.py  channel.py      both data-movement CPUs on one stream; the host channel
  tensor.py  dram.py           device tensors; interleaved DRAM buffers, bank by bank
  formats.py  insn_check.py    bfloat16 and tiles on the host; the instruction check of a kernel
  thread.py                    X280 threads: declaration, arguments, regions, results
  islands.py                   islands: the graph, ranks, the island table, placement (host values)
  island_session.py            the session over several cards (or one card split into islands)
  abi.py  native.py            constants read from the C headers; the image and native module
  power.py                     power sampling: cards, a GPU, the CPU package
  build.py                     out-of-tree builds: host program, Python module, X280 image
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
KMR_SIZES=1 python -m pytest tests/projects/test_kernel_sizes.py   # kernel sizes
python -m komorebi.tools.cxx_format --check
python -m komorebi.tools.comment_lint --check src tests examples experiments packaging tutorial projects
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
assembly of `build.py` against a small fake `build.ninja`, the power sampler against
fake sensors and a fake sysfs tree, and the formatter. Tests
marked `card` need a Tenstorrent card and are skipped unless `KMR_CARD=1` (they live in
`tests/hardware`). The host package's CPU tests run `Run` against a fake dispatcher
(`tests/core/fake_port.py`). The formatter tests are skipped when clang-format is not installed.

The reference tests in `tests/core` (`test_ref_*.py`) cover the scene generator and
the numpy rasterizer and ray tracer with the box tree. The comparisons of the C++ and
CUDA versions with numpy, and a slow rasterizer case, are in `tests/ref`, run by hand
([README](../tests/ref/README.md)); the CUDA ones only with `KMR_GPU=1`.

The CPU tests of a program under `projects/` live in
`tests/projects/test_<project>_*.py`, not in `tests/core`, which is the framework's
suite. They run as a step of their own, with a budget of 5 s for all programs
together (about 1 s each).

`tests/projects/test_kernel_sizes.py` checks that every program's kernels fit the
session's kernel buffer with 1 KiB to spare, without a card: it builds the kernels on
a mock device (tt-metal's Blackhole P150 cluster descriptor) with the runtime's
compiler, so it needs `TT_METAL_HOME` set to a built tree and `clang++-20`, builds the
native module and compiles each program (a few minutes the first time), and runs only
with `KMR_SIZES=1`. The continuous-integration workflow has no tree and skips it.

The continuous-integration workflow runs ruff, black, the CPU tests and the
programs' CPU tests.

## The runtime on the CPU

[`experiments/runtime_harness`](../experiments/runtime_harness/README.md) compiles the
Tensix runtime's device headers and the map/collect kernels unchanged against a stub of
the tt-metal calls they use, runs them as host threads on a simulated grid, and forces
failure paths on it: stop flags, give-ups, lost or late command chunks, mailbox races,
items past a buffer's end. It is a tool run by hand (`python
experiments/runtime_harness/run.py`), not part of the test suite;
`tests/core/test_runtime_harness.py` runs four of its scenarios that pin known
behaviour.

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

Files that must stay as they are (copies of another project's files) are listed in
`.clang-format-ignore`, one path, directory or pattern per line; the formatter skips them
unless they are named on its command line.

Experiments come first; a regression test comes after behaviour settles (see
`experiments/README.md`).

## Known limitations

- **The `host` and `pymodule` builds borrow compile flags from targets of another
  project.** `kmr build host` reads its compile and link flags from the target
  `ttx_bench` in `<tree>/build/build.ninja`, and `kmr build pymodule` from the target
  `_ttx280`. tt-metal itself does not define them: the
  [tt-x280-dispatch](https://github.com/KohakuBlueleaf/tt-x280-dispatch) project adds
  them to a tt-metal build configured with its option (`TT_X280_DISPATCH_DIR`, which its
  `install.sh` sets). Without them the build stops with the statement it could not find.
  The flags come from there because those targets already carry the include directories
  and defines that C++ code using the X280 runtime needs. This dependency ends when
  Komorebi has its own build files. `kmr build image` is not affected: it uses the X280
  runtime's compiler, flags, entry objects and linker script directly.
- The wheels on PyPI (0.2.0) are built for one card model, the Blackhole P150A, and
  run with no tree ([the two wheels](install.md#from-010-the-two-wheels)); a
  `komorebi-tt` wheel built locally without the native module builds it from a tt-metal
  tree on first use ([from a wheel](install.md#from-a-wheel)).
