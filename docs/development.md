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
  cli.py                       the `kmr` command: doctor, build
  layout.py                    the grid, its panels and teams of cores (pure functions)
  stats.py                     timing statistics and error against a reference
  build.py                     out-of-tree builds: host program, Python module, X280 image
  engines/stream.py            micro-op streams: builder, upload encoding, check, simulator
  tools/cxx_format.py          clang-format with the rules it cannot express
  manifest/  lang/             reserved for later tiers (README inside, no code yet)
src/komorebi_native/           the native layer (README inside; directories only)
examples/  projects/           runnable examples; larger programs (README inside)
experiments/  ref/             standalone proofs; CPU references (README inside)
tests/core/                    tests that need no card (reference streams in data/streams)
docs/                          concepts, install, this page
packaging/                     tt-metal.pin: the tt-metal version Komorebi is developed against
```

`import komorebi` needs no tt-metal tree and imports no tt-metal or TT-NN module; the
package boundary test enforces it. Nothing runs at import and no device is opened
except by code that is asked to.

## Checks

```
python -m pytest tests/core                     # CPU tests
python -m komorebi.tools.cxx_format --check
ruff check . && black --check . && blacken-docs -t py313 -l 88 $(git ls-files '*.md')
```

The `dev` extra installs the tools, including clang-format 22 or later (the style file
`.clang-format` uses options older versions reject). Run the commands from the
repository, in the environment of [install](install.md#from-a-clone).

`tests/core` covers the layout functions (panels, teams, shares), statistics, the
package boundary, `kmr doctor` with and without a tree, the parsing and command
assembly of `build.py` against a small fake `build.ninja`, and the formatter. Tests
marked `card` would need a Tenstorrent card and are skipped unless `KMR_CARD=1`; there
are none yet. The formatter tests are skipped when clang-format is not installed.

The continuous-integration workflow runs ruff, black and the CPU tests.

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
- The native layer has no sources yet, so there is nothing to build with
  `komorebi_native/cmake` and no wheel.
