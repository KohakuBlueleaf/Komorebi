# komorebi-runtime

The tt-metal build that Komorebi runs on, as one platform wheel with a locator. It is
the one copy of the runtime that `komorebi` (and later `kohakutt`) depend on, so that
both can run in one process on one card. Nothing of Komorebi's programming model is in
it. Nothing is published: the wheel is built locally.

| File | What it is |
|---|---|
| `tt-metal.pin` | The tt-metal release the runtime is built from (`v0.81.1`). It is the only pin in the repository; `packaging/tt-metal.pin` links to it. |
| `patches/` | Our patches to that release: a successful kernel build's compiler output goes through tt-metal's logger, and eager launches of a thread that opts in publish their fetch-queue entries together. Both come from Kohaku-TT's packaging. |
| `BUILD` | Our build number. The version is the pin plus it: `0.81.1.1`. A dependent requires `>=0.81.1.N,<0.81.2`. |
| `build_tt_metal.sh` | Clones the pin into a directory, applies the patches, configures it with the X280 runtime and without distributed (OpenMPI cannot be carried in a wheel), and builds only what the wheel carries. It installs no system packages: tt-metal's build dependencies must be on the host. |
| `build_wheel.py` | Builds the wheel from that directory. |
| `src/komorebi_runtime/` | The locator and its commands. |

## Building

```
runtime/build_tt_metal.sh /path/to/tt-metal              # KMR_JOBS=N limits the jobs
python3.13 runtime/build_wheel.py --tt-metal /path/to/tt-metal --out dist
```

`build_tt_metal.sh` builds the targets `ttnn` (with it `_ttnncpp.so` and the
libraries), `hw_toolchain`, the two X280 runtime objects, and the static X280 host
runtime (`x280_device`, `x280_state`) that Komorebi's native host module links. A
second run updates the build. `KMR_CCACHE=1` compiles through ccache.

`build_wheel.py` refuses a tree at another tag or with a patch not applied, and a
`pyproject.toml` whose version is not the pin plus `BUILD`. It needs `patchelf`
(`--patchelf PATH`). It prints the wheel's size and its three largest members.

## What the wheel carries

Everything is under `komorebi_runtime/`:

- `lib/`: the libraries `_ttnn.so` loads from the build, under the names they are
  loaded as, each with the rpath `$ORIGIN`: `libtt_metal.so`, `_ttnncpp.so`,
  `libtt-umd.so.0`, `libtracy.so.0.14.1`, `libtt_stl.so`;
- `tree/`: the kernel runtime tree the JIT compiles against, laid out as in a tt-metal
  tree (`runtime/hw`, `tt_metal/hw`, `tt-llk`, the kernel-side headers and sources,
  `tt_metal/sfpi-version`, the list tt-metal's own wheel ships), and the X280 pieces at
  their tree paths: `tt_metal/fabric/experimental/x280/ld/` and `include/`,
  `build/x280/obj/entry_active.o` and `x280_active_start.o`;
- `python/ttnn/`: **interim**. tt-metal's `ttnn` Python package with `_ttnn.so`
  (rpath `$ORIGIN/../../lib`), carried only while Komorebi's host still imports
  `ttnn`. It is on the path only after `komorebi_runtime.activate()`. Its own Python
  dependencies (`loguru`, `numpy`, `torch` and others) are not declared;
- `manifest.json`: the pin, the tt-metal commit, the patches and their sha256, the build
  number, the version, the libraries and every file of the package.

Not inside: the RISC-V compiler (sfpi). `python -m komorebi_runtime sfpi` fetches the
release `tree/tt_metal/sfpi-version` names, checks its sha256, and unpacks it into the
cache directory.

## The locator

Nothing runs at import.

| Call | Answer |
|---|---|
| `installed()` | whether the package holds a built runtime (an install of `runtime/` alone does not) |
| `manifest()`, `pin()`, `version()` | the build's manifest; `"v0.81.1"`; `"0.81.1.1"` |
| `lib_dir()`, `tree()` | the libraries; the kernel runtime tree inside the package |
| `x280()` | the X280 runtime objects in link order, the linker script, the include directory |
| `sfpi_dir()`, `sfpi_bin()` | where the compiler is fetched to; its `bin` directory, or None before the fetch |
| `cache_dir()` | `KMR_CACHE`, else `$XDG_CACHE_HOME/komorebi`, else `~/.cache/komorebi` |
| `root()` | the runtime root: `<cache>/runtime-<version>/`, a directory of links to the tree's entries plus `runtime/sfpi` once the compiler is fetched. It has a tt-metal tree's layout, so it serves as `TT_METAL_RUNTIME_ROOT` and as the tree Komorebi builds X280 images against |
| `activate()` | sets `TT_METAL_RUNTIME_ROOT` to `root()` unless it is set, and puts `python/` on `sys.path` (interim, for `ttnn`) |

```
python -m komorebi_runtime sfpi     # fetch the compiler into the cache
python -m komorebi_runtime doctor   # files present, libraries load, compiler found
python -m komorebi_runtime root     # print the runtime root
```

`doctor` exits with status 1 when a file is missing, a library does not load or the
compiler is not fetched.

## How Komorebi finds it

With `TT_METAL_HOME` set, Komorebi uses that tree, as before. Otherwise, when
`komorebi_runtime` is importable and holds a runtime, Komorebi uses `root()` as its
tree: the X280 objects, linker script, include directory and compiler come from the
package through it. `kmr doctor` says which was used (`from TT_METAL_HOME` or
`from komorebi-runtime 0.81.1.1`). `komorebi`'s extra `runtime` names this package; it
is not yet required.
