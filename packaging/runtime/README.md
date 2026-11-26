# komorebi-runtime

The tt-metal build that Komorebi runs on, as one platform wheel with a locator. It is
the one copy of the runtime that `komorebi-tt` (and later `kohakutt`) depend on, so that
both can run in one process on one card. Nothing of Komorebi's programming model is in
it. It is on PyPI as `komorebi-runtime` (0.81.1.4, with Komorebi 0.2.0, 0.3.0, 0.4.0, 0.5.0, 0.5.1, 0.5.2, 0.5.3 and 0.6.0, for
one card model); `pip install "komorebi-tt[runtime]"` installs it.

| File | What it is |
|---|---|
| `../tt-metal.pin` | The tt-metal release the runtime is built from (`v0.81.1`): `packaging/tt-metal.pin`, the only pin in the repository. |
| `patches/` | Our patches to that release: a successful kernel build's compiler output goes through tt-metal's logger, and eager launches of a thread that opts in publish their fetch-queue entries together (both from Kohaku-TT's packaging); no path of the build machine in the binaries (the X280 objects and images compiled with `-ffile-prefix-map`, the experimental NoC estimator's table looked up relative to the working directory); and 1250, 1350 and 1500 MHz in the X280 clock table (it held 15, 200, 800, 1000 and 1750), the clocks Komorebi's session defaults to 1350 on. The table is compiled into the static X280 host runtime that the native host module links, so a session's clocks are those of the tree its module was built in: a `komorebi-tt` wheel built with `--host-module` from a patched tree has them with any runtime build installed. Builds 4 and later carry the fourth patch (0.81.1.3 was built without it); without it the session falls back to 1000 MHz. |
| `BUILD` | Our build number. The version is the pin plus it: `0.81.1.4`. A dependent requires `>=0.81.1.N,<0.81.2`. |
| `build_tt_metal.sh` | Clones the pin into a directory, applies the patches, configures it with the X280 runtime and without distributed (OpenMPI cannot be carried in a wheel) and with `-ffile-prefix-map=DIR=.` (so no path of the build machine is written into a library), and builds only what the wheel carries. It installs no system packages: tt-metal's build dependencies must be on the host. |
| `build_wheel.py` | Builds the wheel from that directory. |
| `src/komorebi_runtime/` (at the top of the repository) | The locator, the one-time setup steps (`prepare.py`) and the commands. The `komorebi-tt` wheel does not carry it. |

## Building

```
packaging/runtime/build_tt_metal.sh /path/to/tt-metal    # KMR_JOBS=N limits the jobs
python3.13 packaging/runtime/build_wheel.py --tt-metal /path/to/tt-metal --out dist
```

`build_tt_metal.sh` builds the targets `ttnn` (with it `_ttnncpp.so` and the
libraries), `hw_toolchain`, the two X280 runtime objects, the X280 idle image, and the
static X280 host runtime (`x280_device`, `x280_state`) that Komorebi's native host
module links, and the X280 priming program `x280_prime_test` (`packaging/build_wheel.py --host-module` builds it in this tree). A
second run updates the build. `KMR_CCACHE=1` compiles through ccache.

`build_wheel.py` rejects a tree at another tag or with a patch not applied, and a
`pyproject.toml` whose version is not the pin plus `BUILD`. It needs `patchelf`
(`--patchelf PATH`). The platform tag is `manylinux_<glibc floor>_x86_64`, where the
floor is the newest `GLIBC_` symbol version any carried library requires (read with
`readelf -V`; 2.38 for a build on glibc 2.39), so the tag claims no more than the
files need; `--platform` overrides it. It prints the floor, the wheel's size and its
three largest members.

## What the wheel carries

Everything is under `komorebi_runtime/`:

- `lib/`: `_ttnncpp.so` (Kohaku-TT's native library links it) and the libraries it
  loads from the build, under the names they are loaded as, each with the rpath
  `$ORIGIN`: `libtt_metal.so`, `libtt-umd.so.0`, `libtracy.so.0.14.1`, `libtt_stl.so`;
- `tree/`: the kernel runtime tree the JIT compiles against, laid out as in a tt-metal
  tree (`runtime/hw`, `tt_metal/hw`, `tt-llk`, the kernel-side headers and sources,
  `tt_metal/sfpi-version`, the list tt-metal's own wheel ships), and the X280 pieces at
  their tree paths: `tt_metal/fabric/experimental/x280/ld/` and `include/`,
  `build/x280/obj/entry_active.o` and `x280_active_start.o`, and the idle image the
  X280 tiles are booted to, `build/x280/firmware-lim-idle.bin`;
- `bin/x280_prime_test`: the pinned tree's program that primes a card's X280 memory,
  as the tree builds it, with the rpath `$ORIGIN/../lib`. It loads only the libraries
  above; gtest and the X280 host runtime are linked into it statically;
- `manifest.json`: the pin, the tt-metal commit, the patches and their sha256, the build
  number, the version, the libraries and every file of the package.

Not inside: tt-metal's Python packages (`ttnn` and its module `_ttnn.so`): Komorebi's
host goes through tt-metal's C++ API in its native module. Nor the RISC-V compiler
(sfpi). `python -m komorebi_runtime sfpi` fetches the
release `tree/tt_metal/sfpi-version` names, checks its sha256, and unpacks it into the
cache directory.

## The locator

Nothing runs at import.

| Call | Answer |
|---|---|
| `installed()` | whether the package holds a built runtime (an install of `packaging/runtime/` alone does not) |
| `manifest()`, `pin()`, `version()` | the build's manifest; `"v0.81.1"`; `"0.81.1.4"` |
| `lib_dir()`, `tree()` | the libraries; the kernel runtime tree inside the package |
| `x280()` | the X280 runtime objects in link order, the linker script, the include directory, the idle image |
| `prime_program()` | the X280 priming program (absent before 0.81.1.3) |
| `sfpi_dir()`, `sfpi_bin()` | where the compiler is fetched to; its `bin` directory, or None before the fetch |
| `cache_directory()` | `KMR_CACHE`, else `$XDG_CACHE_HOME/komorebi`, else `~/.cache/komorebi` |
| `root()` | the runtime root: `<cache>/runtime-<version>/`, a directory of links to the tree's entries plus `runtime/sfpi` once the compiler is fetched. It has a tt-metal tree's layout, so it serves as `TT_METAL_RUNTIME_ROOT` and as the tree Komorebi builds X280 images against |
| `activate()` | sets `TT_METAL_RUNTIME_ROOT` to `root()` unless it is set |

The commands need the installed wheel ([building](#building)): in a clone the package
holds no built runtime, and they stop.

```
python -m komorebi_runtime sfpi     # fetch the compiler into the cache
python -m komorebi_runtime prime    # prime a card's X280 memory (--card N)
python -m komorebi_runtime doctor   # files present, libraries load, compiler found
python -m komorebi_runtime root     # print the runtime root
```

`sfpi` and `prime` are the lower-level form of Komorebi's `kmr sfpi` and `kmr prime`;
`kmr setup` runs both, in that order.

## Priming a card

A card's X280 memory is prepared once: the ECC of the X280 tiles' local memory (LIM) is
primed, and the card is then reset once (`tt-smi -r N`); the priming survives resets.
`prime` runs the carried program on card N (`--card N`, else the first of
`TT_VISIBLE_DEVICES`, else 0) with the runtime root, slow dispatch and the program's
opt-in (`X280_ALLOW_LIM_ECC_PRIME=1`), and reads its result:

- the card was unprimed: the program opened the X280 runtime, which primed every
  L2CPU tile (WayEnable committed, the top way walked, the marker written); the card
  needs its reset;
- the card was primed before: the open found it ready and wrote nothing to the X280
  tiles (the same open every Komorebi session makes); nothing to do;
- anything else (no `/dev/tenstorrent/N`, the compiler not fetched, the program's
  failure): exit status 1 and the reason.

Priming opens the card and refuses while an X280 tile is out of reset, so nothing else
may use the card while it runs. The program is the pinned tree's own rather than a new
one: the X280 runtime's entry point lives in a static library that no carried shared
library exports, and the program needs no library the wheel does not already carry.

A program that uses `komorebi.Session`, the examples among them, needs nothing set:
Komorebi activates the runtime before it opens the card. A host of one's own sets
`TT_METAL_RUNTIME_ROOT` to what `root` prints.

`doctor` exits with status 1 when a file is missing, a library does not load or the
compiler is not fetched. It names the priming command and does not open the card, so
it does not say whether a card is primed.

## How Komorebi finds it

With `TT_METAL_HOME` set, Komorebi uses that tree, as before. Otherwise, when
`komorebi_runtime` is importable and holds a runtime, Komorebi uses `root()` as its
tree: the X280 objects, linker script, include directory, idle image and compiler come
from the package through it, and a `komorebi-tt` wheel built with `--host-module` loads
its prebuilt `_km_host`, whose rpath `$ORIGIN/../komorebi_runtime/lib` finds this
package's libraries when both are installed in one environment. `kmr doctor` says
which was used (`from TT_METAL_HOME` or `from komorebi-runtime 0.81.1.4`).
`komorebi-tt`'s extra `runtime` names this package; it is not yet required.
