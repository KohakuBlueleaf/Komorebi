---
title: Environment setup
summary: Check that the machine is ready with kmr doctor, run the first example on the card, and read its output.
---

# 2. Environment setup

The [install guide](../install.md) builds everything: the Python package, a tt-metal
tree with its X280 runtime, and the card's one-time preparation. This page does not
repeat it. It shows how to tell that the result works: one command that checks the
software, one example that runs on the card, and what each line of their output means.

## The idea

Three things have to be right for a program to run:

1. **The Python package.** Komorebi is a Python package that needs numpy and nothing
   else. It is installed into a virtual environment, from a clone or from a wheel.
2. **A tt-metal tree.** Komorebi compiles kernels and opens the card through tt-metal
   v0.81.1 with its X280 runtime. `TT_METAL_HOME` names the tree. With the
   `komorebi-runtime` wheel installed, no tree is needed.
3. **A prepared card.** The card's X280 memory is primed once; after that it stays
   primed.

| Command | What it does |
|---|---|
| `kmr doctor` | Prints the Python, the package, the tt-metal tree and its X280 compiler. Opens no card, takes no lock, always exits 0 |
| `python -m komorebi.native` | Builds, ahead of time, the dispatcher's image and the host module that a session needs (about ten seconds, once) |
| `kmr power` | Prints the card's board power once a second, from a second shell |
| `python <example>.py` | Runs a program on the card |
| `TT_LOGGER_LEVEL=error` | An environment variable of tt-metal: hides its info lines |

## Check the software

Activate the environment, point at the tree, and ask:

```bash
source .venv/bin/activate
export TT_METAL_HOME=~/tt-metal TT_METAL_RUNTIME_ROOT=~/tt-metal
kmr doctor
```

```{ .text .km-output }
komorebi      0.1.0
python        3.13.18 (~/.venv/bin/python3)
package       ~/komorebi/src/komorebi
include       ~/komorebi/src/komorebi_native/include
tt-metal      ~/tt-metal
from          TT_METAL_HOME
build         ~/tt-metal/build
x280 runtime  yes
x280 compiler ~/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-gcc
```

The paths are the ones of the machine that produced the output, shortened; yours
differ. Two lines decide whether a card run can work: `x280 runtime` must say `yes`,
and `x280 compiler` must name a file. If `tt-metal` says `not found`, `TT_METAL_HOME` is
unset or wrong.

## Run the first example

`examples/01_mapcollect` is the program of this tutorial's chapters 4 to 9 grown up. For
now it is only a test that the card, the dispatcher and the kernels work together. Four
jobs, each on a core group of 8 cores:

```bash
cd examples/01_mapcollect
TT_LOGGER_LEVEL=error python mapcollect.py --jobs 4 --count 8 --out-pages 64
```

```{ .text .km-output }
[   3.99] kernel buffer: 17792 of 70656 B (text: BRISC 5648, NCRISC 4336, TRISC0 3304, TRISC1 1624, TRISC2 2188)
[   4.04] kernels built and checked (0.87 s)
[   4.04] launch
[   4.12] dispatch kernel started
[   4.12] ready
[   4.12] loaded: 120 cores, depth 4
[   4.12] 4 events
[   4.14] x280 runtime closed
[   4.16] grid returned (0.0 s)
[   4.49] state and trace read back
[   4.58] output 0: 64 of 64 pages read
[   4.58] 1 outputs read back
{"jobs": 4, "seg": "map", "teams": "fixed", "cores": 120, "depth": 4, "count": 8, "tiles": 64, "kernel": {"error_name": "none", "records": 5, "events": 5, "max_busy": 4}, "us_per_job_kernel": 62.476, "us_per_job_host": 153.33, "issue": "many", "host_us_per_record": {"issue": 93.341, "wait_python": 46.193, "wait_native": 19.688}, "latency_us": {"mean": 5.41, "max": 6.24}, "cores_magic_ok": 120, "cores_ended": {"reader_end": 120, "writer_end": 120, "compute_end": 120}, "compute_wrong": 0, "gave_up_cores": 0, "member_records": 32, "core_counts_bad": 0, "first_bad_cores": [], "jobs_checked": 4, "jobs_overwritten": 0, "exact_jobs_checked": 2, "jobs_bad": 0, "first_bad": [], "random_total_rel_rms_max": 0.001701, "tiles_same_as_before": 0, "tiles_without_input": 0, "ok": true, "wall_s": 4.6}
```

Each line before the last is the program's progress log. The time in brackets is
seconds since the start. The last line is a summary in JSON, and the example exits with
status 1 if `ok` is not `true`. The parts to look at:

| Part of the output | Meaning |
|---|---|
| `kernel buffer: 17792 of 70656 B` | The compiled kernels take 17,792 bytes of the 70,656 that L1 gives to kernels. Chapter 3's `load` checks it fits |
| `loaded: 120 cores` | The kernels are on all 120 Tensix cores and the dispatcher is running |
| `4 events` | Four commands were submitted and four reports came back |
| `"jobs_bad": 0`, `"compute_wrong": 0` | The host compared every output tile with numpy: none was wrong |
| `"ok": true` | Everything above held |
| `latency_us` | Microseconds from submit to report, per command. The numbers depend on the card and the host's load |

The times in the log and the figures in the summary differ from run to run and from
card to card. A first run on a machine also compiles the kernels and builds the
dispatcher's image, which takes tens of seconds more and prints compiler messages.

## Notes

- If a run stops with `km.CardBusy`, another process holds the card. The error names its
  process ID. Wait for it or ask whose it is; Komorebi never takes a card from a live
  process.
- A process killed with `kill -9` while a program ran can leave the card needing a reset.
  Stop a program with Ctrl-C or `SIGTERM`: the session catches both and closes the card.
- `TT_LOGGER_LEVEL=error` is only for readability. Leave it out when something fails:
  tt-metal's info lines tell what it was doing.
- The tests that need a card run only with `KMR_CARD=1`. They are for the repository's
  developers; you do not need them.
- Reference: [install](../install.md), [host API: Session](../reference/host.md#session).
