# Core reset: one core of a run back without a reset of the card

Whether a run can get one Tensix core back while the other 119 keep their kernels:
its RISC CPUs reset, its kernels loaded again, the core answering commands again. A
ladder of four steps, each stopped at the first that fails: the host resets a healthy
core; the dispatcher on the X280 does the same; a core whose handler never returns; a
core whose compute unit was left running.

What it shows: a core whose handler never returns, in a data-movement CPU or in a
compute thread, comes back without a card reset and without the other cores noticing
more than the give-up. The dispatcher resets it by itself at the give-up, from a list the
session hands it at load (`session.load(recover=True)`, the default), and the commands
after it are exact. A reset by the dispatcher takes 1.6 to 1.7 ms of its loop; by the
host, about 65 ms.

## Times

All on one Blackhole p150a (card 0), slow dispatch, X280 at 1350 MHz, the map/collect
example on 120 cores (`deadline=2.0`, `wait=4.0`), core 15 reset.

| What | Time | Conditions |
|---|---|---|
| The host's reset and reload of one core (`run.reset_core` without the list) | 65.4 ms: assert 0.4, firmware start 31.2, kernels 33.8 (32.6 ms in an earlier run of the same writes) | a healthy core, the runtime's device calls from the host |
| The dispatcher's reset of one core | 1.69 to 1.70 ms on the X280; 1.8 to 1.9 ms from the host's command to its answer | a write list of 51,008 to 68,864 bytes from the upload region, the core healthy or stuck in a loop |
| Handing the list over at load (`recover=True`) | 3.8 ms; 68,864 bytes of the upload region | every core's arguments in one list |
| The same on two cards as islands, each island its own | 3.4 to 4.0 ms an island; 104,192 bytes an island | two p150a cards, 119 ranks and one link core an island (the link core gets no list), `kernel_bytes=84 * 1024` |
| A stuck command to its recovery | the command's deadline (2 s) plus the give-up's wait (20 ms) plus the reset | no host action |
| The watch on the dispatcher's own loop: a stall to the loop running again | 100.01 ms: the watch's bound (100 ms) plus 1.2 microseconds | [dispatcher_watch](../dispatcher_watch/README.md), a stall made on purpose |

## Results

Core 15 (rank 1 of the group of cores 14 and 15, not a column leader) on a Blackhole
p150a, the map/collect example on all 120 cores; MAP commands on the group and on all
120 before and after each reset, every one checked page by page.

| Step | Recovered | Time of the reset | What the other cores saw |
|---|---|---|---|
| 1. The host resets a healthy core | yes | 65.4 ms (assert 0.4, firmware 31.2, kernels 33.8) | nothing: every MAP before and after exact, no give-up |
| 2. The dispatcher resets a healthy core | yes | 1.70 ms on the X280 (51 KB write list), 1.9 ms from the host's command to its answer | nothing: every MAP exact, no give-up |
| 3. Handler that never returns, in the reader | yes | 1.70 ms | core 14's writer left the command at its 2 s deadline (the give-up); every MAP after exact |
| 3. Handler that never returns, in the math thread | yes | 1.70 ms | the same |
| The give-up resets (no host action), reader then math thread | yes, both | the command's deadline plus the give-up's wait | the timeout names core 15 as recovered; every MAP after exact |
| The same through the session's public calls (`recover.py`): the give-up's reset, then `run.reset_core(15)` | yes | 1.59 ms on the X280 for `reset_core` | `recoveries` names core 15; every MAP after exact |
| Two cards as islands (`islands.py --step 1`): handler that never returns in island 1's core 15, group of island 1's 119 ranks | yes, by island 1's dispatcher | the command's deadline plus the give-up's wait | the timeout starts `island 1:` and names core 15 as recovered; the MAP after exact; `run.stop()` clean |
| The same in one group over both islands' 238 ranks (`islands.py --step 2`) | yes, by island 1's dispatcher | the same | island 0's part done; the timeout names island 1 and its core 15 (rank 134); the MAP after on 238 ranks exact; `run.stop()` clean |
| A handler that never returns on island 0's link core 0 while island 0's ranks read pages island 1 holds (`links.py --step 1`) | no: the dispatcher names it (`link core 0: did not leave the command`), the MAP waiting on it is given up as lost; the host calls `run.reset_link(0, 0)` | the deadline plus 20.2 ms; the lost MAP 2.9 ms later; reset 103.8 ms, open 0.3 ms | the MAP across before and after exact; `run.stop()` clean |
| The same while island 0 crosses a MAP to island 1's ranks (`links.py --step 2`) | the same; the crossing whose line was on the core is given up as lost and its packet number taken again | the deadline plus 20.3 ms; the lost crossing 79 us later; reset 102.1 ms, open 0.3 ms | the crossed MAP before and after exact; lines crossed each way 1, 1, then 2; `run.stop()` clean |
| 4. A compute unit left running | not run yet: it may need a card reset | | |

## What a reset has to write

The RISC reset alone does not bring a core back. Its firmware starts from what the
device open wrote, and the run has since covered part of it:

- The firmware's boot jump at L1 address 0, and the tables its start copies from L1
  scratch (bank to NoC coordinates, bank offsets, logical to virtual coordinates). The
  scratch lies inside the kernels' area, so the first launch of the run overwrote it on
  every core. A reset without them: the firmware never reports DONE (no boot jump), or
  starts and its kernels' DRAM reads go to the wrong coordinates and never complete.
- So the reset writes, for that core alone, what the open writes: the five firmware
  images, the tables (generated as the open generates them), the initial launch
  messages, the go message INIT, both read pointers 0, the boot jump. Then BRISC is
  released and the firmware's DONE awaited.
- Then the core's part of the launch (slow dispatch): kernel binaries, circular buffer
  configuration, runtime arguments, launch message, go. The kernels' start-up clears the
  core's ring and state words; a core that is not a column leader sends only its
  check-in, so nothing else in the run sees the restart.

`Launch.reset_core(x, y, timeout_s)` (the host module) makes those writes from the
host, through the runtime's reset call. `Launch.reset_list(x, y)` gives the same writes
as a write list, which the dispatcher's reset command (`KMD_HANDLER_RESET`,
`x280/kernel/reset.h`) runs on the X280: the soft-reset register through a strict NoC
window, the list from the upload region into the core's L1, a wait of at most a second
for the firmware, then the go.

## What the dispatcher's reset command needs

1. The write list held in the upload region for the run (67.6 to 68.9 KB with every core's arguments), handed over at load.
2. One list for each group of cores holding the same kernels, and which list each core takes (several programs: a list a program); no semaphores.
3. The soft-reset register of the core, one strict NoC window, the only register written.
4. The firmware images, its tables and the boot jump written again: the first launch overwrote them.
5. A bounded wait for the firmware's DONE (1 s), counted as the watchdog's heartbeat.
6. The core not a column leader and not core 0, whose start-up others wait for.
7. Commands already in the core's ring are lost: it restarts at an empty ring.
8. The dispatcher's loop blocked for the reset, 1.7 ms a core.
9. The give-up's records marked so the timeout names the cores it reset.
10. A core whose compute unit is still busy is not known to come back (step 4, not run).

## Running it

On card 0 with tt-metal's environment set, one step a run:

    python experiments/recovery/core_reset/ladder.py --step 1   # the host resets core 15
    python experiments/recovery/core_reset/ladder.py --step 2   # the dispatcher does
    python experiments/recovery/core_reset/ladder.py --step 3 --spin reader   # stuck, then reset
    python experiments/recovery/core_reset/ladder.py --step 5   # the give-up resets, no host action
    python experiments/recovery/core_reset/ladder.py --step 4   # a unit left running
    python experiments/recovery/core_reset/recover.py   # the session's own recovery, public calls
    python experiments/recovery/core_reset/islands.py --step 1   # two cards: island 1's ranks
    python experiments/recovery/core_reset/islands.py --step 2   # two cards: one group over both
    python experiments/recovery/core_reset/links.py --step 1     # two cards: a link core hangs, the host resets it
    python experiments/recovery/core_reset/links.py --step 2     # the same with a crossed command

It prints progress lines and one JSON line: the reset's times, the MAP commands before
and after (on the core's group (14, 2) and on all 120 cores), whether each is exact,
and which kernels gave up. Exit status 0 when every MAP is exact.

`kernels/` holds the map/collect example's reader and compute with one more handler,
SPIN (`kernels/spin.h`): a loop that never returns, in the reader, in the math thread,
or in all three compute threads driving the units.

## Source changes

- `komorebi_native/csrc/host/km_host.cpp`: `Launch.reset_core` (the host's reset) and
  `Launch.reset_list` (the write list; `every=True` for every core's arguments).
- `komorebi/session.py`: the backend keeps the launch (`backend.launch`).
- `komorebi_native/x280/kernel/reset.h`, `main.c`, `km_dispatch.h`, `kernel.h`: the
  reset command (`KMD_HANDLER_RESET`), the kept list, the soft-reset constants.
- `komorebi_native/x280/kernel/inspect.h`: at the end of a give-up's wait, the cores
  still in the command reset from the kept list (`KMD_INSPECT_RESET`).
- `komorebi/state.py`, `komorebi/errors.py`, `komorebi/run.py`: the inspect's reset
  flags, `CommandTimeout.recovered`, `Run.keep_resets`, `Run.reset_core`,
  `Run.recoveries`.
- `komorebi/session.py`: `load(recover=True)` hands the list over at load.
