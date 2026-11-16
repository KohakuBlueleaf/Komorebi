# An X280 core as a rank

What it takes for an X280 core to be a rank of a dynamic run: a message forwarded by a
Tensix core is handled by a C function on the X280, the X280 core's message back is
consumed by a Tensix core, and the run's counters and its end stay exact. The probe
runs the dispatcher's own dynamic run (the two passes over the status lines and the end
command) with the X280 cores in it, and the same round trips as port asks beside it.

## Results

One Blackhole p150a, slow dispatch, AICLK 1350 MHz (two samples a load), the X280 at
the session's 1000 MHz, host load 7.9 to 9.5. 100,000 round trips a run (100,008 where
the rank cores do not divide it), each a message from a rank core to its X280 core and
the reply back; every reply checked by the core that sent the message.

| Rank cores : X280 cores | in flight per core | round trip, mean (p99) | round trips a second | the same work as port asks: ask, mean (p99) | asks a second |
|---|---|---|---|---|---|
| 1 : 1 | 1 | 2.59 µs (3.41) | 0.37 M | 0.87 µs (1.42) | 1.01 M |
| 1 : 1 | 16 | 11.9 µs | 1.32 M | | |
| 10 : 1 (a row) | 1 | 7.2 µs | 1.31 M | 5.5 µs | 1.77 M |
| 10 : 1 | 16 | 120 µs | 1.33 M | | |
| 40 : 4 (a cluster) | 1 | 9.3 µs | 4.10 M | 5.9 µs | 6.64 M |
| 40 : 4 | 16 | 139 µs | 4.50 M | | |
| 12 : 12 | 1 | 2.58 µs (3.79) | 4.28 M | 0.92 µs (1.52) | 11.5 M |
| 12 : 12 | 16 | 13.8 µs | 12.9 M | | |
| 108 : 12 | 1 | 8.1 µs | 11.7 M | 5.3 µs | 19.8 M |
| 108 : 12 | 16 | 133 µs | 12.5 M | | |

A round trip is two messages: one pair moves 2.6 M messages a second, twelve X280 cores
25.8 M (12 : 12) and 24.9 M (108 : 12). An earlier run of the pair: 2.48 µs, 1.17 M
round trips a second with 16 in flight, asks 0.87 µs.

| Cost of one message | |
|---|---|
| Grid to X280, the Tensix core's send: reserve, write, commit | 0.83 µs alone; 0.21 µs a message at 4 a reservation |
| Grid to X280, the X280 core's take: a cache flush and a load per line, the check | 0.74 µs a message at 1.8 a look; 0.46 to 0.68 at 7.9 |
| X280 to grid, the X280 core's store of the reply: four 16-byte stores, two fences | 0.24 to 0.30 µs |
| X280 to grid, the Tensix core's take from its inbox | a few L1 loads (not timed apart) |

Every run was exact: every round trip done once, no reply wrong, nothing after the
last; the grid's sends equal the X280 cores' takes and the X280 cores' sends the grid's
takes (200,000 or 200,016 each way over a load's two message runs); and every run ended
by itself, the dispatcher's end command after its two passes saw every line idle with
equal counts (10 message runs and 5 ask runs). An X280 core takes 1.0 to 1.3 M messages
a second whatever feeds it, which is what bounds every row with 16 in flight; port asks
handle more because an ask is one 16-byte line in local memory, read in about 30
cycles, where a message is a line of DRAM the X280 core must flush and fetch.

## The probe

The program is a dynamic program (`Program.dynamic`); the dispatcher runs its RUN
command as for any dynamic run. Its kernels are the probe's, not the runtime's generic
reader and writer: the reader does everything.

- **Toward the X280.** Each X280 core with a stand-in has a queue of 1,024 messages of
  64 bytes in its cluster's region (`XR_QUEUE`). A rank core reserves positions with
  one atomic add on its stand-in's tail line, writes the messages into the bank with
  the first one's mark 0, waits for the writes, then writes the first 16 bytes with its
  mark: the runtime's commit. The X280 core flushes the line at its head, loads it and
  takes the prefix of valid marks, up to 8. It never reads the grid and never writes
  its queue, so no line it holds can be written back over the grid's.
- **Toward the grid.** Each rank core has an inbox of 64 messages in L1. The X280 core
  stores a reply's last 48 bytes through the library's window onto that core, a fence,
  then its first 16 bytes with the mark; the core takes the prefix of valid marks. The
  core writes its inbox head into a 64-byte credit line in the X280 core's area every
  16 messages; the X280 core reads it only when the inbox might be full.
- **The run's counts and end.** A rank core keeps the runtime's rules: counts per run,
  an idle line after 20,000 cycles with nothing to do, a busy line landed before it
  takes a message again. Each X280 core has a stand-in, a Tensix core of the RUN group
  whose kernel writes no status line: the X280 core writes the stand-in's line itself,
  through a window onto the dispatcher's local memory set non-posted, so a store and
  a fence return once the line landed. The busy line lands before the first reply after
  an idle line; the idle line goes out after 15 µs with nothing in its queue (replies
  are never held). The stand-in tells its X280 core the run's start (its sequence and
  mode) and, once the end command reaches it, the end; the stand-in's tail line is
  the X280 core's queue tail, so the stand-ins are also the X280 cores' counter tiles.
- **The asks.** The same rank cores call their X280 core through the port
  (`km::X280Port::call`, three words), one ask at a time; the X280 cores' lines say idle
  with no messages, and the run ends the same way.

The X280 side is `x280/rank.c` (about 320 lines), the grid side `kernels/reader.cpp`
(about 380): no dispatcher, library or runtime change.

## What a proper build needs

**On the X280 side, for the accounting:** per run a sent and a consumed count; the run's
sequence at its start and its end command; a status line per X280 rank at the
dispatcher, written through a non-posted window, busy before the first send after an
idle line, idle after a quiet time with an empty queue and nothing held; a give-up line
as the cores have. The probe does exactly this; it needed nothing the library lacks
except a store into a core's L1 (it used the port's window directly).

**Elsewhere,** what the probe borrowed or left out:

| Where | What | Lines, roughly |
|---|---|---|
| `x280/kernel/dynamic.h`, `km_dispatch.h`, `kernel.h` | status lines for X280 ranks read in the two passes (the 2 KiB of lines has 8 free, 12 are needed); the run's start and end handed to the rank clusters (as commands to a thread, the path of `experiments/x280/thread_commands`) instead of stand-ins | 80 |
| `km_abi.h` | the X280 ranks' lines, tail lines and inbox layout | 20 |
| `km_messages.hpp` | the writer's flush to an X280 rank's queue (its bank and region instead of the interleaved queue); the reader taking a second source, the L1 inboxes, with credit lines; ranks counted with the X280 ones in `Messages` | 200 |
| `x280/kmx/` | a rank API beside the port's (take, forward, send, consume, as `Messages`), the status window, the store into a core's L1 | 300 |
| `program/description.py`, `session/session.py`, `program/thread.py` | `Program.dynamic(..., x280_ranks=...)`, the queues in the clusters' regions, the inboxes' L1, the words | 120 |
| tests (`tests/core`), the reference page | the layouts and the host's checks; the page's section | 200 |

About 900 lines. The riskiest part is the generic reader: today it reads one queue in
DRAM in order; with X280 ranks it also reads an inbox per X280 cluster in L1, and an
inbox written by four X280 cores needs them to share its positions (an atomic in their
own memory, which they have).

**What a message toward the X280 may carry.** Here one 64-byte line, word 0 the
runtime's mark (60 bytes of the program's). Nothing in the path fixes that size: the
runtime's messages are 64 to 256 bytes, the grid's writes and the commit are the same
for any of them, and the X280 core flushes and loads one more line per 64 bytes (about
0.2 µs each, from the line fill measured in `experiments/x280/x280_threads`; not measured
here). The queue costs `XR_QUEUE` × the message size of a cluster's region per X280
core. Toward the grid the size is bounded by the inboxes' L1: 64 messages of 64 bytes
is 4 KiB a core and an X280 cluster.

**Reading.** The accounting is small and was exact at once: the X280 side of the probe
is one file and every run ended by itself. A proper build is not small: about 900 lines,
most of them in the generic reader and the dispatcher, and a second message source in
the reader is a change to the part every dynamic program runs on. Worth building as a
package of its own, after commands to a thread land. The rate is not the obstacle: an
X280 rank takes 1.0 to 1.3 M messages a second, near a port's 1.1 M asks; the latency is
about 3 times an ask's (2.5 µs against 0.87), most of it the DRAM queue: the
reservation, the writes and the commit on the grid, the flush and the line fill on the
X280.

## Running it

`rank.py` makes the dynamic program (the grid's kernels, the thread `x280/rank.c` on
the clusters used, the ports), runs one RUN per entry of `--windows` and one of asks
with `--ask`, then ends the threads, and prints one JSON line; `ok` is false, and it
exits non-zero, when any check fails.

```sh
python rank.py --ranks 1 --x280 1 --trips 100 --windows 1          # smallest
python rank.py --ranks 1 --x280 1 --windows 1 16 --ask
python rank.py --ranks 10 --x280 1 --windows 1 16 --ask
python rank.py --ranks 40 --x280 4 --windows 1 16 --ask
python rank.py --ranks 12 --x280 12 --windows 1 16 --ask
python rank.py --ranks 108 --x280 12 --windows 1 16 --ask
```

Rank cores are cores 0 to `--ranks` - 1, core i served by X280 core i mod `--x280`
(clusters 1 to 3, four X280 cores each, in order); the stand-ins follow them. Each
rank core keeps a page of stats in L1 (`xr.h`): round trips, the histogram of their
times in AICLK cycles, wrong replies, its counts, the time in its sends; each X280 core
returns its counts and the time in its takes and stores as result words.
