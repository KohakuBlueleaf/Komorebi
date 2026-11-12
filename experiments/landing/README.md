# Landing a 64-byte multicast record

The X280 sends work to the Tensix grid as one 64-byte record, written by one vector store
through a multicast window into the same L1 address on every core. The four 16-byte
chunks of that NoC write do not arrive in a defined order, so a core polling the line can
see a partly written record. This experiment compares two ways for a core to know that a
record has landed completely:

- **Protocol A, echo words.** Words 0, 4, 8 and 12 (one per 16-byte chunk) carry the
  echo word `tag << 16 | seq`. A core accepts the record when all four equal the
  sequence number it waits for.
- **Protocol B, record then sequence word.** The X280 stores the 64-byte body, then
  (after a `fence ow, ow`) one 32-bit sequence word `tag << 16 | seq` at a separate
  address of the same control block, through the same multicast window. A core waits
  only on the sequence word, then reads the body.

There is no work per record, so the protocol itself is what is measured.

## What runs

| part | file |
|---|---|
| X280 image (hart 0): writes records 1..n, paces on the column leaders' acks | `x280/landing_seq.c` |
| Tensix kernel, one per core (RISCV_0): waits, verifies, counts, completes | `kernels/worker.cpp` |
| Layout, payload function and counter indices shared by both | `kernels/landing.h` |
| Device template of tt-x280-dispatch (unchanged copy, see below): check-in, column-leader acks | `kernels/ttx_dispatch.hpp` |
| Host module: X280 runtime, tt-x280-dispatch's `DispatchPlan` with its own `start()` | `host/landing_x280.cpp` |
| Driver: builds the program, runs, closes the X280 runtime, reads the counters | `landing.py` |

- **Record contents.** Every payload word is `landing_word(seq, w, tag)`, a mix of the
  sequence number, the word index and the run's random tag, so every core recomputes
  and checks every payload word of every record it accepts, and a record left in L1 by
  an earlier process never verifies.
- **Pacing.** Depth 1: the X280 waits until every column leader acknowledged record i
  before it writes record i + 1 (slot 0 only). Depth 4: record i goes to slot i % 4 once
  every acknowledgement is at least i - 4 (an independent stream). Protocol B uses four
  body slots and one sequence word; a core accepts when the word is at least its seq.
- **Counters per core**, kept by the kernel and written to L1 and then to one DRAM page
  per core at the end; the host reads them after the X280 runtime is closed:
  - *accepted*: records accepted (must be n on every core);
  - *torn*: accepted records with a payload word that did not verify (for B: the
    sequence word was seen but the body was not yet complete). A torn record stays
    counted; the kernel then re-reads it for diagnosis only (*healed*);
  - *early*: protocol A polls that saw one to three of the four echo words, i.e. a
    partly landed record (not applicable to B, which never looks at the body early);
  - *idle polls*: polls that saw nothing of the record yet;
  - *lost*: a bounded wait gave up (the kernel then completes the last record so the
    run ends instead of hanging);
  - with `--stamp`: wall-clock cycles of the one successful acceptance check
    (`invalidate_l1_cache` plus the loads and compares), min, mean and max.
- **Time per record**: X280 `mtime` (50 MHz) from record 1 issued to the last ack,
  divided by n.

## The copied header

`kernels/ttx_dispatch.hpp` is an unchanged copy of
`src/dispatcher/device/ttx_dispatch.hpp` from
[tt-x280-dispatch](https://github.com/KohakuBlueleaf/tt-x280-dispatch), taken at
revision `cb12789` (the commit that added it; it has not changed since). That project is
licensed under Apache-2.0, the same licence as this repository. The file is listed in
`.clang-format-ignore` at the repository root so the formatter leaves it as it is; to
check the copy, compare it byte for byte with the original at that revision.

## Build and run

Needs this repository installed (`pip install -e .` provides `kmr`), a configured tt-
metal source tree with the X280 runtime built, one Blackhole p150a, and a checkout of
tt-x280-dispatch at the revision above. `build.sh` uses `kmr build`, which borrows the
host module's compile flags from build targets that tt-x280-dispatch adds to a tt-metal
build (see the [known limitations](../../docs/development.md#known-limitations)). The
checkout provides the host header `ttx_dispatch_host.hpp`, which `host/landing_x280.cpp`
includes unchanged. Nothing is built inside either tree.

```
TT_METAL_TREE=<tt-metal> TTX_DISPATCH=<tt-x280-dispatch> ./build.sh
python landing.py --proto a|b --depth 1|4 --records N [--stamp]
```

`landing.py` runs in slow dispatch and prints one JSON line (`ok` is true when every core
finished, accepted all n records and nothing was lost). On a shared machine, `run.sh`
wraps one run in a hang watcher (a tool of the host it was written on, not part of this
repository) that holds the card lock and kills the job on a time limit:
`run.sh <tag> <landing.py args>` with `LANDING_ENV` (environment script that sets
`$PY`, the Python to run), `HANGWATCH` (the watcher's path), `LOCK_DIR` (the directory
of its stop and reset lock files), and optionally `CARD` and `LIMIT`. Logs go to
`../../.logs/`, which git ignores.

## Results

p150a, 120 Tensix cores (12 x 10) at AICLK 1350 MHz, X280 at 1750 MHz, slow dispatch,
posted multicast windows, two panels. Every run below finished with `ok` true: all 120
cores accepted every record, nothing lost. Sizes were run in the order 1, 100, 10^4,
10^5, 10^6 for each protocol and depth; then 10^6 twice more without and three times with
stamping. Host load average (one minute) 6.6 to 12.7 on 96 CPUs.

Torn records per configuration, summed over all its runs (1, 100, 10^4, 10^5, and six
runs of 10^6: 6,110,101 records; core-records = records x 120 cores):

| protocol | depth | core-records checked | torn | early polls (A: partly landed record seen) |
|---|---|---|---|---|
| A | 1 | 733,212,120 | 0 | 61.5 M (0.108 per core-record in the unstamped runs) |
| B | 1 | 733,212,120 | 0 | n/a |
| A | 4 | 733,212,120 | 0 | 36.5 M (0.052 per core-record in the unstamped runs) |
| B | 4 | 733,212,120 | 0 | n/a |

Microseconds per record (X280 tick counter), 10^6 records, each run:

| protocol | depth | runs, no stamp | runs, `--stamp` | acceptance check, cycles (min / mean / max) |
|---|---|---|---|---|
| A | 1 | 1.2849, 1.2849, 1.2851 | 1.3018, 1.3024, 1.3024 | 39 / 39.8 / 49 to 51 |
| B | 1 | 1.3096, 1.3097, 1.3094 | 1.3229, 1.3226, 1.3225 | 29 / 29.6 / 36 |
| A | 4 | 0.7720, 0.7720, 0.7720 | 0.7941, 0.7941, 0.7941 | 34 / 39.2 / 49 |
| B | 4 | 0.7281, 0.7281, 0.7281 | 0.7500, 0.7500, 0.7500 | 24 / 29.2 / 36 |

Stamping reads the wall clock twice per poll and adds 0.013 to 0.022 us per record; the
cycle counts include one wall-clock read (1 cycle when measured back to back).

What was observed:

- Neither protocol produced a torn record in 7.3 x 10^8 core-records each.
- Partly landed records are common: under protocol A, unstamped, there were 0.108 early
  polls per core-record at depth 1 and 0.052 at depth 4 (a poll that saw some but not
  all of the four echo words). A single word checked alone (for example only `seq` in
  word 0) could have accepted some of these records before they were complete.
- Protocol B's check is about 10 cycles (7 ns) cheaper per record. End to end, B is
  0.025 us slower than A at depth 1 (presumably the extra store and fence per record on
  the X280; not measured separately) and 0.044 us faster at depth 4 (cause not
  separated).

Decision: protocol A, the sequence number repeated in every 16-byte chunk of the
record. Both protocols were equally clean here, so the choice rests on what each one
depends on. A's correctness needs only that each 16-byte chunk lands whole (not
documented either, but the payload check would have counted any record that broke it as
torn, and none was). B needs the separate sequence-word write to arrive after all four
chunks of the body write; that held in every run, but it is an observation on this
chip's routes, not a documented ordering guarantee. A costs three words of the 64-byte
record and about 10 cycles per record on each core, and is within 0.05 us per record of
B at both depths.

## Not tested

- Protocol B without the `fence ow, ow` between the body and the sequence word.
- Posted ordering only; one card (p150a), AICLK 1350 MHz, X280 at 1750 MHz, 120 cores
  and the two-panel split of this card.
- Records with work between them, that is, a core busy with something else when the
  record lands. Here the cores are always polling when a line arrives, which is the case
  most likely to show a partial record, but a busy core was not tried.
- Counters are kept in registers during the run and stored to L1 once at the end, not
  updated in L1 per record.
