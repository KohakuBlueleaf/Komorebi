# 18 thread split

Every core pulls its ranges of work from its X280 core, and each answer carries a
**word** beside the range: the overlay to run on it
([the pull's word](../../docs/reference/threads.md#the-pulls-word),
[X280 ports](../../docs/reference/ports.md#x280-ports),
[Overlays](../../docs/reference/overlays.md)). The core loads that overlay if it does
not hold it, calls it on the range, and tells its X280 core what kinds of units the
range held; the X280 thread decides each core's next overlay from those tells.

A unit of work is one result tile. Units come in runs of one kind (0, 1 or 2, mixed
from the run's number); each of the three overlays is fast on one kind and pays a
penalty (cycles spun) on the others:

- `plain` (a reader part): writes the result tile value by value; fast on kind 0;
- `sparse` (a reader part): writes it two values a word; fast on kind 1;
- `dense` (reader, compute and writer parts): the reader fills the input tile, the
  compute cores double it on the vector unit, the writer writes it; fast on kind 2.

The reader has two slots, fewer than the overlays, so changing overlays loads.

| Handler | Arguments | What a core does |
|---|---|---|
| `PASS` | a0 units, a1 the tiles' DRAM address, a2 the pass's first page, a3 the run, a4 the penalty, a5 the lines' DRAM address, a6 the pass, a7 1 for a pass that only pulls | pulls ranges until none is left; loads and calls the overlay each word names, hands dense ranges to compute and the writer, tells the X280 core the range's kinds; writes its line (units per overlay, loads, pulls, cycles in `next()`, its span) |

The thread (`thread/split.c`, on cluster 1, every core's port dealt over its four X280
cores) runs seven passes, each a `kmx_pull_run` with a `pick_word`; every core pulls
the same stripe of ranges in each:

| Pass | The word for a range |
|---|---|
| `plain`, `sparse`, `dense` | that overlay for everything |
| `host` | the overlay the host chose for the core beforehand: its stripe's commonest kind |
| `thread` | the thread's choice: the kind the core's newest tell counted most units of |
| `pick`, `word` | pull only, no work: a `pick` without a word, and a `pick_word` with a fixed one |

The host checks every result tile of the five work passes against numpy, and every
core's line: its units, the overlays it ran (all of one in a single pass, its own in
the host's), its pulls and words in the pull-only passes. It prints a table and one
JSON line.

## The pull, on both sides

```c
/* thread/split.c: the range and the overlay for it */
static uint64_t pick_word(int core, uint32_t index, uint64_t arg) {
    ...
    return KMX_PULL_WORD(range, word);
}
```

```c++
// kernels/reader.cpp
uint32_t first = 0, word = 0;
for (;;) {
    const uint32_t count = port.next(first, word);  // the range and its overlay
    if (count == 0) {
        break;
    }
    km::Overlay overlay(reader, word);
    line[TS_LINE_LOADS] += !overlay.held();
    if (!overlay.load()) {  // nothing moves when the slot holds it
        return;
    }
    overlay.call(range);  // ts_range{first, count, ...}
    port.tell(kinds, word, count);  // the range's kinds, a byte each
}
```

## Files

| File | What it is |
|---|---|
| `split.h` | what the thread, the kernels and the overlays share (plain C): the handler, the overlays, the buffers, a unit's kind, the thread's words, the passes, a line |
| `kernels/reader.cpp` | pulls, loads and calls the overlay each word names, tells the kinds, writes the line |
| `kernels/compute.cpp`, `writer.cpp` | take the reader's jobs; load (the meeting) and call the dense parts |
| `kernels/plain/`, `sparse/`, `dense/` | the overlays' parts |
| `thread/split.c` | the X280 thread: the passes and their picks |
| `thread_split.py` | declares the program, runs it, checks every tile and line, prints the table and one JSON line |

## What it needs

A Komorebi install and a card prepared as in [Running on a card](../../docs/install.md#running-on-a-card).

```
python thread_split.py --units 16384 --per 4 --run 64   # the numbers below
python thread_split.py --units 256 --pull-units 1024     # a quick check
```

`--penalty` (cycles a unit costs off its overlay's kind, default 20,000), `--slots` (the
reader's, default 2). The script exits non-zero if a tile or a line is wrong.

## Numbers

A Blackhole p150a at AICLK 1,350 MHz, all 120 cores, 16,384 units in ranges of 4, runs
of 64, a penalty of 20,000 cycles. Every tile and line exact.

| Pass | Whole pass (thread's clock) | Slowest core | Loads (most on a core) | Units off their kind |
|---|---|---|---|---|
| plain for everything | 4,888 µs | 4,864 µs | 120 (1) | 9,984 |
| sparse for everything (the best single) | 4,222 µs | 4,200 µs | 120 (1) | 11,456 |
| dense for everything | 4,820 µs | 4,810 µs | 120 (1) | 11,328 |
| the host's split, fixed per core | 3,934 µs | 3,901 µs | 49 (1) | 5,944 |
| the thread's choice per range | 3,158 µs | 3,147 µs | 173 (4) | not counted |

The thread's choice is 25% faster than the best single overlay and 20% faster than the
host's split: a core's stripe crosses several runs, and the thread follows them a few
ranges late (the pick for range i + 1 is made when range i arrives, from the tell of
range i - 1). Sparse is the fastest single overlay though plain is off fewer units: the
penalty dominates and sparse writes faster.

The pull with and without a word, 7,680 ranges of one unit and no work (30 cores to each
X280 core): 1,375 µs a pass without a word, 1,333 µs with one; 28,000 and 27,000 cycles
a `next()` (the X280 core serving 30 cores is the limit). In
`examples/10_x280_ports --pull`, 0.66 to 0.70 µs a pull before and after the word came.
With ranges of one unit (`--per 1 --run 32`) the thread's pass is bound by the pulls
(19,000 cycles a `next()`): reading each tell adds to every pick, and a range should be
long beside 30 picks.
