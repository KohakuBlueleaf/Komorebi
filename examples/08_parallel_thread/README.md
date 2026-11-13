# 08 parallel thread

Parallel code in an X280 thread: the work cluster's four harts run one function each
through `kmx_par`, on three patterns. Each pattern runs with one hart and with four
(the same function, given one share or four), is timed on the card, and is checked
exactly against numpy:

| Pattern | What each hart does | Merged by hart 0 |
|---|---|---|
| map | `out[i] = f(in[i])` over its chunks, `f(x) = (x ^ x >> 15) * 2246822519 + 1`; also with the vector unit, and `f` applied 16 times (heavy) | nothing |
| reduction | the sum of its chunks in a register, stored once into its own line | the four partial sums |
| prefix sum | pass 1: the sum of each of its chunks; pass 2: each chunk's exclusive prefix from its base | between the passes: the chunk sums into bases |

The grid takes no part: its kernels do nothing. The thread is
`thread/parallel.c`, with the vector loop in `thread/vmap.c`.

    python parallel_thread.py [--words 262144] [--repeats 3] [--rounds 16] [--seed 1]

Prints one JSON line: the times in microseconds (`us`, the best of `--repeats` runs,
each from a cold cache), the thread's whole time (`thread_ms`), whether each output is
exact (`exact`), four harts against one (`four_harts_x`), and `ok`: the thread
returned code 0 without a library error and every output is exact. The script
exits non-zero without it.

## One hart against four

A Blackhole p150a, work cluster 1, X280 at the session's 1000 MHz, 262,144 words
(1 MiB) per array, best of 3; every output exact:

| Pattern | 1 hart, ms | 4 harts, ms | 4 against 1 | Bound by |
|---|---|---|---|---|
| map | 2.85 | 1.66 | 1.72 | the bank: 2 MiB moved in 1.66 ms, 1.27 GB/s |
| map, vector loop | 1.66 | 1.60 | 1.04 | the bank |
| map, `f` 16 times | 26.46 | 6.69 | 3.96 | arithmetic |
| reduction | 1.81 | 0.82 | 2.21 | the bank: 1 MiB read in 0.82 ms, 1.28 GB/s |
| prefix sum, two passes | 4.82 (one pass: 3.52) | 2.48 | 1.42 against one pass | the bank: 3 MiB moved |

With 65,536 words (256 KiB), about the same ratios: 0.72 / 0.41 ms, 0.41 / 0.40,
6.60 / 1.67, 0.45 / 0.21, 0.94 (one pass) / 0.62. Four harts pay where a word costs
arithmetic; where it only moves, the cluster's DRAM rate (about 1.3 GB/s at 1000 MHz)
is the limit, and one hart with the vector loop reaches it alone.

## The rules

**Who calls what.** `kmx_main` and `kmx_par` run on hart 0; `kmx_par(fn, arg)` runs
`fn(hart, arg)` on harts 0 to 3 and returns when all four have returned (`kmx.c`):

| Call | Hart 0 | Harts 1 to 3 (inside `fn`) |
|---|---|---|
| `kmx_args_get`, `kmx_region`, `kmx_time`, `kmx_stop_requested` | yes | yes |
| `kmx_take`, `kmx_give` | yes: outside `kmx_par` the sweep is split over the four harts; inside, this hart alone | yes, this hart alone |
| `kmx_par` | yes, not inside `fn` | no: the library error `par`, `KMX_STOPPED`, and the run ends failed |
| `kmx_issue`, `kmx_wait`, `kmx_poll` | yes | no: as `kmx_par` |
| `kmx_progress`, `kmx_result` | yes | best not: they write the one result block, without a lock |

`fn` runs on hart 0 too, as share 0: hart 0 does a quarter of the work and then waits.
A call to `kmx_par` with an empty `fn` costs 0.59 microseconds (1,000 calls in 591),
so a share should be worth tens of microseconds at least.

**Shares and results.** Pass one argument; every hart computes its share from its
number. `arg` points into hart 0's memory (here a struct on its stack), which all four
harts address alike. A result comes back in a slot of its own per hart (here
`g_part[hart]`, a global), which hart 0 reads after `kmx_par` returned: `kmx_par`
fences, so what `fn` stored is visible then. Read the argument's fields into locals
before the loop: a `uint32_t` store into the output may alias a `uint32_t` field, and
the compiler then reads the field again on every word (the map took 2.46 ms instead
of 0.72 ms on 65,536 words until `map_range` did this).

**Chunks, not quarters, and where arrays lie.** The cluster's L3 is one cache way of
128 KiB (`experiments/x280_threads/`): two addresses a multiple of 128 KiB apart take
the same place in it, and a loop that reads one such address and writes the other
runs many times slower, as one way would explain: the map with its output a whole
number of ways from its input (`map_aliased`) took 90.6 ms instead of 2.85 ms. So:
- the host script makes every array a whole number of 128 KiB and the input 64 KiB
  longer, so every output starts half a way from the input (`program()`);
- the harts share the work in chunks of 4 KiB taken in turn (hart `h` takes chunks
  `h`, `h + 4`, ...), so the four move through one window together. Quarters put four
  streams a quarter of the array apart; with 65,536 words a quarter is 64 KiB, one
  hart's reads fall on another's writes, and the map took 6.08 ms instead of 0.41
  (`map_quarters`; with 262,144 words, 1.86 against 1.66).

Chunks also make the prefix sum a two-pass scan over chunks: pass 1 stores each
chunk's sum, hart 0 turns the sums into bases (one add per chunk), pass 2 writes each
chunk's prefix from its base.

**False sharing.** A cache line is 64 bytes (the hand-over sweeps by 64). Four harts
adding into four 64-bit counters in one line of a region took 17.88 ms; the same
counters a line apart, 2.54 ms (`sum_4_packed`, `sum_4_padded`). A slot each hart
writes often gets a line of its own; a slot written once per call, like
`g_part[hart]`, does not need one, but it costs nothing.

**Hand-over.** `kmx_take` before reading what the grid or the host wrote, `kmx_give`
after writing what they will read; inside the cluster the four harts are coherent and
need neither. Both sweep every line of the region through the cluster's flush
register, about 18 ns per line of region, clean or dirty: 312 microseconds
for the 1.06 MiB input, 296 for a 1 MiB output just written, the same on one hart as
split over four (`take_1`, `take_4`, `give_1`, `give_4`). Call them once per region and
hand-over, from hart 0 outside `kmx_par`; the example sweeps before every timed run
only to start it cold.

**Vector.** The library's start-up turns the vector unit on, but the thread build's
`-march` is `rv64gc`: the compiler emits no vector code by itself. A function asks for
it with `__attribute__((target("arch=+v")))` and `<riscv_vector.h>` intrinsics
(`thread/vmap.c`). Keep such functions in a file of their own: the toolchain's GCC
15.1 stops with an internal compiler error under `-g` (which the build uses) when a
function without the attribute follows one with it in the same file. The vector map
on one hart (1.66 ms) is as fast as four scalar harts, and four vector harts gain
nothing more: the bank is the limit.

**The deadline.** `deadline=` (seconds, at most 60) counts from the thread's start.
`kmx_par` checks it while it waits and returns `KMX_LATE`, with the library error
`late`; harts 1 to 3 stop serving at it. Neither interrupts a running `fn`: a loop in
`fn` that may be long polls `kmx_stop_requested()`, which harts 1 to 3 may call. The
whole thread takes 0.55 s of a 30 s deadline (`thread_ms`).

## The thread

| Words | `n, repeats, rounds` (the heavy map's `f` count) |
|---|---|
| Regions | `in`, `map_1`, `map_4`, `map_quarters`, `map_aliased` (written 64 KiB in), `vmap_1`, `vmap_4`, `heavy_1`, `heavy_4`, `seq`, `prefix_1`, `prefix_4`, `counters` |
| Codes | 0 done; 1 bad words or regions; 2 a `kmx_par` call was late or stopped |
| Result words | the 20 times of `TIMES` in `parallel_thread.py`, in timer ticks (50 per microsecond), then the four sums as 64-bit pairs |
| Progress | 1 to 8 the maps, 9 to 12 the reductions, 13 to 15 the prefix sums, 16 the hand-overs, 17 the `kmx_par` calls |

## Files

| File | What it is |
|---|---|
| `thread/parallel.c` | the thread: the patterns, their shares, the timings |
| `thread/vmap.c`, `thread/vmap.h` | the map's vector loop |
| `kernels/` | the grid's kernels, which do nothing |
| `parallel_thread.py` | the host script: the regions, the run, the numpy checks |
