# 08 parallel thread

Parallel code in an X280 thread: the work cluster's four cores run one function each
through `kmx_parallel`, on three patterns. Each pattern runs with one core and with four
(the same function, given one partition or four), is timed on the card, and is checked
exactly against numpy:

| Pattern | What each core does | Merged by core 0 |
|---|---|---|
| map | `output[word] = mix(input[word])` over its chunks, `mix(x) = (x ^ x >> 15) * 2246822519 + 1`; also with the vector unit, and `mix` applied 16 times (heavy) | nothing |
| reduction | the sum of its chunks in a register, stored once into its own line | the four partial sums |
| prefix sum | pass 1: the sum of each of its chunks; pass 2: each chunk's exclusive prefix from its base | between the passes: the chunk sums into bases |

The grid takes no part: its kernels do nothing. The thread is
`thread/parallel.c`, with the vector loop in `thread/vmap.c`.

    python parallel_thread.py [--words 262144] [--repeats 3] [--rounds 16] [--seed 1]
                              [--twelve | --fetch]

`--twelve` runs the map, the heavy map and the prefix sum on 1, 4, 8 and 12 cores of
the three work clusters instead ([Twelve cores](#twelve-cores)). `--fetch` puts the
input in DRAM bank 7, which cluster 1 cannot read, and the thread fetches it first
([The input in another bank](#the-input-in-another-bank)).

Prints one JSON line: the times in microseconds (`microseconds`, the best of `--repeats` runs,
each from a cold cache), the thread's whole time (`thread_ms`), whether each output is
exact (`exact`), four cores against one (`four_cores_x`), and `ok`: the thread
returned code 0 without a library error and every output is exact. The script
exits non-zero without it.

## One core against four

A Blackhole p150a, work cluster 1, X280 at the session's 1000 MHz, 262,144 words
(1 MiB) per array, best of 3; every output exact:

| Pattern | 1 core, ms | 4 cores, ms | 4 against 1 | Bound by |
|---|---|---|---|---|
| map | 2.85 | 1.66 | 1.72 | the bank: 2 MiB moved in 1.66 ms, 1.27 GB/s |
| map, vector loop | 1.66 | 1.60 | 1.04 | the bank |
| map, `mix` 16 times | 26.46 | 6.69 | 3.96 | arithmetic |
| reduction | 1.81 | 0.82 | 2.21 | the bank: 1 MiB read in 0.82 ms, 1.28 GB/s |
| prefix sum, two passes | 4.82 (one pass: 3.52) | 2.48 | 1.42 against one pass | the bank: 3 MiB moved |

With 65,536 words (256 KiB), about the same ratios: 0.72 / 0.41 ms, 0.41 / 0.40,
6.60 / 1.67, 0.45 / 0.21, 0.94 (one pass) / 0.62. Four cores pay where a word costs
arithmetic; where it only moves, the cluster's DRAM rate (about 1.3 GB/s at 1000 MHz)
is the limit, and one core with the vector loop reaches it alone.

## The rules

**Who calls what.** `kmx_main` and `kmx_parallel` run on core 0; `kmx_parallel(fn, arg)` spawns
`fn(core, arg)` on cores 1 to 3, runs it on core 0 and joins the three (`kmx.c`):

| Call | Core 0 | Cores 1 to 3 (inside `fn`) |
|---|---|---|
| `kmx_args_get`, `kmx_region`, `kmx_time`, `kmx_stop_requested` | yes | yes |
| `kmx_invalidate`, `kmx_flush` | yes: outside `kmx_parallel` the sweep is split over the four cores; inside, this core alone | yes, this core alone |
| `kmx_parallel`, `kmx_spawn`, `kmx_join` | yes, not inside `fn` | no: the library error `parallel`, `KMX_STOPPED`, and the run ends failed |
| `kmx_submit`, `kmx_wait`, `kmx_poll` | yes | no: as `kmx_parallel` |
| `kmx_progress`, `kmx_result` | yes | best not: they write the one result block, without a lock |

The calls across clusters are in [X280 threads](../../docs/threads.md#twelve-cores).
`fn` runs on core 0 too, as partition 0: core 0 does a quarter of the work and then waits.
A call to `kmx_parallel` with an empty `fn` costs 0.71 microseconds (1,000 calls in 709;
0.59 before `kmx_parallel` was built on spawn and join), so a partition should be worth tens
of microseconds at least.

**Partitions and results.** Pass one argument; every core computes its partition from its
number. `arg` points into core 0's memory (here a struct on its stack), which all four
cores address alike. A result is returned in a slot of its own per core (here
`core_sums[core]`, a global), which core 0 reads after `kmx_parallel` returned: `kmx_parallel`
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
- the cores split the work in chunks of 4 KiB taken in turn (core 0 takes chunks
  0, 4, ...; core 1 takes 1, 5, ...), so the four move through one window together. Quarters put four
  streams a quarter of the array apart; with 65,536 words a quarter is 64 KiB, one
  core's reads fall on another's writes, and the map took 6.08 ms instead of 0.41
  (`map_quarters`; with 262,144 words, 1.86 against 1.66).

Chunks also make the prefix sum a two-pass scan over chunks: pass 1 stores each
chunk's sum, core 0 turns the sums into bases (one add per chunk), pass 2 writes each
chunk's prefix from its base.

**False sharing.** A cache line is 64 bytes (flush and invalidate sweep by 64). Four cores
adding into four 64-bit counters in one line of a region took 17.88 ms; the same
counters a line apart, 2.54 ms (`sum_4_packed`, `sum_4_padded`). A slot each core
writes often gets a line of its own; a slot written once per call, like
`core_sums[core]`, does not need one, but it costs nothing.

**Flush and invalidate.** `kmx_invalidate` before reading what the grid or the host wrote, `kmx_flush`
after writing what they will read; inside the cluster the four cores are coherent and
need neither. Both sweep every line of the region through the cluster's flush
register, about 18 ns per line of region, clean or dirty: 312 microseconds
for the 1.06 MiB input, 296 for a 1 MiB output just written, the same on one core as
split over four (`invalidate_1`, `invalidate_4`, `flush_1`, `flush_4`). Call them once per region and
direction, from core 0 outside `kmx_parallel`; the example sweeps before every timed run
only to start it cold.

**Vector.** The library's start-up turns the vector unit on, but the thread build's
`-march` is `rv64gc`: the compiler does not auto-vectorize. A function asks for
it with `__attribute__((target("arch=+v")))` and `<riscv_vector.h>` intrinsics
(`thread/vmap.c`). Keep such functions in a file of their own: the toolchain's GCC
15.1 stops with an internal compiler error under `-g` (which the build uses) when a
function without the attribute follows one with it in the same file. The vector map
on one core (1.66 ms) is as fast as four scalar cores, and four vector cores gain
nothing more: the bank is the limit.

**The deadline.** `deadline=` (seconds, at most 60) counts from the thread's start.
`kmx_parallel` checks it while it waits and returns `KMX_LATE`, with the library error
`late`; cores 1 to 3 stop serving at it. Neither interrupts a running `fn`: a loop in
`fn` that may be long polls `kmx_stop_requested()`, which cores 1 to 3 may call. The
whole thread takes 0.55 s of a 30 s deadline (`thread_ms`).

## Twelve cores

`thread/clusters.c` runs on clusters 1 to 3 (`prog.thread(clusters=(1, 2, 3))`). Cluster
1 leads; 2 and 3 call `kmx_serve`. With eight or twelve cores the chunks are split into
a contiguous block per cluster (two or three), and within a block the cluster's four
cores take chunks in turn. Each cluster reads and writes its own copy of the regions,
in its own bank, so data moves between banks by `kmx_push` only:

- map: cluster 1 pushes each other block's input into that cluster's `in`;
  `kmx_parallel_all` maps (each cluster invalidates its block of `in` first); a second call has
  clusters 2 and 3 push their outputs into cluster 1's, which invalidates them;
- prefix sum: the input out as for the map; pass 1 (chunk sums, one 64-byte line per
  chunk so each block's sums push whole); clusters 2 and 3 push their sums to cluster
  1 and flush them (a dirty line would later overwrite the bases); cluster 1 turns
  the sums into bases and pushes each block's bases out; pass 2; the outputs back.

The 1-core output is the reference: the thread counts the words of every other run
that differ from it, and the host checks the 1-core and 12-core outputs against numpy.
On a Blackhole p150a, X280 at the session's 1000 MHz, best of 3, every output exact,
times in ms, pushes included:

| Words | Pattern | 1 core | 4 | 8 | 12 |
|---|---|---|---|---|---|
| 16,384 | map | 0.18 | 0.11 | 0.12 | 0.13 |
| | map, `mix` 16 times | 1.66 | 0.42 | 0.31 | 0.31 |
| | prefix sum | 0.31 | 0.13 | 0.18 | 0.18 |
| | the map's input push | - | - | 0.03 | 0.05 |
| 65,536 | map | 0.73 | 0.42 | 0.42 | 0.38 |
| | map, `mix` 16 times | 6.61 | 1.68 | 1.06 | 0.88 |
| | prefix sum | 1.23 | 0.66 | 0.53 | 0.48 |
| | the map's input push | - | - | 0.11 | 0.15 |
| 262,144 | map | 2.89 | 1.66 | 1.94 | 1.61 |
| | map, `mix` 16 times | 26.44 | 6.68 | 4.46 | 3.38 |
| | prefix sum | 4.90 | 2.57 | 2.46 | 2.07 |
| | the map's input push | - | - | 0.54 | 0.62 |
| 1,048,576 | map | 11.56 | 6.63 | 7.35 | 6.42 |
| | map, `mix` 16 times | 105.75 | 26.71 | 17.63 | 13.21 |
| | prefix sum | 19.70 | 10.21 | 9.18 | 7.79 |
| | the map's input push | - | - | 1.71 | 2.34 |

Four cores are one cluster and `kmx_parallel`; eight and twelve pay for the pushes: the
lead's input push reads its bank through the cache, at about 1.2 GB/s. Twelve cores
lose to four on small inputs (at 16,384 words the map and the prefix sum; the heavy
map still gains), and where the work only moves memory they gain little at any size (the map: 6.42 against
6.63 ms at a million words). The prefix sum gains 1.3 times at a million words, the
heavy map, which computes, 2.0 times; a step that wants twelve cores should compute
more per byte it moves, or find its input already in each bank.

## The input in another bank

With `--fetch` the host writes the input into bank 7 (the pages of a DRAM tensor with
16 KiB pages that lie in bank 7, one after another), the thread's `in` starts zero, and
before the patterns the thread copies it in through four Tensix cores of the grid:

```c
const kmx_dram from = {words[WORD_FETCH_ADDRESS], words[WORD_FETCH_BANK], 0, 0};
kmx_fetch(REGION_INPUT, 0, &from, bytes, 0, FETCH_CORES, kmx_time() + KMX_TICKS_PER_SECOND, 0);
```

On a Blackhole p150a, card 1, X280 at 1000 MHz, AICLK 1350 MHz, 262,144 words (1 MiB),
best of 3, host load 5.0 to 5.8, 2026-11-14; every output exact against numpy in both:

| | input in bank 6 (the example's own) | input in bank 7, fetched first |
|---|---|---|
| the fetch (`kmx_fetch`, 4 cores, 1 MiB) | - | 0.49 ms |
| prefix sum, two passes, 4 cores | 2.49 ms | 2.48 ms |
| prefix sum with its input's way in | 2.49 ms | 2.97 ms |
| the same input read by the X280 itself from bank 7 (loads over the NoC, 0.024 GB/s) | - | 44.1 ms |

The fetch's 0.49 ms is mostly the two sweeps of the region's cache lines around the
copy; the cores' copy of the MiB takes 0.02 ms ([bank_fetch](../../experiments/bank_fetch/README.md)).

## The thread

| Words | `array length, repeats, rounds` (how many times the heavy map applies `mix`), then with `--fetch` the input's bank and bank address |
|---|---|
| Regions | `in`, `map_1`, `map_4`, `map_quarters`, `map_aliased` (written 64 KiB in), `vmap_1`, `vmap_4`, `heavy_1`, `heavy_4`, `sequence`, `prefix_1`, `prefix_4`, `counters` |
| Codes | 0 done; 1 bad words or regions; 2 a `kmx_parallel` call was late or stopped; 3 the fetch failed |
| Result words | the 20 times of `TIMES` in `parallel_thread.py`, in timer ticks (50 per microsecond), then the four sums as 64-bit pairs, then the fetch's time (0 without `--fetch`) |
| Progress | 1 to 8 the maps, 9 to 12 the reductions, 13 to 15 the prefix sums, 16 the flushes and invalidates, 17 the `kmx_parallel` calls |

## Files

| File | What it is |
|---|---|
| `thread/parallel.c` | the thread: the patterns, their partitions, the timings |
| `thread/clusters.c` | the `--twelve` thread: three patterns on 1 to 12 cores |
| `thread/vmap.c`, `thread/vmap.h` | the map's vector loop |
| `kernels/` | the grid's kernels, which do nothing |
| `parallel_thread.py` | the host script: the regions, the run, the numpy checks |
