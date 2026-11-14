# 09 bucket sort

The first pass of a bucket sort, driven by an X280 thread ([X280 threads](../../docs/threads.md)):
the keys end up grouped by bucket, buckets in order, the keys of one bucket in input
order (a stable sort by bucket). The pass has three steps, and each runs where it
fits:

1. **COUNT**, on the Tensix group: every core counts its run of the keys per bucket and
   writes its row of the counts table (a counter of 4 bytes per core and bucket).
2. **PLAN**, on 1, 4 or 12 X280 cores: the thread adds the table up column by column
   (a total per bucket), walks the totals once (each becomes the bucket's start), and
   then gives each core's first key of each bucket its place: the bucket's start plus
   the counts of the cores before it. This is a prefix sum over the buckets and, inside
   a bucket, over the cores: serial in nature, and too irregular for the Tensix group.
3. **PLACE**, on the Tensix group: every core reads its row of the plan and writes each
   of its keys, in order, to its place in the output.

A bucket is a key's top `log2(buckets)` bits. The thread submits COUNT and PLACE as
commands and waits for each command's event; the host loads the program, starts it,
reads the output back and checks it against numpy (`reference.py`, written apart from
the card's code).

| Handler | Arguments | What a core does |
|---|---|---|
| `COUNT` (1) | a0 `keys_address`, a1 `key_count`, a2 `log2_buckets` and `blocks` << 8, a3 `banks` (8 bits per block), a4 to a6 the counts tables of the blocks | counts its run of keys per bucket in L1 (a word a bucket), then writes its row of the counts table to each block's cluster, in that cluster's bank |
| `PLACE` (2) | as `COUNT`, a4 to a6 the plan tables, with a7 `output_address` | reads its row of the plan into L1, then writes each key to its place in the output, 4 bytes at a time |

The group is cores 0 to `cores` - 1 of the grid; a core's run of the keys is
`km::partition(key_count, rank, count)`, the same split in the reader, the writer and
`reference.py` (the first `key_count % cores` cores take one key more).

    python bucket_sort.py [--keys 4194304] [--buckets 4096] [--x280-cores 12]
                          [--cores 0] [--repeats 1] [--one-bucket] [--seed 1]

`--buckets` is a power of 2 from 16 to 16,384; `--x280-cores` is 1, 4 or 12 (cluster 1's
core 0, cluster 1's four cores, or the twelve cores of clusters 1 to 3); `--cores 0` is
every Tensix core of the grid; `--repeats` runs the three steps that many times and
keeps the best time of each; `--one-bucket` puts every key in bucket 0.

Prints one line a step while it runs, then one JSON line: the keys, buckets and cores,
the code of each cluster's thread (`codes`), the best time of each step in ms (`ms`:
`count`, `plan`, `place`, `whole`), the keys the plan placed and the rounds done
(`placed`, `rounds`), the thread's whole time (`thread_ms`), the keys of the output
that differ from numpy's (`wrong`) and `ok`: every thread ran and returned code 0
without a library error, `placed` is the key count, `rounds` is `--repeats` and `wrong`
is 0. The script exits non-zero without it.

## The data

- **Keys and output** are tensors of `uint32` in DRAM, 1,024 keys to a 4 KiB page,
  interleaved over the banks; the host pads the last page with zeros, which no core
  reads.
- **The bucket** is one shift of the key. The compute kernel has nothing to do: the
  writer takes the shift as it reads each key from L1.
- **The tables** (counts and plan) are `cores` rows of `buckets` counters. Buckets go in
  chunks of 16, 64 bytes, one line of a row. With 12 X280 cores the chunks are split into
  three blocks, one per work cluster (`bs_block()` in `bucket_sort.h`, which the kernels
  and the thread both call), and each block's table is in its cluster's region in its
  own bank, so that no line and no NoC transfer holds two blocks' buckets.
- **The thread's regions** are `counts`, `plan`, `totals` and `peers`, each cluster with
  its own copy. `totals` holds a counter a bucket, with room for two rows, and the
  thread starts its counters half a row off the rows of `counts`; `counts` is whole
  cache ways plus half a way long. So the rows of `counts`, `plan` and `totals` read
  side by side do not fall on the same place in the cluster's cache (one way of 128 KiB: see
  [the rules of example 08](../08_parallel_thread/README.md#the-rules)).
- **Where an X280 core must flush or invalidate**: the Tensix cores write `counts` over
  the NoC, past the X280 core's cache, so the thread invalidates it before it reads;
  the Tensix cores read `plan` from DRAM, so the thread flushes it after it writes.

## The thread

The thread (`thread/plan.c`) is one image on each cluster taking part. Core 0 of
cluster 1 drives the program; with 12 cores clusters 2 and 3 call `kmx_serve` and run
the lead's `kmx_parallel_all` calls.

| Words | `key_count, log2_buckets, cores, keys_address, output_address, x280_cores, repeats, banks` (`cores` Tensix cores 0 to `cores` - 1; `banks`: the DRAM bank of work cluster c in bits 8 (c - 1)) |
|---|---|
| Regions | `counts`, `plan`, `totals`, `peers` (256 bytes: a line for each cluster, holding its table addresses) |
| Codes | 0 done; 1 bad words or regions; 2 a `kmx_parallel` call failed; 3 a `kmx_push` failed; 4 the command ring failed; 5 a command did not finish |
| Result words | the best time of COUNT, PLAN, PLACE and the whole round in timer ticks (50 per microsecond), the keys the plan placed, the rounds done |
| Progress | per round, 1 COUNT, 2 PLAN, 3 PLACE (4 more for each round after the first) |

The steps of one round:

| Step | Where | What |
|---|---|---|
| 0 | X280, before the first round (12 cores only) | clusters 2 and 3 push the addresses of their tables into the lead's `peers` |
| 1 COUNT | the Tensix group | the thread submits the command and waits for its event (a timeout of one second) |
| 2a totals | the planning cores | each core sums the rows of its chunks of buckets, row after row |
| 2b | clusters 2 and 3 | push their totals to the lead and flush them |
| 2c starts | core 0 of the lead alone | every bucket's total becomes its start, in one walk; the other blocks' starts are pushed back; the walk's sum is the keys placed, which must be the key count |
| 2d plan | the planning cores | each core's first key of a bucket gets its place, row after row; every row's lines are flushed |
| 3 PLACE | the Tensix group | the thread submits the command with the plan's table addresses and waits |

## The kernels

- `reader.cpp`: on every COUNT and PLACE command, reads the core's pages of keys into a
  buffer of two pages (index 0), a page ahead of the writer.
- `writer.cpp`: COUNT counts in a word a bucket of its own L1 (buffer index 16), then
  writes its row of the counts. PLACE first reads its row of the plan, then writes each
  key to its place as a 4-byte NoC write from a 16-byte slot (64 slots): the slot and
  the place in DRAM share their low 4 address bits, the NoC's rule. A place outside the
  output is never written. Each command is reported done only after its writes are
  acknowledged, so the next reader of them, the thread or the host, may look at once.
- `compute.cpp`: takes every command and returns at once.

## What it needs

A Komorebi install, a card prepared as in
[Running on a card](../../docs/install.md#running-on-a-card), and the X280 threads
(clusters 1 to 3 for `--x280-cores 12`).

```
cd examples/09_bucket_sort
python bucket_sort.py
python bucket_sort.py --keys 100003 --buckets 64 --x280-cores 4 --cores 7
python bucket_sort.py --one-bucket --repeats 3
```

A key count that does not divide by `--cores` gives some cores one key more.

## Measured

A Blackhole p150a, card 0, 2026-11-14, AICLK 1,350 MHz (sampled once a second
beside the run: 1,350 MHz in all 19 samples from the card's open to the end), X280 at the
session's 1000 MHz, host load 3.3 to 3.6, the defaults (`python bucket_sort.py`: 4,194,304
keys, 4,096 buckets, 120 Tensix cores, 12 X280 cores, one round). The output matched
numpy in every key (`wrong` 0), every cluster's thread returned code 0 without a
library error, and the plan placed all 4,194,304 keys:

| Step | Time, ms |
|---|---|
| COUNT | 0.697 |
| PLAN | 2.380 |
| PLACE | 3.710 |
| the round | 6.786 |

The counts table is 120 rows of 4,096 counters, 1.97 MB. COUNT took 0.697 ms for the
16.8 MB of keys (24 GB/s, computed from the two), and the round, 6.786 ms, is 618
million keys a second. The thread's whole time (`thread_ms`) was 41.79 ms, and the
script's, from its start to its JSON line (`wall_s`), 33.0 s: the device open, the
upload and the read-back of the output, 16.8 MB in 8 chunks of 2 MB.

One run: the times of one round are not a statistic.

## Files

| File | What it is |
|---|---|
| `bucket_sort.py` | the host script: the tensors, the program, the thread's words and regions, the numpy check, the JSON line |
| `bucket_sort.h` | what the kernels and the thread share: the handlers, the table chunks and blocks, the thread's words and result words |
| `kernels/reader.cpp` | the reader: the core's keys from DRAM into L1, a page at a time |
| `kernels/writer.cpp` | the writer: COUNT and PLACE |
| `kernels/compute.cpp` | takes every command and does nothing |
| `thread/plan.c` | the thread: the commands, the plan on 1, 4 or 12 X280 cores |
| `reference.py` | the numpy reference: the partitions, the counts, the plan and the output |
