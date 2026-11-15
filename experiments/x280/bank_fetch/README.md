# Bank fetch

How fast does an X280 thread get data that lies in a DRAM bank its work cluster cannot
read? A cluster's cache reaches only its own bank (6 for cluster 1, 7 for clusters 2
and 3). Its own loads over the NoC move 0.024 GB/s. With `kmx_fetch` the thread sends
one command and Tensix cores copy the range, DRAM to DRAM, into the thread's region.

## Results

Blackhole p150a, card 1, slow dispatch, AICLK 1350 MHz (sampled; 1343 at least in the
64 MiB run), X280 at 1350 MHz, host load 3.0 to 3.2, 2026-11-14. Bank 0 to work
cluster 1's region in bank 6 on four Tensix cores, best of 5 (1 MiB) or 3 (16 and
64 MiB); every way's bytes checked (a hash of the whole region in the thread against
numpy, and at 1 MiB the region read back whole). The row of `kmx_fetch` with a sweep
is the call before it evicted: card 0, X280 at 1000 MHz, host load 4.2 to 4.9, the
same day.

| Way | 1 MiB | 16 MiB | 64 MiB |
|---|---|---|---|
| the copy (the command's event: dispatcher submit to completion) | 19.6 µs | 269 µs | 1.06 ms, 63.4 GB/s |
| `kmx_fetch_clean`, the whole call (nothing dropped) | 21.8 µs, 48.2 GB/s | 271 µs, 61.9 GB/s | 1.07 ms, 62.7 GB/s |
| `kmx_fetch`, the whole call (`kmx_evict` before the copy) | 40.2 µs, 26.1 GB/s | 289 µs, 58.0 GB/s | 1.08 ms, 62.1 GB/s |
| `kmx_fetch` with one sweep before the copy (the call before) | 192 µs, 5.46 GB/s | 2.99 ms, 5.61 GB/s | 11.9 ms, 5.62 GB/s |
| `kmx_fetch` and `kmx_invalidate_at` of the range (one sweep after it) | 211 µs, 4.98 GB/s | 3.01 ms, 5.58 GB/s | 12.0 ms, 5.61 GB/s |
| a sweep of the range (`kmx_flush_at`), four X280 cores | 219 µs | 2.77 ms | 10.9 ms |
| the same sweep on core 0 alone | 171 µs | 2.72 ms | 10.9 ms |

- **The copy.** One core reads and writes 16 KiB chunks at about 16.7 GB/s (64.6 µs,
  1.01 ms and 4.02 ms); four cores reach the bank's rate (about 63 GB/s), and twelve
  gain nothing more (19.2 µs, 265 µs, 1.05 ms): card 0's run.
- **The sweep.** A sweep writes one Flush64 per line to the cluster's cache controller:
  10.4 ns a line, 6.1 GB/s, the same on one core as on four. The controller takes one
  line at a time, so the sweep costs the range's lines and nothing spreads it.
- **The call.** `kmx_fetch` drops the range's lines from the cache before the copy: a
  dirty line written back later would overwrite the new bytes, and a stale line would
  be read in place of them. From 160 KiB on it calls `kmx_evict`, below that it sweeps
  the range ([the cluster's cache](#the-clusters-cache): the two cost the same at
  about 2,540 lines). Nothing on the cluster touches the range while the cores copy,
  so nothing follows the copy; the thread's first read of a line takes the new bytes.
  The call then costs the copy and 20 µs: 1.08 ms for 64 MiB against 11.9 ms with the
  sweep. When the range has no line in the cache (not read or written since the load,
  its last flush or the last fetch into it), `kmx_fetch_clean` drops nothing. In this
  run the zeroed region's lines were dirty before each `kmx_fetch` and flushed before
  each `kmx_fetch_clean`.
- **The store.** `kmx_store` of 1 MiB from the region back to bank 0 on four cores:
  0.19 to 0.24 ms (one sweep, then the copy), exact: card 0's run. From 160 KiB on a
  store evicts instead of sweeping ([below](#the-clusters-cache)): an evicted dirty
  line is written back, which is what the cores need before they read the range.

On card 1 the same day, the X280 reading the range itself (8-byte loads through a NoC
window) took 44.1 ms for 1 MiB (0.024 GB/s), and `kmx_push` of 1 MiB from cluster 1 to
cluster 2's region (bank 7) on four X280 cores 477 µs (2.20 GB/s): rows from
`experiments/x280/work_clusters/clusters.py push --cluster 1 --bank 7 --bytes 1048576 --pull
1048576` (its pull) and example 08's `--twelve` run (its input push to cluster 2).

## Over islands

`islands.py` opens one session over cards 0 and 1 (two islands, four links) and makes a
64 MiB tensor of 4 KiB pages held by island 0 and the same bytes held by island 1.
Island 0's work cluster 1 thread (`thread/islands.c`) fetches each size from both into
its region with `kmx_fetch` on 16 cores (best of 3), then stores the region with
`kmx_store` into a tensor of each island. The thread names the tensor and the range
only: the copy finds where the pages are. Every word is checked: the region's hash in
the thread, the region and all four stored tensors read back whole. Cards 0 and 1,
2026-11-14, one run:

| Way | 1 MiB | 64 MiB |
|---|---|---|
| `kmx_fetch` of island 0's tensor | 51 µs, 20.7 GB/s | 1.30 ms, 51.6 GB/s |
| `kmx_fetch` of island 1's tensor | 204 µs, 5.15 GB/s | 11.0 ms, 6.10 GB/s |
| `kmx_store` into island 0's tensor | 47 µs | 1.30 ms |
| `kmx_store` into island 1's tensor | 197 µs | 10.7 ms |

The local rows are the table's `kmx_fetch` above (40.2 µs and 1.08 ms on four cores).
Across, each copy core then asked link core 0 for its chunk's pages (a get line, or a
put line for a store), and one CPU on one link moved them all: about 6 GB/s. Now each
copy core asks each run the other island holds as one line straight between the two
DRAMs (all in flight, waited for once), served by every page CPU: 2026-11-15, one
run, every word exact:

| Way | 1 MiB | 64 MiB |
|---|---|---|
| `kmx_fetch` of island 0's tensor | 51 µs | 1.303 ms |
| `kmx_fetch` of island 1's tensor | 107 µs | 1.125 ms, 59.7 GB/s |
| `kmx_store` into island 1's tensor | 102 µs | 1.118 ms |
 Pages crossing are at most one fabric
packet (4352 bytes); the tensors are made before the load (the load writes the table
of where each tensor's pages are). On one island the copy is what it was: the check
for a placed tensor ends at the island count.

## The cluster's cache

`python evict.py shape` has work cluster 1's thread read the cache controller's
registers (reads only) and the cached Zero Device. On card 1, 2026-11-14:

- `CCACHE0_CONFIG` 0x06091004: 4 banks, 16 ways, 512 sets, 64-byte lines, so 128 KiB a
  way. `CCACHE0_WAYENABLE` 0: one way is cache, the other fifteen pieces are the
  uncached scratchpad the image runs from.
- Every one of the 38 way masks is 0xffff, the four cores' L2s (masters 34 to 37)
  included: they evict from way 0, the only cache way. The cache is inclusive of the
  cores' L2 (each 2 banks, 8 ways, 256 sets: 128 KiB) and L1 data caches, so a line
  that leaves way 0 leaves the cluster.
- The cached Zero Device spans 0x0A00_0000 to 0x0A1F_FFFF. The runtime keeps its first
  0x10C0 bytes for pinned input; the 2,092,864 bytes after them read as zeros.

So reading 2,048 lines that fill every set of the one way, 128 KiB of contiguous
addresses, removes every other line from the cluster's caches.

`kmx_evict` does that with one load per line of the Zero Device from 0x0A02_0000, the
second 128 KiB of it. `python evict.py time` times it on core 0 against the same loads
of a 128 KiB pad in the cluster's own DRAM, 20 times each after the thread read 1 MiB
of other lines (the cache full of clean lines), wrote them (full of dirty lines), or
had just evicted (card 1, 2026-11-14, host load 4.9):

| The cache before | `kmx_evict` (the Zero Device) | the pad in DRAM |
|---|---|---|
| clean lines | 21.4 to 21.8 µs | 100.1 to 101.0 µs |
| dirty lines | 24.4 to 26.4 µs | 109.5 to 111.4 µs |
| just evicted | 18.2 µs | 18.4 µs |

The Zero Device answers from the cluster itself; the pad's lines come from DRAM. A
sweep costs 10.4 ns a line, so eviction is cheaper from about 2,540 lines (159 KiB)
on.

`python evict.py proof` tries to make a stale line survive. Each round the thread
writes a new pattern (different in every round and word) into a source region of its
bank and sweeps it out, reads the whole range (the cache then holds its last lines),
writes every fifth line of it (dirty lines), evicts, has four Tensix cores copy the
source into the range (`kmx_fetch_clean`), evicts again and compares every word of the
range with the pattern. The variants:

- `plain`: as above.
- `lead`: the thread also reads the 256 KiB before the range, up to its first byte,
  just before the first eviction, so the prefetcher runs ahead into the range.
- `readers`: cores 1 to 3 read 4 MiB of other memory the whole time.
- `one_evict`: as `lead`, without the second eviction: the order of `kmx_fetch`.
- `no_evict`: as `lead`, without any eviction: the control, which must find stale
  lines.

Card 1, 2026-11-14, host load 2.9 to 7.3; every variant exact (no wrong word):

| Range | plain | lead | readers | one_evict | no_evict (control) |
|---|---|---|---|---|---|
| 4 KiB | 300 rounds | 300 | 300 | 300 | not run |
| 1 MiB | 300 | 300 | 300 | 300 | 1,800 stale lines in 300 rounds |
| 64 MiB | 180 | 180 | 90 | 180 | 4,320 stale lines in 180 rounds |

The control's stale lines are the first lines of the range (6 a round at 1 MiB, 24 at
64 MiB): the prefetcher fetched them during the lead read, after the lead read had
pushed every other line of the range out. One eviction before the copy removes them.

The eviction comes before the copy: a dirty line of the range that left the cache
during or after the copy would be written back over the bytes the cores wrote. A line
the cache holds after the eviction can only come from a read of the range, and nothing
on the cluster reads it while the cores copy, so `one_evict` needs no second eviction.

`kmx_fetch` and `kmx_store` evict only when the thread's start found the cache this
shape: the library reads `CCACHE0_WAYENABLE` (0) and core 0's way mask (way 0 in it)
once, and sweeps otherwise.

`python evict.py store` proves the store the same way. Each round the thread writes a
new pattern over the whole range (the cache then holds its last 2,048 lines, dirty),
the four cores copy the range to a second region of the bank, and the thread sweeps
that region's lines and compares every word with the pattern. Variants: `store`
(`kmx_store`, which evicts), `sweep` (a sweep of the range then the copy: the store
as it was) and `bare` (the copy alone, nothing written back: the control). Card 1,
2026-11-14, host load 2.8 to 2.9; the call's time best to worst:

| Range | `store` | `sweep` | `bare` (control) |
|---|---|---|---|
| 1 MiB, 300 rounds | exact, 64.5 to 69.4 µs | exact, 309 to 311 µs | 613,200 wrong lines, 39.8 to 41.7 µs |
| 64 MiB, 200 rounds | exact, 2.30 to 2.32 ms | exact, 13.27 to 13.35 ms | 408,800 wrong lines, 2.28 to 2.31 ms |

The control's wrong lines are 2,044 a round, the dirty lines the cache held: the cores
read the range's previous pattern there. The copy reads and writes the same bank, so
it takes about twice a fetch's 64 MiB.

## Running it

```sh
python fetch.py --bytes 4096 --cores 1
# card 0's run: one, four and twelve cores, the store
python fetch.py --bytes 1048576 --cores 1 4 12 --repeats 5 --store 1048576
python fetch.py --bytes 16777216 --cores 1 4 12 --repeats 3 --store 1048576
python fetch.py --bytes 67108864 --cores 1 4 12 --repeats 3 --store 1048576
# the table's run
python fetch.py --bytes 1048576 --cores 4 --repeats 5
python fetch.py --bytes 16777216 --cores 4 --repeats 3
python fetch.py --bytes 67108864 --cores 4 --repeats 3
# over islands (cards 0 and 1): 1 MiB and 64 MiB, 16 cores, best of 3
python islands.py
python evict.py shape
python evict.py time --repeats 20
python evict.py proof --bytes 4096 --repeats 300 --variants 0 1 2 3
python evict.py proof --bytes 1048576 --repeats 300
python evict.py proof --bytes 67108864 --repeats 90 --variants 2 --load-each --deadline 58
python evict.py proof --bytes 67108864 --repeats 180 --variants 0 1 --load-each --deadline 58
python evict.py proof --bytes 67108864 --repeats 180 --variants 3 4 --load-each --deadline 58
python evict.py store --bytes 1048576 --repeats 300
python evict.py store --bytes 67108864 --repeats 200 --load-each --deadline 58
```

At 64 MiB a round takes 0.23 s (0.49 s with the readers), so each variant gets a load
of its own (`--load-each`): a thread runs at most 60 s.

`fetch.py` writes the source as a DRAM tensor of 16 KiB pages whose pages in bank
`--bank` (default 0) hold random words one after another, loads a grid whose commands
do nothing (example 08's kernels) and the thread `thread/copy.c` on cluster 1, and
prints one JSON line: per count of cores the best and worst `kmx_fetch` and its event in
µs, then per way (`call`, `clean`, `before`) its best call, rate and whether its hash
matched; the region compared whole up to 1 MiB; the store's status, time and bytes;
the sweep on four cores and on one (`sweep_us`). Before each way the thread zeroes the
region, so each hash shows that way's own copies. `--card` names the card whose clock
is sampled; the session opens `TT_VISIBLE_DEVICES` (the table's run: 1 for both; card 0's run: 0).
