# Bank fetch

How fast does an X280 thread get data that lies in a DRAM bank its work cluster cannot
read? A cluster's cache reaches only its own bank (6 for cluster 1, 7 for clusters 2
and 3). Its own loads over the NoC move 0.024 GB/s. With `kmx_fetch` the thread sends
one command and Tensix cores copy the range, DRAM to DRAM, into the thread's region.

## Results

Blackhole p150a, card 0, slow dispatch, AICLK 1350 MHz throughout (sampled), X280 at the
session's 1000 MHz, host load about 4.2 to 4.9, 2026-11-14. Bank 0 to work cluster 1's
region in bank 6 on four Tensix cores, best of 5 (1 MiB) or 3 (16 and 64 MiB); every
way's bytes checked (a hash of the whole region in the thread against numpy, and at
1 MiB the region read back whole).

| Way | 1 MiB | 16 MiB | 64 MiB |
|---|---|---|---|
| the copy (the command's event: dispatcher submit to completion) | 19.3 µs | 269 µs | 1.06 ms, 63.1 GB/s |
| `kmx_fetch_clean`, the whole call (no sweep) | 21.5 µs, 48.7 GB/s | 270 µs, 62.0 GB/s | 1.07 ms, 62.9 GB/s |
| `kmx_fetch`, the whole call (one sweep before the copy) | 192 µs, 5.46 GB/s | 2.99 ms, 5.61 GB/s | 11.9 ms, 5.62 GB/s |
| `kmx_fetch` and `kmx_invalidate_at` of the range: the earlier call's two sweeps | 363 µs, 2.89 GB/s | 5.71 ms, 2.94 GB/s | 22.8 ms, 2.94 GB/s |
| a sweep of the range (`kmx_flush_at`), four X280 cores | 171 µs | 2.77 ms | 10.9 ms |
| the same sweep on core 0 alone | 171 µs | 2.72 ms | 10.9 ms |

- **The copy.** One core reads and writes 16 KiB chunks at about 16.7 GB/s (64.6 µs,
  1.01 ms and 4.02 ms); four cores reach the bank's rate (about 63 GB/s), and twelve
  gain nothing more (19.2 µs, 265 µs, 1.05 ms).
- **The sweep.** A sweep writes one Flush64 per line to the cluster's cache controller:
  10.4 ns a line, 6.1 GB/s, the same on one core as on four. The controller takes one
  line at a time, so the sweep costs the range's lines and nothing spreads it.
- **The call.** `kmx_fetch` sweeps the range once, before the copy: a dirty line
  written back later would overwrite the new bytes, and a stale line would be read in
  place of them. Nothing on the cluster touches the range while the cores copy, so no
  second sweep follows; the thread's first read of a line takes the new bytes. When the
  range has no line in the cache (not read or written since the load, its last flush
  or the last fetch into it), `kmx_fetch_clean` skips the sweep and the call costs the
  copy: 1.07 ms for 64 MiB against 22.8 ms before. In this run the zeroed region's lines
  were dirty before each `kmx_fetch` and flushed before each `kmx_fetch_clean`.
- **The store.** `kmx_store` of 1 MiB from the region back to bank 0 on four cores:
  0.19 to 0.24 ms (one sweep, then the copy), exact.

On card 1 the same day, the X280 reading the range itself (8-byte loads through a NoC
window) took 44.1 ms for 1 MiB (0.024 GB/s), and `kmx_push` of 1 MiB from cluster 1 to
cluster 2's region (bank 7) on four X280 cores 477 µs (2.20 GB/s): rows from
`experiments/work_clusters/clusters.py push --cluster 1 --bank 7 --bytes 1048576 --pull
1048576` (its pull) and example 08's `--twelve` run (its input push to cluster 2).

## Running it

```sh
python fetch.py --bytes 4096 --cores 1
python fetch.py --bytes 1048576 --cores 1 4 12 --repeats 5 --store 1048576
python fetch.py --bytes 16777216 --cores 1 4 12 --repeats 3 --store 1048576
python fetch.py --bytes 67108864 --cores 1 4 12 --repeats 3 --store 1048576
```

`fetch.py` writes the source as a DRAM tensor of 16 KiB pages whose pages in bank
`--bank` (default 0) hold random words one after another, loads a grid whose commands
do nothing (example 08's kernels) and the thread `thread/copy.c` on cluster 1, and
prints one JSON line: per count of cores the best and worst `kmx_fetch` and its event in
µs, then per way (`call`, `clean`, `before`) its best call, rate and whether its hash
matched; the region compared whole up to 1 MiB; the store's status, time and bytes;
the sweep on four cores and on one (`sweep_us`). Before each way the thread zeroes the
region, so each hash shows that way's own copies. `--card` names the card whose clock
is sampled; the session opens `TT_VISIBLE_DEVICES` (the runs above: 0 for both).
