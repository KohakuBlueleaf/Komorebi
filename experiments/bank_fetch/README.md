# Bank fetch

How fast does an X280 thread get data that lies in a DRAM bank its work cluster cannot
read? A cluster's cache reaches only its own bank (6 for cluster 1, 7 for clusters 2
and 3). Its own loads over the NoC move 0.024 GB/s. With `kmx_fetch` the thread sends
one command and Tensix cores copy the range, DRAM to DRAM, into the thread's region.

## Results

Blackhole p150a, card 1, slow dispatch, AICLK 1350 MHz throughout (sampled), X280 at the
session's 1000 MHz, host load 3.0 to 6.9, 2026-11-14. Bank 0 to work cluster 1's
region in bank 6, best of 3 (64 MiB) or 5 (1 MiB); every byte checked (a hash of the
whole region in the thread against numpy, and at 1 MiB the region read back whole).

| Way | Cores | 1 MiB | 64 MiB |
|---|---|---|---|
| the copy (`kmx_fetch`'s command: dispatcher submit to completion) | 1 | 60.6 µs, 17.3 GB/s | 3.79 ms, 17.7 GB/s |
| | 4 | 19.2 µs, 54.6 GB/s | 1.06 ms, 63.5 GB/s |
| | 12 | 18.9 µs, 55.4 GB/s | 1.05 ms, 63.8 GB/s |
| `kmx_fetch`, the whole call (the cache sweeps included) | 12 | 482 µs, 2.17 GB/s | 30.4 ms, 2.21 GB/s |
| the X280 reads the range itself (8-byte loads through a NoC window) | 1 X280 core | 44.1 ms, 0.024 GB/s | - |
| `kmx_push` from cluster 1 to cluster 2's region (bank 7) | 4 X280 cores | 477 µs, 2.20 GB/s | - |

- **The copy.** One core reads and writes 16 KiB chunks at 17.5 GB/s; four cores reach
  the bank's rate (about 64 GB/s), and twelve gain nothing more. At 1 MiB the
  dispatcher's own path is a few µs of the 19.
- **The call.** `kmx_fetch` drops the region's cache lines before the copy (a dirty line
  written back later would overwrite the new bytes) and again after it, each sweep at
  about 4.5 GB/s; that is 96 % of the call. It is still 90 times faster than the X280
  reading the bank itself and as fast as a producer pushing the data with `kmx_push`,
  which the fetch replaces when the producer is not a thread.
- **The store.** `kmx_store` of 1 MiB from the region back to bank 0 on four cores: 0.25
  to 0.32 ms (one sweep, then the copy), exact.

The X280 row is `experiments/work_clusters/clusters.py push --cluster 1 --bank 7
--bytes 1048576 --pull 1048576` (its pull; its window push gave 2.28 GB/s), the
`kmx_push` row example 08's `--twelve` run (its input push to cluster 2 at eight cores),
both on the same card and day.

## Running it

```sh
python fetch.py --bytes 4096 --cores 1
python fetch.py --bytes 1048576 --cores 1 4 12 --repeats 5 --store 1048576
python fetch.py --bytes 67108864 --cores 1 4 12 --repeats 3 --store 1048576
```

`fetch.py` writes the source as a DRAM tensor of 16 KiB pages whose pages in bank
`--bank` (default 0) hold random words one after another, loads a grid whose commands
do nothing (example 08's kernels) and the thread `thread/copy.c` on cluster 1, and
prints one JSON line: per count of cores the best call and its event in µs, the worst
call, the rate and whether the hash matched; the region compared whole up to 1 MiB; the
store's status, time and bytes. Per count the thread zeroes the region first, so each
count's hash shows its own copies.
