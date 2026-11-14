# Pushing bytes from the cores into the host's memory

Reading a frame back from card memory in slow dispatch runs at 1 to 2 MB/s (about 2 s
for 4 MiB). This experiment shows the other direction: a core writes its bytes over the
NoC to the PCIe tile, which writes them into the host's 1 GiB memory channel that the
driver pinned when the device opened. The host reads the bytes from the mapping it
already holds. Nothing in `src/` changes; the experiment runs raw one-shot kernels
through `Session.launch()`.

## What it does

- `kernels/push.cpp`: one data-movement core pushes `bytes` to channel offset `offset`.
  Mode 0 writes a pattern; mode 1 reads pages of a DRAM tensor into L1 (two buffers of
  32 KiB) and writes each buffer while the next one is read. Every write goes through
  tt-metal's PCIe-aware calls: `noc_async_write_set_pcie_state`,
  `noc_async_write_with_state`, `noc_async_write_clear_pcie_state` for the data and
  `noc_async_write_pcie` for the sequence word.
- The core flushes the data writes (every wait bounded by the wall clock), clears the
  PCIe routing, and only then writes a 16-byte sequence word (nonce, bytes, core,
  magic) as a write of its own. It reports status, bytes, three wall-clock stamps and
  whether the write buffer holds no PCIe routing any more.
- `push.py` opens the session, stops if the open log shows the "NOC address ... does
  not match" warning, finds the channel in `/proc/self/maps`, fills the data region
  with the complement of what is expected (so a missing write shows), waits for every
  core's word, and compares every byte, the bytes after the data (they must not
  change) and the words.

## Bounds it keeps

- Only the PCIe-aware calls write toward the PCIe tile; no register space, no raw NoC
  read.
- The kernel refuses before its first write unless `offset + bytes <= 64 MiB`, the
  sequence word is inside 64 MiB and does not overlap the data, offsets and sizes are
  multiples of 16, and the two buffers fit its L1; the host computes the same slices
  (data from offset 0, words at 32 MiB, 64 bytes a core) and checks them first.
- `clear_pcie_state` after the batch, before the report; the report holds the check.
- Run on card 1 only (`TT_VISIBLE_DEVICES=1` is required), every job through the hang
  watcher, up the ladder: 4 KiB, 1 MiB, 4 MiB, 20 MiB, one core before more.

## Results

Card 1, Blackhole p150a, link 16.0 GT/s x8 (read before every run), AICLK 1350 MHz
throughout (min = mean; the wall clock runs at the nominal AICLK), slow dispatch, host
load 5 to 10.

Step 3, one core, pattern: 4 KiB and 1 MiB arrive exact (0 differing bytes), the
sequence word is seen, the bytes after the data are untouched, the routing is clear,
and the host reads plain loads from its mapping with no cache step. One core writing
the pattern (it fills L1 first) pushes 1.0 GB/s.

Step 4, each core pushes its slice of a DRAM tensor (pages of 4 KiB); card MB/s is the
bytes over the span from the first core's start to the last core's last flushed data
write, the median of three pushes in one process; host seconds is `launch.run()` until
the host sees the last word, the median of three:

| cores | 4 MiB card MB/s | 20 MiB card MB/s | 20 MiB host s |
|---|---|---|---|
| 1 | 13466 | 13496 | 0.0143 |
| 2 | 14799 | 14247 | 0.0366 |
| 4 | 14825 | 13511 | 0.0259 |
| 8 | 14825 | 14952 | 0.0132 |
| 12 | 14803 | 14070 | 0.0353 |

`Session.read_chunks` over the same bytes (2 MiB chunks), host load 5: 4 MiB in 3.86 s
(1.09 MB/s), 20 MiB in 9.67 s (2.17 MB/s). A 20 MiB frame is about 1.5 ms of card
time against 9.7 s: 6,000 times faster; the host's 12 to 36 ms is the launch of a
program (a 1 MiB pattern push also takes 13 ms), which a persistent kernel does not pay.

- One core already reaches 13.5 GB/s; more cores add up to about 15 GB/s, the link's
  limit (x8 at 16 GT/s is 15.8 GB/s before overhead). The core count does not matter;
  the link does.
- Outliers: at 2 cores and more one push in three has a span 2 to 10 times longer
  (for example 12 cores, 4 MiB: 4.0 M cycles instead of 0.38 M), all bytes still exact.
  I think a core started late (the host starts the cores one after another); I did not
  check, as the span is from the first start. The medians leave them out.
- The sequence word is written after the flush of the data writes, as a separate
  write. Seen first, with the data complete, in every push (`exact` in every run); a
  run that wrote the word without the flush was not tried.
- NoC0 only (as `mem_bench` runs); chunks of 32 KiB only; no pushes of other sizes
  than the ladder's.

## Running it

On a card through the hang watcher, with tt-metal's environment set, `FR_LOG` naming the
log the watcher writes and the repository's `src` on the path:

    TT_VISIBLE_DEVICES=1 FR_LOG=<log> python push.py --mode pattern --bytes 4096
    TT_VISIBLE_DEVICES=1 FR_LOG=<log> python push.py --mode dram --bytes 20971520 \
        --cores 8 --reps 3 --readback

It prints one JSON line a push (card cycles, host seconds, the checks) and one for the
`read_chunks` timing.

## What the framework call needs

- Where: a writer-side call in the kernel library, `push_to_host(l1, offset, bytes)`,
  built as `push.cpp` does it (set, writes, flush, clear); the session hands the program
  the channel's offset and size as arguments, never a raw address.
- Flow control: the writer waits for its writes' acknowledgements before it reuses an L1
  buffer, then writes a sequence word; the host cannot slow the card, so a ring in the
  channel needs a host-to-card credit (an L1 word the host writes, or the X280's ring).
- The session owns the channel: it opens at device open in both dispatch modes, and
  only fast dispatch uses it; a session must reserve a region and check every offset
  against it, because a wrong offset may land in host memory outside the channel.
- It must stop at open on the NoC address warning, zero its region before each use
  (the hugepage file keeps old bytes between processes) and map the channel once.
- Unknown: pushes while fast dispatch runs, two cards at once, and the X280 as the
  writer.
