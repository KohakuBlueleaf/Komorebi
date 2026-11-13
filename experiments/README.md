# Experiments

Standalone proofs of one behaviour of the hardware or of a design choice: a small
program, how to run it, and the numbers it prints. An experiment does not import the
rest of the repository's code unless it is testing that code, and it states the
conditions of every number (card, clocks, sizes, load).

The rule is: experiments first; a regression test comes after behaviour settles.
A test pins down behaviour that is understood. Until then, an experiment is the right
place to ask a question and to be wrong.

## Experiments

- [landing](landing/README.md): how a core knows that a 64-byte command written by
  multicast has landed completely. Echo words in every 16-byte chunk against a command
  followed by a sequence word, on a Blackhole p150a with 120 cores. Decision: echo words.
- [tile_probes](tile_probes/README.md): what one Tensix tile costs for a 32-ray walk:
  predicated vector instructions, the "any broadcast enabled" register, a per-broadcast gather
  through the `Dst` window, one box against 32 rays, the compute RISC-V cores, and one
  full walk of a 255-node box tree ray-first against node-first, on one core of a
  Blackhole p150a. Predication is free; the gather costs 240 to 2280 cycles per 32-broadcast
  step depending on how the core copies; the node-first walk is 1.6 to 2.3 times faster
  from 256 rays on, because the ray-first walk is bound by the core's stack and copy
  work.
- [collision](collision/README.md): whether the X280's command multicasts on NoC0 and
  broadcast broadcasts inside a grid column (NoC0 or NoC1, short blocks checked word by
  word and long multicasts back to back) can run at the same time. No lost or wrong
  word and no hang in any arrangement measured; NoC1 recommended for broadcasts while the
  X280 dispatches.
- [noc_alignment](noc_alignment/README.md): what a NoC write shorter than 16 bytes does
  when its source and destination differ in their low four address bits. It lands at
  the destination, carrying the source line's bytes at the destination's offset, and is
  acknowledged as usual: silently wrong. Small writes go from a 16-byte slot to a
  16-byte slot.
- [wall_clock](wall_clock/README.md): whether the Tensix wall clock is one count for
  the whole chip. It is: every core within 90 cycles of core (0, 0), no measurable
  drift, 1343 to 1347 MHz against host time with the driver reporting 1350.
- [feeder_runs](feeder_runs/README.md): the longer runs of the map/collect example
  under the test feeder (a thousand jobs, every group size, groups in flight at once),
  run by hand.
- [stream_checks](stream_checks/README.md): the reference streams of the micro-op
  simulator and a mutation run over its source. Every one of 104 mutants changes some
  saved answer.
- [dispatch_kernel](dispatcher/README.md): what the dispatcher costs on the
  card: microseconds per command, the host's stores into the card (a store right after
  a fence stalls for about 12 microseconds), loads, stores and fences per X280 memory,
  and the stream engine's share of the kernel's loop.
- [graph_checks](graph_checks/README.md): the reference graphs of the graph executor
  and a comparison of its two implementations (Python and the plain-C engine built for
  the host) on random graphs and mutated uploads. No difference found.
- [page_index](page_index/README.md): whether the compute kernel can read page k of a
  circular buffer it waited for in full, by index, with nothing popped before it. Yes:
  two runs, exact. Values put into fixed slots are handed over as the whole buffer.
- [message_cost](message_cost/README.md): one put by distance and many senders
  incrementing one word. A put's latency does not depend on the hop count (1 to 16
  hops); a target in the sender's row costs about 43 cycles more than one in its column.
  118 increments of one word take 1.1 µs, about 12 cycles each.
- [runtime_harness](runtime_harness/README.md): the Tensix runtime's kernels as host
  threads on a simulated grid, with failure paths forced on purpose (stop flags,
  give-ups, lost chunks, mailbox races, items past a buffer's end). It found seven
  defects in the runtime and replays a card hang and two card findings; 35 scenarios,
  run by hand.
- [dram_read](dram_read/README.md): reading a 4 MB bfloat16 tile tensor back to the
  host bank by bank under one barrier against a whole-buffer read (then `ttnn.to_torch`):
  about 2 s against 3 s on one p150a at host load 8, the same array both ways.
- [x280_threads](x280_threads/README.md): a user image on X280 cluster 1 beside the
  dispatcher and a resident grid (100 of 100 cycles clean), DRAM visibility
  between a Tensix core and the X280 (a Flush64 sweep each way is the hand-over), the
  X280's read and write rates (about 1.3 and 1.8 GB/s) and prefix sums over 119 x
  1,024 counters (2.1 ms), and what a thread that ignores its deadline leaves behind.
- [work_clusters](work_clusters/README.md): X280 clusters 1 to 3 at once: the bank
  each reaches (6, 7, 7), about 1.2 GB/s per cluster alone or together (the limit is
  per cluster, not per bank), a push into the other bank at 2.3 GB/s against a pull at
  0.024, and a wake between clusters in 0.66 µs.
- [x280_ports](x280_ports/README.md): ports between X280 cores and Tensix cores with
  no dispatcher in the path: an ask answered in 0.94 µs (9.1 µs with ten Tensix
  cores on each X280 core, 13.1 M a second over 12), exact over a million round trips
  under DRAM load and across a command that times out, and 4 KiB into L1 at 8.1 GB/s
  against 2.1 through DRAM.
- [dyn_probes](dyn_probes/README.md): whether one core can reserve positions in another
  core's queue with one atomic add (yes: 119 cores, exact cover, 266 to 2596 cycles),
  the round trip of an item through DRAM (0.93 µs) and the rate between cores (up to
  1211 M items/s, 120 cores in packets of 32).
- [atomic_answers](atomic_answers/README.md): when the returned value of a non-posted
  atomic add is lost: only when the asked tile submits atomics of its own on the same NoC
  with a changing return address (11 to 35 of 2.38 M); none from tiles that only answer.
  Decision: the dynamic crossing's tails live on counter tiles.
- [overlay_probe](overlay_probe/README.md): a persistent kernel as a loader of kernel
  bodies linked for a fixed L1 slot and kept in DRAM. Loads cost 540 cycles plus 21 a
  KiB; the i-cache must be invalidated after each load (else stale code runs); the
  compute cores are parked in a wait loop outside the slots; 120 cores swap a 20 KiB
  overlay in 11.5 µs with a copy in every DRAM bank.
- [remote_load](remote_load/README.md): a core's slot programmed by another core. A
  Tensix core stores bodies into 119 others in 29 µs; an X280 core decides each core's
  next body (3.5 µs a swap), stores it, or patches a template's constant; the target
  must invalidate its own i-cache: a NoC write to that register wedges the card.
