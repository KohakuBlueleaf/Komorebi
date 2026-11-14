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
  broadcasts inside a grid column (NoC0 or NoC1, short blocks checked word by
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
- [dispatcher](dispatcher/README.md): what the dispatcher costs on the
  card: microseconds per command, the host's stores into the card (a store right after
  a fence stalls for about 12 microseconds), loads, stores and fences per X280 memory,
  and the stream engine's share of the kernel's loop.
- [graph_checks](graph_checks/README.md): the reference graphs of the graph executor
  and a comparison of its two implementations (Python and the plain-C engine built for
  the host) on random graphs and mutated uploads. No difference found.
- [page_index](page_index/README.md): whether the compute kernel can read page k of a
  circular buffer it waited for in full, by index, with nothing popped before it. Yes:
  two runs, exact. Values put into fixed slots reach compute as the whole buffer.
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
  dispatcher and a persistent grid (100 of 100 cycles clean), DRAM visibility
  between a Tensix core and the X280 (a Flush64 sweep each way flushes or invalidates), the
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
- [messages_probes](messages_probes/README.md): whether one core can reserve positions in another
  core's queue with one atomic add (yes: 119 cores, exact cover, 266 to 2596 cycles),
  the round trip of a message through DRAM (0.93 µs) and the rate between cores (up to
  1211 M messages/s, 120 cores in packets of 32).
- [atomic_answers](atomic_answers/README.md): when the returned value of a non-posted
  atomic add is lost: only when the asked tile submits atomics of its own on the same NoC
  with a changing return address (11 to 35 of 2.38 M); none from tiles that only answer.
  Decision: the routed messages' tails live on counter tiles.
- [overlay_probe](overlay_probe/README.md): a persistent kernel as a loader of kernel
  bodies linked for a fixed L1 slot and kept in DRAM. Loads cost 540 cycles plus 21 a
  KiB; the i-cache must be invalidated after each load (else stale code runs); the
  compute cores are parked in a wait loop outside the slots; 120 cores swap a 20 KiB
  overlay in 11.5 µs with a copy in every DRAM bank.
- [remote_load](remote_load/README.md): a core's slot programmed by another core. A
  Tensix core stores bodies into 119 others in 29 µs; an X280 core decides each core's
  next body (3.5 µs a swap), stores it, or patches a template's constant; the target
  must invalidate its own i-cache: a NoC write to that register wedges the card.
- [overlay_costs](overlay_costs/README.md): what an overlay costs inside the runtime.
  Runtime calls through a table of pointers add 12 cycles a call (+9.7% to a reader
  loop, +4.9% to a compute loop); compiled in, an overlay runs as fast as the whole
  kernel and refers only to shared state. Position-independent overlays cost nothing
  measurable and one copy serves any slot. Checking the program before a command costs
  0.1 µs; a switch of five 4 KiB parts 0.79 µs. A live core sets its circular buffers
  up again in 0.68 µs (240,000 switches, every value checked).
- [movement_probe](movement_probe/README.md): both data-movement CPUs of a core on one
  stream, each taking the blocks a pattern gives it through its own half of a paired
  buffer. One core doubles its reads (1.92 to 1.98x); across cores the ratio must follow
  the routes (AAAB reads +18%, BBBBBAAA writes +47% on 120 cores); the K-split matrix
  product gains 10.5% with its chunk reads split AB; attention, the rasterizer and the
  ray tracer are compute-bound.
- [ray_moves](ray_moves/README.md): whether the ray tracer gains when treelets stay in
  their owners' L1 and bundles of rays move between cores: the rays' paths on the host,
  routed messages of 64 bytes to 4 KiB on the card, and a predicted frame. Sponza's
  overview: 292 ms against 348.5 today with 2 KiB pages that no core copies (12.7
  million a second), R = 32, if walking and moving overlap (437 ms if not). Bistro's
  treelets need 156 MiB of L1 against 118: 2256 ms against 523.7.
- [raster_passes](raster_passes/README.md): the rasterizer's per-triangle passes and
  fill with each part on the unit that suits it: block ranges cut by predicted cost, each
  triangle finished on the vector unit, counting and slot writes on the writer core
  (together 12.1 to 6.5 ms a frame at 10^6 triangles, exact), and small triangles as
  vector lanes in the fill (1.5 to 2.1 times faster per small triangle on one core). A
  scalar into all lanes through `SFPCONFIG` is exact but slower than immediates.
- [ray_wide](ray_wide/README.md): whether one ray (or 2 or 4) tested against the child
  boxes of wide nodes (W = 8, 16, 32) speeds up the bounce and second shadow: the counts
  on the host, one node visit's cycles on the card, and a predicted frame. No layout
  beats today: at best (1 x 32, W = 32, boxes through the unpacker) a visit costs 729
  cycles and Sponza's frame is 471 ms against 348.5, Bistro's 680 against 523.7.
- [thread_commands](thread_commands/README.md): a command whose group names a work
  cluster (`KM_GROUP_CLUSTER`) answered by the thread's `kmx_program`: 10,000 from the
  host and 10,000 from another cluster's thread exact, a deadline on a function that
  does not return; 4.74 µs from a thread against 3.60 for a NOP to one core, and no
  cost to the commands that do not use it.
- [x280_rank](x280_rank/README.md): an X280 core as a rank of a dynamic run, its queue
  in its cluster's region and inboxes in the cores' L1, the dispatcher's own end:
  exact, 2.6 µs a round trip (0.87 for an ask), 12.9 M round trips a second on 12 X280
  cores; a proper build is about 900 lines, most in the generic reader.
- [decode_shape](decode_shape/README.md): the shape of an LLM decode step, a chain of
  K-split products (seven 4096 x 4096 a layer, 32 layers of random weights, 20 tokens)
  driven by the host, a stream and an X280 thread: 52.6, 20.9 and 20.9 ms a token, the
  weights read at 143, 360 and 360 GB/s against a ceiling of 503; parts-only products
  without the completion boundary at 458 GB/s; two chains on halves of the grid 1.12
  times one on all of it.
- [host_push](host_push/README.md): reading a frame back by having the cores write it
  over the NoC through the PCIe tile into the host's pinned memory channel, checked byte
  for byte. 4 KiB to 20 MiB exact; 13.5 GB/s from one core and about 15 GB/s from 2 to
  12 (the x8 link) against 1 to 2 MB/s for `Session.read_chunks`. Now in the session,
  both ways: a 20 MiB frame read in 30 ms without a run and 7 ms inside one, against
  10.1 s.
- [texture_lookup](texture_lookup/README.md): what a texture lookup costs on a core:
  pages of texels in L1 through a 4-way table (10 cycles a pixel with every page held,
  17 to 40 through the table, the same on all four CPUs), a 4 KB miss from DRAM in
  about 580 cycles (127 a page with 8 in flight), texel x light on the vector unit 29
  times faster than in integers, and the real reads of Sponza and Bistro replayed:
  1.1 to 2.1 ms a frame over 120 cores with a table of 32 pages.
- [two_cards](two_cards/README.md): two cards. One process a card runs example 01 on
  both at once (under 1 % a command); two sessions in one process do not open. A
  Tensix core writes into the other card's L1 over tt-metal's 1-D fabric in slow
  dispatch, exact both ways; 10.03 GB/s a link a direction from one sending core, 40.1
  GB/s over four links, the same each way at once.
- [islands_mesh](islands_mesh/README.md): both cards as islands of one run. One
  process opens them as one 1 x 2 fabric mesh with one X280 runtime over it, a
  dispatcher on each card and example 01 on each grid: 1.13 and 1.16 µs a NOP command,
  1.20 and 1.21 with both at once, every tile exact. A handler holds a fabric
  connection across commands and writes 64 bytes into the other island's L1, exact
  both ways, 1.6 to 1.7 µs a write command.
- [link_rate](link_rate/README.md): what a link between the two cards carries. One
  sending core a link, plain writes of 4352-byte payloads and one count for a batch,
  carries 28 to 29 GB/s a link, 111 to 115 GB/s over four, the same each way at once,
  wherever the senders sit; a fused write and increment with every packet drops a link
  to 10.6 GB/s. Packets for a DRAM bank or a core's L1 with no kernel running there
  land exact at the same rate. Two CPUs of one core each run a link. The sender's own
  per-packet work, not the link, sets the rate.
- [bank_fetch](bank_fetch/README.md): how fast an X280 thread gets data from a DRAM
  bank its cluster cannot read. `kmx_fetch` has Tensix cores copy it into the thread's
  region: 64 MiB from bank 0 to bank 6 in 3.79 ms on one core, 1.06 ms on four and
  1.05 on twelve (63.8 GB/s, the bank's rate); the whole call 30.4 ms, bound by the two sweeps
  of the region's cache lines; the X280's own loads 0.024 GB/s, `kmx_push` 2.20 GB/s.
