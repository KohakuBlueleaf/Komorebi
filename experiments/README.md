# Experiments

Standalone proofs of one behaviour of the hardware or of a design choice: a small
program, how to run it, and the numbers it prints. An experiment does not import the
rest of the repository's code unless it is testing that code, and it states the
conditions of every number (card, clocks, sizes, load).

The rule is: experiments first; a regression test comes after behaviour settles.
A test pins down behaviour that is understood. Until then, an experiment is the right
place to ask a question and to be wrong.

## noc

How the NoC carries what the runtime sends: a command's landing, collisions, alignment,
messages and casters.

- [landing](noc/landing/README.md): how a core knows that a 64-byte command written by
  multicast has landed completely. Echo words in every 16-byte chunk against a command
  followed by a sequence word, on a Blackhole p150a with 120 cores. Decision: echo words.
- [collision](noc/collision/README.md): whether the X280's command multicasts on NoC0 and
  broadcasts inside a grid column (NoC0 or NoC1, short blocks checked word by
  word and long multicasts back to back) can run at the same time. No lost or wrong
  word and no hang in any arrangement measured; NoC1 recommended for broadcasts while the
  X280 dispatches.
- [noc_alignment](noc/noc_alignment/README.md): what a NoC write shorter than 16 bytes does
  when its source and destination differ in their low four address bits. It lands at
  the destination, carrying the source line's bytes at the destination's offset, and is
  acknowledged as usual: silently wrong. Small writes go from a 16-byte slot to a
  16-byte slot.
- [message_cost](noc/message_cost/README.md): one put by distance and many senders
  incrementing one word. A put's latency does not depend on the hop count (1 to 16
  hops); a target in the sender's row costs about 43 cycles more than one in its column.
  118 increments of one word take 1.1 µs, about 12 cycles each.
- [messages_probes](noc/messages_probes/README.md): whether one core can reserve positions in another
  core's queue with one atomic add (yes: 119 cores, exact cover, 266 to 2596 cycles),
  the round trip of a message through DRAM (0.93 µs) and the rate between cores (up to
  1211 M messages/s, 120 cores in packets of 32).
- [atomic_answers](noc/atomic_answers/README.md): when the returned value of a non-posted
  atomic add is lost: only when the asked tile submits atomics of its own on the same NoC
  with a changing return address (11 to 35 of 2.38 M); none from tiles that only answer.
  Decision: the routed messages' tails live on counter tiles.
- [casters](noc/casters/README.md): every core of a broadcast casting its own part of each
  block into one rectangle at once, an arrangement the NoC documentation warns can lose a
  flit and hang. Ten casters into a column of 10 cores, 80,000 blocks: every word and every
  count right, nothing hung. In the linear layer's prefill every shape was slower than the
  hand-over (1.31 to 2.07 times), so the runtime keeps one caster a rectangle.

## core

One Tensix core and the chip around it: what a tile costs, a circular buffer's pages,
the data-movement CPUs and the wall clock.

- [tile_probes](core/tile_probes/README.md): what one Tensix tile costs for a 32-ray walk:
  predicated vector instructions, the "any broadcast enabled" register, a per-broadcast gather
  through the `Dst` window, one box against 32 rays, the compute RISC-V cores, and one
  full walk of a 255-node box tree ray-first against node-first, on one core of a
  Blackhole p150a. Predication is free; the gather costs 240 to 2280 cycles per 32-broadcast
  step depending on how the core copies; the node-first walk is 1.6 to 2.3 times faster
  from 256 rays on, because the ray-first walk is bound by the core's stack and copy
  work.
- [page_index](core/page_index/README.md): whether the compute kernel can read page k of a
  circular buffer it waited for in full, by index, with nothing popped before it. Yes:
  two runs, exact. Values put into fixed slots reach compute as the whole buffer.
- [movement_probe](core/movement_probe/README.md): both data-movement CPUs of a core on one
  stream, each taking the blocks a pattern gives it through its own half of a paired
  buffer. One core doubles its reads (1.92 to 1.98x); across cores the ratio must follow
  the routes (AAAB reads +18%, BBBBBAAA writes +47% on 120 cores); the K-split matrix
  product gains 10.5% with its chunk reads split AB; attention, the rasterizer and the
  ray tracer are compute-bound.
- [wall_clock](core/wall_clock/README.md): whether the Tensix wall clock is one count for
  the whole chip. It is: every core within 90 cycles of core (0, 0), no measurable
  drift, 1343 to 1347 MHz against host time with the driver reporting 1350.

## x280

The X280 and its dispatcher: the dispatcher's costs, threads, work clusters, ports, a
rank, commands to a thread and a thread's fetch from DRAM.

- [dispatcher](x280/dispatcher/README.md): what the dispatcher costs on the
  card: microseconds per command, the host's stores into the card (a store right after
  a fence stalls for about 12 microseconds), loads, stores and fences per X280 memory,
  and the stream engine's share of the kernel's loop.
- [x280_threads](x280/x280_threads/README.md): a user image on X280 cluster 1 beside the
  dispatcher and a persistent grid (100 of 100 cycles clean), DRAM visibility
  between a Tensix core and the X280 (a Flush64 sweep each way flushes or invalidates), the
  X280's read and write rates (about 1.3 and 1.8 GB/s) and prefix sums over 119 x
  1,024 counters (2.1 ms), and what a thread that ignores its deadline leaves behind.
- [work_clusters](x280/work_clusters/README.md): X280 clusters 1 to 3 at once: the bank
  each reaches (6, 7, 7), about 1.2 GB/s per cluster alone or together (the limit is
  per cluster, not per bank), a push into the other bank at 2.3 GB/s against a pull at
  0.024, and a wake between clusters in 0.66 µs.
- [x280_ports](x280/x280_ports/README.md): ports between X280 cores and Tensix cores with
  no dispatcher in the path: an ask answered in 0.94 µs (9.1 µs with ten Tensix
  cores on each X280 core, 13.1 M a second over 12), exact over a million round trips
  under DRAM load and across a command that times out, and 4 KiB into L1 at 8.1 GB/s
  against 2.1 through DRAM.
- [x280_rank](x280/x280_rank/README.md): an X280 core as a rank of a dynamic run, its queue
  in its cluster's region and inboxes in the cores' L1, the dispatcher's own end:
  exact, 2.6 µs a round trip (0.87 for an ask), 12.9 M round trips a second on 12 X280
  cores; a proper build is about 900 lines, most in the generic reader.
- [thread_commands](x280/thread_commands/README.md): a command whose group names a work
  cluster (`KM_GROUP_CLUSTER`) answered by the thread's `kmx_program`: 10,000 from the
  host and 10,000 from another cluster's thread exact, a deadline on a function that
  does not return; 4.74 µs from a thread against 3.60 for a NOP to one core, and no
  cost to the commands that do not use it.
- [bank_fetch](x280/bank_fetch/README.md): how fast an X280 thread gets data from a DRAM
  bank its cluster cannot read. `kmx_fetch` has Tensix cores copy it into the thread's
  region: 64 MiB from bank 0 to bank 6 in 3.79 ms on one core, 1.06 ms on four and
  1.05 on twelve (63.8 GB/s, the bank's rate); the whole call 1.08 ms on four, the
  cluster's cache emptied first (`kmx_evict`, about 20 µs: one 128 KiB way, proven to
  leave no stale line); the X280's own loads 0.024 GB/s, `kmx_push` 2.20 GB/s.

## code

Where a core's code lives: overlays loaded into a slot, a slot filled by another core,
what an overlay costs and what an address costs a loop.

- [overlay_probe](code/overlay_probe/README.md): a persistent kernel as a loader of kernel
  bodies linked for a fixed L1 slot and kept in DRAM. Loads cost 540 cycles plus 21 a
  KiB; the i-cache must be invalidated after each load (else stale code runs); the
  compute cores are parked in a wait loop outside the slots; 120 cores swap a 20 KiB
  overlay in 11.5 µs with a copy in every DRAM bank.
- [remote_load](code/remote_load/README.md): a core's slot programmed by another core. A
  Tensix core stores bodies into 119 others in 29 µs; an X280 core decides each core's
  next body (3.5 µs a swap), stores it, or patches a template's constant; the target
  must invalidate its own i-cache: a NoC write to that register wedges the card.
- [overlay_costs](code/overlay_costs/README.md): what an overlay costs inside the runtime.
  Runtime calls through a table of pointers add 12 cycles a call (+9.7% to a reader
  loop, +4.9% to a compute loop); compiled in, an overlay runs as fast as the whole
  kernel and refers only to shared state. Position-independent overlays cost nothing
  measurable and one copy serves any slot. Checking the program before a command costs
  0.1 µs; a switch of five 4 KiB parts 0.79 µs. A live core sets its circular buffers
  up again in 0.68 µs (240,000 switches, every value checked).
- [overlay_shift](code/overlay_shift/README.md): two programs in the runtime's form,
  each with overlays of its own numbered from 1, and an overlay added to the live run, in
  one run on 120 cores. The second program's 1 is the run's 3 (its shift 2) and the
  added overlay is 4: 16 commands, 0 wrong; a command after the other program's costs
  about 1.5 µs more, the program's switch.
- [code_address](code/code_address/README.md): loops of assembly copied to chosen L1
  addresses and timed. A taken branch or jump whose target is at or above 1 MB costs 4
  cycles more (about 7.5 to a line that is not the next), a branch never taken and
  straight-line code do not, and data loads and stores above 1 MB are not slower;
  distance, alignment and the words after the loop do not matter.

## recovery

What comes back without a card reset: a core, the dispatcher.

- [core_reset](recovery/core_reset/README.md): whether one Tensix core of a run comes back
  without a card reset. A core whose handler never returns is reset and loaded again by
  the dispatcher at the command's give-up (1.6 to 1.7 ms of its loop), from a list the
  session hands it at load, and the commands after it are exact; by the host the same
  reset takes about 65 ms. A compute unit left running is not tried yet.
- [dispatcher_watch](recovery/dispatcher_watch/README.md): whether the dispatcher comes back by
  itself when its loop on X280 core 0 stalls. Core 2 sees no pass for 100 ms, reports
  the stall to the host and interrupts core 0, which re-enters its loop 1.2
  microseconds later with the run's state kept; the MAP commands after it are exact.

## host

What the host sends and reads: a frame read back by the cores' writes, the start of
cores and a tensor read bank by bank.

- [host_push](host/host_push/README.md): reading a frame back by having the cores write it
  over the NoC through the PCIe tile into the host's pinned memory channel, checked byte
  for byte. 4 KiB to 20 MiB exact; 13.5 GB/s from one core and about 15 GB/s from 2 to
  12 (the x8 link) against 1 to 2 MB/s for `Session.read_chunks`. Now in the session,
  both ways: a 20 MiB frame read in 30 ms without a run and 7 ms inside one, against
  10.1 s.
- [start_spread](host/start_spread/README.md): why some host pushes are slower with two or
  more cores. Each core's own push time never changes; the launch starts later cores up
  to 0.48 ms late (most often 0.476 ms), with or without a push, whatever the bytes, the
  clock, telemetry reads, the other card or link errors.
- [dram_read](host/dram_read/README.md): reading a 4 MB bfloat16 tile tensor back to the
  host bank by bank under one barrier against a whole-buffer read (then `ttnn.to_torch`):
  about 2 s against 3 s on one p150a at host load 8, the same array both ways.

## islands

Several cards as islands of one run: two cards, a mesh, the links' rate, pages, bytes,
collectives and the link service.

- [two_cards](islands/two_cards/README.md): two cards. One process a card runs example 01 on
  both at once (under 1 % a command); two sessions in one process do not open. A
  Tensix core writes into the other card's L1 over tt-metal's 1-D fabric in slow
  dispatch, exact both ways; 10.03 GB/s a link a direction from one sending core, 40.1
  GB/s over four links, the same each way at once.
- [islands_mesh](islands/islands_mesh/README.md): both cards as islands of one run. One
  process opens them as one 1 x 2 fabric mesh with one X280 runtime over it, a
  dispatcher on each card and example 01 on each grid: 1.13 and 1.16 µs a NOP command,
  1.20 and 1.21 with both at once, every tile exact. A handler holds a fabric
  connection across commands and writes 64 bytes into the other island's L1, exact
  both ways, 1.6 to 1.7 µs a write command.
- [link_rate](islands/link_rate/README.md): what a link between the two cards carries. One
  sending core a link, plain writes of 4352-byte payloads and one count for a batch,
  carries 28 to 29 GB/s a link, 111 to 115 GB/s over four, the same each way at once,
  wherever the senders sit; a fused write and increment with every packet drops a link
  to 10.6 GB/s. Packets for a DRAM bank or a core's L1 with no kernel running there
  land exact at the same rate. Two CPUs of one core each run a link. The sender's own
  per-packet work, not the link, sets the rate.
- [islands_pages](islands/islands_pages/README.md): a tensor held by one island written into
  a tensor held by the other by the session's link cores, a page a packet straight
  into the other island's DRAM: 64 MiB over four links (two cores, both CPUs of each)
  in 2.09 ms, 32.1 GB/s, every page exact. Three L1 slots a CPU leave the DRAM reads'
  latency in charge.
- [islands_bytes](islands/islands_bytes/README.md): whether a kernel's read of a few bytes
  of a page the other island holds (`km::read_bytes`) lands exactly those bytes. 118
  ranks, 64 ranges each of 4 to 2,048 bytes at their own offsets: every word exact,
  nothing beside them written; the other island answers one in 1.7 to 1.95 µs.
  `forms.py`: a tensor in pages of the command's size (`km::SizedTensor`) and a range
  of one DRAM bank (`km::BankRange`), read and written by the ranks of both islands:
  183,608 words checked on the cores and both tensors read back, every word exact.
- [islands_collectives](islands/islands_collectives/account/README.md): where the time of
  `km::all_gather` and `km::all_reduce` goes across two cards, and how it grows with the
  tensor. 16 MiB gathered in 132.4 µs and 64 MiB in 381.7 (101 GB/s each way, the links'
  rate); the all-reduce takes 1,488.3 and 5,556.8 µs, the growth mostly the adds.
- [islands_tree](islands/islands_tree/README.md): the ports' reduce as an all-reduce,
  its sum cast back to every rank of the group in the same command: exact on every rank
  of 16 and 118, 10.16 and 21.98 µs against the plain reduce's 7.62 and 10.44 (one
  card, 800 MHz).
- [acceptance](islands/acceptance/README.md): the same programs (attention, SwiGLU, the
  decode layer) unchanged on one island, two cards, a line of three and the 2 x 2 grid:
  every two-card output word-equal to one island, and SwiGLU placed island-aware four
  times faster than across the edge; on a split card the refusals and open defects,
  each named.
- [far_order](islands/far_order/README.md): a put, then a posted add, from one CPU to a
  far island, in one command with the rank that reads at the add: 0 early reads in
  100,000 rounds a case, one link and two links away.
- [link_service](islands/link_service/README.md): the first measure of messages between
  islands: two islands joined by one link, a core on each running a link reader and writer.
  A crossing costs 6.5 µs for one message and 4.7 µs among a thousand round trips; single
  messages reach about 0.6 million a second each way, batched about 4.7 million. The service
  now runs on the island's link core 0; the measuring script is in the history.

## language

The class and tile languages against the hand programs: emitted bodies, programs written
as classes and what a class cannot say yet.

- [attention_class](language/attention_class/README.md): the attention project's
  program as a class, every move a class line (the job table's entries, Q's rows held
  through the job, the K / V cast in bands, the output rows) and every compute step the
  hand's. Beside the hand program in one session, O is equal word for word on t1 and
  causal64, 57.75 us a command against the hand's 57.64 us.
- [bounds](language/bounds/README.md): a 27B-class layer's five products (assumed
  shapes) as classes, W in bfp8_b or bfp4_b, against the card's DRAM and matrix-unit
  bounds on one card and two: decode 1.33 to 2.60 times the memory bound a layer,
  prefill at M 4,096 1.60 to 1.67 times the HiFi2 compute bound, M 256 6 to 12 times.
  A decode layer as one chain is 1.26 (bfp8_b) and 1.74 (bfp4_b) times the bound on one
  card. A product's fixed part is 13 to 20 us; the rest is W's rate (about 400 and 290
  GB/s).
  On two cards as one chain with two gathers: 1.74 and 2.48 times; a W tile's cost is
  the movers' read of a page (about 156 cycles), not compute. W read in DRAM pages
  of 7 tiles on both movers (a hand probe) reaches 477 GB/s in bfp8_b and 489 in bfp4_b.
  The class's chain with W in such pages (`page_tiles`): bfp4_b 814.5 to 657.0 us a
  layer on one card (19.8 to 24.5 tok/s), bfp8_b 1,116.1 to 1,076.5, words unchanged.
  X read once and cast: no form a class says today beats the chain (wider groups 666.5
  us against 658.6; mover B alone reads at about 247 GB/s).
- [dense_class](language/dense_class/README.md): the dense project's linear layer, Y = X W, as
  a class in the project's own two plans (the K split on 112 cores, the grid form on 12 x
  8), its SwiGLU gate a second tile function. Y equals the host's product for integers;
  94.2 us a command at 32 x 4096 x 4096 against the hand's 92.4, and 358.6 against 382.4 at
  1024 x 4096 x 4096.
- [raytrace_class](language/raytrace_class/README.md): whether a program of the ray tracer's
  kind can be written as a class with nothing but its own lines and its arithmetic in
  inline bodies. The trace pass (1,048,576 primary rays on 120 cores) gives hit pages equal
  to the hand's word for word and takes 3,294.9 us a command against the hand's 3,259.9.
- [raster_class](language/raster_class/README.md): the rasterizer's FILL, COUNT, HIST and
  SETUP as two classes beside the hand programs, on the tiny scene. The frame, counters,
  histograms and lists equal the hand's word for word. FILL takes 88.34 us a command
  against 118.90 (the empty tile's planes kept in L1); the passes are within 10 % of the
  hand's.
- [own_buffer](language/own_buffer/README.md): whether a class can give its compute bodies
  buffers that only they use, a float32 tile written in place and a FIFO from the pack CPU
  to the unpack CPU, with no line filling, waiting for or freeing them (`km.own()`). On 8
  cores of 40 tiles every output matched numpy (0 of 8,192 and 0 of 327,680 wrong), 21.9 and
  21.8 us a command.
- [tile_rules](language/tile_rules/README.md): whether the compute bodies that
  `komorebi.dsl.tile` emits for SiLU, a residual add, a scale, the SwiGLU gate and the row
  norm compute what the hand bodies compute, what a unit costs on one core with each, and
  how close the printed cost is to the card. Every emitted body is inside its tolerance;
  the emitted norm takes 1,277 cycles a unit against the hand's 2,640; every body's
  estimate is within 4%, attention's steps within 1% (the accurate softmax step -12%).
- [tile_front](language/tile_front/README.md): the SiLU compute body emitted by
  `komorebi.dsl.tile` against tt-metal's `silu_tile` on one core of a Blackhole p150a.
  The tile body is the same instruction stream modulo registers (the setup differs by
  two counter resets and one `SFPNOP`), the output is equal bit for bit, and both take
  1730 cycles a tile.

## programs

Shapes of programs measured before they are built: the ray tracer's and the rasterizer's
moves, texture lookups and a decode step.

- [ray_moves](programs/ray_moves/README.md): whether the ray tracer gains when treelets stay in
  their owners' L1 and bundles of rays move between cores: the rays' paths on the host,
  routed messages of 64 bytes to 4 KiB on the card, and a predicted frame. Sponza's
  overview: 292 ms against 348.5 today with 2 KiB pages that no core copies (12.7
  million a second), R = 32, if walking and moving overlap (437 ms if not). Bistro's
  treelets need 156 MiB of L1 against 118: 2256 ms against 523.7. The runtime's own
  bodies move 30.3 million 2 KiB bundles a second.
- [direct_ring](programs/direct_ring/README.md): a direct message from one core's L1
  into a declared receiver's against the queues in DRAM: tokens hopping a ring of 118
  ranks, 1.85 µs a hop direct (77 % of hops; the rest forwarded) against 10.66 µs
  through the queues; over pairs that are not offsets, 3.04 against 3.19 µs.
- [ray_wide](programs/ray_wide/README.md): whether one ray (or 2 or 4) tested against the child
  boxes of wide nodes (W = 8, 16, 32) speeds up the bounce and second shadow: the counts
  on the host, one node visit's cycles on the card, and a predicted frame. No layout
  beats today: at best (1 x 32, W = 32, boxes through the unpacker) a visit costs 729
  cycles and Sponza's frame is 471 ms against 348.5, Bistro's 680 against 523.7.
- [raster_passes](programs/raster_passes/README.md): the rasterizer's per-triangle passes and
  fill with each part on the unit that suits it: block ranges cut by predicted cost, each
  triangle finished on the vector unit, counting and slot writes on the writer core
  (together 12.1 to 6.5 ms a frame at 10^6 triangles, exact), and small triangles as
  vector lanes in the fill (1.5 to 2.1 times faster per small triangle on one core). A
  scalar into all lanes through `SFPCONFIG` is exact but slower than immediates.
- [texture_lookup](programs/texture_lookup/README.md): what a texture lookup costs on a core:
  pages of texels in L1 through a 4-way table (10 cycles a pixel with every page held,
  17 to 40 through the table, the same on all four CPUs), a 4 KB miss from DRAM in
  about 580 cycles (127 a page with 8 in flight), texel x light on the vector unit 29
  times faster than in integers, and the real reads of Sponza and Bistro replayed:
  1.1 to 2.1 ms a frame over 120 cores with a table of 32 pages.
- [decode_shape](programs/decode_shape/README.md): the shape of an LLM decode step, a chain of
  K-split products (seven 4096 x 4096 a layer, 32 layers of random weights, 20 tokens)
  driven by the host, a stream and an X280 thread: 52.6, 20.9 and 20.9 ms a token, the
  weights read at 143, 360 and 360 GB/s against a ceiling of 503; parts-only products
  without the completion boundary at 458 GB/s; two chains on halves of the grid 1.12
  times one on all of it.

## checks

References and long runs that are run by hand: the feeder, the stream and graph
references and the runtime harness.

- [feeder_runs](checks/feeder_runs/README.md): the longer runs of the map/collect example
  under the test feeder (a thousand jobs, every group size, groups in flight at once),
  run by hand.
- [stream_checks](checks/stream_checks/README.md): the reference streams of the micro-op
  simulator and a mutation run over its source. Every one of 104 mutants changes some
  saved answer.
- [graph_checks](checks/graph_checks/README.md): the reference graphs of the graph executor
  and a comparison of its two implementations (Python and the plain-C engine built for
  the host) on random graphs and mutated uploads. No difference found.
- [runtime_harness](checks/runtime_harness/README.md): the Tensix runtime's kernels as host
  threads on a simulated grid, with failure paths forced on purpose (stop flags,
  give-ups, lost chunks, mailbox races, items past a buffer's end). It found seven
  defects in the runtime and replays a card hang and two card findings; 35 scenarios,
  run by hand.
