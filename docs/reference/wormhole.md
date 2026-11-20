---
title: Wormhole
summary: What the framework does on Wormhole, feature by feature, from what has run on the chips of a Galaxy Wormhole box: supported, supported with a stated limit, refused by name, or not run.
tags:
  - reference
  - wormhole
---

# Wormhole

Komorebi was written on Blackhole. On Wormhole it runs with the dispatcher on a Tensix
core (Wormhole has no X280), one chip as one session of 79 cores, and the chips of a
Galaxy box as islands, up to 31 of its 32. This page says, for each feature of the
framework, what it does there. Each row is one of:

- **supported**: it ran on Wormhole chips and gave the expected result;
- **supported, limit**: it ran, and the limit stated beside it holds;
- **refused by name**: the build, the load or the call refuses it with an error that
  names what is missing, before anything runs on the cores;
- **fails**: it is neither refused nor working; what is seen is stated;
- **not run**: nothing on Wormhole chips has shown it either way.

A row says supported only for what ran. Where it ran is in the last column:

- *census*: the one-chip census of the card list and the lines outside it (159 lines,
  each run with its own check on one chip of the Galaxy, tt-metal v0.81.1, slow
  dispatch). Its last pass, 2026-11-19: 118 pass, 28 refused by name, 13 fail.
- *box*: runs on the Galaxy's chips, 2026-11-20 (AICLK 1000 MHz): on one chip, and on
  2, 4, 8, 16 and 31 chips as islands (one link a pair, every chip opened under
  `FABRIC_1D_RING`).
- *merge check*: this tree's own runs on the box, 2026-11-20 (below).
- *the line's tests*: the tests that came with a fix on the Wormhole line, 2026-11-20
  and 21, most without a card; not run again in the merge check.

A chip of that box has a DRAM fault and stays out of every session
([hazards](hazards.md#a-dram-bank-that-flips-one-bit)), so 31 islands is the most run.

## The table

### One chip

| Feature | On Wormhole | Limit, or the refusal's name | Ran in |
|---|---|---|---|
| A session whose facts come from the device (the 8 x 10 grid, L1, 32 buffer indices, 12 DRAM banks, the tick rate) | supported | | census |
| Kernels for Wormhole built with no chip (`kmr prebuild --arch wormhole_b0`) | supported | | census, box |
| The X280 dispatcher | refused by name | the chip has no X280; the Tensix dispatcher is the default | census |
| Plain commands, result words, groups, several commands in flight, the job queue | supported | | census |
| Kernels as Python classes | supported, limit | the class's writer must fit NCRISC's 16 KiB of code, below | census |
| Tensor reads and writes through the host channel | supported | | census |
| Channels: `emit`, `arrive`, `take`, reduce by owner and by tree | supported | | census (02_merge_tree) |
| Broadcasts along a row or a column | supported | | census (03_broadcast) |
| Movement: both movers on one stream | supported | | census (11_movement) |
| Overlays, add-ons, several programs in one run, program switches | supported, limit | an overlay part's fetch hung in 4 of 725 runs of `leader_overlay.py`, and `overlays_class.py` left a chip's cores busy in 2 of 38 runs ([hazards](hazards.md#an-overlay-fetch-that-does-not-land-on-wormhole-open), [hazards](hazards.md#a-wormhole-chip-whose-cores-stay-busy-after-a-run-open)) | census |
| A writer over NCRISC's 16 KiB of code | supported, limit | its paths after a command's work run from the service overlay's writer part in L1, at the lowest writer level that fits; still over at level 4, the highest: refused by name ("the writer ... is N B of NCRISC's 16,384 B code region at writer level 4, the highest: M B over") ([hazards](hazards.md#a-writer-near-the-16-kib-of-code-a-wormhole-cores-ncrisc-holds)) | census |
| Routed messages, dynamic handlers, a handler as an overlay | supported | | census (05, 14, 20, 23) |
| X280 cores as ranks of a dynamic run | refused by name | "X280 ranks" | census (24_thread_rank) |
| Micro-op streams and graphs: host-fed and the resident engines on the dispatcher's TRISC 0 | supported | | census (04, 06) |
| The trace | supported | | census |
| The chain written by the host (`run.chain`), and the chain step classes | supported, limit | the classes build at a writer level; a chain whose writer is still over is refused by name | census (29_chain) |
| A command given up at its deadline, `run.inspect`, the reset and reload of a core | supported, limit | `when_it_goes_wrong.py` ended once in 42 runs with the stop's grid timeout on its core 0 | census |
| A thread's C on the dispatcher's own core (TRISC 1): its arguments, result, commands to the cores and from the host (`kmx_program`) | supported | | census (tutorial 32, 33, 42) |
| X280 threads beyond that one: `kmx_load`, regions, ports, many-core calls, threads on clusters | refused by name | "the Tensix dispatcher does not serve ..." the feature | census (24 lines) |
| X280 ports | refused by name | "X280 ports" | census (10_x280_ports) |
| Shared regions | refused by name | not served by the Tensix dispatcher on either chip | not run on Wormhole |
| Tile functions whose words mean the same on Wormhole (add, multiply, multiply and add, abs, relu, maximum), tile products at every fidelity and operand format | supported | the same words as on Blackhole | census ([tile functions](tile.md#on-wormhole)) |
| The reciprocal, sigmoid, silu, division, tanh | supported, limit | from Wormhole's own estimate: within 1.2e-7 of float64 (tanh within its stated 3e-7) | census |
| A comparison with no register free for its Wormhole form; a load macro's template negating an operand another word reads | refused by name | ([tile functions](tile.md#on-wormhole)) | census |
| A hand-written kernel's Blackhole vector word that Wormhole computes otherwise (a MAD word's negate bit, `SFPGT`, `SFPARECIP`) | refused by name | at the load, naming the word and its source line | census |
| A held field's release in a tile function | supported | | box (one chip) |
| A 16-bit operand read by `.row()` into a float32 result (`x * w.row()`) | supported | Komorebi's own row broadcast on Wormhole: the library's wrote 16-bit rows into the float32 Dst | the line's tests |
| A float32 operand and a 16-bit operand copied in one element-wise tile function | supported | each copy sets its own buffer up ([hazards](hazards.md#a-float32-operand-and-a-16-bit-operand-copied-in-one-tile-function)) | the line's tests |
| A tile function with a matrix product and an operand copied beside it | refused by name | at the build, at the call's line, on both chips: two tile functions in two handlers ([hazards](hazards.md#a-tile-function-with-a-product-and-an-operand-copied-beside-it)) | the line's tests |
| A reduce's sum written in another format than float32 (or held across a loop) | supported | the leader's mover moves past the pages compute freed | the line's tests |
| Block-float pages with a negative zero code | refused by name | at a write ([hazards](hazards.md#a-block-float-page-packed-by-hand-with-a-negative-zero)) | the line's tests |
| Dst words from a pack thread (`km::dst`) | supported | | census |
| Code that reads or writes Dst through Blackhole's RISC-V window (`ckernel_dest.h`): the ray tracer, the raster stages | fails | the build stops: `ckernel_dest.h: No such file` (9 lines) | census |
| The raster project's arithmetic handler and `--spread` | refused by name | "the arithmetic handler needs the Dst window" | census |
| The dense project (matrix products, SwiGLU), the attention project, the raster project's fill | supported | attention at `causal64`: a few words in a million differ from Blackhole's by one bf16 unit; the raster fill writes Blackhole's frame word for word | census |

### Islands: chips of a Galaxy

| Feature | On Wormhole | Limit, or the refusal's name | Ran in |
|---|---|---|---|
| Cards as islands: every chip opened as one fabric mesh, islands on the cards asked, the pairs from the machine's Ethernet connections | supported, limit | up to 31 islands; the one-dimensional fabric forms only, the ring form first where the pairs hold a cycle ([hazards](hazards.md#a-two-dimensional-fabric-form-under-komorebis-link-packets)); one link a pair | box, merge check |
| Commands across islands (a command's part from the host, from a dispatcher), a dynamic run's deliveries | supported | example 05 exact on 2, 4, 8, 16 and 31 islands | box |
| The Tensix dispatcher's waves over more than 17 islands | supported | | box (31) |
| The X280 dispatcher over more than 17 islands | refused by name | "the X280 dispatcher's wave queue holds 16: at most 17 islands" (a chip with an X280 only) | not run on Wormhole |
| `km::write_l1`, `km::add_word`, page reads and writes into another island's L1, asks into L1 | supported | | box |
| A put into a tensor another island holds (`km::put_bytes` batches), an ask answered into DRAM, the host's `send_pages` into a tensor and `cross()` | refused by name | no link packet targets a far Wormhole chip's DRAM: "fardram", the island named ([hazards](hazards.md#a-link-packet-for-a-far-wormhole-chips-dram)) | box |
| The collectives over islands from a kernel and from the host (`km::all_reduce`, `km::reduce_to`, `km::all_gather`, `km::broadcast_from`) | supported, limit | the pages land in the far leader's L1, which writes its own DRAM; about 0.4 GB/s (example 27 on 2 islands); a load that cannot hold two slots of one page a port is refused by name; `all_gather` and `all_reduce` ran, `broadcast_from` and `reduce_to` alone did not | box (27, 28 on 2 and 31), merge check (28) |
| A collective given up mid-way, then `run.clear_collectives()` | supported, limit | exact on 2 islands (2 of 2 runs); on 31 islands it ran only before the meetings became markers, and gave wrong pages then (6 of 7 runs); not run on 31 since | box |
| `km.all_gather` in a chain | refused by name | it puts into far DRAM itself | box |
| Example 29's chain over islands | refused by name | its writer is 18,336 B at writer level 4 | box |
| A program whose island build is over NCRISC's 16 KiB and that never needs the far paths: the near build (`KM_ISLAND_NEAR`) | supported | a mixture-of-experts layer class on 2 and 31 islands, each island bit-identical to its one-card run | box |
| A command of a near build given a tensor not whole on its island, or a reduce, port or broadcast across an island's edge | refused by name | at the submit, naming the tensor and the islands | box |
| A load in the runtime's form on islands (a class made with `form="runtime"`, a dict of programs) | supported | the collectives' landing is sized from the plan's programs; a class's reduce across islands on a near load is refused by name | the line's tests |
| Two links a pair | supported, limit | ran in the link checks on four chips (pages, asks, L1 writes and command lines exact, one link core); a program's writer over its code with second links is refused by name with its size and "links=1"; every other run on the box used one link a pair | box |
| A card split into islands | refused by name | its writer with the link services is 40,220 B for two islands, 47,736 B for three | census |
| X280 threads, ports and ranks on islands | refused by name | as on one chip | census |

## The merge check (2026-11-20)

This tree on the box, the Tensix dispatcher, AICLK 1000 MHz:

- `tests/core`: 2,197 passed, 23 failed. Every failure is the box's environment, not
  Wormhole: 8 `test_card_ready.py` tests build a fake Blackhole driver tree and read the
  Galaxy's own; 11 `tests/core/dsl/thread` tests take the host's default dispatcher for
  the X280's; one test compares the installed package's version with the tree's; 3
  prebuilds stop at the box's tt-metal, which has no X280 objects to link.
- Example 28 (`--dtype float32 --pages 256 --rounds 5`, 1 MiB) on chips 0 and 1:
  exact from a kernel and from the host, 2,701.3 µs and 2,704.2 µs on the card (the
  median of 5).
- Example 28 on 31 islands (all chips but 16), the same arguments: exact from a kernel
  and from the host, 17,610.6 µs and 17,453.9 µs on the card (the median of 5). A first
  try had stopped at a chip left with its compute unit not ready by an earlier job; it
  ran after a whole-box reset.

## Not run on Wormhole

- Four links a pair; two links a pair beyond the link checks; chip 16 of the box in a
  session.
- `km::broadcast_from` and `km::reduce_to` by themselves over islands (they share the
  tree's code with the all-gather and the all-reduce, which ran).
- A collective given up mid-way and cleared, on more than 2 islands, with the meetings
  as markers.
- A host collective on a near load (unit tests only), a crossing reduce, port or
  broadcast group of a near load.
- The model- and reference-dependent scripts and most of `experiments/`.
