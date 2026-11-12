# Experiments

Standalone proofs of one behaviour of the hardware or of a design choice: a small
program, how to run it, and the numbers it prints. An experiment does not import the
rest of the repository's code unless it is testing that code, and it states the
conditions of every number (card, clocks, sizes, load).

The rule is: experiments first; a regression test comes after behaviour settles.
A test pins down behaviour that is understood. Until then, an experiment is the right
place to ask a question and to be wrong.

## Experiments

- [landing](landing/README.md): how a core knows that a 64-byte record written by
  multicast has landed completely. Echo words in every 16-byte chunk against a record
  followed by a sequence word, on a Blackhole p150a with 120 cores. Decision: echo words.
- [tile_probes](tile_probes/README.md): what one Tensix tile costs for a 32-ray walk:
  predicated vector instructions, the "any lane enabled" register, a per-lane gather
  through the `Dst` window, one box against 32 rays, the compute RISC-V cores, and one
  full walk of a 255-node box tree ray-first against node-first, on one core of a
  Blackhole p150a. Predication is free; the gather costs 240 to 2280 cycles per 32-lane
  step depending on how the core copies; the node-first walk is 1.6 to 2.3 times faster
  from 256 rays on, because the ray-first walk is bound by the core's stack and copy
  work.
- [collision](collision/README.md): whether the X280's record multicasts on NoC0 and
  lane broadcasts inside a grid column (NoC0 or NoC1, short blocks checked word by
  word and long multicasts back to back) can run at the same time. No lost or wrong
  word and no hang in any arrangement measured; NoC1 recommended for lanes while the
  X280 dispatches.
- [noc_alignment](noc_alignment/README.md): what a NoC write shorter than 16 bytes does
  when its source and destination differ in their low four address bits. It lands at
  the destination, carrying the source line's bytes at the destination's offset, and is
  acknowledged as usual: silently wrong. Small writes go from a 16-byte slot to a
  16-byte slot.
- [stream_checks](stream_checks/README.md): the reference streams of the micro-op
  simulator and a mutation run over its source. Every one of 104 mutants changes some
  saved answer.
- [runtime_harness](runtime_harness/README.md): the Tensix runtime's kernels as host
  threads on a simulated grid, with failure paths forced on purpose (stop words,
  give-ups, lost chunks, mailbox races, items past a buffer's end). It found seven
  defects in the runtime and replays a card hang; 24 scenarios, run by hand.
