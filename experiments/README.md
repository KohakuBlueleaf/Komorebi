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
  through the `Dst` window, one box against 32 rays, and the compute RISC-V cores, on
  one core of a Blackhole p150a. Predication is free; the gather costs 240 to 2280
  cycles per 32-lane step depending on how the core copies.
