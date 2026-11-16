# Islands tree: a reduce whose sum every rank holds

Question: can the ports' reduce leave its sum at every rank of the group in the same
command, and what does that cost beside the plain reduce, whose sum only rank 0 holds?

`tree.py` loads one program (`kernels/`). Each rank reads its P float32 tiles (integers
in [-8, 8], so every sum is exact), the group reduces them by tree, and then:

- the plain reduce: rank 0 writes the sum to the output at its page;
- the all-reduce (`km::all_reduce_arrive`, `km::all_reduce_emit`,
  `km::all_reduce_take`): rank 0 casts the sum back to every other rank, whose compute
  packs it, and every rank writes its copy to the output at its own page.

The host checks every holder's copy against numpy and takes each command's time on the
card from its event.

```
python tree.py --ranks 16 118 --plain 118 --rounds 5   # card 0
```

## Results

2026-11-16, card 0 (p150a) as one island, AICLK 800 MHz, the tree form, one tile a
rank, the median of 5 commands:

| Group | Plain reduce | All-reduce | Every copy |
|---|---:|---:|---|
| 16 ranks | 7.62 µs | 10.16 µs | exact |
| 118 ranks | 10.44 µs | 21.98 µs | exact |

The all-reduce's extra time holds the cast (one multicast a rectangle of the group's
cores, then an atomic a rank for the count), each rank's compute packing the sum, and
each rank's write of its copy to DRAM, which the plain reduce does at rank 0 alone.

The first try counted the cast with one 4-byte multicast from a word at offset 12 of
its 16-byte line to a word at offset 0: the write landed the source line's word 0 (a 0)
and no rank saw the count ([noc_alignment](../../noc/noc_alignment/README.md)). The
count is an atomic a rank since.
