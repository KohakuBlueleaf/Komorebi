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
| 16 ranks | 7.62 µs | 10.48 µs | exact |
| 118 ranks | 10.48 µs | 22.68 µs | exact |

The all-reduce's extra time holds the cast (one multicast a rectangle of the group's
cores, for the pages and then for the count), each rank's compute packing the sum, and
each rank's write of its copy to DRAM, which the plain reduce does at rank 0 alone.
With the count an atomic a rank it was 10.16 and 21.98 µs: not the count.

Across islands (`--islands 2 --ranks 0`: card 0 split in 2, 118 ranks over both), rank
0 writes the sum to island 1's first rank by `km::write_l1`, which casts it on: every
rank exact, 36.6 µs.

The first try counted the cast with one 4-byte multicast from a word at offset 12 of
its 16-byte line to a word at offset 0: the write landed the source line's word 0 (a 0)
and no rank saw the count ([noc_alignment](../../noc/noc_alignment/README.md)). The
count is an atomic a rank since.

## The tree collectives on a copy's pages

`pages/pages.py` calls `km::all_gather`, `km::all_reduce`, `km::reduce_to` and
`km::broadcast_from` from a kernel (every rank of TREE makes the call) and
`run.all_gather`, `run.all_reduce`, `run.reduce_to`, `run.broadcast_from` from the host,
on a float32 copy of 64 pages, each island's copy its own integers in [-8, 8], and
checks every copy against numpy (`reduce_to`: the root's).

```
python pages/pages.py --islands 3 --shape line --rounds 3   # card 0 split in a line
python pages/pages.py --islands 4 --shape ring --rounds 3   # ... in a ring
```

2026-11-16, card 0, the root the last island, the median of 3 commands on the card,
from a kernel (the host's within 1%), every copy exact:

| Islands | all-gather | broadcast_from | all-reduce | reduce_to |
|---|---:|---:|---:|---:|
| 3, a line (116 ranks) | 30.7 µs | 51.7 µs | 328.5 µs | 330.0 µs |
| 4, a ring (112 ranks) | 27.0 µs | 51.4 µs | 331.8 µs | 606.5 µs |

On the ring `reduce_to` adds every page at the root, from both its children; the
all-reduce gives each island the tree of its own share.

With a tree of the user's, `--tree 1 2 3 3` on the ring (a chain rooted at island 3,
`km::tree(low, high)` from the kernel, `tree=[1, 2, 3, 3]` from the host), every copy
exact: broadcast_from 77.5 µs, all-reduce 704.6 µs, reduce_to 629.7 µs.
