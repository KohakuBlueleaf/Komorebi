# A gather by a token list

`item_gather.py`: each of 8 cores reads its page of a token list, 64 entries of
(token, slot), and copies item `token` of a table to item `slot` of the output, item
to item on the movers:

```python
for e in ids.pages(self.rank, 1).entries(Token, size=2, fixed=TOKENS):
    results = km.Stream(out.items(e.slot, 1, size=64), write=True)
    for item in km.Stream(table.items(e.token, 1, size=64)):
        results.push(item)
```

`results.push(item)`, an item pushed into a write stream of items of its size, is a
copy on the movers. Mover A reads each item into its half of the loop's pair, and
mover B writes it from there. Compute takes nothing. Every word is checked against
numpy's `table[tokens]`.

Card 1, 1350 MHz, 2026-11-18, at commit 8a6751d7 (the refusals' commit), one load, a command a size:

| item | copied | us | GB/s | words wrong |
| --- | --- | --- | --- | --- |
| 64 B | 32 KiB (512 items) | 65.04 | 0.504 | 0 |
| 2 KiB | 1 MiB (512 items) | 67.40 | 15.558 | 0 |

The two times are nearly equal. Each token is one read of one item and one write of
it, waited for in turn, so the 64 moves a core makes cost about 1 us each whatever
their size.

## Items of 32 B are refused

An item of a DRAM tensor is a multiple of 64 B. Blackhole reads DRAM at 64-B alignment
(`NOC_DRAM_READ_ALIGNMENT_BYTES`), so a 32-B item that does not start a 64-B line lands
at its offset within the line, not at the start of its buffer page. Run with 32-B
items before the refusal, the gather got 2,104 of 4,096 words wrong. Those are exactly
the items whose token and slot differ in parity: 263 items of 8 words. The 64-B and
2-KiB cases of the same run were exact.

The build now refuses a stream of items of a DRAM tensor whose size is not a multiple
of 64, read or written, by name. `km::items<B>` static_asserts the same for hand
kernels. An L1 tensor's items may be any size that divides its page.

## Two cards

On islands a tensor's word is its table entry, and an items stream reads one island's
tensor. A class built for two cards refuses a stream of items at its line, whether the
table is split over the ranks, copied to each island or held by the other island.
