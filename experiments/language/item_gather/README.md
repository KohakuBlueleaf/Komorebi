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

Card 1, 1350 MHz, 2026-11-18, one load, a command a size:

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

On islands an item the reading core's island holds is read from its bank as on one card;
another island's item is one ask for its bytes, a block's items asked together and
waited for once; a written item goes the same way (a put of its bytes). Items there are
a multiple of 64 B; another size is refused by name.

The same gather, 8 ranks (4 on each of two cards), 512 items, exact in each case; one run
each, the clock not read with the cards open:

| placement | 64 B | 2 KiB |
|---|---:|---:|
| the session's spread (the table on card 0, out split) | 783.0 us | 672.6 us |
| the table on card 1 alone, out on card 0 alone | 799.8 us | 1,416.9 us |

A rank copies its 64 items one at a time (a stream of one item a token), so these are
about 10 to 22 us an item against about 1 us on one card (65 and 67 us above): each
item another card holds is an ask across the link and its answer, waited for alone.
