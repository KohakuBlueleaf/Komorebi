# A decode's weight reads on one card and on two cards' islands

`reads.py` runs the decode's weight reads alone: the K split's groups (gate: groups of
8 of width 28; down: 19 of width 14), each core streaming its K share of its group's
columns of W and dropping the tiles, no product, no tree. On one card a W of half the
columns is read by the same groups; on two cards W is split so each card's groups read
only the pages that card holds. A card's rate is its half of W over the command's time
(median of 10). tok/s is for a two-card decode of 62 layers of the assumed shapes,
products only (13.6 GB a card a token in bfp8_b, 7.2 GB in bfp4_b).

## Result

Blackhole p150a cards 0 and 1, AICLK 1,350 MHz. Before: an island page call looked
the tensor table up for every page. After: once for each run of pages one island holds
at consecutive indices.

| product | format | one card GB/s | two cards before, GB/s a card | after, GB/s a card | tok/s at that rate, before / after |
|---|---|---|---|---|---|
| gate | bfp8_b | 423.3 | 365.9 | 420.7 | 26.9 / 30.9 |
| down | bfp8_b | 434.4 | 383.8 | 428.1 | 28.2 / 31.5 |
| gate | bfp4_b | 325.0 | 257.2 | 317.8 | 35.7 / 44.1 |
| down | bfp4_b | 334.3 | 262.9 | 323.0 | 36.5 / 44.9 |

## Running it

    python experiments/noc/island_reads/reads.py --cards 0     # one card, about 13 s
    python experiments/noc/island_reads/reads.py --cards 0,1   # two cards

It prints one JSON line a product and format.
