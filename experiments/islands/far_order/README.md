# A put, then an add, to a far island

Question: when one data-movement CPU puts a page into another island's DRAM and then
posts `km::add_word` to a rank there, can that rank see the add before the page has
landed? A CPU's lines are documented to land in the order sent
([link_ops](../link_ops/README.md)); this checks it with a reader that acts the
moment the add arrives.

`order.py` runs, for each case `S:T`, ONE command over the flat ranks from island S's
first rank to island T's last rank, so both halves run at once:

- the sender (island S's first rank) waits until the watcher runs, then for each round
  fills a 4 KiB page with words unique to the round, puts it as page `round % 64` of a
  tensor held by island T (`km::put_pages`, not waited for), posts `km::add_word(...,
  km::posted)` of 1 to the watcher's counter, and only then waits for the put;
- the watcher (island T's last rank) polls its counter; at each rise to `n` it reads
  page `(n - 1) % 64` from its own island's DRAM and checks every word against round
  `n - 1`. A word of an older round is an early read. A read after which the counter
  already shows 63 rounds more is set aside (the sender may have begun that page's next
  put); none was.

Both sides are bounded by the wall clock (`--most`, seconds at 1,350 MHz) and report
in their L1, which the script reads back after the command.

```
python experiments/islands/far_order/order.py --cards 0,1 --cases 0:1,1:0 --rounds 100000
python experiments/islands/far_order/order.py --card 0 --islands 3 --graph 0-1,1-2 --cases 0:2,2:0 --rounds 100000
```

## Results

Main at 8e828d52, 2026-11-17, Blackhole p150a cards, AICLK 1350 MHz, 100,000 rounds a
case, kernels ended with no error:

| Where | Case | Links | Rises seen | Pages checked | Early reads | µs a round |
|---|---|---|---|---|---|---|
| Two cards | 0:1 | 1 | 99,120 | 99,120 | 0 | 9.73 |
| | 1:0 | 1 | 100,000 | 100,000 | 0 | 9.90 |
| Card 0 split, line of 3 | 0:2 | 2 | 100,000 | 100,000 | 0 | 13.32 |
| | 2:0 | 2 | 100,000 | 100,000 | 0 | 13.11 |

No add was seen before the put sent ahead of it, one link or two away. A round is a put
line and its pages, a posted add and the wait for the put's confirmation, one round in
flight at a time.
