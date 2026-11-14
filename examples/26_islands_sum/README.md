# 26 islands sum

A sum written for islands: example 02's merge tree, with the island level in the kernel's
own words ([islands](../../docs/concepts/islands.md#acting-by-island)). The same script and
the same kernels run on two cards and on one.

Each core adds its partition of the T input tiles (`km::partition` over the whole group),
then:

1. **Inside its island** (`km::local`): the cores of each island's part reduce to the
   part's first rank, by owner or by tree (a4), as example 02 does on one card.
2. **Across** (`group.leaders()`, `km::to_island(i).put`): each island's leader other
   than the first sends its part to the first leader (`group.leader(0)`), into an arrival
   slot counted from the end of the buffer, on port 7. The first leader waits for one
   part from each other island, adds them in island order and writes the total.

On one card `group.leaders()` is the one leader, `km::local` is the whole group and
nothing crosses: the program is example 02's.

| Command word | Meaning |
|---|---|
| a0, a1, a6 | `input_address`, `tiles` (T), `first_tile` |
| a2, a3 | `output_address`, `first_page` |
| a4 | `form` inside an island: 0 by owner, 1 by tree |
| a7 | `part_tiles` P (1 to 3) |

A part to another island goes through the relay: the leader asks its island's link core
0, whose link 0 CPU reads the pages and sends them over the link to the far link core 0,
which writes them into the arrival slot and bumps the port. The receiving core waits on
its port as for a part from its own island.

## Running it

`sum.py` loads the program, submits SUM on every rank `--rounds` times (from island 0's
dispatcher, which sends the other island's part across), waits for each, and checks the
total against numpy. The input is a copy on every island; integers in [-2, 2], so every
partial sum is exact in float32. The link core's services make the kernels larger than
the runtime's default buffer, so the session opens with `kernel_bytes=76 * 1024`.

```
python sum.py                 # cards 0 and 1, 238 ranks
python sum.py --cards 0       # one card, 120 ranks
```

Blackhole p150a cards, AICLK 1350 MHz, T = 512 tiles, P = 1, tree inside each island, 20
rounds, the median from submit to the event on the host:

| Cards | Ranks | µs a round | Total |
|---|---|---|---|
| 0 and 1 (one link) | 238 | 169.8 (the result word asked: one more pass at the root) | exact |
| 0 | 120 | 158.0 | exact |

A command that crosses completes with the result word of the first part back, not the
group's rank 0's, so the script checks the total itself, not a result word.

## Files

| File | What it is |
|---|---|
| `kernels/sum.h` | handler, buffers, the leaders' port and slots |
| `kernels/reader.cpp` | its partition; the island's parts; at the first leader, the other leaders' |
| `kernels/compute.cpp` | the adds, in rank order inside the island, then in island order |
| `kernels/writer.cpp` | the part to its island's receiver or to the first leader; the total |
| `sum.py` | the program, the run and the check; one JSON line |
