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

A part to another island goes by `km::write_l1`: a line to the leader's link core toward
that island, whose page CPUs send the pages across (any number of links); the far link
core writes them into the arrival slot and bumps the port. The receiving core waits on
its port as for a part from its own island.

The input is read and the total written through Komorebi's page calls
(`km::read_pages`, `km::write_pages` over a `km::Tensor`), so they hold wherever the
session placed the tensors. Placed by the command's group (the session's default), each
rank's tiles are in its own island and nothing is asked for; a page another island holds
is asked for (the input) or put (the total) and counted at the reader's or the
writer's page port, apart from the ports the reduce and the leaders' parts count on.

## Running it

`sum.py` loads the program, submits SUM on every rank `--rounds` times (from island 0's
dispatcher, which sends the other island's part across), waits for each, and checks the
total against numpy. The input is a copy on every island; integers in [-2, 2], so every
partial sum is exact in float32. The link core's services make the kernels larger than
the runtime's default buffer; a session over two cards opens with one that holds them.

```
python sum.py                 # cards 0 and 1, 238 ranks
python sum.py --cards 0       # one card, 120 ranks
```

Blackhole p150a cards, AICLK 1350 MHz, T = 512 tiles, P = 1, tree inside each island, 20
rounds, the median from submit to the event on the host:

| Cards | Ranks | µs a round | Total |
|---|---|---|---|
| 0 and 1 (one link) | 238 | 167.5 (2026-11-16; 165.9 before) | exact |
| 0 | 120 | 158.0 | exact |
| 0 split in 2 (`--islands 2`, a wire) | 118 | 162.18 | exact |
| 0 split in 3 | 114 | 174.19 | exact |
| 0 split in 4 | 108 | 168.37 | exact |

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
