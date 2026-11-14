---
title: "A sum across cards"
part: 6
summary: "a kernel written for islands: each card adds its own part first, and one part from each card crosses a link"
---

# A sum across cards

The program of [[two-cards]] ran on both cards without a change. This lesson writes a
kernel that says where the cards are, in its own words: a sum of the tiles of every
rank, where each card adds its own part and only one part from each other card goes over
a link. It relies on [[reduce]], [[the-leader]], [[partitions]] and [[two-cards]], and it
runs on two cards joined by Ethernet links, or on one card.

## Why say it

What crosses between the cards is slower than what stays inside one: the cores of a card
reach each other over the NoC, and a core of one card reaches a core of the other only
over a link. A [[reduce]] over a group that spans both cards already keeps that in mind
without being told: each card's part reduces as a group of its own, and each part's first
rank sends once to the group's rank 0. That is what the kernels of
[`02_merge_tree`](../../examples/02_merge_tree/README.md) do over two cards, unchanged.
A kernel written for islands takes the two steps apart itself, so that it decides where
each card's part goes and in what order the parts are added:

1. **Inside each card**, the ranks of that card's part of the group reduce to the part's
   first rank, by owner or by tree as in [[reduce]].
2. **Across**, the first rank of each card other than the first sends its part once to
   the first rank of the group, which adds the parts in card order and writes the total.

On one card the second step has nothing to do: the group has one first rank, nothing
crosses, and the program is the sum of [[reduce]]. The same script and the same kernels
run on two cards and on one.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow", "steps": [{"t": "every rank adds its share of the tiles", "gem": "amber"}, {"t": "inside each card: the ranks of its part reduce\nto the part's first rank (km::local)"}, {"t": "across: card 1's first rank sends its part\nto card 0's first rank, once", "gem": "amber"}, {"t": "card 0's first rank adds the parts\nand writes the total"}]}
</script>
<figcaption>The sum written for two cards: one part crosses, not one for each rank of the other card.</figcaption>
</figure>

## The general form

Four calls say it, each with an answer on one card:

| Call | What it does | On one card |
|---|---|---|
| `km::reduce_arrive(..., km::local)`, `km::reduce_emit(..., km::local)`, `km::reduce(..., km::local, ...)` | the reduce stays inside this core's card | the whole group |
| `group.leaders()` | the group of the first rank of each card's part: `.size` of them, `.rank` this core's place among them (`.size` for a core that is not one) | one leader, the group's first rank |
| `group.leader(j)` | the first rank of part `j` of the group, the parts in card order (`leader(0)` is the group's first rank) | the group's first rank |
| `km::to_island(i).put(writer, command, arrival, cb, pages, rank, port, slot)` | sends `pages` of buffer `cb` to an arrival slot of `rank` on card `i`, counted at `port`; it gives the command up if `rank` is not on card `i` | a send inside the card |

`km::local` is the scope; `group.local()` is the part of the group on this core's card.
A core that is not the first of its card's part has nothing to do in step 2.

## One line at a time

The program is [`26_islands_sum`](../../examples/26_islands_sum/README.md): `sum.py`, and
the kernels `reader.cpp`, `compute.cpp` and `writer.cpp` with `sum.h`. The handler, the
buffers and the place where the other cards' parts arrive are in the header:

```{ .cpp }
--8<-- "examples/26_islands_sum/kernels/sum.h:10:24"
```

`kCbArrive` is the arrival buffer, `kArrivePages` (118) pages. The reduce inside a card
uses its slots from the start; the part of card `j`, for `j` above 0, is written into
slot `cross_slot(part_tiles, j)`, which counts from the end, so the two never meet.
`kCrossPort` is the port at the first rank that counts those parts.

In `reader.cpp`, each core reads its share of the `T` input tiles, which is the
`km::partition` of [[partitions]] over the whole group, and then waits for the parts it
is owed:

```{ .cpp }
--8<-- "examples/26_islands_sum/kernels/reader.cpp:47:49"
```

```{ .cpp }
--8<-- "examples/26_islands_sum/kernels/reader.cpp:79:86"
```

`group_of` is the whole group, as in [[reduce]]. The tiles are read with
`km::read_pages` (line 64), as in [[two-cards]], so they are found wherever the session
put them. Then `reduce_arrive(..., km::local)` waits for the parts sent inside the card,
and, only at the first leader (`leaders.rank == 0`) and only when there are other cards
(`leaders.size > 1`), `km::arrive` waits for one part from each other card on `kCrossPort`.

The writer sends:

```{ .cpp }
--8<-- "examples/26_islands_sum/kernels/writer.cpp:109:134"
```

`group.local().rank != 0` is every core that is not the first of its card's part: it
sends its sum to its owner or its tree partner inside the card with `reduce_emit`.
The first core of a card's part (the rest of the function) has the card's sum; if it is
not the group's first leader (`leaders.rank != 0`), it puts that sum into the first
leader's arrival slot, on the card of that leader, `km::island_of(root)`. A part that
goes to another card crosses a link: the call asks the link core of its card, which
sends it, and the receiving core waits on its port as it does for a part from its own
card. The first leader falls through to the total:

```{ .cpp }
--8<-- "examples/26_islands_sum/kernels/writer.cpp:135:137"
--8<-- "examples/26_islands_sum/kernels/writer.cpp:151:160"
```

`km::reduce_result` waits for the sum, and `km::write_pages` writes it into the output
tensor wherever that tensor's page is held. (Between the two pieces the writer sets the
command's result word when it was asked for one, as in [[a-result]].)

The compute kernel adds in the same order, so the answer does not depend on which part
arrived first: its own tiles, then its card's parts, then at the first leader the other
cards' parts in card order:

```{ .cpp }
--8<-- "examples/26_islands_sum/kernels/compute.cpp:76:86"
```

The script decides where the input lives and sends one command:

```{ .python }
--8<-- "examples/26_islands_sum/sum.py:54:60"
```

By default the input is a copy on every card (`{"copy": True}`), so every rank's tiles
are in its own card's DRAM and nothing is asked for. With `--far` the input is held by
the last card's ranks alone, so the ranks of the other cards ask across for theirs,
rank 0 among them, while it counts the reduce on its ports.

```{ .python }
--8<-- "examples/26_islands_sum/sum.py:103:110"
```

The output is a copy as well. The session is opened over the cards with one link, so
each card has one link core and two cards have 238 ranks:

```{ .python }
--8<-- "examples/26_islands_sum/sum.py:93:94"
--8<-- "examples/26_islands_sum/sum.py:97:97"
```

The command is given to card 0's own run, so that the card's dispatcher sends the other
card's part across:

```{ .python }
--8<-- "examples/26_islands_sum/sum.py:114:121"
```

`island0.submit("sum", (0, ranks), words)` is a group of every rank, in flat ranks. A
command that crosses completes with the result word of the first part back, not the
group's rank 0's, so the script does not use a result word: it reads card 0's copy of the
output after the run has stopped and compares it with numpy:

```{ .python }
--8<-- "examples/26_islands_sum/sum.py:138:141"
```

## The whole program

=== "sum.py"

    ``` { .python }
    --8<-- "examples/26_islands_sum/sum.py"
    ```

=== "kernels/sum.h"

    ``` { .cpp }
    --8<-- "examples/26_islands_sum/kernels/sum.h"
    ```

=== "kernels/reader.cpp"

    ``` { .cpp }
    --8<-- "examples/26_islands_sum/kernels/reader.cpp"
    ```

=== "kernels/compute.cpp"

    ``` { .cpp }
    --8<-- "examples/26_islands_sum/kernels/compute.cpp"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "examples/26_islands_sum/kernels/writer.cpp"
    ```

The script prints one JSON line with `ok`, true when the total is exact, every round
was done and every kernel ended with no error. Run it with `python sum.py` for cards 0 and 1,
or `python sum.py --cards 0` for one card. The example's page records these runs
([`26_islands_sum`](../../examples/26_islands_sum/README.md#running-it)): Blackhole p150a
cards, AICLK 1,350 MHz, 512 input tiles, one tile a part, the tree inside each card, 20
rounds, the median from submit to the event on the host, every total exact:

| Cards | Ranks | µs a round |
|---|---|---|
| 0 and 1 (one link) | 238 | 165.9 |
| 0 | 120 | 158.0 |
| 0 split in 2 (`--islands 2`) | 118 | 162.18 |
| 0 split in 3 | 114 | 174.19 |
| 0 split in 4 | 108 | 168.37 |

The numbers depend on the machine, as in every lesson. Two cards take 7.9 µs more than
one card for a sum over twice the ranks. The kernels of example 02 over two cards, unchanged
and by tree, took 172.6 µs against 162.8 on one island of 119 ranks; those rows asked for
a result word, one more pass over the total at the root, and the rows above did not. The
rows below the second are one card described as two to four islands, which is the way to
check a program written for islands with one card
([islands](../reference/islands.md#a-card-split-into-islands)).

## Common mistakes

- Sending a part to a rank that is not on the card named. `km::to_island(i).put` checks
  that `rank` is on card `i` and gives the command up if it is not. Take the card from
  the rank, `km::to_island(km::island_of(root))`, as the writer does.
- Putting two cards' parts into the same arrival slot. A slot holds one part; the second
  overwrites the first. The part of card `j` goes to its own slot, here counted from the
  end of the buffer.
- Waiting for the cards' parts on every rank. Only the first leader is sent any, and a
  rank that waits on `kCrossPort` without being sent anything waits until its wait runs
  out. The reader tests `leaders.rank == 0 && leaders.size > 1`.
- Reading a result word for the total. A command that crosses completes with the first
  part back; write the total into a tensor and read that.

That is the last lesson. The [islands](../concepts/islands.md) page says what an island
is and what the links between cards carry, the
[islands reference](../reference/islands.md) lists every call of the host and of the
kernel, and [every primitive](every-primitive.md) is the table of the whole model.
