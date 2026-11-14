---
title: "Two cards"
part: 6
summary: "one session over two cards: one group of 240 cores, a tensor split over the cards, and a kernel that does not change"
---

# Two cards

A session can open two Blackhole cards at once. Each card's chip is an **island**: its own
grid of cores, its own dispatcher, its own DRAM, and the other card only over the links
between them. The cores of both cards are one group of ranks, 0 to 235 on two cards of 120
cores (the last two cores of each card serve the links, below), and the kernels of one
card's program run on all of them without a change. This
lesson relies on [[core-group]], [[partitions]], [[tiles-and-tensors]] and
[[when-it-goes-wrong]], and it needs two cards joined by Ethernet links in one machine.

## Why a session over two cards

More cores is more work in one command, and more DRAM is more data in one tensor. A program
that spreads its work over a group by rank should not have to learn that some of the ranks
are on another card. So Komorebi keeps the ranks **flat**: island 0's cores are ranks 0 to
117, island 1's are ranks 118 to 235, and `me() - km::first(command)` in a kernel is the
rank in the group, whichever card the core is on. What the host does for you is what the
card boundary asks for:

- a command for a group is cut by card, and each card's dispatcher runs its own part of it;
- a tensor is placed by card, so a rank that takes its part of a tensor finds it in its own
  card's DRAM;
- an error or a time-out says which card it came from.

A kernel can still reach the other card, and when it does the crossing is slower than
the NoC. `km::read_pages` and `km::write_pages` ask for and put pages the other card
holds, and a message, a reduce and a broadcast cross the links too. This lesson is the
part that needs no change in a kernel; [[a-sum-across-cards]] is the part that does. A
command for a group over both cards is two commands, one for each card, and the host
sends both before it waits.

## One line at a time

The program is [`islands.py`](../../examples/01_mapcollect/islands.py) of
[`01_mapcollect`](../../examples/01_mapcollect/README.md), with the program and the kernels
of that example. Run it with `--whole` and both cards visible and free: the commands are
then one group of every rank. The lines below are the ones of the file that this lesson
talks about.

The session takes the cards as a list, `km.Session(cards=[0, 1])`, and nothing else (the
script opens it in `open_session`). Every card is opened in one mesh, one X280 runtime is
opened over it, and each card gets a session of its own, a dispatcher and its grid. The
session uses every link between the two cards (four on two p150a), and each card gives
the last of its cores to the links: a link core carries what crosses, and is not a rank.
The kernel buffer (160 KiB) is sized for the link services the kernels carry. The number
of ranks is the cores of both cards less their link cores:

```{ .python }
--8<-- "examples/01_mapcollect/islands.py:158:164"
```

`session.graph` is the host's picture of the islands: `graph.first(i)` is island `i`'s
first rank and `graph.count(i)` its number of ranks, 0 and 118 for island 0, 118 and 118
for island 1 with four links (two link cores a card).

The tensors come next. The input has two tiles for each rank of both cards; the output has
one:

```{ .python }
--8<-- "examples/01_mapcollect/islands.py:166:169"
--8<-- "examples/01_mapcollect/islands.py:178:179"
```

A tensor is one logical tensor: each card holds only its part. With no hint its pages
are split along the leading dimension over the ranks in rank order, with the rule of
`km::partition`; `split=d` splits dimension `d` instead and `group=` names the ranks that
split it (`--split`, `--group` here). So a kernel is given `source.address` and
`source.accessor_args()` as on one card, and rank `r`, which reads the tiles `2r` and
`2r + 1`, finds them in its own card's DRAM; `km::read_pages` asks the other card for a
page it holds. `session.read_bf16(source)` assembles the whole tensor from the card that
holds each page:

```{ .python }
--8<-- "examples/01_mapcollect/islands.py:239:239"
```

The program is loaded on every card, one after the other, and each card's cores get the
island table that tells them where the ranks are:

```{ .python }
--8<-- "examples/01_mapcollect/islands.py:229:229"
```

A command for a group of every rank:

```{ .python }
--8<-- "examples/01_mapcollect/islands.py:373:377"
```

`submit` takes the group `(0, ranks)`, the host cuts it into the part on each card, and
`wait` sends every card's queued commands first, so the cards run their parts at once, and
then waits for all of them. The result for one command is not one event but a tuple, one
event for each card that ran a part, in card order. The ticket that `submit` returns is the
handle of a command over several cards.

When the run is stopped, the output is read back in the same way:

```{ .python }
--8<-- "examples/01_mapcollect/islands.py:272:272"
```

## What changes in the kernel

Nothing. The reader of the example asks for its rank and its share the way it does on one
card:

```{ .cpp }
--8<-- "examples/01_mapcollect/kernels/reader.cpp:15:15"
```

```{ .cpp }
--8<-- "examples/01_mapcollect/kernels/reader.cpp:21:23"
```

`reader.me()` is the flat rank, `km::count(command)` is the whole group's size, and the
partition of the tiles is the one of [[partitions]]. A kernel that wants to know where it
is can ask, and the answers cost nothing:

| Call | What it answers |
|---|---|
| `km::island()`, `km::islands()` | this core's card (0 or 1), and the number of cards of the run (2) |
| `km::island_of(rank)`, `km::crosses(rank)` | the card that holds `rank`, and whether a send to it would cross a link |
| `km::hops(i)` | the links between this card and card `i` (0 for itself, 1 for the other) |
| `group.local()` | the part of a group that is on this card |

A program that never asks is unchanged. A program that asks gets the same answers on one
card as it did before: card 0, one card, and no crossing. The
[islands](../concepts/islands.md) page has every call and the table behind them.

## The whole program

=== "islands.py"

    ``` { .python }
    --8<-- "examples/01_mapcollect/islands.py"
    ```

The script prints one JSON line, and exits with a failure if a check failed. The
[islands reference](../reference/islands.md#measured) records one run, with its conditions:
two Blackhole p150a cards joined by four Ethernet links, one process, the example's program
and kernels unchanged, the X280 at 1,350 MHz, taken when every core of both cards was a
rank (240 ranks; with the link cores of today it is 236). In it, with `--whole`, the map command ran on
one group of all 240 ranks, where rank `r` adds the tiles `2r` and `2r + 1` of its own
card's part of the input. The 240 tiles of the last timed batch were bit-exact against the sums
of `mapcollect.py`, and both cards' kernels ended with no error. The time of a command
from submit to event, pipelined, was:

| What | µs a command |
|---|---|
| NOP on island 0's 120 ranks | 1.195 |
| NOP on all 240 ranks, a part on each card | 2.410 |
| MAP on all 240 ranks, the input split over both cards | 5.215 |

The numbers depend on the machine, as in every lesson. The NOP on one card costs what it
costs on a card opened alone, and a command over both cards costs the host two commands'
work, once for each card.

With `--hang` the same script hangs a command on a rank of card 1 and asks what that rank
was doing, by its flat rank as everywhere else:

```{ .python }
--8<-- "examples/01_mapcollect/islands.py:414:414"
```

`run.inspect((rank, 1))` has island 1's dispatcher read the core of that rank; the answer
names it `rank 128 (island 1, core 10)`, the card and the core added as information. In
the recorded run the inspect named the reader waiting for a port arrival, the deadline
gave the command up, the time-out named island 1 and its ranks, the run went on, and 16
pages of island 1 read inside the run were exact. The way to read
such a time-out is [Recover from a command that does not return](../guides/recovery.md).

## Common mistakes

- Reading a page that another card holds with a plain `TensorAccessor` from a kernel.
  `source.address` over cards is not a bank address but the tensor's entry in the
  tensor table; `km::read_pages` and `km::write_pages` find the card and the part.
- Giving a dynamic run (the messages of [[routed-message]]) a group of only some of the
  ranks over two cards. Its RUN takes every rank, `(0, session.ranks)`; another group
  raises `ValueError`. `km::crosses(rank)` tells a kernel that a send would cross.
- Taking the result of a command for one event. Over two cards it is a tuple, one event for
  each card with a part.
- Opening cards another process holds, or naming cards that are not the visible ones. A
  card held by another process is refused (`km.CardBusy`), and `TT_VISIBLE_DEVICES` naming
  other cards raises `ValueError`.

The next lesson writes a kernel for islands: [[a-sum-across-cards]]. The
[islands](../concepts/islands.md) page says what an island is and what the links between
two cards carry, and the [islands reference](../reference/islands.md) lists every call of
the host.
