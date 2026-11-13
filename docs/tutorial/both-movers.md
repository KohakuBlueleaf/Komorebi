---
title: "Both movers"
part: 3
summary: "both data-movement CPUs of a core carry one stream: a pattern deals its blocks between them"
---

# Both movers

A core has two data-movement CPUs. So far one of them read, with `reader.cpp`, and the
other wrote, with `writer.cpp`. Here both take part in one stream of tiles: a pattern
says which of the two moves each block, and each moves its blocks through a buffer of its
own. This lesson relies on [[reading-from-dram]], [[writing-to-dram]] and
[[circular-buffers]].

## Why both

Each data-movement CPU sends its requests over its own NoC: the first over NoC 0, the
second over NoC 1 ([[hello-core]]). One CPU that reads tile after tile keeps one NoC busy
and leaves the other idle, and on one core it is the CPU that sets the pace, not DRAM. Two
CPUs that each read every other block keep two reads in flight on two NoCs.

`reader.cpp` and `writer.cpp` are roles: a CPU loaded with `reader.cpp` reads because
your code tells it to, not because it can do nothing else. When one stream is the work of
a command, both CPUs can carry it. In this part they are called the **movers**: mover A
is the CPU of `reader.cpp`, on NoC 0, and mover B the CPU of `writer.cpp`, on NoC 1.

## The general form

A stream is cut into **blocks** of a few tiles. A **pattern** is a word of the letters
A and B: block k goes to the mover of letter k, counted round the word. `"AB"` gives the
movers every other block; `"AAAB"` gives mover A three blocks of every four.

```python
km.Pattern(LETTERS).word  # a command argument
km.BufferPair(INDEX_A, INDEX_B, DTYPE, page_bytes=BYTES, pages=PAGES)
km.Program(NAME, kernels=FOLDER, compile_args={"movement": ARGS}, ...)
```

```cpp
km::Movement movement;
const km::Stream stream(DIRECTION, TENSOR, FIRST_TILE, TILES, BLOCK_TILES, 1,
                        {INDEX_A, INDEX_B}, km::Pattern::of(WORD));
movement.move(stream);  // this mover's blocks of the stream; false: it gave up
```

- `FOLDER` holds `movement.cpp` and `compute.cpp`. The one `movement.cpp` is built for
  both data-movement CPUs; `km::Movement` is a `km::Reader` on mover A and a `km::Writer`
  on mover B, so the loop of [[hello-core]] runs as before.
- A **buffer pair** is two circular buffers, one for each mover: its **halves**.
  `km.BufferPair` declares both with one format and page size, `PAGES` pages each.
- `DIRECTION` is `km::kMoveRead` (tiles from DRAM into the mover's half) or
  `km::kMoveWrite` (tiles from its half to DRAM). `TILES` tiles from `FIRST_TILE` make the
  stream, `BLOCK_TILES` to a block; the last block may be shorter.
- Both movers call `move` with the same stream. Each moves only the blocks the pattern
  gives it, waits for room or data in its own half, and returns when its blocks are done.
- A half is filled by one CPU and emptied by one CPU for as long as the program is
  loaded, as every buffer is ([[circular-buffers]]).

## One line at a time

```{ .cpp }
--8<-- "tutorial/23_both_movers/kernels/movement.cpp:run"
```

This is the copy of [[circular-buffers]], with the loop of `reader.cpp` and the loop of
`writer.cpp` replaced by two streams. `read` brings the tiles from the source into the
halves, `write` sends them on from the halves to the destination. `move` takes both: each
mover reads one of its blocks into its half, then writes it on from there, so no
compute kernel stands between them and each half is filled and emptied by its own mover.

```{ .python }
--8<-- "tutorial/23_both_movers/both_movers.py:program"
```

The pair holds 16 pages in each half: two blocks of 8 tiles, so that a mover can read
its next block while it writes the last one. `"movement"` gives the same tensor
arguments to both movers.

```{ .python }
--8<-- "tutorial/23_both_movers/both_movers.py:copy"
```

The same command twice, on one core: with pattern `"A"` mover A copies every block
alone, with `"AB"` the movers take every other block.

## The whole program

=== "both_movers.py"

    ``` { .python }
    --8<-- "tutorial/23_both_movers/both_movers.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/23_both_movers/kernels/handlers.h"
    ```

=== "kernels/movement.cpp"

    ``` { .cpp }
    --8<-- "tutorial/23_both_movers/kernels/movement.cpp"
    ```

=== "kernels/compute.cpp"

    ``` { .cpp }
    --8<-- "tutorial/23_both_movers/kernels/compute.cpp"
    ```

```{ .text .km-output }
--8<-- "tutorial/23_both_movers/output.txt"
```

One mover copied the 256 tiles in 39.6 microseconds, both in 21.6: nearly twice as
fast, as two CPUs each did half of the reads and half of the writes on two NoCs. The
last line is the program's check: both copies hold every tile of the source.

## Common mistakes

- Expecting both movers to be faster everywhere. On one core they are. On many cores
  the two NoCs reach DRAM at different rates, and the best pattern follows them: on the
  whole card `"AAAB"` reads fastest ([Movement](../reference/movement.md#choosing-a-pattern)).
- Giving a mover's stream to a buffer the compute kernel fills in another command. A half
  keeps the CPU that fills it and the CPU that empties it for the whole load; `move`
  gives up if they change.
- A buffer that is not a whole number of blocks. Each block is one item of
  `BLOCK_TILES` pages, and a half must hold a whole number of them.

A compute kernel that takes blocks from a pair takes them in order, block k from the
half `km::block_half(pair, pattern, k)`; the [Movement](../reference/movement.md)
reference page has the rules.

That is the end of Part 3. A core can now read tiles from DRAM, work on them with the
compute kernel and the matrix unit, write them back with one or both of its
data-movement CPUs, and a group can do that for a whole tensor. Part 4 lets cores pass
tiles to each other, without a trip to DRAM.

Next: [[why-cores-talk]].
