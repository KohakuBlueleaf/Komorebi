---
title: "Casting a word"
part: 4
summary: "the leader of a row reads data, decides what the row does next and tells it with a word"
---

# Casting a word

In [[broadcast]] the sender cast tiles. A cast can also carry one 32-bit **word**, with a
block of tiles or alone. This is how one core leads a group: it looks at data the
others have not seen, decides what they all do next, and tells them in one cast. This
lesson relies on [[broadcast]] and [[the-leader]].

## Why a word

The cores of a group usually run the same code on different data. Sometimes the next
step depends on something only one core has seen: a value it read, a count it made.
That core, the **leader**, could write the answer to DRAM and every other core could
read it back; a cast is quicker. The steps themselves are already in every core's
kernel: the word only says which one to take.

A broadcast that carries words alone needs no buffer: nothing goes to compute, and no
tile moves. Its numbers on the host end in `(..., 0, 0)`: a window of 0 blocks of 0
pages.

## The general form

Reader of every core of the broadcast, in the function given to `run`:

```cpp
km::Broadcast NAME(reader, command, FIRST_WORD);
if (!NAME.ok() || !NAME.included()) {
    return;
}
for (uint32_t cast = 0; cast < COUNT; cast++) {
    if (NAME.sender()) {
        ... decide WORD ...
        if (!NAME.cast(cast, WORD)) { return; }
    } else if (!NAME.arrive(cast)) {
        return;
    }
    ... act on NAME.word() ...
}
NAME.finish();
```

- `cast(cast, WORD)` sends the word to every other core of the broadcast. On a broadcast
  with a buffer it is called after `reserve` and sends the block with it, as in
  [[broadcast]].
- `arrive(cast)` waits until that cast has reached this core.
- `word()` is the word of the last cast, on the leader as on the others, so every core
  can run the same line after the `if`.
- Before each cast, the leader waits until every core took the last word. A word is
  never lost, but each cast of a word costs a trip to the farthest core and back.

## One line at a time

```{ .python }
--8<-- "tutorial/45_casting_a_word/casting_a_word.py:plan"
```

The same plan as in [[broadcast]], with `(0, 0, 0, 0)`: broadcast 0, no buffer, a window
of 0 blocks of 0 pages. The program's two buffers are not the broadcast's: one holds
the values the leader reads, the other the total each core writes.

```{ .cpp }
--8<-- "tutorial/45_casting_a_word/kernels/reader.cpp:run"
```

The leader reads one page of eight values. For each value it casts `kDouble` if the
value is even and `kAddThree` if it is odd. Every core, the leader too, starts from a
total of 1 and doubles it or adds three, as the word says. After `finish`, each core
writes its total to its row of `totals`.

```{ .python }
--8<-- "tutorial/45_casting_a_word/casting_a_word.py:follow"
```

One command, `exclusive=True`, to the six cores of the row.

## The whole program

=== "casting_a_word.py"

    ``` { .python }
    --8<-- "tutorial/45_casting_a_word/casting_a_word.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/45_casting_a_word/kernels/handlers.h"
    ```

=== "kernels/reader.cpp"

    ``` { .cpp }
    --8<-- "tutorial/45_casting_a_word/kernels/reader.cpp"
    ```

=== "kernels/compute.cpp"

    ``` { .cpp }
    --8<-- "tutorial/45_casting_a_word/kernels/compute.cpp"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/45_casting_a_word/kernels/writer.cpp"
    ```

```{ .text .km-output }
--8<-- "tutorial/45_casting_a_word/output.txt"
```

Only core 0 read the values, and every core ended with the same total: each one took
the same eight steps in the same order.

## Common mistakes

- `cast(cast)` without a word on a broadcast without a buffer, or `reserve` on it. There
  is no block to send: the core gives up (`broadcast_begin`).
- Using `word()` before `arrive` returned true. It is still the word of the cast before.
- Expecting a word on a block cast with `cast(block)`. A block cast without a word gives
  `word()` 0.

[examples/15_leader](../../examples/15_leader/) runs a million casts down every column of
the card and checks every word that every core took; the reference is
[Broadcast](../reference/broadcast.md).

Next: [[routed-message]] sends to a core that the data picks.
