---
title: "A word with each range"
part: 5
summary: "a pull: cores take ranges of work from their X280 core one after another, and a word with each range tells them what to do with it"
---

# A word with each range

Cores can take their work from an X280 core a piece at a time: each asks for the next
range when it is ready, and the answer can carry a word that says what to do with that
range. This lesson relies on [[core-asks-x280]].

## Why a pull

If every core gets an equal share of the work at the start, the slowest share sets the
time. In a **pull** the cores take ranges one after another instead, so a core that
finishes early simply takes more. The X280 core that serves a core hands out its ranges,
and it can decide, range by range, which one comes next and what the core should do with
it. That second part is the **word**: one 32-bit number that comes beside the range. A
word can name a way to process the range, for example one of several pieces of code
(`examples/18_thread_split` names an overlay, [[overlays]]).

## The general form

In the thread, a function that gives a core its next range and its word, and the call
that runs a command and serves its pulls:

```c
static uint64_t pick_word(int core, uint32_t index, uint64_t argument) {
    return KMX_PULL_WORD(RANGE, WORD);  /* RANGE: KMX_PULL_NONE when no range is left */
}

const kmx_pull pull = {UNITS, PER, KMX_PULL_COUNT, 0, 0, ARGUMENT, pick_word};
kmx_event event;
int status = kmx_pull_run(&command, &pull, &event);
```

On the Tensix core, in `reader.cpp`:

```cpp
uint32_t first = 0, word = 0;
while (const uint32_t count = port.next(first, word)) {
    ... units first to first + count - 1, as word says ...
}
```

- The work is the units 0 to `UNITS - 1`, cut into ranges of `PER` units: range `r` is
  the units from `r * PER`, `PER` of them (fewer in the last one).
- `pick_word(core, index, argument)` is called on the X280 core that serves `core`, when
  that core asks for its range number `index` (0, 1, 2, ... for each core). It returns the
  range and the word, put together by `KMX_PULL_WORD(RANGE, WORD)`. `argument` is the
  `ARGUMENT` of `kmx_pull`.
- `kmx_pull_run(&command, &pull, &event)` submits the command, like `kmx_port_run` in
  [[core-asks-x280]], and answers every core's asks with `pick_word` until the command
  ends.
- `port.next(first, word)` asks for the next range and waits for it. It returns the
  range's count and sets `first` and `word`; it returns 0 when no range is left.
- A `kmx_pull` without `pick_word` hands out the ranges in order from a counter, and the
  word is then 0; `port.next(first)` takes a range without a word.

## One line at a time

```{ .python }
--8<-- "tutorial/47_a_word_with_each_range/a_word_with_each_range.py:ports"
```

Tensix cores 0 to 3 are all served by core 0 of cluster 1.

```{ .c }
--8<-- "tutorial/47_a_word_with_each_range/thread/pick.c:pick"
```

The numbers 0 to 63 make 16 ranges of 4. Core `c` gets ranges `c`, `c + 4`, `c + 8` and
`c + 12`, then none. The word is 1 for an even range and 2 for an odd one.

```{ .c }
--8<-- "tutorial/47_a_word_with_each_range/thread/pick.c:run"
```

The thread runs the command and serves the pulls until its event comes. After it, the
thread reads each core's last tell and returns the four totals.

```{ .cpp }
--8<-- "tutorial/47_a_word_with_each_range/kernels/reader.cpp:pull"
```

Each core takes ranges until there are none. With word 1 it adds the numbers, with word
2 their squares, and after each range it tells the X280 core its total so far.

```{ .python }
--8<-- "tutorial/47_a_word_with_each_range/a_word_with_each_range.py:check"
```

Python works the totals out the same way, to check them.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow",
 "steps": [
  {"t": "core 1: port.next asks\nfor its next range", "gem": "amber"},
  {"t": "thread: pick_word(1, 0) gives\nrange 1 and word 2", "gem": "sapphire"},
  {"t": "core 1: numbers 4 to 7, word 2:\nadds 16 + 25 + 36 + 49", "gem": "amber"},
  {"t": "core 1: asks again; pick_word(1, 1)\ngives range 5 and word 2", "gem": "sapphire"},
  {"t": "... until pick_word gives\nKMX_PULL_NONE and next() returns 0"}]}
</script>
<figcaption>Each answer is a range and a word; the core does what the word says.</figcaption>
</figure>

## The whole program

=== "a_word_with_each_range.py"

    ``` { .python }
    --8<-- "tutorial/47_a_word_with_each_range/a_word_with_each_range.py"
    ```

=== "thread/pick.c"

    ``` { .c }
    --8<-- "tutorial/47_a_word_with_each_range/thread/pick.c"
    ```

=== "kernels/reader.cpp"

    ``` { .cpp }
    --8<-- "tutorial/47_a_word_with_each_range/kernels/reader.cpp"
    ```

`compute.cpp` and `writer.cpp` are the empty ones of [[hello-core]]. Run it from the folder
`tutorial/47_a_word_with_each_range`:

```bash
python a_word_with_each_range.py
```

```{ .text .km-output }
--8<-- "tutorial/47_a_word_with_each_range/output.txt"
```

Cores 1 and 3 got the odd ranges, so they added squares, and their totals are far larger.

## Common mistakes

- Setting both `pick` and `pick_word` in `kmx_pull`. `kmx_pull_run` returns `KMX_BAD`
  and submits nothing.
- Using the word after `next()` returned 0. With no range left the word is 0.
- A pick that never returns `KMX_PULL_NONE`. The core keeps getting ranges; return
  `KMX_PULL_NONE` once a core's share is done.
- Doing slow work in `pick_word`. It runs on the X280 core while other cores wait for
  their answers: one X280 core serving many cores answers them one after another.

Next: [[measuring]] starts Part 6 with how long things take.
