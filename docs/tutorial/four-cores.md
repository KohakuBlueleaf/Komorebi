---
title: "Four cores"
part: 5
summary: "one function run on all four cores of an X280 cluster, and a sum that comes out four times faster"
---

# Four cores

A cluster has four X280 cores, and `kmx_parallel` runs one function on all four, each with
its own number. This lesson relies on [[hello-thread]] and [[partitions]].

## Why four cores

`kmx_main` runs on core 0 alone, so a thread that only uses `kmx_main` uses a quarter of
its cluster. Work that can be cut into independent pieces can use all four, in the way
that [[partitions]] cut work among Tensix cores: each core takes a part of the numbers,
and one core adds the parts together at the end.

## The general form

```c
static void piece(int core, void* argument) {
    BODY
}

int status = kmx_parallel(piece, &VALUE);
```

- `piece` is a function that you write, with exactly this form: it takes the number of
  the core that runs it, 0 to 3, and a pointer.
- `argument` is the pointer you passed. It is the same on all four cores. The function
  casts it back to what it points to, for instance `*(const uint32_t*)argument`. Pass
  `0` when there is nothing to pass.
- `BODY` is the work of one core. It uses `core` to choose its share, as a Tensix core
  uses its rank.
- `kmx_parallel(piece, &VALUE)` starts `piece` on cores 1, 2 and 3, runs it on core 0, and
  returns when all four have finished. It returns `KMX_OK`, or another code when it could
  not start.
- Only core 0 calls `kmx_parallel`, and not from inside a `piece`.
- Inside `piece`, a core may read the arguments, the clock and its regions. It may not
  submit commands, and it should not call `kmx_result`.

The cores share memory, so a core hands its piece to core 0 through an ordinary array. The
array is a global variable, and a global variable of a thread must start with a value that
is not zero, and carry the attribute `__attribute__((section(".data")))`, because the
thread's image has no place for variables that start as zero.

## One line at a time

```{ .c }
--8<-- "tutorial/34_four_cores/thread/four.c:partials"
```

Four lines of 16 numbers, one line for each core: a core writes to the first number of its
own line. A line of 64 bytes is the unit in which the cores' caches exchange memory, so
each core's piece lies in a line of its own. This lesson does not measure what sharing a line would cost. The `= {1}` is the
non-zero start, and the attribute puts the array where the image allows it.

```{ .c }
--8<-- "tutorial/34_four_cores/thread/four.c:sum"
```

`scrambled_sum(first, last)` adds up a number made from each `i` from `first` up to, but
not including, `last`. The number is scrambled by `scramble` so that the loop has real
work and no formula for its sum. It is ordinary C: a loop, a function, no library.

```{ .c }
--8<-- "tutorial/34_four_cores/thread/four.c:piece"
```

Every core adds up its quarter of the numbers 0 to `n - 1`: `n` comes through the pointer,
`share` is a quarter of it, and core `c` takes the numbers from `c * share` up to, but not
including, `(c + 1) * share`. Core 0 and core 3 run the same code on different numbers, and
each stores its sum in its own line.

```{ .c }
--8<-- "tutorial/34_four_cores/thread/four.c:parallel"
```

The call, timed with the thread's clock. Before it, the same thread added all the numbers
on core 0 alone, and timed that too. After it, core 0 adds the four pieces.

```{ .python }
--8<-- "tutorial/34_four_cores/four_cores.py:check"
```

Python computes the same sum with numpy, so that it can check the thread's answer, and
prints the two times.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow",
 "steps": [
  {"t": "kmx_main on core 0: adds all the\nnumbers alone, and keeps the time", "gem": "sapphire"},
  {"t": "kmx_parallel(piece, &n)\ncores 0, 1, 2 and 3 each run piece", "gem": "sapphire"},
  {"t": "each core adds its quarter and stores\nthe sum in its own line of partial"},
  {"t": "kmx_parallel returns when all four\nhave finished"},
  {"t": "kmx_main: adds the four lines,\nreturns both sums and both times"}]}
</script>
<figcaption>The same sum, alone and on four cores.</figcaption>
</figure>

## The whole program

=== "four_cores.py"

    ``` { .python }
    --8<-- "tutorial/34_four_cores/four_cores.py"
    ```

=== "thread/four.c"

    ``` { .c }
    --8<-- "tutorial/34_four_cores/thread/four.c"
    ```

The kernels are the empty ones of [[hello-core]]. Run it from the folder
`tutorial/34_four_cores`:

```bash
python four_cores.py
```

```{ .text .km-output }
--8<-- "tutorial/34_four_cores/output.txt"
```

Both sums equal the one numpy computed. Four cores did the work in a quarter of the time of
one, 4.0 times as fast: the work is arithmetic, every core has its own share of it, and
nothing has to wait for another core. Work that mostly moves memory does not scale like
this; [[twelve-cores]] comes back to it.

## Common mistakes

- A global array that starts as zero. `load` stops with `komorebi.builder.flags.BuildError`, and
  the linker's message in it ends with
  `x280.ld: per-hart images must carry no .bss -- entry.S zeroes none`. Give the array a
  non-zero first value and the `.data` attribute.
- A number of items that four does not divide. With `n = 4000002` the shares are
  `1000000` each, and the last two numbers are in nobody's piece. The sum comes out
  wrong, and nothing says so.
- Calling `kmx_submit` inside `piece`. Cores 1 to 3 may not, and the run ends with the
  library error `parallel`. Collect what you need in the array and submit from `kmx_main`.

Next: [[sharing-memory]] is about the memory that Python and a thread both use.
