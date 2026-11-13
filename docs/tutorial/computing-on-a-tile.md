---
title: "Computing on a tile"
part: 3
summary: "the compute kernel: unpack two tiles, add them, pack the sum"
---

# Computing on a tile

The compute kernel takes tiles from buffers, lets the core's arithmetic units work on them,
and puts the result tile in another buffer, in three steps: unpack, one operation, pack.
This lesson relies on [[circular-buffers]], [[three-kernels]] and [[tiles-and-tensors]].

## Why three steps

The arithmetic units do not read L1 themselves. Of the three CPUs that run the compute
kernel ([[the-card]]), the first, **unpack**, moves tiles from buffers into the units'
working registers. The second, **math**, runs the operation on what is there. The third,
**pack**, moves the result from the registers back into a buffer. You write one function
for all three; each CPU runs the part that is its own. Between them they pass the tile
through the **destination register**, a set of working tiles inside the unit, numbered
from 0.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow", "steps": [{"t": "buffers 0 and 1 hold tile A and tile B", "gem": "amber"}, {"t": "unpack (CPU 3): A and B go to the unit"}, {"t": "math (CPU 4): A + B, left in register 0"}, {"t": "pack (CPU 5): register 0 goes to buffer 16"}, {"t": "buffer 16 holds the sum", "gem": "amber"}]}
</script>
<figcaption>Two tiles in, one tile out: unpack and pack touch the buffers; math works on the register between them.</figcaption>
</figure>

## The general form

```cpp
compute_kernel_hw_startup(FIRST, SECOND, OUTPUT);
add_init(FIRST, SECOND);
compute.run([&](const km_command_t& command) {
    compute.wait_front(FIRST, 1);
    compute.wait_front(SECOND, 1);
    tile_regs_acquire();
    add_tiles(FIRST, SECOND, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    compute.reserve_back(OUTPUT, 1);
    pack_tile(0, OUTPUT);
    cb_push_back(OUTPUT, 1);
    tile_regs_release();
    cb_pop_front(FIRST, 1);
    cb_pop_front(SECOND, 1);
});
```

- `FIRST`, `SECOND` and `OUTPUT` are the indexes of the three buffers: two to read, one to
  write.
- `compute_kernel_hw_startup` and `add_init` are set-up, done once, before `run`. They
  tell the units which buffers and which operation to expect.
- `wait_front` waits until the reader has pushed a tile into the buffer, as the writer
  did in [[three-kernels]].
- `tile_regs_acquire()` takes the destination register for this kernel. `tile_regs_commit()`
  says "math is done"; `tile_regs_wait()` is where pack waits for it. `tile_regs_release()`
  gives the register back.
- `add_tiles(FIRST, SECOND, 0, 0, 0)` adds the first tile of buffer `FIRST` to the first
  tile of `SECOND` and leaves the sum in register 0. The last number is the register; the
  two before it are which tile of each buffer.
- `compute.reserve_back` and `cb_push_back` are the reader's `reserve` and `push` on the
  output side; `pack_tile(0, OUTPUT)` copies register 0 into the tile just reserved.
- Every tile taken with `wait_front` is released with `cb_pop_front`.

## One line at a time

```{ .python }
--8<-- "tutorial/20_computing_on_a_tile/computing_on_a_tile.py:tensors"
```

The numbers are whole numbers from -8 to 8, in four tiles, stored as `"bfloat16"`, a
2-byte number with a fraction. Whole numbers this small are exact in it, so the sum can be
checked exactly. The destination is one tile of zeros.

```{ .python }
--8<-- "tutorial/20_computing_on_a_tile/computing_on_a_tile.py:program"
```

Three buffers of one tile: 0 and 1 for the two inputs, 16 for the output. `dict(...)`
holds what the three have in common.

```{ .cpp }
--8<-- "tutorial/20_computing_on_a_tile/kernels/reader.cpp:run"
```

The reader puts one tile of the source in each of the two input buffers.

```{ .cpp }
--8<-- "tutorial/20_computing_on_a_tile/kernels/compute.cpp:run"
```

The compute kernel is the general form with the checks: if a wait gives up, it returns.

```{ .python }
--8<-- "tutorial/20_computing_on_a_tile/computing_on_a_tile.py:add"
```

One command: add tile 1 and tile 2 of the source, write the result as tile 0 of the
destination. The writer of this lesson is that of [[writing-to-dram]], reading from
buffer 16.

## The whole program

=== "computing_on_a_tile.py"

    ``` { .python }
    --8<-- "tutorial/20_computing_on_a_tile/computing_on_a_tile.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/20_computing_on_a_tile/kernels/handlers.h"
    ```

=== "kernels/reader.cpp"

    ``` { .cpp }
    --8<-- "tutorial/20_computing_on_a_tile/kernels/reader.cpp"
    ```

=== "kernels/compute.cpp"

    ``` { .cpp }
    --8<-- "tutorial/20_computing_on_a_tile/kernels/compute.cpp"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/20_computing_on_a_tile/kernels/writer.cpp"
    ```

```{ .text .km-output }
--8<-- "tutorial/20_computing_on_a_tile/output.txt"
```

The first row of the two source tiles, and of the destination: -6 + 3 is -3, 4 + 4 is 8,
0 + 5 is 5. The script checks all 1024 numbers against numpy.

## Common mistakes

- Forgetting `cb_pop_front` on an input. The next command finds the buffer full, as in
  [[reading-from-dram]].
- Changing the order of the register calls. Keep the order of the general form:
  acquire, operation, commit, wait, pack, release.
- Expecting a fraction from `"bfloat16"` to be exact. It keeps about three decimal digits;
  whole numbers up to 256 are exact.

Next: [[the-matrix-unit]] replaces the addition with a product of tiles.
