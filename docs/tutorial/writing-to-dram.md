---
title: "Writing to DRAM"
part: 3
summary: "a core writes a tile from its L1 to a tensor in DRAM: the first program that changes data"
---

# Writing to DRAM

The writer of a core can copy a tile from the core's L1 to a tile of a tensor in DRAM, and
that is how a program puts something where Python can read it. This lesson relies on
[[reading-from-dram]] and [[tiles-and-tensors]].

## Why the writer writes

A result that is a single number can travel back in the result word ([[a-result]]).
A result that is a tile cannot: it is 1024 numbers. Its way out is DRAM: the writer copies
it to a tensor that Python made, and Python reads the tensor. With a read and a write, a
program can now change data, and the smallest program that does is a copy: read a tile in,
write the tile out somewhere else.

## The general form

In the writer:

```cpp
constexpr auto ARGUMENTS = TensorAccessorArgs<0>();
...
const auto tensor = TensorAccessor(ARGUMENTS, ADDRESS);
writer.wait_front(INDEX, 1);
noc_async_write_page(TILE, tensor, get_read_ptr(INDEX));
noc_async_write_barrier();
cb_pop_front(INDEX, 1);
```

In Python, one more entry of `compile_args`:

```python
compile_args = {"reader": SOURCE.accessor_args(), "writer": DESTINATION.accessor_args()}
```

- `ARGUMENTS` and `TensorAccessor(...)` are as in [[reading-from-dram]], here for the
  destination. `ADDRESS` is `destination.address`, passed as a command argument.
- `writer.wait_front(INDEX, 1)` waits until a tile is in the buffer.
- `noc_async_write_page(TILE, tensor, where)` starts a copy of the L1 data at `where` into
  tile number `TILE` of the tensor. `get_read_ptr(INDEX)` is the address of the tile in the
  buffer.
- `noc_async_write_barrier()` waits until every write that was started is finished.
- `cb_pop_front(INDEX, 1)` says "I am done with the tile", and frees the room for the
  reader.

<figure class="km-diagram">
<script type="application/json">
{"kind": "boxes", "boxes": [{"id": "src", "x": 0, "y": 0, "w": 150, "h": 64, "t": "DRAM\nsource tensor\n4 tiles"}, {"id": "l1", "x": 230, "y": 0, "w": 150, "h": 64, "t": "L1 of core 0\nthe buffer\n1 tile", "gem": "amber"}, {"id": "dst", "x": 460, "y": 0, "w": 150, "h": 64, "t": "DRAM\ndestination tensor\n2 tiles"}], "arrows": [{"from": "src", "to": "l1", "kind": "data", "t": "reader: read", "dy": -12, "dx": -40}, {"from": "l1", "to": "dst", "kind": "data", "t": "writer: write", "dy": -12, "dx": -40}]}
</script>
<figcaption>A tile goes from one tensor through the L1 of a core to another: the reader brings it in, the writer takes it out.</figcaption>
</figure>

## One line at a time

```{ .python }
--8<-- "tutorial/17_writing_to_dram/writing_to_dram.py:destination"
```

The tensor to write into, made of zeros: two tiles. Python makes it before `load`, just
like the source.

```{ .cpp }
--8<-- "tutorial/17_writing_to_dram/kernels/writer.cpp:run"
```

The writer's half of the copy. The command says where the destination is (argument 2) and
which tile of it to write (argument 3). Once the tile is in the buffer, the writer copies
it out, waits until the write is done, and frees the buffer. The reader is that of
[[reading-from-dram]]: arguments 0 and 1 are the source and its tile.

```{ .python }
--8<-- "tutorial/17_writing_to_dram/writing_to_dram.py:copy"
```

Two commands: tile 2 of the source to tile 0 of the destination, and tile 3 to tile 1.
Each one waits before the next: the buffer holds one tile.

```{ .python }
--8<-- "tutorial/17_writing_to_dram/writing_to_dram.py:check"
```

After `stop`, Python reads the raw pages of both tensors with `session.read`, prints the
first number of each destination tile, and checks that each page is the same bytes as
the source page it was copied from.

## The whole program

=== "writing_to_dram.py"

    ``` { .python }
    --8<-- "tutorial/17_writing_to_dram/writing_to_dram.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/17_writing_to_dram/kernels/handlers.h"
    ```

=== "kernels/reader.cpp"

    ``` { .cpp }
    --8<-- "tutorial/17_writing_to_dram/kernels/reader.cpp"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/17_writing_to_dram/kernels/writer.cpp"
    ```

The compute file is the empty one.

```{ .text .km-output }
--8<-- "tutorial/17_writing_to_dram/output.txt"
```

The numbers of the source count up from 1000, row by row, so tile 2 starts at
1000 + 2 x 1024 = 3048, and tile 3 at 4072. The destination began as zeros; now its first
tile starts with 3048 and its second with 4072, and the script has checked that every
byte is the same.

## Common mistakes

- Reading the tensor too early. `session.read` does not wait for the cores. Read after the
  commands have been waited for ([[commands]]).
- Writing a tile number the destination does not have. The tensor has two tiles: tile 2 is
  past its end.
- Forgetting `noc_async_write_barrier()`. The command may be reported done before the
  tile is in DRAM.

Next: [[three-kernels]] looks at the three kernels as a whole.
