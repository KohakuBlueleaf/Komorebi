---
title: "Reading from DRAM"
part: 3
summary: "a core copies one tile from DRAM into its own L1"
---

# Reading from DRAM

The reader of a core can copy one tile of a tensor from DRAM into the core's L1, where the
core can use it. This lesson relies on [[tiles-and-tensors]], [[arguments]] and
[[the-card]].

## Why the core must copy

A core works only on what is in its own L1 ([[the-card]]). Data in DRAM is out of its
reach until it is copied in. The reader is the CPU for that, and it does it over the NoC:
it asks for tile number so-and-so of a tensor and says where in L1 the tile is to land.

A copy takes time and is not finished when the call returns. The reader has to wait for
it, and then has to tell the others that the tile has arrived.

## The general form

The kernel needs three things: a place in L1 for the tile, a way to name the tensor, and
the copy itself.

In Python, before `load`:

```python
program = km.Program(
    NAME,
    kernels=FOLDER,
    handlers=HEADER,
    buffers=[km.Buffer(index=INDEX, format=DTYPE, page_bytes=BYTES, pages=PAGES)],
    compile_args={"reader": TENSOR.accessor_args()},
)
```

In the reader:

```cpp
constexpr auto ARGUMENTS = TensorAccessorArgs<0>();
...
const auto tensor = TensorAccessor(ARGUMENTS, ADDRESS);
reader.reserve(INDEX, PAGES);
noc_async_read_page(TILE, tensor, get_write_ptr(INDEX));
noc_async_read_barrier();
cb_push_back(INDEX, PAGES);
```

- `km.Buffer(...)` sets aside room in the L1 of every core. `INDEX` is its number, 0 to 31,
  and `DTYPE` and `BYTES` say what one tile of the tensor is: `"uint32"` and 4096 bytes
  for the tiles of the last lesson. `PAGES` is how many tiles fit; this lesson uses one.
  The `cb_` in the C++ names below belongs to the kind of buffer that
  [[circular-buffers]] explains. For now a buffer is a numbered room in L1.
- `compile_args` hands the kernel what it needs to find the tensor's pages. In the
  kernel, `TensorAccessorArgs<0>()` receives it, and `TensorAccessor(ARGUMENTS, ADDRESS)`
  is an object that turns a tile number into a place in DRAM. `ADDRESS` is where the
  tensor starts. Python has it as `tensor.address`; it travels in a command argument.
- `reader.reserve(INDEX, PAGES)` waits until the buffer has room for `PAGES` tiles.
- `noc_async_read_page(TILE, tensor, where)` starts a copy of tile number `TILE` of the
  tensor to the L1 address `where`. `get_write_ptr(INDEX)` is the address of the room in
  the buffer. The call returns at once; the copy goes on.
- `noc_async_read_barrier()` waits until every copy that was started is finished.
- `cb_push_back(INDEX, PAGES)` says "these tiles are now in the buffer". Whoever waits
  for them is let through.

<figure class="km-diagram">
<script type="application/json">
{"kind": "boxes", "boxes": [{"id": "dram", "x": 0, "y": 0, "w": 150, "h": 64, "t": "DRAM\nthe source tensor:\n4 tiles"}, {"id": "l1", "x": 230, "y": 0, "w": 150, "h": 64, "t": "L1 of core 0\nthe buffer:\nroom for 1 tile", "gem": "amber"}, {"id": "py", "x": 460, "y": 0, "w": 130, "h": 64, "t": "Python\nthe result", "gem": "sapphire"}], "arrows": [{"from": "dram", "to": "l1", "kind": "data", "t": "reader: noc_async_read_page", "dy": -12, "dx": -70}, {"from": "l1", "to": "py", "kind": "reply", "t": "writer: result", "dy": -12, "dx": -30}]}
</script>
<figcaption>One tile goes from DRAM to the L1 of core 0; the writer sends a number of it to Python.</figcaption>
</figure>

## One line at a time

The tensor first, in Python:

```{ .python }
--8<-- "tutorial/16_reading_from_dram/reading_from_dram.py:source"
```

Four tiles; the number at place `p` of tile `t` is `1000 * t + p`, so that a number tells
where it came from. (`positions` is 0 to 1023, and the numpy shapes line up tile numbers
and positions.)

```{ .python }
--8<-- "tutorial/16_reading_from_dram/reading_from_dram.py:program"
```

The program names the buffer and gives the reader what it needs to find `source`. The
buffer holds one tile of 4096 bytes in the format of the tensor.

```{ .cpp }
--8<-- "tutorial/16_reading_from_dram/kernels/reader.cpp:run"
```

The command's arguments say which tensor and which tile. The reader makes the accessor,
waits for room, starts the copy, waits for it to finish, and announces the tile.

```{ .cpp }
--8<-- "tutorial/16_reading_from_dram/kernels/writer.cpp:run"
```

The writer is the one that can send a result ([[a-result]]), so it takes the number out of
the tile. `wait_front` waits until the reader has announced the tile; `get_read_ptr` is the
address of the tile in L1; the `reinterpret_cast` says "read it as an array of `uint32_t`";
`tile[position]` is the number at that place. `cb_pop_front` frees the room for the next
command.

```{ .python }
--8<-- "tutorial/16_reading_from_dram/reading_from_dram.py:fetch"
```

Three commands, each fetching one tile and asking for one number of it.

## The whole program

=== "reading_from_dram.py"

    ``` { .python }
    --8<-- "tutorial/16_reading_from_dram/reading_from_dram.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/16_reading_from_dram/kernels/handlers.h"
    ```

=== "kernels/reader.cpp"

    ``` { .cpp }
    --8<-- "tutorial/16_reading_from_dram/kernels/reader.cpp"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/16_reading_from_dram/kernels/writer.cpp"
    ```

The compute file is the empty one.

```{ .text .km-output }
--8<-- "tutorial/16_reading_from_dram/output.txt"
```

Each number is `1000 * tile + position`: the tile that was asked for arrived, and its
number at that place. The script checks it. It asks only for positions below 16 because
that is where the row-by-row order of the host array and the order of the numbers in the tile
agree; beyond them a tile keeps its numbers in four blocks of 16 x 16.

## Common mistakes

- Starting the copy and not waiting. `noc_async_read_page` returns at once, and without
  `noc_async_read_barrier()` the tile may not be in L1 yet when it is used.
- Typing `page_bytes` by hand. It must be the size of one tile of the tensor; write
  `source.page_bytes` instead of 4096 and it cannot go wrong when the dtype changes.
- Forgetting `cb_pop_front` in the writer. The first command works. The second finds the
  buffer still full, the reader waits for room that never comes, and the command ends
  with `CommandTimeout` ([[when-it-goes-wrong]]).
- Taking a number of the tile at a place that is not in the first row of the first 16
  columns, and expecting the host array's order. The numbers inside a tile are kept in
  four blocks of 16 x 16.

Next: [[writing-to-dram]] sends a tile the other way.
