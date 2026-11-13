---
title: "The matrix unit"
part: 3
summary: "replacing the addition by a product of two tiles, on the unit the card is built around"
---

# The matrix unit

The core's arithmetic has a unit built to multiply two 32 x 32 tiles as matrices in one
call, and using it is the same program as the addition with a different operation. This
lesson relies on [[computing-on-a-tile]].

## Why a matrix unit

The addition of the last lesson does one operation per number: 1024 of them for a tile. A
matrix product of two 32 x 32 tiles does 32 x 32 x 32 = 32,768 multiplications and as many
additions, and nearly all of the work of a neural network is products of this kind. The
card has a unit that takes the two tiles and produces the product in a single call,
instead of the loops you would write for 32,768 steps. That is why the data is in
tiles: the unit's size is the tile.

The product of tiles is the matrix product: number `(row, column)` of the result is the
sum, over `k` from 0 to 31, of number `(row, k)` of the left tile times number
`(k, column)` of the right tile. It is not the product of numbers in the same place.

## The general form

```cpp
compute_kernel_hw_startup<SrcOrder::Reverse>(LEFT, RIGHT, OUTPUT);
matmul_init(LEFT, RIGHT);
...
tile_regs_acquire();
matmul_tiles(LEFT, RIGHT, 0, 0, 0);
```

and in Python, in `km.Program`:

```python
compute = {"math_fidelity": "HiFi4"}
```

- It is the program of [[computing-on-a-tile]] with three changes. `matmul_init` replaces
  `add_init`. `matmul_tiles` replaces `add_tiles`, with the same five arguments.
- `SrcOrder::Reverse` is a setting that the matrix operation needs, and the others do
  not. Write it as shown.
- `LEFT` is the buffer of the left tile of the product and `RIGHT` that of the right: the
  order matters, `A x B` is not `B x A`.
- `math_fidelity` says how many passes the unit makes over the numbers. `"HiFi4"` is the
  most accurate and the slowest; for bfloat16 products it keeps them exact where
  the operands allow.

<figure class="km-diagram">
<script type="application/json">
{"kind": "anatomy", "parts": [{"t": "matmul_tiles(", "n": "multiply two tiles\\nas matrices"}, {"t": "kFirstInput", "n": "buffer of\\nthe left tile"}, {"t": ", "}, {"t": "kSecondInput", "n": "buffer of\\nthe right tile"}, {"t": ", 0, 0, ", "n": "first tile of each\\nbuffer"}, {"t": "0", "n": "register\\nfor the product"}, {"t": ")"}]}
</script>
<figcaption>The call that replaces the addition.</figcaption>
</figure>

## One line at a time

```{ .cpp }
--8<-- "tutorial/21_the_matrix_unit/kernels/compute.cpp:run"
```

The body of the compute kernel is that of the last lesson. The line with `matmul_tiles`
is the only one that differs, and `kHandlerMultiply` is the handler for the product.

```{ .python }
--8<-- "tutorial/21_the_matrix_unit/the_matrix_unit.py:tensors"
```

The numbers are now 0, 1 and 2. Whole numbers keep the product exact: the largest possible
number of the result is 32 x 2 x 2 = 128, and `"bfloat16"` holds whole numbers up to 256
exactly.

```{ .python }
--8<-- "tutorial/21_the_matrix_unit/the_matrix_unit.py:check"
```

numpy's `@` is the matrix product. The script compares all 1024 numbers of the result with
it. The three prints show a row, a column and the first number of the product: the
first number is the first row dotted with the first column.

## The whole program

=== "the_matrix_unit.py"

    ``` { .python }
    --8<-- "tutorial/21_the_matrix_unit/the_matrix_unit.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/21_the_matrix_unit/kernels/handlers.h"
    ```

=== "kernels/compute.cpp"

    ``` { .cpp }
    --8<-- "tutorial/21_the_matrix_unit/kernels/compute.cpp"
    ```

The reader and the writer are those of lesson 20, with the handler renamed.

```{ .text .km-output }
--8<-- "tutorial/21_the_matrix_unit/output.txt"
```

## Common mistakes

- Expecting `A x B` to equal `B x A`. Swap the tile numbers in the command (`2, 1` for
  `1, 2`) and the product is a different tile.
- Omitting `SrcOrder::Reverse` in the start-up call. The matrix operation needs it.
- Numbers that grow past what `"bfloat16"` keeps. A product of larger whole numbers is
  rounded, and the numpy check fails; compare with a tolerance for such data.

Next: [[a-whole-map]] puts a group of cores to work on a whole tensor.
