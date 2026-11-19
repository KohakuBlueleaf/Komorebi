# The IR's operations (being built)

!!! note "Being built"
    The Komorebi IR and its command `komo` are being built. This page lists the operations of the
    language as the specification and its lessons write them.

`//` starts a comment and is no operator.

Each operation has one form, the one the lessons write; another spelling is refused (`K111` in
[the IR's checks](checks.md)). An operand is either **by position**, in the order given, or **by
name**, `name = value`, whose position carries no meaning. An operand marked *optional* may be left
out; every other operand must be given (`K103`).

A number written in the text has the type its place needs: the other operand's, a parameter's, a
field's, the declared type of a `let`, a `var` or a result. Beside a tile it has the tile's format, or
`f32` beside a `bf16` or a block-float tile. Where the place gives none, a whole number is `u32` and
one with a point or an exponent is `f32`. A number its type cannot hold is refused (`K115`): `let a =
-1` is refused, `let a: i32 = -1` is not. The values the language gives (`worker.rank`, `group.size`
and their kin) are `u32`.

The "Written in" column names a lesson (T1 to T16) or the topic of the specification that writes the
form. What the language names and the build does not have yet is listed at the end; it is refused by
name (`K010`).

## Values and control flow

| Operation | Operands | Types | Yields | Written in |
|---|---|---|---|---|
| `let x = e`, `let x: T = e` | a name, a value | any value type | binds once (`K104`) | T1, T2 |
| `var x: T = e` | a name, a type, a value | any value type | a mutable binding | T2, T4 |
| `x = e`, `x += e`, `x -= e` | a `var`, or an element of state | as declared | | T2, T3, T4 |
| `if c { } else { }` | a condition | `bool` | | T2 |
| `while c { }` | a condition | `bool` | control bodies only (`K005`) | T11 |
| `for i in range(n) { }` | `n` by position | an integer | `i` from 0 to n - 1 | T2 |
| `for b in stream(...) { }` | a stream | | each block in order | T1 |
| `for a, b in zip(s1, s2) { }` | two streams by position | | the two streams block by block | T4 |
| `for b, w in br.arrive() { }` | a broadcast | | each block and its word | T6 |
| `return e`, `return` | a value, *optional* | the declared result | ends the body | T2 |
| Integer operators `+ - * / % & \| ^ << >>` | two integers | integers of one type (`K119`), but the amount of a shift may be of any integer type; a number written in the text takes the other operand's type | an integer; arithmetic wraps modulo 2^n; `/` divides rounding toward zero and `%` is its remainder; by zero, the run fails (`divide_by_zero`), and a constant zero is refused (`K114`); a shift by a negative amount likewise (`out_of_range`, `K115`) | Handler bodies, Order, state and failure |
| `-a` | one by position | a number, a tile, or a reduction's values | the same, negated; the one unary operator | Grammar |
| `min(a, b)`, `max(a, b)` | two by position | numbers of one type (`K119`) | a number of that type | T2 |
| `u8(x)` ... `i64(x)`, `f16(x)`, `f32(x)`, `f64(x)` | one by position | a scalar | `x` converted to the type named: integer to integer wraps in the new width; integer to float is its value; float to integer drops the fraction, and a value the type cannot hold is the failure `out_of_range`. A number written in the text as `x` has the type named as its place, and one that type cannot hold is refused (`K115`): `u32(-1)` is refused, all ones is written `0xffffffff` | Numbers |
| Comparisons `== != < > <= >=` | two values | scalars of one type (`K119`; a number written in the text takes the other operand's type), or two `bool` for `==` and `!=` | `bool` | T2, Types and data |
| `c && d`, `c \|\| d` | two conditions | `bool` | `bool`; there is no negation operator: write `c == false` | T11 |

## Compute on tiles

| Operation | Operands | Types | Yields | Written in |
|---|---|---|---|---|
| `a + b`, `a - b`, `a * b`, `a / b` | two by position | tiles, or a tile and a scalar; or a reduction's values with a scalar or with values of the same axis (never with a tile: `K116`); an integer tile takes an integer scalar or tile, never a float (`K117`) | a tile, or a reduction's values; `/` divides the elements | T1, T4, Numbers |
| `neg(a)`, `abs(a)`, `relu(a)`, `square(a)`, `floor(a)` | one by position | a tile, or a reduction's values | the same kind | Compute functions |
| `maximum(a, b)`, `minimum(a, b)` | two by position | as for `+` | a tile, or a reduction's values | T4 |
| `where(a OP b, x, y)` | a comparison, then two values, by position | `OP` one of `> < >= <= == !=`; as for `+` | a tile, or a reduction's values | T4 |
| `exp`, `exp2`, `tanh`, `erf`, `log`, `sigmoid`, `silu`, `sqrt`, `rsqrt`, `reciprocal` | one by position | a tile, or a reduction's values | the same kind | T4 |
| `reduce_sum(a, axis = row)`, `reduce_max(...)`, `reduce_min(...)` | `a` by position, `axis` by name | a tile; `row` or `col` | a value for each row (`axis = row`) or each column (`axis = col`): not a tile; stored, pushed, given to `matmul` or mixed with a tile, it is refused (`K116`) | T4, Compute functions |
| `v.col()` | none | a value for each row; or a tile | spread along each row; of a tile, its column 0 spread | T4, Compute functions |
| `w.row()` | none | a value for each column; or a tile | spread along each column; of a tile, its row 0 spread | T4, Compute functions |
| `matmul(a, b)` | two by position | tiles | a tile | T4, T13 |
| `a.T` | none | a tile | the tile turned | T4 |
| `zeros` | none | | a tile, or the scalars, of zeros | T4, T16 |
| A number constant, a scalar parameter | | | a scalar | T1, T2 |
| A call of a `func`, a `combine` or an `extern func` | its parameters, by position | as declared; one tile for a parameter `tile<F>` (`K203`) | its result | T1, T8 |

The function body of a `func` uses only these (`K202`). Operands of different formats may be mixed.

## Views of a tensor

| Operation | Operands | Types | Yields | Written in |
|---|---|---|---|---|
| `t.tiles(first, n)` | two by position | a tensor; integers | a view of n consecutive tiles | T1 |
| `t.tiles[r0:r1, c0:c1]`, `t.tiles[r0:r1:s, c0:c1:s]` | two ranges, each with a step *optional* | a tensor; integers | a rectangle of tiles, walked row by row | T7, Types and data |
| `v.T` | none | a rectangle view | the rectangle walked column by column, each tile as it is (a tile is turned by its own `.T`) | T7, Compute functions |
| `t.items(size = B)` | `size` by name | a tensor whose format is an integer type, `f32` or `f64` (`K208`); an integer | a view of all the tensor's items of B bytes | T7, Types and data |
| `t.items(first, n, size = B)` | `first` and `n` by position, `size` by name | a tensor; integers | a view of n items of B bytes from item `first` | Types and data |
| `v[i]` | one by position | a view; an integer | one tile, read; an index past the view's end is `out_of_range` (`K115`) | Handler bodies |
| `v.words()` | none | a view of one tile (`K207`) of a tensor whose format is an integer type, `f32` or `f64` (`K208`) | the scalars of the tile, read by index | T2, Types and data |
| `v.records(R, count = c)` | `R` by position, `count` by name, always given | a view of one tile of a tensor whose format is an integer type, `f32` or `f64` (`K208`); a record type; c records that fit the tile (`K115`) | the records, read by index and field | T7, Types and data |
| `items.put(src, at = i)` | `src` by position, `at` by name | an items view; a state member | writes one item | T7 |
| `partition(view, rank, size)` | three by position | a view; integers | this rank's contiguous part | T1 |
| `stream(view, block = n)` | `view` by position, `block` by name, always given | a view; an integer | the blocks in order: `tile<F>`, or `tile<F>[n]` for n above 1 | T1 |
| `zip(s1, s2)` | two by position | streams of the same length (`K115`) | the two together | T4 |
| `sink(view, block = n)` | `view` by position, `block` by name, always given | a view; an integer | a write stream | T1 |
| `out.push(t)` | one by position | a write stream; a tile | writes the next block | T1 |
| `store view <- value` | a view, a value | a view; a tile (a store of scalars: `K010`) | writes it, visible at completion | T4 |
| `past(y)` | a name | a tensor parameter of a chain | y as an earlier call wrote it; in a chain's call only (`K602`) | T10 |

A tensor is reached only through a view (`K107`).

## State

| Operation | Operands | Types | Yields | Written in |
|---|---|---|---|---|
| `s.fill(stream)` | one by position | a member; a stream that holds exactly the member's tiles (`K115`) | fills the member whole | T3 |
| `s.zero()` | none | a member | clears it, and it is filled | T3 |
| `s[i] = x` | an index, a value | a member; its element; an index inside the member (`K115`) | writes one element; it fills the member only when it is marked `@filled(element)` | T3 |
| `s[i]` | an index | a member; an index inside it (`K115`) | one element, read | T3, T7 |
| `s.bump(i)` | one by position | a member of integers | the value it read; it then adds one to the element | T3 |

## Identity

| Operation | Operands | Yields | Written in |
|---|---|---|---|
| `worker.rank` | none | the rank in this invocation's group | T1 |
| `worker.is_leader` | none | `bool`: true on one worker of the group | T2 |
| `worker.id` | none | the worker's position in the loaded program | T12, T15 |
| `worker.row`, `worker.col` | none | the position on a grid; in a handler on a grid only (`K105`) | T6 |
| `group.size` | none | the number of workers in the group | T1 |
| `group.id(rank)` | one by position | the id of another rank of the group | T12 |
| `group.split(parts = n)`, `group.split(size = k)` | one by name | sub-groups contiguous in rank, as the operand `within` of a reduce | T5 |
| `group.leaders(parts = n)`, `group.leaders(size = k)` | one by name | their leaders, as the operand `among` of a reduce | T5 |

## Collectives

| Operation | Operands | Types | Yields | Written in |
|---|---|---|---|---|
| `reduce(x, with = f, within = g, among = g, publish = [v])` | `x` by position; `with` by name; `within`, `among`, `publish` by name, *optional* | a value; a `combine` or one of `sum max min and or` (`K401`); sub-groups; views | the combination, at the leader | T5 |
| `all_reduce(x, with = f, publish = [v])` | `x` by position; `with` by name; `publish` *optional* | in a handler, a value a worker holds, and the group's workers take part; in a control program, a replicated tensor, and the islands take part; `with` as for `reduce` | the same combination at every participant | T5, T12, Handler bodies, Islands |
| `gather(x)` | one by position | a value | `tile<F>[group.size]` at the leader, indexed by rank | T5 |
| `broadcast(along = row, sender = c)` | both by name; `sender` *optional*, 0 when left out | `row` or `col`; an integer | a broadcast; in a handler on a grid only (`K402`) | T6 |
| `br.is_sender` | none | | `bool` | T6 |
| `br.cast(block, word = w)` | `block` by position, `word` by name, *optional* | a tile; a scalar | sends the block, with the word when given | T6 |
| `br.arrive()` | none | | the blocks and their words, in order; always both | T6 |
| `sync(publish = [v])` | by name | views | a point the whole group reaches | T15 |

## Dynamic runs

In a dynamic run only (`K503`).

| Operation | Operands | Types | Yields | Written in |
|---|---|---|---|---|
| `M { field: value, ... }` | every field of `M` (`K113`) | scalars | a message | T9 |
| `send(M { ... }, to = rank, publish = [v])` | the message by position; `to` by name; `publish` *optional* | a message; an integer; views | `bool`: false when refused | T9, T15 |
| `forward(m, with = { ... }, to = rank, publish = [v])` | `m` by position; `to` by name, always given; `with`, `publish` by name, *optional* | the arrived message; some of its fields; an integer; views | sends the same unit of work on | T9, T15 |
| `consume m` | the arrived message | | ends the unit of work | T9 |
| `offer(M { ... }, to = rank)` | the message by position; `to` by name | a message; an integer, to a declared peer (`K502`) | `bool`: false while the peer holds the last one | Dynamic runs |
| `m.field` | none | an arrived message | the field | T9 |

## Control programs

In a control program only (`K703`).

| Operation | Operands | Types | Yields | Written in |
|---|---|---|---|---|
| `invoke(P.h(args), on = group(first, count), after = [a], after_end = [a])` | the call by position; `on` by name; `after`, `after_end` by name, *optional* | a handler of a program and its arguments; a group; handles | a handle | T11, T16 |
| `group(first, count)`, `group(count)` | by position; `first` *optional*: left out, the runtime chooses it | integers | a group of workers | T11, Control programs |
| `await h` | a handle | | the invocation's result; fails the control program if it failed | T11 |
| `outcome(h)` | one by position | a handle | an outcome: `.failed` (`bool`), `.reason`, `.value` | T16 |
| `o.reason == name` | a failure's name | one of `view_end`, `empty_state`, `send_refused`, `capacity`, `given_up`, `dependency`, `not_a_peer`, `over_bound`, `divide_by_zero`, `out_of_range` (`K704`) | `bool` | Order, state and failure |

## Islands

| Operation | Operands | Types | Yields | Written in |
|---|---|---|---|---|
| `island.count` | none | | the number of islands | T12 |
| `island.of(id)` | one by position | a worker's id | its island | T12 |
| `island.crosses(id)` | one by position | a worker's id | `bool` | T12 |
| `island.hops(id)` | one by position | a worker's id | an integer | T12 |
| `reduce_to(t, to = i, with = f)` | `t` by position; `to`, `with` by name | a replicated tensor; an island; a combine | in a control program only (`K801`) | T12 |
| `broadcast_from(t, from = i)` | `t` by position; `from` by name | a replicated tensor; an island | in a control program only (`K801`) | T12 |

## Named, not built yet

The language names these and the build does not have them yet. Each is refused by name (`K010`):

- `all_gather`, a collective with the participants of `all_reduce`: in a handler the group's workers,
  in a control program the islands. Its operands are not written yet.
- A `store` of scalars.
- A message's body: `@body(n)` on a message, `m.body`, and `as = M` on a `forward`.
- Several tile shapes in one module.
- A function on a block of several tiles, `tile<F>[n]`.
