# The IR's operations (being built)

!!! note "Being built"
    The Komorebi IR and its command `komo` are being built. This page lists the operations of the
    language as the specification and its lessons write them.

Each operation has one form, the one the lessons write; another spelling is refused (`K111` in
[the IR's checks](checks.md)). An operand is either **by position**, in the order given, or **by
name**, `name = value`, whose position carries no meaning. An operand marked *optional* may be left out; every other operand must be given
(`K103`).

The "Written in" column names a lesson (T1 to T16) or the topic of the specification that writes the
form. Below the tables, the forms the specification writes in two ways, and the points it does not
write, are listed: until they are settled the checker accepts only the form in the tables.

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
| Integer operators `+ - * // % & \| ^ << >>` | two integers | integers | an integer; arithmetic wraps modulo 2^n | Handler bodies |
| `min(a, b)`, `max(a, b)` | two by position | integers | an integer | T2 |
| Comparisons `== != < > <= >=` | two values | scalars, or two `bool` for `==` and `!=` | `bool` | T2, Types and data |

## Compute on tiles

| Operation | Operands | Types | Yields | Written in |
|---|---|---|---|---|
| `a + b`, `a - b`, `a * b`, `a / b` | two by position | tiles, or a tile and a scalar | a tile | T1, T4 |
| `neg(a)`, `abs(a)`, `relu(a)`, `square(a)`, `floor(a)` | one by position | a tile | a tile | Compute functions |
| `maximum(a, b)`, `minimum(a, b)` | two by position | tiles, or a tile and a scalar | a tile | T4 |
| `where(a OP b, x, y)` | a comparison, then two values, by position | `OP` one of `> < >= <= == !=`; tiles or scalars | a tile | T4 |
| `exp`, `exp2`, `tanh`, `erf`, `log`, `sigmoid`, `silu`, `sqrt`, `rsqrt`, `reciprocal` | one by position | a tile | a tile | T4 |
| `reduce_sum(a, axis = row)`, `reduce_max(...)`, `reduce_min(...)` | `a` by position, `axis` by name | a tile | one value for each row | T4 |
| `v.col()` | none | a value for each row | it spread along the row | T4 |
| `w.row()` | none | a tile | a row spread along the columns | T4 |
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
| `t.tiles[r0:r1, c0:c1]` | two ranges | a tensor; integers | a rectangle of tiles, walked row by row | T7 |
| `v.T` | none | a rectangle view | the rectangle turned | T7 |
| `t.items(size = B)` | `size` by name | a tensor; an integer | a view of items of B bytes | T7 |
| `v[i]` | one by position | a view; an integer | one tile, read | Handler bodies |
| `v.words()` | none | a view of one tile (`K207`) | the scalars of the tile, read by index | T2 |
| `v.records(R, count = c)` | `R` by position, `count` by name | a view of one tile; a record type | the records, read by index and field | T7 |
| `items.put(src, at = i)` | `src` by position, `at` by name | an items view; a state member | writes one item | T7 |
| `partition(view, rank, size)` | three by position | a view; integers | this rank's contiguous part | T1 |
| `stream(view, block = n)` | `view` by position, `block` by name | a view; an integer | the blocks in order: `tile<F>`, or `tile<F>[n]` for n above 1 | T1 |
| `zip(s1, s2)` | two by position | streams | the two together | T4 |
| `sink(view, block = n)` | `view` by position, `block` by name | a view; an integer | a write stream | T1 |
| `out.push(t)` | one by position | a write stream; a tile | writes the next block | T1 |
| `store view <- value` | a view, a value | a view; a tile | writes it, visible at completion | T4 |
| `past(y)` | a name | a tensor parameter of a chain | y as an earlier call wrote it; in a chain's call only (`K602`) | T10 |

A tensor is reached only through a view (`K107`).

## State

| Operation | Operands | Types | Yields | Written in |
|---|---|---|---|---|
| `s.fill(stream)` | one by position | a member; a stream | fills the member whole | T3 |
| `s.zero()` | none | a member | clears it, and it is filled | T3 |
| `s[i] = x` | an index, a value | a member; its element | writes one element | T3 |
| `s[i]` | an index | a member | one element, read | T3, T7 |
| `s.bump(i)` | one by position | a member of integers | reads the element, then adds one | T3 |

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
| `all_gather` | not written | in a handler, a value a worker holds, among the group's workers; in a control program, a replicated tensor, among the islands | not written | Islands |
| `gather(x)` | one by position | a value | `tile<F>[group.size]` at the leader, indexed by rank | T5 |
| `broadcast(along = row, sender = c)` | both by name; `sender` *optional*, 0 when left out | `row` or `col`; an integer | a broadcast; in a handler on a grid only (`K402`) | T6 |
| `br.is_sender` | none | | `bool` | T6 |
| `br.cast(block, word = w)` | `block` by position, `word` by name | a tile; a scalar | sends the block with the word | T6 |
| `br.arrive()` | none | | the blocks and their words, in order | T6 |
| `sync(publish = [v])` | by name | views | a point the whole group reaches | T15 |

## Dynamic runs

In a dynamic run only (`K503`).

| Operation | Operands | Types | Yields | Written in |
|---|---|---|---|---|
| `M { field: value, ... }` | every field of `M` (`K113`) | scalars | a message | T9 |
| `send(M { ... }, to = rank, publish = [v])` | the message by position; `to` by name; `publish` *optional* | a message; an integer; views | `bool`: false when refused | T9, T15 |
| `forward(m, with = { ... }, to = rank, as = M, publish = [v])` | `m` by position; `with`, `to` by name; `as`, `publish` *optional* | the arrived message; its fields; an integer; a message type; views | sends the same unit of work on | T9, T15 |
| `consume m` | the arrived message | | ends the unit of work | T9 |
| `offer(M { ... }, to = rank)` | the message by position; `to` by name | a message; an integer, to a declared peer (`K502`) | `bool`: false while the peer holds the last one | Dynamic runs |
| `m.field` | none | an arrived message | the field | T9 |

## Control programs

In a control program only (`K703`).

| Operation | Operands | Types | Yields | Written in |
|---|---|---|---|---|
| `invoke(P.h(args), on = group(first, count), after = [a], after_end = [a])` | the call by position; `on` by name; `after`, `after_end` by name, *optional* | a handler of a program and its arguments; a group; handles | a handle | T11, T16 |
| `group(first, count)` | two by position | integers | a group of workers | T11 |
| `await h` | a handle | | the invocation's result; fails the control program if it failed | T11 |
| `outcome(h)` | one by position | a handle | an outcome: `.failed` (`bool`), `.reason`, `.value` | T16 |
| `o.reason == name` | a failure's name | one of `view_end`, `empty_state`, `send_refused`, `capacity`, `given_up`, `dependency`, `not_a_peer`, `over_bound` (`K704`) | `bool` | Order, state and failure |

## Islands

| Operation | Operands | Types | Yields | Written in |
|---|---|---|---|---|
| `island.count` | none | | the number of islands | T12 |
| `island.of(id)` | one by position | a worker's id | its island | T12 |
| `island.crosses(id)` | one by position | a worker's id | `bool` | T12 |
| `island.hops(id)` | one by position | a worker's id | an integer | T12 |
| `reduce_to(t, to = i, with = f)` | `t` by position; `to`, `with` by name | a replicated tensor; an island; a combine | in a control program only (`K801`) | T12 |
| `broadcast_from(t, from = i)` | `t` by position; `from` by name | a replicated tensor; an island | in a control program only (`K801`) | T12 |

## Not yet settled

Forms the specification writes in two ways:

| Operation | One way | The other | Meanwhile, the lessons' form |
|---|---|---|---|
| Items of a tensor | `t.items(size = B)` (T7, Handler bodies) | `t.items(first, n, size = B)` (Types and data) | `size` only |
| The block of a stream | `stream(view, block = n)` in every lesson | `zip(stream(a), stream(b))`, the block left out (Handler bodies) | `block` is given |
| The fields of a `forward` | `forward(m, with = { ... }, to = rank)` (T9) | `forward(m, as = Hit, to = worker.rank)`, with no `with`, in the direction for message bodies (Dynamic runs) | `with` is given |

Forms the specification names and does not write:

- The operands of `all_gather`, and what it yields: it is named among the collectives, never written.
- A group given as a count only ("lets the runtime choose the first rank", Control programs).
- The step of a rectangle `t.tiles[r0:r1, c0:c1]` ("with an optional step", Types and data).
- `axis = col` on `reduce_sum`, `reduce_max` and `reduce_min`, and the type of their result.
- The element type of `v.words()`.
- `store` of scalars ("Write a tile or scalars", Handler bodies).
- Whether `word` may be left out of `br.cast` and `count` out of `.records`.
- `&&`, written in T11's control program and not in the list of operators; `||` and a negation.
- A message's body, `@body(n)` and `m.body`.
