# The IR's operations (being built)

!!! note "Being built"
    The Komorebi IR and its command `komo` are being built. This page lists the operations of the
    language as the specification and its lessons write them.

`//` starts a comment and is no operator.

A name is the name its author gave (R50). A name that is a word of the text (`let`, `var`, `if`,
`else`, `for`, `in`, `while`, `return`, `store`, `consume`, `await`, `true`, `false`, `past`,
`fit`) is written between backticks, `` `store` ``, wherever a name is; a type named by a word a
type is written with (`tile`, `tensor`, `view`, `bool`, `u32` and the other scalar types) is
written between them where a type is. Any name may be written between backticks; the canonical
layout (`komo fmt`) writes them exactly where they are needed and takes the others away. The
backticks are not part of the name: the byte code, a backend and a message use the name itself
(`` handler `store` `` is the handler `store`).

A bare name means the language's wherever the language has a meaning for it in that place, so no
line changes its meaning with the module's names: `zip(stream(a), stream(b))` is the language's
`zip` in every module. An item of the module takes backticks only where its kind stands beside a
language name of the same spelling: a function (`func`, `combine`, `extern func`) named by one of
the language's calls, a conversion (`u32`) or a combination (`sum max min and or`) is
`` func `square` `` and is called `` `square`(x) ``; a value (a local, a parameter, a loop's
name, a state member, a module constant) named `zeros`, `worker`, `group` or `island`; a record
or a message named by a type's word. Such an item declared bare is refused (`K120`). The
canonical layout writes its backticks at its declaration and at every use; the checker takes a
bare use as the module's item only where the language has no meaning for the name in that place
(`` `group` + 1 `` is the parameter, `group.id(0)` the language's beside it). A name between
backticks is the module's item only, so where no item of the module can stand it is an unknown
name (`K101`): a word an attribute takes (`` @on(`group`) ``, `` @scope(`worker`) ``), a member
or a method of the language's values (`` worker.`rank` ``, `` x.`tiles`(0, 1) ``) and an
operand's name (`` `axis` = row ``). A handler, a run, a chain, a program, a field and a message
member stand where only an author's name does, so `handler zip`, `var sum` and a parameter `row`
are written bare.

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

A tile a compute operation yields has the format its place states: the declared type of a `let` or a
`var`, a function's result, the parameter it is given to, the tensor it is written to. Where the place
states none, it has the first operand's. Among the float formats this is a notation only: a float
tile of any format may stand where a float tile of another is declared. Integer formats agree
(`K102`). `return matmul(x, w)` of two `bf16` tiles, in a function returning `tile<f32>`, yields a
`tile<f32>`.

## Constant expressions

A **constant expression** is made of whole numbers written in the text, module parameters bound at
build (`@bind(build)`), the operators `+ - * / % << >> & | ^`, `min(a, b)`, `max(a, b)` and
brackets. Program compile evaluates it with the language's integer rules (`/` rounds toward zero).
Its value must fit the type its place asks, `u32` where the place is a count, a size, a rank or a
grid side (`K115`). A division or a remainder by zero is refused (`K114`); a parameter bound at
load or at the invocation inside one is refused (`K102`: a constant expression names only
parameters bound at build); a parameter named before its own declaration is refused (`K101`). At a
place of a float type (a float parameter's value) it is made of numbers with a point, float
parameters bound at build, `+ - * /` and brackets. In the program structure a constant expression
is the number it evaluates to, as a number written in the text is.

A constant expression is taken in these places:

- A module parameter's value: `param H: u32 = 2 * D @bind(build)`, of any binding of `H` (`= fit`
  is unchanged).
- Each side of `@on(grid[R, C])` on a handler and a chain, above 0 (`K115`).
- Each number of `@peers(offsets = [d, ...])` and `@peers(pairs = [(s, r), ...])`.
- A count in a type: `u32[N * 2]`, `tile<F>[N * 2]`; or the bare name of one module parameter of
  any binding.
- The size of an item, `t.items(size = B)` and `t.items(first, n, size = B)`; another value is
  refused (`K102`).
- The block of `stream(view, block = n)` and `sink(view, block = n)`; or the bare name of one
  module parameter of any binding; another value is refused (`K209`).

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
| `for b, w in br.arrive() { }` | a broadcast | | each block, a tile of the format the casts send, and its word | T6 |
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
| `matmul(a, b)` | two by position | tiles; in a function on blocks, blocks `[r, k]` by `[k, c]` (`K102`) | a tile; of blocks, a block `[r, c]` | T4, T13 |
| `a.T` | none | a tile | the tile turned | T4 |
| `zeros` | none | | a tile, or the scalars, of zeros | T4, T16 |
| A number constant, a scalar parameter | | | a scalar | T1, T2 |
| A call of a `func`, a `combine` or an `extern func` | its parameters, by position | as declared; one tile for a parameter `tile<F>` (`K203`); a block of the declared shape for a parameter `tile<F>[n]` or `tile<F>[r, c]` (`K102`) | its result | T1, T8 |

The function body of a `func` uses only these (`K202`). Operands of different formats may be mixed.

### A function on blocks

A `func` may be declared on blocks: `func f(x: tile<bf16>[8], k: f32) -> tile<bf16>[8]`. A block is
a matrix of tiles: `tile<F>[r, c]` is r rows of c tiles, and `tile<F>[n]` is `[1, n]`, one row of n
tiles. Its rows are a number written in the text or a module parameter bound at build (its value
is known at compile); one bound at load or at the invocation, or `group.size`, there is refused
(`K102`). Its columns are a number in the text or a module parameter, the block of the
stream the blocks come from. A block of more than one row stands only as a parameter or the result
of a function (`K010` elsewhere).

A handler may hold a block of one row in a `var` of `zeros`: `var acc: tile<f32>[WIDTH] = zeros`.
It is that invocation's own value, started at zeros. `acc += e` adds a block of its shape (`K102`
otherwise), and `store view <- acc` writes its tiles in the view's order; a view of another count
of tiles fails at the run. A frontend writes a field its handler fills, reads and frees in one
invocation this way; what one invocation leaves for another is a state member. A function on
one tile given such vars, of one shape, runs on each of their tiles and gives a block of that
shape (`store t.tiles[0:1, c:c + WIDTH] <- gate(h, g)`); given another block in a handler it is
`K203`.

Its body is typed on blocks: a parameter is the block it is declared, and a tile `tile<F>` is
`[1, 1]`. An elementwise operation (the arithmetic, the elementary functions, `where`, a
conversion) takes operands of one shape, a scalar going to every tile, and gives that shape
(`K102`). `matmul` takes an `[r, k]` by a `[k, c]` and gives an `[r, c]`, whose tile (i, j) is the
sum over k, in order, of the products of the left's tile (i, k) and the right's tile (k, j); its
result may be a tile, `[1, 1]`. A function on one tile called in it, given blocks of one shape, is
that function on each tile; a function on blocks is given blocks of its shapes. An elementwise body
runs on each tile of the blocks in turn; a last block that is shorter gives a shorter result. A
function on one tile is never given a block (`K203`).

A reduction along a row (`reduce_sum`, `reduce_max`, `reduce_min` with `axis = row`) runs along
each of the block's r rows: its c tiles side by side, a row of c times the tile's width, combined in
order from the row's first tile's first column. It gives one value for each row of each row of
tiles, of the block's shape. `.col()` of those values spreads each row's value over every tile of
its row of tiles, and `.row()` of a tile parameter spreads that tile's row 0 down its own tile. On
one tile the same body keeps its meaning: the tile's own rows. This is how a row wider than one
tile is normalised (`x - reduce_max(x, axis = row).col()`). A reduction along a column, `.col()` of
anything else, `.row()` of anything else and `.T` in its body are refused by name (`K010`), as is a `combine` or an `extern func` on blocks. What a backend computes of a row
is its own statement ([backends](backends.md)).

## Views of a tensor

| Operation | Operands | Types | Yields | Written in |
|---|---|---|---|---|
| `t.tiles(first, n)` | two by position | a tensor; integers | a view of n consecutive tiles | T1 |
| `t.tiles[r0:r1, c0:c1]`, `t.tiles[r0:r1:s, c0:c1:s]` | two ranges, each with a step *optional* | a tensor; integers | a rectangle of tiles, walked row by row | T7, Types and data |
| `v.T` | none | a rectangle view | the rectangle walked column by column, each tile as it is (a tile is turned by its own `.T`) | T7, Compute functions |
| `t.items(size = B)` | `size` by name | a tensor whose format is an integer type, `f32` or `f64` (`K208`); a constant expression (`K102`) | a view of all the tensor's items of B bytes | T7, Types and data |
| `t.items(first, n, size = B)` | `first` and `n` by position, `size` by name | a tensor; integers; B a constant expression (`K102`) | a view of n items of B bytes from item `first` | Types and data |
| `v[i]` | one by position | a view; an integer | one tile, read, of the view's format; through a `view` of no format (a parameter declared `view`), of the format its place states: a `let` with a type, an argument, a result; with no such place it is refused (`K102`); an index past the view's end is `out_of_range` (`K115`) | Handler bodies |
| `v.words()` | none | a view of one tile (`K207`) of a tensor whose format is an integer type, `f32` or `f64` (`K208`) | the scalars of the tile, read by index | T2, Types and data |
| `v.records(R, count = c)` | `R` by position, `count` by name, always given | a view of one tile of a tensor whose format is an integer type, `f32` or `f64` (`K208`); a record type; c records that fit the tile (`K115`) | the records, read by index and field | T7, Types and data |
| `items.put(src, at = i)` | `src` by position, `at` by name | an items view; a state member | writes one item | T7 |
| `partition(view, rank, size)` | three by position | a view; integers | this rank's contiguous part | T1 |
| `stream(view, block = n)`, `stream(view, block = [r, c])` | `view` by position, `block` by name, always given | a view; a constant expression or a module parameter (`K209`); r a number written in the text or a module parameter bound at build (`K102`) | the blocks in order: `tile<F>`, or `tile<F>[n]` for n above 1; of `[r, c]` with r above 1, the blocks `tile<F>[r, c]` of a rectangle view they divide, a row of blocks after another | T1 |
| `zip(s1, s2, ...)` | two or more by position | streams of the same number of blocks (`K115` where the compile knows each stream's tiles and block) | a block of each, together | T4 |
| `sink(view, block = n)` | `view` by position, `block` by name, always given | a view; a constant expression or a module parameter (`K209`); `[1, n]` is `n`, more rows `K010` | a write stream | T1 |
| `out.push(t)` | one by position | a write stream; a tile | writes the next block | T1 |
| `store view <- value` | a view, a value | a view; a tile, or a block of one row; or scalars: one scalar, a state member of scalars named whole, or the words of a tile, into a view of one tile (`K102` where the text shows more), of a format with stated bytes (`K208`), of the format's scalar type (`K102`), no more than the tile's rows times columns (`K115` where the text shows the count; at the run `out_of_range`) | writes it, visible at completion; a block's tiles in the view's order; scalars, the mirror of `.words()`: the tile's first k scalars, in the order `.words()` reads them, are the k values given, every other scalar of the tile is 0, and the whole tile is written | T4 |
| `past(y)` | a name | a tensor parameter of a chain | y as an earlier call wrote it; in a chain's call only (`K602`) | T10 |

A tensor is reached only through a view (`K107`).

## State

| Operation | Operands | Types | Yields | Written in |
|---|---|---|---|---|
| `s.fill(stream)` | one by position | a member; a stream that holds exactly the member's tiles (`K115`) | fills the member whole | T3 |
| `s.zero()` | none | a member | clears it, and it is filled | T3 |
| `s[i] = x` | an index, a value | a member; its element; an index inside the member (`K115`) | writes one element; it fills the member only when it is marked `@filled(element)` | T3 |
| `s[i]` | an index | a member; an index inside it (`K115`) | one element, read | T3, T7 |
| `s` | none | a member of one element (declared with no count), read by its name | its element 0, read: `s[0]` | T13 |
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
| `all_gather(x, publish = [v])` | `x` by position; `publish` by name, *optional* | a tile a worker holds, and the group's workers take part; views | `tile<F>[group.size]` at every worker of the group, indexed by rank: each worker's value kept apart, nothing combined (`gather` delivered to every participant, as `all_reduce` is `reduce` delivered to every participant) | Handler bodies |
| `gather(x)` | one by position | a value | `tile<F>[group.size]` at the leader, indexed by rank | T5 |
| `broadcast(along = row, sender = c)` | both by name; `sender` *optional*, 0 when left out | `row` or `col`; an integer | a broadcast, of the format of the tiles its casts send; some cast sends on it (`K404`); in a handler on a grid only (`K402`) | T6 |
| `br.is_sender` | none | | `bool` | T6 |
| `br.cast(block, word = w)` | `block` by position, `word` by name, *optional* | a tile; a scalar | sends the block, with the word when given; every cast on one broadcast sends one format (`K403`) | T6 |
| `br.arrive()` | none | | the blocks and their words, in order; always both | T6 |
| `sync(publish = [v])` | by name | views | a point the whole group reaches | T15 |

## Dynamic runs

In a dynamic run only (`K503`).

| Operation | Operands | Types | Yields | Written in |
|---|---|---|---|---|
| `M { field: value, ... }` | every field of `M` (`K113`) | scalars | a message | T9 |
| `message M @body(n) { ... }` | one, a whole number written in the text, above 0 and a multiple of 4 (`K004`) | | each `M` carries a body of up to n bytes beside its fields | Dynamic runs |
| `send(M { ... }, to = rank, body = src, publish = [v])` | the message by position; `to` by name; `body`, `publish` by name, *optional* | a message; an integer; u32 scalars: `m.body`, the words of a tile of `u32`, or a state member of `u32` named whole, given only to a type with `@body` and no longer than it (`K102`, `K115`); views | `bool`: false when refused. Without `body`, the body is empty; a state member not filled fails `empty_state`, a body longer than the type takes `out_of_range` | T9, T15 |
| `forward(m, as = T, with = { ... }, to = rank, publish = [v])` | `m` by position; `to` by name, always given; `as`, `with`, `publish` by name, *optional* | the arrived message; a message type (`K101`); some fields of `T` (of `m`'s type without `as`); an integer; views | sends the same unit of work on, its body unchanged. With `as`, as a `T`: each field of `T` from `with`, else from `m`'s field of the same name and scalar type (`K113`); `m`'s body fits `T`'s (`K102`) | T9, T15 |
| `consume m` | the arrived message | | ends the unit of work | T9 |
| `offer(M { ... }, to = rank, body = src)` | the message by position; `to` by name; `body` by name, *optional* | a message; an integer, to a declared peer (`K502`); u32 scalars, as `send`'s | `bool`: false while the peer holds the last one | Dynamic runs |
| `m.field` | none | an arrived message | the field | T9 |
| `m.body` | none | the arrived message, of a type with `@body` (`K102`) | its body; a field named `body` is written between backticks (`K120`) | Dynamic runs |
| `b.words()` | none | a message's body | its words, `u32` scalars, indexed `w[i]`; an index at or past the words the body holds fails `out_of_range` | Dynamic runs |

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
| `all_gather(t)` | one by position | in a control program, a replicated tensor (`K102`), and the islands take part | each island's value is what it wrote into its own copy since the copies were last equal (when the tensor was made, and after each `all_reduce`, `broadcast_from` or `all_gather` over it; a `reduce_to(t, to = i, ...)` counts as island i writing every place of `t`). After it every copy holds, at each place: written by one island, that island's value; written by none, what it held (the copies already agree there); written on several islands, one of those islands' values on every copy, which one the backend's (the language leaves that order open, as for any two unordered writes). After it the copies are equal | Islands |

## Tile shapes

A tile is the block of elements compute functions take and return; its shape is the module's own
choice. A tensor is held by the execution context and addressed in tiles by the code that reads or
writes it: the tile shape belongs to the code, not to the tensor. Two handlers of different shapes
may address one tensor; each sees the same elements, cut into its own tiles.

| Rule | Where it is checked |
|---|---|
| A module has one or more `tile [R, C]` lines. The first is the module's default shape. The same shape twice is refused | `K112` |
| `@tile(R, C)` names a shape of the module: two whole numbers written in the text (another argument form `K004`), one of the module's `tile` lines (`K101`) | Program compile |
| `@tile` stands on a `func`, a `combine`, an `extern func`, a handler, a run or a `state` block; anywhere else (a control program, a chain, a chain's call, a state member, a parameter, a message) it is refused. A declaration of that list without `@tile` has the default shape | `K006` |
| The shape of a declaration is that of every tile its body computes on, of every block of its streams and sinks, of every tile of every view it makes (`t.tiles(first, n)` counts the tensor's tiles of that shape, row of tiles after row of tiles), of the tile `.words()`, `.records()` and `.items()` lay their bytes by, and, for a `state` block, of its members of tiles | |
| A call of a `func` or an `extern func` from a body of another shape is refused | `K102` |
| A `combine` named by `reduce` or `all_reduce` in a handler body of another shape is refused | `K102` |
| A handler or run body that reads or writes a state member of tiles of a `state` block of another shape is refused; a member of scalars has no shape | `K102` |
| In a control program, `all_reduce` and `reduce_to` over the copies of a replicated tensor apply a `combine` in its own shape, a provided combination (`sum`, `max`, ...) element by element; `broadcast_from` and `all_gather` move elements and have no shape | |
| A chain calls handlers of any shapes; a control program invokes handlers of any shapes | |
| The proofs on views (`past`, `K601`; a race shown at compile, `K701`; a stream that writes where it reads, `K206`) compare two views only in bodies of one shape. Between two shapes a `past` is not shown, and is refused, the message saying the two calls address the tensor in different tile shapes; no race is claimed at compile | `K601` |
| A tensor has a padded shape in elements, fixed when the execution context makes it. An invocation of a handler of shape `[R, C]` given a tensor whose padded rows are not a multiple of R, or columns not a multiple of C, is invalid: "a tensor whose padded shape is not whole tiles of [R, C]" | At the invocation |
| Every read and write of a tile goes through the body's shape. Element (r, c) of the tensor is element (r mod R, c mod C) of tile `(r / R) * (padded cols / C) + c / C` in shape `[R, C]`, whatever shape the tensor is held in | At the run |
| The record of reads and writes that finds races between invocations no edge orders is exact between shapes: two accesses meet when they share an element | At the run |

## Named, not built yet

The language names these and the build does not have them yet. Each is refused by name (`K010`):

- In a function on blocks: a reduction along a column, `.col()` of other than a reduction along a
  row, `.row()` of other than a tile parameter, and `.T`; a `combine` or an `extern func` on blocks.
- A block of more than one row other than a function's parameter or result, and a `sink` of one.
