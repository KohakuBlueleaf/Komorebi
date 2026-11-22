# The IR's checks (being built)

!!! note "Being built"
    The Komorebi IR and its command `komo` are being built. This page lists the rules the build
    checks; it grows with the checker.

Every rule of the IR that a program can break has one code. This page gives, for each, the sentence
of the specification that states it, with the topic it belongs to, and who finds a program that
breaks it:

- **program compile**: always, before any run (`komo check`);
- **program compile, arguments known**: before the run, where the values the rule depends on are
  written in the module (a control program's invocations with their arguments and groups);
- **the run**: a named failure of the invocation, on every backend;
- **the interpreter**: an invalid program that only the CPU interpreter finds, in the run it executes.

A rule about what the module does as a whole (`K205`, `K206`, `K301`, `K405`, `K406`, `K601`,
`K701`) is checked on the module lowered to the program structure, with the effects of each body
derived; a module the build cannot lower yet is refused with `K010`, naming what it does not build,
so that no rule is skipped without saying so.

A rule found two ways (program compile where it can show it, the run otherwise) is one rule with one
code. Where the text leaves a point open, the build takes the stricter reading, since a program
refused today can be allowed later without breaking any program; such a row says "a reading the build
uses".

## The codes

`K` and three digits. The first digit is the part of the language the rule belongs to:

| Codes | Part |
|---|---|
| `K0xx` | The text, the module's declarations, attributes |
| `K1xx` | Names and types |
| `K2xx` | Functions, external functions, views and streams |
| `K3xx` | State |
| `K4xx` | Collectives |
| `K5xx` | Dynamic runs |
| `K6xx` | Chains |
| `K7xx` | Control programs and the order between invocations |
| `K8xx` | Islands |
| `K9xx` | Failures of an invocation found only in the run; from `K950`, a byte-code file refused; from `K960`, a backend's refusal |

A program compile diagnostic is written `file:line:col: code: text`. Every case named below is a
module of `tests/ir/corpus/golden/`, in `invalid/` (refused by program compile) or in `run_fails/`
(checks clean, and its run fails), with a `.expect` file beside it.

The operations and their operands, which several rows below refer to, are listed in
[the IR's operations](operations.md).

## The text, declarations and attributes

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K001` | The text follows the grammar | Grammar | Program compile (parse) | `k001_parse` |
| `K002` | A `state` declaration is a block | Programs, state, handlers: "Always a block: `state { ... }` with the shared attributes on it. One form, with no second spelling for a single member" | Program compile (parse) | `k002_state_not_block` |
| `K003` | An undotted attribute is one of the language's own | Grammar: "A closed set: the language's own. A name outside it is an error at program compile, never a warning: an undotted attribute changes what a program means" | Program compile | `k003_unknown_attribute` |
| `K004` | An attribute of the language's own takes only the arguments it is given for: `@scope(worker)`, `@lifetime(program)`, `@filled(element)`, `@bind` with one of `build`, `load`, `invoke`, `@on(group)` or `@on(grid[R, C])` on a handler or a chain (R and C each a constant expression, [operations](operations.md#constant-expressions)), `@body(n)` on a message (n a whole number of bytes written in the text, above 0 and a multiple of 4), and no argument list after `@exclusive` and `@export`. Another argument, a missing argument list or more than one argument is an error at the attribute. `@on` on a chain's call is `K604`'s; a name between backticks is `K101`'s; `@tile(R, C)` takes two whole numbers written in the text ([operations](operations.md#tile-shapes)) | Grammar, a reading the build uses: nothing is supplied silently, and an attribute changes what a program means | Program compile | `k004_scope_argument`, `k004_lifetime_argument`, `k004_filled_argument`, `k004_on_argument`, `k004_export_arguments`, `k004_bind_argument`, `k004_body_argument`, `k004_impl_argument`, `k004_peers_offsets_number`, `k004_peers_offsets_pair`, `k004_peers_pairs_number`, `k004_peers_pairs_triple`, `k004_peers_unknown_label`, `k004_peers_unlabelled`, `k004_tile_argument` |
| `K005` | No `while` in a handler body | Grammar: "`while` is allowed in a control body and not in a handler body" | Program compile | `k005_while_in_handler` |
| `K006` | A known attribute only where it has a meaning; `@tile` on a `func`, a `combine`, an `extern func`, a handler, a run or a `state` block only | Grammar, a reading the build uses: "A known attribute where it has no meaning (`@export` on a handler) [...] Both are errors. Nothing is supplied silently" | Program compile | `k006_export_on_handler`, `k006_replicated_on_scalar`, `k006_attribute_on_func`, `k006_tile_on_control` |
| `K007` | A declaration carries the attribute every example gives it: `@on` on a handler and on a chain, `@dynamic` on a run, `@bind` on a parameter, `@scope` and `@lifetime` on a `state` | Grammar, a reading the build uses: "a declaration without the attribute every example gives it (a handler with no `@on`, a run with no `@dynamic`, a parameter with no `@bind`, a `state` with no `@scope` and `@lifetime`) [...] Both are errors. Nothing is supplied silently"; Chains, a reading the build uses: "A chain states `@on` as a handler does" | Program compile | `k007_handler_without_on`, `k007_chain_without_on`, `k007_run_without_dynamic`, `k007_param_without_bind`, `k007_state_without_scope`, `k007_extern_without_impl` |
| `K008` | `= fit` only on a parameter bound at load | Module, for `@bind(load)`: "The module is loaded on a machine. `= fit` means the largest value such that declared state fits the worker memory"; a reading the build uses: "`= fit` on a parameter not bound at load" is an error | Program compile | `k008_fit_not_load` |
| `K009` | A module states in `requires` what its content uses: `dynamic` for a dynamic run, `group = grid` for a handler or a chain on a grid, `impl("name")` for an external function with no fallback | Module: a module must state an ability its content uses: "A missing line is an error that names the construct and the line to add. The block is then always complete: a reader and a load see what a module needs without analysis"; `group = line` is every target's and is never required; `islands >= n` and `local_state >= ...` are stated only; Chains, a reading the build uses: "a chain on a grid needs `group = grid`" | Program compile | `k009_requires_dynamic`, `k009_requires_grid`, `k009_requires_grid_chain`, `k009_requires_impl` |
| `K010` | What the language names and the build does not have yet is refused by name: in a function on blocks, an operation on a whole tile other than the row form and a product of blocks (a reduction along a column, `.col()` of other than a reduction along a row, `.row()` of other than a tile parameter, `.T`), and a `combine` or an `extern func` on blocks; a block of more than one row other than a function's parameter or result, and a `sink` of one; a module the build cannot lower to the program structure yet, whose rules on what it does are then not checked | Module, a reading the build uses: "One diagnostic says so by name: `all_gather`; a `store` of scalars; [...]; a function on a block of several tiles" | Program compile | `k010_func_on_block` |
| `K011` | The text nests no deeper than the tool reads: an expression at most 256 deep (each operand, sign, operator of a chain, member, call, index and bracket counts one level as it nests), and blocks at most 256 deep | Grammar, a reading the build uses: "The language has no limit. The tool reads an expression nested up to 256 deep and refuses a deeper one by a name of its own, apart from a grammar error". A limit of this tool, not of the language | Program compile (parse) | `k011_too_deep` |

## Names and types

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K101` | Every name is declared, a name in a type's count included; a constant expression names a parameter after its declaration only; `@tile(R, C)` names a shape one of the module's `tile` lines declares | The pipeline: program compile "Checks names and types" | Program compile | `k101_unknown_name`, `k101_count_undeclared`, `k101_parameter_named_before_it`, `k101_tile_shape_not_declared` |
| `K102` | Every operand has the type its operation takes; a tile of an integer format stands only where its own format is declared; a count in a type is a constant expression or a module parameter; a constant expression names only parameters bound at build; the size of an item is a constant expression; a state member is tiles or scalars; a store of scalars writes one tile, of the scalar type of the tensor's format; a tile read through a view of no format has a place that states its format; the rows of a block `[r, c]` and of a stream's `block = [r, c]` are a number written in the text or a module parameter bound at build; in a function on blocks an elementwise operation takes operands of one shape (a scalar any), `matmul` an `[r, k]` by a `[k, c]`, and a call blocks of the callee's shapes (or, of a function on one tile, blocks of one shape); a block stored into a view of as many tiles, where the compile knows both counts; a body calls a function, names a `combine` in a `reduce` or an `all_reduce` (in a handler), and reads or writes a state member of tiles, each of its own tile shape ([operations](operations.md#tile-shapes)) | The pipeline, as above; Programs, state, handlers: "`-> T`: one result of a scalar type"; Numbers, a reading the build uses: "Among the float formats this is a notation only: a float tile of any format may stand where a float tile of another is declared. Integer formats must agree"; Compute functions, a reading the build uses: a tile read through a `view` of no format takes the format its place states | Program compile | `k102_wrong_type`, `k102_view_read_no_format`, `k102_block_count`, `k102_store_scalars_many_tiles`, `k102_store_scalars_other_type`, `k102_constant_names_load_parameter`, `k102_item_size_not_constant`, `k102_state_member_tensor`, `k102_state_member_view`, `k102_state_member_record`, `k102_all_gather_not_tile`, `k102_count_too_large`, `k102_gather_not_tile`, `k102_list_not_a_value`, `k102_no_value_await`, `k102_no_value_bracketed`, `k102_no_value_cast`, `k102_no_value_fill`, `k102_no_value_forward`, `k102_no_value_push`, `k102_no_value_sync`, `k102_no_value_zero`, `k102_param_record`, `k102_param_tensor`, `k102_param_tile`, `k102_param_tile_value`, `k102_param_view`, `k102_partition_stream_no_format`, `k102_range_alone`, `k102_range_as_value`, `k102_records_no_format`, `k102_reduce_not_tile`, `k102_sink_no_format`, `k102_spread_not_tile`, `k102_state_scalars_alone`, `k102_state_scalars_as_value`, `k102_stream_alone`, `k102_stream_no_format`, `k102_stream_read_as_value`, `k102_string_not_a_value`, `k102_tuple_not_a_value`, `k102_where_condition_and`, `k102_where_condition_bool`, `k102_where_condition_true`, `k102_words_no_format`, `k102_zeros_let`, `k102_zeros_operand`, `k102_zeros_pushed`, `k102_zip_as_value`, `k102_call_across_tile_shapes`, `k102_combine_of_another_tile_shape`, `k102_state_of_another_tile_shape` |
| `K103` | Every operand an operation takes is given | Handler bodies, for `partition(view, rank, size)`: "This rank's contiguous part of a view" | Program compile | `k103_missing_operand` |
| `K104` | A `let` is not assigned again | Handler bodies: "`let` (single assignment), `var` (mutable)" | Program compile | `k104_assign_let` |
| `K105` | `worker.row` and `worker.col` only in a handler on a grid | Programs, state, handlers: "`@on(grid[R, C])`, which adds `worker.row` and `worker.col`" | Program compile | `k105_row_outside_grid` |
| `K106` | A format the module uses is declared | Module: "Element formats the module uses, each defined by its set of values [...]. Declared so a reader and a load see them without analysis" | Program compile | `k106_format_not_declared` |
| `K107` | A tensor is reached only through a view | Handler bodies: "A tensor is reached only through a view; `view[i]` is one tile and is a read" | Program compile | `k107_tensor_indexed` |
| `K108` | A message's and a record's fields are scalars | Types and data, for `message`: "Named fields of scalar types, with an optional body of bytes" | Program compile | `k108_message_field`, `k108_record_field` |
| `K109` | A boolean is bound, tested and compared; it is not the type of state, a field, a parameter or a result | Types and data, for `bool`: "The result of a comparison, a `send` or an `offer`. It may be bound, tested and compared for equality; it is not a type of state, of a field, of a parameter or of a result" | Program compile | `k109_bool_parameter`, `k109_bool_state`, `k109_bool_module_parameter` |
| `K110` | A format is one of the language's formats | Numbers, a reading the build uses: "Closed: `f16 f32 f64`, `bf16`, the integers, `bfp8`, `bfp4`, `bfp2`. Another name is an error" | Program compile | `k110_unknown_format` |
| `K111` | Each operand in its own form, by position or by name | Handler bodies, a reading the build uses: "Each operation has one form, the one the lessons write: `partition(view, rank, size)`, `stream(view, block = n)`, `reduce(x, with = f)`, `group.split(parts = n)`. Another spelling is an error"; the forms are listed in [the IR's operations](operations.md) | Program compile | `k111_split_positional`, `k111_stream_positional` |
| `K112` | A name is declared once; a tile shape is declared once (one `tile` line a shape) | Programs, state, handlers, a reading the build uses: "one name declared twice" is an error | Program compile | `k112_name_twice`, `k112_tile_shape_twice` |
| `K113` | A message is made with every one of its fields; a `forward` with `as = T` gives each field of `T`, from `with` or from the arrived message's field of the same name and scalar type | Dynamic runs, a reading the build uses: "a message made without one of its fields" is an error | Program compile | `k113_message_missing_field`, `k113_forward_as_field_missing` |
| `K114` | An integer is not divided by zero, a constant expression's included | Order, state and failure, a reading the build uses: "`divide_by_zero`: an integer `/` or `%` by zero, of scalars or of integer tiles; a float tile follows IEEE [...] Where the build can show one before the run, it refuses the program" | Program compile, where the divisor is a constant zero; otherwise the run, as the failure `divide_by_zero` | `k114_divide_by_zero`, `k114_constant_divides_by_zero` |
| `K115` | An index or a count fits what it addresses (a store of scalars no more than one tile holds); a constant expression's value fits the type its place asks, and a side of a grid is above 0; a number written in the text fits the type its place gives it (a negative number at an unsigned place is refused: `let a = -1` has no place and is `u32`, so the text writes `let a: i32 = -1`); the number of a conversion takes the type named as its place, so `u32(-1)` is refused (all ones is written `0xffffffff`) | Order, state and failure, a reading the build uses: "`out_of_range`: an index past the end of a state member, of a tile's scalars or records, of a view (`v[i]`, `put(at = i)`); a `fill` whose stream does not hold exactly the member's tiles; a `zip` of two streams of different lengths; a negative shift amount. Where the build can show one before the run, it refuses the program"; Numbers, a reading the build uses: "float to integer drops the fraction, and a value the type cannot hold (or a NaN) is the failure `out_of_range`" | Program compile, where the text shows the index and the end, both counts, or the number; otherwise the run, as the failure `out_of_range` | `k115_index_past_end`, `k115_fill_count`, `k115_zip_lengths`, `k115_records_too_many`, `k115_conversion`, `k115_store_scalars_too_many`, `k115_grid_side_zero`, `k115_item_size_too_large`, `k115_islands_too_large`, `k115_local_state_bytes_too_large`, `k115_local_state_tiles_too_large`, `k115_tile_line_too_large` |
| `K116` | A reduction's values are spread into a tile before they meet one | Compute functions, a reading the build uses: "`reduce_sum(a, axis = row)` yields a value for each row, which is not a tile: operations element by element apply to it (with a scalar or with another of its kind), and `.col()` makes it a tile [...] Used in any other place (stored, pushed, given to `matmul`, mixed with a tile) it is an error" | Program compile | `k116_reduction_with_tile` |
| `K117` | An integer tile does not meet a float | Numbers, a reading the build uses: "An integer tile with a float (scalar or tile) is an error: the language has no conversion between integer and float tiles yet" | Program compile | `k117_integer_tile_float` |
| `K119` | The two operands of a scalar operation or comparison (and of `x += e`) have one type; a number written in the text takes the other operand's type; a value of another type is converted by naming the type (`u32(x)`). The amount of a shift may be of any integer type | Numbers, a reading the build uses: "An error: the two operands of a scalar operation or comparison have one type. Nothing is converted silently. A number written in the text takes the other operand's type. A value of another type is converted by naming the type: `f32(n)`, `u32(x)`, `i64(a)`. Integer to integer wraps in the new width; integer to float is its value; float to integer drops the fraction, and a value the type cannot hold (or a NaN) is the failure `out_of_range`. The amount of a shift may be of any integer type" | Program compile | `k119_mixed_scalars` |
| `K118` | A module that uses a tile, a tensor or a view has its `tile` line; reported once, at the first use | Module, a reading the build uses: "A module that names a tile, a tensor or a view needs its tile line" | Program compile | `k118_no_tile_line` |
| `K120` | A bare name that is the language's where an item's kind stands is the language's; the module's item of that name is written between backticks, at its declaration and at every use. A function (`func`, `combine`, `extern func`) against the language's calls, the conversions and the combinations `sum max min and or`; a value (a local, a parameter, a loop's name, a state member, a module constant) against `zeros worker group island`; a record or a message against the words a type is written with; a message's field against `body`, the language's after a message (`m.body`). A handler, a run, a chain, a program, a field or a message member against the words of the text only, which do not read bare | R50: "a name is the name its author gave"; "a bare name means the language's wherever the language has a meaning for it in that place" | Program compile | `k120_bare_word` |

Formats may be mixed: operands of different formats in one operation, and a tile of one format
written to a tensor of another, are accepted with no rule, since a format is a notation (Numbers).

## Functions, views and streams

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K201` | A `func` takes no view | Compute functions: "A `func` is pure and takes no view" | Program compile | `k201_func_takes_view` |
| `K202` | A function body uses the compute family only | The hierarchy: "A function body uses the compute family only" | Program compile | `k202_func_not_compute` |
| `K203` | A block of several tiles is not given to a function on one tile in a handler, other than the handler's vars of a block of one row of one shape, which it runs on each tile | Handler bodies: "A block of n tiles is `tile<F>[n]`. Giving it to a function declared on one tile is a type error" | Program compile | `k203_block_to_tile_func` |
| `K204` | An `extern func` that takes a view states what it touches | Compute functions: "An `extern func` may take one and states what it touches with `@reads` and `@writes`" | Program compile | `k204_extern_view_unstated` |
| `K205` | A `fallback` body touches only what `@reads` and `@writes` state | Compute functions: "its `fallback` body may touch what those state and nothing else" | Program compile | `k205_fallback_touches` |
| `K206` | A stream writes where it reads only on exactly the view it reads | Handler bodies: "A stream may write where it reads when the view written is exactly the view read [...] Any other overlap is refused at program compile" | Program compile (needs the derived effects) | `k206_stream_overlap` |
| `K207` | `.words()` and `.records()` read one tile | Handler bodies, for `view.words()` and `view.records(R, count = c)`: "Scalars or records from one tile"; a reading the build uses: "`.words()` or `.records()` on a view shown to hold more than one tile" is an error | Program compile, where the view's count is known | `k207_words_many_tiles` |
| `K208` | `items`, `records` and `words` read the bytes of a tensor whose format states them, and a store of scalars writes them | Types and data, a reading the build uses: "Only on a tensor whose format is an integer type, `f32` or `f64`: the other formats are notations with no stated bytes, and the build refuses the program" | Program compile | `k208_words_bf16`, `k208_store_scalars_no_stated_bytes` |
| `K209` | The block of a `stream` or a `sink` is a constant expression or a module parameter; a handler's parameter, another value of the run or `group.size` is refused | Handler bodies, a reading the build uses: `stream(view, block = n)` hands "the view's blocks, in order", and the count of tiles in a block is known before the run | Program compile | `k209_stream_block` |

## State

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K301` | A state member is read only when it is filled | Programs, state, handlers: "Reading what is not filled is a failure of the invocation, on every backend. Program compile refuses a member that some body reads and no body of the program fills; any other such read is found at the run, since whether a member is filled depends on what ran before" | Program compile (needs the derived effects), for a member that no body of the program fills; otherwise the run, as the failure `empty_state` | `k301_empty_state_body`; the run: `k301_empty_state_chain`, `k301_empty_state_invocations` |

## Collectives

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K401` | A reduce combines with a `combine` function or a provided one | Handler bodies: "A `combine` function used in a reduce is one for which every grouping is an acceptable result [...] `sum`, `max`, `min`, `and`, `or` are provided and commutative" | Program compile | `k401_reduce_with_func` |
| `K402` | A broadcast needs a handler on a grid | Handler bodies: "Needs `group = grid`"; a reading the build uses: "Refused: `along = row` and `along = col` need `@on(grid[R, C])`" | Program compile | `k402_broadcast_on_line` |
| `K403` | One broadcast carries one format: every cast on it sends tiles of the same format, and its arrivals are tiles of that format | Handler bodies: a broadcast's `cast` and `arrive` | Program compile | `k403_broadcast_formats` |
| `K404` | A broadcast is sent on: some cast sends on it, else it carries nothing and its arrivals have no format | Handler bodies: a broadcast's `cast` and `arrive` | Program compile | `k404_broadcast_no_cast` |
| `K405` | A value that a collective gives only the leader (`reduce`, `gather`) is used only where the leader alone runs | A reading the build uses, on how the reference interpreter runs a program: "A value a collective gives only the leader is absent on the other ranks, and a use of it there is an invalid program" | Program compile (needs the derived effects), where the body shows it: a use outside every branch on the rank, or in the `else` of `worker.is_leader`; a `reduce ... among` takes it from the leaders only | `k405_leader_value` |
| `K406` | Every member of a group takes part in its collectives | A reading the build uses, on how the reference interpreter runs a program: "[...] so is a collective that not every member reaches" | Program compile (needs the derived effects), where the body shows it: a collective inside a branch whose condition depends on the worker's rank or identity | `k406_collective_branch` |

## Dynamic runs

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K501` | Every arrived message is forwarded or consumed exactly once | Dynamic runs: "every arrived message is forwarded or consumed exactly once"; The pipeline: program compile checks "every message forwarded or consumed once" | Program compile | `k501_message_kept`, `k501_message_twice` |
| `K502` | An `offer` goes to a declared peer | Dynamic runs: "An `offer` to a rank that is not a declared peer is refused at program compile where the rank is known, and is a failure of the run where it is not"; a reading the build uses: "Known means a constant rank, or `(worker.rank + d) % group.size` with a constant d" | Program compile where the rank is known; otherwise the run | `k502_offer_no_peer`, `k502_offer_offset_two` |
| `K503` | `send`, `forward`, `consume` and `offer` only in a dynamic run | Dynamic runs, a reading the build uses: "`send`, `forward`, `consume` or `offer` outside a dynamic run" is an error | Program compile | `k503_send_in_handler` |

## Chains

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K601` | Every block a `past` reads was written by an earlier call; what the build proves is on views: the tiles a call reads through `past(y)` lie, worker by worker, inside a view an earlier call opened for writing (a sink's view, a store's target, a put's items), not that every tile was pushed; the views of two handlers of different tile shapes are not compared, and a `past` between them is refused | Chains: "`past(y)` reads y as an earlier call wrote it. The build proves that every block read was written before, or refuses, naming the call, the tensor, the tile and the worker" | Program compile (needs the derived effects) | `k601_past_unproven`, `k601_past_across_tile_shapes` |
| `K602` | `past` only in a chain's call | Chains, a reading the build uses: "`past` outside a chain's call" is an error | Program compile | `k602_past_in_handler` |
| `K603` | An operand of a chain's call is one of the chain's parameters, `past(y)`, or a constant (a number, a negated number, `true`, `false`, a module parameter bound at build). A module parameter bound at load or at the invocation is refused for now: the program structure has no form for it yet | Chains, a reading the build uses: a chain's call takes the chain's parameters, `past(y)` or constants; any other expression there is an error | Program compile | `k603_chain_operand` |
| `K604` | A call's `@on` is `group[first:end]`, a range of the chain's group: each end a number or a module parameter bound at build, the first below the end, and the end within the chain's group where its size is stated (`@on(grid[R, C])`); otherwise at the invocation | R47: "A call in a chain may name a part of the chain's group ... A range past the chain's group is refused where the group's size is known" | Program compile | `k604_call_part` |

A call of a chain may read what an earlier call wrote without `past`: a chain means its calls in the
written order. `past(y)` asks the build for the proof of `K601`; leaving it out is not an error.

## Control programs and order

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K701` | No race: two actions on one place, one a write, with no edge between them; two invocations on the two arms of one `if` never both run | Order, state and failure: "A race is two actions on the same place, at least one of them a write, with no edge [...] between them. The IR gives a race no meaning: it is an invalid program"; "The CPU interpreter reports every race that happens in the run it executes, and program compile what it can show before a run" | Program compile, arguments known (needs the derived effects); the interpreter, for a race in the run it executes | `k701_race_order`, `k701_race_halves` |
| `K702` | A control body uses values, control flow, `invoke`, `await` and results; of the island queries, `island.count` and `island.of(id)` only, since `island.crosses(id)` and `island.hops(id)` ask about the asking worker's own island | The hierarchy: "A control body uses values, control flow, `invoke`, `await` and results"; Islands: the collectives over the copies are "called from a control program"; a reading the build uses: a control program runs on no worker, so it has no island of its own to ask about | Program compile | `k702_control_not_values`, `k702_island_hops_in_control` |
| `K703` | `invoke` and `await` only in a control program | Control programs, a reading the build uses: "`invoke` or `await` outside a control program" is an error | Program compile | `k703_invoke_in_handler` |
| `K704` | A failure's name is one of the ten | Order, state and failure: the names are "`view_end`, `empty_state`, `send_refused`, `capacity`, `given_up`, `dependency`, `not_a_peer`, `over_bound`, and [...] `divide_by_zero` and `out_of_range`. Another name is an error" | Program compile | `k704_unknown_reason` |

## Islands

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K801` | `reduce_to` and `broadcast_from` only in a control program | Islands: "Only `reduce_to` and `broadcast_from`, called from a control program, move data between copies"; a reading the build uses: "`reduce_to` or `broadcast_from` in a handler" is an error | Program compile | `k801_reduce_to_in_handler` |

## Failures found in the run

The named failures of an invocation (Order, state and failure): "Failures, by name. A control program
compares the name: `o.reason == capacity`." Each failure ends the invocation without a result; its
name is what `o.reason` yields. The rows `K9xx` can only show in a run and have no case; `K114`,
`K115`, `K301` and `K502` are rules that program compile finds where it can show them, and the run
finds otherwise.

| Code | Failure | Its name | Stated | Found by |
|---|---|---|---|---|
| `K901` | A view past its tensor's end | `view_end` | Types and data: "A view past its tensor's end is a stated failure: nothing is read or written and the invocation does not complete. It is never a wrong read" | The run |
| `K301` | A read of empty state, where whether the member is filled depends on what ran before | `empty_state` | Programs, state, handlers: "any other such read is found at the run, since whether a member is filled depends on what ran before" | The run (cases in `run_fails/`) |
| `K902` | A `send` that is refused and whose result is not used | `send_refused` | Dynamic runs: "New work is bounded, so a send may be refused; it yields a boolean, and unused, a refusal gives up by name" | The run |
| `K903` | A refusal by the backend: one of its limits exceeded | `capacity` | Order, state and failure: "A refusal by the backend: one of its limits exceeded" | The backend |
| `K904` | A give-up by the service | `given_up` | The execution interface: "an invocation that is given up has no completion, and other invocations are unaffected" | The run |
| `K905` | A failed dependency | `dependency` | Control programs: "If `a` failed, the invocation does not start and fails with \"a dependency failed\"" | The run |
| `K502` | An `offer` to a rank that is not a declared peer, where the rank is not known before the run | `not_a_peer` | Dynamic runs: "An `offer` to a rank that is not a declared peer [...] is a failure of the run where it is not" | The run |
| `K907` | An invocation passes a parameter above its declared largest value | `over_bound` | Module, for `@bind(invoke)`: "Each invocation. The declared value is the largest allowed" | The run |
| `K114` | An integer divided by zero, by `/` or `%`, where the divisor is not a constant | `divide_by_zero` | Order, state and failure: "An integer divided by zero, by `/` or `%`" | The run |
| `K115` | An index or a count that does not fit what it addresses, or a float converted to an integer type that cannot hold it, where the text does not show it | `out_of_range` | Order, state and failure: "An index or a count that does not fit what it addresses"; Numbers, a reading the build uses: "a value the type cannot hold (or a NaN) is the failure `out_of_range`" | The run |

One more rule is found only in the run, and is not a failure of an invocation:

| Code | Rule | Stated | Found by |
|---|---|---|---|
| `K906` | An `await` of a failed invocation fails the control program | Control programs: "`await h` fails the control program when `h` failed" | The run |

## A byte-code file

A byte-code program (`.komc`) is refused when its bytes are not one this build writes; no rule of
the language is broken, and no failure has these names. [The byte code](bytecode.md) gives the
layout. A program loaded to run is accepted only where its text would check: after `K955`, the rules
of Verify run on it, each refused with its own code.

| Code | Refused | Found by |
|---|---|---|
| `K950` | Not a byte-code program: shorter than the magic and the version, or another magic | The reader |
| `K951` | A byte-code program of a newer version than the build reads, refused by name | The reader |
| `K952` | A byte-code program of an older version, refused by name until the language is released | The reader |
| `K953` | Bytes this version's writer would not write: a part cut short or longer than its walk, a number out of its range, bytes after the last part | The reader |
| `K954` | A module that reads whole and does not validate | The reader |
| `K955` | A module whose derived parts (a body's effects, a chain's dependencies, the requirement set) are not the ones Derive gives it again | Loading it |

## A backend refuses

A backend states what it provides, implements, allows and computes; a program that asks for more is
refused by name before anything is emitted, at the construct that asks (What a backend states:
"A backend that cannot provide a requirement, or exceeds one of its own limits, refuses the module
and names what. It never substitutes other behaviour silently"). What depends on the machine
(islands, worker memory) is compared when the program is loaded on it.

| Code | Refused | Found by |
|---|---|---|
| `K960` | A requirement the backend does not provide: `group = grid`, `dynamic`, `islands >= n`, `local_state >= ...` (the amounts on the machine, at load) | The backend's check, at compile and at load |
| `K961` | An external function with no fallback whose implementation the backend does not have | The backend's check |
| `K962` | A program over one of the backend's limits: handlers in a program, parameters or argument words of a handler, run or chain, fields of a message; a tile function the backend's compute unit cannot take, with the unit's reason | The backend's check |
| `K963` | A format the module declares that the backend does not compute | The backend's check |
| `K964` | A construct the backend does not emit yet | The backend's check |
| `K965` | A module whose tile shape the backend does not compute on | The backend's check |
| `K966` | A machine given at load that is not a machine: its islands' sizes not summing to its workers, an island of no worker, a connection naming no island, islands not connected | A backend's load and the reference interpreter's, with the same text |

A limit only the run can show stays the run's failure `capacity` (`K903`).
