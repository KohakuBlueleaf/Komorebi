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
| `K9xx` | Failures of an invocation found only in the run |

The code `K004` is not used: the rule it stood for became part of `K010`.

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
| `K005` | No `while` in a handler body | Grammar: "`while` is allowed in a control body and not in a handler body" | Program compile | `k005_while_in_handler` |
| `K006` | A known attribute only where it has a meaning | Grammar, a reading the build uses: "A known attribute where it has no meaning (`@export` on a handler) [...] Both are errors. Nothing is supplied silently" | Program compile | `k006_export_on_handler`, `k006_replicated_on_scalar` |
| `K007` | A declaration carries the attribute every example gives it: `@on` on a handler and on a chain, `@dynamic` on a run, `@bind` on a parameter, `@scope` and `@lifetime` on a `state` | Grammar, a reading the build uses: "a declaration without the attribute every example gives it (a handler with no `@on`, a run with no `@dynamic`, a parameter with no `@bind`, a `state` with no `@scope` and `@lifetime`) [...] Both are errors. Nothing is supplied silently"; Chains, a reading the build uses: "A chain states `@on` as a handler does" | Program compile | `k007_handler_without_on`, `k007_chain_without_on`, `k007_run_without_dynamic`, `k007_param_without_bind`, `k007_state_without_scope` |
| `K008` | `= fit` only on a parameter bound at load | Module, for `@bind(load)`: "The module is loaded on a machine. `= fit` means the largest value such that declared state fits the worker memory"; a reading the build uses: "`= fit` on a parameter not bound at load" is an error | Program compile | `k008_fit_not_load` |
| `K009` | A module states in `requires` what its content uses: `dynamic` for a dynamic run, `group = grid` for a handler or a chain on a grid, `impl("name")` for an external function with no fallback | Module: a module must state an ability its content uses: "A missing line is an error that names the construct and the line to add. The block is then always complete: a reader and a load see what a module needs without analysis"; `group = line` is every target's and is never required; `islands >= n` and `local_state >= ...` are stated only; Chains, a reading the build uses: "a chain on a grid needs `group = grid`" | Program compile | `k009_requires_dynamic`, `k009_requires_grid`, `k009_requires_grid_chain`, `k009_requires_impl` |
| `K010` | What the language names and the build does not have yet is refused by name: `all_gather`; a `store` of scalars; a message body, `@body` and `as =` on a `forward`; several tile shapes in a module; a function on a block of several tiles | Module, a reading the build uses: "One diagnostic says so by name: `all_gather`; a `store` of scalars; a message body and `as =` on a `forward` [...]; several tile shapes in a module; a function on a block of several tiles" | Program compile | `k010_all_gather`, `k010_store_scalars`, `k010_message_body`, `k010_forward_as`, `k010_two_tile_shapes`, `k010_func_on_block` |
| `K011` | The text nests no deeper than the tool reads: an expression at most 256 deep (each operand, sign, operator of a chain, member, call, index and bracket counts one level as it nests), and blocks at most 256 deep | Grammar, a reading the build uses: "The language has no limit. The tool reads an expression nested up to 256 deep and refuses a deeper one by a name of its own, apart from a grammar error". A limit of this tool, not of the language | Program compile (parse) | `k011_too_deep` |

## Names and types

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K101` | Every name is declared | The pipeline: program compile "Checks names and types" | Program compile | `k101_unknown_name` |
| `K102` | Every operand has the type its operation takes | The pipeline, as above; Programs, state, handlers: "`-> T`: one result of a scalar type" | Program compile | `k102_wrong_type` |
| `K103` | Every operand an operation takes is given | Handler bodies, for `partition(view, rank, size)`: "This rank's contiguous part of a view" | Program compile | `k103_missing_operand` |
| `K104` | A `let` is not assigned again | Handler bodies: "`let` (single assignment), `var` (mutable)" | Program compile | `k104_assign_let` |
| `K105` | `worker.row` and `worker.col` only in a handler on a grid | Programs, state, handlers: "`@on(grid[R, C])`, which adds `worker.row` and `worker.col`" | Program compile | `k105_row_outside_grid` |
| `K106` | A format the module uses is declared | Module: "Element formats the module uses, each defined by its set of values [...]. Declared so a reader and a load see them without analysis" | Program compile | `k106_format_not_declared` |
| `K107` | A tensor is reached only through a view | Handler bodies: "A tensor is reached only through a view; `view[i]` is one tile and is a read" | Program compile | `k107_tensor_indexed` |
| `K108` | A message's fields are scalars | Types and data, for `message`: "Named fields of scalar types, with an optional body of bytes" | Program compile | `k108_message_field` |
| `K109` | A boolean is bound, tested and compared; it is not the type of state, a field, a parameter or a result | Types and data, for `bool`: "The result of a comparison, a `send` or an `offer`. It may be bound, tested and compared for equality; it is not a type of state, of a field, of a parameter or of a result" | Program compile | `k109_bool_parameter`, `k109_bool_state` |
| `K110` | A format is one of the language's formats | Numbers, a reading the build uses: "Closed: `f16 f32 f64`, `bf16`, the integers, `bfp8`, `bfp4`, `bfp2`. Another name is an error" | Program compile | `k110_unknown_format` |
| `K111` | Each operand in its own form, by position or by name | Handler bodies, a reading the build uses: "Each operation has one form, the one the lessons write: `partition(view, rank, size)`, `stream(view, block = n)`, `reduce(x, with = f)`, `group.split(parts = n)`. Another spelling is an error"; the forms are listed in [the IR's operations](operations.md) | Program compile | `k111_split_positional`, `k111_stream_positional` |
| `K112` | A name is declared once | Programs, state, handlers, a reading the build uses: "one name declared twice" is an error | Program compile | `k112_name_twice` |
| `K113` | A message is made with every one of its fields | Dynamic runs, a reading the build uses: "a message made without one of its fields" is an error | Program compile | `k113_message_missing_field` |
| `K114` | An integer is not divided by zero | Order, state and failure, a reading the build uses: "`divide_by_zero`: an integer `/` or `%` by zero, of scalars or of integer tiles; a float tile follows IEEE [...] Where the build can show one before the run, it refuses the program" | Program compile, where the divisor is a constant zero; otherwise the run, as the failure `divide_by_zero` | `k114_divide_by_zero` |
| `K115` | An index or a count fits what it addresses | Order, state and failure, a reading the build uses: "`out_of_range`: an index past the end of a state member, of a tile's scalars or records, of a view (`v[i]`, `put(at = i)`); a `fill` whose stream does not hold exactly the member's tiles; a `zip` of two streams of different lengths; a negative shift amount. Where the build can show one before the run, it refuses the program" | Program compile, where the text shows the index and the end, or both counts; otherwise the run, as the failure `out_of_range` | `k115_index_past_end`, `k115_fill_count`, `k115_zip_lengths`, `k115_records_too_many` |
| `K116` | A reduction's values are spread into a tile before they meet one | Compute functions, a reading the build uses: "`reduce_sum(a, axis = row)` yields a value for each row, which is not a tile: operations element by element apply to it (with a scalar or with another of its kind), and `.col()` makes it a tile [...] Used in any other place (stored, pushed, given to `matmul`, mixed with a tile) it is an error" | Program compile | `k116_reduction_with_tile` |
| `K117` | An integer tile does not meet a float | Numbers, a reading the build uses: "An integer tile with a float (scalar or tile) is an error: the language has no conversion between integer and float tiles yet" | Program compile | `k117_integer_tile_float` |
| `K118` | A module that uses a tile, a tensor or a view has its `tile` line; reported once, at the first use | Module, a reading the build uses: "A module that names a tile, a tensor or a view needs its tile line" | Program compile | `k118_no_tile_line` |

Formats may be mixed: operands of different formats in one operation, and a tile of one format
written to a tensor of another, are accepted with no rule, since a format is a notation (Numbers).

## Functions, views and streams

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K201` | A `func` takes no view | Compute functions: "A `func` is pure and takes no view" | Program compile | `k201_func_takes_view` |
| `K202` | A function body uses the compute family only | The hierarchy: "A function body uses the compute family only" | Program compile | `k202_func_not_compute` |
| `K203` | A block of several tiles is not given to a function on one tile | Handler bodies: "A block of n tiles is `tile<F>[n]`. Giving it to a function declared on one tile is a type error" | Program compile | `k203_block_to_tile_func` |
| `K204` | An `extern func` that takes a view states what it touches | Compute functions: "An `extern func` may take one and states what it touches with `@reads` and `@writes`" | Program compile | `k204_extern_view_unstated` |
| `K205` | A `fallback` body touches only what `@reads` and `@writes` state | Compute functions: "its `fallback` body may touch what those state and nothing else" | Program compile | `k205_fallback_touches` |
| `K206` | A stream writes where it reads only on exactly the view it reads | Handler bodies: "A stream may write where it reads when the view written is exactly the view read [...] Any other overlap is refused at program compile" | Program compile | `k206_stream_overlap` |
| `K207` | `.words()` and `.records()` read one tile | Handler bodies, for `view.words()` and `view.records(R, count = c)`: "Scalars or records from one tile"; a reading the build uses: "`.words()` or `.records()` on a view shown to hold more than one tile" is an error | Program compile, where the view's count is known | `k207_words_many_tiles` |
| `K208` | `items`, `records` and `words` read the bytes of a tensor whose format states them | Types and data, a reading the build uses: "Only on a tensor whose format is an integer type, `f32` or `f64`: the other formats are notations with no stated bytes, and the build refuses the program" | Program compile | `k208_words_bf16` |

## State

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K301` | A state member is read only when it is filled | Programs, state, handlers: "Reading what is not filled is a failure of the invocation, on every backend. Program compile refuses a member that some body reads and no body of the program fills; any other such read is found at the run, since whether a member is filled depends on what ran before" | Program compile, for a member that no body of the program fills; otherwise the run, as the failure `empty_state` | `k301_empty_state_body`; the run: `k301_empty_state_chain`, `k301_empty_state_invocations` |

## Collectives

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K401` | A reduce combines with a `combine` function or a provided one | Handler bodies: "A `combine` function used in a reduce is one for which every grouping is an acceptable result [...] `sum`, `max`, `min`, `and`, `or` are provided and commutative" | Program compile | `k401_reduce_with_func` |
| `K402` | A broadcast needs a handler on a grid | Handler bodies: "Needs `group = grid`"; a reading the build uses: "Refused: `along = row` and `along = col` need `@on(grid[R, C])`" | Program compile | `k402_broadcast_on_line` |

## Dynamic runs

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K501` | Every arrived message is forwarded or consumed exactly once | Dynamic runs: "every arrived message is forwarded or consumed exactly once"; The pipeline: program compile checks "every message forwarded or consumed once" | Program compile | `k501_message_kept`, `k501_message_twice` |
| `K502` | An `offer` goes to a declared peer | Dynamic runs: "An `offer` to a rank that is not a declared peer is refused at program compile where the rank is known, and is a failure of the run where it is not"; a reading the build uses: "Known means a constant rank, or `(worker.rank + d) % group.size` with a constant d" | Program compile where the rank is known; otherwise the run | `k502_offer_no_peer`, `k502_offer_offset_two` |
| `K503` | `send`, `forward`, `consume` and `offer` only in a dynamic run | Dynamic runs, a reading the build uses: "`send`, `forward`, `consume` or `offer` outside a dynamic run" is an error | Program compile | `k503_send_in_handler` |

## Chains

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K601` | Every block a `past` reads was written by an earlier call | Chains: "`past(y)` reads y as an earlier call wrote it. The build proves that every block read was written before, or refuses, naming the call, the tensor, the tile and the worker" | Program compile (needs the derived effects) | `k601_past_unproven` |
| `K602` | `past` only in a chain's call | Chains, a reading the build uses: "`past` outside a chain's call" is an error | Program compile | `k602_past_in_handler` |
| `K603` | An operand of a chain's call is one of the chain's parameters, `past(y)`, or a constant (a number, a negated number, `true`, `false`, a module parameter bound at build). A module parameter bound at load or at the invocation is refused for now: the program structure has no form for it yet | Chains, a reading the build uses: a chain's call takes the chain's parameters, `past(y)` or constants; any other expression there is an error | Program compile | `k603_chain_operand` |

A call of a chain may read what an earlier call wrote without `past`: a chain means its calls in the
written order. `past(y)` asks the build for the proof of `K601`; leaving it out is not an error.

## Control programs and order

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K701` | No race: two actions on one place, one a write, with no edge between them | Order, state and failure: "A race is two actions on the same place, at least one of them a write, with no edge [...] between them. The IR gives a race no meaning: it is an invalid program"; "The CPU interpreter reports every race that happens in the run it executes, and program compile what it can show before a run" | Program compile, arguments known (needs the derived effects); the interpreter, for a race in the run it executes | `k701_race_order`, `k701_race_halves` |
| `K702` | A control body uses values, control flow, `invoke`, `await` and results | The hierarchy: "A control body uses values, control flow, `invoke`, `await` and results"; Islands: the collectives over the copies are "called from a control program" | Program compile | `k702_control_not_values` |
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
| `K115` | An index or a count that does not fit what it addresses, where the text does not show it | `out_of_range` | Order, state and failure: "An index or a count that does not fit what it addresses" | The run |

One more rule is found only in the run, and is not a failure of an invocation:

| Code | Rule | Stated | Found by |
|---|---|---|---|
| `K906` | An `await` of a failed invocation fails the control program | Control programs: "`await h` fails the control program when `h` failed" | The run |
