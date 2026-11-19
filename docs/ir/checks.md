# The IR's checks (being built)

!!! note "Being built"
    The Komorebi IR and its command `komo` are being built. This page lists the rules the build
    checks; it grows with the checker.

Every rule of the IR that a program can break has one code. This page gives, for each, the sentence
of the specification that states it (with its section, or its row among the settled points), and who
finds a program that breaks it:

- **program compile**: always, before any run (`komo check`);
- **program compile, arguments known**: before the run, where the values the rule depends on are
  written in the module (a control program's invocations with their arguments and groups);
- **the run**: a named failure of the invocation, on every backend;
- **the interpreter**: an invalid program that only the CPU interpreter finds, in the run it executes.

A rule found two ways (program compile where it can show it, the run otherwise) is one row with one
code.

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
| `K9xx` | Failures of an invocation found in the run |

A program compile diagnostic is written `file:line:col: code: text`. Every case named below is a
module in `tests/ir/corpus/golden/invalid/`, with a `.expect` file beside it.

## The text, declarations and attributes

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K001` | The text follows the grammar | Section 24, the grammar | Program compile (parse) | `k001_parse` |
| `K002` | A `state` declaration is a block | Row 38: "Always a block: `state { ... }` with the shared attributes on it. One form, with no second spelling for a single member" | Program compile (parse) | `k002_state_not_block` |
| `K003` | An undotted attribute is one of the language's own | Row 40: "A closed set: the language's own. A name outside it is an error at program compile, never a warning" | Program compile | `k003_unknown_attribute` |
| `K004` | One tile shape in a module, for now | Section 25, F1: "Several tile shapes in one module (row 18) have no syntax: the grammar has one `tile [R, C]`"; the build takes one shape a module until a form is settled | Program compile | `k004_two_tile_shapes` |
| `K005` | No `while` in a handler body | Section 24: "`while` is allowed in a control body and not in a handler body" | Program compile | `k005_while_in_handler` |

## Names and types

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K101` | Every name is declared | Section 20: program compile "checks names and types" | Program compile | `k101_unknown_name` |
| `K102` | Every operand has the type its operation takes | Section 20, as above; section 11: "`-> T`: one result of a scalar type" | Program compile | `k102_wrong_type` |
| `K103` | Every operand an operation takes is given | Section 12: "`partition(view, rank, size)`: this rank's contiguous part of a view" | Program compile | `k103_missing_operand` |
| `K104` | A `let` is not assigned again | Section 12: "`let` (single assignment), `var` (mutable)" | Program compile | `k104_assign_let` |
| `K105` | `worker.row` and `worker.col` only in a handler on a grid | Section 11: "`@on(grid[R, C])`, which adds `worker.row` and `worker.col`" | Program compile | `k105_row_outside_grid` |
| `K106` | A format the module uses is declared | Section 7: "Element formats the module uses, each defined by its set of values (section 9). Declared so a reader and a load see them without analysis" | Program compile | `k106_format_not_declared` |
| `K107` | A tensor is reached only through a view | Section 12: "A tensor is reached only through a view; `view[i]` is one tile and is a read" | Program compile | `k107_tensor_indexed` |
| `K108` | A message's fields are scalars | Section 8: "`message`: Named fields of scalar types, with an optional body of bytes" | Program compile | `k108_message_field` |

## Functions, views and streams

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K201` | A `func` takes no view | Section 10: "A `func` is pure and takes no view" | Program compile | `k201_func_takes_view` |
| `K202` | A function body uses the compute family only | Section 6: "A function body uses the compute family only" | Program compile | `k202_func_not_compute` |
| `K203` | A block of several tiles is not given to a function on one tile | Section 12: "A block of n tiles is `tile<F>[n]`. Giving it to a function declared on one tile is a type error" | Program compile | `k203_block_to_tile_func` |
| `K204` | An `extern func` that takes a view states what it touches | Section 10: "An `extern func` may take one and states what it touches with `@reads` and `@writes`" | Program compile | `k204_extern_view_unstated` |
| `K205` | A `fallback` body touches only what `@reads` and `@writes` state | Section 10: "its `fallback` body may touch what those state and nothing else" | Program compile | `k205_fallback_touches` |
| `K206` | A stream writes where it reads only on exactly the view it reads | Section 12: "A stream may write where it reads when the view written is exactly the view read [...] Any other overlap is refused at program compile" | Program compile | `k206_stream_overlap` |

## State

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K301` | A state member is read only when it is filled | Section 11: "Reading what is not filled is a failure of the invocation, on every backend; inside one body, and across the calls of one chain, program compile finds it before the run" | Program compile inside one body and across a chain's calls (needs the derived effects); otherwise the run, as the failure "a read of empty state" | `k301_empty_state_body`, `k301_empty_state_chain` |

## Collectives

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K401` | A reduce combines with a `combine` function or a provided one | Section 12: "A `combine` function used in a reduce is one for which every grouping is an acceptable result [...] `sum`, `max`, `min`, `and`, `or` are provided and commutative" | Program compile | `k401_reduce_with_func` |

## Dynamic runs

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K501` | Every arrived message is forwarded or consumed exactly once | Section 13: "every arrived message is forwarded or consumed exactly once"; section 20: program compile checks "every message forwarded or consumed once" | Program compile | `k501_message_kept`, `k501_message_twice` |
| `K502` | An `offer` goes to a declared peer | Section 13 and row 43: "An `offer` to a rank that is not a declared peer is refused at program compile where the rank is known, and is a failure of the run where it is not" | Program compile where the rank is known; otherwise the run | `k502_offer_no_peer` |

## Chains

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K601` | Every block a `past` reads was written by an earlier call | Section 14: "`past(y)` reads y as an earlier call wrote it. The build proves that every block read was written before, or refuses, naming the call, the tensor, the tile and the worker" | Program compile (needs the derived effects) | `k601_past_unproven` |

## Control programs and order

| Code | Rule | Stated | Found by | Case |
|---|---|---|---|---|
| `K701` | No race: two actions on one place, one a write, with no edge between them | Section B2: "A race is two actions on the same place, at least one of them a write, with no edge of B1 between them. The IR gives a race no meaning: it is an invalid program"; row 29: "The CPU interpreter reports every race that happens in the run it executes, and program compile what it can show before a run" | Program compile, arguments known (needs the derived effects); the interpreter, for a race in the run it executes | `k701_race_order`, `k701_race_halves` |
| `K702` | A control body uses values, control flow, `invoke`, `await` and results | Section 6: "A control body uses values, control flow, `invoke`, `await` and results"; section 16: the island collectives are called from a control program | Program compile | `k702_control_not_values` |

## Failures found in the run

These are the named failures of an invocation (section B3). No module can show them before the run;
they have no case in the corpus.

| Code | Failure | Stated | Found by |
|---|---|---|---|
| `K901` | A view past its tensor's end | Section 8: "A view past its tensor's end is a stated failure: nothing is read or written and the invocation does not complete. It is never a wrong read" | The run |
| `K902` | A `send` that is refused and whose result is not used | Section 13: "New work is bounded, so a send may be refused; it yields a boolean, and unused, a refusal gives up by name" | The run |
| `K903` | A refusal by the backend | Section 4: "A backend that cannot provide a requirement, or exceeds one of its own limits, refuses the module and names what" | The backend, at load or at the run |
| `K904` | A give-up by the service | Section 19: "an invocation that is given up has no completion, and other invocations are unaffected" | The run |
| `K905` | A failed dependency | Section 15: "If `a` failed, the invocation does not start and fails with \"a dependency failed\"" | The run |
| `K906` | An `await` of a failed invocation fails the control program | Section 15: "`await h` fails the control program when `h` failed" | The run |

A read of empty state across invocations is `K301`, and an `offer` whose destination is not known
before the run is `K502`: one rule each, found by the run where program compile cannot show it.
