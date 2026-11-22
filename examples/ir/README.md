# IR examples

Work in progress: the Komorebi IR is being built, and these files change with it.

## The lessons

`lessons/` holds the IR lessons of the tutorial, each the same file as the test corpus's
(`tests/ir/corpus/golden/lessons/`), with the run scripts that run it on the interpreter
([docs/ir/run-scripts.md](../../docs/ir/run-scripts.md)). Beside each script, `<script>.report`
is the report it prints; the test `examples` runs every script and compares.

| Lesson | Shows | Scripts |
|---|---|---|
| `t01_scale.kom` | One handler over a tensor | `t01_scale_group1`, `t01_scale_group3`, `t01_scale_group4`, `t01_scale_group7`, `t01_scale_ragged`, `t01_scale_view_end` |
| `t02_count.kom` | Parameters, requirements, integers and a result | `t02_count` |
| `t03_gain.kom` | State that stays between invocations | `t03_gain`, `t03_gain_report` |
| `t04_compute.kom` | Compute functions | `t04_compute` |
| `t05_total.kom` | Combining over the group | `t05_total` |
| `t06_rows.kom` | A grid and a broadcast | `t06_rows` |
| `t07_views.kom` | Views, items and records | `t07_views` |
| `t08_trace.kom` | An external function: the circuit slot | `t08_trace` |
| `t09_walk.kom` | A dynamic run | `t09_walk`, `t09_walk_live` |
| `t10_steps.kom` | A chain | `t10_steps` |
| `t11_solve.kom` | A control program | `t11_solve` |
| `t12_spread.kom` | Islands | `t12_spread_two`, `t12_spread_line`, `t12_spread_star`, `t12_spread_merge`, `t12_spread_report` |
| `t13_hinted.kom` | Hints | `t13_hinted` |
| `t15_handoff.kom` | Handing data over | `t15_handoff` |
| `t16_retry.kom` | A failure as a value | `t16_retry` |

Each script's first line says what it runs and what to look for in its report. To run one with
the repository's Rust toolchain:

```
cargo run -q --bin komo -- run examples/ir/lessons/t01_scale.kom examples/ir/lessons/t01_scale_group3.run
```

## Modules the DSL frontend writes

Each `.kom` file directly in this folder is the Komorebi IR module the DSL frontend writes for a
program it takes end to end: the first lesson (T1) and the corpus's programs 1 to 7. Its first
line names the Python class it comes from, in `tests/ir/tensix/`. The files are what
`komorebi.dsl.ir.built.module` writes today, in the writer's own layout: the frontend's tests fail
when one is not, and `python tests/ir/frontend/ir_examples.py --write` writes them again
([docs/ir/frontends.md](../../docs/ir/frontends.md)). To check one, and to print the program it
compiles to: `cargo run -q --bin komo -- check examples/ir/p7_swiglu.kom` and
`cargo run -q --bin komo -- print examples/ir/p7_swiglu.kom`.
