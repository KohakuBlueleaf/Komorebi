# IR examples

Each `.kom` file is the Komorebi IR module the DSL frontend writes for a program it takes end to
end: the first lesson (T1) and the corpus's programs 1 to 7. Its first line names the Python class
it comes from, in `tests/ir/tensix/`. The files are what `komorebi.dsl.ir.built.module` writes
today, in the writer's own layout: the frontend's tests fail when one is not, and
`python tests/ir/frontend/ir_examples.py --write` writes them again
([docs/ir/frontends.md](../../docs/ir/frontends.md)).

To check one, and to print the program it compiles to, with the repository's Rust toolchain:

```
cargo run -q --bin komo -- check examples/ir/p7_swiglu.kom
cargo run -q --bin komo -- print examples/ir/p7_swiglu.kom
```
