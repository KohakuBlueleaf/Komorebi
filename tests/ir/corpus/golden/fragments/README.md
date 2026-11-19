# Fragments

The code fragments of the specification that are not whole modules, each made one: a `module` line
named by its topic, statements inside a `control c()` and program items inside a `program P`.
Each is in the layout `komo fmt` writes; the round-trip test parses, writes back and compares it.
Fragments with `...` in place of a body are left out.

Four carry the header lines a whole module needs (`tile`, `format`, and `requires` where they use
a dynamic run) and check clean: `count_program`, `forward_program`, `state_filled` and
`sweep_halves`; the golden test asserts it. The others name what the specification declares around
them, so they are read and written back only.
