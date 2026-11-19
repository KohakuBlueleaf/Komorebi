# Fragments

The code fragments of the specification that are not whole modules, each made one: a `module` line
named by its topic, statements inside a `control c()` and program items inside a `program P`.
Each is in the layout `komo fmt` writes; the round-trip test parses, writes back and compares it.
Fragments with `...` in place of a body are left out.

Five carry the header lines a whole module needs (`tile`, `format`, and `requires` where they use
a dynamic run or an external function) and check clean: `count_program`, `forward_program`,
`state_filled`, `sweep_halves` and `tile_functions`; the golden test asserts it. The others name what the specification declares around
them, so they are read and written back only.
