# Fragments

The code fragments of the specification that are not whole modules, each made one: a `module` line
named by its topic, statements inside a `control c()` and program items inside a `program P`.
Each is in the layout `komo fmt` writes; the round-trip test parses, writes back and compares it.
Fragments with `...` in place of a body are left out. Whether one checks clean is the checker's to say.
