# A frontend's text and its origin

!!! note "Being built"
    The Komorebi IR and its command `komo` are being built. This page says how a frontend (a
    language that produces IR modules, such as Komorebi's Python classes) hands its modules to the
    core.

A frontend prints its module as IR text. The text is checked and compiled like any other module, so
every refusal of [the IR's checks](checks.md) applies to it, by name, before any backend runs.

The text does not say where it came from: a `.kom` text has no syntax for an origin. The frontend
gives its **origin** beside the text, and the core puts it in the module's notes (P in
[the byte-code program](bytecode.md)). There it changes only the full hash.

## What the origin holds

| Part | What it is |
|---|---|
| files | the frontend's source files, as it resolved them; a place names one by its index |
| lines | the source lines its places cover: (file, line number from 1, the line as written), in (file, number) order, each once |
| map | ranges of the text's lines, each with the place of the source it was printed for: (text's first line, text's last line, file, first line, last line) |
| renames | (the safe name the text writes, the source's name), for values and parameters |

## From the text's lines to places

After the module is lowered, each anchored thing whose span starts on line L of the text takes the
place of the **narrowest** range of the map that holds L. An anchor that no range holds has no place.
The spans are then cleared: a frontend's module has an origin and no spans.

Two ranges are nested or apart. Two ranges over the same lines, or two that cross, are refused.

A frontend therefore gives these ranges:
- one range over a whole declaration, mapped to the declaration's line. It places the declaration,
  its parameters and the lines nothing narrower holds.
- one range over each statement it prints, mapped to the source statement's lines. It places that
  statement's instructions.
- for a compound statement, the range runs from its first line to its closing brace. It places the
  loop or the branch itself, and its inner statements have narrower ranges of their own.

A diagnostic of the text is placed by the same rule, so a refusal names the source's line as well as
the text's.

## Names

A source's name may be a word of the IR (`group`, `tile`, `stream`), which cannot be written as a name
in the text. The frontend writes a safe name, and gives the source's name in `renames`. After
lowering, each name of a value, a slot or a parameter that is a safe name is replaced by its source
name. The renames are applied all at once, so a source's name may be another's safe name. A renamed
name stays in P, which only the full hash covers.

## What is refused

An origin is refused by what is wrong with it, before the module is used:
- a range on line 0, or one whose first line is after its last;
- a range in a file past the files;
- two ranges over the same lines, or two that cross;
- a safe name given twice, or one that names no value and no parameter;
- whatever the module's own rules refuse of an origin (`K954` in [the byte-code program](bytecode.md)),
  such as a place on a line the lines do not hold.

## From Python

`komorebi_ir.check(text, origin)` and `komorebi_ir.compile(text, origin)` take a
`komorebi_ir.Origin(files, lines, map, renames)`. A diagnostic then has a `place`: the file, and its
first and last lines. A refused origin raises `IrError`, and its message says what is wrong.

## Komorebi's classes

`komorebi.dsl.ir.module(cls, **make)` writes a class written as today's DSL writes it (a
`km.Program` with `km.handler` methods) as an IR module. It reads the handlers with `ast`, as the
class compiler does, and gives a `Made`: the text and `made.origin` for `komorebi_ir.compile`. It
does not import `komorebi_ir`, and today's make does not call it.

It takes the lines of the first lesson: a partition of a tensor's pages, a write stream and a read
loop over a partition, and a push of a tile function's result. Every other line is refused by name,
with the class's line, and so is a make keyword that would change the module. The make's tensors
give the module's formats; a tensor's pages must be one tile each, since an IR view is in tiles.
`l1=`, `depth=` and `log=` are not part of the module.

A partition the class applies to a second tensor (`y[src]`) is written as the same range of that
tensor, partitioned alike. The Tensix backend prints one partition for the two.
