---
title: Tile functions
summary: What komorebi.dsl.tile turns into a compute body for the vector unit, what it refuses and why, the cost it prints, and the gate an emitted body passes.
tags:
  - reference
  - compute
---

# Tile functions

Status: experimental, nothing in the runtime imports it. Five bodies emitted from tile
functions ran on one Blackhole core against the hand bodies they stand in for; each
computes the same expression as its hand body within the stated tolerance
([tile_rules](../../experiments/tile_rules/README.md)).

A **tile function** is a Python function traced into the tile IR. `komorebi.dsl.tile`
turns it into a **unit program**: the vector-unit (SFPU) instructions the math CPU
issues for one unit of work (one tile, or one row of tiles), the calls around them
(copies into `Dst`, packs), the rules applied as passes, the hazard checks, and a cost.
`emit` writes the unit program as a C++ header: `<name>_init()` once, `<name>(dst)` per
unit, raw Tensix words only (no floats, every immediate known when the header is built).

## Writing one

```python
from komorebi.dsl.tile import emit, trace, unit_program
from komorebi.dsl.tile.ir import F32, KernelIR, TensorParam

a, b = TensorParam("a", False, F32), TensorParam("b", False, F32)
y = TensorParam("y", True, F32)
ir = KernelIR("gate", [a, b, y], [], {})
with trace.Tracer(ir):
    xs, ys, gs = trace.TensorProxy(a), trace.TensorProxy(y), trace.TensorProxy(b)
    for t in trace.grid(trace.num_tiles(xs)):
        x, g = trace.load(xs.tiles[t]), trace.load(gs.tiles[t])
        trace.store(ys.tiles[t], trace.silu(x) * g)

unit = unit_program(ir)  # or unit_program(ir, load_macro=True)
print(unit.dump())  # passes, checks, cost
header = emit(unit)  # km_tile::gate_init(), km_tile::gate(dst)
```

| Call | What it does |
|---|---|
| `unit_program(ir, dest="fp32", load_macro=False)` | the stream form: one store a tile, of an expression over up to 3 whole input tiles (`Dst` tiles 0, 1, 2). `load_macro=True` adds the load-macro rule (below) |
| `rows.row_program(ir, width, rows_tile, cols_tile)` | the row form: a row of `width` tiles (1 or 2) in `Dst` tiles 0 .. width - 1, a per-row value in `Dst` tile `rows_tile`, a per-column operand in `Dst` tile `cols_tile` |
| `emit(unit)` | the C++ header; `UnitError` if a word does not encode |
| `unit.dump()` | the program as text: setup, loop nests, each pass and what it did, each check and its verdict, the rules not checked, the cost on each CPU |
| `unit.cost(cpu)` | `(instructions, cycles a unit, terms)` on `"unpack"`, `"math"` or `"pack"` |

## What a tile function may contain

**Stream form.** Each tile of the output is an expression of the same tile of up to
three inputs: `+`, `-`, `*`, unary minus, Python number constants, `trace.sigmoid`,
`trace.silu` (`x * sigmoid(x)`) and `trace.rsqrt`. Inputs are bf16 or fp32 tensors;
`Dst` is fp32. Constants become `SFPLOADI` immediates, or one of the three programmable
constant registers (LReg 12 to 14) when that saves words.

**Row form.** One reduction, `trace.reduce_sum(f(row), exact=True)`, along a row of
tiles loaded once; a per-row value computed from it (constants, `+`, `*`, `rsqrt`); the
stored row an expression of the row, the per-row value broadcast with `.col()` and one
resident row (a weight, `trace.resident(...)`) broadcast with `.row()`. The row norm
`x * rsqrt(mean(x^2) + eps) * w` is this form:

```python
ms = trace.reduce_sum(row * row, exact=True) * (1.0 / 64)
r = trace.rsqrt(ms + 1e-6)
trace.store(ys.tiles[t, :], row * r.col() * weight.row())
```

## Rules

Each rule is a pass; `dump()` prints what each one did.

| Rule | Kind | What it does |
|---|---|---|
| fold_negation | algebra | a negation disappears into the multiply-add that makes it or reads it (a sign bit) |
| fuse_mad | algebra | a product read only by an add becomes one `SFPMAD` (one rounding in place of two) |
| constant_registers | layout | the constants the rows load move into LReg 12 to 14, most words saved first; equal constants share one |
| hoist | schedule | the setup of those registers moves out of the rows, once a stream |
| allocate | layout | each value an LReg 0 .. 7 for its live range |
| replay | schedule | the first row's instructions recorded into the replay buffer as it runs, replayed by the others |
| load_macro | schedule | a row of loads, one operation a sub-unit and a store becomes one `SFPLOADMACRO` a row |
| per-row lanes | layout | a per-row value lives as lanes of one tile: slot `(h, q)` of the rows tile, lane `l` holding row `16 h + 4 q + l / 8`, the same in all 8 lanes of its lane row; the sum along a row is a multiply-add chain then three lane rotations (`SFPSHFT2`, 1, 2 and 4 lanes) |
| broadcast | layout | `.col()` reads the per-row slot once a half-row; `.row()` reads the per-column operand once a vector column; no other broadcast is built |

## What is refused, and why

| Refusal | Why |
|---|---|
| "the unit program takes one store" | one output tile a unit is what the loop nest and the pack calls are built for |
| "the unit program reads at most 3 operand tiles" | the operands sit in `Dst` tiles 0 .. 2 |
| "the unit program reads whole-tile operands" | a slice or a broadcast operand needs the row form |
| "vector function f is not built" / "vector operation k is not built" | only the functions and operations listed above have a lowering |
| "the unit program takes no runtime scalars" | every immediate is built into the header; a run-time word would need building on the core |
| "tensors are bf16 or fp32" | the loads and stores are fp32 `Dst` accesses |
| "the unit program is built for fp32 Dst" | the address walk assumes 64 address units a tile |
| row form: "rows of 1 or 2 tiles" | the per-column operand tile holds two tiles of columns |
| row form: one reduction, `reduce_sum(..., exact=True)` along the row; one `.col()`; one loaded row; `.row()` of a resident row only | what the stat and apply phases are built for |
| load macro: "the row computes with N instructions (M MAD-class); a macro arms at most one a sub-unit" | a macro has one slot each for the simple, MAD and round sub-units and one for the store; SiLU (45 instructions) and the gate (46) do not fit, and run as plain rows |
| load macro: a lane-flag instruction, a store of a value other than the last, a store more than 7 cycles after its load, more than 4 templates | the macro's slots run unguarded, its delays are 3 bits, the unit holds 4 templates |

A refused load macro is not an error: the pass records "refused: reason" and the rows
stay plain.

## Checks

After the passes, each check repairs what it can and prints a verdict: `mad-reader`
(an `SFPNOP` between a multiply-add and a reader the unit does not hold back: `SFPIADD`,
`SFPSHFT`, `SFPSHFT2` lane modes, `SFPSETSGN` / `SFPSETEXP` without an immediate, the
half writes of `SFPLOADI`, `SFPCONFIG` reading LReg 0, as the ISA's `SFPMAD` page lists
them), `config-nop`, `constants` (set before use, LReg 11 restored to -1.0),
`predication` (every lane on at the end), `dst-load` (the matrix unit's writes finished
before the first `SFPLOAD`), `immediates`, and in the row form `lanes` (each per-row slot
stored complete, the same value in the 8 lanes of a lane row) and `broadcast` (each
stored vector followed back to the loads it reads). The rules no check covers yet are
printed as "not checked".

## The cost printed

`cost(cpu)` adds up a price for each instruction, each loop pass and each stall it
predicts; the calls (`copy_tile`, `pack_tile`, the syncs) are listed and not priced.
Every price names the one-core measurement it comes from
([tile_probes](../../experiments/tile_probes/README.md), section 8):

| Term | Cycles |
|---|---|
| a word pushed | 1 |
| a word replayed | 1.004 |
| a pass of a loop the core runs | N + 2 |
| a multiply-add result read by the next instruction | 1 stall (not when the reader is a store) |
| an `SFPSHFT2` lane mode | 1 stall (the ISA's statement) |
| a word built at run time | 4.14 |
| a word from a table | 9.37 |
| one `SFPLOADMACRO` row | 1.046 |

Against the card (math CPU, between two `tensix_sync()`, less the empty span), cycles a
tile:

| Body | Estimate | Card | Error |
|---|---|---|---|
| SiLU | 1593.6 | 1592 | +0.1% |
| add | 182.6 | 183.5 | -0.5% |
| add, load macro | 91.6 | 93 | -1.5% |
| scale by a constant | 150.4 | 155 | -3.0% |
| scale, load macro | 59.5 | 62 | -4.0% |
| gate `silu(a) * b` | 1657.6 | 1656.5 | +0.1% |
| row norm (a row of 2 tiles) | 753.0 | 771.5 | -2.4% |

The cost covers the vector unit only. A whole unit also pays the copies into `Dst` and
the pack, which the card shows to be most of an add or a scale.

## The gate

An emitted body passes when it computes the same expression as the hand body it
replaces, within a stated tolerance; bit equality is not a goal. The tolerance is the
largest relative error against float64 of the same fp32 inputs, fixed before the run:
2^-23 for one fp32 rounding (add, scale), 1e-6 for SiLU and the gate, 1e-5 for the
norm (a sum of 64 squares, a 23-bit reciprocal square root, two products). The hand
bodies' own errors are reported against the same float64 reference.
