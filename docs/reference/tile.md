---
title: Tile functions
summary: What komorebi.dsl.tile turns into a compute body for the vector unit, what it refuses and why, the cost it prints, and the gate an emitted body passes.
tags:
  - reference
  - compute
---

# Tile functions

Status: experimental, nothing in the runtime imports it. The bodies emitted from tile
functions for SiLU, a residual add, a scale, the SwiGLU gate, the row norm, attention's
exponent step and its online softmax step ran on one Blackhole core against the hand
bodies they stand in for; each computes the same expression as its hand body within the
stated tolerance ([tile_rules](../../experiments/tile_rules/README.md)). The matrix
unit's words of a tile product, emitted from the library's no-MOP order, gave the
library call's result bit for bit.

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
| `steps.step_program(ir, width, state_tile, fidelity="HiFi4")` | the step form: a row of `width` tiles (1 to 3) in `Dst` tiles 0 .. width - 1, stored over them, and a state tile of per-row values in `Dst` tile `state_tile`, read and written in place; `fidelity` is the matrix unit's, for a row sum the cost model gives it |
| `row_operand(row, width)` | the per-column operand tile `.row()` reads, placed from a plain row of 32 `width` values (1024 values in face order, written once by the host) |
| `matrix.product_program(name, fidelity)` | one tile product with the math CPU's words emitted (the library's unpack and `mm_no_mop_init_short` are the caller's) |
| `matrix.tile_products(fidelity, ct_dim, rt_dim, dst)` | the words of one `matmul_block_no_mop` call in issue order, each with its `Dst` block |
| `matrix.block_cost(fidelity, ct_dim, rt_dim, kt_dim=1)` | the priced cycles and words of `kt_dim` K steps of `matmul_block` (the MOP form issues the same words as the no-MOP form) |
| `emit(unit)` | the C++ header; `UnitError` if a word does not encode, a check refuses (below) |
| `unit.dump()` | the program as text: setup, loop nests, each pass and what it did, each check and its verdict, the rules not checked, the cost on each CPU |
| `unit.cost(cpu)` | `(instructions, cycles a unit, terms)` on `"unpack"`, `"math"` or `"pack"` |

## What a tile function may contain

**Stream form.** Each tile of the output is an expression of the same tile of up to
three inputs: `+`, `-`, `*`, unary minus, Python number constants, `trace.sigmoid`,
`trace.silu` (`x * sigmoid(x)`) and `trace.rsqrt`. Inputs are bf16 or fp32 tensors;
`Dst` is fp32. Constants become `SFPLOADI` immediates, or one of the constant registers
when that saves words: the three programmable ones (LReg 12 to 14), then LReg 11, which
the unit sets after its setup and restores to -1.0 at its end.

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

**Transcendentals at a stated error.** `trace.exp(x, error=None, over=None,
exact_at_zero=False)` and `trace.exp2(...)`: `error` is the relative error the program
accepts for inputs in `over = (lo, hi)`; the lowering takes the cheapest template whose
bound over that range meets it, and `dump()` names the template and its bound. Without
`error` the accurate template is taken. `exact_at_zero=True` makes the result 1 exactly
at 0 (a rescale factor that must be none when nothing moved).

| Function | Template | Bound | Range |
|---|---|---|---|
| `exp` | flash: `2^(a log2 e)`, the integer part shifted into the exponent, a degree-2 polynomial for the fraction | 1.75e-3 | [-87, 88] |
| `exp` | accurate: Cody-Waite reduction, degree-6 polynomial | 3e-7 | all |
| `exp2` | flash | 1.75e-3 | [-126, 127] |
| `exp2` | accurate: `exp(a ln 2)` | 6e-6 | all |

**Step form.** The online softmax step of attention is this form: per row of score
tiles the reference max that may lag, the old state's rescale factor, P, the new sum,
and one word saying whether any row's reference moved:

```python
x = trace.load(ss.tiles[t, :]) * A  # A = scale log2 e
state = trace.load_state(sts.tiles[t])
m_old, l_old = state[0], state[1]
m_new = trace.lagging_max(m_old, trace.reduce_max(x, exact=True), 16.0)
alpha = trace.exp2(m_old - m_new, error=2e-3, over=(-inf, 0.0), exact_at_zero=True)
prob = trace.exp2(x - m_new.col(), error=2e-3, over=(-inf, 0.0))
l_new = alpha * l_old + trace.reduce_sum(prob, exact=True)
trace.store(ps.tiles[t, :], prob)
moved = trace.word_max(m_new - m_old)
trace.store_state(outs.tiles[t], [m_new, l_new, alpha], word=moved)
```

Each per-row value lives as lanes of one slot of the state tile (slot `k` of row group
`(h, q)` at `Dst` row `16 h + 4 q + (0, 2, 32, 34)[k]`); the reductions run in levels,
one pass over the row a level and row group. `word_max` reduces a per-row value to one
word (the largest over the rows) in the last slot, which a core reads as a word.

## Rules

Each rule is a pass; `dump()` prints what each one did.

| Rule | Kind | What it does |
|---|---|---|
| fold_negation | algebra | a negation disappears into the multiply-add that makes it or reads it (a sign bit) |
| fuse_mad | algebra | a product read only by an add becomes one `SFPMAD` (one rounding in place of two) |
| constant_registers | layout | the constants the rows load move into LReg 12 to 14 and then LReg 11, most words saved first; equal constants share one; LReg 11 is set in the unit and restored to -1.0 (the library's value) after it |
| whole-unit constants | layout | step form: a constant is held in a register for the whole step when the cost model prices that cheaper than loading it once a group and level |
| hoist | schedule | the setup of those registers moves out of the rows, once a stream |
| templates | algebra | each transcendental takes the cheapest template meeting its stated error (above) |
| flash bias | algebra | the flash template's bias joins the argument's subtracted term: `p k + (639 - r k)` is one multiply-add, the per-row part computed once a row group |
| drop_copies | algebra | a move whose source is not written again before its reader disappears |
| common_values | algebra | an instruction computed again from the same operands is computed once for all its readers; constants are keyed by their bits (an `SFPLOADI` itself is not shared: that would lengthen register lives) |
| negated_zero | algebra | the -0.0 a row loads becomes one load |
| resident_constants | layout | two-word polynomial constants stay in free LRegs for the whole tile when 8 work registers allow |
| interleave | schedule | in the row and step forms, independent instructions fill the multiply-add and `SFPSWAP` stall slots under a live-register limit (retried with a tighter limit, down to 3; the limit counts live values as the allocator does); a constant load is placed at its reader; a constant register's setup (`SFPLOADI` into LReg 0, `SFPCONFIG`) stays whole |
| matrix_sum | algebra | step form: a row sum becomes products of the row with a tile of ones on the matrix unit (P packed and unpacked again, then a second part of the step adds it), when its cost (`matrix_cost` of the words, plus the second part) is lower than the vector unit's |
| max_macro | schedule | step form: a run of loads each followed by a max swap into one accumulator becomes `SFPLOADMACRO` + `SFPNOP` a load, the loaded register alternating between two of LReg 0 .. 3; the swap is an instruction template in a slot no word of the unit writes as `VD` |
| store_step | schedule | stream form: the `INCRWC` after a row's last store becomes that store's address mode (`ADDR_MOD_6`, `Dst` + 4) |
| per_row_prepass | algebra | stream form: a per-row operand's term computed once a block in a pass first, when the cost model prices the pass cheaper than the words it saves |
| max_scale | algebra | step form: a positive scale moves out of a max (exact) |
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
| exp / exp2: "no template meets relative error E over [lo, hi]" | no template's bound meets the stated error over that range |
| step form: 1 to 3 row tiles, one row store and one state store, one row tensor, one state tile, exact sums or maxima, a slot read after the level that writes it, "the step's values do not fit the 8 work registers" | what the levels and the state layout are built for |
| "a state tile holds 4 per-row values, or 3 and a word" | 4 slots a row group |
| row form: one reduction, `reduce_sum(..., exact=True)` along the row; one `.col()`; one loaded row; `.row()` of a resident row only | what the stat and apply phases are built for |
| load macro: "the row computes with N instructions (M MAD-class); a macro arms at most one a sub-unit" | a macro has one slot each for the simple, MAD and round sub-units and one for the store; SiLU (45 instructions) and the gate (46) do not fit, and run as plain rows |
| load macro: a lane-flag instruction, a store of a value other than the last, a store more than 7 cycles after its load, more than 4 templates | the macro's slots run unguarded, its delays are 3 bits, the unit holds 4 templates |
| matrix_sum: at LoFi and HiFi2; "the step without its sums does not fit the registers"; "kept on the vector unit (cheaper)" | P sits in SrcB, whose low bits only phases 2 and 3 multiply; the second build must still allocate |
| max_macro: "a multiply-add just before the run"; "no second register of LReg 0 .. 3 free to alternate"; "no template free (4 a unit, less those the code writes as VD)" | the template's swap reads its registers unstalled; the loaded register is LReg 0 .. 3; a word whose `VD` is 12 + k stores template k |
| emit: "macro k runs template t, which the init does not store"; "... writes VD 12 + t in the tile's code" | the macro would run whatever template an earlier body left, or, from the second tile on, the word that overwrote it |
| emit: "buffer B is pushed by X and Y" | a buffer's received counter takes one producer's count: a second producer's pages never reach the wait (a state kept in place is neither pushed nor popped) |

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
stored vector followed back to the loads it reads). `emit` adds two that refuse:
`macro-templates` (each template a macro runs is the last word of the init to write its
`VD` 12 + k, and no word of the tile's code writes it: the tile's code repeats, and the
card's init runs once a stream) and `one-producer` (each buffer the unpack CPU waits for
or the pack CPU pushes has one producer). The rules no check covers yet are printed as
"not checked".

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
| an instruction other than `SFPNOP` right after an `SFPSWAP` | 1 stall (the ISA's statement) |

The matrix unit's words are priced by `unit.matrix_cost(words)` (probe 11 and the
rows of section 8): one word a cycle for `MVMUL`, `ELWMUL`, `ELWADD`, `MOVA2D`,
`MOVB2D`; a word adding into a `Dst` block waits until 6 cycles after that block's last
write (K words apart: max(1, 6 / K) cycles a word; 6 or more blocks rotating: 1); 8
cycles from the last word to a sync. The moves overwrite and do not wait.
`unit.rotate_blocks(words)` is the schedule rule that follows: each block's words keep
their order (its sum rounds the same) and a block is added to again as late as the
order allows. The library's `matmul_block` and `matmul_block_no_mop` issue the same
words (`matrix.tile_products`): per fidelity phase 16 `MVMUL`, `Dst` blocks 0 .. 7 then
0 .. 7 again, then a `SETRWC` releasing a Src bank; `matrix.block_cost` prices them,
41.0 cycles for one tile product at HiFi2, 73.0 at HiFi4, 24.0 at LoFi, and down to
33.5 a tile and K step at HiFi2 in a 4 x 2 block of 4 K steps.

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
| row norm (a row of 2 tiles), first rules | 753.0 | 771.5 | -2.4% |
| row norm, with interleave and common_values | 677.0 | 689.5 | -1.8% |
| SiLU, resident constants | 1533.6 | 1532 | +0.1% |
| gate, resident constants | 1597.6 | 1597 | +0.0% |
| exp step, accurate (one tile, a per-row operand) | 1113.5 | 1115 | -0.1% |
| exp step at 2e-3 (flash) | 443.2 | 444 | -0.2% |
| softmax step at 2e-3 (3 score tiles and the state) | 2350.0 | 2369 | -0.8% |
| softmax step, accurate | 4950.0 | 5630 | -12% |
| softmax step, + constants held for the step | 2291.0 | 2311 | -0.9% |
| softmax step, + LReg 11 a constant register | 2289.0 | 2308 | -0.8% |
| softmax step, + interleave filling the stall slots | 2015.0 | 2034 | -0.9% |
| softmax step, + max macro | 1939.0 | 2045 | -5.2% |
| softmax step, + row sums on the matrix unit: first part | 1659.0 | 1761 | -5.8% |
| the same, second part (from the products' issue) | 246.0 | 366 | -33% |
| exp step at 2e-3, + the store's address mode | 401.0 | 402 | -0.2% |

The accurate softmax step is 26 KB of straight-line code; the 12% is not explained by
the price list (a candidate, not measured: instruction fetch past what the core
holds). The max macro (the hand body's form) saved nothing on the card where the price
list gives it 76 cycles: its 88 load-and-max pairs cost about 3 cycles each either way,
not explained. The second part of the matrix row sums waits for the unpacker of P (a
second copy of P through a buffer), which the price list does not see. The emitted tile
product measured the same span as the library call (1147.0 and 1145.5 cycles a unit,
both bound by the reader's tile reads, not by the 34 words).

The cost of a stream, row or step program covers the vector unit only. A whole unit also pays the copies into `Dst` and
the pack, which the card shows to be most of an add or a scale.

## The gate

An emitted body passes when it computes the same expression as the hand body it
replaces, within a stated tolerance; bit equality is not a goal. The tolerance is the
largest relative error against float64 of the same fp32 inputs, fixed before the run:
2^-23 for one fp32 rounding (add, scale), 1e-6 for SiLU and the gate, 1e-5 for the
norm (a sum of 64 squares, a 23-bit reciprocal square root, two products), and the
error the program states where it states one (2e-3 on P, alpha and l for the softmax
step, as the hand body's exp delivers). The hand bodies' own errors are reported
against the same float64 reference.

For tolerances, not for bit checks: the matrix unit's sum is not one rounding of the
exact sum, and the packer's fp32 to bf16 at round-to-nearest sends ties to the larger
magnitude. A tile product of bf16 operands at HiFi2 is held to 2^-4 against float64
(SrcB's last mantissa bit is multiplied only in phases 2 and 3: about 2^-8 a term).
