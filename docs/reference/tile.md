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
stated tolerance ([tile_rules](../../experiments/language/tile_rules/README.md)). The matrix
unit's words of a tile product, emitted from the library's no-MOP order, gave the
library call's result bit for bit, and a body with tile products inside it
(`silu(a @ w1) * (a @ w3)`) ran within its tolerance.

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
| `unit_program(ir, dest="fp32", load_macro=False, rows=False, fidelity="HiFi4")` | the stream form: one store a tile, of an expression over up to 3 whole input tiles or tile products (`Dst` tiles 0, 1, 2). `load_macro=True` adds the load-macro rule (below); `rows=True` the band (below); `fidelity` the products' (below) |
| `rows.row_program(ir, width, rows_tile, cols_tile, rows=False)` | the row form: a row of `width` tiles (1 or 2) in `Dst` tiles 0 .. width - 1, a per-row value in `Dst` tile `rows_tile`, a per-column operand in `Dst` tile `cols_tile`; `rows=True` the band |
| `steps.step_program(ir, width, state_tile, fidelity="HiFi4")` | the step form: a row of `width` tiles (1 to 3) in `Dst` tiles 0 .. width - 1, stored over them, and a state tile of per-row values in `Dst` tile `state_tile`, read and written in place; `fidelity` is the matrix unit's, for a row sum the cost model gives it |
| `row_operand(row, width)` | the per-column operand tile `.row()` reads, placed from a plain row of 32 `width` values (1024 values in face order, written once by the host) |
| `matrix.product_program(name, fidelity)` | one tile product with the math CPU's words emitted (the library's unpack and `mm_no_mop_init_short` are the caller's) |
| `matrix.tile_products(fidelity, ct_dim, rt_dim, dst)` | the words of one `matmul_block_no_mop` call in issue order, each with its `Dst` block |
| `matrix.block_cost(fidelity, ct_dim, rt_dim, kt_dim=1)` | the priced cycles and words of `kt_dim` K steps of `matmul_block` (the MOP form issues the same words as the no-MOP form) |
| `emit(unit)` | the C++ header; `UnitError` if a word does not encode, a check refuses (below) |
| `unit.dump()` | the program as text: setup, loop nests, each pass and what it did, each check and its verdict, the rules not checked, the cost on each CPU |
| `unit.cost(cpu, rows=32)` | `(instructions, cycles a unit, terms)` on `"unpack"`, `"math"` or `"pack"`; `rows`: the live rows a band body is priced at |

## The band: the live rows only

A decode step holds a token in a few rows of each tile; the rest of the rows hold
nothing anyone reads. With `rows=True` the emitted body is `<name>(dst, rows)`: `rows`
is a runtime argument (a word of the caller's command, 1 to 32; 0 or more than 32
means the whole tile) and the body runs the bands of 4 rows that hold one. Rows past
the last band are left as they were in `Dst`. Without `rows` the code is the whole
tile's, unchanged.

- **Stream form.** Faces 0 and 1 (rows 0 to 15) run in a loop of two, faces 2 and 3 in
  a loop of two that runs past 16 rows. A face records its first row pass into the
  replay buffer as it runs, then a loop the core runs replays the face's other bands
  (`(rows + 3) / 4` bands in faces 0 and 1, the rest in 2 and 3). A band is 2 row
  passes, or 1 when a pass covers 4 rows (a load macro of two row steps, a per-row
  operand's 4-row block). Each face's `Dst` is set from the face's base by the two
  `SETRWC` that end the face, so the rows a face ran do not move the next face. The
  per-row prepass (a per-row operand's term once a block) runs the whole tile.
- **Row form.** Each instruction belongs to the first band whose stored value it
  feeds (followed back from the stores through the registers); bands are 8 rows, the
  two row groups the schedule interleaves. A run of instructions of one band runs
  under `if (rows > first row)`; a run that ends in a multiply-add ends in an
  `SFPNOP` (the run after it may not run).
- **The cost.** `unit.cost("math", rows)` prices the body at `rows` live rows: the
  words of the bands that run; the loop of bands and the tests of the rows are priced
  at nothing, as the card ran them (the core's loop and test run ahead of the words it
  hands the unit). At 32 rows a band body is priced as the whole tile.

On the card (`experiments/language/tile_rules`, bodies 24 to 26, math cycles a tile less the
sync; the estimate in brackets):

| Body | 1 row | 9 rows | 17 rows | 32 rows | Whole tile |
|---|---:|---:|---:|---:|---:|
| add, load macro | 35.0 (24.2) | 48.5 (40.6) | 71.0 (67.0) | 93.5 (91.6) | 93.0 (91.6) |
| silu(a) b | 220.0 (210.3) | 603.0 (587.3) | 992.0 (974.0) | 1568.0 (1539.6) | not run (1539.6) |
| norm, a row of 2 tiles | 183.0 (173) | 349.0 (337) | 519.0 (509) | 682.0 (673) | 689.5 (673) |

Every live band equals the whole tile's result within the case's tolerance, and the
rows past the band are the input's bit for bit. At 32 rows the band costs what the
whole tile does (add 93.5 against 93.0; the norm's 682 is under its whole tile's).
The estimate is low by about 10 cycles a body at every row count (add 35.0 against
24.2 at one row, the norm 10 at each), a cost the model does not have.

| Refused under the band | Why |
|---|---|
| load macro: "a macro turn of N row steps; a band is 2" | a macro's register turn would cross a band; the row stays plain |
| row form: "the row form's band takes no lane flags" | a flag set in a run that does not run would change the lanes of the next |

## A product in the trace

`a @ w` of two tiles the unit reads (`trace.load(x.tiles[t])` or a resident tile) is an
operand of the stream form like a load: its result is a `Dst` tile the vector code
after it reads, so `silu(a @ w1) * (a @ w3)` is one body, with no pack and no unpack
between the products and the gate.

- **The words.** Each product is the library's no-MOP product at the unit's
  `fidelity` (`matrix.tile_products`, one K step of one 32 x 32 tile a side: 16 `MVMUL`
  a fidelity phase and two `SETRWC`), after the RISC-V's write of its tile's `Dst`
  base, in operand order, before the vector code. The vector code's first word is the
  stall of the vector unit on the matrix unit (`STALLWAIT`), so it reads the products
  complete. A product stored as it is (`y = a @ w`) emits the product's words alone,
  the words `matrix.product_program` emits.
- **The caller.** The unpack CPU unpacks each product's operands in the same order,
  `llk_unpack_AB_matmul(a, w, ...)` (a into SrcB, w into SrcA: the plain placement;
  the `km.srcb` / `km.srca` markers of [placement](formats.md#placement) are refused in
  a traced tile function), after
  `mm_no_mop_init_short(a, w, ...)` once and the body's `_init()`; the compute kernel's
  `math_fidelity` is the unit's. `Dst` is clear when the body starts (the pack's release
  clears it): a product adds into its tile. `unit.products` names each product's tile
  and operands (`UnitProduct(dst, a, w)`); `unit.operands` holds `None` at a product's
  tile, the name of the input copied into `Dst` at the others; the header names the
  products in its comment.
- **The cost.** The products' words are priced with the vector code
  (`matrix_cost`: a word a cycle, the last product's result 8 cycles to the stall).
- **The band.** With `rows=True` the vector code runs the live rows; the products run
  the whole tile.

On the card ([tile_rules](../../experiments/language/tile_rules/README.md), bf16 operands at
HiFi4, cycles a unit less the sync): silu(a @ w1) 1579 against 1548.6 estimated, within
3.0e-4 of float64; silu(a @ w1) (a @ w3) 1707 against 1678.6, within 4.6e-4, and 356
against 349.3 with the band at one live row. The gate less its vector code as that ran
alone (1568) is 139.0 cycles: the two products' estimate.

| Refused | Why |
|---|---|
| "a product after vector code: the left operand of `@` is a silu, computed on the vector unit" (or a product, its result in `Dst`) | the matrix unit reads its operands unpacked from a buffer into SrcA and SrcB: the value has to be packed and given to a second body |
| "the unpack feeds a product one 32 x 32 tile a side (x.tiles[t]); the left operand of `@` is a row block" | one unpack call feeds one tile a side; a K loop over a row is not lowered |
| "N products: Dst holds 4 operand tiles of a body" | each product's result stays in its own `Dst` tile for the vector code |

## What a tile function may contain

**Stream form.** Each tile of the output is an expression of the same tile of up to
four inputs: `+`, `-`, `*`, `/`, unary minus, `abs`, Python number constants,
`trace.sigmoid`, `trace.silu` (`x * sigmoid(x)`), `trace.rsqrt`, `trace.sqrt`
(`x * rsqrt(x)`, 0 at 0), `trace.reciprocal` (the unit's estimate and two Newton
steps, as tt-metal's `sfpu_reciprocal_iter<2>`; `x / y` is `x * reciprocal(y)`),
`trace.maximum` and `trace.minimum` (one `SFPSWAP` on copies: its larger or its
smaller result). The four inputs sit in `Dst` tiles 0 .. 3, the half of the fp32 `Dst`
a call holds. On the card ([tile lines](../../experiments/language/tile_lines/README.md), bf16
inputs): `x / y` within 1.6e-7 of float64, `sqrt` 9.5e-8, `reciprocal` 1.1e-7, `minimum`
and `abs` exact, four inputs `x * y + z * w` 6.0e-8 (largest relative errors); none is
guarded outside its domain ([hazards](hazards.md#a-tile-functions-sqrt-reciprocal-and-division-outside-their-domain)). Inputs are fp32 tensors or tensors
of the B family (bf16, bfp8_b, bfp4_b, bfp2_b), which run on an fp32 `Dst`, or tensors
of the A family (fp16, bfp8, bfp4, bfp2), which run on a 16-bit `Dst`: the same words,
each `SFPLOAD` and `SFPSTORE` in the FP16A mode, the vector code in fp32 between them
([tile formats, the A family in a tile function](formats.md#the-a-family-in-a-tile-function)).
The two families are not mixed in one function. Operands of different formats of one
family are taken: the call sets SrcA's format again before each unpack into it (a
product's W, each copied operand) when those are not of one format. Constants become `SFPLOADI` immediates, or one of the constant registers
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

Below the flash template's range the argument is clamped (`max(u, 512)` on `u = a k +
639`), so the result is under 2^-126 and finite for any argument down to minus
infinity; the card reads 0 there (one core, arguments -127.5 to -1e4). The clamp's
`SFPSWAP` takes a copy of 512 in a work register as its `VD`: the init leaves
`LaneConfig`'s back door open, and a word with a `VD` of 12 or more stores an
instruction template and runs nothing (a clamp against LReg 12 gave NaN at -127.5 and
1.76e38 at -128 on the card).

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
| common_values | algebra | an instruction computed again from the same operands is computed once for all its readers; constants are keyed by their bits (an `SFPLOADI` itself is not shared: that would lengthen register lives); a load no instruction reads is dropped, and `SFPCONFIG` reads LReg 0 unless it takes `Imm16` |
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
| band | schedule | with `rows=True`: the face loop or the row groups run for the live rows only (above) |
| per-row lanes | layout | a per-row value lives as lanes of one tile: slot `(h, q)` of the rows tile, lane `l` holding row `16 h + 4 q + l / 8`, the same in all 8 lanes of its lane row; the sum along a row is a multiply-add chain then three lane rotations (`SFPSHFT2`, 1, 2 and 4 lanes) |
| broadcast | layout | `.col()` reads the per-row slot once a half-row; `.row()` reads the per-column operand once a vector column; no other broadcast is built |

## What is refused, and why

| Refusal | Why |
|---|---|
| "the unit program takes one store" | one output tile a unit is what the loop nest and the pack calls are built for |
| "the unit program reads at most 4 operand tiles" | the operands sit in `Dst` tiles 0 .. 3, the half of the fp32 `Dst` a call holds |
| "the unit program reads whole-tile operands" | a slice or a broadcast operand needs the row form |
| "vector function f is not built" / "vector operation k is not built" | only the functions and operations listed above have a lowering (tanh, exp2 outside a template and the comparisons other than `>`, `<`, `==`, `!=` have none) |
| "the unit program takes no runtime scalars" | every immediate is built into the header; a run-time word would need building on the core |
| "a tile function takes tiles of float32 or the B family ..., or of the A family ...; given float16 with float32" | an fp32 `Dst` reads the A family's datums 2^-15 off, and a 16-bit `Dst` holds no float32 |
| "takes tiles of float16 (the A family, a 16-bit Dst) and is called on a row" | the row form and the step form keep each row's value, its sums and the state word in `Dst` between their levels, which a 16-bit `Dst` would cut to fp16 (the word is an int32) |
| the load macro: "refused: fp16a Dst" (a pass's record, not a refusal) | the macro's load and store are built for the fp32 `Dst`; an A-family unit runs its rows plain |
| row form: "rows of 1 or 2 tiles" | the per-column operand tile holds two tiles of columns |
| exp / exp2: "no template meets relative error E over [lo, hi]" | no template's bound meets the stated error over that range |
| step form: 1 to 3 row tiles, one row store and one state store, one row tensor, one state tile, exact sums or maxima, a slot read after the level that writes it, "the step's values do not fit the 8 work registers" | what the levels and the state layout are built for |
| "a state tile holds 4 per-row values, or 3 and a word" | 4 slots a row group |
| row form: one reduction, `reduce_sum(..., exact=True)` along the row; one `.col()`; one loaded row; `.row()` of a resident row only | what the stat and apply phases are built for |
| load macro: "the row computes with N instructions (M MAD-class); a macro arms at most one a sub-unit" | a macro has one slot each for the simple, MAD and round sub-units and one for the store; SiLU (45 instructions) and the gate (46) do not fit, and run as plain rows |
| load macro: a lane-flag instruction, a store of a value other than the last, a store more than 7 cycles after its load, more than 4 templates | the macro's slots run unguarded, its delays are 3 bits, the unit holds 4 templates |
| load macro: "N rows a macro turn do not divide a face" | a face is 16 rows; the rows stay plain (`abs(x)` alone: a turn of 3) |
| matrix_sum: at LoFi and HiFi2; "the step without its sums does not fit the registers"; "kept on the vector unit (cheaper)" | P sits in SrcB, whose low bits only phases 2 and 3 multiply; the second build must still allocate |
| max_macro: "a multiply-add just before the run"; "no second register of LReg 0 .. 3 free to alternate"; "no template free (4 a unit, less those the code writes as VD)" | the template's swap reads its registers unstalled; the loaded register is LReg 0 .. 3; a word whose `VD` is 12 + k stores template k |
| emit: "macro k runs template t, which the init does not store"; "... writes VD 12 + t in the tile's code" | the macro would run whatever template an earlier body left, or, from the second tile on, the word that overwrote it |
| emit: "W (...) has VD v: with the back door open it stores template v - 12 and runs nothing" | any vector word but `SFPCONFIG` and `SFPLOADMACRO` with a `VD` of 12 or more is a template store while `LaneConfig`'s back door is open (the ISA's `SFPCONFIG` page); only the init's template stores may have one |
| emit: "buffer B is pushed by X and Y" | a buffer's received counter takes one producer's count: a second producer's pages never reach the wait (a state kept in place is neither pushed nor popped) |

What the limits above would take, where a user would write the line: a slice or a
broadcast operand in the whole-tile form (`x * w.row()` in the stream form) needs the
row form's per-column load inside the stream form, about 100 lines; two stores of one
call need a second result tile in `Dst` and a second pack, about 60 lines; `>=` and
`<=` are the swapped `SFPGT` with its flags inverted, about 10 lines; tanh needs
tt-metal's fp32 polynomial lowered and its error checked, about 40 lines.

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
stored vector followed back to the loads it reads). `emit` adds three that refuse:
`macro-templates` (each template a macro runs is the last word of the init to write its
`VD` 12 + k, and no word of the tile's code writes it: the tile's code repeats, and the
card's init runs once a stream), `back-door` (no other word has a `VD` of 12 or more)
and `one-producer` (each buffer the unpack CPU waits for
or the pack CPU pushes has one producer). The rules no check covers yet are printed as
"not checked".

## The cost printed

`cost(cpu)` adds up a price for each instruction, each loop pass and each stall it
predicts; the calls (`copy_tile`, `pack_tile`, the syncs) are listed and not priced.
Every price names the one-core measurement it comes from
([tile_probes](../../experiments/core/tile_probes/README.md), section 8):

| Term | Cycles |
|---|---|
| a word pushed | 1 |
| a word replayed | 1.004 |
| a pass of a loop the core runs | N + 2 |
| a pass of a band's loop, a band's test of the rows | 0 (the band on the card, above) |
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
| exp2 at 2e-3 (one tile), the clamp against LReg 12 | 478.1 | 478.0 | +0.0% |
| the same, the clamp's `VD` a work register | 510.2 | 511.0 | -0.2% |
| softmax step, the clamp's `VD` a work register: first part | 1777.0 | 1878.5 | -5.4% |
| the same, second part | 246.0 | 368 | -33% |
| `a * rsqrt(s.col())` | 487.4 | 490.0 | -0.5% |
| `sigmoid(a + s.col())` | 1453.5 | 1454.5 | -0.1% |

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
