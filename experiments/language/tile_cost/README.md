# Tile cost: what one tile function costs on the card, and why

Question: for one compute core of a Blackhole p150a, what do the unpack, math and pack
CPUs and the shared `Dst` spend on one tile function's cycle, which of those add up and
which overlap, and can a short rule built from measured rows predict a published
program's time from its compiler cost line?

`docs/reference/tile.md` prices only the vector unit's words (its "cycles a tile"). The
copy into `Dst`, the pack, the synchronisations and a command's own cost are listed
there and not priced. This experiment measures them.

## How it runs

| part | file |
|---|---|
| compute: one cycle of acquire, copies, body, commit, wait, packs, release; matrix products; format changes; a clock spin (`PB_MODE` 0, 1, 2, 4) | `kernels/compute.cpp` |
| reader (RISCV_1) and writer (RISCV_0): fill the input buffers, write the info page | `kernels/reader.cpp`, `kernels/writer.cpp` |
| layout of the info page, wall clock and counter reads | `kernels/cost.hpp` |
| driver: builds on a mock device, runs on one card core, prints per-CPU spans | `probe.py` |
| command-level probes (event microseconds): a stream of N tiles on C cores; loops that wait for their reads; strided blocks and resident fields; the shift's product loops | `scale.py`, `waits.py`, `taps.py` |
| the cost rule, checked against 18 published programs | `model.py` |

```
python probe.py list                      # the sections
python probe.py build --only copy_bf16_g4 # mock device: the kernels build
python probe.py card --only sync copy_bf16_g4 --save out.json   # one core
python scale.py --cores 1 8 36 --tiles 4 8 16 32 64 --repeats 5 [--dtype float32]
python waits.py --repeats 5 [--only loops strided apply]
python taps.py --repeats 5
python model.py                           # the table below, off the card
```

- **Spans.** The three compute CPUs start together (a barrier keyed by a fresh nonce
  each launch, so stale words in the info page cannot satisfy it). Each reads the wall
  clock (AICLK cycles) around its own part. The pack CPU then reads tt-llk's
  performance counters (85 values: unpacker and packer busy, FPU and SFPU active,
  thread stalls). The empty span (one `tensix_sync`) is 10 cycles and is taken off.
- **Repeats.** A section runs 5 times in one launch; the first is dropped and the
  tables give the median and the min .. max of the other 4.
- **Clock.** sysfs `tt_aiclk` reads 800 MHz on an unopened card (the idle clock) and
  1,350 MHz in every sample taken with the session open during a spin. A 100 M and a
  300 M cycle spin took 0.0982 s and 0.2444 s of host time (1,368 MHz by the
  difference). The throttle word the library's `matmul_block` reads
  (`MEM_L1_ARC_FW_SCRATCH`, L1 address 16) read 0 in every section. Cycles are
  converted at 1,350 MHz: 1 us = 1,350 cycles, 1 cycle = 0.74 ns.
- **Commands.** A command's time is the event's microseconds. Tables give the median of
  the commands after the first of each handler: the first command is 0.1 to 1.0 us
  longer on a command under 10 us, and it is the number the READMEs print.

## The machine, in words

One tile function's cycle with a half-sync `Dst` (the library's and the compiler's mode):

- The **unpack CPU** waits for a page of the input buffer (`cb_wait_front`), the
  unpacker writes the tile into SrcA (and SrcB), the CPU pops the page. It runs ahead
  of the math CPU by one SrcA bank.
- The **math CPU** acquires a `Dst` half (a `SEMWAIT` on the MATH_PACK semaphore, which
  stalls while the semaphore is at its maximum, 2 in half sync), copies SrcA into `Dst`
  (`STALLWAIT` on SrcA valid, then 8 `MOVA2D`), issues the body's vector words one after
  another on the one vector unit, and commits (posts MATH_PACK once the vector unit is
  idle, then flips the half).
- The **pack CPU** waits for the post (`SEMWAIT`, stalls while zero), packs the tile into
  the output buffer, pushes it (`cb_push_back` stores the buffer's received count from
  the pack CPU's own count), and releases the half (waits for the packer, clears the
  half with `ZEROACC`, takes the semaphore back).

Sources read: `tt_llk_blackhole/common/inc/cmath_common.h:233-310`,
`tt_llk_blackhole/llk_lib/llk_pack_common.h:25-80`,
`tt_metal/hw/ckernels/blackhole/metal/llk_io/llk_io_pack.h:61-71`, `llk_io_unpack.h:31-33`.

What is serial and what overlaps (all measured):

- The pack of cycle i hides behind the unpack of cycle i + 1: copy and pack of a bf16
  tile is 49.0 cycles against 45.2 (unpack alone) + 33.4 (pack alone) = 78.6 for the
  sum.
- A bf16 tile copied through SrcA hides behind the body: a loop is the body plus 34
  cycles, and about 30 more for each further tile copied.
- An fp32 tile unpacked straight into `Dst` does not hide: 126 to 130 cycles for every
  tile copied.
- The three CPUs pass a page along one after the other when one page is in flight; a
  second and third page in flight bring a page's turn down to the longest CPU's own
  work (see the table).

What synchronises, and what each costs:

| sync | cost |
|---|---|
| MATH_PACK: acquire + commit + wait + release with no work | 20.6 cycles a cycle (15.3 ns) |
| SrcA valid before a math word reading it | inside the unpack's 45 cycles |
| the buffer calls and the hand-over between the three CPUs, one page in flight | about 115 cycles over the 54.2-cycle bare copy-and-pack loop |
| `tensix_sync` with nothing issued | 10 cycles |

A buffer has one producer. The pack CPU's push stores its own running count as the
buffer's received count, so a reader's earlier push is overwritten and the unpack CPU
waits for ever. The tile compiler and the class compiler refuse this at build
(`dsl/tile/emit.py` `check_buffers`; `dsl/compile/pages/field_fills.py`;
`dsl/compile/tiles/bodies.py` `take_own`); a push through a variable name is not seen.

## The cost table

Cycles a tile at 1,350 MHz, one core, median (min .. max).

| format | unpack only | copy + pack | pack only | unpacker busy | packer busy |
|---|---|---|---|---|---|
| bf16 | 45.2 (44.8 .. 45.9) | 49.0 (48.6 .. 49.6) | 33.4 (33.3 .. 33.9) | 32.0 | 26.8 |
| bfp8_b | 45.2 (44.7 .. 45.2) | 49.0 (48.7 .. 49.7) | 35.3 (34.5 .. 35.4) | 30.1 | 27.8 |
| bfp4_b | 45.2 (44.6 .. 45.2) | 48.9 (48.2 .. 49.1) | 34.1 (34.0 .. 34.7) | 30.1 | 27.5 |
| fp32 | 45.8 (45.2 .. 46.0) | 55.3 (54.9 .. 55.8) | 49.2 (49.0 .. 49.4) | 42.1 | 42.5 |
| fp32 straight to `Dst` | 125.8 (125.5 .. 126.5) | 133.8 (133.2 .. 134.1) | not run | 47.3 | not run |

The format does not change the unpack window, only the busy count. One tile a cycle
(G = 1) of bf16 copy + pack is 54.2 (54.0 .. 54.4).

| row | cycles | ns |
|---|---|---|
| acquire + commit + wait + release, no work | 20.6 (20.5 .. 20.6) | 15.3 |
| a page's turn through the three CPUs, bf16, 1 page in flight | 169.0 (168.8 .. 171.7) | 125 |
| the same, fp32, 1 page | 201.9 (201.5 .. 205.5) | 150 |
| the same, bf16, 2 pages / 3 pages in flight | 86.6 / 76.2 | 64 / 56 |
| loop of one bf16 tile, no format change | 60.9 | 45 |
| unpack reconfig to the same format | +41.3 | 31 |
| pack reconfig to the same format | +121.2 | 90 |
| both reconfigs, same format | +120.9 (they overlap) | 90 |
| a format change, bf16 to bfp8_b or fp32, both | +131 | 97 |

Vector bodies. "Body" is the loop on fp32 tiles straight into `Dst` less the copy's
share. "Est." is tile.md's price for the same unit program.

| op | loop, fp32 to `Dst` | loop, bf16 via SrcA | body | est. | error |
|---|---|---|---|---|---|
| add (2 copies) | 425.8 | 227 | 167 | 150.4 | +11% |
| mul (2) | 425.9 | not run | 167 | 150.4 | +11% |
| square | 274.6 | 176 | 142 | 118 | +20% |
| exp (accurate) | 1122.6 | 1026 | 990 | 998 | -0.8% |
| exp2 | 1188 | not run | 1056 | 1063 | -0.7% |
| exp2 at error 2e-3 | 638 | not run | 505 | 510 | -1% |
| reciprocal | 500 | 404 | 368 | 375 | -2% |
| rsqrt | 787 | not run | 655 | 632 | +4% |
| sqrt | 851 | not run | 718 | 696 | +3% |
| tanh | 2587 | not run | 2454 | 2464 | -0.4% |
| sigmoid | 1508 | not run | 1375 | 1384 | -0.6% |
| SiLU | 1630 | 1531 | 1498 | 1476 | +1.5% |
| erf | 2592 | not run | 2460 | 2470 | -0.4% |
| maximum (2) | 486 | not run | 228 | 247 | -8% |
| compare-select (2) | 494 | 295 | 235 | 247 | -5% |
| row sum, band of 4 tiles | 959 | 562 | 449 | 465 | -3% |
| row max, band of 4 tiles | 1198 | not run | 687 | 705 | -3% |

Thirteen of the 17 bodies agree with tile.md's price within 4%. The small ones (square,
add, mul) are 11 to 20% above it: 34 cycles of loop overhead is a large share of them.
The spread of the 4 repeats is under 1.2% in every row.

Matrix products, the whole loop divided by the products it holds (cycles a product):

| fidelity | 1x1x1 | 1x1x4 | 2x2x4 | FPU words a product (counter) |
|---|---|---|---|---|
| LoFi | 59.2 | 32.4 | 19.3 | 16 |
| HiFi2 | 60.3 | 40.8 | 36.4 | 32 |
| HiFi4 | 90.4 | 72.8 | 68.3 | 64 |

The operand formats (bf16, bfp8_b, bfp4_b) do not change a product: HiFi2 1x1x4 is 40.8,
41.2, 41.0, 41.1 for bf16 x bf16, bf16 x bfp8_b, bf16 x bfp4_b, bfp8_b x bfp8_b.
tile.md's 41.0 and 73.0 agree with the HiFi2 and HiFi4 loops; its LoFi 24.0 does not
(32.4 here: the loop's pack and synchronisation are in it). A 4x2 block was not run.

A command's own terms (event microseconds):

| term | value |
|---|---|
| fixed term of a command, 1 / 8 / 36 cores (a line through 4 .. 64 tiles, at 0 tiles) | 3.7 / 3.9 / 5.0 us |
| a core's data movement, read and write together | 28 to 30 GB/s |
| a loop that waits for its own read (1 tile stream in, 1 out), 1 core | 1.06 us |
| a loop of products that waits for its reads, 1 core / 36 cores | 1.3 us / 1.5 to 2.5 us |
| a block of 2 tiles at every other tile row (contiguous: 0.28) | 0.51 us |
| a resident fp32 field read by `.col()`: fill / first field a tile / second a tile | 1.06 us / 152 cycles (112 ns) / 283 cycles (209 ns) |
| a field held in `Dst` for the stream: its fill and two loads, a command (per-channel rows 0.94, the apply 0.50) | 0.72 us |

## The cost rule and the published programs

The rule (`model.py`), with the rows above:

1. A loop of a tile function over a stream: body + 34 + 30 (copies - 1) cycles for bf16
   or fp32 through SrcA; body + 132 + 130 (copies - 1) for fp32 straight into `Dst`.
   Each resident field read by `.col()` or `.row()` adds 152 (the first) or 283 (the
   second) cycles a tile and a fill of 1.06 us; a field held in `Dst` for the stream
   (the compiler's cost line says "held in DST") adds 0.72 us a command instead.
2. A stream: time = F + tiles x max(loop, bytes moved a tile / 29.5 GB/s), F = 3.7 / 3.9 /
   5.0 us on 1 / 8 / 36 cores.
3. Loops that each wait for their own reads (a convolution's taps): from the product-loop
   rows (1.3 us a loop on one core; on 36 cores 5.16, 7.66, 11.40, 17.38 us for 1, 2, 4, 8
   loops, then 1.5 us each). A loop whose reads go out in the move of the loop before
   (the compiler takes plain read loops one after another in one move) saves 0.08 us:
   the convolution's 15 such loops took it from 29.9 to 28.7 us, values word for word
   unchanged (2026-11-18, card 1, 1,350 MHz).
4. Blocks at every other tile row: 3.3 us + 0.51 us a block.
5. The chunk form (a row of more than 2 tiles): F + (body + passes x tiles x 126 + field
   tiles x 126 + passes x tiles / 2 x 20.6 cycles) / 1,350.
6. The body is the compiler's own printed cost line.

Predicted against measured (warm median of the commands after the first, 1,350 MHz).
"Free": no constant was fitted to that program. "Read back": the constants came from a
probe of the same shape in this folder, so the row checks the probe, not the rule.

| program | rule us | measured us | error | printed in its README (first command) | |
|---|---|---|---|---|---|
| SiLU, 8 cores x 8 tiles | 12.8 | 13.0 | -1.2% | 13.2 | free |
| GELU, sigmoid form | 14.7 | 14.9 | -1.1% | 15.2 | free |
| GELU, erf form, 16 tiles | 36.6 | 36.9 | -0.9% | 38.0 | free |
| residual add then SiLU | 13.6 | 14.1 | -3.6% | 14.2 | free |
| per-channel scale and shift (`.col()`), fields held | 6.5 | 6.9 | -6.5% | 7.9 | read back |
| per-channel (`.row()`), fields held | 6.5 | 6.9 | -6.5% | 7.1 | free |
| 3x3 shifts, 36 cores, 4 loops | 11.4 | 11.3 | +0.9% | 11.6 | read back |
| 3x3 convolution, 32 cores, 18 loops (15 in the move before) | 31.2 | 28.7 | +8.6% | 30.4 | read back |
| group norm squares, 36 cores | 5.6 | 5.4 | +2.9% | 6.6 | free |
| group norm column sums, 2 cores | 11.5 | 13.8 | -17.0% | 14.2 | miss |
| group norm lane sum, 2 cores | 4.4 | 4.2 | +3.8% | 4.3 | free |
| group norm apply, 2 cores, fields held | 12.9 | 12.4 | +3.6% | 14.0 | read back |
| attention scores, 8 cores | 5.1 | 5.4 | -4.6% | 6.5 | free |
| attention mask and scale | 5.6 | 5.8 | -3.2% | 6.0 | free |
| attention weighted sum | 5.0 | 6.1 | -17.9% | 6.3 | miss |
| row softmax (4 tiles), 8 cores | 14.2 | 16.6 | -14.7% | 16.6 | on the line |
| RMS norm (64 tiles a row), 4 cores, values held | 34.7 | 30.6 | +13.4% | 32.2 | free |
| layer norm (64 tiles a row), 4 cores, values held | 56.0 | 63.0 | -11.2% | 64.7 | miss |

Ten of 18 programs are within 5% and 16 within 15%. Two miss by more than 15%
(column sums, weighted sum); the softmax is at -14.7%, RMS norm at +13.4%, layer norm
at -11.2%, the convolution at +8.6% and per-channel rows and columns at -6.5% (the
held-field term is one constant fitted across rows and the apply). The norms' rows were measured with the
row's values tile held through the write pass (before: RMS norm 33.1 us, +4.2%; layer
norm 66.6 us, -16.4%); the convolution's with its read loops in one move (before: 29.9
us, +8.4%). Why the others miss is not isolated:

- Column sums (0.14 us over an iteration) and weighted sum (0.27 us over): the second
  stream's page read may not overlap the first's.
- Layer norm (7.0 us over; 10.9 before its values tile was held) and softmax (2.4 us
  over) are the chunk form with more than one reduction level; RMS norm, with one
  level, is +13.4%. The chunk passes may cost
  more than 126 cycles a tile a pass. Before the hold, two 128 KB fills at about 4 us
  each would have brought layer norm to -3.3% but put RMS norm at +17%, so fills alone
  did not explain it.

Two things the table also says about the printed cost lines. The compiler's softmax cost
line (12,211 cycles) is 9.0 us at 1,350 MHz against a 16.6 us command, and the layer
norm's (36,180 cycles) is 26.8 us against 63.0: the rest is the command's fixed term, the
fp32 copies of each pass and the synchronisations, which the line does not count. The
split-norm experiment's 13,468 cycles are 10.0 us at 1,350 MHz, not 16.8 us (that figure
used 800 MHz, the idle clock).

## The gap between emitted and hand code

- `experiments/language/tile_front`: the emitted SiLU body is 1,512 cycles a tile
  against `silu_tile`'s 1,730. The SiLU unit here is 1,498 cycles of body (1,531 with the
  bf16 copy), within 1% of the emitted figure. The 218-cycle difference matches the 84
  fewer words the emitted body runs (381 against 465), which are constant loads and
  `Dst` address bumps (inferred from those counts).
- Where emitted code costs more than the vector body, the table says what the extra is:
  one page in flight (169.0 cycles a turn against 86.6 with a second page); an fp32 tile
  copied into `Dst` (126 to 130 cycles each); a resident field copied into `Dst` every
  tile (152 and 283 cycles a tile: 12.4 of a group norm apply's 23.4 us, before a pushed
  call's `.col()` and `.row()` fields were held in `Dst`: 12.4 us now); a row read into
  `Dst` once a pass (126 cycles a tile a pass); a loop that waits for its read (1.06 to
  1.3 us a loop).
- The pack of a bf16 tile (33.4 cycles) and the unpack (45.2) cost the same in emitted
  and hand code: a hand body gains only where it overlaps what an emitted one chains.

## The bound

Least time for a list of ops on N tiles: F + N x max(unpack, math, pack, data movement)
with perfect overlap; F + N x (their sum) with none.

| program | bound | sum | measured |
|---|---|---|---|
| SiLU, 8 tiles, 8 cores | 12.8 us | 13.4 us | 13.0 us (at the bound) |
| GELU erf, 16 tiles | 36.2 us | 36.8 us | 36.9 us (at the sum) |
| RMS norm, 64-tile row | 18.8 us (the body alone) | 34.7 us (rule) | 30.6 us |
| layer norm, 64-tile row | 30.6 us (the body alone) | 56.0 us (rule) | 63.0 us |

Emitted elementwise and norm programs sit near the sum of their parts, not near the
longest part. Copy and pack alone: 49.0 cycles measured, 78.6 for the sum, 45.2 for the
longest CPU. A core's data movement (28 to 30 GB/s) bounds a stream of two inputs and one
output before the vector unit does when the body is under about 200 cycles.

## What the language cannot say

Nothing here is built; each row gives the card's price and what a user would write.

| thing | worth | what a user would write |
|---|---|---|
| a second page in flight between the three CPUs (rows in flight in the step form) | a turn falls from 169.0 to 86.6 cycles with 2 pages and 76.2 with 3 | the loop order already expresses it; the compiler would have to skew steps |
| the packer adding into L1 for a value kept through `Dst` between steps | 190 cycles (0.14 us) a 2-tile round trip through SrcA, 350 (0.26 us) to `Dst` | `self.o[i] += p @ v`; `self.f += self.g(xs, ws)` on an fp32 field already does it for a product |
| a macro exponential for an `exp2` of a stated error | 1,056 against 505 cycles a tile (0.78 against 0.37 us) | `km.exp2(s, error=2e-3, over=(lo, hi))`, taken when `over` ends at 127 or below |
| one loop over several streams as one accumulation | a convolution's 18 dependent loops cost 1.5 us each on 36 cores; one move over their reads saved only 0.08 us a loop, so the reads' round trip is not that cost | one Stream over a list of tile views |
| a short row kept in `Dst` across the chunk form's passes | 8 extra copies a row on the softmax: 1,008 cycles (0.75 us); a 64-tile row does not fit | not expressible |
| a buffer written by a mover and by compute | hangs a core; refused at build except when pushed through a variable | a refusal that also sees pushes through a variable |

## Not measured

- A 4x2 product block, and LoFi alone (24.0 estimated, 32.4 in the loop).
- Unpack of a float16 tile; the pack of an fp32 tile unpacked straight into `Dst`.
- The vector-unit mutex; one `cb_wait_front` or `cb_reserve_back` alone (only the four
  calls with the hand-over, about 115 cycles).
- A page's turn in an unpack-to-`Dst` buffer: that hand-written kernel hung a core once
  and is not in this folder. The turn rows above were measured with a one-page and a
  multi-page buffer on the pack CPU's own push; those kernels are not kept either.
- Why a handler's first command is 0.1 to 1.0 us longer.
