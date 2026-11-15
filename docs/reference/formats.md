---
title: Tile formats
summary: The card's native tile formats and their bytes, the two exponent families and the rule of one family a kernel, what the unpacker and the packer do with each, the host's block-float encode against the card's, the significand bits a product multiplies on each side at each fidelity, and the placement of a product's operands (km.srcb, km.srca) with its measured cost.
tags:
  - reference
  - compute
---

# Tile formats

A tile is 32 x 32 values in four 16 x 16 faces. Its format is what a buffer, a tensor and
a field of a class are declared in; the unpacker turns it into the source registers'
form (SrcA and SrcB, 19 bits a datum) or into `Dst`, and the packer turns `Dst` back.
Every number on this page was measured on one p150a
([experiments/core/formats](../../experiments/core/formats/README.md)), unless it says
otherwise.

## The formats

| Format (`dtype`) | Class name | Family | Significand bits | Bytes a tile | Host values |
| --- | --- | --- | --- | --- | --- |
| `float32` | `km.fp32` | either | 24 (11 in a product, tf32) | 4096 | float32 |
| `bfloat16` | `km.bf16` | B | 8 | 2048 | float32 |
| `bfp8_b` (`bfloat8_b`) | `km.bfp8_b` | B | 7, a shared exponent | 1088 | float32 |
| `bfp4_b` (`bfloat4_b`) | `km.bfp4_b` | B | 3, a shared exponent | 576 | float32 |
| `bfp2_b` | `km.bfp2_b` | B | 1, a shared exponent | 320 | float32 |
| `float16` | `km.fp16` | A | 11 | 2048 | float32 |
| `bfp8` | `km.bfp8` | A | 7, a shared exponent | 1088 | float32 |
| `bfp4` | `km.bfp4` | A | 3, a shared exponent | 576 | float32 |
| `bfp2` | `km.bfp2` | A | 1, a shared exponent | 320 | float32 |

Significand bits count the hidden bit: bfloat16 is 1 + 7, float16 1 + 10. A block-float
tile is 64 exponent bytes, one for each row of 16 datums of a face, then the datums'
sign and magnitude (8, 4 or 2 bits each), with no hidden bit: a datum smaller than its
row's largest keeps fewer significant bits. `bfloat8_b` and `bfloat4_b` are the earlier
names of `bfp8_b` and `bfp4_b` and stay. A block format has no element size
(`formats.itemsize` refuses it) and no row layout: its tensors are tiles
(`formats.buffer_bytes(shape, dtype, "tile")`, `formats.tile_bytes(dtype)`).
`session.tensor(data, "bfp8_b", "tile")` encodes on the host; `tensor.read()` decodes.

## The exponent families

The B family has bfloat16's 8-bit exponent, the A family float16's 5-bit one. float32
and the integer formats go with either. The unpacker and the packer hold one exponent
width for every buffer of a kernel, so **a kernel's buffers are of one family**:
`km.Program` refuses a program whose buffers mix them, naming one buffer of each
("buffer 2 (bfp8) is of the A family ... and buffer 0 (bfloat16) of the B family"). A
class's product of the two families is refused at its line ("its operands are of one
exponent family"), any other mix of its buffers when it is made.

**The A family needs a 16-bit `Dst`.** With `fp32_dest_acc_en`, every A datum arrives in
`Dst` 2^-15 too large (through SrcA or unpacked straight to `Dst`: zero becomes 2^-15,
infinity 65536), an fp32 `Dst` packed to float16 is garbage, and a pack to bfp8 is all
zeros. With a 16-bit `Dst` the A copies are exact within float16's normal range:
subnormals become 0 and -0 becomes +0, infinity and NaN packed to float32 read 65536 and
98304, and a host-encoded A row whose exponent saturated (31) reads as NaN. So a
program with an A buffer and `fp32_dest_acc_en` is refused ("... 16-bit Dst"), and a
class of A buffers computes in a 16-bit `Dst` and refuses a float32 tile unpacked to
`Dst`. A tile function's code runs on an fp32 `Dst`, so it takes float32 and the B
family, not the A family. An A-family product loses about 1 % where its products are
near 1e-5 (the 5-bit exponent's range).

## Unpack and pack

The unpacker gives every format to the source registers as 19 bits: a sign, an 8-bit
exponent and a 10-bit mantissa (11 significant bits). A float32 operand is cut to that,
tf32: its twelfth significant bit is never multiplied. float16 fits whole; bfloat16
and the block formats leave the low bits zero.

The B packer rounds each datum to 7 significant bits at its own exponent (ties away
from zero), takes the largest of those exponents as the row's, aligns each datum to it
and rounds again (ties away) to the 7-bit magnitude, then cuts the magnitude to 3 or
1 bits for bfp4_b and bfp2_b. It never writes a negative zero. `bfp8_pack_precise`
changed no byte.

## The host encode and the card's

`formats.bfp_encode` is tt-metal's `pack_as_bfp_tiles`, bit for bit (checked against the
bound function on a mock device): each row of 16 datums shares its largest exponent,
each magnitude is rounded once to the width, half to even, and a zero has no sign. The
A family's exponent is rebiased and clamped to 0 .. 31. It is not the card's packer:
of 1024 datums of one test tile the two differ in 235 for bfp8_b (and 9 of 64 row
exponents), 377 for bfp4_b, 244 for bfp2_b, in the last bit of the magnitude. A tensor
the host encoded and the same values the card packed are therefore not equal byte for
byte; compare values with a tolerance of the format's last bit, or let one side make
both.

## The bits a product multiplies

A product multiplies SrcB (its left operand, a library product's `in0`) by SrcA (its
right, `in1`) in phases: phase 0 the top 5 bits of SrcA by the top 7 of SrcB, phase 1
SrcA's next 5 by SrcB's top 7, phase 2 SrcA's top 5 by SrcB's low 4, phase 3 the low
bits of both. A fidelity runs the first 1 (LoFi), 2 (HiFi2), 3 or 4 phases. Of the 11
significant bits each side holds, it multiplies:

| Fidelity | SrcB (left) | SrcA (right) |
| --- | --- | --- |
| LoFi | 7 | 5 |
| HiFi2 | 7 | 10 |
| HiFi3 | 11 | 10 |
| HiFi4 | 11 | 10 |

SrcA's eleventh bit is never multiplied. An operand keeps the smaller of its format's
significand bits and its side's: bfloat16 keeps 7 in SrcB and 8 in SrcA at HiFi2, all 8
on both sides at HiFi3 and HiFi4; a bfp8_b operand keeps all 7 in SrcB from LoFi and in
SrcA from HiFi2. `formats.bits_used(dtype, side, fidelity)` is this table. All 37
programs of the measurement equal this model exactly, at every fidelity and format.

The sum is not exact even when every product is: 32 products of bfloat16 operands at
HiFi4 summed in an fp32 `Dst` are 2.9e-4 (median, unbiased) from the float64 sum.

**The fidelity is a kernel's, not a product's.** The library's product reads the compute
kernel's `MATH_FIDELITY` at compile time, so every product of a kernel runs at one
fidelity; a class with products of two fidelities is refused ("the fidelity is the
compute kernel's").

## Placement

Which operand goes into SrcB matters at LoFi and HiFi2, and the weight's side is
usually the one to give up bits. A product tile function places its operands with a
marker:

```python
@km.tile_function(fidelity="HiFi2")
def project(x, w):
    return x @ km.srcb(w)  # w in SrcB, x in SrcA
```

The math is always `x @ w`. `km.srcb(...)` puts its operand in SrcB, `km.srca(...)` in
SrcA; one marker places both operands, and both marked for one side is refused. Without
a marker, or marked the plain way (`km.srcb(x) @ w`), the left operand is in SrcB and
nothing moves. Marked the other way (`x @ km.srcb(w)` or `km.srca(x) @ w`), the class
builds the **turned** product: it computes (X W)^T = W^T X^T, so

- W's tensor is stored with each tile transposed, which the host states when it writes
  it: `session.tensor(w, "bfp8_b", "tile", transposed_tiles=True)` (the tensor's
  `transposed_tiles`; `tensor.write` and `tensor.read` keep it). The build refuses a
  turned product whose W tensor is not stored so, and a plain one whose W tensor is;
- X's tile is read with the unpacker's transpose;
- `Dst` is transposed back to the plain layout (`transpose_dest`): `acc`'s tiles before
  the loop and after it, the field sums' each subblock before its pack.

A block-float W stored with transposed tiles shares an exponent along 16 rows of W's
column instead of 16 columns of its row; "W stored" in the table below is each layout's
own loss. The turn is built for the three product forms of a class: the field product,
the stream product into `Dst` and the field sums. A product of two broadcasts' blocks
refuses a turned marker, and a tile function the tile compiler traces is refused when
it holds a marker ("km.srcb marks an operand of a tile function `return a @ b` that a
handler's product line adds").

The build prints both placements of every product with the bits each operand keeps,
whether or not it is marked:

```text
placements (each product: the significand bits of 11 each operand keeps in the matrix unit, either way):
  ProofHiFi2.plain8, line 85: self.sums += self.plain(xb, wb): left bfloat16, right bfp8_b, HiFi2; plain (left in SrcB): left keeps 7, right 7; turned (right in SrcB): left 8, right 7; built plain, no marker
```

The program keeps the lines as `program.placements`.

**Measured.** Y = X W, 32 x 1024 x 1024 on 8 cores (each 4 columns of Y, X and W streamed
in step into an fp32 field), X bfloat16, W float32 normal / 32 stored in each format;
error against float64 of the given values, the median of 3 commands, the weight bytes a
command reads ([placement.py](../../experiments/core/formats/README.md#the-placement-of-a-linear-layer)):

| Fidelity | W | Form | RMS rel. | Max / RMS | µs | W bytes | W stored |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LoFi | bfloat16 | no marker | 2.62e-2 | 0.117 | 17.9 | 2,097,152 | 1.66e-3 |
| LoFi | bfloat16 | `km.srcb(w)` | 2.66e-2 | 0.107 | 18.6 | 2,097,152 | 1.66e-3 |
| LoFi | bfp8_b | no marker | 2.21e-2 | 0.094 | 17.4 | 1,114,112 | 7.26e-3 |
| LoFi | bfp8_b | `km.srcb(w)` | 2.50e-2 | 0.108 | 17.9 | 1,114,112 | 7.26e-3 |
| LoFi | bfp4_b | no marker | 0.118 | 0.511 | 17.1 | 589,824 | 0.117 |
| LoFi | bfp4_b | `km.srcb(w)` | 0.119 | 0.509 | 17.7 | 589,824 | 0.117 |
| HiFi2 | bfloat16 | no marker | 4.40e-3 | 0.023 | 17.9 | 2,097,152 | 1.66e-3 |
| HiFi2 | bfloat16 | `km.srcb(w)` | 4.39e-3 | 0.022 | 18.9 | 2,097,152 | 1.66e-3 |
| HiFi2 | bfp8_b | no marker | 8.33e-3 | 0.038 | 17.4 | 1,114,112 | 7.26e-3 |
| HiFi2 | bfp8_b | `km.srcb(w)` | 7.34e-3 | 0.030 | 18.2 | 1,114,112 | 7.26e-3 |
| HiFi2 | bfp4_b | no marker | 0.118 | 0.511 | 17.0 | 589,824 | 0.117 |
| HiFi2 | bfp4_b | `km.srcb(w)` | 0.118 | 0.501 | 18.0 | 589,824 | 0.117 |
| HiFi4 | bfloat16 | no marker | 1.70e-3 | 0.009 | 18.3 | 2,097,152 | 1.66e-3 |
| HiFi4 | bfloat16 | `km.srcb(w)` | 1.70e-3 | 0.008 | 20.1 | 2,097,152 | 1.66e-3 |
| HiFi4 | bfp8_b | no marker | 7.28e-3 | 0.031 | 17.7 | 1,114,112 | 7.26e-3 |
| HiFi4 | bfp8_b | `km.srcb(w)` | 7.34e-3 | 0.030 | 19.7 | 1,114,112 | 7.26e-3 |
| HiFi4 | bfp4_b | no marker | 0.119 | 0.513 | 17.4 | 589,824 | 0.117 |
| HiFi4 | bfp4_b | `km.srcb(w)` | 0.118 | 0.501 | 19.7 | 589,824 | 0.117 |

At HiFi4 the turned product equals the plain one within the sum's noise. A bfp8_b
weight in SrcB keeps all its 7 bits from LoFi and X keeps its 8 in SrcA from HiFi2, so
the marked HiFi2 result is the HiFi4 result bit for bit; unmarked, HiFi2 cuts X to 7
bits. At LoFi the unmarked placement is the better one for bfp8_b: SrcA's 5 bits cut a
block-float weight, whose smaller datums have fewer than 7 bits, less than they cut X.
The turn costs 0.5 to 2 µs a command here (the `Dst` transposes, about 180 cycles a
tile, and the matrix unit set up again after them). The build does not choose a
placement: it prints both, and the marker is the program's choice.

## Not in this version

- **lf8** (an 8-bit float the packer can write) is not built: no buffer, tensor or
  class format names it.
- **A choice of placement by the build.** The build prints both placements' bits; it
  does not pick one.
- **A turned product of two broadcasts' blocks**, and markers in a tile function the
  tile compiler traces.
- **A fidelity a product.** One a compute kernel (above).
- **The A family in a tile function**, which runs on an fp32 `Dst`.
- **The card's packer on the host.** `formats.bfp_encode` is tt-metal's encode, not the
  packer's rounding.
