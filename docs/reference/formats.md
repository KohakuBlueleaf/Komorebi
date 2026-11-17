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
| `float32` | `km.fp32` | either | 24 (tf32 in a product: 11; a class product of two streams only) | 4096 | float32 |
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
`Dst`. A tile function of the A family runs on a 16-bit `Dst` too
([below](#the-a-family-in-a-tile-function)). An A-family product loses about 1 % where its products are
near 1e-5 (the 5-bit exponent's range).

## Unpack and pack

The unpacker gives every format to the source registers as 19 bits: a sign, an 8-bit
exponent and a 10-bit mantissa (11 significant bits). float16 fits whole; bfloat16
and the block formats leave the low bits zero. A float32 operand is cut to that, tf32
(its twelfth significant bit never multiplied), only when the kernel sets the sources'
format to tf32 itself: a hand kernel's float32 x float32 product does, and is proven on
the card (the probe below; `projects/decode`'s attend kernel). The library's tables give a
float32 buffer to its source register as float32, and a class product built on them gave
wrong words (1e37 or NaN). A class product of two streams in step sets a float32 side as
tf32 itself (`km::tf32_formats`, `device/core/km_tf32.hpp`, in place of the tables'
formats; every later unpack of the kernel then sets its own again), and on the card each
float32 pair is right (float32 x bfp8_b 4.8e-4, x bfloat16 4.9e-4, bfloat16 x float32
8.4e-4, float32 x float32 1.2e-3 relative to float64:
[class pairs](../../experiments/core/formats/README.md#a-class-products-operands)).
Still refused at its line ("a product's left operand is float32: ..."): a float32
operand of any other class product (a field, a cast, a held stream's row) and of a
turned product. Stream a bfloat16 copy of the tensor, or declare the field `km.bf16`.

The B packer rounds each datum to 7 significant bits at its own exponent (ties away
from zero), takes the largest of those exponents as the row's, aligns each datum to it
and rounds again (ties away) to the 7-bit magnitude, then cuts the magnitude to 3 or
1 bits for bfp4_b and bfp2_b. It never writes a negative zero. `bfp8_pack_precise`
changed no byte.

## The host encode and the card's

A block tile's parts: `formats.bfp_tile_bytes(bits)` (1088, 576, 320 for 8, 4, 2
bits), `BFP_WIDTH` (the magnitude bits of each: 7, 3, 1), `bfp_unpack_codes(raw, bits)`
(the exponents `[N, 64]` and datum codes `[N, 1024]` of `[N, tile bytes]` tiles) and
`bfp_pack_codes(exponents, codes, bits)` (its inverse).

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
  `transposed_tiles`; `tensor.write` and `tensor.read` keep it; set on an empty tensor
  before its first write, refused after the host wrote its data). The build refuses a
  turned product whose W tensor is not stored so, and a plain one whose W tensor is;
- X's tile is read with the unpacker's transpose;
- `Dst` is transposed back to the plain layout (`transpose_dest`): `acc`'s tiles before
  the loop and after it, the field sums' each subblock before its pack.

A block-float W stored with transposed tiles shares an exponent along 16 rows of W's
column instead of 16 columns of its row; "W stored" in the table below is each layout's
own loss. The turn is built for the four product forms of a class: the field product,
the stream product into `Dst`, the field sums and the product of two broadcasts' blocks
(W's tiles of the block's columns into SrcB, each row of X's subblock read transposed,
the subblock's `Dst` turned back before its pack; the W the build checks is the tensor
the block casts).

A tile function the tile compiler traces (vector code with its products, as
`km.silu(a @ w1) * (a @ w3)`) takes a plain marker (`km.srcb(a) @ w1`), which moves
nothing, and refuses a turning one at the call's line: its vector code runs on the
products in `Dst` within one body, where a turned product would leave (A W)^T, and the
transpose back is a library call between the body's words. Write that product as a
tile function `return a @ km.srcb(w)` of its own.

The command is checked too: a command's tensor that a product reads as W (by the
handler's parameter) is refused at submit when its `transposed_tiles` differs from the
product's placement, naming the tensor and the product's line; through a merged
program too (`km.merge` of made programs, a class deriving two classes), by its handler
or its class's:

```text
apply: tensor w is not stored with its tiles transposed, and Sums.apply, line 153: self.sums += self.turned(xb, wb) reads it as W of a turned product; write it with session.tensor(..., transposed_tiles=True)
```

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

The A family's turn and the broadcasts' turn, measured the same way
([placement_more.py](../../experiments/core/formats/README.md#the-turns-other-forms-float16-and-two-broadcasts-blocks)):
float16 X and W with the sums in the 16-bit `Dst` (32 x 1024 x 1024, 8 cores), and two
broadcasts' blocks on a 2 x 2 rectangle (X bfloat16, W bfp8_b, K 1024):

| Product | Fidelity | Form | RMS rel. | Max / RMS | µs |
| --- | --- | --- | --- | --- | --- |
| float16 | LoFi | no marker | 3.00e-2 | 0.141 | 17.0 |
| float16 | LoFi | `km.srcb(w)` | 2.99e-2 | 0.119 | 17.2 |
| float16 | HiFi2 | no marker | 6.71e-3 | 0.040 | 17.1 |
| float16 | HiFi2 | `km.srcb(w)` | 6.73e-3 | 0.034 | 17.4 |
| float16 | HiFi4 | no marker | 3.53e-3 | 0.022 | 17.4 |
| float16 | HiFi4 | `km.srcb(w)` | 3.51e-3 | 0.021 | 17.5 |
| broadcasts | HiFi4 | no marker | 7.17e-3 | 0.029 | 24.6 |
| broadcasts | HiFi4 | `km.srcb(w)` | 7.23e-3 | 0.029 | 24.7 |

Each turn equals its plain product within the sums' noise. Two float16 operands carry
11 bits each, so the two placements cut alike (one keeps 5 and the other 7 at LoFi);
the marker matters when the operands' bits differ.

## The A family in a tile function

A tile function of fp16, bfp8, bfp4 or bfp2 tiles runs its whole unit program on a
16-bit `Dst`, as a class of A buffers computes: its operands are copied into `Dst`
through SrcA (exact, above), each `SFPLOAD` and `SFPSTORE` takes the FP16A mode where
the B family's take `Dst`'s own fp32, and the vector code runs in fp32 in the vector
unit's registers between them. The words are the same as the B family's, one for one
(the gate: 1336 words, 1678.6 cycles a tile in both), so a tile costs the same. The
families do not mix in one function ("given float16 with float32": an fp32 `Dst` reads
the A family 2^-15 off, a 16-bit one holds no float32).

What passes through the 16-bit `Dst`, and so through its 11 significant bits:

- each operand tile, copied in: exact;
- each product in the trace: the matrix unit accumulates its sums there, in the 16-bit
  format, **truncated** at each accumulation;
- a per-row operand (a state tile's value) that the stream form's per-row pass
  computes once and stores over the operand, then loads again;
- the result, stored once: **truncated** to the 16-bit format. On the card, x * y of
  fp16 tiles (an exact product in the registers) read back equal to the product
  rounded toward zero in 99.998 % of datums, to it rounded to nearest even in 50 %.

The 16-bit `Dst`'s format is not IEEE fp16 at its top: its largest exponent is a number,
not infinity. A product's sums above 65,504 read back as themselves (truncated) up to
131,008 and saturate there; none read infinity. Under 2^-14 a value reads 0: an operand
under 2^-14 is 0 from its unpack, so its terms drop from a product's sums, and a result
under 2^-14 stores as 0. A result above 65,504 packed to an fp16 tensor takes fp16's
exponent 31, which the host reads as infinity or NaN (above); packed to float32 it
reads as itself.

| Step | A family (16-bit `Dst`) | B family (fp32 `Dst`) |
| --- | --- | --- |
| an operand tile in | fp16 11 bits; bfp8 7, bfp4 3, bfp2 1 | bf16 8; bfp8_b 7, bfp4_b 3, bfp2_b 1 |
| a product's operands | as the build prints ([above](#the-bits-a-product-multiplies)): HiFi4, fp16 x fp16: 11 and 10 | bf16 x bf16: 8 and 8 |
| a product's sums in `Dst` | 11, truncated at each accumulation; 2^-14 to 131,008 (saturating) | fp32 |
| the vector code | fp32 registers | fp32 registers |
| the result stored to `Dst` | 11, truncated | fp32 |

**Measured** ([tile_a_family.py](../../experiments/core/formats/README.md#a-tile-function-of-the-a-family)):
the gate `silu(a @ w1) * (a @ w3)` of each 32 x 32 tile, 64 tiles a core on 12 cores,
a normal and w1, w3 normal / sqrt(32), the result packed to float32; the median of 3
commands. "Computed" is the card's result against the float64 gate of the values the
tensors hold (the computation's own error); "the inputs' formats" is the float64 gate
of those values against the float64 gate of the float32 inputs (the storage's error).

| Tensors | Computed: RMS rel. | Max / RMS | The inputs' formats: RMS rel. | µs |
| --- | --- | --- | --- | --- |
| fp16 | 1.48e-3 | 0.050 | 4.35e-4 | 87.6 |
| fp16 a, bfp8 w | 1.16e-3 | 0.035 | 9.94e-3 | 87.0 |
| bf16 | 4.57e-4 | 0.012 | 3.48e-3 | 87.2 |
| bf16 a, bfp8_b w | 4.56e-4 | 0.016 | 1.02e-2 | 86.7 |

The A family's computation is the less precise (its products' sums and its result are
truncated to 11 bits, the B family's kept in fp32), its inputs the more precise; with
both, fp16 is about 1.5e-3 against bf16's 3.5e-3, and a bfp8 weight's storage dominates
either family. x * y alone, fp16: 4.13e-4, each datum its product truncated; fp16 x
bfp8 3.95e-4; bf16 x bfp8_b exact.

## Not in this version

- **lf8** (an 8-bit float the packer can write) is not built: no buffer, tensor or
  class format names it.
- **A choice of placement by the build.** The build prints both placements' bits; it
  does not pick one.
- **A turned product in a tile function the tile compiler traces** (above).
- **A fidelity a product.** One a compute kernel (above).
- **The A family in a tile function's row and step forms** (below).
- **The card's packer on the host.** `formats.bfp_encode` is tt-metal's encode, not the
  packer's rounding.
