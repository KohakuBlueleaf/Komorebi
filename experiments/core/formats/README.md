# The card's tile formats, measured

What the matrix unit, the unpacker and the packer of a Blackhole card do with each
native tile format, one tile a case on core (0, 0), each case against a host model:

- which significand bits of each operand of a product each fidelity multiplies;
- what the A family (5-bit exponent: fp16, bfp8 / bfp4 / bfp2) keeps through unpack,
  Dst and pack;
- how the packer rounds to bfp8_b, bfp4_b and bfp2_b, against the host's encode;
- block tiles at odd tile offsets in a buffer;
- the transposed product `Y^T = W^T X^T` (the right tile read transposed, then a
  transpose in Dst) against the plain `Y = X W`, and what the Dst transpose costs.

Bits are counted as significand bits, the hidden one included, out of the register's
11. A format's own: fp16 and fp32 (held as tf32) 11, bf16 8, bfp8 7, bfp4 3, bfp2 1.

## Results

One Blackhole p150a, card AICLK 800 MHz; `results/card1.json` (56 programs) and
`results/card2.json` (11 programs).

### The bits a product multiplies

Of a product `left @ right`, the left operand is in SrcB and the right in SrcA. The
probe sets one mantissa bit at a time on each side (rows of the left against a row of
ones on the right, and the mirror), and a table of every left bit against every right
bit for the cross terms. In all 37 programs (bf16 with bf16, bfp8_b, bfp4_b, bfp2_b on
either side, fp32 with fp32, fp16 with fp16, bfp8 with fp16) the card's tile equals
the phase model below datum for datum:

| Fidelity | SrcB (left) | SrcA (right) | Phases |
| --- | --- | --- | --- |
| LoFi | 7 | 5 | 0 |
| HiFi2 | 7 | 10 | 0, 1 |
| HiFi3 | 11 | 10 | 0, 1, 2 (no low x low cross term) |
| HiFi4 | 11 | 10 | 0, 1, 2, 3 |

The phases: 0 multiplies SrcA's top 5 by SrcB's top 7; 1 SrcA's next 5 by SrcB's top
7; 2 SrcA's top 5 by SrcB's low 4; 3 the two low slices. SrcA's lowest significand bit
is never multiplied. The bits an operand keeps are the smaller of its format's and its
side's: a bf16 operand keeps 8 on SrcA from HiFi2 and on SrcB from HiFi3, 5 on SrcA
at LoFi, 7 on SrcB at LoFi and HiFi2. An fp32 operand is held as tf32: its 11th and
12th mantissa bits never reach the product.

### The B packer against the host encode

An fp32 tile in Dst (unpacked to Dst unchanged) packed to each block format, 1024
datums of designed rows (random, ties for each width, small negatives, a wide spread of
exponents, magnitudes that round over the largest, bits under bf16's):

| Format | Datums differing from the host encode | Exponents differing (of 64) |
| --- | --- | --- |
| bfp8_b | 235 | 9 |
| bfp4_b | 377 | 9 |
| bfp2_b | 244 | 9 |

The card's bytes equal, for every datum of the three formats, this model: each datum
is first rounded to 7 significant bits at its own exponent (nearest, ties away from
zero); the row's largest of those exponents is shared; each datum is aligned to it and
rounded again (nearest, ties away) to the 7-bit magnitude; bfp4_b and bfp2_b then keep
its top 3 or 1 bits (truncation). The host's encode (tt-metal's `pack_as_bfp_tiles`)
rounds the unrounded datum once, to nearest even, straight to the width. So the two
differ at ties, where the first rounding carries into the exponent, and for bfp4 and
bfp2 nearly everywhere the bits below the width are at least a half. No negative zero
was written. `bfp8_pack_precise` changed no byte.

### The A family

A products into an fp32 Dst of values near 1 multiply the same bits as the table above.
Everything else of the A family in an fp32 Dst is wrong:

- an fp16 or bfp8 datum unpacked, through SrcA or straight to Dst, arrives as itself
  plus 2^-15 at fp16's precision: a zero becomes 2^-15, an infinity 65536, values under
  2^-3 move visibly;
- fp32 Dst packed to fp16 is garbage (1019 of 1024 datums), to bfp8 all zero.

With a 16-bit Dst (`fp32_dest_acc_en` off) the A family is right within fp16's normal
range: fp16 through unpack and pack is exact except subnormals (to 0) and negative zero
(to +0); bfp8 likewise (its datums under 2^-14 to 0); infinities and NaN survive to
fp16 and become 65536 and 98304 in fp32. A row the host encode saturated (exponent 31)
reads as NaN. A product of two A operands is within fp16's rounding of the model where
the products are of ordinary size, and loses about 1 % where they are near 1e-5.

### Odd offsets

bfp4_b and bfp2_b tiles 1 and 2 of a three-tile buffer (576 and 320 bytes a tile) as
a product's right operand: right, within the error of the matrix unit's sum below
(1.8e-4 of the tile's RMS, 9.6e-4 at most).

### The transposed product

`W^T X^T` with the right tile (X) read transposed and the result transposed in Dst,
against `X W`: each is its own model at LoFi, HiFi2 and HiFi4 within the error of the
sum (3.2e-4 of the RMS, 1.6e-3 at most). At LoFi and HiFi2 the two differ by which
operand lost its bits. The Dst transpose of one fp32 tile took 166 to 189 wall-clock
cycles; the product itself 891 to 1010.

### The sum

The 32 products of a bf16 dot product are exact at HiFi4, but their sum in an fp32
Dst is not: against the float64 sum of the same products the median relative error is
2.9e-4 (about 2^-12), unbiased. This bounds a product's precision whatever its
operands' bits; a single-term product (the bits probe) is exact.

### The placement of a linear layer

`placement.py`: Y = X W, 32 x 1024 x 1024 on 8 cores (each 4 columns of Y, X and W
streamed in step, the sums in an fp32 field), X bf16, W float32 normal / 32 stored in
each format; the tile function `return x @ w` (no marker: X in SrcB) against
`return x @ km.srcb(w)` (W stored with its tiles transposed, the product turned). Error
against the float64 product of the given values; "W stored" is the format's own loss;
time the median of 3 commands; the weight bytes a command reads.

| fidelity | W | form | rms rel | max / rms | us | W bytes | W stored |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LoFi | bf16 | no marker | 2.62e-2 | 0.117 | 17.9 | 2,097,152 | 1.66e-3 |
| LoFi | bf16 | km.srcb(w) | 2.66e-2 | 0.107 | 18.6 | 2,097,152 | 1.66e-3 |
| LoFi | bfp8_b | no marker | 2.21e-2 | 0.094 | 17.4 | 1,114,112 | 7.26e-3 |
| LoFi | bfp8_b | km.srcb(w) | 2.50e-2 | 0.108 | 17.9 | 1,114,112 | 7.26e-3 |
| LoFi | bfp4_b | no marker | 0.118 | 0.511 | 17.1 | 589,824 | 0.117 |
| LoFi | bfp4_b | km.srcb(w) | 0.119 | 0.509 | 17.7 | 589,824 | 0.117 |
| HiFi2 | bf16 | no marker | 4.40e-3 | 0.023 | 17.9 | 2,097,152 | 1.66e-3 |
| HiFi2 | bf16 | km.srcb(w) | 4.39e-3 | 0.022 | 18.9 | 2,097,152 | 1.66e-3 |
| HiFi2 | bfp8_b | no marker | 8.33e-3 | 0.038 | 17.4 | 1,114,112 | 7.26e-3 |
| HiFi2 | bfp8_b | km.srcb(w) | 7.34e-3 | 0.030 | 18.2 | 1,114,112 | 7.26e-3 |
| HiFi2 | bfp4_b | no marker | 0.118 | 0.511 | 17.0 | 589,824 | 0.117 |
| HiFi2 | bfp4_b | km.srcb(w) | 0.118 | 0.501 | 18.0 | 589,824 | 0.117 |
| HiFi4 | bf16 | no marker | 1.70e-3 | 0.009 | 18.3 | 2,097,152 | 1.66e-3 |
| HiFi4 | bf16 | km.srcb(w) | 1.70e-3 | 0.008 | 20.1 | 2,097,152 | 1.66e-3 |
| HiFi4 | bfp8_b | no marker | 7.28e-3 | 0.031 | 17.7 | 1,114,112 | 7.26e-3 |
| HiFi4 | bfp8_b | km.srcb(w) | 7.34e-3 | 0.030 | 19.7 | 1,114,112 | 7.26e-3 |
| HiFi4 | bfp4_b | no marker | 0.119 | 0.513 | 17.4 | 589,824 | 0.117 |
| HiFi4 | bfp4_b | km.srcb(w) | 0.118 | 0.501 | 19.7 | 589,824 | 0.117 |

The turned product is right end to end: at HiFi4 it equals the plain one within the
sum's noise. A bfp8_b or bfp4_b weight in SrcB uses all its bits from HiFi2, and X's 8
in SrcA, so its HiFi2 result is its HiFi4 result bit for bit; unmarked, HiFi2 cuts X
to 7 bits (8.33e-3 against 7.34e-3). At LoFi the unmarked placement is the better one
for bfp8_b: SrcA's 5 bits cut a block-float weight (whose small datums have fewer bits
than 7) less than they cut X. The turn costs 0.5 to 2 us a command (each subblock's
Dst transposed back, the matrix unit set up again). bfp4_b is its storage's error at
every fidelity. The commands are not bound by the weight's bytes at this size.

## Running it

    python experiments/core/formats/probe.py build [--set 2]   # mock device: builds every case
    python experiments/core/formats/probe.py card [--set 2]    # one card

`build` fills the runtime's kernel cache on a mock device (and checks formats.py's
block-float encode and decode against tt-metal's, bit for bit); `card` reads the same
cache, runs each case once and writes `results/card<set>.json`.

    python experiments/core/formats/placement.py build   # mock device: the three programs
    python experiments/core/formats/placement.py card    # one card: results/placement.json
