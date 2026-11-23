# Upscale: QuickSRNet-Medium 2x on one card

QuickSRNet-Medium 2x takes a 512 x 512 RGB frame and gives a 1,024 x 1,024 one. It is
seven 3 x 3 convolutions followed by a pixel shuffle. Here it runs on one Blackhole
p150a as ONE chain command of one class. The result is checked against the host's
emulation of the card's formats.

- Layer 1: 3 to 32 channels.
- Layers 2 to 6: 32 to 32 channels.
- Layer 7: 32 to 12 channels.
- After every layer: Hardtanh(0, 1), zero padding.
- At the end: the pixel shuffle by 2.

## Files

- `model.py`: the network in numpy.
  - The float pass.
  - The emulation of the card's arithmetic.
  - The weight tiles.
  - bicubic, PSNR.
- `host.py`: checks A and B on the host. These are the float32 and int8-trained passes,
  bicubic, and the emulation.
- `upscale.py`: the class.
- `upscale_move.hpp`: mover B's shift step.
- `upscale_compute.hpp`: the compute bodies.
- `run.py`: the card run, with the check against the emulation, the timing and the
  stamps.

## Running it

```
python projects/upscale/host.py <weights folder> <picture>... [--out <folder>]
python projects/upscale/run.py <weights folder> <picture>... [--out <folder>] [--stamps]
```

The weights folder holds `float32.npz`, and `host.py` also reads `int8.npz`. Each holds
the checkpoint's `state_dict` as plain arrays, named `cnn.0.weight` ... `conv_last.bias`.

A picture is a 1,024 x 1,024 PNG:

- Its 2 x 2 box average is the 512 input.
- The picture itself is the target.
- The values fed are display values, as the PNG stores them. The network was trained
  on display values.

## The layout

An activation is a bfloat16 tile tensor of 517 tensor rows by 16 tiles:

- Tensor row R holds image row R - 1.
- Rows 0 and 513 to 516 are padding, never read as data.
- A tile is 32 neighbouring pixels of one image row (tile rows) by 32 channels (tile
  columns).
- Two such tensors alternate as each layer's input and output.

The weights are 84 bfp8_b tiles, read into L1 once by `load` and kept there:

- Each layer has ten tiles: nine taps and the bias.
- Each tap is stored as the turned product's left operand (out channels by in channels).
- The bias tile holds the bias in column 0.
- Then come the shuffle's 14 tiles of 0 and 1.

## The handler

Every core runs the same `conv(x, y, layer)`. The rows are dealt by `self.rank`: rank g
writes image rows 5g to 5g + 4.

- Cores 0 to 102 carry rows. A chain call writes equal blocks, so 512 rows at 5 a core
  take 103 cores.
- Rows 512 to 514 of the last core are junk. They are never read, because the shift
  step zeroes rows outside the image.

A layer, on one core:

1. **Read (class line, mover A).** `band.fill` reads 7 input rows: the band and one
   halo row on each side, 112 tiles.
2. **The shift (`km.cxx` on mover B).** Each band row is copied to three versions: as
   it is, and shifted by one pixel either way. The section below says how.
3. **The products (`km.cxx` on compute).** Each output tile is ten block products in one
   fp32 DST tile:
   - In SrcB: the weight tile (bfp8_b).
   - In SrcA: the version tile, read transposed (bf16). This is the turned product
     of `projects/transport`.
   - Nine taps, then the bias tile times a ones tile.
   - Then Hardtanh, by `relu_max` at 1.0.
   - Then `transpose_dest` back to pixels by channels.
   - Then a bf16 pack into the write stream's block: one image row of 16 tiles.
4. **Write (class line).** The write stream writes the rows to `y`.

`last(x, result, layer)` does the same read and shift, then the pixel shuffle. For each
tile column:

1. **The five rows' convolutions.** Z_r is packed to bf16 without the turn.
2. **Gather by channel.** Z'_j = sum over r of P_jr Z_r, with 0/1 tiles. Row 6r + 3i + c
   of Z'_j holds channel 4c + 2i + j, the torch order.
3. **Spread the pixels.** O_h^T = sum over j of Q_hj Z'_j^T: output pixel x of half h is
   pixel 16h + x/2 of parity x % 2.
4. **Pack.** O_h is turned and untilized as float32 into a 32-row field: one row per
   output row and colour.

Mover B then writes that field's 30 rows to `result`. The output is float32,
row-major: page 3Y + c is output row Y of colour c, 1,024 values.

`forward` is the `@km.chain` of six `conv` calls and one `last`. Each record waits for
the marks of the record before it, by a covering range: a band's halo rows come from
the neighbouring cores.

The lines that needed `km.cxx`: the shift copy, the products, the ready token, and the
last layer's open, close and write of its field. The reads, the write stream, the kept
fields and the chain are class lines.

## The shift

The brief proposed reading a shifted tile from L1 one pixel earlier or later. The
unpacker cannot do this:

- It reads a tile as four contiguous 512 B faces of 16 x 16.
- A pixel (a tile row) is face row r of faces 0 and 1, or of faces 2 and 3.
- So moving the start address by one pixel (32 B, one face row) shifts the pixel only
  inside a face. Rows 15 and 31 of a face pair come from the wrong face.

The smallest copy that fixes this, for each row and each shift:

- One 32 KB local NoC read from the row, 32 B later or earlier.
- After a barrier, four 32 B face rows a tile are copied from the neighbouring face,
  or the neighbouring tile (zero at the image's edge).

The unshifted version is one 32 KB copy. Mover B writes all three versions so that
compute reads only the copies, and the band field is freed.

## Measured (card 1, 2026-11-23, AICLK 1350 MHz)

Card against the emulation (all seven layers as one command):

| frame | card vs target dB | emulation vs target | card vs emulation dB | 8-bit > 1/255 off | most (8-bit) |
| --- | --- | --- | --- | --- | --- |
| sponza overview | 31.550 | 31.551 | 62.20 | 0.066 % | 3 |
| bistro interior | 33.785 | 33.784 | 63.11 | 0.021 % | 3 |
| bistro exterior | 27.135 | 27.134 | 60.05 | 0.118 % | 4 |

On the host (sponza), the float32 network gives 31.71 dB and bicubic 31.64 dB. On these
rendered frames the network adds -0.04 to +0.19 dB over bicubic. The emulation loses
0.16 to 0.19 dB against the float pass, mostly from the bfp8_b weights.

Time:

- 32 calls of the chain command: median 830.0 us, least 827.2 us. The host wall time
  for a call is 980 us.
- Per layer, from the chain stamps (median and slowest core, in us):

| layer | 1 | 2 | 3 | 4 | 5 | 6 | 7 + shuffle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| busy | 119 / 140 | 103 / 114 | 102 / 110 | 101 / 110 | 100 / 109 | 100 / 109 | 114 / 124 |
| waiting for the layer before | 0 / 0 | 7 / 17 | 7 / 25 | 8 / 24 | 8 / 23 | 8 / 24 | 8 / 24 |

Bytes per layer:

- Read: 23.6 MB, since 7 rows are read for 5 because of the halo.
- Written: 16.9 MB. The last layer writes 12.7 MB of float32.

**What limits it.** The test timed layer 1 alone as a command, 32 calls (median, in us):

| version | us |
| --- | --- |
| as built | 140.6 |
| without the products | 104.6 |
| without the shift copies | 126.3 |
| DRAM read and write only | 92.8 |

So the DRAM traffic is the floor: 40.5 MB a layer at about 0.44 TB/s. The shift adds
about 14 us and the products about 36 us. Neither overlaps the read, because a core's
products start after its whole band is in and shifted.

## Limits

- The frame is 512 x 512 with 3 input channels. The rows a core and the core count
  are constants of `upscale.py` and `run.py`.
- At most 32 channels a layer: one tile.
- 17 of the 120 cores are idle, and the last core computes 3 junk rows. The slowest
  core has 5 rows either way.
- The L1 buffers take 1,292,288 B of the 1,351,616 B left. The three shifted versions
  of 7 rows are most of it (672 KB).
- The output is float32 in colour planes per row, not interleaved RGB, and not 8-bit.

## Weights

QuickSRNet (Berger et al., arXiv 2303.04336). Code and weights are from the AIMET model
zoo (Qualcomm Innovation Center), BSD-3-Clause.

The float32 checkpoint pickles an optimizer object. So it was converted to `.npz` with
a restricted unpickler that builds only the tensors from the zip's raw bytes and
imports no class. The int8 checkpoint holds quantisation-aware-trained floats in the
same channel order.
