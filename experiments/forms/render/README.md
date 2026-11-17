# forms: rendering

A renderer's per-pixel passes, each written as classes with the language as it is and
checked against numpy on one Blackhole p150a (card 1): a texel fetch, a texture
filter's and a depth test's arithmetic, a vertex stage, a separable 3x3 filter over
tiles with a halo, a small MLP over pixel blocks, and integer work. Each script prints
one JSON line, `ok` when every check is within its bound, and exits 1 otherwise.

```bash
cd experiments/forms/render
python fetch.py  # --held: the held form and its probes, wrong today
python filter.py
python denoise.py
python mlp.py
python integers.py
python frame.py
python thread.py
python dynamic.py  # --fan: V2, its first run lost a send for want of quota
python int_tile.py
python operands.py
python field_rows.py
python treelet.py
python denoise.py --cards 0,1
python fetch.py --cards 0,1
```

The scripts use `../nn/forms.py`'s helpers. Where a direct form exists (`km.where`,
`km.floor`), the script runs it beside the way round it, with both errors and times.
The times are one command's event microseconds. Each script ran once on card 1
(denoise and fetch also once on cards 0 and 1). The first run of filter, denoise
and mlp failed only on bounds the scripts set too tight (1e-6, 1e-4), which assumed
exact sums and a bit-exact silu; the bounds below are what the formats explain.

## Texel fetch (`fetch.py`)

Each pixel's texel address is a word of a u32 page, and the result is
`texels[address]`, checked word for word.

- **DRAM**: an `entries` loop over a core's address page. For each entry a body on
  mover B reads the texture page the address names (`km::read_pages`) and takes the
  texel, then writes the result page with `km::write_pages`. 8 cores x 1,024 texels
  from a 64-page texture: **exact**, 580.7 us.
- **Held in L1** (`--held`; 4 pages, a field on every core; a body on pack gathers from it
  into the write stream's page): **wrong**. 32,761 of 32,768 texels are wrong. Three
  probes of the same body on one core and one page narrow it:

  | The body writes into `out` | Wrong of 1,024 |
  |---|---|
  | a constant | 0 |
  | the field's word 0 (`self.tex`) | 0 |
  | a copy of the stream's page (`addr`) | 1,024 (909 of them 0) |

  The out page and the field reach pack's body right; the page at `addr` is not the
  stream's page. Not fixed.

The item-stream form (`tex.items(address, 1, size=4)` in the entries loop) is not
run here.

## Filter, depth and vertex arithmetic (`filter.py`)

float32 tiles, 8 cores x 2 tiles. Each error is over the rms of the numpy result.

| Form | rms | max | us |
|---|---|---|---|
| bilinear as three `lerp` passes through two scratch tensors | 3.4e-8 | 1.3e-7 | 7.2, 6.2, 6.3 |
| `km.floor` | 0 | 0 | 5.1 |
| floor by the magic constant, less 1 where it rounded up (no `where`) | 0 | 0 | 5.0 |
| frac, `u - km.floor(u)` | 2.6e-9 | 5.2e-8 | |
| clamp, `km.minimum(km.maximum(u, 0), 1)` | 0 | 0 | |
| nearer depth, `km.where(z_new < z_old, z_new, z_old)` | 0 | 0 | 5.7 |
| nearer depth by arithmetic, the difference x 2^64 clamped to 0..1 as the mask | 1.8e-8 | 7.3e-8 | 5.8 |
| x' from 4 float words of the command | 3.9e-8 | 2.1e-7 | 6.4 |
| the camera as `[x y z 1] @ M`, bf16, HiFi4 (16 words do not fit a command) | 1.3e-4 | 5.5e-4 | 3.3 |

frac, floor and clamp ran in one command (9.1 us). u is uniform in -40..40, with
-2, -1.5, -0.5, 0, 0.5, 1.5, 2.5 and -1e-7 among the values, and z ties
included. The camera's error is against float64 on the stored bf16 values, with
the bound at 2.9e-4: every product is exact at HiFi4, but their sum in `Dst` is not
([formats](../../../docs/reference/formats.md#the-bits-a-product-multiplies):
32 products, 2.9e-4).

## A 3x3 filter over tiles with a halo (`denoise.py`)

A 4 x 4-tile bf16 frame, padded on the host by one zero tile on each side, 16 cores.
The filter is separable:

- **Horizontal pass**: the 3 tiles of a row as one stream block, times the taps as a
  96 x 32 matrix held in L1. It writes into a padded bf16 mid frame.
- **Vertical pass**: the roles are swapped. The 3 tiles of a column are held, and the
  taps stream by as a 32 x 96 matrix on the left; the stream tile is always the left
  operand.
- **Vertical pass, taps on the left**: the 32 x 96 taps held in L1 (filled once by
  `load_taps`), the column's 3 tiles streaming, `acc += self.left(xb, self.vtaps)` with
  `left` written `return b @ a` (a product whose left field is filled).

On card 0 (2026-11-18), the taps on the left give the swapped form's values to the bit
(shift exact; blur rms 1.5e-3, max 1.3e-2 against numpy with mid in bf16; rms 1.7e-3
against float64) in 3.6 us (shift) and 3.5 us (blur), against 4.6 us for the swapped
form in the same run. The horizontal pass took 3.6 and 3.5 us there.

| Taps | Against | rms | max | us (horizontal, vertical) |
|---|---|---|---|---|
| (1, 0, 0), a shift | numpy | 0 | 0 | 3.8, 4.6 |
| (0.25, 0.5, 0.25) | numpy, mid rounded to bf16 | 1.5e-3 | 1.3e-2 | 3.5, 4.6 |

Rounding mid to bf16 is itself 1.7e-3 from float64. The card's sums are a little off
the exact ones ([formats](../../../docs/reference/formats.md#the-bits-a-product-multiplies)),
so some of mid's values round to the neighbouring bf16. The script checks the blur
against float64 at rms 2.5e-3. That number was not printed in the run above; the
script prints it now.

## MLP over pixel blocks (`mlp.py`)

`y = silu(x W1) W2` on 120 cores, 2 pixel rows a core. Each row is 32 pixels by 64
features, with 64 hidden and 64 out. W1 and the hidden layer are held in L1, and the
products run at HiFi4. W2 is (a) streamed from DRAM for each row, or (c) cast once
along a column of cores and held.

| Form | rms against float64 | max | us |
|---|---|---|---|
| (a) W2 streamed | 1.76e-3 | 1.4e-2 | 21.9 |
| (c) W2 cast | 1.76e-3 | 1.4e-2 | 17.8 |

The two forms give the same values. The hidden layer rounded to bf16 by numpy is
itself 1.69e-3 from float64, and the card is 1.17e-3 from that model. The check is
rms 2.5e-3 against float64.

## Integers (`integers.py`)

- **(a)** Bit operations on a handler's int words (`& | ^ << >> % min max`) pick the
  page a command writes: pages 560 and 333, as numpy reckons them.
- **(b)** A depth comparison of two stream tiles as a 0/1 mask, by `km.where` and by
  arithmetic: both exact, 5.6 us each.
- **(c)** A radix pass, one byte of 1,024 u32 keys a core on 8 cores, is a body on
  mover B. It reads the page, counts, places the keys and writes the page back: exact
  against numpy's stable sort by that byte, 52.3 us.

Tiles are floats: integer values compare exactly in float32 tiles up to 2^24, and in
bf16 tiles up to 256 (8 significant bits). Integer tiles in a tile function are not a
form here; integer work on page data is a body.

## A frame's passes (`frame.py`)

A class of five handlers over 16 float32 tiles, run on 8 cores:

1. **vertex**: a tile function of x, y and z.
2. **bin**: a body on mover B, one core. It sorts the screen tiles into list pages
   by each tile's bin (a u32 page of 4 bin ids, every bin used) and writes each
   bin's first page and count.
3. **raster**: one core a bin, an `entries` loop over its bin's first page and
   count; it reads and writes those pages.
4. **shade**.
5. **resolve**.

| Form | Image against numpy (float32, list order) | us |
|---|---|---|
| 5 commands | rms 3.1e-8, max 3.1e-7 | 7.7, 20.8, 8.4, 6.6, 4.8 |
| split at the scatter: a chain (vertex), bin and raster as commands, a chain (shade, resolve) | the same values, bit for bit | 8.0, 20.6, 8.3, 13.1 |
| one chain through bin | refused in this run; builds since (below) | |

Each bin's first page and count are right in both forms. One chain of the five calls
is refused: "ChainRefused: Frame.raster, line 122: e.first is not a number the call's
words and the class's constants give; it still runs as a command". Raster writes
`zbuf.pages(e.first, e.pages)`, a place the table's word gives. That refusal is
fixed since: one chain through bin is bit-equal to the five commands (46.7 us against
48.3 us, measured with the fix, not rerun here), and the chain of this script builds
on the mock and runs there to its end.

## The frame from an X280 thread (`thread.py`)

The same class, with one more handler (tally: the bins in use, a word it returns).
Its X280 thread ([thread/frame.c](thread/frame.c)) submits vertex, bin and tally.
It then runs raster on as many cores as tally's word says (the event's result word),
then shade and resolve. Bin 3 holds no tile, so raster ran on 3 cores.

- The thread's code is 0, and its result word is 3, as numpy counts the bins in use.
- The image is rms 3.1e-8 from numpy. The thread took 72.6 us for the six commands.

## Dynamic work (`dynamic.py`)

16 cores, 16 seed rays, depth 3. Each core reports its totals in results words of
an L1 tensor, as example 05 does.

- **V1b, one body**: a ray reads the scene word its origin picks, from 256 words in
  an L1 tensor on every core. While its depth is above 0 it spawns a ray at
  origin + hit and forwards itself, each at depth - 1, to the core its origin picks.
  At depth 0, or when the quota runs out, it is consumed. The walk, the quota test,
  send, forward and consume are all in one body. Rays consumed: 128, origin sum
  102,310, none for want of quota: equal to a Python walk of the same rules.
  47.8 us.
- **V2, lines** (`--fan`): a message spawns `fanout` (a word of the message, 3) children at
  depth - 1 and is consumed; the depth-0 ones are counted. **Wrong**: 431 leaves of
  432, origin sum 93,085 against 93,096. Exactly one leaf, origin 11, is missing;
  the run's event is "done", 55.7 us. The cause was a `self.send` with no quota left,
  dropped unseen: a core's quota was then max_live over the load's 118 ranks, 34
  sends at 4,096. It is now max_live over the run's ranks: 292 a rank for this 16-core
  group (14 ranks) at 4,096 ([dynamic](../../../docs/reference/dynamic.md)). Since the
  fix such a send ends the run with `km.MessagesGaveUp`, naming the
  core and what to change ([dynamic](../../../docs/reference/dynamic.md)). The fix's
  own proof (worker fix's job, card 0 and cards 0,1, 2026-11-18): V2 at max_live
  1,024 ends with `km.MessagesGaveUp` ("the run was given up by core 0: quota ...")
  and stops clean; at 16,384 it is exact. V2 is not rerun here.

max_live and the queues are 4,096 each: 34 sends a core when these ran, enough for
V1b's walk (it counts a ray that finds no quota, and none did) and not always for V2's
fan; by today's rule the 16-core group's ranks get 292 each.

## Ways round, each beside its direct form

Each script tries the direct form first and reports its refusal at the build, then
runs the way round on card 0 (one run each, 2026-11-18). None needs a change to the
framework.

- **A uint32 tile** (`int_tile.py`): packed RGBA8 texels, a uint32 a texel. The
  direct form, a tile function of the uint32 tile, is refused: "a tile function takes
  tiles of float32 or the B family ... given with uint32". The way round is a body on
  mover B that unpacks each page of words into four float32 pages, one a channel,
  then a tile function of the four, (0.299 r + 0.587 g + 0.114 b) a / 65025: 0 of
  65,536 words wrong, rms 5.2e-8. 16 pages on 8 cores: the unpack 313.0 us, the shade
  6.9 us. Each core unpacks its 2 pages in parallel with the others, 156.5 us a page
  (1,024 texels) on one mover, a word at a time; the direct form would be about one shade
  pass, so the way round costs about 45 times it. The faster form the docs name, a
  compute body with sfpi integers ([classes](../../../docs/reference/classes.md)'s
  "Not in this version"), is not built here: recorded, with this price.
- **6 operand tiles** (`operands.py`): the bilinear filter of four texels, fx and fy in
  one pass. The tile function of 6 operands is refused: "the unit program reads at
  most 4 operand tiles; write this body inline: km.cxx(..., gives=km.Tile)". The
  inline body copies the tiles into `Dst` 0 to 3 in turn and computes the two lerps
  and the third there: rms 3.7e-8, 11.1 us, against the three lerp passes through two
  scratch tensors of `filter.py` at 7.2 + 6.2 + 6.3 = 19.7 us. `out.push(km.cxx(...))`
  is refused ("push one tile of the loop"): name the body's tile first, then push it.
- **A field's rows by command** (`field_rows.py`): y = x W, W of K tile rows held in
  a field, K 4 then 8, 8 cores, 2 tile rows each. (a) A build a size, the class
  constant `ROWS` given at the make: 7.0 and 10.0 us. (b) One build, the field sized
  to 8 rows, W stored with zero rows to 8 and K a word of the command: 6.8 and 9.9
  us, each load 3.7 and 3.2 us; its errors equal (a)'s (rms 3.1e-4, 3.2e-4 from
  float64). (b) needs `handlers={"apply": [km.UPTO(x=8)]}`: without it, an x of
  another width is refused ("x is 4 tiles wide, and the handler's rectangles of x were
  built for 8").
- **A DRAM read in a dynamic run** (`treelet.py`): a scene of two treelets in DRAM,
  one held in L1 at a time. Run 1 walks treelet 0 and bins each ray that reaches
  depth 0 (46.0 us); a plain handler reads treelet 1 from DRAM into every core's
  scene words (5.2 us); run 2 seeds the binned rays and walks treelet 1 (100.3 us).
  16 cores, 16 seeds, depth 3. Binned 128 and walked 1,024 rays, origin sum
  1,694,950, none for want of quota: equal to a Python walk of the same rules; the
  scene after the load is treelet 1 on every core. These runs were measured when a
  core's quota was max_live over the load's ranks (118 on a card): the first run, at
  max_live 4,096 (34 sends a core), counted 773 rays, 157 of them ended for want of
  quota, since run 2's 128 seeds come from bins of 1 to 16 rays and one core makes
  up to 104 sends (Python's count of these rules). At 16,384 (138 a core) none was
  short. The quota is now max_live over the run's ranks (14 for this group): 292 a
  rank at 4,096, above the 104 sends one core makes; 4,096 has not been run since.

## Two cards (`--cards 0,1`)

Cards 0 and 1 as islands; each command's group straddles the cards' edge.

- **denoise.py**: a 14 x 16-tile frame on 224 ranks from rank 6, half on each card.
  The tensors have no placement hint, so each lies wholly on card 0 (the session
  warns "not in proportion to their ranks"). Card 1's ranks read every tile across
  the link: "horizontal on ranks 6..229 moved 448 pages across the link". The shift is exact. The blur is 1.70e-3 from float64,
  where rounding mid to bf16 alone is 1.70e-3 (bound 2.5e-3). us: horizontal
  605.9, vertical 1,722.5 (one card, 4 x 4 tiles: 3.8 and 4.6).
- **fetch.py**: the DRAM form on 8 ranks from 114, 4 on each card: exact, 13,914.1
  us (580.7 on one card). The texture lies wholly on card 0, so card 1's 4 ranks
  read every texel page across the link: "gather on ranks 114..121 moved 4104 pages
  across the link (asks 4100, puts 8)". The held form is not run on two cards.

The fetch with `--place` (cards 0 and 1, 8 ranks, us): the group and tensors on
island 0, 700.5; straddling with no hint, 13,875.2, with 4,104 pages across the link;
straddling with the address pages on their own ranks' card and the texture
`copy=True`, 634.5, with none across. One card: 580.7. The 4,100 single-page asks
moved 17.1 MB in 13,875 us, about 1.2 GB/s, against 40.6 to 48.3 GB/s for bulk
`km::read_pages` of the other island's pages
([islands](../../../docs/reference/islands.md#what-costs-time)).
