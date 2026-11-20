# A layer's own attention on one Wormhole chip (DeepSeek-V4.1-Flash, pieces 7, 8, 9)

The query path (wq_a 5120 -> 1280, q_norm, wq_b 1280 -> 64 heads x 512), the key-value path
(wkv 5120 -> 512, kv_norm: one key-value head for the 64 query heads), the rotary positions
on the last 64 of the 512, the sliding-window attention over the layer's last 128 tokens
with the attention sink, and the output path (wo_a over 8 groups, wo_b): `inference/model.py`
lines 614-791. One command activation is one row tile [32, K]: a decode token in row 0, or
a prefill block of 32 tokens.

| file | what |
|---|---|
| `proj.py` | class `Projections`: the five products (bounds' K split: a group of `split` ranks for each block of Y columns, fp32 sums up a tree, HiFi4, weights bfp8_b) and the two norms (row form, one core) |
| `window.py` | class `Window`: rope as a swap product (`rope_swap`) and an element-wise `x c + (x P) s` (`rope`); the window insert (`insert_move`, `insert`); the rope tables (theta 10000 without YaRN for compress ratio 0, compress_rope_theta with YaRN otherwise) |
| `attend.py`, `flash_local.hpp` | class `AttendLocal`: experiments/language/attention_class's ATTEND (projects/attention's compute) plus a bias tile on each score tile; the 64 heads are the query tile rows of ONE head, so the key-values are broadcast once for all of them; the mask table |
| `layer.py` | `Shared` (tables), `LayerTensors` (a layer's weights, sink tiles, window), `Stack` (the three classes in the runtime's form, loaded as a dict, the cores switching program between them), `Layer` |
| `piece.py` | the galaxy driver's call interface: `make(session, weights, layers)` |
| `verify_chip.py` | galaxy.driver.verify's comparison on a one-chip session |
| `check_*.py` | each part against float64, random values |

## The key list

The 64 query heads are the query tile rows of one head (a tile row a head, its 32 rows the
tokens), so every head shares the one key-value head's broadcast, and q and o keep wq_b's
page order ([32, 64 x 512] row tile = [64 x 32, 512]). The plan's sizes: one query tile row
a job (64 rows on the 72 cores of the 8 x 9 broadcast rectangle, and the L1 of a program
beside the others), 6 key tiles a block (block 0 the sink's and the window's), a broadcast
length of 9 (a whole column).

A layer's key list (K = V) is one tensor [22 x 32, 512] bf16 (the driver's key list): key
tile 0 zero (the sink's), key tiles 1-4 the window's ring (position p in slot p mod 128, row
32 + p mod 128), key tile 5 a prefill block's own keys, key tiles 6-21 up to 512 selected
compressed entries (written by the compressed attention's gather), whose visibility is a mask [16 x 32, 32] of
0 / -1e30 a query row and entry, copied as the bias of those key tiles. The call reads key
tiles 0 .. 5 + ceil(n / 32) for n entries (0 for layers 0-1). Key tile 0's bias holds sink_h / scale in column 0 (and -1e30 in the others), and
its K and V rows are zero, so it adds exp(sink_h - m) to the row sum and nothing to O. Which
keys a row sees is a mask variant of one table (decode at pos: min(pos, 127); a prefill
block at s, s a multiple of 32: 128 + s / 32 mod 4, + 4 when s >= 128). A decode step puts
the token's key into its slot first and attends over key tiles 0-4. A prefill step puts the
block's keys into tile 5, attends over tiles 0-5, then copies them into their ring tile.

## The rotary positions and the insert

Every tensor of window.py is seen 512 wide (16 tiles a row): q and o as [64 x 32, 512]
(tile row h: head h's 32 tokens, wq_b's [32, 64 x 512] in the same pages), kv as [32, 512].

rope_swap then rope rotate the last 64 of the 512 columns (tiles 14 and 15) of `heads` tile
rows (a rank a tile row), adjacent pairs taken as complex numbers (model.py
apply_rotary_emb): x c + (x P) s, P the tile that swaps adjacent columns (its product into
a scratch y), c and s the position tables' tiles (s holds -sin in even columns and sin in
odd ones; the inverse rotation's table the opposite), written in place. Tile row t of a
table holds positions t .. t + 31, so a command's row 0 (a decode token) or its 32 rows (a
prefill block) start at t. The product and the multiply are two handlers: a tile function
with a product beside copied operands is refused
([hazards](../../../docs/reference/hazards.md#a-tile-function-with-a-product-and-an-operand-copied-beside-it)).

insert_move then insert: tile row `row` of the window becomes old m + e kv (kv's tile row
0, a rank a column tile; e kv into a scratch `moved`), e and m a variant's tiles of the
insert tables: variant r < 32 puts kv's row 0 into row r (e a 1 at (r, 0), m zero in row r
and one elsewhere), variant 32 the whole tile (e the identity, m zero). Every product and
multiply is of ones and zeros, so the window holds kv's words.

## Why three programs, not one merged one

The three classes together need more circular buffers than the 26 indices a program has
(the build refuses at the window's rope: 9 pairs and 6 singles taken before the attention's).
Each class is therefore made in the runtime's form, and the three are loaded as a dict. A
step switches program 4 times (projections, window, attention, window, projections), at
13 to 41 us a switch. A switch waits until every buffer of both programs is empty, so the
attention builds its constant tiles in every command and frees them at the end.

## Measured (one chip, 79 ranks, AICLK 1000 MHz, each command waited for)

Random weights, against float64 of the stored values (`check_layer.py`, prefill blocks at 0, 32,
64, 96 and 128, then decodes at 160 and later; both rope kinds): y rel. RMS 5.1e-3 to 6.1e-3.
A step's commands add up to: prefill block 1.22 to 1.29 ms, decode token 1.20 to 1.21 ms. Of a
decode: projections 907 us (wq_a 68, q_norm 42, wq_b 251, wkv 53, kv_norm 23, wo_a 211,
wo_b 263), attention 113 us, rope of q, kv and o 98 us, insert 18 us, program switches 64 us.
The host's round trip between commands is not counted.

Key list of the window and 512 entries (`check_attend.py`, random, 64 heads): decode 247 us
(22 key tiles), 200 entries 180 us, a prefill block with 16 entries 127 us; rel. RMS 2.4e-3
against float64, largest error 1.8% of the RMS.

All products are at HiFi3: HiFi4 on Wormhole gives rare power-of-two errors (seen in this
project's checks).

Real weights (the checkpoint's E4M3 x E8M0, held as bfp8_b) against the reference's golden
outputs (en). The reference rounds each E4M3 linear's input (and o before wo_a, and the stored
window) to E4M3 in blocks of 32. This chip keeps bf16. A float64 product of golden's input
without that rounding differs from golden's output as the chip does (wq_a layer 0: 1.36% on
the host, 1.37% on the chip).

Known difference from the reference: activations stay bf16. The reference rounds each
E4M3 linear's input, o before wo_a and the stored window to E4M3 in blocks of 32; the chip
does not (the linears' 1.1-3.6% against the golden outputs and the attn chain's 4-5% are this).

Not here: making the compressed entries and choosing them (pieces 10-12, the compressed attention); more
than one chip; the weights loader's arenas (the piece converts the checkpoint itself).
