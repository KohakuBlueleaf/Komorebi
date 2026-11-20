# A layer's own attention on one Wormhole chip (DeepSeek-V4.1-Flash, pieces 7, 8, 9)

The query path (wq_a 5120 -> 1280, q_norm, wq_b 1280 -> 64 heads x 512), the key-value path
(wkv 5120 -> 512, kv_norm: one key-value head for the 64 query heads), the rotary positions
on the last 64 of the 512, the sliding-window attention over the layer's last 128 tokens
with the attention sink, and the output path (wo_a over 8 groups, wo_b): `inference/model.py`
lines 614-791. One command activation is one row tile [32, K]: a decode token in row 0, or
a prefill block of 32 tokens.

| file | what |
|---|---|
| `proj.py` | class `Projections`: the five products (bounds' K split: a group of `split` ranks for each block of Y columns, fp32 sums up a tree, HiFi2, weights bfp8_b) and the two norms (row form, one core) |
| `window.py` | class `Window`: rope as a swap product (`rope_swap`) and an element-wise `x c + (x P) s` (`rope`); the window insert (`insert_move`, `insert`); the rope tables (theta 10000 without YaRN for compress ratio 0, compress_rope_theta with YaRN otherwise) |
| `attend.py`, `flash_local.hpp` | class `AttendLocal`: experiments/language/attention_class's ATTEND (projects/attention's compute) plus a bias tile on each score tile; the 64 heads are the query tile rows of ONE head, so the key-values are broadcast once for all of them; the mask table |
| `layer.py` | `Shared` (tables), `LayerTensors` (a layer's weights, sink tiles, window), `Stack` (the three classes in the runtime's form, loaded as a dict, the cores switching program between them), `Layer` |
| `piece.py` | the driver's call interface (`projects/galaxy/driver/interface.py`): `make(session, weights, layers)` |
| `verify_chip.py` | galaxy.driver.verify's comparison on a one-chip session or on joined islands |
| `check_*.py` | each part against float64, random values |

## The key list

The 64 query heads are the query tile rows of one head (a tile row a head, its 32 rows the
tokens), so every head shares the one key-value head's broadcast, and q and o keep wq_b's
page order ([32, 64 x 512] row tile = [64 x 32, 512]). The plan's sizes: one query tile row
a job (64 rows on the 72 cores of the 8 x 9 broadcast rectangle, and the L1 of a program
beside the others), 6 key tiles a block (block 0 the sink's and the window's), a broadcast
length of 9 (a whole column).

A layer's key list (K = V) is one tensor [22 x 32, 512] bf16: key
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
with a product beside copied operands is refused by the build.

insert_move then insert: tile row `row` of the window becomes old m + e kv (kv's tile row
0, a rank a column tile; e kv into a scratch `moved`), e and m a variant's tiles of the
insert tables: variant r < 32 puts kv's row 0 into row r (e a 1 at (r, 0), m zero in row r
and one elsewhere), variant 32 the whole tile (e the identity, m zero). Every product and
multiply is of ones and zeros, so the window holds kv's words.

## Compressed layers

Layers 2-39 attend over the window and the selected compressed entries. Until the compressed
attention's gather writes them on the chip, the piece takes them from the host:
`piece.compress(layer, x, qr, start_pos, offset)` returns what the reference's
`compress_kv` returns (the compressed entries [1, n, 512] and, a row, the chosen ones as
offset + j, -1 for none; offset the window's rows the reference concatenates before them:
the block's 32 at position 0, else 128). A step runs wq_a and q_norm, reads qr back, calls
it, writes the chosen entries (in entry order, at most 512) into key tiles 6 on and their
mask, then runs the rest with the call over 6 + ceil(m / 32) key tiles. In the driver it is
the reference's own `Path.compress_kv` (its compressor and indexer keep their state on the
host); `verify_chip.py` gives golden's `compress_kv` outputs of the same step.

The `sparse_attn` site builds the same key list from its kv ([window rows | compressed
entries], the window's the block's rows at a prefill, the 128 slots at a decode): the window
rows from key 32, only the entries topk_idxs chooses (at most 512) from key tile 6. A check
of a prompt recorded from a decode on (the 1,100-token prompt) seeds the layer's window from
the recorded state of that step (`piece.set_window`).

## Weights

The linears come from the weights loader (`projects/galaxy/weights`, its subset load):
`make` takes the driver's W when it holds the layers' bfp8_b linears on this chip, else loads
them itself (`load(session, names)`). A weight is pages `first` .. of its arena, a tile
tensor of one tile column, its tiles in row-major order of its held shape [in, out] (wo_a
[8, 4096, 1024]: the stacked [8 x 4096, 1024]). A product reads its W block as the walk
`w.tiles[at + k0 * C + c : at + k1 * C + c : C, 0:width]` (C the weight's tile columns), the
first page `at` a command word. The norms' and the sink's values are read from the same
converted cache on the host (`host.held`) and written in the norm's page layout. A check's
own random weights are a tile tensor seen as one column, `at` = 0.

## Ranks a command takes

Every command runs on ranks 0 .. n - 1 of the first island, n at most 75: a Wormhole card
joined to others keeps 75 of its 79 cores as ranks (4 are link cores). No reduce's group
crosses an island, so each is marked `local=True`.

| class | handler | ranks |
|---|---|---|
| `Projections` | wq_a | 75 (5 blocks of 8 columns, K split 15) |
| | wq_b | 64 (64 blocks of 16 columns, no split) |
| | wkv | 74 (2 blocks of 8 columns, K split 37) |
| | wo_a | 64 (8 groups of 2 blocks of 16 columns, K split 4) |
| | wo_b | 60 (20 blocks of 8 columns, K split 3) |
| | norm_q, norm_kv | 1 |
| `Window` | rope_swap, rope of q and o | 64 (a head's tile row each) |
| | rope_swap, rope of kv | 1 |
| | insert_move, insert | 16 (a column tile each) |
| `AttendLocal` | attend | 72 (the 8 x 9 broadcast rectangle) |

The dict's load puts the attention on the rectangle, the window and the projections on the
first island's ranks left (3 when joined, 7 on one card), and the projections also on every
other island's ranks; a switch loads a program on the first island's ranks.

A joined island's L1 holds the service overlay in a slot of its own (58,816 B), so the
broadcast holds one block in flight (`attend.BROADCAST_WINDOW = 1`, 384 KiB), sized for a
joined island's L1; it may grow back to 2 when that overlay can share the reader's slots.
A joined session opens with `kernel_bytes=104_448` (`piece.KERNEL_BYTES`, the
Projections' kernels): `make` raises `KernelBufferError` on a larger kernel buffer.

## Why three programs, not one merged one

The three classes together need more circular buffers than the 26 indices a program has
(the build refuses at the window's rope: 9 pairs and 6 singles taken before the attention's).
Each class is therefore made in the runtime's form, and the three are loaded as a dict. A
step switches program 4 times (projections, window, attention, window, projections), at
13 to 41 us a switch. A switch waits until every buffer of both programs is empty, so the
attention builds its constant tiles in every command and frees them at the end.

## Measured (one chip, 79 ranks, AICLK 1000 MHz)

A step's commands add up to: prefill block 1.22 to 1.29 ms, decode token 1.20 to 1.21 ms
(random weights, `check_layer.py`). Of a decode: projections 907 us (wq_a 68, q_norm 42,
wq_b 251, wkv 53, kv_norm 23, wo_a 211, wo_b 263), attention 113 us, rope of q, kv and o
98 us, insert 18 us, program switches 64 us. The key list of the window and 512 entries (`check_attend.py`): decode 247 us, 200
entries 180 us, a prefill block with 16 entries 127 us.

A step's commands are queued at once (`Stack.queue`: each exclusive, sent to the card every
4) and waited for once; a compressed layer waits once in the middle, for q_norm, to read qr
for the host's hook. `piece.attn_tensors` runs the attn site on an x and a y already on the
chip. Wall of one decode call (layer 0; layer 2 with golden's entries in brackets): every
command waited for 6.4 ms; queued with the host's write of x and read of y 4.1 (5.9) ms;
queued on chip tensors 2.75 (4.6) ms, of which the host's 16 submits take 1.6 ms (about
0.1 ms each in Python) and, in layer 2, the look in the middle 1.4 ms (the wait, the read of
qr 0.33, the hook, the write of the entries and their mask 0.57).

On two joined islands (two cards, one link, `verify_chip.py --cards`), with the host's
write of x and read of y: a decode call 4.8 to 5.5 ms (layer 0) and 6.6 to 6.9 ms (layer
2), its commands 1.46 ms (1.15 to 1.21 ms on one card).

Every product runs at HiFi2 (the box's one setting). The checks compare against float64 of
the stored values and the recorded reference; a result is right or a clear outlier is named.

Known difference, for the driver's README: activations stay bf16. The reference rounds each
E4M3 linear's input, o before wo_a and the stored window to E4M3 in blocks of 32; the chip
does not.

Not here: making the compressed entries and choosing them (pieces 10-12); more
than one island; the layer's home island.
