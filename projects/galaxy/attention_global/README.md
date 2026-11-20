# The global attention of DeepSeek-V4.1-Flash on a Wormhole chip

The global part of the model's attention: the compressor that pools `ratio` tokens
into one key-value latent of 512 (layers 2, 8, 14 at ratio 2, layer 20 at ratio 1), the
indexer that scores the compressed entries and keeps the best 512 (Full layers 2, 8, 14,
20; Reindex layers 24, 28, 32, 36 inside layer 20's candidate blocks; Reuse layers take the
latest choice), and the attention of the 64 query heads over the window and the selected
entries in one softmax with the head's sink (model.py 429-611, 775-782).

| File | What |
|---|---|
| `piece.py` | the piece of the driver's interface: `make(session, weights, layers)` serves the sites `compressor`, `indexer` and `sparse_attn`; `make_compressor`, `make_indexer` and `make_sparse_attn` serve one site each and load only its class. The classes (`Compress`, `Index`, `Attend`, runtime form, `slots(count=1, service="shared")`) share one run, the cores switching program between sites |
| `compressor.py` | step 1: the compressor on one chip (synthetic weights), against float64 |
| `indexer.py` | step 2: the indexer's decode on one chip and the host's top-k, against float64 |
| `attention.py` | step 3: the decode attention over 128 window keys and 512 gathered entries (the gather by 0/1 selection tiles), against float64 |

## What runs where (version 1)

On the chip: every product (the compressor's projection, the indexer's wq_b and
weights_proj, a decode query's index scores, the gather, the attention's Q K^T and P V),
the compressor's gated pool of a group's two slots (ratio 2) or bf16 rounding (ratio 1)
and its RMSNorm, and the attention's softmax with the sink. Every class runs with one set
of compute options: HiFi2, float32 Dst, full Dst sync. On the host: the E4M3 rounding of
qr before wq_b, the index keys (wk, k_norm, rotary, FP4) and the index query's rotary
and FP4 rounding, the top-k and the candidate blocks, and the prefill's index scores.

## The handlers

- `Compress` (`piece.py`): `linear_x` Y = X W for one row tile of X (K = 160 tiles),
  groups of `split` cores each a 1/split of K, the group's sums up a tree, the leader
  writing `width` float32 tiles at pages y0 + group x width (the state's slot row tile);
  `pool` (ratio 2) core c: tile c of the pooled latent from the state's row tiles 0 and
  1 (kv in columns 0 to 511, scores in 512 to 1023), the softmax of the two slots'
  scores, written bf16; `rounded` (ratio 1) core c: tile c of the slot, written bf16;
  `norm` one core: RMSNorm of each row of the pooled latent times the weight.
- `Index` (`piece.py`): `linear_x` for wq_b (qr padded to K = 160 tiles) and
  weights_proj (its weight padded to 4,096 columns); `column` w^T as I W^T by a cast of
  W turned; `score` core r takes entry row tiles r, r + size, ...: relu(K Q^T) w^T into
  page r; `compact` column 0 of 32 score pages into one packed page (sum_b S_b E'_b, E'_b
  a 1 at (0, b)).
- `Attend` (`piece.py`): `qk` S tile (b, k) = q's row tile b times kt's column tile k
  (q's tile column 16 selects a mask row of kt's row tile 16); `softmax` core b: the row
  of keys + 2 tiles of S (the sink tile, a masked tile) times the scale, into P (bf16);
  `pv` O tile (b, c) = P's row tile b times the keys' column c; `gather` below.
- The step scripts' classes (`compressor.py`, `indexer.py`, `attention.py`) are the
  same math on synthetic weights, each checked against float64; `indexer.py` adds
  `qrel` (q's 32 index heads as the rows of a [32, 128] matrix, times the position's
  rotation) and `append` (a new row into a cache's row tile), `attention.py` `unrope`
  (o's last 64 dims rotated back by the query's position).

## The gather and the attention's key list

`Attend.gather` copies the chosen rows of the compressed cache into a key list: key list
row tile `e0 + b`, four column tiles a core, is the product of a 0/1 selection tile row
(row j holds a 1 at the cache row entry j names; an unreachable choice, -1, a zero row)
and the cache. With `e0 = ENTRY0` (6) it writes the attention's key list in its layout:
tile 0 the sink, tiles 1 to 4 the window, tile 5 a prefill block's own keys, tiles 6 to
21 the selected entries (entry j in row 192 + j). `entry_mask(seen)` gives the entries'
mask beside it ([16 x 32, 32]: tile t, row i, column c 0 where query row i sees entry
32 t + c, else -1e30). In a step of a compressed layer: the compressor (at decode only
when the token completes a group), the indexer, the gather and the mask, then the
attention over the key list.

## Verified against golden/ (driver's verify, real weights, one chip)

Prompts en and zh, prefill (32 tokens) then decode0 and decode1; layers 2, 14 (Full,
ratio 2), 20 (Full, ratio 1, candidate source), 3 and 21 (Reuse), 24 (Reindex). The long
prompt (1,100 tokens, decode0 and decode1; layers 2, 3, 20, 21, 24), where the indexer
chooses 512 of 550 (layer 2) and of 1,100 (layers 20, 24), the index key caches started
from the recorded ones (no prefill is recorded there).

| Site | en, zh (prefill, decode0, decode1) | long (decode0, decode1) |
|---|---|---|
| `compressor` (2, 14, 20) | right (decode0 at ratio 2: None, as the reference) | right (layers 2, 20) |
| `indexer` (2, 14, 20, 24) | equal, 24 of 24 | layer 20 equal; layers 2 and 24: the same choice up to entries tied at the 512th score (below) |
| `sparse_attn` (2, 3, 14, 20, 21, 24) | right | right (128 window keys + 512 gathered entries) |

Too close to call: at layers 2 and 24 of the long prompt many entries share the 512th
score's bf16 value exactly, and the 2 to 9 entries that differ are among them (one is a
bf16 step from it). The step scripts check what golden/ cannot reach: `compressor.py`
the chain on synthetic weights, `indexer.py` the choice at 1,000 entries, `attention.py`
the gather (exact).

Times on the chip a call (medians of the logged commands, decode): see the verify log's
`piece:` lines; a program switch costs 140 to 340 us of the host's wait, a read over this
machine's PCIe about 1 ms. A make's time is the classes' build and load: `Attend` about
140 s (its six handler sets), `Compress` 13 s, `Index` under 2 s with the build cache
warm; the weights' preparation under 1 s. A driver that needs only the compressor and the
indexer makes those (`make_compressor`, `make_indexer`).

## Known limits

- The softmax of a decode call runs on 2 cores (one per 32 heads). The attention of the
  model is the local attention's single call over the key list; this one checks the math.
- The gather uploads the whole compressed cache each call (the chip keeps no copy between
  calls yet).
- The piece's sites run in their own run on the piece's own chip; the selected rows reach
  the local attention's key list through the host until both share one load.
