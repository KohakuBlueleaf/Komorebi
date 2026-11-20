# Galaxy parts: the norms, the residual streams, Engram and the head

Pieces 4, 5, 6, 17 and 18 of DeepSeek-V4.1-Flash on Wormhole. Every class is built in the
runtime form with one set of compute options (`COMPUTE`: HiFi2, DST in full sync, fp32
DST), so that one load can hold them all and switch between them.

## The norms (`norms.py`)

y = x rsqrt(mean(x^2) + eps) w over a row of 5,120 values (attn_norm, ffn_norm, the final
norm), 1,280 (q_norm) or 512 (kv_norm), one tile row of tokens (row 0 the decode token),
bf16 in and out. One class serves every width: a row of T tiles is split over T / EACH
cores (the group given at the submit): each core sums the squares of its own EACH tiles,
the sums are all-reduced over the group, and each core writes its own EACH tiles
normalized as x rsqrt(sum + n eps) (w sqrt(n)), the same as the reference's form with the
mean's 1 / n moved out of the root, so that no constant depends on the width. The weight
row w sqrt(n) is held float32 (the norm reads its row by `.row()`), n eps in a float32
tensor of EACH tiles.

## The head and the choice (`head.py`)

logits = x W^T over the 129,280 ids (W bf16, 5,120 x 129,280 in the tile order: 4,040 tile
columns), and the largest logit's id (of equal largest logits the first). Each island
holds one slice of W as the loader holds it (head.sN: 1,010 tile columns, W^T in the tile
order). Three commands an island, all on its own ranks:

1. `logits` (73 cores): a core's 14 columns (13 on the last 12 ranks) over the whole K in
   two passes, x and W streamed in step, the sums in DST, written to the core's 14 slots
   of the float32 logits tensor y.
2. `choose` (73 cores): a core's largest logit (a row function over its WIDTH tiles),
   all-reduced by max over the group; then each of its logits that equals the group's
   largest gives its id (from the tile tensor of ids), any other NOWHERE, into the same
   tiles of c; the group's largest into page `rank` of g.
3. `least` (73 cores): a core's smallest id of c (as the largest of the negated ids),
   all-reduced by max; the leader writes it, negated, to r.

The host reads g's and r's first page of each island and keeps the largest logit (of equal
ones the smallest id). Every row of the tile row (32 tokens) is chosen alike. The two
classes are one list load (`beside`): one shared layout of buffer indices.

## The four residual streams and their mixing (`mhc.py`)

The reference's `Block.hc_mixes`, `hc_pre` and `hc_post` (inference/model.py), s tokens (1
or up to 32) in one tile row.

- `mixes`: mixes = (x W^T) rsqrt(mean(x^2) + eps) over the flattened 4 x 5,120 = 20,480
  values of a token (x bf16, W = hc_fn held [20,480, 24] float32, padded to 32 columns);
  40 cores each take 16 of the 640 tile rows of K: each core's product sums to its page of
  m, their squares' sums all-reduced and the rsqrt to m's page 40 by the leader; `gather`
  (one core) adds the 40 pages and multiplies by the rsqrt.
- `pre` and `post`: each a matrix product out = D Z on the matrix unit, Z the streams
  stacked (tile row m of Z [5 x 32, 5,120] bf16 is stream m: the residual's 4, then the
  sublayer's output x) and D float32 tiles that are diagonal (a coefficient a token): pre:
  out [32, 5,120] = sum_m pre_m Z_m (D's fifth tile zero); post: out row block k = post_k x
  + sum_j comb[j, k] r_j. 40 cores, 4 tile columns each, the sums in DST.

The 24 mixes become pre, post and comb on the HOST in this version (hc_split_sinkhorn:
sigmoids and 20 Sinkhorn steps on 24 values a token). That costs a host round trip twice a
layer (once around the attention, once around the feed-forward), so it is the first thing
to move onto the box when the token loop is timed.

## Engram's projections (`engram.py`)

The reference's `Engram.forward` after the lookup (site `engram_mix` of layers 1 and 14):
the table rows are looked up on the host, everything after them runs here.

- `key_value` (50 cores): kv = rows W (rows [s, 24 x 256 = 6,144] bf16, W = wkv held
  [6,144, 25,600] bf16, exact of the released E4M3 x E8M0), a core's 16 tile columns over
  the whole K in two passes of 8, the sums in DST, rounded to bf16 as the reference's
  Linear. The reference first rounds its input to E4M3 in blocks of 32 with a power-of-two
  scale: the looked-up rows are E4M3 values times a power of two already, so that rounding
  leaves them unchanged and is not done here.
- `dots`, `norms`, `gate`, `mix` (4 groups of 10 cores, one a stream c): the key of stream
  c is kv's tile columns [160 c, 160 c + 160), the value [640, 800); a core takes 16 tiles
  of the row. `dots` writes h q k_w key to p (h the stream, q and k_w the released q_weight
  and k_weight rows); `norms` sums h^2 and key^2, all-reduced over the group, and writes
  rsqrt((mean h^2 + eps)(mean key^2 + eps)) / sqrt(5120) to its page of a; `gate` sums p,
  all-reduced, and writes the signed sqrt of dot = that sum times a's factor (|dot| at
  least 1e-6) to its page of g; `mix` writes h + sigmoid(that) value in float32 (rsqrt and
  sigmoid are two setups of the vector unit: two functions), which the host rounds to bf16
  as the reference's .to(bfloat16).

## The driver's interface (`piece.py`)

Sites: the norms (attn_norm, ffn_norm, q_norm, kv_norm, final_norm), the residual streams'
mixing (hc_attn_mixes, hc_ffn_mixes, hc_attn_pre, hc_ffn_pre, hc_head_pre, hc_attn_post,
hc_ffn_post), Engram after the lookup (engram_mix) and the head and the choice (head,
choose: the vocabulary in four quarters one after another on the chip; on the box one
quarter an island).

One instance a part: `make_norms`, `make_mixing`, `make_engram` and `make_head` each give
a piece of that part's sites only, which loads that part once and keeps it. `make` gives
every site on one chip: a session holds one loaded run, so it keeps one part loaded at a
time and changes part when a site needs another (the old part's run stopped and its
tensors freed). Every weight a part reads is written when the part is made (all its
layers) and never again: a call only points the commands at its layer's tensors.

The mixing's time is the host's: in the driver, 240 calls a decode step cost 1.02 s
(4.3 ms a call), almost all of it the round trips of each call's inputs and outputs and
of the split on the host. The one load with its tensors kept on the box removes those
round trips next.

The weights come from the weights loader (`projects/galaxy/weights/`). The large ones are
read by the kernels from the loader's arenas: the head's four slices `head.s0` .. `head.s3`
and the Engram layers' `wkv`, each a weight's pages from its `first` page of a one-column
arena (the driver's W when it holds them on the chip, else a subset load in `make`). The
small ones (the norms' weights, `hc_*_fn`, `hc_*_scale`, `hc_*_base`, Engram's `q_weight`
and `k_weight`) are read on the host from the same converted cache (`host.held`) and
written to the piece's own tensors. A slice has 1,010 tile columns over 73 cores of 14
slots: rank r starts at column 14 r - min(r, 12), so the first 12 ranks' last slot repeats
the next rank's first column (with its id: a repeat cannot change the choice).
`piece.spent` is the last call's box microseconds by command.
