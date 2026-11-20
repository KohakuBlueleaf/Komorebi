# The model's weights on the Wormhole Galaxy box

DeepSeek-V4.1-Flash's backbone (40 layers, 384 routed experts a layer, the head) placed on the
31 islands of a Galaxy box (32 Wormhole chips, one left out), converted to the chip's tile formats,
and loaded into the islands' DRAM, each weight found by name.

| File | What |
|---|---|
| `ckpt.py` | the released checkpoint read with numpy: safetensors headers, memory-mapped tensors, the released values of BF16, F32, FP8 E4M3 (+ E8M0 per 32 x 32) and FP4 E2M1 (+ E8M0 per 32) |
| `placement.py` | the placement: which weight is on which island, in which format, at which page of which arena; writes `placement.json` |
| `home.py` | the placement "home": every weight but the routed experts on island 0; writes `placement-home.json` from `placement.json` |
| `measure_formats.py` | what each conversion does to the released values, on real tensors (relative RMS, largest error, share kept exactly) |
| `convert.py` | the converted cache: every placed weight as the bytes its island's arena holds (one file an arena), resumable, a CRC a weight |
| `check_cache.py` | the cache checked: every weight's CRC against the conversion's log, and no block-float negative zero code (a datum the card reads as -inf) |
| `loader.py` | `load(session, names=None, islands=None, verify="sample", resume=False)`: the arenas made and written from the cache, every weight a `Handle` (arena, word, first page, pages, shape, format, island); a read-back check |
| `host.py` | the cache on the host as the values the chip holds: `held(name)` (held shape), `as_released(checkpoint name)` (the checkpoint's shape, for the reference on as-held weights); `select(layers, kinds, formats)` names by layer and kind |
| `load_box.py` | the whole backbone into a 31-island session, checked and timed |
| `try_load.py`, `probe_arena.py` | development: a load of some weights onto a few cards; the largest arena on one card |

## Placement

Island i is the i-th card of `[0..15, 17..31]`. Routed expert e of every layer is on island
e mod 31 (islands 0-11 hold 13 a layer, 12-30 hold 12). The rest of layer L is on its home island
`[12, ..., 30, 4, 5][L mod 21]`. The Engram projections of layers 1 and 14 are on islands 6 and 7, the
head in four slices of 32,320 vocabulary rows on islands 0-3, the final norm on island 0. The Engram
tables and the embedding stay on the host.

Every weight is held in a tile layout as a product's right operand `[in, out]`; a vector `[N]` as
`[1, N]`; `wo_a`'s 8 groups as `[8, 4096, 1024]`. An island holds one tensor (an arena) per format it
uses, of one-tile pages; a weight is a run of pages of it, its tiles in row-major order. That is 102
tensor-table entries for the whole backbone instead of one a weight.

## Home

A second placement, selected by name: `GALAXY_PLACEMENT=home` (or
`load(..., placement=loader.placement_path("home"))`) takes `placement-home.json` in place of
`placement.json`; the loader, `host.held`, `host.as_released` and `host.select` follow it. With
the variable unset the default placement is used, unchanged.

Every weight but the routed experts is on island 0: each layer's attention projections, norms,
sink, mixing, router, shared expert, compressor and indexer, the Engram projections, the head's
four slices and the final norm. A piece that submits its commands on island 0's ranks then reads
all of them from its own DRAM. Routed expert e of every layer is on island 1 + e mod 30 (islands
1-24 hold 13 a layer, 25-30 hold 12).

Each weight is placed once. A weight the default holds in two formats is held in `bfp8_b`, but
the Engram `wkv` and the indexer's `wq_b` in `bfloat16`, as their pieces read them. Both copies of
every FP8 weight would be 20.3 GB, more than an island holds.

| Island | Holds | Bytes | Left of 12 GiB |
|---|---|---|---|
| 0 | every weight but the routed experts, and the relay | 9,384,378,368 | 3,500,523,520 |
| 1-24 | 13 routed experts a layer | 10,351,411,200 | 2,533,490,688 |
| 25-30 | 12 routed experts a layer | 9,555,148,800 | 3,329,753,088 |

Island 0, by kind (bytes): attention projections (`bfp8_b`) 5,381,160,960; shared expert
(`bfp8_b`) 1,504,051,200; head 1,323,827,200; Engram projections 630,456,320; mixing (`float32`)
210,370,560; router and its bias 159,252,480; compressor and indexer 123,895,808; norms and sinks
31,457,280; the relay 19,906,560. The put batches' landing (1,023,426,560 B at 31 islands, a
formula) and the shared load's reserve come out of what is left.

The relay (`relay@bfp4_b`) is NOT a weight of the model. It is pages the experts piece reads at
weight 0 on an island of its tree that holds no expert: such an island's command reads pages 0 to
34,559 of that island's `bfp4_b` arena, and its result is multiplied by 0. The home island is in
every tree and island 0 holds no expert, so the placement gives island 0 one run of 34,560
`bfp4_b` pages first in that arena (a copy of layer 0's expert 0, so the values are finite). It
is kept apart from the weights (`relay` in `placement-home.json`, not in `entries`), and the
loader adds it to any load that takes a `bfp4_b` weight.

`placement-home.json` is made from `placement.json` and keeps each weight's place in the
converted cache (`cache`: its island and first page there), so the cache is not converted again:

```
python home.py $GALAXY_LLM    # placement-home.json beside placement.json
```

On a session of fewer islands the loader folds the placement this way: island 0 stays on island
0, and island p > 0 goes onto island 1 + (p - 1) mod (islands - 1). So on 8 islands the experts go
onto islands 1-7.

## Formats, against the released values (real tensors)

| Stored | Held as | Error |
|---|---|---|
| BF16 | bfloat16 | exact |
| F32 | float32 | exact |
| FP8 E4M3 + E8M0 per 32 x 32 | bfloat16, and bfp8_b | bf16 exact; bfp8_b relative RMS 3.4e-3 to 4.4e-3 |
| FP4 E2M1 + E8M0 per 32 | bfp4_b | relative RMS 0.107 to 0.112 (a block of 16 whose largest value is 4 or 6 times its scale loses 0.5 and 1.5 times it); bfp8_b would be exact but does not fit the box's DRAM |

## Running

```
export GALAXY_LLM=<folder>        # holds ckpt/ (the released checkpoint), placement.json, cache/
python placement.py $GALAXY_LLM/ckpt $GALAXY_LLM                        # placement.json
python convert.py $GALAXY_LLM/ckpt $GALAXY_LLM/placement.json $GALAXY_LLM/cache/v1
python load_box.py --report load.json                                   # whole box, flock -x
```

## The converted cache

One file an arena, `i{island:02d}-{dtype}.bin`, page p at byte p x tile bytes; a log an island,
`i{island:02d}.done`, a line a weight written: name, CRC-32 of its bytes, its relative RMS against the
released values (measured for every weight but the routed experts, of which a sample; -1 where not
measured). Resumable: a weight in the log is not converted again.

## The loader

`load(session, names=None, islands=None, verify="sample", resume=False, threads=None)`:

- `names`: None (all: a session of the 31 islands), a list of names (`host.select`), or a predicate on
  a name. A session of fewer islands than the placement's takes only `names`; the arenas then hold
  only the named weights and `h.first` is the page in that session's arena.
- `islands`: {placement island: session island}; default placement island p onto session island
  p mod (the session's islands); for "home", the fold in "Home".
- `verify`: "sample" reads back the first, middle and last page of every weight and compares them
  with the cache; "full" every page; None nothing.
- `resume`: a 256 MiB range whose first and last pages already hold the cache's bytes is not
  written again (a reload into the same allocation).
- `threads`: islands written and checked at once (default 8). The whole box, 328.5 GB: 31 at once
  333.9 s, 8 at once 215.4 s, every sampled page equal both times. A range under 80 KiB goes the
  device API's way, which writes only whole tensors, so a short tail is written as the last range of
  80 KiB (its first pages again). Per GB on one island: about 0.57 s the card's transfer, 0.65 s the
  write's negative-zero check, 0.24 s the cache read.

A weight is pages `first .. first + pages - 1` of its arena (one tile tensor of one-tile pages an
island and chip format), its tiles in row-major order of its held shape.

## On the host

```python
from host import held, as_released, select

held("layers.3.ffn.experts.7.w1@bfp4_b")  # float32 [5120, 2304], as held
as_released("layers.3.ffn.experts.7.w1.weight")  # float32 [2304, 5120], as held
select(layers=[3], kinds=["attn."], formats=["bfp8_b", "float32"])
```

`as_released` gives a checkpoint tensor in its own logical shape ([out, in]; a vector [N]; FP4
unpacked) with the values the chip holds, so the reference can run on the as-held weights; of a
weight held in two formats it takes the compact one (bfp8_b for FP8) unless `dtype` names the other.
