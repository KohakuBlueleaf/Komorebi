# Galaxy experts: pieces 13, 14 and 15

This folder holds the mixture-of-experts feed-forward of one layer of DeepSeek-V4.1-Flash on
the Wormhole Galaxy islands: the router (piece 13), the shared expert (14) and the routed
experts (15). It follows the driver's call interface (`projects/galaxy/driver/interface.py`):

```python
from projects.galaxy.experts import piece

p = piece.make(session, W, layers)  # W: the weights loader's handles
weights, indices = p("gate", L, x)  # x [s, 5120] bf16 -> [s, 6] f32, [s, 6] i64
y = p("shared", L, x)  # -> [s, 5120] bf16
y = p("routed", L, x, weights, indices)  # -> [s, 5120] f32
y = p("ffn", L, x)  # [1, s, 5120] -> bf16(routed + shared)
```

`p.calls` keeps each call's times: wall clock, the load, the switches, host writes, the
host's wait, and the cards' command time of each step in microseconds. With
`EXPERTS_LOG=1`, each call also prints them as a JSON line.

A layer whose weights were not loaded is refused by name (`KomorebiError`), at `make` or
at the call, before anything reaches the cards.

The verify tool runs it like this on four chips joined as four islands (the loader puts
placement island `p` on session island `p` mod 4):

```bash
TT_VISIBLE_DEVICES=20,21,22,23 python -m projects.galaxy.driver.verify \
    --golden <golden> --piece projects.galaxy.experts.piece:make \
    --cards 20-23 --load "3@ffn." --layers 3 --prompts en --steps prefill,decode0
```

It serves the sites gate, shared, routed and ffn of every layer it was made with. Every
product runs at HiFi2 with a float32 Dst in full sync, in the class and in the tree.

| File | What it is |
|---|---|
| `piece.py` | the sites: the tensors on the islands, the lists, the commands |
| `moe.py` | the class `Moe`: the router and the experts, reading the weights where the loader put them |
| `tree.py` | the block down a tree of the islands and the sums up it (hand kernels in `kernels/`) |
| `kernels/` | the tree's reader, writer and compute kernels (the runtime's form) and their command words (`tree.h`) |
| `check_piece.py` | the sites against golden, and with `--stored` against float64 of the stored pages |

## The weights where the loader put them

Each island has one arena per chip format: a tile tensor with one tile column. A weight is a
run of pages from `first`, holding the row-major tiles of its held `[in, out]` shape. To read
tile row `r`, columns `c .. c + w`, of a weight with `C` tile columns, a handler walks the
arena with a stride: `arena.tiles[first + r * C + c : ... : C, 0 : w]`. That reads `w`
contiguous pages a row, with rows `C` pages apart. The stream's count is a constant
(`stop = start + N * C`), so a product can pair it with the activations' stream.

A list place is one table row of three words: the first pages of the expert's `w1`, `w3`
and `w2`. A routed expert's run is `w1`, `w3`, `w2`, each 11,520 pages of bfp4_b. The
shared expert reads its bfp8_b copy (`EXPERTS_SHARED_FORMAT=bfloat16` picks the exact copy).

## One load, kept across calls

`Moe` holds every handler of the three pieces:

- `route` computes the router's logits, `x Wg`, in float32. The 160 K tiles are split over
  5 cores for each score tile, and the partial sums meet in a tree reduce.
- `up` computes `h | g` of each list place (K split, tree reduce).
- `gate` computes `silu(min(h, 10)) * clamp(g, +-10) * r`. Here `r` is the token's weight
  for that expert, 0 for a token that did not pick it.
- `down` computes `w2` of that, summed over the list.

The list length `e` is a command word. The word `eight` picks the shared expert's weight
format in `up` and `down`.

`make()` builds the class and the tree's program once. The first call loads them as one
run, and that run stays for every later call and layer:

```python
tree.ahead(moe)  # Moe's parts built at the load
tree.slots(count=1, service="shared")
run = session.load({tree: (0, session.ranks)})  # the tree: the dict's program
run.addon(moe)  # Moe, a class in the runtime form
run.wait(run.load(moe, (0, session.ranks)))  # a switch: the cores now run Moe
```

A switch happens only when a step needs the other program. A routed call that follows
another site switches three times: to the tree (down), to Moe (the experts) and to the
tree (up). Gate and shared need none while the cores hold Moe. A call's switches and
commands are queued, each command exclusive, sent to the cards every 4, and waited for
once at the end; `ffn` waits twice, after the router (the host takes the top 6) and at
the end. Both programs share one set of compute options. The tree's buffers sit
at indices where Moe holds bfloat16 or nothing (`kernels/tree.h`), since an index holds
one format in the load. Moe's reduces are marked `local=True`, since each group is on one
island. The tree's child slots are the most any island has as the home, so one build
serves every layer.

Moe is made with small copy tensors of the arenas' forms. Each command names the arena of
its own island.

`make(session, W, layers, run=r)` hands the two programs to a one load another module
makes: `r.take({tree: every rank, moe: every rank})`, the tree first, since a hand
program can only be a dict's. The piece then loads nothing and goes through `r`'s
`load`, `submit`, `wait` and `store`. Every class added beside it must hold bfloat16 or
no buffer at indices 0, 1, 18, 20, 21, 22, 26 and 27, and uint32 or none at 23. The shared run says what
the cores hold (`run.holds`); the piece asks it before a switch, since another piece may
have switched them between two calls.

## The sites

- **gate**:
  1. `route` runs on the layer's home island, the island that holds the gate weight.
  2. The host computes the scores, `sqrt(softplus(logit))` in float32, and takes the top
     6 of score + bias.
  3. The host computes the weights: the chosen scores, normalised, times 1.5.
- **shared**: `up`, `gate`, `down` on the home island with one list place and a weight
  of 1.
- **routed**: each island that holds a chosen expert runs the list of its experts the
  block's tokens chose, as `up`, `gate`, `down` on all those islands at once.
  1. The block goes down a breadth-first tree of the islands from the home island,
     pruned to the islands it needs.
  2. An island on the way that holds no chosen expert runs one place at weight 0.
  3. The sums come up the same tree, added on Tensix at each island.

  Down the tree, the experts, and up the tree are three switches of the one load.
- **ffn**: gate, then routed and shared on the same block. The host adds the two and
  rounds to bf16.

## Known limits

- The router's scores and the top 6 are taken on the host. The logits are the box's.
- At a near-tie of the 6th and 7th keys (3e-5 to 3e-4 apart in float64), the box's logits
  (bf16 operands at HiFi2) can pick the other expert: 1 or 2 tokens of a 32-token block
  in layers 0 and 20 of the fixed prompts.
- In the runtime form the cards' command times are longer than in the kernel form (layer
  3, a 32-token block, four islands): `up` of the experts 2,741 against 2,367 us, `down`
  1,320 against 1,105 us, `route` 66 against 49 us. Not looked into.

## The tree kernels (kernels/)

`tree.h`: the block's activation goes down a spanning tree of the islands into each
island's own copy in DRAM, and the islands' results come up the same tree, added on Tensix
at each island (its own slice and its children's, fp32 in DST), the sum written into the
home island's DRAM. S cores an island, n tiles (bf16, 2 KiB) each: core s takes from and
puts to core s of its neighbours. The tree comes in the command: a parent an island. The
runtime's form (km_program): a program of one load that switches program.

`reader.cpp`: an island's core s takes the sync pages of its children; down, the home reads
its slice of x for the writer, another island takes its parent's block and writes it into
its own copy; up, it reads its own y slice for compute, takes its children's sums, and the
home writes compute's total into its DRAM.

`writer.cpp`: the sync; the block down (the home's slice from its reader, another island's
from its parent, copied to the buffer it sends from and put to each child); the sums up
(compute's sum put to its parent; the home's reader writes it); the home's ticks from after
the sync to its end. The stamp handler's word 0 picks the writer's ticks after the sync (0)
or the home reader's on the way up (1).
