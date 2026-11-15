# The decode layer, as a class

## The whole layer as one class

`decode_layer.py` is the layer of the decode project as one class, `DecodeLayer`: qkv,
attend, o, gate_up and down, each a handler in the hand's plan (112 cores in groups of
7, 120 in 15, 120 in 15, 96 in 2, 120 in 15). Each product is the hand's K split in the
class's lines. The group's root then runs the hand's operation as an inline body that
takes the reduce's sum (`km.cxx(ADD, takes=[whole, residual, out, part, ...])`). The
operations are in `layer.hpp`, the hand's C++ on the class's buffers. Attend's cores
merge their states up a tree by `km.reduce(self.state, form="tree",
combine=MERGE)`, and the root appends the token's key and value to the cache from mover
bodies with `writes=[keys]` and `writes=[values]`. `run_layer.py` runs the hand's
program (`card.Decode`, a command a step) and then the class, on the same x, weights and
cache at T 256 (position 255). It compares every output word with the hand's and with
float64 (`model.layer`), then times ten layers each way.

Card 0, a Blackhole p150a, AICLK 800 MHz, 2026-11-16, a command a step (each waited for,
the dispatcher's microseconds; the class's median of ten layers, the hand's mean of ten
tokens):

| Step | Words that differ from the hand's | Against float64, hand / class (max rel) | Hand, us | Class, us |
|---|---|---|---:|---:|
| qkv | qkv 0 of 131,072 | 1.13e-2 / 1.13e-2 | 63.83 | 64.39 |
| attend | a 0 of 65,536 | 1.00e-2 / 1.00e-2 | 50.24 | 47.27 |
| o | x1 0 of 65,536, squares2 0 of 8,192 | 3.79e-3 / 3.79e-3 | 33.90 | 36.14 |
| gate_up | gated 0 of 196,608 | 7.43e-3 / 7.43e-3 | 131.24 | 140.64 |
| down | x 0 of 65,536, squares1 0 of 8,192 | 6.54e-3 / 6.54e-3 | 72.65 | 72.88 |
| layer | | | 363.17 | 361.23 |

The five handlers are resident (170,368 B of kernel text, so the session opens with
`kernel_bytes` 200 KiB). An earlier run with them as overlays gave the same words in
430.87 us a layer, each step 10 to 20 us slower. The overlay loads before each command
are the likely difference; they were not timed apart.

**The bodies, and which could become tile functions.** Tile functions of fields
written, with division, `km.sqrt`, `km.reciprocal`, `km.minimum`, tile expressions in
a line and float parameters, are on main.

| Body | What it does | As a tile function |
|---|---|---|
| ADD (o, down) | sum + residual into x, and each row's sum of squares of x as a diagonal tile | The add, yes. The squares are a product of x with x transposed, keeping the diagonal: not a tile function today |
| GATE (gate_up) | 1 / rms from the squares, then silu(gate) x up on the scaled sums | The 1 / rms and the product, yes (`km.sqrt`, `km.reciprocal`, a division on the fields of sums). silu needs an exp, which was not checked here |
| SCALE (qkv) | 1 / rms, then each head's norm (over 128 columns: a row mean by a product with the mean tile) and the rotation (rows of the rope table) | The scaling, yes. The head's mean is a product with a constant tile, and the rotation reads rows a mover put into tiles: not today |
| ATTEND, MERGE, FINISH | scores Q K^T, the running max, exp, sum, P V; the merge's rescale; the output over the sum | The arithmetic per tile, yes. The products Q K^T and P V over a core's keys and the row max are not tile functions today |
| QK_READ, ATTEND_KEYS, ATTEND_VALUES, TOKEN | movers' reads of rows of the rope table and the cache, the append, the token's copy | No: they move data, they do not compute it |

**As a chain: open.** `chain_calls` now describes calls of several groups:
- a write or read under the leader's `if` after the reduce, or under `if self.rank % G
  == 0:`, is each group's root's;
- a K share written out over each group's ranks is the call's share of K.

On that basis the layer as the host's list and as a `km.chain` method gave equal
records. On the card, the class with the chain's walk hung on its first plain command,
qkv. A second run without the walk, with the same driver change, hung the same way. Both
needed a card reset, and the cause is not found. So `DecodeLayer` here is the class that
ran (the table above), and the chain form is not in this folder.

**Overlays in a chain.** A chain's walk refuses overlay handlers, but the hand's
decode keeps its operations as overlays inside its chain. A class with overlay handlers
in a chain would need the walk to load a record's handler parts before the record runs
(the service overlay's load, from the chain's case), and the next record's load to
overlap the current one, as the hand's chain does. That is a gap, not a design limit:
the walk and the overlay service exist apart, and joining them is the work.

## The products alone (layer_class.py)

**Fixed.** Until 2026-11-16 `layer_class.py` ran only qkv and o unless given
`--known-wrong`: gate_up and down, run after another product, gave wrong sums that
changed from run to run, and a run of the four in a mixed order hung the card. The class
compiler's field reduce did not count mover B's send in unpack's own count of the
field's pages, and its products let pack's reserve steer the sections of all three
compute threads. Both are fixed, the same mixed order is right, and the flag is gone.

Question: can the decode project's layer
([projects/decode](../../../projects/decode/README.md)) be written as a class
([Programs written as classes](../../../docs/reference/classes.md)): its five calls
(qkv, attend, o, gate_up, down) as handlers, the layer as a chain
([chains](../../../docs/reference/classes.md#chains)), the arithmetic bodies the hand's,
inline and unchanged? And does one layer give the hand's output, word for word, at T
256?

Not yet. Each of the four products is the hand's K split, and the class's lines state
it: `layer_class.py`'s handlers `qkv`, `o`, `gate_up` and `down`, each in its plan (16
groups of 7, 8 of 15, 48 of 2, 8 of 15), give the hand's Y word for word for qkv and o,
in the hand's time. What the hand does after the sum (each root's operation: the norms,
the rotation, the residual, silu), attend's merge, and the layer as a chain each need a
line the class does not have yet. The handlers stop there: the hand's operation bodies
are not written over in lines, and no line is built here.

```python
@km.handler
def o(
    self,
    attended: km.Tensor,
    w_o: km.Tensor,
    x1: km.Tensor,
    k_tiles: int,
    pattern: km.Pattern,
):
    share = self.rank % self.O_GROUP  # the rank in its group: km::partition's index
    first = self.rank // self.O_GROUP * self.WIDTH
    k0 = share * (k_tiles // self.O_GROUP) + min(share, k_tiles % self.O_GROUP)
    k1 = (share + 1) * (k_tiles // self.O_GROUP) + min(
        share + 1, k_tiles % self.O_GROUP
    )
    for ab, wb in zip(
        km.Stream(attended.tiles[0:1, k0:k1], block=self.CHUNK, pattern=pattern),
        km.Stream(
            w_o.tiles[k0:k1, first : first + self.WIDTH],
            block=self.W_BLOCK,
            pattern=pattern,
        ),
    ):
        self.sums += self.project(ab, wb)
    whole = km.reduce(self.sums, form="tree", group=self.O_GROUP)
    if self.rank == self.leader:
        x1.tiles[0:1, first : first + self.WIDTH].write(
            whole
        )  # the hand: its operation
```

## The products, beside the hand's

`layer_class.py` loads the hand's program (`card.Decode`, a command a step, context 256)
and runs each product with no operation after the sum (`a6` 0: the split's writer writes
the group's sum), then loads the class and runs its handler on the same A and W. A is a
random row tile, W uniform of variance 1 / K, both bf16. Each Y is compared word for word
and against float64 of the same A and W.

Card 0, a Blackhole p150a, AICLK 800 MHz, 2026-11-16; 20 commands a product, each
waited for, the dispatcher's median:

| Product (plan) | Words of Y that differ | Against float64, hand / class (RMS) | Hand, us | Class, us |
|---|---|---|---:|---:|
| qkv (16 x 7, AB) | 0 of 131,072 | 4.26e-3 / 4.26e-3 | 50.05, 50.21 | 49.00, 48.83 |
| o (8 x 15, AB) | 0 of 65,536 | 4.25e-3 / 4.25e-3 | 31.90, 31.90 | 32.23, 32.35 |
| gate_up (48 x 2, AAAB) | 18,504 and 18,156 of 393,216 | 4.25e-3 / 0.134 | 126.76, 126.82 | 133.29, 133.36 |
| down (8 x 15, AB) | 57,160 and 57,078 of 65,536 | 4.25e-3 / 0.538 | 71.01, 70.73 | 67.48, 67.52 |

(Two runs: two values in a cell; the table is from before the fix.) The class's gate_up
and down were wrong in this order (qkv, o, gate_up, down, each 21 times), and the count
changed from run to run. Run alone (down, then gate_up; each over part of K too: down at
K 120, 150, 165, 180 and 192 tiles, gate_up at 6, 14, 16, 18 and 64, twice each) every
one was right, 4.23e-3 to 4.29e-3 against float64. A run of the four in a mixed order
(gate_up first) hung: a command did not end and the grid did not return.

The cause: a field of sums is emptied by unpack on the group's root and by mover B's
send on the other ranks, and unpack writes its own count of the pages it took whole into
the field's count. A core that had sent rewound the count at its next turn as root, so
a handler of another group size after it sent early (wrong sums) or, with the field
full, waited forever. Pack's give-up there then left unpack and math inside the compute
unit, which is why the hang needed a card reset. With both fixed, the same mixed order of
17 commands (gate_up, down, qkv, gate_up, o, gate_up, down, qkv, down, o, down, o, o,
gate_up, qkv, qkv, down) is right on card 0: each 4.25e-3 to 4.29e-3 against float64,
the kernels ending with no error. The table's word-for-word comparison with the hand is
not run again here.

The class's text is 80,160 B (mover A 18,704, mover B 16,060, unpack 14,000, math
15,352, pack 15,348), the hand's program 62,224 B with its four operations as overlays.

Found on the way: the class's reduces share one arrival buffer (`kArrival0`), and the
build sized it by the first handler's reduce (its slots) and the last one's (its slot's
pages). A later handler whose tree had more rounds got too few slots, and its reader
gave up (`bad_item` in a port) on every core; a reduce of fewer tiles after one of more
made the slots overlap. It is now sized for the largest reduce
(`tests/core/test_class_reduce_arrival.py`). The runs above had `o` first, so four slots
of 8 tiles for all four products: that is not what made gate_up and down wrong.

```bash
python layer_class.py                   # the four: the hand's, then the class's
python layer_class.py --skip-hand       # the class alone
python layer_class.py --products qkv o  # some of them
```

## The lines the layer still needs

Each with two forms it could take.

| # | Where | What the hand does | Form 1 | Form 2 |
|---|---|---|---|---|
| 1 | qkv, o, gate_up, down | The group's root calls an operation on the sum where it would pack it (`split_op.hpp`): o and down add the residual's block and make the rows' squares, qkv and gate_up scale by 1 / rms (qkv also each head's norm and the rotation, gate_up silu and the gate). Today a body cannot take the reduce's result: `whole is a reduced here` (and a tile function takes only a loop's tiles, `self.total = whole` is refused) | `km.cxx(ADD, takes=[whole, residual, out])` at the root: `whole_buffer` the group's sum in float32, waited for, freed after the body, as the hand's hook | `km.reduce(self.sums, form="tree", group=S, then=ADD, takes=[residual, out])`: the body the root runs on the sum, named at the reduce |
| 2 | o, down, in a chain | Each writes two tensors, x and its rows' squares (a tile a group), and the next norm reads both. A chain's call has one output: the second write is refused | `kc.Call` with several outputs, each with its block a group and its own marks; the class deriving each write | The squares as tiles of the output (x one tile column wider, the root writing block g and tile N + g): one output, read by the next call as a range |
| 3 | qkv, in a chain | The rotary table's rows at the token's position: 32-byte rows of a page (`km::ask_bytes`), put into tile faces. As an inline body on mover A taking the table, the class builds it, but a chain cannot describe a tensor a body reads (refused) | `km.cxx(ROTARY, takes=[qk, position], reads=[qk])`: the body states what it reads, the build treats it as an outside input | `qk.rows(first, count, size=DC_ROPE_ROW)`: a read stream whose item is a row, put into faces by a compute or mover body |
| 4 | attend | A KV head's group merges its cores' attention states (row max, sum, O) up a tree over the ports, a merge body at each round, not a sum | `km.reduce(state, form="tree", group=S, combine=MERGE)`: the arrival merged into the core's state by an inline body | Port lines: `km.Port(self).send(state, to=rank)` and `.arrive()` a round, the merge a body between |
| 5 | attend | The token's key and value written as rows of the cache at the position (a row inside a tile) | `keys.rows(position, rows).write(k)`: a write of rows of a tile tensor | A body on mover B taking `keys` (built today), with `writes=[keys]` for a chain |

The layer as a chain needs 1 to 5.
