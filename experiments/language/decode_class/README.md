# The decode layer, as a class

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

(Two runs: two values in a cell.) The class's gate_up and down are wrong in this order
(qkv, o, gate_up, down, each 21 times), and the count changes from run to run. Run alone
(down, then gate_up; each over part of K too: down at K 120, 150, 165, 180 and 192 tiles,
gate_up at 6, 14, 16, 18 and 64, twice each) every one is right, 4.23e-3 to 4.29e-3
against float64. A run of the four in a mixed order (gate_up first) hung: a command did
not end and the grid did not return. So one handler leaves something that a handler of
another group size or pattern after it meets; what, is not found.

The class's text is 79,840 B (mover A 18,704, mover B 16,060, unpack 13,308, math
15,248, pack 15,836), the hand's program 62,224 B with its four operations as overlays.

Found on the way: the class's reduces share one arrival buffer (`kArrival0`), and the
build sizes it by the first handler's reduce. A later handler whose tree has more rounds
gets too few slots, and its reader gives up (`bad_item` in a port) on every core. `o` is
the first handler for that reason: its tree of 15 has four rounds, qkv's of 7 three.

```bash
python layer_class.py                   # the hand's four products, then the class's
python layer_class.py --skip-hand       # the class alone
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
