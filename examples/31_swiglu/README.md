# 31 swiglu

The README's SwiGLU layer, `y = (silu(x W1) * (x W3)) W2`, as a Python class: the class
in `swiglu_class.py` is the README's block as printed. x is [32, 1,024], W1 and W3 are
[1,024, 3,072], and W2 is [3,072, 1,024], all bf16. The hidden is bf16 and y is float32.

| Handler | Cores | Reads | Writes |
|---|---|---|---|
| `up` | 12, 8 hidden columns each | x's row of tiles, its 8 columns of W1 and of W3, 2 K tiles a block | `hidden`: `gate(self.h, self.g)` of its two fields of sums |
| `down` | 4, 8 columns of y each | the hidden's row (the call before's output), its 8 columns of W2 | `y`: its field of sums |

`up` adds two products a block into two fields, reading x's block once for both, and
writes the vector unit's gate of the two fields. `forward`, a `km.chain` method, puts
`up` on cores 0 to 11 and `down` on cores 0 to 3 as one command. `down`'s record waits
for every `up` block it reads.

```
python examples/31_swiglu/swiglu_class.py build   # on a mock device, no card
python examples/31_swiglu/swiglu_class.py card
```

`card` loads the class twice. The first time it runs a command for each step (`up`,
then `down`, each waited for). The second time it runs the chain. Each form runs six
times, and the five after the first are timed. hidden and y are read after the run's
stop (a read waits for the cores) and compared against float64 of the stored bf16
inputs. The session gives an 80 KiB kernel buffer: the class with the chain's walk
holds 81,664 B of text.

## Measured (one card)

| | hidden rms | hidden largest | y rms | y largest | us a call |
|---|---:|---:|---:|---:|---:|
| a command a step (up + down) | 6.6e-3 | 6.7e-3 | 9.4e-3 | 9.2e-3 | 104.9 |
| the chain, one command | 6.6e-3 | 6.7e-3 | 9.4e-3 | 9.2e-3 | 105.7 |

- "largest" is the largest error over the largest value.
- The products run at HiFi2, as the README's `project` states. y against float64 of
  the card's own hidden is 4.1e-3 rms. These are within the dense project's SwiGLU
  bound (2e-2 rms, 3e-2 largest; [SWIGLU](../../projects/dense/SWIGLU.md)).
- The steps' time is the sum of the two commands' card times, with no host turn
  between them. The chain's time includes its walk and `down`'s wait for `up`'s marks.
- The hand program, projects/dense SwiGLU, holds every weight in each core's L1 (D x H
  tiles a weight). At this shape that is 6 MB a weight, so it cannot run here.

## up and down as overlays

`swiglu_overlays.py` makes the same handlers overlays and runs the chain at each slot
count:

```python
class OverlaySwiGLU(SwiGLU):
    up = km.handler(overlay=True)(SwiGLU.up.function)
    down = km.handler(overlay=True)(SwiGLU.down.function)
```

In a chain, an overlay handler's record loads its overlay, waits for its input whole,
then calls the part ([classes, overlay handlers in the walk](../../docs/reference/classes.md#chains)).
The resident `down` reads its W2 columns while `up`'s blocks are still being marked;
as an overlay it reads nothing until every block of the hidden it reads is marked.
Card 0, 2026-11-16, AICLK 800 MHz, one job, 50 timed chains a form after a first
command; hidden and y equal word for word to the resident chain's in every form:

| The chain | us on the card | us from the host | Slots a part |
|---|---:|---:|---:|
| resident | 105.67 | 246.77 | |
| overlays, the default slots | 111.92 | 253.93 | 2 |
| overlays, 1 slot | 115.70 | 257.13 | 1 |

- At the default slots (a slot a handler) both overlays stay held after the first
  command, so a record loads nothing: the 6.25 us over the resident chain is the wait
  for the whole input (`down`'s W2 reads no longer start ahead of the hidden's marks).
- At 1 slot each record loads its overlay (both, every chain): 3.78 us more for two
  loads, each waiting for the core's other CPUs to end the record before.

```
python examples/31_swiglu/swiglu_overlays.py             # resident, default slots, 1 slot
```

## the hidden sent direct

`swiglu_direct.py` lets `down` take the hidden direct, and adds a chain that sends it:

```python
class DirectSwiGLU(SwiGLU):
    down = km.handler(direct=("hidden",))(SwiGLU.down.function)

    @km.chain
    def forward_direct(self, x, w1, w3, w2, hidden, y):
        self.up(x, w1, w3, hidden, cores=(0, 12))
        self.down(km.direct(hidden), w2, y, cores=(0, 4))  # up's blocks into L1
```

Each of `up`'s 12 roots keeps its 8 tiles of the hidden in its write buffer and writes
them into the landing area of each of `down`'s 4 cores: 16 KiB a reader, 64 KiB a
root. It marks those cores once its own writes are acknowledged. A `down` core reads
the hidden from its area (a local copy into its stream buffer), not from DRAM. The
same class runs the plain `forward` chain through DRAM. The hidden has no DRAM copy
after the direct chain, so `hidden.read()` then raises, naming `up`'s call.

Card 0, 2026-11-16, one job. Each chain was run once after its own load, and y was
equal word for word in both. Then both chains ran in turn in one load, 50 timed each
after a first command. The AICLK was 800 MHz before the job and 1,350 MHz after it.
The two chains alternated, so they compare with each other, not with the table above.

| The chain | us on the card |
|---|---:|
| through DRAM | 105.93 |
| the hidden direct | 110.86 |

- The direct chain is slower by 4.93 us. `down` is bound by its W2 reads (96 x 8
  tiles, 1.5 MiB a core), which start ahead of the hidden's marks in both chains. Its
  192 KiB of hidden a core, read from DRAM in the plain chain, overlapped with those
  reads. In the direct chain, each root's 64 KiB send has to be acknowledged before it
  marks. That puts 12 roots' sends into 4 cores' L1 on the path before `down` can use
  the hidden.
- Not measured, expected: a direct input gains where the reader's DRAM read of its
  input is on that path and the roots' sends are short (a small input to few readers).

```
python examples/31_swiglu/swiglu_direct.py               # both chains, every word of y
```
