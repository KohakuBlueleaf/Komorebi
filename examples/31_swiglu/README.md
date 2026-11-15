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
