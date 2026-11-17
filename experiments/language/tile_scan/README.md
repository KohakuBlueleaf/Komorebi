# A scan's state in a tile function

Question: does `h = f(x_t, h)` in a stream loop, h a DST tile made before the loop,
give on the card the recurrence float64 gives, token after token, and does a scan cut
into two commands (the second started from the state the first wrote) give the states
of one?

`scan.py` makes the class `Scan` with one core's two handlers on tiles of a token's
32 x 32 values:

```python
@km.tile_function
def gated(a, h, u):
    return a * h + u
```

`run_all` steps h from `km.tile.zeros(km.fp32)` over 64 tokens, pushing every h_t into
a float32 tensor, and writes the last; `run_half` steps 32 tokens from `first`, h
started from a field filled from a state tensor (`h = self.state`), and writes its last
state. The script runs `run_all`, then `run_half` from a zero state and `run_half` again
from the state the first half wrote. a is uniform in [0.5, 1), u normal; the class is
made for float32 and bfloat16 a and u, the states float32, compared with float64 on the
stored values.

    python scan.py                # one card (--cards N: that card)
    kmr prebuild --cards 0,1 scan.py --cards 0,1   # the same class for two cards, on the mock

## Measured (one card, one core)

2026-11-18, card 1, AICLK 800 MHz before the job, 0 compiled (prebuilt), 13 s with the
open. The largest error of the 64 states over the largest state, against float64 on
the stored a and u:

| a and u | 64 tokens | the last token | two commands of 32 against one of 64 |
|---|---|---|---|
| float32 | 6.6e-8 | 5.6e-8 | 0 lanes differ; last states equal |
| bf16 | 8.8e-8 | 9.2e-8 | 0 lanes differ; last states equal |

`run_all`'s written last state equals its 64th pushed one, and the second half's
written state equals it too. The last token's error is within the 64 tokens' largest.
