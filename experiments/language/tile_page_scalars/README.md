# A tile function's scalars from a tensor page

Question: do a tensor page's words, read as float words (`read(km.fp32)`), give a tile
function's run-time scalars on the card what the same command's float words give, and
what does reading them cost?

`camera.py` makes the class `Camera` on 12 cores, 4 tiles of x, y and z a core:

```python
@km.tile_function
def clip(x, y, z, m0: float, m1: float, m2: float, m3: float):
    return x * m0 + y * m1 + z * m2 + m3
```

`row` reads the camera's row r, `m0, m1, m2, m3 = cam.pages(r, 1).read(km.fp32)[0:4]`,
from a float32 tensor of 3 pages (a 3 x 4 camera, a row a page) and pushes `clip`;
`row_words` takes the same 4 numbers as command float words; `scaled` pushes `a * m0`,
a tile expression of the page's first word. Each row is a command of its own (no
build between them), each command run 5 times for its median microseconds; the class
is made for float32 and bfloat16 x, y and z, each result compared with numpy (float64)
on the stored values and `row`'s with `row_words`' lane by lane.

    python camera.py                # one card (--cards N: that card)
    python camera.py --cards 0,1    # two cards, the group across the islands' edge
    kmr prebuild --cards 0,1 camera.py --cards 0,1   # the same for two cards, on the mock

## Measured

2026-11-18, 0 compiled (prebuilt), AICLK 800 MHz before each job. One card (card 1,
14 s with the open) and two (cards 0 and 1, the 12 ranks 6 on each island, tensors
where the session puts them). Median microseconds of 5 runs a command; the largest
error over the largest value against float64 on the stored tiles; 0 lanes beyond 1e-6
of it in every command:

| Command | float32, one card | bf16, one card | float32, two cards | bf16, two cards | error |
|---|---|---|---|---|---|
| `row` (page words), rows 0 to 2 | 9.8 to 10.0 | 8.6 to 8.8 | 294.5 to 296.3 | 293.4 to 293.6 | 6.1e-8 to 1.0e-7 |
| `row_words` (command words) | 9.6 | 8.1 to 8.2 | 275.8 to 276.4 | 274.0 to 274.9 | the same |
| `scaled` (`a * m0`) | 6.4 | 6.4 | 136.8 | 136.5 | 3.8e-8 |

`row` equals `row_words` lane for lane at every row, format and card count. `row`
takes 0.2 to 0.7 us a command more than `row_words` on one card, and 18 to 21 us more
with the group across the edge (where the camera tensor's page lies was not measured).
