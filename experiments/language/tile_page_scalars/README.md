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
