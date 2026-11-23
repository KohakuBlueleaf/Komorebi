"""QuickSRNet-Medium 2x as a class: one program on every core, the weights kept in L1,
each layer a handler call over the cores' bands of image rows, the seven layers one
chain (README.md, "The handler")."""

import komorebi as km

ROWS = 5  # image rows a core
TILES = 16  # tiles an image row: 512 pixels
BAND = (ROWS + 2) * TILES  # the band's input tiles: its rows and one row each side
TAPS = 3 * BAND  # three versions of each input row
WEIGHTS = 84  # 7 layers of 10 tiles, then the shuffle's 14
OUT_ROWS = 6 * ROWS  # output rows a core: 2 a row, 3 colours each

INCLUDE_MOVE = '#include "../../../../projects/upscale/upscale_move.hpp"\n'
INCLUDE_COMPUTE = '#include "../../../../projects/upscale/upscale_compute.hpp"\n'
SHIFT = (
    INCLUDE_MOVE + f"upscale::shift(band, taps, ones, ready, first, {BAND // TILES});"
)
TAKE = "compute.wait_front(ready, 1);\ncb_pop_front(ready, 1);"
CONV = INCLUDE_COMPUTE + "upscale::conv_row<taps, weights, ones>(out_buffer, r, layer);"
OPEN = f"cb_reserve_back(rows, {OUT_ROWS + 2});"
LAST = (
    INCLUDE_COMPUTE
    + "upscale::last_column<taps, weights, ones, zs, zp, rows>(t, layer);"
)
CLOSE = f"cb_push_back(rows, {OUT_ROWS + 2});"
WRITE = f"""
if (!movement.wait_front(rows, {OUT_ROWS + 2})) {{
    return false;
}}
if (!km::write_pages(
    movement, command, result, out_first, {OUT_ROWS}, get_read_ptr(rows))) {{
    return false;
}}
noc_async_write_barrier();
cb_pop_front(rows, {OUT_ROWS + 2});
"""


class Upscale(km.Program, compute={"math_fidelity": "HiFi2", "fp32_dest_acc_en": True}):
    """load: the weights into L1 once. conv: one 3 x 3 layer, image rows 5 rank to
    5 rank + 4 of y from the band of x around them. last: the last layer and the pixel
    shuffle into the frame's rows."""

    weights: km.L1[km.bfp8_b, WEIGHTS, 32, 32]  # tiles of the turned product, kept
    ones: km.L1[km.bf16, 1, 32, 32]  # column 0 ones: the bias's operand, kept
    band: km.L1[km.bf16, BAND, 32, 32]  # the input rows as read
    taps: km.L1[km.bf16, TAPS, 32, 32] = km.own(push=None)  # mover B's versions
    ready: km.L1[km.bf16, 1, 32, 32] = km.own(push="mover_b")  # taps written
    zs: km.L1[km.bf16, ROWS, 32, 32] = km.own(push="pack")  # the last layer's Z_r
    zp: km.L1[km.bf16, 2, 32, 32] = km.own(push="pack")  # its Z'_j
    rows: km.L1[km.fp32, OUT_ROWS + 2, 32, 32] = km.own(push="pack")  # the frame's

    @km.handler
    def load(self, w: km.Tensor, o: km.Tensor):
        self.weights.fill(km.Stream(w.pages(0, WEIGHTS)))
        self.ones.fill(km.Stream(o.pages(0, 1)))

    @km.handler
    def conv(self, x: km.Tensor, y: km.Tensor, layer: int):
        first = self.rank * ROWS  # x's row of band row 0: image row 5 rank - 1
        self.band.fill(km.Stream(x.pages(first * TILES, BAND)))
        km.cxx(
            SHIFT,
            takes=[self.band, self.taps, self.ones, self.ready, first],
            on="mover_b",
        )
        km.cxx(TAKE, takes=[self.ready])
        out = km.Stream(
            y.pages((first + 1) * TILES, ROWS * TILES), write=True, block=TILES
        )
        for r in range(ROWS):
            km.cxx(CONV, takes=[out, r, layer, self.taps, self.weights, self.ones])

    @km.handler
    def last(self, x: km.Tensor, result: km.Tensor, layer: int):
        first = self.rank * ROWS
        self.band.fill(km.Stream(x.pages(first * TILES, BAND)))
        km.cxx(
            SHIFT,
            takes=[self.band, self.taps, self.ones, self.ready, first],
            on="mover_b",
        )
        km.cxx(TAKE, takes=[self.ready])
        km.cxx(OPEN, takes=[self.rows], on="pack")
        for t in range(TILES):
            km.cxx(
                LAST,
                takes=[
                    t,
                    layer,
                    self.taps,
                    self.weights,
                    self.ones,
                    self.zs,
                    self.zp,
                    self.rows,
                ],
            )
        km.cxx(CLOSE, takes=[self.rows], on="pack")
        out_first = self.rank * OUT_ROWS
        km.cxx(
            WRITE,
            takes=[result, out_first, self.rows],
            writes=[(result, self.rank * OUT_ROWS, OUT_ROWS)],
            on="mover_b",
        )

    @km.chain
    def forward(self, frame, a, b, result):
        self.conv(frame, a, 0)
        self.conv(a, b, 1)
        self.conv(b, a, 2)
        self.conv(a, b, 3)
        self.conv(b, a, 4)
        self.conv(a, b, 5)
        self.last(b, result, 6)
