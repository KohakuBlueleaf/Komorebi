"""A window past a short block (no card): a window into a cast that may be shorter
than its slot is checked at run time (past it the slot holds an earlier block),
unless the sums started from a mask row, whose -inf covers it; a window the build
knows is inside is not checked."""

from pathlib import Path

from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor

K, N, SUB, KB = 2, 2, 2, 4
MASK_PAGES = 4  # 2 mask rows of SUB tiles


def tensors():
    def made(shape, address, dtype="bfloat16"):
        pages = shape[0] * shape[1] // 1024
        page = formats.tile_bytes(dtype)
        return Tensor(Pages(page, pages, address), shape, dtype, "tile")

    return {
        "x": made((4 * 32, K * 32), 0x100000),
        "kt": made((8 * 32, K * 32), 0x1000000),
        "init": made((32, 32 * 4), 0x2000000, "float32"),
        "y": made((4 * 32, SUB * 32), 0x4000000, "float32"),
    }


@km.tile_function
def score(x, w):
    return x @ w


@km.tile_function
def start(row, state):
    return row, (state[0],)


@km.tile_function
def keep(row, state):
    return row, (state[0],)


class Unmasked(km.Program):
    @km.handler
    def run(self, x: km.Tensor, kt: km.Tensor, init: km.Tensor, y: km.Tensor, n: int):
        column = km.Broadcast(self, along="column", window=1, mover="A")
        xs = km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=K)
        if column.sender():
            kv = column.cast(kt.tiles[0 : min(0 + KB, n), 0:K].T)
        else:
            kv = column.arrive()
        for c in range(KB // SUB):
            s = km.tile.zeros(km.fp32, SUB)
            s += score(xs[0], kv[:, c * SUB : c * SUB + SUB])
            y.pages(c * SUB, SUB).write(s)


class Masked(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    mask: km.L1[km.fp32, 2, SUB, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, kt: km.Tensor, init: km.Tensor, y: km.Tensor, n: int):
        self.state.zero()
        self.mask.fill(km.Stream(init.pages(0, MASK_PAGES)))
        column = km.Broadcast(self, along="column", window=1, mover="A")
        xs = km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=K)
        if column.sender():
            kv = column.cast(kt.tiles[0 : min(0 + KB, n), 0:K].T)
        else:
            kv = column.arrive()
        for c in range(KB // SUB):
            s = start(self.mask[c % 2], self.state[0])
            s += score(xs[0], kv[:, c * SUB : c * SUB + SUB])
            y.pages(c * SUB, SUB).write(keep(s, self.state[0]))


class Known(km.Program):
    @km.handler
    def run(self, x: km.Tensor, kt: km.Tensor, init: km.Tensor, y: km.Tensor, n: int):
        column = km.Broadcast(self, along="column", window=1, mover="A")
        xs = km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=K)
        if column.sender():  # noqa: SIM108 (the compiler's form)
            kv = column.cast(kt.tiles[0:KB, 0:K].T)
        else:
            kv = column.arrive()
        s = km.tile.zeros(km.fp32, SUB)
        s += score(xs[0], kv[:, 2 : 2 + SUB])
        y.pages(0, SUB).write(s)


def compute_of(cls):
    program = cls(l1=1 << 20, log=quiet, rect=(0, 0, 2, 4), **tensors())
    return (Path(program.built.folder) / "compute.cpp").read_text()


CHECK = "// the window inside the cast's rectangle"


def test_a_window_of_a_short_block_is_checked_before_its_products():
    text = compute_of(Unmasked)
    check = text.index(CHECK)
    gave_up = text.index("compute.gave_up(km::kGaveBadItem, km::kWhereCInput);", check)
    product = text.index("matmul_block(km_held_xs, ", gave_up)
    assert check < gave_up < product
    assert "((c * 2)) + 2 <= (" in text[check - 200 : check]


def test_sums_started_from_a_mask_row_are_not_checked():
    assert CHECK not in compute_of(Masked)


def test_a_window_the_build_knows_is_not_checked():
    assert CHECK not in compute_of(Known)
