"""A product whose left side is a FILLED field (no card): `acc += f(self.v, xb)` with
self.v filled in the handler or kept by another, waited for once and freed at the
end (or kept); `return b @ a` read as the same product with its operands swapped;
refused: a body of neither form, a swapped call off the loop's block."""

from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.errors import ProgramError

K = 3  # the field's tiles: a row of 3 (a 3-tap vertical filter)


def tensors():
    def made(shape, address, dtype="bfloat16"):
        pages = shape[0] * shape[1] // 1024
        page = formats.tile_bytes(dtype)
        return Tensor(Pages(page, pages, address), shape, dtype, "tile")

    return {
        "img": made((8 * 32, 4 * 32), 0x100000),
        "v": made((32, K * 32), 0x1000000),
        "out": made((4 * 32, 4 * 32), 0x2000000, "float32"),
    }


@km.tile_function(fidelity="HiFi4")
def left(a, b):
    return a @ b


@km.tile_function(fidelity="HiFi4")
def swapped(a, b):
    return b @ a


@km.tile_function(fidelity="HiFi4")
def neither(a, b):
    return a @ b + a


class Vertical(km.Program):
    taps: km.L1[km.bf16, 1, K, 32, 32]

    @km.handler
    def run(self, img: km.Tensor, v: km.Tensor, out: km.Tensor):
        self.taps.fill(km.Stream(v.tiles[0:1, 0:K]))
        r = self.rank // 4
        c = self.rank % 4
        acc = km.tile.zeros(km.fp32, 1)
        for xb in km.Stream(img.tiles[r : r + K, c : c + 1], block=K):
            acc += left(self.taps, xb)
        out.pages(self.rank, 1).write(acc)


class Swapped(km.Program):
    taps: km.L1[km.bf16, 1, K, 32, 32]

    @km.handler
    def run(self, img: km.Tensor, v: km.Tensor, out: km.Tensor):
        self.taps.fill(km.Stream(v.tiles[0:1, 0:K]))
        r = self.rank // 4
        c = self.rank % 4
        acc = km.tile.zeros(km.fp32, 1)
        for xb in km.Stream(img.tiles[r : r + K, c : c + 1], block=K):
            acc += swapped(xb, self.taps)
        out.pages(self.rank, 1).write(acc)


class Kept(km.Program):
    taps: km.L1[km.bf16, 1, K, 32, 32]

    @km.handler
    def load(self, v: km.Tensor):
        self.taps.fill(km.Stream(v.tiles[0:1, 0:K]))

    @km.handler
    def run(self, img: km.Tensor, out: km.Tensor):
        r = self.rank // 4
        c = self.rank % 4
        acc = km.tile.zeros(km.fp32, 1)
        for xb in km.Stream(img.tiles[r : r + K, c : c + 1], block=K):
            acc += left(self.taps, xb)
        out.pages(self.rank, 1).write(acc)


class Neither(km.Program):
    taps: km.L1[km.bf16, 1, K, 32, 32]

    @km.handler
    def run(self, img: km.Tensor, v: km.Tensor, out: km.Tensor):
        self.taps.fill(km.Stream(v.tiles[0:1, 0:K]))
        acc = km.tile.zeros(km.fp32, 1)
        for xb in km.Stream(img.tiles[0:K, 0:1], block=K):
            acc += neither(xb, self.taps)
        out.pages(self.rank, 1).write(acc)


class SwappedOff(km.Program):
    taps: km.L1[km.bf16, 1, K, 32, 32]

    @km.handler
    def run(self, img: km.Tensor, v: km.Tensor, out: km.Tensor):
        self.taps.fill(km.Stream(v.tiles[0:1, 0:K]))
        acc = km.tile.zeros(km.fp32, 1)
        for xb in km.Stream(img.tiles[0:K, 0:1], block=K):
            acc += swapped(self.taps, xb)
        out.pages(self.rank, 1).write(acc)


def compute_of(cls):
    program = cls(l1=1 << 20, log=quiet, **tensors())
    return (Path(program.built.folder) / "compute.cpp").read_text()


def test_a_filled_field_is_waited_once_taken_on_the_left_and_freed_at_the_end():
    text = compute_of(Vertical)
    waited = text.index("km_ok && compute.wait_front(kFieldTaps, kFieldTapsPages);")
    init = text.index("matmul_block_init(kFieldTaps, kRead0.mover_a, 0, 1, 1, 3);")
    call = text.index("matmul_block(kFieldTaps, km_half0, ")
    freed = text.rindex("cb_pop_front(kFieldTaps, kFieldTapsPages);")
    assert waited < init < call < freed
    assert text.count("compute.wait_front(kFieldTaps, kFieldTapsPages)") == 1


def code(text):
    """The handlers' lines, less the source comments (lines and names differ)."""
    body = text[text.index("case kHandlerRun:") :].split("\n")[1:]
    return [line for line in body if "// line" not in line]


def test_return_b_at_a_is_the_same_product_with_its_operands_swapped():
    assert code(compute_of(Swapped)) == code(compute_of(Vertical))


def test_a_field_another_handler_fills_is_kept_between_commands():
    text = compute_of(Kept)
    run = text[text.index("case kHandlerRun:") :]
    assert "matmul_block(kFieldTaps, " in run
    assert "cb_pop_front(kFieldTaps, kFieldTapsPages);" not in run


@pytest.mark.parametrize(
    "cls, refusal",
    [
        (Neither, r"is neither `return a @ b` nor `return b @ a`"),
        (SwappedOff, r"swapped is `return b @ a`, read as a field on the left"),
    ],
)
def test_refusals_name_both_forms(cls, refusal):
    with pytest.raises(ProgramError, match=rf"line \d+, `.*{refusal}"):
        compute_of(cls)
