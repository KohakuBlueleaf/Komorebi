"""A product's placement where the tile functions are traced or the operands are two
broadcasts' blocks, and the check at submit (no card): a traced function takes a plain
marker and refuses a turning one at its call's line; the broadcasts' product turns; a
command's W tensor stored otherwise than its product takes it is refused."""

import pytest

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import placements_of
from komorebi.errors import ProgramError


class Pages:
    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor(dtype, turned=False, shape=(256, 256)):
    made = Tensor(Pages(formats.tile_bytes(dtype)), shape, dtype, "tile")
    made.transposed_tiles = turned
    return made


class TracedPlain(km.Program):
    @km.tile_function
    def gate(a, w1, w3):
        return km.silu(km.srcb(a) @ w1) * (a @ w3)

    @km.handler
    def run(self, a: km.Tensor, w1: km.Tensor, w3: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, p, q in zip(
            km.Stream(a.pages(0, 8), block=8),
            km.Stream(w1.pages(0, 8), block=8),
            km.Stream(w3.pages(0, 8), block=8),
        ):
            results.push(self.gate(x, p, q))


class TracedTurned(km.Program):
    @km.tile_function
    def gate(a, w1, w3):
        return km.silu(a @ km.srcb(w1)) * (a @ w3)

    @km.handler
    def run(self, a: km.Tensor, w1: km.Tensor, w3: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, p, q in zip(
            km.Stream(a.pages(0, 8), block=8),
            km.Stream(w1.pages(0, 8), block=8),
            km.Stream(w3.pages(0, 8), block=8),
        ):
            results.push(self.gate(x, p, q))


def traced_tensors():
    return {name: tensor("bfloat16") for name in ("a", "w1", "w3", "out")}


def test_a_traced_function_takes_a_plain_marker():
    compiled = compile_class(TracedPlain, {}, traced_tensors())
    built = sorted(
        line.rsplit("built ", 1)[1]
        for line in placements_of(compiled, traced_tensors())[1:]
    )
    assert built == ["plain, marked", "plain, no marker"]


def test_a_traced_function_refuses_a_turning_marker_at_its_line():
    with pytest.raises(ProgramError) as refused:
        compile_class(TracedTurned, {}, traced_tensors())
    text = str(refused.value)
    assert "results.push(self.gate(x, p, q))" in text
    assert "gate's `a @ km.srcb(w1)` (line 49) turns the product" in text
    assert "DST's transpose back is a library call" in text


def grid(turned):
    class Grid(km.Program):
        part: km.L1[km.fp32, 2, 2, 32, 32]

        if turned:

            @km.tile_function(fidelity="HiFi4")
            def product(x, w):
                return x @ km.srcb(w)

        else:

            @km.tile_function(fidelity="HiFi4")
            def product(x, w):
                return x @ w

        @km.handler
        def grid(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
            along = km.Broadcast(self, along="row", window=2)
            down = km.Broadcast(self, along="column", window=2)
            for chunk in range(4):
                k0 = chunk * 2
                if along.sender():
                    xs = along.cast(x.tiles[0:2, k0 : k0 + 2].read())
                else:
                    xs = along.arrive()
                if down.sender():
                    ws = down.cast(w.tiles[k0 : k0 + 2, 0:2].read())
                else:
                    ws = down.arrive()
                self.part += self.product(xs, ws)
            y.tiles[0:2, 0:2].write(self.part)

    return Grid


def test_the_broadcasts_product_turns_with_w_transposed():
    tensors = {"x": tensor("bfloat16"), "w": tensor("bfp8_b", True)}
    tensors["y"] = tensor("float32")
    compiled = compile_class(grid(True), {}, tensors)
    (line,) = placements_of(compiled, tensors)[1:]
    assert line.endswith(
        "built turned, marked: W's tiles stored transposed, X's read so"
    )
    plain = {**tensors, "w": tensor("bfp8_b")}
    with pytest.raises(ProgramError, match="is turned and tensor w is not stored"):
        placements_of(compile_class(grid(True), {}, plain), plain)


class Sums(km.Program):
    sums: km.L1[km.fp32, 1, 4, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def turned(x, w):
        return x @ km.srcb(w)

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:2], block=2),
            km.Stream(w.tiles[0:2, 0:4], block=8),
        ):
            self.sums += self.turned(xb, wb)
        y.tiles[0:1, 0:4].write(self.sums)


def test_a_commands_weight_is_checked_against_its_products_placement():
    tensors = {"x": tensor("bfloat16"), "w": tensor("bfp8_b", True)}
    tensors["y"] = tensor("float32")
    Sums(log=lambda line: None, l1=1_000_000, **tensors)
    assert Sums.apply.args(tensors["x"], tensors["w"], tensors["y"])
    with pytest.raises(ProgramError) as refused:
        Sums.apply.args(tensors["x"], tensor("bfp8_b"), tensors["y"])
    text = str(refused.value)
    assert "apply: tensor w is not stored with its tiles transposed" in text
    assert "Sums.apply, line 146: self.sums += self.turned(xb, wb)" in text
    assert "W of a turned product" in text
