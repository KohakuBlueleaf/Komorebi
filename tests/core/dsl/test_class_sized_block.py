"""Sizes given at each submit, km.UPTO among a handler's sets: a stream's block (the
pair sized at the most, the block a word, the host refusing a value that does not
divide the most, by name; the same class built for its constant unchanged), a
tensor's width, and how km.UPTO sits beside listed sets and km.ANY (no card)."""

import pytest
from test_class_thread_header import Pages, grid, quiet, twice

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.front import _sized
from komorebi.errors import ProgramError


class Blocks(km.Program):
    @km.handler
    def scale(self, a: km.Tensor, out: km.Tensor, tiles: int,
              block: km.constant[int] = 8):  # fmt: skip
        results = km.Stream(out.pages(0, tiles), write=True, block=block)
        for x in km.Stream(a.pages(0, tiles), block=block):
            results.push(twice(x))


class Sizes(km.Program):
    """Two constants: size listed, block given at each submit."""

    @km.handler
    def scale(self, a: km.Tensor, out: km.Tensor, tiles: int,
              size: km.constant[int] = 2, block: km.constant[int] = 8):  # fmt: skip
        results = km.Stream(out.pages(0, tiles * size), write=True, block=block)
        for x in km.Stream(a.pages(0, tiles * size), block=block):
            results.push(twice(x))


def made(cls, handlers=None):
    return cls(l1=1 << 20, log=quiet, a=grid(16), out=grid(16, 0x9000),
               handlers=handlers)  # fmt: skip


def text(program, name):
    return (program.built.folder / name).read_text()


UPTO_8 = {"scale": [km.UPTO(block=8)]}


def test_the_block_is_the_command_s_word_and_the_pairs_its_most():
    fixed = made(Blocks)
    fixed_text = {name: text(fixed, name) for name in ("movement.cpp", "compute.cpp")}
    sized = made(Blocks, UPTO_8)
    for name, before in fixed_text.items():
        said = text(sized, name)
        assert "const uint32_t block = command.a3;" in said
        assert "[[maybe_unused]] const uint32_t kRead0Block = block;" in said
        assert "[[maybe_unused]] const uint32_t kWrite0Block = block;" in said
        assert "Block = block" not in before
    assert sized.buffers == fixed.buffers  # the pairs at the most
    assert Blocks.scale.words[3] == ("block", "int")


def test_a_block_above_its_most_is_refused_by_name():
    program = made(Blocks, UPTO_8)
    a, out = grid(16), grid(16, 0x9000)
    assert list(program.scale.args(a, out, 16, 4))[2:] == [16, 4]
    with pytest.raises(ProgramError, match=r"block=9; a size given at each submit"):
        program.scale.args(a, out, 16, 9)
    with pytest.raises(ProgramError, match=r"block=3; .* divides 8"):
        program.scale.args(a, out, 16, 3)
    with pytest.raises(ProgramError, match=r"block=0"):
        program.scale.args(a, out, 16, 0)
    with pytest.raises(ProgramError, match=r"no argument 'block'"):
        program.scale.args(a, out, 16)


def test_a_program_made_before_the_class_was_made_again_is_refused():
    sized = made(Blocks, UPTO_8)
    made(Blocks)
    with pytest.raises(ProgramError, match=r"made with km.UPTO \{'block': 8\}"):
        sized.scale.args(grid(16), grid(16, 0x9000), 16, 4)


def test_listed_sets_and_km_any_beside_km_upto():
    program = made(Sizes, {"scale": [km.UPTO(block=8), {"size": 2}, {"size": 1}]})
    assert [values for values, _ in program.constant_handlers["scale"]] == [
        {"size": 2},
        {"size": 1},
    ]
    words = program.scale.args(grid(16), grid(16, 0x9000), 4, 1, 2)
    assert list(words)[2:] == [4, 2] and words.constants == {"size": 1}
    with_any = made(Sizes, {"scale": [km.UPTO(block=8), km.ANY]})
    assert "scale" in with_any.first_use.opted


@pytest.mark.parametrize(
    ("handlers", "said"),
    [
        ({"scale": [km.UPTO(block=8), {"block": 4}]}, r"block is given at each"),
        ({"scale": [km.UPTO(block=8), km.ANY(block=16)]}, r"block is given at each"),
        ({"scale": [km.UPTO(block=8), km.UPTO(block=4)]}, r"km.UPTO 2 times"),
        ({"scale": [km.UPTO]}, r"names no size"),
        ({"scale": [km.UPTO(width=8)]}, r"names no parameter of it, and is not rect"),
        ({"scale": [km.UPTO(tiles=8)]}, r"tiles is neither a constant int"),
        ({"scale": [km.UPTO(block=2.5)]}, r"block is a constant int: its most is"),
        ({"scale": [km.UPTO(rect=(4, 4))]}, r"reads no self.x, self.y or broadcast"),
        ({"scales": [km.UPTO(block=8)]}, r"names 'scales', not a handler"),
    ],
)
def test_km_upto_refusals(handlers, said):
    with pytest.raises(ProgramError, match=said):
        made(Blocks, handlers)


def test_a_block_with_rows_is_refused():
    class Rows(km.Program):
        @km.handler
        def scale(self, a: km.Tensor, out: km.Tensor, block: km.constant[int] = 8):
            results = km.Stream(out.pages(0, 16), write=True)
            for x in km.Stream(a.pages(0, 16), block=block, row=2):
                results.push(twice(x))

    with pytest.raises(ProgramError, match=r"of rows of a rectangle or of row="):
        made(Rows, UPTO_8)
    with pytest.raises(ProgramError, match=r"most 0, an int"):
        _sized(0)


class Wide(km.Program):
    @km.handler
    def stripe(self, out: km.Tensor, x: km.Tensor):
        c = self.rank * 2
        results = km.Stream(out.pages(self.rank * 4, 4), write=True, block=2)
        for tile in km.Stream(x.tiles[0:2, c : c + 2], block=2):
            results.push(twice(tile))


def tall(columns, address=0x1000):
    """A bfloat16 tensor of two rows of `columns` tiles."""
    return Tensor(Pages(2048, 2 * columns, address), (64, columns * 32), "bfloat16",
                  "tile")  # fmt: skip


def test_a_width_given_at_each_submit_is_the_rows_pitch():
    program = Wide(l1=1 << 20, log=quiet, out=grid(16, 0x9000), x=tall(8),
                   handlers={"stripe": [km.UPTO(x=64)]})  # fmt: skip
    said = text(program, "movement.cpp")
    assert "const uint32_t x_columns = command.a2;" in said
    assert "x_columns" in said.split("const km::Stream km_read0")[1].split(";")[0]
    assert Wide.stripe.words == [("out", "tensor"), ("x", "tensor"), ("x", "columns")]
    assert list(program.stripe.args(grid(16, 0x9000), tall(16)))[2] == 16
    assert list(program.stripe.args(grid(16, 0x9000), tall(64)))[2] == 64
    with pytest.raises(ProgramError, match=r"x is 2080 columns wide; .* at most 64"):
        program.stripe.args(grid(16, 0x9000), tall(65))
