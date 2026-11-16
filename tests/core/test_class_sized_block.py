"""A stream's block given at each submit (the internal hook, front._sized(8)): the
pair sized at the most, the handler's block a word, the host refusing a value that
does not divide the most, by name; a constant block unchanged (no card)."""

import pytest
from test_class_thread_header import Pages, grid, quiet, twice

import komorebi as km
from komorebi.dsl.front import _sized
from komorebi.errors import ProgramError
from komorebi.tensor import Tensor


class Sized(km.Program):
    @km.handler
    def scale(self, a: km.Tensor, out: km.Tensor, tiles: int,
              block: int = _sized(8)):  # fmt: skip
        results = km.Stream(out.pages(0, tiles), write=True, block=block)
        for x in km.Stream(a.pages(0, tiles), block=block):
            results.push(twice(x))


class Fixed(km.Program):
    @km.handler
    def scale(self, a: km.Tensor, out: km.Tensor, tiles: int):
        results = km.Stream(out.pages(0, tiles), write=True, block=8)
        for x in km.Stream(a.pages(0, tiles), block=8):
            results.push(twice(x))


def made(cls):
    return cls(l1=1 << 20, log=quiet, a=grid(16), out=grid(16, 0x9000))


def text(program, name):
    return (program.built.folder / name).read_text()


def test_the_block_is_the_command_s_word_and_the_pairs_its_most():
    sized, fixed = made(Sized), made(Fixed)
    for name in ("movement.cpp", "compute.cpp"):
        said = text(sized, name)
        assert "const uint32_t block = command.a3;" in said
        assert "[[maybe_unused]] const uint32_t kRead0Block = block;" in said
        assert "[[maybe_unused]] const uint32_t kWrite0Block = block;" in said
        assert "Block = block" not in text(fixed, name)
    assert sized.buffers == fixed.buffers  # the pairs at the most
    assert Sized.scale.words[3] == ("block", "int")


def test_a_block_above_its_most_is_refused_by_name():
    program = made(Sized)
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


def test_refusals():
    class Rows(km.Program):
        @km.handler
        def scale(self, a: km.Tensor, out: km.Tensor, block: int = _sized(8)):
            results = km.Stream(out.pages(0, 16), write=True)
            for x in km.Stream(a.pages(0, 16), block=block, row=2):
                results.push(twice(x))

    with pytest.raises(ProgramError, match=r"of rows of a rectangle or of row="):
        made(Rows)
    with pytest.raises(ProgramError, match=r"most 0, an int"):
        _sized(0)


WIDTH_64 = _sized(64)  # a width given at each submit, at most 64 tiles


class Wide(km.Program):
    @km.handler
    def stripe(self, out: km.Tensor, x: km.Tensor = WIDTH_64):
        c = self.rank * 2
        results = km.Stream(out.pages(self.rank * 4, 4), write=True, block=2)
        for tile in km.Stream(x.tiles[0:2, c : c + 2], block=2):
            results.push(twice(tile))


def tall(columns, address=0x1000):
    """A bfloat16 tensor of two rows of `columns` tiles."""
    return Tensor(Pages(2048, 2 * columns, address), (64, columns * 32), "bfloat16",
                  "tile")  # fmt: skip


def test_a_width_given_at_each_submit_is_the_rows_pitch():
    program = Wide(l1=1 << 20, log=quiet, out=grid(16, 0x9000), x=tall(8))
    said = text(program, "movement.cpp")
    assert "const uint32_t x_columns = command.a2;" in said
    assert "x_columns" in said.split("const km::Stream km_read0")[1].split(";")[0]
    assert Wide.stripe.words == [("out", "tensor"), ("x", "tensor"), ("x", "columns")]
    assert list(program.stripe.args(grid(16, 0x9000), tall(16)))[2] == 16
    assert list(program.stripe.args(grid(16, 0x9000), tall(64)))[2] == 64
    with pytest.raises(ProgramError, match=r"x is 2080 columns wide; .* at most 64"):
        program.stripe.args(grid(16, 0x9000), tall(65))
