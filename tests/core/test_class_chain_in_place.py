"""A direct input in place (km.handler(direct=..., in_place=True)): the reading stream's
own pair takes the roots' sends, its refusals by name; a root's block sent as written,
in pieces too (no card)."""

import pytest
from chain_fixture import Handlers, quiet, run_of, tensors

import komorebi as km
from komorebi import chain as kc
from komorebi.errors import ProgramError

GIVEN = tensors()
X, Y, Z, S, T = (GIVEN[name] for name in "xyzst")


class Mix(km.Program):
    """The halves of s (8 cores, 2 tiles each) summed whole on each of 4 cores, s in
    place."""

    K = km.constant(16)

    @km.handler
    def halves(self, a: km.Tensor, out: km.Tensor):
        first = self.rank * 2  # 2 tiles, each by .write (a write stream is not sent)
        left = km.tile.zeros(km.fp32)
        for tile in km.Stream(a.pages(first, 1)):
            left += tile
        out.pages(first, 1).write(left)
        right = km.tile.zeros(km.fp32)
        for tile in km.Stream(a.pages(first + 1, 1)):
            right += tile
        out.pages(first + 1, 1).write(right)

    @km.handler(direct=("s",), in_place=True)
    def total(self, s: km.Tensor, out: km.Tensor):
        whole = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(0, self.K)):
            whole += tile
        out.pages(self.rank, 1).write(whole)

    @km.chain
    def both(self, x, s, out):
        self.halves(x, s, cores=(0, 8))
        self.total(km.direct(s), out, cores=(0, 4))

    @km.chain
    def again(self, x, s, out):
        self.halves(x, s, cores=(0, 8))
        self.total(s, out, cores=(0, 4))
        self.halves(x, s, cores=(0, 8))
        self.total(km.direct(s), out, cores=(0, 4))


class Copied(Mix):
    total = km.handler(direct=("s",))(Mix.total.function)


def made(cls):
    return cls(l1=1 << 20, log=quiet, a=X, out=Y, s=S)


def text(program, name):
    return (program.built.folder / name).read_text()


def case(source, handler):
    start = source.index(f"case kHandler{handler}: {{")
    end = source.find("case kHandler", start + 1)
    return source[start : end if end > 0 else len(source)]


def test_the_stream_s_own_pair_takes_the_input_and_only_its_handler_uses_it():
    program = made(Mix)
    assert program.direct_in == ("total", "s", "kRead0")
    header = text(program, "mix.h")
    pair = next(line for line in header.splitlines() if "kDirectIn =" in line)
    first, second = (int(n) for n in pair.split("{")[1].rstrip("};").split(","))
    assert max(first, second) < 32  # the movers' streams take buffers under 32
    assert 'kDirectInPattern = km::pattern("AAAB")' in header
    assert "kDirectArea =" not in header and "kDirectAreaTiles = 16" in header
    movement, compute = text(program, "movement.cpp"), text(program, "compute.cpp")
    assert "km::in_place(S(s), km_direct_area(movement).base != 0)" in movement
    assert "kRead0" not in case(movement, "Total")
    assert "kRead0" not in case(compute, "Total")
    assert "kDirectIn" in case(compute, "Total")
    assert "kRead0" in case(movement, "Halves") and "kDirectIn" not in case(
        compute, "Halves"
    )
    assert "km::InPlaceLanding{" in movement and "LinearLanding" not in movement
    (pair,) = [p for p in program.pairs if p.mover_a == first]
    assert pair.counts() == (12, 4)  # 16 blocks of 1 under AAAB


def test_a_copied_input_lands_in_the_area_and_both_are_sent_as_written():
    for cls, landing in ((Copied, "km::LinearLanding{"), (Mix, "km::InPlaceLanding{")):
        movement = text(made(cls), "movement.cpp")
        assert landing in movement
        assert "km::chain_send_writes(" in movement
        assert "km::chain_mark<true>(" in movement
        assert "km_kept" not in movement and "km_direct_keep" not in movement
    assert "km::direct_or(S(s), km_direct_area(movement))" in text(
        made(Copied), "movement.cpp"
    )


def test_in_place_of_a_partition_s_share_is_refused():
    class Summed(km.Program):
        sum = km.handler(direct=("s",), in_place=True)(Handlers.sum.function)

        @km.chain
        def once(self, s, t, tiles: int):
            self.sum(s, t, tiles)

    why = r"Summed.sum: in_place for s: s is read by 1 streams, not in pages in order"
    with pytest.raises(ProgramError, match=why):
        Summed(l1=1 << 20, log=quiet, s=S, t=T)


def test_in_place_without_constant_tiles_is_refused():
    class Counted(km.Program):
        @km.handler(direct=("s",), in_place=True)
        def total(self, s: km.Tensor, out: km.Tensor, tiles: int):
            whole = km.tile.zeros(km.fp32)
            for tile in km.Stream(s.pages(0, tiles)):
                whole += tile
            out.pages(self.rank, 1).write(whole)

        @km.chain
        def once(self, s, out, tiles: int):
            self.total(s, out, tiles, cores=(0, 4))

    why = (
        r"Counted.total: in_place for s: its stream's tiles \(tiles\) and repetitions "
        r"\(1\) must be constants of the class"
    )
    with pytest.raises(ProgramError, match=why):
        Counted(l1=1 << 20, log=quiet, s=S, out=Y)


def test_two_direct_inputs_in_place_are_refused():
    class Two(Mix):
        scale = km.handler(direct=("a",))(Handlers.scale.function)

    why = r"Two.total: in_place for s: the class declares 2 direct inputs"
    with pytest.raises(ProgramError, match=why):
        made(Two)


def test_an_earlier_call_of_the_in_place_handler_is_refused():
    program = made(Mix)
    run = run_of(program)
    assert run.chain(program.both(X, S, Y)).calls[1].direct is not None
    why = (
        r"call 3 \(Mix.total\): its direct input lands in place, in the halves of "
        r"total's stream, which call 1 \(Mix.total\) may still be reading"
    )
    with pytest.raises(kc.ChainRefused, match=why):
        run.chain(program.again(X, S, Y))


class Pieces(km.Program):
    """Each core's 2 tiles of out written in 2 pieces (and with a gap: refused)."""

    @km.handler
    def halves(self, a: km.Tensor, out: km.Tensor):
        first = self.rank * 2
        left = km.tile.zeros(km.fp32)
        for tile in km.Stream(a.pages(first, 1)):
            left += tile
        out.pages(first, 1).write(left)  # the first piece, before the second's sum
        right = km.tile.zeros(km.fp32)
        for tile in km.Stream(a.pages(first + 1, 1)):
            right += tile
        out.pages(first + 1, 1).write(right)

    @km.handler
    def apart(self, a: km.Tensor, out: km.Tensor):
        first = self.rank * 2
        left = km.tile.zeros(km.fp32)
        for tile in km.Stream(a.pages(first, 1)):
            left += tile
        out.pages(first, 1).write(left)  # the first piece, before the second's sum
        right = km.tile.zeros(km.fp32)
        for tile in km.Stream(a.pages(first + 1, 1)):
            right += tile
        out.pages(first + 2, 1).write(right)

    @km.chain
    def joined(self, a, out):
        self.halves(a, out, cores=(0, 8))

    @km.chain
    def gapped(self, a, out):
        self.apart(a, out, cores=(0, 8))


def test_a_root_s_block_in_pieces_is_one_block_and_a_gap_is_refused():
    program = Pieces(l1=1 << 20, log=quiet, a=X, out=Y)
    run = run_of(program)
    (call,) = run.chain(program.joined(X, Y)).calls
    assert call.width == 2
    why = (
        r"line \d+: group 0 writes out pages 0 to 2 with a gap or an overlap; a "
        r"chain's call writes block g of its output, tiles g x 2 to g x 2 \+ 1"
    )
    with pytest.raises(kc.ChainRefused, match=why):
        run.chain(program.gapped(X, Y))
