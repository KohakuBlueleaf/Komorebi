"""A once-only object made at one loop depth and used at another (no card): a tile of
DST, a reduce's sum, a cast block, a filled field, a drained stream. Each wrong form is
refused at the consuming line, naming the right forms; each right form still builds.
A chain of a broadcast's handler is refused where its cast reads a tensor in an if."""

import pytest
from chain_fixture import quiet, run_of
from class_lines_fixture import Pages
from class_movers_fixture import nocs
from class_movers_fixture import tensors as mover_tensors

import komorebi as km
from komorebi.chain import calls as kc
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import compute_source, movement_source


def grid(dtype="bfloat16"):
    return Tensor(Pages(2048), (256, 128), dtype, "tile")


def given():
    both = {name: grid() for name in ("x", "y", "out", "w")}
    both["lists"] = Tensor(Pages(4096), (8, 1024), "uint32", "row")
    return both


def built(cls):
    """The class's movement and compute text, in the kernel form."""
    compiled = compile_class(cls, {}, given())
    return movement_source(compiled), compute_source(compiled)


@km.tile_function
def project(x, w):
    return x @ w


class WriteTotalInLoop(km.Program):
    """A running total: a tile made outside a loop, written in it, a page a turn of
    the sum so far (carried through an fp32 buffer at the write)."""

    @km.handler
    def h(self, x: km.Tensor, out: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for i in range(2):
            for tile in km.Stream(x.pages(i * 2, 2)):
                total += tile
            out.pages(i, 1).write(total)


class ZerosInLoop(km.Program):
    """The right form: the tile made in the loop that writes it."""

    @km.handler
    def h(self, x: km.Tensor, out: km.Tensor):
        for i in range(2):
            total = km.tile.zeros(km.fp32)
            for tile in km.Stream(x.pages(i * 2, 2)):
                total += tile
            out.pages(i, 1).write(total)


class AccumulateAcrossRange(km.Program):
    """The right form: added to in a loop, written after it."""

    @km.handler
    def h(self, x: km.Tensor, out: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for i in range(2):
            for tile in km.Stream(x.pages(i * 2, 2)):
                total += tile
        out.pages(0, 1).write(total)


class ReduceTotalOutside(km.Program):
    """A running total reduced a turn: the own total carried around the reduce."""

    @km.handler
    def h(self, x: km.Tensor, out: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for i in range(2):
            for tile in km.Stream(x.pages(i * 2, 2)):
                total += tile
            whole = km.reduce(total, form="tree")
            if self.rank == self.leader:
                out.pages(i, 1).write(whole)


class WriteTotalInIf(km.Program):
    """A running total's write under an if: its carry would be taken unconditionally
    at the turn's end."""

    @km.handler
    def h(self, x: km.Tensor, out: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for i in range(2):
            for tile in km.Stream(x.pages(i * 2, 2)):
                total += tile
            if i == 1:
                out.pages(i, 1).write(total)


class ReduceInLoop(km.Program):
    """The right form: the tile, the reduce and the write all in the loop."""

    @km.handler
    def h(self, x: km.Tensor, out: km.Tensor):
        for i in range(2):
            total = km.tile.zeros(km.fp32)
            for tile in km.Stream(x.pages(i * 2, 2)):
                total += tile
            whole = km.reduce(total, form="tree")
            if self.rank == self.leader:
                out.pages(i, 1).write(whole)


class ReduceWriteInLoop(km.Program):
    """A leader's sum made once and written in a loop: held across it, a copy a turn."""

    @km.handler
    def h(self, x: km.Tensor, out: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(x.pages(0, 2)):
            total += tile
        whole = km.reduce(total, form="tree")
        for i in range(2):
            if self.rank == self.leader:
                out.pages(i, 1).write(whole)


class ReduceWriteNotLeader(km.Program):
    """A sum written in a loop outside the leader's if: still refused."""

    @km.handler
    def h(self, x: km.Tensor, out: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(x.pages(0, 2)):
            total += tile
        whole = km.reduce(total, form="tree")
        for i in range(2):
            out.pages(i, 1).write(whole)


class ReduceWriteOnce(km.Program):
    """The right form: the sum written once, outside every loop."""

    @km.handler
    def h(self, x: km.Tensor, out: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(x.pages(0, 2)):
            total += tile
        whole = km.reduce(total, form="tree")
        if self.rank == self.leader:
            out.pages(0, 1).write(whole)


class SumTakenInLoop(km.Program):
    sums: km.L1[km.fp32, 1, 4, 32, 32]

    @km.tile_function
    def project(x, w):
        return x @ w

    @km.handler
    def add(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        k0 = self.rank % 2 * 2
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0 : k0 + 2], block=2),
            km.Stream(w.tiles[k0 : k0 + 2, 0:4], block=8),
        ):
            self.sums += self.project(xb, wb)
        whole = km.reduce(self.sums, form="tree", group=2)
        if self.rank == self.leader:
            for r in range(2):  # noqa: B007 (a turn each)
                km.cxx("f(whole_buffer, whole_index);", takes=[whole])


class SumTakenOnce(SumTakenInLoop):
    """The right form: the group's sum taken once, by the leader."""

    @km.handler
    def add(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        k0 = self.rank % 2 * 2
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0 : k0 + 2], block=2),
            km.Stream(w.tiles[k0 : k0 + 2, 0:4], block=8),
        ):
            self.sums += self.project(xb, wb)
        whole = km.reduce(self.sums, form="tree", group=2)
        if self.rank == self.leader:
            km.cxx("f(whole_buffer, whole_index);", takes=[whole])


class CastUsedInRange(km.Program):
    """A block cast once and taken in a loop: held, read each turn, freed after."""

    @km.handler
    def h(self, x: km.Tensor, out: km.Tensor):
        row = km.Broadcast(self, along="row", window=2)
        if row.sender():  # noqa: SIM108 (the compiler's form)
            tiles = row.cast(x.pages(self.y, 1).read())
        else:
            tiles = row.arrive()
        total = km.tile.zeros(km.fp32)
        for i in range(2):  # noqa: B007 (a turn each)
            for tile in tiles:
                total += tile
        out.pages(self.rank, 1).write(total)


class CastInRange(km.Program):
    """The right form: the cast in the loop that takes it."""

    @km.handler
    def h(self, x: km.Tensor, out: km.Tensor):
        row = km.Broadcast(self, along="row", window=2)
        total = km.tile.zeros(km.fp32)
        for i in range(2):
            if row.sender():
                tiles = row.cast(x.pages(self.y * 2 + i, 1).read())
            else:
                tiles = row.arrive()
            for tile in tiles:
                total += tile
        out.pages(self.rank, 1).write(total)


class FillInRange(km.Program):
    """A field filled each turn and read by that turn's product: freed at the end of
    the turn."""

    weight: km.L1[km.bf16, 2, 2, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        for i in range(2):
            self.weight.fill(km.Stream(w.pages(i * 4, 4)))
            acc = km.tile.zeros(km.fp32, 2)
            for block in km.Stream(x.pages(i * 2, 2), block=2):
                acc += project(block, self.weight)
            y.pages(i * 2, 2).write(acc)


class FillThenProductAfter(km.Program):
    """A field filled in a loop and read by a product after it: still refused (the
    product reads one fill of several)."""

    weight: km.L1[km.bf16, 2, 2, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        for i in range(2):
            self.weight.fill(km.Stream(w.pages(i * 4, 4)))
        acc = km.tile.zeros(km.fp32, 2)
        for block in km.Stream(x.pages(0, 2), block=2):
            acc += project(block, self.weight)
        y.pages(0, 2).write(acc)


class FillOutsideRange(km.Program):
    """The right form: the field filled once, the product in the loop."""

    weight: km.L1[km.bf16, 2, 2, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        self.weight.fill(km.Stream(w.pages(0, 4)))
        for i in range(2):
            acc = km.tile.zeros(km.fp32, 2)
            for block in km.Stream(x.pages(i * 2, 2), block=2):
                acc += project(block, self.weight)
            y.pages(i * 2, 2).write(acc)


class Drained(km.Program):
    """The fields of the drained-stream probes."""

    out: km.L1[km.u32, 2, 32, 32] = km.own(push="pack")
    next_slots: km.L1[km.u32, 1, 32, 32] = km.own(push="mover_b")


class WithInLoopDrainAfter(Drained):
    @km.handler
    def setup(self, lists: km.Tensor, count: int):
        for r in range(2):  # noqa: B007 (a turn each)
            with km.Stream(self.out, mover="B") as out:
                km.cxx("bin(out);", takes=[out], on="pack")
        for page in out:
            slot = self.next_slots.bump(page[1])
            lists.items(size=128).put(page, at=slot)


class WithInLoopDrainInside(Drained):
    """The right form: the with and its drain in the same loop."""

    @km.handler
    def setup(self, lists: km.Tensor, count: int):
        for r in range(2):  # noqa: B007 (a turn each)
            with km.Stream(self.out, mover="B") as out:
                km.cxx("bin(out);", takes=[out], on="pack")
            for page in out:
                slot = self.next_slots.bump(page[1])
                lists.items(size=128).put(page, at=slot)


class EndInLoop(Drained):
    @km.handler
    def setup(self, lists: km.Tensor, count: int):
        out = km.Stream(self.out, mover="B")
        for r in range(2):  # noqa: B007 (a turn each)
            km.cxx("bin(out);", takes=[out], on="pack")
            out.end()
        for page in out:
            slot = self.next_slots.bump(page[1])
            lists.items(size=128).put(page, at=slot)


class DrainInLoopEndOutside(Drained):
    @km.handler
    def setup(self, lists: km.Tensor, count: int):
        out = km.Stream(self.out, mover="B")
        km.cxx("bin(out);", takes=[out], on="pack")
        out.end()
        for r in range(2):  # noqa: B007 (a turn each)
            for page in out:
                slot = self.next_slots.bump(page[1])
                lists.items(size=128).put(page, at=slot)


class BodyInLoopEndedAfter(Drained):
    """The right form of a stream made in a loop and drained after it: every turn's
    pages, drained once."""

    @km.handler
    def setup(self, lists: km.Tensor, count: int):
        out = km.Stream(self.out, mover="B")
        for r in range(2):  # noqa: B007 (a turn each)
            km.cxx("bin(out);", takes=[out], on="pack")
        out.end()
        for page in out:
            slot = self.next_slots.bump(page[1])
            lists.items(size=128).put(page, at=slot)


class Ended(Drained):
    """The right form: made, ended and drained outside every loop."""

    @km.handler
    def setup(self, lists: km.Tensor, count: int):
        out = km.Stream(self.out, mover="B")
        km.cxx("bin(out);", takes=[out], on="pack")
        out.end()
        for page in out:
            slot = self.next_slots.bump(page[1])
            lists.items(size=128).put(page, at=slot)


@pytest.mark.parametrize(
    ("cls", "message"),
    [
        (
            WriteTotalInIf,
            r"WriteTotalInIf\.h, line \d+, `out\.pages\(i, 1\)\.write\(total\)`: "
            r"total is a running total \(made at line \d+, outside this loop\): it "
            r"is written or reduced once a turn, as a line of the range loop's own "
            r"body; move the line out of the if or inner loop",
        ),
        (
            ReduceWriteNotLeader,
            r"ReduceWriteNotLeader\.h, line \d+, `out\.pages\(i, 1\)\.write\(whole\)`: "
            r"whole was made at line \d+, outside the loop this write stands in: a "
            r"reduce makes one sum each time it runs.*; write it where it is "
            r"reduced, in the same loop .*or reduce and write both outside the loop",
        ),
        (
            FillThenProductAfter,
            r"FillThenProductAfter\.apply, line \d+, `acc \+= project\(block, "
            r"self\.weight\)`: self\.weight is filled in a loop \(line \d+\) and a "
            r"product frees it once, at the handler's end.*; fill it before the loops, "
            r"once \(self\.weight\.fill\(km\.Stream\(\.\.\.\)\)\), or take it with a "
            r"body in the loop that fills it",
        ),
        (
            WithInLoopDrainAfter,
            r"WithInLoopDrainAfter\.setup, line \d+, `for page in out:`: out was "
            r"made at line \d+ in 1 loop and its drain stands in no loop: .*make, "
            r"end and drain out in the same loop \(a with block per turn, and `for "
            r"page in out:` after it in that loop\), or all three outside the "
            r"loops; for every turn's pages drained once, make out before the loop "
            r"\(out = km\.Stream\(\.\.\.\)\), its body lines in the loop, and "
            r"out\.end\(\) and the drain after it",
        ),
        (
            EndInLoop,
            r"EndInLoop\.setup, line \d+, `out\.end\(\)`: out was made at line \d+ "
            r"in no loop and its end stands in 1 loop: ",
        ),
        (
            DrainInLoopEndOutside,
            r"DrainInLoopEndOutside\.setup, line \d+, `for page in out:`: out was "
            r"made at line \d+ in no loop and its drain stands in 1 loop: ",
        ),
    ],
)
def test_a_once_only_object_used_at_another_loop_depth_is_refused(cls, message):
    with pytest.raises(Refused, match=message):
        compile_class(cls, {}, given())


@pytest.mark.parametrize(
    ("cls", "lines"),
    [
        (ZerosInLoop, ["tile_regs_acquire"]),
        (WriteTotalInLoop, ["cb_push_back(kCarryTotal, 1);", "copy_tile(kCarryTotal"]),
        (ReduceTotalOutside, ["cb_push_back(kCarryTotal, 1);", "km::reduce"]),
        (AccumulateAcrossRange, ["tile_regs_acquire"]),
        (ReduceInLoop, ["km::reduce"]),
        (ReduceWriteOnce, ["km::reduce"]),
        (SumTakenOnce, ["whole"]),
        (CastInRange, ["km::broadcast_receive("]),
        (CastUsedInRange, ["km::broadcast_receive(", "km::broadcast_done("]),
        (FillOutsideRange, ["kFieldWeight"]),
        (FillInRange, ["this turn's fill, freed for the next"]),
        (ReduceWriteInLoop, ["the sum held across the loop, copied for this turn"]),
        (SumTakenInLoop, ["held across the loop, freed"]),
        (WithInLoopDrainInside, ["km_pushed_out"]),
        (Ended, ["km_pushed_out"]),
        (BodyInLoopEndedAfter, ["km_pushed_out"]),
    ],
)
def test_the_right_forms_still_build(cls, lines):
    movement, compute = built(cls)
    for line in lines:
        assert line in movement or line in compute, (cls.__name__, line)


def test_a_tile_made_in_a_loop_and_written_after_it_is_refused():
    class ZerosThenWrite(km.Program):
        @km.handler
        def h(self, x: km.Tensor, out: km.Tensor):
            for i in range(2):
                total = km.tile.zeros(km.fp32)
                for tile in km.Stream(x.pages(i * 2, 2)):
                    total += tile
            out.pages(0, 1).write(total)

    with pytest.raises(Refused, match=r"total was made at line \d+, inside a loop"):
        compile_class(ZerosThenWrite, {}, given())


class Casts(km.Program):
    """A broadcast's handler stated in a chain: its cast reads a tensor under an if."""

    @km.handler
    def first(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        if column.sender():
            ws = column.cast(w.tiles[0:8, 0:4].read())
        else:
            ws = column.arrive()
        acc = km.tile.zeros(km.fp32, 4)
        for xb in km.Stream(x.tiles[self.y : self.y + 1, 0:8], block=2, pattern="A"):
            acc += project(xb, ws)
        y.pages(self.rank * 4, 4).write(acc)

    @km.chain
    def steps(self, x, w, y):
        self.first(x, w, y)


@pytest.mark.parametrize("cores", [None, (0, 8), (0, 4), (0, 7)])
def test_a_chain_of_a_broadcast_handler_is_refused_at_the_cast_if(cores):
    program = Casts(
        l1=1 << 20, log=quiet, rect=(0, 0, 2, 4), chain=True, **mover_tensors()
    )
    program.rectangle["nocs"] = nocs()
    program.args(0)
    assert program.rectangle["group"] == (0, 8) and "first" in program.broadcast_lines
    shared = mover_tensors()
    with pytest.raises(
        kc.ChainRefused,
        match=(
            r"Casts\.first, line \d+: a read or write under an if on the command's "
            r"words: which cores read or write it is not known before the command "
            r"runs; it still runs as a command"
        ),
    ):
        call = Casts.first(shared["x"], shared["w"], shared["y"])
        run_of(program).chain([call], cores=cores)


class FieldWriteInLoop(km.Program):
    """A tile made outside a loop, packed into a field in it."""

    h: km.L1[km.bf16, 4, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w1: km.Tensor, w2: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, 4)
        for r in range(2):
            for xb, wb in zip(
                km.Stream(x.tiles[r : r + 1, 0:8], block=2),
                km.Stream(w1.tiles[0:8, 0:4], block=8),
            ):
                acc += project(xb, wb)
            self.h.write(acc)


def test_a_tile_made_outside_a_loop_and_packed_into_a_field_in_it_is_refused():
    both = {name: grid() for name in ("x", "w1", "w2", "y")}
    with pytest.raises(
        Refused,
        match=(
            r"FieldWriteInLoop\.run, line \d+, `self\.h\.write\(acc\)`: acc was made "
            r"at line \d+, outside the loop this write stands in"
        ),
    ):
        compile_class(FieldWriteInLoop, {}, both)


def test_a_field_filled_each_turn_is_freed_in_that_turn():
    """FillInRange: compute waits for the fill and frees it inside the turn, mover A
    reserves room for each turn's fill, and nothing frees it after the loop."""
    movement, compute = built(FillInRange)
    body = compute[compute.index("for (uint32_t i = 0;") :]
    turn = body[: body.index("this turn's fill, freed for the next")]
    assert "compute.wait_front(kFieldWeight, kFieldWeightPages)" in turn
    assert compute.count("cb_pop_front(kFieldWeight, kFieldWeightPages);") == 1
    loop = movement[movement.index("for (uint32_t i = 0;") :]
    assert "movement.reserve(kFieldWeight, kFieldWeightPages)" in loop


def test_a_sum_held_across_a_loop_is_freed_after_it():
    """ReduceWriteInLoop: each turn copies the sum into the write's own pages (no
    free in the loop); the sum is freed once, after the loop, by the leader."""
    movement, compute = built(ReduceWriteInLoop)
    body = compute[compute.index("for (uint32_t i = 0;") :]
    after = body[body.index("held across the loop, freed") :]
    turn = body[: body.index("held across the loop, freed")]
    assert "copy_tile(" in turn and "cb_pop_front(kReduce" not in turn
    assert after.count("cb_pop_front(") == 1
    loop = movement[movement.index("for (uint32_t i = 0;") :]
    assert loop.count("km::write_pages(") == 1
