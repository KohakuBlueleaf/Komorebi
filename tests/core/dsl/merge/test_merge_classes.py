"""Classes merged into one program (km.merge of made class programs, and a class
deriving two classes): the numbering, each class's own constants, a tensor name two
classes give different examples, a shared and a clashing field, the markers, the
pick of a handler to become an overlay, and the refusals (no card)."""

import pytest
from class_constants_fixture import grid, project

import komorebi as km
from komorebi.dsl.compile.records import Refused
from komorebi.errors import KernelSizeError, ProgramError
from komorebi.program.description import KernelSizes
from komorebi.session.run import Run

TENSORS = {"x": grid(1, 8), "w": grid(8, 8), "y": grid(1, 8)}


def quiet(line):
    pass


@km.tile_function
def doubled(a):
    return a + a


class Sums(km.Program):
    K_TILES = km.constant(4)
    sums: km.L1[km.fp32, 1, 2, 32, 32]

    @km.handler
    def split(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0 : self.K_TILES], block=1),
            km.Stream(w.tiles[0 : self.K_TILES, 0:2], block=2),
        ):
            self.sums += project(xb, wb)
        y.tiles[0:1, 0:2].write(self.sums)

    @km.handler
    def copy(self, x: km.Tensor, y: km.Tensor, tiles: int):
        out = km.Stream(y.pages(0, tiles), write=True, block=2)
        for t in km.Stream(x.pages(0, tiles), block=2):
            out.push(doubled(t))


class Copies(km.Program):
    K_TILES = km.constant(8)
    sums: km.L1[km.fp32, 1, 2, 32, 32]  # as Sums's: one field

    @km.handler
    def copy(self, x: km.Tensor, y: km.Tensor, tiles: int):
        out = km.Stream(y.pages(0, tiles), write=True, block=self.K_TILES)
        for t in km.Stream(x.pages(0, tiles), block=self.K_TILES):
            out.push(doubled(t))


class Wider(km.Program):
    sums: km.L1[km.fp32, 1, 4, 32, 32]  # Sums's name, another shape

    @km.handler
    def widen(self, x: km.Tensor, y: km.Tensor, tiles: int):
        out = km.Stream(y.pages(0, tiles), write=True, block=2)
        for t in km.Stream(x.pages(0, tiles), block=2):
            out.push(doubled(t))


def made(cls, **tensors):
    return cls(l1=1 << 20, log=quiet, **(tensors or TENSORS))


def test_merged_programs_numbering_constants_and_tensors():
    sums = made(Sums)
    copies = made(Copies, x=grid(1, 8), y=grid(1, 8))  # its own x and y
    merged = km.merge(sums, copies, log=quiet)
    assert merged.handlers == {"split": 1, "sums_copy": 2, "copies_copy": 3}
    assert int(merged.handler(copies.copy)) == 3
    assert int(merged.handler(sums, "copy")) == 2 and int(merged.handler("split")) == 1
    with pytest.raises(ValueError, match="'copy' is in sums and copies"):
        merged.handler("copy")
    run = Run.__new__(Run)
    run.programs = [merged]
    assert run._handler(copies.copy) == (3, 0)
    assert [b for b in merged.buffers if b.format == "float32"]  # one field sums


def test_a_class_deriving_two_classes():
    class Layer(Sums, Copies):
        pass

    layer = made(Layer)
    assert int(Layer.split) == 1 and layer.handlers["copies_copy"] == 3
    with pytest.raises(ProgramError, match="a handler of Sums and Copies"):
        _ = Layer.copy
    assert int(layer.handler(Copies, "copy")) == 3
    run = Run.__new__(Run)
    run.programs = [layer]
    assert run._handler(Layer.split) == (1, 0)
    with pytest.raises(ProgramError, match="each class keeps its own"):
        Layer(l1=1 << 20, log=quiet, K_TILES=2, **TENSORS)


def test_a_clashing_field_and_the_markers():
    with pytest.raises(Refused, match="field sums is .* in Sums and .* in Wider"):
        km.merge(made(Sums), made(Wider), log=quiet)
    sums, copies = made(Sums), made(Copies)
    marked = km.merge(sums, copies, overlay=[copies.copy], log=quiet)
    assert list(marked.compiled.overlays) == ["copies_copy"]
    assert "copies_copy" in marked.overlays
    whole = km.merge(sums, copies, overlay=[copies], resident=["split"], log=quiet)
    assert list(whole.compiled.overlays) == ["copies_copy"]
    with pytest.raises(ProgramError, match="marked both overlay and resident"):
        km.merge(sums, copies, overlay=["split"], resident=["split"], log=quiet)


def test_the_pick_and_the_refusals():
    merged = km.merge(made(Sums), made(Copies), resident=["split"], log=quiet)
    sizes = merged.code_sizes()
    sizes_error = KernelSizeError("m", KernelSizes({"BRISC": 1}, 99, 9), 9)
    assert merged.spill(sizes_error)
    largest = max((size, name) for name, size in sizes.items() if name != "split")[1]
    assert list(merged.compiled.overlays) == [largest]
    assert merged.spill(sizes_error) and not merged.spill(sizes_error)
    assert "split" not in merged.compiled.overlays  # marked resident
    hand = km.Program("hand", kernels={}, buffers=[], handlers={"a": 1})
    with pytest.raises(ProgramError, match="a class in the kernel form merged with"):
        km.merge(made(Sums), hand)


def test_a_spill_during_a_load_keeps_the_defines_the_load_added():
    merged = km.merge(made(Sums), made(Copies), log=quiet)
    built = dict(merged.defines)
    merged.defines = {**built, "KM_PAGE_LINKS": 1}  # as an island load puts them
    seen = []

    class Session:
        def load(self, **arguments):
            seen.append(dict(merged.defines))
            if len(seen) == 1:  # the first build does not fit: a handler spills
                raise KernelSizeError("m", KernelSizes({"BRISC": 1}, 99, 9), 9)
            return "run"

    assert merged.fits_kernels(Session(), {}) == "run"
    assert merged.compiled.overlays and all(d["KM_PAGE_LINKS"] == 1 for d in seen)
    assert {k: v for k, v in seen[1].items() if k != "KM_PAGE_LINKS"} == built


class Both(Sums, Copies):
    """Sums and Copies as one class (a class deriving two)."""


def test_a_spills_rebuild_keeps_what_the_make_was_given():
    given = {"depth": 2, "guard": False, "broadcast_wait": True}
    both = Both(l1=1 << 20, log=quiet, defines={"MY_WORD": "3"}, **given, **TENSORS)
    before = {key: getattr(both, key) for key in given} | {"defines": both.defines}
    assert both.spill(KernelSizeError("m", KernelSizes({"BRISC": 1}, 99, 9), 9))
    assert both.compiled.overlays  # built again, a handler as an overlay
    assert {key: getattr(both, key) for key in given} == {
        key: before[key] for key in given
    }
    assert both.defines == before["defines"] and both.defines["MY_WORD"] == "3"
