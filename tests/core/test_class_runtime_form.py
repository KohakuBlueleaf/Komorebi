"""A class in the runtime's form (form="runtime"): the build key and the folders, the
sources and arguments the runtime takes, the refusals, a class made beside other
programs, and each program's per-core args in a run of several (no card)."""

from pathlib import Path

import pytest

import komorebi as km
from komorebi import overlay, programs
from komorebi.dsl.program import form_key
from komorebi.errors import ProgramError
from komorebi.tensor import Tensor


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tiles(count):
    """A bfloat16 tensor of `count` tiles."""
    return Tensor(Pages(2048, count), (count * 32, 32), "bfloat16", "tile")


def line_of(text):
    """The number of the line of this file that holds `text`."""
    lines = Path(__file__).read_text().split("\n")
    return next(n for n, line in enumerate(lines, start=1) if text in line)


def quiet(line):
    pass


def made(cls, **given):
    return cls(source=tiles(64), target=tiles(64), l1=1 << 20, log=quiet, **given)


@km.tile_function
def doubled(x):
    return x + x


class Copy(km.Program):
    @km.handler
    def copy(self, source: km.Tensor, target: km.Tensor, count: int):
        out = km.Stream(target.pages(self.rank * count, count), write=True)
        for tile in km.Stream(source.pages(self.rank * count, count)):
            out.push(doubled(tile))


class Visit(km.Message):
    hops: int


class Walk(km.Program):
    @km.handler
    def walk(self, source: km.Tensor, target: km.Tensor, hops: int):
        for message in self.arrive(Visit):  # a dynamic handler
            self.forward(hops, message)


class Started(km.Program):
    @km.handler
    def copy(self, source: km.Tensor, target: km.Tensor):
        km.cxx("prepare();", at="start")  # a body at the kernel's start
        km.Stream(source.pages(0, 1)).drain()


class StartedUp(km.Program, startup=km.Startup("ones", "ones", "ones")):
    ones: km.L1[km.bf16, 1, 32, 32] = km.own(push=None)

    @km.handler
    def copy(self, source: km.Tensor, target: km.Tensor):
        km.Stream(source.pages(0, 1)).drain()


class Ended(km.Program):
    @km.handler
    def copy(self, source: km.Tensor, target: km.Tensor):
        km.cxx("noc_async_write_barrier();", on="mover_b", at="end")  # at the end
        km.Stream(source.pages(0, 1)).drain()


class Overlaid(km.Program):
    @km.handler(overlay=True)
    def copy(self, source: km.Tensor, target: km.Tensor, count: int):
        out = km.Stream(target.pages(self.rank * count, count), write=True)
        for tile in km.Stream(source.pages(self.rank * count, count)):
            out.push(doubled(tile))


def test_the_form_is_in_the_key_and_the_kernel_forms_key_is_as_it_was():
    assert form_key("ab12cd34", "kernel") == "ab12cd34"
    runtime = form_key("ab12cd34", "runtime")
    assert runtime != "ab12cd34" and len(runtime) == 8
    assert form_key("ab12cd34", "kernel", {0, 16}) not in ("ab12cd34", runtime)
    assert form_key("ab12cd34", "runtime", {0}) != runtime


def test_the_two_forms_of_one_class():
    kernel, runtime = made(Copy), made(Copy, form="runtime")
    assert kernel.form() == "kernel" and runtime.form() == "program"
    assert kernel.built.folder != runtime.built.folder
    assert set(kernel.compile_args) == {"movement"}
    assert set(runtime.compile_args) == {"reader", "writer"}
    assert runtime.compile_args["reader"] == kernel.compile_args["movement"]
    sources = runtime.sources()
    assert (
        sources["reader"]
        == sources["writer"]
        == runtime.built.folder / ("movement.cpp")
    )
    assert [b.index for b in runtime.buffers] == [b.index for b in kernel.buffers]
    movement = sources["reader"].read_text()
    assert "static bool km_handlers(" in movement
    assert "void km_program(km::Reader& reader, const km_command_t& command)" in (
        movement
    )
    assert "bool km_program(km::Writer& writer, const km_command_t& command)" in (
        movement
    )
    assert "kernel_main" not in movement
    compute = sources["compute"].read_text()
    head = "void km_program(km::Compute& compute, const km_command_t& command) {"
    assert head in compute and "compute_kernel_hw_startup(" not in compute
    first_in = min(b.index for b in runtime.buffers if b.index < 16)
    first_out = min(b.index for b in runtime.buffers if b.index >= 16)
    starts = compute.index(head) + len(head)
    assert compute.index(f"reconfig_data_format({first_in}, {first_in});") > starts
    assert f"pack_reconfig_data_format({first_out});" in compute


def test_an_unknown_form_is_refused():
    with pytest.raises(ProgramError, match="form"):
        made(Copy, form="overlay")


def test_what_the_runtimes_form_has_no_place_for_is_refused_by_its_line():
    line = line_of("def walk(self, source")
    with pytest.raises(ProgramError, match=rf"Walk.walk \(line {line}\): a dynamic"):
        made(Walk, form="runtime")
    line = line_of('on="mover_b", at="end")')
    with pytest.raises(ProgramError, match=rf"kernel's end \(line {line}\)"):
        made(Ended, form="runtime")
    made(Ended)  # each is built in the kernel form


def test_an_overlay_handler_is_an_overlay_of_the_runtime_form_program():
    program = made(Overlaid, form="runtime")
    assert set(program.overlays) == {"copy"}
    case = program.sources()["reader"].read_text()
    assert "km::Overlay km_part(movement, kOverlayCopy);" in case


def hand(tmp_path, name, buffers, compute=None, args=None):
    """A hand program in the runtime's form."""
    folder = tmp_path / name
    folder.mkdir()
    text = "void km_program(km::{}& kernel, const km_command_t& command) {{}}\n"
    for kernel, kind in (("reader", "Reader"), ("writer", "Writer")):
        (folder / f"{kernel}.cpp").write_text(text.format(kind))
    (folder / "compute.cpp").write_text(text.format("Compute"))
    return km.Program(
        name,
        kernels=folder,
        buffers=[km.Buffer(i, "float32", 4096, 2) for i in buffers],
        handlers={"one": 1},
        compute=compute or {},
        args=args,
    )


def test_a_class_made_beside_programs_leaves_their_indices_and_takes_their_modes(
    tmp_path,
):
    modes = {"unpack_to_dest_mode": {1: "UnpackToDestFp32"}}
    other = hand(tmp_path, "other", [0, 1, 16], {"fp32_dest_acc_en": True, **modes})
    alone, beside = made(Copy, form="runtime"), made(Copy, form="runtime", beside=other)
    assert {b.index for b in alone.buffers} & {0, 16}
    assert not {b.index for b in beside.buffers} & {0, 1, 16}
    assert beside.compute["unpack_to_dest_mode"] == {1: "UnpackToDestFp32"}
    assert "unpack_to_dest_mode" not in alone.compute
    assert beside.built.folder != alone.built.folder
    with pytest.raises(ProgramError, match="beside= takes programs"):
        made(Copy, form="runtime", beside=[other, "gate"])


def test_each_programs_per_core_args_have_their_place_in_a_run_of_several(tmp_path):
    first = hand(tmp_path, "first", [0], args=lambda core: ([core, 7], [core], []))
    second = hand(tmp_path, "second", [1])
    third = hand(tmp_path, "third", [2], args=lambda core: ([9], [], [core, 5, 6]))
    placed = programs.plan([first, second, third], 4)
    assert [p.args_at for p in placed.programs] == [0, 2, 2]
    assert programs.core_args(placed, 3) == ([3, 7, 9], [3, 0], [0, 0, 3, 5, 6])
    defines = programs.program_overlay(third).defines
    assert defines["KM_PROGRAM_ARGS_AT"] == "2"
    assert "KM_PROGRAM_ARGS_AT" not in programs.program_overlay(first).defines
    alone = programs.plan(third, 4)
    assert alone.programs[0].args_at == 0


def test_a_class_programs_own_overlays_take_its_compile_args(tmp_path):
    program = made(Overlaid, form="runtime")
    other = hand(tmp_path, "other", [30], compute=program.compute)
    placed = programs.plan([program, other], 120)
    (own,) = programs.own_overlays(placed.programs[0])
    assert own.compile_args["reader"] == program.compile_args["reader"]
    assert overlay.part_defines(own, "reader")[1] == program.compile_args["reader"]
    (bare,) = programs.own_overlays(hand_with_overlay(tmp_path))
    assert bare.compile_args == {} and overlay.part_defines(bare, "reader")[1] is None


def hand_with_overlay(tmp_path):
    """A hand program in the runtime's form with an overlay of its own, no args."""
    program = hand(tmp_path, "owner", [3])
    part = tmp_path / "owner" / "part.cpp"
    part.write_text("uint32_t km_overlay(km::MoverRuntime& kernel) { return 1; }\n")
    program.overlay("part", {"reader": part}, number=1)
    return program


def test_a_kernel_form_class_in_a_run_of_several_is_told_the_form(tmp_path):
    other = hand(tmp_path, "other", [30])
    with pytest.raises(ProgramError, match=r'kernel_main; make the class with form="'):
        programs.plan([made(Copy), other], 120)
    with pytest.raises(ProgramError, match=r"which refuses it: Walk.walk \(line"):
        programs.plan({made(Walk): (0, 60), other: (60, 60)}, 120)
    with pytest.raises(ProgramError, match=r"kernel_main$"):
        programs.plan([hand_kernel(tmp_path), other], 120)


def hand_kernel(tmp_path):
    """A hand program in the kernel form."""
    folder = tmp_path / "kernel"
    folder.mkdir()
    for kernel in ("reader", "writer", "compute"):
        (folder / f"{kernel}.cpp").write_text("void kernel_main() {}\n")
    return km.Program("kernel", kernels=folder, handlers={"one": 1})
