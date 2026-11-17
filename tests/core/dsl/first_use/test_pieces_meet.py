"""The pieces meeting each other: km.ANY on a class made beside another program,
km.merge of classes in the runtime's form and of such a class with a hand program, a
merged class's handler named by its class, each refusal of a pair that does not meet,
the runtime's kernels given the added handlers' path (no card)."""

import shutil

import pytest
from class_first_use_fixture import FakeRun
from class_handler_constants_fixture import Gate, quiet
from pieces_meet_fixture import made

import komorebi as km
from komorebi.dsl import first_use
from komorebi.dsl.first_use import constant_handler
from komorebi.errors import ProgramError
from komorebi.program import several as programs
from komorebi.session import kernels as session_kernels


@km.tile_function
def thrice(a):
    return a + a + a


class Plain(km.Program):
    @km.handler
    def plain(self, a: km.Tensor, out: km.Tensor, tiles: int):
        results = km.Stream(out.pages(0, tiles), write=True, block=4)
        for x in km.Stream(a.pages(0, tiles), block=4):
            results.push(thrice(x))


class Other(km.Program):
    @km.handler
    def scale(self, a: km.Tensor, out: km.Tensor, tiles: int):
        results = km.Stream(out.pages(0, tiles), write=True, block=4)
        for x in km.Stream(a.pages(0, tiles), block=4):
            results.push(thrice(x))


class Both(Plain, Other):
    pass


def hand(tmp_path, compute):
    """A hand program in the runtime's form (km_program in each kernel)."""
    text = "void km_program(km::{}& kernel, const km_command_t& command) {{}}\n"
    kinds = {"reader": "Reader", "writer": "Writer", "compute": "Compute"}
    for kernel, kind in kinds.items():
        (tmp_path / f"{kernel}.cpp").write_text(text.format(kind))
    return km.Program(
        "hand",
        kernels=tmp_path,
        buffers=[km.Buffer(30, "bfloat16", 2048, 2)],
        handlers={"one": 1},
        compute=compute,
    )


class Adding(FakeRun):
    """A run's addon that also records the compile arguments given."""

    def addon(self, name, parts, handler=False, defines=None, compile_args=None):
        self.added[name] = type("Added", (), {"number": 20 + len(self.added)})()
        self.adds.append((name, parts, defines, compile_args))
        return self.added[name].number


def test_a_first_use_beside_another_program_is_built_at_the_indices_it_leaves():
    first = made(Gate, form="runtime")
    second = made(
        Gate, handlers={"gate": [{"size": 8}, km.ANY]}, form="runtime", beside=first
    )
    assert second.first_use.reserved == {b.index for b in first.buffers}
    folder = first_use.folder_of(second, Gate.gate, {"size": 4})
    shutil.rmtree(folder, ignore_errors=True)  # an earlier test run wrote it
    run = Adding([first, second])
    number, built = constant_handler(second, Gate.gate, {"size": 4}, run=run)
    assert (number, built.name) == (20, "gate_size4")
    ((_, parts, defines, compile_args),) = run.adds
    header = (folder / "gate.h").read_text()
    assert "constexpr km::BufferPair kRead0 = {4, 5};" in header  # first has 0..3
    assert defines == {"KM_ADDED_HANDLERS": "1"}
    assert compile_args == {k: second.kernel_compile_args(k) for k in compile_args}
    assert compile_args["reader"] == [2, 2048, 2, 2048]
    # the tile function's header from the folder above the part's (no class folder
    # on a runtime-form load's include path)
    assert '#include "../tile_twice.hpp"' in parts["compute"].read_text()


def test_a_kernel_form_first_use_is_added_with_its_defines_only():
    program = made(Gate, handlers={"gate": km.ANY})
    assert first_use.added_with(program) == {"defines": {"KM_ADDED_HANDLERS": "1"}}


def test_a_merge_of_runtime_form_classes_is_in_that_form():
    merged = km.merge(
        made(Plain, form="runtime"), made(Other, form="runtime"), l1=1 << 20, log=quiet
    )
    assert merged.form() == "program"
    assert merged.handlers == {"plain": 1, "scale": 2}
    run = FakeRun([merged])
    assert run._handler(Other.scale) == (2, 0)  # the class's handler, its number here
    kernel = km.merge(made(Plain), made(Other), l1=1 << 20, log=quiet)
    assert kernel.form() == "kernel"  # classes in the kernel form: one class program


def test_a_runtime_form_class_merges_with_a_hand_program(tmp_path):
    plain = made(Plain, form="runtime")
    merged = km.merge(plain, hand(tmp_path, plain.compute), l1=1 << 20, log=quiet)
    assert merged.form() == "program"
    assert merged.handlers == {"plain": 1, "one": 2}
    assert int(merged.handler(plain.handler("plain"))) == 1
    assert FakeRun([merged])._handler(Plain.plain) == (1, 0)


def test_the_refusals(tmp_path):
    plain = made(Plain)
    kernel_form = r"plain: a class in the kernel form merged with programs in the run"
    with pytest.raises(ProgramError, match=kernel_form):
        km.merge(plain, hand(tmp_path, plain.compute), l1=1 << 20, log=quiet)
    with pytest.raises(ProgramError, match=kernel_form):
        km.merge(plain, made(Other, form="runtime"), l1=1 << 20, log=quiet)
    sets = made(Gate, handlers={"gate": km.ANY}, form="runtime")
    with pytest.raises(ProgramError, match=r"gate: made with sets of its handlers'"):
        km.merge(sets, hand(tmp_path, sets.compute), l1=1 << 20, log=quiet)
    why = r"Both: form= at the make of a class merging Plain, Other: it is built as"
    with pytest.raises(ProgramError, match=why):
        made(Both, form="runtime")
    with pytest.raises(ProgramError, match=r"Both: form= and beside= at the make"):
        made(Both, form="runtime", beside=plain)
    merged = km.merge(made(Plain), made(Other), l1=1 << 20, log=quiet)
    why = r'a merge of classes made with form="runtime" is in that form'
    with pytest.raises(ProgramError, match=why):
        programs.plan([merged, made(Other, form="runtime")], 120)


def test_the_runtime_s_kernels_get_the_added_handlers_path():
    opted = made(Gate, handlers={"gate": km.ANY}, form="runtime")
    beside = made(Gate, handlers={"gate": km.ANY}, form="runtime", beside=opted)
    assert session_kernels.added_path([opted, beside]) == [("KM_ADDED_HANDLERS", "1")]
    assert session_kernels.added_path([opted, made(Other, form="runtime")]) == []
    assert session_kernels.added_path([]) == []
