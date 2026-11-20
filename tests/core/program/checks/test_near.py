"""The near build (KM_ISLAND_NEAR): an island load whose writer is over the code memory
at the highest level is built again without the far paths when the host can keep it
near; a command is then refused at its submit when a tensor it is given is not whole on
its island, and a host collective at its call."""

from types import SimpleNamespace

import pytest

import komorebi as km
from komorebi.device import remembered
from komorebi.device.tensor import Tensor
from komorebi.errors import KomorebiError
from komorebi.islands import local as island_local
from komorebi.islands.run import IslandRun
from komorebi.program.checks import near
from komorebi.session import kernels as kernel_module
from komorebi.session.parts.load import LoadMixin

KERNEL = "void kernel_main() {}\n"
PUTS = """\
#pragma once
// km::put_pages(core, command, 1, 0, 64, 0, 1, run): a comment
inline bool send(const km::Writer& writer, const km_command_t& command) {
    return km::put_pages (writer, command, 1, 0, 64, 0, 1, 0);
}
"""
FAR = (24_056, 16_384)
SAID = (
    "without the far paths (with them the writer is 24,056 B of 16,384 B at the "
    "highest writer level)"
)


def program(folder, header="#pragma once\n", defines=None, tensors=None):
    """A class's program as the check sees it: its kernels, its defines, its make."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "reader.cpp").write_text('#include "puts.h"\n' + KERNEL)
    (folder / "puts.h").write_text(header)
    (folder / "writer.cpp").write_text(KERNEL)
    (folder / "compute.cpp").write_text(KERNEL)
    made = km.Program("gate", kernels=folder, defines=defines or {"KM_TENSOR_TABLE": 1})
    made.make_given = {"tensors": tensors or {}}
    return made


def tensor(copy, parts=()):
    return SimpleNamespace(placement=SimpleNamespace(copy=copy, parts=parts))


def test_a_class_of_copies_whose_movers_cross_nowhere_may_be_built_near(tmp_path):
    made = program(tmp_path / "a", tensors={"x": tensor(True)})
    assert near.applies([made]) and near.refusal([made]) is None


def test_each_reason_against_the_near_build_is_named(tmp_path):
    one = program(tmp_path / "b", defines={"KM_PAGE_LINKS": 1})
    assert not near.applies([one])
    assert near.refusal([one]) == "gate is not a program of several islands"
    by_hand = km.Program("sums", kernels=tmp_path / "b", defines={"KM_TENSOR_TABLE": 1})
    assert near.refusal([by_hand]) == (
        "sums's kernels are given by hand: the host cannot tell which of its command "
        "words are tensors"
    )
    split = program(tmp_path / "c", tensors={"w": tensor(False, ((0, 1), (1, 1)))})
    assert near.refusal([split]) == "gate's tensor w at its make is not a copy"
    chain = program(
        tmp_path / "d", defines={"KM_TENSOR_TABLE": 1, "KM_CHAIN_GATHER": 1}
    )
    assert near.refusal([chain]) == "gate's chain gathers (km.all_gather in a chain)"
    wired = program(tmp_path / "e1", defines={"KM_TENSOR_TABLE": 1, "KM_WIRE": 1})
    assert near.refusal([wired]) == "gate runs on a card split into islands"
    dynamic = program(tmp_path / "e2")
    dynamic.messages = object()
    assert near.refusal([dynamic]) == (
        "gate is dynamic: its link cores carry its messages"
    )
    threads = program(tmp_path / "e")
    threads.threads = {"t": object()}
    assert near.refusal([threads]) == (
        "gate's X280 threads send commands the host never checks"
    )


def test_a_mover_naming_a_far_call_is_refused_with_its_file_and_line(tmp_path):
    made = program(tmp_path / "f", header=PUTS)
    why = near.refusal([made])
    assert why.startswith("gate: ") and why.endswith("puts.h:4 calls km::put_pages")


def test_the_movers_of_a_near_build_take_its_define_and_lose_the_page_links(
    tmp_path,
):
    made = program(tmp_path / "g")
    given = [("KM_TENSOR_TABLE", "1"), ("KM_PAGE_LINKS", "1")]
    given += [("KM_LINK_NO_FAR_DRAM", "1"), ("KM_LINK_THIN_SECOND", "1")]
    made.near = FAR
    for kernel in ("reader", "writer"):
        built = kernel_module.kernel_build(made, kernel, given, "wormhole_b0")
        names = [name for name, _ in built["defines"]]
        assert near.DEFINE in names and "KM_TENSOR_TABLE" in names
        assert not set(near.DROPPED) & set(names)
    built = kernel_module.kernel_build(made, "compute", given, "wormhole_b0")
    assert (near.DEFINE, "1") not in built["defines"]
    assert ("KM_PAGE_LINKS", "1") in built["defines"]
    made.near = None
    built = kernel_module.kernel_build(made, "writer", given, "wormhole_b0")
    assert (near.DEFINE, "1") not in built["defines"]
    assert ("KM_PAGE_LINKS", "1") in built["defines"]


def test_the_far_builds_bytes_are_read_from_the_runtimes_refusal():
    error = RuntimeError(
        "/c/kernels/movement/123//ncrisc/ncrisc.elf: segment[0] [0x4c0,+0x5df8) "
        "overflows region:0 limit of 0x4000 bytes"
    )
    assert kernel_module.writer_over_bytes(error) == FAR
    assert near.line(*FAR) == SAID


def handler(*places):
    def call(self, *values):
        del self, values

    return SimpleNamespace(name="gate", function=call, places=lambda: list(places))


class Gate:
    name = "gate"
    near = FAR


def test_a_tensor_not_whole_on_the_island_that_runs_the_command_is_refused():
    method = handler("x", "w")
    copy, here = tensor(True), tensor(False, ((0, 2), (0, 0)))
    there = tensor(False, ((0, 0), (0, 2)))
    island_local.check_near(Gate(), method, (copy, here), 0)
    with pytest.raises(KomorebiError) as caught:
        island_local.check_near(Gate(), method, (copy, there), 0)
    assert str(caught.value) == (
        "Gate.gate: its tensor w is held by island 1, and the command runs on island "
        f"0, but the program was built {SAID}: such a load reads and writes the pages "
        "of the island that runs the command alone; give w as a copy, or as a tensor "
        "of the island whose ranks the command takes"
    )
    split = tensor(False, ((0, 1), (1, 1)))
    with pytest.raises(KomorebiError, match="its tensor x is split over islands"):
        island_local.check_near(Gate(), method, (split, copy), 1)
    bank = SimpleNamespace(island=1, bank=0, entry=lambda banks: b"")
    with pytest.raises(KomorebiError, match="its tensor w is held by island 1"):
        island_local.check_near(Gate(), method, (copy, bank), 0)
    island_local.check_near(Gate(), method, (copy, bank), 1)


def test_a_load_built_with_the_far_paths_is_not_looked_at():
    far = SimpleNamespace(name="gate", near=None)
    split = tensor(False, ((0, 1), (1, 1)))
    island_local.check_near(far, handler("x"), (split,), 0)
    island_local.check_near(Gate(), handler("x"), (split,), None)  # one card


def test_a_collective_of_the_host_is_refused_on_a_near_load():
    island_local.near_collective([SimpleNamespace(name="gate", near=None)], "x")
    with pytest.raises(KomorebiError) as caught:
        island_local.near_collective([Gate()], "run.all_reduce()")
    assert str(caught.value) == (
        f"run.all_reduce(): gate was built {SAID}, and a collective crosses islands"
    )


def test_commands_given_as_words_are_refused_on_a_near_load():
    # submit_many's rows and a handler named by number hold no tensor to check
    island_local.near_words([SimpleNamespace(name="gate", near=None)], "submit_many")
    with pytest.raises(KomorebiError) as caught:
        island_local.near_words([Gate()], "submit_many")
    assert str(caught.value) == (
        f"submit_many: gate was built {SAID}, and a command given as words cannot be "
        "checked against the island that runs it: submit each command as the "
        "class's handler with its tensors (run.submit(program.handler, group, "
        "args=(...)))"
    )
    islands = SimpleNamespace(
        runs=[SimpleNamespace(programs=[Gate()])], _check_half_added=lambda named: None
    )
    with pytest.raises(KomorebiError, match="^submit_many: gate was built"):
        IslandRun._submit_many(islands, 1, [0, 8], [8, 8], None, False)


def test_a_reduce_whose_group_crosses_an_islands_edge_is_refused_on_a_near_load():
    gate = Gate()
    gate.port_handlers = {"gate"}
    gate.reduce_groups = {"gate": (4, "line 9")}
    method = handler("x")
    copy = tensor(True)
    island_local.check_near(gate, method, (copy,), 0, (0, 8), (0, 8))
    island_local.check_near(gate, method, (copy,), 1, (0, 16), (8, 16))
    with pytest.raises(KomorebiError) as caught:
        island_local.check_near(gate, method, (copy,), 1, (6, 8), (8, 16))
    assert str(caught.value).startswith(
        "Gate.gate: a group of 4 ranks of the command (6, 8) holds ranks of island 1 "
        "(ranks 8 to 15) and of another, and the handler reduces, uses ports or "
        f"broadcasts, but the program was built {SAID}"
    )
    del gate.reduce_groups
    with pytest.raises(KomorebiError, match="Gate.gate: its group of the command"):
        island_local.check_near(gate, method, (copy,), 0, (0, 16), (0, 8))
    gate.port_handlers = set()
    island_local.check_near(gate, method, (copy,), 0, (0, 16), (0, 8))


OVER = (
    "/c/kernels/movement/123//ncrisc/ncrisc.elf: segment[0] [0x4c0,+0x5df8) "
    "overflows region:0 limit of 0x4000 bytes"
)


class Pages:
    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tiles(count):
    """A float32 tensor of `count` tiles in a row."""
    return Tensor(Pages(4096, count), (32, count * 32), "float32", "tile")


class Sums(km.Program):
    @km.handler
    def sums(self, s: km.Tensor, t: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            total += tile
        whole = km.all_reduce(total)
        t.pages(self.rank, 1).write(whole)


def test_a_class_s_reduce_across_an_islands_edge_is_refused_on_a_near_load():
    # its reducing handler stays a port handler after init, so a near load's submit
    # refuses its group across an island's edge by name, before its movers give up
    made = Sums(s=tiles(8), t=tiles(8), l1=1 << 20, log=lambda line: None)
    assert made.port_handlers == {"sums"}
    made.near = FAR
    copy = tensor(True)
    island_local.check_near(made, Sums.sums, (copy, copy), 0, (0, 4), (0, 4))
    with pytest.raises(KomorebiError) as caught:
        island_local.check_near(made, Sums.sums, (copy, copy), 1, (0, 8), (4, 8))
    assert str(caught.value).startswith(
        "Sums.sums: its group of the command (0, 8) holds ranks of island 1 (ranks 4 "
        "to 7) and of another, and the handler reduces, uses ports or broadcasts, but "
        f"the program was built {SAID}"
    )
    assert str(caught.value).endswith(
        "submit groups that lie on one island, or make the class with "
        'form="runtime" (its handlers\' code in parts in L1, its movers built with '
        "the page links)"
    )


@pytest.fixture(autouse=True)
def no_elf_refused_yet(monkeypatch):
    """Each test's process memory of the ELFs the runtime refused, empty."""
    monkeypatch.setattr(kernel_module, "_elf_over", {})


class Levels:
    """A load's _build_levels over a build that fits from `fits` on ((near, level)),
    each (level, near) it was built at in `walked`."""

    def __init__(self, arch, fits=None):
        self.arch, self.fits, self.walked, self.logged = arch, fits, [], []
        self._dispatcher = None

    def log(self, line):
        self.logged.append(line)

    def describe(self, made):
        def build():
            self.walked.append((made.writer_level, made.near))
            if self.fits is not None and (bool(made.near), made.writer_level) >= (
                self.fits
            ):
                return [], None
            raise RuntimeError(OVER)

        return SimpleNamespace(build=build)

    def release_l1(self):
        """The attempt's L1 tensors given back: said in `walked`."""
        self.walked.append("released")


def builds(levels):
    """The (level, near) of each build of `levels`, its releases left out."""
    return [each for each in levels.walked if each != "released"]


def build_levels(levels, made, monkeypatch, kept=None):
    """_build_levels of `made`, its kept level and near build in `kept` (a dict, also
    what the walk keeps)."""
    kept = {} if kept is None else kept
    monkeypatch.setattr(
        remembered, "writer_level", lambda program, arch: kept.get("level", 0)
    )
    monkeypatch.setattr(
        remembered,
        "keep_writer_level",
        lambda program, arch, level: kept.__setitem__("level", level),
    )
    monkeypatch.setattr(
        remembered, "writer_near", lambda program, arch: kept.get("near")
    )
    monkeypatch.setattr(
        remembered,
        "keep_writer_near",
        lambda program, arch, far: kept.__setitem__("near", far),
    )
    backend = SimpleNamespace(describe=levels.describe, release_l1=levels.release_l1)
    return LoadMixin._build_levels(levels, backend, made, None, (made,))


def test_blackhole_walks_level_0_alone_and_never_the_near_build(tmp_path, monkeypatch):
    made = program(tmp_path / "h", tensors={"x": tensor(True)})
    levels = Levels("blackhole")
    with pytest.raises(km.ProgramError, match="build failed"):
        build_levels(levels, made, monkeypatch)
    assert builds(levels) == [(0, None)]


def test_wormhole_walks_its_levels_then_the_near_build_from_level_0(
    tmp_path, monkeypatch
):
    made = program(tmp_path / "i", tensors={"x": tensor(True)})
    levels = Levels("wormhole_b0", fits=(True, 2))
    build_levels(levels, made, monkeypatch)
    assert builds(levels) == [
        *((k, None) for k in range(5)),
        (0, FAR),
        (1, FAR),
        (2, FAR),
    ]
    assert made.near == FAR and made.writer_level == 2
    assert f"built again {SAID}, from writer level 0" in levels.logged[-3]


def test_a_wormhole_load_that_cannot_be_near_says_why_with_its_size(
    tmp_path, monkeypatch
):
    made = program(tmp_path / "j", header=PUTS)
    levels = Levels("wormhole_b0")
    with pytest.raises(km.ProgramError) as caught:
        build_levels(levels, made, monkeypatch)
    text = str(caught.value)
    assert text.startswith(
        "gate: the writer (movement) is 24,056 B of NCRISC's 16,384 B code region at "
        "writer level 4, the highest: 7,672 B over; it is not built without the far "
        "paths: gate: "
    )
    assert text.endswith("puts.h:4 calls km::put_pages")
    assert builds(levels) == [(k, None) for k in range(5)]


def test_a_near_build_over_at_its_highest_level_is_refused_with_its_size(
    tmp_path, monkeypatch
):
    made = program(tmp_path / "k", tensors={"x": tensor(True)})
    levels = Levels("wormhole_b0")
    with pytest.raises(km.ProgramError, match="at writer level 4, the highest"):
        build_levels(levels, made, monkeypatch)
    assert builds(levels) == [
        *((k, None) for k in range(5)),
        *((k, FAR) for k in range(5)),
    ]


def test_the_next_load_of_a_near_build_builds_near_at_first(tmp_path, monkeypatch):
    # the next island's load: no far build the process saw fail is asked for again
    made = program(tmp_path / "l", tensors={"x": tensor(True)})
    kept = {}
    build_levels(Levels("wormhole_b0", fits=(True, 1)), made, monkeypatch, kept)
    assert kept == {"level": 1, "near": FAR}
    again = Levels("wormhole_b0", fits=(True, 1))
    build_levels(again, made, monkeypatch, kept)
    assert builds(again) == [(1, FAR)]


def test_a_kept_near_build_no_longer_near_is_refused_before_any_build(
    tmp_path, monkeypatch
):
    made = program(tmp_path / "m", header=PUTS)
    levels = Levels("wormhole_b0")
    kept = {"level": 0, "near": FAR}
    with pytest.raises(km.ProgramError) as caught:
        build_levels(levels, made, monkeypatch, kept)
    assert "(its last build's); it is not built without the far paths" in str(
        caught.value
    )
    assert builds(levels) == []


def test_a_class_refused_then_loaded_again_is_refused_at_once_by_name(
    tmp_path, monkeypatch
):
    # the runtime's ELF cache would wait on its refused ELF without end: never asked
    made = program(tmp_path / "n", header=PUTS)
    first = Levels("wormhole_b0")
    with pytest.raises(km.ProgramError, match="at writer level 4, the highest"):
        build_levels(first, made, monkeypatch)
    assert builds(first) == [(k, None) for k in range(5)]
    again = Levels("wormhole_b0")
    with pytest.raises(km.ProgramError) as caught:
        build_levels(again, made, monkeypatch)
    assert builds(again) == []
    text = str(caught.value)
    assert text.startswith("already refused in this process: gate: the writer")
    assert "at writer level 4, the highest" in text
    assert "already refused in this process: /c/kernels/movement/123//ncrisc" in str(
        caught.value.__cause__
    )


def test_a_blackhole_build_refused_then_asked_again_is_refused_by_name(
    tmp_path, monkeypatch
):
    made = program(tmp_path / "o", tensors={"x": tensor(True)})
    with pytest.raises(km.ProgramError, match="build failed"):
        build_levels(Levels("blackhole"), made, monkeypatch)
    again = Levels("blackhole")
    with pytest.raises(km.ProgramError, match="already refused in this process"):
        build_levels(again, made, monkeypatch)
    assert builds(again) == []


def test_each_attempt_over_gives_its_l1_back_before_the_next_describe(
    tmp_path, monkeypatch
):
    # else its control blocks alternate between two addresses, and two islands that
    # walked a different number of levels place them apart (_check_links refuses)
    made = program(tmp_path / "p", tensors={"x": tensor(True)})
    levels = Levels("wormhole_b0", fits=(True, 1))
    build_levels(levels, made, monkeypatch)
    over = [(k, None) for k in range(5)] + [(0, FAR)]
    assert levels.walked == [
        *(step for each in over for step in (each, "released")),
        (1, FAR),
    ]
