"""A class added to a live run (run.addon(program)): the refusals by name, the switch
area's size (grown into the L1 the load leaves, or buffers=), a class named ahead
(Program.ahead(program)) sizing the area and taking the next field (no card)."""

import dataclasses
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from class_handler_constants_fixture import Pages, grid, quiet

import komorebi as km
from komorebi import formats
from komorebi.builder.flags import BuildError
from komorebi.device import native
from komorebi.device.tensor import Tensor
from komorebi.errors import ProgramError
from komorebi.islands import run as island_run
from komorebi.program import overlay, thread
from komorebi.program import several as programs
from komorebi.program.addon import code as added
from komorebi.program.addon import program as added_program
from komorebi.session import kernels as session_kernels
from komorebi.session.run import Run

pytestmark = pytest.mark.tree  # needs a tt-metal tree: the tree-tests lane


# The open device the tables are for: named, so no test asks the runtime (its cluster
# would open the cards).
BLACKHOLE = SimpleNamespace(arch=lambda: "blackhole", buffer_indices=lambda: 64)


def native_module():
    """The native module, or the test skipped where no tt-metal tree is named."""
    try:
        return native.load_module()
    except BuildError as error:
        pytest.skip(str(error))


@km.tile_function
def doubled(a):
    return a + a


class Twice(km.Program):
    """The loaded class: each core's tile of a, doubled, into out."""

    @km.handler
    def twice(self, a: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(self.rank, 1), write=True, block=1)
        for x in km.Stream(a.pages(self.rank, 1), block=1):
            results.push(doubled(x))


class Project(km.Program):
    """The added class: a row of tiles of x times w, summed in an fp32 field."""

    sums: km.L1[km.fp32, 1, 4, 32, 32]

    @km.tile_function
    def project(x, w):
        return x @ w

    @km.handler
    def rows(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        r = self.rank
        for xb, wb in zip(
            km.Stream(x.tiles[r : r + 1, 0:4], block=4),
            km.Stream(w.tiles[0:4, 0:4], block=16),
        ):
            self.sums += self.project(xb, wb)
        y.tiles[r : r + 1, 0:4].write(self.sums)


class Low(Twice, compute={"math_fidelity": "LoFi"}):
    """The loaded class at another fidelity."""


def low(**given):
    return Low(l1=1 << 20, log=quiet, a=grid(2, 1), out=grid(2, 1), **given)


def project(**given):
    return Project(l1=1 << 20, log=quiet, x=grid(2, 4), w=grid(4, 4), y=grid(2, 4),
                   **given)  # fmt: skip


def twice(**given):
    return Twice(l1=1 << 20, log=quiet, a=grid(2, 1), out=grid(2, 1), **given)


def backend(**given):
    cbs = [(29, "uint32", 64, 1), (30, "uint32", 64, 1)]
    values = {
        "switch_start": 0x20000,
        "switch_shift": 0x4000,
        "switch_base": 0x1F000,
        "switch_reserve": 0x40000,
        "slot_index": 28,
        "trace_cap": 0,
        "island": None,
        "_runtime_cbs": lambda: cbs,
    }
    return SimpleNamespace(**(values | given))


def run_of(loaded, switch=True, **given):
    fields = {
        "programs": [loaded],
        "_switch": {"records": {0: 0x1000}} if switch else None,
        "island_ranks": 120,
        "cores": 120,
        "_workers": 120,
        "_holds": [np.ones(120, bool)],
        "backend": backend(),
    }
    return SimpleNamespace(**(fields | given))


def test_a_class_beside_the_added_one_takes_it_and_the_refusals_name_each_cause():
    other = project(form="runtime")
    loaded = twice(form="runtime", beside=other)
    assert added_program.refused(run_of(loaded), other) == ""
    cases = [
        (run_of(loaded), project(), r"made in the runtime's form"),
        (run_of(loaded, switch=False), other, r"not loaded as a dict"),
        (
            run_of(low(form="runtime")),
            other,
            r"compute option math_fidelity is HiFi4 and the loaded kernels. LoFi",
        ),
        (run_of(loaded, island_ranks=118), other, r"an island run"),
        (run_of(loaded, backend=backend(trace_cap=8)), other, r"a traced run"),
        (run_of(loaded, backend=backend(slot_index=0)), other, r"buffer indices \[0\]"),
        (run_of(loaded, programs=[loaded, other]), other, r"of this run already"),
    ]
    for run, program, why in cases:
        assert re.search(why, added_program.refused(run, program))
    with pytest.raises(added.AddError, match=r"not loaded as a dict"):
        added_program.add(run_of(loaded, switch=False), other)


def test_a_program_that_runs_chains_is_not_added_to_a_switching_run():
    other = project(form="runtime")
    loaded = twice(form="runtime", beside=other)
    other.buffers.append(km.Buffer(programs.chain_buffer(), "uint32", 2816, 1))
    why = r"buffer 28 \(the chain's words, kChainBuffer\).*does not zero"
    assert re.search(why, added_program.refused(run_of(loaded), other))


def test_per_core_args_and_word_fields_go_in_a_row_a_core_and_dynamic_is_refused():
    other = project(form="runtime")
    loaded = twice(form="runtime", beside=other)
    other.args = lambda core: ([core, 7], [core + 100], [2 * core, 3, 4])
    assert "named with Program.ahead" in added_program.refused(run_of(loaded), other)
    built = run_of(loaded, backend=backend(switch_args=True))
    assert added_program.refused(built, other) == ""
    rows, row_bytes = added_program.args_defines(other, 3)
    assert rows[2] == [2, 7, 102, 4, 3, 4] and row_bytes == programs.CB_ALIGN
    c = overlay.constants()
    at = c["kOverlayArea"] + c["kOverlayRecord"] + 4 * c["kRecordArgsL1"]
    assert other.defines["KM_ARGS_L1"] == str(at)
    assert (other.defines["KM_ARGS_WRITER"], other.defines["KM_ARGS_COMPUTE"]) == (
        "2",
        "3",
    )
    other.args = None
    assert added_program.args_defines(other, 3) == ([], 0)
    other.word_fields = {"counts": 4}  # its tensor's address: compute's word 0
    other.args = lambda core: ([], [], [0x1F000])
    assert added_program.refused(built, other) == ""
    assert added_program.args_defines(other, 2) == ([[0x1F000]] * 2, programs.CB_ALIGN)
    assert other.defines["KM_ARGS_COMPUTE"] == "0"
    other.args, other.word_fields = None, {}
    other.messages = SimpleNamespace(handlers={}, parts={})
    assert "a dynamic handler" in added_program.refused(run_of(loaded), other)


class Descriptor:
    def __init__(self):
        self.grown = []

    def grow_cb(self, index, extra):
        self.grown.append((index, extra))


def area(given, rows, free=900_000):
    made = backend(added_buffers=given, switch_pad=7, switch_reserve=0x3000)
    made.card = SimpleNamespace(l1=lambda: (0, free), log=quiet, cb_align=64)
    descriptor = Descriptor()
    added_program.grow_area(made, descriptor, rows, None)
    return made, descriptor


def test_the_switch_area_grows_into_what_the_load_leaves_or_to_buffers():
    rows = [("circular buffers", 0x3000), ("slots", 100_000)]
    made, descriptor = area(None, rows)
    left = 900_000 - 0x3000 - 100_000
    extra = (left - added_program.KEEP) // programs.CB_ALIGN * programs.CB_ALIGN
    assert descriptor.grown == [(7, extra)]
    assert made.switch_reserve == 0x3000 + extra == rows[0][1]
    have = 0x1F000 + 0x3000 - 0x20000
    made, descriptor = area(have + 1000, [("circular buffers", 0x3000)])
    assert descriptor.grown == [(7, 1024)]
    assert area(0, [("circular buffers", 0x3000)])[1].grown == []
    with pytest.raises(ProgramError, match=r"buffers=10,000,000: .* would take"):
        area(10_000_000, [("circular buffers", 0x3000)])
    with pytest.raises(ValueError, match=r"buffers=-1"):
        added_program.buffers_given(-1)


def test_a_class_named_ahead_takes_the_next_field_and_sizes_the_area():
    other = project(form="runtime")
    loaded = twice(form="runtime", beside=other).ahead(other)
    with pytest.raises(ProgramError, match=r"a class ahead takes nothing more"):
        loaded.ahead(other, {"reader": "x.cpp"})
    placed = programs.plan({loaded: (0, 60)}, 120)
    assert added_program.ahead_programs(placed) == [other]
    assert other.program_field == 1
    made = backend(switch_shift=0)
    made.card = SimpleNamespace(hand_nocs=lambda program: None, cb_align=64)
    end = added_program.layout_of(made, other)[1]
    reserve = added_program.pad_reserve(made, placed, 0x20000)
    assert reserve >= end + programs.CB_ALIGN
    assert made.switch_pad not in {b.index for b in [*loaded.buffers, *other.buffers]}
    assert made.switch_loaded == 1 and made.switch_args is False
    made.card.module = native_module()
    made.card.device = BLACKHOLE
    (part,) = added_program.ahead_overlays(made, placed, 0, {})
    assert part.number == overlay.constants()["kOverlayService"] - 2
    assert part.recipes == {} and part.name == "project"
    assert added_program.ahead_overlays(made, placed, 1, {}) == []
    assert added_program.refused(run_of(loaded), other) == ""


@dataclasses.dataclass(frozen=True)
class Recipe:
    includes: tuple
    elf: Path = Path("kernel.elf")


def descriptors_text(tables, sizes):
    """A descriptors file as tt-metal's build writes one: each table on its line."""
    lines = []
    for name, values in [*tables.items(), *sizes.items()]:
        joined = ",".join(str(value) for value in values)
        lines.append(f"constexpr uint8_t {name}[64] = {{\n    {joined}\n}};")
    return "\n".join(lines) + "\n"


def test_the_added_class_s_parts_get_its_formats_in_a_copy_of_the_tables(tmp_path):
    other = project(form="runtime")
    loaded = twice(form="runtime")
    module = native_module()
    run = run_of(loaded)
    run.backend.card = SimpleNamespace(module=module, device=BLACKHOLE)
    run.backend.switch_pad, run.backend.switch_loaded = 27, 1
    names, modes = added_program._loaded_formats(run)
    fp32 = bool(loaded.compute.get("fp32_dest_acc_en"))
    listed = [modes.get(index, "Default") for index in range(64)]
    made = module.format_tables(names, fp32, listed, False, "blackhole")
    base = dict(zip(added_program.TABLES, made))
    sizes = {name: [1088] * 64 for name in added_program.SIZES}
    (tmp_path / "chlkc_descriptors.h").write_text(descriptors_text(base, sizes))
    (tmp_path / "trisc0").mkdir()
    recipe = Recipe(("-I..",), tmp_path / "trisc0" / "kernel.elf")
    made = added_program.own_recipes(run, other, {"unpack": recipe})["unpack"]
    folder = Path(made.includes[0][2:])
    assert made.includes[1:] == recipe.includes and folder != tmp_path
    text = (folder / "chlkc_descriptors.h").read_text()
    for buffer in other.buffers:
        size = added_program._table(text, "unpack_tile_size")[buffer.index]
        assert size == formats.tile_bytes(buffer.format)
        assert added_program._table(text, "unpack_src_format")[buffer.index] != 255
    fp32_field = [b.index for b in other.buffers if b.format == "float32"]
    assert fp32_field and all(
        added_program._table(text, "unpack_dst_format")[index] == 0  # Float32 in Dst
        for index in fp32_field
    )
    kept = [i for i in range(64) if i not in {b.index for b in other.buffers}]
    for name in added_program.TABLES:
        assert [added_program._table(text, name)[i] for i in kept] == [
            base[name][i] for i in kept
        ]
    wrong = dict(base, unpack_dst_format=[7] * 64)
    (tmp_path / "chlkc_descriptors.h").write_text(descriptors_text(wrong, sizes))
    with pytest.raises(added.AddError, match=r"unpack_dst_format .* is not the one"):
        added_program.own_recipes(run, other, {"unpack": recipe})


class Total(km.Program):
    """An added class of float32 buffers: a tile a core, all-reduced (its out at 16)."""

    @km.handler
    def total(self, x: km.Tensor, y: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for a in km.Stream(x.pages(self.rank, 1), block=1):
            total += a
        whole = km.all_reduce(total, local=True)
        y.pages(self.rank, 1).write(whole)


def float32_grid(rows, columns):
    return Tensor(Pages(4096, rows * columns), (rows * 32, columns * 32), "float32",
                  "tile")  # fmt: skip


def test_an_added_class_takes_its_own_formats_where_the_loaded_one_has_others(tmp_path):
    """The loaded kernels read no buffer by its format: the added class's float32
    buffers at the loaded class's bfloat16 indices get its own entries in its copy;
    the switch area's own index stays refused by name."""
    loaded = twice(form="runtime")
    other = Total(l1=1 << 20, log=quiet, form="runtime", x=float32_grid(1, 2),
                  y=float32_grid(1, 2))  # fmt: skip
    module = native_module()
    run = run_of(loaded)
    run.backend.card = SimpleNamespace(module=module, device=BLACKHOLE)
    run.backend.switch_pad, run.backend.switch_loaded = 27, 1
    names, modes = added_program._loaded_formats(run)
    shared = [b.index for b in other.buffers if names[b.index] not in ("", b.format)]
    assert 16 in shared and names[16] == "bfloat16"
    assert added_program.refused(run, other) == ""
    fp32 = bool(loaded.compute.get("fp32_dest_acc_en"))
    listed = [modes.get(index, "Default") for index in range(64)]
    base = module.format_tables(names, fp32, listed, False, "blackhole")
    base = dict(zip(added_program.TABLES, base))
    sizes = {name: [1088] * 64 for name in added_program.SIZES}
    (tmp_path / "chlkc_descriptors.h").write_text(descriptors_text(base, sizes))
    (tmp_path / "trisc0").mkdir()
    recipe = Recipe(("-I..",), tmp_path / "trisc0" / "kernel.elf")
    made = added_program.own_recipes(run, other, {"unpack": recipe})["unpack"]
    text = (Path(made.includes[0][2:]) / "chlkc_descriptors.h").read_text()
    mine, own = list(names), other.compute.get("unpack_to_dest_mode", {})
    for buffer in other.buffers:
        mine[buffer.index] = buffer.format
        listed[buffer.index] = own.get(buffer.index, "Default")
    wanted = module.format_tables(mine, fp32, listed, False, "blackhole")
    for name, values in zip(added_program.TABLES, wanted):
        assert [added_program._table(text, name)[i] for i in shared] == [
            values[i] for i in shared
        ]
    sizes = added_program._table(text, "unpack_tile_size")
    assert all(sizes[i] == 4096 for i in shared)
    run.backend.switch_pad = 8  # an index the class takes, the area's own
    with pytest.raises(added.AddError, match=r"buffer 8 is float32 .* switch area's"):
        added_program.own_recipes(run, other, {"unpack": recipe})


def test_a_mover_part_takes_the_tables_of_tt_metal_s_data_movement_build(tmp_path):
    """tt-metal builds the reader's and writer's tables with its default options (no
    fp32 Dst): a load with an fp32 Dst checks them so, and compute's with its own."""
    loaded = project(form="runtime")
    other = twice(form="runtime", beside=loaded)
    module = native_module()
    run = run_of(loaded)
    run.backend.card = SimpleNamespace(module=module, device=BLACKHOLE)
    run.backend.switch_pad, run.backend.switch_loaded = 27, 1
    names, _ = added_program._loaded_formats(run)
    assert loaded.compute.get("fp32_dest_acc_en")
    plain = module.format_tables(names, False, ["Default"] * 64, False, "blackhole")
    plain = dict(zip(added_program.TABLES, plain))
    sizes = {name: [1088] * 64 for name in added_program.SIZES}
    (tmp_path / "chlkc_descriptors.h").write_text(descriptors_text(plain, sizes))
    (tmp_path / "ncrisc").mkdir()
    recipe = Recipe(("-I..",), tmp_path / "ncrisc" / "kernel.elf")
    made = added_program.own_recipes(run, other, {"writer": recipe})["writer"]
    assert made.includes[1:] == recipe.includes
    with pytest.raises(added.AddError, match=r"unpack_dst_format .* is not the one"):
        added_program.own_recipes(run, other, {"unpack": recipe})


def test_the_load_reserves_a_record_row_for_every_field_an_added_class_may_take():
    other = project(form="runtime")
    placed = programs.plan({twice(form="runtime", beside=other): (0, 60)}, 120)
    service = overlay.constants()["kOverlayService"]
    assert added_program.spare_fields(placed) == range(1, service - 1)
    switch = {"records": {0: 0x100}, "spare": {1: (0x140, 1), 2: (0x180, 2)}}
    switch["refused"] = None
    assert thread.load_records(switch) == (0x100, 0x40, 1)
    switch["records"][1] = switch["spare"].pop(1)[0]  # the add's row
    assert thread.load_records(switch) == (0x100, 0x40, 2)


def test_an_added_field_past_the_reserved_rows_is_refused():
    other = project(form="runtime")
    run = run_of(twice(form="runtime", beside=other))
    run._switch["spare"], run._check_alive = {}, lambda: None
    with pytest.raises(added.AddError, match=r"field 1 is past the record rows"):
        added_program.add(run, other)


class Port:
    def __init__(self):
        self.writes = []

    def work_write(self, cluster, offset, data):
        self.writes.append((cluster, offset, bytes(data)))


def test_an_add_raises_the_count_a_live_thread_s_kmx_load_reads():
    ring = thread.RingPlacement(1, 2, 3, 4, 8, 16)
    port = Port()
    live = SimpleNamespace(
        records=(0x100, 0x40, 1), ring=ring, buffer=0, on_card=True, running=True,
        port=port, thread=SimpleNamespace(cluster=1),
    )  # fmt: skip
    switch = {
        "records": {0: 0x100, 1: 0x140},
        "spare": {2: (0x180, 2)},
        "refused": None,
    }
    thread.ThreadRun.write_records(live, switch)
    ((cluster, offset, line),) = port.writes
    assert (cluster, offset) == (1, thread.layout()["KMX_LIM_RING"])
    assert line == thread.ring_line(ring, 0, (0x100, 0x40, 2))
    assert np.frombuffer(line[-12:], np.uint32).tolist() == [0x100, 0x40, 2]


def test_a_command_of_an_added_class_to_cores_that_did_not_load_it_is_refused():
    """The host refuses it (its holdings row starts empty); the dispatcher refuses it
    from any source by its per-core table (reason 13, KMD_REJECT_PROGRAM), so no CPU of
    a core that does not hold the class takes it."""
    other = project(form="runtime")
    loaded = twice(form="runtime", beside=other)
    run = run_of(loaded, check_programs=True, _nop=0)
    run._holds = [np.ones(120, bool), np.zeros(120, bool)]
    run.programs = [loaded, other]
    run._part = lambda first, count: (first, first + count)
    with pytest.raises(ValueError, match=r"group \(60, 60\) does not hold project"):
        Run._check_holds(run, 1, 1, 60, 60)
    run._holds[1][60:] = True  # its load on cores 60 to 119
    Run._check_holds(run, 1, 1, 60, 60)


class Overlaid(km.Program):
    """An added class with two overlay handlers of its own."""

    @km.handler(overlay=True)
    def first(self, a: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(self.rank, 1), write=True, block=1)
        for x in km.Stream(a.pages(self.rank, 1), block=1):
            results.push(doubled(x))

    @km.handler(overlay=True)
    def second(self, a: km.Tensor, out: km.Tensor):  # the same body, another overlay
        results = km.Stream(out.pages(self.rank, 1), write=True, block=1)
        for x in km.Stream(a.pages(self.rank, 1), block=1):
            results.push(doubled(x))


def test_an_added_class_s_own_overlays_take_free_numbers_below_every_field():
    other = Overlaid(l1=1 << 20, log=quiet, a=grid(2, 1), out=grid(2, 1),
                     form="runtime")  # fmt: skip
    loaded = twice(form="runtime", beside=other).ahead(other)
    assert "need the entry command's code" in added_program.refused(
        run_of(loaded), other
    )
    ready = run_of(loaded, backend=backend(added_entry=True))
    assert added_program.refused(ready, other) == ""
    placed = programs.plan({loaded: (0, 60)}, 120)
    assert added_program.ahead_own_kernels(placed) == {"reader", "writer", "compute"}
    assert added_program.ahead_parts(placed) == 15  # its part and two own, 5 parts each
    assert (
        added_program.ahead_parts(programs.plan({twice(form="runtime"): (0, 60)}, 120))
        == 0
    )
    assert added_program.ahead_programs(placed) == [other]
    assert (other.program_field, other.overlay_shift) == (1, 0)
    assert [each.number for each in added_program.own_overlays(other)] == [1, 2]
    service = overlay.constants()["kOverlayService"]
    assert added_program.spare_fields(placed) == range(1, service - 1 - 2)
    made = backend(added_own={1, 2})
    made.room = SimpleNamespace(groups=[SimpleNamespace(numbers={1, 2, 62})], added={})
    run = run_of(loaded, backend=made)
    assert added_program._own_shift(run, other, 2) == 2  # numbers 3 and 4
    made.added_own = set(range(1, service - 3))
    with pytest.raises(added.AddError, match=r"no overlay numbers left"):
        added_program._own_shift(run, other, 2)


def test_an_island_run_takes_an_added_class_through_its_island_run_alone():
    loaded, other = twice(form="runtime"), twice(form="runtime")
    island = run_of(loaded, island_ranks=118)
    why = added_program.refused(island, other)
    assert "one island's run of an island run" in why
    assert added_program.refused(island, other, island=True) == ""
    split = run_of(loaded, backend=backend(split=("graph",)))
    assert "a card split into islands" in added_program.refused(split, other)
    behind = run_of(loaded, island_ranks=118)
    behind.programs = [loaded, twice(form="runtime")]
    with pytest.raises(added.AddError, match=r"holds? \[1, 2\] programs: an added "):
        added_program.add_islands([island, behind], other, {"KM_TENSOR_TABLE": 1})
    assert "KM_TENSOR_TABLE" not in other.defines  # the island's defines off again


def test_an_island_run_s_added_class_takes_tensors_made_after_the_load(monkeypatch):
    made = SimpleNamespace(placement="split")  # made while the run lives
    program = twice(form="runtime")
    program.make_given = {"tensors": {"a": made}}
    handed = []
    session = SimpleNamespace(
        _hand_rectangles=handed.append, _island_defines=lambda: {"KM_TENSOR_TABLE": 1}
    )
    run = SimpleNamespace(session=session, runs=("island 0", "island 1"))
    monkeypatch.setattr(added_program, "add_islands", lambda runs, p, defines: 3)
    assert island_run.IslandRun.addon(run, program) == 3
    assert handed == [program]
    run._addon_code = lambda name, parts, **options: (
        "code",
        name,
    )  # test_island_addon_code
    assert island_run.IslandRun.addon(run, "parts") == ("code", "parts")
    run.session = None
    with pytest.raises(added.AddError, match=r"a card split into islands"):
        island_run.IslandRun.addon(run, program)


def test_an_add_whose_parts_do_not_fit_the_room_is_refused_before_any_write():
    room = added.Room(SimpleNamespace(page_bytes=100), 2, 5, [], set(), True, None)
    other = twice(form="runtime")
    added_program._check_room(room, other, [{"reader": b"a" * 150}, {"x": b"b" * 50}])
    with pytest.raises(added.AddError, match=r"400 B of the room .*4 rows of 100 B"):
        added_program._check_room(room, other, [{"r": b"a" * 250, "w": b"c" * 10}])


def test_a_loaded_program_with_own_overlays_gets_two_slots_in_their_kernels():
    own = Overlaid(l1=1 << 20, log=quiet, a=grid(2, 1), out=grid(2, 1),
                   form="runtime")  # fmt: skip
    plain = twice(form="runtime")
    built = [SimpleNamespace(slots={"reader": (1, 4160), "writer": (1, 4032)},
                             service=8640)]  # fmt: skip
    declared = SimpleNamespace(count=None)
    alone = programs.plan({own: (0, 120)}, 120)
    made = session_kernels._common_slots(alone, declared, built, 4)
    assert made.slots == {"reader": (2, 4160), "writer": (2, 4032)}
    plain_alone = programs.plan({plain: (0, 120)}, 120)
    made = session_kernels._common_slots(plain_alone, declared, built, 4)
    assert made.slots == {"reader": (1, 4160), "writer": (1, 4032)}  # pays nothing
    with pytest.raises(ProgramError, match=r"slots\(count=1\) for the reader"):
        session_kernels._common_slots(alone, SimpleNamespace(count=1), built, 4)
    lowered = SimpleNamespace(slots={"reader": (1, 4160), "writer": (2, 4032)})
    with pytest.raises(ProgramError, match=r"reader slots lowered to 1"):
        session_kernels._own_slots_kept(alone, lowered)
    session_kernels._own_slots_kept(
        plain_alone, lowered
    )  # no own overlay: one is enough


def switching_pair(first_service, second_service):
    first, second = twice(form="runtime"), project(form="runtime")
    first.slots(count=1, service=first_service)
    second.slots(count=1, service=second_service)
    return programs.plan({first: (0, 60), second: (60, 60)}, 120)


def test_a_switching_run_puts_the_service_overlay_in_the_reader_slots_when_asked():
    made = overlay.Built({}, {"reader": (1, 56320), "writer": (1, 30848)}, 58816,
                         58816)  # fmt: skip
    totals = {}
    for service in (None, "shared"):
        placed = switching_pair(service, service)
        assert placed.switch
        declared = session_kernels.switch_slots(placed)
        common = session_kernels._common_slots(placed, declared, [made], 4)
        common = session_kernels.fit_slots(common, declared, 1 << 20, [], quiet)
        checked = session_kernels.switch_group_slots(made, declared, True)
        rows = session_kernels.slot_rows(common.slots, common.service, common.why)
        total = sum(size for _, size in rows)
        group = session_kernels.slot_rows(checked.slots, checked.service)
        assert sum(size for _, size in group) == total  # the run's slots, counted
        totals[service] = (common.service, common.slots, total)
    assert totals[None] == (58816, {"reader": (1, 56320), "writer": (1, 30848)},
                            145_984)  # fmt: skip
    assert totals["shared"] == (0, {"reader": (1, 58816), "writer": (1, 30848)},
                                89_664)  # fmt: skip
    plain = session_kernels.switch_group_slots(made, overlay.Slots(), True)
    assert plain is made  # the option not given: the group's check as before
    alone = session_kernels.switch_group_slots(made, overlay.Slots(service="shared"),
                                               False)  # fmt: skip
    assert alone is made  # one program: fit_slots places it (overlay.choose)


def test_programs_of_a_switching_run_that_ask_two_service_places_are_refused():
    with pytest.raises(ProgramError, match=r"share one set of slots"):
        session_kernels.switch_slots(switching_pair("shared", None))
    with pytest.raises(ProgramError, match=r"slots\(service='own'\) on"):
        session_kernels.switch_slots(switching_pair("shared", "own"))
    one = programs.plan([twice(form="runtime")], 120)
    assert session_kernels.switch_slots(one) is one.programs[0].overlay_slots


def test_a_class_made_after_the_load_takes_the_shift_of_its_kind_named_ahead():
    named = twice(form="runtime")
    named.program_field, named.added_own_shift = 1, 4
    loaded = SimpleNamespace(added_ahead_programs=[named])
    runs = [SimpleNamespace(programs=[loaded])]
    later = twice(form="runtime")  # its tensors made while the run lives
    assert added_program._ahead_twin_shift(runs, later, 1) == 4
    assert added_program._ahead_twin_shift(runs, later, 2) is None  # another field
    assert added_program._ahead_twin_shift(runs, named, 1) is None  # itself
    named.added_live = True  # added already: its numbers are taken
    assert added_program._ahead_twin_shift(runs, later, 1) is None


def test_an_added_part_over_its_slot_says_how_to_keep_more(monkeypatch):
    def too_big(*args):
        raise overlay.OverlayError(
            "overlay columns, math part: 2148 bytes (code, data and .bss) do not fit "
            "its slot of 2112 bytes"
        )

    monkeypatch.setattr(overlay, "build_part_cached", too_big)
    each = SimpleNamespace(slots={"math": (1, 2112)}, recipes={"math": None})
    made = SimpleNamespace(name="columns")
    with pytest.raises(
        added.AddError, match=r"name the class ahead, or give the loaded"
    ):
        added._part(None, each, made, "math", "math")
