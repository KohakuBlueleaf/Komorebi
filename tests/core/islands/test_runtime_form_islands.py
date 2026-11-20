"""A load in the runtime's form (form="runtime", a dict of two, a km_program) on two
linked Wormhole islands (no card): each island takes its part of the plan; the landing
reads the plan's programs and is sized from its buffers' L1; an island's build refused
for its writer names the plan's programs."""

from types import SimpleNamespace

import pytest

import komorebi as km
from komorebi.device import mock
from komorebi.device.tensor import Tensor
from komorebi.errors import KomorebiError, ProgramError
from komorebi.islands.parts.link_plan import LinkPlanMixin
from komorebi.islands.parts.load import LoadMixin
from komorebi.program import several as programs
from komorebi.program.checks import far_collectives, page_calls

RANKS = 4  # an island's
WORMHOLE = {"KM_TENSOR_TABLE": 1, "KM_PAGE_LINKS": 1, "KM_LINK_NO_FAR_DRAM": 1}
WRITER_OVER = (
    "build failed: TT_THROW @ tt_elffile.cpp:405: tt::exception\ninfo:\n"
    "/cache/kernels/writer/13895347754378637234//ncrisc/ncrisc.elf: segment[0] "
    "[0xffc00000,+0x4758) overflows region:0 limit of 0x4000 bytes, reduce the "
    "size of code"
)
SUMS = (
    "void km_program(km::Reader& reader, const km_command_t& command) {\n"
    "    km::all_reduce(reader, command, 0, 0, 1, 0, km::Sum{}, 0, 0);\n"
    "}\n"
)
EMPTY = "void km_program(km::{kind}& kernel, const km_command_t& command) {{}}\n"


class Pages:
    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tiles(count):
    """A float32 tensor of `count` tiles in a row."""
    return Tensor(Pages(4096, count), (32, count * 32), "float32", "tile")


class Maximum(km.Program):
    part: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, source: km.Tensor, target: km.Tensor):
        acc = km.tile.zeros(km.fp32)
        for tile in km.Stream(source.pages(self.rank, 1)):
            acc += tile
        self.part.write(acc)
        km.all_reduce(self.part, op="max")


class Copy(km.Program):
    @km.handler
    def copy(self, source: km.Tensor, target: km.Tensor):
        km.Stream(source.pages(0, 1)).drain()


def runtime(cls):
    given = {"source": tiles(64), "target": tiles(64), "l1": 1 << 20}
    return cls(**given, log=lambda line: None, form="runtime")


def written(folder, name, pages, reader=None):
    """A program in the runtime's form from its sources (km_program each), with one
    input buffer of `pages` pages of 2048 B and one output buffer of 2."""
    folder.mkdir(parents=True)
    for kernel, kind in (("reader", "Reader"), ("writer", "Writer")):
        text = reader if kernel == "reader" and reader else EMPTY.format(kind=kind)
        (folder / f"{kernel}.cpp").write_text(text)
    (folder / "compute.cpp").write_text(EMPTY.format(kind="Compute"))
    buffers = [
        km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=pages),
        km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
    ]
    return km.Program(name, kernels=folder, buffers=buffers, handlers={name: 1})


class Islands(LoadMixin, LinkPlanMixin):
    """Two islands of Wormhole chips, one link between them: the islands' load as it
    is, the cards stood in (each island's session records what it is given; the link
    plan's part that reads the programs, the collectives' landing, is run)."""

    arch = "wormhole_b0"
    ASKS_BY = "core"

    def __init__(self, free=1 << 20, refused=None, defines=WORMHOLE):
        self.given, self.landing, self.refused = [], "not asked", refused
        self.said = []
        self.log = self.said.append
        self.defines = defines
        self.run, self._memory, self.split, self.links = None, None, 1, 1
        self._regions, self._seconds = [], {}
        self.graph = SimpleNamespace(
            ranks=2 * RANKS,
            first=lambda island: island * RANKS,
            count=lambda island: RANKS,
            links=[SimpleNamespace(a=0, b=1, count=2)],
        )
        nocs = SimpleNamespace(grid=(RANKS, 1))
        self.sessions = tuple(
            SimpleNamespace(
                load=self.island_load,
                largest_page=2048,
                l1=lambda: (0, free),
                nocs=lambda: nocs,
                _last_total=0,
            )
            for _ in range(2)
        )

    def island_load(self, one, island=None, link=None, tensors=None, **options):
        held = one.programs if isinstance(one, programs.Plan) else [one]
        self.given.append((one, [dict(p.defines) for p in held]))
        if self.refused:
            raise self.refused
        return SimpleNamespace(backend=None)

    def _island_defines(self):
        return dict(self.defines)

    def _tables_for_load(self):
        return [0, 0]

    def _links(self, programs=None):
        self.landing = self._tree_landing_of(programs)
        return [None, None]


@pytest.fixture
def islands(monkeypatch):
    monkeypatch.setattr(mock, "active", lambda: True)  # a mock run for the launch
    return Islands


def groups(plan):
    return [(g.first, g.count, g.programs) for g in plan.groups]


def test_a_runtime_form_class_loads_on_wormhole_islands(islands):
    made, program = islands(), runtime(Maximum)
    assert program.form() == "program"
    made.load(program)
    plans = [one for one, _ in made.given]
    assert [type(one) for one in plans] == [programs.Plan] * 2
    assert [groups(one) for one in plans] == [[(0, RANKS, (0,))]] * 2
    assert all(seen[0]["KM_LINK_NO_FAR_DRAM"] == 1 for _, seen in made.given)
    # km.all_reduce in a class is the ports' tile reduce: across islands it writes
    # the far rank's L1 (km::write_l1), not a copy's DRAM, so it needs no landing
    (movement,) = page_calls.sources(plans[0].programs[0])
    assert "km::all_reduce_emit(" in movement.read_text()
    assert not far_collectives.calls(plans[0].programs)
    assert made.landing is None
    assert "KM_LINK_NO_FAR_DRAM" not in program.defines  # put back after the load


def test_a_dict_of_two_runtime_form_classes_loads_on_wormhole_islands(islands):
    made, logits, choice = islands(), runtime(Maximum), runtime(Copy)
    made.load({logits: (0, 2 * RANKS - 1), choice: (2 * RANKS - 1, 1)})
    first, second = (one for one, _ in made.given)
    assert first.switch and second.switch
    assert groups(first) == [(0, RANKS, (0,))]
    assert groups(second) == [(0, RANKS - 1, (0,)), (RANKS - 1, 1, (1,))]
    assert made.landing is None


def test_a_plan_whose_mover_calls_a_collective_gets_the_landing(islands, tmp_path):
    # the switch plan keeps its largest program's buffers (880,640 + 4,096 B) on
    # every core: 163,840 B of 1 MiB left, 8,192 B a slot (the largest alone)
    small = written(tmp_path / "sums", "sums", 18, reader=SUMS)
    large = written(tmp_path / "large", "large", 430)
    made = islands()
    made.load({small: (0, RANKS), large: (RANKS, RANKS)})
    assert made.landing == (2, 8192)


def test_an_island_s_writer_over_its_region_names_the_plan_s_programs(islands):
    made = islands(
        refused=ProgramError(WRITER_OVER),
        defines=WORMHOLE | {"KM_LINK_THIN_SECOND": 1},
    )
    logits, choice = runtime(Maximum), runtime(Copy)
    with pytest.raises(KomorebiError) as caught:
        made.load({logits: (0, 2 * RANKS - 1), choice: (2 * RANKS - 1, 1)})
    assert "maximum, copy: the writer is 18,264 B of NCRISC's 16,384 B" in str(
        caught.value
    )
