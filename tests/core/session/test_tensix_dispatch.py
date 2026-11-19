"""The Tensix dispatcher's host side without a card: its port's interface, the cores it
leaves the program, its rectangles and start words, and what it refuses by name."""

from types import SimpleNamespace

import pytest
from fake_port import FakeBackend, FakePort

import komorebi as km
from komorebi import _env, abi, layout
from komorebi.device import chips, mock
from komorebi.device.events import Event
from komorebi.dsl.program import ranks_rectangle, session_nocs
from komorebi.errors import NotServed
from komorebi.islands import link_area
from komorebi.islands import local as island_local
from komorebi.islands.parts import load as island_load
from komorebi.islands.parts.mesh import cards_graph
from komorebi.program import thread as thread_module
from komorebi.session import dispatch, kernels
from komorebi.session.backend import backend as card_backend
from komorebi.session.dispatch import tensix
from komorebi.session.run import Run

BLACKHOLE = (12, 10)  # a P150A's worker grid
WORMHOLE = (8, 10)
HEADERS = abi.headers()


def fake_nocs(grid, noc0_x, whole=False):
    """layout.Nocs of `grid`: column x at NoC0 x noc0_x[x], rows from 2."""
    order = layout.core_order(grid)
    words = [noc0_x[x] | (y + 2) << 16 for x, y in order]
    return layout.nocs(grid, words, words, whole)


def port_of(grid, noc0_x, whole=False):
    session = SimpleNamespace(nocs=lambda: fake_nocs(grid, noc0_x, whole))
    return tensix.TensixDispatch(session)


def blackhole_port():
    return port_of(BLACKHOLE, chips.chip("blackhole").tensix_noc0_x[:12])


def wormhole_port():
    return port_of(WORMHOLE, [1, 2, 3, 4, 6, 7, 8, 9], whole=True)  # chips.whole_grid


def test_the_port_has_every_member_of_the_interface():
    port = blackhole_port()
    missing = [name for name in dispatch.MEMBERS if not hasattr(port, name)]
    assert missing == []
    assert dispatch.kind(port) == dispatch.TENSIX


def test_the_program_gets_the_grid_less_the_dispatcher_core():
    assert dispatch.ranks(blackhole_port()) == 119
    assert dispatch.ranks(wormhole_port()) == 79
    assert dispatch.workers(WORMHOLE, dispatch.X280) == 80
    x280 = SimpleNamespace(grid_x=12, grid_y=10)  # no kind: the X280's port
    assert dispatch.ranks(x280) == 120


def test_the_dispatcher_is_the_last_core_outside_every_group():
    port = wormhole_port()
    assert (port.x, port.y) == (7, 9)
    groups = abi.column_groups(WORMHOLE, dispatch.ranks(port))
    assert port.core not in groups and len(groups) == 79
    assert {slot for _, _, _, slot in groups.values()} == set(range(8))
    assert groups[7] == (7, 0, 8, 7)  # the last column: its leader and 8 peers


def test_rectangles_reach_every_worker_once():
    rects = tensix.rectangles(blackhole_port().nocs)
    assert len(rects) == 2 and sum(receivers for _, _, receivers in rects) == 119
    # Wormhole: one rectangle over the grid, its sender counted with loopback
    assert tensix.rectangles(wormhole_port().nocs) == [(1 | 2 << 16, 9 | 11 << 16, 79)]
    loopback = chips.chip("wormhole_b0").multicast_loopback
    assert loopback and not chips.chip("blackhole").multicast_loopback
    assert tensix.rectangles(wormhole_port().nocs, loopback)[0][2] == 80


def test_a_class_s_default_rectangle_holds_its_ranks_only():
    """With no rect= a class's broadcast takes the whole rows of the ranks and its
    self.x, self.y the grid with the ranks' group; the X280's ranks change nothing."""
    nocs = wormhole_port().nocs
    program = SimpleNamespace(rectangle={"nocs": None})
    session_nocs(program, nocs, None, 79)
    assert program.rectangle["ranks"] == 79
    session_nocs(program, nocs, None, 80)  # the X280's: the whole grid
    assert program.rectangle["ranks"] is None
    args, group, rect = ranks_rectangle(SimpleNamespace(broadcasts=[]), nocs, 79)
    assert (group, rect) == ((0, 79), None)
    assert args(78)[2] == [6, 9, 1, 8, 10]  # x, y, inside, columns, rows
    buffer = SimpleNamespace(index_a=3, block=2)
    rows = SimpleNamespace(broadcasts=[("row", buffer, 2, "A", None, 0)])
    _, group, rect = ranks_rectangle(rows, nocs, 79)
    assert (group, rect) == ((0, 72), (0, 0, 8, 9))
    _, group, rect = ranks_rectangle(rows, blackhole_port().nocs, 119)
    assert (group, rect) == ((0, 108), (0, 0, 12, 9))


def test_the_cores_name_its_core_and_its_control_block():
    port = blackhole_port()
    words = dispatch.start_words(port, 0x1000, HEADERS)
    assert words == {
        "dispatcher_coordinates": 14 | 11 << 16,  # NoC0 x 14 (column 11), row 9
        "ack_base": 0x1000 + HEADERS["KM_CONTROL_DONE"],
        "ready_base": 0x1000 + HEADERS["KM_CONTROL_BOOT"],
        "mailbox": 0x1000 + HEADERS["KM_CONTROL_JOB"],
    }


def test_the_choice_of_dispatcher(monkeypatch):
    monkeypatch.delenv(dispatch.VARIABLE, raising=False)
    assert dispatch.choose(None, True) == dispatch.X280
    assert dispatch.choose(None, False) == dispatch.TENSIX
    assert dispatch.choose("tensix", True) == dispatch.TENSIX
    monkeypatch.setenv(dispatch.VARIABLE, "tensix")
    assert dispatch.choose(None, True) == dispatch.TENSIX
    with pytest.raises(ValueError, match="one of x280, tensix"):
        dispatch.choose("riscv", True)


def test_not_served_names_the_dispatcher_and_the_ability():
    error = tensix.refused("chains (run.chain)")
    assert isinstance(error, km.NotServed) and isinstance(error, km.KomorebiError)
    assert (error.dispatcher, error.ability) == ("tensix", "chains (run.chain)")
    assert str(error) == "the Tensix dispatcher does not serve chains (run.chain)"


def tensix_run():
    port = FakePort()
    port.kind = dispatch.TENSIX
    return Run(FakeBackend(port), BLACKHOLE, ranks=119, island_ranks=119)


def test_a_run_and_its_stand_in_send_commands_to_the_thread():
    for run in (tensix_run(), mock.Run(host=tensix_run())):  # a prebuild's run too
        run.submit(0, cluster=1)  # kmx_program's: a line to the thread
    assert len(tensix_run().thread_words(7)) == 3  # no refusal: as submit sends them


def test_a_run_takes_groups_inside_its_ranks_only():
    run = tensix_run()
    run.submit(0, (0, 119))
    with pytest.raises(ValueError, match="not inside 119 ranks"):
        run.submit(0, (0, 120))


def program(**fields):
    base = {"name": "p", "threads": {}, "x280_ports": None, "messages": None}
    base |= {"overlays": {}, "defines": {}}
    return SimpleNamespace(**(base | fields))


@pytest.mark.parametrize(
    "given, placed, options, ability",
    [
        ({"threads": {2: object()}}, None, {}, r"threads on clusters \[2\]"),
        ({"x280_ports": {"noc": 0}}, None, {}, "X280 ports"),
    ],
)
def test_a_load_refuses_by_name_what_it_does_not_serve(given, placed, options, ability):
    one = program(**given)
    with pytest.raises(NotServed, match=f"does not serve {ability}"):
        tensix.refuse_load(one, placed, one, options)


@pytest.mark.parametrize(
    "given",
    [
        {},
        {"overlays": {"o": 1}},
        {"defines": {"KM_ADDED_HANDLERS": 1}},
        {"messages": object()},
    ],
)
def test_a_load_of_one_program_with_its_overlays_is_served(given):
    one = program(**given)
    tensix.refuse_load(one, SimpleNamespace(programs=[one], switch=None), one, {})
    tensix.refuse_load(one, None, one, {"trace": 64})  # the cores' own trace


def test_an_islands_load_with_its_link_cores_is_served():
    one = program(defines={"KM_PAGE_LINKS": 1})
    tensix.refuse_load(one, None, one, {"island": (object(), 1), "link": object()})


def test_prepare_takes_an_islands_word_into_the_kernels_arguments():
    port = blackhole_port()
    word = 119 | 119 << 16 | 0 << 32  # island 1 of two: flat ranks 119 on, base 0
    port.prepare(16, 64, 128, 10.0, 1.0, 0.0, 0, island=word)
    assert port.options["island"] == word
    assert port.options["link"] is None  # no link line: nothing crosses
    port.prepare(16, 64, 128, 10.0, 1.0, 0.0, 0)
    assert port.options["island"] == 0  # one island: the kernel as before
    c = tensix.constants()
    assert c["kTdIsland"] + 1 == c["kTdIslandBase"] < c["kTdLinkIsland"]
    assert c["kTdIsland"] > c["kTdThreadScale"]
    assert c["kTdLinkEntries"] + 3 * c["kTdLinkIslandsMax"] == c["kTdArgs"]


def link_line(islands, island=1):
    graph = cards_graph([(8, 10)] * islands, chain(islands), per=1, dispatcher_cores=1)
    tiles = [(7, 9)] * islands
    return graph, link_area.line(graph, island, 78, 0x30000, tiles, 0)


def chain(islands):
    return [(i, i + 1) for i in range(islands - 1)]


def test_prepare_takes_the_link_line_into_the_links_arguments():
    """The host's link line (link_area.line) as td_link.hpp's arguments: this island,
    link core 0 and its area, an inbox part's packets, each island's ranks, threads and
    dispatcher tile, and the link block's bytes."""
    h = abi.headers()
    graph, line = link_line(2)
    port = blackhole_port()
    port.prepare(16, 64, 128, 10.0, 1.0, 0.0, 0, island=78 | 78 << 16, link=list(line))
    link = port.options["link"]
    assert (link["island"], link["islands"], link["core"]) == (1, 2, 78)
    assert link["area"] == 0x30000 and link["slots"] == link_area.slots(2)
    assert link["entries"][0::3] == [
        graph.first(i) | graph.count(i) << 16 for i in (0, 1)
    ]
    assert link["entries"][2::3] == [7 | 9 << 16] * 2
    part = h["KMD_LINK_COUNT_BYTES"] + link["slots"] * h["KM_LINK_PACKET_BYTES"]
    assert link["bytes"] == tensix.constants()["kTlInbox"] + part


def test_more_islands_than_the_links_arguments_hold_are_refused():
    most = tensix.constants()["kTdLinkIslandsMax"]
    _, line = link_line(most + 1)
    with pytest.raises(NotServed, match=f"commands across {most + 1} islands"):
        tensix.link_words(line)


def test_islands_whose_link_blocks_differ_are_refused():
    ports = [SimpleNamespace(link_base=0x20000), SimpleNamespace(link_base=0x20040)]
    with pytest.raises(km.KomorebiError, match="link blocks at different"):
        island_load.check_link_blocks(ports)
    island_load.check_link_blocks([SimpleNamespace(link_base=0x20000)] * 2 + [None])


def test_prepare_refuses_the_x280_kernels_options():
    port = blackhole_port()
    assert port.prepare(16, 64, 128, 10.0, 1.0, 0.0, 0, channel_region=1 << 28) > 0
    with pytest.raises(NotServed, match="work_depth"):
        port.prepare(16, 64, 128, 10.0, 1.0, 0.0, 0, work_depth=5)
    assert port.prepare(16, 64, 128, 10.0, 1.0, 0.0, 0, programs=[0, 1]) > 0


def test_a_dynamic_run_on_one_island_is_served_and_its_lines_lie_in_the_block():
    port = blackhole_port()
    port.prepare(16, 64, 128, 10.0, 1.0, 0.0, 0, messages_handler=0x0705)
    assert port.options["messages"] == 0x0705
    for word in (1 << 36 | 0x05, 3 << 48 | 1 << 36 | 0x05):  # X280 ranks, counters
        with pytest.raises(NotServed, match="X280 ranks"):
            port.prepare(16, 64, 128, 10.0, 1.0, 0.0, 0, messages_handler=word)
    port.prepare(16, 64, 128, 10.0, 1.0, 0.0, 0, messages_handler=1 << 40 | 0x05)
    assert (port.options["messages"], port.options["deliver"]) == (0x05, 1)  # waves
    module = SimpleNamespace(kernel_buffer=lambda device: (0, 1000, 1100))
    card = SimpleNamespace(module=module, device=None, cb_align=16)
    c = tensix.constants()
    block, size = tensix.block_buffer(card)
    lines = port.messages_lines(card)
    assert lines == block + c["kTbLines"] and lines % 16 == 0
    assert lines + 16 * HEADERS["KM_CORES_MAX"] <= block + size


def test_each_cores_program_goes_to_the_kernel_a_byte_a_core():
    words = tensix.held_words([0, 1, 2, 3, 5])
    c = tensix.constants()
    assert len(words) == c["kTdTag"] - c["kTdHeld"] == HEADERS["KM_CORES_MAX"] // 4
    assert words[0] == 0x03020100 and words[1] == 0xFFFFFF05
    assert set(words[2:]) == {0xFFFFFFFF}
    assert tensix.held_words([]) == [0xFFFFFFFF] * len(words)


def test_the_block_is_its_cores_first_buffer_at_a_64_byte_boundary():
    module = SimpleNamespace(kernel_buffer=lambda device: (0, 1000, 1100))
    card = SimpleNamespace(module=module, device=None, cb_align=16)
    block, size = tensix.block_buffer(card)  # buffers from 100, the first at 112
    assert block == 128 and size == 16 + tensix.constants()["kTbBytes"]


def test_a_load_of_several_programs_with_a_switch_is_served():
    one, two = program(), program(name="q")
    placed = SimpleNamespace(programs=[one, two], switch={"records": {0: 0x1000}})
    tensix.refuse_load(one, placed, one, {})


def test_event_times_by_the_dispatchers_clock():
    assert Event(1, "done", 0, 50).microseconds == 1.0  # the X280's 50 MHz
    ticks = chips.tick_mhz("blackhole", 1350)
    assert Event(1, "done", 0, 1350, None, ticks).microseconds == 1.0
    assert chips.tick_mhz("wormhole_b0", 1000) == pytest.approx(985.5)
    assert chips.tick_mhz("blackhole", None) is None


def test_a_card_split_into_islands_is_refused_with_the_tensix_dispatcher():
    with pytest.raises(NotServed, match="a card split into islands"):
        km.Session(card=0, islands=2, dispatcher="tensix")


def test_the_dispatchers_tile_is_its_own_core():
    port = wormhole_port()
    word = int(port.nocs.words[port.core])
    assert (port.l2_x, port.l2_y) == (word & 0xFFFF, word >> 16)


def test_the_default_group_is_every_core_the_program_runs_on():
    run = tensix_run()
    assert run._group(None) == (0, 119)
    x280 = Run(FakeBackend(FakePort()), BLACKHOLE)
    assert x280._group(None) == (0, 120)


def test_the_inspect_block_off_a_card_is_empty():
    read = blackhole_port().inspect(4)
    assert read["given_up"] == 0
    assert (
        read["block"].size * 4
        == HEADERS["KMD_INSPECT_HEAD"] + 4 * HEADERS["KMD_INSPECT_RECORD"]
    )


def test_the_session_names_its_dispatcher_and_ranks(monkeypatch):
    monkeypatch.delenv(dispatch.VARIABLE, raising=False)
    session = SimpleNamespace(
        _dispatch_kind=lambda module: "tensix",
        _module=object(),
        tree=None,
        grid=WORMHOLE,
    )
    assert km.Session.dispatcher.fget(session) == "tensix"
    session.dispatcher = "tensix"
    assert km.Session.ranks.fget(session) == 79
    assert km.Session.dispatcher.fset is None  # read-only


def test_no_overlay_is_built_against_the_dispatchers_kernel():
    assert kernels._runs_no_program(tensix.directory() / "dispatcher.cpp")
    reader = tensix.directory().parent / "program" / "reader.cpp"
    assert not kernels._runs_no_program(reader)


def test_a_timeout_names_the_command_by_the_handle_submit_returned():
    handle = tensix_run().submit(0, (0, 1))
    error = km.CommandTimeout(handle, 0, 1, (0, 1), [])
    assert handle == 1 and str(error).startswith("command 1 timed out")


def test_a_wait_that_runs_out_reports_the_tensix_dispatcher_without_a_marker():
    backend = SimpleNamespace(port=blackhole_port())
    backend.port.native = SimpleNamespace(is_open=True)  # its port, as on a card
    report = card_backend._Backend.kernel_report(backend)
    assert report == "the dispatcher runs; it keeps no pass marker to probe"


def test_an_ability_whose_row_is_gone_is_served(monkeypatch):
    for name in ("inspect", "recovery", "trace", "engines", "islands"):
        assert name not in tensix.REFUSED
    assert tensix.refusal("split").ability == "a card split into islands"
    monkeypatch.delitem(tensix.REFUSED, "split")
    assert tensix.refusal("split") is None
    tensix.refuse("split", "run.load")


def test_a_queued_command_goes_to_the_tensix_dispatcher():
    run = tensix_run()
    assert "queue" not in tensix.REFUSED
    run.submit(0, 8, queued=True)  # the kernel picks the 8 cores


def test_each_split_launches_its_loops():
    assert tensix.LOOPS["A"] == (0, None) and tensix.SPLIT == "B"
    assert tensix.LOOPS["B"] == (0, 1) and tensix.LOOPS["B'"] == (1, 0)
    assert tensix.constants()["kTbBytes"] % 64 == 0


def test_a_stand_in_island_run_names_one_command_as_a_submit():
    one = island_local.one_command(0, [0], [1], None)
    assert one == (0, (0, 1), ())  # the call IslandRun.submit_many makes of it
    assert island_local.one_command(0, [0, 1], [1, 1], None) is None


def test_no_x280_part_refuses_a_thread_image_by_name(monkeypatch, tmp_path):
    monkeypatch.setattr(_env, "has_x280_runtime", lambda tree: False)
    with pytest.raises(
        NotServed, match=r"does not serve X280 threads \(make of C, line 3\)"
    ):
        tensix.refuse_x280_image(tmp_path, "make of C, line 3")
    monkeypatch.setattr(_env, "has_x280_runtime", lambda tree: True)
    tensix.refuse_x280_image(tmp_path, "make of C, line 3")  # a tree with one: none


def a_thread(tmp_path, **fields):
    source = tmp_path / "hello.c"
    source.write_text("int kmx_main(int core) { return core; }\n")
    fields.setdefault("sources", [source])
    return thread_module.Thread(regions={"unused": 64}, words=[20, 22], **fields)


def test_one_thread_on_cluster_1_is_served_on_the_dispatchers_core(tmp_path):
    one = program(threads={1: a_thread(tmp_path)})
    tensix.refuse_load(one, None, one, {})
    asked = program(threads={1: a_thread(tmp_path, handlers={"relay": 7})})
    tensix.refuse_load(asked, None, asked, {})  # commands to it (kmx_program): served
    shared = program(threads={1: a_thread(tmp_path, shared=("s",))})
    with pytest.raises(NotServed, match="a thread's shared regions"):
        tensix.refuse_load(shared, None, shared, {})
    vector = tmp_path / "vmap.c"
    vector.write_text(
        "#include <riscv_vector.h>\nint kmx_main(int core) { return 0; }\n"
    )
    wide = program(threads={1: a_thread(tmp_path, sources=[vector])})
    with pytest.raises(
        NotServed, match=r"vector extension \(riscv_vector.h in vmap.c\)"
    ):
        tensix.refuse_load(wide, None, wide, {})
    tensix.refuse_load(wide, None, wide, {}, trisc=False)  # an X280 thread: served


def test_a_threads_kernel_is_its_defines_the_shim_its_sources_then_the_cores(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("KMR_CACHE", str(tmp_path / "cache"))
    thread = a_thread(tmp_path, defines={"kRounds": "4u"})
    path = tensix.thread_kernel(thread)
    lines = path.read_text().splitlines()
    assert lines[1:] == [
        "#define kRounds 4u",
        "#define KM_TD_THREAD 1",
        f'#include "{tensix.directory() / tensix.THREAD}"',
        f'#include "{(tmp_path / "hello.c").resolve()}"',
        f'#include "{tensix.directory() / tensix.COMPUTE}"',  # the core's one kernel
    ]
    assert tensix.thread_kernel(thread) == path  # written once, by its text


def test_a_threads_build_that_trisc_1_cannot_hold_is_refused_by_name():
    call = (
        "load.c:21:19: error: call to 'kmx_load' declared with attribute error: "
        "not on the dispatcher's core: its command ring serves kmx_submit, kmx_wait "
        "and kmx_poll alone\n"
    )
    with pytest.raises(NotServed, match=r"kmx_load in load\.c, a thread on its core"):
        raise tensix.thread_build_refusal("log: " + call)
    local = (
        "kernels/thread-efe16166/24225/trisc1/trisc1.elf: segment[1] "
        "[0xffb00020,+0x1010) overflows region:1 limit of 0x720 bytes, reduce"
    )
    with pytest.raises(NotServed, match="statics take 4,112 B .* holds 1,824 B"):
        raise tensix.thread_build_refusal(local)
    assert tensix.thread_build_refusal("trisc0 build failed: a syntax error") is None


def test_the_threads_area_on_a_stand_in_returns_as_the_x280s_does(monkeypatch):
    monkeypatch.setattr(mock, "active", lambda: True)
    port = wormhole_port()
    layout_of = thread_module.layout()
    block = bytes(range(64)) * (layout_of["KMX_LIM_BYTES"] // 64)
    port.work_start(1, None, block)
    assert port.work_wait(1, 0.0)["ready"]
    assert port.work_read(1, 0, 64) == bytes(range(64))
    with pytest.raises(NotServed, match="a thread on cluster 2"):
        port.work_read(2, 0, 64)
    c = tensix.constants()
    assert c["kTaLim"] + layout_of["KMX_LIM_BYTES"] == c["kTaRing"]  # then its ring
