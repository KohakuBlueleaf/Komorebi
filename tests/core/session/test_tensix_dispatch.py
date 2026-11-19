"""The Tensix dispatcher's host side without a card: its port's interface, the cores it
leaves the program, its rectangles and start words, and what it refuses by name."""

from types import SimpleNamespace

import pytest
from fake_port import FakeBackend, FakePort

import komorebi as km
from komorebi import abi, layout
from komorebi.device import chips
from komorebi.device.events import Event
from komorebi.errors import NotServed
from komorebi.session import dispatch, kernels
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


@pytest.mark.parametrize(
    "call, ability",
    [
        (lambda run: run.submit(0, (0, 1), queued=True), "the job queue"),
        (lambda run: run.submit(0, (0, 1), cluster=1), "X280 threads"),
        (lambda run: run.submit(0, (0, 1), part=True), "islands and links"),
        (lambda run: run.submit_many(0, [0], [1], part=True), "islands and links"),
        (lambda run: run.reset_core(0), "recovery beyond the stop"),
        (lambda run: run.inspect(), "the inspect"),
        (lambda run: run.upload(b"", "stream"), "stream/graph engines"),
        (lambda run: run.run_hostfed(None), "stream/graph engines"),
        (lambda run: run.thread_words(0), "X280 threads"),
    ],
)
def test_a_run_refuses_by_name_what_it_does_not_serve(call, ability):
    with pytest.raises(NotServed, match=f"does not serve {ability}"):
        call(tensix_run())


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
        ({"threads": {1: object()}}, None, {}, "X280 threads"),
        ({"messages": object()}, None, {}, "dynamic runs"),
        ({}, None, {"island": (1, 0)}, "islands and links"),
        ({"defines": {"KM_PAGE_LINKS": 1}}, None, {}, "islands and links"),
        ({}, None, {"trace": 64}, "the trace"),
    ],
)
def test_a_load_refuses_by_name_what_it_does_not_serve(given, placed, options, ability):
    one = program(**given)
    with pytest.raises(NotServed, match=f"does not serve {ability}"):
        tensix.refuse_load(one, placed, one, options)


@pytest.mark.parametrize(
    "given", [{}, {"overlays": {"o": 1}}, {"defines": {"KM_ADDED_HANDLERS": 1}}]
)
def test_a_load_of_one_program_with_its_overlays_is_served(given):
    one = program(**given)
    tensix.refuse_load(one, SimpleNamespace(programs=[one], switch=None), one, {})


def test_prepare_refuses_the_x280_kernels_options():
    port = blackhole_port()
    assert port.prepare(16, 64, 128, 10.0, 1.0, 0.0, 0, channel_region=1 << 28) > 0
    with pytest.raises(NotServed, match="messages_handler"):
        port.prepare(16, 64, 128, 10.0, 1.0, 0.0, 0, messages_handler=5)
    assert port.prepare(16, 64, 128, 10.0, 1.0, 0.0, 0, programs=[0, 1]) > 0


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


def test_islands_are_refused_with_the_tensix_dispatcher():
    with pytest.raises(NotServed, match="islands and links"):
        km.Session(cards=[0, 1], dispatcher="tensix")


def test_the_default_group_is_every_core_the_program_runs_on():
    run = tensix_run()
    assert run._group(None) == (0, 119)
    x280 = Run(FakeBackend(FakePort()), BLACKHOLE)
    assert x280._group(None) == (0, 120)


def test_a_deadline_finds_no_command_given_up():
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
