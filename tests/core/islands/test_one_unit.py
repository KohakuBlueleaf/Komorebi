"""Two cards and a split card offer one card's calls: every public name of Session,
Run and Tensor is offered by the island class with the same keywords, or refused by
name; the same script on the mock gives the same shapes on all three."""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from komorebi import abi
from komorebi.device.tensor import Tensor
from komorebi.dsl.front.front import HandlerMethod
from komorebi.errors import KomorebiError, ProgramError
from komorebi.islands import run as island_run
from komorebi.islands import same
from komorebi.islands.records import IslandTensor
from komorebi.islands.run import IslandRun
from komorebi.islands.session import IslandSession
from komorebi.session import state as device_state
from komorebi.session.run import Run
from komorebi.session.session import Session

# The card classes' own workings, used by no script: not offered on islands
INTERNAL = {
    Session: {
        "grid_timeout", "island", "kernel_followed", "l1_held", "l1_rows", "l1_taken",
        "largest_page", "link_area", "link_plan", "link_plans", "load_islands",
        "log_given", "page_bounds", "page_bounds_list", "slow_code", "stuck",
        "tensor_table", "transfer_chunk", "transfer_cores", "transfer_cycles",
    },
    Run: {
        "arch", "batch", "dram", "far_cores", "grid", "half_added", "island_cores",
        "island_first", "island_number", "island_ranks", "island_span", "link_events",
        "links_named", "named_cores", "own_commands", "port", "ranks", "send_islands",
        "thread_first", "thread_program", "transfer",
    },
    Tensor: {"kept", "loaded", "sent"},
}  # fmt: skip
# Keywords of the card's calls that are the island split's own (a part, a chain's)
OWN_KEYWORDS = {"part", "span", "records_at", "checked"}
# Not offered on islands yet
NOT_YET = {Session: {}, Run: {}, Tensor: {}}
REFUSED = {Run: same.RUN_REFUSED, Tensor: same.TENSOR_REFUSED, Session: {}}
PAIRS = ((Session, IslandSession), (Run, IslandRun), (Tensor, IslandTensor))


def public(cls):
    """Every public name of `cls`: its class's, and each `self.name = ...` of its
    classes' code (what its instances hold)."""
    names = {name for name in dir(cls) if not name.startswith("_")}
    for klass in cls.__mro__[:-1]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(klass)))
        names |= {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and not node.attr.startswith("_")
        }
    return names


@pytest.mark.parametrize("card, islands", PAIRS)
def test_every_name_of_one_card_is_offered_refused_or_internal(card, islands):
    theirs = public(islands)
    missing = []
    for name in sorted(public(card) - INTERNAL[card]):
        if NOT_YET[card].get(name, ()) is None or name in REFUSED[card]:
            continue
        if name not in theirs:
            missing.append(name)
    assert missing == []


@pytest.mark.parametrize("card, islands", PAIRS)
def test_a_shared_call_takes_one_card_s_keywords_and_defaults(card, islands):
    for name in sorted(set(dir(card)) & set(dir(islands))):
        ours, theirs = getattr(card, name), getattr(islands, name)
        if name.startswith("_") or not (callable(ours) and callable(theirs)):
            continue
        mine = inspect.signature(theirs).parameters
        later = NOT_YET[card].get(name) or set()
        for key, parameter in inspect.signature(ours).parameters.items():
            if key in OWN_KEYWORDS | later or parameter.kind is parameter.VAR_KEYWORD:
                continue
            assert key in mine, f"{card.__name__}.{name}: {key}"
            assert mine[key].default == parameter.default, f"{name}: {key}"


def test_a_name_islands_do_not_offer_is_refused_by_what_to_call():
    run = IslandRun.__new__(IslandRun)
    with pytest.raises(AttributeError, match=r"run\.upload on islands: .*island\(i\)"):
        _ = run.upload
    with pytest.raises(AttributeError, match=r"trace\(\)\.command\(ticket\)"):
        _ = run.sequence
    assert not hasattr(run, "backend") and not hasattr(run, "no_such_name")
    tensor = IslandTensor.__new__(IslandTensor)
    with pytest.raises(AttributeError, match=r"tensor\.parts\[i\]\.buffer"):
        _ = tensor.buffer


def test_one_value_where_every_island_has_it_else_refused_by_name():
    assert same.same("x", [(12, 10), (12, 10)], "own") == (12, 10)
    with pytest.raises(KomorebiError, match=r"session\.l1\(\): the islands' differ"):
        session = SimpleNamespace(sessions=[1, 2])
        same.SessionCalls._same(session, "l1()", lambda one: one)
    two = SimpleNamespace(cards=(0, 1))
    with pytest.raises(KomorebiError, match=r"session\.cards"):
        same.SessionCalls.card.fget(two)
    assert same.SessionCalls.card.fget(SimpleNamespace(cards=(3,))) == 3


def two_islands():
    """An IslandRun over two fake runs of 4 ranks and a link core each."""
    graph = SimpleNamespace(
        rank_cores=[[0, 1, 2, 3], [0, 1, 2, 3]],
        first=lambda island: 4 * island,
    )

    class Part:
        cores = 5
        recovery_refused = None
        stalls = ()

        def __init__(self):
            self.arrived, self.kept, self.groups = set(), [], {}

        def _arrived(self, handles):
            return np.isin(handles, list(self.arrived))

        def _check_alive(self):
            pass

        def _step(self, handles, upto, timeout):
            self.arrived |= set(handles.tolist())

        def keep_resets(self, exclude, timeout):
            self.kept.append(sorted(exclude))
            return True

        def group(self, handle):
            return self.groups.get(handle)

    run = IslandRun.__new__(IslandRun)
    run.graph, run.runs, run._next = graph, (Part(), Part()), 3
    run._parts = [np.array([0, 1, 2, 0]), np.array([0, 0, 1, 0])]
    run.wait_s, run.send, run.queued_on = 0.0, lambda: None, {}
    run.wait = lambda ticket, timeout: ("event", ticket)
    return run


def test_wait_any_returns_the_first_ticket_whose_every_part_arrived():
    run = two_islands()
    run.runs[0].arrived = {2}  # ticket 2: island 0 done, island 1 not yet
    assert run.wait_any([2, 1]) == ("event", 1)  # 1 is whole after one round
    with pytest.raises(ValueError, match="never submitted"):
        run.wait_any([3])


def test_keep_resets_leaves_out_each_island_s_link_cores_and_the_ranks_given():
    run = two_islands()
    assert run.keep_resets(exclude=[5]) is True
    assert [part.kept[0] for part in run.runs] == [[4], [1, 4]]


def test_a_ticket_s_group_and_the_islands_reasons():
    run = two_islands()
    run.runs[0].groups[1] = (0, 6)
    assert run.group(1) == (0, 6) and run.group(9) is None
    run.runs[1].recovery_refused = "semaphores"
    run.runs[1].stalls = [{"stage": 1}]
    assert run.recovery_refused == "island 1: semaphores"
    assert run.stalls == [{"stage": 1, "island": 1}]


def test_the_island_run_takes_one_card_s_keyword_names():
    for name, keyword in (
        ("wait", "handles"),
        ("event", "handle"),
        ("forget", "handles"),
        ("given_up", "handle"),
        ("inspect", "group"),
        ("reset_core", "core"),
        ("addon", "name"),
    ):
        assert (
            keyword in inspect.signature(getattr(island_run.IslandRun, name)).parameters
        )


def block(**changed):
    """A clean result block of one island, with `changed` keys."""
    h = abi.headers()
    words = np.zeros(h["KMD_R_WORDS"], np.uint32)
    words[h["KMD_R_STATE"]] = h["KMD_STATE_DONE"]
    words[h["KMD_R_COMMANDS"]] = 5
    return device_state.result_block(words, h) | {"cleaned": True, **changed}


def test_the_islands_result_reads_as_one_card_s_block_each_island_s_kept():
    h = abi.headers()
    deadline = {"error": h["KMD_ERROR_DEADLINE"], "error_name": "deadline"}
    blocks = [block(control=1, result=7), block(control=2, result=7, **deadline)]
    result = device_state.island_result(blocks)
    assert set(result) == set(blocks[0]) | {"island"} and result.islands == blocks
    assert (result["error_name"], result["island"]) == ("deadline", 1)
    assert result["commands"] == 10 and result["state_name"] == "done"
    assert result["control"] is None and result["result"] == 7  # differ / alike
    clean = device_state.island_result([block(), block()])
    assert clean["error_name"] == "none" and clean["island"] is None
    ended = device_state.island_result([block(), block(ended="core 3")])
    assert ended["ended"] == "island 1: core 3" and not ended.get("failed")


def test_an_island_tensor_is_not_read_or_written_while_a_run_is_loaded():
    tensor = IslandTensor.__new__(IslandTensor)
    tensor.shape, tensor.dtype, tensor.layout, tensor.parts = (
        (64, 32),
        "bf16",
        "row",
        (),
    )
    tensor.in_dram, tensor.sent, tensor.loaded = True, None, lambda: True
    with pytest.raises(KomorebiError, match=r"session\.read\(tensor\) reads its pages"):
        tensor.read_bytes()
    tensor.loaded = lambda: False
    tensor.session = SimpleNamespace(read=lambda one: [np.zeros((2, 4), np.uint8)])
    assert tensor.read_bytes().shape == (8,)


def test_transposed_tiles_is_refused_once_the_tensor_holds_data_on_both_kinds():
    class Buffer:
        page_size, num_pages, address = 2048, 2, 0x1000

        def write(self, raw):
            pass

    card = Tensor(Buffer(), (32, 64), "bfloat16", "tile")
    card.transposed_tiles = True  # empty: before its data, as a build marks it
    card.write(np.zeros((32, 64), np.float32))
    card.transposed_tiles = True  # the same: no change
    island = IslandTensor.__new__(IslandTensor)
    island.parts, island.shape, island._transposed = (card,), (32, 64), True
    island.dtype, island.layout, island.written = "bfloat16", "tile", True
    for tensor in (card, island):
        with pytest.raises(KomorebiError, match="set after its data was written"):
            tensor.transposed_tiles = False


def test_a_block_goes_with_every_island_s_part():
    given = []

    class Part:
        programs = (SimpleNamespace(),)  # no messages: no quota to check

        def submit(self, *args, **options):
            given.append(options)
            return 7

        def _submit_words(self, *checked):  # its part's checks: none refuse
            pass

        def _handler(self, handler):
            return 0, 0

    run = IslandRun.__new__(IslandRun)
    run.graph = SimpleNamespace(split=lambda group: [(0, 0, 2), (1, 0, 2)])
    run.runs, run._next, run._parts = (Part(), Part()), 1, [np.zeros(4, int)] * 2
    run.cores, run.submit_s, run.linked, run._counts = 8, 0.0, False, None
    run._half_added, run.program, run.crossing_line = {}, None, "off"
    assert run.submit("h", (2, 4), (1,), block=b"words") == 1
    assert [(one["block"], one["part"]) for one in given] == [(b"words", True)] * 2


def test_the_threads_of_every_island_by_flat_number_the_first_as_run_thread():
    def thread(cluster):
        return SimpleNamespace(thread=SimpleNamespace(cluster=cluster))

    held = ({1: thread(1), 2: thread(2)}, {1: thread(1)})
    runs = tuple(SimpleNamespace(_thread_runs=lambda one=one: one) for one in held)
    run = IslandRun.__new__(IslandRun)
    run.runs, run.graph = runs, SimpleNamespace(thread_firsts=[1, 4])
    assert sorted(run.threads) == [1, 2, 4]
    assert run.thread is held[0][1] and run.threads[4] is held[1][1]


def test_recovery_reads_as_one_card_s_with_each_island_s_and_none_if_one_kept_none():
    def kept(cores, size):
        return {"offset": 0, "bytes": size, "lists": [(0, size)], "cores": cores,
                "seconds": 0.5}  # fmt: skip

    run = IslandRun.__new__(IslandRun)
    run.graph = SimpleNamespace(rank_cores=((3, 4), (3, 5)), first=lambda i: 2 * i)
    run.runs = (SimpleNamespace(recovery=kept([3, 4], 8)),
                SimpleNamespace(recovery=kept([5, 9], 4)))  # fmt: skip
    got = run.recovery
    assert (got["bytes"], got["cores"], got["seconds"]) == (12, [0, 1, 3], 1.0)
    assert len(got["lists"]) == 2 and got.islands[1]["bytes"] == 4
    run.runs[1].recovery = None
    assert run.recovery is None


def test_a_turned_product_s_weight_is_checked_on_an_island_tensor_as_on_one_card():
    handler = SimpleNamespace(name="apply", placed={"w": (True, "apply's product")})
    island = IslandTensor.__new__(IslandTensor)
    island.parts, island._transposed, island.written = (), False, False
    island.address, island.dtype, island.page_bytes, island.placement = 0, "", 0, None
    with pytest.raises(ProgramError, match="tensor w is not stored with its tiles"):
        HandlerMethod.placed_check(handler, "w", island)
    island.transposed_tiles = True
    HandlerMethod.placed_check(handler, "w", island)


class L1Part:
    """A card session's L1 tensor as empty() gives it: rows a core, its address."""

    def __init__(self, shape, at):
        self.shape, self.address, self.page_bytes = shape, at, 4 * shape[1]
        self.aligned_page_bytes, self.freed = self.page_bytes, False

    def free(self):
        self.freed = True


def l1_session(addresses, split=1):
    """An island session over cards whose L1 tensors land at `addresses` in turn."""
    made, at = [], iter(addresses)

    class Card:
        def empty(self, shape, dtype, layout, shard_grid=None, address=None):
            made.append(L1Part(shape, next(at) if address is None else address))
            return made[-1]

    session = IslandSession.__new__(IslandSession)
    session.split = split
    session.sessions = (Card(),) * split if split > 1 else (Card(), Card())
    session.run = None
    return session, made


def test_an_l1_tensor_is_a_row_a_core_of_every_island_at_one_address():
    session, made = l1_session([0x18000, 0x18000])
    tensor = session.empty((2 * 6, 4), "uint32", shard_grid=(3, 2))
    assert (tensor.address, tensor.shard_grid, tensor.in_dram) == (
        0x18000,
        (3, 2),
        False,
    )
    assert [part.shape for part in made] == [(6, 4), (6, 4)]
    assert tensor.placement.part(1).tolist() == list(range(6, 12))
    assert tensor.pages == 12 and tensor.placement.where(7) == (1, 1)


def test_a_split_card_s_l1_tensor_is_the_card_s_every_island_holding_it():
    session, made = l1_session([0x18000], split=2)
    tensor = session.empty((6, 4), "uint32", shard_grid=(3, 2))
    assert len(made) == 1 and tensor.parts == (made[0], made[0])
    assert tensor.placement.copy and tensor.pages == 6


def test_an_l1_tensor_at_two_addresses_is_refused_naming_the_cause():
    session, made = l1_session([0x18000, 0x18040])
    with pytest.raises(KomorebiError, match="made on one island alone"):
        session.empty((12, 4), "uint32", shard_grid=(3, 2))
    assert all(part.freed for part in made)


def test_an_l1_tensor_of_other_rows_or_with_a_placement_hint_is_refused():
    session, _ = l1_session([0x18000] * 4)
    with pytest.raises(ValueError, match=r"\[2 islands x 6 cores, words\]"):
        session.empty((6, 4), "uint32", shard_grid=(3, 2))
    with pytest.raises(ValueError, match="no split, group, like, copy or page_tiles"):
        session.empty((12, 4), "uint32", copy=True, shard_grid=(3, 2))
    with pytest.raises(ValueError, match="address= is an L1 tensor's"):
        session.empty((12, 4), "uint32", address=0x18000)


def test_an_event_is_one_card_s_once_every_part_arrived_and_parts_say_each():
    done = SimpleNamespace(kind="done", ticks=5, result=9)
    arrived = {1: done, 2: None}
    run = IslandRun.__new__(IslandRun)
    run.runs = tuple(SimpleNamespace(event=arrived.get) for _ in range(2))
    run._parts = [np.array([0, 1]), np.array([0, 1])]
    assert run.event(1).kind == "done" and run.event(1).result == 9
    run._parts[1] = np.array([0, 2])
    assert run.event(1) is None and run.event_parts(1) == (done, None)


def queued_run(split=1):
    """An island run over two islands of 3 and 2 ranks whose parts answer submit."""
    given, arrived = [], set()

    class Part:
        def __init__(self, island):
            self.island, self.next = island, 0

        def submit(self, handler, count, args, result=False, queued=False, block=None):
            given.append((self.island, count, queued))
            self.next += 1
            return self.next

        def _arrived(self, handles):
            return np.isin(handles, [h for i, h in arrived if i == self.island])

        def group(self, handle):
            return (1, 2)  # core 1 of its island, the queue's choice

    run = IslandRun.__new__(IslandRun)
    run.runs = (Part(0), Part(1))
    run.graph = SimpleNamespace(
        count=lambda i: (3, 2)[i],
        first=lambda i: (0, 3)[i],
        rank_cores=((0, 1, 2), (0, 1)),
    )
    run.session = SimpleNamespace(split=split)
    run._next, run._parts, run.submit_s, run._half_added = (
        1,
        [np.zeros(8, int)] * 2,
        0.0,
        {},
    )
    run._queued, run.queued_on = [[], []], {}
    return run, given, arrived


def test_a_queued_command_goes_whole_to_the_island_with_fewest_unfinished():
    run, given, arrived = queued_run()
    first = run.submit("h", 2, (1,), queued=True)
    second = run.submit("h", 2, (1,), queued=True)
    assert [island for island, _, _ in given] == [0, 1] and given[0][1:] == (2, True)
    arrived.add((0, 1))  # island 0's done: it has none unfinished again
    run.submit("h", 2, (1,), queued=True)
    run.submit("h", 3, (1,), queued=True)  # only island 0 has 3 ranks
    assert [island for island, _, _ in given] == [0, 1, 0, 0]
    assert run.group(first) == (1, 2) and run.group(second) == (4, 2)  # flat


def test_a_queued_command_past_every_island_exclusive_or_on_a_split_card_refused():
    run, _, _ = queued_run()
    with pytest.raises(ValueError, match="one island's ranks"):
        run.submit("h", 4, queued=True)
    with pytest.raises(ValueError, match="cannot be exclusive"):
        run.submit("h", 2, queued=True, exclusive=True)
    with pytest.raises(KomorebiError, match="split into islands"):
        queued_run(split=2)[0].submit("h", 2, queued=True)


def test_a_closed_island_session_closes_each_islands_device_before_the_mesh():
    done = []

    class Closed:
        def __init__(self, name):
            self.name = name

        def close(self):
            done.append(self.name)

    session = IslandSession.__new__(IslandSession)
    session._memory, session.split, session.run, session.log = None, 1, None, print
    session.mesh = Closed("mesh")
    session.sessions = [
        SimpleNamespace(stuck=False, device=Closed(f"device {i}"), close=lambda: None)
        for i in range(2)
    ]
    session.close()
    assert done == ["device 0", "device 1", "mesh"]
    assert session.mesh is None and [s.device for s in session.sessions] == [None] * 2
