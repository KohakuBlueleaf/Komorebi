"""Two cards and a split card offer one card's calls: every public name of Session,
Run and Tensor is offered by the island class with the same keywords, or refused by
name; the same script on the mock gives the same shapes on all three."""

import ast
import inspect
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from komorebi import abi
from komorebi.device.tensor import Tensor
from komorebi.errors import KomorebiError
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
        "grid_timeout", "island", "l1_held", "l1_rows", "l1_taken", "largest_page",
        "link_area", "link_plan", "link_plans", "load_islands", "log_given",
        "page_bounds", "page_bounds_list", "slow_code", "stuck", "tensor_table",
        "transfer_chunk", "transfer_cores", "transfer_cycles",
    },
    Run: {
        "batch", "dram", "far_cores", "grid", "half_added", "island_cores",
        "island_first", "island_ranks", "island_span", "link_events", "links_named",
        "named_cores", "own_commands", "port", "ranks", "thread_first",
        "thread_program", "transfer",
    },
    Tensor: {"kept", "loaded", "sent"},
}  # fmt: skip
# Keywords of the card's calls that are the island split's own (a part, a chain's)
OWN_KEYWORDS = {"part", "span", "records_at"}
# Not offered on islands yet
NOT_YET = {
    Session: {"tensor": {"shard_grid"}, "empty": {"shard_grid", "address"}},
    Run: {"submit": {"queued"}},
    Tensor: {"shard_grid": None},
}
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
    run.wait_s, run.send = 0.0, lambda: None
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


SCRIPT = Path(__file__).with_name("one_unit_script.py")
# what islands refuse by name that one card gives
REFUSED_ON_ISLANDS = {"run.sequence", "run.upload"}


@pytest.mark.sizes
def test_the_same_script_gives_the_same_shapes_on_one_card_two_cards_a_split_card():
    shapes = {}
    for mode, cards in (("one", "0"), ("two", "0,1"), ("split", "0")):
        done = subprocess.run(
            [sys.executable, "-m", "komorebi.cli", "prebuild", "--cards", cards]
            + [str(SCRIPT), mode],
            capture_output=True,
            text=True,
            timeout=600,
            env=os.environ,
        )
        lines = [
            line for line in done.stdout.splitlines() if line.startswith("SHAPES ")
        ]
        assert done.returncode == 0 and lines, done.stdout[-2000:] + done.stderr[-2000:]
        shapes[mode] = json.loads(lines[-1][len("SHAPES ") :])
    for mode in ("two", "split"):
        for name, one in shapes["one"].items():
            if name in REFUSED_ON_ISLANDS:
                assert shapes[mode][name] == ["refused", "AttributeError"], name
            else:
                assert shapes[mode][name] == one, (mode, name)


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
        def submit(self, *args, **options):
            given.append(options)
            return 7

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
