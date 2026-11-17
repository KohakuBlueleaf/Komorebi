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

from komorebi.device.tensor import Tensor
from komorebi.errors import KomorebiError
from komorebi.islands import run as island_run
from komorebi.islands import same
from komorebi.islands.records import IslandTensor
from komorebi.islands.run import IslandRun
from komorebi.islands.session import IslandSession
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
    Session: {
        "tensor": {"shard_grid", "transposed_tiles"},
        "empty": {"shard_grid", "address"},
    },
    Run: {
        "submit": {"queued", "block"},
        "thread": None, "threads": None, "thread_result": None, "thread_results": None,
    },
    Tensor: {"shard_grid": None, "transposed_tiles": None},
}  # fmt: skip
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
