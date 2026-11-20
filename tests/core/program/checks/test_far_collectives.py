"""The host's collectives over islands, once the host has seen a command given up at
its deadline on a load whose movers call a collective, are refused at the call before
anything is queued, naming the leaders marked, until run.clear_collectives()."""

from types import SimpleNamespace

import numpy as np
import pytest

import komorebi as km
from komorebi import abi
from komorebi.errors import KomorebiError
from komorebi.islands.parts import link_plan
from komorebi.islands.run import IslandRun
from komorebi.session import state

KERNEL = "void kernel_main() {}\n"
READER = """\
#include "reduce.h"
// km::all_reduce(reader, command, ...): a comment
const char* said = "km::reduce_to(";
void kernel_main() { km::all_reduce_keeps(group); }
"""
REDUCE = """\
#pragma once
inline bool sum(const km::Reader& reader, const km_command_t& command) {
    return km::all_reduce (reader, command, 0, 0, 1, 0, km::Sum{}, 0, 0);
}
"""


def program(folder, reduce=REDUCE):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "reader.cpp").write_text(READER)
    (folder / "reduce.h").write_text(reduce)
    (folder / "writer.cpp").write_text(KERNEL)
    (folder / "compute.cpp").write_text(KERNEL)
    return km.Program("sums", kernels=folder, defines={"KM_TENSOR_TABLE": 1})


def given_up_run(programs, marked, seen=1):
    """An IslandRun of two islands whose host has seen `seen` give-ups at a deadline,
    its generation rows marking `marked` (coll_marked's rows); no command can be
    queued (its Runs have no queue)."""
    run = IslandRun.__new__(IslandRun)
    run.runs = (
        SimpleNamespace(_given_up_seen=seen, programs=programs),
        SimpleNamespace(_given_up_seen=0, programs=programs),
    )
    run._counts = lambda: None
    run._coll_clear_at, run._coll_movers = 0, None
    run.session = SimpleNamespace(arch="blackhole", coll_marked=lambda: marked)
    return run


def copy_tensor(pages=4):
    return SimpleNamespace(
        pages=pages,
        placement=SimpleNamespace(copy=True),
        dtype="float32",
        aligned_page_bytes=4096,
        address=0x9000,
    )


def test_a_host_collective_after_a_seen_give_up_is_refused_naming_the_marked(tmp_path):
    h = abi.headers()
    deadline = h["kGaveDeadline"] | h["kWhereAsk"] << 8
    marked = [
        (1, 7, state.gave_up(deadline, 5), 5, 0),  # its begin
        (0, 3, state.gave_up(deadline, 5), 5, 1),  # its team met
        (30, 9, None, 5, 3),
    ]
    run = given_up_run([program(tmp_path / "a")], marked)
    with pytest.raises(KomorebiError) as caught:
        run.all_reduce(copy_tensor(), copy_tensor(8))  # refused before any queue
    assert str(caught.value) == (
        "run.all_reduce(): refused before anything was queued: islands 0, 1 and 30: "
        "a collective was given up on this load and the host has not yet cleared it; "
        "the tensors it was reducing are not whole: write them again, then "
        "run.clear_collectives(); the host saw a command given up at its deadline on "
        "this load, whose movers call a collective; the leaders marked, earliest "
        "stage first: island 1's leader (core 7) gave up at its begin, before its team "
        "met (deadline in ask) in the command of sequence 5; island 0's leader (core "
        "3) gave up at its team met, its shares on their way (deadline in ask) in the "
        "command of sequence 5; island 30's leader (core 9) gave up at its shares in, "
        "its posts not yet confirmed (no give-up recorded) in the command of sequence 5"
    )
    for call in (
        lambda: run.all_gather(copy_tensor()),
        lambda: run.broadcast_from(copy_tensor(), root=1),
        lambda: run.reduce_to(copy_tensor(), copy_tensor(8), root=1),
    ):
        with pytest.raises(KomorebiError, match="refused before anything was queued"):
            call()


def test_the_refusal_says_so_when_no_leader_is_marked(tmp_path):
    run = given_up_run([program(tmp_path / "a")], [])
    with pytest.raises(KomorebiError) as caught:
        run.all_gather(copy_tensor())
    assert str(caught.value).startswith(
        "run.all_gather(): refused before anything was queued: an island: a "
        "collective was given up on this load"
    )
    assert str(caught.value).endswith(
        "; no leader is marked (the command given up was not a collective, or its "
        "marks were not written)"
    )


def test_no_refusal_after_the_clear_without_a_give_up_or_without_a_collective(
    tmp_path,
):
    calling = [program(tmp_path / "a")]
    for run in (
        given_up_run(calling, [], seen=0),  # no give-up seen
        given_up_run([program(tmp_path / "b", "#pragma once\n")], []),
    ):
        run._coll_cleared("run.all_reduce()")
    run = given_up_run(calling, [])
    run._coll_clear_at = 1  # what clear_collectives() keeps: the give-ups it saw
    run._coll_cleared("run.all_reduce()")
    run.runs[1]._given_up_seen = 1  # a give-up after the clear
    with pytest.raises(KomorebiError, match="refused before anything was queued"):
        run._coll_cleared("run.all_reduce()")


def test_the_clear_keeps_the_give_ups_it_saw():
    run = given_up_run([], [])
    run.drain = lambda timeout: None
    run.session.raise_generation = lambda: 2
    assert run.clear_collectives() == 2
    assert run._coll_clear_at == 1


def test_the_marked_rows_earliest_stage_first_with_what_they_gave_up_in():
    h = abi.headers()
    words = np.zeros((4, 16), np.uint32)
    words[:, h["KM_COLL_GEN_NOW"]] = 2
    for core, stage, sequence in ((1, 3, 9), (2, 0, 9)):
        words[core, h["KM_COLL_GEN_MARK"]] = 2
        words[core, h["KM_COLL_GEN_GAVE"]] = h["kGaveDeadline"] | h["kWhereAsk"] << 8
        words[core, h["KM_COLL_GEN_SEQUENCE"]] = sequence
        words[core, h["KM_COLL_GEN_STAGE"]] = stage
    words[3, h["KM_COLL_GEN_MARK"]] = 1  # marked under an earlier generation: not now
    self = SimpleNamespace(
        _gen_words=lambda: [(None, None, {0: 0, 1: 0, 2: 1, 3: 1}, words)]
    )
    marked = link_plan.LinkPlanMixin.coll_marked(self)
    assert [(i, c, s, st) for i, c, _, s, st in marked] == [(1, 2, 9, 0), (0, 1, 9, 3)]
    assert (marked[0][2]["code"], marked[0][2]["where"]) == ("deadline", "ask")
