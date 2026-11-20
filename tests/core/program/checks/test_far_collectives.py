"""On a chip whose links may not target a far DRAM (Wormhole) the collectives over
islands land in the far leader's L1: a load whose movers call one makes that landing,
refused by name below two slots of one page; the host's collectives without it, or after
a give-up seen and not cleared, are refused at the call, before anything is queued."""

from types import SimpleNamespace

import numpy as np
import pytest

import komorebi as km
from komorebi import abi, errors
from komorebi.errors import KomorebiError
from komorebi.islands.parts import link_plan
from komorebi.islands.parts.load import LoadMixin
from komorebi.islands.run import IslandRun
from komorebi.program.checks import far_collectives
from komorebi.session import kernels, state

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
HAZARD = "docs/reference/hazards.md#a-link-packet-for-a-far-wormhole-chips-dram"


def program(folder, reduce=REDUCE, far=True):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "reader.cpp").write_text(READER)
    (folder / "reduce.h").write_text(reduce)
    (folder / "writer.cpp").write_text(KERNEL)
    (folder / "compute.cpp").write_text(KERNEL)
    defines = {"KM_TENSOR_TABLE": 1} | ({"KM_LINK_NO_FAR_DRAM": 1} if far else {})
    return km.Program("sums", kernels=folder, defines=defines)


def test_a_call_in_an_included_file_wants_the_landing(tmp_path):
    assert far_collectives.wanted([program(tmp_path / "a")])
    assert not far_collectives.wanted([program(tmp_path / "b", far=False)])
    assert not far_collectives.wanted(
        [program(tmp_path / "c", reduce="#pragma once\n")]
    )


def test_comments_strings_and_other_names_are_not_calls(tmp_path):
    program(tmp_path / "c", reduce="#pragma once\n")
    assert far_collectives.scan(tmp_path / "c" / "reader.cpp") == []


@pytest.mark.parametrize(
    "name", ["km::reduce_to", "km::all_gather", "km::broadcast_from"]
)
def test_each_collective_is_found(tmp_path, name):
    program(tmp_path / "d", reduce=f"#pragma once\nbool f() {{ return {name}(c); }}\n")
    (found,) = far_collectives.scan(tmp_path / "d" / "reader.cpp")
    assert found[1:] == (2, name)


def plan(tmp_path, left, page=2048, arch="wormhole_b0", buffers=0, calls=True):
    said = []
    made = program(tmp_path / "p", reduce=REDUCE if calls else "#pragma once\n")
    made.buffers = [SimpleNamespace(page_bytes=buffers, pages=1)] if buffers else []
    session = SimpleNamespace(largest_page=page, l1=lambda: (0, left))
    return (
        SimpleNamespace(arch=arch, sessions=[session], log=said.append),
        [made],
        said,
    )


def test_the_landing_takes_whole_pages_up_to_its_chunk_from_half_the_l1_left(tmp_path):
    self, programs, said = plan(tmp_path, 1 << 20, buffers=200_000)
    assert link_plan.LinkPlanMixin._tree_landing_of(self, programs) == (2, 32768)
    assert said == [
        "collectives' landing: 2 slots a port of 32768 B (16 pages of 2048 B), "
        "295552 B a core"
    ]


def test_little_l1_left_gives_fewer_pages_a_slot_then_the_named_floor(tmp_path):
    self, programs, said = plan(tmp_path, 120_000)
    assert link_plan.LinkPlanMixin._tree_landing_of(self, programs) == (2, 6144)
    self, programs, said = plan(tmp_path, 40_000)
    assert link_plan.LinkPlanMixin._tree_landing_of(self, programs) == (2, 2048)
    assert "(the floor: 1 pages of 2048 B)" in said[0]


def test_a_load_that_cannot_hold_two_slots_of_one_page_is_refused_by_name(tmp_path):
    self, programs, _ = plan(tmp_path, 30_000)
    with pytest.raises(KomorebiError) as caught:
        link_plan.LinkPlanMixin._tree_landing_of(self, programs)
    assert str(caught.value) == (
        "the collectives' landing (km::all_reduce and the others on wormhole_b0: "
        "their pages land in the far leader's L1) needs 19072 B of L1 on every core "
        "at least, 2 slots of one 2048 B page a port, at most half of what the load "
        "leaves (30000 B): fewer or smaller buffers"
    )


def test_no_landing_on_blackhole_or_for_a_load_that_calls_no_collective(tmp_path):
    self, programs, said = plan(tmp_path, 1 << 20, arch="blackhole")
    assert link_plan.LinkPlanMixin._tree_landing_of(self, programs) is None
    self, programs, said = plan(tmp_path, 1 << 20, calls=False)
    assert link_plan.LinkPlanMixin._tree_landing_of(self, programs) is None
    assert said == []


def test_the_island_table_holds_the_landing_s_two_words_and_its_bytes():
    graph = SimpleNamespace(
        islands=[SimpleNamespace(cores=range(2))] * 2,
        table=lambda island: np.zeros(4, np.uint32).tobytes(),
        children=lambda island: np.zeros(2, np.uint32),
        count=lambda island: 2,
    )

    class Words(np.ndarray):
        address, aligned_page_bytes = 0x5000, 32

    session = SimpleNamespace(
        grid=(2, 1), tensor=lambda words, *a, **k: words.view(Words)
    )
    self = SimpleNamespace(
        graph=graph,
        arch="wormhole_b0",
        split=1,
        sessions=[session, session],
        _tree_landing=(2, 64),
        _coll_gen={},
        _far_tiles=lambda island: [np.zeros(2, np.uint32), np.zeros(2, np.uint32)],
    )
    words = link_plan.LinkPlanMixin._table_of(self, session)
    # after 6 x 2 words the meeting words (their number, 2, then 2), the generation's
    # two, the landing's two
    assert list(words[0, 12:19]) == [2, 0, 0, 0x5000, 32, 2, 64]
    # in DRAM, a page a core: generation 1, no mark, led under none, no drop
    assert self._coll_gen[0].tolist() == [[1] + [0] * 15] * 2
    # 6 x 2 words, 3 meeting words, the four, 64 B to align, the generation row (8),
    # credits (8 x 8 words), 9 slots of 32 + 64 B
    assert words.shape == (2, -(-(12 + 3 + 4 + 16 + 8 + 64 + 9 * 24) // 16) * 16)
    self._tree_landing = None  # no landing: the table as on any load, the landing's
    words = link_plan.LinkPlanMixin._table_of(self, session)  # two words 0
    assert list(words[0, 15:19]) == [0x5000, 32, 0, 0]
    assert words.shape == (2, 48)


def test_a_host_collective_on_a_load_without_the_landing_is_refused_by_name():
    session = SimpleNamespace(arch="wormhole_b0", _tree_landing=None)
    with pytest.raises(KomorebiError) as caught:
        IslandRun._far_dram(SimpleNamespace(session=session), "run.all_reduce()")
    assert str(caught.value) == (
        "run.all_reduce(): on wormhole_b0 a collective's pages land in the far "
        "leader's L1, and this load made no landing (no mover of its programs calls a "
        f"collective; {HAZARD}): load a program whose mover calls one"
    )
    session._tree_landing = (2, 4096)
    IslandRun._far_dram(SimpleNamespace(session=session), "run.all_reduce()")
    IslandRun._far_dram(SimpleNamespace(session=SimpleNamespace(arch="blackhole")), "x")
    IslandRun._far_dram(SimpleNamespace(session=None), "x")


def test_a_chains_gather_is_refused_on_wormhole_even_with_the_landing():
    # km_chain.hpp's gather puts into far DRAM itself: not converted
    session = SimpleNamespace(arch="wormhole_b0", _tree_landing=(2, 4096))
    with pytest.raises(KomorebiError) as caught:
        IslandRun._far_dram(
            SimpleNamespace(session=session), "km.all_gather in a chain", landed=False
        )
    assert str(caught.value) == (
        "km.all_gather in a chain writes a copy's pages into the other islands' DRAM: "
        "On Wormhole no link packet targets a DRAM address of the far chip "
        f"({HAZARD})"
    )


@pytest.mark.parametrize(
    "what, port, island, said",
    [
        ("Slot", 2, 7, "a free slot in island 7's leader's landing (port 2)"),
        ("Credits", 0, 1, "the last credits from island 1's leader (port 0)"),
        ("Share", 0xF, 30, "the pages of island 30's share"),
        (
            "Header",
            3,
            4,
            "a slot from island 4 whose header names no page here (port 3)",
        ),
        ("Write", 1, 2, "a slot from island 2 written into DRAM (port 1)"),
        ("None", 0xF, 0, "island 0: this load made no landing for it"),
        (
            "Gen",
            1,
            5,
            "island 5's generation words in its DRAM, written (the slots it dropped) "
            "(port 1)",
        ),
    ],
)
def test_a_landing_wait_given_up_names_what_its_port_and_its_island(
    what, port, island, said
):
    h = abi.headers()
    word = (
        h["kGaveLimit"]
        | h["kWhereLand"] << 8
        | (h[f"kLand{what}"] << 4 | port) << 16
        | island << 24
    )
    gave = state.gave_up(word, 5)
    assert (gave["where"], gave["waited"], gave["island"]) == (
        "land",
        what.lower(),
        island,
    )
    text = errors.describe_core({"core": 3, "reader": gave})
    assert (
        text == f"core 3 reader gave up (limit) in land: a collective's landing: {said}"
    )


def given_up_run(programs, marked, seen=1):
    """An IslandRun of 31 islands whose host has seen `seen` give-ups at a deadline,
    its generation rows marking `marked` (coll_marked's rows); no command can be
    queued (its Runs have no queue)."""
    run = IslandRun.__new__(IslandRun)
    run.runs = (
        SimpleNamespace(_given_up_seen=seen, programs=programs),
        *(SimpleNamespace(_given_up_seen=0, programs=programs) for _ in range(30)),
    )
    run._counts = lambda: None
    run._coll_clear_at, run._coll_movers = 0, None
    run._parts, run._next = [np.zeros(8, np.int64) for _ in run.runs], 1
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
        (0, 3, state.gave_up(deadline, 11), 11, 1),  # its team met
        (30, 9, None, 3, 3),
    ]
    run = given_up_run([program(tmp_path / "a", far=False)], marked)
    with pytest.raises(KomorebiError) as caught:
        run.all_reduce(copy_tensor(), copy_tensor(8))  # refused before any queue
    assert str(caught.value) == (
        "run.all_reduce(): refused before anything was queued: islands 0, 1 and 30: "
        "a collective was given up on this load and the host has not yet cleared it; "
        "the tensors it was reducing are not whole: write them again, then "
        "run.clear_collectives(); the host saw a command given up at its deadline on "
        "this load, whose movers call a collective; the leaders marked, earliest "
        "stage first: island 1's leader (core 7) gave up at its begin, before its team "
        "met (deadline in ask) in its island's command 5 (that island's own count); "
        "island 0's leader (core 3) gave up at its team met, its shares on their way "
        "(deadline in ask) in its island's command 11 (that island's own count); "
        "island 30's leader (core 9) gave up at its shares in, its posts not yet "
        "confirmed (no give-up recorded) in its island's command 3 (that island's own "
        "count)"
    )
    for call in (
        lambda: run.all_gather(copy_tensor()),
        lambda: run.broadcast_from(copy_tensor(), root=1),
        lambda: run.reduce_to(copy_tensor(), copy_tensor(8), root=1),
    ):
        with pytest.raises(KomorebiError, match="refused before anything was queued"):
            call()


def test_the_refusal_says_so_when_no_leader_is_marked(tmp_path):
    run = given_up_run([program(tmp_path / "a", far=False)], [])
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
    calling = [program(tmp_path / "a", far=False)]
    for run in (
        given_up_run(calling, [], seen=0),  # no give-up seen
        given_up_run([program(tmp_path / "b", "#pragma once\n", far=False)], []),
    ):
        run._coll_cleared("run.all_reduce()")
    run = given_up_run(calling, [])
    run._coll_clear_at = 1  # what clear_collectives() keeps: the give-ups it saw
    run._coll_cleared("run.all_reduce()")
    run.runs[1]._given_up_seen = 1  # a give-up after the clear
    with pytest.raises(KomorebiError, match="refused before anything was queued"):
        run._coll_cleared("run.all_reduce()")


def test_the_refusal_names_one_command_by_the_ticket_the_host_gave_it(tmp_path):
    marked = [(0, 3, None, 11, 1), (1, 7, None, 4, 0)]  # each island's own count
    run = given_up_run([program(tmp_path / "a", far=False)], marked)
    run._next = 6
    run._parts[0][5], run._parts[1][5] = 40, 12  # ticket 5: handle 40, and 12
    run.runs[0].sequence = {40: 11}.get
    run.runs[1].sequence = {12: 4}.get
    with pytest.raises(KomorebiError) as caught:
        run.all_gather(copy_tensor())
    assert str(caught.value).endswith(
        "island 0's leader (core 3) gave up at its team met, its shares on their way "
        "(no give-up recorded) in ticket 5; island 1's leader (core 7) gave up at its "
        "begin, before its team met (no give-up recorded) in ticket 5"
    )


def test_the_clear_keeps_the_give_ups_it_saw():
    run = given_up_run([], [])
    run._coll_movers = True  # its movers call a collective: READ and SETTLE built
    done = []

    def quiet(timeout, named=None):  # a probe given up on island 1 meanwhile
        done.append("quiet")
        run.runs[1]._given_up_seen += 1
        run.runs[1]._probe_given_up = 1

    run.drain = lambda timeout: done.append("drain")
    run.session._quiet = quiet
    run.session.raise_generation = lambda: done.append("raise") or 2
    assert run.clear_collectives() == 2
    assert done == ["drain", "quiet", "raise"]  # no line anywhere, then the raise
    assert run._coll_clear_at == 1  # the probe's give-up is not counted
    run._coll_cleared("run.all_reduce()")


def test_a_run_ended_in_the_clear_s_drain_raises_its_own_error():
    run = given_up_run([], [])
    done = []
    ended = KomorebiError("the PAGES command: not sent: core 75 ... load it again")

    def drain(timeout):  # a probe of the drain's crossings read went late
        run.runs[0]._failure = ended
        raise ended

    run.drain = drain
    run.session._quiet = lambda timeout, named=None: done.append("quiet")
    run.session.raise_generation = lambda: done.append("raise") or 2
    with pytest.raises(KomorebiError) as caught:
        run.clear_collectives()
    assert caught.value is ended and done == []


def test_a_clear_on_a_load_calling_no_collective_reads_nothing_and_raises():
    run = given_up_run([], [])  # no mover calls a collective: no READ was built
    done = []
    run.drain = lambda timeout: done.append("drain")
    run.session._quiet = lambda timeout, named=None: pytest.fail("a READ was sent")
    run.session.raise_generation = lambda: done.append("raise") or 2
    assert run.clear_collectives() == 2
    assert done == ["drain", "raise"]


def test_only_a_load_whose_movers_call_a_collective_builds_the_clear_s_read(
    monkeypatch,
):
    calls = {"yes": True, "no": False}
    monkeypatch.setattr(far_collectives, "calls", lambda programs: calls[programs[0]])
    session = SimpleNamespace(links=1)
    assert LoadMixin._coll_quiet(session, ["yes"]) == {"KM_COLL_QUIET": 1}
    assert LoadMixin._coll_quiet(session, ["no"]) == {}
    session.links = 0  # no links: nothing to read
    assert LoadMixin._coll_quiet(session, ["yes"]) == {}
    assert "KM_COLL_QUIET" in kernels.LINK_DEFINES  # the link cores' kernels too


def test_a_clear_whose_links_are_not_quiet_raises_and_does_not_raise_the_generation():
    run = given_up_run([], [])
    run._coll_movers = True
    run.drain = lambda timeout: None

    def not_quiet(timeout, named=None):
        raise KomorebiError("clear_collectives: the links were not quiet")

    run.session._quiet = not_quiet
    run.session.raise_generation = lambda: pytest.fail("the generation was raised")
    with pytest.raises(KomorebiError, match="not quiet"):
        run.clear_collectives()
    assert run._coll_clear_at == 0  # the host's collectives stay refused


def quiet_rows(n=4, rooms=2):
    """A READ of two islands joined by one link core each, every count balanced:
    (read, h, n, rooms, far), the rows' words at their KM_QUIET_* places."""
    h = abi.headers()
    far = -(-(12 + 4 * h["KM_FAR_LINES"]) * n // 64) * 64
    page = h["KM_QUIET_FAR"] + far + -(-(40 * rooms + 4) // 64) * 64
    put, tail = h["KM_QUIET_PUT"] // 4, h["KM_PUT_TAIL"]
    links = {}
    for island in (0, 1):
        words = np.zeros(page // 4, np.uint32)
        words[put] = 9  # numbered, taken by its one page CPU, sent and taken across
        words[put + (h["KM_PUT_TAKEN"] - tail) // 4] = 9
        words[put + (h["KM_PUT_CPUS"] - tail) // 4] = 1
        words[put + (h["KM_PUT_DONE"] - tail) // 4] = 9
        words[put + (h["KM_PUT_CONFIRMED"] - tail) // 4] = 9
        ask = h["KM_QUIET_ASK"] // 4
        words[ask + (h["KM_ASK_TAKEN"] - h["KM_ASK_TAIL"]) // 4] = 3
        words[ask + (h["KM_ASK_SERVED"] - h["KM_ASK_TAIL"]) // 4] = 3
        places = h["KM_QUIET_FAR"] // 4
        words[places + 2] = words[places + n + 2] = 5  # far lines to island 2, done
        forward = (h["KM_QUIET_FAR"] + far) // 4
        words[forward + 4] = words[forward + 8 * rooms + 1] = 7  # room 1 landed, taken
        words[forward + 9 * rooms] = words[forward + 4 * rooms] = 6  # room 0 sent, back
        links[(island, 0)] = (1 - island, 0, 0, words)
    own = np.zeros(32, np.uint32)
    number = (h["KM_OWN_PUT_NUMBER"] - h["KM_OWN_PUT_SENT"]) // 4
    confirmed = (h["KM_OWN_PUT_CONFIRMED"] - h["KM_OWN_PUT_SENT"]) // 4
    own[16 + number] = own[16 + confirmed] = 0x0100_0004  # the writer's, confirmed
    return {"links": links, "ranks": {(0, 5): own}}, h, n, rooms, far


def test_a_quiet_graph_balances_and_each_rule_fails_while_a_line_is_held():
    read, h, n, rooms, far = quiet_rows()
    assert link_plan.quiet_faults(read, h, n, rooms, far) == []
    put, tail = h["KM_QUIET_PUT"] // 4, h["KM_PUT_TAIL"]
    ask, places = h["KM_QUIET_ASK"] // 4, h["KM_QUIET_FAR"] // 4
    forward = (h["KM_QUIET_FAR"] + far) // 4
    ring = h["KM_QUIET_RING"] // 4
    held = {  # a word as a line held there leaves it, and what the clear says
        put: "island 0's link core 0 (toward island 1): put queue lines numbered 10, "
        "taken by its page CPUs [9]",
        put + (h["KM_PUT_DONE"] - tail) // 4: "put lines sent across 10, taken "
        "across 9",
        h["KM_QUIET_QUEUE"] // 4 + h["KM_QUEUE_PENDING_COUNT"]: "1 lines taken, not "
        "yet done",
        ask + (h["KM_ASK_TAKEN"] - h["KM_ASK_TAIL"]) // 4: "gets sent across 4, "
        "served back [3]",
        ring + h["KM_PUT_RING_TAIL"] // 4: "CPU 0's ring words written 1, sent 0",
        places + 2: "far put lines to island 2 sent 6, done there 5",
        places + 3 * n: "far places hold 1 lines with pages counted",
        forward + 9 * rooms: "forward room 0's packets sent on 7, taken there 6",
        forward + 4: "forward room 1's packets landed 8, passed on 7",
    }
    for at, said in held.items():
        read, *_ = quiet_rows()
        read["links"][(0, 0)][-1][at] += 1
        faults = link_plan.quiet_faults(read, h, n, rooms, far)
        assert len(faults) == 1 and said in faults[0], (at, faults)
    read, *_ = quiet_rows()
    read["ranks"][(0, 5)][16 + 1] += 1  # the writer's line numbered, not confirmed
    assert link_plan.quiet_faults(read, h, n, rooms, far) == [
        "island 0's core 5: CPU 1's put lines (parity 0) numbered 0x1000005, "
        "confirmed 0x1000004"
    ]
    read, *_ = quiet_rows()  # a refused line is confirmed (KM_OWN_REFUSED)
    read["ranks"][(0, 5)][16 + 3] |= h["KM_OWN_REFUSED"]
    assert link_plan.quiet_faults(read, h, n, rooms, far) == []


def quiet_session(reads):
    """A session whose _quiet_read hands `reads` in turn (the last again), and whose
    ENTRY commands are kept."""
    _read, _h, n, rooms, far = quiet_rows()
    sent, given = [], iter(reads)
    last = []

    def next_read(timeout):
        last[:] = [next(given, last[0] if last else None)]
        return last[0]

    return (
        SimpleNamespace(
            _quiet_page=lambda: (0, n, rooms, far),
            _quiet_read=next_read,
            graph=SimpleNamespace(islands=[0, 1]),
            run=SimpleNamespace(_served=lambda *args: sent.append(args)),
        ),
        sent,
    )


def test_the_clear_settles_after_two_reads_in_a_row_the_same_and_balanced():
    h = abi.headers()
    moving, *_ = quiet_rows()
    moving["links"][(0, 0)][-1][h["KM_QUIET_PUT"] // 4] += 1  # a line held
    quiet, *_ = quiet_rows()
    session, sent = quiet_session([moving, quiet, quiet])
    link_plan.LinkPlanMixin._quiet(session, timeout=5.0)
    settle = [h["KM_ENTRY_QUIET"], h["KM_QUIET_SETTLE"]]
    assert sent == [(h["KM_HANDLER_ENTRY"], [settle, settle], 5.0)]


def test_a_line_held_past_the_limit_fails_the_clear_by_name_without_a_settle():
    h = abi.headers()
    moving, *_ = quiet_rows()
    moving["links"][(1, 0)][-1][h["KM_QUIET_PUT"] // 4] += 1
    session, sent = quiet_session([moving])
    with pytest.raises(KomorebiError) as caught:
        link_plan.LinkPlanMixin._quiet(session, timeout=0.05)
    assert str(caught.value) == (
        "clear_collectives: the links were not quiet within 0.05 s, so no rank was "
        "settled and the generation was not raised: island 1's link core 0 (toward "
        "island 0): put queue lines numbered 10, taken by its page CPUs [9]; a line "
        "of a given-up command is still on its way or lost: this load cannot be made "
        "safe: load it again"
    )
    assert sent == []


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


def test_a_read_refused_while_a_core_settles_fails_the_clear_with_the_run_alive():
    session, sent = quiet_session([])

    def refused(timeout):
        raise errors._CoresNotFree(
            "island 1: the ENTRY command: refused before it was sent: rank 78"
        )

    session._quiet_read = refused
    with pytest.raises(KomorebiError) as caught:
        link_plan.LinkPlanMixin._quiet(session, timeout=1.0)
    assert str(caught.value) == (
        "clear_collectives: a core still waits for a line of the given-up command "
        "after 1 s, so no rank was settled and the generation was not raised; the "
        "run goes on: call it again once the line passes (a core waits up to the "
        "load's wait limit for its own lines): island 1: the ENTRY command: refused "
        "before it was sent: rank 78"
    )
    assert sent == []  # no SETTLE
