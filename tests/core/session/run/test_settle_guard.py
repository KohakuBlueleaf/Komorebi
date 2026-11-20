"""After a give-up the host has seen, a core may still be settling it, and a runtime
command it cannot take ends the run at its deadline: such a command waits first for a
NOP probe to complete on the cores it addresses, or is refused by name with the run
alive (fake dispatcher)."""

import time
from types import SimpleNamespace

import pytest
from fake_port import CONSTANTS, FakeBackend, FakePort

from komorebi import errors
from komorebi.errors import KomorebiError, _CoresNotFree
from komorebi.islands.run import IslandRun
from komorebi.session import state
from komorebi.session.run import Run, _still_waiting

GRID = (12, 10)
PUSH, NOP = CONSTANTS["KM_HANDLER_PUSH"], CONSTANTS["KM_HANDLER_NOP"]


class Reqs(set):
    """FakePort's give_up_req as a set: every command number in it is given up."""

    def __eq__(self, req):
        return isinstance(req, int) and set.__contains__(self, req)

    __hash__ = None


def guarded_run(given_up=()):
    """A run whose PUSH completes at once, its port giving up the commands numbered
    in `given_up` (a set the test may grow); the handlers the port took, in order."""
    port = FakePort()
    run = Run(FakeBackend(port), GRID)
    run._deadline_s = 0.05
    taken = []
    port.give_up_req = Reqs(given_up)
    done = CONSTANTS["KM_EVENT_DONE"]

    def push(req, words):
        taken.append(PUSH)
        port._event(req, done)

    port.handlers[PUSH] = push
    return run, port, taken


class Until:
    """FakePort's give_up_req: every command is given up until `until`
    (time.monotonic())."""

    def __init__(self, until):
        self.until = until

    def __eq__(self, req):
        return isinstance(req, int) and time.monotonic() < self.until

    __hash__ = None


def handlers_sent(port, after=0):
    return [port.ring[req][1] & 0xFF for req in sorted(port.ring) if req > after]


def test_no_probe_while_no_give_up_was_seen():
    run, port, _ = guarded_run()
    run.transfer(PUSH, (0, 2, 0), [], 1.0)
    assert handlers_sent(port) == [PUSH]


def test_after_a_give_up_a_runtime_command_waits_for_a_probe_once_per_group():
    run, port, taken = guarded_run({1})
    given = run.submit(1, (0, 2))
    with pytest.raises(KomorebiError, match="given up"):
        run.wait(given)
    assert run._unsure
    run.transfer(PUSH, (0, 2, 0), [], 1.0)
    run.transfer(PUSH, (0, 2, 0), [], 1.0)  # known free: no probe again
    run.transfer(PUSH, (4, 1, 0), [], 1.0)  # another group: its own probe
    run.transfer(PUSH, (1, 2, 0), [], 1.0)  # overlapping a known-free one: probed
    assert handlers_sent(port, after=given) == [NOP, PUSH, PUSH, NOP, PUSH, NOP, PUSH]
    assert taken == [PUSH] * 4
    assert run._given_up_seen == 1 and run._probe_given_up == 0


def test_a_probe_given_up_is_sent_again_and_is_not_the_user_s():
    run, port, _ = guarded_run({1, 2, 3})  # the command, then two probes
    given = run.submit(1, (0, 2))
    with pytest.raises(KomorebiError, match="given up"):
        run.wait(given)
    run._deadline_s = 5.0
    run.transfer(PUSH, (0, 2, 0), [], 5.0)
    assert handlers_sent(port, after=given) == [NOP, NOP, NOP, PUSH]
    assert run._given_up_seen == 3 and run._probe_given_up == 2
    assert list(run._given_up) == [given]  # the user's alone: the probes forgotten
    assert run._unsure  # still: only the group probed is known free


def test_cores_not_free_within_the_time_refuse_the_command_by_name_the_run_alive():
    run, port, taken = guarded_run(set(range(1, 10**4)))  # the command, every probe
    given = run.submit(1, (0, 2))
    with pytest.raises(KomorebiError, match="given up"):
        run.wait(given)
    with pytest.raises(_CoresNotFree) as caught:
        run.transfer(PUSH, (0, 2, 0), [], 0.05)
    assert str(caught.value).startswith(
        "the PUSH command: refused before it was sent: a core still waits for a "
        "line of a given-up command after 0.05 s, and a runtime command it cannot "
        "take ends the run at its deadline; the run goes on: "
    )
    assert PUSH not in handlers_sent(port) and taken == []
    assert not run.stopped and run._unsure
    port.give_up_req = Reqs()  # the line passed: the same command goes now
    run.transfer(PUSH, (0, 2, 0), [], 1.0)
    assert taken == [PUSH]


def test_the_dispatcher_s_own_commands_and_program_commands_pass_unprobed():
    run, port, _ = guarded_run({1})
    given = run.submit(1, (0, 2))
    with pytest.raises(KomorebiError, match="given up"):
        run.wait(given)
    run.wait(run.submit(1, (0, 2)))
    run.wait(run._queue_command(CONSTANTS["KM_HANDLER_PING"], None))
    inspect = CONSTANTS["KMD_HANDLER_INSPECT"]
    run.wait(run._queue_command(inspect, (0, 2), flags=run._flag_dispatcher))
    assert NOP not in handlers_sent(port, after=given)


def probing_run(errors):
    """A Run whose probes' waits raise `errors` in turn (then complete), its log
    kept in `said`."""
    run = Run.__new__(Run)
    run._known_free, run._deadline_s, run._budget_cap = set(), 5.0, None
    run._budget_given = None
    run._probing, run._probe_given_up, run._cluster_base = False, 0, 0xFF00
    run._nop, run._name_cores, said = NOP, None, []
    run.recovery_refused = "a card split into islands"
    run._log = said.append
    run._queue_command = lambda *args, **kwargs: 7
    run.forget = lambda handles: None
    pending = list(errors)

    def wait(handle, timeout):
        if pending:
            raise pending.pop(0)

    run.wait = wait
    return run, said


LIMIT = {"code": "limit", "where": "clear", "slot": 0, "sequence": 9}


def test_a_probe_s_give_up_that_reset_a_core_says_so_in_the_log():
    cores = [{"core": 0, "reader": LIMIT}]
    gave = errors.CommandTimeout(7, 0, 9, (0, 2), cores, given_up=True, recovered=[0])
    run, said = probing_run([gave])
    Run._free_for(run, PUSH, (0, 2), 0, None)
    assert said == [
        "the PUSH command: before it, cores still in a given-up command whose own "
        "wait ended were reset and loaded again: core 0 reader gave up (limit) in clear"
    ]
    assert (0, 2, 0) in run._known_free


def test_a_core_ended_and_not_reset_stops_the_probe_at_once_by_name():
    cores = [{"core": 36, "reader": LIMIT}, {"core": 37}]
    gave = errors.CommandTimeout(7, 0, 9, (36, 2), cores, given_up=True)
    run, said = probing_run([gave, gave, gave])  # every probe given up
    with pytest.raises(KomorebiError) as caught:
        Run._free_for(run, PUSH, (36, 2), 0, 30.0)
    assert str(caught.value).startswith(
        "the PUSH command: not sent: core 36 reader gave up (limit) in clear: ended "
        "at its own wait limit while it settled a given-up command, and nothing "
        "resets it (no reset list: a card split into islands); it takes no command "
        "again, so the run cannot go on with it: load it again; the dispatcher's "
        "report: command 7 was given up"
    )
    assert not isinstance(caught.value, _CoresNotFree)  # never "call it again"
    assert said == [] and (36, 2, 0) not in run._known_free


def test_a_late_probe_names_the_core_that_ended_at_its_own_limit():
    limit = {"code": "limit", "where": "pages", "slot": 0, "sequence": 9}
    late = errors.CommandTimeout(
        7, 0, 9, (CONSTANTS["KM_GROUP_LINK_BASE"], 4), [{"core": 75, "reader": limit}]
    )
    run, _ = probing_run([late])
    with pytest.raises(KomorebiError) as caught:
        Run._free_for(run, PUSH, (CONSTANTS["KM_GROUP_LINK_BASE"], 4), 0, None)
    assert str(caught.value).startswith(
        "the PUSH command: not sent: core 75 reader gave up (limit) in pages: ended "
        "at its own wait limit while it settled a given-up command, and the "
        "dispatcher could not reset it; the run has ended by that core's rule: load "
        "it again; the dispatcher's report: command 7 timed out"
    )


def test_a_refusal_prints_its_budget_short():
    run, port, _ = guarded_run(set(range(1, 10**4)))
    given = run.submit(1, (0, 2))
    with pytest.raises(KomorebiError, match="given up"):
        run.wait(given)
    with pytest.raises(_CoresNotFree, match=r"after 0\.05 s, and"):
        run.transfer(PUSH, (0, 2, 0), [], 0.05 + 3e-7)


def test_a_give_up_in_the_links_service_names_its_wait():
    word = CONSTANTS["kGaveLimit"] | CONSTANTS["kWherePages"] << 8
    assert state.gave_up(word, 9)["where"] == "pages"


def test_one_island_s_give_up_marks_every_island_and_probes_count_apart():
    runs = [
        SimpleNamespace(
            _unsure=False, _known_free={(0, 1, 0)}, _given_up_seen=s, _probe_given_up=p
        )
        for s, p in ((3, 2), (1, 0))
    ]
    island_run = SimpleNamespace(runs=runs)
    IslandRun._settling(island_run)
    assert all(run._unsure and not run._known_free for run in runs)
    assert IslandRun._given_ups(island_run) == 2  # the user's: 1 and 1


def test_the_refusal_does_not_name_leaders_it_could_not_read():
    def not_free():
        raise _CoresNotFree("the PUSH command: refused before it was sent")

    runs = [
        SimpleNamespace(_given_up_seen=s, _probe_given_up=0, _deadline_s=0.5)
        for s in (0, 1)
    ]
    island_run = IslandRun.__new__(IslandRun)
    island_run.runs = runs
    island_run._coll_clear_at, island_run._coll_movers = 0, ["a mover"]
    island_run.session = SimpleNamespace(coll_marked=not_free)
    with pytest.raises(KomorebiError) as caught:
        island_run._coll_cleared("run.all_reduce()")
    assert str(caught.value).endswith(
        "the host saw a command given up at its deadline on this load, whose movers "
        "call a collective; the leaders were not read: a core is still settling"
    )
    assert str(caught.value).startswith(
        "run.all_reduce(): refused before anything was queued: island 1: "
    )
    assert all(run._budget_cap is None for run in runs)


def test_a_probe_of_link_cores_given_up_has_no_far_part():
    island_run = SimpleNamespace(
        graph=SimpleNamespace(split=lambda group: pytest.fail("a link group cut"))
    )
    link_group = (CONSTANTS["KM_GROUP_LINK_BASE"], 4)
    assert IslandRun._far_cores(island_run, 0, link_group) == []


def test_a_drain_s_crossings_read_finds_cores_free_within_the_drain_s_time():
    caps = []
    runs = [
        SimpleNamespace(drain=lambda timeout: 0, _budget_cap=None) for _ in range(2)
    ]
    island_run = IslandRun.__new__(IslandRun)
    island_run.runs, island_run._measured = runs, {}
    island_run.send = lambda: None
    island_run._crossed = lambda tickets: caps.append([r._budget_cap for r in runs])
    island_run._drain(5.0)
    assert len(caps) == 1 and all(0 < cap <= 5.0 for cap in caps[0])
    assert all(run._budget_cap is None for run in runs)


def test_a_refusal_names_the_cores_still_waiting_before_the_idle_ones():
    idle = {
        "core": 1,
        "live": {"reader": {"where": "next"}, "writer": {"where": "writer"}},
    }
    settling = {"core": 0, "live": {"reader": {"where": "clear"}}}
    assert _still_waiting(settling) and not _still_waiting(idle)
    assert not _still_waiting({"core": 2})


def test_a_caller_s_limit_given_replaces_each_command_s_own_time():
    run, port, taken = guarded_run()
    port.give_up_req = Until(time.monotonic() + 0.3)  # the command, then the probes
    given = run.submit(1, (0, 2))
    with pytest.raises(KomorebiError, match="given up"):
        run.wait(given)
    with pytest.raises(_CoresNotFree, match=r"after 0\.05 s, and"):
        run.transfer(PUSH, (0, 2, 0), [], 0.05)  # its own time: too short
    run._budget_given = time.monotonic() + 5.0  # a clear's limit, given
    run.transfer(PUSH, (0, 2, 0), [], 0.05)
    assert taken == [PUSH] and not run.stopped


def test_a_clear_gives_its_limit_to_its_drain_and_its_raise():
    seen = []
    runs = [SimpleNamespace(_budget_given=None) for _ in range(2)]

    def said(*args):
        seen.append([run._budget_given for run in runs])
        return 1

    island_run = IslandRun.__new__(IslandRun)
    island_run.runs, island_run.drain = runs, said
    island_run.session = SimpleNamespace(raise_generation=said)
    island_run._coll_movers, island_run._counts = [], None  # no READ: no links
    started = time.monotonic()
    assert island_run.clear_collectives(30.0) == 1
    assert len(seen) == 2  # the drain, then the raise
    assert all(started + 30.0 <= at <= time.monotonic() + 30.0 for at in sum(seen, []))
    assert all(run._budget_given is None for run in runs)
