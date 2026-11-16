"""A drain does not raise for a command given up; drain_given_up counts them, so a
script sees it without reading each event (fake dispatcher)."""

from types import SimpleNamespace

from fake_port import FakeBackend, FakePort

from komorebi.islands import session as island_session
from komorebi.session.run import Run

GRID = (12, 10)


def test_a_drain_counts_the_commands_given_up_and_the_next_drain_starts_at_0():
    port = FakePort()
    run = Run(FakeBackend(port), GRID)
    assert run.drain_given_up == 0
    handles = [run.submit(1, (i, 1)) for i in range(3)]
    port.give_up_req = handles[1]
    assert run.drain() == 3
    assert run.drain_given_up == 1 and run.given_up(handles[1]) is not None
    assert run.event(handles[0]).kind == "done"
    run.submit(1, (0, 1))
    assert run.drain() == 1 and run.drain_given_up == 0


def test_an_island_run_s_count_is_every_island_s_parts():
    runs = [SimpleNamespace(drain_given_up=n) for n in (0, 2)]
    count = island_session.IslandRun.drain_given_up.fget(SimpleNamespace(runs=runs))
    assert count == 2


def test_an_island_run_names_a_ticket_s_first_given_up_part():
    def run_of(error):
        return SimpleNamespace(given_up=lambda handle: error)

    island_run = SimpleNamespace(
        runs=[run_of(None), run_of("stuck")],
        _parts=[[0, 4], [0, 9]],
        _error=lambda error, island: f"island {island}: {error}",
    )
    given_up = island_session.IslandRun.given_up
    assert given_up(island_run, 1) == "island 1: stuck"
    island_run.runs[1] = run_of(None)
    assert given_up(island_run, 1) is None
