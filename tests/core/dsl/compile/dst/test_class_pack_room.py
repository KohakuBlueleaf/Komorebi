"""Compute's reserve is decided by pack alone (true on unpack and math), so in a class's
generated compute its result guards only the pack lines; the sections, their hand-over
and the products run on all three threads. A pack's give-up then ends with the others,
not with unpack and math held inside the unit (no card)."""

from types import SimpleNamespace

from chain_fixture import quiet, tensor
from class_pack_room_fixture import check
from class_reduce_arrival_fixture import Small, tiles

from komorebi.session.run import Run


def compute_text(cls, **tensors):
    program = cls(l1=1 << 20, log=quiet, **tensors)
    return (program.built.folder / "compute.cpp").read_text()


def test_a_field_reduce_packs_into_its_sums_in_l1():
    text = compute_text(
        Small,
        x=tiles(1, 30, 0x10000),
        w=tiles(30, 8, 0x20000),
        y=tiles(1, 8, 0x90000),
        s=tensor(120, 0xA0000),
        t=tensor(1, 0x120000),
    )
    seen = check(text)
    # L1 accumulation into a held sum: a plain pack passes today's shapes too
    assert any(line.startswith("pack_tile<true>") for line in seen)
    assert any(line.startswith("pack_reconfig_l1_acc") for line in seen)


def test_a_timeout_says_what_the_deadline_read_before_the_stop():
    core = {"core": 3, "live": {"math": {"where": "the destination", "sequence": 9}}}
    idle = {"core": 4, "live": {}}
    run = SimpleNamespace(_timeout_cores=[core, idle], _reset_failed=(-1, 0))
    nothing = SimpleNamespace(_timeout_cores=[], _reset_failed=(-1, 0))
    seen = Run._timeout_seen(run)
    assert seen == (
        "at the deadline: core 3 math waits in the destination; "
        "1 cores read at it named no wait"
    )
    assert Run._timeout_seen(nothing) == ""
