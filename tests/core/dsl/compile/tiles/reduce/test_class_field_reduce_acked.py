"""A field of sums is emptied by unpack on the group's root and by mover B's send
elsewhere; unpack keeps its own count, so a sender adds mover B's pop to it after the
push, as the hand's K split does. Without it, groups of two sizes rewound the count:
wrong sums, then a hang (no card)."""

from chain_fixture import quiet, tensor
from test_class_reduce_arrival import Large, Small, tiles

ACKED = "UNPACK((get_local_cb_interface(kFieldSums).tiles_acked += kFieldSumsPages));"


def handler_cases(cls):
    program = made(cls)
    text = (program.built.folder / "compute.cpp").read_text()
    cases = text.split("case kHandler")[1:]
    return {case.split(":")[0]: case for case in cases}


def made(cls):
    return cls(
        l1=1 << 20,
        log=quiet,
        x=tiles(1, 30, 0x10000),
        w=tiles(30, 8, 0x20000),
        y=tiles(1, 8, 0x90000),
        s=tensor(120, 0xA0000),
        t=tensor(1, 0x120000),
    )


def test_each_field_reduce_counts_the_send_on_unpack():
    cases = handler_cases(Small)
    for name in ("Two", "Fifteen"):  # a field of sums, groups of 2 and of 15
        case = cases[name]
        guard = "if (km_ok && !km::reduce_send(km::digit_group(command, compute.me("
        assert guard in case and ACKED in case
        assert case.index("cb_push_back(kFieldSums, kFieldSumsPages);") < case.index(
            ACKED
        )
    assert ACKED not in cases["Seven"]  # a reduce of one tile: mover B alone takes it


def test_whichever_order_the_handlers_come_in():
    cases = handler_cases(Large)
    assert all(ACKED in cases[name] for name in ("Two", "Fifteen", "First"))
