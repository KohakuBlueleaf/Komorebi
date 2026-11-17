"""The handlers' reduces share one arrival buffer (kArrival0): it is sized for the
largest of them, a slot of the most tiles a reduce sends and as many slots as the most
rounds, whichever handler comes first (no card)."""

from types import SimpleNamespace

from chain_fixture import quiet, tensor
from class_reduce_arrival_fixture import Large, Small, tiles

import komorebi as km
from komorebi.dsl.compile.tiles import reduce
from komorebi.dsl.islands import island_reduce


class Apart(km.Program):
    """Small's tree of 2 (8 tiles, one round) and its tree of 7 (one tile, three
    rounds): no reduce takes 3 slots of 8 tiles."""

    sums: km.L1[km.fp32, 1, 8, 32, 32]
    two = Small.two
    seven = Small.seven
    project = Small.project


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


def arrival(cls, name="kArrival0"):
    program = made(cls)
    header = program.built.folder / f"{cls.__name__.lower()}.h"
    line = next(
        line for line in header.read_text().splitlines() if f"Arrival {name} " in line
    )
    return tuple(int(word) for word in line.split("{")[1].split("}")[0].split(","))


def test_a_shared_arrival_is_sized_for_the_largest_reduce():
    index, pages, per = arrival(Small)
    assert per == 8  # a slot of the most tiles a reduce sends (the sums' 8, not 1)
    assert pages == 4 * 8  # the most rounds: the tree of 15's four


def test_whichever_handler_comes_first():
    assert arrival(Large)[1:] == arrival(Small)[1:] == (32, 8)


def test_a_tree_over_the_commands_group_has_a_slot_each_island_part_more():
    # over islands: the root part's rounds and a slot each other part (a line of 3
    # islands of 47, 34, 35 ranks: 6 + 2 = 8, past the flat tree's 7: given up)
    def slots(islands, size=None):
        made = SimpleNamespace(compiled=SimpleNamespace(islands=islands))
        return reduce.tree_slots(made, size)

    assert slots((1, 0)) == reduce.TREE_ROUNDS == 7  # one island: as before
    assert slots((3, 47)) == 8 and slots((2, 118)) == 8
    assert slots((2, 40)) == 7  # 6 + 1: the flat tree's 7 still the most
    assert slots((1, 0), size=15) == 4  # group=S on one island: its own rounds


def test_a_tree_in_groups_of_s_over_islands_has_a_slot_each_island_it_can_touch_more():
    """group=S on islands: the largest part's rounds and a slot each further island
    the group can touch (km::reduce_slots), as over the command's group; a group of 7
    split 5 | 2 at an island's edge needs 3 + 1, past one island's 3."""

    def slots(islands, size):
        made = SimpleNamespace(compiled=SimpleNamespace(islands=islands))
        return reduce.tree_slots(made, size)

    assert slots((1, 0), 7) == 3
    assert slots((3, 47), 7) == 3 + 2 and slots((2, 59), 7) == 3 + 1
    assert slots((4, 29), 64) == 5 + 3  # larger than an island: parts of 29 at most
    assert slots((4, 29), 2) == 1  # split 1 | 1: no round and a slot, as one island


def device_slots(sizes, base, size):
    """km::reduce_slots (km_group.hpp) of the group of `size` ranks from flat rank
    `base` on islands of `sizes` ranks: its own rounds on one island, else the root
    part's rounds (its ranks on the first island) and a slot each further part."""
    edges = [sum(sizes[:i]) for i in range(len(sizes) + 1)]
    low = next(i for i in range(len(sizes)) if base < edges[i + 1])
    high = next(i for i in range(len(sizes)) if base + size - 1 < edges[i + 1])
    if low == high:
        return island_reduce.rounds(size)
    return island_reduce.rounds(edges[low + 1] - base) + high - low


def test_the_slots_are_the_most_any_group_takes_wherever_the_command_puts_it():
    """group_slots against km::reduce_slots for every S and every first rank: never
    fewer; on two islands exactly the most (max(rounds(S), rounds(S - 1) + 1))."""
    for sizes in ([118, 118], [59, 59], [47, 34, 35], [29, 29, 29, 29]):
        total, islands = sum(sizes), (len(sizes), max(sizes))
        for size in range(1, total + 1):
            most = max(
                device_slots(sizes, base, size) for base in range(total - size + 1)
            )
            built = island_reduce.group_slots(size, islands)
            assert built >= max(1, most), (sizes, size)
            if len(sizes) == 2:
                assert built == max(1, most), (sizes, size)
    assert [island_reduce.group_slots(s, (2, 118)) for s in (2, 3, 9, 10)] == [
        1,
        2,
        4,
        5,
    ]


def test_each_reduce_takes_its_own_slots_of_the_arrival_sized_for_the_largest():
    assert arrival(Apart)[1:] == (8, 8)  # 1 x 8 and 3 x 1: not 3 x 8
    assert arrival(Apart, "kArrival0Per1")[1:] == (8, 1)  # seven's: 8 slots of 1
    movement = (made(Apart).built.folder / "movement.cpp").read_text()
    assert "km::reduce_arrive(movement, command, kArrival0Per1," in movement
    assert "km::reduce_arrive(movement, command, kArrival0," in movement


def test_a_class_whose_reduces_share_one_part_size_has_no_other_constant():
    class Same(km.Program):
        sums: km.L1[km.fp32, 1, 8, 32, 32]
        two, fifteen, project = Small.two, Small.fifteen, Small.project

    assert arrival(Same)[1:] == (32, 8)  # four rounds of 8, as before
    assert "kArrival0Per" not in (made(Same).built.folder / "same.h").read_text()
