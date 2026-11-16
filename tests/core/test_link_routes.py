"""The link service's routes on the host: each entry's ports toward an island, the
graph's diameter, the 2 x 2 grid with wrap, graphs given as island pairs, the link
words a pair and the graphs the session refuses by name."""

import pytest

from komorebi import islands
from komorebi.island_session import (
    IslandSession,
    _split_graph,
    cards_graph,
    check_wired,
    link_slots,
    link_tail,
)


def cores(parts, size=30, columns=10):
    return [
        islands.Island(tuple(range(i * size, (i + 1) * size)), columns)
        for i in range(parts)
    ]


def test_two_islands_reach_each_other_by_their_one_port():
    graph = islands.graph_of(cores(2), islands.line_pairs(2, links=4))
    table = islands.read_table(graph.table(0), ports=True)
    assert table[0][2:] == (islands.NO_LINK, 0, 0)
    assert table[1][2:] == (0, 1, 0b1)
    assert graph.diameter() == 1


def test_a_ring_of_four_takes_both_ports_to_the_far_island():
    graph = islands.graph_of(cores(4), islands.ring_pairs(4))
    assert graph.neighbours(0) == (1, 3)
    assert graph.ports_toward(0, 2) == 0b11  # both ways round are shortest
    assert graph.ports_toward(0, 1) == 0b01
    assert graph.ports_toward(0, 3) == 0b10
    assert graph.diameter() == 2


def test_a_line_of_four_has_one_port_and_diameter_three():
    graph = islands.graph_of(cores(4), islands.line_pairs(4))
    assert graph.ports_toward(0, 3) == 0b1
    assert graph.ports_toward(1, 3) == 0b10  # neighbours of 1: (0, 2)
    assert graph.diameter() == 3


def test_the_two_by_two_grid_with_wrap_doubles_each_pair():
    graph = islands.graph_of(cores(4), islands.grid_pairs(2, 2))
    assert {(link.a, link.b, link.count) for link in graph.links} == {
        (0, 1, 2),
        (0, 2, 2),
        (1, 3, 2),
        (2, 3, 2),
    }
    assert graph.ports_toward(0, 3) == 0b11  # through 1 or through 2
    assert graph.diameter() == 2
    assert len(graph.ports[0]) == 4  # four links: north, south, east, west


def test_a_larger_torus_routes_by_both_dimensions():
    graph = islands.graph_of(cores(32, size=8, columns=4), islands.grid_pairs(4, 8))
    assert graph.diameter() == 2 + 4
    assert all(len(graph.neighbours(i)) == 4 for i in range(32))


def test_a_split_card_wires_the_pairs_given_a_pair_k_times_k_wires():
    def pairs(given, parts=4):
        return {
            (link.a, link.b, link.count)
            for link in _split_graph(10, 12, parts, given).links
        }

    assert pairs(islands.line_pairs(4)) == {(0, 1, 1), (1, 2, 1), (2, 3, 1)}
    assert islands.parse_pairs("0-1,1-2,1-2") == [(0, 1), (1, 2), (1, 2)]
    assert pairs(islands.ring_pairs(3), 3) == {(0, 1, 1), (1, 2, 1), (0, 2, 1)}
    assert pairs([(0, 1), (1, 0), (2, 1), (3, 2)]) == {(0, 1, 2), (1, 2, 1), (2, 3, 1)}
    assert pairs(islands.grid_pairs(2, 2)) == {
        (0, 1, 2),
        (1, 3, 2),
        (2, 3, 2),
        (0, 2, 2),
    }
    star = _split_graph(10, 12, 4, [(0, 1), (0, 2), (0, 3)])  # three neighbours of 0
    assert len(star.link_cores[0]) == 3 and star.diameter() == 2
    # the next link names a link core, two links a core: 0 toward 1, 1 toward 2
    grid = _split_graph(10, 12, 4, islands.grid_pairs(2, 2))
    assert [islands.read_table(grid.table(0))[i][2] for i in (1, 2)] == [0, 1]
    # three wires take two link cores toward that neighbour, one toward the next
    odd = _split_graph(10, 12, 3, [(0, 1)] * 3 + [(0, 2)])
    assert odd.link_cores_toward(0, 1) == (0, 2) and odd.link_cores_toward(0, 2) == (
        2,
        1,
    )
    assert [islands.read_table(odd.table(0))[i][2] for i in (1, 2)] == [0, 2]
    with pytest.raises(ValueError, match=r"pair \(1, 1\): two different islands of 4"):
        _split_graph(10, 12, 4, [(0, 1), (1, 1)])
    with pytest.raises(ValueError, match="island 2 has no path to island 0"):
        _split_graph(10, 12, 4, [(0, 1), (2, 3)])


def test_a_pair_s_links_take_two_slots_a_link_core_and_more_than_four_are_refused():
    odd = _split_graph(10, 12, 3, [(0, 1)] * 3 + [(0, 2)])
    none = 0xFF
    assert link_slots(odd, 0, lambda there, k: 10 * there + k) == (
        10,
        11,
        12,
        none,
        20,
        none,
    )
    check_wired(_split_graph(10, 12, 3, [(0, 1)] * 4 + [(1, 2)] * 3), 3)
    with pytest.raises(ValueError, match="islands 0 and 1: 5 links, at most 4 a pair"):
        check_wired(_split_graph(10, 12, 3, [(0, 1)] * 5 + [(1, 2)]), 3)


def test_more_neighbours_than_ports_or_islands_than_a_byte_are_refused():
    star = islands.Graph(cores(6, size=10), [islands.Link(0, i) for i in range(1, 6)])
    with pytest.raises(ValueError, match="island 0: 5 neighbours, a card's ports 4"):
        check_wired(star, 1)
    check_wired(islands.graph_of(cores(32, 4, 2), islands.ring_pairs(32)), 1)
    with pytest.raises(ValueError, match="257 islands: 1 to ISLANDS_MAX 256"):
        islands.graph_of(cores(257, size=2, columns=2), islands.line_pairs(257))


def test_a_load_of_more_than_16_islands_defines_its_arrays_and_message_tail():
    assert link_tail(32) - link_tail(4) == 16 * 16
    assert link_tail(2) == link_tail(16)


def test_cards_wire_the_pairs_given_and_graph_needs_cards_or_a_split():
    assert IslandSession([0, 1, 2, 3]).links == 1  # the fabric's count at the open
    graph = cards_graph([(12, 10)] * 4, islands.line_pairs(4, links=2))
    assert [len(cores) for cores in graph.link_cores] == [1, 2, 2, 1]
    assert graph.link_cores[1] == (118, 119) and graph.diameter() == 3
    assert [islands.read_table(graph.table(1))[i][2] for i in (0, 2, 3)] == [0, 1, 1]
    assert graph.ranks == 4 * 120 - 6
    ring = cards_graph([(12, 10)] * 4, islands.ring_pairs(4))  # a link a pair
    assert [len(cores) for cores in ring.link_cores] == [2] * 4 and ring.diameter() == 2
    with pytest.raises(ValueError, match="graph=: cards, or a card split into islands"):
        IslandSession([0], graph=[(0, 1)])
    with pytest.raises(ValueError, match="links=2 with graph=: a pair k times"):
        IslandSession([0, 1], links=2, graph=[(0, 1)])
    assert IslandSession([0], islands=4, graph=islands.ring_pairs(4)).links == 1


@pytest.mark.parametrize(
    "form,count",
    [("ring", n) for n in (3, 4, 5, 8, 16, 32)]
    + [("grid", rc) for rc in ((2, 2), (2, 4), (4, 4), (4, 8))],
)
def test_routes_and_masks_hold_on_rings_and_grids_up_to_32_islands(form, count):
    if form == "ring":
        n = count
        graph = islands.graph_of(cores(n, size=8, columns=4), islands.ring_pairs(n))
        diameter = count // 2
    else:
        rows, columns = count
        n = rows * columns
        graph = islands.graph_of(
            cores(n, size=8, columns=4), islands.grid_pairs(rows, columns)
        )
        diameter = rows // 2 + columns // 2
    assert graph.diameter() == diameter
    for source in range(n):
        assert len(graph.neighbours(source)) <= islands.PORTS_MAX
        for target in range(n):
            hops, mask = graph.hops(source, target), graph.ports_toward(source, target)
            if source == target:
                assert (hops, mask) == (0, 0)
                continue
            near = graph.neighbours(source)
            ways = [near[p] for p in range(len(near)) if mask >> p & 1]
            assert ways and all(graph.hops(w, target) == hops - 1 for w in ways)
            assert graph.next_island(source, target) in ways
