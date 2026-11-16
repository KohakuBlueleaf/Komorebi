"""The link service's routes on the host: each entry's ports toward an island, the
graph's diameter, the 2 x 2 grid with wrap, a split card's shapes and the graphs the
session refuses by name."""

import pytest

from komorebi import islands
from komorebi.island_session import IslandSession, _split_graph, check_wired, link_tail


def cores(parts, size=30, columns=10):
    return [
        islands.Island(tuple(range(i * size, (i + 1) * size)), columns)
        for i in range(parts)
    ]


def test_two_islands_reach_each_other_by_their_one_port():
    graph = islands.line(cores(2), links=4)
    table = islands.read_table(graph.table(0), ports=True)
    assert table[0][2:] == (islands.NO_LINK, 0, 0)
    assert table[1][2:] == (0, 1, 0b1)
    assert graph.diameter() == 1


def test_a_ring_of_four_takes_both_ports_to_the_far_island():
    graph = islands.ring(cores(4))
    assert graph.neighbours(0) == (1, 3)
    assert graph.ports_toward(0, 2) == 0b11  # both ways round are shortest
    assert graph.ports_toward(0, 1) == 0b01
    assert graph.ports_toward(0, 3) == 0b10
    assert graph.diameter() == 2


def test_a_line_of_four_has_one_port_and_diameter_three():
    graph = islands.line(cores(4))
    assert graph.ports_toward(0, 3) == 0b1
    assert graph.ports_toward(1, 3) == 0b10  # neighbours of 1: (0, 2)
    assert graph.diameter() == 3


def test_the_two_by_two_grid_with_wrap_doubles_each_pair():
    graph = islands.grid(cores(4), 2, 2)
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
    graph = islands.grid(cores(32, size=8, columns=4), 4, 8)
    assert graph.diameter() == 2 + 4
    assert all(len(graph.neighbours(i)) == 4 for i in range(32))


def test_split_shapes_wire_the_pairs_of_their_shape():
    pairs = lambda shape, parts=4: {  # noqa: E731
        (link.a, link.b, link.count)
        for link in _split_graph(10, 12, parts, 1, shape).links
    }
    assert pairs("line") == {(0, 1, 1), (1, 2, 1), (2, 3, 1)}
    assert pairs("ring") == {(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 3, 1)}
    assert pairs("ring", 3) == {(0, 1, 1), (1, 2, 1), (0, 2, 1)}  # the triangle
    assert pairs("grid") == {(0, 1, 2), (1, 3, 2), (2, 3, 2), (0, 2, 2)}
    assert len(_split_graph(10, 12, 4, 1, "mesh").links) == 6


def test_the_session_wires_a_ring_and_the_grid_and_refuses_three_wires_by_name():
    check_wired(_split_graph(10, 12, 4, 1, "ring"), 4)
    check_wired(_split_graph(10, 12, 3, 1, "ring"), 3)
    check_wired(_split_graph(10, 12, 4, 1, "grid"), 4)  # two wires a pair
    with pytest.raises(ValueError, match="3 wires a pair: beyond two islands a link"):
        check_wired(_split_graph(10, 12, 3, 3, "line"), 3)


def test_more_neighbours_than_ports_or_islands_than_a_byte_are_refused():
    star = islands.Graph(cores(6, size=10), [islands.Link(0, i) for i in range(1, 6)])
    with pytest.raises(ValueError, match="island 0: 5 neighbours, a card's ports 4"):
        check_wired(star, 1)
    check_wired(islands.ring(cores(32, size=4, columns=2)), 1)  # no table bound
    with pytest.raises(ValueError, match="257 islands: 1 to ISLANDS_MAX 256"):
        islands.line(cores(257, size=2, columns=2))


def test_a_load_of_more_than_16_islands_defines_its_arrays_and_message_tail():
    assert link_tail(32) - link_tail(4) == 16 * 16
    assert link_tail(2) == link_tail(16)


def test_three_cards_and_a_shape_without_a_split_are_refused():
    with pytest.raises(ValueError, match="3 cards are not wired yet"):
        IslandSession([0, 1, 2])
    with pytest.raises(ValueError, match="shape='ring': a card split into islands"):
        IslandSession([0, 1], shape="ring")


@pytest.mark.parametrize(
    "shape,count",
    [("ring", n) for n in (3, 4, 5, 8, 16, 32)]
    + [("grid", rc) for rc in ((2, 2), (2, 4), (4, 4), (4, 8))],
)
def test_routes_and_masks_hold_on_rings_and_grids_up_to_32_islands(shape, count):
    if shape == "ring":
        graph, n = islands.ring(cores(count, size=8, columns=4)), count
        diameter = count // 2
    else:
        rows, columns = count
        n = rows * columns
        graph = islands.grid(cores(n, size=8, columns=4), rows, columns)
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
