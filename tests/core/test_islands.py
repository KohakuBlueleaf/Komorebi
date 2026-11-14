"""Islands on the host: the graph, ranks, the table a core holds, partition, where a
tensor's pages live, trees and thread numbers. One island answers as today."""

import pytest

from komorebi import islands
from komorebi.islands import NO_LINK, Island, Link


def test_one_island_is_todays_run():
    graph = islands.one(120, 12, threads=3)
    assert graph.link_cores == ((),)
    assert graph.ranks == 120 and graph.resolve(77) == (0, 77)
    assert graph.core_of(77) == (0, 77)
    assert graph.split((8, 16)) == [(0, 8, 16)]
    assert graph.hops(0, 0) == 0 and graph.next_link(0, 0) == NO_LINK
    assert graph.neighbours(0) == ()
    assert islands.read_table(graph.table(0)) == [(0, 120, NO_LINK, 0)]
    assert [graph.thread(n) for n in (1, 2, 3)] == [(0, 1), (0, 2), (0, 3)]
    assert islands.partition(10, 0, 1) == (0, 10)
    placement = islands.place(graph, (256, 256), "tile", split=1)
    assert placement.parts == ((0, 8),) and placement.where(63) == (0, 63)
    assert islands.tree(graph, (0, 120)) == islands.Tree(0, {0: 0}, {}, ())


def test_partition_is_contiguous_and_differs_by_at_most_one():
    parts = [islands.partition(10, part, 4) for part in range(4)]
    assert parts == [(0, 3), (3, 3), (6, 2), (8, 2)]
    with pytest.raises(ValueError):
        islands.partition(10, 4, 4)


def test_link_cores_are_spread_over_rows_and_columns():
    cores = islands.spread(range(120), 12, 4)
    assert len({core // 12 for core in cores}) == 4
    assert len({core % 12 for core in cores}) == 4


def test_a_card_split_in_four_as_a_line():
    graph = islands.split_card(120, 12, 4, links=2)
    assert [len(cores) for cores in graph.link_cores] == [2, 4, 4, 2]
    assert [graph.count(i) for i in range(4)] == [28, 26, 26, 28]
    assert [graph.first(i) for i in range(4)] == [0, 28, 54, 80] and graph.ranks == 108
    assert graph.resolve(54) == (2, 0) and graph.resolve(53) == (1, 25)
    island, core = graph.core_of(0)
    assert island == 0 and core not in graph.link_cores[0]
    assert graph.split((20, 40)) == [(0, 20, 8), (1, 0, 26), (2, 0, 6)]
    assert graph.hops(0, 3) == 3 and graph.next_island(0, 3) == 1
    assert graph.next_link(1, 0) == 0 and graph.next_link(1, 3) == 2
    assert graph.ports[1] == ((0, 0), (0, 1), (2, 0), (2, 1))
    assert islands.read_table(graph.table(1)) == [
        (0, 28, 0, 1),
        (28, 26, NO_LINK, 0),
        (54, 26, 2, 1),
        (80, 28, 2, 2),
    ]
    assert len(graph.table(1)) == 4 * islands.ENTRY_BYTES


def test_a_ring_and_any_connected_graph_route_by_shortest_path():
    ring = islands.ring([Island(tuple(range(16)), 4)] * 5)
    assert ring.hops(0, 4) == 1 and ring.next_island(0, 3) == 4
    assert ring.next_island(0, 2) == 1
    star = islands.Graph(
        [Island(tuple(range(8)), 4)] * 4, [Link(0, 1), Link(0, 2), Link(0, 3, 2, 30e9)]
    )
    assert star.hops(1, 3) == 2 and star.next_island(1, 3) == 0
    assert star.between(3, 0).rate == 30e9
    assert star.neighbours(0) == (1, 2, 3)


def test_limits_name_themselves():
    with pytest.raises(ValueError, match="CORES_MAX"):
        islands.one(129, 12)
    with pytest.raises(ValueError, match="ISLANDS_MAX"):
        islands.Graph([Island((0, 1), 2)] * 257)
    with pytest.raises(ValueError, match="THREAD_NUMBERS"):
        islands.line([Island((0, 1, 2), 3, threads=3)] * 86)
    with pytest.raises(ValueError, match="no path"):
        islands.Graph([Island((0, 1), 2)] * 2)
    with pytest.raises(ValueError, match="no rank"):
        islands.Graph([Island((0,), 1), Island((0, 1), 2)], [Link(0, 1)])


def test_thread_numbers_are_flat_by_island():
    graph = islands.line([Island(tuple(range(8)), 4, 3), Island(tuple(range(8)), 4, 2)])
    assert list(graph.thread_numbers(1)) == [4, 5]
    assert graph.thread(5) == (1, 2) and graph.thread(3) == (0, 3)


def test_pages_split_by_rank_order_and_hints():
    graph = islands.split_card(120, 12, 2)  # 59 and 59 ranks
    split = islands.place(graph, (118, 4))  # rows: rank r's row on its island
    assert split.parts == ((0, 59), (59, 59)) and split.part_pages(1) == 59
    rank = 70
    start, _ = islands.partition(118, rank, 118)
    assert islands.island_of(split, start) == graph.resolve(rank)[0]
    group = islands.place(graph, (30, 4), group=(50, 10))  # 9 ranks on 0, 1 on 1
    assert group.parts == ((0, 27), (27, 3)) and group.where(28) == (1, 1)
    copy = islands.place(graph, (30, 4), copy=True)
    assert islands.island_of(copy, 5, island=1) == 1 and copy.part_pages(0) == 30
    assert islands.page_dims((2, 64, 96), "tile") == (2, 2, 3)
    with pytest.raises(ValueError, match="inside a page"):
        islands.place(graph, (5, 7), split=1)


@pytest.mark.parametrize("parts", [1, 2, 3])
def test_a_split_along_the_second_dimension_is_each_islands_columns(parts):
    graph = islands.split_card(120, 12, parts)
    columns = 2 * graph.ranks  # two tile columns a rank, two tile rows
    tiles = islands.place(graph, (64, 32 * columns), "tile", split=1)
    for page in (0, columns - 1, columns + 77, 2 * columns - 1):
        holder, index = tiles.where(page)
        assert holder == graph.resolve(page % columns // 2)[0]
        assert tiles.part(holder)[index] == page
    assert sum(tiles.part_pages(i) for i in range(parts)) == 2 * columns
    assert tiles.runs(columns, columns) == [
        (i, columns + 2 * graph.first(i), 2 * graph.count(i), 2 * graph.count(i))
        for i in range(parts)
    ]
    like = islands.place(graph, (64, 32), "tile", split=1, group=(0, 1))
    assert like.parts[0] == (0, 1) and like.part_pages(0) == 2


def test_trees_fold_inside_an_island_then_cross_once():
    graph = islands.split_card(120, 12, 4)
    tree = islands.tree(graph, (10, 100), root=40)
    assert tree.root == 1 and tree.leaders == {0: 10, 1: 40, 2: 57, 3: 85}
    assert tree.parent == {0: 1, 2: 1, 3: 2}
    assert tree.order == (3, 0, 2)


def test_a_crossing_passes_through_an_island_without_part_of_the_group():
    star = islands.Graph(
        [Island(tuple(range(8)), 4)] * 4, [Link(0, 1), Link(0, 2), Link(0, 3)]
    )
    tree = islands.tree(star, (star.first(1), star.ranks - star.first(1)))
    assert tree.root == 1 and tree.parent == {2: 1, 3: 1}
