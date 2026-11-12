"""Panels, teams and shares of the core grid."""

import pytest

from komorebi.layout import (
    Team,
    core_order,
    panels,
    share,
    split_range,
)

GRID = (12, 10)
# Logical columns 0 to 5 sit at NoC0 x 1 to 6, columns 6 to 11 at NoC0 x 11 to 16.
NOC0_X = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
PANELS = [(0, 5), (6, 11)]


def test_panels_split_where_noc0_is_not_consecutive():
    assert panels(NOC0_X) == PANELS


@pytest.mark.parametrize(
    ("noc0_x", "expected"),
    [
        (list(range(1, 13)), [(0, 11)]),
        ([1, 2, 4, 5, 7], [(0, 1), (2, 3), (4, 4)]),
        ([3], [(0, 0)]),
        ([], []),
        ([5, 4, 3], [(0, 0), (1, 1), (2, 2)]),
    ],
)
def test_panels_other_grids(noc0_x, expected):
    assert panels(noc0_x) == expected


def test_panels_cover_every_column_once():
    flat = [x for x0, x1 in panels(NOC0_X) for x in range(x0, x1 + 1)]
    assert flat == list(range(len(NOC0_X)))


def test_core_order_is_row_by_row():
    order = core_order(GRID)
    assert len(order) == len(set(order)) == 120
    assert order[:3] == [(0, 0), (1, 0), (2, 0)]
    assert order[12] == (0, 1) and order[-1] == (11, 9)


@pytest.mark.parametrize("total", [0, 1, 5, 12, 98, 120])
@pytest.mark.parametrize("count", [1, 2, 3, 7, 12, 120, 130])
@pytest.mark.parametrize("start", [0, 7])
def test_share_assigns_every_element_exactly_once(total, count, start):
    items = range(start, start + total)
    parts = [share(items, i, count) for i in range(count)]
    assert [x for part in parts for x in part] == list(items)
    sizes = [len(part) for part in parts]
    assert max(sizes) - min(sizes) <= 1
    assert sizes == sorted(sizes, reverse=True)


def test_share_gives_the_first_members_the_extra_item():
    assert [list(share(range(10), i, 4)) for i in range(4)] == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7],
        [8, 9],
    ]


def test_share_with_more_members_than_items_leaves_the_last_empty():
    assert [len(share(range(3), i, 5)) for i in range(5)] == [1, 1, 1, 0, 0]


@pytest.mark.parametrize(("index", "count"), [(-1, 4), (4, 4), (0, 0), (0, -1)])
def test_share_rejects_a_bad_member(index, count):
    with pytest.raises(ValueError):
        share(range(10), index, count)


def test_share_rejects_a_strided_range():
    with pytest.raises(ValueError):
        share(range(0, 10, 2), 0, 2)


def test_split_range_covers_every_item_once():
    cores = core_order(GRID)
    ranges = split_range(98, cores)
    assert list(ranges) == cores
    covered = [
        i for start, count in ranges.values() for i in range(start, start + count)
    ]
    assert covered == list(range(98))
    assert max(c for _, c in ranges.values()) - min(c for _, c in ranges.values()) <= 1


def test_split_range_needs_a_core():
    with pytest.raises(ValueError):
        split_range(5, [])


def test_split_range_small_example():
    assert split_range(5, ["a", "b", "c"]) == {"a": (0, 2), "b": (2, 2), "c": (4, 1)}


def test_team_cores_are_a_range_of_the_core_order():
    team = Team(13, 5)
    assert team.members == range(13, 18)
    assert team.cores(GRID) == [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]


@pytest.mark.parametrize("team", [Team(118, 3), Team(120, 1), Team(0, 121)])
def test_team_that_does_not_fit_the_grid_is_rejected(team):
    with pytest.raises(ValueError, match="does not fit"):
        team.cores(GRID)
    with pytest.raises(ValueError, match="does not fit"):
        team.rectangles(GRID, PANELS)


@pytest.mark.parametrize(("first", "count"), [(-1, 3), (5, 0), (5, -5), (5, -30)])
def test_team_needs_a_first_core_and_at_least_one_member(first, count):
    with pytest.raises(ValueError):
        Team(first, count)


def test_team_share_splits_a_range_among_its_members():
    team = Team(10, 4)
    parts = [team.share(range(9), i) for i in range(team.count)]
    assert [list(p) for p in parts] == [[0, 1, 2], [3, 4], [5, 6], [7, 8]]


def test_teams_made_with_share_tile_the_grid():
    teams = [Team(p.start, len(p)) for p in (share(range(120), i, 7) for i in range(7))]
    cores = [c for team in teams for c in team.cores(GRID)]
    assert cores == core_order(GRID)


def test_whole_grid_is_one_rectangle_per_panel():
    assert Team(0, 120).rectangles(GRID, PANELS) == [(0, 0, 5, 9), (6, 0, 11, 9)]


def test_rectangles_of_a_partial_team():
    assert Team(3, 20).rectangles(GRID, PANELS) == [
        (3, 0, 5, 0),
        (6, 0, 11, 0),
        (0, 1, 5, 1),
        (6, 1, 10, 1),
    ]


def test_rectangles_of_a_team_inside_one_row_and_panel():
    assert Team(14, 3).rectangles(GRID, PANELS) == [(2, 1, 4, 1)]
    assert Team(4, 4).rectangles(GRID, PANELS) == [(4, 0, 5, 0), (6, 0, 7, 0)]


@pytest.mark.parametrize(
    "bad_panels", [[(0, 3)], [(0, 5)], [(0, 5), (7, 11)], [(0, 5), (5, 11)], []]
)
def test_rectangles_need_panels_that_tile_the_columns(bad_panels):
    with pytest.raises(ValueError, match="do not tile"):
        Team(0, 5).rectangles(GRID, bad_panels)


def test_rectangles_cover_the_team_exactly_and_stay_in_a_panel():
    for first in range(0, 120, 1):
        for count in range(1, 120 - first + 1):
            team = Team(first, count)
            covered = []
            for x0, y0, x1, y1 in team.rectangles(GRID, PANELS):
                assert any(p0 <= x0 and x1 <= p1 for p0, p1 in PANELS), team
                covered += [
                    (x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)
                ]
            assert sorted(covered, key=lambda c: (c[1], c[0])) == team.cores(GRID), team


def test_rectangles_use_only_the_panels_given():
    one_panel = [(0, 11)]
    assert Team(3, 20).rectangles(GRID, one_panel) == [(3, 0, 11, 0), (0, 1, 10, 1)]
