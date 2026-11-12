"""Panels, teams and shares of the core grid."""

import pytest

from komorebi.layout import (
    LANE_NONE,
    Team,
    blackhole_noc0_x,
    core_order,
    lane_args,
    lane_rect,
    lanes,
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
    assert panels([1, 2, 4, 5, 7]) == [(0, 1), (2, 3), (4, 4)]
    assert panels([]) == []
    assert blackhole_noc0_x(0b11000000) == NOC0_X  # columns 7 and 10 harvested


def test_core_order_is_row_by_row():
    order = core_order(GRID)
    assert len(order) == len(set(order)) == 120
    assert order[:3] == [(0, 0), (1, 0), (2, 0)]
    assert order[12] == (0, 1) and order[-1] == (11, 9)


def test_share_gives_the_first_members_the_extra_item():
    assert [list(share(range(10), i, 4)) for i in range(4)] == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7],
        [8, 9],
    ]
    assert [len(share(range(3), i, 5)) for i in range(5)] == [1, 1, 1, 0, 0]


def test_share_rejects_a_bad_member_or_a_strided_range():
    for index, count in [(-1, 4), (4, 4), (0, 0)]:
        with pytest.raises(ValueError):
            share(range(10), index, count)
    with pytest.raises(ValueError):
        share(range(0, 10, 2), 0, 2)


def test_split_range_gives_each_core_a_start_and_a_count():
    assert split_range(5, ["a", "b", "c"]) == {"a": (0, 2), "b": (2, 2), "c": (4, 1)}
    with pytest.raises(ValueError):
        split_range(5, [])


def test_team_cores_are_a_range_of_the_core_order():
    team = Team(13, 5)
    assert team.members == range(13, 18)
    assert team.cores(GRID) == [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]


def test_team_that_does_not_fit_the_grid_is_rejected():
    team = Team(118, 3)
    with pytest.raises(ValueError, match="does not fit"):
        team.cores(GRID)
    with pytest.raises(ValueError, match="does not fit"):
        team.rectangles(GRID, PANELS)


def test_team_needs_a_first_core_and_at_least_one_member():
    for first, count in [(-1, 3), (5, 0)]:
        with pytest.raises(ValueError):
            Team(first, count)


def test_team_share_splits_a_range_among_its_members():
    team = Team(10, 4)
    parts = [team.share(range(9), i) for i in range(team.count)]
    assert [list(p) for p in parts] == [[0, 1, 2], [3, 4], [5, 6], [7, 8]]


def test_whole_grid_is_one_rectangle_per_panel():
    assert Team(0, 120).rectangles(GRID, PANELS) == [(0, 0, 5, 9), (6, 0, 11, 9)]


def test_rectangles_of_a_partial_team_stay_in_a_row_and_a_panel():
    assert Team(3, 20).rectangles(GRID, PANELS) == [
        (3, 0, 5, 0),
        (6, 0, 11, 0),
        (0, 1, 5, 1),
        (6, 1, 10, 1),
    ]
    assert Team(14, 3).rectangles(GRID, PANELS) == [(2, 1, 4, 1)]


def test_rectangles_need_panels_that_tile_the_columns():
    with pytest.raises(ValueError, match="do not tile"):
        Team(0, 5).rectangles(GRID, [(0, 5), (7, 11)])
    assert Team(3, 20).rectangles(GRID, [(0, 11)]) == [(3, 0, 11, 0), (0, 1, 10, 1)]


def test_column_lanes_send_from_the_start_of_the_nocs_direction():
    down = lanes(GRID, PANELS, (2, 1, 2, 3), "column", 0)
    assert [lane.cores for lane in down] == [
        ((2, 1), (2, 2), (2, 3)),
        ((3, 1), (3, 2), (3, 3)),
    ]
    up = lanes(GRID, PANELS, (2, 1, 2, 3), "column", 1)
    assert up[0].sender == (2, 3) and up[0].cores[-1] == (2, 1)


def test_row_lanes_are_cut_at_the_panel_boundary():
    found = lanes(GRID, PANELS, (4, 0, 4, 1), "row", 1)
    assert [lane.cores for lane in found] == [((5, 0), (4, 0)), ((7, 0), (6, 0))]


def test_lane_rect_corners_follow_the_noc_and_need_consecutive_noc0_steps():
    noc0 = {(x, y): (NOC0_X[x], y + 2) for x, y in core_order(GRID)}
    table = {(x, y): (x + 1, y + 2) for x, y in core_order(GRID)}  # another space
    row = lanes(GRID, PANELS, (6, 4, 3, 1), "row", 0)[0]
    assert lane_rect(row, noc0, table, 0) == (8 | 6 << 16, 9 | 6 << 16)
    column = lanes(GRID, PANELS, (0, 0, 1, 10), "column", 1)[0]
    assert lane_rect(column, noc0, table, 1) == (1 | 10 << 16, 1 | 2 << 16)
    assert (
        lane_rect(lanes(GRID, PANELS, (0, 0, 1, 1), "row", 0)[0], noc0, table, 0)
        is None
    )
    across = lanes(GRID, [(0, 11)], (5, 0, 2, 1), "row", 0)[0]  # NoC0 x 6 then 11
    with pytest.raises(ValueError, match="not NoC0 step"):
        lane_rect(across, noc0, table, 0)


def test_lane_args_give_ranks_inside_the_rectangle_and_none_outside():
    noc0 = {(x, y): (NOC0_X[x], y + 2) for x, y in core_order(GRID)}
    team = Team(12, 24)
    args = lane_args(
        GRID, PANELS, team, (0, 1, 2, 2), "column", 1, noc0, noc0, (1, 3, 2, 4)
    )
    assert sorted(args) == list(team.members)
    assert args[24]["rank"] == 0 and args[12]["rank"] == 1 and args[12]["sender"] == 24
    assert args[12]["receivers"] == 1 and args[12]["team"] == 12 | 24 << 16
    assert args[13]["mc_start"] == 2 | 3 << 16 and args[14]["rank"] == LANE_NONE
    with pytest.raises(ValueError, match="not inside"):
        lane_args(
            GRID,
            PANELS,
            Team(13, 20),
            (0, 1, 2, 2),
            "column",
            1,
            noc0,
            noc0,
            (1, 3, 2, 4),
        )
