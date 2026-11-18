"""Host side of the matrix product: the reference, the plan and the tile order in
which the kernels read A and W, emulated in numpy."""

import numpy as np
import pytest

from komorebi import layout
from projects.dense import plan as dense_plan
from reference import dense as ref

CONSTANTS = dense_plan.K
T = ref.TILE


def tiles(x):
    """x [r*32, c*32] as a list of 32 x 32 tiles in row-major tile order (the
    order of a tile-layout tensor's pages)."""
    tile_rows, tile_columns = x.shape[0] // T, x.shape[1] // T
    return list(
        x.reshape(tile_rows, T, tile_columns, T)
        .transpose(0, 2, 1, 3)
        .reshape(tile_rows * tile_columns, T, T)
    )


def emulate(plan, activations, weights):
    """Y as the kernels compute it: the reader's pages, the compute loop's tile
    indices into the A and W buffers, the writer's output pages."""
    activation_tiles, weight_tiles = tiles(activations), tiles(weights)
    y = [None] * (plan.mt * plan.nt)
    count = dense_plan.group_size(plan, 120)
    for i in range(count):
        for tile_row in layout.partition(range(plan.mt), i, count):
            row = (
                activation_tiles[tile_row * plan.kt : (tile_row + 1) * plan.kt]
                if plan.row_item
                else None
            )
            for first_column in range(0, plan.nt, plan.ct):
                accumulators = [np.zeros((T, T)) for _ in range(plan.ct)]
                for first_depth in range(0, plan.kt, plan.kc):
                    a_item = (
                        row
                        if plan.row_item
                        else activation_tiles[tile_row * plan.kt + first_depth :][
                            : plan.kc
                        ]
                    )
                    row_offset = first_depth if plan.row_item else 0
                    if plan.resident:
                        wbuf = weight_tiles
                    else:
                        first = first_depth * plan.nt + first_column
                        wbuf = [
                            weight_tiles[first + i * plan.nt + j]
                            for i in range(plan.kc)
                            for j in range(plan.ct)
                        ]
                    for depth in range(plan.kc):
                        base = (
                            depth * plan.ct
                            if not plan.resident
                            else (first_depth + depth) * plan.nt + first_column
                        )
                        for j in range(plan.ct):
                            accumulators[j] += (
                                a_item[row_offset + depth] @ wbuf[base + j]
                            )
                for j in range(plan.ct):
                    y[tile_row * plan.nt + first_column + j] = accumulators[j]
    full = np.array(y).reshape(plan.mt, plan.nt, T, T).transpose(0, 2, 1, 3)
    return full.reshape(plan.mt * T, plan.nt * T)


def test_bf16_rounds_to_nearest_even():
    x = np.array([1.0, 1 + 2**-8, 1 + 3 * 2**-8, -2.5, 1e-40], np.float32)
    got = ref.bf16(x)
    assert got.tolist()[:4] == [1.0, 1.0, 1 + 2**-6, -2.5]
    assert (got.view(np.uint32) & 0xFFFF == 0).all()


def test_errors_are_relative_to_the_reference():
    y = np.ones((64, 96))
    assert ref.errors(y, y) == {"max_rel": 0.0, "rms_rel": 0.0}
    z = y.copy()
    z[40, 70] = 2.0
    assert ref.errors(z, y)["max_rel"] == 1.0


def test_chunk_and_block():
    assert [dense_plan.chunk(depth) for depth in (1, 12, 28, 32, 64, 96, 37)] == [
        1,
        12,
        28,
        32,
        32,
        32,
        1,
    ]
    assert [dense_plan.block(count) for count in (1, 2, 3, 4, 6, 12, 7)] == [
        1,
        2,
        3,
        4,
        3,
        4,
        1,
    ]


def test_plan_resident_until_w_overflows():
    plan = dense_plan.plan(512, 12, 12)
    assert plan.resident and (plan.kc, plan.ct) == (12, 4) and plan.row_item
    assert (
        plan.l1_bytes()
        == (24 + 144 + 8) * dense_plan.TILE_BYTES + CONSTANTS["kProfBytes"]
    )
    big = dense_plan.plan(512, 32, 32, budget=1_000_000)
    assert not big.resident and big.ws_pages == 2 * 32 * 4 and big.a_pages == 64
    assert dense_plan.plan(512, 32, 32, budget=100_000) is None
    assert (
        dense_plan.plan(4, 4, 4, form=CONSTANTS["kFormStream"]).form
        == CONSTANTS["kFormStream"]
    )


def test_product_args_follow_the_header():
    plan = dense_plan.plan(8, 4, 8)
    words = dense_plan.product_args(plan, 100, 200, 300, CONSTANTS["kModeReadOnly"])
    assert len(words) == 10
    assert words[:2] == [100, 200] and words[9] == 300
    assert words[2:9] == [
        8,
        4,
        8,
        plan.kc,
        plan.ct,
        CONSTANTS["kModeReadOnly"],
        plan.form,
    ]
    assert dense_plan.load_args(plan, 300)[0] == 300 and dense_plan.load_args(
        plan, 300
    )[3:] == [4, 8]
    assert dense_plan.flops(plan) == 2 * 8 * 4 * 8 * 32**3


@pytest.mark.parametrize(
    "shape,form",
    [
        ((3, 4, 8), "kFormResident"),
        ((3, 40, 6), "kFormResident"),
        ((3, 4, 8), "kFormStream"),
        ((2, 40, 6), "kFormStream"),
    ],
)
def test_kernel_tile_order_gives_the_product(shape, form):
    mt, kt, nt = shape
    plan = dense_plan.plan(mt, kt, nt, form=CONSTANTS[form])
    rng = np.random.default_rng(1)
    activations = rng.standard_normal((mt * T, kt * T))
    weights = rng.standard_normal((kt * T, nt * T))
    assert np.allclose(emulate(plan, activations, weights), activations @ weights)


def emulate_grid(plan, activations, weights, grid=(12, 10)):
    """Y as the grid kernels compute it: the blocks the readers fill (A rows stride
    kb, W rows stride bn), compute's tile indices, the writer's dropped tiles."""
    activation_tiles, weight_tiles = tiles(activations), tiles(weights)
    y = np.full((plan.mt, plan.nt, T, T), np.nan)
    for tile_row, tile_column in plan.places(grid).values():
        m_end, n_end = min(tile_row + plan.pcm, plan.mt), min(
            tile_column + plan.pcn, plan.nt
        )
        for blk in range(plan.nbh * plan.nbw):
            first_row, first_column = (
                tile_row + blk // plan.nbw * plan.bm,
                tile_column + blk % plan.nbw * plan.bn,
            )
            acc = np.zeros((plan.bm, plan.bn, T, T))
            for kblk in range(plan.nk):
                first_depth = kblk * plan.kb
                activation_block = [np.zeros((T, T))] * (plan.bm * plan.kb)
                weight_block = [np.zeros((T, T))] * (plan.kb * plan.bn)
                for i in range(plan.bm):
                    for j in range(min(plan.kb, plan.kt - first_depth)):
                        if first_row + i < plan.mt:
                            activation_block[i * plan.kb + j] = activation_tiles[
                                (first_row + i) * plan.kt + first_depth + j
                            ]
                for i in range(min(plan.kb, plan.kt - first_depth)):
                    for j in range(plan.bn):
                        if first_column + j < plan.nt:
                            weight_block[i * plan.bn + j] = weight_tiles[
                                (first_depth + i) * plan.nt + first_column + j
                            ]
                inner = plan.kb_last if kblk == plan.nk - 1 else plan.kb
                for i in range(plan.bm):
                    for j in range(plan.bn):
                        for depth in range(inner):
                            acc[i, j] += (
                                activation_block[i * plan.kb + depth]
                                @ weight_block[depth * plan.bn + j]
                            )
            for i in range(plan.bm):
                for j in range(plan.bn):
                    if first_row + i < m_end and first_column + j < n_end:
                        y[first_row + i, first_column + j] = acc[i, j]
    return y.transpose(0, 2, 1, 3).reshape(plan.mt * T, plan.nt * T)


@pytest.mark.parametrize("layout", dense_plan.LAYOUTS)
def test_grid_tile_order_gives_the_product(layout):
    plan = dense_plan.grid_plan(13, 5, 14, layout=layout)
    plan = dense_plan.Grid(
        **(plan.__dict__ | {"bm": 2, "bn": 2, "kb": 2, "sh": 1, "sw": 2})
    )
    rng = np.random.default_rng(1)
    activations = rng.integers(-2, 3, (plan.mt * T, plan.kt * T)).astype(np.float64)
    weights = rng.integers(-2, 3, (plan.kt * T, plan.nt * T)).astype(np.float64)
    assert np.array_equal(
        emulate_grid(plan, activations, weights), activations @ weights
    )


def test_grid_rule_layouts_fit():
    shapes = {(128, 128, 128): "2d", (512, 32, 32): "rows", (1, 128, 128): "cols"}
    for (mt, kt, nt), lay in shapes.items():
        plan = dense_plan.grid_plan(mt, kt, nt)
        assert plan.layout == lay and plan.l1_bytes() <= dense_plan.L1_BUDGET
        plan.check((12, 10))
    plan = dense_plan.grid_plan(128, 128, 128)
    assert (plan.rows, plan.cols, plan.pcm, plan.pcn, plan.sh, plan.sw) == (
        10,
        11,
        13,
        12,
        1,
        4,
    )
    assert plan.group((12, 10)) == (0, 119) and plan.rect((12, 10)) == (0, 0, 11, 10)


def test_an_island_s_grid_plan_keeps_its_group_within_the_island_s_ranks():
    whole = dense_plan.grid_plan(138, 256, 256)  # an island of 120 ranks: 2d
    assert whole.layout == "2d" and whole.group((12, 10)) == (0, 119)
    with pytest.raises(ValueError, match="reaches past 118 ranks"):
        whole.check((12, 10), ranks=118)
    plan = dense_plan.grid_plan(138, 256, 256, ranks=118)  # two link cores held
    assert (plan.layout, plan.rows, plan.cols) == ("2d", 10, 10)  # one column fewer
    assert plan.group((12, 10)) == (0, 118)
    assert dense_plan.grid_plan(138, 256, 256, ranks=120) == whole
    rows = dense_plan.island_plan(138, 256, 256, (12, 10), 1_380_000, 118, ("rows",))
    assert rows.layout == "rows" and rows.group((12, 10))[1] <= 118
    rows.check((12, 10), ranks=118)
    assert dense_plan.island_plan(138, 256, 256, (12, 10), 1, 118) is None  # no L1


def emulate_split(split, activations, weights, form):
    """Y as the K-split kernels compute it: each core's partition of K in chunks (A rows
    stride kc, W rows stride ow), then the parts summed by owner or by tree."""
    activation_tiles, weight_tiles = tiles(activations), tiles(weights)
    y = np.full((split.mt, split.nt, T, T), np.nan)
    outputs = split.oh * split.ow
    for group_index in range(split.groups()):
        tile_row, tile_column = (
            group_index // split.nbw() * split.oh,
            group_index % split.nbw() * split.ow,
        )
        parts = []
        for part in range(split.s):
            depth_indices = layout.partition(range(split.kt), part, split.s)
            acc = np.zeros((outputs, T, T))
            for first_depth in range(0, len(depth_indices), split.kc):
                depth_count = min(split.kc, len(depth_indices) - first_depth)
                activation_block = [np.zeros((T, T))] * (split.oh * split.kc)
                weight_block = [np.zeros((T, T))] * (split.kc * split.ow)
                for j in range(depth_count):
                    depth = depth_indices[first_depth + j]
                    for i in range(split.oh):
                        if tile_row + i < split.mt:
                            activation_block[i * split.kc + j] = activation_tiles[
                                (tile_row + i) * split.kt + depth
                            ]
                    for column in range(split.ow):
                        if tile_column + column < split.nt:
                            weight_block[j * split.ow + column] = weight_tiles[
                                depth * split.nt + tile_column + column
                            ]
                for i in range(split.oh):
                    for column in range(split.ow):
                        for depth in range(depth_count):
                            acc[i * split.ow + column] += (
                                activation_block[i * split.kc + depth]
                                @ weight_block[depth * split.ow + column]
                            )
            parts.append(acc)
        if form == "owner":
            total = parts[0] + sum(parts[1:], np.zeros_like(parts[0]))
        else:
            level = 0
            while 1 << level < split.s:
                for j in range(0, split.s, 2 << level):
                    if j + (1 << level) < split.s:
                        parts[j] = parts[j] + parts[j + (1 << level)]
                level += 1
            total = parts[0]
        for i in range(split.oh):
            for column in range(split.ow):
                if tile_row + i < split.mt and tile_column + column < split.nt:
                    y[tile_row + i, tile_column + column] = total[i * split.ow + column]
    return y.transpose(0, 2, 1, 3).reshape(split.mt * T, split.nt * T)


@pytest.mark.parametrize("form", list(dense_plan.REDUCE_FORMS))
@pytest.mark.parametrize("s, oh, ow, kc", [(3, 2, 2, 2), (5, 1, 3, 1), (1, 3, 1, 4)])
def test_split_tile_order_gives_the_product(form, s, oh, ow, kc):
    split = dense_plan.split_plan(5, 7, 4, s, oh, ow, kc)
    rng = np.random.default_rng(1)
    activations = rng.integers(-2, 3, (split.mt * T, split.kt * T)).astype(np.float64)
    weights = rng.integers(-2, 3, (split.kt * T, split.nt * T)).astype(np.float64)
    assert np.array_equal(
        emulate_split(split, activations, weights, form), activations @ weights
    )


def test_split_plans_of_the_narrow_shapes_fit():
    for (m, k, n), (s, oh, ow) in {
        (128, 896, 128): (7, 1, 4),
        (32, 4096, 4096): (7, 1, 8),
    }.items():
        split = dense_plan.split_plan(m // T, k // T, n // T, s, oh, ow)
        assert split.cores() <= 120 and split.l1_bytes() <= dense_plan.L1_BUDGET
        assert len(split.words()) == dense_plan.S["kSplitWords"]
    with pytest.raises(ValueError):
        dense_plan.split_plan(1, 4, 128, 5, 1, 8)  # a group larger than K's tiles


def test_a_program_s_plans_with_smaller_items_are_refused_by_name():
    # decode's plans: o's 8-tile blocks, then qkv at 4-tile blocks (32 groups of 3);
    # on the card the narrow plan's items left the rings where o's did not fit
    wide = dense_plan.split_plan(1, 64, 64, 15, 1, 8, 2, arrive=64, depth=5)
    narrow = dense_plan.split_plan(1, 64, 128, 3, 1, 4, 2, arrive=64, depth=5)
    assert len(dense_plan.split_table([wide, wide])) == 2 * len(wide.words())
    with pytest.raises(ValueError, match=r"plan 1 \(64 x 128 tiles, block 1 x 4\)"):
        dense_plan.split_table([wide, narrow])
