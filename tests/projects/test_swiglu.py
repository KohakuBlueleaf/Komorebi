"""Host side of SwiGLU on the dense kernels: the plan's L1 fit, the LAYER commands'
chain through the held buffers, and the tile order of that chain emulated in numpy."""

from collections import deque

import numpy as np

from komorebi import layout
from projects.dense import swiglu_plan
from ref import dense as ref

CONSTANTS = swiglu_plan.K
T = ref.TILE


def tiles(x):
    """x [r*32, c*32] as 32 x 32 tiles in row-major tile order."""
    tile_rows, tile_columns = x.shape[0] // T, x.shape[1] // T
    return list(
        x.reshape(tile_rows, T, tile_columns, T)
        .transpose(0, 2, 1, 3)
        .reshape(tile_rows * tile_columns, T, T)
    )


def emulate(plan, x, weights, words):
    """Y as one core after another runs the call's LAYER commands: x's row from
    DRAM or a held buffer's tiles in FIFO order, W's blocks, the epilogue, the pack."""
    buffers = {cb: tiles(weight) for cb, weight in zip(swiglu_plan.W_CBS, weights)}
    activation_tiles, y = tiles(x), {}
    for i in range(plan.group):
        held = {cb: deque() for cb in swiglu_plan.HOLDS}
        for _, _, mt, kt, nt, ct, wcb, epilogue, slots, _ in words:
            a_at, b_at, y_at = slots & 0xFF, slots >> 8 & 0xFF, slots >> 16
            for tile_row in layout.partition(range(mt), i, plan.group):
                row = None
                if wcb and a_at == swiglu_plan.DRAM:
                    row = activation_tiles[tile_row * kt : (tile_row + 1) * kt]
                elif wcb:
                    row = [held[a_at].popleft() for _ in range(kt)]
                for first_column in range(0, nt, ct):
                    if wcb:
                        accumulators = [
                            sum(
                                row[depth] @ buffers[wcb][depth * nt + first_column + j]
                                for depth in range(kt)
                            )
                            for j in range(ct)
                        ]
                    else:
                        accumulators = [held[a_at].popleft() for _ in range(ct)]
                    if epilogue in (CONSTANTS["kEpSilu"], CONSTANTS["kEpGate"]):
                        accumulators = [ref.silu(tile) for tile in accumulators]
                    if epilogue in (CONSTANTS["kEpMul"], CONSTANTS["kEpGate"]):
                        accumulators = [
                            tile * held[b_at].popleft() for tile in accumulators
                        ]
                    for j in range(ct):
                        if y_at == swiglu_plan.DRAM:
                            y[tile_row * nt + first_column + j] = accumulators[j]
                        else:
                            held[y_at].append(accumulators[j])
    out = np.array([y[depth] for depth in range(len(y))]).reshape(
        plan.mt, plan.dt, T, T
    )
    return out.transpose(0, 2, 1, 3).reshape(plan.mt * T, plan.dt * T)


def test_the_shapes_fit_and_a_wide_layer_does_not():
    for tile_row, width, hidden in swiglu_plan.SHAPES.values():
        for gate_pass in (False, True):
            assert swiglu_plan.make(tile_row, width, hidden, gate_pass, 119).fits()
    plan = swiglu_plan.make(16384, 256, 512, False, 119)
    assert (plan.mt, plan.group, plan.rows()) == (512, 119, 5)
    assert dict(plan.buffers())[CONSTANTS["kCbHold1"]] == 5 * 16
    assert not swiglu_plan.make(16384, 1024, 4096, False, 119).fits()


def test_commands_chain_through_the_held_buffers():
    plan = swiglu_plan.make(256, 256, 512, False, 119)
    first_command, second_command, third_command = swiglu_plan.commands(plan, 100, 200)
    assert (
        first_command[0] == second_command[0] == 100
        and third_command[1] == 200
        and len(first_command) == 10
    )
    assert first_command[8] == swiglu_plan.where(y=CONSTANTS["kCbHold0"])
    assert second_command[8] == swiglu_plan.where(
        b=CONSTANTS["kCbHold0"], y=CONSTANTS["kCbHold1"]
    )
    assert third_command[8] == swiglu_plan.where(
        a=CONSTANTS["kCbHold1"]
    ) and third_command[3:5] == [16, 8]
    gate = swiglu_plan.commands(swiglu_plan.make(256, 256, 512, True, 119), 100, 200)
    assert [command[6] for command in gate] == [
        CONSTANTS["kCbW"],
        CONSTANTS["kCbW3"],
        0,
        CONSTANTS["kCbW2"],
    ]


def test_kernel_tile_order_gives_swiglu():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (5 * T, 2 * T))
    weights = [
        rng.uniform(-0.5, 0.5, plan) for plan in ((2 * T, 3 * T), (2 * T, 3 * T))
    ]
    weights.append(rng.uniform(-0.5, 0.5, (3 * T, 2 * T)))
    for gate_pass in (False, True):
        plan = swiglu_plan.make(5 * T, 2 * T, 3 * T, gate_pass, 2)
        got = emulate(plan, x, weights, swiglu_plan.commands(plan, 0, 0))
        assert np.allclose(got, ref.swiglu(x, *weights))
