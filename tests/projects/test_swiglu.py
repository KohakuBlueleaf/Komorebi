"""Host side of SwiGLU on the dense kernels: the plan's L1 fit, the LAYER commands'
chain through the held buffers, and the tile order of that chain emulated in numpy."""

from collections import deque

import numpy as np

from komorebi import layout
from projects.dense import swiglu_plan as sp
from ref import dense as ref

K = sp.K
T = ref.TILE


def tiles(x):
    """x [r*32, c*32] as 32 x 32 tiles in row-major tile order."""
    r, c = x.shape[0] // T, x.shape[1] // T
    return list(x.reshape(r, T, c, T).transpose(0, 2, 1, 3).reshape(r * c, T, T))


def emulate(s, x, w, words):
    """Y as one member after another runs the call's LAYER commands: x's row from
    DRAM or a held buffer's tiles in FIFO order, W's blocks, the epilogue, the pack."""
    buf = {cb: tiles(v) for cb, v in zip(sp.W_CBS, w)}
    px, y = tiles(x), {}
    for i in range(s.group):
        held = {cb: deque() for cb in sp.HOLDS}
        for _, _, mt, kt, nt, ct, wcb, ep, at, _ in words:
            a_at, b_at, y_at = at & 0xFF, at >> 8 & 0xFF, at >> 16
            for m in layout.partition(range(mt), i, s.group):
                row = None
                if wcb and a_at == sp.DRAM:
                    row = px[m * kt : (m + 1) * kt]
                elif wcb:
                    row = [held[a_at].popleft() for _ in range(kt)]
                for j0 in range(0, nt, ct):
                    if wcb:
                        dst = [
                            sum(row[k] @ buf[wcb][k * nt + j0 + j] for k in range(kt))
                            for j in range(ct)
                        ]
                    else:
                        dst = [held[a_at].popleft() for _ in range(ct)]
                    if ep in (K["kEpSilu"], K["kEpGate"]):
                        dst = [ref.silu(t) for t in dst]
                    if ep in (K["kEpMul"], K["kEpGate"]):
                        dst = [t * held[b_at].popleft() for t in dst]
                    for j in range(ct):
                        if y_at == sp.DRAM:
                            y[m * nt + j0 + j] = dst[j]
                        else:
                            held[y_at].append(dst[j])
    out = np.array([y[k] for k in range(len(y))]).reshape(s.mt, s.dt, T, T)
    return out.transpose(0, 2, 1, 3).reshape(s.mt * T, s.dt * T)


def test_the_shapes_fit_and_a_wide_layer_does_not():
    for m, d, h in sp.SHAPES.values():
        for gate_pass in (False, True):
            assert sp.make(m, d, h, gate_pass, 119).fits()
    s = sp.make(16384, 256, 512, False, 119)
    assert (s.mt, s.group, s.rows()) == (512, 119, 5)
    assert dict(s.buffers())[K["kCbHold1"]] == 5 * 16
    assert not sp.make(16384, 1024, 4096, False, 119).fits()


def test_commands_chain_through_the_held_buffers():
    s = sp.make(256, 256, 512, False, 119)
    r1, r2, r3 = sp.commands(s, 100, 200)
    assert r1[0] == r2[0] == 100 and r3[1] == 200 and len(r1) == 10
    assert r1[8] == sp.where(y=K["kCbHold0"])
    assert r2[8] == sp.where(b=K["kCbHold0"], y=K["kCbHold1"])
    assert r3[8] == sp.where(a=K["kCbHold1"]) and r3[3:5] == [16, 8]
    gate = sp.commands(sp.make(256, 256, 512, True, 119), 100, 200)
    assert [w[6] for w in gate] == [K["kCbW"], K["kCbW3"], 0, K["kCbW2"]]


def test_kernel_tile_order_gives_swiglu():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (5 * T, 2 * T))
    w = [rng.uniform(-0.5, 0.5, s) for s in ((2 * T, 3 * T), (2 * T, 3 * T))]
    w.append(rng.uniform(-0.5, 0.5, (3 * T, 2 * T)))
    for gate_pass in (False, True):
        s = sp.make(5 * T, 2 * T, 3 * T, gate_pass, 2)
        got = emulate(s, x, w, sp.commands(s, 0, 0))
        assert np.allclose(got, ref.swiglu(x, *w))
