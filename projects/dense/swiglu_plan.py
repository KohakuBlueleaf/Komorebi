"""The plan of SwiGLU, y = (silu(x W1) * (x W3)) W2, on the dense persistent kernels as
LAYER commands, as pure functions: the shapes, the buffers per core and the L1 fit, and
the commands of a call. Sizes are in 32 x 32 tiles (M rows, D model, H hidden)."""

from dataclasses import dataclass

from projects.dense.plan import L1_BUDGET, MAX_CHUNK, TILE, TILE_BYTES, K, block

SHAPES = {
    "sw256": (256, 256, 512),
    "sw4k": (4096, 256, 512),
    "sw16k": (16384, 256, 512),
    "sw16k384": (16384, 384, 384),
}
W_CBS = (K["kCbW"], K["kCbW3"], K["kCbW2"])
HOLDS = (K["kCbHold0"], K["kCbHold1"], K["kCbHold2"])
DRAM = K["kDram"]


@dataclass(frozen=True)
class Swiglu:
    """[mt, dt] rows of x through W1, W3 [dt, ht] and W2 [ht, dt], all resident; a
    core's rows held in L1 between commands; the gate fused or a pass of its own."""

    mt: int
    dt: int
    ht: int
    gate_pass: bool
    group: int

    def rows(self):
        """Most row tiles any core owns."""
        return -(-self.mt // self.group)

    def buffers(self):
        """(index, pages) of the program's buffers."""
        buffers = [(K["kCbA"], 2 * self.dt), (K["kCbOut"], 2 * block(self.dt))]
        buffers += [(cb, self.dt * self.ht) for cb in W_CBS]
        holds = 3 if self.gate_pass else 2
        return buffers + [(cb, self.rows() * self.ht) for cb in HOLDS[:holds]]

    def l1_bytes(self):
        pages = sum(count for _, count in self.buffers())
        return pages * TILE_BYTES + K["kProfBytes"] + K["kLoadedBytes"]

    def fits(self, budget=L1_BUDGET):
        return self.l1_bytes() <= budget and max(self.dt, self.ht) <= MAX_CHUNK


def make(rows, model_width, hidden_width, gate_pass, workers):
    """The SwiGLU plan of an [rows, model_width] x with `hidden_width` on `workers`
    cores."""
    m_tiles = rows // TILE
    return Swiglu(
        m_tiles,
        model_width // TILE,
        hidden_width // TILE,
        gate_pass,
        min(m_tiles, workers),
    )


def where(a=DRAM, b=DRAM, y=DRAM):
    """Word a8 of LAYER: where A, B and Y live."""
    return a | b << 8 | y << 16


def layer_args(
    plan, a_address, y_address, k_tiles, n_tiles, w_cb, epilogue, held_buffers
):
    """Command words a0..a9 of LAYER."""
    return [
        a_address,
        y_address,
        plan.mt,
        k_tiles,
        n_tiles,
        block(n_tiles),
        w_cb,
        epilogue,
        held_buffers,
        0,
    ]


def loads(plan, w1, w3, w2):
    """LOAD words of the three W (DRAM addresses) into their buffers."""
    dimensions = ((plan.dt, plan.ht), (plan.dt, plan.ht), (plan.ht, plan.dt))
    return [
        [address, cb, 0, *shape]
        for address, cb, shape in zip((w1, w3, w2), W_CBS, dimensions)
    ]


def commands(plan, x, y):
    """The LAYER words of one call, x and y the DRAM input and output: three commands
    (the gate fused into the second product) or four (a gate pass of its own)."""
    (
        model_tiles,
        hidden_tiles,
        (w1_buffer, w3_buffer, w2_buffer),
        (hold0, hold1, hold2),
    ) = (plan.dt, plan.ht, W_CBS, HOLDS)
    if not plan.gate_pass:
        return [
            layer_args(
                plan,
                x,
                0,
                model_tiles,
                hidden_tiles,
                w1_buffer,
                K["kEpSilu"],
                where(y=hold0),
            ),
            layer_args(
                plan,
                x,
                0,
                model_tiles,
                hidden_tiles,
                w3_buffer,
                K["kEpMul"],
                where(b=hold0, y=hold1),
            ),
            layer_args(
                plan,
                0,
                y,
                hidden_tiles,
                model_tiles,
                w2_buffer,
                K["kEpNone"],
                where(a=hold1),
            ),
        ]
    return [
        layer_args(
            plan,
            x,
            0,
            model_tiles,
            hidden_tiles,
            w1_buffer,
            K["kEpNone"],
            where(y=hold0),
        ),
        layer_args(
            plan,
            x,
            0,
            model_tiles,
            hidden_tiles,
            w3_buffer,
            K["kEpNone"],
            where(y=hold1),
        ),
        layer_args(
            plan,
            0,
            0,
            hidden_tiles,
            hidden_tiles,
            0,
            K["kEpGate"],
            where(a=hold0, b=hold1, y=hold2),
        ),
        layer_args(
            plan,
            0,
            y,
            hidden_tiles,
            model_tiles,
            w2_buffer,
            K["kEpNone"],
            where(a=hold2),
        ),
    ]


def flops(plan):
    """Floating-point operations of the three products of one call."""
    return 3 * 2 * plan.mt * plan.dt * plan.ht * TILE**3
