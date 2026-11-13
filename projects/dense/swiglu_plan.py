"""The plan of SwiGLU, y = (silu(x W1) * (x W3)) W2, on the dense resident kernels as
LAYER records, as pure functions: the shapes, the buffers per core and the L1 fit, and
the records of a call. Sizes are in 32 x 32 tiles (M rows, D model, H hidden)."""

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
    member's rows held in L1 between records; the gate fused or a pass of its own."""

    mt: int
    dt: int
    ht: int
    gate_pass: bool
    team: int

    def rows(self):
        """Most row tiles any member owns."""
        return -(-self.mt // self.team)

    def buffers(self):
        """(index, pages) of the program's buffers."""
        out = [(K["kCbA"], 2 * self.dt), (K["kCbOut"], 2 * block(self.dt))]
        out += [(cb, self.dt * self.ht) for cb in W_CBS]
        holds = 3 if self.gate_pass else 2
        return out + [(cb, self.rows() * self.ht) for cb in HOLDS[:holds]]

    def l1_bytes(self):
        pages = sum(n for _, n in self.buffers())
        return pages * TILE_BYTES + K["kProfBytes"] + K["kLoadedBytes"]

    def fits(self, budget=L1_BUDGET):
        return self.l1_bytes() <= budget and max(self.dt, self.ht) <= MAX_CHUNK


def make(m, d, h, gate_pass, workers):
    """The SwiGLU plan of an [m, d] x with hidden width h on `workers` cores."""
    mt = m // TILE
    return Swiglu(mt, d // TILE, h // TILE, gate_pass, min(mt, workers))


def where(a=DRAM, b=DRAM, y=DRAM):
    """Word a8 of LAYER: where A, B and Y live."""
    return a | b << 8 | y << 16


def layer_args(s, a, y, kt, nt, w_cb, ep, at):
    """Record words a0..a9 of LAYER."""
    return [a, y, s.mt, kt, nt, block(nt), w_cb, ep, at, 0]


def loads(s, w1, w3, w2):
    """LOAD words of the three W (DRAM addresses) into their buffers."""
    dims = ((s.dt, s.ht), (s.dt, s.ht), (s.ht, s.dt))
    return [[w, cb, 0, *kn] for w, cb, kn in zip((w1, w3, w2), W_CBS, dims)]


def records(s, x, y):
    """The LAYER words of one call, x and y the DRAM input and output: three records
    (the gate fused into the second product) or four (a gate pass of its own)."""
    d, h, (w1, w3, w2), (h0, h1, h2) = s.dt, s.ht, W_CBS, HOLDS
    if not s.gate_pass:
        return [
            layer_args(s, x, 0, d, h, w1, K["kEpSilu"], where(y=h0)),
            layer_args(s, x, 0, d, h, w3, K["kEpMul"], where(b=h0, y=h1)),
            layer_args(s, 0, y, h, d, w2, K["kEpNone"], where(a=h1)),
        ]
    return [
        layer_args(s, x, 0, d, h, w1, K["kEpNone"], where(y=h0)),
        layer_args(s, x, 0, d, h, w3, K["kEpNone"], where(y=h1)),
        layer_args(s, 0, 0, h, h, 0, K["kEpGate"], where(a=h0, b=h1, y=h2)),
        layer_args(s, 0, y, h, d, w2, K["kEpNone"], where(a=h2)),
    ]


def flops(s):
    """Floating-point operations of the three products of one call."""
    return 3 * 2 * s.mt * s.dt * s.ht * TILE**3
