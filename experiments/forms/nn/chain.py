"""A small model as ONE chain of calls of three classes merged into one program, against
numpy and against the same calls as commands: L layers of layer norm of a row of 64
tiles, a product by W1, SiLU, a product by W2 and a residual add, 5 L calls.
Example: python chain.py --layers 4"""

import argparse
import sys

import numpy as np
from forms import (
    bf16,
    card_options,
    error,
    finish,
    log,
    open_session,
    random,
    read,
    tiles,
    timed,
    words,
)

import komorebi as km
from komorebi.dsl.tile import row_operand

WIDE = 64  # tiles of the row: 2,048 values
CHUNKS = WIDE // 2  # the norm's field tiles: one for each chunk of 2 tiles
CORES = 16  # the products' and the activations' group
EPS = 1e-5


@km.tile_function(tolerance=1e-5)
def layer_norm(row, weight, bias):
    mean = km.reduce_sum(row, exact=True) * (1.0 / (32 * WIDE))
    d = row - mean.col()
    var = km.reduce_sum(d * d, exact=True) * (1.0 / (32 * WIDE))
    return d * km.rsqrt(var + EPS).col() * weight.row() + bias.row()


@km.tile_function(tolerance=1e-6)
def silu(a):
    return km.silu(a)


@km.tile_function(tolerance=1e-6)
def add(a, b):
    return a + b


class Norm(km.Program):
    """The norm of the row: one core (its call's cores=(0, 1))."""

    weight: km.L1[km.fp32, CHUNKS, 32, 32]
    bias: km.L1[km.fp32, CHUNKS, 32, 32]

    @km.handler
    def norm(self, x: km.Tensor, gamma: km.Tensor, beta: km.Tensor, out: km.Tensor):
        first = self.rank * WIDE
        self.weight.fill(km.Stream(gamma.pages(0, CHUNKS)))
        self.bias.fill(km.Stream(beta.pages(0, CHUNKS)))
        results = km.Stream(out.pages(first, WIDE), write=True, block=8)
        for row in km.Stream(x.pages(first, WIDE), block=8, row=WIDE):
            results.push(layer_norm(row, self.weight, self.bias))


class Proj(km.Program):
    """y = x W: core r writes y's tiles r WIDTH to r WIDTH + WIDTH - 1."""

    WIDTH = km.constant(4)  # tiles of y a core: 16 cores x 4 = 64
    W_BLOCK = 2 * WIDTH  # W's tiles a stream block: 2 K rows
    sums: km.L1[km.fp32, 1, WIDTH, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project_tile(a, b):
        return a @ b

    @km.handler
    def project(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        first = self.rank * self.WIDTH
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:WIDE], block=2),
            km.Stream(w.tiles[0:WIDE, first : first + self.WIDTH], block=self.W_BLOCK),
        ):
            self.sums += self.project_tile(xb, wb)
        y.tiles[0:1, first : first + self.WIDTH].write(self.sums)


class Act(km.Program):
    """SiLU, and the residual add, over `tiles` pages a core."""

    @km.handler
    def silu_layer(self, a: km.Tensor, out: km.Tensor, tiles: int):
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=4)
        for x in km.Stream(a.pages(first, tiles), block=4):
            results.push(silu(x))

    @km.handler
    def residual(self, a: km.Tensor, r: km.Tensor, out: km.Tensor, tiles: int):
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=4)
        for x, y in zip(
            km.Stream(a.pages(first, tiles), block=4),
            km.Stream(r.pages(first, tiles), block=4),
        ):
            results.push(add(x, y))


class Model(Norm, Proj, Act):
    """The three classes' handlers in one program: a chain's calls are one program's."""


def field_pages(values):
    """A .row() field of the chunk form: row_operand of each chunk's 64 values."""
    pages = [row_operand(values[64 * c : 64 * c + 64], 2) for c in range(CHUNKS)]
    return np.asarray(pages, np.float32)


def calls(x, layers, normed, h, g, past):
    """The 5 calls of each layer; `past` marks x where the chain reads it written
    5 calls back (km.past), and is the identity for commands."""
    out = []
    for w1, w2, gamma, beta in layers:
        out += [
            (Model.norm, (0, 1), (x, gamma, beta, normed)),
            (Model.project, (0, CORES), (normed, w1, h)),
            (Model.silu_layer, (0, CORES), (h, g, WIDE // CORES)),
            (Model.project, (0, CORES), (g, w2, h)),
            (Model.residual, (0, CORES), (h, past(x), x, WIDE // CORES)),
        ]
    return out


def truncated(values):
    """The left operand of a HiFi2 product: 7 of a bf16 value's 8 significant bits."""
    bits = np.asarray(values, np.float32).view(np.uint32) & np.uint32(0xFFFE0000)
    return bits.view(np.float32).astype(np.float64)


def model(x, weights, stored):
    """The layers in float64; `stored`: each tensor written rounded to bf16 and each
    product's left operand truncated as HiFi2 does (the card's path)."""
    keep = bf16 if stored else (lambda v: v)
    left = truncated if stored else (lambda v: v)
    for w1, w2, gamma, beta in weights:
        m = x.mean(axis=1, keepdims=True)
        v = ((x - m) ** 2).mean(axis=1, keepdims=True)
        n = keep((x - m) / np.sqrt(v + EPS) * gamma + beta)
        h = keep(left(n) @ w1)
        g = keep(h / (1.0 + np.exp(-h)))
        x = keep(x + keep(left(g) @ w2))
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    card_options(parser)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    x0 = random(rng, (32, WIDE * 32))
    weights = [
        (random(rng, (WIDE * 32, WIDE * 32), 1.0 / 45.25),  # 1 / sqrt(2048)
         random(rng, (WIDE * 32, WIDE * 32), 1.0 / 45.25),
         1.0 + 0.1 * random(rng, WIDE * 32), 0.1 * random(rng, WIDE * 32))
        for _ in range(args.layers)
    ]  # fmt: skip
    report = {"layers": args.layers, "calls": 5 * args.layers, "cores": CORES}
    with open_session(args) as session:
        x_cmd, x_chain = tiles(session, x0), tiles(session, x0)
        normed, h, g = (tiles(session, np.zeros_like(x0)) for _ in range(3))
        layers = [
            (tiles(session, w1), tiles(session, w2),
             session.tensor(field_pages(gamma), "float32"),
             session.tensor(field_pages(beta), "float32"))
            for w1, w2, gamma, beta in weights
        ]  # fmt: skip
        w1, _, gamma, beta = layers[0]
        program = Model(
            x=x_cmd, gamma=gamma, beta=beta, out=normed, w=w1, y=h, a=h, r=h,
            chain=True, session=session, log=log,
        )  # fmt: skip
        run = session.load(program)
        try:
            us = [
                timed(run, handler, group, call)
                for handler, group, call in calls(
                    x_cmd, layers, normed, h, g, lambda t: t
                )
            ]
            report["commands_us"] = round(sum(us), 1)
            log(f"{len(us)} commands: {report['commands_us']} us")
            chain_calls = [
                handler(*call, cores=group)
                for handler, group, call in calls(
                    x_chain, layers, normed, h, g, km.past
                )
            ]
            chain = run.chain(chain_calls, cores=(0, CORES))
            event = run.wait(run.submit(chain), timeout=30.0)
            if event.kind != "done":
                raise RuntimeError(f"the chain: {event}")
            report["chain_us"] = round(event.microseconds, 1)
            log(f"one chain of {len(chain_calls)} calls: {report['chain_us']} us")
        finally:
            run.stop()
        got_cmd, got_chain = words(session, x_cmd), words(session, x_chain)
        got = read(session, x_chain)
    report["chain_equals_commands"] = bool(np.array_equal(got_cmd, got_chain))
    report["chain_vs_commands"] = {
        "max": 0.0 if report["chain_equals_commands"] else 1.0
    }
    report["vs_card_path"] = error(got, model(x0, weights, stored=True))
    report["vs_float64"] = error(got, model(x0, weights, stored=False))
    log(f"chain = commands: {report['chain_equals_commands']}")
    log(f"card path {report['vs_card_path']}, float64 {report['vs_float64']}")
    finish(
        report,
        [("chain_vs_commands", "max", 0.0), ("vs_card_path", "rms", 0.01),
         ("vs_float64", "rms", 0.05)],
    )  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
