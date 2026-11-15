"""The decode layer's five calls as a class, as far as the class's lines go: each
product (qkv, o, gate_up, down) as the hand's K split in its plan, run beside the hand's
split with no operation after the sum on the same A and W, Y compared word for word. The
operations after the sum and attend stop at a missing line (README.md)."""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi import formats
from projects.decode import card
from projects.decode import model as md
from projects.dense import card as dense_card
from projects.dense import plan as pl

STARTED = time.monotonic()
PRODUCTS = ("qkv", "o", "gate_up", "down")
RIGHT = ("qkv", "o")  # the products a run gives right; gate_up and down: README.md
GROUPS = {"qkv": 7, "o": 15, "gate_up": 2, "down": 15}  # md.plans' on 120 cores
KERNEL_BYTES = 84 * 1024  # the hand's program, as its bench opens it with the chain


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class Layer(km.Program):
    """The layer's four products, Y = A W for a row tile of A by the hand's K split:
    groups of S ranks, a group WIDTH columns of Y, a rank its km::partition of K in
    CHUNK-tile chunks on both movers, summed up a tree, the root writing the sum (the
    hand's split with no operation). attend: not stated (README.md)."""

    WIDTH = km.constant(md.BLOCK_WIDTH)  # every plan's 1 x 8 block
    CHUNK = km.constant(md.CHUNK)
    W_BLOCK = CHUNK * WIDTH
    QKV_GROUP = km.constant(GROUPS["qkv"])
    O_GROUP = km.constant(GROUPS["o"])
    GATE_UP_GROUP = km.constant(GROUPS["gate_up"])
    DOWN_GROUP = km.constant(GROUPS["down"])
    sums: km.L1[km.fp32, 1, WIDTH, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def o(
        self,
        attended: km.Tensor,
        w_o: km.Tensor,
        x1: km.Tensor,
        k_tiles: int,
        pattern: km.Pattern,
    ):
        share = self.rank % self.O_GROUP  # the rank in its group: km::partition's index
        first = self.rank // self.O_GROUP * self.WIDTH
        k0 = share * (k_tiles // self.O_GROUP) + min(share, k_tiles % self.O_GROUP)
        k1 = (share + 1) * (k_tiles // self.O_GROUP) + min(
            share + 1, k_tiles % self.O_GROUP
        )
        for ab, wb in zip(
            km.Stream(attended.tiles[0:1, k0:k1], block=self.CHUNK, pattern=pattern),
            km.Stream(
                w_o.tiles[k0:k1, first : first + self.WIDTH],
                block=self.W_BLOCK,
                pattern=pattern,
            ),
        ):
            self.sums += self.project(ab, wb)
        whole = km.reduce(self.sums, form="tree", group=self.O_GROUP)
        if self.rank == self.leader:
            x1.tiles[0:1, first : first + self.WIDTH].write(whole)

    @km.handler
    def qkv(
        self,
        x: km.Tensor,
        w_qkv: km.Tensor,
        qkv: km.Tensor,
        k_tiles: int,
        pattern: km.Pattern,
    ):
        share = (
            self.rank % self.QKV_GROUP
        )  # the rank in its group: km::partition's index
        first = self.rank // self.QKV_GROUP * self.WIDTH
        k0 = share * (k_tiles // self.QKV_GROUP) + min(share, k_tiles % self.QKV_GROUP)
        k1 = (share + 1) * (k_tiles // self.QKV_GROUP) + min(
            share + 1, k_tiles % self.QKV_GROUP
        )
        for ab, wb in zip(
            km.Stream(x.tiles[0:1, k0:k1], block=self.CHUNK, pattern=pattern),
            km.Stream(
                w_qkv.tiles[k0:k1, first : first + self.WIDTH],
                block=self.W_BLOCK,
                pattern=pattern,
            ),
        ):
            self.sums += self.project(ab, wb)
        whole = km.reduce(self.sums, form="tree", group=self.QKV_GROUP)
        if self.rank == self.leader:
            qkv.tiles[0:1, first : first + self.WIDTH].write(whole)

    @km.handler
    def gate_up(
        self,
        h: km.Tensor,
        w_gate_up: km.Tensor,
        gated: km.Tensor,
        k_tiles: int,
        pattern: km.Pattern,
    ):
        share = (
            self.rank % self.GATE_UP_GROUP
        )  # the rank in its group: km::partition's index
        first = self.rank // self.GATE_UP_GROUP * self.WIDTH
        k0 = share * (k_tiles // self.GATE_UP_GROUP) + min(
            share, k_tiles % self.GATE_UP_GROUP
        )
        k1 = (share + 1) * (k_tiles // self.GATE_UP_GROUP) + min(
            share + 1, k_tiles % self.GATE_UP_GROUP
        )
        for ab, wb in zip(
            km.Stream(h.tiles[0:1, k0:k1], block=self.CHUNK, pattern=pattern),
            km.Stream(
                w_gate_up.tiles[k0:k1, first : first + self.WIDTH],
                block=self.W_BLOCK,
                pattern=pattern,
            ),
        ):
            self.sums += self.project(ab, wb)
        whole = km.reduce(self.sums, form="tree", group=self.GATE_UP_GROUP)
        if self.rank == self.leader:
            gated.tiles[0:1, first : first + self.WIDTH].write(whole)

    @km.handler
    def down(
        self,
        g: km.Tensor,
        w_down: km.Tensor,
        x2: km.Tensor,
        k_tiles: int,
        pattern: km.Pattern,
    ):
        share = (
            self.rank % self.DOWN_GROUP
        )  # the rank in its group: km::partition's index
        first = self.rank // self.DOWN_GROUP * self.WIDTH
        k0 = share * (k_tiles // self.DOWN_GROUP) + min(
            share, k_tiles % self.DOWN_GROUP
        )
        k1 = (share + 1) * (k_tiles // self.DOWN_GROUP) + min(
            share + 1, k_tiles % self.DOWN_GROUP
        )
        for ab, wb in zip(
            km.Stream(g.tiles[0:1, k0:k1], block=self.CHUNK, pattern=pattern),
            km.Stream(
                w_down.tiles[k0:k1, first : first + self.WIDTH],
                block=self.W_BLOCK,
                pattern=pattern,
            ),
        ):
            self.sums += self.project(ab, wb)
        whole = km.reduce(self.sums, form="tree", group=self.DOWN_GROUP)
        if self.rank == self.leader:
            x2.tiles[0:1, first : first + self.WIDTH].write(whole)


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(md.MODELS), default="qwen3-1.7b")
    parser.add_argument(
        "--products", nargs="*", default=list(RIGHT), choices=list(PRODUCTS)
    )
    parser.add_argument(
        "--known-wrong",
        action="store_true",
        help="allow gate_up and down: run after another product they gave wrong sums, "
        "changing run to run, and a run of the four hung the card (2026-11-16); "
        "under investigation",
    )
    parser.add_argument("--context", type=int, default=256, help="the cache's rows")
    parser.add_argument("--runs", type=int, default=20, help="timed commands a way")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--skip-hand", action="store_true", help="the class alone (its build)"
    )
    args = parser.parse_args()
    wrong = sorted(set(args.products) - set(RIGHT))
    if wrong and not args.known_wrong:
        parser.error(
            f"{', '.join(wrong)}: wrong sums and a hang on 2026-11-16, under "
            "investigation; --known-wrong to run them anyway"
        )
    return args


def operands(session, model, name, rng):
    """A (one row tile), W and two Ys of product `name`, bf16 values of the scale the
    bench's weights have (W variance 1 / K)."""
    k, n = (model.products() | model.shapes())[name]
    a = md.bf16_random(rng, (md.TILE, k), np.sqrt(3.0))[1]
    w = md.bf16_random(rng, (k, n), np.sqrt(3.0 / k))[1]
    zeros = np.zeros((md.TILE, n), np.float32)
    return {
        "want": a.astype(np.float64) @ w.astype(np.float64),
        "a": dense_card.bf16_tensor(session, a),
        "w": dense_card.bf16_tensor(session, w),
        "hand": dense_card.bf16_tensor(session, zeros),
        "class": dense_card.bf16_tensor(session, zeros),
    }


NAMES = {  # each handler's tensor parameters: A, W, Y
    "qkv": ("x", "w_qkv", "qkv"),
    "o": ("attended", "w_o", "x1"),
    "gate_up": ("h", "w_gate_up", "gated"),
    "down": ("g", "w_down", "x2"),
}


def examples(tensors):
    """An example of each handler's tensor parameters, each its product's own."""
    given = {}
    for name in PRODUCTS:
        t = tensors.get(name, next(iter(tensors.values())))
        given |= dict(zip(NAMES[name], (t["a"], t["w"], t["class"])))
    return given


def timed(run, submit, runs):
    """The dispatcher's microseconds of `runs` commands, each waited for: the median."""
    times = []
    for _ in range(runs + 1):
        event = run.wait(submit(), timeout=5.0)
        if event.kind != "done":
            raise km.KomorebiError(f"{event}")
        times.append(event.microseconds)
    return round(float(np.median(times[1:])), 2)


def main():
    args = parse()
    model = md.MODELS[args.model]
    rng = np.random.default_rng(args.seed)
    result = {"model": args.model, "context": args.context, "products": {}}
    plans = md.plans(model, halves=(md.DEPTH, 1))  # the hand's, a command a step
    with km.Session(log=log, kernel_bytes=KERNEL_BYTES) as session:
        made = {name: operands(session, model, name, rng) for name in PRODUCTS}
        tensors = {name: made[name] for name in args.products}
        result["hand_kernel"], hand_us = "none", {}
        if not args.skip_hand:
            result["hand_kernel"], hand_us = hand_products(
                session, model, args, tensors
            )
        program = Layer(session=session, log=log, **examples(made))
        for name in tensors:
            if plans[md.plan_of(model, name)].s != GROUPS[name]:
                raise ValueError(f"{name}: the plan's group is not {GROUPS[name]}")
        run = session.load(program, wait=1.0, deadline=0.5)
        for name, t in tensors.items():
            plan = plans[md.plan_of(model, name)]
            pattern = md.PATTERNS.get(name, "AB")
            k_tiles = (model.products() | model.shapes())[name][0] // md.TILE
            values = (t["a"], t["w"], t["class"], k_tiles, pattern)
            handler = getattr(program, name)
            cores = (0, plan.cores())
            us = timed(
                run,
                lambda h=handler, w=values, c=cores: run.submit(
                    h, c, args=w, exclusive=True
                ),
                args.runs,
            )
            got = {way: session.read(t[way])[0] for way in ("hand", "class")}
            same = got["hand"].view(np.uint16) == got["class"].view(np.uint16)
            shape = t["want"].shape
            for way, raw in got.items():  # against float64 of the same A and W
                values = formats.from_bytes(raw, shape, "bfloat16", "tile")
                errors = md.errors(values, t["want"])
                result["products"].setdefault(name, {})[f"{way}_errors"] = errors
            result["products"][name] |= {
                "plan": f"{plan.groups()} x {plan.s}",
                "class_us": us,
                "words_differ": int(np.sum(~same)),
                "words": int(same.size),
            }
        run.stop()
        result["kernel"] = run.result()["error_name"]
    for name, us in hand_us.items():
        result["products"][name]["hand_us"] = us
    result["ok"] = bool(
        result["kernel"] == "none"
        and result["hand_kernel"] == "none"
        and all(p["words_differ"] == 0 for p in result["products"].values())
    )
    result["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(result))
    return 0 if result["ok"] else 1


def hand_products(session, model, args, tensors):
    """Each product by the hand's program (projects/decode, a command a step), with
    no operation after the sum, into its hand Y: (the kernels' end, {name: us})."""
    decode = card.Decode(session, model, 1, 0.5, log, context=args.context, chain=False)
    hand, tree, times = decode.run, pl.REDUCE_FORMS["tree"], {}
    for name, t in tensors.items():
        plan = decode.plans[md.plan_of(model, name)]
        index = decode.plan_index[md.plan_of(model, name)]
        words = [t["a"].address, t["w"].address, t["hand"].address, tree, 0, index]
        words += [0, 0, 0, 0]  # a6 0: no operation after the sum
        cores = (0, plan.cores())
        times[name] = timed(
            hand,
            lambda w=words, c=cores: hand.submit("split", c, w, exclusive=True),
            args.runs,
        )
    hand.stop()
    return hand.result()["error_name"], times


if __name__ == "__main__":
    sys.exit(main())
