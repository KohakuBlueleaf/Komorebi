"""Where a wide decode layer's time goes: the five products of bounds.py as one
chain against the memory bound and against their commands' sum, a command's times by
core from the run's trace and by K, on one card or two. README.md.
Example: python decode_cost.py --cards 0 --part chain --format bfp8_b bfp4_b"""

import argparse
import json
import sys

import numpy as np

import komorebi as km
from experiments.language.bounds import bounds as bd
from komorebi import abi
from komorebi.bench import power
from komorebi.dsl.chain import chain_stamps

# The layer's calls in order: product, handler, input, output, K tiles; up reads o's
# output two calls back. The input of o is q's first 128 tiles (attention's place).
CALLS = (
    ("qkv", "qkv", "x", "q", bd.HIDDEN // 32),
    ("o", "o", "q", "h", bd.HEADS * bd.HEAD_DIM // 32),
    ("gate", "gate_up", "h", "g", bd.HIDDEN // 32),
    ("up", "gate_up", "h", "u", bd.HIDDEN // 32),
    ("down", "down", "u", "y", bd.MLP // 32),
)
# A handler's tensor parameters (a tensor's width is the build's, so o and down,
# whose X differ, are two handlers).
PARAMS = {h: (f"x_{h}", f"w_{h}", f"y_{h}") for h in ("qkv", "o", "down")}
PARAMS["gate_up"] = ("x_gu", "w_gu", "y_gu")
# A product's group width and size on one card and on two. One class holds the five,
# so gate and up take o's width (bounds.py's 28 x 5 does not fit beside the others).
PLAN = {
    "qkv": (16, (7, 14)),
    "o": (14, (10, 19)),
    "gate": (14, (2, 4)),
    "up": (14, (2, 4)),
    "down": (14, (10, 19)),
}
SIZES = {"x": bd.HIDDEN, "q": bd.QKV, "h": bd.HIDDEN, "g": bd.MLP, "u": bd.MLP}
SIZES["y"] = bd.HIDDEN
TERMS = ("o", "gate")  # the products split by term (--part terms)
log = bd.log


class Layer(km.Program):
    """The five decode products by bounds.py's K split, with K a word: groups of S_*
    cores, a group W16 or W14 columns of Y, each group's sums reduced by a tree; with
    the chain's walk, so the layer is one chain."""

    S_QKV = km.constant(7)
    S_O = km.constant(10)
    S_G = km.constant(2)
    W16 = km.constant(16)
    W14 = km.constant(14)
    CHUNK = km.constant(2)
    B16 = CHUNK * W16  # W's blocks
    B14 = CHUNK * W14
    sums16: km.L1[km.fp32, 1, W16, 32, 32]
    sums14: km.L1[km.fp32, 1, W14, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def qkv(
        self,
        x_qkv: km.Tensor,
        w_qkv: km.Tensor,
        y_qkv: km.Tensor,
        k: int,
        col0: int,
    ):
        share = self.rank % self.S_QKV  # km::partition's share of K
        first = col0 + self.rank // self.S_QKV * self.W16
        k0 = share * (k // self.S_QKV) + min(share, k % self.S_QKV)
        k1 = (share + 1) * (k // self.S_QKV) + min(share + 1, k % self.S_QKV)
        for xb, wb in zip(
            km.Stream(x_qkv.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(w_qkv.tiles[k0:k1, first : first + self.W16], block=self.B16),
        ):
            self.sums16 += self.project(xb, wb)
        whole = km.reduce(self.sums16, form="tree", group=self.S_QKV)
        if self.rank == self.leader:
            y_qkv.tiles[0:1, first : first + self.W16].write(whole)

    @km.handler
    def o(
        self,
        x_o: km.Tensor,
        w_o: km.Tensor,
        y_o: km.Tensor,
        k: int,
        col0: int,
    ):
        share = self.rank % self.S_O
        first = col0 + self.rank // self.S_O * self.W14
        k0 = share * (k // self.S_O) + min(share, k % self.S_O)
        k1 = (share + 1) * (k // self.S_O) + min(share + 1, k % self.S_O)
        for xb, wb in zip(
            km.Stream(x_o.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(w_o.tiles[k0:k1, first : first + self.W14], block=self.B14),
        ):
            self.sums14 += self.project(xb, wb)
        whole = km.reduce(self.sums14, form="tree", group=self.S_O)
        if self.rank == self.leader:
            y_o.tiles[0:1, first : first + self.W14].write(whole)

    @km.handler
    def down(
        self,
        x_down: km.Tensor,
        w_down: km.Tensor,
        y_down: km.Tensor,
        k: int,
        col0: int,
    ):
        share = self.rank % self.S_O
        first = col0 + self.rank // self.S_O * self.W14
        k0 = share * (k // self.S_O) + min(share, k % self.S_O)
        k1 = (share + 1) * (k // self.S_O) + min(share + 1, k % self.S_O)
        for xb, wb in zip(
            km.Stream(x_down.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(w_down.tiles[k0:k1, first : first + self.W14], block=self.B14),
        ):
            self.sums14 += self.project(xb, wb)
        whole = km.reduce(self.sums14, form="tree", group=self.S_O)
        if self.rank == self.leader:
            y_down.tiles[0:1, first : first + self.W14].write(whole)

    @km.handler
    def gate_up(
        self,
        x_gu: km.Tensor,
        w_gu: km.Tensor,
        y_gu: km.Tensor,
        k: int,
        col0: int,
    ):
        share = self.rank % self.S_G
        first = col0 + self.rank // self.S_G * self.W14
        k0 = share * (k // self.S_G) + min(share, k % self.S_G)
        k1 = (share + 1) * (k // self.S_G) + min(share + 1, k % self.S_G)
        for xb, wb in zip(
            km.Stream(x_gu.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(w_gu.tiles[k0:k1, first : first + self.W14], block=self.B14),
        ):
            self.sums14 += self.project(xb, wb)
        whole = km.reduce(self.sums14, form="tree", group=self.S_G)
        if self.rank == self.leader:
            y_gu.tiles[0:1, first : first + self.W14].write(whole)


def groups_of(session, name):
    """A product's command groups and col0 (PLAN): one card one group; two cards one
    group over both islands centred on the edge, then each island's half alone.
    [(where, group, col0)]."""
    width, split = PLAN[name]
    graph = bd.islands_of(session)
    nt = bd.PRODUCTS[name]["n"] // 32
    if graph is None:
        return [("card", (0, nt // width * split[0]), 0)]
    cores, edge = nt // 2 // width * split[1], graph.first(1)
    return [
        ("both", (edge - cores, 2 * cores), 0),
        ("island 0", (edge - cores, cores), 0),
        ("island 1", (edge, cores), nt // 2),
    ]


def tensors_of(session, fmt, seed, check_cols):
    """X, the five Ws (bounds.py's pages) and the outputs, placed as bounds.py places
    them (two cards: W and Y split over a centred group, X a copy)."""
    t, checked = {}, {}
    for name, _, source, out, _ in CALLS:
        hints = bd.hints_of(session, name)
        pages, cols, wcols = bd.weights(name, fmt, seed, check_cols)
        shape = bd.PRODUCTS[name]
        t["w_" + name] = session.empty(
            (shape["k"], shape["n"]), fmt, "tile", **hints["w"]
        )
        session.write(t["w_" + name], pages)
        checked[name] = (cols, wcols)
        t[out] = session.empty((32, SIZES[out]), "bfloat16", "tile", **hints["y"])
        if source == "x":
            x = bd.activations(1, bd.HIDDEN, seed)
            t["x"] = session.tensor(x, "bfloat16", "tile", **hints["x"])
    return t, checked


def calls_of(session, t):
    """The layer's five calls (a class's calls, each on its product's group)."""
    out = []
    for name, handler, source, target, k in CALLS:
        _, group, col0 = groups_of(session, name)[0]
        given = km.past(t[source]) if name == "up" else t[source]
        method = getattr(Layer, handler)
        out.append(method(given, t["w_" + name], t[target], k, col0, cores=group))
    return out


def words(session, t, checked):
    """Each product's relative RMS against float64 of its input as the card left it
    (the step before's bfloat16) and the float32 W, on the checked columns."""
    out = {}
    for name, _, source, target, k in CALLS:
        got = bd.row_tile(session, t[target], 0, SIZES[target])
        given = bd.row_tile(session, t[source], 0, SIZES[source])[:, : k * 32]
        cols, wcols = checked[name]
        out[name] = bd.rms(got[:, cols], given @ wcols.astype(np.float64))
    return out


def parts_us(event):
    """An event's microseconds by island part (one card: one)."""
    events = event if isinstance(event, list) else getattr(event, "parts", (event,))
    if any(part.kind != "done" for part in events):
        raise km.KomorebiError(f"not done: {event}")
    return [part.microseconds for part in events]


def crossed(run, before):
    """What crossed the link since `before` (None on one card)."""
    if before is None:
        return None
    return {k: int(v) for k, v in (run.crossings() - before).total().items()}


def crossings_now(run):
    return run.crossings() if hasattr(run, "crossings") else None


def made(session, t, chunk, stamps=False):
    """The Layer class at the session's group sizes, each handler's tensors its
    product's (the examples)."""
    at = 1 if bd.islands_of(session) is not None else 0
    examples = {}
    for name, handler, source, target, _ in CALLS:
        x, w, y = PARAMS[handler]
        examples |= {x: t[source], w: t["w_" + name], y: t[target]}
    return Layer(
        log=log,
        session=session,
        chain=True,
        chain_stamps=stamps,
        S_QKV=PLAN["qkv"][1][at],
        S_O=PLAN["o"][1][at],
        S_G=PLAN["gate"][1][at],
        CHUNK=chunk,
        **examples,
    )


def part_chain(session, fmt, args):
    """The layer as one chain and as five commands, both timed `repeat` times, the
    words after each; on one card the chain's stamps' terms (a second load); on two
    each command also on each island alone, with what crossed the link."""
    t, checked = tensors_of(session, fmt, args.seed, args.check_cols)
    graph = bd.islands_of(session)
    out = {"format": fmt, "cards": 2 if graph else 1, "chunk": args.chunk}
    program = made(session, t, args.chunk)
    run = session.load(program)
    try:
        chain = run.chain(calls_of(session, t))
        before = crossings_now(run)
        first = parts_us(run.wait(run.submit(chain), timeout=10.0))
        out["chain_crossed"] = crossed(run, before)
        out["chain_words"] = words(session, t, checked)
        times = [
            parts_us(run.wait(run.submit(chain), timeout=10.0))
            for _ in range(args.repeat)
        ]
        out["chain_first_us"] = first
        out["chain_us"] = float(np.median([max(p) for p in times]))
        out["chain_parts_us"] = np.median(np.array(times), axis=0).tolist()
        chain.free()
        out["commands"] = commands(session, run, program, t, args.repeat)
        out["command_words"] = words(session, t, checked)
    finally:
        run.stop()
    out["kernel"] = run.result()["error_name"] if graph is None else "stopped"
    if graph is None:
        out["chain_terms"] = stamped(session, fmt, t, args)
    for tensor in t.values():
        tensor.free()
    bound = sum(bd.bounds_of(n, fmt, 1, out["cards"])["memory_us"] for n, *_ in CALLS)
    out["bound_us"] = bound
    out["commands_sum_us"] = sum(
        c["us"] for c in out["commands"] if c["where"] in WHOLE
    )
    out["chain_ratio"] = out["chain_us"] / bound
    out["commands_ratio"] = out["commands_sum_us"] / bound
    return out


WHOLE = ("card", "both")


def commands(session, run, program, t, repeat):
    """Each call of the layer as a plain command on each of its groups, `repeat` times:
    the median microseconds (the slower part), each part's and what crossed."""
    out = []
    for name, handler, source, target, k in CALLS:
        for where, group, col0 in groups_of(session, name):
            args = (t[source], t["w_" + name], t[target], k, col0)
            method = getattr(program, handler)
            before = crossings_now(run)
            times = [
                parts_us(run.wait(run.submit(method, group, args=args), timeout=10.0))
                for _ in range(repeat)
            ]
            each = crossed(run, before)
            out.append(
                {
                    "product": name,
                    "where": where,
                    "group": list(group),
                    "us": float(np.median([max(p) for p in times])),
                    "parts_us": np.median(np.array(times), axis=0).tolist(),
                    "crossed_a_command": (
                        None
                        if each is None
                        else {a: b / repeat for a, b in each.items()}
                    ),
                }
            )
            log(json.dumps(out[-1]))
    return out


def stamped(session, fmt, t, args):
    """One card: the chain again, made with chain_stamps=True, `repeat` chains; each
    record's terms (us, median and largest over its cores)."""
    program = made(session, t, args.chunk, stamps=True)
    run = session.load(program)
    try:
        chain = run.chain(calls_of(session, t))
        rows = chain_stamps.tensor_for(session, 120)
        chain_stamps.read(session, run, program, rows, (0, 120))
        for _ in range(args.repeat):
            parts_us(run.wait(run.submit(chain), timeout=10.0))
        held = chain_stamps.read(session, run, program, rows, (0, 120))
        mhz = power.aiclk_mhz()
        terms = chain_stamps.terms(held, mhz, len(CALLS))
        chain.free()
    finally:
        run.stop()
    rows.free()
    return [{"product": c[0]} | term for c, term in zip(CALLS, terms)]


def part_terms(session, fmt, args):
    """One card: o and gate as plain commands at full K and at fewer K tiles, traced:
    each command's dispatcher time and its cores' terms (summarized's)."""
    t, _ = tensors_of(session, fmt, args.seed, args.check_cols)
    program = made(session, t, args.chunk)
    handles = []
    run = session.load(program, trace=args.trace)
    try:
        for name, handler, source, target, k in CALLS:
            if name not in TERMS:
                continue
            _, group, col0 = groups_of(session, name)[0]
            split = PLAN[name][1][0]
            for kk in sorted({k, k // 2, k // 4, 2 * split}, reverse=True):
                args_ = (t[source], t["w_" + name], t[target], kk, col0)
                for _ in range(args.repeat):
                    handle = run.submit(getattr(program, handler), group, args=args_)
                    event = run.wait(handle, timeout=10.0)
                    handles.append((name, kk, split, handle, parts_us(event)[0]))
    finally:
        run.stop()
    trace = run.trace()
    for tensor in t.values():
        tensor.free()
    return summarized(trace, handles, fmt)


def summarized(trace, handles, fmt):
    """part_terms's rows by product and K, medians over the repeats: the take's spread,
    a leaf's take to its outputs written (its K share and send), the root's outputs
    after the last leaf's (the tree and Y's write), the completion after it."""
    h = abi.headers()
    seen, done, sent = h["kTraceSeen"], h["kTraceDone"], h["kTraceSent"]
    rows = {}
    for name, kk, split, handle, us in handles:
        words = trace.command(handle)
        if words is None or not len(words):  # not traced (a prebuild's mock run)
            continue
        at = ((words[:, :3] - words[:, seen].min()) % 2**32) / trace.mhz
        share = np.arange(len(at)) % split
        root, leaf = share == 0, share % 2 == 1
        last_leaf = at[leaf, done].max()
        terms = {
            "dispatcher_us": us,
            "cores_span": float(at[:, sent].max()),
            "outside_cores": us - float(at[:, sent].max()),
            "take_spread": float(at[:, seen].max()),
            "leaf_take_to_done": float(np.median(at[leaf, done] - at[leaf, seen])),
            "last_leaf_done": float(last_leaf),
            "root_after_last_leaf": float(at[root, done].max() - last_leaf),
            "completion_after_root": float(at[:, sent].max() - at[root, done].max()),
        }
        rows.setdefault((name, kk, split), []).append(terms)
    out = []
    for (name, kk, split), each in rows.items():
        out.append(
            {"product": name, "format": fmt, "k_tiles": kk, "group": split}
            | {"k_a_core": round(kk / split, 2), "rounds": int(np.ceil(np.log2(split)))}
            | {
                key: round(float(np.median([e[key] for e in each])), 2)
                for key in each[0]
            }
        )
        log(json.dumps(out[-1]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0", help="0, or 0,1 (the output sharded)")
    parser.add_argument("--part", choices=("chain", "terms"), default="chain")
    parser.add_argument("--format", nargs="+", default=["bfp8_b"])
    parser.add_argument("--chunk", type=int, default=2, help="X tiles a block")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--trace", type=int, default=255, help="--part terms")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--check-cols", type=int, default=256)
    parser.add_argument("--kernel-bytes", type=int)
    args = parser.parse_args()
    cards = [int(card) for card in args.cards.split(",")]
    opened = {"cards": cards} if len(cards) > 1 else {"card": cards[0]}
    if args.kernel_bytes:
        opened["kernel_bytes"] = args.kernel_bytes
    if args.part == "terms" and len(cards) > 1:
        sys.exit("--part terms is one card's")
    reports = []
    with km.Session(log=log, **opened) as session:
        for fmt in args.format:
            if args.part == "chain":
                reports.append(part_chain(session, fmt, args))
                log(json.dumps(reports[-1], default=float))
            else:
                reports += part_terms(session, fmt, args)
    for report in reports:
        print(json.dumps(report, default=float), flush=True)
    print(json.dumps(power.clocked({"cards": cards}, card=cards[0])), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
