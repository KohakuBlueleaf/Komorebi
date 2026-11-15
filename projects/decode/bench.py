"""Decode on one card: T tokens of L decoder layers, started by one command to an X280
thread, which submits every layer's commands; one layer checked against numpy; the
time of each command, and the account of a token against the DRAM read ceiling.
Prints one JSON line. Usage in README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402

import komorebi as km  # noqa: E402
from komorebi import power  # noqa: E402
from komorebi.session import visible_card  # noqa: E402
from projects.decode import card  # noqa: E402
from projects.decode import model as md  # noqa: E402
from projects.dense import bench as dense_bench  # noqa: E402

CEILING_GBS = 495.6  # the card's DRAM read, 120 cores on both movers (dense ceilings)
MATMUL_PER_CORE = 4.70e9 / 120  # tile products a second a core (the card's measured)
D_HEAD = 4  # decode.h DC_HEAD_TILES: a score tile is 4 products, a head's P V 4 a key
# a loose bound on a layer of bf16 steps (bf16 alone: 6e-3 a product, 1e-2 the gate)
CHECK_RMS, CHECK_MAX = 2e-2, 5e-2
# The layer twice from one x at each cache length, the second after every core's
# change of role. Checked: what the card keeps of a layer, and the cache's new row.
CHECK_RUNS = 2
CHECKED = ("q", "k", "v", "a", "x1", "gated", "x")
# The cache's rows a KV head and layer, and the lengths T timed (the last token
# attends over T rows) and checked (in this order: a check appends to row T - 1).
CONTEXT, LENGTHS = 4096, (4096, 1024, 256)
# attend alone first, (groups, first position, tokens): one token in a tile, at a
# tile's last row, 5 and 32 tokens across two tiles.
ALONE = ((1, 1000, 1), (8, 4095, 1), (8, 30, 5), (8, 1000, 32))
# The layer's check: one token at each length, then 32 at the first.
CHECK_ROWS = 32
CASES = {
    "full": {},
    "bare": {"bare": True},
    "parts": {"parts": True},
    "attend_read": {"attend": "read"},
    "attend_local": {"attend": "local"},
    "attend_exp_twice": {"attend": "exp_twice"},
    "overlap": {"overlap": True},
    "parts_apart": {"parts": True, "apart": True},
}
TIMED = ("full", "bare", "parts", "attend_read", "attend_local", "attend_exp_twice")
PRODUCT_STEPS = ("qkv", "o", "gate_up", "down")
SPANNED = ("q", "k", "v", "o")  # overlapped: one span
TRACED = 512  # commands a core stamps with --trace (16 bytes of L1 each)
TRACE_FIELDS = ("kTraceSeen", "kTraceDone", "kTraceSent", "kTraceExtra")
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=list(md.MODELS), default="qwen3-1.7b")
    parser.add_argument("--config", default=None, help="a config.json instead")
    parser.add_argument("--layers", type=int, default=None, help="default: the model's")
    parser.add_argument("--tokens", type=int, default=10)
    parser.add_argument("--cases", nargs="*", default=list(TIMED), choices=list(CASES))
    parser.add_argument("--context", type=int, default=CONTEXT, help="the cache's rows")
    parser.add_argument("--lengths", type=int, nargs="*", default=list(LENGTHS))
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument(
        "--loop-check", action="store_true", help="one token through every layer"
    )
    parser.add_argument("--pause", type=float, default=1.0, help="x a case's time")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", default=None)
    parser.add_argument("--chunk", type=int, default=md.CHUNK, help="K tiles a chunk")
    parser.add_argument(
        "--pattern", choices=("AB", "AAAB"), default=None, help="every product's"
    )
    parser.add_argument(
        "--trace", default=None, help="an .npz of every core's stamps of each command"
    )
    parser.add_argument(
        "--deadline",
        type=float,
        default=None,
        help="a command's deadline in s (default: from the run's size)",
    )
    parser.add_argument(
        "--bodies",
        nargs="*",
        default=list(card.DEFAULT_BODIES),
        choices=list(card.BODIES),
        help="the operations whose vector bodies are emitted (bodies.py)",
    )
    return parser.parse_args()


def patterns(args):
    """md.plans()'s patterns: --pattern on every product, else the model's."""
    return None if args.pattern is None else dict.fromkeys(md.PRODUCTS, args.pattern)


def bounded(errors):
    """errors with "ok": inside the check's bound."""
    return errors | {
        "ok": errors["rms_rel"] <= CHECK_RMS and errors["max_rel"] <= CHECK_MAX
    }


def appended(decode, caches, position, k, v, heads, rows=1):
    """Whether the cache's rows from `position` of the first `heads` KV heads hold the
    tokens' k and v (rows 0 to rows - 1) bit for bit; the host's copy of layer 0's
    cache takes them."""
    exact = True
    for index, (name, tokens) in enumerate((("keys", k), ("values", v))):
        for row in range(rows):
            got = decode.cache_rows(0, name, position + row)[:heads]
            want = np.asarray(tokens[row], np.float32).reshape(-1, got.shape[1])[:heads]
            exact &= bool(np.array_equal(got, want))
            caches[index][:heads, position + row] = got
    return exact


def attend_alone(decode, model, caches, seed, groups, position, rows):
    """ATTEND from the host on `groups` KV heads' groups, `rows` tokens from `position`,
    from a random qkv, against md.attention() over layer 0's cache: errors of a's live
    rows, the append."""
    rng = np.random.default_rng([seed, groups, position, rows])
    shape = (md.TILE, model.products()["qkv"][1])
    qkv = md.bf16_random(rng, shape, np.sqrt(3.0))[1]
    decode.write_activation("qkv", qkv)
    decode.attend(groups, position, rows)
    width = model.heads * model.head_dim
    q, k, v = np.split(qkv, [width, width + model.kv_heads * model.head_dim], axis=1)
    tokens = [
        np.concatenate(
            [
                cache[:, :position],
                token[:rows].reshape(rows, model.kv_heads, -1).swapaxes(0, 1),
            ],
            axis=1,
        )
        for cache, token in zip(caches, (k, v))
    ]
    columns = groups * width // model.kv_heads
    want = md.bf16(md.attention(model, q, *tokens, rows=rows))[:rows, :columns]
    got = decode.read_activation("a")[:rows, :columns]
    out = bounded(md.errors(got, want))
    out["append_exact"] = appended(decode, caches, position, k, v, groups, rows)
    return {"groups": groups, "position": position, "rows": rows} | out


def check(decode, model, weights, caches, seed, length, rows=1):
    """One layer from a known x on the card against model.layer(), `rows` tokens ending
    at row length - 1 of layer 0's cache, CHECK_RUNS times: the larger errors of the
    runs on the live rows, a tensor; whether the cache's new rows are the tokens'."""
    position = length - rows
    start = md.start(seed, model)
    cached = tuple(cache[:, :position].copy() for cache in caches)
    want = md.layer(model, start, weights, cached, rows)
    out, exact = {}, True
    width = {name: model.shapes()[name][1] for name in ("q", "k", "v")}
    for _ in range(CHECK_RUNS):
        decode.write_activation("x", start)
        decode.job(1, 1, position=position, rows=rows)
        got = {
            name: decode.read_activation(name)
            for name in ("qkv", "a", "x1", "gated", "x")
        }
        got["q"], got["k"], got["v"] = np.split(
            got["qkv"], [width["q"], width["q"] + width["k"]], axis=1
        )
        exact &= appended(
            decode, caches, position, got["k"], got["v"], model.kv_heads, rows
        )
        for name in CHECKED:
            errors = md.errors(got[name][:rows], want[name][:rows])
            if name in out:
                errors = {key: max(out[name][key], errors[key]) for key in errors}
            out[name] = errors
    return {name: bounded(errors) for name, errors in out.items()} | {
        "rows": rows,
        "append_exact": exact,
    }


def chain(model, layers, seed, context, length):
    """x after one token through `layers` layers by model.layer(), each over its own
    random cache at position length - 1 (computed before the card is opened)."""
    x, position = md.start(seed, model), length - 1
    for layer in range(layers):
        keys, values = md.cache(model, seed, layer, context)
        cached = (keys[:, :position], values[:, :position])
        x = md.layer(model, x, md.layer_weights(model, seed, layer), cached)["x"]
        log(f"loop reference: layer {layer}")
    return x


def loop_check(decode, model, caches, seed, layers, length, want):
    """One token through every layer from one LAYERS command against chain()'s x (the
    token's row); the host's copy of layer 0's cache takes its new row."""
    position = length - 1
    decode.write_activation("x", md.start(seed, model))
    decode.job(1, layers, position=position)
    out = md.errors(decode.read_activation("x")[:1], want[:1])
    for index, name in enumerate(("keys", "values")):
        caches[index][:, position] = decode.cache_rows(0, name, position)
    return {"layers": layers, "length": length} | out


def measure(decode, case, tokens, layers, index, length):
    """One job of `tokens` x `layers` under the power and clock samplers, the last
    token at the cache's row length - 1."""
    meter = power.PowerSampler(index)
    with meter, dense_bench.Clock(index) as clock:
        meter.mark("job")
        started = time.perf_counter()
        position = length - tokens
        got = decode.job(tokens, layers, timeout=60.0, position=position, **CASES[case])
        seconds = time.perf_counter() - started
        meter.mark("after")
    return got | {
        "case": case,
        "length": length,
        "tokens": tokens,
        "layers": layers,
        "host_seconds": round(seconds, 4),
        "mean_w": meter.result.phases["job"].mean_w,
        "aiclk_mhz": clock.summary(),
    }


def per_layer(result, slot):
    """A case's microseconds a layer: the layer, each step's events, the span of the
    overlapped products."""
    count = result["tokens"] * result["layers"]
    return {
        "layer": slot["microseconds"] / count,
        "steps": {name: step["sum_us"] / count for name, step in slot["steps"].items()},
        "span": slot["span_us"] / count,
    }


def busiest(length, group):
    """Key tiles of the busiest core of a KV head's group, the last token at row
    length - 1: the full tiles' largest share, or the root's (the smallest) and its
    own tile."""
    full = (length - 1) // md.TILE
    return max(-(-full // group), full // group + 1)


def attend_terms(model, cases, length, tokens, group):
    """attend's terms a layer: the cache's K and V at the ceiling (the mean rows the
    tokens attend over), the busiest core's tile products at the card's rate and its
    exponents at their measured time (exp_twice less full) = HARDWARE; the read below
    the ceiling (attend_read), compute not under the read (attend_local), the merge."""
    needed = {"full", "attend_read", "attend_local", "attend_exp_twice"}
    if not needed <= cases.keys():
        return []
    step = {name: cases[name]["steps"]["attend"] for name in needed}
    rows = length - (tokens - 1) / 2
    cache_bytes = 2 * model.kv_heads * rows * model.head_dim * 2
    keys = busiest(length, group)
    products = (2 * keys * (2 * D_HEAD + 1) + 2) / MATMUL_PER_CORE * 1e6
    exps = step["attend_exp_twice"] - step["full"]
    read = cache_bytes / (CEILING_GBS * 1e3)
    compute = step["attend_local"] - step["attend_read"]
    return [
        (
            "HARDWARE",
            f"attend: the cache's K and V at the ceiling ({rows:.1f} rows)",
            read,
        ),
        (
            "HARDWARE",
            f"attend: {2 * keys * (2 * D_HEAD + 1) + 2} tile products, busiest core",
            products,
        ),
        ("HARDWARE", f"attend: {2 * keys} exponent tiles, measured", exps),
        (
            "OURS",
            "attend: the read below the ceiling (q a core, the append, the command)",
            step["attend_read"] - read,
        ),
        ("OURS", "attend: compute not under the read", compute - products - exps),
        ("OURS", "attend: the merge's rounds", step["full"] - step["attend_local"]),
    ]


def account(model, cases, attend=()):
    """{case: terms a layer} of `full` and `overlap`: each weight at the ceiling, its
    read below it (parts), its boundary (bare less parts), its operation (full less
    bare; qkv's and gate_up's hold the norm), attend's terms, the thread; overlapped,
    q to o as one span. Needs full, bare, parts."""
    if not {"full", "bare", "parts"} <= cases.keys():
        return {}
    full, bare, parts = (cases[name]["steps"] for name in ("full", "bare", "parts"))
    if "parts_apart" in cases:  # q, k, v's parts, apart
        parts = parts | {name: cases["parts_apart"]["steps"][name] for name in md.APART}
    ceiling = {
        name: 2 * k * n / (CEILING_GBS * 1e3)
        for name, (k, n) in (model.shapes() | model.products()).items()
    }

    def product(step, own):
        """The terms of a product whose own time is `own`."""
        terms = [
            ("HARDWARE", f"{step}: W at the ceiling", ceiling[step]),
            (
                "OURS",
                f"{step}: the read below the ceiling",
                parts[step] - ceiling[step],
            ),
            ("OURS", f"{step}: the boundary", bare[step] - parts[step]),
        ]
        return terms + [("OURS", f"{step}: its operation", own - bare[step])]

    out = {}
    for name in ("full", "overlap"):
        if name not in cases:
            continue
        case = cases[name]
        steps, terms = case["steps"], []
        spanned = SPANNED if name == "overlap" else ()
        if spanned:
            own = sum(parts[step] for step in spanned) + full["o"] - bare["o"]
            terms += [
                (
                    "HARDWARE",
                    "q, k, v, o: W at the ceiling",
                    sum(ceiling[s] for s in spanned),
                ),
                (
                    "OURS",
                    "q, k, v, o: the read below the ceiling",
                    sum(parts[s] - ceiling[s] for s in spanned),
                ),
                ("OURS", "o: its operation", full["o"] - bare["o"]),
                (
                    "OURS",
                    "q, k, v, o: the boundary left and q, k, v's scale, overlapped",
                    case["span"] - own,
                ),
            ]
        for step in PRODUCT_STEPS:
            if step not in spanned and not (spanned and step == "qkv"):
                terms += product(step, steps[step])
        if name == "full":
            terms += list(attend)
        commands = case["span"] + sum(
            value for step, value in steps.items() if step not in spanned
        )
        terms.append(
            (
                "OURS",
                "the thread between commands (submit, wait)",
                case["layer"] - commands,
            )
        )
        rows = [{"kind": k, "term": t, "us": round(us, 2)} for k, t, us in terms]
        out[name] = {
            "layer_us": round(case["layer"], 2),
            "ceiling_us": round(model.layer_bytes() / (CEILING_GBS * 1e3), 2),
            "terms": rows,
            "unaccounted_us": round(case["layer"] - sum(r["us"] for r in rows), 2),
        }
    return out


def waiting(run):
    """After a failure: each core that a kernel of is in a runtime wait, the waits by
    kernel with their command's sequence (the dispatcher's inspect)."""
    try:
        cores = run.inspect(timeout=2.0)
    except km.KomorebiError as error:
        return f"no inspect: {error}"
    out = []
    for core in cores:
        roles = {
            role: f"{state['where']} @{state['sequence']}"
            for role, state in core.get("live", {}).items()
            if isinstance(state, dict) and state["where"] is not None
        }
        if roles:
            out.append({"core": core["core"]} | roles)
    return out


def main():
    args = parse()
    model = md.from_config(args.config) if args.config else md.MODELS[args.model]
    layers = args.layers or model.layers
    report = {"model": model.name, "layers": layers, "tokens": args.tokens}
    report["bodies"] = args.bodies
    report["layer_bytes"] = model.layer_bytes()
    report["plans"] = {
        f"{k}x{n}": plan.words()
        for (k, n), plan in md.plans(
            model, chunk=args.chunk, patterns=patterns(args)
        ).items()
    }
    index = visible_card()
    deadline = args.deadline or round(max(1.0, args.tokens * layers * 2e-3 + 0.5), 3)
    results, ok = [], True
    if args.loop_check:
        looped = chain(model, layers, args.seed, args.context, args.lengths[0])
    with km.Session(log=log) as session:
        decode = card.Decode(
            session,
            model,
            layers,
            deadline,
            log,
            trace=TRACED if args.trace else 0,
            chunk=args.chunk,
            patterns=patterns(args),
            context=args.context,
            bodies=args.bodies,
        )
        log(
            f"loaded: {len(decode.plans)} plans, deadline {deadline} s, "
            f"{decode.keys_most} key tiles a core at most"
        )
        write_s, first, caches = 0.0, None, None
        for layer in range(layers):
            weights = md.layer_weights(model, args.seed, layer)
            write_s += decode.write_layer(layer, weights)
            cache = md.cache(model, args.seed, layer, args.context)
            write_s += decode.write_cache(layer, *cache)
            if layer == 0:
                first, caches = weights, cache
            log(f"layer {layer} and its cache written ({write_s:.2f} s of writes)")
        cache_bytes = 2 * decode.weights[0]["keys"].pages * md.TILE * md.TILE * 2
        written = layers * (model.layer_bytes() + cache_bytes)
        report["write_gbs"] = round(written / write_s / 1e9, 2)
        try:
            if args.loop_check:  # first: the caches are as written
                report["loop"] = loop_check(
                    decode, model, caches, args.seed, layers, args.lengths[0], looped
                )
                log(f"loop check: {json.dumps(report['loop'])}")
            if not args.no_check:
                report["alone"] = []
                for groups, position, rows in ALONE:
                    got = attend_alone(
                        decode, model, caches, args.seed, groups, position, rows
                    )
                    report["alone"].append(got)
                    log(f"attend alone: {json.dumps(got)}")
                    ok &= got["ok"] and got["append_exact"]
                report["check"] = {}
                checks = [(length, 1) for length in args.lengths]
                for length, rows in checks + [(args.lengths[0], CHECK_ROWS)]:
                    if not ok:
                        break
                    got = check(decode, model, first, caches, args.seed, length, rows)
                    report["check"][f"{length}x{rows}"] = got
                    ok &= got["append_exact"] and all(
                        errors["ok"] for name, errors in got.items() if name in CHECKED
                    )
                    log(f"check, T {length}, {rows} rows: {json.dumps(got)}")
            for length in args.lengths if ok else ():
                for case in args.cases:
                    log(f"case {case}, T {length}: {args.tokens} x {layers} layers")
                    result = measure(decode, case, args.tokens, layers, index, length)
                    log(f"case {case}: {json.dumps(result)}")
                    results.append(result)
                    end = time.monotonic() + args.pause * result["host_seconds"]
                    while (left := end - time.monotonic()) > 0:
                        time.sleep(min(left, 10.0))
        except km.KomorebiError as error:
            report["failed"] = f"{type(error).__name__}: {error}"
            log(f"failed: {report['failed']}")
            ok = False
            report["thread_result"] = decode.run.thread_result(timeout=0.5)._asdict()
            report["waiting"] = waiting(decode.run)
            log(f"waiting: {json.dumps(report['waiting'])}")
        decode.run.stop()
        log("run stopped")
        stats = decode.stats()
        if args.trace:
            trace = decode.run.trace()
            fields = {name: trace.h[name] for name in TRACE_FIELDS}
            np.savez(args.trace, rows=trace.rows, mhz=trace.mhz, **fields)
    layered = {}
    for result in results:
        slot = card.decode_slot(stats[result["stats_slot"]])
        result["thread"] = slot
        layered.setdefault(result["length"], {})[result["case"]] = per_layer(
            result, slot
        )
        token_ms = slot["microseconds"] / result["tokens"] / 1e3
        result["ms_per_token"] = round(token_ms, 3)
        result["tokens_per_second"] = round(1e3 / token_ms, 1)
        result["weight_gbs"] = round(layers * model.layer_bytes() / token_ms / 1e6, 1)
    group = md.plans(model)[md.plan_of(model, "o")].s
    report["account"] = {
        length: account(
            model, cases, attend_terms(model, cases, length, args.tokens, group)
        )
        for length, cases in layered.items()
    }
    report["ceiling_ms_per_token"] = round(
        layers * model.layer_bytes() / (CEILING_GBS * 1e6), 3
    )
    report["cases"] = results
    report["load_avg"] = Path("/proc/loadavg").read_text().split()[:3]
    report["ok"] = ok
    print(json.dumps(report, default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
