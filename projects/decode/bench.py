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
# a loose bound on a layer of bf16 steps (bf16 alone: 6e-3 a product, 1e-2 the gate)
CHECK_RMS, CHECK_MAX = 2e-2, 5e-2
# The layer twice from one x: fused and waited, then q, k, v apart and overlapped (the
# second after every core's change of role). Checked: what the card keeps of a layer.
CHECK_RUNS = 2
CHECKED = ("q", "k", "v", "x1", "gated", "x")
CASES = {
    "full": {},
    "bare": {"bare": True},
    "parts": {"parts": True},
    "overlap": {"overlap": True},
    "parts_apart": {"parts": True, "apart": True},
}
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
    parser.add_argument("--cases", nargs="*", default=list(CASES), choices=list(CASES))
    parser.add_argument("--no-check", action="store_true")
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
    return parser.parse_args()


def patterns(args):
    """md.plans()'s patterns: --pattern on every product, else the model's."""
    return None if args.pattern is None else dict.fromkeys(md.PRODUCTS, args.pattern)


def check(decode, model, weights, seed):
    """One layer from a known x on the card against model.layer(), CHECK_RUNS times:
    the larger errors of the runs, a tensor."""
    start = md.start(seed, model)
    want = md.layer(model, start, weights)
    out = {}
    width = {name: model.shapes()[name][1] for name in ("q", "k", "v")}
    for run in range(CHECK_RUNS):
        decode.write_activation("x", start)
        decode.job(1, 1, overlap=run == 1)
        got = {
            name: decode.read_activation(name) for name in ("qkv", "x1", "gated", "x")
        }
        got["q"] = got["qkv"][:, : width["q"]]
        if run == 1:  # apart: k and v in their own tensors
            got |= {name: decode.read_activation(name) for name in ("k", "v")}
        else:
            got["k"], got["v"] = np.split(
                got["qkv"][:, width["q"] :], [width["k"]], axis=1
            )
        for name in CHECKED:
            errors = md.errors(got[name], want[name])
            if name in out:
                errors = {key: max(out[name][key], errors[key]) for key in errors}
            out[name] = errors
    for errors in out.values():
        errors["ok"] = errors["rms_rel"] <= CHECK_RMS and errors["max_rel"] <= CHECK_MAX
    return out


def measure(decode, case, tokens, layers, index):
    """One job of `tokens` x `layers` under the power and clock samplers."""
    meter = power.PowerSampler(index)
    with meter, dense_bench.Clock(index) as clock:
        meter.mark("job")
        started = time.perf_counter()
        got = decode.job(tokens, layers, timeout=60.0, **CASES[case])
        seconds = time.perf_counter() - started
        meter.mark("after")
    return got | {
        "case": case,
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


def account(model, cases):
    """{case: terms a layer} of `full` and `overlap`: each weight at the ceiling, its
    read below it (parts), its boundary (bare less parts), its operation (full less
    bare; qkv's and gate_up's hold the norm), the thread; overlapped, q to o as one
    span. Needs full, bare, parts."""
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


def main():
    args = parse()
    model = md.from_config(args.config) if args.config else md.MODELS[args.model]
    layers = args.layers or model.layers
    report = {"model": model.name, "layers": layers, "tokens": args.tokens}
    report["layer_bytes"] = model.layer_bytes()
    report["plans"] = {
        f"{k}x{n}": plan.words()
        for (k, n), plan in md.plans(
            model, chunk=args.chunk, patterns=patterns(args)
        ).items()
    }
    index = visible_card()
    deadline = round(max(1.0, args.tokens * layers * 2e-3 + 0.5), 3)
    results, ok = [], True
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
        )
        log(f"loaded: {len(decode.plans)} plans, deadline {deadline} s")
        write_s, first = 0.0, None
        for layer in range(layers):
            weights = md.layer_weights(model, args.seed, layer)
            write_s += decode.write_layer(layer, weights)
            first = weights if layer == 0 else first
            log(f"layer {layer} written ({write_s:.2f} s of writes)")
        report["write_gbs"] = round(layers * model.layer_bytes() / write_s / 1e9, 2)
        try:
            if not args.no_check:
                report["check"] = check(decode, model, first, args.seed)
                ok &= all(errors["ok"] for errors in report["check"].values())
                log(f"check: {json.dumps(report['check'])}")
            for case in args.cases:
                log(f"case {case}: {args.tokens} tokens of {layers} layers")
                result = measure(decode, case, args.tokens, layers, index)
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
        layered[result["case"]] = per_layer(result, slot)
        token_ms = slot["microseconds"] / result["tokens"] / 1e3
        result["ms_per_token"] = round(token_ms, 3)
        result["tokens_per_second"] = round(1e3 / token_ms, 1)
        result["weight_gbs"] = round(layers * model.layer_bytes() / token_ms / 1e6, 1)
    report["account"] = account(model, layered)
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
