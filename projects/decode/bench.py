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

import komorebi as km  # noqa: E402
from komorebi import power  # noqa: E402
from komorebi.session import visible_card  # noqa: E402
from projects.decode import card  # noqa: E402
from projects.decode import model as md  # noqa: E402
from projects.dense import bench as dense_bench  # noqa: E402

CEILING_GBS = 495.6  # the card's DRAM read, 120 cores on both movers (dense ceilings)
# a loose bound on a layer of bf16 steps (bf16 alone: 6e-3 a product, 1e-2 the gate)
CHECK_RMS, CHECK_MAX = 2e-2, 5e-2
CHECK_RUNS = 2  # the layer twice from the same x: the second after every core's roles
CHECKED = ("h1", "q", "k", "v", "x1", "h", "up", "gated", "x")
READ_BACK = ("q", "k", "v", "h", "up", "gated", "x")  # h1 and x1 are overwritten
CASES = {"full": {}, "bare": {"bare": True}, "parts": {"parts": True}}
PRODUCT_STEPS = ("q", "k", "v", "o", "up", "gate", "down")
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
    parser.add_argument(
        "--cases", nargs="*", default=["full", "bare"], choices=list(CASES)
    )
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--pause", type=float, default=1.0, help="x a case's time")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", default=None)
    return parser.parse_args()


def check(decode, model, weights, seed):
    """One layer from a known x on the card against model.layer(), CHECK_RUNS times:
    the larger errors of the runs, a tensor."""
    start = md.start(seed, model)
    want = md.layer(model, start, weights)
    out = {}
    for _ in range(CHECK_RUNS):
        decode.write_activation("x", start)
        decode.job(1, 1)
        for name in READ_BACK:
            errors = md.errors(decode.read_activation(name), want[name])
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


def account(model, result, slot, bare=None):
    """A layer's microseconds by term: HARDWARE (the weight read at the ceiling), OURS
    (each command's time above it, the operations', the thread's between commands) and
    what is left; from the thread's clock and each command's event."""
    count = result["tokens"] * result["layers"]
    layer_us = slot["microseconds"] / count
    steps = {name: step["sum_us"] / count for name, step in slot["steps"].items()}
    shapes = model.shapes()
    hardware = {
        name: 2 * k * n / (CEILING_GBS * 1e3) for name, (k, n) in shapes.items()
    }
    terms = []
    for step in PRODUCT_STEPS:
        if not steps.get(step):
            continue
        ceiling = hardware[step]
        terms.append(
            {"term": f"{step}: W at the ceiling", "kind": "HARDWARE", "us": ceiling}
        )
        plain = steps[step]
        if bare is not None and step in ("o", "gate", "down"):
            plain = bare["steps"][step] / bare["count"]
            terms.append(
                {
                    "term": f"{step}: its operation",
                    "kind": "OURS",
                    "us": steps[step] - plain,
                }
            )
        terms.append(
            {
                "term": f"{step}: the product above the ceiling",
                "kind": "OURS",
                "us": plain - ceiling,
            }
        )
    terms += [
        {"term": f"{step}: the command", "kind": "OURS", "us": steps[step]}
        for step in ("norm1", "norm2")
        if steps.get(step)
    ]
    commands = sum(steps.values())
    terms.append(
        {
            "term": "the thread between commands (submit, wait)",
            "kind": "OURS",
            "us": layer_us - commands,
        }
    )
    for term in terms:
        term["us"] = round(term["us"], 2)
    total = sum(term["us"] for term in terms)
    return {
        "layer_us": round(layer_us, 2),
        "ceiling_us": round(model.layer_bytes() / (CEILING_GBS * 1e3), 2),
        "steps_us": {name: round(value, 2) for name, value in steps.items()},
        "terms": terms,
        "unaccounted_us": round(layer_us - total, 2),
    }


def main():
    args = parse()
    model = md.from_config(args.config) if args.config else md.MODELS[args.model]
    layers = args.layers or model.layers
    report = {"model": model.name, "layers": layers, "tokens": args.tokens}
    report["layer_bytes"] = model.layer_bytes()
    report["plans"] = {
        f"{k}x{n}": plan.words() for (k, n), plan in md.plans(model).items()
    }
    index = visible_card()
    deadline = round(max(1.0, args.tokens * layers * 2e-3 + 0.5), 3)
    results, ok = [], True
    with km.Session(log=log) as session:
        decode = card.Decode(session, model, layers, deadline, log)
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
    slots = {
        result["case"]: card.decode_slot(stats[result["stats_slot"]])
        for result in results
    }
    bare = None
    if "bare" in slots:
        result = next(r for r in results if r["case"] == "bare")
        bare = {
            "steps": {n: s["sum_us"] for n, s in slots["bare"]["steps"].items()},
            "count": result["tokens"] * result["layers"],
        }
    for result in results:
        slot = slots[result["case"]]
        result["thread"] = slot
        result["account"] = account(
            model, result, slot, bare if result["case"] == "full" else None
        )
        token_ms = slot["microseconds"] / result["tokens"] / 1e3
        result["ms_per_token"] = round(token_ms, 3)
        result["tokens_per_second"] = round(1e3 / token_ms, 1)
        result["weight_gbs"] = round(layers * model.layer_bytes() / token_ms / 1e6, 1)
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
