"""Komorebi-nn on one card: the decode layer through the executor's list beside the
decode's own thread, and what an entry costs driven three ways (a closed list, an open
list, the host submitting each call). Prints one JSON line. Usage in README.md."""

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
from projects.nn import nn  # noqa: E402

LENGTHS = (4096, 1024, 256)  # the cache lengths T timed, as the decode bench's
CHECKED = ("qkv", "a", "x1", "gated", "x")  # what a layer leaves, compared bit for bit
COST_ENTRIES = 200  # the cost lists' length
COST_TIMES = 5  # a closed cost list's replays
COST_CORES = (1, 120)  # the cheapest call on one core and on every core
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--layers", type=int, default=None, help="default: the model's")
    parser.add_argument("--tokens", type=int, default=10)
    parser.add_argument("--lengths", type=int, nargs="*", default=list(LENGTHS))
    parser.add_argument("--check", action="store_true", help="one layer every way")
    parser.add_argument("--costs", action="store_true", help="an entry's cost by term")
    parser.add_argument("--decode", action="store_true", help="the layers timed")
    parser.add_argument("--pause", type=float, default=1.0, help="x a case's time")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", default=None)
    return parser.parse_args()


def layer_state(decode):
    """What a layer leaves (CHECKED) and layer 0's cache rows, as float32."""
    return {name: decode.read_activation(name) for name in CHECKED}


def appended(decode, position):
    """Layer 0's cache rows at `position`: keys and values, [KV heads, 128] each."""
    return [decode.cache_rows(0, name, position) for name in ("keys", "values")]


def check(decode, executor, model, seed, length):
    """One token through layer 0 at row length - 1, from one x, six ways: the decode's
    thread a command a step and as a chain, the list closed (both), open, the host's
    calls. Every tensor and the cache's new row against the first, bit for bit."""
    position = length - 1
    start = md.start(seed, model)
    calls = nn.layer_calls(decode, 0, position)
    ways = {}

    def hand(chain):
        decode.job(1, 1, position=position, chain=chain)

    def closed(chain):
        entries = nn.decode_entries(decode, 1, 1, position, chain)
        executor.write(entries)
        executor.closed(entries)

    def opened():
        listed = executor.open(len(calls) + 1)
        for call in calls:
            listed.append(call)
        listed.sync()

    runs = {
        "hand_commands": lambda: hand(False),
        "hand_chain": lambda: hand(True),
        "list_commands": lambda: closed(False),
        "list_chain": lambda: closed(True),
        "open_commands": opened,
        "host_commands": lambda: nn.drive(decode.run, calls),
    }
    for name, go in runs.items():
        decode.write_activation("x", start)
        go()
        ways[name] = (layer_state(decode), appended(decode, position))
        log(f"check: {name} ran")
    want, rows = ways["hand_commands"]
    out = {}
    for name, (got, got_rows) in ways.items():
        same = {key: bool(np.array_equal(got[key], want[key])) for key in CHECKED}
        same["cache_row"] = all(
            np.array_equal(a, b) for a, b in zip(got_rows, rows, strict=True)
        )
        out[name] = same
    return out


def per_entry(terms, count):
    """A run's terms (us) a entry: the thread's parts and the rest."""
    out = {name: round(terms[name] / count, 3) for name in nn.TERMS[:5]}
    parts = ("fetch", "submit", "wait", "idle", "check")
    out["wake"] = round((terms["wait"] - terms["command"]) / count, 3)
    out["rest"] = round((terms["total"] - sum(terms[n] for n in parts)) / count, 3)
    out["idle"] = round(terms["idle"] / count, 3)
    out["check"] = round(terms["check"] / count, 3)
    return out


def costs(decode, executor):
    """COST_ENTRIES cheapest calls a list, on each of COST_CORES: closed (one run,
    then COST_TIMES replays), open (appended while it runs, then a sync), the host's
    calls; the host's read of a result after it. Thread terms are read after stop."""
    out, at = [], 0
    for cores in COST_CORES:
        call = nn.nop_call(decode, (0, cores))
        entries = nn.Entries(at)
        for _ in range(COST_ENTRIES):
            entries.command(call)
        started = time.perf_counter()
        executor.write(entries)
        written = time.perf_counter() - started
        _, first = executor.closed(entries)
        started = time.perf_counter()
        _, replays = executor.closed(entries, times=COST_TIMES)
        closed_s = time.perf_counter() - started
        log(f"costs: {cores} cores closed ({written * 1e3:.2f} ms to write)")
        at += COST_ENTRIES
        opened = {}
        for fence in (True, False):
            listed = executor.open(at, fence=fence)
            started = time.perf_counter()
            for _ in range(COST_ENTRIES):
                listed.append(call)
            appended_s = time.perf_counter() - started
            listed.sync()
            synced_s = time.perf_counter() - started - appended_s
            at = listed.next
            opened["fenced" if fence else "checked"] = {
                "slot": listed.slot,
                "host_us": {
                    name: round(seconds * 1e6 / COST_ENTRIES, 3)
                    for name, seconds in listed.times.items()
                },
                "append_us": round(appended_s * 1e6 / COST_ENTRIES, 3),
                "sync_us": round(synced_s * 1e6, 1),
            }
            log(f"costs: {cores} cores open, fence {fence}")
        started = time.perf_counter()
        host = nn.drive(decode.run, [call] * COST_ENTRIES)
        host_s = time.perf_counter() - started
        log(f"costs: {cores} cores host")
        started = time.perf_counter()
        decode.session.read_tiles_bf16(decode.tensors["x"], 0, 1)
        tile_s = time.perf_counter() - started
        started = time.perf_counter()
        decode.read_activation("x")
        row_s = time.perf_counter() - started
        host = np.array(host, np.float64)
        out.append(
            {
                "cores": cores,
                "write_list_us": round(written * 1e6, 1),
                "closed_slots": [first, replays],
                "closed_host_us": round(closed_s * 1e6 / COST_TIMES / COST_ENTRIES, 3),
                "open": opened,
                "host_us": {
                    "submit": round(host[:, 0].mean() * 1e6, 3),
                    "wait": round(host[:, 1].mean() * 1e6, 3),
                    "command": round(host[:, 2].mean() / nn.TICKS_PER_MICROSECOND, 3),
                    "total": round(host_s * 1e6 / COST_ENTRIES, 3),
                },
                "read_tile_us": round(tile_s * 1e6, 1),
                "read_x_us": round(row_s * 1e6, 1),
            }
        )
    return out


def measure(decode, executor, index, way, tokens, layers, length):
    """One way of `tokens` x `layers` layers, the last token at row length - 1, under
    the power and clock samplers: the host's seconds and the thread's slot."""
    position = length - tokens
    meter = power.PowerSampler(index)
    with meter, dense_bench.Clock(index) as clock:
        meter.mark("job")
        if way == "host":
            calls = [
                call
                for token in range(tokens)
                for layer in range(layers)
                for call in nn.layer_calls(decode, layer, position + token)
            ]
            started = time.perf_counter()
            host = nn.drive(decode.run, calls)
            seconds, slot = time.perf_counter() - started, None
        else:
            entries = nn.decode_entries(
                decode, tokens, layers, position, way == "chain"
            )
            executor.write(entries)
            started = time.perf_counter()
            _, slot = executor.closed(entries)
            seconds, host = time.perf_counter() - started, None
        meter.mark("after")
    out = {
        "way": way,
        "length": length,
        "tokens": tokens,
        "layers": layers,
        "host_seconds": round(seconds, 4),
        "slot": slot,
        "mean_w": meter.result.phases["job"].mean_w,
        "aiclk_mhz": clock.summary(),
    }
    if host is not None:
        host = np.array(host, np.float64)
        steps = host[:, 2].reshape(-1, len(nn.STEPS)) / nn.TICKS_PER_MICROSECOND
        count = tokens * layers
        out["layer_us"] = round(seconds * 1e6 / count, 2)
        out["steps_us"] = dict(
            zip(nn.STEPS, np.round(steps.sum(0) / count, 2).tolist(), strict=True)
        )
        out["host_submit_us"] = round(host[:, 0].sum() * 1e6 / count, 2)
        out["host_wait_us"] = round(host[:, 1].sum() * 1e6 / count, 2)
    return out


def layer_account(result, words):
    """A list way's microseconds a layer from its stats slot: the layer, its commands'
    events (by step a command a step), the thread's terms."""
    count = result["tokens"] * result["layers"]
    terms = nn.slot_terms(words)
    entries = count * len(nn.STEPS)
    ticks = nn.entry_ticks(words, entries).reshape(count, len(nn.STEPS))
    out = {
        "layer_us": round(terms["total"] / count, 2),
        "commands_us": round(terms["command"] / count, 2),
        "thread_us": round((terms["total"] - terms["command"]) / count, 2),
        "terms_us": per_entry(terms, count),
        "status": terms["status"],
        "entries_run": terms["entries"],
    }
    if result["way"] == "commands":
        steps = ticks.sum(0) / count / nn.TICKS_PER_MICROSECOND
        out["steps_us"] = dict(zip(nn.STEPS, np.round(steps, 2).tolist(), strict=True))
    return out


def main():
    args = parse()
    model = md.MODELS["qwen3-1.7b"]
    layers = args.layers or model.layers
    report = {"model": model.name, "layers": layers, "tokens": args.tokens}
    index = visible_card()
    results, ok = [], True
    with km.Session(log=log, kernel_bytes=card.CHAIN_KERNEL_BYTES) as session:
        decode = nn.Decode(session, model, layers, 2.0, log)
        executor = nn.Executor(session, decode.run)
        log(f"loaded: the list at bank {executor.bank} {executor.base:#x}")
        write_s = 0.0
        for layer in range(layers):
            write_s += decode.write_layer(
                layer, md.layer_weights(model, args.seed, layer)
            )
            cache = md.cache(model, args.seed, layer, decode.context)
            write_s += decode.write_cache(layer, *cache)
            log(f"layer {layer} and its cache written ({write_s:.2f} s of writes)")
        try:
            if args.check:
                report["check"] = check(
                    decode, executor, model, args.seed, args.lengths[0]
                )
                ok &= all(all(way.values()) for way in report["check"].values())
                log(f"check: {json.dumps(report['check'])}")
            if args.costs and ok:
                report["costs"] = costs(decode, executor)
                log(f"costs: {json.dumps(report['costs'])}")
            for length in args.lengths if args.decode and ok else ():
                for way in ("commands", "chain", "host"):
                    log(f"{way}, T {length}: {args.tokens} x {layers} layers")
                    result = measure(
                        decode, executor, index, way, args.tokens, layers, length
                    )
                    log(f"{way}: {json.dumps(result)}")
                    results.append(result)
                    end = time.monotonic() + args.pause * result["host_seconds"]
                    while (left := end - time.monotonic()) > 0:
                        time.sleep(min(left, 10.0))
        except km.KomorebiError as error:
            report["failed"] = f"{type(error).__name__}: {error}"
            log(f"failed: {report['failed']}")
            ok = False
            try:
                threads = decode.run.thread_results(timeout=0.5)
                report["threads"] = {c: r._asdict() for c, r in threads.items()}
            except km.KomorebiError as failed:
                report["threads"] = str(failed)
        decode.run.stop()
        log("run stopped")
        stats = executor.stats()
    for result in results:
        if result["slot"] is not None:
            result |= layer_account(result, stats[result["slot"]])
    for row in report.get("costs", ()):
        first, replays = row.pop("closed_slots")
        row["closed_first"] = per_entry(nn.slot_terms(stats[first]), COST_ENTRIES)
        row["closed"] = per_entry(
            nn.slot_terms(stats[replays]), COST_ENTRIES * COST_TIMES
        )
        for form in row["open"].values():
            slot = nn.slot_terms(stats[form.pop("slot")])
            form["thread"] = per_entry(slot, COST_ENTRIES) | {
                "polls": round(slot["polls"] / (COST_ENTRIES + 1), 2),
                "stale": slot["stale"],
            }
    report["decode"] = results
    report["load_avg"] = Path("/proc/loadavg").read_text().split()[:3]
    report["ok"] = ok
    print(json.dumps(report, default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
