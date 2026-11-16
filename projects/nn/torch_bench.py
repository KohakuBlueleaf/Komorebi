"""Komorebi-nn under torch: model A (the MLP half) and model B (the decode layer) eager
and compiled at three stages, each against torch float32 on the CPU, a call's time
split into named terms. Off the card (--prepare) it writes the inputs, the references
and torch's CPU times; on the card it reads them. Prints one JSON line."""

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import komorebi as km  # noqa: E402
from komorebi import abi, formats  # noqa: E402
from komorebi.bench import power  # noqa: E402
from komorebi.session.parts import transfer as transfer_part  # noqa: E402
from projects.decode import model as md  # noqa: E402
from projects.decode import real  # noqa: E402
from projects.nn import front, models, nn, table  # noqa: E402

MODEL = md.MODELS["qwen3-1.7b"]
CONTEXT = 4096  # the cache's rows a KV head, as the decode bench's
CHECK_RMS, CHECK_MAX = 2e-2, 5e-2  # decode's bound (its README), fixed before the run
# The compiled stages measured: no fusion and no chain; fusion; fusion and the chain.
STAGES = {
    "compile": {"fuse": False, "chain": False},
    "fused": {"fuse": True, "chain": False},
    "chained": {"fuse": True, "chain": True},
    "joint": {"fuse": True, "chain": True, "joint": True},  # the compile's removal
    # a call as one host submission: the list pulls x and pushes y
    "one": {"fuse": True, "chain": True, "joint": True, "one": True},
    # the same with the plain transfers' spin (300 us), the long one's comparison
    "one_spin300": {
        "fuse": True,
        "chain": True,
        "joint": True,
        "one": True,
        "spin": 3e-4,
    },
}
TERMS = (  # the host's terms (table.Device.clock), seconds summed a call
    "in_write",
    "squares",
    "dispatch",
    "planner",
    "allocate",
    "build",
    "store",
    "fence",
    "link",
    "open",
    "sync",
    "run",
    "fallback_read",
    "host",
    "fallback_write",
    "write",
    "out_read",
    "list_write",
    "weights",
    "in_pages",
    "copy_in",
    "submit",
    "submit_queue",
    "submit_hand",
    "submit_store",
    "spin",
    "copy_out",
    "out_values",
)
INSIDE_BODY = (
    "copy_in",
    "submit_queue",
    "submit_hand",
    "submit_store",
    "spin",
    "copy_out",
)
INSIDE_DISPATCH = (  # what the eager dispatch's own time holds besides itself
    "planner",
    "allocate",
    "build",
    "store",
    "fence",
    "link",
    "open",
    "sync",
    "fallback_read",
    "host",
    "fallback_write",
)
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prepare", action="store_true", help="off the card")
    parser.add_argument("--data", required=True, help="the .npz --prepare writes")
    parser.add_argument("--snapshot", default=None, help="the model's snapshot folder")
    parser.add_argument("--models", nargs="*", default=["a", "b"], choices=["a", "b"])
    parser.add_argument("--ways", nargs="*", default=["eager", *STAGES])
    parser.add_argument("--calls", type=int, default=8, help="timed calls a way")
    parser.add_argument("--mirror", action="store_true", help="eager: host mirrors")
    parser.add_argument("--tag", default="", help="a way's name suffix in the report")
    parser.add_argument("--position", type=int, default=CONTEXT - 1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=16, help="torch's, --prepare")
    parser.add_argument("--channel", action="store_true", help="plain transfers, guard")
    parser.add_argument("--json", default=None)
    return parser.parse_args()


def build(name, args):
    """Model `name`'s bf16 module and its input (bf16 values)."""
    if name == "a":
        module = models.ModelA(MODEL, models.model_a_weights(MODEL, args.seed))
        x = md.start(args.seed, MODEL)
    else:
        weights = real.layer_weights(real.Checkpoint(args.snapshot), 0)
        keys, values = models.cache(MODEL, args.seed, args.position, CONTEXT)
        module = models.ModelB(
            MODEL, models.model_b_weights(weights), keys, values, args.position
        )
        x = md.start(args.seed, MODEL)[:1]
    return module.eval(), torch.from_numpy(x).to(torch.bfloat16)


def prepare(args):
    """The inputs, torch float32's outputs and its CPU time a call (args.threads)."""
    torch.set_num_threads(args.threads)
    out = {}
    for name in args.models:
        module, x = build(name, args)
        reference = copy.deepcopy(module).float()
        with torch.no_grad():
            want = reference(x.float())
            times = []
            for _ in range(args.calls + 1):
                started = time.perf_counter()
                reference(x.float())
                times.append(time.perf_counter() - started)
        out[f"{name}_x"] = x.float().numpy()
        out[f"{name}_want"] = want.numpy()
        out[f"{name}_cpu_us"] = np.array(times[1:]) * 1e6
        log(f"model {name}: torch float32 CPU {np.mean(times[1:]) * 1e6:.1f} us a call")
    np.savez(args.data, **out)


def snapshot(device):
    return dict(device.clock), Counter(device.fallbacks), len(device.runs)


def series(device, call, count):
    """`call` once (the first: its compile, its weights) then `count` times, each
    call's host terms, fallbacks, list runs and wall: (first, [calls], last output)."""
    rows = []
    for _ in range(count + 1):
        clock, fallbacks, runs = snapshot(device)
        started = time.perf_counter()
        out = call()
        wall = time.perf_counter() - started
        device.collect()  # a one-submission call's RUN event, off its path
        rows.append(
            {
                "wall": wall,
                "clock": {
                    name: device.clock[name] - clock.get(name, 0.0)
                    for name in (*TERMS, "call_body")
                },
                "fallbacks": dict(Counter(device.fallbacks) - fallbacks),
                "runs": list(range(runs, len(device.runs))),
            }
        )
    return rows[0], rows[1:], out


def run_model(device, name, args, data):
    module, x = build(name, args)
    want = data[f"{name}_want"]
    out = {"cpu_us": round(float(np.mean(data[f"{name}_cpu_us"])), 1)}
    torch.set_num_threads(1)
    for way in args.ways:
        log(f"model {name}: {way}")
        device.mirror = args.mirror and way == "eager"
        with torch.no_grad():
            if way == "eager":
                card_module = front.card_module(copy.deepcopy(module), device)
                first, rows, got = series(
                    device,
                    lambda m=card_module: m(front.to_card(x, device)).numpy(),
                    args.calls,
                )
                described = {"mirror": device.mirror}
            else:
                torch._dynamo.reset()
                options = {"device": device, **STAGES[way]}
                compiled = torch.compile(module, backend="komorebi", options=options)
                first, rows, got = series(
                    device,
                    lambda c=compiled: c(x).float().numpy(),
                    args.calls,
                )
                described = {}
        if way.startswith("one"):
            described = {"spans": list(device.ones[-1].spans)}
        errors = md.errors(np.asarray(got, np.float32), want)
        ok = errors["rms_rel"] <= CHECK_RMS and errors["max_rel"] <= CHECK_MAX
        out[way + args.tag] = {
            "errors": errors,
            "ok": ok,
            "first": first,
            "calls": rows,
            **described,
        }
        log(
            f"model {name} {way}: {json.dumps(errors)} ok {ok}, a call "
            f"{np.mean([r['wall'] for r in rows]) * 1e6:.1f} us"
        )
    return out


def random_pages(rng, columns):
    """A [32, columns] tile tensor's pages of random bf16 values."""
    values = rng.standard_normal((table.TILE, columns)).astype(np.float32)
    return md.tile_pages(formats.bf16_bits(values))


def plain_transfers(device, calls):
    """A plain Session.write and read of x's joint pages (72) and y's (64), without the
    words' spin (WORDS_SPIN_S 0, the old wait) and with it: us per call, mean and
    median over `calls`."""
    session, out = device.session, {}
    rng = np.random.default_rng(3)
    for name, columns in (("x_joint", 2304), ("y", 2048)):
        tensor = session.empty((table.TILE, columns), "bfloat16", "tile")
        pages = random_pages(rng, columns)
        for spin in (0.0, transfer_part.WORDS_SPIN_S):
            kept, transfer_part.WORDS_SPIN_S = transfer_part.WORDS_SPIN_S, spin
            writes, reads = [], []
            for _ in range(calls + 1):
                started = time.perf_counter()
                session.write(tensor, pages)
                written = time.perf_counter()
                (got,) = session.read(tensor)
                writes.append(written - started)
                reads.append(time.perf_counter() - written)
            transfer_part.WORDS_SPIN_S = kept
            if not np.array_equal(got, pages):
                raise km.KomorebiError(f"{name}: the read is not the write")
            out[f"{name}_spin_{round(spin * 1e6)}us"] = {
                "pages": tensor.pages,
                "write_us": [round(np.mean(writes[1:]) * 1e6, 1),
                             round(np.median(writes[1:]) * 1e6, 1)],
                "read_us": [round(np.mean(reads[1:]) * 1e6, 1),
                            round(np.median(reads[1:]) * 1e6, 1)],
            }  # fmt: skip
        tensor.free()
    return out


def guard(device):
    """A thread's PULL and PUSH with the session's region word (pages back as sent),
    then each with the whole channel's region word: refused, the region untouched.
    Returns what was seen; the refusals' tick words are read after the stop."""
    session, h = device.session, abi.headers()
    tensor = session.empty((table.TILE, 2048), "bfloat16", "tile")
    rng = np.random.default_rng(5)
    pages = random_pages(rng, 2048)
    one = table.OneSubmission(device, nn.Entries(0), [tensor], [tensor])
    pull, push = one.local[id(one.pulls[0])], one.local[id(one.pushes[0])]
    (back,) = one(
        lambda: pull.__setitem__(slice(None), pages.reshape(-1)),
        lambda: [push.reshape(pages.shape).copy()],
    )
    device.collect()
    out = {"right_region_round_trip": bool(np.array_equal(back, pages))}
    out["refused"] = []
    for move in one.moves:
        args = list(move.call.args)
        args[h["KM_TRANSFER_REGION"]] = h["KM_CHANNEL_BYTES"]
        entries = nn.Entries(device.reserve(1))
        entries.command(move.call._replace(args=tuple(args)), nn.NN["NN_KIND_TRANSFER"])
        device.write_list(entries)
        region = session.channel[: session._region.bytes]
        region[move.offset : move.offset + tensor.pages * tensor.aligned_page_bytes] = (
            0xA5
        )
        words = one.channel_words(move.words, move.count)
        words[:] = 0xA5A5A5A5
        before = (np.array(region[move.offset : move.offset + 4096]), words.copy())
        try:
            device.executor.closed(entries)
            result = "ran"
        except km.KomorebiError as error:
            result = str(error)
        out["refused"].append(
            {
                "handler": int(move.call.handler),
                "run": result,
                "entry": entries.first,
                "slot": device.executor.slot - 1,
                "data_unchanged": bool(
                    np.array_equal(region[move.offset : move.offset + 4096], before[0])
                ),
                "words_unchanged": bool(np.array_equal(words, before[1])),
            }
        )
    device.ones.remove(one)
    tensor.free()
    return out


def account(result, stats):
    """A way's terms in us a call (means over its timed calls), with its card time
    from the stats (the commands' events, the thread's total) and what is left."""
    rows = result["calls"]
    count = len(rows)
    terms = {
        name: sum(row["clock"][name] for row in rows) * 1e6 / count for name in TERMS
    }
    if terms["dispatch"]:
        terms["dispatch"] -= sum(terms[name] for name in INSIDE_DISPATCH)
    body = sum(row["clock"].get("call_body", 0.0) for row in rows) * 1e6 / count
    if body:  # one submission: the call's own Python beside torch's wrappers
        terms["call_python"] = body - sum(terms[name] for name in INSIDE_BODY)
    wall = sum(row["wall"] for row in rows) * 1e6 / count
    runs = [index for row in rows for index in row["runs"]]
    if all(index in stats for index in runs):
        card_us = sum(stats[index]["command"] for index in runs) / count
        thread_us = sum(stats[index]["total"] for index in runs) / count
        idle_us = sum(stats[index]["idle"] for index in runs) / count
    else:
        card_us = thread_us = idle_us = None
    fallbacks = Counter()
    for row in rows:
        fallbacks.update(row["fallbacks"])
    host = sum(terms[name] for name in TERMS)
    ones = [stats[index] for index in runs if "pull" in stats.get(index, {})]
    one = {}
    if ones:
        thread = {
            name: round(sum(run[name] for run in ones) / len(ones), 2)
            for name in nn.TERMS
        }
        one = {
            "pull_us": round(sum(r["pull"] for r in ones) / len(ones) / 50, 2),
            "push_us": round(sum(r["push"] for r in ones) / len(ones) / 50, 2),
            "thread_terms_us": thread,
            "card_span_cycles": round(float(np.mean(result.get("spans", [0]))), 1),
        }
    return {
        "wall_us": round(wall, 1),
        "terms_us": {name: round(value, 1) for name, value in terms.items() if value},
        "python_us": round(wall - host, 1),
        **({"outside_call_us": round(wall - body, 1)} if body else {}),
        "card_commands_us": None if card_us is None else round(card_us, 1),
        "thread_us": None if thread_us is None else round(thread_us, 1),
        "thread_idle_us": None if idle_us is None else round(idle_us, 1),
        "lists_a_call": round(len(runs) / count, 2),
        "fallbacks_a_call": {k: v / count for k, v in sorted(fallbacks.items())},
        "first_call_us": round(result["first"]["wall"] * 1e6, 1),
        **({"one": one} if one else {}),
    }


def main():
    args = parse()
    if args.prepare:
        prepare(args)
        return
    data = np.load(args.data)
    report = {"model": MODEL.name, "position": args.position, "calls": args.calls}
    results = {}
    with km.Session(log=log) as session:  # its kernel buffer follows the kernels
        device = table.Device(session, MODEL, 2.0, log)
        log("loaded")
        try:
            if args.channel:
                report["plain"] = plain_transfers(device, args.calls * 4)
                log(f"plain transfers: {json.dumps(report['plain'])}")
                report["guard"] = guard(device)
                log(f"guard: {json.dumps(report['guard'])}")
            for name in args.models:
                results[name] = run_model(device, name, args, data)
        except km.KomorebiError as error:
            report["failed"] = f"{type(error).__name__}: {error}"
            log(f"failed: {report['failed']}")
        device.decode.run.stop()
        log("run stopped")
        stats = device.stats()
        if "guard" in report:  # the refused transfers' tick words: kind | reason << 8
            words = device.executor.stats()
            for refused in report["guard"]["refused"]:
                at = nn.NN["NN_ST_ENTRY_TICKS"] + refused["entry"]
                refused["event_word"] = int(words[refused["slot"]][at])
    for name, ways in results.items():
        for way, result in ways.items():
            if isinstance(result, dict):
                result["account"] = account(result, stats)
                log(f"{name} {way}: {json.dumps(result['account'])}")
                del result["calls"], result["first"]
    report["results"] = results
    guarded = report.get("guard")
    if guarded is not None:
        refusal = 3 | 1 << 8  # KM_EVENT_REJECTED, reason 1 (KMD_REJECT_HANDLER)
        guarded["ok"] = guarded["right_region_round_trip"] and all(
            r["run"] != "ran"
            and r["data_unchanged"]
            and r["words_unchanged"]
            and r.get("event_word") == refusal
            for r in guarded["refused"]
        )
    report["ok"] = (
        (guarded is None or guarded["ok"])
        and "failed" not in report
        and all(
            way["ok"]
            for ways in results.values()
            for way in ways.values()
            if isinstance(way, dict)
        )
    )
    print(json.dumps(power.clocked(report), default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
