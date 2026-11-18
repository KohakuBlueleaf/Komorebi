"""The two-card decode on cards 0 and 1 (card.py): `layer` random layers, each output
against md.layer() and the layer's time with and without its crossings (stamps: each
record's, a card each); `real` the model's own weights against torch_reference.py's
tokens and hidden states. Prints one JSON line."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import komorebi as km
from projects.decode import card as one
from projects.decode import model as md
from projects.decode import real
from projects.decode.two import card as two

STARTED = time.monotonic()
SEED = 7
MICRO = one.TICKS_PER_MICROSECOND


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("layer", "real", "probe"))
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--stamps", action="store_true")
    parser.add_argument("--local", action="store_true", help="also each crossing local")
    parser.add_argument("--token", action="store_true", help="tok/s: a token a command")
    parser.add_argument(
        "--probe", default="apart", choices=("together", "serial", "apart")
    )
    parser.add_argument("--serial", action="store_true", help="island 1 posts second")
    parser.add_argument("--slots", default="1,1,3", help="reader, writer, compute")
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--json", default=None)
    return parser.parse_args()


def chain_us(decode, slot):
    """Each card's chain commands' event microseconds of a stats slot (after stop)."""
    words = decode.stats()[:, slot]
    at = md.D["DC_ST_STEPS"] + md.D["DC_ST_STEP_WORDS"] * md.D["DC_STEP_CHAIN"]
    return [(int(w[at]) | int(w[at + 1]) << 32) / MICRO for w in words]


def stamps(decode):
    """Each card's chain stamps since the last read, then zeroed: [card, ranks,
    kernels, rows, words] sums of AICLK cycles (one.Decode.chain_stamps' form)."""
    out = []
    runs = decode.run.runs
    for island, run in enumerate(runs):
        first, count = decode.firsts[island], decode.counts[island]
        tensor = decode.stamp_tensors[island]
        args = [tensor.address, 0, 0, 0, one.CHAIN["kChainStampsMode"], 0, 0, 0, 0, 0]
        event = run.wait(
            run.submit("split", (first, count), args, exclusive=True, part=True),
            timeout=5.0,
        )
        if event.kind != "done":
            raise km.KomorebiError(f"stamps of island {island}: {event}")
    for island, s in enumerate(decode.session.sessions):
        (pages,) = s.read(decode.stamp_tensors[island])
        first, count = decode.firsts[island], decode.counts[island]
        words = pages.reshape(-1).view(np.uint32).reshape(-1, pages.shape[1] // 4)
        words = words[first : first + count, : one.CHAIN["kChainStampBytes"] // 4]
        shape = (one.CHAIN["kChainStampKernels"], one.CHAIN["kChainStampRecords"], -1)
        out.append(words.reshape(count, *shape))
    return np.stack(out)


def stamp_account(words, jobs, leader):
    """Per card, a record row's medians over its ranks (us a job): taken to marked,
    to done (reader), and the leader's writer op (loaded to called) of the crossings."""
    rows = ("qkv", "attend", "o", "gate_up", "down", "G1", "G2+G4", "G3")
    w = words.astype(np.float64) / 1350.0 / jobs  # AICLK cycles at 1.35 GHz
    out = []
    for card in w:
        reader, writer = card[:, 0], card[:, 1]
        row = {}
        for r, name in enumerate(rows):
            live = reader[:, r, one.CHAIN["kChainStampDone"]] > 0
            if r >= 5:  # the leader alone
                op = writer[leader, r, 13] - writer[leader, r, 12]
                row[name] = {
                    "marked": round(writer[leader, r, 4], 2),
                    "op": round(op, 2),
                }
                continue
            if not live.any():
                continue
            row[name] = {
                "marked": round(float(np.median(reader[live, r, 4])), 2),
                "done": round(float(np.median(reader[live, r, 6])), 2),
            }
        out.append(row)
    return out


def port_words(run, core, port):
    """Port `port`'s count and want of a core, from the run's control blocks (state.py
    decodes the first KM_PORTS only)."""
    h = km.abi.headers()
    block = run._blocks[core]
    at = h["KM_CONTROL_PORT"] // 4 + port * h["KM_PORT_STRIDE"] // 4
    return [int(block[at + h["KM_PORT_COUNT"]]), int(block[at + h["KM_PORT_WANT"]])]


def failed_cores(decode):
    """After a failed job: the run stopped, each island's ranks that gave up (where,
    on which kernel) and how many commands each completed."""
    try:
        decode.run.stop()
    except km.KomorebiError as error:
        log(f"stop: {error}")
    out = []
    ports = {"left": km.abi.headers()["KM_PORT_LEFT"]}
    ports |= {"right": ports["left"] + 1, "pages": km.abi.headers()["KM_PORT_PAGES"]}
    ports["pages_writer"] = ports["pages"] + 1
    for island, run in enumerate(decode.run.runs):
        try:
            state = run.state()[: decode.counts[island]]
        except km.KomorebiError as error:
            out.append(str(error))
            continue
        cores, kinds = [], {}
        for core in state:
            gave = {k: core[k] for k in ("reader", "writer", "compute") if core[k]}
            key = repr(sorted((k, v["where"], v.get("slot")) for k, v in gave.items()))
            kinds.setdefault(key, []).append(core["core"])
            if core["core"] == decode.leader:
                leader = {"completed": core["completed"]} | gave
                leader["ports"] = {
                    n: port_words(run, core["core"], p) for n, p in ports.items()
                }
                cores.append({"leader": leader})
        cores.append({"by kind": {k: [len(v), v[:6]] for k, v in kinds.items()}})
        links = run.state()[decode.counts[island] :]
        cores.append(
            {
                "links": [
                    {"core": c["core"], "completed": c["completed"]}
                    | {k: c[k] for k in ("reader", "writer", "compute") if c[k]}
                    for c in links
                ]
            }
        )
        result = run.result()
        cores.append(
            {
                "result": {
                    k: v
                    for k, v in result.items()
                    if "name" in k or k in ("error", "slot", "sequence", "req")
                }
            }
        )
        out.append(cores)
    return out


def guarded(decode, report, name, **job):
    """decode.job(**job); on a failure the cores that gave up into report[name]."""
    try:
        return decode.job(**job)
    except km.KomorebiError as error:
        report[name] = {"error": str(error), "cores": failed_cores(decode)}
        log(f"{name}: {report[name]}")
        raise


def probe(args, session, report):
    """The crossing alone in the decode's program, sent from the host to each leader
    (no chain): both parts back to back, then island 0's first and island 1's 2 ms
    later, then the reverse. Each crossing's pages exact on the far card."""
    model = md.MODELS["qwen3-1.7b"]
    slots = dict(zip(("reader", "writer", "compute"), map(int, args.slots.split(","))))
    decode = two.TwoDecode(
        session, model, 1, 1.0, log, args.context, args.stamps, slots, wait=2.0
    )
    x = md.start(SEED, model)
    decode.write_activation("x", x)
    want = decode.read_activation("x", 0)
    words = decode.words(0)
    t = two.T
    base = [
        decode.copies["x"].address,
        0,
        32,
        words[md.D["DC_WORD_REDUCE"]] | 1 << md.D["DC_ROWS_SHIFT"],
        one.S["kSplitOp"],
        words[t["DC2_WORD_PLAN_CROSS"]] & 0xFF,
        words[t["DC2_WORD_CROSS"]],
        0,
        0,
        decode.leader,
    ]
    tries = {  # the two cards' posts: at once, island 1 second, 50 ms apart
        "together": (("together", (0, 1), 0.0, False),),
        "serial": (("serial", (0, 1), 0.0, True),),
        "apart": (("0 first", (0, 1), 0.05, False),),
    }[args.probe]
    for name, order, gap, serial in tries:
        pages = md.tile_pages(real.formats.bf16_bits(x))
        for island in (0, 1):  # each card holds only its own half before
            mine = pages.copy()
            mine[(1 - island) * 32 : (2 - island) * 32] = 0
            tensor = decode.copies["x"]
            decode.session.sessions[island].write(tensor.parts[island], mine)
        before = decode.run.crossings()  # the link cores' counts, around the try
        handles = {}
        for island in order:
            run = decode.run.runs[island]
            first = decode.firsts[island] + decode.leader
            words_i = list(base)
            words_i[1] = island * 32
            words_i[9] = decode.leader | (two.CROSS_SERIAL if serial else 0)
            handles[island] = run.submit("split", (first, 1), words_i, part=True)
            run.store()  # on its card now, not at its wait
            if gap:
                time.sleep(gap)
        events = {}
        for island, handle in handles.items():
            try:
                events[island] = decode.run.runs[island].wait(handle, timeout=3.0).kind
            except km.KomorebiError as error:
                events[island] = str(error)
        got = [decode.read_activation("x", island) for island in (0, 1)]
        report[f"probe {name}"] = {
            "events": events,
            "exact": [bool(np.array_equal(g, want)) for g in got],
            "crossed": crossed(decode.run.crossings() - before),
        }
        log(f"probe {name}: {report[f'probe {name}']}")
        if any(kind != "done" for kind in events.values()) and "cores" not in report:
            report["cores"] = [name, failed_cores(decode)]  # the first failure's
    decode.run.stop()


def crossed(diff):
    """{"link island->there": its lines, asks, pages, puts} of a Crossings difference
    (puts: put lines and their confirmations, as the link cores counted them)."""
    out = {}
    for (link, island, there), row in sorted(diff.rows.items()):
        counts = {f: row[f] for f in ("lines", "asks", "pages", "puts") if row[f]}
        if counts:
            out[f"{link} {island}->{there}"] = counts
    return out


def check(decode, want):
    """Every word of x1, a, gated and x on both cards against md.layer()'s, each half
    apart (card c's own: its output columns), and each card's q, k, v of its heads."""
    out = {}
    for name in ("a", "x1", "gated", "x"):
        for island in (0, 1):
            got = decode.read_activation(name, island)[:1]
            out[f"{name}{island}"] = md.errors(got, want[name][:1])
            half = got.shape[1] // 2
            for c in (0, 1):
                part = slice(c * half, (c + 1) * half)
                key = f"{name}{island} half{c}"
                out[key] = md.errors(got[:, part], want[name][:1, part])
    model = decode.model
    for island in (0, 1):
        (raw,) = decode.session.sessions[island].read(decode.local[island]["qkv"])
        qkv = two.untilize_row(raw)
        q_width = model.heads // 2 * model.head_dim
        kv_width = model.kv_heads // 2 * model.head_dim
        got = np.split(qkv[:1], [q_width, q_width + kv_width], axis=1)
        for name, width, values in zip("qkv", (q_width, kv_width, kv_width), got):
            mine = slice(island * width, (island + 1) * width)
            out[f"{name}{island}"] = md.errors(values, want[name][:1, mine])
    return out


def layer_mode(args, session, report):
    model = md.MODELS["qwen3-1.7b"]
    slots = dict(zip(("reader", "writer", "compute"), map(int, args.slots.split(","))))
    decode = two.TwoDecode(
        session,
        model,
        args.layers,
        1.0,
        log,
        args.context,
        args.stamps,
        slots,
        wait=2.0,
        serial=args.serial,
    )
    report["plans"] = {
        name: [p.kt, p.nt, p.s, p.groups()] for name, p in decode.plans.items()
    }
    report["keys_most"] = decode.keys_most
    position = args.context - 1
    weights = [md.layer_weights(model, SEED, layer) for layer in range(args.layers)]
    caches = [md.cache(model, SEED, layer, position) for layer in range(args.layers)]
    written = 0.0
    for layer in range(args.layers):
        written += decode.write_layer(layer, weights[layer])
        written += decode.write_cache(layer, *caches[layer])
    log(f"weights written ({written:.2f} s)")
    x = md.start(SEED, model)
    decode.write_activation("x", x)
    want = md.layer(model, x, weights[0], caches[0], rows=1)
    guarded(
        decode,
        report,
        "failed local",
        tokens=1,
        layers=1,
        position=position,
        local=True,
    )
    decode.write_activation("x", x)  # the local job wrote only its own half of x
    guarded(decode, report, "failed cross", tokens=1, layers=1, position=position)
    report["check"] = check(decode, want)
    log(f"check: { {k: round(v['rms_rel'], 4) for k, v in report['check'].items()} }")
    if args.stamps:
        stamps(decode)  # zeroed after the check
    runs = {}
    forms = [("cross", False)] + ([("local", True)] if args.local else [])
    for name, local in forms:
        slots_used = []
        for _ in range(args.reps):
            slots_used.append(decode.slot)
            decode.job(1, args.layers, position=position, local=local, token=args.token)
        runs[name] = slots_used
        if args.stamps:
            jobs = args.reps * (1 if args.token else args.layers)
            report[f"stamps {name}"] = stamp_account(
                stamps(decode), jobs, decode.leader
            )
    decode.run.stop()
    for name, slots_used in runs.items():
        per = np.array([chain_us(decode, slot) for slot in slots_used])  # [reps, cards]
        report[f"{name} chain us a job"] = {
            "median": np.round(np.median(per, axis=0), 2).tolist(),
            "least": np.round(per.min(axis=0), 2).tolist(),
        }
        if args.token:
            report[f"{name} tok/s"] = round(1e6 / float(np.median(per.max(axis=1))), 1)


def real_mode(args, session, report):
    reference = np.load(args.reference)
    prompt, want = reference["prompt"].tolist(), reference["tokens"].tolist()
    hidden = reference["hidden"]
    model = md.from_config(Path(args.snapshot) / "config.json")
    checkpoint = real.Checkpoint(args.snapshot)
    embedding = checkpoint.get("model.embed_tokens.weight")
    head = real.formats.bf16_values(checkpoint.get("lm_head.weight"))
    last_norm = real.formats.bf16_values(checkpoint.get("model.norm.weight"))
    steps = len(prompt) + len(want) - 1
    slots = dict(zip(("reader", "writer", "compute"), map(int, args.slots.split(","))))
    decode = two.TwoDecode(
        session,
        model,
        model.layers,
        1.0,
        log,
        args.context,
        False,
        slots,
        serial=args.serial,
    )
    written = 0.0
    empty = np.zeros((model.kv_heads, 0, model.head_dim), np.float32)
    for layer in range(model.layers):
        written += decode.write_layer(layer, real.layer_weights(checkpoint, layer))
        written += decode.write_cache(layer, empty, empty)
    log(f"weights written ({written:.2f} s)")
    tokens, per_layer, step_us = list(prompt), [], []
    try:
        for position in range(steps):
            decode.slot = 0
            x = np.zeros((md.TILE, model.hidden), np.float32)
            x[0] = real.formats.bf16_values(embedding[tokens[position]])
            decode.write_activation("x", x)
            if position < hidden.shape[0]:
                errors = []
                for layer in range(model.layers):
                    decode.job(1, 1, first=layer, position=position, token=args.token)
                    decode.slot = 0
                    row = decode.read_activation("x", layer % 2)[0]
                    if layer == model.layers - 1:
                        row = real.final_norm(row, last_norm, model.eps)
                    errors.append(real.relative_rms(row, hidden[position, layer]))
                per_layer.append(errors)
                log(f"position {position}: per layer {np.round(errors, 4).tolist()}")
            else:
                ran = decode.job(1, model.layers, position=position, token=args.token)
                step_us.append(max(ran["thread_ticks"]) / MICRO)
            row = decode.read_activation("x", position % 2)[0]
            logits = head @ real.final_norm(row, last_norm, model.eps)
            if position >= len(prompt) - 1:
                tokens.append(int(np.argmax(logits)))
                log(f"position {position}: token {tokens[-1]}")
    except km.KomorebiError as error:
        report["failed"] = f"{type(error).__name__}: {error}"
        log(f"failed: {report['failed']}")
    decode.run.stop()
    got = tokens[len(prompt) :]
    report |= {
        "card_tokens": got,
        "reference_tokens": want,
        "first_different": next(
            (i for i, (a, b) in enumerate(zip(got, want)) if a != b), None
        ),
        "per_layer_rms": per_layer,
        "step_us": step_us,
        "tokens_per_second": (
            round(1e6 / float(np.median(step_us)), 1) if step_us else None
        ),
    }


def main():
    args = parse()
    report = {
        "mode": args.mode,
        "context": args.context,
        "slots": args.slots,
        "serial": args.serial,
    }
    report |= {"layers": args.layers, "stamps": args.stamps, "token": args.token}
    with km.Session(cards=[0, 1], log=log) as session:
        try:
            if args.mode == "probe":
                probe(args, session, report)
            elif args.mode == "layer":
                layer_mode(args, session, report)
            else:
                real_mode(args, session, report)
        except km.KomorebiError as error:
            report["failed"] = str(error)
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print("TWO", json.dumps(report), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    sys.exit(0 if "failed" not in report else 1)


if __name__ == "__main__":
    main()
