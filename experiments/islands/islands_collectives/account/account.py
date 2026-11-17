"""The collectives' account: km::all_gather and km::all_reduce from a kernel and from
the host, each call's time on the card and, with the stamps (KM_COLL_STAMPS), where it
goes, by island and step. Prints one JSON line a size and form; --json keeps it."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi.layout import partition

KERNELS = Path(__file__).resolve().parent / "kernels"
CONSTANTS = km.abi.constants(KERNELS / "account.h", known=km.abi.headers().constants)
TILE = 32
STAMPS = 32  # words a rank (km::kCsStamps)
# km::CollStamp, by index
NAMES = [
    "entry", "met", "posted", "arrived", "confirmed", "released", "left_arrived",
    "left_go", "left_folded", "left_passed", "right_arrived", "right_go",
    "right_folded", "right_passed", "kept_go", "kept_folded", "kept_met",
]  # fmt: skip
READ, ADD, WRITE, WORDS = 20, 21, 22, 23
SAMPLE_ABOVE = 256  # pages: larger copies are checked by sample (2 MB reads at most)
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--islands", type=int, default=0, help="card 0 split, a line")
    parser.add_argument("--sizes", default="2,256", help="pages (float32 tiles)")
    parser.add_argument("--calls", default="gather,reduce")
    parser.add_argument("--forms", default="kernel,host", help="in this order")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--stamps", action="store_true")
    parser.add_argument("--wait", type=float, default=0.5)
    parser.add_argument(
        "--store",
        default="wait",
        help="the kernel form's command stored by the wait, or first: island by "
        "island (old), the other island first (reversed), or send(); a list, in turn",
    )
    parser.add_argument(
        "--inspect", action="store_true", help="once after the load: run.inspect()"
    )
    parser.add_argument("--json")
    return parser.parse_args()


def store(run, how):
    """Stores the queued command on every island as `how` says before the wait;
    the host's microseconds each island's store took, island order."""
    took = [0.0] * len(run.runs)
    if how == "send":
        start = time.perf_counter()
        run.send()
        return [round((time.perf_counter() - start) * 1e6, 2)]
    order = range(len(run.runs))
    for island in reversed(order) if how == "reversed" else order:
        start = time.perf_counter()
        run.runs[island].store()
        took[island] = round((time.perf_counter() - start) * 1e6, 2)
    return took


def aiclk(cards):
    """Each card's AICLK now (MHz), from the driver's sysfs."""
    out = []
    for card in cards:
        path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
        try:
            out.append(int(path.read_text().split()[0]))
        except (OSError, ValueError, IndexError):
            out.append(None)
    return out


def write_copy(session, tensor, island, values):
    raw, _ = formats.to_bytes(values, tensor.dtype, tensor.layout)
    pages = raw.reshape(tensor.pages, tensor.page_bytes)
    session.sessions[island].write(tensor.parts[island], pages)


def read_copy(session, tensor, island):
    (raw,) = session.sessions[island].read(tensor.parts[island])
    return formats.from_bytes(raw, tensor.shape, tensor.dtype, tensor.layout)


def sampled(session, tensor, island, first, pages):
    """Pages first .. first + pages - 1 of island `island`'s copy, their values."""
    (raw,) = session.sessions[island].read(
        tensor.parts[island], first=first, pages=pages
    )
    shape = (pages * TILE, TILE)
    return formats.from_bytes(raw, shape, tensor.dtype, tensor.layout)


def wrong_shares(session, tensor, want):
    """[(island, share)] of the copies' shares that are not `want`'s: whole up to
    SAMPLE_ABOVE pages, else by sample (each share's first, middle and last 8)."""
    islands = len(session.sessions)
    bad = []
    for island in range(islands):
        whole = tensor.pages <= SAMPLE_ABOVE
        got = read_copy(session, tensor, island) if whole else None
        for c in range(islands):
            share = partition(range(tensor.pages), c, islands)
            if whole:
                runs = [(share.start, share.stop - share.start)]
            else:
                middle = (share.start + share.stop) // 2
                runs = [(share.start, 8), (middle, 8), (share.stop - 8, 8)]
            for start, pages in runs:
                rows = slice(start * TILE, (start + pages) * TILE)
                values = (
                    got[rows]
                    if whole
                    else sampled(session, tensor, island, start, pages)
                )
                if not np.array_equal(values, want[rows]):
                    bad.append((island, c))
                    break
    return bad


def account(session, stamps, mhz, call_us):
    """Per island: its leader's steps (us from its entry, in time order), its ranks'
    entries and releases, the folds' cycles; the call's time beside them."""
    out = []
    for island in range(len(session.sessions)):
        (raw,) = session.sessions[island].read(stamps.parts[island])
        words = np.frombuffer(raw.tobytes(), np.uint32).reshape(-1, STAMPS)
        first, count = session.graph.first(island), session.graph.count(island)
        mine = words[first : first + count].astype(np.int64)
        clock = mhz[min(island, len(mhz) - 1)] or 1350
        entry = mine[:, 0]
        base = int(entry.min())

        def us(cycles, base=base, clock=clock):
            return round(((int(cycles) - base) % 2**32) / clock, 2)

        lead = mine[0]
        steps = sorted(
            ((us(lead[k]), NAMES[k]) for k in range(len(NAMES)) if lead[k]),
            key=lambda step: step[0],
        )
        released = mine[:, 5]
        folds = {
            "read_us": round(int(mine[:, READ].max()) / clock, 2),
            "add_us": round(int(mine[:, ADD].max()) / clock, 2),
            "write_us": round(int(mine[:, WRITE].max()) / clock, 2),
            "words_a_rank": int(mine[:, WORDS].max()),
        }
        if folds["words_a_rank"]:
            k = int(np.argmax(mine[:, WORDS]))
            folds["add_cycles_a_word"] = round(
                int(mine[k, ADD]) / int(mine[k, WORDS]), 1
            )
        out.append(
            {
                "island": island,
                "leader_steps_us": steps,
                "entries_us": [0.0, us(entry.max())],
                "released_us": [us(released.min()), us(released.max())],
                "span_us": us(released.max()),
                "folds": folds,
            }
        )
    spans = [row["span_us"] for row in out]
    return {
        "call_us": call_us,
        "outside_us": round(call_us - max(spans), 2),
        "islands": out,
    }


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    opened = {"cards": cards}
    if args.islands:
        opened = {"card": cards[0], "islands": args.islands, "line": True}
    sizes = [int(size) for size in args.sizes.split(",")]
    stores_how = args.store.split(",")
    if set(stores_how) - {"wait", "old", "reversed", "send"}:
        sys.exit(f"--store {args.store}: wait, old, reversed, send")
    rows, ok = [], True
    generator = np.random.default_rng(1)
    with km.Session(**opened, log=log) as session:
        ranks, islands = session.ranks, len(session.sessions)
        made, stamps = {}, None
        for pages in sizes:  # every tensor before the load
            zeros = np.zeros((pages * TILE, TILE), np.float32)
            made[pages] = (
                session.tensor(zeros, "float32", "tile", copy=True),
                session.tensor(
                    np.concatenate([zeros, zeros]), "float32", "tile", copy=True
                ),
            )
            if stamps is None:  # the third tensor: one stamped build for any sizes
                stamps = session.tensor(
                    np.zeros((ranks, STAMPS), np.uint32), "uint32", copy=True
                )
        defines = {"KM_COLL_STAMPS": stamps.address} if args.stamps else {}
        program = km.Program(
            "collectives_account",
            kernels=KERNELS,
            buffers=[
                km.Buffer(
                    CONSTANTS["kCbFold"], "uint32", CONSTANTS["kFoldBytes"],
                    CONSTANTS["kFoldPages"],
                ),
                km.Buffer(
                    CONSTANTS["kCbStamps"], "uint32", CONSTANTS["kStampsBytes"], 1
                ),
            ],  # fmt: skip
            handlers=KERNELS / "account.h",
            defines=defines,
        )
        run = session.load(program, wait=args.wait)
        log(f"loaded: {islands} islands, {ranks} ranks")
        if args.inspect:  # every rank's state, each island's group in its own ranks
            seen = run.inspect()
            row = {
                "call": "inspect",
                "ranks": len(seen),
                "islands": sorted({one["island"] for one in seen}),
                "in_order": [one["rank"] for one in seen] == list(range(ranks)),
            }
            ok &= row["ranks"] == ranks and row["in_order"]
            rows.append(row)
            print(json.dumps(row), flush=True)
        nop = km.abi.headers()["KM_HANDLER_NOP"]  # the command's own time, no call
        card = []
        for _ in range(args.rounds if "nop" in args.calls.split(",") else 0):
            events = run.wait(run.submit(nop, (0, ranks), []))
            card.append(max(e.microseconds for e in events))
        if card:
            row = {"call": "nop", "card_us": round(float(np.median(card)), 2)}
            row["card_us_all"] = [round(c, 2) for c in card]
            rows.append(row)
            print(json.dumps(row), flush=True)
        for pages in sizes:
            tensor, scratch = made[pages]
            for call in [c for c in args.calls.split(",") if c != "nop"]:
                partials = [
                    generator.integers(-8, 9, (pages * TILE, TILE)).astype(np.float32)
                    for _ in range(islands)
                ]
                whole = np.sum(partials, axis=0) if call == "reduce" else partials[0]
                for form, how in [
                    (one, how)
                    for one in args.forms.split(",")
                    for how in (stores_how if one == "kernel" else ["wait"])
                ]:
                    card, kinds, bad, stores = [], set(), [], []
                    for _ in range(args.rounds):
                        if call == "gather":  # island c's share, zeros elsewhere
                            for c in range(islands):
                                share = partition(range(pages), c, islands)
                                mine = np.zeros_like(whole)
                                rows_ = slice(share.start * TILE, share.stop * TILE)
                                mine[rows_] = whole[rows_]
                                write_copy(session, tensor, c, mine)
                        else:
                            for c, partial in enumerate(partials):
                                write_copy(session, tensor, c, partial)
                        mhz = aiclk(cards)
                        if form == "kernel":
                            handler = "gather" if call == "gather" else "reduce"
                            words = [tensor.address, 0, pages, scratch.address, 1]
                            ticket = run.submit(handler, (0, ranks), words)
                            if how != "wait":
                                stores.append(store(run, how))
                            events = run.wait(ticket)
                        elif call == "gather":
                            events = run.all_gather(tensor)
                        else:
                            events = run.all_reduce(tensor, scratch)
                        if events is not None:
                            card.append(max(e.microseconds for e in events))
                            kinds |= {e.kind for e in events}
                        bad = bad or wrong_shares(session, tensor, whole)
                    row = {
                        "call": call, "form": form, "pages": pages,
                        "bytes": pages * 4096, "islands": islands,
                        "card_us": round(float(np.median(card)), 2) if card else None,
                        "card_us_all": [round(c, 2) for c in card],
                        "events": sorted(kinds), "wrong_shares": bad, "aiclk": mhz,
                        "store": how, "stores_us": stores,
                    }  # fmt: skip
                    if args.stamps and card:
                        row["account"] = account(session, stamps, mhz, card[-1])
                    ok &= not bad
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        errors = [r["error_name"] for r in run.stop().islands]
    ok &= set(errors) == {"none"}
    summary = {
        "ok": bool(ok),
        "kernel_errors": errors,
        "wall_s": round(time.monotonic() - STARTED, 1),
    }
    print(json.dumps(summary), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, **summary}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
