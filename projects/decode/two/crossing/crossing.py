"""One crossing of the two-card decode alone, on cards 0 and 1: V1 (the roots mark
their card's leader, which puts the card's half across as one line) or V2 (each root
puts its block across); the far leader counts the half and releases its card. Beside
km::all_gather of the same pages; prints one JSON line."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

KERNELS = Path(__file__).resolve().parent / "kernels"
PAGE = 2048  # a bf16 tile page
STAMPS = 16  # words a rank
WAYS = {"V1": 1, "V2": 2}
NAMES = ("entry", "written", "marked", "met", "posted", "arrived", "released")
NAMES += ("confirmed",)
# The decode's four crossings of a layer (form A, a card's half): roots, their stride
# among the card's ranks, a root's pages (the product's groups, as plans() makes them)
CROSSINGS = {"G2": (4, 16, 8), "G3": (24, 4, 4)}
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--crossings", default="G2,G3")
    parser.add_argument("--wait", type=float, default=0.5, help="wait limit, seconds")
    return parser.parse_args()


def card_us(events):
    """A command's microseconds on the card (submit to done): each island's."""
    return [event.microseconds for event in events]


def read_part(session, tensor, island):
    (raw,) = session.sessions[island].read(tensor.parts[island])
    return raw.view(np.uint32).reshape(tensor.pages, -1)


def exact(session, x, half, salt):
    """Words wrong in each island's copy (word w of page p: salt + p x 512 + w)."""
    want = salt + np.arange(2 * half * PAGE // 4, dtype=np.uint64)
    want = (want % (1 << 32)).astype(np.uint32).reshape(2 * half, -1)
    return [int(np.sum(read_part(session, x, i)[: 2 * half] != want)) for i in range(2)]


def account(session, stamps, graph, roots, stride, salt):
    """Per island, from its leader's entry (its card's clock, cycles): the leader's
    stages, the roots' median written and marked, the ranks' median release."""
    out = {}
    for island in range(2):
        words = read_part(session, stamps, island).astype(np.int64)
        first, count = graph.first(island), graph.count(island)
        mine = words[first : first + count]
        if not np.all(mine[:, 8] == salt):
            out[f"island {island}"] = "stale stamps"
            continue
        base = mine[0, 0]
        rel = (mine[:, :8] - base) % (1 << 32)
        rel = np.where(mine[:, :8] == 0, -1, rel)
        leader = {
            name: int(rel[0, k]) for k, name in enumerate(NAMES) if rel[0, k] >= 0
        }
        index = np.arange(count)
        is_root = (index % stride == 0) & (index // stride < roots)
        median = {}
        for k, name in enumerate(NAMES):
            for who, rows in (("roots", is_root), ("others", ~is_root)):
                values = rel[rows, k]
                values = values[values >= 0]
                if len(values):
                    median[f"{who} {name}"] = int(np.median(values))
        out[f"island {island}"] = {"leader": leader, "median": median}
    return out


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    report = {"cards": cards, "runs": args.runs}
    with km.Session(cards=cards, log=log) as session:
        ranks, graph = session.ranks, session.graph
        half_most = max(r * p for r, _, p in CROSSINGS.values())
        x = session.tensor(
            np.zeros((32, 32 * 2 * half_most), np.float32),
            "bfloat16",
            "tile",
            copy=True,
        )
        stamps = session.tensor(np.zeros((ranks, STAMPS), np.int32), "int32", copy=True)
        block = max(p for _, _, p in CROSSINGS.values())
        program = km.Program(
            "crossing",
            kernels=KERNELS,
            buffers=[
                km.Buffer(0, "bfloat16", PAGE, block),
                km.Buffer(1, "uint32", 4 * STAMPS, 1),
            ],
            handlers=KERNELS / "crossing.h",
        )
        run = session.load(program, wait=args.wait)
        salt = 1 << 20
        for name in args.crossings.split(","):
            roots, stride, pages = CROSSINGS[name]
            half = roots * pages
            for way in ("V1", "V2", "gather"):
                handler = "gather" if way == "gather" else "cross"
                times = []
                for _ in range(args.runs):
                    salt += 1 << 20
                    words = [x.address, stamps.address, half, roots, stride, pages]
                    words += [WAYS.get(way, 0), salt]
                    times.append(
                        card_us(run.wait(run.submit(handler, (0, ranks), words)))
                    )
                row = {
                    "card us": round(float(np.median([max(t) for t in times])), 2),
                    "islands us": [
                        round(float(np.median([t[i] for t in times])), 2)
                        for i in (0, 1)
                    ],
                    "pages a half": half,
                }
                if way != "gather":
                    row["wrong"] = exact(session, x, half, salt)
                row["stamps"] = account(session, stamps, graph, roots, stride, salt)
                report[f"{name} {way}"] = row
                log(f"{name} {way}: {row['card us']} us, {row.get('wrong')}")
        report["kernel"] = [r["error_name"] for r in run.stop().islands]
    report["ok"] = set(report["kernel"]) == {"none"} and all(
        row.get("wrong", [0, 0]) == [0, 0]
        for key, row in report.items()
        if isinstance(row, dict)
    )
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print("CROSSING", json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
