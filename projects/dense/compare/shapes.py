"""The shapes of the one-card and two-card comparison with Kohaku-TT, in the order they
are measured, the names of their result files, and the README table made from them.
`python -m projects.dense.compare.shapes list|table`."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"

# Large K against M and N first (the gain should grow with it), the shapes where K is
# small last. (M, K, N), all bfloat16.
SHAPES = [
    (8192, 65536, 8192),
    (4096, 32768, 4096),
    (2048, 16384, 2048),
    (8192, 32768, 8192),
    (4096, 16384, 4096),
    (2048, 8192, 2048),
    (4096, 16384, 16384),
    (16384, 16384, 16384),
    (8192, 16384, 8192),
    (4096, 8192, 4096),
    (2048, 4096, 2048),
    (8192, 8192, 8192),
    (4096, 4096, 4096),
    (8192, 2048, 8192),
    (16384, 4096, 16384),
]
CUBES = (4096, 8192, 16384)


def parse(name):
    """(M, K, N) of a name `MxKxN`."""
    m_rows, k_depth, n_columns = (int(part) for part in name.split("x"))
    return m_rows, k_depth, n_columns


def name(shape):
    return "x".join(str(value) for value in shape)


def card_name(cards):
    return "x".join(str(card) for card in cards)


def result_path(folder, shape, system, cards, variant=None):
    """The JSON of a point: `shape-<MxKxN>-<ours|ktt>-<0|0x1>[-<variant>].json`."""
    tail = f"-{variant}" if variant else ""
    return Path(folder) / f"shape-{name(shape)}-{system}-{card_name(cards)}{tail}.json"


def ours_path(shape, cards):
    """Our result file of a point; the cubes' are the curve's."""
    if shape[0] == shape[1] == shape[2] and shape[0] in CUBES:
        return RESULTS / f"linear-curve-{shape[0]}-{card_name(cards)}.json"
    return result_path(RESULTS, shape, "ours", cards)


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def ours_times(report):
    """(single median, loop) microseconds of one of our reports; None, None when the
    plan refused or the point failed."""
    linear = (report or {}).get("linear") or {}
    loop = linear.get("loop")
    if not loop or not linear.get("ok"):
        return None, None
    singles = sorted(loop["single_microseconds"])
    return singles[len(singles) // 2], loop["microseconds_per_call"]


def ktt_times(report):
    if not report:
        return None, None
    loop = (report.get("loop") or {}).get("microseconds_per_call")
    return report["synced_microseconds_median"], loop


def text(value, digits=1):
    return "-" if value is None else f"{value:,.{digits}f}"


def ratio(one, two):
    """One card's time over two cards': above 1, two cards are faster."""
    return None if one is None or two is None else one / two


def passes(report):
    """The slower island's put, sync and own pass in microseconds (slowest core)."""
    found = ((report or {}).get("linear") or {}).get("passes")
    if not found:
        return None
    slower = max(found, key=lambda one: sum(one["cycles"].values()))
    return slower["microseconds_at_mhz"]


def table(variant="default"):
    """The two README tables as text: times and ratios, then our crossing."""
    head = "| M x K x N | Ours 1 card single / loop, us | Ours 2 cards | Ratio "
    head += "(loop) | Kohaku-TT 1 card | Kohaku-TT 2 cards | Ratio (loop) |"
    rows = [head, "|---|---|---|---|---|---|---|"]
    crossing = [
        "| M x K x N | Crossed a call, MB | Put | Sync | Own columns | in ms |",
        "|---|---|---|---|---|---|",
    ]
    for shape in SHAPES:
        ours = [ours_times(load(ours_path(shape, c))) for c in ([0], [0, 1])]
        theirs = [
            ktt_times(load(result_path(RESULTS, shape, "ktt", c, "default")))
            for c in ([0], [0, 1])
        ]
        cells = [f"{text(one)} / {text(loop)}" for one, loop in ours + theirs]
        pair = []
        for (one_s, one_l), (two_s, two_l) in (ours[:2], theirs[:2]):
            single, looped = ratio(one_s, two_s), ratio(one_l, two_l)
            pair.append(f"{text(single, 2)} ({text(looped, 2)})")
        rows.append(
            f"| {name(shape)} | {cells[0]} | {cells[1]} | {pair[0]} | "
            f"{cells[2]} | {cells[3]} | {pair[1]} |"
        )
        two = load(ours_path(shape, [0, 1]))
        found = passes(two)
        if found:
            crossed = two["linear"]["crossed_bytes"]["blocks"] / 1e6
            times = " | ".join(
                f"{found[key] / 1e3:.2f}" for key in ("put", "sync", "own")
            )
            crossing.append(f"| {name(shape)} | {crossed:,.1f} | {times} | |")
    return "\n".join(rows + [""] + crossing)


def main():
    if sys.argv[1:] == ["list"]:
        print(*(name(shape) for shape in SHAPES), sep="\n")
    elif sys.argv[1:] == ["table"]:
        print(table())
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
