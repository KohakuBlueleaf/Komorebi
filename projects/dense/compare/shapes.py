"""The shapes of the one-card and two-card comparison with Kohaku-TT, in the order they
are measured, the names of their result files, and the README table made from them.
`python -m projects.dense.compare.shapes list|table`."""

import json
import statistics
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


def ktt_report(shape, cards, variant="default"):
    """Kohaku-TT's report of a point: its single call is the nine-call run's median
    where `results/singles9` has one (the first calls after the first product are
    slow), its loop that of the first run."""
    first = load(result_path(RESULTS, shape, "ktt", cards, variant))
    nine = load(result_path(RESULTS / "singles9", shape, "ktt", cards, variant))
    return first, nine


def ktt_times(shape, cards, variant="default"):
    """(single, loop) microseconds; single is the steady median of nine calls when
    one was run."""
    first, nine = ktt_report(shape, cards, variant)
    if not first:
        return None, None
    single = (nine or first)["synced_microseconds_median"]
    if nine:  # the nine-call median skips the slow calls after the first product
        single = statistics.median(nine["synced_microseconds"][3:])
    return single, first["loop"]["microseconds_per_call"]


def text(value, digits=1):
    return "-" if value is None else f"{value:,.{digits}f}"


def ratio(one, two):
    """One card's time over two cards': above 1, two cards are faster."""
    return None if one is None or two is None else one / two


def clock(report):
    """Mean AICLK of a loop, MHz: the lower of the cards'."""
    loop = ((report or {}).get("linear") or report or {}).get("loop") or {}
    means = [value[1] for value in (loop.get("aiclk_mhz") or {}).values()]
    return round(min(means)) if means else None


def pair(one, two):
    """'single (loop)' of a ratio pair of (single, loop) tuples."""
    return f"{text(ratio(one[0], two[0]), 2)} ({text(ratio(one[1], two[1]), 2)})"


def tables():
    """The README tables: times and ratios, then the clocks of the loops."""
    head = "| M x K x N | Ours 1 card | Kohaku-TT 1 card "
    head += "| Kohaku-TT 2 cards | ratio | Kohaku-TT 2 cards, K split | ratio |"
    rows = [head, "|---|---|---|---|---|---|---|"]
    clocks = [
        "| M x K x N | AICLK mean, MHz: ours 1, Kohaku-TT 1, Kohaku-TT 2 |",
        "|---|---|",
    ]
    for shape in SHAPES:
        one = load(ours_path(shape, [0]))
        k1 = ktt_times(shape, [0])
        k2 = ktt_times(shape, [0, 1])
        ks = ktt_times(shape, [0, 1], "ksplit")
        cells = [f"{text(s)} / {text(loop)}" for s, loop in (ours_times(one), k1, k2)]
        ksplit_cell = f"{text(ks[0])} / {text(ks[1])}"
        rows.append(
            f"| {name(shape)} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{pair(k1, k2)} | {ksplit_cell} | {pair(k1, ks)} |"
        )
        means = [
            clock(one),
            clock(load(result_path(RESULTS, shape, "ktt", [0], "default"))),
            clock(load(result_path(RESULTS, shape, "ktt", [0, 1], "default"))),
        ]
        clocks.append(f"| {name(shape)} | {' / '.join(text(c, 0) for c in means)} |")
    return "\n".join(rows + [""] + clocks)


def main():
    if sys.argv[1:] == ["list"]:
        print(*(name(shape) for shape in SHAPES), sep="\n")
    elif sys.argv[1:] == ["table"]:
        print(tables())
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
