"""The README's tables of where SUM, HSUM and PLACE spend their time, from a run of
run_timed.py (all its loads): each stage on the slowest core, the region class's
route beside the hand's text, both stamped; and the CPUs that run beside mover B.
Example: python timed_tables.py build/timed.npz"""

import sys

import numpy as np

MHZ = 1350.0  # the Tensix clock the trace and the stamps count


def wall(words, base):
    """Microseconds after `base` of 32-bit wall-clock words."""
    return ((np.asarray(words).astype(np.int64) - base) % (1 << 32)) / MHZ


def frame(kept, who, name):
    """One stamped command's slowest core: its trace, its CPUs' stamps."""
    trace = kept[f"{who}/{name}_stamped/0/trace"].astype(np.int64)
    if not len(trace):  # a prebuild's run: its trace is empty
        raise SystemExit("timed_tables.py: no trace rows (reads were zeros)")
    stamps = kept[f"{who}/{name}_stamped/stamps"].astype(np.int64)
    base = trace[:, 0].min()
    seen, done, sent = (wall(trace[:, k], base) for k in range(3))
    core = int(np.argmax(done))
    a, b, p = (stamps[core, 16 * k : 16 * k + 16] for k in range(3))
    return {
        "core": core,
        "host": float(kept[f"{who}/{name}_stamped/us"][0]),
        "sent": sent[0],
        "seen": seen[core],
        "done": done[core],
        "a0": wall(a[0], base),
        "b0": wall(b[0], base),
        "b_end": wall(b[15], base),
        "p0": wall(p[0], base),
        "p_first": wall(p[1], base),
        "p_last": wall(p[3], base),
        "a": a / MHZ,
        "b": b / MHZ,
        "p": p / MHZ,
        "items": int(p[4]),
    }


def stages(name, c, h):
    """(stage, CPU, class us, hand us) along mover B of the slowest core."""
    out = [
        ("submit to the first take, completion back", "host", c["host"] - c["sent"],
         h["host"] - h["sent"]),
        ("the reader takes the command", "mover A", c["seen"], h["seen"]),
        ("mover B's handler starts", "mover B", c["b0"] - c["seen"],
         h["b0"] - h["seen"]),
    ]  # fmt: skip
    c_span, h_span = c["b_end"] - c["b0"], h["b_end"] - h["b0"]
    if name == "place":
        c_lines = (c["b"][1], c["b"][2], c["b"][3])
        h_lines = (h["b"][1], h["b"][3] - h["b"][4] - h["b"][2], h["b"][2])
        out += [
            ("the region's reads land", "mover B", c_lines[0], h_lines[0]),
            ("the body after the reads", "mover B", c_lines[1], h_lines[1]),
            ("directory and end slot written", "mover B", c_lines[2], h_lines[2]),
            ("the handler's other lines", "mover B", c_span - sum(c_lines),
             h_span - sum(h_lines)),
        ]  # fmt: skip
    else:
        h_issue = h["b"][3] - h["b"][1] - h["b"][2]
        out += [
            ("prefixes written (hand: reads issued, sums, writes issued)", "mover B",
             c["b"][5], h_issue),
            ("read wait", "mover B", 0.0, h["b"][1]),
            ("flush wait", "mover B", 0.0, h["b"][2]),
            ("totals written", "mover B", c["b"][7], 0.0),
            ("the handler's other lines", "mover B", c_span - c["b"][5] - c["b"][7],
             h_span - h["b"][3]),
        ]  # fmt: skip
    return out + [
        ("handler end to outputs written", "mover B", c["done"] - c["b_end"],
         h["done"] - h["b_end"]),
        ("slowest core done to the completion", "leader", c["sent"] - c["done"],
         h["sent"] - h["done"]),
    ]  # fmt: skip


def main():
    kept = np.load(sys.argv[1])
    for name in ("sum", "hsum", "place"):
        c, h = frame(kept, "route", name), frame(kept, "hand_route", name)
        print(f"\n{name.upper()}: class core {c['core']}, hand's text core {h['core']}")
        print(
            "| Stage | CPU | Class us | Hand us | Difference |\n|---|---|---|---|---|"
        )
        rows = stages(name, c, h)
        for stage, cpu, class_us, hand_us in rows:
            print(f"| {stage} | {cpu} | {class_us:.2f} | {hand_us:.2f} | "
                  f"{class_us - hand_us:+.2f} |")  # fmt: skip
        total_c, total_h = sum(r[2] for r in rows), sum(r[3] for r in rows)
        print(
            f"| command | | {total_c:.2f} | {total_h:.2f} | {total_c - total_h:+.2f} |"
        )
        if name == "place":
            print(f"beside: mover A's fill {c['a'][1]:.2f}")
            continue
        span = c["p_last"] - c["p_first"]
        print(
            f"beside: mover A's moves {c['a'][5]:.2f} (moving {c['a'][6]:.2f}); pack's "
            f"first item at {c['p_first']:.2f}, {c['items']} items, sums "
            f"{c['p'][2]:.2f}, waits and hand-overs {span - c['p'][2]:.2f}"
        )


if __name__ == "__main__":
    main()
