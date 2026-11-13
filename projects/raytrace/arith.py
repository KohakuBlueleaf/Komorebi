"""Arithmetic check: the slab test, triangle test, reciprocal and nearest-hit update on
the card's vector unit against the reference's numpy expressions, bit for bit, on
scene rays with their boxes and triangles, random ones and edge cases; RECIP: 1 / a
for every mantissa of the given exponents.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from projects.raytrace import arithmetic, card, packets  # noqa: E402

K = packets.K
F = np.float32
VEC = packets.VEC
OUT_NAMES = arithmetic.OUT_NAMES


def recip_check(session, exponent, workers):
    """1 / a on the card for every mantissa of one exponent field, against numpy."""
    pages = (1 << 23) // K["kRecipPerPage"]
    directory = arithmetic.contiguous(pages, workers)
    output, profile, results, times, _ = card.run_handler(
        session,
        K["kHandlerRecip"],
        directory,
        None,
        pages,
        K["kRecipPerPage"],
        args=[exponent],
        deadline_s=20.0,
    )
    values = ((exponent << 23) | np.arange(1 << 23, dtype=np.uint32)).view(np.float32)
    with np.errstate(all="ignore"):
        want = packets.bits(F(1.0) / values)
    got = output.reshape(-1)
    bad = got != want
    normal = (want & 0x7F800000) != 0
    first_bad = np.nonzero(bad)[0][:5]
    return {
        "exponent": exponent,
        "run": card.check_run(results, workers),
        "values": int(values.size),
        "differ": int(bad.sum()),
        "differ_normal_results": int((bad & normal).sum()),
        "examples": [
            {
                "a": int(values.view(np.uint32)[row]),
                "got": int(got[row]),
                "want": int(want[row]),
            }
            for row in first_bad
        ],
        "microseconds": float(card.command_microseconds(results)[-1]),
    }


def math_check(session, generator, count, workers):
    """a / b and sqrt(x) on the card against numpy, bit for bit."""
    count = (count + VEC - 1) // VEC * VEC
    dividend, divisor, radicand = arithmetic.math_inputs(generator, count)
    pages = arithmetic.math_pages(dividend, divisor, radicand)
    directory = arithmetic.contiguous(len(pages), workers)
    output, profile, results, times, _ = card.run_handler(
        session,
        K["kHandlerMath"],
        directory,
        pages,
        len(pages),
        K["kMaOutVectors"] * VEC,
        deadline_s=20.0,
    )
    report = arithmetic.math_compare(dividend, divisor, radicand, output)
    report["run"] = card.check_run(results, workers)
    report["microseconds"] = float(card.command_microseconds(results)[-1])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=1 << 20)
    parser.add_argument("--scene", default="s10k")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--recip", type=int, nargs="*", default=[])
    parser.add_argument("--math", type=int, default=0, help="rows of division and sqrt")
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    generator = np.random.default_rng(arguments.seed)
    count = (arguments.n + VEC - 1) // VEC * VEC
    report = {"n": count, "scene": arguments.scene}
    rows, tree = arithmetic.scene_inputs(generator, count, arguments.scene)
    edge = arithmetic.edge_inputs(generator, rows, tree)
    arithmetic.bests(generator, rows)
    pages = arithmetic.input_pages(rows)
    report["edge_rows"] = int(edge.sum())
    card.log(f"inputs: {count} rows, {int(edge.sum())} edge cases, {len(pages)} pages")
    with card.open_session() as session:
        workers = card.workers(session)
        directory = arithmetic.contiguous(len(pages), workers)
        output, profile, results, times, _ = card.run_handler(
            session,
            K["kHandlerArith"],
            directory,
            pages,
            len(pages),
            K["kArOutVectors"] * VEC,
            deadline_s=20.0,
        )
        report["run"] = card.check_run(results, workers)
        report["times"] = times
        report["recip"] = [
            recip_check(session, exponent, workers) for exponent in arguments.recip
        ]
        if arguments.math:
            report["math"] = math_check(session, generator, arguments.math, workers)
    got = output.reshape(len(pages), K["kArOutVectors"], VEC).transpose(1, 0, 2)
    got = {name: got[vector].reshape(-1) for vector, name in enumerate(OUT_NAMES)}
    got = {
        name: values.view(np.int32 if name == "new_tri" else np.float32)
        for name, values in got.items()
    }
    report["check"] = arithmetic.compare(rows, got, edge)
    report["ok"] = exact(report, workers)
    print(json.dumps(report), flush=True)
    if arguments.json:
        with open(arguments.json, "w") as handle:
            json.dump(report, handle, indent=1)
    return report["ok"]


def exact(report, workers):
    """Every run clean; no bit differs but the one known kind (the sign of a zero
    exit: min(exit * scale + 0, best) gives +0 for -0); reciprocals of normals exact."""
    check = report["check"]
    ok = card.ran_cleanly(report["run"], workers)
    for name, entry in check.items():
        if isinstance(entry, dict) and "differ_bits" in entry:
            allowed = entry["differ_only_zero_sign"] if name == "exit" else 0
            ok &= entry["differ_bits"] == allowed
    ok &= check["box_hit_flag_differs"] == 0
    for recip_report in report["recip"]:
        ok &= (
            card.ran_cleanly(recip_report["run"], workers)
            and recip_report["differ_normal_results"] == 0
        )
    if "math" in report:
        math_report = report["math"]
        ok &= card.ran_cleanly(math_report["run"], workers)
        ok &= (
            math_report["divide"]["differ_bits"]
            == math_report["sqrt"]["differ_bits"]
            == 0
        )
    return bool(ok)


if __name__ == "__main__":
    started = time.time()
    ok = main()
    print(f"wall {time.time() - started:.1f}s", flush=True)
    sys.exit(0 if ok else 1)
