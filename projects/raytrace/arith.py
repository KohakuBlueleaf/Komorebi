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


def recip_check(session, exponent, n_workers):
    """1 / a on the card for every mantissa of one exponent field, against numpy."""
    pages = (1 << 23) // K["kRecipPerPage"]
    directory = arithmetic.contiguous(pages, n_workers)
    out, prof, res, times, _ = card.run_handler(
        session, K["kHandlerRecip"], directory, None, pages, K["kRecipPerPage"],
        args=[exponent], deadline_s=20.0,
    )  # fmt: skip
    a = ((exponent << 23) | np.arange(1 << 23, dtype=np.uint32)).view(np.float32)
    with np.errstate(all="ignore"):
        want = packets.bits(F(1.0) / a)
    got = out.reshape(-1)
    bad = got != want
    normal = (want & 0x7F800000) != 0
    idx = np.nonzero(bad)[0][:5]
    return {
        "exponent": exponent,
        "run": card.check_run(res, n_workers),
        "values": int(a.size),
        "differ": int(bad.sum()),
        "differ_normal_results": int((bad & normal).sum()),
        "examples": [
            {"a": int(a.view(np.uint32)[i]), "got": int(got[i]), "want": int(want[i])}
            for i in idx
        ],
        "microseconds": float(card.command_microseconds(res)[-1]),
    }


def math_check(session, rng, n, n_workers):
    """a / b and sqrt(x) on the card against numpy, bit for bit."""
    n = (n + VEC - 1) // VEC * VEC
    a, b, x = arithmetic.math_inputs(rng, n)
    pages = arithmetic.math_pages(a, b, x)
    directory = arithmetic.contiguous(len(pages), n_workers)
    out, prof, res, times, _ = card.run_handler(
        session, K["kHandlerMath"], directory, pages, len(pages),
        K["kMaOutVectors"] * VEC, deadline_s=20.0,
    )  # fmt: skip
    report = arithmetic.math_compare(a, b, x, out)
    report["run"] = card.check_run(res, n_workers)
    report["microseconds"] = float(card.command_microseconds(res)[-1])
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=1 << 20)
    p.add_argument("--scene", default="s10k")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--recip", type=int, nargs="*", default=[])
    p.add_argument("--math", type=int, default=0, help="rows of division and sqrt")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    n = (args.n + VEC - 1) // VEC * VEC
    report = {"n": n, "scene": args.scene}
    rows, tree = arithmetic.scene_inputs(rng, n, args.scene)
    edge = arithmetic.edge_inputs(rng, rows, tree)
    arithmetic.bests(rng, rows)
    pages = arithmetic.input_pages(rows)
    report["edge_rows"] = int(edge.sum())
    card.log(f"inputs: {n} rows, {int(edge.sum())} edge cases, {len(pages)} pages")
    with card.open_session() as session:
        workers = card.workers(session)
        directory = arithmetic.contiguous(len(pages), workers)
        out, prof, res, times, _ = card.run_handler(
            session, K["kHandlerArith"], directory, pages, len(pages),
            K["kArOutVectors"] * VEC, deadline_s=20.0,
        )  # fmt: skip
        report["run"] = card.check_run(res, workers)
        report["times"] = times
        report["recip"] = [recip_check(session, e, workers) for e in args.recip]
        if args.math:
            report["math"] = math_check(session, rng, args.math, workers)
    got = out.reshape(len(pages), K["kArOutVectors"], VEC).transpose(1, 0, 2)
    got = {name: got[k].reshape(-1) for k, name in enumerate(OUT_NAMES)}
    got = {
        k: v.view(np.int32 if k == "new_tri" else np.float32) for k, v in got.items()
    }
    report["check"] = arithmetic.compare(rows, got, edge)
    report["ok"] = exact(report, workers)
    print(json.dumps(report), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
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
    for r in report["recip"]:
        ok &= card.ran_cleanly(r["run"], workers) and r["differ_normal_results"] == 0
    if "math" in report:
        m = report["math"]
        ok &= card.ran_cleanly(m["run"], workers)
        ok &= m["divide"]["differ_bits"] == m["sqrt"]["differ_bits"] == 0
    return bool(ok)


if __name__ == "__main__":
    t0 = time.time()
    ok = main()
    print(f"wall {time.time() - t0:.1f}s", flush=True)
    sys.exit(0 if ok else 1)
