"""The whole backbone into the 31 islands' DRAM, checked, timed (exclusive lock).

python load_box.py [--verify sample|full] [--threads N] [--report FILE] [--resume]
[--converted]"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import convert  # noqa: E402
import loader  # noqa: E402
import placement  # noqa: E402

import komorebi as km  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", default="sample")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--converted",
        action="store_true",
        help="only the weights the cache's logs hold now (a load while converting)",
    )
    a = ap.parse_args()
    t0 = time.time()
    with km.Session(cards=placement.CARDS, links=1) as session:
        t_open = time.time() - t0
        print(
            f"open: {len(session.sessions)} islands, {session.ranks} ranks, "
            f"{t_open:.1f} s",
            flush=True,
        )
        names = None
        if a.converted:
            names = [
                n
                for i in range(len(placement.CARDS))
                for n in convert.read_done(loader.CACHE, i)
            ]
        W = loader.load(
            session, names, verify=a.verify or None, threads=a.threads, resume=a.resume
        )
        report = {"open_s": t_open, **W.report, "total_s": time.time() - t0}
        bad = report.get("verify", {}).get("bad", [])
        report["verify_bad_count"] = len(bad)
        if "verify" in report:
            report["verify"]["bad"] = bad[:50]
        print(json.dumps(report), flush=True)
        if a.report:
            with open(a.report, "w") as f:
                json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
