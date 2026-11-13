"""The map/collect example's kernels fed by the test feeder on the last core (no X280);
every MAP member's tile is checked against numpy. One JSON line.

Example: python mapcollect_fed.py --jobs 20 --teams fixed --count 8"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fed  # noqa: E402
import records as rec  # noqa: E402

import komorebi as km  # noqa: E402
from komorebi import formats  # noqa: E402

K = rec.K
KERNELS = HERE.parents[2] / "examples" / "01_mapcollect" / "kernels"
SEGMENTS = rec.constants(KERNELS / "mapcollect.h")
SEG_MAP, SEG_STALL = SEGMENTS["kSegMap"], SEGMENTS["kSegStall"]
SEG_ALIGN, SEG_LATE = SEGMENTS["kSegAlign"], SEGMENTS["kSegLate"]
SEG_BAD_ITEM, SEG_RESULT = SEGMENTS["kSegBadItem"], SEGMENTS["kSegResult"]
TILE = 32


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--jobs", type=int, default=20)
    p.add_argument("--seg", choices=["map", "nop", "align"], default="map")
    p.add_argument(
        "--release", type=int, default=8000, help="align: cycles after issue"
    )
    p.add_argument(
        "--teams", choices=["fixed", "random", "pair", "overlap"], default="fixed"
    )
    p.add_argument("--count", type=int, default=8, help="team size (fixed)")
    p.add_argument(
        "--team-list", help="first:count per job, comma-separated (overrides --teams)"
    )
    p.add_argument("--first", type=int, default=0, help="first core (fixed)")
    p.add_argument("--tiles", type=int, default=64, help="input tiles T")
    p.add_argument("--depth", type=int, default=4, help="ring slots used")
    p.add_argument("--out-pages", type=int, default=2048, help="pages per output")
    p.add_argument("--out-tensors", type=int, default=1, help="output tensors")
    p.add_argument("--stall", type=int, default=-1, help="job index made a STALL")
    p.add_argument("--stall-member", type=int, default=0)
    p.add_argument("--late", type=int, default=-1, help="job index made a LATE")
    p.add_argument("--late-member", type=int, default=0)
    p.add_argument("--late-cycles", type=int, default=0)
    p.add_argument("--slow", type=int, default=-1, help="MAP job with slow compute")
    p.add_argument("--bad-item", type=int, default=-1, help="job index made a BAD_ITEM")
    p.add_argument("--bad-side", choices=["writer", "reader"], default="writer")
    p.add_argument("--result", type=int, default=-1, help="job made a flagged RESULT")
    p.add_argument("--result-value", type=int, default=0x5EED, help="its a0")
    p.add_argument("--slow-member", type=int, default=0)
    p.add_argument("--slow-cycles", type=int, default=0)
    p.add_argument("--wait", type=float, default=2.0, help="wait limit, seconds")
    p.add_argument(
        "--exit-at", type=int, default=-1, help="job index made an EXIT (team of 1)"
    )
    p.add_argument("--deadline", type=float, default=0.05, help="per record, seconds")
    p.add_argument("--raw", type=int, default=-1, help="job index issued raw")
    p.add_argument("--raw-slot", type=int, default=0, help="slot its header names")
    p.add_argument("--exit-team", default="", help="first:count kept on the EXIT")
    p.add_argument(
        "--row", type=int, help="cores per row for completion (0: one level)"
    )
    p.add_argument("--trace", action="store_true")
    p.add_argument("--trace-cap", type=int, help="records traced (default: all)")
    p.add_argument("--excl", action="store_true", help="KM_F_EXCL on every MAP record")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dump", type=Path, help="save records, trace and state (npz)")
    return p.parse_args(argv)


def make_teams(args, workers, rng):
    if args.team_list:
        teams = [tuple(map(int, t.split(":"))) for t in args.team_list.split(",")]
        args.jobs = len(teams)
        return teams
    if args.teams == "fixed":
        return [(args.first, args.count)] * args.jobs
    if args.teams == "pair":
        half = workers // 2
        return [
            (0, half) if j % 2 == 0 else (half, workers - half)
            for j in range(args.jobs)
        ]
    if args.teams == "overlap":  # two halves that share a quarter, alternating
        half, quarter = workers // 2, workers // 4
        return [(0, half) if j % 2 == 0 else (quarter, half) for j in range(args.jobs)]
    return rec.random_teams(rng, args.jobs, workers)


def place_outputs(teams, pages, tensors):
    """Global output page of member 0 per job: a ring of `tensors` x `pages` pages
    in which no job crosses a tensor."""
    base, at = [], 0
    for _, count in teams:
        if count > pages:
            raise ValueError(f"a team of {count} needs more than {pages} output pages")
        if at % pages + count > pages:
            at = (at // pages + 1) * pages
        if at >= tensors * pages:
            at = 0
        base.append(at)
        at += count
    return base


def checked_jobs(teams, base, pages):
    """Jobs whose output pages no later job overwrote."""
    owner = np.full(pages, -1)
    for j, ((_, count), b) in enumerate(zip(teams, base)):
        owner[b : b + count] = j
    return [
        j
        for j, ((_, c), b) in enumerate(zip(teams, base))
        if (owner[b : b + c] == j).all()
    ]


def members(args, teams, res, states):
    """Cores whose runs, completions, slot refusals or last seen and reported sequence
    differ from what the teams of the issued and acknowledged records give."""
    workers, recs, s = len(states), res.records, res.summary
    runs, done = np.zeros((2, workers), int), np.zeros((2, workers), int)  # low, high
    refused = np.zeros(workers, int)
    last = {}  # slot: (seq, cores reporting) of the last record issued there
    for j, (first, count) in enumerate(teams):
        seq, slot, acked = (
            int(recs["seq"][j]),
            int(recs["slot"][j]),
            recs["status"][j] == 1,
        )
        is_exit = j == args.exit_at
        team = slice(0, workers) if is_exit else slice(first, first + count)
        if seq == 0:
            continue
        if j == args.raw and args.raw_slot != slot:  # not run by the cores that see it
            refused[team] += 1
            continue
        runs[1, team] += 1
        runs[0, team] += acked
        done[1, team] += not is_exit
        done[0, team] += acked and not is_exit
        if acked:
            last[slot] = (seq, team)
    if s["status"] == "done" and not 0 <= args.exit_at < len(teams):
        runs[1] += 1  # the feeder's own EXIT, on slot 0
        runs[0] += bool(s["exit_acked"])
        last[0] = (s["issued"] + 1, slice(0, workers))
    bad = []
    for c in states:
        k = c["core"]
        ok = runs[0, k] <= c["runs"] <= runs[1, k]
        ok &= done[0, k] <= c["completed"] <= done[1, k]
        ok &= c["slot_wrong"] == refused[k]
        if s["status"] == "done":
            for slot, (seq, team) in last.items():
                member = team.start <= k < team.stop
                ok &= c["last"][slot] == seq and (c["reported"][slot] == seq) == member
        if not ok:
            bad.append(k)
    return bad


def member_sums(x, count):
    """Float64 sum of each member's share of the tiles of x: [count, 32, 32]."""
    tiles = x.reshape(-1, TILE, TILE)
    out = np.zeros((count, TILE, TILE))
    for m in range(count):
        start, length = rec.share(len(tiles), m, count)
        out[m] = tiles[start : start + length].sum(0)
    return out


def trace_phases(res, teams, n):
    """Per job from the trace: microseconds of each phase (see the README)."""
    tr = res.trace.astype(np.int64)
    rows = res.records
    mhz = res.mhz
    out = []
    for j in range(n):
        first, count = teams[j]
        seq = int(rows["seq"][j])
        if seq == 0 or seq > tr.shape[1]:
            continue
        t = tr[first : first + count, seq - 1]
        issue = int(rows["issue"][j])
        rel = rec.wrap(t - issue)
        seen, done, sent = rel[:, 0], rel[:, 1], rel[:, 2]
        out.append(
            {
                "count": count,
                "first_seen": seen.min() / mhz,
                "last_seen": seen.max() / mhz,
                "last_done": done.max() / mhz,
                "last_member_sent": (sent[1:].max() if count > 1 else sent[0]) / mhz,
                "leader_sent": sent[0] / mhz,
                "push_after_last_done": (sent[0] - done.max()) / mhz,
                "acked": int(rec.wrap(rows["acked"][j] - issue)) / mhz,
                "passes_mean": float(t[:, 3].mean()),
            }
        )
    return out


def overlaps(res, teams, n):
    """Consecutive jobs on disjoint teams where a member of the second took its record
    before the first job's leader pushed its completion (wall clock, from the trace)."""
    tr = res.trace.astype(np.int64)
    seqs = res.records["seq"]
    found = []
    for j in range(n - 1):
        (fa, ca), (fb, cb) = teams[j], teams[j + 1]
        sa, sb = int(seqs[j]), int(seqs[j + 1])
        disjoint = fa + ca <= fb or fb + cb <= fa
        if not disjoint or 0 in (sa, sb) or max(sa, sb) > tr.shape[1]:
            continue
        pushed = tr[fa, sa - 1, 2]
        seen = tr[fb : fb + cb, sb - 1, 0].min()
        lead = int(rec.wrap(pushed - seen))
        if 0 < lead < 1 << 31:
            found.append(lead / res.mhz)
    return {
        "pairs": n - 1,
        "overlapping": len(found),
        "lead_us_mean": round(float(np.mean(found)) if found else 0.0, 3),
    }


def in_flight_together(res, teams, n, window=K["KM_RING_SLOTS"]):
    """Pairs of jobs on overlapping teams (at most `window` apart) where a member of
    the later one took it before the earlier one's leader pushed (from the trace)."""
    tr = res.trace.astype(np.int64)
    seqs = res.records["seq"]
    pairs = together = 0
    for a in range(n):
        for b in range(a + 1, min(n, a + 1 + window)):
            (fa, ca), (fb, cb) = teams[a], teams[b]
            sa, sb = int(seqs[a]), int(seqs[b])
            if fa + ca <= fb or fb + cb <= fa or 0 in (sa, sb):
                continue
            if max(sa, sb) > tr.shape[1]:
                continue
            pairs += 1
            seen = tr[fb : fb + cb, sb - 1, 0].min()
            together += 0 < int(rec.wrap(tr[fa, sa - 1, 2] - seen)) < 1 << 31
    return {"pairs": pairs, "together": together}


def summarize_phases(phases):
    keys = [k for k in phases[0] if k != "count"] if phases else []
    return {k: round(float(np.mean([p[k] for p in phases])), 3) for k in keys}


def main():
    args = parse()
    rng = np.random.default_rng(args.seed)
    t_start = time.time()
    with km.Session(log=fed.log) as session:
        fed.log(f"device open {time.time() - t_start:.1f}s")
        out = run(session, args, rng)
    out["wall_s"] = round(time.time() - t_start, 1)
    print(json.dumps(out), flush=True)


def run(session, args, rng):
    runner = fed.Fed(session)
    teams = make_teams(args, runner.workers, rng)
    workers, n, tiles = runner.workers, args.jobs, args.tiles
    fed.log(f"grid {runner.grid}, {workers} workers, aiclk {runner.mhz} MHz")

    # Exact: integers in [-2, 2] (sums exact in bf16 for T <= 64). Random: uniform in
    # [-1, 1], rounded to bf16. Job j reads tiles j to j + T - 1 (outputs all differ).
    x_exact = rng.integers(-2, 3, ((tiles + n) * TILE, TILE)).astype(np.float32)
    x_rand = rng.uniform(-1, 1, ((tiles + n) * TILE, TILE)).astype(np.float32)
    x_rand = formats.bf16_values(formats.bf16_bits(x_rand))

    def tile_tensor(a):
        return session.tensor(a, "bfloat16", "tile")

    inputs = [tile_tensor(x_exact), tile_tensor(x_rand)]
    pages = args.out_pages
    zeros = np.zeros((pages * TILE, TILE), dtype=np.float32)
    outs = [tile_tensor(zeros) for _ in range(args.out_tensors)]
    base = place_outputs(teams, pages, args.out_tensors)

    records = []
    flags = K["KM_F_EXCL"] if args.excl else 0
    for j, (first, count) in enumerate(teams):
        if j == args.stall:
            records.append(rec.record(SEG_STALL, first, count, [args.stall_member]))
        elif j == args.bad_item:
            side = [0, int(args.bad_side == "reader")]
            records.append(rec.record(SEG_BAD_ITEM, first, count, side))
        elif j == args.result:
            value, flag = [args.result_value], K["KM_F_RESULT"]
            records.append(rec.record(SEG_RESULT, first, count, value, flag))
        elif j == args.late:
            late = [args.late_member, args.late_cycles]
            records.append(rec.record(SEG_LATE, first, count, late))
        elif j == args.exit_at:  # the feeder sets the team to every worker
            x = tuple(map(int, args.exit_team.split(":"))) if args.exit_team else (0, 1)
            records.append(rec.record(K["KM_SEG_EXIT"], *x))
        elif args.seg == "align":
            records.append(rec.record(SEG_ALIGN, first, count))
        elif args.seg == "nop":
            records.append(rec.record(K["KM_SEG_NOP"], first, count))
        else:
            words = [
                inputs[j % 2].address,
                tiles,
                outs[base[j] // pages].address,
                base[j] % pages,
                args.slow_cycles if j == args.slow else 0,
                args.slow_member if j == args.slow else 0xFFFFFFFF,
                j,  # first input tile
            ]
            records.append(rec.record(SEG_MAP, first, count, words, flags))
        if j == args.raw:  # issued as built: the slot it names and its team kept
            records[-1][rec.RECORD_FIELDS.index("hdr")] |= args.raw_slot << 16

    kernels = fed.Kernels(
        reader=KERNELS / "reader.cpp",
        compute=KERNELS / "compute.cpp",
        writer=KERNELS / "writer.cpp",
        reader_ct=inputs[0].accessor_args(),
        writer_ct=outs[0].accessor_args(),
        compute_options={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        cbs=[
            (SEGMENTS["kCbIn"], "bfloat16", 2048, SEGMENTS["kInPages"]),
            (SEGMENTS["kCbOut"], "bfloat16", 2048, 2),
        ],
    )
    trace_cap = (n + 4) // 4 * 4 if args.trace else 0  # 64-byte pages
    trace_cap = args.trace_cap if args.trace and args.trace_cap else trace_cap
    res = runner.run(
        kernels,
        records,
        tensors=[*inputs, *outs],
        depth=args.depth,
        deadline_s=args.deadline,
        wait_s=args.wait,
        trace_cap=trace_cap,
        release=args.release if args.seg == "align" else 0,
        row=args.row,
        raw=args.raw + 1 if args.raw >= 0 else 0,
    )
    if args.dump:
        np.savez(
            args.dump,
            teams=np.asarray(teams),
            base=np.asarray(base),
            seq=res.records["seq"],
            slot=res.records["slot"],
            issue=res.records["issue"],
            acked=res.records["acked"],
            state=res.state,
            trace=res.trace if res.trace is not None else np.zeros(0),
        )
    s = res.summary
    out = {
        "jobs": n,
        "seg": args.seg,
        "teams": args.teams,
        "count": args.count if args.teams == "fixed" else None,
        "tiles": tiles,
        "depth": args.depth,
        "excl": args.excl,
        "held": int(res.records["held"].sum()),
        "result": int(res.records["result"][args.result]) if args.result >= 0 else None,
        "aiclk_mhz": res.mhz,
        "feeder": s,
        "us_per_record": round(s["span_cycles"] / res.mhz / max(s["done"], 1), 3),
        "issue_us_mean": round(s["issue_cycles"] / res.mhz / max(s["issued"], 1), 3),
        "host_program_s": round(res.host_s, 2),
    }
    lat = rec.wrap(res.records["acked"] - res.records["issue"])[
        res.records["status"] == 1
    ]
    if len(lat):
        out["latency_us"] = {
            "mean": round(float(lat.mean()) / res.mhz, 3),
            "min": round(float(lat.min()) / res.mhz, 3),
            "max": round(float(lat.max()) / res.mhz, 3),
        }
    if 0 <= args.exit_at < n:  # issued only when every earlier record was acked
        r, k = res.records, args.exit_at
        gap = rec.wrap(r["issue"][k] - r["acked"][:k])
        out["exit"] = {
            "held": bool(r["held"][k]),
            "status": int(r["status"][k]),
            "after_every_ack": bool(((gap > 0) & (gap < 1 << 31)).all()),
        }
    states = [rec.core_state(res.state, c) for c in range(workers)]
    out["runs"] = sum(c["runs"] for c in states)
    if "exit" in out:  # cores whose report of the EXIT landed
        slot, seq = int(res.records["slot"][args.exit_at]), int(
            res.records["seq"][args.exit_at]
        )
        out["exit"]["reported_by"] = sum(c["reported"][slot] == seq for c in states)
    out["slot_wrong"] = {c["core"]: c["slot_wrong"] for c in states if c["slot_wrong"]}
    out["trace_dropped"] = {
        c["core"]: c["trace_dropped"] for c in states if c["trace_dropped"]
    }
    out["membership_bad"] = members(args, teams, res, states)[:16]
    out["cores_magic_ok"] = sum(c["magic_ok"] for c in states)
    out["cores_ended"] = {
        k: sum(c[k] != 0 for c in states)
        for k in ("reader_end", "writer_end", "compute_end")
    }
    out["compute_wrong"] = sum(sum(c["compute_wrong"]) for c in states)
    passes = sum(c["passes"] for c in states)
    out["next_cycles_per_pass"] = round(
        sum(c["next_cycles"] for c in states) / max(passes, 1), 1
    )
    parts = ("reader", "writer", "compute")
    gave = [
        {"core": c["core"], **{k: c[k] for k in parts}}
        for c in states
        if any(c[k] for k in parts)
    ]
    out["gave_up_cores"] = len(gave)
    out["gave_up_where"] = {
        w: sum(1 for g in gave for k in parts if g[k] and g[k]["where"] == w)
        for w in sorted({g[k]["where"] for g in gave for k in parts if g[k]})
    }
    out["gave_up_busy"] = [  # cores that gave up anywhere but waiting for a record
        {"core": g["core"], **{k: g[k] for k in parts if g[k]}}
        for g in gave
        if any(g[k] and g[k]["where"] != "next" for k in parts)
    ][:16]
    if s["status"] == "timeout":
        first, count = teams[s["index"]]
        team = range(first, first + count)
        out["timed_out"] = {"slot": s["slot"], "seq": s["seq"], "team": [first, count]}
        out["not_reported"] = [
            c for c in team if states[c]["reported"][s["slot"]] != s["seq"]
        ]
        out["named"] = [
            {"core": g["core"], **{k: g[k] for k in parts if g[k]}}
            for g in gave
            if any(g[k] and g[k]["where"] in ("stall", "complete") for k in parts)
        ]
    if args.trace and res.trace is not None:
        phases = trace_phases(res, teams, n)
        if args.seg == "align":  # release at issue + args.release cycles
            for p in phases:
                p["push_after_release"] = p["leader_sent"] - args.release / res.mhz
        out["trace_us"] = summarize_phases(phases)
        out["overlap"] = overlaps(res, teams, n)
        out["overlapping_teams"] = in_flight_together(res, teams, n)

    if args.seg == "map" and s["status"] == "done":
        out.update(check(args, teams, base, outs, [x_exact, x_rand]))
    return out


def check(args, teams, base, outs, xs):
    pages = args.out_pages * len(outs)
    got = []
    for i, t in enumerate(outs):
        got.append(t.read())
        fed.log(f"output {i + 1} of {len(outs)} read back")
    got = np.concatenate(got)
    got = got.reshape(pages, TILE, TILE)
    fed.log("outputs read back")
    last = args.exit_at if args.exit_at >= 0 else len(teams)
    jobs = [
        j
        for j in checked_jobs(teams, base, pages)
        if j not in (args.stall, args.late, args.bad_item, args.result) and j < last
    ]
    bad, rel_max, exact_jobs = [], 0.0, 0
    for j in jobs:
        _, count = teams[j]
        x = xs[j % 2][j * TILE : (j + args.tiles) * TILE]  # tiles j to j + T - 1
        ref = member_sums(x.astype(np.float64), count)
        mine = got[base[j] : base[j] + count].astype(np.float64)
        if j % 2 == 0:
            exact_jobs += 1
            if not np.array_equal(mine, ref):
                bad.append(
                    {
                        "job": j,
                        "team": teams[j],
                        "base": base[j],
                        "wrong_members": int((mine != ref).any((1, 2)).sum()),
                    }
                )
        else:
            total, ref_total = mine.sum(0), ref.sum(0)
            rel = float(
                np.sqrt(((total - ref_total) ** 2).mean() / (ref_total**2).mean())
            )
            rel_max = max(rel_max, rel)
            if rel > 1e-2:
                bad.append(
                    {"job": j, "team": teams[j], "base": base[j], "rel_rms": rel}
                )
    return {
        "jobs_checked": len(jobs),
        "exact_jobs_checked": exact_jobs,
        "jobs_bad": len(bad),
        "first_bad": bad[:5],
        "random_total_rel_rms_max": float(f"{rel_max:.3e}"),
    }


if __name__ == "__main__":
    main()
