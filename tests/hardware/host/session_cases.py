"""Card cases of the host package that the example does not cover, one per run
(`python session_cases.py CASE`, CASES below); prints one JSON line with `ok`."""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

import komorebi as km

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "01_mapcollect"
T0 = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def program(kernels):
    """The example's kernels with no tensors: only NOP and STALL records are sent."""
    return km.Program(
        "mapcollect",
        kernels=kernels,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
        ],
        compile_args={"reader": [2, 2048], "writer": [2, 2048]},
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        segments=Path(kernels) / "mapcollect.h",
    )


def case_compile_error(session):
    """A compute kernel that fails to compile: ProgramError, no run loaded; then the
    good program loads and runs in the same session (no image was left running)."""
    out, before = {}, session.run
    with tempfile.TemporaryDirectory() as tmp:
        broken = Path(tmp) / "kernels"
        shutil.copytree(EXAMPLE / "kernels", broken)
        compute = broken / "compute.cpp"
        compute.write_text(compute.read_text() + "\n#error deliberate compile error\n")
        try:
            session.load(program(broken))
            out["raised"] = None
        except km.ProgramError as err:
            out["raised"] = type(err).__name__
            out["names_error"] = "deliberate compile error" in str(err)
    out["run_after_error"] = session.run is not before
    run = session.load(program(EXAMPLE / "kernels"))
    out["nop_after"] = run.wait(run.issue(0, (0, 8))).kind
    run.stop()
    out["ok"] = (
        out["raised"] == "ProgramError"
        and out["names_error"]
        and not out["run_after_error"]
        and out["nop_after"] == "done"
        and run.result()["error"] == 0
    )
    return out


def case_exclusive(session):
    """A STALL-free check of `exclusive`: a record overlapping one in flight is held
    until that one completed (trace: its members took it after the leader pushed)."""
    run = session.load(program(EXAMPLE / "kernels"), trace=64)
    pairs = []
    for k in range(20):
        a = run.issue(0, (0, 60))
        b = run.issue(0, (30, 60), exclusive=k % 2 == 1)
        pairs.append((a, b, k % 2 == 1))
    run.drain()
    run.stop()
    trace = run.trace()
    leads = [(excl, trace.lead(a, b)) for a, b, excl in pairs]
    out = {
        "exclusive_leads_us": [round(v, 3) for e, v in leads if e],
        "plain_leads_us": [round(v, 3) for e, v in leads if not e],
    }
    out["ok"] = all(v < 0 for e, v in leads if e)
    return out


def case_queue_order(session, rounds=1000, jobs=32, count=8, busy=200):
    """Each round: BUSY on every core (its leader reads a page `busy` times), then
    `jobs` queued PUTs of `count` cores, which wait in the kernel's queue, each writing
    the round's value into its page, then an exclusive SUM on every core counting the
    pages that hold it. SUM must wait for every queued job: `jobs` each round."""
    prog, pages = queue_program(session)
    run = session.load(prog)
    at, counts, t0 = pages.address, [], time.monotonic()
    for k in range(rounds):
        run.issue(3, None, args=[at, busy])
        for j in range(jobs):
            run.issue(1, count, args=[at, j, k + 1], queued=True)
        total = run.issue(2, None, args=[at, jobs, k + 1], exclusive=True, result=True)
        counts.append(run.wait(total).result)
    run.drain()
    took = time.monotonic() - t0
    run.stop()
    short = [(k, c) for k, c in enumerate(counts) if c != jobs]
    out = {"rounds": rounds, "rounds_short": len(short), "first_short": short[:8]}
    out["ms_per_round"] = round(took / rounds * 1e3, 3)
    out["ok"] = not short and run.result()["error"] == 0
    return out


def queue_program(session):
    """The queue cases' program and its tensor: 64 pages of 64 bytes, uint32."""
    pages = session.tensor(np.zeros((64, 16), np.uint32), "uint32")
    prog = km.Program(
        "queue_order",
        kernels=Path(__file__).resolve().parent / "queue_kernels",
        buffers=[km.Buffer(index=16, format="uint32", page_bytes=64, pages=1)],
        compile_args={"writer": pages.accessor_args()},
    )
    return prog, pages


def case_queue_cost(session, windows=10, window=48, count=8, reads=(0, 100)):
    """Where a queued record's time goes: `windows` x `window` records of `count` cores,
    queued or on fixed disjoint teams, each window issued and waited for at once: PUTs,
    or BUSY records whose leader reads a page `reads` times."""
    prog, pages = queue_program(session)
    at, out = pages.address, {}
    for mode in [(m, r) for r in reads for m in ("plain", "queued")]:
        run = session.load(prog)
        events, t0 = [], time.monotonic()
        for w in range(windows):
            tickets = []
            for j in range(window):
                if mode[0] == "queued":
                    team, queued = count, True
                else:
                    team, queued = (count * (j % (run.cores // count)), count), False
                seg, args = (3, [at, mode[1]]) if mode[1] else (1, [at, j, w + 1])
                tickets.append(run.issue(seg, team, args=args, queued=queued))
            events += run.wait(tickets)
        wall = time.monotonic() - t0
        run.stop()
        res, n = run.result(), windows * window
        out[f"{mode[0]}_{mode[1]}_reads"] = {
            "host_us_per_record": round(wall / n * 1e6, 2),
            "host_issue_us": round(run.issue_s / n * 1e6, 2),
            "kernel_us_per_record": round(res["last"] / 50 / n, 2),
            "issue_to_done_us": round(
                float(np.mean([e.ticks for e in events])) / 50, 2
            ),
            "passes_per_record": round(res["passes"] / n, 1),
            "max_busy": res["max_busy"],
            "all_done": all(e.kind == "done" for e in events),
        }
    out["ok"] = all(v["all_done"] for v in out.values())
    return out


def bf16_tiles(session, x):
    return session.tensor(x, "bfloat16", "tile")


def case_read(session):
    """Tensors read by banks equal what was written and a whole-buffer read (a uint32
    row-major one of 37 pages over every bank, two bfloat16 ones of 6 tiles, one
    barrier), also while a run is loaded. The read's speed is an experiment
    (experiments/dram_read)."""
    rng = np.random.default_rng(3)
    words = rng.integers(0, 1 << 32, (37, 24), dtype=np.uint64).astype(np.uint32)
    x = rng.integers(-4, 5, (96, 64)).astype(np.float32)  # exact in bfloat16
    w = session.tensor(words, "uint32")
    a, b = bf16_tiles(session, x), bf16_tiles(session, 2 * x)
    raw, got_a, got_b = session.read(w)[0], *session.read_bf16(a, b)
    out = {
        "words": bool((raw.reshape(-1).view(np.uint32) == words.reshape(-1)).all()),
        "tiles": bool((got_a == x).all() and (got_b == 2 * x).all()),
        "as_whole": bool((got_a == a.read()).all()),
    }
    run = session.load(program(EXAMPLE / "kernels"))
    run.wait(run.issue(0, (0, 8)))
    log("read while the run is loaded")
    (during,) = session.read_bf16(a)
    out["during_run"] = bool((during == x).all())
    run.stop()
    out["ok"] = all(out[k] for k in ("words", "tiles", "as_whole"))
    out["ok"] = out["ok"] and out["during_run"] and run.result()["error"] == 0
    return out


def case_start_error(session):
    """A record deadline under one timer tick: the kernel refuses its arguments before
    it opens a window. StartError with its result; the host stops the grid (every
    reader gave up on the stop word, not its idle limit); then a run works."""
    out = {}
    t0 = time.monotonic()
    try:
        session.load(program(EXAMPLE / "kernels"), deadline=1e-9, idle=60.0)
        out["raised"] = None
    except km.StartError as err:
        out["raised"] = type(err).__name__
        out["result"] = err.result is not None
        out["message"] = str(err)[:200]
    out["start_error_s"] = round(time.monotonic() - t0, 2)
    failed = session.run
    readers = {}
    for c in failed.state():
        r = c["reader"]
        key = f"{r['code']} in {r['where']}" if r else "none"
        readers[key] = readers.get(key, 0) + 1
    out["readers"] = readers
    run = session.load(program(EXAMPLE / "kernels"))
    out["nop_after"] = run.wait(run.issue(0, None)).kind
    run.stop()
    out["ok"] = (
        out["raised"] == "StartError"
        and out["result"]
        and readers.get("stop in next", 0) == 120
        and out["nop_after"] == "done"
        and run.result()["error"] == 0
    )
    return out


CASES = {
    "compile-error": case_compile_error,
    "exclusive": case_exclusive,
    "queue-cost": case_queue_cost,
    "queue-order": case_queue_order,
    "read": case_read,
    "start-error": case_start_error,
}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("case", choices=sorted(CASES))
    args = p.parse_args()
    with km.Session(log=log) as session:
        out = {"case": args.case, **CASES[args.case](session)}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
