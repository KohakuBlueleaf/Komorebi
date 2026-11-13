"""MAP jobs with random team sizes decided by a micro-op stream: run on the host, on
the X280 issuing each record, and on the X280 submitting them to the job queue, each
bit-exact against numpy; then three streams that end in their error events. Prints one
JSON line with `ok` and exits non-zero without it. See README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import dram, native
from komorebi.engines import stream as ks
from komorebi.layout import share
from komorebi.state import wrap

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
TILE = 32
STEP = 61  # the input window moves this many tiles per job
AFTER = 20  # jobs of the clean stream after the error cases
T0 = time.monotonic()


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def read_bf16_chunked(session, tensor, name):
    """A bfloat16 tile tensor as session.read_bf16 gives it, read in chunks of 2 MB
    with a line per chunk: a whole read of an output takes a minute or more."""
    total, parts = tensor.pages, []
    for first, pages in session.read_chunks(tensor):
        parts.append(pages)
        log(f"{name}: {first + len(pages)} of {total} pages read")
    rows, cols = tensor.shape[-2:]
    return dram.untilize_bf16(np.concatenate(parts), rows, cols)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--jobs", type=int, default=1000)
    p.add_argument("--tiles", type=int, default=64, help="input tiles T per job")
    p.add_argument("--in-tiles", type=int, default=512, help="tiles of the input")
    p.add_argument(
        "--ways",
        default="hostfed,issue,submit,errors",
        help="comma-separated: hostfed, issue (X280, depth 1), submit (X280, queued), "
        "errors (three streams that end in an error, then a clean one)",
    )
    p.add_argument("--depth", type=int, default=16, help="ring slots used")
    p.add_argument("--deadline", type=float, default=1.0, help="per record, seconds")
    p.add_argument("--seed", type=int, default=1, help="the stream's xorshift seed")
    p.add_argument("--card", type=int, default=None)
    p.add_argument("--no-trace", action="store_true", help="no per-record trace")
    return p.parse_args(argv)


def job_stream(form, jobs, cores, window, addr, tiles, seed):
    """The stream of every way: per job a team size from RAND, the output page after
    the previous job's, the input window moved by STEP. `form` "issue": a computed
    first core, ISSUE and WAIT 0; "submit": SUBMIT, the queue picks the team."""
    s = ks.Stream(req=40 if form == "issue" else 41, seed=seed)
    j, x, count, first, room, base, page = s.regs(7)
    n_cores, n_jobs, wrap = s.const(cores), s.const(jobs), s.const(window)
    for r in (j, base, page):
        s.set(r, 0)
    top = s.label()
    s.rand(x)
    s.mod(count, x, n_cores)
    s.addi(count, count, 1)  # 1 to every core
    map_args = [addr["in"], tiles, addr["out"], base, page]
    if form == "issue":
        s.sub(room, n_cores, count)
        s.addi(room, room, 1)
        s.shr(first, x, 8)
        s.mod(first, first, room)
        s.issue(addr["map"], (first, count), args=map_args)
        s.wait(0)
    else:
        s.submit(addr["map"], count, args=map_args)
    s.add(base, base, count)
    s.addi(page, page, STEP)
    inside = s.forward()
    s.jlt(page, wrap, inside)
    s.sub(page, page, wrap)
    s.bind(inside)
    s.addi(j, j, 1)
    s.jlt(j, n_jobs, top)
    s.wait(0)
    s.halt(j)
    return s


def simulated(stream, cores):
    """The records the kernel issues for `stream`, from the package's simulator:
    [(first, count, output page, first input tile)]."""
    sim = stream.simulate(n_cores=cores, max_passes=1_000_000)
    if sim.status != "ended":
        raise SystemExit(f"the simulator ends the stream with {sim.status}")
    return [(r.first, r.count, r.args[3], r.args[4]) for r in sim.records]


def expected(x, jobs, tiles):
    """Every output tile: member m of a job adds its share of the job's input tiles
    (float64 over integer inputs, exact in bf16)."""
    sums = np.cumsum(x.reshape(-1, TILE, TILE).astype(np.float64), 0)
    prefix = np.concatenate([np.zeros((1, TILE, TILE)), sums])
    pages = sum(c for _, c, _, _ in jobs)
    out = np.zeros((pages, TILE, TILE))
    for _, count, base, start in jobs:
        for m in range(count):
            part = share(range(start, start + tiles), m, count)
            out[base + m] = prefix[part.stop] - prefix[part.start]
    return out


def membership(jobs, cores):
    """Records each core is a member of."""
    edge = np.zeros(cores + 1, np.int64)
    for first, count, _, _ in jobs:
        edge[first] += 1
        edge[first + count] -= 1
    return np.cumsum(edge)[:cores]


def traced(trace):
    """Each traced record in sequence order: (first, count, took, done) with took the
    first member's reader time and done the last member's writer time, in us."""
    h = trace.h
    seen = trace.rows[:, :, h["kTraceSeen"]]
    done = trace.rows[:, :, h["kTraceDone"]]
    at = np.flatnonzero(seen.any(0))
    if not at.size:
        return []
    zero = seen[:, at][seen[:, at] != 0].min()
    out = []
    for s in at:
        cores = np.flatnonzero(seen[:, s])
        took = wrap(seen[cores, s] - zero).min() / trace.mhz
        end = wrap(done[cores, s] - zero).max() / trace.mhz
        out.append((int(cores[0]), len(cores), float(took), float(end)))
    return out


def most_in_flight(records):
    """The most records whose (took, done) intervals overlap at one time."""
    edges = sorted([(r[2], 1) for r in records] + [(r[3], -1) for r in records])
    now = most = 0
    for _, d in edges:
        now += d
        most = max(most, now)
    return most


def main():
    args = parse()
    out = {"jobs": args.jobs, "tiles": args.tiles, "seed": args.seed}
    out["image_bytes"] = native.image_path().stat().st_size
    with km.Session(card=args.card, log=log) as session:
        log("device open")
        out.update(run_ways(session, args))
    out["ok"] = verdict(out, args)
    out["wall_s"] = round(time.monotonic() - T0, 1)
    print(json.dumps(out), flush=True)
    sys.exit(0 if out["ok"] else 1)


def verdict(out, args):
    """Every way exact with every core's count right, as simulated and ended with
    its halt value; the queue with more than one record in flight; each error stream
    ended with its reason and the jobs after them exact."""
    ok = out["same_jobs_issue_submit"]
    for way, got in out["ways"].items():
        ok &= got["tiles_wrong"] == 0 and got["cores_bad"] == 0
        ok &= got["status"] == ("ended" if way == "hostfed" else "end")
        ok &= got.get("halt", args.jobs) == args.jobs
        if way != "submit":
            ok &= got.get("as_simulated", True)
        elif "traced" in got:
            ok &= got["most_in_flight"] > 1
    if "errors" in out:
        e = out["errors"]
        reasons = [e[k].get("reason") for k in ("bad_opcode", "steps", "never_fits")]
        after = e["after"]
        ok &= reasons == ["opcode", "steps", "never_fits"]
        ok &= after["halt"] == min(AFTER, args.jobs) and after["kernel_error"] == 0
        ok &= after["cores_bad"] == 0 and after["tiles_wrong"] == 0
    return bool(ok)


def run_ways(session, args):
    gx, gy = session.grid
    cores = gx * gy
    ways = args.ways.split(",")
    tiles, in_tiles = args.tiles, max(args.in_tiles, args.tiles + STEP)
    window = in_tiles - tiles + 1
    rng = np.random.default_rng(args.seed)
    # Integers in [-2, 2]: every partial sum of up to 64 tiles is exact in bf16.
    x = rng.integers(-2, 3, (in_tiles * TILE, TILE)).astype(np.float32)

    def tile_tensor(a):
        return session.tensor(a, "bfloat16", "tile")

    # The jobs, from the simulator of the ISSUE stream: the same for every way.
    probe = {"in": 0, "out": 0, "map": 1}
    issue_jobs = simulated(
        job_stream("issue", args.jobs, cores, window, probe, tiles, args.seed), cores
    )
    submit_jobs = simulated(
        job_stream("submit", args.jobs, cores, window, probe, tiles, args.seed), cores
    )
    # The queue may issue a later, smaller job first: the same jobs, in another order.
    same = sorted(j[1:] for j in issue_jobs) == sorted(j[1:] for j in submit_jobs)
    want = expected(x, issue_jobs, tiles)
    pages = len(want)
    log(f"{len(issue_jobs)} jobs, {pages} output tiles, {cores} cores")

    # Tensors before any load: a tensor write waits for idle cores.
    inp = tile_tensor(x)
    size = {w: pages for w in ways}
    if "errors" in ways:  # only the clean stream's pages
        size["errors"] = sum(c for _, c, _, _ in issue_jobs[: min(AFTER, args.jobs)])
    outs = {w: tile_tensor(np.zeros((size[w] * TILE, TILE), np.float32)) for w in ways}
    program = km.Program(
        "uop_stream",
        kernels=KERNELS,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),  # kCbIn
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),  # kCbOut
        ],
        compile_args={
            "reader": inp.accessor_args(),
            "writer": outs[ways[0]].accessor_args(),
        },
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        depth=args.depth,
        segments=KERNELS / "uop_stream.h",
    )
    result = {"cores": cores, "same_jobs_issue_submit": same, "ways": {}}
    for way in ways:
        if way == "errors":
            result["errors"] = error_cases(
                session, program, args, inp, outs[way], want, issue_jobs, cores, window
            )
            continue
        addr = {
            "in": inp.address,
            "out": outs[way].address,
            "map": program.seg("map"),
        }
        result["ways"][way] = one_way(
            session, program, args, way, addr, issue_jobs, cores, window
        )
        got = read_bf16_chunked(session, outs[way], way).reshape(-1, TILE, TILE)
        wrong = int(np.count_nonzero((got.astype(np.float64) != want).any((1, 2))))
        result["ways"][way]["tiles_wrong"] = wrong
        log(f"{way}: {wrong} tiles wrong of {pages}")
    return result


def one_way(session, program, args, way, addr, jobs, cores, window):
    """Loads the program, runs the jobs one way, stops; times and core counts."""
    trace = 0 if args.no_trace else len(jobs)
    run = session.load(program, deadline=args.deadline, trace=trace)
    log(f"{way}: loaded")
    n = len(jobs)
    out = {}
    form = "submit" if way == "submit" else "issue"
    stream = job_stream(form, args.jobs, cores, window, addr, args.tiles, args.seed)
    t0 = time.perf_counter()
    if way == "hostfed":
        res = run.run_hostfed(stream, timeout=60.0)
        out["status"] = res.status
        sent = [(r.first, r.count, r.args[3], r.args[4]) for r in res.records]
        out["as_simulated"] = sent == jobs
    else:
        up = run.upload(stream)
        out["upload_bytes"] = up.bytes
        ticket = run.start(up)
        end = run.wait_engine(ticket, timeout=60.0)
        out["status"], out["halt"] = end.kind, end.value
        run.release(up)
    out["us_per_job"] = round((time.perf_counter() - t0) * 1e6 / n, 3)
    run.stop()
    log(f"{way}: {out['us_per_job']} us per job")
    if trace:
        recs = traced(run.trace())[:n]  # the stop's EXIT may follow in the trace
        out["traced"] = len(recs)
        if way == "issue":  # one record in flight: the order is the simulator's
            out["as_simulated"] = [r[:2] for r in recs] == [j[:2] for j in jobs]
        out["most_in_flight"] = most_in_flight(recs)
    states = run.state()
    runs = np.array([c["runs"] for c in states])
    done = np.array([c["completed"] for c in states])
    out["records"] = run.result()["records"]
    out["member_records"] = int(done.sum())
    if way != "submit":  # the queue picks the teams: only the total is known
        member = membership(jobs, cores)
        out["cores_bad"] = int(
            np.count_nonzero((runs != member + 1) | (done != member))
        )
    else:
        out["cores_bad"] = int(done.sum() != sum(c for _, c, _, _ in jobs))
    return out


def error_cases(session, program, args, inp, out_t, want, jobs, cores, window):
    """An upload with a bad opcode, a stream past its step limit and a job larger than
    the grid, each ending in its error event; then the first jobs of the ISSUE stream
    on the same run, exact, and a clean stop."""
    addr = {"in": inp.address, "out": out_t.address}
    addr["map"] = program.seg("map")
    run = session.load(program, deadline=args.deadline)
    bad = bytearray(job_stream("issue", 4, cores, window, addr, args.tiles, 1).encode())
    bad[64] = 0xEE  # the first micro-op's opcode
    steps = ks.Stream(req=42, max_steps=1000)
    (i,) = steps.regs(1)
    top = steps.label()
    steps.addi(i, i, 1)
    steps.jmp(top)
    big = ks.Stream(req=43)
    (n,) = big.regs(1)
    big.set(n, cores + 1)
    big.submit(addr["map"], n, args=[addr["in"], args.tiles, addr["out"], 0, 0])
    big.wait(0)
    big.halt(n)
    out = {}
    for name, obj in (
        ("bad_opcode", bytes(bad)),
        ("steps", steps),
        ("never_fits", big),
    ):
        up = run.upload(obj, kind="stream")
        try:
            end = run.wait_engine(run.start(up), timeout=10.0)
            out[name] = {"ended": end.kind}
        except km.EngineError as e:
            out[name] = {"reason": e.event.reason_name, "pc": e.event.pc}
        except km.EngineRefused as e:
            out[name] = {"refused": e.reason_name}
        run.release(up)
        log(f"errors: {name} {out[name]}")
    k = min(AFTER, len(jobs))
    clean = job_stream("issue", k, cores, window, addr, args.tiles, args.seed)
    up = run.upload(clean)
    out["after"] = {"halt": run.wait_engine(run.start(up), timeout=10.0).value}
    run.release(up)
    result = run.stop()
    out["after"]["kernel_error"] = result["error"]
    done = np.array([c["completed"] for c in run.state()])
    out["after"]["cores_bad"] = int(
        np.count_nonzero(done != membership(jobs[:k], cores))
    )
    pages = sum(c for _, c, _, _ in jobs[:k])
    got = read_bf16_chunked(session, out_t, "errors").reshape(-1, TILE, TILE)
    wrong = (got.astype(np.float64) != want[:pages]).any((1, 2))
    out["after"]["tiles_wrong"] = int(np.count_nonzero(wrong))
    return out


if __name__ == "__main__":
    main()
