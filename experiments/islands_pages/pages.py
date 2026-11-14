"""Pages written across a link: a tensor held by island 0 copied into a tensor held
by island 1, sent by island 0's link cores straight into island 1's DRAM; read back by
the host and compared. Prints one JSON line. See README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import islands

KERNELS = Path(__file__).resolve().parent / "kernels"
NO_READ = 0x10  # KM_PAGES_SHAPE_NO_READ
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--links", type=int, default=4)
    parser.add_argument("--mib", type=int, default=64, help="MiB written across")
    parser.add_argument("--page", type=int, default=4096, help="bytes a page")
    parser.add_argument(
        "--where",
        action="store_true",
        help="no crossing: ask cores of both islands which island holds pages",
    )
    parser.add_argument(
        "--ask",
        type=int,
        nargs="+",
        metavar="MIB",
        help="island 0's ranks read MIB of a tensor island 1 holds (km::read_pages)",
    )
    parser.add_argument(
        "--put",
        type=int,
        nargs="+",
        metavar="MIB",
        help="island 0's ranks write MIB into a tensor island 1 holds (write_pages)",
    )
    parser.add_argument("--step", type=int, default=32, help="pages a call (L1)")
    parser.add_argument(
        "--shapes",
        type=lambda text: int(text, 0),
        nargs="+",
        metavar="SHAPE",
        help="the host's cross of --mib, once a push shape (KM_PAGES_SHAPE_*)",
    )
    parser.add_argument(
        "--chunk", type=int, default=0, help="KiB of a link CPU's L1 (0: the default)"
    )
    parser.add_argument(
        "--packet", type=int, help="payload bytes a packet (default: the session's)"
    )
    parser.add_argument(
        "--soak",
        choices=("ask", "put"),
        help="island 0's ranks ask (or put) --step pages a call over --mib, in rounds,"
        " until --calls calls or --seconds; every asked page checked on the core",
    )
    parser.add_argument(
        "--runs",
        choices=("one", "both"),
        help="every rank of island 0 (both: of both islands at once) puts its share "
        "of --mib to the other island in runs of --run-pages, --piece pages a piece",
    )
    parser.add_argument("--run-pages", type=int, default=64, help="pages a run")
    parser.add_argument("--piece", type=int, default=4, help="pages a piece")
    parser.add_argument("--calls", type=int, default=3_000_000, help="the soak's")
    parser.add_argument("--seconds", type=float, default=100.0, help="the soak's most")
    return parser.parse_args()


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    report = {"cards": cards, "links": args.links, "page": args.page}
    if args.ask or args.put:
        args.mib = max(args.ask or args.put)
    pages = args.mib * (1 << 20) // args.page
    data = np.random.default_rng(1).integers(
        0, 1 << 32, (pages, args.page // 4), dtype=np.uint64
    )
    data = data.astype(np.uint32)
    if args.soak:  # the pattern a soak's core checks: word 0 the page, the last ~page
        data[:, 0] = np.arange(pages, dtype=np.uint32)
        data[:, -1] = ~np.arange(pages, dtype=np.uint32)
    with km.Session(
        cards=cards, links=args.links, log=log, packet=args.packet
    ) as session:
        report["packet"] = session.packet
        if args.chunk:
            for one in session.sessions:
                one.transfer_chunk = args.chunk * 1024
        graph = session.graph
        own = [(graph.first(i), graph.count(i)) for i in range(2)]
        report["link_cores"] = [list(cores) for cores in session.link_cores]
        report["channels"] = [list(channels) for channels in session.channels]
        if args.runs:
            report.update(runs(session, args, data, own, log))
            report["wall_s"] = round(time.monotonic() - STARTED, 1)
            print(json.dumps(report), flush=True)
            sys.exit(0 if report["ok"] else 1)
        if args.soak:
            report.update(soak(session, args, data, own, log))
            report["wall_s"] = round(time.monotonic() - STARTED, 1)
            print(json.dumps(report), flush=True)
            sys.exit(0 if report["ok"] else 1)
        moving = args.ask or args.put
        # MOVE: ASK reads a source island 1 holds, PUT writes a target it holds
        source = session.tensor(data, "uint32", group=own[1 if args.ask else 0])
        # one target a size: ASK's on island 0, PUT's on island 1
        targets = [
            session.empty(data.shape, "uint32", group=own[0 if args.ask else 1])
            for _ in moving or ()
        ]
        if moving:
            tensors = (source, *targets)
        elif args.shapes:  # one target a shape that reads; one for those that do not
            reading = [shape for shape in args.shapes if not shape & NO_READ]
            targets = [
                session.empty(data.shape, "uint32", group=own[1])
                for _ in [*reading, None]
            ]
            tensors = (source, *targets)
        else:
            target = session.empty(data.shape, "uint32", group=own[1])
            split = session.empty(data.shape, "uint32")
            copy = session.empty(data.shape, "uint32", copy=True)
            tensors = (source, target, split, copy)
        report["parts"] = [list(tensor.placement.parts) for tensor in tensors]
        log(f"source {source.placement.parts}, target {tensors[1].placement.parts}")
        program = km.Program(
            "pages",
            kernels=KERNELS,
            handlers=KERNELS / "pages.h",
            buffers=[  # MOVE's: a call's pages
                km.Buffer(
                    index=0,
                    format="bfloat16",
                    page_bytes=args.step * args.page,
                    pages=1,
                )
            ],
            compile_args={"reader": source.accessor_args()},
        )
        run = session.load(program, deadline=2.0)
        log("loaded on both islands")
        if args.where:
            report.update(where(session, program, tensors))
            run.stop()
            report["ok"] = report["wrong_answers"] == 0
            print(json.dumps(report), flush=True)
            sys.exit(0 if report["ok"] else 1)
        if args.shapes:
            report.update(shapes(session, args, source, targets, data))
            try:
                results = run.stop()
                report["kernel_errors"] = [result["error_name"] for result in results]
            except km.KomorebiError as error:
                report["kernel_errors"] = str(error)[:300]
            log(f"stopped: {report['kernel_errors']}")
            report["ok"] = bool(
                report["kernel_errors"] == ["none", "none"]
                and all(row.get("wrong_pages", 0) == 0 for row in report["rows"])
            )
            print(json.dumps(report), flush=True)
            sys.exit(0 if report["ok"] else 1)
        if moving:
            report.update(move(session, run, program, args, source, targets, data))
            try:
                results = run.stop()
                report["kernel_errors"] = [result["error_name"] for result in results]
            except km.KomorebiError as error:
                report["kernel_errors"] = str(error)[:300]
            log(f"stopped: {report['kernel_errors']}")
            report["ok"] = bool(
                report["kernel_errors"] == ["none", "none"]
                and all(row["wrong_pages"] == 0 for row in report["rows"])
            )
            print(json.dumps(report), flush=True)
            sys.exit(0 if report["ok"] else 1)
        timeout = 5.0 + pages * args.page / 1e9
        try:
            crossed = session.cross(source, target, timeout=timeout)
            report.update(crossed)
            report["gb_s"] = round(crossed["bytes"] / crossed["seconds"] / 1e9, 2)
            log(f"crossed {crossed['bytes']:,} B in {crossed['seconds'] * 1e3:.3f} ms")
        except km.KomorebiError as error:
            report["error"] = str(error)[:400]
            log(f"cross failed: {report['error']}")
        try:
            results = run.stop()
            report["kernel_errors"] = [result["error_name"] for result in results]
        except km.KomorebiError as error:
            report["kernel_errors"] = str(error)[:300]
        log(f"stopped: {report['kernel_errors']}")
        if "error" in report:
            links(session, report)
        (got,) = session.read(target)
        got = got.view(np.uint32).reshape(data.shape)
        report["wrong_pages"] = int((got != data).any(axis=1).sum())
        log(f"target read back from island 1: {report['wrong_pages']} pages wrong")
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    report["ok"] = bool(
        report["wrong_pages"] == 0 and report["kernel_errors"] == ["none", "none"]
    )
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


def move(session, run, program, args, source, targets, data):
    """MOVE of each size by island 0's ranks (ASK: the source across, PUT: the
    target), one command each (its event's ticks: submit to completion), what
    crossed beside it; then each target's pages read back and compared."""
    handler = program.handler("move")
    mine = run.island(0)
    _, chunk = session.staging(0)
    log(f"link CPUs' L1: {chunk:,} B a CPU")
    group = (session.graph.first(0), session.graph.count(0))
    name = "ASK" if args.ask else "PUT"
    rows = []
    for mib, target in zip(args.ask or args.put, targets):
        pages = mib * (1 << 20) // args.page
        words = [source.address, target.address, pages, args.step]
        before = run.crossings()
        event = mine.wait(mine.submit(handler, group, words), timeout=30.0)
        moved = run.crossings() - before
        seconds = event.ticks / 50e6  # the X280's mtime: 50 MHz
        row = {
            "mib": mib,
            "ms": round(seconds * 1e3, 3),
            "gb_s": round(pages * args.page / seconds / 1e9, 2),
            "crossed": {f"{a}->{b}": moved.total(a, b) for a, b in ((0, 1), (1, 0))},
        }
        log(f"{name} {mib} MiB: {row['ms']} ms, {row['gb_s']} GB/s")
        log(f"crossed:\n{moved}")
        rows.append(row)
    for row, target in zip(rows, targets):
        pages = row["mib"] * (1 << 20) // args.page
        (got,) = session.read(target, first=0, pages=pages)
        got = got.view(np.uint32).reshape(pages, -1)
        row["wrong_pages"] = int((got != data[:pages]).any(axis=1).sum())
        log(f"{row['mib']} MiB read back: {row['wrong_pages']} pages wrong")
    return {"mode": name.lower(), "group": list(group), "chunk": chunk, "rows": rows}


def runs(session, args, data, own, log):
    """RUNS by island 0's ranks (both: by both islands' at once, each into the
    other), one command an island (its event's ticks); the targets read back."""
    islands_putting = (0, 1) if args.runs == "both" else (0,)
    sources = [session.tensor(data, "uint32", group=own[i]) for i in islands_putting]
    targets = [
        session.empty(data.shape, "uint32", group=own[1 - i]) for i in islands_putting
    ]
    reports = [
        session.empty(data.shape, "uint32", group=own[i]) for i in islands_putting
    ]
    program = km.Program(
        "pages",
        kernels=KERNELS,
        handlers=KERNELS / "pages.h",
        buffers=[
            km.Buffer(
                index=0, format="bfloat16", page_bytes=args.piece * args.page, pages=1
            )
        ],
        compile_args={"reader": sources[0].accessor_args()},
    )
    run = session.load(program, deadline=2.0)
    handler = program.handler("runs")
    _, chunk = session.staging(0)
    log(f"link CPUs' L1: {chunk:,} B a CPU")
    pages = data.shape[0]
    out = {"mode": f"runs-{args.runs}", "chunk": chunk, "rows": [], "error": None}
    submitted = []
    before = run.crossings()
    for i, source, target, report in zip(islands_putting, sources, targets, reports):
        words = [
            source.address,
            target.address,
            pages,
            args.run_pages,
            args.piece,
            report.address,
        ]
        submitted.append(
            (i, run.island(i), run.island(i).submit(handler, own[i], words))
        )
    for i, island, event in submitted:
        try:
            done = island.wait(event, timeout=30.0)
        except km.KomorebiError as error:
            out["error"] = str(error)[:600]
            log(f"RUNS island {i} failed: {out['error']}")
            continue
        seconds = done.ticks / 50e6
        row = {
            "from": i,
            "ms": round(seconds * 1e3, 3),
            "gb_s": round(pages * args.page / seconds / 1e9, 2),
        }
        log(f"RUNS {args.mib} MiB from island {i}: {row['ms']} ms, {row['gb_s']} GB/s")
        row.update(steps(session, reports[islands_putting.index(i)], own[i][1], log))
        out["rows"].append(row)
    moved = run.crossings() - before
    for row in out["rows"]:
        row["links"] = ring_shares(moved, row["from"], row["ms"], log)
    wrong = 0
    for target in targets:
        (got,) = session.read(target)
        wrong += int(
            (got.view(np.uint32).reshape(data.shape) != data).any(axis=1).sum()
        )
    out["wrong_pages"] = wrong
    log(f"targets read back: {wrong} pages wrong")
    try:
        results = run.stop()
        out["kernel_errors"] = [result["error_name"] for result in results]
    except km.KomorebiError as error:
        out["kernel_errors"] = str(error)[:300]
    out["ok"] = bool(
        out["error"] is None and wrong == 0 and out["kernel_errors"] == ["none", "none"]
    )
    return out


RUNS_STEPS = ("line", "read", "piece", "wait")  # pages.h kRuns*, in order


def steps(session, reports, ranks, log):
    """Each rank's RUNS report (pages.h): the share of its cycles in each step, the
    mean over the ranks (and the least and most), and its pieces."""
    (got,) = session.read(reports, first=0, pages=ranks)
    words = got.view(np.uint32).reshape(ranks, -1)[:, : 6 + len(RUNS_STEPS)]
    good = words[:, 0] == 0x52554E53
    if not good.all():
        log(f"RUNS reports: {int((~good).sum())} of {ranks} missing")
        return {"reports_missing": int((~good).sum())}
    total = words[:, 2].astype(np.float64)
    shares = {}
    for k, name in enumerate(RUNS_STEPS):
        share = words[:, 4 + k] / total
        shares[name] = [
            round(float(share.mean()), 3),
            round(float(share.min()), 3),
            round(float(share.max()), 3),
        ]
    for k, name in enumerate(("ring turn", "ring write")):
        share = words[:, 4 + len(RUNS_STEPS) + k] / total
        shares[name] = [
            round(float(share.mean()), 3),
            round(float(share.min()), 3),
            round(float(share.max()), 3),
        ]
    pieces = words[:, 3]
    per_piece = float((words[:, 4 + RUNS_STEPS.index("piece")] / pieces).mean())
    log(
        "RUNS steps, share of a rank's cycles (mean, least, most): "
        + ", ".join(f"{name} {v[0]} ({v[1]}..{v[2]})" for name, v in shares.items())
        + f"; {int(pieces.mean())} pieces a rank, {per_piece:,.0f} cycles a put_piece,"
        + f" {float(total.mean()):,.0f} cycles a rank"
    )
    return {"steps": shares, "pieces": int(pieces.mean()), "piece_cycles": per_piece}


AICLK_HZ = 1.35e9  # the cores' clock (the wall clock the counters read)


def ring_shares(moved, source, ms, log):
    """Each link CPU's put ring from island `source`: the share of the command's time
    it sent (its own send, then waiting for the router's slots)."""
    cycles = ms * 1e-3 * AICLK_HZ
    out = {}
    for (link, here, _), row in sorted(moved.rows.items()):
        if here != source or not row["ring_sending"]:
            continue
        sending, router = row["ring_sending"], row["ring_router"]
        out[link] = {
            "send": round((sending - router) / cycles, 3),
            "router": round(router / cycles, 3),
            "pages": row["pages"],
        }
    log(
        f"RUNS link CPUs from island {source}, share of the command's time: "
        + ", ".join(
            f"link {k} send {v['send']} router {v['router']}" for k, v in out.items()
        )
    )
    return out


def soak(session, args, data, own, log):
    """SOAK commands by island 0's ranks, rounds each, until --calls calls or
    --seconds: asks of --step pages from island 1 (each page checked on the core), or
    puts into island 1 (its target read back at the end); each rank's report."""
    pages = data.shape[0]
    asks = args.soak == "ask"
    source = session.tensor(data, "uint32", group=own[1 if asks else 0])
    target = session.empty(data.shape, "uint32", group=own[1])
    reports = session.empty(data.shape, "uint32", group=own[0])
    program = km.Program(
        "pages",
        kernels=KERNELS,
        handlers=KERNELS / "pages.h",
        buffers=[
            km.Buffer(
                index=0, format="bfloat16", page_bytes=args.step * args.page, pages=1
            )
        ],
        compile_args={"reader": source.accessor_args()},
    )
    run = session.load(program, deadline=2.0)
    handler = program.handler("soak")
    mine = run.island(0)
    group = own[0]
    ranks = group[1]
    per_round = sum(
        -(-(pages // ranks + (r < pages % ranks)) // args.step) for r in range(ranks)
    )
    log(
        f"SOAK {args.soak}: {pages} pages of {args.page} B, {args.step} a call, "
        f"{per_round} calls a round over {ranks} ranks"
    )
    out = {"mode": f"soak-{args.soak}", "calls": 0, "commands": 0, "error": None}
    out["reports_bad"], out["pages_bad"] = [], 0

    def reported():
        """Each rank's report of the command just run: the bad ones kept."""
        (got,) = session.read(reports, first=0, pages=ranks)
        rows = [row[:8].tolist() for row in got.view(np.uint32).reshape(ranks, -1)]
        bad = [
            row
            for row in rows
            if row[0] != 0x534F414B
            or row[2] != 0
            or row[4] != 0xFFFFFFFF
            or row[5] != 0xFFFFFFFF
        ]
        out["reports_bad"] += bad[: 8 - len(out["reports_bad"])]
        out["pages_bad"] += int(sum(row[2] for row in rows if row[0] == 0x534F414B))
        return not bad

    rounds, started = 1, time.monotonic()
    while out["calls"] < args.calls and time.monotonic() - started < args.seconds:
        words = [
            source.address,
            target.address,
            pages,
            args.step,
            rounds,
            reports.address,
            args.page,
        ]
        begun = time.monotonic()
        try:
            mine.wait(mine.submit(handler, group, words), timeout=60.0)
        except km.KomorebiError as error:
            out["error"] = str(error)[:600]
            log(f"SOAK command {out['commands']} failed: {out['error']}")
            reported()
            break
        took = time.monotonic() - begun
        if not reported():
            log(f"SOAK command {out['commands']}: a rank reports a bad page")
            break
        out["calls"] += per_round * rounds
        out["commands"] += 1
        log(f"SOAK: {rounds} rounds in {took:.2f} s, {out['calls']:,} calls")
        # a command near a second: a core's wait limit (about 2.4 s) ends a longer one
        rounds = max(1, min(int(rounds * 1.0 / max(took, 1e-3)), rounds * 8))
    out["seconds"] = round(time.monotonic() - started, 1)
    log(f"reports: {len(out['reports_bad'])} ranks with a bad page or a failure")
    if not asks:
        (back,) = session.read(target)
        back = back.view(np.uint32).reshape(data.shape)
        out["wrong_pages"] = int((back != data).any(axis=1).sum())
        log(f"target read back: {out['wrong_pages']} pages wrong")
    try:
        results = run.stop()
        out["kernel_errors"] = [result["error_name"] for result in results]
    except km.KomorebiError as error:
        out["kernel_errors"] = str(error)[:300]
    out["ok"] = bool(
        out["error"] is None
        and not out["reports_bad"]
        and out["pages_bad"] == 0
        and out.get("wrong_pages", 0) == 0
        and out["kernel_errors"] == ["none", "none"]
        and out["calls"] >= args.calls
    )
    log(
        f"SOAK {args.soak}: {out['calls']:,} calls in {out['seconds']} s, "
        f"ok {out['ok']}"
    )
    return out


def shapes(session, args, source, targets, data):
    """The host's cross of the source into island 1, once a push shape; a shape that
    reads into a target of its own, read back and compared after all have run."""
    buffer, chunk = session.staging(0)
    log(f"link CPUs' L1: {chunk:,} B a CPU at {buffer:#x}")
    timeout = 5.0 + data.nbytes / 1e9
    rows, reading = [], iter(targets[:-1])
    for shape in args.shapes:
        target = targets[-1] if shape & NO_READ else next(reading)
        row = {"shape": hex(shape)}
        try:  # as session.cross, with each command's event: X280 ticks, 50 MHz
            started = time.perf_counter()
            far = session.wait_pages(1, session._across(0, 1), source.pages)
            sent = session.send_pages(
                source, 0, source.pages, target, there=1, shape=shape
            )
            send, landed = session.wait_together([sent, (1, far)], timeout)
            wall = time.perf_counter() - started
            row["wall_ms"] = round(wall * 1e3, 3)
            row["send_ms"] = round(send.ticks / 50e3, 3)
            row["far_ms"] = round(landed.ticks / 50e3, 3)
            row["gb_s"] = round(data.nbytes / (send.ticks / 50e6) / 1e9, 2)
            log(
                f"shape {shape:#04x}: send {row['send_ms']} ms ({row['gb_s']} GB/s), "
                f"far count {row['far_ms']} ms, wall {row['wall_ms']} ms"
            )
        except km.KomorebiError as error:
            row["error"] = str(error)[:400]
            log(f"shape {shape:#04x} failed: {row['error']}")
            rows.append(row)
            break
        if not shape & NO_READ:
            row["target"] = target
        rows.append(row)
    for row in rows:
        target = row.pop("target", None)
        if target is not None:
            (got,) = session.read(target)
            got = got.view(np.uint32).reshape(data.shape)
            row["wrong_pages"] = int((got != data).any(axis=1).sum())
            log(f"shape {row['shape']} read back: {row['wrong_pages']} pages wrong")
    return {"mode": "shapes", "chunk": chunk, "rows": rows}


def where(session, program, tensors):
    """km::island_of(tensor, page) on the first and the last core of each island, for
    the first and last page of every part, against islands.island_of on the host."""
    asked, wrong = [], 0
    handler = program.handler("where")
    for island in range(2):
        run = session.run.island(island)
        cores = (0, session.graph.count(island) - 1)
        for tensor in tensors:
            pages = {0, tensor.pages - 1}
            for start, length in tensor.placement.parts:
                pages |= {start, start + length - 1} if length else set()
            for core in cores:
                for page in sorted(pages):
                    rank = session.graph.first(island) + core  # groups are flat
                    event = run.wait(
                        run.submit(
                            handler, (rank, 1), [tensor.address, page], result=True
                        )
                    )
                    want = islands.island_of(tensor.placement, page, island)
                    asked.append(
                        [island, core, tensor.address, page, event.result, want]
                    )
                    wrong += int(event.result != want)
    log(f"{len(asked)} answers of km::island_of, {wrong} wrong")
    return {"answers": asked, "wrong_answers": wrong}


def links(session, report):
    """After a failed crossing: the link cores' state words and port counts."""
    for island, cores in enumerate(session.link_cores):
        try:
            states = session.run.island(island).state()
        except km.KomorebiError as error:
            log(f"island {island}: no state ({error})")
            continue
        for core in cores:
            state = states[core]
            log(f"island {island} link core {core}: {state}")
            report[f"link_{island}_{core}"] = str(state)[:600]


if __name__ == "__main__":
    main()
