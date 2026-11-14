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
    with km.Session(cards=cards, links=args.links, log=log) as session:
        if args.chunk:
            for one in session.sessions:
                one.transfer_chunk = args.chunk * 1024
        graph = session.graph
        own = [(graph.first(i), graph.count(i)) for i in range(2)]
        report["link_cores"] = [list(cores) for cores in session.link_cores]
        report["channels"] = [list(channels) for channels in session.channels]
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
    return {"mode": name.lower(), "group": list(group), "rows": rows}


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
