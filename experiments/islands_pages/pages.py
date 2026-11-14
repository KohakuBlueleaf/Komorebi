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
    return parser.parse_args()


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    report = {"cards": cards, "links": args.links, "page": args.page}
    pages = args.mib * (1 << 20) // args.page
    data = np.random.default_rng(1).integers(
        0, 1 << 32, (pages, args.page // 4), dtype=np.uint64
    )
    data = data.astype(np.uint32)
    with km.Session(cards=cards, links=args.links, log=log) as session:
        graph = session.graph
        own = [(graph.first(i), graph.count(i)) for i in range(2)]
        report["link_cores"] = [list(cores) for cores in session.link_cores]
        report["channels"] = [list(channels) for channels in session.channels]
        source = session.tensor(data, "uint32", like=own[0])
        target = session.empty(data.shape, "uint32", like=own[1])
        split = session.empty(data.shape, "uint32")
        copy = session.empty(data.shape, "uint32", copy=True)
        tensors = (source, target, split, copy)
        report["parts"] = [list(tensor.placement.parts) for tensor in tensors]
        log(f"source {source.placement.parts}, target {target.placement.parts}")
        program = km.Program(
            "pages",
            kernels=KERNELS,
            handlers=KERNELS / "pages.h",
            compile_args={
                "writer": [w for tensor in tensors for w in tensor.placement_args()]
            },
        )
        run = session.load(program, deadline=2.0)
        log("loaded on both islands")
        if args.where:
            report.update(where(session, program, tensors))
            run.stop()
            report["ok"] = report["wrong_answers"] == 0
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


def where(session, program, tensors):
    """km::island_of(tensor, page) on the first and the last core of each island, for
    the first and last page of every part, against islands.island_of on the host."""
    asked, wrong = [], 0
    handler = program.handler("where")
    for island in range(2):
        run = session.run.island(island)
        cores = (0, session.graph.count(island) - 1)
        for index, tensor in enumerate(tensors):
            pages = {0, tensor.pages - 1}
            for start, length in tensor.placement.parts:
                pages |= {start, start + length - 1} if length else set()
            for core in cores:
                for page in sorted(pages):
                    event = run.wait(
                        run.submit(handler, (core, 1), [index, page], result=True)
                    )
                    want = islands.island_of(tensor.placement, page, island)
                    asked.append([island, core, index, page, event.result, want])
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
