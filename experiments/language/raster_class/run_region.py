"""SUM, HSUM and PLACE on the card for one scene: the hand program, the class, or both
in one session, each program in one load with an idle X280 thread whose two regions
the commands write and read; inputs the host's counters, histograms, items region.
Example: python run_region.py --program both --scene tiny"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(HERE)]

from region_class import REGION_PAGES, RegionPasses  # noqa: E402

import komorebi as km  # noqa: E402
from projects.raster import (  # noqa: E402
    card,
    card_passes,
    card_thread,
    lists,
    passes,
    run,
)
from reference import raster  # noqa: E402

S, K, F = card_passes.S, lists.K, card_thread.F
THREAD = HERE / "thread" / "idle.c"
REGION_BYTES = S["kCoreWordsAt"] + 16 * S["kMaxFillCores"]
FRAME_WORDS = [16384 + 4096 * i for i in range(16)]  # a directory's frame words


def items_region(entries, rows, workers):
    """The items region the thread's middle pass leaves for PLACE (setup.h's layout),
    from the host's middle pass: the items table, each item's core, count and header
    slot, each core's first page, pages, items and end slot; and the pages' count."""
    total = entries.sum(axis=0)
    partition = lists.assign(passes.cost_of(entries, rows), workers, "balanced")
    region = np.zeros(REGION_BYTES // 4, dtype=np.uint32)
    region[: S["kItemsBytes"] // 4] = passes.item_table(
        [(tile, 0, 32) for tile in range(lists.TILES)]
    )
    region[S["kFrameWordsAt"] // 4 : S["kFrameWordsAt"] // 4 + 16] = FRAME_WORDS
    item_words = region[S["kItemWordsAt"] // 4 :]
    core_words = region[S["kCoreWordsAt"] // 4 :]
    first_page = 0
    for core, own in enumerate(partition):
        slot = 16 * first_page
        for item in own:
            item_words[2 * item] = core | int(total[item]) << 8
            if total[item]:
                item_words[2 * item + 1] = slot
                slot += 1 + int(total[item])
        pages = -(-(slot + 1 - 16 * first_page) // lists.PAGE_SLOTS)
        core_words[4 * core : 4 * core + 4] = [first_page, pages, len(own), slot]
        first_page += pages
    return region, first_page


def inputs(scene, workers):
    """The host's part: COUNT's counters and HIST's histograms of the cut tiles (the
    heavy ones, or the first 256 with entries), the items region and its middle."""
    records = raster.setup(scene)
    triangles = len(scene["indices"])
    entries, rows = passes.counters(records, triangles, workers)
    starts, ends = passes.histograms(records, triangles, workers)
    heavy = passes.heavy_tiles(entries, rows, workers)
    cut = [tile for tile, _ in heavy] or list(np.flatnonzero(entries.sum(0))[:256])
    halves = np.stack([starts[:, cut], ends[:, cut]], axis=2).astype(np.uint16)
    counters = np.zeros((workers, S["kCountBytes"] // 4), dtype=np.uint32)
    counters[:, : 2 * lists.TILES] = np.concatenate([entries, rows], axis=1)
    region, list_pages = items_region(entries, rows, workers)
    return {
        "counters": counters,
        "histograms": halves.view(np.uint32).reshape(workers, -1),
        "cut": len(cut),
        "region": region,
        "list_pages": list_pages,
        "middle": passes.middle(entries, rows, workers),
    }


def tensors(session, given, workers):
    """One program's DRAM tensors: the inputs, and zeros for what the passes write."""
    return {
        "counters": card.tensor(session, given["counters"]),
        "prefixes": card.tensor(session, np.zeros((workers, S["kSlotsBytes"] // 4))),
        "histograms": card.tensor(session, given["histograms"]),
        "hprefix": card.tensor(
            session, np.zeros((workers, S["kHprefixPageBytes"] // 4))
        ),
        "directory": card.tensor(session, np.zeros((workers, S["kDirPageBytes"] // 4))),
        "lists": card.tensor(
            session, np.zeros((given["list_pages"], S["kListPageBytes"] // 4))
        ),
    }


def with_thread(program, given):
    """The program with the idle thread on cluster 1: regions sums and items."""
    program.thread(
        cluster=card_thread.CLUSTER,
        sources=[THREAD],
        regions={"sums": F["RF_SUMS_BYTES"], "items": given["region"]},
        deadline=10.0,
    )
    return program


def submitted(loaded, commands, who, workers):
    """Each command in turn on every core, waited for; its microseconds."""
    times = []
    try:
        for name, handler, words in commands:
            handle = loaded.submit(handler, (0, workers), args=words)
            event = loaded.wait(handle, timeout=10.0)
            card.log(f"{who} {name}: {event.kind}")
            times.append(event.microseconds)
    finally:
        loaded.stop()
    return times


def hand(session, given, placed, workers):
    """The project's passes program: SUM, HSUM, PLACE, the region's bank and address
    in the command's words."""
    loaded = session.load(
        with_thread(card_passes.program(), given), deadline=1.0, thread="card"
    )
    bank, sums = loaded.threads[card_thread.CLUSTER].regions.address("sums")
    _, items = loaded.threads[card_thread.CLUSTER].regions.address("items")
    words = {
        "sum": [0, placed["counters"].address, placed["prefixes"].address, sums],
        "hsum": [given["cut"], placed["histograms"].address],
        "place": [items, placed["directory"].address, 0, 0, 0],
    }
    words["sum"] += [0, workers, workers, 0, bank + 1]
    words["hsum"] += [placed["hprefix"].address, sums, 64 * given["cut"]]
    words["hsum"] += [workers, workers, 0, bank + 1]
    words["place"] += [workers, workers, placed["lists"].address, bank + 1]
    commands = [
        (name, S[f"kHandler{name.capitalize()}"], words[name])
        for name in ("sum", "hsum", "place")
    ]
    return loaded, submitted(loaded, commands, "hand", workers)


def of_class(session, given, placed, workers):
    """The class: sum_tiles, hsum_tiles, place_items, each region by its name."""
    program = RegionPasses(log=card.log, session=session, depth=1, **placed)
    loaded = session.load(with_thread(program, given), deadline=1.0, thread="card")
    p = placed
    commands = [
        (
            "sum",
            RegionPasses.sum_tiles,
            (p["counters"], p["prefixes"], "sums", workers),
        ),
        (
            "hsum",
            RegionPasses.hsum_tiles,
            (
                given["cut"],
                p["histograms"],
                p["hprefix"],
                "sums",
                workers,
                given["cut"],
            ),
        ),
        (
            "place",
            RegionPasses.place_items,
            ("items", p["directory"], workers, p["lists"]),
        ),
    ]
    return loaded, submitted(loaded, commands, "class", workers)


def results(session, loaded, placed):
    """What the passes wrote: prefixes, heavy prefixes, directory, lists, sums."""
    names = ("prefixes", "hprefix", "directory", "lists")
    read = card.read(session, *(placed[name] for name in names))
    out = {name: pages.reshape(-1) for name, pages in zip(names, read)}
    out["sums"] = loaded.threads[card_thread.CLUSTER].read("sums").view(np.uint32)
    return out


def against_host(given, got, workers):
    """The hand's directory (its frame words apart) and list marks against the host's
    middle pass of the same items: the region's layout checked."""
    directory = got["directory"].reshape(workers, -1).copy()
    frame = directory[:, K["kDirFrame"] : K["kDirFrame"] + 16].copy()
    directory[:, K["kDirFrame"] :] = 0
    pages = given["middle"]["pages"].reshape(-1)
    return {
        "directory_differ": int((directory != given["middle"]["directory"]).sum()),
        "frame_words_right": bool((frame == FRAME_WORDS).all()),
        "lists_differ": int((got["lists"][: pages.size] != pages).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--program", choices=("hand", "class", "both"), required=True)
    parser.add_argument("--scene", default="tiny")
    arguments = parser.parse_args()
    runs = [("hand", hand), ("class", of_class)]
    runs = [r for r in runs if arguments.program in (r[0], "both")]
    scene, _ = run.load(arguments.scene)
    ok = True
    (HERE / "build").mkdir(exist_ok=True)
    with km.Session(log=card.log) as session:
        workers = card.workers(session)
        given = inputs(scene, workers)
        assert REGION_BYTES == 4096 * REGION_PAGES
        placed = {name: tensors(session, given, workers) for name, _ in runs}
        card.log(f"{given['cut']} heavy tiles, {given['list_pages']} list pages")
        got = {}
        for name, program in runs:
            loaded, microseconds = program(session, given, placed[name], workers)
            status = card.check_run(loaded, workers, 3, microseconds)
            got[name] = results(session, loaded, placed[name])
            report = {
                "program": name,
                "run": status,
                "microseconds": dict(zip(("sum", "hsum", "place"), microseconds)),
            }
            if name == "hand":
                report["hand_against_host"] = against_host(given, got[name], workers)
            ok = ok and status["ok"]
            print(json.dumps(report, default=int), flush=True)
            np.savez(HERE / "build" / f"region_{name}.npz", **got[name])
        if len(got) == 2:
            differ = {
                key: int((got["hand"][key] != got["class"][key]).sum())
                for key in got["hand"]
            }
            words = {key: int(value.size) for key, value in got["hand"].items()}
            ok = ok and not any(differ.values())
            print(
                json.dumps({"class_against_hand": differ, "words": words}), flush=True
            )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
