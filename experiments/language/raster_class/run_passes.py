"""COUNT, HIST and SETUP on the card for one scene: the hand program, the class, or both
in one session, each in one load; HIST's and SETUP's inputs the host's middle pass on
the reference's counters. Counters, histograms, lists and times to build/passes_*.npz.
Example: python run_passes.py --program both --scene tiny"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(HERE)]

from passes_class import SetupPasses, made  # noqa: E402

import komorebi as km  # noqa: E402
from projects.raster import card, card_passes, passes, run  # noqa: E402
from reference import raster  # noqa: E402

S = card_passes.S


def inputs(scene, workers):
    """The host's part: the mesh blocks, the camera page, the tiles HIST writes (the
    heavy tiles, or the first 256 with entries when none is heavy) and the middle
    pass on the reference's counters (dict)."""
    blocks = passes.mesh_blocks(scene)
    records = raster.setup(scene)
    entries, rows = passes.counters(records, len(scene["indices"]), workers)
    heavy = passes.heavy_tiles(entries, rows, workers)
    cut = heavy or [(int(tile), 1) for tile in np.flatnonzero(entries.sum(0))[:256]]
    bands = None
    if heavy:
        starts, ends = passes.histograms(records, len(scene["indices"]), workers)
        tiles = [tile for tile, _ in heavy]
        starts, ends = starts[:, tiles], ends[:, tiles]
        bands = passes.items_of(entries, rows, heavy, starts, ends)
    middle = passes.middle(entries, rows, workers, bands=bands)
    heavy_page = np.zeros((1, S["kHeavyBytes"] // 4), dtype=np.uint32)
    heavy_page[0, 0] = len(cut)
    heavy_page[0, 1 : 1 + len(cut)] = [tile for tile, _ in cut]
    next_slots = np.zeros((workers, S["kSlotsBytes"] // 4), dtype=np.uint32)
    next_slots[:, : len(middle["items"])] = middle["next"]
    return {
        "blocks": blocks,
        "camera": card_passes.camera_page(scene),
        "heavy": heavy_page,
        "cut": len(cut),
        "middle": middle,
        "next_slots": next_slots,
        "items": passes.item_table(middle["items"]),
    }


def tensors(session, given, workers, rows):
    """One program's DRAM tensors; `rows`, the slots' and items' page words (the
    hand's 2,048, the class's fields 1,024)."""
    return {
        "mesh": card.tensor(session, given["blocks"]),
        "camera": card.tensor(session, given["camera"]),
        "counters": card.tensor(session, np.zeros((workers, S["kCountBytes"] // 4))),
        "heavy": card.tensor(session, given["heavy"]),
        "histograms": card.tensor(session, np.zeros((workers, 16 * given["cut"]))),
        "lists": card.tensor(session, given["middle"]["pages"]),
        "slots": card.tensor(session, given["next_slots"].reshape(-1, rows)),
        "items": card.tensor(session, given["items"].reshape(-1, rows)),
    }


def hand(session, given, placed):
    """The project's passes program: COUNT, HIST, SETUP in one load (microseconds)."""
    workers = card.workers(session)
    blocks = len(given["blocks"])
    loaded = session.load(card_passes.program(), deadline=1.0)
    card.log("hand loaded: COUNT, HIST, SETUP")
    count = [placed["mesh"].address, placed["counters"].address, blocks]
    count += [placed["camera"].address, 0, workers]
    hist = [placed["heavy"].address, placed["histograms"].address, 0, 0]
    hist += [64 * given["cut"], workers]
    setup = [placed["mesh"].address, placed["lists"].address, blocks]
    setup += [placed["camera"].address, placed["slots"].address, workers]
    setup += [given["middle"]["slots"], placed["items"].address]
    events = []
    try:
        for handler, words in (("Count", count), ("Hist", hist), ("Setup", setup)):
            handle = loaded.submit(
                S[f"kHandler{handler}"], group=(0, workers), args=words
            )
            events.append(loaded.wait(handle, timeout=10.0))
            card.log(f"hand {handler}: {events[-1].kind}")
    finally:
        loaded.stop()
    return loaded, [event.microseconds for event in events]


def of_class(session, given, placed):
    """The class: count_tiles, hist_tiles, setup_lists in one load (microseconds)."""
    workers = card.workers(session)
    blocks = len(given["blocks"])
    examples = {
        "mesh": placed["mesh"],
        "counters": placed["counters"],
        "camera": placed["camera"],
        "heavy": placed["heavy"],
        "histograms": placed["histograms"],
        "lists": placed["lists"],
        "slot_pages": placed["slots"],
        "item_words": placed["items"],
    }
    program = made(session, examples, card.log)
    loaded = session.load(program, deadline=1.0)
    card.log("class loaded: count_tiles, hist_tiles, setup_lists")
    count = (placed["mesh"], placed["counters"], blocks, placed["camera"])
    hist = (placed["heavy"], placed["histograms"], 64 * given["cut"])
    setup = (placed["mesh"], placed["lists"], blocks, placed["camera"])
    setup += (placed["slots"], given["middle"]["slots"], placed["items"])
    events = []
    try:
        for name, handler, words in (
            ("count_tiles", SetupPasses.count_tiles, count),
            ("hist_tiles", SetupPasses.hist_tiles, hist),
            ("setup_lists", SetupPasses.setup_lists, setup),
        ):
            handle = loaded.submit(handler, (0, workers), args=words)
            events.append(loaded.wait(handle, timeout=10.0))
            card.log(f"class {name}: {events[-1].kind}")
    finally:
        loaded.stop()
    return loaded, [event.microseconds for event in events]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--program", choices=("hand", "class", "both"), required=True)
    parser.add_argument("--scene", default="tiny")
    arguments = parser.parse_args()
    runs = {"hand": [("hand", hand, 2048)], "class": [("class", of_class, 1024)]}
    runs = runs.get(
        arguments.program, [("hand", hand, 2048), ("class", of_class, 1024)]
    )
    scene, _ = run.load(arguments.scene)
    ok = True
    (HERE / "build").mkdir(exist_ok=True)
    with km.Session(log=card.log) as session:
        workers = card.workers(session)
        given = inputs(scene, workers)
        # every tensor written before the first load (none while a program runs)
        placed = {
            name: tensors(session, given, workers, rows) for name, _, rows in runs
        }
        card.log(f"{len(given['blocks'])} blocks, {given['cut']} tiles for HIST")
        microseconds = {}
        for name, program, _ in runs:
            loaded, microseconds[name] = program(session, given, placed[name])
            status = card.check_run(loaded, workers, 3, microseconds[name])
            report = {
                "program": name,
                "run": status,
                "microseconds": dict(
                    zip(("count", "hist", "setup"), microseconds[name])
                ),
            }
            ok = ok and status["ok"]
            print(json.dumps(report, default=int), flush=True)
        for name, _, _ in runs:
            counters, histograms, list_pages = card.read(
                session,
                placed[name]["counters"],
                placed[name]["histograms"],
                placed[name]["lists"],
            )
            np.savez(
                HERE / "build" / f"passes_{name}.npz",
                counters=counters,
                histograms=histograms,
                lists=list_pages,
                microseconds=np.array(microseconds[name]),
            )
            card.log(f"{name}: counters, histograms and lists saved")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
