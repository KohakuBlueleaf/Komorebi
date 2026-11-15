"""The fill pass on the card from the host-binned path's lists of one scene: the hand
program (the project's kernels), the class (load_empty once, then fill_tiles), or both
in one session; frame, profile pages and times to build/<program>.npz.
Example: python run_fill.py --program both --scene tiny --reps 5"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(HERE)]

from fill_class import PLANES, FillPass, made  # noqa: E402

import komorebi as km  # noqa: E402
from projects.raster import card, lists, run  # noqa: E402

K = lists.K
# The empty tile's planes, as the hand's writer builds them: depth 0, the background
# colour, triangle -1.
EMPTY_WORDS = (0, 0x3D4CCCCD, 0x3D4CCCCD, 0x3DA3D70A, 0xBF800000)


def aiclk():
    """The card's AICLK in MHz (card 1), or 0 when it cannot be read."""
    try:
        return int(Path("/sys/class/tenstorrent/tenstorrent!1/tt_aiclk").read_text())
    except (OSError, ValueError):
        return 0


def hand(session, prepared, uploaded, reps):
    """The project's own program and driver loop: (run status, microseconds)."""
    program = card.program(uploaded)
    loaded = session.load(program, deadline=1.0)
    card.log(f"hand loaded; {reps} FILL commands")
    microseconds = card.loop(
        loaded, K["kHandlerFill"], card.words(uploaded, PLANES), reps
    )
    loaded.stop()
    status = card.check_run(loaded, card.workers(session), reps, microseconds)
    return status, microseconds


def of_class(session, prepared, uploaded, reps):
    """The class: the same lists, its own directory and frame, one load_empty, then
    `reps` commands of fill, one at a time: (run status, microseconds)."""
    workers = card.workers(session)
    planes = np.repeat(np.array(EMPTY_WORDS, np.uint32)[:, None], 1024, axis=1)
    tensors = {
        "planes": card.tensor(session, planes),
        "directory": uploaded["dir"],
        "lists": uploaded["lists"],
        "frame": uploaded["frame"][0],
        "profile": uploaded["prof"],
    }
    program = made(session, tensors, card.log)
    loaded = session.load(program, deadline=1.0)
    card.log(f"class loaded; load_empty, {reps} fill_tiles commands")
    args = tuple(tensors[name] for name in ("directory", "lists", "frame", "profile"))
    try:
        handles = [
            loaded.submit(FillPass.load_empty, (0, workers), args=(tensors["planes"],))
        ]
        handles += [
            loaded.submit(FillPass.fill_tiles, (0, workers), args=args)
            for _ in range(reps)
        ]
        events = loaded.wait(handles, timeout=10.0 + 1.0 * reps)
    finally:
        loaded.stop()
    events = events if isinstance(events, list) else [events]
    microseconds = np.array([event.microseconds for event in events])
    print(json.dumps({"load_empty_microseconds": float(microseconds[0])}), flush=True)
    status = card.check_run(loaded, workers, reps + 1, microseconds)
    return status, microseconds[1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--program", choices=("hand", "class", "both"), required=True)
    parser.add_argument("--scene", default="tiny")
    parser.add_argument("--reps", type=int, default=5)
    arguments = parser.parse_args()
    runs = {"hand": [("hand", hand)], "class": [("class", of_class)]}.get(
        arguments.program, [("hand", hand), ("class", of_class)]
    )
    scene, _ = run.load(arguments.scene)
    ok = True
    (HERE / "build").mkdir(exist_ok=True)
    with km.Session(log=card.log) as session:
        workers = card.workers(session)
        prepared = run.prepare(scene, workers)
        outputs = len(prepared["tiles"]) * PLANES
        # every tensor written before the first load (none while a program runs)
        uploads = {
            name: card.upload(
                session, prepared["pages"], prepared["directory"], outputs
            )
            for name, _ in runs
        }
        card.log(f"uploaded {len(prepared['pages'])} list pages for {len(runs)} runs")
        records = {}
        for name, program in runs:
            status, microseconds = program(
                session, prepared, uploads[name], arguments.reps
            )
            report = {
                "program": name,
                "run": status,
                "fill_microseconds": [float(value) for value in microseconds],
                "median_microseconds": float(np.median(microseconds)),
                "mhz": aiclk(),
            }
            ok = ok and status["ok"]
            records[name] = report
            print(json.dumps(report, default=int), flush=True)
        for name, _ in runs:
            (profile,) = card.read(session, uploads[name]["prof"])
            frame = card.read_frame(session, uploads[name])
            np.savez(
                HERE / "build" / f"{name}.npz",
                frame=frame,
                profile=profile,
                microseconds=np.array(records[name]["fill_microseconds"]),
            )
            card.log(f"{name}: frame and profile saved")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
