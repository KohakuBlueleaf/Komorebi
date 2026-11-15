"""Where SUM, HSUM and PLACE spend their time, on the card for one scene: the hand
program, the region class and region_timed.py's handlers, each load traced, with each
command's per-core trace and stamps kept in build/timed.npz for timed_tables.py.
Example: python run_timed.py --scene tiny"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(HERE)]

from region_class import REGION_PAGES, RegionPasses  # noqa: E402
from region_timed import (  # noqa: E402
    RegionBlock,
    RegionBlockTimed,
    RegionFn,
    RegionHand,
    RegionPlain,
    RegionRoute,
)
from run_region import (  # noqa: E402
    REGION_BYTES,
    inputs,
    results,
    tensors,
    with_thread,
)

import komorebi as km  # noqa: E402
from projects.raster import card, card_passes, card_thread, run  # noqa: E402

S = card_passes.S
REPEAT = 3  # each timed command this many times, for its spread
TRACE = 64
TIMED = {
    "route": RegionRoute,
    "block": RegionBlock,
    "hand": RegionHand,
    "plain": RegionPlain,
    "btimed": RegionBlockTimed,
    "fn": RegionFn,
}


def traced(session, program, given, commands, workers, kept, who):
    """Loads the program traced, runs each (name, handler, words) once in turn on every
    core; keeps each command's microseconds and per-core trace rows in `kept` under
    who/name/n; returns the run (stopped) and the per-command microseconds."""
    loaded = session.load(
        with_thread(program, given), deadline=1.0, thread="card", trace=TRACE
    )
    handles, times = [], {}
    try:
        for name, handler, words in commands:
            words = [w(loaded) if callable(w) else w for w in words]
            handle = loaded.submit(handler, (0, workers), args=words)
            event = loaded.wait(handle, timeout=10.0)
            if event.kind != "done":
                card.log(f"{who} {name}: {event.kind}")
            times.setdefault(name, []).append(event.microseconds)
            handles.append((name, len(times[name]) - 1, handle))
    finally:
        loaded.stop()
    trace = loaded.trace()
    kept[f"{who}/mhz"] = np.array(trace.mhz)
    for name, n, handle in handles:
        rows = trace.command(handle)
        if rows is not None:
            kept[f"{who}/{name}/{n}/trace"] = np.asarray(rows)
    for name, values in times.items():
        kept[f"{who}/{name}/us"] = np.array(values)
    return loaded, times


def region_word(name):
    """A region's bank address, from the loaded run's thread."""

    def word(loaded):
        return loaded.threads[card_thread.CLUSTER].regions.address(name)[1]

    return word


def bank_word(loaded):
    """The hand's a8: the regions' bank + 1."""
    return loaded.threads[card_thread.CLUSTER].regions.address("sums")[0] + 1


def hand_commands(given, p, profile, workers):
    """The hand's SUM with no cores (its cost of a command), SUM, HSUM, PLACE, and a
    last PLACE with its writer profile (a9)."""
    sums, items = region_word("sums"), region_word("items")
    sum_words = [0, p["counters"].address, p["prefixes"].address, sums, 0]
    hsum_words = [given["cut"], p["histograms"].address, p["hprefix"].address, sums]
    hsum_words += [64 * given["cut"], workers, workers, 0, bank_word]
    place_words = [items, p["directory"].address, 0, 0, 0, workers, workers]
    place_words += [p["lists"].address, bank_word]
    none = sum_words + [0, workers, 0, bank_word]
    commands = [("none", S["kHandlerSum"], none)] * REPEAT
    whole = sum_words + [workers, workers, 0, bank_word]
    commands += [("sum", S["kHandlerSum"], whole)] * REPEAT
    commands += [("hsum", S["kHandlerHsum"], hsum_words)] * REPEAT
    commands += [("place", S["kHandlerPlace"], place_words)] * REPEAT
    commands += [("place_profile", S["kHandlerPlace"], place_words + [profile.address])]
    return commands


def class_commands(given, p, workers):
    """The region class's three handlers, each REPEAT times."""
    sum_words = (p["counters"], p["prefixes"], "sums", workers)
    hsum_words = (given["cut"], p["histograms"], p["hprefix"], "sums", workers)
    hsum_words += (given["cut"],)
    place_words = ("items", p["directory"], workers, p["lists"])
    commands = [("sum", RegionPasses.sum_tiles, sum_words)] * REPEAT
    commands += [("hsum", RegionPasses.hsum_tiles, hsum_words)] * REPEAT
    commands += [("place", RegionPasses.place_items, place_words)] * REPEAT
    return commands


def timed_commands(route, given, p, stamps, workers):
    """region_timed's handlers of one route ("route", "block" or "hand"): each REPEAT
    times unstamped, then once stamped (stamps[pass]); "route" leads with idle."""
    cut = given["cut"]
    sum_words = (p["counters"], p["prefixes"], "sums", workers)
    hsum_words = (cut, p["histograms"], p["hprefix"], "sums", workers, cut)
    place_words = ("items", p["directory"], workers, p["lists"])
    made = TIMED[route]
    commands = [("idle", RegionRoute.idle, ())] * REPEAT if route == "route" else []
    for name, words in (
        ("sum", sum_words),
        ("hsum", hsum_words),
        ("place", place_words),
    ):
        handler = getattr(made, f"{name}_{route}", None)
        if handler is None:
            continue
        if route in ("block", "plain", "fn"):
            commands += [(name, handler, words)] * REPEAT
            continue
        commands += [(name, handler, words + (0, stamps[name]))] * REPEAT
        commands += [(f"{name}_stamped", handler, words + (1, stamps[name]))]
    if route == "plain":
        ranges = ("items", stamps["directory"], workers, stamps["lists"])
        commands += [("ranges", RegionPlain.place_ranges, ranges)] * REPEAT
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="tiny")
    parser.add_argument("--loads", default="hand,class,route,block,hand_route")
    parser.add_argument("--out", default="timed.npz")
    parser.add_argument("--against", default=None, help="an earlier out's hand words")
    arguments = parser.parse_args()
    scene, _ = run.load(arguments.scene)
    (HERE / "build").mkdir(exist_ok=True)
    kept, got, report = {}, {}, {}
    ok = True
    with km.Session(log=card.log) as session:
        workers = card.workers(session)
        given = inputs(scene, workers)
        assert REGION_BYTES == 4096 * REGION_PAGES
        names = ("hand", "class", "route", "block", "hand_route", "plain", "btimed")
        names += ("ranges", "fn")
        placed = {name: tensors(session, given, workers) for name in names}
        profile = card.tensor(session, np.zeros((workers, S["kSProfPageBytes"] // 4)))
        stamps = {
            route: {
                name: card.tensor(session, np.zeros((workers, 64)))
                for name in ("sum", "hsum", "place")
            }
            for route in ("route", "hand_route", "btimed")
        }
        loads = [
            (
                "hand",
                card_passes.program,
                hand_commands(given, placed["hand"], profile, workers),
            ),
            ("class", None, class_commands(given, placed["class"], workers)),
        ]
        routes = [("route", "route"), ("block", "block"), ("hand", "hand_route")]
        routes += [("plain", "plain"), ("btimed", "btimed"), ("fn", "fn")]
        for route, name in routes:
            extra = placed["ranges"] if route == "plain" else stamps.get(name)
            loads.append(
                (
                    name,
                    route,
                    timed_commands(route, given, placed[name], extra, workers),
                )
            )
        chosen = arguments.loads.split(",")
        for who, make, commands in [load for load in loads if load[0] in chosen]:
            if make is None:
                program = RegionPasses(
                    log=card.log, session=session, depth=1, **placed[who]
                )
            elif make in TIMED:
                extra = {"stamps": stamps[who]["sum"]} if who in stamps else {}
                program = TIMED[make](
                    log=card.log, session=session, depth=1, **placed[who], **extra
                )
            else:
                program = make()
            loaded, times = traced(
                session, program, given, commands, workers, kept, who
            )
            done = [us for values in times.values() for us in values]
            status = card.check_run(loaded, workers, len(commands), done)
            got[who] = results(session, loaded, placed[who])
            if who == "plain":  # place_ranges's own directory and lists
                got["ranges"] = dict(got[who])
                for key in ("directory", "lists"):
                    (pages,) = card.read(session, placed["ranges"][key])
                    got["ranges"][key] = pages.reshape(-1)
            ok = ok and status["ok"]
            report[who] = {
                "run": status,
                "us": {k: [round(v, 2) for v in vs] for k, vs in times.items()},
            }
            print(json.dumps({who: report[who]}, default=int), flush=True)
        for route in stamps:
            for name, tensor in stamps[route].items():
                (pages,) = card.read(session, tensor)
                kept[f"{route}/{name}_stamped/stamps"] = pages.reshape(workers, -1)
        (pages,) = card.read(session, profile)
        kept["hand/place_profile/profile"] = pages.reshape(workers, -1)
    if "hand" not in got and arguments.against:
        earlier = np.load(HERE / "build" / arguments.against)
        got["hand"] = {
            key: earlier[f"hand/out/{key}"]
            for key in ("prefixes", "hprefix", "directory", "lists", "sums")
        }
    for who in got:
        for key, value in got[who].items():
            kept[f"{who}/out/{key}"] = value
    compared = {  # what each load writes: the batched routes no PLACE, fn only PLACE
        "block": ("prefixes", "hprefix", "sums"),
        "btimed": ("prefixes", "hprefix", "sums"),
        "fn": ("directory", "lists"),
    }
    differ = {
        who: {
            key: int((got["hand"][key] != got[who][key]).sum())
            for key in compared.get(who, got["hand"])
        }
        for who in got
        if who != "hand" and "hand" in got
    }
    ok = ok and not any(any(counts.values()) for counts in differ.values())
    print(json.dumps({"against_hand": differ}), flush=True)
    np.savez(HERE / "build" / arguments.out, **kept)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
