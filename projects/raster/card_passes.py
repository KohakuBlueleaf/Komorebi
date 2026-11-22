"""COUNT and SETUP on the card: program, mesh and camera upload, LANES check of the
records, and a frame (COUNT, the host's middle pass, SETUP, the fill of `card.py`)
checked against the reference. `python -m projects.raster.card_passes --scene tiny
[--frame]` prints one JSON line; exits 1 unless exact."""

import argparse
import contextlib
import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi.bench import power
from projects.raster import card, lists, passes, run
from reference import raster

KERNELS = lists.HERE / "kernels"
SETUP_H = KERNELS / "setup.h"
S = {
    name: int(value, 0) for name, value in lists.CONSTEXPR.findall(SETUP_H.read_text())
}


def program():
    """The passes' three kernels on every core; one command in flight per core."""
    return km.Program(
        "raster_passes",
        kernels={
            "reader": KERNELS / "setup_reader.cpp",
            "compute": KERNELS / "setup_compute.cpp",
            "writer": KERNELS / "setup_writer.cpp",
        },
        buffers=[
            km.Buffer(S["kCbMesh"], "uint32", S["kMeshPageBytes"], S["kMeshPages"]),
            km.Buffer(S["kCbRec"], "uint32", S["kRecBytes"], S["kRecPages"]),
            km.Buffer(S["kCbSlots"], "uint32", S["kSlotsBytes"], 1),
            km.Buffer(S["kCbWork"], "uint32", S["kWorkBytes"], 1),
            km.Buffer(S["kCbCount"], "uint32", S["kCountBytes"], 1),
            km.Buffer(S["kCbOut"], "uint32", S["kOutBytes"], S["kOutPages"]),
            km.Buffer(S["kCbHist"], "uint32", S["kHistBytes"], 1),
            km.Buffer(S["kCbItems"], "uint32", S["kItemsBytes"], 1),
            km.Buffer(S["kCbSProf"], "uint32", S["kSProfBytes"], 1),
        ],
        compute={"fp32_dest_acc_en": True, "enable_trisc2_rvv": "if_present"},
        depth=1,
        handlers=SETUP_H,
    )


def camera_page(scene):
    """The camera's 16 float32 numbers as one page."""
    camera = np.asarray(scene["camera"], dtype=np.float32)
    return camera.view(np.uint32).reshape(1, S["kCamWords"])


def expected_lanes(scene):
    """Per triangle (padded to blocks): the reference's record words where it keeps the
    whole triangle, its box, and the triangles kept whole (dict of arrays)."""
    triangle_count = len(scene["indices"])
    records = raster.setup(scene)
    packed = lists.pack_records(records)
    camera_x, camera_y, camera_z = raster.camera_space(scene)
    beyond = (camera_z > scene["camera"][15])[scene["indices"]].sum(axis=1)
    whole = beyond == 3
    first = np.searchsorted(records["tri"], np.arange(triangle_count))
    kept = np.zeros(triangle_count, dtype=bool)
    has_record = first < len(records["tri"])
    kept[has_record] = (
        records["tri"][first[has_record]] == np.arange(triangle_count)[has_record]
    )
    kept &= whole
    return {
        "cards": packed,
        "first": first,
        "kept": kept,
        "crossing": (beyond == 1) | (beyond == 2),
        "recs": records,
    }


def check_lanes(scene, got, seen):
    """Differences between the card's records (blocks x 32 lanes, 32 words) and the
    reference over the triangles `seen`, as counts."""
    expected = expected_lanes(scene)
    triangle_count = len(scene["indices"])
    got = got.reshape(-1, 32)[:triangle_count]
    seen = seen[:triangle_count]
    near = np.float32(scene["camera"][15]).view(np.int32)
    corner_z = got[:, S["kRawZ"] : S["kRawZ"] + 3].view(np.int32)
    area = got[:, S["kRawArea"]] & 0x7FFFFFFF
    box = got[:, S["kRawX0"] : S["kRawY1"] + 1].astype(np.int64) - S["kMagic"]
    # the pack thread's rule: every corner beyond the near distance, a non-zero area,
    # a box that is not empty
    kept = (
        (corner_z > near).all(axis=1)
        & (area != 0)
        & (box[:, 0] <= box[:, 1])
        & (box[:, 2] <= box[:, 3])
    )
    both = kept & expected["kept"] & seen
    reference_words = expected["cards"][expected["first"][both]]
    words = [*range(0, 9), *range(12, 24)]
    differ = got[both][:, words] != reference_words[:, words]
    box_differ = box[both] != expected["recs"]["box"][expected["first"][both]]
    return {
        "triangles": int(triangle_count),
        "kept_card": int((kept & seen).sum()),
        "kept_ref_whole": int((expected["kept"] & seen).sum()),
        "checked": int(seen.sum()),
        "kept_disagree": int(((kept != expected["kept"]) & seen).sum()),
        "crossing": int((expected["crossing"] & seen).sum()),
        "words_differ": int(differ.sum()),
        "records_differ": int(differ.any(axis=1).sum()),
        "differ_by_word": {
            int(word): int(count) for word, count in zip(words, differ.sum(0)) if count
        },
        "box_differ": int(box_differ.any(axis=1).sum()),
    }


def lists_check(session, uploaded, scene, records, middle, sample):
    """The lists SETUP wrote against `lists.build_lists` of the reference's records
    (the record number word aside): every page, or `sample` pages spread over them,
    read in chunks of at most 2 MB."""
    counts, offsets, members = raster.bin_triangles(records)
    packed = lists.pack_records(records)
    packed[:, lists.K["kRecIndex"]] = 0
    want, directory = lists.build_lists(packed, offsets, members, middle["tiles_of"])
    total = len(want)
    if sample and sample < total:
        first_pages = np.linspace(0, total - 1, sample).round().astype(int)
        chunks = [(int(start), 1) for start in sorted(set(first_pages))]
    else:
        pages_per_chunk = (2 << 20) // lists.K["kInPageBytes"]
        chunks = [
            (start, min(pages_per_chunk, total - start))
            for start in range(0, total, pages_per_chunk)
        ]
    differ = 0
    for first, pages in chunks:
        (got,) = session.read(uploaded["lists"], first=first, pages=pages)
        differ += int((got.view(np.uint32) != want[first : first + pages]).any(1).sum())
        if pages > 1:
            card.log(f"lists pages {first} + {pages} of {total}")
    return {
        "pages": int(total),
        "pages_checked": int(sum(pages for _, pages in chunks)),
        "pages_differ": differ,
        "directory_equal": bool(np.array_equal(directory, middle["directory"])),
    }


def count_only(args):
    """COUNT alone over `args.cores` cores: the counters against `passes.counters`."""
    scene, _ = run.load(args.scene)
    blocks = passes.mesh_blocks(scene)
    records = raster.setup(scene)
    report = {"scene": args.scene, "blocks": len(blocks)}
    with km.Session(log=card.log) as session:
        workers = card.workers(session)
        cores = args.cores or workers
        mesh = card.tensor(session, blocks)
        camera = card.tensor(session, camera_page(scene))
        counters = card.tensor(session, np.zeros((workers, S["kCountBytes"] // 4)))
        running = session.load(program(), deadline=1.0)
        command_words = [mesh.address, counters.address, len(blocks)]
        command_words += [camera.address, 0, cores]
        event = running.wait(
            running.submit(S["kHandlerCount"], group=(0, workers), args=command_words),
            10.0,
        )
        report["event"] = {"kind": event.kind, "microseconds": event.microseconds}
        running.stop()
        report["status"] = card.check_run(running, workers, 1, [event.microseconds])
        (raw,) = card.read(session, counters)
        raw = raw.view(np.uint32).reshape(workers, 2, lists.TILES)[:cores]
        want_entries, want_rows = passes.counters(records, len(scene["indices"]), cores)
        report["entries"] = int(want_entries.sum())
        report["cnt_differ"] = int((raw[:, 0] != want_entries).sum())
        report["rows_differ"] = int((raw[:, 1] != want_rows).sum())
        # HIST over the tiles to cut (every tile with entries when none is to be cut)
        entries, rows = raw[:, 0].astype(np.int64), raw[:, 1].astype(np.int64)
        heavy = passes.heavy_tiles(entries, rows, workers)
        report["heavy"] = len(heavy)
        cut_tiles = heavy or [
            (int(tile), 1) for tile in np.flatnonzero(entries.sum(0))[:256]
        ]
        base_words = [mesh.address, 0, len(blocks), camera.address, 0, cores]
        starts, ends = hist(session, base_words, cut_tiles, cores, {})
        want_starts, want_ends = passes.histograms(
            records, len(scene["indices"]), cores
        )
        histogram_tiles = [tile for tile, _ in cut_tiles]
        report["hist_tiles"] = len(histogram_tiles)
        report["hist_differ"] = int(
            (starts != want_starts[:, histogram_tiles]).sum()
            + (ends != want_ends[:, histogram_tiles]).sum()
        )
    report["ok"] = bool(
        report["status"]["ok"]
        and report["cnt_differ"] == 0
        and report["rows_differ"] == 0
        and report["hist_differ"] == 0
    )
    return report


def frame(args, session=None):
    """COUNT, the host's middle pass, SETUP, FILL: one frame, each part timed (in
    `session`, or a session of its own)."""
    scene, _ = run.load(args.scene)
    blocks = passes.mesh_blocks(scene)
    records = raster.setup(scene)
    report = {"scene": args.scene, "blocks": len(blocks), "times": {}}
    times = report["times"]
    session_context = (
        contextlib.nullcontext(session) if session else km.Session(log=card.log)
    )
    with session_context as session:
        workers = card.workers(session)
        cores = args.cores or workers
        started = time.perf_counter()
        mesh = card.tensor(session, blocks)
        camera = card.tensor(session, camera_page(scene))
        counters = card.tensor(session, np.zeros((workers, S["kCountBytes"] // 4)))
        times["upload_mesh_s"] = time.perf_counter() - started
        group = (0, workers)
        base_words = [mesh.address, 0, len(blocks), camera.address, 0, cores]

        started = time.perf_counter()
        running = session.load(program(), deadline=1.0)
        times["load_passes_s"] = time.perf_counter() - started
        command_words = list(base_words)
        command_words[1] = counters.address
        event = running.wait(
            running.submit(S["kHandlerCount"], group=group, args=command_words),
            timeout=10.0,
        )
        report["count_microseconds"] = event.microseconds
        running.stop()
        report["count_status"] = card.check_run(
            running, workers, 1, [event.microseconds]
        )
        started = time.perf_counter()
        (raw,) = card.read(session, counters)
        times["read_counters_s"] = time.perf_counter() - started
        raw = raw.view(np.uint32).reshape(workers, 2, lists.TILES)[:cores]
        entries, rows = raw[:, 0].astype(np.int64), raw[:, 1].astype(np.int64)
        want_entries, want_rows = passes.counters(records, len(scene["indices"]), cores)
        report["counters_equal"] = bool(
            np.array_equal(entries, want_entries) and np.array_equal(rows, want_rows)
        )

        heavy = passes.heavy_tiles(entries, rows, workers)
        report["heavy"] = len(heavy)
        bands = None
        if heavy:
            starts, ends = hist(session, base_words, heavy, cores, times)
            want_starts, want_ends = passes.histograms(
                records, len(scene["indices"]), cores
            )
            histogram_tiles = [tile for tile, _ in heavy]
            report["hist_equal"] = bool(
                np.array_equal(starts, want_starts[:, histogram_tiles])
                and np.array_equal(ends, want_ends[:, histogram_tiles])
            )
            started = time.perf_counter()
            bands = passes.items_of(entries, rows, heavy, starts, ends)
            times["items_s"] = time.perf_counter() - started
        started = time.perf_counter()
        middle = passes.middle(entries, rows, workers, bands=bands)
        times["middle_s"] = time.perf_counter() - started
        report["items"] = len(middle["items"])
        started = time.perf_counter()
        uploaded = card.upload(
            session, middle["pages"], middle["directory"], lists.TILES * 5
        )
        next_slots = np.zeros((workers, S["kSlotsBytes"] // 4), dtype=np.uint32)
        next_slots[:cores, : len(middle["items"])] = middle["next"]
        slots_tensor = card.tensor(session, next_slots)
        items_tensor = card.tensor(session, passes.item_table(middle["items"])[None, :])
        times["upload_lists_s"] = time.perf_counter() - started
        report["list_pages"] = len(middle["pages"])

        started = time.perf_counter()
        running = session.load(program(), deadline=1.0)
        times["load_passes2_s"] = time.perf_counter() - started
        command_words = list(base_words)
        command_words[1] = uploaded["lists"].address
        command_words[4] = slots_tensor.address
        command_words += [middle["slots"], items_tensor.address]
        event = running.wait(
            running.submit(S["kHandlerSetup"], group=group, args=command_words),
            timeout=10.0,
        )
        report["setup_microseconds"] = event.microseconds
        running.stop()
        report["setup_status"] = card.check_run(
            running, workers, 1, [event.microseconds]
        )
        if args.lists >= 0:
            report["lists"] = lists_check(
                session, uploaded, scene, records, middle, args.lists
            )

        got = None if args.setup else fill_frame(session, uploaded, report)
    succeeded = all(report[part]["ok"] for part in ("count_status", "setup_status"))
    succeeded = (
        succeeded and report["counters_equal"] and report.get("hist_equal", True)
    )
    if "lists" in report:
        lists_report = report["lists"]
        succeeded = (
            succeeded
            and lists_report["pages_differ"] == 0
            and lists_report["directory_equal"]
        )
    if got is not None:
        counts, offsets, members = raster.bin_triangles(records)
        tiles = list(range(lists.TILES))
        report["check"] = run.check(
            args.scene, None, records, offsets, members, tiles, counts, got
        )
        succeeded = (
            succeeded and report["fill_status"]["ok"] and run.check_ok(report["check"])
        )
    report["ok"] = bool(succeeded)
    return report


def hist(session, base, heavy, cores, times):
    """HIST: the heavy tiles' start and end counts COUNT kept on every core, as
    (cores, heavy, GROUPS) twice."""
    workers = card.workers(session)
    page = np.zeros((1, S["kHeavyBytes"] // 4), dtype=np.uint32)
    page[0, 0] = len(heavy)
    page[0, 1 : 1 + len(heavy)] = [tile for tile, _ in heavy]
    started = time.perf_counter()
    heavy_tensor = card.tensor(session, page)
    histograms_tensor = card.tensor(session, np.zeros((workers, 16 * len(heavy))))
    times["upload_heavy_s"] = time.perf_counter() - started
    started = time.perf_counter()
    running = session.load(program(), deadline=1.0)
    times["load_hist_s"] = time.perf_counter() - started
    command_words = [
        heavy_tensor.address,
        histograms_tensor.address,
        0,
        0,
        64 * len(heavy),
        cores,
    ]
    event = running.wait(
        running.submit(S["kHandlerHist"], group=(0, workers), args=command_words),
        timeout=10.0,
    )
    running.stop()
    status = card.check_run(running, workers, 1, [event.microseconds])
    if not status["ok"]:
        raise RuntimeError(f"HIST: {status}")
    started = time.perf_counter()
    (raw,) = card.read(session, histograms_tensor)
    times["read_hist_s"] = time.perf_counter() - started
    raw = raw.view(np.uint32).reshape(workers, len(heavy), 16)[:cores]
    counts = raw.view(np.uint16).reshape(cores, len(heavy), 2, passes.GROUPS)
    return counts[:, :, 0].astype(np.int64), counts[:, :, 1].astype(np.int64)


def fill_frame(session, uploaded, report):
    """The fill of `card.py` over the lists in place: the frame's pages, timed."""
    times = report["times"]
    workers = card.workers(session)
    started = time.perf_counter()
    fill = session.load(card.program(uploaded), deadline=1.0)
    times["load_fill_s"] = time.perf_counter() - started
    microseconds = card.loop(fill, lists.K["kHandlerFill"], card.words(uploaded, 5), 1)
    report["fill_microseconds"] = float(microseconds[0]) if len(microseconds) else None
    fill.stop()
    report["fill_status"] = card.check_run(fill, workers, 1, microseconds)
    started = time.perf_counter()
    got = card.read_frame(session, uploaded)
    times["read_frame_s"] = time.perf_counter() - started
    return got


WINDOW = 510  # blocks per read: 2 MB of records


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="tiny")
    parser.add_argument("--cores", type=int, default=0, help="0: every core")
    parser.add_argument(
        "--windows", type=int, default=0, help="check this many windows (0: all)"
    )
    parser.add_argument("--json", default=None)
    parser.add_argument("--dump", default=None, help="save the raw records (.npy)")
    parser.add_argument("--frame", action="store_true", help="a frame, not LANES")
    parser.add_argument("--count", action="store_true", help="COUNT alone, not LANES")
    parser.add_argument("--setup", action="store_true", help="frame without the fill")
    parser.add_argument(
        "--lists",
        type=int,
        default=0,
        help="frame: list pages checked (0: all, -1: none)",
    )
    args = parser.parse_args()
    if args.frame or args.count or args.setup:
        report = count_only(args) if args.count else frame(args)
        print(json.dumps(power.clocked(report)), flush=True)
        if args.json:
            with open(args.json, "w") as file:
                json.dump(report, file, indent=1)
        sys.exit(0 if report["ok"] else 1)
    scene, _ = run.load(args.scene)
    blocks = passes.mesh_blocks(scene)
    starts = list(range(0, len(blocks), WINDOW))
    if args.windows and args.windows < len(starts):
        picked = np.linspace(0, len(starts) - 1, args.windows).round().astype(int)
        starts = [starts[i] for i in sorted(set(picked))]
    report = {"scene": args.scene, "blocks": len(blocks), "windows": len(starts)}
    got = np.zeros((len(blocks) * 32, 32), dtype=np.uint32)
    seen = np.zeros(len(blocks) * 32, dtype=bool)
    with km.Session(log=card.log) as session:
        workers = card.workers(session)
        cores = args.cores or workers
        mesh = card.tensor(session, blocks)
        camera = card.tensor(session, camera_page(scene))
        output_tensor = card.tensor(session, np.zeros((WINDOW, S["kRecBytes"] // 4)))
        running = session.load(program(), deadline=1.0)
        microseconds, succeeded = [], True
        for first_block in starts:
            block_count = min(WINDOW, len(blocks) - first_block)
            command_words = [mesh.address, output_tensor.address, len(blocks)]
            command_words += [camera.address, first_block, cores, block_count]
            handle = running.submit(
                S["kHandlerLanes"], group=(0, workers), args=command_words
            )
            event = running.wait(handle, timeout=10.0)
            microseconds.append(event.microseconds)
            if event.kind != "done":
                succeeded = False
                break
            (page,) = session.read(output_tensor, first=0, pages=block_count)
            got[32 * first_block : 32 * (first_block + block_count)] = page.view(
                np.uint32
            ).reshape(-1, 32)
            seen[32 * first_block : 32 * (first_block + block_count)] = True
            card.log(f"window {first_block} of {len(blocks)}: {block_count} blocks")
        running.stop()
        status = card.check_run(running, workers, len(microseconds), microseconds)
    report["event_microseconds"] = microseconds
    report["status"] = status
    if args.dump:
        np.save(args.dump, got)
    report["check"] = check_lanes(scene, got, seen)
    report["ok"] = bool(
        succeeded
        and status["ok"]
        and report["check"]["words_differ"] == 0
        and report["check"]["kept_disagree"] == 0
        and report["check"]["box_differ"] == 0
    )
    print(json.dumps(power.clocked(report)), flush=True)
    if args.json:
        with open(args.json, "w") as file:
            json.dump(report, file, indent=1)
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
