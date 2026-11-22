"""A frame driven by the work cluster's thread (`thread/frame.c`), each step it
reaches checked against the host's (`passes`) and the image against the reference;
the mesh uploaded once, the camera at the load. `python -m projects.raster.card_thread
--scene tiny [--steps N] [--no-image]` prints one JSON line; exits 1 unless exact."""

import argparse
import contextlib
import json
import re
import sys
import time

import numpy as np

import komorebi as km
from komorebi.bench import power
from komorebi.program.thread import BANK_OF_CLUSTER, TIMER_HZ
from projects.raster import card, card_passes, lists, passes, run
from reference import raster

THREAD = lists.HERE / "thread"
KERNELS = lists.HERE / "kernels"
FRAME_H = THREAD / "frame.h"
F = {
    name: int(value.rstrip("u"), 0)
    for name, value in re.findall(
        r"^#define (RF_\w+)\s+(\d+u?)\b", FRAME_H.read_text(), re.M
    )
}
# expressions in frame.h
F["RF_W_PROF"] = F["RF_W_FRAME"] + F["RF_FRAME_MAX"]
F["RF_W_REPEAT"] = F["RF_W_PROF"] + 1
F["RF_W_SPROF"] = F["RF_W_REPEAT"] + 1
for i, name in enumerate(["COUNTERS", "HIST", "SLOTS", "HPREFIX"]):
    F[f"RF_W_{name}"] = F["RF_W_SPROF"] + 1 + i
F["RF_W_BALANCE"] = F["RF_W_HPREFIX"] + 1
F["RF_W_DISPLAY"] = F["RF_W_BALANCE"] + 1
F["RF_W_SHOWPROF"] = F["RF_W_DISPLAY"] + 1
for i, name in enumerate(["SHADE", "TEXTURES", "SHADE_TABLE", "SHADEPROF"]):
    F[f"RF_W_{name}"] = F["RF_W_SHOWPROF"] + 1 + i
for i, name in enumerate(["MESH", "BLOCKS", "LISTS", "LIST_SLOTS", "DIR", "SHADE"]):
    F[f"RF_W_M_{name}"] = F["RF_W_SHADEPROF"] + 1 + i
F["RF_W_M_FRAMES"] = F["RF_W_M_SHADE"] + 1
F["RF_W_M_FRAME"] = F["RF_W_M_FRAMES"] + 1
F["RF_W_SURFACE"] = F["RF_W_M_FRAME"] + F["RF_FRAME_MAX"]
F["RF_W_SUN_PAGES"] = F["RF_W_SURFACE"] + 1
F["RF_WORDS"] = F["RF_W_SUN_PAGES"] + 1
F["RF_HIST_BYTES"] = 64 * F["RF_HIST_MAX"]
F["RF_CAMERA_BYTES"] = F["RF_AREAS"] * F["RF_AREA_BYTES"]
S, K = card_passes.S, lists.K
SHOW_H = KERNELS / "show.h"
SHOW = {
    name: int(value, 0) for name, value in lists.CONSTEXPR.findall(SHOW_H.read_text())
}
SHADE_H = KERNELS / "shade.h"
SH = {
    name: int(value, 0) for name, value in lists.CONSTEXPR.findall(SHADE_H.read_text())
}
CLUSTER = 1
PAGES_PER_ITEM = 5  # the frame's pages per tile (card.words)


def same_numbers():
    """The numbers frame.h takes from the kernels' headers, as pairs that differ."""
    pairs = {
        "RF_HANDLER_COUNT": S["kHandlerCount"],
        "RF_HANDLER_SETUP": S["kHandlerSetup"],
        "RF_HANDLER_HIST": S["kHandlerHist"],
        "RF_HANDLER_PLACE": S["kHandlerPlace"],
        "RF_HANDLER_SUM": S["kHandlerSum"],
        "RF_HANDLER_HSUM": S["kHandlerHsum"],
        "RF_BASE_AT": S["kBaseAt"],
        "RF_HPREFIX_ADDR_AT": S["kHprefixAddrAt"],
        "RF_HPREFIX_BYTES": S["kHprefixPageBytes"],
        "RF_SUMS_HEAVY_AT": S["kSumsHeavyAt"],
        "RF_HANDLER_FILL": K["kHandlerFill"],
        "RF_PAGES_PER_ITEM": PAGES_PER_ITEM,
        "RF_FRAME_MAX": K["kDirFrameMax"],
        "RF_COUNTER_BYTES": S["kCountBytes"],
        "RF_HEAVY_BYTES": S["kHeavyBytes"],
        "RF_SLOTS_BYTES": S["kSlotsBytes"],
        "RF_ITEMS_BYTES": S["kItemsBytes"],
        "RF_FRAME_WORDS_AT": S["kFrameWordsAt"],
        "RF_ITEM_WORDS_AT": S["kItemWordsAt"],
        "RF_CORE_WORDS_AT": S["kCoreWordsAt"],
        "RF_ITEMS_REGION_BYTES": S["kCoreWordsAt"] + 16 * S["kMaxFillCores"],
        "RF_RANGES_AT": S["kRangesAt"],
        "RF_COSTS_AT": S["kCostsAt"],
        "RF_COST_KNOTS": S["kCostKnots"],
        "RF_MAX_CORES": S["kMaxFillCores"],
        "RF_COSTS_AT + costs <= RF_PROFILES_AT": (
            min(F["RF_COSTS_AT"] + 128 * S["kMaxFillCores"], F["RF_PROFILES_AT"]),
            F["RF_COSTS_AT"] + 128 * S["kMaxFillCores"],
        ),
        "RF_PROFILES_AT + 64 <= RF_AREA_BYTES": (
            min(F["RF_PROFILES_AT"] + 64, F["RF_AREA_BYTES"]),
            F["RF_PROFILES_AT"] + 64,
        ),
        "RF_HANDLER_SHOW": SHOW["kHandlerShow"],
        "RF_SHOW_TABLE_BYTES": 4 * SHOW["kShowTableWords"],
        "RF_DISPLAY_PAGES": lists.TILES,
        "RF_HANDLER_SHADE": SH["kHandlerShade"],
        "RF_HANDLER_LIGHT": SH["kHandlerLight"],
        "RF_SHOW_RAW": SHOW["kShowRaw"],
        "RF_SUN_TABLE_AT + table": (
            F["RF_SUN_TABLE_AT"] + SH["kLightTableBytes"],
            F["RF_SUN_BYTES"],
        ),
        "RF_SHADE_ADDR_AT": S["kShadeAddrAt"],
        "kRecShade": (SH["kRecShade"], S["kRecShadeWord"]),
        # PLACE's list and directory words against the fill's
        "kLayDirIds": (S["kLayDirIds"], K["kDirIds"]),
        "kLayDirFrame": (S["kLayDirFrame"], K["kDirFrame"]),
        "kLayItemEmpty": (S["kLayItemEmpty"], K["kItemEmpty"]),
        "kLayRow0Shift": (S["kLayRow0Shift"], K["kItemRow0Shift"]),
        "kLayRow1Shift": (S["kLayRow1Shift"], K["kItemRow1Shift"]),
        "kLayMarkTile": (S["kLayMarkTile"], K["kMarkTile"]),
        "kLayMarkEnd": (S["kLayMarkEnd"], K["kMarkEnd"]),
        "kLayHdrTile": (S["kLayHdrTile"], K["kHdrTile"]),
        "kLayHdrCount": (S["kLayHdrCount"], K["kHdrCount"]),
        "kLayHdrRow0": (S["kLayHdrRow0"], K["kHdrRow0"]),
        "kLayHdrRow1": (S["kLayHdrRow1"], K["kHdrRow1"]),
        "kDirPageBytes": (S["kDirPageBytes"], K["kDirBytes"]),
    }
    pairs = {
        name: value if isinstance(value, tuple) else (F[name], value)
        for name, value in pairs.items()
    }
    return {name: pair for name, pair in pairs.items() if pair[0] != pair[1]}


def camera_region(scene, profiles=None):
    """The camera's numbers, then room for COUNT's and SETUP's ranges and costs (an area
    per layer and camera, frame.h RF_AREAS); at RF_PROFILES_AT `profiles` (live.py: the
    masked layer's profile tensors), else 0."""
    region = np.zeros((1, F["RF_CAMERA_BYTES"] // 4), dtype=np.uint32)
    region[0, : S["kCamWords"]] = card_passes.camera_page(scene)[0]
    if profiles is not None:
        at = F["RF_PROFILES_AT"] // 4
        region[0, at : at + len(profiles)] = profiles
    return region


def regions(scene, workers, show_table=None, sun=None, profiles=None):
    """The thread's regions in frame.h's order: the camera's page, then sizes; SHOW's
    block holds `show_table` (live.py), else nothing; the sun region `sun` (live.py,
    lit: the sun's camera and LIGHT's table, light.sun_region), else nothing;
    `profiles`: camera_region's."""
    work = 256 << 10  # frame.c work_t: middle.h scratch and items (about 130 KB)
    return {
        "camera": camera_region(scene, profiles),
        "heavy": F["RF_HEAVY_BYTES"],
        "items": F["RF_ITEMS_REGION_BYTES"],
        "work": work,
        "sums": F["RF_SUMS_BYTES"],
        "times": F["RF_TIMES_BYTES"],
        "show": (
            F["RF_SHOW_TABLE_BYTES"]
            if show_table is None
            else np.asarray(show_table, dtype=np.uint32)
        ),
        "push": F["RF_PUSH_BYTES"],
        "sun": F["RF_SUN_BYTES"] if sun is None else np.asarray(sun, dtype=np.uint32),
        "sun_times": F["RF_TIMES_BYTES"],
    }


def shade_buffers():
    """SHADE's buffers (kernels/shade.h): planes, table, asks, answers, the page slots
    and the white texel's page, the pack thread's space and its lit space."""
    return [
        km.Buffer(
            SH["kCbShadePlanes"], "uint32", K["kOutPageBytes"], SH["kShadePlanePages"]
        ),
        km.Buffer(SH["kCbShadeTable"], "uint32", SH["kShadeTableBytes"], 1),
        km.Buffer(
            SH["kCbShadeAsk"], "uint32", SH["kShadeAskBytes"], SH["kShadeAskPages"]
        ),
        km.Buffer(
            SH["kCbShadeDone"], "uint32", SH["kShadeDoneBytes"], SH["kShadeDonePages"]
        ),
        km.Buffer(
            SH["kCbShadeSlots"], "uint32", SH["kTexPageBytes"], SH["kShadeSlots"] + 1
        ),
        km.Buffer(SH["kCbShadeWork"], "uint32", SH["kShadeWorkBytes"], 1),
        km.Buffer(SH["kCbLightWork"], "uint32", SH["kLightWorkBytes"], 1),
    ]


def program(scene, words, workers, show_table=None, sun=None, profiles=None):
    """The frame program: the passes' and the fill's kernels behind one entry each
    (kernels/frame_*.cpp), all their buffers, the frame thread on work cluster 1; with
    `show_table` (live.py) it serves FRAME and SUN commands; `sun`, `profiles`: the
    regions' (regions)."""
    fill_program, passes_program = card.program(), card_passes.program()
    frame_program = km.Program(
        "raster_frame",
        kernels={
            role: KERNELS / f"frame_{role}.cpp"
            for role in ("reader", "compute", "writer")
        },
        buffers=[
            *fill_program.buffers,
            *passes_program.buffers,
            km.Buffer(
                SHOW["kCbShow"], "uint32", SHOW["kShowPageBytes"], SHOW["kShowPages"]
            ),
            *shade_buffers(),
        ],
        compile_args=card.compile_args(),  # the fill's tensors: DRAM tensors' args
        compute=fill_program.compute,
        defines=card.spread_parts()[2],
        depth=1,
        handlers={
            **fill_program.handlers,
            **passes_program.handlers,
            "show": SHOW["kHandlerShow"],
            "shade": SH["kHandlerShade"],
            "light": SH["kHandlerLight"],
        },
    )
    frame_program.thread(
        cluster=CLUSTER,
        sources=[THREAD / "frame.c", THREAD / "middle.c"],
        includes=[THREAD],
        regions=regions(scene, workers, show_table, sun, profiles),
        words=words,
        deadline=10.0,
        **(
            {"handlers": {"frame": 1, "sun": 2}, "defines": {"RF_SERVE": "1"}}
            if show_table is not None
            else {}
        ),
    )
    return frame_program


def layer_tensors(session, blocks, workers, list_slots):
    """A mesh's own DRAM tensors: the mesh (once per scene), lists, directory, frame."""
    pages = -(-list_slots // lists.PAGE_SLOTS)
    output_pages = lists.TILES * 5
    chunks = -(-output_pages // card.CHUNK_PAGES)
    frame = [
        card.tensor(
            session,
            np.zeros(
                (min(card.CHUNK_PAGES, output_pages - chunk * card.CHUNK_PAGES), 1024)
            ),
        )
        for chunk in range(chunks)
    ]
    return {
        "mesh": card.tensor(session, blocks),
        "lists": card.tensor(session, np.zeros((pages, lists.PAGE_WORDS))),
        "dir": card.tensor(session, np.zeros((workers, K["kDirBytes"] // 4))),
        "frame": frame,
        "list_slots": pages * lists.PAGE_SLOTS,
    }


def tensors(session, blocks, workers, list_slots):
    """The DRAM tensors: mesh (once per scene), lists, directory, frame, profiles, and
    the pages only the cores read and write."""
    uploaded = {
        **layer_tensors(session, blocks, workers, list_slots),
        "prof": card.tensor(session, np.zeros((workers, 1024))),
        "sprof": card.tensor(session, np.zeros((workers, S["kSProfPageBytes"] // 4))),
        "counters": card.tensor(
            session, np.zeros((workers, F["RF_COUNTER_BYTES"] // 4))
        ),
        "hist": card.tensor(session, np.zeros((workers, F["RF_HIST_BYTES"] // 4))),
        "slots": card.tensor(session, np.zeros((workers, F["RF_SLOTS_BYTES"] // 4))),
        "hprefix": card.tensor(
            session, np.zeros((workers, F["RF_HPREFIX_BYTES"] // 4))
        ),
    }
    session.device.finish()  # the uploads land before any frame runs
    return uploaded


def words(uploaded, blocks, cores, workers, steps, balance=True):
    thread_words = [0] * F["RF_WORDS"]
    thread_words[F["RF_W_BALANCE"]] = int(balance)
    thread_words[F["RF_W_CORES"]] = cores
    thread_words[F["RF_W_WORKERS"]] = workers
    thread_words[F["RF_W_BANK"]] = BANK_OF_CLUSTER[CLUSTER]
    thread_words[F["RF_W_MESH"]] = uploaded["mesh"].address
    thread_words[F["RF_W_BLOCKS"]] = blocks
    thread_words[F["RF_W_LISTS"]] = uploaded["lists"].address
    thread_words[F["RF_W_LIST_SLOTS"]] = uploaded["list_slots"]
    thread_words[F["RF_W_DIR"]] = uploaded["dir"].address
    thread_words[F["RF_W_STEPS"]] = steps
    thread_words[F["RF_W_FRAME_PAGES"]] = card.CHUNK_PAGES
    thread_words[F["RF_W_FRAMES"]] = len(uploaded["frame"])
    for index, frame_tensor in enumerate(uploaded["frame"]):
        thread_words[F["RF_W_FRAME"] + index] = frame_tensor.address
    thread_words[F["RF_W_PROF"]] = uploaded["prof"].address
    thread_words[F["RF_W_SPROF"]] = uploaded["sprof"].address
    for name in ("counters", "hist", "slots", "hprefix"):
        thread_words[F[f"RF_W_{name.upper()}"]] = uploaded[name].address
    return thread_words


def masked_words(thread_words, layer, blocks, shade_address):
    """The masked layer's words (frame.h RF_W_M_*) into `thread_words`: its tensors
    (layer_tensors), blocks and shade tensor."""
    thread_words[F["RF_W_M_MESH"]] = layer["mesh"].address
    thread_words[F["RF_W_M_BLOCKS"]] = blocks
    thread_words[F["RF_W_M_LISTS"]] = layer["lists"].address
    thread_words[F["RF_W_M_LIST_SLOTS"]] = layer["list_slots"]
    thread_words[F["RF_W_M_DIR"]] = layer["dir"].address
    thread_words[F["RF_W_M_SHADE"]] = shade_address
    thread_words[F["RF_W_M_FRAMES"]] = len(layer["frame"])
    for index, frame_tensor in enumerate(layer["frame"]):
        thread_words[F["RF_W_M_FRAME"] + index] = frame_tensor.address
    return thread_words


def result_dict(result):
    """The thread's result: code, what the middle pass found, each part in
    microseconds."""
    out_words = list(result.words) + [0] * (F["RF_OUT_WORDS"] - len(result.words))
    microseconds_per_tick = 1e6 / TIMER_HZ
    names = ["count", "sum", "hist", "hsum", "middle", "place", "setup", "fill"]
    return {
        "ran": result.ran,
        "state": result.state,
        "error": result.error,
        "code": result.code,
        "where": result.where,
        "thread_microseconds": result.microseconds,
        "slots": out_words[F["RF_O_SLOTS"]],
        "items": out_words[F["RF_O_ITEMS"]],
        "marks": out_words[F["RF_O_MARKS"]],
        "heavy": out_words[F["RF_O_HEAVY"]],
        "microseconds": {
            part: out_words[F[f"RF_O_{part.upper()}"]] * microseconds_per_tick
            for part in names
        },
        "middle_microseconds": {
            part: out_words[F[f"RF_O_{part}"]] * microseconds_per_tick
            for part in (
                "TAKE",
                "HEAVY_MICROSECONDS",
                "CUT",
                "SORT",
                "ASSIGN",
                "COMPACT",
                "PAGES",
                "GIVE",
            )
        },
    }


def expected(scene, records, cores, workers):
    """The host's middle pass on the reference's counts: what the thread must find."""
    triangle_count = len(scene["indices"])
    entries, rows = passes.counters(records, triangle_count, cores)
    heavy = passes.heavy_tiles(entries, rows, workers)
    bands = None
    if heavy:
        histogram_tiles = [tile for tile, _ in heavy]
        starts, ends = passes.histograms(records, triangle_count, cores)
        bands = passes.items_of(
            entries, rows, heavy, starts[:, histogram_tiles], ends[:, histogram_tiles]
        )
    middle = passes.middle(entries, rows, workers, bands=bands)
    pages = middle["pages"].reshape(-1, lists.SLOT)
    middle["found"] = {
        "slots": int(middle["slots"]),
        "items": len(middle["items"]),
        "marks": int((pages[:, 0] >= K["kMarkTile"]).sum()),
        "heavy": len(heavy),
    }
    return middle


def slots_differ(session, uploaded, records, middle):
    """Differing list slots by kind (header, end, record), for a small frame."""
    counts, offsets, members = raster.bin_triangles(records)
    packed = lists.pack_records(records)
    packed[:, K["kRecIndex"]] = 0
    want, _ = lists.build_lists(packed, offsets, members, middle["tiles_of"])
    (got,) = session.read(uploaded["lists"], first=0, pages=len(want))
    want = want.reshape(-1, lists.SLOT)
    got = got.view(np.uint32).reshape(-1, lists.SLOT)
    differs = (got != want).any(axis=1)
    kind = np.where(
        want[:, 0] == K["kMarkTile"],
        "header",
        np.where(want[:, 0] == K["kMarkEnd"], "end", "record"),
    )
    report = {
        name: int((differs & (kind == name)).sum())
        for name in ("header", "end", "record")
    }
    first_differing = np.flatnonzero(differs)[:3]
    report["first"] = [
        {
            "slot": int(slot),
            "want": want[slot, :6].tolist(),
            "got": got[slot, :6].tolist(),
        }
        for slot in first_differing
    ]
    return report


def sprof_dict(profile_pages, cores):
    """Per pass, each core's writer total and pack counters from the profile pages of
    the last frame (wall ticks), and the slowest core against the mean."""
    report = {}
    for name, handler in (("count", S["kHandlerCount"]), ("setup", S["kHandlerSetup"])):
        page = profile_pages[:cores, 32 * (handler - S["kHandlerCount"]) :][
            :, :32
        ].astype(np.int64)
        total = page[:, S["kSProfWriter"] + 2]
        pack = page[:, S["kSProfPack"] : S["kSProfPack"] + 8]
        report[name] = {
            "total": total.tolist(),
            "kept": pack[:, 4].tolist(),
            "entries": pack[:, 6].tolist(),
            "max_over_mean": float(total.max() / max(total.mean(), 1)),
        }
    return report


def frame(args, session=None):
    scene, _ = run.load(args.scene)
    blocks = passes.mesh_blocks(scene)
    records = raster.setup(scene)
    report = {"scene": args.scene, "blocks": len(blocks), "times": {}}
    times = report["times"]
    mismatches = same_numbers()
    if mismatches:
        raise SystemExit(f"frame.h differs from the kernels' headers: {mismatches}")
    session_context = (
        contextlib.nullcontext(session) if session else km.Session(log=card.log)
    )
    with session_context as session:
        workers = card.workers(session)
        cores = args.cores or workers
        started = time.perf_counter()
        slots = args.list_slots or 2 * len(scene["indices"]) + 65536
        uploaded = tensors(session, blocks, workers, slots)
        times["upload_s"] = time.perf_counter() - started
        thread_words = words(
            uploaded, len(blocks), cores, workers, args.steps, args.balance
        )
        thread_words[F["RF_W_REPEAT"]] = args.repeat
        started = time.perf_counter()
        running = session.load(
            program(scene, thread_words, workers), deadline=1.0, trace=16
        )
        times["load_s"] = time.perf_counter() - started
        started = time.perf_counter()
        result = running.thread_result(timeout=15.0)
        times["thread_wait_s"] = time.perf_counter() - started
        try:
            running.stop()
        except km.errors.KernelFailed as error:
            report["stop_error"] = str(error)
            trace = running.trace()
            headers = km.abi.headers()
            rows = trace.rows[:workers]
            seen, done = (
                rows[:, :, headers["kTraceSeen"]],
                rows[:, :, headers["kTraceDone"]],
            )
            report["trace"] = {
                "seen_cores": (seen != 0).sum(0).tolist(),
                "done_cores": (done != 0).sum(0).tolist(),
                "not_done": [
                    [
                        int(core)
                        for core in np.flatnonzero(
                            (seen[:, point] != 0) & (done[:, point] == 0)
                        )
                    ]
                    for point in range(rows.shape[1])
                ],
            }
            print(json.dumps(power.clocked(report), default=str), flush=True)
        report["thread"] = result_dict(result)
        report["x280_mhz"] = session.pll_mhz
        report["status"] = card.check_run(running, workers, 0, [])
        (profile_pages,) = card.read(session, uploaded["sprof"])
        report["sprof"] = sprof_dict(
            profile_pages.view(np.uint32).reshape(workers, -1), cores
        )
        # Regions are not read back (a region read reads every bank's pages of the
        # regions' tensor at once): the tensors the thread's commands wrote are.
        middle = expected(scene, records, cores, workers)
        found = {key: report["thread"][key] for key in middle["found"]}
        report["found_equal"] = found == middle["found"]
        succeeded = (
            result.ran
            and result.code == 0
            and report["found_equal"]
            and report["status"]["ok"]
        )
        if args.steps >= 3:
            report["lists"] = card_passes.lists_check(
                session, uploaded, scene, records, middle, args.lists
            )
            (directories,) = card.read(session, uploaded["dir"])
            directories = directories.view(np.uint32).reshape(workers, -1).copy()
            frame_words = directories[:, K["kDirFrame"] :].copy()
            directories[:, K["kDirFrame"] :] = 0
            report["dir_equal"] = bool(np.array_equal(directories, middle["directory"]))
            want = [card.CHUNK_PAGES] + [
                frame_tensor.address for frame_tensor in uploaded["frame"]
            ]
            report["dir_frame_equal"] = bool(
                (frame_words[:, : len(want)] == want).all()
                and not frame_words[:, len(want) :].any()
            )
            if report["lists"]["pages_differ"] and len(middle["pages"]) <= 1024:
                report["slots_differ"] = slots_differ(
                    session, uploaded, records, middle
                )
            succeeded = succeeded and report["lists"]["pages_differ"] == 0
            succeeded = succeeded and report["dir_equal"] and report["dir_frame_equal"]
        if args.steps >= 4 and args.image:
            started = time.perf_counter()
            image = card.read_frame(session, uploaded)
            times["read_frame_s"] = time.perf_counter() - started
            counts, offsets, members = raster.bin_triangles(records)
            report["check"] = run.check(
                args.scene,
                None,
                records,
                offsets,
                members,
                list(range(lists.TILES)),
                counts,
                image,
            )
            succeeded = succeeded and run.check_ok(report["check"])
    report["ok"] = bool(succeeded)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="tiny")
    parser.add_argument("--cores", type=int, default=0, help="0: every core")
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="2 the middle pass, 3 SETUP, 4 FILL and the image",
    )
    parser.add_argument(
        "--list-slots", type=int, default=0, help="0: from the triangles"
    )
    parser.add_argument(
        "--lists", type=int, default=0, help="list pages checked (0: all, -1: none)"
    )
    parser.add_argument(
        "--no-image", dest="image", action="store_false", help="no image read-back"
    )
    parser.add_argument(
        "--no-balance",
        dest="balance",
        action="store_false",
        help="equal block partitions",
    )
    parser.add_argument(
        "--repeat", type=int, default=0, help="frames in a row (0: one)"
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    report = frame(args)
    print(json.dumps(power.clocked(report)), flush=True)
    if args.json:
        with open(args.json, "w") as file:
            json.dump(report, file, indent=1)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
