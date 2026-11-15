"""The thread-driven frame with this folder's kernels and thread: frames in a row,
the per-part profile of COUNT and SETUP on every core, block ranges given by the host
(`--ranges predicted`), the image against the reference. One JSON line.
Run: python experiments/programs/raster_passes/frames.py --scene s1m --frames 10 --image
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi.program import handlers_from_header  # noqa: E402
from projects.raster import (  # noqa: E402
    bench,
    card,
    card_passes,
    card_thread,
    lists,
    passes,
    run,
)
from reference import raster  # noqa: E402

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
THREAD = HERE / "thread"
SETUP_H = KERNELS / "setup.h"
S = {
    name: int(value, 0) for name, value in lists.CONSTEXPR.findall(SETUP_H.read_text())
}
F = card_thread.F
CLUSTER = card_thread.CLUSTER

# The pack thread's parts of a block, in the profile page from kSProfParts.
PARTS = [
    "copy_in",
    "vector_unit",
    "records_out",
    "lane_scan",
    "per_triangle",
    "per_entry",
    "out_copies",
    "clipped",
]


def passes_buffers():
    """The passes' buffers, sized by this folder's setup.h."""
    return [
        km.Buffer(S["kCbMesh"], "uint32", S["kMeshBytes"], S["kMeshPages"]),
        km.Buffer(S["kCbRec"], "uint32", S["kRecBytes"], S["kRecPages"]),
        km.Buffer(S["kCbSlots"], "uint32", S["kSlotsBytes"], 1),
        km.Buffer(S["kCbWork"], "uint32", S["kWorkBytes"], 1),
        km.Buffer(S["kCbCount"], "uint32", S["kCountBytes"], 1),
        km.Buffer(S["kCbOut"], "uint32", S["kOutBytes"], S["kOutPages"]),
        km.Buffer(S["kCbHist"], "uint32", S["kHistBytes"], 1),
        km.Buffer(S["kCbItems"], "uint32", S["kItemsBytes"], 1),
        km.Buffer(S["kCbSProf"], "uint32", S["kSProfBytes"], 1),
    ]


def program(scene, words, workers, defines, camera=None):
    """The frame program of `card_thread` built from this folder's copies; `camera`:
    the camera's region in place of `card_thread.camera_region`."""
    fill_program = card.program()
    regions = card_thread.regions(scene, workers)
    if camera is not None:
        regions["camera"] = camera
    frame_program = km.Program(
        "raster_passes_probe",
        kernels={
            role: KERNELS / f"frame_{role}.cpp"
            for role in ("reader", "compute", "writer")
        },
        buffers=[*fill_program.buffers, *passes_buffers()],
        compute=fill_program.compute,
        defines=defines,
        depth=1,
        handlers={**fill_program.handlers, **handlers_from_header(SETUP_H)},
    )
    frame_program.thread(
        cluster=CLUSTER,
        sources=[THREAD / "frame.c", THREAD / "middle.c"],
        includes=[THREAD],
        regions=regions,
        words=words,
        deadline=10.0,
    )
    return frame_program


def block_work(scene, records, blocks):
    """Per mesh block: kept records, their (record, tile) entries and the triangles
    crossing the near distance, from the reference (what COUNT finds)."""
    block_of = records["tri"] // 32
    box = records["box"].astype(np.int64)
    tiles = (box[:, 1] // 32 - box[:, 0] // 32 + 1) * (
        box[:, 3] // 32 - box[:, 2] // 32 + 1
    )
    kept = np.bincount(block_of, minlength=blocks)
    entries = np.bincount(block_of, weights=tiles, minlength=blocks)
    _, _, camera_z = raster.camera_space(scene)
    beyond = (camera_z > scene["camera"][15])[scene["indices"]].sum(axis=1)
    crossing = (beyond == 1) | (beyond == 2)
    clipped = np.bincount(np.flatnonzero(crossing) // 32, minlength=blocks)
    return {"kept": kept, "entries": entries, "clipped": clipped}


def cut(costs, cores):
    """Contiguous block ranges of about equal summed cost: cores + 1 boundaries."""
    total = np.cumsum(costs)
    wanted = total[-1] * np.arange(1, cores) / cores
    inner = np.searchsorted(total, wanted, side="left") + 1
    return np.concatenate([[0], np.minimum(inner, len(costs)), [len(costs)]])


def camera_with_ranges(scene, bounds):
    """The camera's region with every core's range at kRangesAt + 64 x core."""
    region = card_thread.camera_region(scene)
    for core in range(len(bounds) - 1):
        word = (S["kRangesAt"] + 64 * core) // 4
        region[0, word] = bounds[core]
        region[0, word + 1] = bounds[core + 1]
    return region


def profile_dict(profile_pages, cores):
    """Per pass and core, every counter of the profile page (wall-clock cycles), and
    per counter the slowest core's, the mean and the maximum."""
    report = {}
    for name, handler in (("count", S["kHandlerCount"]), ("setup", S["kHandlerSetup"])):
        offset = 32 * (handler - S["kHandlerCount"])
        page = profile_pages[:cores, offset : offset + 32].astype(np.int64)
        columns = {
            "writer_total": page[:, S["kSProfWriter"] + 2],
            "writer_wait": page[:, S["kSProfWriter"]],
            "writer_write": page[:, S["kSProfWriter"] + 1],
            "reader_total": page[:, S["kSProfReader"] + 2],
            "reader_wait": page[:, S["kSProfReader"]],
            "reader_read": page[:, S["kSProfReader"] + 1],
            "unpack_page_wait": page[:, S["kSProfUnpack"]],
            "unpack_back_wait": page[:, S["kSProfUnpack"] + 1],
            "unpack_total": page[:, S["kSProfUnpack"] + 2],
            "pack_mail": page[:, S["kSProfPack"]],
            "pack_vector": page[:, S["kSProfPack"] + 1],
            "pack_scalar": page[:, S["kSProfPack"] + 2],
            "pack_total": page[:, S["kSProfPack"] + 3],
            "kept": page[:, S["kSProfPack"] + 4],
            "clipped_count": page[:, S["kSProfPack"] + 5],
            "entries": page[:, S["kSProfPack"] + 6],
            "out_wait": page[:, S["kSProfPack"] + 7],
        }
        for index, part in enumerate(PARTS):
            columns[part] = page[:, S["kSProfParts"] + index]
        slowest = int(np.argmax(columns["writer_total"]))
        report[name] = {
            "slowest_core": slowest,
            "slowest": {key: int(value[slowest]) for key, value in columns.items()},
            "mean": {key: float(value.mean()) for key, value in columns.items()},
            "max": {key: int(value.max()) for key, value in columns.items()},
            "per_core": {key: value.tolist() for key, value in columns.items()},
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="tiny")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--cores", type=int, default=0, help="0: every core")
    parser.add_argument(
        "--ranges",
        default="equal",
        choices=["equal", "history", "predicted"],
        help="equal partitions, the last frame's ticks (main's), or the host's cut",
    )
    parser.add_argument(
        "--weights",
        default="0,1,0,0",
        help="predicted: cost of a block, a triangle kept, an entry, a clipped one",
    )
    parser.add_argument("--define", action="append", default=[], help="NAME=VALUE")
    parser.add_argument("--image", action="store_true", help="read and check it")
    parser.add_argument(
        "--no-profile", action="store_true", help="no profile page (frame time only)"
    )
    parser.add_argument(
        "--lists", type=int, default=-1, help="list pages checked (0: all, -1: none)"
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    scene, _ = run.load(args.scene)
    blocks = passes.mesh_blocks(scene)
    records = raster.setup(scene)
    defines = dict(item.split("=", 1) for item in args.define)
    report = {"scene": args.scene, "blocks": len(blocks), "args": vars(args)}
    index = bench.card_index()
    with km.Session(log=card.log) as session:
        workers = card.workers(session)
        slots = 2 * len(scene["indices"]) + 65536
        uploaded = card_thread.tensors(session, blocks, workers, slots)
        card.log("uploaded")
        balance = {"equal": 0, "history": 1, "predicted": 2}[args.ranges]
        cores = args.cores or workers
        words = card_thread.words(uploaded, len(blocks), cores, workers, 4, balance)
        words[F["RF_W_REPEAT"]] = args.frames
        if args.no_profile:
            words[F["RF_W_SPROF"]] = 0
        camera = None
        if args.ranges == "predicted":
            work = block_work(scene, records, len(blocks))
            weights = [float(value) for value in args.weights.split(",")]
            costs = weights[0] + sum(
                weight * work[key]
                for weight, key in zip(weights[1:], ("kept", "entries", "clipped"))
            )
            bounds = cut(costs, cores)
            camera = camera_with_ranges(scene, bounds)
            report["bounds"] = bounds.tolist()
        frame_program = program(scene, words, workers, defines, camera)
        report["load_average"] = Path("/proc/loadavg").read_text().split()[:3]
        with bench.ClockSampler(index) as clock:
            running = session.load(frame_program, deadline=1.0)
            card.log("loaded")
            result = running.thread_result(timeout=15.0)
            running.stop()
        card.log("stopped")
        report["clock"] = clock.figures()
        report["x280_mhz"] = session.pll_mhz
        report["thread"] = card_thread.result_dict(result)
        frames = (list(result.words) + [0] * F["RF_OUT_WORDS"])[F["RF_O_FRAMES"]]
        report["frames_run"] = frames
        report["per_frame_microseconds"] = {
            part: value / max(frames, 1)
            for part, value in report["thread"]["microseconds"].items()
        }
        report["frame_microseconds"] = result.microseconds / max(frames, 1)
        report["status"] = card.check_run(running, workers, 0, [])
        if not args.no_profile:
            (profile_pages,) = card.read(session, uploaded["sprof"])
            report["profile"] = profile_dict(
                profile_pages.view(np.uint32).reshape(workers, -1), cores
            )
        succeeded = result.code == 0 and report["status"]["ok"]
        if args.lists >= 0:
            middle = card_thread.expected(scene, records, cores, workers)
            report["lists"] = card_passes.lists_check(
                session, uploaded, scene, records, middle, args.lists
            )
            succeeded = succeeded and report["lists"]["pages_differ"] == 0
        if args.image:
            image = card.read_frame(session, uploaded)
            counts, offsets, members = raster.bin_triangles(records)
            check = run.check(
                args.scene,
                None,
                records,
                offsets,
                members,
                list(range(lists.TILES)),
                counts,
                image,
            )
            report["image_differs"] = check["any_differs"]
            succeeded = succeeded and run.check_ok(check)
    report["ok"] = bool(succeeded)
    print(json.dumps(report), flush=True)
    if args.json:
        with open(args.json, "w") as file:
            json.dump(report, file, indent=1)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
