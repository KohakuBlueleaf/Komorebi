"""COUNT and SETUP on the card: the program, the mesh and camera upload, the LANES
check of the vector unit's records against the reference, and a frame (COUNT, the
host's step, SETUP, then the fill of `card.py`) checked against the reference's image
and lists. `python -m projects.raster.card_passes --scene tiny [--frame]` prints one
JSON line; exits 1 unless exact."""

import argparse
import contextlib
import json
import sys
import time

import numpy as np

import komorebi as km
from projects.raster import card, lists, passes, run
from ref import raster

KERNELS = lists.HERE / "kernels"
SETUP_H = KERNELS / "setup.h"
S = {
    name: int(value, 0) for name, value in lists.CONSTEXPR.findall(SETUP_H.read_text())
}


def program():
    """The passes' three kernels on every core; one record in flight per core."""
    return km.Program(
        "raster_passes",
        kernels={
            "reader": KERNELS / "setup_reader.cpp",
            "compute": KERNELS / "setup_compute.cpp",
            "writer": KERNELS / "setup_writer.cpp",
        },
        buffers=[
            km.Buffer(S["kCbMesh"], "uint32", S["kMeshBytes"], S["kMeshPages"]),
            km.Buffer(S["kCbRec"], "uint32", S["kRecBytes"], S["kRecPages"]),
            km.Buffer(S["kCbSlots"], "uint32", S["kSlotsBytes"], 1),
            km.Buffer(S["kCbWork"], "uint32", S["kWorkBytes"], 1),
            km.Buffer(S["kCbCount"], "uint32", S["kCountBytes"], 1),
            km.Buffer(S["kCbOut"], "uint32", S["kOutBytes"], S["kOutPages"]),
        ],
        compute={"fp32_dest_acc_en": True, "enable_trisc2_rvv": True},
        depth=1,
        segments=SETUP_H,
    )


def camera_page(scene):
    """The camera's 16 float32 numbers as one page."""
    cam = np.asarray(scene["camera"], dtype=np.float32)
    return cam.view(np.uint32).reshape(1, S["kCamWords"])


def expected_lanes(scene):
    """Per triangle (padded to blocks): the reference's record words where it keeps the
    whole triangle, its box, and the triangles kept whole (dict of arrays)."""
    n = len(scene["indices"])
    recs = raster.setup(scene)
    cards = lists.pack_records(recs)
    xc, yc, zc = raster.camera_space(scene)
    beyond = (zc > scene["camera"][15])[scene["indices"]].sum(axis=1)
    whole = beyond == 3
    first = np.searchsorted(recs["tri"], np.arange(n))
    kept = np.zeros(n, dtype=bool)
    hit = first < len(recs["tri"])
    kept[hit] = recs["tri"][first[hit]] == np.arange(n)[hit]
    kept &= whole
    return {
        "cards": cards,
        "first": first,
        "kept": kept,
        "crossing": (beyond == 1) | (beyond == 2),
        "recs": recs,
    }


def check_lanes(scene, got, seen):
    """Differences between the card's records (blocks x 32 lanes, 32 words) and the
    reference over the triangles `seen`, as counts."""
    exp = expected_lanes(scene)
    n = len(scene["indices"])
    got = got.reshape(-1, 32)[:n]
    seen = seen[:n]
    near = np.float32(scene["camera"][15]).view(np.int32)
    z = got[:, S["kRawZ"] : S["kRawZ"] + 3].view(np.int32)
    area = got[:, S["kRawArea"]] & 0x7FFFFFFF
    box = got[:, S["kRawX0"] : S["kRawY1"] + 1].astype(np.int64) - S["kMagic"]
    # the pack thread's rule: every corner beyond the near distance, a non-zero area,
    # a box that is not empty
    kept = (
        (z > near).all(axis=1)
        & (area != 0)
        & (box[:, 0] <= box[:, 1])
        & (box[:, 2] <= box[:, 3])
    )
    both = kept & exp["kept"] & seen
    rec = exp["cards"][exp["first"][both]]
    words = [*range(0, 9), *range(12, 24)]
    differ = got[both][:, words] != rec[:, words]
    box_differ = box[both] != exp["recs"]["box"][exp["first"][both]]
    return {
        "triangles": int(n),
        "kept_card": int((kept & seen).sum()),
        "kept_ref_whole": int((exp["kept"] & seen).sum()),
        "checked": int(seen.sum()),
        "kept_disagree": int(((kept != exp["kept"]) & seen).sum()),
        "crossing": int((exp["crossing"] & seen).sum()),
        "words_differ": int(differ.sum()),
        "records_differ": int(differ.any(axis=1).sum()),
        "differ_by_word": {int(w): int(c) for w, c in zip(words, differ.sum(0)) if c},
        "box_differ": int(box_differ.any(axis=1).sum()),
    }


def lists_check(session, up, scene, recs, middle, sample):
    """The lists SETUP wrote against `lists.build_lists` of the reference's records
    (the record number word aside): every page, or `sample` pages spread over them,
    read in chunks of at most 2 MB."""
    counts, offsets, members = raster.bin_triangles(recs)
    cards = lists.pack_records(recs)
    cards[:, lists.K["kRecIndex"]] = 0
    want, directory = lists.build_lists(cards, offsets, members, middle["tiles_of"])
    total = len(want)
    if sample and sample < total:
        firsts = np.linspace(0, total - 1, sample).round().astype(int)
        chunks = [(int(f), 1) for f in sorted(set(firsts))]
    else:
        per = (2 << 20) // lists.K["kInPageBytes"]
        chunks = [(f, min(per, total - f)) for f in range(0, total, per)]
    differ = 0
    for first, n in chunks:
        (got,) = session.read(up["lists"], first=first, pages=n)
        differ += int((got.view(np.uint32) != want[first : first + n]).any(1).sum())
        if n > 1:
            card.log(f"lists pages {first} + {n} of {total}")
    return {
        "pages": int(total),
        "pages_checked": int(sum(n for _, n in chunks)),
        "pages_differ": differ,
        "directory_equal": bool(np.array_equal(directory, middle["directory"])),
    }


def count_only(args):
    """COUNT alone over `args.cores` cores: the counters against `passes.counters`."""
    scene, _ = run.load(args.scene)
    blocks = passes.mesh_blocks(scene)
    recs = raster.setup(scene)
    out = {"scene": args.scene, "blocks": len(blocks)}
    with km.Session(log=card.log) as session:
        workers = card.workers(session)
        cores = args.cores or workers
        dev = session.device
        mesh = card.tensor(dev, blocks)
        cam = card.tensor(dev, camera_page(scene))
        counters = card.tensor(dev, np.zeros((workers, S["kCountBytes"] // 4)))
        prog = session.load(program(), deadline=1.0)
        a = [mesh.buffer_address(), counters.buffer_address(), len(blocks)]
        a += [cam.buffer_address(), 0, cores]
        ev = prog.wait(prog.issue(S["kSegCount"], team=(0, workers), args=a), 10.0)
        out["event"] = {"kind": ev.kind, "us": ev.us}
        prog.stop()
        out["status"] = card.check_run(prog, workers, 1, [ev.us])
        (raw,) = card.read(session, counters)
    raw = raw.view(np.uint32).reshape(workers, 2, lists.TILES)[:cores]
    want_cnt, want_rows = passes.counters(recs, len(scene["indices"]), cores)
    out["entries"] = int(want_cnt.sum())
    out["cnt_differ"] = int((raw[:, 0] != want_cnt).sum())
    out["rows_differ"] = int((raw[:, 1] != want_rows).sum())
    out["ok"] = bool(
        out["status"]["ok"] and out["cnt_differ"] == 0 and out["rows_differ"] == 0
    )
    return out


def frame(args, session=None):
    """COUNT, the host's step, SETUP, FILL: one frame, each step timed (in `session`,
    or a session of its own)."""
    scene, _ = run.load(args.scene)
    blocks = passes.mesh_blocks(scene)
    recs = raster.setup(scene)
    out = {"scene": args.scene, "blocks": len(blocks), "times": {}}
    tm = out["times"]
    own = contextlib.nullcontext(session) if session else km.Session(log=card.log)
    with own as session:
        workers = card.workers(session)
        cores = args.cores or workers
        dev = session.device
        t0 = time.perf_counter()
        mesh = card.tensor(dev, blocks)
        cam = card.tensor(dev, camera_page(scene))
        counters = card.tensor(dev, np.zeros((workers, S["kCountBytes"] // 4)))
        tm["upload_mesh_s"] = time.perf_counter() - t0
        team = (0, workers)
        base = [mesh.buffer_address(), 0, len(blocks), cam.buffer_address(), 0, cores]

        t0 = time.perf_counter()
        prog = session.load(program(), deadline=1.0)
        tm["load_passes_s"] = time.perf_counter() - t0
        a = list(base)
        a[1] = counters.buffer_address()
        ev = prog.wait(prog.issue(S["kSegCount"], team=team, args=a), timeout=10.0)
        out["count_us"] = ev.us
        prog.stop()
        out["count_status"] = card.check_run(prog, workers, 1, [ev.us])
        t0 = time.perf_counter()
        (raw,) = card.read(session, counters)
        tm["read_counters_s"] = time.perf_counter() - t0
        raw = raw.view(np.uint32).reshape(workers, 2, lists.TILES)[:cores]
        cnt, rows = raw[:, 0].astype(np.int64), raw[:, 1].astype(np.int64)
        want_cnt, want_rows = passes.counters(recs, len(scene["indices"]), cores)
        out["counters_equal"] = bool(
            np.array_equal(cnt, want_cnt) and np.array_equal(rows, want_rows)
        )

        t0 = time.perf_counter()
        mid = passes.middle(cnt, rows, workers)
        tm["middle_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        up = card.upload(dev, mid["pages"], mid["directory"], lists.TILES * 5)
        nxt = np.zeros((workers, lists.TILES), dtype=np.uint32)
        nxt[:cores] = mid["next"]
        slots_t = card.tensor(dev, nxt)
        tm["upload_lists_s"] = time.perf_counter() - t0
        out["list_pages"] = len(mid["pages"])

        t0 = time.perf_counter()
        prog = session.load(program(), deadline=1.0)
        tm["load_passes2_s"] = time.perf_counter() - t0
        a = list(base)
        a[1] = up["lists"].buffer_address()
        a[4] = slots_t.buffer_address()
        a.append(mid["slots"])
        ev = prog.wait(prog.issue(S["kSegSetup"], team=team, args=a), timeout=10.0)
        out["setup_us"] = ev.us
        prog.stop()
        out["setup_status"] = card.check_run(prog, workers, 1, [ev.us])
        if args.lists >= 0:
            out["lists"] = lists_check(session, up, scene, recs, mid, args.lists)

        got = None if args.setup else fill_frame(session, up, out)
    ok = all(out[k]["ok"] for k in ("count_status", "setup_status"))
    ok = ok and out["counters_equal"]
    if "lists" in out:
        lc = out["lists"]
        ok = ok and lc["pages_differ"] == 0 and lc["directory_equal"]
    if got is not None:
        counts, offsets, members = raster.bin_triangles(recs)
        tiles = list(range(lists.TILES))
        out["check"] = run.check(
            args.scene, None, recs, offsets, members, tiles, counts, got
        )
        ok = ok and out["fill_status"]["ok"] and run.check_ok(out["check"])
    out["ok"] = bool(ok)
    return out


def fill_frame(session, up, out):
    """The fill of `card.py` over the lists in place: the frame's pages, timed."""
    tm = out["times"]
    workers = card.workers(session)
    t0 = time.perf_counter()
    fill = session.load(card.program(), deadline=1.0)
    tm["load_fill_s"] = time.perf_counter() - t0
    us = card.loop(fill, lists.K["kSegFill"], card.words(up, 5), 1)
    out["fill_us"] = float(us[0]) if len(us) else None
    fill.stop()
    out["fill_status"] = card.check_run(fill, workers, 1, us)
    t0 = time.perf_counter()
    got = card.read_frame(session, up)
    tm["read_frame_s"] = time.perf_counter() - t0
    return got


WINDOW = 510  # blocks per read: 2 MB of records


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", default="tiny")
    p.add_argument("--cores", type=int, default=0, help="0: every core")
    p.add_argument(
        "--windows", type=int, default=0, help="check this many windows (0: all)"
    )
    p.add_argument("--json", default=None)
    p.add_argument("--dump", default=None, help="save the raw records (.npy)")
    p.add_argument("--frame", action="store_true", help="a frame, not LANES")
    p.add_argument("--count", action="store_true", help="COUNT alone, not LANES")
    p.add_argument("--setup", action="store_true", help="frame without the fill")
    p.add_argument(
        "--lists",
        type=int,
        default=0,
        help="frame: list pages checked (0: all, -1: none)",
    )
    args = p.parse_args()
    if args.frame or args.count or args.setup:
        out = count_only(args) if args.count else frame(args)
        print(json.dumps(out), flush=True)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(out, f, indent=1)
        sys.exit(0 if out["ok"] else 1)
    scene, _ = run.load(args.scene)
    blocks = passes.mesh_blocks(scene)
    starts = list(range(0, len(blocks), WINDOW))
    if args.windows and args.windows < len(starts):
        pick = np.linspace(0, len(starts) - 1, args.windows).round().astype(int)
        starts = [starts[i] for i in sorted(set(pick))]
    out = {"scene": args.scene, "blocks": len(blocks), "windows": len(starts)}
    got = np.zeros((len(blocks) * 32, 32), dtype=np.uint32)
    seen = np.zeros(len(blocks) * 32, dtype=bool)
    with km.Session(log=card.log) as session:
        workers = card.workers(session)
        cores = args.cores or workers
        dev = session.device
        mesh = card.tensor(dev, blocks)
        cam = card.tensor(dev, camera_page(scene))
        res = card.tensor(dev, np.zeros((WINDOW, S["kRecBytes"] // 4)))
        run_ = session.load(program(), deadline=1.0)
        us, ok = [], True
        for w0 in starts:
            n = min(WINDOW, len(blocks) - w0)
            a = [mesh.buffer_address(), res.buffer_address(), len(blocks)]
            a += [cam.buffer_address(), w0, cores, n]
            t = run_.issue(S["kSegLanes"], team=(0, workers), args=a)
            ev = run_.wait(t, timeout=10.0)
            us.append(ev.us)
            if ev.kind != "done":
                ok = False
                break
            (page,) = session.read(res, first=0, pages=n)
            got[32 * w0 : 32 * (w0 + n)] = page.view(np.uint32).reshape(-1, 32)
            seen[32 * w0 : 32 * (w0 + n)] = True
            card.log(f"window {w0} of {len(blocks)}: {n} blocks")
        run_.stop()
        status = card.check_run(run_, workers, len(us), us)
    out["event_us"] = us
    out["status"] = status
    if args.dump:
        np.save(args.dump, got)
    out["check"] = check_lanes(scene, got, seen)
    out["ok"] = bool(
        ok
        and status["ok"]
        and out["check"]["words_differ"] == 0
        and out["check"]["kept_disagree"] == 0
        and out["check"]["box_differ"] == 0
    )
    print(json.dumps(out), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    if not out["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
