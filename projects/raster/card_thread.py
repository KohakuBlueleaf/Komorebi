"""A frame conducted by the work cluster's thread (`thread/frame.c`), each step it
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
from komorebi.thread import BANK_OF_CLUSTER, TIMER_HZ
from projects.raster import card, card_passes, lists, passes, run
from ref import raster

THREAD = lists.HERE / "thread"
KERNELS = lists.HERE / "kernels"
FRAME_H = THREAD / "frame.h"
F = {
    name: int(value.rstrip("u"), 0)
    for name, value in re.findall(
        r"^#define (RF_\w+) (\d+u?)\b", FRAME_H.read_text(), re.M
    )
}
# expressions in frame.h
F["RF_W_PROF"] = F["RF_W_FRAME"] + F["RF_FRAME_MAX"]
F["RF_W_REPEAT"] = F["RF_W_PROF"] + 1
F["RF_WORDS"] = F["RF_W_REPEAT"] + 1
S, K = card_passes.S, lists.K
CLUSTER = 1
PAGES_PER_ITEM = 5  # the frame's pages per tile (card.words)


def same_numbers():
    """The numbers frame.h takes from the kernels' headers, as pairs that differ."""
    pairs = {
        "RF_SEG_COUNT": S["kSegCount"],
        "RF_SEG_SETUP": S["kSegSetup"],
        "RF_SEG_HIST": S["kSegHist"],
        "RF_SEG_PLACE": S["kSegPlace"],
        "RF_SEG_FILL": K["kSegFill"],
        "RF_PAGES_PER_ITEM": PAGES_PER_ITEM,
        "RF_HDR_TILE": K["kHdrTile"],
        "RF_HDR_COUNT": K["kHdrCount"],
        "RF_HDR_ROW0": K["kHdrRow0"],
        "RF_HDR_ROW1": K["kHdrRow1"],
        "RF_DIR_FRAME": K["kDirFrame"],
        "RF_FRAME_MAX": K["kDirFrameMax"],
        "RF_COUNTER_BYTES": S["kCountBytes"],
        "RF_HEAVY_BYTES": S["kHeavyBytes"],
        "RF_SLOTS_BYTES": S["kSlotsBytes"],
        "RF_ITEMS_BYTES": S["kItemsBytes"],
        "RF_DIR_BYTES": S["kDirPageBytes"],
        "RF_MARK_BYTES": S["kMarkBytes"],
    }
    return {k: (F[k], v) for k, v in pairs.items() if F[k] != v}


def regions(scene, workers):
    """The thread's regions in frame.h's order: the camera's page, then sizes."""
    work = 256 << 10  # frame.c work_t: middle.h scratch, items, marks (about 170 KB)
    return {
        "camera": card_passes.camera_page(scene),
        "counters": workers * F["RF_COUNTER_BYTES"],
        "heavy": F["RF_HEAVY_BYTES"],
        "hist": workers * F["RF_HIST_MAX"] * 64,
        "slots": workers * F["RF_SLOTS_BYTES"],
        "items": F["RF_ITEMS_BYTES"],
        "dir": workers * F["RF_DIR_BYTES"],
        "marks": (2048 + 248) * F["RF_MARK_BYTES"],
        "work": work,
    }


def program(scene, words, workers):
    """The frame program: the passes' and the fill's kernels behind one entry each
    (kernels/frame_*.cpp), both programs' buffers, with the frame thread on work
    cluster 1."""
    fill, passes_ = card.program(), card_passes.program()
    prog = km.Program(
        "raster_frame",
        kernels={
            k: KERNELS / f"frame_{k}.cpp" for k in ("reader", "compute", "writer")
        },
        buffers=[*fill.buffers, *passes_.buffers],
        compute=fill.compute,
        depth=1,
        segments={**fill.segments, **passes_.segments},
    )
    prog.thread(
        cluster=CLUSTER,
        sources=[THREAD / "frame.c", THREAD / "middle.c"],
        includes=[THREAD],
        regions=regions(scene, workers),
        words=words,
        deadline=10.0,
    )
    return prog


def tensors(dev, blocks, workers, list_slots):
    """The DRAM tensors: mesh (once per scene), lists, directory, frame, profile."""
    pages = -(-list_slots // lists.PAGE_SLOTS)
    out_pages = lists.TILES * 5
    chunks = -(-out_pages // card.CHUNK_PAGES)
    frame = [
        card.tensor(
            dev,
            np.zeros((min(card.CHUNK_PAGES, out_pages - i * card.CHUNK_PAGES), 1024)),
        )
        for i in range(chunks)
    ]
    return {
        "mesh": card.tensor(dev, blocks),
        "lists": card.tensor(dev, np.zeros((pages, lists.PAGE_WORDS))),
        "dir": card.tensor(dev, np.zeros((workers, K["kDirBytes"] // 4))),
        "frame": frame,
        "prof": card.tensor(dev, np.zeros((workers, 1024))),
        "list_slots": pages * lists.PAGE_SLOTS,
    }


def words(up, blocks, cores, workers, steps):
    w = [0] * F["RF_WORDS"]
    w[F["RF_W_CORES"]] = cores
    w[F["RF_W_WORKERS"]] = workers
    w[F["RF_W_BANK"]] = BANK_OF_CLUSTER[CLUSTER]
    w[F["RF_W_MESH"]] = up["mesh"].buffer_address()
    w[F["RF_W_BLOCKS"]] = blocks
    w[F["RF_W_LISTS"]] = up["lists"].buffer_address()
    w[F["RF_W_LIST_SLOTS"]] = up["list_slots"]
    w[F["RF_W_DIR"]] = up["dir"].buffer_address()
    w[F["RF_W_STEPS"]] = steps
    w[F["RF_W_FRAME_PAGES"]] = card.CHUNK_PAGES
    w[F["RF_W_FRAMES"]] = len(up["frame"])
    for i, t in enumerate(up["frame"]):
        w[F["RF_W_FRAME"] + i] = t.buffer_address()
    w[F["RF_W_PROF"]] = up["prof"].buffer_address()
    return w


def result_dict(r):
    """The thread's result: code, what the step found, each part in microseconds."""
    o = list(r.words) + [0] * (F["RF_OUT_WORDS"] - len(r.words))
    us = 1e6 / TIMER_HZ
    names = ["count", "hist", "middle", "place", "setup", "fill"]
    return {
        "ran": r.ran,
        "state": r.state,
        "error": r.error,
        "code": r.code,
        "where": r.where,
        "thread_us": r.us,
        "slots": o[F["RF_O_SLOTS"]],
        "items": o[F["RF_O_ITEMS"]],
        "marks": o[F["RF_O_MARKS"]],
        "heavy": o[F["RF_O_HEAVY"]],
        "us": {n: o[F[f"RF_O_{n.upper()}"]] * us for n in names},
        "middle_us": {
            n: o[F[f"RF_O_{n}"]] * us
            for n in ("TAKE", "HEAVY_US", "CLEAR", "LAYOUT", "NEXT", "PAGES", "GIVE")
        },
    }


def expected(scene, recs, cores, workers):
    """The host's step on the reference's counts: what the thread must find."""
    n = len(scene["indices"])
    cnt, rows = passes.counters(recs, n, cores)
    heavy = passes.heavy_tiles(cnt, rows, workers)
    bands = None
    if heavy:
        ht = [t for t, _ in heavy]
        first, last = passes.histograms(recs, n, cores)
        bands = passes.items_of(cnt, rows, heavy, first[:, ht], last[:, ht])
    mid = passes.middle(cnt, rows, workers, bands=bands)
    pages = mid["pages"].reshape(-1, lists.SLOT)
    mid["found"] = {
        "slots": int(mid["slots"]),
        "items": len(mid["items"]),
        "marks": int((pages[:, 0] >= K["kMarkTile"]).sum()),
        "heavy": len(heavy),
    }
    return mid


def slots_differ(session, up, recs, mid):
    """Differing list slots by kind (header, end, record), for a small frame."""
    counts, offsets, members = raster.bin_triangles(recs)
    cards = lists.pack_records(recs)
    cards[:, K["kRecIndex"]] = 0
    want, _ = lists.build_lists(cards, offsets, members, mid["tiles_of"])
    (got,) = session.read(up["lists"], first=0, pages=len(want))
    want = want.reshape(-1, lists.SLOT)
    got = got.view(np.uint32).reshape(-1, lists.SLOT)
    bad = (got != want).any(axis=1)
    kind = np.where(
        want[:, 0] == K["kMarkTile"],
        "header",
        np.where(want[:, 0] == K["kMarkEnd"], "end", "record"),
    )
    out = {k: int((bad & (kind == k)).sum()) for k in ("header", "end", "record")}
    first = np.flatnonzero(bad)[:3]
    out["first"] = [
        {"slot": int(i), "want": want[i, :6].tolist(), "got": got[i, :6].tolist()}
        for i in first
    ]
    return out


def frame(args, session=None):
    scene, _ = run.load(args.scene)
    blocks = passes.mesh_blocks(scene)
    recs = raster.setup(scene)
    out = {"scene": args.scene, "blocks": len(blocks), "times": {}}
    tm = out["times"]
    bad = same_numbers()
    if bad:
        raise SystemExit(f"frame.h differs from the kernels' headers: {bad}")
    own = contextlib.nullcontext(session) if session else km.Session(log=card.log)
    with own as session:
        workers = card.workers(session)
        cores = args.cores or workers
        dev = session.device
        t0 = time.perf_counter()
        slots = args.list_slots or 2 * len(scene["indices"]) + 65536
        up = tensors(dev, blocks, workers, slots)
        tm["upload_s"] = time.perf_counter() - t0
        w = words(up, len(blocks), cores, workers, args.steps)
        t0 = time.perf_counter()
        prog = session.load(program(scene, w, workers), deadline=1.0)
        tm["load_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        r = prog.thread_result(timeout=15.0)
        tm["thread_wait_s"] = time.perf_counter() - t0
        prog.stop()
        out["thread"] = result_dict(r)
        out["x280_mhz"] = session.pll_mhz
        out["status"] = card.check_run(prog, workers, 0, [])
        # Regions are not read back (a region read reads every bank's pages of the
        # regions' tensor at once): the tensors the thread's records wrote are.
        mid = expected(scene, recs, cores, workers)
        found = {k: out["thread"][k] for k in mid["found"]}
        out["found_equal"] = found == mid["found"]
        ok = r.ran and r.code == 0 and out["found_equal"] and out["status"]["ok"]
        if args.steps >= 3:
            out["lists"] = card_passes.lists_check(
                session, up, scene, recs, mid, args.lists
            )
            (dirs,) = card.read(session, up["dir"])
            dirs = dirs.view(np.uint32).reshape(workers, -1).copy()
            frame_words = dirs[:, K["kDirFrame"] :].copy()
            dirs[:, K["kDirFrame"] :] = 0
            out["dir_equal"] = bool(np.array_equal(dirs, mid["directory"]))
            want = [card.CHUNK_PAGES] + [t.buffer_address() for t in up["frame"]]
            out["dir_frame_equal"] = bool(
                (frame_words[:, : len(want)] == want).all()
                and not frame_words[:, len(want) :].any()
            )
            if out["lists"]["pages_differ"] and len(mid["pages"]) <= 1024:
                out["slots_differ"] = slots_differ(session, up, recs, mid)
            ok = ok and out["lists"]["pages_differ"] == 0
            ok = ok and out["dir_equal"] and out["dir_frame_equal"]
        if args.steps >= 4 and args.image:
            t0 = time.perf_counter()
            img = card.read_frame(session, up)
            tm["read_frame_s"] = time.perf_counter() - t0
            counts, offsets, members = raster.bin_triangles(recs)
            out["check"] = run.check(
                args.scene,
                None,
                recs,
                offsets,
                members,
                list(range(lists.TILES)),
                counts,
                img,
            )
            ok = ok and run.check_ok(out["check"])
    out["ok"] = bool(ok)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", default="tiny")
    p.add_argument("--cores", type=int, default=0, help="0: every core")
    p.add_argument(
        "--steps", type=int, default=4, help="2 the step, 3 SETUP, 4 FILL and the image"
    )
    p.add_argument("--list-slots", type=int, default=0, help="0: from the triangles")
    p.add_argument(
        "--lists", type=int, default=0, help="list pages checked (0: all, -1: none)"
    )
    p.add_argument(
        "--no-image", dest="image", action="store_false", help="no image read-back"
    )
    p.add_argument("--json", default=None)
    args = p.parse_args()
    out = frame(args)
    print(json.dumps(out), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
