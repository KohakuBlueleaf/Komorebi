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
from komorebi.thread import BANK_OF_CLUSTER, TIMER_HZ
from projects.raster import card, card_passes, lists, passes, run
from ref import raster

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
F["RF_WORDS"] = F["RF_W_BALANCE"] + 1
F["RF_HIST_BYTES"] = 64 * F["RF_HIST_MAX"]
S, K = card_passes.S, lists.K
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
        "RF_MAX_CORES": S["kMaxFillCores"],
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
    pairs = {k: v if isinstance(v, tuple) else (F[k], v) for k, v in pairs.items()}
    return {k: v for k, v in pairs.items() if v[0] != v[1]}


def camera_region(scene):
    """The camera's numbers, then room for COUNT's and SETUP's ranges and costs."""
    out = np.zeros((1, F["RF_CAMERA_BYTES"] // 4), dtype=np.uint32)
    out[0, : S["kCamWords"]] = card_passes.camera_page(scene)[0]
    return out


def regions(scene, workers):
    """The thread's regions in frame.h's order: the camera's page, then sizes."""
    work = 256 << 10  # frame.c work_t: middle.h scratch and items (about 130 KB)
    return {
        "camera": camera_region(scene),
        "heavy": F["RF_HEAVY_BYTES"],
        "items": F["RF_ITEMS_REGION_BYTES"],
        "work": work,
        "sums": F["RF_SUMS_BYTES"],
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
        handlers={**fill.handlers, **passes_.handlers},
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


def tensors(session, blocks, workers, list_slots):
    """The DRAM tensors: mesh (once per scene), lists, directory, frame, profiles, and
    the pages only the cores read and write."""
    pages = -(-list_slots // lists.PAGE_SLOTS)
    out_pages = lists.TILES * 5
    chunks = -(-out_pages // card.CHUNK_PAGES)
    frame = [
        card.tensor(
            session,
            np.zeros((min(card.CHUNK_PAGES, out_pages - i * card.CHUNK_PAGES), 1024)),
        )
        for i in range(chunks)
    ]
    up = {
        "mesh": card.tensor(session, blocks),
        "lists": card.tensor(session, np.zeros((pages, lists.PAGE_WORDS))),
        "dir": card.tensor(session, np.zeros((workers, K["kDirBytes"] // 4))),
        "frame": frame,
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
        "list_slots": pages * lists.PAGE_SLOTS,
    }
    session.device.finish()  # the uploads land before any frame runs
    return up


def words(up, blocks, cores, workers, steps, balance=True):
    w = [0] * F["RF_WORDS"]
    w[F["RF_W_BALANCE"]] = int(balance)
    w[F["RF_W_CORES"]] = cores
    w[F["RF_W_WORKERS"]] = workers
    w[F["RF_W_BANK"]] = BANK_OF_CLUSTER[CLUSTER]
    w[F["RF_W_MESH"]] = up["mesh"].address
    w[F["RF_W_BLOCKS"]] = blocks
    w[F["RF_W_LISTS"]] = up["lists"].address
    w[F["RF_W_LIST_SLOTS"]] = up["list_slots"]
    w[F["RF_W_DIR"]] = up["dir"].address
    w[F["RF_W_STEPS"]] = steps
    w[F["RF_W_FRAME_PAGES"]] = card.CHUNK_PAGES
    w[F["RF_W_FRAMES"]] = len(up["frame"])
    for i, t in enumerate(up["frame"]):
        w[F["RF_W_FRAME"] + i] = t.address
    w[F["RF_W_PROF"]] = up["prof"].address
    w[F["RF_W_SPROF"]] = up["sprof"].address
    for name in ("counters", "hist", "slots", "hprefix"):
        w[F[f"RF_W_{name.upper()}"]] = up[name].address
    return w


def result_dict(r):
    """The thread's result: code, what the middle pass found, each part in
    microseconds."""
    o = list(r.words) + [0] * (F["RF_OUT_WORDS"] - len(r.words))
    us = 1e6 / TIMER_HZ
    names = ["count", "sum", "hist", "hsum", "middle", "place", "setup", "fill"]
    return {
        "ran": r.ran,
        "state": r.state,
        "error": r.error,
        "code": r.code,
        "where": r.where,
        "thread_microseconds": r.microseconds,
        "slots": o[F["RF_O_SLOTS"]],
        "items": o[F["RF_O_ITEMS"]],
        "marks": o[F["RF_O_MARKS"]],
        "heavy": o[F["RF_O_HEAVY"]],
        "microseconds": {n: o[F[f"RF_O_{n.upper()}"]] * us for n in names},
        "middle_microseconds": {
            n: o[F[f"RF_O_{n}"]] * us
            for n in (
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


def expected(scene, recs, cores, workers):
    """The host's middle pass on the reference's counts: what the thread must find."""
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


def sprof_dict(sp, cores):
    """Per pass, each core's writer total and pack counters from the profile pages of
    the last frame (wall ticks), and the slowest core against the mean."""
    out = {}
    for name, handler in (("count", S["kHandlerCount"]), ("setup", S["kHandlerSetup"])):
        page = sp[:cores, 32 * (handler - S["kHandlerCount"]) :][:, :32].astype(
            np.int64
        )
        total = page[:, S["kSProfWriter"] + 2]
        pack = page[:, S["kSProfPack"] : S["kSProfPack"] + 8]
        out[name] = {
            "total": total.tolist(),
            "kept": pack[:, 4].tolist(),
            "entries": pack[:, 6].tolist(),
            "max_over_mean": float(total.max() / max(total.mean(), 1)),
        }
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
        t0 = time.perf_counter()
        slots = args.list_slots or 2 * len(scene["indices"]) + 65536
        up = tensors(session, blocks, workers, slots)
        tm["upload_s"] = time.perf_counter() - t0
        w = words(up, len(blocks), cores, workers, args.steps, args.balance)
        w[F["RF_W_REPEAT"]] = args.repeat
        t0 = time.perf_counter()
        prog = session.load(program(scene, w, workers), deadline=1.0, trace=16)
        tm["load_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        r = prog.thread_result(timeout=15.0)
        tm["thread_wait_s"] = time.perf_counter() - t0
        try:
            prog.stop()
        except km.errors.KernelFailed as e:
            out["stop_error"] = str(e)
            tr = prog.trace()
            h = km.abi.headers()
            rows = tr.rows[:workers]
            seen, done = rows[:, :, h["kTraceSeen"]], rows[:, :, h["kTraceDone"]]
            out["trace"] = {
                "seen_cores": (seen != 0).sum(0).tolist(),
                "done_cores": (done != 0).sum(0).tolist(),
                "not_done": [
                    [
                        int(c)
                        for c in np.flatnonzero((seen[:, k] != 0) & (done[:, k] == 0))
                    ]
                    for k in range(rows.shape[1])
                ],
            }
            print(json.dumps(out, default=str), flush=True)
        out["thread"] = result_dict(r)
        out["x280_mhz"] = session.pll_mhz
        out["status"] = card.check_run(prog, workers, 0, [])
        (sp,) = card.read(session, up["sprof"])
        out["sprof"] = sprof_dict(sp.view(np.uint32).reshape(workers, -1), cores)
        # Regions are not read back (a region read reads every bank's pages of the
        # regions' tensor at once): the tensors the thread's commands wrote are.
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
            want = [card.CHUNK_PAGES] + [t.address for t in up["frame"]]
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
        "--steps",
        type=int,
        default=4,
        help="2 the middle pass, 3 SETUP, 4 FILL and the image",
    )
    p.add_argument("--list-slots", type=int, default=0, help="0: from the triangles")
    p.add_argument(
        "--lists", type=int, default=0, help="list pages checked (0: all, -1: none)"
    )
    p.add_argument(
        "--no-image", dest="image", action="store_false", help="no image read-back"
    )
    p.add_argument(
        "--no-balance",
        dest="balance",
        action="store_false",
        help="equal block partitions",
    )
    p.add_argument("--repeat", type=int, default=0, help="frames in a row (0: one)")
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
