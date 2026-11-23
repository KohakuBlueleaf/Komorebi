"""Running the host-binned rasterizer on a card through the host package: the lists,
directory, frame and profile tensors are written before the load, then FILL commands go
to every core under the dispatcher, one in flight at a time."""

import re
import time

import numpy as np

import komorebi as km
from komorebi.device import chips
from projects.raster import lists

K = lists.K
KERNELS = lists.HERE / "kernels"
MIDDLE_H = lists.HERE / "thread" / "middle.h"
SETUP_H = KERNELS / "setup.h"
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


# The accessor arguments of a DRAM tensor of interleaved pages, as Session.tensor makes
# them (its configuration word, then its page bytes): for a build with no tensors.
DRAM_ARGS = 2


def compile_args(uploaded=None):
    """The reader's and the writer's tensor arguments (km::Tensor, from 0): directory
    and lists; directory, frame and profile pages; of `uploaded`'s tensors, or (none)
    those of DRAM tensors of their page sizes."""

    def args(name, page_bytes):
        if uploaded is None:
            return [DRAM_ARGS, page_bytes]
        tensor = uploaded[name][0] if name == "frame" else uploaded[name]
        return tensor.tensor_args()

    return {
        "reader": args("dir", K["kDirBytes"]) + args("lists", K["kInPageBytes"]),
        "writer": args("dir", K["kDirBytes"])
        + args("frame", K["kOutPageBytes"])
        + args("prof", K["kOutPageBytes"]),
    }


# The fill's per-triangle numbers: False (the default) as immediates the pack thread
# builds; True (RASTER_SPREAD) spread over the lanes by the reader, unpacked into Dst.
SPREAD = False


def spread_parts():
    """The option's buffer, compute settings and defines, empty without it."""
    if not SPREAD:
        return [], {}, {}
    tiles = [
        km.Buffer(K[name], "uint32", K["kSpreadBytes"], K["kSpreadPages"])
        for name in ("kCbSpread", "kCbSpreadOdd")
    ]
    tasks = km.Buffer(
        K["kCbSpreadTask"], "uint32", K["kSpreadTaskBytes"], K["kSpreadTaskPages"]
    )
    modes = {buffer.index: "UnpackToDestFp32" for buffer in tiles}
    return [*tiles, tasks], {"unpack_to_dest_mode": modes}, {"RASTER_SPREAD": "1"}


def program(uploaded=None, held=None):
    """The rasterizer's three kernels on every core, one command in flight per core (an
    event's time its own); tensors' arguments from `uploaded` (upload's), or those of
    DRAM tensors; built for the capacity `held` (card_thread.Capacity; by default the
    default frame's tiles, the host-binned path's)."""
    spread_buffers, spread_compute, spread_defines = spread_parts()
    return km.Program(
        "raster",
        kernels=KERNELS,
        buffers=[
            km.Buffer(K["kCbIn"], "uint32", K["kInPageBytes"], K["kInPages"]),
            km.Buffer(K["kCbDirReader"], "uint32", K["kDirBytes"], 1),
            km.Buffer(K["kCbDirWriter"], "uint32", K["kDirBytes"], 1),
            km.Buffer(K["kCbOut"], "uint32", K["kOutPageBytes"], K["kOutPages"]),
            km.Buffer(K["kCbEmpty"], "uint32", K["kOutPageBytes"], K["kPlanes"]),
            *spread_buffers,
        ],
        compile_args=compile_args(uploaded),
        compute={
            "math_fidelity": "HiFi4",
            "fp32_dest_acc_en": True,
            "enable_trisc2_rvv": "if_present",
            **spread_compute,
        },
        defines={**spread_defines, **capacity_defines(held)},
        depth=1,
        handlers=lists.RASTER_H,
    )


def workers(session):
    """The cores the program runs on: the core numbered `index` is at (index % x,
    index // x); the grid's, less a Tensix dispatcher's; on islands every flat rank."""
    return session.ranks


def tensor(session, rows, **placed):
    """A row-major uint32 array as a DRAM tensor (one page per row); `placed`, on a
    session of islands, its placement (split, group)."""
    return session.tensor(np.asarray(rows, dtype=np.uint32), "uint32", **placed)


def read(session, *tensors):
    """The tensors' words, [pages, page words] uint32 each, read by DRAM banks one
    tensor at a time (each at most 2 MB) with a progress line per tensor."""
    pages_read = []
    for number, source_tensor in enumerate(tensors):
        (pages,) = session.read(source_tensor)
        pages_read.append(pages.view(np.uint32))
        log(f"read {number + 1} of {len(tensors)}: {pages.nbytes} bytes")
    return pages_read


def read_frame(session, uploaded):
    """The frame's pages, its tensors read one by one."""
    return np.concatenate(read(session, *uploaded["frame"]))


# A tile is cut into bands above SPLIT of a core's mean load (thread/middle.h RM_SPLIT).
SPLIT = float(re.search(r"#define RM_SPLIT ([\d.]+)", MIDDLE_H.read_text())[1])
# The base word's heavy index field (kernels/setup.h kBaseSlotBits): 1 + index < 2^9.
HEAVY_FIELD = 1 << (
    32 - int(re.search(r"kBaseSlotBits = (\d+);", SETUP_H.read_text())[1])
)
# The host-stepped passes' tables (card_passes standalone, passes.py, experiments): a
# frame there is the default 1,024 tiles; the frame program sizes them from its cores.
STEPPED_HEAVY, STEPPED_BAND_ITEMS = 256, 1024


def heavy_capacity(held=None):
    """(heavy tiles, bands' items) the tables of a capacity `held` hold: `cores` / SPLIT
    and one (a frame has fewer: a tile is cut above SPLIT of a core's mean), the heavy
    at most its tiles, the items to 32; without cores the host-stepped passes'.
    ValueError past the base word's heavy index field."""
    if held is None or held.cores is None:
        return STEPPED_HEAVY, STEPPED_BAND_ITEMS
    most = int(held.cores / SPLIT) + 1
    heavy = min(held.tiles, most)
    if heavy >= HEAVY_FIELD:
        raise ValueError(
            f"{held.cores} fill cores can cut {heavy} heavy tiles: the base word's "
            f"heavy index holds {HEAVY_FIELD - 1} (kernels/setup.h kBaseSlotBits), "
            f"at most {int((HEAVY_FIELD - 2) * SPLIT)} cores"
        )
    return heavy, -(-most // 32) * 32  # the items page's rows on 64 bytes (setup.h)


def capacity_defines(held=None):
    """The kernels' build defines of a capacity `held` (card_thread.Capacity; None: the
    default frame's tiles): RASTER_TILES, RASTER_HEAVY, RASTER_BAND_ITEMS
    (heavy_capacity), and RASTER_WIDE, RASTER_RECOUNT when set."""
    heavy, band_items = heavy_capacity(held)
    counts = {"RASTER_HEAVY": str(heavy), "RASTER_BAND_ITEMS": str(band_items)}
    if held is None:
        return {"RASTER_TILES": str(lists.TILES), **counts}
    return {
        "RASTER_TILES": str(held.tiles),
        **counts,
        **({"RASTER_WIDE": "1"} if held.wide else {}),
        **({"RASTER_RECOUNT": "1"} if held.recount else {}),
    }


# Frame pages per tensor: at most 2 MB, a whole number of 3- and 5-page items.
CHUNK_PAGES = 510


def chunk_pages(pages):
    """Pages per tensor of a frame of `pages`: CHUNK_PAGES, or as many more (a multiple
    of 15) as keep it in the kDirFrameMax tensors the directory names."""
    most = K["kDirFrameMax"]
    if -(-pages // CHUNK_PAGES) <= most:
        return CHUNK_PAGES
    each = -(-pages // most)  # pages a tensor, then up to a multiple of 15
    return -(-each // 15) * 15


def even(session, steps):
    """On a session of two islands, the hint that splits `steps` rows over k ranks on
    each side of the edge, k the most that divides them evenly, so each island holds
    as many as its ranks take; {} on one card or where no k does (the session then
    warns that the split is not in proportion)."""
    if not hasattr(session, "graph") or len(session.graph.islands) != 2:
        return {}
    edge = session.graph.first(1)
    for k in range(min(edge, session.graph.count(1)), 0, -1):
        if steps % (2 * k) == 0:
            return {"split": 0, "group": (edge - k, 2 * k)}
    return {}


def upload(session, pages, directory, output_pages, frame_placed=None):
    """The DRAM tensors of a run: lists, directory (naming the frame's tensors), the
    frame in tensors of CHUNK_PAGES pages (placed by `frame_placed` on a session of
    islands, else evenly over the edge), and the profile pages."""
    chunks = -(-output_pages // CHUNK_PAGES)
    if chunks > K["kDirFrameMax"]:
        raise ValueError(f"a frame of {output_pages} pages needs {chunks} tensors")
    sizes = [
        min(CHUNK_PAGES, output_pages - chunk * CHUNK_PAGES) for chunk in range(chunks)
    ]
    frame = [
        tensor(session, np.zeros((size, 1024)), **(frame_placed or even(session, size)))
        for size in sizes
    ]
    directory = np.array(directory, dtype=np.uint32)
    frame_column = K["kDirFrame"]
    directory[:, frame_column] = CHUNK_PAGES
    directory[:, frame_column + 1 : frame_column + 1 + chunks] = [
        chunk_tensor.address for chunk_tensor in frame
    ]
    return {
        "lists": tensor(session, pages, **even(session, len(pages))),
        "dir": tensor(session, directory),
        "frame": frame,
        "prof": tensor(session, np.zeros((len(directory), 1024), dtype=np.uint32)),
    }


def words(uploaded, pages_per_item):
    """FILL's arguments: directory, lists, (unused), pages per item, profile."""
    return [
        uploaded["dir"].address,
        uploaded["lists"].address,
        0,
        pages_per_item,
        uploaded["prof"].address,
    ]


def refuse(session, handler):
    """Raises on what the session's chip does not run: without the Dst window
    (chips.Chip.dst_window) the arithmetic handler's part tiles, and SPREAD."""
    if chips.chip(session.arch).dst_window:
        return
    if handler == K["kHandlerArith"]:
        raise ValueError(
            f"the arithmetic handler needs the Dst window ({session.arch})"
        )
    if SPREAD:
        raise ValueError(f"SPREAD needs the Dst window ({session.arch})")


def run_lists(
    session,
    handler,
    pages,
    directory,
    output_pages,
    pages_per_item,
    repetitions=1,
    frame=True,
    args=None,
    frame_placed=None,
):
    """Loads the program, runs `reps` commands of `handler` over the lists and directory
    (`args` in place of the tensors' addresses), stops, reads back. Returns (output
    pages or None, profile pages, status, command microseconds, host seconds)."""
    if K["kOutPages"] % pages_per_item:
        raise ValueError(f"{pages_per_item}-page items do not divide the output buffer")
    if len(directory) != workers(session):
        raise ValueError(f"{len(directory)} directory pages for {workers(session)}")
    refuse(session, handler)
    times = {}
    started = time.perf_counter()
    uploaded = upload(session, pages, directory, output_pages, frame_placed)
    times["upload_s"] = time.perf_counter() - started
    started = time.perf_counter()
    run = session.load(program(uploaded), deadline=1.0)
    times["load_s"] = time.perf_counter() - started
    log(f"loaded; {repetitions} commands over {len(pages)} list pages")
    microseconds = loop(
        run, handler, args or words(uploaded, pages_per_item), repetitions
    )
    times["commands_s"] = time.perf_counter() - started - times["load_s"]
    run.stop()
    log("stopped")
    started = time.perf_counter()
    (profile_pages,) = read(session, uploaded["prof"])
    frame_pages = read_frame(session, uploaded) if frame else None
    times["readback_s"] = time.perf_counter() - started
    log(f"read back in {times['readback_s']:.1f} s")
    status = check_run(run, len(directory), repetitions, microseconds)
    return frame_pages, profile_pages, status, microseconds, times


def loop(run, handler, args, repetitions, batch=512):
    """Submits `reps` commands of `handler` on every core in batches and waits for them;
    returns each command's microseconds from submit to acknowledgement."""
    microseconds = []
    while len(microseconds) < repetitions:
        commands = min(batch, repetitions - len(microseconds))
        cores = getattr(
            run, "ranks", run.cores
        )  # a card's run: less a Tensix dispatcher
        handles = run.submit_many(handler, 0, cores, np.tile(args, (commands, 1)))
        events = run.wait(list(handles), timeout=10.0 + commands * 0.01)
        # over islands an event is its parts, one an island: the slowest part's time
        parts = [(event,) if hasattr(event, "kind") else event for event in events]
        microseconds += [max(part.microseconds for part in event) for event in parts]
        if any(part.kind != "done" for event in parts for part in event):
            break
    return np.array(microseconds)


def check_run(run, cores, repetitions, microseconds):
    """The run's end as a dict, `ok` when every core ended cleanly and every command
    finished; raises nothing."""
    states = run.state()
    writer_column = km.abi.headers()["kStWriter"]
    if hasattr(run, "runs"):  # a session of islands: each island's ranks, in order
        state_words = np.concatenate(
            [
                np.asarray(island_run.state_words())[list(cores_of)]
                for island_run, cores_of in zip(run.runs, run.graph.rank_cores)
            ]
        )
    else:
        state_words = np.asarray(run.state_words())
    writer_words = state_words[:cores, writer_column].astype(np.int64)
    bad_cores = np.flatnonzero((writer_words & 0xFF) == K["kGaveBadItem"])
    ended = {
        key: sum(state[key] != 0 for state in states[:cores])
        for key in ("reader_end", "writer_end", "compute_end")
    }
    gave = [
        {"core": state["core"], "reader": state["reader"], "writer": state["writer"]}
        for state in states[:cores]
        if state["reader"] or state["writer"] or state["compute"]
    ]
    results = run.result()
    kernels = [
        (one or {}).get("error_name")
        for one in (results if isinstance(results, list) else [results])
    ]
    result = {
        "kernel": kernels[0] if len(kernels) == 1 else kernels,
        "commands": int(len(microseconds)),
        "cores_ended": ended,
        "compute_wrong": sum(sum(state["compute_wrong"]) for state in states[:cores]),
        "bad_items": [
            {"core": int(core), "where": int(writer_words[core] >> 8 & 0xFF)}
            for core in bad_cores
        ],
        "gave_up": gave,
    }
    result["ok"] = bool(
        len(microseconds) == repetitions
        and all(count == cores for count in ended.values())
        and result["compute_wrong"] == 0
        and not gave
        and not result["bad_items"]
    )
    return result
