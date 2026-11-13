"""Running the host-binned rasterizer on a card through the host package: the lists,
directory, frame and profile tensors are written before the load, then FILL records go
to every core under the dispatch kernel, one in flight at a time."""

import time

import numpy as np
import torch
import ttnn

import komorebi as km
from projects.raster import lists

K = lists.K
KERNELS = lists.HERE / "kernels"
T0 = time.monotonic()


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def program():
    """The rasterizer's three kernels on every core; one record in flight per core, so
    a record's event time is its own, issue to the last core's acknowledgement."""
    return km.Program(
        "raster",
        kernels=KERNELS,
        buffers=[
            km.Buffer(K["kCbIn"], "uint32", K["kInPageBytes"], K["kInPages"]),
            km.Buffer(K["kCbDirReader"], "uint32", K["kDirBytes"], 1),
            km.Buffer(K["kCbDirWriter"], "uint32", K["kDirBytes"], 1),
            km.Buffer(K["kCbOut"], "uint32", K["kOutPageBytes"], K["kOutPages"]),
            km.Buffer(K["kCbEmpty"], "uint32", K["kOutPageBytes"], K["kPlanes"]),
        ],
        compute={
            "math_fidelity": "HiFi4",
            "fp32_dest_acc_en": True,
            "enable_trisc2_rvv": True,
        },
        depth=1,
        segments=lists.RASTER_H,
    )


def workers(session):
    """The cores of the open card's grid; core c is (c % x, c // x)."""
    grid = session.device.compute_with_storage_grid_size()
    return grid.x * grid.y


def tensor(device, rows):
    """A row-major uint32 array as an int32 DRAM tensor (one page per row)."""
    data = np.ascontiguousarray(np.asarray(rows, dtype=np.uint32)).view(np.int32)
    return ttnn.from_torch(
        torch.from_numpy(data),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def read(session, *tensors):
    """The tensors' words, [pages, page words] uint32 each, read by DRAM banks one
    tensor at a time (each at most 2 MB) with a progress line per tensor."""
    out = []
    for i, t in enumerate(tensors):
        (pages,) = session.read(t)
        out.append(pages.view(np.uint32))
        log(f"read {i + 1} of {len(tensors)}: {pages.nbytes} bytes")
    return out


def read_frame(session, up):
    """The frame's pages, its tensors read one by one."""
    return np.concatenate(read(session, *up["frame"]))


# Frame pages per tensor: at most 2 MB, a whole number of 3- and 5-page items.
CHUNK_PAGES = 510


def upload(device, pages, directory, out_pages):
    """The DRAM tensors of a run: lists, directory (naming the frame's tensors), the
    frame in tensors of CHUNK_PAGES pages, and the profile pages."""
    chunks = -(-out_pages // CHUNK_PAGES)
    if chunks > K["kDirFrameMax"]:
        raise ValueError(f"a frame of {out_pages} pages needs {chunks} tensors")
    frame = [
        tensor(device, np.zeros((min(CHUNK_PAGES, out_pages - i * CHUNK_PAGES), 1024)))
        for i in range(chunks)
    ]
    directory = np.array(directory, dtype=np.uint32)
    at = K["kDirFrame"]
    directory[:, at] = CHUNK_PAGES
    directory[:, at + 1 : at + 1 + chunks] = [t.buffer_address() for t in frame]
    return {
        "lists": tensor(device, pages),
        "dir": tensor(device, directory),
        "frame": frame,
        "prof": tensor(device, np.zeros((len(directory), 1024), dtype=np.uint32)),
    }


def words(up, per_item):
    """FILL's arguments: directory, lists, (unused), pages per item, profile."""
    return [
        up["dir"].buffer_address(),
        up["lists"].buffer_address(),
        0,
        per_item,
        up["prof"].buffer_address(),
    ]


def run_lists(
    session, seg, pages, directory, out_pages, per_item, reps=1, frame=True, args=None
):
    """Loads the program, runs `reps` records of `seg` over the lists and directory
    (`args` in place of the tensors' addresses), stops, reads back. Returns (output
    pages or None, profile pages, status, record microseconds, host seconds)."""
    if K["kOutPages"] % per_item:
        raise ValueError(f"{per_item}-page items do not divide the output buffer")
    if len(directory) != workers(session):
        raise ValueError(f"{len(directory)} directory pages for {workers(session)}")
    times = {}
    t0 = time.perf_counter()
    up = upload(session.device, pages, directory, out_pages)
    times["upload_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    run = session.load(program(), deadline=1.0)
    times["load_s"] = time.perf_counter() - t0
    log(f"loaded; {reps} records over {len(pages)} list pages")
    us = loop(run, seg, args or words(up, per_item), reps)
    times["records_s"] = time.perf_counter() - t0 - times["load_s"]
    run.stop()
    log("stopped")
    t0 = time.perf_counter()
    (prof,) = read(session, up["prof"])
    out = read_frame(session, up) if frame else None
    times["readback_s"] = time.perf_counter() - t0
    log(f"read back in {times['readback_s']:.1f} s")
    status = check_run(run, len(directory), reps, us)
    return out, prof, status, us, times


def loop(run, seg, args, reps, batch=512):
    """Issues `reps` records of `seg` on every core in batches and waits for them;
    returns each record's microseconds from issue to acknowledgement."""
    us = []
    while len(us) < reps:
        n = min(batch, reps - len(us))
        tickets = run.issue_many(seg, 0, run.cores, np.tile(args, (n, 1)))
        events = run.wait(list(tickets), timeout=10.0 + n * 0.01)
        us += [e.us for e in events]
        if any(e.kind != "done" for e in events):
            break
    return np.array(us)


def check_run(run, cores, reps, us):
    """The run's end as a dict, `ok` when every core ended cleanly and every record
    finished; raises nothing."""
    states = run.state()
    st_writer = km.abi.headers()["kStWriter"]
    word = np.asarray(run.state_words())[:cores, st_writer].astype(np.int64)
    bad = np.flatnonzero((word & 0xFF) == K["kGaveBadItem"])
    ended = {
        k: sum(c[k] != 0 for c in states[:cores])
        for k in ("reader_end", "writer_end", "compute_end")
    }
    gave = [
        {"core": c["core"], "reader": c["reader"], "writer": c["writer"]}
        for c in states[:cores]
        if c["reader"] or c["writer"] or c["compute"]
    ]
    out = {
        "kernel": run.result()["error_name"],
        "records": int(len(us)),
        "cores_ended": ended,
        "compute_wrong": sum(sum(c["compute_wrong"]) for c in states[:cores]),
        "bad_items": [{"core": int(c), "where": int(word[c] >> 8 & 0xFF)} for c in bad],
        "gave_up": gave,
    }
    out["ok"] = bool(
        len(us) == reps
        and all(v == cores for v in ended.values())
        and out["compute_wrong"] == 0
        and not gave
        and not out["bad_items"]
    )
    return out
