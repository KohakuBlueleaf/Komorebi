"""Running the host-binned rasterizer on a card through the host package: the lists,
directory, frame and profile tensors are written before the load, then FILL commands go
to every core under the dispatcher, one in flight at a time."""

import time

import numpy as np

import komorebi as km
from projects.raster import lists

K = lists.K
KERNELS = lists.HERE / "kernels"
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def program():
    """The rasterizer's three kernels on every core; one command in flight per core, so
    a command's event time is its own, submit to the last core's acknowledgement."""
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
        handlers=lists.RASTER_H,
    )


def workers(session):
    """The cores of the open card's grid; the core numbered `index` is at
    (index % x, index // x)."""
    x, y = session.grid
    return x * y


def tensor(session, rows):
    """A row-major uint32 array as a DRAM tensor (one page per row)."""
    return session.tensor(np.asarray(rows, dtype=np.uint32), "uint32")


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


# Frame pages per tensor: at most 2 MB, a whole number of 3- and 5-page items.
CHUNK_PAGES = 510


def upload(session, pages, directory, output_pages):
    """The DRAM tensors of a run: lists, directory (naming the frame's tensors), the
    frame in tensors of CHUNK_PAGES pages, and the profile pages."""
    chunks = -(-output_pages // CHUNK_PAGES)
    if chunks > K["kDirFrameMax"]:
        raise ValueError(f"a frame of {output_pages} pages needs {chunks} tensors")
    frame = [
        tensor(
            session,
            np.zeros((min(CHUNK_PAGES, output_pages - chunk * CHUNK_PAGES), 1024)),
        )
        for chunk in range(chunks)
    ]
    directory = np.array(directory, dtype=np.uint32)
    frame_column = K["kDirFrame"]
    directory[:, frame_column] = CHUNK_PAGES
    directory[:, frame_column + 1 : frame_column + 1 + chunks] = [
        chunk_tensor.address for chunk_tensor in frame
    ]
    return {
        "lists": tensor(session, pages),
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
):
    """Loads the program, runs `reps` commands of `handler` over the lists and directory
    (`args` in place of the tensors' addresses), stops, reads back. Returns (output
    pages or None, profile pages, status, command microseconds, host seconds)."""
    if K["kOutPages"] % pages_per_item:
        raise ValueError(f"{pages_per_item}-page items do not divide the output buffer")
    if len(directory) != workers(session):
        raise ValueError(f"{len(directory)} directory pages for {workers(session)}")
    times = {}
    started = time.perf_counter()
    uploaded = upload(session, pages, directory, output_pages)
    times["upload_s"] = time.perf_counter() - started
    started = time.perf_counter()
    run = session.load(program(), deadline=1.0)
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
        handles = run.submit_many(handler, 0, run.cores, np.tile(args, (commands, 1)))
        events = run.wait(list(handles), timeout=10.0 + commands * 0.01)
        microseconds += [event.microseconds for event in events]
        if any(event.kind != "done" for event in events):
            break
    return np.array(microseconds)


def check_run(run, cores, repetitions, microseconds):
    """The run's end as a dict, `ok` when every core ended cleanly and every command
    finished; raises nothing."""
    states = run.state()
    writer_column = km.abi.headers()["kStWriter"]
    writer_words = np.asarray(run.state_words())[:cores, writer_column].astype(np.int64)
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
    result = {
        "kernel": run.result()["error_name"],
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
