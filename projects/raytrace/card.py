"""Running the ray tracer's kernels on a card through the host package: one
`komorebi.Session`, a program load per handler run, records to every core under the
dispatcher; DRAM tensors in, hit pages and per-core profiles out.
"""

import os
import threading
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import session as km_session
from projects.raytrace import packets

K = packets.K
KERNELS = packets.HERE / "kernels"
KERNEL_BYTES = 80 << 10  # kernel buffer: the default 70656 B is too small since SORT
T0 = time.monotonic()


def log(msg):
    """One progress line with seconds since start, flushed (the watcher reads it)."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def aiclk_mhz():
    """AICLK of the card this process uses, from the driver (MHz)."""
    return km_session.aiclk_mhz()


def hwmon(card):
    """The hwmon folder (name `blackhole`) of a card, found through its device."""
    dev = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/device").resolve()
    for h in Path("/sys/class/hwmon").glob("hwmon*"):
        name = (h / "name").read_text().strip() if (h / "name").exists() else ""
        if name == "blackhole" and (h / "device").resolve() == dev:
            return h
    return None


class Power:
    """Samples the card's power1_input (watts) in a thread while running."""

    def __init__(self, period=0.005):
        card = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
        self.dir = hwmon(card)
        self.period = period
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def read(self):
        return int((self.dir / "power1_input").read_text()) / 1e6

    def limit(self):
        if self.dir is None:
            return None
        return int((self.dir / "power1_max").read_text()) / 1e6

    def idle(self, seconds=1.0):
        """Mean power over `seconds` (call before the job)."""
        if self.dir is None:
            return None
        vals = []
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            vals.append(self.read())
            time.sleep(self.period)
        return float(np.mean(vals))

    def __enter__(self):
        if self.dir is None:
            return self

        def loop():
            while not self._stop.is_set():
                self.samples.append((time.perf_counter(), self.read()))
                time.sleep(self.period)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def tensor(session, rows):
    """A row-major uint32 array as a DRAM tensor (one page per row)."""
    return session.tensor(np.asarray(rows, dtype=np.uint32), "uint32")


def read(session, *tensors):
    """Tensors written by `tensor`, read back as uint32 rows."""
    return [
        pages.view(np.uint32).reshape(len(pages), -1)
        for pages in session.read(*tensors)
    ]


def open_session():
    """The session every driver runs in (`with card.open_session() as session`)."""
    return km.Session(log=log, kernel_bytes=KERNEL_BYTES)


def workers(session):
    """Cores of the grid: every one is a worker under the dispatcher."""
    x, y = session.nocs().grid
    return x * y


def program(tree_pages=1, sizes=None):
    """The ray tracer's program; sizes: bytes of the treelet buffers (state, lists,
    slots), small when absent. One record at a time: its event time is its own."""
    sizes = sizes or {}
    pages = [
        (K["kCbIn"], K["kInPageBytes"], K["kInPages"]),
        (K["kCbTree"], K["kTreePageBytes"], max(tree_pages, 1)),
        (K["kCbDirReader"], K["kDirReaderBytes"], 1),
        (K["kCbDirWriter"], K["kDirBytes"], 1),
        (K["kCbOut"], K["kOutPageBytes"], K["kOutPages"]),
        (K["kCbWork"], K["kWorkBytes"], K["kWorkPages"]),
        (K["kCbSignal"], K["kSignalBytes"], 1),
        (
            K["kCbState"],
            K["kStTileBytes"],
            sizes.get("state", 0) // K["kStTileBytes"] or 1,
        ),
        (K["kCbLists"], sizes.get("lists", 64), 1),
        (K["kCbSlots"], sizes.get("slots", 64), 1),
    ]
    # Packet states are unpacked into Dst and packed back as float32 tiles (Dst held
    # whole); the input and output pages share the packer's and unpacker's format.
    tiles = (K["kCbIn"], K["kCbOut"], K["kCbState"])
    return km.Program(
        "raytrace",
        kernels=KERNELS,
        buffers=[
            km.Buffer(i, "float32" if i in tiles else "uint32", page, n)
            for i, page, n in pages
        ],
        compute={
            "math_fidelity": "HiFi4",
            "fp32_dest_acc_en": True,
            "enable_trisc2_rvv": True,
            "dst_full_sync_en": True,
            "unpack_to_dest_mode": {K["kCbState"]: "UnpackToDestFp32"},
        },
        handlers=packets.HEADER,
        depth=1,
    )


class Ran:
    """What a handler run left: each record's event, the kernel's result block and
    every core's state words (read after the stop)."""

    def __init__(self, events, result, state, host_s):
        self.events = events
        self.result = result
        self.state = state
        self.host_s = host_s


def check_run(res, n_workers):
    """Events, kernel result and kernel ends as a dict; raises nothing."""
    states = res.state[:n_workers]
    return {
        "done": sum(ev.kind == "done" for ev in res.events),
        "commands": len(res.events),
        "kernel_error": res.result["error"],
        "cores_ended": {
            k: sum(c[k] != 0 for c in states)
            for k in ("reader_end", "writer_end", "compute_end")
        },
        "compute_wrong": sum(sum(c["compute_wrong"]) for c in states),
        "gave_up": [
            {"core": c["core"], "reader": c["reader"], "writer": c["writer"]}
            for c in states
            if c["reader"] or c["writer"]
        ],
    }


def ran_cleanly(run, n_workers):
    """Every record done, the kernel's error none, every kernel ended, nothing gave
    up (a check_run dict)."""
    ended = set(run["cores_ended"].values())
    return (
        run["done"] == run["commands"]
        and run["kernel_error"] == 0
        and ended == {n_workers}
        and run["compute_wrong"] == 0
        and not run["gave_up"]
    )


def command_us(res):
    """Submit to completion of each record, in microseconds (the X280's timer)."""
    return np.array([ev.us for ev in res.events], dtype=np.float64)


def run_handler(
    session,
    handler,
    directory,
    inputs,
    out_rows,
    out_words,
    tree=None,
    args=(),
    reps=1,
    deadline_s=5.0,
    power=None,
    extra=(),
    sizes=None,
):
    """Runs `reps` rounds of a handler (after a LOAD of `tree`) over a directory
    (workers, 16) or a list of them (a record each per round) and input pages, into
    out_rows pages of out_words; args[0] is a3, extra tensors and args[1:] follow a4.
    Returns (outputs, profiles summed over directories, Ran, times, power)."""
    n = workers(session)
    chunks = directory if isinstance(directory, list) else [directory]
    for d in chunks:
        if len(d) != n:
            raise ValueError(f"{len(d)} directory entries for {n} workers")
    times = {}
    t0 = time.perf_counter()
    dir_ts = [tensor(session, d) for d in chunks]
    in_t = tensor(session, inputs if inputs is not None else np.zeros((1, 32)))
    out_t = tensor(session, np.zeros((max(out_rows, 1), out_words), dtype=np.uint32))
    prof_rows = np.zeros((n, K["kOutPageBytes"] // 4), dtype=np.uint32)
    prof_ts = [tensor(session, prof_rows) for _ in chunks]
    extra_t = [tensor(session, rows) for rows in extra]
    commands = []
    tree_pages = 0
    tree_t = None
    if tree is not None:
        # One push of the whole image into a buffer of exactly its pages.
        if tree.ndim != 2 or tree.shape[1] * 4 != K["kTreePageBytes"]:
            raise ValueError(f"tree image {tree.shape} is not whole tree pages")
        tree_t = tensor(session, tree)
        tree_pages = len(tree)
        commands.append((K["kHandlerLoad"], [tree_t.address, tree.nbytes]))
    times["upload_s"] = time.perf_counter() - t0
    a3 = list(args)[:1] or [0]
    tail = [t.address for t in extra_t] + list(args)[1:]
    for _ in range(reps):
        for dir_t, prof_t in zip(dir_ts, prof_ts):
            words = [dir_t.address, in_t.address]
            words += [out_t.address, *a3, prof_t.address, *tail]
            commands.append((handler, words))
    log(f"run: handler {handler}, {len(commands)} commands, {out_rows} output pages")
    meter = power if power is not None else Power()
    with meter:
        run = session.load(program(tree_pages, sizes), deadline=deadline_s)
        mark = getattr(meter, "mark", None)
        t0 = time.perf_counter()
        if mark is not None:
            mark("program")
        handles = [run.submit(s, args=w) for s, w in commands]
        events = run.wait(handles, timeout=10.0 + deadline_s * len(commands))
        host_s = time.perf_counter() - t0
        if mark is not None:
            mark("after")
        run.stop()
    times["program_s"] = host_s
    res = Ran(
        events if isinstance(events, list) else [events],
        run.result(),
        run.state(),
        host_s,
    )
    log(f"ran: {len(res.events)} commands, error {res.result['error']}; reading back")
    t0 = time.perf_counter()
    out = read_rows(session, out_t, "hits")
    profs = read(session, *prof_ts)
    prof = sum(p[:n].astype(np.uint64) for p in profs)  # the last round's
    times["readback_s"] = time.perf_counter() - t0
    log(f"read back in {times['readback_s']:.1f}s")
    return out[:out_rows], prof, res, times, meter.samples


READ_LIMIT = 4 << 20  # bytes in one read-back call; larger tensors go in chunks
CHUNK = 2 << 20  # bytes per chunk


def read_rows(session, tensor, name):
    """A tensor read back as uint32 rows: in one call up to READ_LIMIT, else in chunks
    of CHUNK with a progress line per chunk."""
    size = tensor.pages * tensor.page_bytes
    t0 = time.perf_counter()
    if size <= READ_LIMIT:
        (rows,) = read(session, tensor)
    else:
        parts = []
        for first, pages in session.read_chunks(tensor, CHUNK):
            parts.append(pages.view(np.uint32).reshape(len(pages), -1))
            log(
                f"read back {name}: pages {first} + {len(pages)} of {size / 1e6:.1f} MB"
            )
        rows = np.concatenate(parts)
    log(f"read back {name}: {size / 1e6:.2f} MB in {time.perf_counter() - t0:.1f}s")
    return rows


def read_each(session, tensors, names):
    """Tensors read back one at a time (read_rows)."""
    return [read_rows(session, t, name) for t, name in zip(tensors, names)]


def run_commands(session, commands, tree_pages, deadline_s=5.0, power=None, sizes=None):
    """One program load: the records (handler, words) submitted in order, one at a time,
    each waited for; returns the Ran and the host seconds from submit to the last."""
    meter = power if power is not None else Power()
    with meter:
        run = session.load(program(tree_pages, sizes), deadline=deadline_s)
        t0 = time.perf_counter()
        handles = [run.submit(handler, args=words) for handler, words in commands]
        events = run.wait(handles, timeout=10.0 + deadline_s * len(commands))
        host_s = time.perf_counter() - t0
        run.stop()
    events = events if isinstance(events, list) else [events]
    ran = Ran(events, run.result(), run.state(), host_s)
    # The host's side of the records: one 64-byte command and one event each.
    ran.host = {
        "commands": len(commands),
        "submit_s": run.submit_s,
        "wait_s": run.wait_s,
    }
    ran.host.update(native_s=run.native_s, command_bytes=64 * len(commands))
    return ran, meter.samples


PROF = (
    "wait_in",
    "load",
    "pop",
    "box",
    "tri",
    "drain",
    "out",
    "wait_out",
    "packets",
    "pops",
    "empty",
    "inner",
    "leaves",
    "error",
    "total",
    "emits",
    "visits",
    "items",
    "wait_work",
    "st_load",
    "st_store",
    "phase1",
    "phase2",
    "phase3",
    "batches",
    "deferred",
)
READER = (
    "wait",
    "read",
    "total",
    "pages",
    "bin",
    "fetch",
    "slot_wait",
    "work_wait",
    "signal",
    "treelet_reads",
    "treelet_pages",
    "slot_hits",
    "items",
    "visits",
    "rounds",
    "groups",
)


def profile_rows(prof, n_workers):
    """Per core: the pack thread's, reader's and writer's counters (dict of arrays)."""
    p = prof[:n_workers].astype(np.int64)
    out = {name: p[:, k] for k, name in enumerate(PROF)}
    for k, name in enumerate(READER):
        out[f"reader_{name}"] = p[:, K["kProfReader"] + k]
    for k, name in enumerate(("wait", "write", "total", "pages")):
        out[f"writer_{name}"] = p[:, K["kProfWriter"] + k]
    return out
