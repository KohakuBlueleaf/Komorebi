"""Running the ray tracer's kernels on a card through the host package: one
`komorebi.Session`, a program load per handler run, commands to every core under the
dispatcher; DRAM tensors in, hit pages and per-core profiles out.
"""

import os
import threading
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import chain as kc
from komorebi.bench import power
from projects.raytrace import packets

K = packets.K
KERNELS = packets.HERE / "kernels"
T0 = time.monotonic()


def log(message):
    """One progress line with seconds since start, flushed (the watcher reads it)."""
    print(f"[{time.monotonic() - T0:7.2f}] {message}", flush=True)


def aiclk_mhz():
    """AICLK of the card this process uses, from the driver (MHz)."""
    return power.aiclk_mhz()


def hwmon(card):
    """The hwmon folder (name `blackhole`) of a card, found through its device."""
    device = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/device").resolve()
    for folder in Path("/sys/class/hwmon").glob("hwmon*"):
        name = (
            (folder / "name").read_text().strip() if (folder / "name").exists() else ""
        )
        if name == "blackhole" and (folder / "device").resolve() == device:
            return folder
    return None


class Power:
    """Samples the card's power1_input (watts) in a thread while running."""

    def __init__(self, period=0.005):
        card = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
        self.folder = hwmon(card)
        self.period = period
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def read(self):
        return int((self.folder / "power1_input").read_text()) / 1e6

    def limit(self):
        if self.folder is None:
            return None
        return int((self.folder / "power1_max").read_text()) / 1e6

    def idle(self, seconds=1.0):
        """Mean power over `seconds` (call before the job)."""
        if self.folder is None:
            return None
        readings = []
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            readings.append(self.read())
            time.sleep(self.period)
        return float(np.mean(readings))

    def __enter__(self):
        if self.folder is None:
            return self

        def loop():
            while not self._stop.is_set():
                self.samples.append((time.perf_counter(), self.read()))
                time.sleep(self.period)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exception):
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
    return km.Session(log=log)


def workers(session):
    """Cores of the grid: every one is a worker under the dispatcher."""
    x, y = session.nocs().grid
    return x * y


# The accessor arguments of a DRAM tensor of interleaved pages, as Session.tensor makes
# them (its configuration word, then its page bytes).
DRAM_ARGS = 2


def compile_args():
    """The reader's and the writer's km::Tensor arguments (from 0): the tree's and the
    treelets' pages, the directory's; the directory's, the profile pages'."""
    tree = [DRAM_ARGS, K["kTreePageBytes"]]
    directory = [DRAM_ARGS, K["kDirBytes"]]
    profile = [DRAM_ARGS, K["kOutPageBytes"]]
    return {"reader": tree + directory, "writer": directory + profile}


CHAIN_PAGE = 2816  # a chain's words on each core (km_chain.hpp's kChainAreaBytes)


def program(tree_pages=1, sizes=None):
    """The ray tracer's program; sizes: bytes of the treelet buffers (state, lists,
    slots), small when absent. One command at a time: its event time is its own."""
    sizes = sizes or {}
    specs = [
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
        (kc.words()["kChainBuffer"], CHAIN_PAGE, 1),
    ]
    # Packet states are unpacked into Dst and packed back as float32 tiles (Dst held
    # whole); the input and output pages share the packer's and unpacker's format.
    tiles = (K["kCbIn"], K["kCbOut"], K["kCbState"])
    return km.Program(
        "raytrace",
        kernels=KERNELS,
        buffers=[
            km.Buffer(
                index, "float32" if index in tiles else "uint32", page_bytes, count
            )
            for index, page_bytes, count in specs
        ],
        compile_args=compile_args(),
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
    """What a handler run left: each command's event, the kernel's result block and
    every core's state words (read after the stop)."""

    def __init__(self, events, result, state, host_s):
        self.events = events
        self.result = result
        self.state = state
        self.host_s = host_s


def check_run(ran, worker_count):
    """Events, kernel result and kernel ends as a dict; raises nothing."""
    states = ran.state[:worker_count]
    return {
        "done": sum(event.kind == "done" for event in ran.events),
        "commands": len(ran.events),
        "kernel_error": ran.result["error"],
        "cores_ended": {
            field: sum(core_state[field] != 0 for core_state in states)
            for field in ("reader_end", "writer_end", "compute_end")
        },
        "compute_wrong": sum(sum(core_state["compute_wrong"]) for core_state in states),
        "gave_up": [
            {
                "core": core_state["core"],
                "reader": core_state["reader"],
                "writer": core_state["writer"],
            }
            for core_state in states
            if core_state["reader"] or core_state["writer"]
        ],
    }


def ran_cleanly(run, worker_count):
    """Every command done, the kernel's error none, every kernel ended, nothing gave
    up (a check_run dict)."""
    ended = set(run["cores_ended"].values())
    return (
        run["done"] == run["commands"]
        and run["kernel_error"] == 0
        and ended == {worker_count}
        and run["compute_wrong"] == 0
        and not run["gave_up"]
    )


def command_microseconds(ran):
    """Submit to completion of each command, in microseconds (the X280's timer)."""
    return np.array([event.microseconds for event in ran.events], dtype=np.float64)


def run_handler(
    session,
    handler,
    directory,
    inputs,
    output_rows,
    output_words,
    tree=None,
    args=(),
    repetitions=1,
    deadline_s=5.0,
    power=None,
    extra=(),
    sizes=None,
):
    """Runs `repetitions` rounds of a handler (after a LOAD of `tree`) over a directory
    (workers, 16) or a list of them (a command each per round) and input pages, into
    `output_rows` pages of `output_words`; args[0] is a3, extra tensors and args[1:]
    follow a4. Returns (outputs, profile sums per directory, Ran, times, power)."""
    worker_count = workers(session)
    chunks = directory if isinstance(directory, list) else [directory]
    for entry in chunks:
        if len(entry) != worker_count:
            raise ValueError(
                f"{len(entry)} directory entries for {worker_count} workers"
            )
    times = {}
    started = time.perf_counter()
    directory_tensors = [tensor(session, entry) for entry in chunks]
    input_tensor = tensor(session, inputs if inputs is not None else np.zeros((1, 32)))
    output_tensor = tensor(
        session, np.zeros((max(output_rows, 1), output_words), dtype=np.uint32)
    )
    profile_zeros = np.zeros((worker_count, K["kOutPageBytes"] // 4), dtype=np.uint32)
    profile_tensors = [tensor(session, profile_zeros) for _ in chunks]
    extra_tensors = [tensor(session, rows) for rows in extra]
    commands = []
    tree_pages = 0
    tree_tensor = None
    if tree is not None:
        # One push of the whole image into a buffer of exactly its pages.
        if tree.ndim != 2 or tree.shape[1] * 4 != K["kTreePageBytes"]:
            raise ValueError(f"tree image {tree.shape} is not whole tree pages")
        tree_tensor = tensor(session, tree)
        tree_pages = len(tree)
        commands.append((K["kHandlerLoad"], [tree_tensor.address, tree.nbytes]))
    times["upload_s"] = time.perf_counter() - started
    third_word = list(args)[:1] or [0]
    tail = [extra_tensor.address for extra_tensor in extra_tensors] + list(args)[1:]
    for _ in range(repetitions):
        for directory_tensor, profile_tensor in zip(directory_tensors, profile_tensors):
            words = [directory_tensor.address, input_tensor.address]
            words += [output_tensor.address, *third_word, profile_tensor.address, *tail]
            commands.append((handler, words))
    log(f"run: handler {handler}, {len(commands)} commands, {output_rows} output pages")
    meter = power if power is not None else Power()
    with meter:
        run = session.load(program(tree_pages, sizes), deadline=deadline_s)
        mark = getattr(meter, "mark", None)
        started = time.perf_counter()
        if mark is not None:
            mark("program")
        handles = [
            run.submit(command_handler, args=command_words)
            for command_handler, command_words in commands
        ]
        events = run.wait(handles, timeout=10.0 + deadline_s * len(commands))
        host_s = time.perf_counter() - started
        if mark is not None:
            mark("after")
        run.stop()
    times["program_s"] = host_s
    ran = Ran(
        events if isinstance(events, list) else [events],
        run.result(),
        run.state(),
        host_s,
    )
    log(f"ran: {len(ran.events)} commands, error {ran.result['error']}; reading back")
    started = time.perf_counter()
    output = read_rows(session, output_tensor, "hits")
    profile_pages = read(session, *profile_tensors)
    profile = sum(
        page[:worker_count].astype(np.uint64) for page in profile_pages
    )  # the last round's
    times["readback_s"] = time.perf_counter() - started
    log(f"read back in {times['readback_s']:.1f}s")
    return output[:output_rows], profile, ran, times, meter.samples


READ_LIMIT = 4 << 20  # bytes in one read-back call; larger tensors go in chunks
CHUNK = 2 << 20  # bytes per chunk


def read_rows(session, tensor, name):
    """A tensor read back as uint32 rows: in one call up to READ_LIMIT, else in chunks
    of CHUNK with a progress line per chunk."""
    size = tensor.pages * tensor.page_bytes
    started = time.perf_counter()
    if size <= READ_LIMIT:
        (rows,) = read(session, tensor)
    else:
        pieces = []
        for first, pages in session.read_chunks(tensor, CHUNK):
            pieces.append(pages.view(np.uint32).reshape(len(pages), -1))
            log(
                f"read back {name}: pages {first} + {len(pages)} of {size / 1e6:.1f} MB"
            )
        rows = np.concatenate(pieces)
    log(
        f"read back {name}: {size / 1e6:.2f} MB in {time.perf_counter() - started:.1f}s"
    )
    return rows


def read_each(session, tensors, names):
    """Tensors read back one at a time (read_rows)."""
    return [read_rows(session, entry, name) for entry, name in zip(tensors, names)]


def frame_chain(run, frame, cores):
    """The frame's commands as one chain: each record on every core, one core a group,
    each core's record marked by its own block of the one before (rays stay there)."""
    own = kc.Range(0, 1, 1)
    calls = [
        kc.Call(
            handler,
            tuple(words),
            (0, cores),
            1,
            1,
            output=f"record {index}",
            input=f"record {index - 1}" if index else None,
            ranges=(own,) if index else (),
            name=f"record {index}",
        )
        for index, (handler, words) in enumerate(frame)
    ]
    return run.chain(calls, K["kChainMode"])


def run_chain(
    session,
    load,
    frame,
    reps,
    tree_pages,
    cores,
    deadline_s=5.0,
    power=None,
    sizes=None,
):
    """One program load: `load` (LOAD) as a command, then the frame's commands as one
    chain submitted `reps` times, each waited for; the Ran's events: LOAD's, then one
    a frame."""
    meter = power if power is not None else Power()
    with meter:
        run = session.load(program(tree_pages, sizes), deadline=deadline_s)
        started = time.perf_counter()
        events = [run.wait(run.submit(load[0], args=load[1]), timeout=10.0)]
        chain = frame_chain(run, frame, cores)
        events.extend(
            run.wait(run.submit(chain), timeout=10.0 + deadline_s) for _ in range(reps)
        )
        host_s = time.perf_counter() - started
        run.stop()
    ran = Ran(events, run.result(), run.state(), host_s)
    ran.host = {"commands": 1 + reps, "chain_records": len(frame)}
    return ran, meter.samples


def run_commands(session, commands, tree_pages, deadline_s=5.0, power=None, sizes=None):
    """One program load: the commands (handler, words) submitted in order, one at a
    time, each waited for; returns the Ran and the host seconds from submit to the
    last."""
    meter = power if power is not None else Power()
    with meter:
        run = session.load(program(tree_pages, sizes), deadline=deadline_s)
        started = time.perf_counter()
        handles = [run.submit(handler, args=words) for handler, words in commands]
        events = run.wait(handles, timeout=10.0 + deadline_s * len(commands))
        host_s = time.perf_counter() - started
        run.stop()
    events = events if isinstance(events, list) else [events]
    ran = Ran(events, run.result(), run.state(), host_s)
    # The host's side of the commands: one 64-byte command and one event each.
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


def profile_rows(profile, worker_count):
    """Per core: the pack thread's, reader's and writer's counters (dict of arrays)."""
    rows = profile[:worker_count].astype(np.int64)
    result = {name: rows[:, word] for word, name in enumerate(PROF)}
    for word, name in enumerate(READER):
        result[f"reader_{name}"] = rows[:, K["kProfReader"] + word]
    for word, name in enumerate(("wait", "write", "total", "pages")):
        result[f"writer_{name}"] = rows[:, K["kProfWriter"] + word]
    return result
