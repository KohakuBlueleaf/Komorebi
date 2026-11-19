"""The dispatcher on one Tensix core, the grid's last: its kernel in the load's launch,
its block in L1, its area of the host channel past the session's region, and the
native port (TensixPort) that writes its host ring and reads its events."""

import functools
import hashlib
import random
import re
import struct
import time
from pathlib import Path

import numpy as np

from komorebi import _env, abi
from komorebi.device import chips, mock, native
from komorebi.errors import NotServed
from komorebi.program import thread as thread_module
from komorebi.session.dispatch import TENSIX

OUT, IN = "dispatcher.cpp", "td_in.cpp"  # its loops: out takes and submits, in sweeps
COMPUTE = "td_compute.cpp"  # its core's compute kernel by TRISC: engines, thread, watch
THREAD = "td_thread.hpp"  # its TRISC 1 part: a thread's sources are included after it
# A thread's build that failed on what TRISC 1 lacks: a call td_thread.hpp refuses, or
# statics over its local memory (tt-metal's check of the ELF's local segment).
THREAD_CALL = re.compile(
    r"([^\s:]+):\d+:\d+: error: call to '(kmx_\w+)' declared with attribute error: "
    r"not on the dispatcher's core: ([^\n]+)"
)
VECTOR = re.compile(r"^\s*#\s*include\s*[<\"]riscv_vector\.h[>\"]", re.M)
THREAD_LOCAL = re.compile(
    r"thread-\w+/\S*?/trisc1/trisc1\.elf: segment\[\d+\] "
    r"\[0x[0-9a-f]+,\+0x([0-9a-f]+)\) overflows region:\d+ limit of 0x([0-9a-f]+) bytes"
)
# The processors (0 BRISC, 1 NCRISC) of out and in by split: "A" one loop on BRISC,
# "B" out on BRISC and in on NCRISC, "B'" the other way round.
LOOPS = {"A": (0, None), "B": (0, 1), "B'": (1, 0)}
SPLIT = "B"
DEFINES = {}  # the kernels' defines of a bench (KM_TD_SELF_FEED): none in a product run
# The result's words past the X280's: the ring bound's changes, in's passes and sweeps.
TAIL = {
    "bound_writes": "kTrBoundWrites",
    "bound_high": "kTrBoundHigh",
    "bound_low": "kTrBoundLow",
    "in_passes": "kTrInPasses",
    "in_sweeps": "kTrInSweeps",
}
# The dynamic run's words of the X280's core block, from the result's.
RUN_WORDS = {
    "KMD_H_MESSAGES_GAVE": "kTrRunGave",
    "KMD_H_MESSAGES_CHECKS": "kTrRunChecks",
    "KMD_H_MESSAGES_QUIET": "kTrRunQuietAt",
}
CLOSE_WAIT_S = 1.0  # a close waits this long for the kernel's end after its stop flag
POLL_S = 0.0005  # between two looks at the host words while waiting for the end


def directory():
    """The dispatcher kernel's sources: its two loops and their headers."""
    return native.native_directory() / "kernels" / "dispatcher"


@functools.cache
def constants():
    """dispatcher.h's constants (kTd*, kTb*, kTh*, kTr*)."""
    return abi.constants(directory() / "dispatcher.h")


# What this dispatcher does not serve yet, one row an ability: the slice that serves one
# deletes its row, and every call that asks for it (refuse) then passes.
REFUSED = {
    "threads": "X280 threads",
    "split": "a card split into islands",
}


def refused(ability):
    """NotServed of the Tensix dispatcher for `ability`."""
    return NotServed(TENSIX, ability)


def refusal(name, where=None):
    """NotServed for REFUSED's row `name`, asked by `where`; None once it is served."""
    if name not in REFUSED:
        return None
    return refused(REFUSED[name] + (f" ({where})" if where else ""))


def refuse(name, where=None):
    """Raises refusal(name, where) unless the dispatcher serves it."""
    error = refusal(name, where)
    if error is not None:
        raise error


def refuse_x280_image(tree, where):
    """Raises the refusal of X280 threads when `tree` has no X280 runtime: an image
    for the X280 cannot be built there, and a Tensix dispatcher serves no thread."""
    if not _env.has_x280_runtime(tree):
        refuse("threads", where)


def rectangles(nocs, loopback=False):
    """[(NoC 0 start, end, receivers)] of the command line's multicast: one rectangle a
    panel over every row, the dispatcher's panel counting it out (the sender) unless
    sent with `loopback` (chips.multicast_loopback)."""
    grid_x, grid_y = nocs.grid
    out = []
    for x0, x1 in nocs.panels:
        a, b = nocs.noc_xy[(x0, 0)], nocs.noc_xy[(x1, grid_y - 1)]
        cores = (x1 - x0 + 1) * grid_y
        holds = x0 <= grid_x - 1 <= x1
        out.append(
            (a[0] | a[1] << 16, b[0] | b[1] << 16, cores - int(holds and not loopback))
        )
    if len(out) > constants()["kTdRectsMax"]:
        raise refused(
            f"a grid of {len(out)} panels (at most {constants()['kTdRectsMax']})"
        )
    return out


def thread_refusal(owner, trisc=True):
    """Why the threads of `owner` cannot run, or None: one thread, on cluster 1, with no
    ports or shared regions; on TRISC 1 (`trisc`) no vector unit."""
    threads = owner.threads
    if owner.x280_ports is not None:
        return "X280 ports"
    if not threads:
        return None
    if len(threads) > 1 or 1 not in threads:
        return f"threads on clusters {sorted(threads)} (one, cluster 1, on its core)"
    thread = threads[1]
    if thread.shared:
        return "a thread's shared regions"
    for path in thread.source_paths() if trisc else ():  # TRISC 1 has no vector unit
        if VECTOR.search(path.read_text(errors="replace")):
            return f"the vector extension (riscv_vector.h in {path.name})"
    return None


def thread_kernel(thread, programs=()):
    """The dispatcher core's compute kernel with `thread` (komorebi.program.thread's
    Thread) on TRISC 1: its defines, td_thread.hpp, its sources, then td_compute.cpp;
    written once."""
    lines = [f"// The thread of {', '.join(p.name for p in thread.source_paths())}."]
    for name, value in sorted(thread.all_defines(programs).items()):
        lines.append(f"#define {name} {value}")
    lines.append("#define KM_TD_THREAD 1")
    lines.append(f'#include "{directory() / THREAD}"')
    lines += [f'#include "{path.resolve()}"' for path in thread.source_paths()]
    lines.append(f'#include "{directory() / COMPUTE}"')  # the core's one compute kernel
    text = "\n".join(lines) + "\n"
    key = hashlib.sha256(text.encode()).hexdigest()[:16]
    out = native.cache_directory() / "td-thread" / f"thread-{key}.cpp"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_suffix(f".{random.getrandbits(32):08x}")
        part.write_text(text)
        part.replace(out)
    return out


def thread_build_refusal(text):
    """NotServed for a thread's build that failed (its log `text`) on what TRISC 1 of
    the dispatcher's core lacks, naming the call or both sizes; else None."""
    call = THREAD_CALL.search(text)
    if call is not None:
        return refused(
            f"{call[2]} in {Path(call[1]).name}, a thread on its core ({call[3]})"
        )
    local = THREAD_LOCAL.search(text)
    if local is not None:
        wanted, most = int(local[1], 16), int(local[2], 16)
        return refused(
            f"a thread whose statics take {wanted:,} B of TRISC 1's local memory, "
            f"which holds {most:,} B for them"
        )
    return None


def refuse_load(program, placed, owner, options, trisc=True):
    """NotServed for a load of what this dispatcher does not serve (it serves plain
    commands, chains, overlays, add-ons, several programs, dynamic runs, an island of
    cards, one thread on its core): the first one it asks."""
    del placed, options  # an island's load: its island word (prepare)
    thread = thread_refusal(owner, trisc)
    if thread is not None:
        raise refused(f"{thread} (load of {program.name})")


def block_buffer(card):
    """(the block's L1 address, its buffer's bytes): the buffer is the first on the
    dispatcher's core, at the buffers' base (as tt-metal places it), the block at its
    first 64-byte boundary in it; the kernel checks the address (td::block_placed)."""
    _, free, top = card.module.kernel_buffer(card.device)
    at = -(-(top - free) // card.cb_align) * card.cb_align
    block = -(-at // 64) * 64
    return block, block - at + constants()["kTbBytes"]


def link_words(line):
    """The host's link line (link_area.line) as the kernel's link arguments: this
    island, the islands, link core 0, its link area, an inbox part's packets, 3 words an
    island (its ranks, threads, dispatcher tile) and the link block's bytes; None when
    the line is empty."""
    if not line:
        return None
    c, h = constants(), abi.headers()
    words = struct.unpack(f"<{len(line) // 4}I", line)
    head = h["KMD_LINK_BYTES"] // 4
    if words[h["KMD_L_MARK"]] != h["KMD_LINK_MARK"]:
        raise ValueError("a link line without its mark")
    islands = words[h["KMD_L_ISLANDS"]]
    if islands > c["kTdLinkIslandsMax"]:
        raise refused(
            f"commands across {islands} islands (at most {c['kTdLinkIslandsMax']})"
        )
    entry = h["KMD_LINK_ENTRY_BYTES"] // 4
    entries = [
        words[head + entry * island + k] for island in range(islands) for k in range(3)
    ]
    slots = words[h["KMD_L_SLOTS"]]
    part = h["KMD_LINK_COUNT_BYTES"] + slots * h["KM_LINK_PACKET_BYTES"]
    return {
        "island": words[h["KMD_L_ISLAND"]],
        "islands": islands,
        "core": words[h["KMD_L_CORE"]],
        "area": words[h["KMD_L_AREA"]],
        "slots": slots,
        "entries": entries,
        "bytes": c["kTlInbox"] + (islands - 1) * part,
    }


def held_words(programs):
    """The kernel's table of each core's program field (kTdHeld): core c in byte c % 4
    of word c / 4, 0xFF for a core past `programs` (none held)."""
    most = abi.headers()["KM_CORES_MAX"]
    held = [field & 0xFF for field in programs] + [0xFF] * (most - len(programs))
    return [
        sum(field << 8 * k for k, field in enumerate(held[at : at + 4]))
        for at in range(0, most, 4)
    ]


class TensixDispatch:
    """The session's port to the Tensix dispatcher (komorebi.session.dispatch.Dispatch):
    the host ring's calls go to the native TensixPort once the load made it."""

    kind = TENSIX  # its core: the grid's last, outside every group
    idle = True  # nothing of it runs before its load's launch
    upload_bytes = 0
    lim_base = 0
    tick_mhz = 50.0  # set by add_kernels: its chip's wall clock

    def __init__(self, session):
        self.session = session
        self.nocs = session.nocs()
        self.grid_x, self.grid_y = self.nocs.grid
        self.core = self.grid_x * self.grid_y - 1
        self.x, self.y = self.core % self.grid_x, self.core // self.grid_x
        self.card = None
        self.native = None  # the TensixPort, made by add_kernels on a card
        self.lists = None  # the kept reset lists' DRAM tensor (keep_lists)
        self.block = None  # its block's L1 address, on its core alone
        self.control = 0
        self.options = {}
        self.tag = 0
        self.close_failed = ""
        self._aborted = None
        self._batch = 64
        self.tail = {}  # the result's words past the X280's (TAIL), once read
        self.parts = []
        self.thread = None  # thread_kernel's source of the load's thread, if it has one
        self.thread_includes = []  # the thread's own include directories
        self.thread_area = None  # its area's L1 address (kTa*), on its core alone
        self.x280 = None  # an X280 thread's runtime (the native Dispatch, no kernel)
        self._lim = None  # a mock's thread area: bytes, as the work_* calls see it

    @property
    def l2_x(self):
        """Its core's NoC x (an island's link line names every dispatcher's tile)."""
        return int(self.nocs.words[self.core]) & 0xFFFF

    @property
    def l2_y(self):
        """Its core's NoC y."""
        return int(self.nocs.words[self.core]) >> 16

    # The host ring's calls (km::HostRing) -----------------------------------------

    @property
    def batch(self):
        return self.native.batch if self.native is not None else self._batch

    @batch.setter
    def batch(self, value):
        self._batch = value
        if self.native is not None:
            self.native.batch = value

    @property
    def settle_microseconds(self):
        return 0.0  # the device API's writes: no fence to keep apart

    @settle_microseconds.setter
    def settle_microseconds(self, value):
        pass

    @property
    def queued(self):
        return self.native.queued if self.native is not None else 0

    @property
    def is_open(self):
        return self.native is not None and self.native.is_open

    def enqueue(self, commands):
        return self.native.enqueue(commands)

    def wait(self, wanted, upto, timeout_s):
        return self.native.wait(wanted, upto, timeout_s)

    def wait_events(self, timeout_s):
        return self.native.wait_events(timeout_s)

    def poll(self):
        return self.native.poll()

    def publish(self):
        self.native.publish()

    def host_word(self, word):
        return self.native.host_word(word)

    def wait_ready(self, timeout_s):
        if self.native is None:
            return abi.headers()["KMD_STATE_READY"]
        return self.native.wait_ready(timeout_s)

    # The load ---------------------------------------------------------------------

    def core_table(self):
        """Core index -> translated NoC x | y << 16, KM_CORES_MAX words."""
        words = list(self.nocs.words)
        most = abi.headers()["KM_CORES_MAX"]
        return np.array(words + [0] * (most - len(words)), dtype=np.uint64)

    def prepare(
        self, depth, host_depth, event_depth, t_start_s, t_command_s, t_idle_s,
        state_ddr, **options,
    ):  # fmt: skip
        """The run's depths and limits (the X280's prepare's first arguments); the
        X280's other options name what this dispatcher does not serve. Returns a tag."""
        del state_ddr  # the X280 kernel's state: none here
        served = {
            "channel_region", "upload_bytes", "first_row", "rows", "programs",
            "messages_handler", "island", "link",
        }  # fmt: skip
        handlers = int(options.get("messages_handler") or 0)
        deliver = handlers >> 40 & 1  # messages to other islands as deliveries
        if handlers >> 32 & ~(1 << 8):  # X280 ranks, their counter tiles
            raise refused("a dynamic run's X280 ranks (messages_handler)")
        for name, value in options.items():
            if name not in served and value not in (0, None, [], b"", ()):
                raise refused(f"{name} (the X280 kernel's {name!r})")
        self.options = {
            "depth": depth,
            "host_depth": host_depth,
            "event_depth": event_depth,
            "start": t_start_s,
            "deadline": t_command_s,
            "idle": t_idle_s,
            "programs": list(options.get("programs") or []),  # each core's, at start
            "messages": handlers & abi.M32,  # the dynamic handlers, one a byte
            "deliver": deliver,
            # an island's flat ranks and first core (0: one island)
            "island": int(options.get("island") or 0),
            # its link line (link_area.line): commands across islands, td_link.hpp
            "link": link_words(bytes(options.get("link") or b"")),
        }
        self.tag = random.randrange(1, 0xFFFF, 2)
        return self.tag

    def start_words(self, control):
        """The cores' start-up words that name it: its coordinates; acks, READY words
        and core 0's hello into its own control block."""
        h = abi.headers()
        return {
            "dispatcher_coordinates": int(self.nocs.words[self.core]),
            "ack_base": control + h["KM_CONTROL_DONE"],
            "ready_base": control + h["KM_CONTROL_BOOT"],
            "mailbox": control + h["KM_CONTROL_JOB"],
        }

    def keep_lists(self, blob):
        """The reset lists `blob` (load(recover=True)) at the start of the session's
        channel region, for the dispatcher to copy into a DRAM tensor of pages of
        kTdStageBytes (no command of the cores): (its address, the channel offset)."""
        page = constants()["kTdStageBytes"]
        self.lists = self.session.empty((-(-len(blob) // page), page // 4), "uint32")
        channel = self.session.channel
        if channel is not None:  # the region is idle during a load
            channel[: len(blob)] = np.frombuffer(blob, np.uint8)
        return self.lists.address, 0

    def messages_lines(self, card):
        """Where a dynamic run's status lines land: in its block (kTbLines)."""
        return block_buffer(card)[0] + constants()["kTbLines"]

    def add_kernels(self, backend, launch, ranks, groups):
        """Its kernel on its core in `launch`, its block in L1, and on a card its port
        with its area of the host channel zeroed (before the launch)."""
        c, h, card = constants(), abi.headers(), backend.card
        self.card = card
        area = self.session._channel_region()  # its area: just past the region
        if not mock.active() and not area:
            raise refused("a session without its mapped host channel (channel=0)")
        here = [(self.x, self.y, self.x, self.y)]
        block, size = block_buffer(card)
        size += c["kTaBytes"] + c["kTeBytes"]  # the thread's area, then the engines'
        if self.thread is not None or self.x280 is not None:
            self.thread_area = block + c["kTbBytes"]
        link = self.options["link"]
        self.link_base = 0
        if link is not None:  # the link block after the engines' area
            self.link_base = block + c["kTbBytes"] + c["kTaBytes"] + c["kTeBytes"]
            size += link["bytes"]
        launch.cb(here, c["kTdBlockCb"], "uint32", size, size)  # this core's L1 alone
        self.block = block
        self.control = backend.control_addr
        options = self.options
        mhz = backend.mhz  # the wall clock's ticks a microsecond (chips.tick_mhz)
        self.tick_mhz = mhz  # its events' ticks a microsecond
        loopback = chips.chip(self.session.arch).multicast_loopback
        rects = rectangles(self.nocs, loopback)
        values = {
            "kTdControl": self.control,
            "kTdRanks": ranks,
            "kTdCores": backend.cores_tensor.address,
            "kTdBlock": block,
            "kTdHostDepth": options["host_depth"],
            "kTdEventDepth": options["event_depth"],
            "kTdArea": area,
            "kTdDepth": options["depth"],
            "kTdDeadline": abi.limit_units(options["deadline"], mhz),
            "kTdBootLimit": abi.limit_units(options["start"], mhz),
            "kTdIdleLimit": (
                abi.limit_units(options["idle"], mhz) if options["idle"] else 0
            ),
            "kTdGroups": groups,
            "kTdRects": len(rects),
            "kTdLoopback": int(loopback),
            "kTdTicksUs": round(mhz * 65536),
            "kTdPrograms": int(bool(options["programs"])),
            "kTdTag": self.tag,
            "kTdMessages": options["messages"],
            "kTdIsland": options["island"] & abi.M32,
            "kTdIslandBase": options["island"] >> 32,
        }
        if link is not None:
            values |= {
                "kTdLinkIsland": (
                    link["island"] | link["islands"] << 16 | options["deliver"] << 31
                ),
                "kTdLinkCore": link["core"],
                "kTdLinkArea": link["area"],
                "kTdLinkBase": self.link_base,
                "kTdLinkSlots": link["slots"],
            }
        if self.thread is not None or self.x280 is not None:
            values["kTdThreadScale"] = round(
                thread_module.TIMER_HZ * 2**32 / (mhz * 1e6)
            )
        if self.x280 is not None:  # in and out copy the area's words to the X280's
            info = self.x280.work_info(1)
            values["kTdThreadTile"] = info["virtual_x"] | info["virtual_y"] << 16
            values["kTdThreadWork"] = (
                info["lim_base"] + thread_module.layout()["KMX_LIM_BYTES"]
            )
        args = [0] * c["kTdArgs"]
        for name, value in values.items():
            args[c[name]] = int(value) & abi.M32
        for r, rect in enumerate(rects):
            args[c["kTdRect"] + 3 * r : c["kTdRect"] + 3 * r + 3] = rect
        words = held_words(options["programs"])
        args[c["kTdHeld"] : c["kTdHeld"] + len(words)] = words
        if link is not None:
            at = c["kTdLinkEntries"]
            args[at : at + len(link["entries"])] = link["entries"]
        includes = [
            str(h.abi.parents[2]),
            str(native.kernel_directory()),
            str(native.engines_directory()),  # the graph executor, on TRISC 0
        ]
        defines = [(name, str(value)) for name, value in DEFINES.items()]
        out, inside = LOOPS[SPLIT]
        if inside is None:
            defines.append(("KM_TD_ALONE", "1"))
        compute = directory() / COMPUTE  # with a thread, its kernel holding the thread
        if self.thread is not None:
            compute = self.thread
            includes = [
                *includes, str(directory()), str(native.kmx_directory()),
                *self.thread_includes,
            ]  # fmt: skip
        for source, processor in ((OUT, out), (IN, inside), (compute, 2)):
            if processor is not None:
                launch.kernel(
                    str(directory() / source), here, processor, [], defines, includes,
                    [(self.x, self.y, args)],
                )  # fmt: skip
        if mock.active():
            return
        module = card.module
        self.native = module.TensixPort(
            card.device, self.x, self.y, block + c["kTbRing"], block + c["kTbRead"],
            area, c["kThBytes"],
        )  # fmt: skip
        self.native.prepare(options["host_depth"], options["event_depth"])
        self.native.batch = self._batch

    def start(self, image):
        """Nothing to start: its kernel was launched with the grid's."""
        del image

    def upload(self, offset, data):
        """Writes `data` (whole 64-byte lines) at `offset` of the engines' upload region
        (kTeUpload of their area, on its core alone), as the X280's port does in LIM."""
        c = constants()
        if offset % 64 or len(data) % 64 or offset + len(data) > c["kTeUploadBytes"]:
            raise ValueError(
                f"an upload of {len(data)} bytes at {offset}: whole 64-byte lines "
                f"inside the region's {c['kTeUploadBytes']} bytes"
            )
        if self.native is None:  # a mock: nothing runs it
            return
        address = self.block + c["kTbBytes"] + c["kTaBytes"] + c["kTeUpload"] + offset
        words = list(struct.unpack(f"<{len(data) // 4}I", bytes(data)))
        self.card.module.write_l1(self.card.device, self.x, self.y, address, words)

    # The thread on its core: the X280's work_* calls of komorebi.program.thread's
    # ThreadRun, over its area (kTa*), at the X280's LIM offsets from kTaLim

    def _thread_cluster(self, cluster):
        if cluster != 1 or self.thread_area is None and not mock.active():
            raise refused(
                f"a thread on cluster {cluster} (one, cluster 1, on its core)"
            )

    # An X280 thread on cluster 1 (Blackhole): the runtime boots its cluster with no
    # X280 kernel; its ring is in this core's area, its events in the cluster's LIM.

    def thread_ring(self):
        """The X280 thread's RingPlacement: commands and the read line in this core's
        area (written over the NoC), consumed and events in the cluster's own LIM."""
        c, h = constants(), abi.headers()
        info = self.x280.work_info(1)
        work = info["lim_base"] + thread_module.layout()["KMX_LIM_BYTES"]
        return thread_module.RingPlacement(
            self.thread_area + c["kTaRing"],
            work + h["KMD_WORK_CONSUMED"],
            work + h["KMD_WORK_EVENTS"],
            self.thread_area + c["kTaRead"],
            c["kTaRingDepth"],
            c["kTaEventDepth"],
            remote=int(self.nocs.words[self.core]),
        )

    def work_info(self, cluster):
        return self.x280.work_info(cluster)

    def work_running(self, cluster):
        return self.x280 is not None and self.x280.work_running(cluster)

    def work_abort(self, cluster):
        return self.x280.work_abort(cluster) if self.x280 is not None else True

    def work_write(self, cluster, offset, data):
        if self.x280 is not None:
            return self.x280.work_write(cluster, offset, data)
        self._thread_cluster(cluster)
        data = bytes(data) + bytes(-len(data) % 4)
        if self.native is None:  # a mock: kept here
            lim = self._lim = self._lim or bytearray(
                thread_module.layout()["KMX_LIM_BYTES"]
            )
            lim[offset : offset + len(data)] = data
            return
        words = list(struct.unpack(f"<{len(data) // 4}I", data))
        address = self.thread_area + constants()["kTaLim"] + offset
        self.card.module.write_l1(self.card.device, self.x, self.y, address, words)

    def work_read(self, cluster, offset, n):
        if self.x280 is not None:
            return self.x280.work_read(cluster, offset, n)
        self._thread_cluster(cluster)
        if self.native is None:
            got = bytes((self._lim or b"")[offset : offset + n])
            return got + bytes(n - len(got))
        address = self.thread_area + constants()["kTaLim"] + offset
        words = self.card.module.read_l1(
            self.card.device, self.x, self.y, address, n // 4
        )
        return struct.pack(f"<{n // 4}I", *(int(w) for w in words))

    def work_start(self, cluster, image, block):
        """The argument block into the area, then the start line: TRISC 1 starts.
        An X280 thread: its work area zeroed, then the runtime starts its image."""
        if self.x280 is not None:  # the area as big as this core's (kTaWork)
            zeros = bytes(constants()["kTaState"] - constants()["kTaWork"])
            self.x280.work_write(
                cluster, thread_module.layout()["KMX_LIM_BYTES"], zeros
            )
            return self.x280.work_start(cluster, str(image), block)
        del image  # its kernel was launched with the grid's
        self.work_write(cluster, 0, block)
        if self.native is None:  # a mock: the thread returned, state done
            result = struct.pack("<I", thread_module.layout()["KMX_STATE_DONE"])
            self.work_write(cluster, thread_module.layout()["KMX_LIM_RESULT"], result)
            return 0
        address = self.thread_area + constants()["kTaStart"]
        self.card.module.write_l1(self.card.device, self.x, self.y, address, [self.tag])
        return 0

    def work_wait(self, cluster, timeout_s):
        """{"ready": whether the thread's result block is final within timeout_s}."""
        if self.x280 is not None:
            return self.x280.work_wait(cluster, timeout_s)
        layout = thread_module.layout()
        final = (layout["KMX_STATE_DONE"], layout["KMX_STATE_FAILED"])
        deadline = time.monotonic() + timeout_s
        while True:
            state = struct.unpack(
                "<I", self.work_read(cluster, layout["KMX_LIM_RESULT"], 4)
            )
            if state[0] in final:
                return {"ready": True, "failed": ""}
            if time.monotonic() > deadline:
                return {"ready": False, "failed": ""}
            time.sleep(POLL_S)

    def work_stop(self, cluster, timeout_s):
        """Nothing to stop: TRISC 1 returns from kmx_main, or at the run's end."""
        if self.x280 is not None:
            return self.x280.work_stop(cluster, timeout_s)
        del timeout_s
        self._thread_cluster(cluster)
        return {"cleaned": True, "failed": ""}

    # The end ----------------------------------------------------------------------

    def _state(self):
        return self.native.words(0, 1)[0]

    def _final(self, timeout):
        """Whether the kernel's state is final within `timeout` seconds."""
        done = abi.headers()["KMD_STATE_DONE"]
        deadline = time.monotonic() + timeout
        while self._state() < done:
            if time.monotonic() > deadline:
                return False
            time.sleep(POLL_S)
        return True

    def _stop_core(self):
        """1 in its own stop flag: the loop ends (KMD_ERROR_STOPPED)."""
        address = self.control + abi.headers()["KM_CONTROL_STOP"]
        self.card.module.write_l1(self.card.device, self.x, self.y, address, [1])

    def _blocks(self):
        """{words: the result block, x280_cores: the X280's core block's form, only
        KMD_H_ABANDONED and RUN_WORDS set} as the kernel wrote them; `tail` its own
        counts, `parts` a KM_TD_PARTS build's ticks by part."""
        c, h = constants(), abi.headers()
        words = [int(w) for w in self.native.words(c["kThResult"], c["kTrWords"])]
        cores = [0] * h["KMD_H_WORDS"]
        cores[h["KMD_H_ABANDONED"]] = words[c["kTrAbandoned"]]
        for word, key in RUN_WORDS.items():  # the dynamic run's, as the X280 reports
            cores[h[word]] = words[c[key]]
        self.parts = words[c["kTrParts"] : c["kTrParts"] + c["kTrPartsCount"]]
        self.tail = {name: words[c[key]] for name, key in TAIL.items()}
        return {"words": words[: h["KMD_R_WORDS"]], "x280_cores": cores}

    def finish(self, timeout_s):
        """Waits (timeout_s) for the kernel's end after the run's STOP, its stop flag
        set if it did not end; its result block, as the X280's finish."""
        final = self._final(timeout_s)
        failed = ""
        if not final:
            self._stop_core()
            final = self._final(CLOSE_WAIT_S)
            failed = f"the Tensix dispatcher did not end within {timeout_s} s"
        return {
            "result_ready": final,
            "result": None,
            "cleaned": True,
            "failed": failed if final else failed + ", nor after its stop flag",
            **self._blocks(),
        }

    def close(self):
        """Ends the kernel if it runs (its stop flag) and closes the port; whether it
        ended (else close_failed says so)."""
        if self.x280 is not None:  # the X280 runtime's thread: closed first
            self.x280.close()
            self.x280 = None
        if self.native is None or not self.native.is_open:
            return True
        ended = self._state() >= abi.headers()["KMD_STATE_DONE"]
        if not ended:
            self._stop_core()
            ended = self._final(CLOSE_WAIT_S)
        self._aborted = self._blocks() if ended else None
        self.native.close()
        self.block = None
        self.close_failed = (
            "" if ended else "the Tensix dispatcher did not end after its stop flag"
        )
        return ended

    def aborted(self):
        """After a close that ended the kernel: {words, x280_cores}, else None."""
        return self._aborted

    def report(self):
        """What the kernel is doing, from its host words (a wait that ran out)."""
        if not self.is_open:
            return "the Tensix dispatcher's port is closed"
        h = abi.headers()
        state, consumed, told, error = (int(w) for w in self.native.words(0, 4))
        return (
            f"the Tensix dispatcher (core {self.x}, {self.y}): state {state} "
            f"(READY {h['KMD_STATE_READY']}), commands taken {consumed}, events "
            f"written {told}, error {error}"
        )

    def probe(self, gap_s):
        """As the X280's probe: whether it runs (its words; no pass marker)."""
        del gap_s
        return {"running": self.is_open, "failed": "", "words": None}

    def inspect(self, count):
        """While the kernel runs: its inspect block with its first `count` records and
        its count of commands given up, read from its core's L1 (the X280's: LIM)."""
        c, h = constants(), abi.headers()
        words = (h["KMD_INSPECT_HEAD"] + count * h["KMD_INSPECT_RECORD"]) // 4
        if mock.active() or self.native is None:
            return {"block": np.zeros(words, np.uint32), "given_up": 0}
        read = self.card.module.read_l1
        device = self.card.device
        block = read(device, self.x, self.y, self.block + c["kTbInspect"], words)
        at = self.block + c["kTbHand"] + c["kTxWords"] + 4 * c["kTwGiveUps"]
        given_up = read(device, self.x, self.y, at, 1)[0]
        return {"block": np.asarray(block, np.uint32), "given_up": int(given_up)}
