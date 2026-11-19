"""The dispatcher on one Tensix core, the grid's last: its kernel in the load's launch,
its block in L1, its area of the host channel past the session's region, and the
native port (TensixPort) that writes its host ring and reads its events."""

import functools
import random
import time

import numpy as np

from komorebi import _env, abi
from komorebi.device import chips, mock, native
from komorebi.errors import NotServed
from komorebi.session.dispatch import TENSIX

OUT, IN = "dispatcher.cpp", "td_in.cpp"  # its loops: out takes and submits, in sweeps
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
    "dynamic": "dynamic runs",
    "islands": "islands and links",
    "trace": "the trace",
    "recovery": "recovery beyond the stop",
    "inspect": "the inspect of a run",
    "engines": "stream/graph engines",
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


def refuse_load(program, placed, owner, options):
    """NotServed for a load of what this dispatcher does not serve (it serves plain
    commands, chains, overlays, add-ons, several programs): the first one it asks."""
    asks = [
        (bool(owner.threads) or owner.x280_ports is not None, "threads"),
        (program.messages is not None, "dynamic"),
        (
            options.get("island") is not None
            or options.get("link") is not None
            or "KM_PAGE_LINKS" in program.defines,
            "islands",
        ),
        (bool(options.get("trace")), "trace"),
    ]
    for asked, name in asks:
        if asked:
            refuse(name, f"load of {program.name}")


def block_buffer(card):
    """(the block's L1 address, its buffer's bytes): the buffer is the first on the
    dispatcher's core, at the buffers' base (as tt-metal places it), the block at its
    first 64-byte boundary in it; the kernel checks the address (td::block_placed)."""
    _, free, top = card.module.kernel_buffer(card.device)
    at = -(-(top - free) // card.cb_align) * card.cb_align
    block = -(-at // 64) * 64
    return block, block - at + constants()["kTbBytes"]


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
        self.block = None  # its block's L1 address, on its core alone
        self.control = 0
        self.options = {}
        self.tag = 0
        self.close_failed = ""
        self._aborted = None
        self._batch = 64
        self.tail = {}  # the result's words past the X280's (TAIL), once read
        self.parts = []

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
        served = {"channel_region", "upload_bytes", "first_row", "rows", "programs"}
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
        }
        args = [0] * c["kTdArgs"]
        for name, value in values.items():
            args[c[name]] = int(value) & abi.M32
        for r, rect in enumerate(rects):
            args[c["kTdRect"] + 3 * r : c["kTdRect"] + 3 * r + 3] = rect
        words = held_words(options["programs"])
        args[c["kTdHeld"] : c["kTdHeld"] + len(words)] = words
        includes = [str(h.abi.parents[2]), str(native.kernel_directory())]
        defines = [(name, str(value)) for name, value in DEFINES.items()]
        out, inside = LOOPS[SPLIT]
        if inside is None:
            defines.append(("KM_TD_ALONE", "1"))
        for source, processor in ((OUT, out), (IN, inside)):
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
        KMD_H_ABANDONED set} as the kernel wrote them; `tail` its own counts, `parts` a
        KM_TD_PARTS build's ticks by part."""
        c, h = constants(), abi.headers()
        words = [int(w) for w in self.native.words(c["kThResult"], c["kTrWords"])]
        cores = [0] * h["KMD_H_WORDS"]
        cores[h["KMD_H_ABANDONED"]] = words[c["kTrAbandoned"]]
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
        """A deadline's look (run.inspect itself is refused): no inspect block, no
        command given up; the run stops with the timeout."""
        h = abi.headers()
        words = (h["KMD_INSPECT_HEAD"] + count * h["KMD_INSPECT_RECORD"]) // 4
        return {"block": np.zeros(words, np.uint32), "given_up": 0}
