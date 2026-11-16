"""The process's sessions: the exit hooks, the open of a card and the chip lock, the
runtime's timeout at the exit and the card this process sees."""

import atexit
import contextlib
import os
import signal
import weakref
from pathlib import Path

from komorebi.builder import prebuild
from komorebi.errors import KomorebiError

CHIP_LOCK = "/dev/shm/TT_UMD_LOCK.CHIP_IN_USE_{card}_PCIe"  # UMD's lock per card
CHIP_LOCK_BYTES, CHIP_LOCK_PID = 56, 52  # its size; the owner PID's byte offset
MOCK = "TT_METAL_MOCK_CLUSTER_DESC_PATH"  # set: the device is a mock, it holds no chip

# The runtime's timeout on its waits for the cores, the one at device close included;
# unset, it is 0, which means no timeout. Read when the process first opens a device.
OP_TIMEOUT = "TT_METAL_OPERATION_TIMEOUT_SECONDS"
OP_TIMEOUT_DEFAULT = 30.0  # the runtime's own example value


# sessions holding a device they opened or an X280 runtime; weak, so that the exit
# hooks never keep a session alive
_LIVE = weakref.WeakSet()
_HOOKED = []  # [True] once the exit hooks are installed
# What this process opened: "mesh" (cards as a fabric mesh), "alone" (a card by itself)
_OPENED = {"mesh": [], "alone": []}


def refuse_reopen(kind, cards):
    """Refuses opening `cards` as `kind` ("mesh" or "alone") in a process that already
    opened the other kind; then records the attempt (see the hazards page)."""
    other = "alone" if kind == "mesh" else "mesh"
    if _OPENED[other]:
        was = ",".join(str(card) for card in _OPENED[other][0])
        now = ",".join(str(card) for card in cards)
        said = {"mesh": f"cards {was} as a fabric mesh", "alone": f"card {was} alone"}
        raise KomorebiError(
            f"{'cards ' + now + ' as a mesh' if kind == 'mesh' else 'card ' + now}: "
            f"this process opened {said[other]}; tt-metal would re-initialise its "
            "context over every card the process holds, which left both cards' "
            "Ethernet cores inactive until a board reset: open it in a new process"
        )
    _OPENED[kind].append(tuple(cards))


def _release_all(why, close_device=True):
    """The exit hooks' clean-up: every live session's run is stopped through the X280
    runtime (with a timeout), then the device it opened is closed."""
    for session in list(_LIVE):
        session._release(why, close_device)


def _on_sigterm(signum, frame):
    """SIGTERM: the runtimes are closed while the devices are open, then the process
    exits through its `with` blocks and the exit hook, which close the devices."""
    _release_all("SIGTERM", close_device=False)
    raise SystemExit(128 + signum)


def _hook_exit():
    """atexit for every session; SIGTERM too when the process has no handler of its
    own (signals are set from the main thread only)."""
    if _HOOKED:
        return
    _HOOKED.append(True)
    atexit.register(_release_all, "exit")
    if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        with contextlib.suppress(ValueError):  # not the main thread
            signal.signal(signal.SIGTERM, _on_sigterm)


def _command(pid):
    """A process's command line, shortened, or "?"."""
    try:
        words = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return "?"
    return " ".join(word.decode(errors="replace") for word in words if word)[:160]


def aiclk_mhz(card=None):
    """The Tensix clock of the card this process uses, from the driver (MHz)."""
    card = card if card is not None else visible_card()
    path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
    return int(path.read_text().split()[0])


def exit_timeout(environ, log):
    """Sets OP_TIMEOUT to its default in `environ` when unset or empty and logs it; a
    value already there is kept. The timeout in seconds (0: none; None: not a number).
    """
    value = environ.get(OP_TIMEOUT, "")
    if value.strip():
        try:
            return max(float(value), 0.0)
        except ValueError:
            return None  # the runtime rejects it at the open
    environ[OP_TIMEOUT] = f"{OP_TIMEOUT_DEFAULT:g}"
    log(f"{OP_TIMEOUT} unset: set to {OP_TIMEOUT_DEFAULT:g} s (the exit's timeout)")
    return OP_TIMEOUT_DEFAULT


def exit_wait(environ=None):
    """How long the device's close waits for the cores, as a phrase."""
    value = (os.environ if environ is None else environ).get(OP_TIMEOUT, "")
    try:
        seconds = float(value)
    except ValueError:
        seconds = 0.0
    return f"up to {seconds:g} s" if seconds > 0 else "without a timeout"


def visible_card():
    return int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0])


def chip_holder(card, path=None):
    """The PID of another live process that holds `card`'s chip lock, else None. Read
    from UMD's lock file (a pthread mutex, a 64-bit word, the owner's TID and PID)
    without taking it; UMD keeps the lock from a process's first open to its exit."""
    if prebuild.active() or os.environ.get(MOCK):  # a mock: no chip to hold
        return None
    # the file's mtime cannot date the PID: UMD writes the lock through a mapping,
    # which leaves the mtime unchanged
    path = Path(path or CHIP_LOCK.format(card=card))
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) != CHIP_LOCK_BYTES:
        return None
    pid = int.from_bytes(data[CHIP_LOCK_PID : CHIP_LOCK_PID + 4], "little", signed=True)
    if pid <= 0 or pid == os.getpid():
        return None
    try:  # the owner holds it only while it maps the file: a zombie, a reused PID
        # or a process that died holding the lock does not
        with open(f"/proc/{pid}/maps") as maps:
            return pid if any(str(path) in line for line in maps) else None
    except (FileNotFoundError, ProcessLookupError):
        return None  # the process ended
    except PermissionError:
        return pid  # alive, another user's
