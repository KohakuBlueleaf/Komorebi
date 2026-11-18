"""The process's sessions: the exit hooks, the open of a card and the chip lock, the
runtime's timeout at the exit, the card this process sees and its driver's state."""

import atexit
import os
import signal
import time
import weakref
from pathlib import Path

from komorebi.bench.power import SYS, driver_number, visible_card
from komorebi.device import chips, mock
from komorebi.errors import CardNotReady, KomorebiError

CHIP_LOCK = "/dev/shm/TT_UMD_LOCK.CHIP_IN_USE_{card}_PCIe"  # UMD's, by driver number
CHIP_LOCK_BYTES, CHIP_LOCK_PID = 56, 52  # its size; the owner PID's byte offset
MOCK = "TT_METAL_MOCK_CLUSTER_DESC_PATH"  # set: the device is a mock, it holds no chip

# The runtime's timeout on its waits for the cores, the one at device close included;
# unset, it is 0, which means no timeout. Read when the process first opens a device.
OP_TIMEOUT = "TT_METAL_OPERATION_TIMEOUT_SECONDS"
OP_TIMEOUT_DEFAULT = 30.0  # the runtime's own example value


# sessions holding a device they opened or an X280 runtime; weak, so that the exit
# hooks never keep a session alive
_LIVE = weakref.WeakSet()
_HOOKED = []  # [True] once the atexit hook is installed
_SIGTERM_HOOKED = []  # [True] once the SIGTERM hook is installed (or the process's own)
_SIGTERM_SAID = []  # [True] once an open off the main thread said it could not
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


def _hook_exit(log=None):
    """atexit for every session; SIGTERM too when the process has no handler of its
    own. Signals are set from the main thread only: an open on another thread says so
    once (`log`), and a later open on the main thread installs it."""
    if not _HOOKED:
        _HOOKED.append(True)
        atexit.register(_release_all, "exit")
    if _SIGTERM_HOOKED:
        return
    if signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
        _SIGTERM_HOOKED.append(True)  # the process's own handler: left as it is
        return
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except ValueError:  # not the main thread
        if log is not None and not _SIGTERM_SAID:
            _SIGTERM_SAID.append(True)
            log(
                "SIGTERM hook not installed: this open is not on the main thread (a "
                "SIGTERM then ends the process without stopping its runs; an open on "
                "the main thread installs it)"
            )
        return
    _SIGTERM_HOOKED.append(True)


def _command(pid):
    """A process's command line, shortened, or "?"."""
    try:
        words = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return "?"
    return " ".join(word.decode(errors="replace") for word in words if word)[:160]


def clock_said(card, when, arch, cause=None, log=print, driver=None):
    """One line when card `card`'s clock (the driver's device `driver`, when given)
    reads below its architecture's busy clock (chips) at `when` ("the open", "the
    close"): the clock, the cause, the way round. Nothing on a mock or unreadable."""
    if mock.active() or os.environ.get(MOCK):
        return None
    busy_mhz = chips.chip(arch).busy_mhz
    try:
        mhz = aiclk_mhz(card, driver)
        for _ in range(10 if when == "the open" else 0):  # a raise still in flight
            if mhz >= busy_mhz:
                break
            time.sleep(0.05)
            mhz = aiclk_mhz(card, driver)
    except (OSError, ValueError, IndexError):
        return None
    if mhz >= busy_mhz:
        return mhz
    if cause is not None:
        because = (
            f"{cause}; the way round: one process a card, or open both cards in one "
            "session (cards=[...])"
        )
    else:
        because = "throttled" if when == "the close" else "the cause not known"
    log(
        f"komorebi: card {card}'s clock is {mhz} MHz at {when}, below {busy_mhz} MHz "
        f"({because}): the times taken on it are at that clock"
    )
    return mhz


def aiclk_mhz(card=None, driver=None):
    """The Tensix clock of the card this process uses (or of the driver's device
    `driver`, the open device's own number), from the driver (MHz)."""
    card = card if card is not None else visible_card()
    driver = driver if driver is not None else driver_number(card)
    path = Path(f"/sys/class/tenstorrent/tenstorrent!{driver}/tt_aiclk")
    return int(path.read_text().split()[0])


DEV = Path("/dev/tenstorrent")  # the driver's device nodes, N for card N
HEARTBEAT_S = 0.3  # the wait between two heartbeat reads (a live card: about 3 counts)
# The driver's files a card state reads, below its sysfs entry, by the state's key.
CARD_FILES = {
    "type": "tt_card_type",
    "firmware": "tt_fw_bundle_ver",
    "aiclk": "tt_aiclk",
}
RESET = chips.RESET  # a card's remedy, unless its card type has its own (reset)


def card_arch(card):
    """The architecture of card `card` before it opens: the mock cluster's on the
    stand-in card, else its driver's card type (chips.card_arch)."""
    if mock.active() or os.environ.get(MOCK):
        return mock.cluster_arch()
    entry = Path(SYS) / "class" / "tenstorrent" / f"tenstorrent!{driver_number(card)}"
    return chips.card_arch(_driver_file(entry / CARD_FILES["type"]))


def reset(card, state=None):
    """The remedy for card `card` when it is not ready, by its card type (chips.reset),
    from `state` (card_states) or read from its driver files."""
    if state is None:
        entry = (
            Path(SYS) / "class" / "tenstorrent" / f"tenstorrent!{driver_number(card)}"
        )
        card_type = _driver_file(entry / CARD_FILES["type"])
    else:
        card_type = state.get("type")
    return chips.reset(card, card_type)


_READY = set()  # the cards this process found ready: an open checks a card once


def _driver_file(path):
    """A driver file's text, stripped, or None when it cannot be read."""
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def card_entries(root=None, dev=None):
    """The cards the driver lists, by sysfs entry (below `root`, default /sys) or by
    device node (in `dev`, default /dev/tenstorrent), sorted."""
    entries = (Path(SYS if root is None else root) / "class" / "tenstorrent").glob("*")
    found = {entry.name.partition("!")[2] for entry in entries}
    found |= {node.name for node in Path(DEV if dev is None else dev).glob("*")}
    return sorted(int(name) for name in found if name.isdigit())


def card_states(cards, root=None, dev=None, lock=None, wait=HEARTBEAT_S):
    """Each card's state from the driver's files, a dict by card: `entry` and `node`
    (whether they exist), CARD_FILES' values, the two heartbeat reads `wait` apart
    (one wait for all), `holder` (chip_holder; `lock` a CHIP_LOCK-like pattern). A
    file that cannot be read is None and named in `missing`. Opens no device."""
    base = Path(SYS if root is None else root) / "class" / "tenstorrent"
    dev = Path(DEV if dev is None else dev)
    number = {card: driver_number(card, root) for card in cards}  # the driver's
    beats = {
        card: _driver_file(base / f"tenstorrent!{number[card]}/tt_heartbeat")
        for card in cards
    }
    if any(beat is not None for beat in beats.values()):
        time.sleep(wait)
    states = {}
    for card in cards:
        entry = base / f"tenstorrent!{number[card]}"
        state = {"card": card, "driver": number[card], "entry": entry.is_dir()}
        state["node"] = dev / str(number[card])
        state["node_exists"] = state["node"].exists()
        state.update(
            {key: _driver_file(entry / name) for key, name in CARD_FILES.items()}
        )
        state["heartbeat"] = (beats[card], _driver_file(entry / "tt_heartbeat"))
        state["wait"] = wait
        names = {**CARD_FILES, "heartbeat": "tt_heartbeat"}
        state["missing"] = [
            names[key]
            for key in names
            if state[key] is None or (key == "heartbeat" and None in state[key])
        ]
        state["holder"] = chip_holder(card, lock and lock.format(card=number[card]))
        states[card] = state
    return states


def card_problems(state):
    """What keeps a card from opening, by its state (card_states), as phrases; none
    when its entry and node exist and its heartbeat advanced."""
    said = []
    if not state["entry"]:
        said.append(f"the driver has no sysfs entry tenstorrent!{state['driver']}")
    if not state["node_exists"]:
        said.append(f"its device node {state['node']} is missing")
    before, after = state["heartbeat"]
    if state["entry"] and (before is None or after is None):
        said.append("its heartbeat (tt_heartbeat) could not be read")
    elif state["entry"] and before == after:
        said.append(f"its heartbeat stayed at {after} over {state['wait']:g} s")
    return said


def card_ready(cards, root=None, dev=None):
    """Refuses with CardNotReady the first card of `cards` that card_problems names;
    each card once a process. Nothing on the stand-in card (a prebuild's mock, or a
    script's own mock cluster): no card is opened there."""
    if mock.active() or os.environ.get(MOCK):
        return
    todo = [card for card in cards if card not in _READY]
    if not todo:
        return
    states = card_states(todo, root, dev)
    for card in todo:
        said = card_problems(states[card])
        if said:
            raise CardNotReady(
                f"card {card} is not ready: {'; '.join(said)} (the open would hang or "
                f"fail): {reset(card, states[card])}",
                card=card,
                state=states[card],
            )
        _READY.add(card)


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


def chip_holder(card, path=None):
    """The PID of another live process that holds `card`'s chip lock, else None. Read
    from UMD's lock file (a pthread mutex, a 64-bit word, the owner's TID and PID)
    without taking it; UMD keeps the lock from a process's first open to its exit."""
    if mock.active() or os.environ.get(MOCK):  # a mock: no chip to hold
        return None
    # the file's mtime cannot date the PID: UMD writes the lock through a mapping,
    # which leaves the mtime unchanged
    path = Path(path or CHIP_LOCK.format(card=driver_number(card)))
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
