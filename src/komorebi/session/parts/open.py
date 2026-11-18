"""The session's card connection: opens and closes the device, maps the host channel."""

import contextlib
import ctypes
import faulthandler
import os
import sys
import threading
from collections.abc import Callable
from typing import Any

from komorebi import abi
from komorebi.device import channel as channel_module
from komorebi.device import mock, native
from komorebi.errors import (
    CardBusy,
    ChannelError,
    KomorebiError,
)
from komorebi.session import process

# The last bytes _output_seen writes into each pipe: the echo ends there, not at the
# pipe's end, which a process the runtime starts (MPI's orted) keeps open for good
_END = b"\0komorebi: the open's output ends\0"


def _echo(source, target, seen):
    """Copies a pipe to a file descriptor as the bytes come, keeping them in `seen`,
    until _END (held back while it may be arriving) or the pipe's end."""
    held = b""
    with contextlib.suppress(OSError):
        while data := os.read(source, 1 << 16):
            held += data
            at = held.find(_END)
            if at >= 0:
                held = held[:at] + held[at + len(_END) :]
                break
            keep = len(_END) - 1
            out, held = held[: len(held) - keep], held[len(held) - keep :]
            if out:
                seen.append(out)
                os.write(target, out)
        if held:
            seen.append(held)
            os.write(target, held)
    os.close(source)


def _handler_on_stderr():
    """True when the fault handler is on as the interpreter itself enabled it
    (PYTHONFAULTHANDLER, -X faulthandler): on descriptor 2, all threads. A handler a
    program enabled on a file of its own is left where it is."""
    asked = os.environ.get("PYTHONFAULTHANDLER") or "faulthandler" in sys._xoptions
    return bool(asked) and faulthandler.is_enabled()


@contextlib.contextmanager
def _output_seen():
    """While the block runs, this process's stdout and stderr (the descriptors, so the
    runtime's own lines too) pass through pipes to where they went; yields the bytes.
    The interpreter's fault handler (_handler_on_stderr) writes to a copy of the real
    stderr meanwhile (a crash ends the pipe unread), then to descriptor 2 again."""
    seen, threads, kept = [], [], []
    flush = ctypes.CDLL(None).fflush
    sys.stdout.flush()
    sys.stderr.flush()
    flush(None)
    fault = os.dup(2) if _handler_on_stderr() else None
    if fault is not None:
        faulthandler.enable(file=fault, all_threads=True)
    for fd in (1, 2):
        source, sink = os.pipe()
        kept.append(os.dup(fd))
        os.dup2(sink, fd)
        os.close(sink)
        thread = threading.Thread(target=_echo, args=(source, kept[-1], seen))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    try:
        yield seen
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        flush(None)
        for fd, saved in zip((1, 2), kept, strict=True):
            with contextlib.suppress(OSError):
                os.write(fd, _END)
            os.dup2(saved, fd)
        if fault is not None:
            faulthandler.enable(file=2, all_threads=True)
            os.close(fault)
        for thread in threads:
            thread.join(timeout=5.0)
        for saved in kept:
            os.close(saved)


# TT_VISIBLE_DEVICES values this process's sessions set and did not restore yet: a
# later open re-sets one of these (a stuck session's), never a value the user gave
_VISIBLE_SET = set()
_UNSET = object()


class OpenMixin:
    """The session's card connection: opens and closes the device, maps the channel."""

    # the state of the session's other parts this one reads or sets, owned there
    _channel_mapped: bool  # Session
    _memory: Any  # Session
    _own_device: bool  # Session
    _region: int | None  # Session
    _worker_l1: int  # Session
    card: int | None  # Session
    device: Any  # Session
    log: Callable  # Session
    tree: Any  # Session
    _starting: Any  # LoadMixin
    run: Any  # LoadMixin

    def _init_open(self):
        """The connection's state before open: no device module, no channel mapped."""
        self.stuck = False  # the grid did not return: the runtime closes it at exit
        self._module = None
        self.channel = None
        self._channel_said = False  # the fallback has been logged
        # a prebuild: the device the card's channel would be mapped for (None: none)
        self._mock_channel = None
        self._clock_card = None  # the card whose clock the open and close read
        self._driver = None  # its driver's number, as the open device says
        self._clock_cause = None  # why it was below the busy clock, when known
        # TT_VISIBLE_DEVICES before this session's open set it (_UNSET: not set)
        self._visible_before = _UNSET
        if mock.active():  # a prebuild: from here a read may give zeros (its writes)
            mock.SESSIONS[0] += 1

    def open(self):
        """Opens the card (slow dispatch) if the session was not given a device."""
        if self.device is not None:
            raise KomorebiError(
                "open: this session's device is open already (a session given its "
                "device, or one opened): close() it first, or make a new Session"
            )
        if self.card is not None:
            visible = os.environ.get("TT_VISIBLE_DEVICES")
            ours = visible in _VISIBLE_SET
            listed = [one.strip() for one in (visible or "").split(",")]
            if visible is not None and not ours and str(self.card) not in listed:
                raise ValueError(f"card {self.card} is not visible ({visible})")
            if self._visible_before is _UNSET:
                self._visible_before = None if ours else visible
            os.environ["TT_VISIBLE_DEVICES"] = str(self.card)
            _VISIBLE_SET.add(str(self.card))
        os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
        if process.exit_timeout(os.environ, self.log) == 0:
            self.log(
                f"{process.OP_TIMEOUT}=0: an exit after a grid that did not return "
                "hangs"
            )
        card = self.card if self.card is not None else process.visible_card()
        holder = process.chip_holder(card)
        if holder is not None:
            raise CardBusy(
                f"card {card} is held by process {holder} "
                f"({process._command(holder)}): an open would wait until that process "
                "exits",
                pid=holder,
            )
        process.card_ready([card])  # the driver's files: CardNotReady before tt-metal
        before = [opened for (opened,) in process._OPENED["alone"] if opened != card]
        process.refuse_reopen("alone", [card])
        mock.choose_cluster()  # set before the native module loads (issue 64717)
        self._module = self._module or native.load_module(self.tree)
        if self._region is None:
            with mock.cluster_visible():  # program cache off
                self.device = self._module.Device(self._worker_l1)
        else:
            with _output_seen() as seen, mock.cluster_visible():  # the driver's warning
                self.device = self._module.Device(self._worker_l1)
        process._hook_exit(self.log)
        process._LIVE.add(self)
        self._own_device = True
        self._clock_card = card
        self._driver = self._driver_number(card)
        cause = (
            f"opened after card {before[0]} in this process: tt-metal raised only the "
            "first card's clock"
            if before
            else None
        )
        self._clock_cause = cause
        process.clock_said(card, "the open", cause, self._said, self._driver)
        self._dram_banks = int(self._module.dram_banks(self.device))
        if self._region is not None:
            try:
                self._open_channel(b"".join(seen).decode(errors="replace"))
            except BaseException:  # any error: the device closed (`with` never exits)
                self.channel = None
                self._close_device()
                raise

    def _driver_number(self, card):
        """The driver's number of the open device (Device.driver_device), its clock's
        and lock's; refused (the device closed) when it is not the one the lock check
        took for `card` (process.driver_number). None on a mock."""
        if mock.active():
            return None
        number, bus = self.device.driver_device()
        if number >= 0 and number != process.driver_number(card):
            self._close_device()
            raise KomorebiError(
                f"card {card} opened as the driver's device {number} (PCI bus "
                f"{bus:#04x}), not {process.driver_number(card)} as its lock was read"
            )
        return number if number >= 0 else None

    def _said(self, line):
        """A line the user must see: printed to stderr, and into the log."""
        print(line, file=sys.stderr, flush=True)
        self.log(line)

    def _map_channel(self):
        """A session given its device maps the channel at its first transfer. It did
        not see the open, so the driver's warning is unknown to it: a transfer whose
        sequence words do not arrive raises ChannelError."""
        if self._channel_mapped or self._region is None:
            return
        self._channel_mapped = True
        self._open_channel("")

    def _mock_mapped(self):
        """A prebuild: whether the card's channel would be mapped for this device (a
        device opened again maps it again, as the card's must)."""
        return self._mock_channel is not None and self._mock_channel is self.device

    def _open_channel(self, said):
        """Maps the region of the card's channel, or logs why reads go the device API's
        way; ChannelError (the device closed) if the open logged the warning."""
        if mock.active():  # the card's would be mapped: its loads are built so
            self._mock_channel = self.device
            return
        warning = channel_module.warning_in(said)
        if warning is not None:
            self.close()
            raise ChannelError(
                f"the device open logged '{warning}': the cores' writes to the host "
                "channel would not land in this process's mapping; close the other "
                "process holding the card's memory, or open with channel=0"
            )
        with open("/proc/self/maps") as maps:
            address, where = channel_module.find(
                maps.read(), self.device.host_channel()
            )
        if address is None:
            self.log(f"host channel not used ({where}): reads go through the device")
            self._channel_said = True
            return
        self.channel = channel_module.view(address, self._region)
        channel_module.words_view(
            self.channel, self._region, abi.headers()["KM_CORES_MAX"]
        )[:] = 0
        self.log(
            f"host channel {where}: a region of {self._region.bytes >> 20} MiB from 0"
        )

    def close(self):
        """Stops a loaded run through the X280 runtime, then closes the device if the
        session opened it. A session on a given device is closed before that device."""
        try:
            if self._memory is not None:
                self._memory.closed()
        finally:
            self._close_run()

    def _close_run(self):
        """close's own part: the run stopped, then the device closed."""
        try:
            if self.run is not None and not self.run.stopped:
                self.run.abort()
        finally:
            if self.run is None or self.run.stopped:  # else the runtime needs it
                self._close_device()

    def _close_device(self):
        if self.stuck:
            self.log(
                "device not closed by the session: the grid did not return; at the "
                f"process's exit the runtime waits {process.exit_wait()} for the "
                "cores, then puts them in reset (a stuck thread recovers at the next "
                "open, a compute unit left running does not)"
            )
            return
        if self.device is not None and self._own_device:
            process.clock_said(
                self._clock_card, "the close", self._clock_cause, self._said,
                self._driver,
            )  # fmt: skip
            self.channel = None  # a view over the driver's mapping: gone with it
            self.device.close()
            self.device = None
        self._restore_visible()
        process._LIVE.discard(self)

    def _restore_visible(self):
        """TT_VISIBLE_DEVICES as it was before this session's open, unless it was
        changed since (another session's open)."""
        before, self._visible_before = self._visible_before, _UNSET
        if before is _UNSET:
            return
        _VISIBLE_SET.discard(str(self.card))
        if os.environ.get("TT_VISIBLE_DEVICES") != str(self.card):
            return
        if before is None:
            os.environ.pop("TT_VISIBLE_DEVICES", None)
        else:
            os.environ["TT_VISIBLE_DEVICES"] = before

    def _reopen_device(self, worker_l1):
        """Closes the card and opens it again with the runtime's `worker_l1` (a new
        kernel buffer, kernel_buffer.reopen); the channel's mapping goes with it."""
        self.channel = None
        self._close_device()
        self._worker_l1 = worker_l1
        self.open()

    def _release(self, why, close_device):
        """The exit hooks' timed clean-up of this session: closes the X280 runtime of
        a load in progress or a loaded run, then (`close_device`) closes the session.
        Errors are logged, not raised: the process is ending. The close runs even when
        the stop raised."""
        try:
            if self._starting is not None:
                self.log(f"{why}: closing the X280 runtime of the load in progress")
                if not self._starting.close():
                    self.log(f"{why}: not restored: {self._starting.close_failed}")
            if self.run is not None and not self.run.stopped:
                self.log(f"{why}: stopping the loaded run through the X280 runtime")
                self.run.abort()
        except Exception as error:  # noqa: BLE001 (the process ends: logged)
            self.log(f"{why}: clean-up raised {type(error).__name__}: {error}")
        finally:
            if close_device:
                try:
                    self.close()
                except Exception as error:  # noqa: BLE001 (as above)
                    self.log(f"{why}: close raised {type(error).__name__}: {error}")
