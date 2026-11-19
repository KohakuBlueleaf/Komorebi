"""The dispatcher behind a session's run: the calls the session, the run and the backend
make on it (Dispatch), which one a session takes, and what tells the two apart. The
X280's is the native module's Dispatch; the Tensix core's is dispatch.tensix's."""

import os
from typing import Any, Protocol

X280 = "x280"
TENSIX = "tensix"
KINDS = (X280, TENSIX)
VARIABLE = "KMR_DISPATCHER"  # the choice of a session that names none (a card check)


class Dispatch(Protocol):
    """What the session, its run and its backend call on a dispatcher's port."""

    grid_x: int
    grid_y: int
    batch: int
    settle_microseconds: float
    is_open: bool
    idle: bool
    queued: int
    close_failed: str

    def core_table(self) -> Any: ...
    def prepare(self, depth, host_depth, event_depth, t_start_s, t_command_s,
    t_idle_s, state_ddr, **options) -> int: ...  # fmt: skip

    def start(self, image) -> None: ...
    def wait_ready(self, timeout_s) -> int: ...
    def enqueue(self, commands) -> int: ...
    def wait(self, wanted, upto, timeout_s) -> Any: ...
    def wait_events(self, timeout_s) -> Any: ...
    def poll(self) -> Any: ...
    def publish(self) -> None: ...
    def host_word(self, word) -> int: ...
    def finish(self, timeout_s) -> dict: ...
    def close(self) -> bool: ...
    def aborted(self) -> dict | None: ...
    def probe(self, gap_s) -> dict: ...


MEMBERS = (
    "grid_x", "grid_y", "batch", "settle_microseconds", "is_open", "idle", "queued",
    "close_failed", "core_table", "prepare", "start", "wait_ready", "enqueue", "wait",
    "wait_events", "poll", "publish", "host_word", "finish", "close", "aborted",
    "probe",
)  # fmt: skip


def choose(asked, has_x280):
    """The session's dispatcher: `asked`, else KMR_DISPATCHER, else the X280 where the
    build has its runtime and the Tensix core where it has not."""
    kind = asked or os.environ.get(VARIABLE) or (X280 if has_x280 else TENSIX)
    if kind not in KINDS:
        raise ValueError(f"dispatcher={kind!r}: one of {', '.join(KINDS)}")
    return kind


def kind(port):
    """X280 or TENSIX: the port's dispatcher (the mock's stand-in is an X280's)."""
    return getattr(port, "kind", X280)


DISPATCHER_CORES = {X280: 0, TENSIX: 1}  # Tensix cores a dispatcher takes


def workers(grid, kind):
    """The cores a load's program runs on: the grid's, less the dispatcher's own."""
    return grid[0] * grid[1] - DISPATCHER_CORES[kind]


def ranks(port):
    """workers() of a port's grid and dispatcher."""
    return workers((port.grid_x, port.grid_y), kind(port))


def start_words(port, control, abi_headers):
    """The cores' start-up words that name the dispatcher: its NoC coordinates, where
    their acks and READY words go, and core 0's hello."""
    if kind(port) == TENSIX:
        return port.start_words(control)
    base = port.lim_base
    return {
        "dispatcher_coordinates": port.l2_x | port.l2_y << 16,
        "ack_base": base + abi_headers["KMD_LIM_ACKS"],
        "ready_base": base + abi_headers["KMD_LIM_READY"],
        "mailbox": base + abi_headers["KMD_LIM_MAILBOX"],
    }
