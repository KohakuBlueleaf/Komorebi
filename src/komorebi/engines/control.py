"""The host's side of the dispatcher's engines: uploads placed in the kernel's
upload region, the RUN and CANCEL commands, and the engines' events as typed values.
A `Run` keeps one `Engines` once an engine call is made (docs/reference/host.md).
"""

import bisect
import re
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from komorebi import abi
from komorebi.engines import graph as graph_module
from komorebi.engines import stream as stream_module
from komorebi.errors import (
    EngineError,
    EngineRejected,
    EngineTimeout,
    KomorebiError,
    WaitTimeout,
)

LINE = 64
READY_WAIT_S = 1.0  # the ready block follows the READY state word closely


def codes(abi_headers=None):
    """The host's constants (abi.headers) with those of km_stream.h and km_graph.h."""
    abi_headers = abi_headers or abi.headers()
    abi_directory = abi_headers.abi.parent
    return {
        **abi_headers.constants,
        **abi.constants(
            abi_headers.abi, abi_directory / "km_stream.h", abi_directory / "km_graph.h"
        ),
    }


def word_offset(path, name, member):
    """Byte offset of `member` among the leading uint32_t members of struct `name`
    (the members after the first other one are not read)."""
    text, _ = abi._clean(Path(path).read_text(), path)
    body = re.search(rf"typedef\s+struct\s*\{{([^{{}}]*)\}}\s*{name}\s*;", text)
    if body is None:
        raise abi.HeaderError(f"no struct {name} in {path}")
    names = []
    for decl in body[1].split(";"):
        words = abi.FIELD.match(" ".join(decl.split()))
        if words is None:
            break
        names += [field_name.strip() for field_name in words[1].split(",")]
    if member not in names:
        raise abi.HeaderError(f"{path}: struct {name}: no uint32_t {member} up front")
    return 4 * names.index(member)


def offsets(abi_headers=None):
    """Where the host reads the headers' words: the ready block's region size (a host
    buffer word) and the byte offsets of each kind's req and of a graph's flags."""
    abi_headers = abi_headers or abi.headers()
    abi_directory = abi_headers.abi.parent
    ready = abi.struct_fields(abi_headers.dispatch, "kmd_ready_t")
    return {
        "ready_bytes": ready.index("upload_bytes"),
        "stream": word_offset(abi_directory / "km_stream.h", "km_stream_t", "req"),
        "graph": word_offset(abi_directory / "km_graph.h", "km_graph_t", "req"),
        "flags": word_offset(abi_directory / "km_graph.h", "km_graph_t", "flags"),
    }


def _lines(size):
    return -(-size // LINE) * LINE


@dataclass(eq=False)
class Upload:
    """An upload in the region: `offset` and `bytes` there (the region keeps whole
    lines), `kind` ("stream" or "graph"), the `req` of its events (0 when the upload
    is shorter than its header's req word), the header's `keep` bit, and its `state`:
    "idle", "running" or "kept" (a kept graph's lines stay the kernel's)."""

    offset: int
    bytes: int
    kind: str
    req: int
    keep: bool
    state: str = "idle"
    data: bytes = field(default=b"", repr=False)
    graph: object = field(default=None, repr=False)  # the Graph, for names
    valid: bool = field(default=True, repr=False)  # the host's check passes


@dataclass(eq=False)
class EngineHandle:
    """One start of an engine: `kind`, the `req` of its events, the `upload`, every
    event so far (`events`) and the end or error event once it came (`end`).
    `unknown`: its RUN was not answered in time, so it may run; cancel settles it."""

    kind: str
    req: int
    upload: Upload
    restart: bool = False
    drop: bool = False  # a CANCEL that drops a kept graph
    events: list = field(default_factory=list, repr=False)
    end: object = field(default=None, repr=False)
    bind: int | None = field(default=None, repr=False)  # its binding line
    unknown: bool = False
    commands: tuple = field(default=(), repr=False)  # handles of its RUN and PING


class EngineEvent(NamedTuple):
    """An engine's event. `kind`: "event" (a stream's EVENT, `value` its register),
    "round" (a graph's round ended, `value` rounds done), "end" (a stream: `value`
    HALT's register; a graph: `registers` = (register 0's low 24 bits, register 1)) or
    "error" (`reason`, `reason_name`, and the stream's `pc` or the graph's `node`)."""

    handle: EngineHandle
    kind: str
    value: int | None = None
    registers: tuple | None = None
    reason: int | None = None
    reason_name: str | None = None
    pc: int | None = None
    node: int | None = None


class Region:
    """First fit over the upload region: pieces of whole lines, kept by offset."""

    def __init__(self, size):
        self.size = size
        self.used = []  # (offset, bytes), sorted

    def take(self, wanted):
        """The offset of `wanted` free bytes (a multiple of LINE), or None."""
        cursor = 0
        for offset, length in self.used:
            if offset - cursor >= wanted:
                break
            cursor = offset + length
        if cursor + wanted > self.size:
            return None
        bisect.insort(self.used, (cursor, wanted))
        return cursor

    def free(self, offset):
        self.used = [piece for piece in self.used if piece[0] != offset]

    def largest(self):
        ends = [0] + [offset + length for offset, length in self.used]
        starts = [offset for offset, _ in self.used] + [self.size]
        return max(start - end for start, end in zip(starts, ends))


class Engines:
    """The uploads and engines of one Run, and the routing of their events."""

    def __init__(self, run):
        self.run = run
        constants = codes(run.h)
        self.constants = constants
        self.header_offsets = offsets(run.h)
        self.reasons = {
            value: name[len("KM_STREAM_ERROR_") :].lower()
            for name, value in constants.items()
            if "_STREAM_ERROR_" in name
        }
        self.kinds = {
            constants["KM_EVENT_STREAM"]: ("stream", "event"),
            constants["KM_EVENT_STREAM_END"]: ("stream", "end"),
            constants["KM_EVENT_STREAM_ERROR"]: ("stream", "error"),
            constants["KM_EVENT_GRAPH"]: ("graph", "round"),
            constants["KM_EVENT_GRAPH_END"]: ("graph", "end"),
            constants["KM_EVENT_GRAPH_ERROR"]: ("graph", "error"),
        }
        self.first_kind, self.last_kind = min(self.kinds), max(self.kinds)
        host = [
            constants[name]
            for name in ("KM_EVENT_DONE", "KM_EVENT_TIMEOUT", "KM_EVENT_PONG")
        ]
        host.append(constants["KM_EVENT_REJECTED"])
        if any(self.first_kind <= kind <= self.last_kind for kind in host):
            raise KomorebiError(
                "the device headers give an engine's event kind the number of a "
                "command's (done, timeout, pong or rejected): the headers this build "
                "read disagree, a defect of the framework's build, not of the program"
            )
        self.region = None
        self.uploads = []
        self.active = {}  # (kind, req) -> the handle of the engine running
        self.unclaimed = []  # (kind, req, status, ticks) of no started engine

    # The region and uploads ----------------------------------------------------

    def _region(self):
        """The region the kernel's ready block names (its size; the host writes by
        offset)."""
        if self.region is None:
            port, ready_address = self.run.port, self.constants["KMD_HOST_READY"]
            deadline = time.monotonic() + READY_WAIT_S
            while port.host_word(ready_address) != self.constants["KMD_READY_MARK"]:
                if time.monotonic() > deadline:
                    raise KomorebiError("the dispatcher published no ready block")
            self.region = Region(
                port.host_word(ready_address + self.header_offsets["ready_bytes"])
            )
            self.region.used = sorted(self.run._reserved)  # the kept reset list
        return self.region

    def upload(self, content, kind=None):
        self.run._check_alive()
        graph = None
        if isinstance(content, stream_module.Stream):
            data, kind = content.encode(), "stream"
            self.run.check_templates(content.templates())
        elif isinstance(content, graph_module.Graph):
            data, kind, graph = content.encode(), "graph", content
            self.run.check_templates(content.templates())
        else:
            data = bytes(content)
            if kind not in ("stream", "graph"):
                raise ValueError("encoded bytes need kind='stream' or kind='graph'")
        if not data:
            raise ValueError("an empty upload")
        region = self._region()
        padded_bytes = _lines(len(data))
        offset = region.take(padded_bytes)
        if offset is None:
            raise ValueError(
                f"an upload of {len(data)} bytes does not fit: the region has "
                f"{region.size} bytes, the largest free piece {region.largest()}"
            )
        req = self._word(data, self.header_offsets[kind])
        keep = False
        valid = True
        if kind == "graph":
            flags = self._word(data, self.header_offsets["flags"])
            keep = bool(flags & self.constants["KM_GRAPH_KEEP"])
            valid = graph_module.check(data) is None
        try:
            self.run.port.upload(offset, data + bytes(padded_bytes - len(data)))
        except BaseException:
            region.free(offset)
            raise
        created = Upload(offset, len(data), kind, req, keep, data=data, graph=graph)
        created.valid = valid  # the kernel's check, which the host does not enforce
        self.uploads.append(created)
        return created

    @staticmethod
    def _word(data, offset):
        """The word at byte `offset`, or 0 when the upload is shorter."""
        return (
            struct.unpack_from("<I", data, offset)[0] if len(data) >= offset + 4 else 0
        )

    def release(self, upload, timeout):
        self._mine(upload)
        if upload.state == "running":
            raise KomorebiError(
                f"the {upload.kind} at {upload.offset} runs: wait or cancel"
            )
        if upload.state == "kept":
            deadline = (
                time.monotonic() + timeout
            )  # one deadline for the command and the end
            engine = EngineHandle("graph", upload.req, upload, drop=True)
            self.active[("graph", upload.req)] = engine
            try:
                self._command(
                    self.constants["KM_HANDLER_CANCEL"], [upload.req], timeout
                )
            except BaseException:
                self.active.pop(("graph", upload.req), None)
                raise
            self._wait(engine, max(deadline - time.monotonic(), 0.0))
        self.uploads.remove(upload)
        self._region().free(upload.offset)

    def _mine(self, upload):
        if not any(other is upload for other in self.uploads):
            raise ValueError("not an upload of this run, or released")

    # Starting and cancelling ------------------------------------------------------

    def start(self, upload, bindings, kept, timeout):
        self._mine(upload)
        if upload.state != "idle":
            raise KomorebiError(
                f"the {upload.kind} at {upload.offset} is {upload.state}"
            )
        self._free_req(upload.req)
        if upload.kind == "stream":
            if bindings or kept is not None:
                raise ValueError("a stream takes no bindings and is never kept")
            engine = EngineHandle("stream", upload.req, upload)
            return self._run(engine, [upload.offset, upload.bytes], timeout)
        if kept is not None and bool(kept) != upload.keep:
            self._set_keep(upload, bool(kept))
        engine = EngineHandle("graph", upload.req, upload)
        args = [upload.offset, upload.bytes, self.constants["KM_GRAPH_RUN_UPLOAD"]]
        return self._run(engine, args + self._bindings(engine, bindings), timeout)

    def restart(self, req, bindings, timeout):
        upload = next(
            (
                candidate
                for candidate in self.uploads
                if candidate.state == "kept" and candidate.req == req
            ),
            None,
        )
        if upload is None:
            raise ValueError(f"no kept graph that is not running has req {req}")
        engine = EngineHandle("graph", req, upload, restart=True)
        args = [req, 0, self.constants["KM_GRAPH_RUN_KEPT"]]
        return self._run(engine, args + self._bindings(engine, bindings), timeout)

    def cancel(self, engine, timeout):
        if engine.end is not None and not engine.unknown:
            return  # it has ended: nothing to cancel
        try:
            self._command(self.constants["KM_HANDLER_CANCEL"], [engine.req], timeout)
        except EngineRejected as error:
            ended = engine.end is not None or engine.unknown  # or its RUN was rejected
            if not ended or error.reason != self.constants["KMD_REJECT_UNKNOWN"]:
                raise  # unless it ended while the command was on its way
        except WaitTimeout as error:
            raise EngineTimeout(
                f"{engine.kind} {engine.req}: CANCEL not answered in {timeout} s",
                engine,
            ) from error
        if engine.unknown:
            self._settle_unknown(engine)

    def stop(self, timeout):
        """Cancels every engine still running or unknown and waits for its end, all
        within `timeout` (the kernel's EXIT waits for them); EngineTimeout if not."""
        deadline = time.monotonic() + timeout
        for engine in list(self.active.values()):
            if engine.drop or engine.end is not None:
                continue  # a kept graph's drop is on its way
            unknown = engine.unknown
            try:
                self.cancel(engine, max(deadline - time.monotonic(), 0.0))
            except EngineRejected:
                if not unknown:
                    raise  # an unknown one's RUN was rejected: it never ran
        for engine in list(self.active.values()):
            self._wait(engine, max(deadline - time.monotonic(), 0.0))

    def _settle_unknown(self, engine):
        """The RUN of `engine` was answered before the command answered now: raises its
        rejection (it never ran) or leaves `engine` running or ended."""
        engine.unknown = False
        command, ping = engine.commands
        self.run.forget([ping])
        try:
            self._answer(command)
        except EngineRejected:
            self._not_started(engine)
            raise

    def _free_req(self, req):
        busy = {busy_req for _, busy_req in self.active}
        busy |= {
            other_upload.req
            for other_upload in self.uploads
            if other_upload.state == "kept"
        }
        if req in busy:
            raise ValueError(f"req {req} belongs to a running or kept engine")

    def _set_keep(self, upload, keep):
        flags_offset = self.header_offsets["flags"]
        if upload.bytes < flags_offset + 4:
            raise ValueError("the upload is too short to carry the keep bit")
        line = bytearray(upload.data[:LINE].ljust(LINE, b"\0"))
        flags = struct.unpack_from("<I", line, flags_offset)[0]
        bit = self.constants["KM_GRAPH_KEEP"]
        struct.pack_into(
            "<I", line, flags_offset, flags | bit if keep else flags & ~bit
        )
        self.run.port.upload(upload.offset, bytes(line))
        upload.data = bytes(line[: len(upload.data)]) + upload.data[LINE:]
        upload.keep = keep

    def _bindings(self, engine, bindings):
        """[binding block offset, count]: the values in one line of the region."""
        graph = engine.upload.graph
        if isinstance(bindings, dict) or (graph is not None and not bindings):
            if graph is None:
                raise ValueError("bindings by name need an upload made from a Graph")
            bindings = graph.binding_values(**(bindings or {}))
        values = [int(value) for value in bindings]
        if len(values) > self.constants["KM_GRAPH_BINDINGS"]:
            raise ValueError(f"{len(values)} bindings, at most 16")
        if any(not 0 <= value <= abi.M32 for value in values):
            raise ValueError("bindings are 32-bit unsigned words")
        if not values:
            return [0, 0]
        offset = self._region().take(LINE)
        if offset is None:
            raise ValueError("no free line in the upload region for the bindings")
        line = struct.pack(f"<{len(values)}I", *values).ljust(LINE, b"\0")
        try:
            self.run.port.upload(offset, line)
        except BaseException:
            self._region().free(offset)
            raise
        engine.bind = offset
        return [offset, len(values)]

    def _run(self, engine, args, timeout):
        """KM_HANDLER_RUN for `engine`; its events are routed to it from now on."""
        self.active[(engine.kind, engine.req)] = engine
        engine.upload.state = "running"
        try:
            self._command(self.constants["KM_HANDLER_RUN"], args, timeout, engine)
        except WaitTimeout as error:
            engine.unknown = True  # the kernel may have started it
            raise EngineTimeout(
                f"{engine.kind} {engine.req}: RUN not answered in {timeout} s; "
                "it may run until cancelled",
                engine,
            ) from error
        except BaseException:
            self._not_started(engine)
            raise
        return engine

    def _not_started(self, engine):
        if self.active.get((engine.kind, engine.req)) is engine and engine.end is None:
            del self.active[(engine.kind, engine.req)]
            engine.upload.state = "kept" if engine.restart else "idle"
            self._free_bind(engine)

    def _command(self, handler, args, timeout, engine=None):
        """Sends a RUN or CANCEL and a PING behind it, and waits for the PONG: the
        kernel answers the command before the PING, with an event only to reject it.
        """
        run = self.run
        handle = run._queue_command(handler, None, args)
        ping = run._queue_command(self.constants["KM_HANDLER_PING"], None)
        if engine is not None:
            engine.commands = (handle, ping)
        run.wait(ping, timeout)
        self._answer(handle)

    def _answer(self, handle):
        """Settles a command that was answered (a later PONG came): EngineRejected if
        the kernel rejected it."""
        run = self.run
        event = run.event(handle)
        run._settle(handle)
        run.forget([handle])
        if event is not None and event.kind == "rejected":
            raise EngineRejected(handle, event.auxiliary, run._rejections)

    # Events ---------------------------------------------------------------------

    def take(self, rows):
        """Files the engines' events of a block (index, req, status, ticks); returns
        the other rows."""
        kind = rows[:, 2] & 0xFF
        mine = (kind >= self.first_kind) & (kind <= self.last_kind)
        if not mine.any():
            return rows
        for _, req, status, ticks in rows[mine].tolist():
            self._file(req, status, ticks)
        return rows[~mine]

    def _file(self, req, status, ticks):
        family, kind = self.kinds[status & 0xFF]
        engine = self.active.get((family, req))
        if engine is None:
            self.unclaimed.append((family, req, status, ticks))
            return
        auxiliary = status >> 8
        if kind == "error":
            reason, where = auxiliary & 0xFF, auxiliary >> 8
            event = EngineEvent(
                engine, kind, reason=reason, reason_name=self._name(reason)
            )
            if family == "stream":
                event = event._replace(pc=where)
            else:
                event = event._replace(
                    node=None if where == self.constants["KM_GRAPH_NO_NODE"] else where
                )
        elif kind == "end" and family == "graph":
            event = EngineEvent(engine, kind, registers=(auxiliary, ticks))
        else:
            event = EngineEvent(
                engine, kind, value=auxiliary if kind == "round" else ticks
            )
        engine.events.append(event)
        if kind in ("end", "error"):
            engine.end = event
            del self.active[(family, req)]
            self._ended(engine, event)

    def _name(self, reason):
        return self.reasons.get(reason, "unknown")

    def _ended(self, engine, event):
        """The upload is idle again, or kept: a graph started from an upload that
        passes the check with the keep bit is kept whatever its end, unless the
        kernel did not take it in (a bad range or binding block, a full kept table),
        and a restarted one stays kept."""
        self._free_bind(engine)
        upload = engine.upload
        rejected = event.kind == "error" and event.reason in (
            self.constants["KM_STREAM_ERROR_HEADER"],
            self.constants["KM_STREAM_ERROR_KEPT_FULL"],
        )
        kept = engine.kind == "graph" and not engine.drop
        kept = kept and (
            engine.restart or (upload.keep and upload.valid and not rejected)
        )
        upload.state = "kept" if kept else "idle"

    def _free_bind(self, engine):
        if engine.bind is not None:
            self._region().free(engine.bind)
            engine.bind = None

    def wait(self, engine, timeout):
        """The end event of `engine`; EngineError for an error, EngineTimeout."""
        event = self._wait(engine, timeout)
        if event.kind == "error":
            raise EngineError(event)
        return event

    def _wait(self, engine, timeout):
        run = self.run
        run._hand()
        deadline = time.monotonic() + timeout
        while engine.end is None:
            run._check_alive()
            left = deadline - time.monotonic()
            if left <= 0:
                raise EngineTimeout(
                    f"{engine.kind} {engine.req}: no end event in {timeout} s", engine
                )
            run.port.publish()  # an engine's posts wait for event room
            run._step_any(np.asarray([run._handed - 1], np.int64), left)
        return engine.end
