"""The host's side of the dispatch kernel's engines: uploads placed in the kernel's
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
from komorebi.engines import graph as gr
from komorebi.engines import stream as st
from komorebi.errors import (
    EngineError,
    EngineRefused,
    EngineTimeout,
    KomorebiError,
)

LINE = 64
READY_WAIT_S = 1.0  # the ready block follows the READY state word closely


def codes(h=None):
    """The host's constants (abi.headers) with those of km_stream.h and km_graph.h."""
    h = h or abi.headers()
    d = h.abi.parent
    return {**h.constants, **abi.constants(h.abi, d / "km_stream.h", d / "km_graph.h")}


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
        names += [w.strip() for w in words[1].split(",")]
    if member not in names:
        raise abi.HeaderError(f"{path}: struct {name}: no uint32_t {member} up front")
    return 4 * names.index(member)


def offsets(h=None):
    """Where the host reads the headers' words: the ready block's region size (a host
    buffer word) and the byte offsets of each kind's req and of a graph's flags."""
    h = h or abi.headers()
    d = h.abi.parent
    ready = abi.struct_fields(h.dispatch, "kmd_ready_t")
    return {
        "ready_bytes": ready.index("upload_bytes"),
        "stream": word_offset(d / "km_stream.h", "km_stream_t", "req"),
        "graph": word_offset(d / "km_graph.h", "km_graph_t", "req"),
        "flags": word_offset(d / "km_graph.h", "km_graph_t", "flags"),
    }


def _lines(n):
    return -(-n // LINE) * LINE


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
class EngineTicket:
    """One start of an engine: `kind`, the `req` of its events, the `upload`, every
    event so far (`events`) and the end or error event once it came (`end`)."""

    kind: str
    req: int
    upload: Upload
    restart: bool = False
    drop: bool = False  # a CANCEL that drops a kept graph
    events: list = field(default_factory=list, repr=False)
    end: object = field(default=None, repr=False)
    bind: int | None = field(default=None, repr=False)  # its binding line


class EngineEvent(NamedTuple):
    """An engine's event. `kind`: "event" (a stream's EVENT, `value` its register),
    "round" (a graph's round ended, `value` rounds done), "end" (a stream: `value`
    HALT's register; a graph: `regs` = (register 0's low 24 bits, register 1)) or
    "error" (`reason`, `reason_name`, and the stream's `pc` or the graph's `node`)."""

    ticket: EngineTicket
    kind: str
    value: int | None = None
    regs: tuple | None = None
    reason: int | None = None
    reason_name: str | None = None
    pc: int | None = None
    node: int | None = None


class Region:
    """First fit over the upload region: pieces of whole lines, kept by offset."""

    def __init__(self, size):
        self.size = size
        self.used = []  # (offset, bytes), sorted

    def take(self, n):
        """The offset of `n` free bytes (a multiple of LINE), or None."""
        at = 0
        for offset, m in self.used:
            if offset - at >= n:
                break
            at = offset + m
        if at + n > self.size:
            return None
        bisect.insort(self.used, (at, n))
        return at

    def free(self, offset):
        self.used = [u for u in self.used if u[0] != offset]

    def largest(self):
        ends = [0] + [o + m for o, m in self.used]
        starts = [o for o, _ in self.used] + [self.size]
        return max(s - e for s, e in zip(starts, ends))


class Engines:
    """The uploads and engines of one Run, and the routing of their events."""

    def __init__(self, run):
        self.run = run
        c = codes(run.h)
        self.c = c
        self.at = offsets(run.h)
        self.reasons = {
            v: n[len("KM_SERR_") :].lower() for n, v in c.items() if "_SERR_" in n
        }
        self.kinds = {
            c["KM_EV_STREAM"]: ("stream", "event"),
            c["KM_EV_STREAM_END"]: ("stream", "end"),
            c["KM_EV_STREAM_ERROR"]: ("stream", "error"),
            c["KM_EV_GRAPH"]: ("graph", "round"),
            c["KM_EV_GRAPH_END"]: ("graph", "end"),
            c["KM_EV_GRAPH_ERROR"]: ("graph", "error"),
        }
        self.lo, self.hi = min(self.kinds), max(self.kinds)
        host = [c[k] for k in ("KM_EV_DONE", "KM_EV_TIMEOUT", "KM_EV_PONG")]
        host.append(c["KM_EV_REFUSED"])
        if any(self.lo <= k <= self.hi for k in host):
            raise KomorebiError("an engine's event kind is also a command's")
        self.region = None
        self.uploads = []
        self.active = {}  # (kind, req) -> the ticket of the engine running
        self.unclaimed = []  # (kind, req, status, ticks) of no started engine

    # The region and uploads ----------------------------------------------------

    def _region(self):
        """The region the kernel's ready block names (its size; the host writes by
        offset)."""
        if self.region is None:
            port, at = self.run.port, self.c["KMD_HOST_READY"]
            by = time.monotonic() + READY_WAIT_S
            while port.host_word(at) != self.c["KMD_READY_MARK"]:
                if time.monotonic() > by:
                    raise KomorebiError("the dispatch kernel published no ready block")
            self.region = Region(port.host_word(at + self.at["ready_bytes"]))
        return self.region

    def upload(self, obj, kind=None):
        self.run._check_alive()
        graph = None
        if isinstance(obj, st.Stream):
            data, kind = obj.encode(), "stream"
        elif isinstance(obj, gr.Graph):
            data, kind, graph = obj.encode(), "graph", obj
        else:
            data = bytes(obj)
            if kind not in ("stream", "graph"):
                raise ValueError("encoded bytes need kind='stream' or kind='graph'")
        if not data:
            raise ValueError("an empty upload")
        region = self._region()
        n = _lines(len(data))
        offset = region.take(n)
        if offset is None:
            raise ValueError(
                f"an upload of {len(data)} bytes does not fit: the region has "
                f"{region.size} bytes, the largest free piece {region.largest()}"
            )
        req = self._word(data, self.at[kind])
        keep = False
        valid = True
        if kind == "graph":
            flags = self._word(data, self.at["flags"])
            keep = bool(flags & self.c["KM_GRAPH_KEEP"])
            valid = gr.check(data) is None
        try:
            self.run.port.upload(offset, data + bytes(n - len(data)))
        except BaseException:
            region.free(offset)
            raise
        up = Upload(offset, len(data), kind, req, keep, data=data, graph=graph)
        up.valid = valid  # the kernel's check, which the host does not enforce
        self.uploads.append(up)
        return up

    @staticmethod
    def _word(data, at):
        """The word at byte `at`, or 0 when the upload is shorter."""
        return struct.unpack_from("<I", data, at)[0] if len(data) >= at + 4 else 0

    def release(self, up, timeout):
        self._mine(up)
        if up.state == "running":
            raise KomorebiError(f"the {up.kind} at {up.offset} runs: wait or cancel")
        if up.state == "kept":
            t = EngineTicket("graph", up.req, up, drop=True)
            self.active[("graph", up.req)] = t
            try:
                self._command(self.c["KM_SEG_CANCEL"], [up.req], timeout)
            except BaseException:
                self.active.pop(("graph", up.req), None)
                raise
            self._wait(t, timeout)
        self.uploads.remove(up)
        self._region().free(up.offset)

    def _mine(self, up):
        if not any(u is up for u in self.uploads):
            raise ValueError("not an upload of this run, or released")

    # Starting and cancelling ------------------------------------------------------

    def start(self, up, bindings, kept, timeout):
        self._mine(up)
        if up.state != "idle":
            raise KomorebiError(f"the {up.kind} at {up.offset} is {up.state}")
        self._free_req(up.req)
        if up.kind == "stream":
            if bindings or kept is not None:
                raise ValueError("a stream takes no bindings and is never kept")
            t = EngineTicket("stream", up.req, up)
            return self._run(t, [up.offset, up.bytes], timeout)
        if kept is not None and bool(kept) != up.keep:
            self._set_keep(up, bool(kept))
        t = EngineTicket("graph", up.req, up)
        args = [up.offset, up.bytes, self.c["KM_GRUN_UPLOAD"]]
        return self._run(t, args + self._bindings(t, bindings), timeout)

    def restart(self, req, bindings, timeout):
        up = next((u for u in self.uploads if u.state == "kept" and u.req == req), None)
        if up is None:
            raise ValueError(f"no kept graph that is not running has req {req}")
        t = EngineTicket("graph", req, up, restart=True)
        args = [req, 0, self.c["KM_GRUN_KEPT"]]
        return self._run(t, args + self._bindings(t, bindings), timeout)

    def cancel(self, t, timeout):
        if t.end is not None:
            return  # it has ended: nothing to cancel
        try:
            self._command(self.c["KM_SEG_CANCEL"], [t.req], timeout)
        except EngineRefused as err:
            if t.end is None or err.reason != self.c["KMD_REFUSE_UNKNOWN"]:
                raise  # unless it ended while the command was on its way

    def _free_req(self, req):
        busy = {r for _, r in self.active}
        busy |= {u.req for u in self.uploads if u.state == "kept"}
        if req in busy:
            raise ValueError(f"req {req} belongs to a running or kept engine")

    def _set_keep(self, up, keep):
        at = self.at["flags"]
        if up.bytes < at + 4:
            raise ValueError("the upload is too short to carry the keep bit")
        line = bytearray(up.data[:LINE].ljust(LINE, b"\0"))
        flags = struct.unpack_from("<I", line, at)[0]
        bit = self.c["KM_GRAPH_KEEP"]
        struct.pack_into("<I", line, at, flags | bit if keep else flags & ~bit)
        self.run.port.upload(up.offset, bytes(line))
        up.data = bytes(line[: len(up.data)]) + up.data[LINE:]
        up.keep = keep

    def _bindings(self, t, bindings):
        """[binding block offset, count]: the values in one line of the region."""
        graph = t.upload.graph
        if isinstance(bindings, dict) or (graph is not None and not bindings):
            if graph is None:
                raise ValueError("bindings by name need an upload made from a Graph")
            bindings = graph.binding_values(**(bindings or {}))
        values = [int(v) for v in bindings]
        if len(values) > self.c["KM_GRAPH_BINDINGS"]:
            raise ValueError(f"{len(values)} bindings, at most 16")
        if any(not 0 <= v <= abi.M32 for v in values):
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
        t.bind = offset
        return [offset, len(values)]

    def _run(self, t, args, timeout):
        """KM_SEG_RUN for ticket `t`; its events are routed to it from now on."""
        key = (t.kind, t.req)
        self.active[key] = t
        state = t.upload.state
        t.upload.state = "running"
        try:
            self._command(self.c["KM_SEG_RUN"], args, timeout)
        except BaseException:
            if self.active.get(key) is t and t.end is None:
                del self.active[key]
                t.upload.state = state
                self._free_bind(t)
            raise
        return t

    def _command(self, seg, args, timeout):
        """Sends a RUN or CANCEL and a PING behind it, and waits for the PONG: the
        kernel answers the command before the PING, with an event only to refuse it.
        """
        run = self.run
        ticket = run._queue_command(seg, None, args)
        run.wait(run._queue_command(self.c["KM_SEG_PING"], None), timeout)
        ev = run.event(ticket)
        run._settle(ticket)
        run.forget([ticket])
        if ev is not None and ev.kind == "refused":
            raise EngineRefused(ticket, ev.aux, run._refusals)

    # Events ---------------------------------------------------------------------

    def take(self, rows):
        """Files the engines' events of a block (index, req, status, ticks); returns
        the other rows."""
        kind = rows[:, 2] & 0xFF
        mine = (kind >= self.lo) & (kind <= self.hi)
        if not mine.any():
            return rows
        for _, req, status, ticks in rows[mine].tolist():
            self._file(req, status, ticks)
        return rows[~mine]

    def _file(self, req, status, ticks):
        family, kind = self.kinds[status & 0xFF]
        t = self.active.get((family, req))
        if t is None:
            self.unclaimed.append((family, req, status, ticks))
            return
        aux = status >> 8
        if kind == "error":
            reason, where = aux & 0xFF, aux >> 8
            ev = EngineEvent(t, kind, reason=reason, reason_name=self._name(reason))
            if family == "stream":
                ev = ev._replace(pc=where)
            else:
                ev = ev._replace(
                    node=None if where == self.c["KM_GRAPH_NO_NODE"] else where
                )
        elif kind == "end" and family == "graph":
            ev = EngineEvent(t, kind, regs=(aux, ticks))
        else:
            ev = EngineEvent(t, kind, value=aux if kind == "round" else ticks)
        t.events.append(ev)
        if kind in ("end", "error"):
            t.end = ev
            del self.active[(family, req)]
            self._ended(t, ev)

    def _name(self, reason):
        return self.reasons.get(reason, "unknown")

    def _ended(self, t, ev):
        """The upload is idle again, or kept: a graph started from an upload that
        passes the check with the keep bit is kept whatever its end, unless the
        kernel did not take it in (a bad range or binding block, a full kept table),
        and a restarted one stays kept."""
        self._free_bind(t)
        up = t.upload
        refused = ev.kind == "error" and ev.reason in (
            self.c["KM_SERR_HEADER"],
            self.c["KM_SERR_KEPT_FULL"],
        )
        kept = t.kind == "graph" and not t.drop
        kept = kept and (t.restart or (up.keep and up.valid and not refused))
        up.state = "kept" if kept else "idle"

    def _free_bind(self, t):
        if t.bind is not None:
            self._region().free(t.bind)
            t.bind = None

    def wait(self, t, timeout):
        """The end event of `t`; EngineError for an error, EngineTimeout."""
        ev = self._wait(t, timeout)
        if ev.kind == "error":
            raise EngineError(ev)
        return ev

    def _wait(self, t, timeout):
        run = self.run
        run._hand()
        by = time.monotonic() + timeout
        while t.end is None:
            run._check_alive()
            left = by - time.monotonic()
            if left <= 0:
                raise EngineTimeout(f"{t.kind} {t.req}: no end event in {timeout} s", t)
            run.port.publish()  # an engine's posts wait for event room
            run._step_any(np.asarray([run._handed - 1], np.int64), left)
        return t.end
