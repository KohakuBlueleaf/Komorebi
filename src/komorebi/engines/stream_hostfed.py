"""Host-fed streams: the simulator's interpreter runs the stream on the host and submits
its commands through a `Run`'s host ring, so a stream gives the same commands and
registers host-fed as when the dispatcher runs it."""

import time

from komorebi.engines import graph as gr
from komorebi.engines import stream as st
from komorebi.engines.graph_hostfed import HostfedCommand, HostfedResult
from komorebi.errors import CommandTimeout, WaitTimeout


def is_stream(obj):
    """Whether `obj` is a Stream or a stream's upload."""
    if isinstance(obj, st.Stream):
        return True
    data = obj if isinstance(obj, bytes | bytearray) else b""
    return len(data) >= 4 and int.from_bytes(data[:4], "little") == st.MAGIC


def run_hostfed(run, stream, bindings=None, timeout=10.0):
    """Runs `stream` (a Stream or its upload) on `run`; returns a `HostfedResult`.
    A command's timeout ends the stream with reason 8; WaitTimeout if it does not end
    within `timeout` seconds."""
    if bindings:
        raise ValueError("a stream takes no bindings")
    data = stream.encode() if isinstance(stream, st.Stream) else bytes(stream)
    bad = st.check(data, len(data))
    if bad:
        raise st.StreamFormatError(bad)
    h = st._header(data)
    ops, templates = st._parse(data, h)
    sim = _Hostfed(run, ops, templates, h)
    return sim.go(time.monotonic() + timeout)


class _Hostfed(st._Sim):
    """The simulator's stream with SUBMIT and ENQUEUE sent through a Run: a command is
    pending until its event arrives, a result word is the latest one that arrived."""

    def __init__(self, run, ops, templates, h):
        super().__init__(ops, templates, h, run.cores, None, None, None)
        self.run = run
        self.flight = {}  # handle -> (pc, wants the result)
        self.hostfed = []

    def post(self, kind, auxiliary):
        """An event as the kernel posts it: an EVENT's or HALT's full register in
        `extra`, as the kernel's ticks."""
        extra = 0
        if kind in (st.EVENT_STREAM, st.EVENT_STREAM_END):
            extra = self.registers[self.ops[self.pc].a]
        self.events.append(gr.Event(kind, self.h["req"], auxiliary, extra, None))

    def do_submit(self, o):
        t, registers = self.templates[o.immediate], self.registers
        first = (
            t.rec[2] & 0xFFFF
            if t.first_register == st.TEMPLATE_NONE
            else registers[t.first_register]
        )
        count = (
            t.rec[2] >> 16
            if t.count_register == st.TEMPLATE_NONE
            else registers[t.count_register]
        )
        if st.group_outside(first, count, self.core_count):
            return self.fail(st.ERROR_GROUP, self.pc)
        self._send(self.patched(t), first, count, False)
        return "ok"

    def do_enqueue(self, o):
        count = self.registers[o.a]
        if count == 0 or count > self.core_count:
            return self.fail(st.ERROR_NEVER_FITS, self.pc)
        self._send(self.patched(self.templates[o.immediate]), None, count, True)
        return "ok"

    def _send(self, words, first, count, queued):
        handler, flags = words[1] & 0xFF, words[1] >> 8 & 0xFF
        args = [words[i] for i in st.ARG_WORD]
        result = bool(flags & st.FLAG_RESULT)
        if queued:
            handle = self.run.submit(handler, count, args, result=result, queued=True)
        else:
            excl = bool(flags & st.FLAG_EXCLUSIVE)
            handle = self.run.submit(handler, (first, count), args, excl, result=result)
        self.flight[handle] = (self.pc, result)
        self.pending += 1
        rec = HostfedCommand(
            self.pc, 1, handle, handler, first, count, flags & ~st.FLAG_QUEUE,
            tuple(args), queued,
        )  # fmt: skip
        self.hostfed.append(rec)

    def go(self, deadline):
        while True:
            if self.error is not None:  # a command's deadline passed
                return self.finish(deadline)
            self.now += 1
            state = self.step_stream()
            if state in ("error", "end"):
                return self.finish(deadline)
            if state == "stall":
                if not self.flight:  # a WAIT or HALT with nothing pending never stalls
                    raise RuntimeError("the stream stalled with nothing in flight")
                self.wait(deadline)
            elif time.monotonic() > deadline:
                raise WaitTimeout("the stream did not end in time")

    def wait(self, deadline):
        """Waits for the first command in flight, then takes every other completion
        already seen."""
        left = deadline - time.monotonic()
        try:
            event = self.run.wait_any(self.flight, timeout=max(left, 0.0))
        except CommandTimeout as err:
            pc = self.flight.pop(err.handle, (self.pc,))[0]
            self.fail(st.ERROR_DEADLINE, pc)
            return
        self.done(event.handle, event)
        for handle in sorted(self.flight):
            event = self.run.event(handle)
            if event is not None:
                self.run.forget([handle])
                self.done(handle, event)

    def done(self, handle, event):
        _, result = self.flight.pop(handle)
        self.pending -= 1
        if result:
            self.last_result = event.result

    def finish(self, deadline):
        status = "ended" if self.error is None else "error"
        if self.error and self.error.reason != st.ERROR_DEADLINE and self.flight:
            self.run.drain(timeout=max(deadline - time.monotonic(), 0.0))
        return HostfedResult(
            status, self.error, list(self.registers), 1, self.hostfed, self.events
        )
