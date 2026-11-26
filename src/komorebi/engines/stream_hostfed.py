"""Host-fed streams: the simulator's interpreter runs the stream on the host and submits
its commands through a `Run`'s host ring, so a stream gives the same commands and
registers host-fed as when the dispatcher runs it."""

import time

from komorebi.engines import graph as graph_module
from komorebi.engines import stream as stream_module
from komorebi.engines.graph_hostfed import HostfedCommand, HostfedResult
from komorebi.errors import CommandTimeout, WaitTimeout


def is_stream(candidate):
    """Whether `candidate` is a Stream or a stream's upload."""
    if isinstance(candidate, stream_module.Stream):
        return True
    data = candidate if isinstance(candidate, bytes | bytearray) else b""
    return len(data) >= 4 and int.from_bytes(data[:4], "little") == stream_module.MAGIC


def run_hostfed(run, stream, bindings=None, timeout=10.0):
    """Runs `stream` (a Stream or its upload) on `run`; returns a `HostfedResult`.
    A command's timeout ends the stream with reason 8; WaitTimeout if it does not end
    within `timeout` seconds."""
    if bindings:
        raise ValueError("a stream takes no bindings")
    if isinstance(stream, stream_module.Stream):
        data = stream.encode(run._handler)  # its handlers' numbers in this run
    else:
        data = bytes(stream)
    bad = stream_module.check(data, len(data))
    if bad:
        raise stream_module.StreamFormatError(bad)
    header = stream_module._header(data)
    ops, templates = stream_module._parse(data, header)
    simulation = _Hostfed(run, ops, templates, header)
    return simulation.go(time.monotonic() + timeout)


class _Hostfed(stream_module._Sim):
    """The simulator's stream with SUBMIT and ENQUEUE sent through a Run: a command is
    pending until its event arrives, a result word is the latest one that arrived."""

    def __init__(self, run, ops, templates, header):
        # a job's cores: the run's ranks (the grid's, less a Tensix dispatcher's core)
        super().__init__(ops, templates, header, run.ranks, None, None, None)
        self.run = run
        self.flight = {}  # handle -> (pc, wants the result)
        self.hostfed = []

    def post(self, kind, auxiliary):
        """An event as the kernel posts it: an EVENT's or HALT's full register in
        `extra`, as the kernel's ticks."""
        extra = 0
        if kind in (stream_module.EVENT_STREAM, stream_module.EVENT_STREAM_END):
            extra = self.registers[self.ops[self.pc].a]
        self.events.append(
            graph_module.Event(kind, self.h["req"], auxiliary, extra, None)
        )

    def do_submit(self, operation):
        template, registers = self.templates[operation.immediate], self.registers
        first = (
            template.rec[2] & 0xFFFF
            if template.first_register == stream_module.TEMPLATE_NONE
            else registers[template.first_register]
        )
        count = (
            template.rec[2] >> 16
            if template.count_register == stream_module.TEMPLATE_NONE
            else registers[template.count_register]
        )
        if stream_module.group_outside(first, count, self.run.ranks, template.flags):
            return self.fail(stream_module.ERROR_GROUP, self.pc)
        self._send(self.patched(template), first, count, False)
        return "ok"

    def do_enqueue(self, operation):
        count = self.registers[operation.a]
        if count == 0 or count > self.core_count:
            return self.fail(stream_module.ERROR_NEVER_FITS, self.pc)
        self._send(self.patched(self.templates[operation.immediate]), None, count, True)
        return "ok"

    def _send(self, words, first, count, queued):
        handler, flags = words[1] & 0xFF, words[1] >> 8 & 0xFF
        named = self.run.field_handler(handler, words[1] >> 24)
        args = [words[i] for i in stream_module.ARG_WORD]
        result = bool(flags & stream_module.FLAG_RESULT)
        if queued:
            handle = self.run.submit(named, count, args, result=result, queued=True)
        elif stream_module.cluster_of(first, count) is not None:
            cluster = stream_module.cluster_of(first, count)
            handle = self.run.submit(named, args=args, result=result, cluster=cluster)
        else:
            exclusive = bool(flags & stream_module.FLAG_EXCLUSIVE)
            handle = self.run.submit(
                named, (first, count), args, exclusive, result=result
            )
        self.flight[handle] = (self.pc, result)
        self.pending += 1
        record = HostfedCommand(
            self.pc, 1, handle, handler, first, count,
            flags & ~stream_module.FLAG_QUEUE, tuple(args), queued, words[1] >> 24,
        )  # fmt: skip
        self.hostfed.append(record)

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
        except CommandTimeout as error:
            pc = self.flight.pop(error.handle, (self.pc,))[0]
            self.fail(stream_module.ERROR_DEADLINE, pc)
            return
        self.done(event.handle, event)
        for handle in sorted(self.flight):
            event = self.run.event(handle)
            if event is not None:
                self.run.forget([handle])
                self.done(handle, event)

    def done(self, handle, event):
        pc, result = self.flight.pop(handle)
        self.pending -= 1
        if event.kind == "timeout":  # a thread's late command, which ends no run
            if self.error is None:
                self.fail(stream_module.ERROR_DEADLINE, pc)
        elif result:
            self.last_result = event.result

    def finish(self, deadline):
        status = "ended" if self.error is None else "error"
        if (
            self.error
            and self.error.reason != stream_module.ERROR_DEADLINE
            and self.flight
        ):
            self.run.drain(timeout=max(deadline - time.monotonic(), 0.0))
        return HostfedResult(
            status, self.error, list(self.registers), 1, self.hostfed, self.events
        )
