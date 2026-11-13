"""Host-fed graphs: the graph engine runs on the host and submits its commands through a
`Run`'s host ring, so a graph gives the same commands and registers host-fed as when
the dispatcher runs it.
"""

import time
from collections import namedtuple

from komorebi.engines import graph as gr
from komorebi.engines import stream as st
from komorebi.engines.graph_native import NativeExecutor
from komorebi.errors import CommandTimeout, WaitTimeout

HostfedCommand = namedtuple(
    "HostfedCommand", "node run handle handler first count flags args queued"
)
HostfedResult = namedtuple(
    "HostfedResult", "status error registers runs commands events"
)


def run_hostfed(run, graph, bindings=None, timeout=10.0, engine=None):
    """Runs `graph` (a Graph or upload bytes) on `run` with `bindings` (a dict by name
    for a Graph, else a list); returns a `HostfedResult`. A command's timeout ends the
    graph with reason 8; WaitTimeout if it does not end within `timeout` seconds."""
    if isinstance(graph, gr.Graph):
        data = graph.encode()
        if bindings is None or isinstance(bindings, dict):
            bindings = graph.binding_values(**(bindings or {}))
    else:
        data = bytes(graph)
    bad = gr.check(data)
    if bad:
        raise gr.GraphFormatError(bad)
    eng = engine if engine is not None else NativeExecutor()
    if eng.start(data, list(bindings or ()), run.cores):
        raise ValueError(f"{len(bindings)} bindings, at most {gr.BINDINGS}")
    return _Hostfed(run, eng, data).go(time.monotonic() + timeout)


class _Hostfed:
    """The executor's submit function over a Run, and the loop of steps and waits."""

    def __init__(self, run, eng, data):
        self.run, self.eng = run, eng
        self.req = gr.HEADER.unpack_from(data)[6]
        self.flight = {}  # handle -> (node, wants the result)
        self.commands, self.events = [], []

    def submit(self, node, words):
        """Queues the command on the Run (a queued one by its count: the kernel picks
        its group, so its `first` is None here)."""
        handler, flags = words[1] & 0xFF, words[1] >> 8 & 0xFF
        first, count = words[2] & 0xFFFF, words[2] >> 16
        queued = bool(flags & st.FLAG_QUEUE)
        args = [words[i] for i in st.ARG_WORD]
        result = bool(flags & st.FLAG_RESULT)
        if queued:
            first = None
            handle = self.run.submit(handler, count, args, result=result, queued=True)
        else:
            excl = bool(flags & st.FLAG_EXCLUSIVE)
            handle = self.run.submit(handler, (first, count), args, excl, result=result)
        self.flight[handle] = (node, result)
        rec = HostfedCommand(
            node, self.eng.runs, handle, handler, first, count, flags & ~st.FLAG_QUEUE,
            tuple(args), queued,
        )  # fmt: skip
        self.commands.append(rec)
        return gr.SUBMITTED

    def go(self, deadline):
        eng = self.eng
        while True:
            code = eng.step(self.submit)
            if code == gr.ROUND:
                self.events.append(
                    gr.Event(gr.EVENT_GRAPH, self.req, eng.runs - 1, 0, None)
                )
            elif code in (gr.END, gr.ERROR):
                return self.finish(code, deadline)
            if self.flight:
                self.wait(deadline)
            elif time.monotonic() > deadline:
                raise WaitTimeout("the graph did not end in time")

    def wait(self, deadline):
        """Waits for the first command in flight to finish, then takes every other
        completion already seen."""
        left = deadline - time.monotonic()
        try:
            event = self.run.wait_any(self.flight, timeout=max(left, 0.0))
        except CommandTimeout as err:
            node = self.flight.pop(err.handle, (gr.NO_NODE,))[0]
            self.eng.stop(st.ERROR_DEADLINE, node)
            return
        self.done(event.handle, event)
        for handle in sorted(self.flight):
            event = self.run.event(handle)
            if event is not None:
                self.run.forget([handle])
                self.done(handle, event)

    def done(self, handle, event):
        node, result = self.flight.pop(handle)
        self.eng.complete(node, event.result if result else 0)

    def finish(self, code, deadline):
        eng = self.eng
        registers = list(eng.registers)
        if code == gr.END:
            self.events.append(
                gr.Event(
                    gr.EVENT_GRAPH_END,
                    self.req,
                    registers[0] & 0xFFFFFF,
                    registers[1],
                    None,
                )
            )
            return HostfedResult(
                "ended", None, registers, eng.runs, self.commands, self.events
            )
        error = gr.GraphError(*eng.error)
        auxiliary = error.reason | error.node << 8
        self.events.append(gr.Event(gr.EVENT_GRAPH_ERROR, self.req, auxiliary, 0, None))
        if error.reason != st.ERROR_DEADLINE and self.flight:
            self.run.drain(timeout=max(deadline - time.monotonic(), 0.0))
        return HostfedResult(
            "error", error, registers, eng.runs, self.commands, self.events
        )
