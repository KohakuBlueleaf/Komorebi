"""Host-fed graphs: the graph engine runs on the host and issues its records through a
`Run`'s host ring, so a graph gives the same records and registers host-fed as when
the dispatch kernel runs it.
"""

import time
from collections import namedtuple

from komorebi.engines import graph as gr
from komorebi.engines import stream as st
from komorebi.engines.graph_native import NativeExecutor
from komorebi.errors import RecordTimeout, WaitTimeout

HostfedRecord = namedtuple(
    "HostfedRecord", "node run ticket seg first count flags args queued"
)
HostfedResult = namedtuple("HostfedResult", "status error regs runs records events")


def run_hostfed(run, graph, bindings=None, timeout=10.0, engine=None):
    """Runs `graph` (a Graph or upload bytes) on `run` with `bindings` (a dict by name
    for a Graph, else a list); returns a `HostfedResult`. A record's timeout ends the
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
    """The executor's issue function over a Run, and the loop of steps and waits."""

    def __init__(self, run, eng, data):
        self.run, self.eng = run, eng
        self.req = gr.HEADER.unpack_from(data)[6]
        self.flight = {}  # ticket -> (node, wants the result)
        self.records, self.events = [], []

    def issue(self, node, words):
        """Queues the record on the Run (a queued one by its count: the kernel picks
        its team, so its `first` is None here)."""
        seg, flags = words[1] & 0xFF, words[1] >> 8 & 0xFF
        first, count = words[2] & 0xFFFF, words[2] >> 16
        queued = bool(flags & st.F_QUEUE)
        args = [words[i] for i in st.ARG_WORD]
        result = bool(flags & st.F_RESULT)
        if queued:
            first = None
            ticket = self.run.issue(seg, count, args, result=result, queued=True)
        else:
            excl = bool(flags & st.F_EXCL)
            ticket = self.run.issue(seg, (first, count), args, excl, result=result)
        self.flight[ticket] = (node, result)
        rec = HostfedRecord(
            node, self.eng.runs, ticket, seg, first, count, flags & ~st.F_QUEUE,
            tuple(args), queued,
        )  # fmt: skip
        self.records.append(rec)
        return gr.ISSUED

    def go(self, deadline):
        eng = self.eng
        while True:
            code = eng.step(self.issue)
            if code == gr.ROUND:
                self.events.append(
                    gr.Event(gr.EV_GRAPH, self.req, eng.runs - 1, 0, None)
                )
            elif code in (gr.END, gr.ERROR):
                return self.finish(code, deadline)
            if self.flight:
                self.wait(deadline)
            elif time.monotonic() > deadline:
                raise WaitTimeout("the graph did not end in time")

    def wait(self, deadline):
        """Waits for the first record in flight to finish, then takes every other
        completion already seen."""
        left = deadline - time.monotonic()
        try:
            ev = self.run.wait_any(self.flight, timeout=max(left, 0.0))
        except RecordTimeout as err:
            node = self.flight.pop(err.ticket, (gr.NO_NODE,))[0]
            self.eng.stop(st.ERR_DEADLINE, node)
            return
        self.done(ev.ticket, ev)
        for ticket in sorted(self.flight):
            ev = self.run.event(ticket)
            if ev is not None:
                self.run.forget([ticket])
                self.done(ticket, ev)

    def done(self, ticket, ev):
        node, result = self.flight.pop(ticket)
        self.eng.complete(node, ev.result if result else 0)

    def finish(self, code, deadline):
        eng = self.eng
        regs = list(eng.regs)
        if code == gr.END:
            self.events.append(
                gr.Event(gr.EV_GRAPH_END, self.req, regs[0] & 0xFFFFFF, regs[1], None)
            )
            return HostfedResult(
                "ended", None, regs, eng.runs, self.records, self.events
            )
        error = gr.GraphError(*eng.error)
        aux = error.reason | error.node << 8
        self.events.append(gr.Event(gr.EV_GRAPH_ERROR, self.req, aux, 0, None))
        if error.reason != st.ERR_DEADLINE and self.flight:
            self.run.drain(timeout=max(deadline - time.monotonic(), 0.0))
        return HostfedResult("error", error, regs, eng.runs, self.records, self.events)
