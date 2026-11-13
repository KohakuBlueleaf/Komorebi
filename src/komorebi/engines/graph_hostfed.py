"""Host-fed graphs: the graph engine runs on the host and submits its commands through a
`Run`'s host ring, so a graph gives the same commands and registers host-fed as when
the dispatcher runs it.
"""

import time
from collections import namedtuple

from komorebi.engines import graph as graph_module
from komorebi.engines import stream as stream_module
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
    if isinstance(graph, graph_module.Graph):
        data = graph.encode()
        if bindings is None or isinstance(bindings, dict):
            bindings = graph.binding_values(**(bindings or {}))
    else:
        data = bytes(graph)
    bad = graph_module.check(data)
    if bad:
        raise graph_module.GraphFormatError(bad)
    executor = engine if engine is not None else NativeExecutor()
    if executor.start(data, list(bindings or ()), run.cores):
        raise ValueError(f"{len(bindings)} bindings, at most {graph_module.BINDINGS}")
    return _Hostfed(run, executor, data).go(time.monotonic() + timeout)


class _Hostfed:
    """The executor's submit function over a Run, and the loop of steps and waits."""

    def __init__(self, run, executor, data):
        self.run, self.executor = run, executor
        self.req = graph_module.HEADER.unpack_from(data)[6]
        self.flight = {}  # handle -> (node, wants the result)
        self.commands, self.events = [], []

    def submit(self, node, words):
        """Queues the command on the Run (a queued one by its count: the kernel picks
        its group, so its `first` is None here)."""
        handler, flags = words[1] & 0xFF, words[1] >> 8 & 0xFF
        first, count = words[2] & 0xFFFF, words[2] >> 16
        queued = bool(flags & stream_module.FLAG_QUEUE)
        args = [words[i] for i in stream_module.ARG_WORD]
        result = bool(flags & stream_module.FLAG_RESULT)
        if queued:
            first = None
            handle = self.run.submit(handler, count, args, result=result, queued=True)
        else:
            exclusive = bool(flags & stream_module.FLAG_EXCLUSIVE)
            handle = self.run.submit(
                handler, (first, count), args, exclusive, result=result
            )
        self.flight[handle] = (node, result)
        record = HostfedCommand(
            node, self.executor.runs, handle, handler, first, count,
            flags & ~stream_module.FLAG_QUEUE, tuple(args), queued,
        )  # fmt: skip
        self.commands.append(record)
        return graph_module.SUBMITTED

    def go(self, deadline):
        executor = self.executor
        while True:
            code = executor.step(self.submit)
            if code == graph_module.ROUND:
                self.events.append(
                    graph_module.Event(
                        graph_module.EVENT_GRAPH, self.req, executor.runs - 1, 0, None
                    )
                )
            elif code in (graph_module.END, graph_module.ERROR):
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
            node = self.flight.pop(err.handle, (graph_module.NO_NODE,))[0]
            self.executor.stop(stream_module.ERROR_DEADLINE, node)
            return
        self.done(event.handle, event)
        for handle in sorted(self.flight):
            event = self.run.event(handle)
            if event is not None:
                self.run.forget([handle])
                self.done(handle, event)

    def done(self, handle, event):
        node, result = self.flight.pop(handle)
        self.executor.complete(node, event.result if result else 0)

    def finish(self, code, deadline):
        executor = self.executor
        registers = list(executor.registers)
        if code == graph_module.END:
            self.events.append(
                graph_module.Event(
                    graph_module.EVENT_GRAPH_END,
                    self.req,
                    registers[0] & 0xFFFFFF,
                    registers[1],
                    None,
                )
            )
            return HostfedResult(
                "ended", None, registers, executor.runs, self.commands, self.events
            )
        error = graph_module.GraphError(*executor.error)
        auxiliary = error.reason | error.node << 8
        self.events.append(
            graph_module.Event(
                graph_module.EVENT_GRAPH_ERROR, self.req, auxiliary, 0, None
            )
        )
        if error.reason != stream_module.ERROR_DEADLINE and self.flight:
            self.run.drain(timeout=max(deadline - time.monotonic(), 0.0))
        return HostfedResult(
            "error", error, registers, executor.runs, self.commands, self.events
        )
