"""The exceptions of a run, each with the facts that name what went wrong."""


class KomorebiError(RuntimeError):
    """Base of every error a session or run raises."""


class ProgramError(KomorebiError):
    """The program could not be launched (a kernel failed to compile, a bad argument).

    No dispatch kernel image was started.
    """


class CardBusy(KomorebiError):
    """Another live process holds the card's chip lock; opening it would wait for that
    process to exit. `pid` is the holder."""

    def __init__(self, message, pid=None):
        super().__init__(message)
        self.pid = pid


class StartError(KomorebiError):
    """The dispatch kernel did not report READY; `result` is its result block."""

    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result


class GridTimeout(KomorebiError):
    """The Tensix grid's kernels did not return within the session's bound after the
    stop word. The device is left open (closing it would wait for the grid); the card
    may need a reset."""


class WaitTimeout(KomorebiError):
    """A host-side wait ran out; the run is still alive. `pending` lists the tickets."""

    def __init__(self, message, pending=()):
        super().__init__(message)
        self.pending = list(pending)


class RecordRefused(KomorebiError):
    """The dispatch kernel consumed the command and issued nothing; the run goes on."""

    def __init__(self, ticket, reason, names=None):
        name = (names or {}).get(reason, "unknown")
        super().__init__(
            f"command {ticket} refused by the dispatch kernel (reason {reason}: {name})"
        )
        self.ticket = ticket
        self.reason = reason
        self.reason_name = name


class KernelFailed(KomorebiError):
    """The dispatch kernel finished with an error; the run is stopped.

    `result` is the decoded result block, `cores` the cores that did not report.
    """

    def __init__(self, message, result=None, cores=()):
        super().__init__(message)
        self.result = result
        self.cores = list(cores)


class RecordTimeout(KernelFailed):
    """A record's team missed its deadline: the command, slot, sequence and team, and
    after the stop the members that did not report and where each was waiting."""

    def __init__(self, ticket, slot, seq, team, cores, result=None):
        named = "; ".join(describe_core(c) for c in cores) or "none named"
        first, count = team if team else (None, None)
        super().__init__(
            f"command {ticket} timed out: slot {slot}, sequence {seq}, "
            f"team first {first} count {count}; not reported: {named}",
            result=result,
            cores=cores,
        )
        self.ticket = ticket
        self.slot = slot
        self.seq = seq
        self.team = team


def describe_core(core):
    """One core of a not-reported list as text: its index and its waits."""
    parts = [f"core {core['core']}"]
    for kernel in ("reader", "writer"):
        gave = core.get(kernel)
        if gave:
            parts.append(f"{kernel} gave up ({gave['code']}) in {gave['where']}")
    if len(parts) == 1:
        parts.append("no give-up recorded")
    return " ".join(parts)


class EngineRefused(RecordRefused):
    """The dispatch kernel refused a RUN or CANCEL: nothing was started or cancelled,
    the run goes on. `reason` and `reason_name` as for RecordRefused."""


class EngineError(KomorebiError):
    """An engine ended with an error event: its `reason` and `reason_name`, the
    stream's program counter `pc` or the graph's `node` (None when the error is not
    tied to one), and the `event`. The run goes on unless the kernel stopped too."""

    def __init__(self, event):
        t = event.ticket
        where = f"pc {event.pc}" if event.pc is not None else f"node {event.node}"
        super().__init__(
            f"{t.kind} {t.req} ended with error {event.reason} ({event.reason_name}) "
            f"at {where}"
        )
        self.event = event
        self.ticket = t
        self.reason = event.reason
        self.reason_name = event.reason_name
        self.pc = event.pc
        self.node = event.node


class EngineTimeout(WaitTimeout):
    """An engine's end event did not arrive in time; the engine may still run."""

    def __init__(self, message, ticket):
        super().__init__(message)
        self.ticket = ticket
