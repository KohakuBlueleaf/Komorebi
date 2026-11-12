"""The exceptions of a run, each with the facts that name what went wrong."""


class KomorebiError(RuntimeError):
    """Base of every error a session or run raises."""


class ProgramError(KomorebiError):
    """The program could not be launched (a kernel failed to compile, a bad argument).

    No dispatch kernel image was started.
    """


class StartError(KomorebiError):
    """The dispatch kernel did not report READY; `result` is its result block."""

    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result


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
