"""The exceptions of a run, each with the facts that name what went wrong."""


class KomorebiError(RuntimeError):
    """Base of every error a session or run raises."""


class ProgramError(KomorebiError):
    """The program could not be launched (a kernel failed to compile, a bad argument).

    No dispatcher image was started.
    """


class KernelInstructionError(ProgramError):
    """A kernel's build holds an instruction its core lacks; nothing was loaded.
    `findings` lists each one (thread, instruction, function, source line)."""

    def __init__(self, program, findings):
        first = findings[0]
        where = f" at {first.line}" if first.line else ""
        super().__init__(
            f"{program}: {first.source} on {first.thread} holds {first.instruction} "
            f"({first.group}, which the core lacks) in {first.function}{where}; "
            f"{len(findings)} such instruction(s) in all. Not loaded "
            "(docs/reference/host.md, 'Instructions the cores lack')."
        )
        self.findings = findings


class KernelSizeError(ProgramError):
    """A program's kernels do not fit the session's kernel buffer; nothing was loaded.
    `sizes` is the build's `KernelSizes` (total None when the runtime did not say)."""

    def __init__(self, program, sizes, most, refused=None):
        thread = max(sizes.text, key=sizes.text.get)
        total = f"{sizes.total} B" if sizes.total else "more"
        advice = (
            f"The session did not follow them ({refused})"
            if refused
            else "Give the session more (Session(kernel_bytes=...))"
        )
        super().__init__(
            f"{program}: its kernels take {total} of the {sizes.limit} B kernel buffer "
            f"(largest: {thread}, {sizes.text[thread]} B of text); L1 allows up to "
            f"{most} B. {advice}, or build a kernel for size (Program(opt={{'reader': "
            "'Os', 'writer': 'Os'}))."
        )
        self.sizes = sizes


class L1BudgetError(ProgramError):
    """A program's buffers, control blocks and overlay slots do not fit a core's L1;
    nothing was loaded. `table` is the L1 budget table of one core."""

    def __init__(self, program, table, held=0, refused=None, taken=0):
        kept = (
            f" The session keeps {held:,} B of every core's L1 at the top (counted in "
            "the table)."
            if held
            else ""
        )
        if taken:
            kept += (
                f" {taken:,} B of it the session held before this load (its L1 "
                "tensors, a run not stopped): free them (del the tensor, run.stop()) "
                "or load the program of most L1 first."
            )
        smaller = f" ({refused})" if refused else ""
        super().__init__(
            f"{program}: its buffers and slots do not fit a core's L1. Fewer or "
            "smaller slots (Program.slots(count=..., bytes=...)), fewer buffer pages, "
            f"or a smaller kernel buffer{smaller}.{kept} The L1 budget of a core:\n"
            f"{table}"
        )
        self.table = table
        self.held = held
        self.taken = taken


class L1LayoutError(L1BudgetError):
    """Programs that share cores declare one buffer index two ways: they share one L1
    layout, so an index both declare is declared identically. Nothing was loaded."""

    def __init__(self, reason, table):
        ProgramError.__init__(
            self,
            f"{reason}: programs that share cores share one L1 layout (disjoint "
            f"buffer indices, or one index declared identically). The L1 budget of a "
            f"core:\n{table}",
        )
        self.table = table


class SlowCodeError(ProgramError):
    """A program's kernels and overlay slots would reach above `line` of L1, where a
    taken branch costs about 4 cycles more; nothing was loaded."""

    def __init__(self, base, size, line):
        super().__init__(
            f"its overlay slots ({size:,} B from {base:#x}, above the kernel buffer) "
            f"would end at {base + size:#x}, above {line:#x} of L1, where every taken "
            "branch costs about 4 cycles more. Fewer or smaller slots "
            "(Program.slots(count=..., bytes=...)) or a smaller kernel buffer; "
            "Session(slow_code=True) loads it anyway."
        )
        self.end = base + size


class CardBusy(KomorebiError):
    """Another live process holds the card's chip lock; opening it would wait for that
    process to exit. `pid` is the holder."""

    def __init__(self, message, pid=None):
        super().__init__(message)
        self.pid = pid


class ChannelError(KomorebiError):
    """The host channel cannot be used as asked: the open logged the driver's warning
    about its NoC address, or a transfer does not fit the session's region."""


class StartError(KomorebiError):
    """The dispatcher did not report READY; `result` is its result block."""

    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result


class GridTimeout(KomorebiError):
    """The Tensix grid's kernels did not return within the session's timeout after the
    stop flag. The session does not close the device; the runtime closes it at the
    process's exit (a timed wait for the cores, then their reset). A stuck thread is
    recovered by the next open; a compute unit left running is not (a card reset)."""


class ThreadFailed(KomorebiError):
    """An X280 thread did not return in time or ended with a library error; the
    program's other threads were stopped. `cluster` names it, `results` holds every
    thread's ThreadResult by cluster."""

    def __init__(self, message, cluster, results):
        super().__init__(message)
        self.cluster = cluster
        self.results = results


class WaitTimeout(KomorebiError):
    """A host-side wait ran out; the run is still alive. `pending` lists the handles."""

    def __init__(self, message, pending=()):
        super().__init__(message)
        self.pending = list(pending)


class CommandRejected(KomorebiError):
    """The dispatcher consumed the command and submitted nothing; the run goes on."""

    def __init__(self, handle, reason, names=None):
        name = (names or {}).get(reason, "unknown")
        super().__init__(
            f"command {handle} rejected by the dispatcher (reason {reason}: {name})"
        )
        self.handle = handle
        self.reason = reason
        self.reason_name = name


class KernelFailed(KomorebiError):
    """The dispatcher finished with an error; the run is stopped.

    `result` is the decoded result block, `cores` the cores that did not report.
    """

    def __init__(self, message, result=None, cores=()):
        super().__init__(message)
        self.result = result
        self.cores = list(cores)


# What a link core's give-up says (KMD_LINK_*), in a CommandTimeout's text.
LINK_SAYS = {
    "gave": "its command given up, the core left it",
    "hung": "did not leave the command: reset it with run.reset_link",
    "lost": "held this command's traffic, did not leave: reset it with run.reset_link",
}


class CommandTimeout(KernelFailed):
    """A command's group missed its deadline: command, slot, sequence, group, and cores:
    after the stop, those that did not report and where each waited; `given_up`: the
    dispatcher gave it up, the run goes on, the cores as its inspect read them;
    `recovered`: those reset again; `ranks`: by flat rank; `links`: (link core, what).
    """

    def __init__(
        self,
        handle,
        slot,
        sequence,
        group,
        cores,
        result=None,
        given_up=False,
        recovered=(),
        ranks=False,
        links=(),
    ):
        named = "; ".join(describe_core(core) for core in cores) or "none named"
        unit = "ranks" if ranks else "cores"
        first, count = group if group else (None, None)
        what = (
            "was given up, the run goes on; at the deadline"
            if given_up
            else "timed out; not reported"
        )
        back = (
            f"; recovered (reset and loaded again): {unit} {list(recovered)}"
            if recovered
            else ""
        )
        on = "".join(
            f"; link core {k}: {LINK_SAYS.get(say, say)}"
            for k, say in dict(links).items()
        )
        super().__init__(
            f"command {handle} {what}: slot {slot}, sequence {sequence}, "
            f"group first {first} count {count}; {unit}: {named}{back}{on}",
            result=result,
            cores=cores,
        )
        self.recovered = list(recovered)
        self.links = list(links)
        self.handle = handle
        self.slot = slot
        self.sequence = sequence
        self.group = group
        self.given_up = given_up


def _no_free_slot(gave):
    """The words after an overlay's give-up for want of a slot."""
    slots = gave.get("slots", 0)
    return (
        f": no free slot for overlay {gave.get('overlay')}: {slots} slot"
        f"{'' if slots == 1 else 's'}, all held"
    )


def describe_core(core):
    """One core of a not-reported list, or of an inspect, as text: its index (its flat
    rank and island in a run over islands), its give-ups, the runtime waits of its
    reader and writer, and of compute's threads while they run a command."""
    where = f"(island {core.get('island')}, core {core['core']})"
    if "island" not in core:
        parts = [f"core {core['core']}"]
    elif core.get("rank") is None:
        parts = [f"link core {core.get('link', core['core'])} {where}"]
    else:
        parts = [f"rank {core['rank']} {where}"]
    line = (core.get("live") or {}).get("line")
    for kernel in ("reader", "writer"):
        gave = core.get(kernel)
        if gave:
            at = f" at line {line['line']}" if line and line["kernel"] == kernel else ""
            parts.append(f"{kernel} gave up ({gave['code']}) in {gave['where']}{at}")
            if gave["where"] == "overlayslots":
                parts[-1] += _no_free_slot(gave)
    for kernel, words in (core.get("live") or {}).items():
        if not isinstance(words, dict) or not words.get("where"):
            continue
        if kernel in ("reader", "writer") or words["sequence"]:
            parts.append(f"{kernel} waits in {words['where']}")
    if len(parts) == 1:
        parts.append("no give-up recorded")
    return " ".join(parts)


class EngineRejected(CommandRejected):
    """The dispatcher rejected a RUN or CANCEL: nothing was started or cancelled,
    the run goes on. `reason` and `reason_name` as for CommandRejected."""


class EngineError(KomorebiError):
    """An engine ended with an error event: its `reason` and `reason_name`, the
    stream's program counter `pc` or the graph's `node` (None when the error is not
    tied to one), and the `event`. The run goes on unless the kernel stopped too."""

    def __init__(self, event):
        handle = event.handle
        where = f"pc {event.pc}" if event.pc is not None else f"node {event.node}"
        super().__init__(
            f"{handle.kind} {handle.req} ended with error {event.reason} "
            f"({event.reason_name}) at {where}"
        )
        self.event = event
        self.handle = handle
        self.reason = event.reason
        self.reason_name = event.reason_name
        self.pc = event.pc
        self.node = event.node


class EngineTimeout(WaitTimeout):
    """An engine's end event did not arrive in time; the engine may still run."""

    def __init__(self, message, handle):
        super().__init__(message)
        self.handle = handle
