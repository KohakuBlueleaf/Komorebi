"""A run over islands (IslandRun): one Run a piece, the flat group split by island on
the host, events joined, links' calls served."""

import time

import numpy as np

from komorebi import abi
from komorebi.chain import calls as chain_module
from komorebi.device import tensor as tensor_module
from komorebi.device.events import IslandEvent
from komorebi.dsl.front import front
from komorebi.errors import CommandTimeout, KomorebiError, MessagesGaveUp, WaitTimeout
from komorebi.islands import local as island_local
from komorebi.islands import records as island_records
from komorebi.islands.same import RunCalls
from komorebi.islands.trees import TreeCalls
from komorebi.program.addon import code as added
from komorebi.program.addon import program as added_program
from komorebi.program.description import Program
from komorebi.session import state as device_state

GAVE_WAIT = 2.0  # seconds a timeout waits for another island's named give-up


def _named(error, island):
    """`error` with the island it came from: its message starts "island i: " and
    `error.island` is i (once: an error raised again keeps its first island)."""
    if getattr(error, "island", None) is None:
        error.island = island
        if error.args:
            error.args = (f"island {island}: {error.args[0]}", *error.args[1:])
    return error


def _left(deadline):
    return max(deadline - time.monotonic(), 0.0)


def _one_command(handler, firsts, counts, args):
    """(handler, group, words) when submit_many's arguments are one command (one
    handler, one first and count, at most one row of integer words), else None: those
    submit takes as they are (any other form, a wrong one too, goes submit_many's way).
    """
    if not isinstance(handler, str | int):  # a Handler is an int
        number = np.asarray(handler)
        if number.ndim or number.dtype.kind not in "iu":
            return None
        handler = int(number)
    first, count = np.asarray(firsts), np.asarray(counts)
    if first.size != 1 or count.size != 1 or first.ndim > 1 or count.ndim > 1:
        return None
    if first.dtype.kind not in "iu" or count.dtype.kind not in "iu":
        return None
    words = ()
    if args is not None:
        args = np.asarray(args)
        if args.ndim != 2 or args.shape[0] != 1 or args.dtype.kind not in "iu":
            return None
        words = args[0].tolist()
    elif first.ndim + count.ndim == 0:
        return None  # no row at all (submit_many refuses it)
    return handler, (int(first.reshape(-1)[0]), int(count.reshape(-1)[0])), words


def _handler_label(handler):
    """A handler as a line names it: program.method, the method, or its number."""
    method = getattr(handler, "method", handler)
    if hasattr(method, "name"):
        program = getattr(handler, "program", None)
        return f"{program.name}.{method.name}" if program else method.name
    program = getattr(handler, "program", None)
    return f"{program.name} handler {int(handler)}" if program else f"handler {handler}"


class IslandRun(TreeCalls, RunCalls):
    """The runs of one program on every island, as one run over flat ranks. A ticket
    is a command of a group of flat ranks; its parts went to their islands' dispatchers
    and it completes when every part has. Errors name the island (`error.island`)."""

    def __init__(self, graph, runs, linked=False, counts=None):
        self.graph = graph
        self.runs = tuple(runs)
        self._counts = counts  # () -> Crossings of the links, read now (None: no link)
        # a dynamic program's messages cross: a dynamic RUN takes every rank, and link
        # core 0 of each island serves it (a PAGES command of mode MESSAGES)
        self.linked = linked
        self.session = None  # the IslandSession with links: a link core's reset
        self._links_reset = {}  # (island, k) -> its run's link events at its reset
        self.cores = graph.ranks
        self.program = self.runs[0].program
        self._next = 1  # the next ticket
        self._queued = [[] for _ in self.runs]  # each island's queued part handles
        self.queued_on = {}  # a queued ticket's island (_submit_queued)
        self.submit_s = 0.0  # host time in submit() and submit_many(), as Run's
        self.wait_s = 0.0  # host time in wait(), wait_any() and drain()
        self._chain_last = 0  # the last chain's ticket (_chain_after)
        self._timed_out = set()  # tickets whose timeout wait or drain raised (stop)
        self.stop_named = []  # the stop's failures of those tickets, not raised again
        # per island, the handle of each ticket's part on that island (0: none)
        self._parts = [np.zeros(1024, np.int64) for _ in self.runs]
        # a (handler, group)'s first command's crossing: "log", "error" or "off"; a
        # traced run starts "off" (its COUNTS commands would take the trace's slots)
        traced = any(getattr(run.backend, "trace_cap", 0) for run in self.runs)
        self.crossing_line = "off" if traced else "log"
        self._crossed_seen, self._measured = set(), {}
        # an add a later island's device refused: ("number", n) or ("program", p) ->
        # why a command naming it is refused (_half_added_error)
        self._half_added = {}
        for run in self.runs:  # each island's own Run refuses them too
            run.half_added = self._half_added
        for island, run in enumerate(self.runs):  # a time-out names every island's part
            run.far_cores = lambda group, island=island: self._far_cores(island, group)
            run.named_cores = lambda island=island: self._named_cores(island)

    def _check_local(self, handler, group):
        """check_local over this run's islands."""
        island_local.check_local(self.runs, self.graph, handler, group)

    def island(self, island):
        """Island `island`'s own Run (local ranks)."""
        return self.runs[island]

    def _grow(self, tickets):
        while self._next + tickets > len(self._parts[0]):
            self._parts = [
                np.concatenate([parts, np.zeros_like(parts)]) for parts in self._parts
            ]

    def chain(self, calls, mode=0, flags=0, cores=None, prefetch=True):
        """Run.chain over flat ranks: each island's records in its own DRAM, walked on
        its own cores; a mark or a credit goes to a rank of any island. A class's calls
        are described with island 0's program (`cores` their flat ranks)."""
        if any(isinstance(call, front.HandlerCall) for call in calls):
            group = (0, self.cores) if cores is None else tuple(cores)
            starts = tuple(self.graph.first(i) for i in range(len(self.runs)))
            if getattr(self.session, "card_dram", False):
                starts = (0,)  # one tensor in the card's DRAM: no copy to gather
            for call in calls:
                if isinstance(call, front.HandlerCall):
                    self._check_local(call.method, call.cores or group)
                    island_local.check_direct(
                        self.graph, call.method, call.cores or group, call.values
                    )
            calls = self.runs[0]._class_calls(calls, None, prefetch, group, starts)
        chain_module.build(calls)
        return self._chain_parts(calls, mode, flags)

    def _chain_parts(self, calls, mode, flags):
        """Each island's records (Run.chain), at one bank address on every island when
        the session's region has room (chain_records), so a thread's or a dispatcher's
        one chain command (a7) names them on every island."""
        region = getattr(self.session, "records", None)
        at = region.take(64 * len(calls)) if region is not None else None
        chains = []
        try:
            for run in self.runs:  # each kept, to be freed if a later one fails
                made = run.chain(calls, mode, flags, records_at=at)
                chains.append(made)
        except KomorebiError as error:
            for made in chains:
                made.free()
            if at is not None:
                region.give(at, 64 * len(calls))
            raise self._error(error, len(chains)) from None
        made = chain_module.IslandChain(chains, chains[0].group)
        if at is not None:
            made.held = (region, at, 64 * len(calls))
        return made

    def _submit_chain(self, chain, words):
        """An island chain as one ticket a piece of kChainRecordsMax records (those
        before the last waited for): a part an island it touches, each with its own
        island's records (the command's a7), after the chain before it is done."""
        pieces = chain.chains[0].pieces
        for piece in range(pieces):
            self._chain_after(self._chain_last)
            parts = self.graph.split(chain.group)
            for island, _first, _count in parts[1:]:  # refused before any is queued
                one = chain.chains[island]
                self._part_checked(
                    island, one.handler, one.group, one.args(words, piece), True
                )
            self._grow(1)
            ticket = self._next
            for island, _first, _count in parts:
                one = chain.chains[island]
                try:
                    handle = self.runs[island].submit(
                        one.handler,
                        one.group,
                        one.args(words, piece),
                        exclusive=True,
                        part=True,
                    )
                except KomorebiError as error:
                    raise self._error(error, island) from None
                self._parts[island][ticket] = handle
            self._next += 1
            self._chain_last = ticket
            if piece + 1 < pieces:
                self.wait(ticket)
        first = chain.chains[0]  # a class's: what it sends direct and writes (Run's)
        tensor_module.sent_direct(first.written, first.direct)
        return ticket

    def _chain_after(self, ticket, timeout=10.0):
        """Holds a chain until the chain before it (`ticket`, 0: none) is done on every
        island (its event kept for wait()): a record's marks word, its own within a
        command, is not counted by two chains at once, as one island's dispatcher
        orders them."""
        if not ticket:
            return
        self.send()
        deadline = time.monotonic() + timeout
        for island, run in enumerate(self.runs):
            handle = int(self._parts[island][ticket])
            row = handle - run._base
            if handle == 0 or row < 0 or run._gone[row] or run._arrived(handle):
                continue
            try:
                run._wait(np.asarray([handle], np.int64), _left(deadline), False)
            except KomorebiError as error:
                raise self._error(error, island) from None

    def submit(
        self,
        handler,
        group=None,
        args=(),
        exclusive=False,
        result=False,
        queued=False,
        cluster=None,
        block=None,
        **words,
    ):
        """A command for `handler` on flat ranks [first, first + count) (default
        all): a part an island it touches; its ticket. A chain: one ticket. `queued`:
        `group` a count, all on one island (_submit_queued). `cluster`: a thread's flat
        number, that island's thread alone. `block`: each part's own copy."""
        if isinstance(handler, chain_module.IslandChain):
            return self._submit_chain(handler, words)
        if words:
            raise TypeError(f"submit: unknown arguments {sorted(words)}")
        started = time.perf_counter()
        try:
            if queued and cluster is None:
                return self._submit_queued(
                    handler, group, args, exclusive, result, block
                )
            return self._submit(handler, group, args, exclusive, result, cluster, block)
        finally:
            self.submit_s += time.perf_counter() - started

    def _submit(self, handler, group, args, exclusive, result, cluster, block):
        """submit's command (not a chain): its parts queued, its ticket; each part's
        block in its own island's DRAM (Run.submit's a7 and a8)."""
        if cluster is not None:
            return self._submit_thread(
                handler, cluster, group, args, exclusive, result, block
            )
        self._check_half_added([handler])
        group = (0, self.cores) if group is None else tuple(group)
        parts = self.graph.split(group)
        self._check_local(handler, group)
        island_local.check_direct(self.graph, handler, group, args)
        for island, _first, _count in parts[1:]:  # refused before any part is queued
            self._part_checked(island, handler, group, args, exclusive, result)
        self._x280_holder(handler, parts)
        self._check_quota(handler, parts)
        if self._dynamic(handler):
            self._serve_links(group)
        before = self._first_use(handler, group)  # read before the command is queued
        self._grow(1)
        ticket = self._next
        for island, _first, _count in parts:
            try:  # the flat group word unchanged, marked a part: the dispatcher cuts
                # its part and does not cross the line (the other parts went as well)
                handle = self.runs[island].submit(
                    handler,
                    group,
                    args,
                    exclusive=exclusive,
                    result=result,
                    block=block,
                    part=True,
                )
            except KomorebiError as error:
                raise self._error(error, island) from None
            self._parts[island][ticket] = handle
        self._next += 1
        if before is not None:  # earlier commands still out: counted with it
            label = _handler_label(handler)
            self._measured[ticket] = (label, group, before, self._out_before(ticket))
        return ticket

    def _submit_queued(self, handler, count, args, exclusive, result, block):
        """A queued command (Run.submit's `queued`): whole on the island with the
        fewest of this run's queued commands unfinished (the lowest such island) among
        those with `count` ranks, whose dispatcher places it in its own ranks; one
        ticket, one part, `group(ticket)` flat once done."""
        if getattr(self.session, "split", 1) > 1:
            raise KomorebiError(
                "submit(..., queued=True) on a card split into islands: not proven on "
                "a split card yet; run.island(i).submit(..., queued=True) queues it on "
                "island i's ranks"
            )
        if exclusive:
            raise ValueError("a queued command cannot be exclusive")
        ranks = [self.graph.count(i) for i in range(len(self.runs))]
        count = max(ranks) if count is None else int(count)
        fits = [i for i, held in enumerate(ranks) if count <= held]
        if count < 1 or not fits:
            raise ValueError(
                f"queued count {count}: a queued command runs on one island's ranks "
                f"(the islands have {ranks})"
            )
        for island, run in enumerate(self.runs):
            held = np.asarray(self._queued[island], np.int64)
            self._queued[island] = (
                held[~run._arrived(held)].tolist() if held.size else []
            )
        island = min(fits, key=lambda i: (len(self._queued[i]), i))
        self._check_half_added([handler])
        self._grow(1)
        ticket = self._next
        try:
            handle = self.runs[island].submit(
                handler, count, args, result=result, queued=True, block=block
            )
        except KomorebiError as error:
            raise self._error(error, island) from None
        self._parts[island][ticket] = handle
        self._queued[island].append(handle)
        self.queued_on[ticket] = island
        self._next += 1
        return ticket

    def _submit_thread(self, handler, number, group, args, exclusive, result, block):
        """A command to thread `number` (flat over islands, graph.thread): its
        island's run gives it to that island's work cluster; one ticket, one part."""
        if number.__class__ is not int:
            raise ValueError(f"cluster {number!r}: a thread's flat number")
        try:
            island, cluster = self.graph.thread(number)
        except ValueError:
            served = [
                f"island {i}: {list(self.graph.thread_numbers(i))}"
                for i in range(len(self.runs))
            ]
            raise ValueError(
                f"cluster {number}: no thread of that number ({'; '.join(served)})"
            ) from None
        self._check_half_added([handler])
        self._grow(1)
        ticket = self._next
        try:
            handle = self.runs[island].submit(
                handler, group, args, exclusive, result, cluster=cluster, block=block
            )
        except KomorebiError as error:
            raise self._error(error, island) from None
        self._parts[island][ticket] = handle
        self._next += 1
        return ticket

    def _first_use(self, handler, group):
        """The links' counts before a (handler, group)'s first command (a COUNTS
        command an island), or None: seen before, no link, crossing_line "off", or
        a run whose messages cross."""
        if self.crossing_line not in ("log", "error", "off"):
            raise ValueError(
                f"crossing_line={self.crossing_line!r}: 'log', 'error' or 'off'"
            )
        key = (_handler_label(handler), group)
        if self.crossing_line == "off" or self._counts is None or self.linked:
            return None  # a run whose messages cross: its counts wait behind the RUN
        if key in self._crossed_seen:
            return None
        self._crossed_seen.add(key)
        return self.crossings()

    def _out_before(self, ticket):
        """The tickets before `ticket` whose events the host has not yet filed."""
        out = set()
        for island, run in enumerate(self.runs):
            handles = self._parts[island][1:ticket]
            rows = handles - run._base
            inside = (handles != 0) & (rows >= 0)  # below the base: filed and gone
            filed = np.zeros_like(inside)
            held = inside & (rows < run._has.size)
            filed[held] = run._has[rows[held]]
            late = inside & ~filed
            out.update((np.flatnonzero(late) + 1).tolist())
        return len(out)

    def _crossed(self, tickets, events=None):
        """After the events of first commands among `tickets` were seen done: the
        counts read again, a line each when pages, asks or puts crossed; under
        "error" raised here, the events on the error's `events`."""
        mine = [int(t) for t in tickets if int(t) in self._measured]
        if not mine:
            return
        after, raised = self.crossings(), []
        for ticket in mine:
            label, group, before, out = self._measured.pop(ticket)
            moved = (after - before).total()
            if not (moved["pages"] or moved["asks"] or moved["puts"]):
                continue
            others = out + self._next - ticket - 1  # before it, and queued after it
            many = "s" if others > 1 else ""
            flight = f" (with {others} other command{many} in flight)" if others else ""
            line = (
                f"{label} on ranks {group[0]}..{group[0] + group[1] - 1} moved "
                f"{moved['pages']} pages across the link (asks {moved['asks']}, puts "
                f"{moved['puts']}){flight}; if that is not meant, pass its tensors "
                f"split=0, group=({group[0]}, {group[1]}) for {label}'s group, or "
                "copy=True if every rank reads all of it"
            )
            if self.crossing_line == "error":
                raised.append(f"{line} (ticket {ticket} has run)")
            else:
                getattr(self.session, "log", print)(line)
        if raised:
            error = KomorebiError(
                f"{'; '.join(raised)} (crossing_line='error': its events on .events)"
            )
            error.events = events
            raise error

    def load(self, program, group):
        """Run.load over flat ranks: the cores of `group` hold `program`, one of the
        dict session.load took; a part on each island it touches. Its ticket."""
        group = tuple(group)
        parts = self.graph.split(group)
        for island, _first, _count in parts:  # every part refused before one is sent
            try:
                self.runs[island].load_field(program)
            except KomorebiError as error:
                raise self._error(error, island) from None
        self._grow(1)
        ticket = self._next
        for island, _first, _count in parts:
            try:
                handle = self.runs[island].load(program, group, part=True)
            except KomorebiError as error:
                raise self._error(error, island) from None
            self._parts[island][ticket] = handle
        self._next += 1
        return ticket

    def addon(
        self,
        name,
        parts=None,
        handler=False,
        number=None,
        defines=None,
        timeout=10.0,
        message=False,
        compile_args=None,
    ):
        """Run.addon(name) over islands: a class one more program of every island's
        run, all or none (added_program.add_islands), one field and overlay numbering;
        its tensors made before the load or live. Returns its program field; run.load
        then loads it. Code (a name, `parts` and Run.addon's options): _addon_code."""
        if not isinstance(name, Program):
            options = {"handler": handler, "defines": defines, "message": message}
            options["compile_args"] = compile_args
            options |= {"timeout": timeout, "number": number}
            return self._addon_code(name, parts, **options)
        program = name
        session = self.session
        if session is None:
            raise added.AddError(
                f"{program.name}: a card split into islands (one launch, its programs "
                "in the kernel form)"
            )
        session._hand_rectangles(program)
        try:
            return added_program.add_islands(
                list(self.runs), program, session._island_defines()
            )
        except KomorebiError as error:
            island = getattr(error, "added_island", 0)
            if getattr(error, "added_step", None) == "commit" and island > 0:
                raise self._half_added_error(
                    error, island, f"class {program.name}", ("program", program)
                ) from None
            raise self._error(error, island) from None

    def _addon_code(self, name, parts, timeout=10.0, number=None, **options):
        """Run.addon(name, parts) over islands: every island's checks and build first,
        the number island 0's (or `number`) on every island, then each island's parts
        and entries; the number. A split card: refused as a class is."""
        if parts is None:
            raise TypeError(f"run.addon({name!r}): parts, as Program.overlay's")
        if self.session is None:
            raise added.AddError(
                f"{name}: a card split into islands (one launch, its programs in the "
                "kernel form)"
            )
        ready = []
        for island, run in enumerate(self.runs):
            try:
                ready.append(
                    added.prepared(
                        run,
                        name,
                        parts,
                        options.get("handler", False),
                        number,
                        options.get("defines"),
                        options.get("message", False),
                        options.get("compile_args"),
                    )
                )
            except KomorebiError as error:
                raise self._error(error, island) from None
            number = ready[0][0].number  # island 0's on every island
        for island, run in enumerate(self.runs):
            try:
                added.committed(run, ready[island], timeout)
            except KomorebiError as error:
                if island == 0:
                    raise self._error(error, island) from None
                for one in self.runs:  # given out again on no island
                    for group in one.backend.room.groups:
                        group.numbers.add(number)
                raise self._half_added_error(
                    error, island, f"overlay {name} ({number})", ("number", number)
                ) from None
        return number

    def _half_added_error(self, error, island, what, key):
        """An add island `island`'s device refused after islands 0 .. island - 1 took
        it (their entries written, which no command takes back): `key` marked
        unusable on every island; the AddError that says so."""
        self._half_added[key] = (
            f"{what}: island {island} refused its add after islands 0 to {island - 1} "
            "took it; it is marked unusable on every island"
        )
        refused = added.AddError(
            f"{self._half_added[key]} (a command naming it is refused; the islands "
            f"would not agree on it): island {island}: {error}"
        )
        refused.island = island
        return refused

    def _check_half_added(self, handlers):
        """Refuses a command naming an add marked unusable (_half_added_error)."""
        added.refuse_half_added(self._half_added, handlers)

    @property
    def check_programs(self):
        """Whether the host refuses a command to cores not holding its program (every
        island's run); False: the dispatchers alone refuse."""
        return all(run.check_programs for run in self.runs)

    @check_programs.setter
    def check_programs(self, value):
        for run in self.runs:
            run.check_programs = value

    def forget(self, handles):
        """Run.forget of every part of the tickets `handles` on its island."""
        tickets = [handles] if isinstance(handles, int) else list(handles)
        for island, run in enumerate(self.runs):
            handles = [int(self._parts[island][t]) for t in tickets]
            run.forget([handle for handle in handles if handle])

    def submit_many(self, handler, firsts, counts, args=None, exclusive=False):
        """n commands at once, as Run.submit_many takes them, the groups in flat
        ranks; each island gets its parts in one submit_many. The tickets, a range. A
        dynamic RUN over a link goes as submit, in its row's place (_runs_apart)."""
        started = time.perf_counter()
        try:
            return self._submit_many(handler, firsts, counts, args, exclusive)
        finally:
            self.submit_s += time.perf_counter() - started

    def _submit_many(self, handler, firsts, counts, args, exclusive):
        rows = () if args is None else args  # a class's words: their constants' handler
        named = handler if isinstance(handler, list | tuple) else [handler]
        self._check_half_added(list(np.ravel(np.asarray(named, object))))
        lows, manys = np.broadcast_arrays(np.ravel(firsts), np.ravel(counts))
        for row in range(lows.size):  # a local=True reduce's groups on one island
            self._check_local(named[row % len(named)], (lows[row], manys[row]))
        handler = self.runs[0]._words_built_for(handler, rows, "submit_many")
        one = _one_command(handler, firsts, counts, args)
        if one is not None:  # one command: submit's Python numbers, not numpy's
            handler, group, words = one
            ticket = self.submit(handler, group, words, exclusive=exclusive)
            return range(ticket, ticket + 1)
        scalar = isinstance(handler, str) or np.ndim(handler) == 0
        handlers, first, count = np.broadcast_arrays(
            np.asarray(0 if scalar else handler),
            np.asarray(firsts, np.int64),
            np.asarray(counts, np.int64),
        )
        if args is not None:
            args = np.asarray(args)
            shape = np.broadcast_shapes(first.shape, (args.shape[0],))
            handlers, first, count = (
                np.broadcast_to(column, shape) for column in (handlers, first, count)
            )
        if first.ndim != 1:
            raise ValueError("handlers, firsts, counts and arguments: one row each")
        bad = (count < 1) | (first < 0) | (first + count > self.cores)
        if bad.any():
            i = int(np.argmax(bad))
            raise ValueError(
                f"group ({int(first[i])}, {int(count[i])}) of {self.cores} ranks"
            )
        rows = len(first)
        x280 = getattr(getattr(self.program, "messages", None), "x280_ranks", 0)
        for row in range(rows if x280 else 0):
            self._x280_holder(
                handler if scalar else handlers[row],
                self.graph.split((int(first[row]), int(count[row]))),
            )
        dynamic = [
            row
            for row in range(rows if self.linked else 0)
            if self._dynamic(handler if scalar else handlers[row])
        ]
        if dynamic:
            return self._runs_apart(
                dynamic, handler, scalar, handlers, first, count, args, exclusive
            )
        return self._queue_many(
            handler, scalar, handlers, first, count, args, exclusive
        )

    def _runs_apart(
        self, dynamic, handler, scalar, handlers, first, count, args, exclusive
    ):
        """submit_many's rows in their order: each dynamic RUN over a link as submit
        (its links served first), the rows between as one submit_many each; the
        tickets one range, an island's commands queued in the rows' order."""
        start, at = self._next, 0
        for row in [*dynamic, len(first)]:
            if row > at:
                part = slice(at, row)
                self._queue_many(
                    handler,
                    scalar,
                    None if scalar else handlers[part],
                    first[part],
                    count[part],
                    None if args is None else args[part],
                    exclusive,
                )
            if row < len(first):
                named = handler if scalar else handlers[row].item()
                words = () if args is None else args[row].tolist()
                group = (int(first[row]), int(count[row]))
                self.submit(named, group, words, exclusive=exclusive)
            at = row + 1
        return range(start, self._next)

    def _queue_many(self, handler, scalar, handlers, first, count, args, exclusive):
        """submit_many's rows (checked, none a dynamic RUN over a link) as one
        submit_many an island; the tickets, a range."""
        rows = len(first)
        self._grow(rows)
        tickets = np.arange(self._next, self._next + rows)
        touched = []
        for island in range(len(self.runs)):
            low = np.maximum(first, self.graph.first(island))
            high = np.minimum(
                first + count, self.graph.first(island) + self.graph.count(island)
            )
            if (high > low).any():
                touched.append((island, high > low))
        for island, mine in touched[1:]:  # refused before any island's rows are sent
            try:
                self.runs[island]._many_words(
                    handler if scalar else handlers[mine],
                    first[mine],
                    count[mine],
                    None if args is None else args[mine],
                    exclusive,
                )
            except KomorebiError as error:
                raise self._error(error, island) from None
        for island, mine in touched:
            run = self.runs[island]
            named = handler if scalar else handlers[mine]
            try:  # the flat group words unchanged, marked parts (as in submit)
                handles = run.submit_many(
                    named,
                    first[mine],
                    count[mine],
                    None if args is None else args[mine],
                    exclusive,
                    part=True,
                )
            except KomorebiError as error:
                raise self._error(error, island) from None
            self._parts[island][tickets[mine]] = np.asarray(handles, np.int64)
        self._next += rows
        return range(int(tickets[0]), int(tickets[0]) + rows) if rows else range(0)

    def _part_checked(self, island, handler, group, args, exclusive, result=False):
        """Island `island`'s part of a command checked as its submit checks it, nothing
        queued (a block's bytes are checked by the first part's), named by island."""
        try:
            self.runs[island]._submit_words(
                handler, group, args, exclusive, result, False, None, None, True
            )
        except KomorebiError as error:
            raise self._error(error, island) from None

    def _x280_holder(self, handler, parts):
        """Refuses a dynamic RUN with no part on island 0 when the program has X280
        ranks: island 0 holds them, and only its part starts their thread."""
        messages = getattr(self.program, "messages", None)
        if messages is None or not messages.x280_ranks or len(self.runs) < 2:
            return
        number, _ = self.runs[0]._handler(handler)
        if number in messages.handlers and all(part[0] != 0 for part in parts):
            raise ValueError(
                f"{_handler_label(handler)}: a dynamic RUN of a program with X280 "
                "ranks takes ranks of island 0, which holds the X280 ranks"
            )

    def _dynamic(self, handler):
        """Whether `handler` is a dynamic handler of a program whose messages cross."""
        if not self.linked:
            return False
        number, _ = self.runs[0]._handler(handler)
        return number in self.program.messages.handlers

    def _check_quota(self, handler, parts):
        """A dynamic RUN whose ranks, counted as the device counts them (each island's
        part without its counter tiles, then the X280 ranks), leave every rank no
        send (max_live // ranks == 0): refused by name before anything is queued. A
        load checks each island's own ranks only."""
        number, field = self.runs[0]._handler(handler)
        messages = getattr(self.runs[0].programs[field], "messages", None)
        if messages is None or number not in messages.handlers:
            return
        ranks = sum(max(count - messages.counters, 0) for _, _, count in parts)
        ranks += messages.x280_ranks
        if messages.max_live // max(ranks, 1) == 0:
            raise ValueError(
                f"{_handler_label(handler)}: a dynamic RUN over {ranks} ranks "
                f"with max_live {messages.max_live} leaves every rank no send (each "
                f"rank's quota is max_live // the RUN's ranks, over every island): "
                f"raise max_live (and queue_messages) to {ranks} or more, or run it "
                "over fewer ranks"
            )

    def _serve_links(self, group):
        """Before a dynamic RUN over a link: its group is every rank, and link core 0
        of each island (each link core, over more than two islands; between two cards
        the link cores of its message services) is given the RUN's group word to serve
        its messages across."""
        if group != (0, self.cores):
            raise ValueError(
                f"group {group}: a dynamic RUN of a program with hold_links=True takes "
                f"every rank (0, {self.cores}); hold_links=False (the default) runs "
                f"any group"
            )
        h = abi.headers()
        words = [0] * 10
        words[h["KM_PAGES_FROM"]] = group[0] | group[1] << 16
        words[h["KM_PAGES_COUNTER"]] = h["KM_PAGES_MESSAGES"] << 24 | 1 << 16
        many = len(self.runs) > 2
        # two cards: the link cores of every service (a link each, two a link core)
        services = self.session.message_services() if self.session else 1
        cores = (services + 1) // 2
        for island, run in enumerate(self.runs):
            try:
                run._queue_command(
                    h["KM_HANDLER_PAGES"],
                    self.graph.link_group(island, 0, None if many else cores),
                    words,
                )
            except KomorebiError as error:
                raise self._error(error, island) from None

    def send(self):
        """Stores every island's queued commands on its card, so that the islands run
        their parts at once (a run over a link needs every part started): all handed
        first, then stored by one native call (every island's lines, one fence), then
        the events read meanwhile filed."""
        for island, run in enumerate(self.runs):
            try:
                run._hand()
            except KomorebiError as error:
                raise self._error(error, island) from None
        ports = [run.port for run in self.runs]
        together = getattr(type(ports[0]), "store_together", None)
        if together is not None and all(type(port) is type(ports[0]) for port in ports):
            # one native call: every island's lines, then one fence
            read = together(ports, [run._store_upto() for run in self.runs])
        else:
            read = []
            for island, run in enumerate(self.runs):
                try:
                    read.append(run._store_handed())
                except KomorebiError as error:
                    raise self._error(error, island) from None
        for island, (run, rows) in enumerate(zip(self.runs, read)):
            if len(rows):
                try:
                    run._take(rows)
                except KomorebiError as error:
                    raise self._error(error, island) from None

    def wait(self, handles, timeout=10.0):
        """Sends what is queued and waits, within one `timeout`, for every part of the
        tickets `handles`. Per ticket the tuple of its parts' events in island order
        (one, or a list for a list); the first holds the group's first rank, so its
        result word is the command's, as on one island."""
        started = time.perf_counter()
        try:
            return self._wait_tickets(handles, timeout)
        finally:
            self.wait_s += time.perf_counter() - started

    def _wait_tickets(self, tickets, timeout):
        single = isinstance(tickets, int | np.integer)
        wanted = np.asarray([tickets] if single else list(tickets), np.int64)
        if not wanted.size:
            return []
        if wanted.min() < 1 or wanted.max() >= self._next:
            raise ValueError("a ticket that was never submitted")
        deadline = time.monotonic() + timeout
        self.send()
        self._store(wanted, deadline)
        parts = []
        for island, run in enumerate(self.runs):
            handles = self._parts[island][wanted]
            mine = handles != 0
            if not mine.any():
                continue
            try:  # every island's parts arrived before any is forgotten: a ticket
                # whose wait runs out on a later island is waited for again whole
                run.wait_kept(handles[mine].tolist(), _left(deadline))
            except WaitTimeout as error:
                error.pending = self._tickets(island, error.pending)
                raise self._error(error, island) from None
            except KomorebiError as error:
                raise self._raised(error, island, deadline) from None
            parts.append((island, mine, handles[mine].tolist()))
        events = {}
        for island, mine, handles in parts:
            got = [self.runs[island].event(handle) for handle in handles]
            self.runs[island].forget(handles)
            for ticket, event in zip(wanted[mine].tolist(), got, strict=True):
                events.setdefault(ticket, []).append(event)
        out = [IslandEvent(events[ticket]) for ticket in wanted.tolist()]
        self._crossed(wanted, out[0] if single else out)
        return out[0] if single else out

    def _store(self, wanted, deadline):
        """Until the tickets' parts arrived or the deadline: one native round an island
        in turn (store what its ring has room for, file its events), so that every
        island is fed while the host waits; Run.wait then returns or raises."""
        pending = {}
        for island, run in enumerate(self.runs):  # what the stores filed is not waited
            handles = self._parts[island][wanted]
            handles = np.unique(handles[handles != 0])
            pending[island] = handles[~run._arrived(handles)]
        while any(handles.size for handles in pending.values()):
            if time.monotonic() > deadline:
                return
            for island, handles in pending.items():
                if not handles.size:
                    continue
                run = self.runs[island]
                try:
                    run._check_alive()
                    run._step(handles, int(handles[-1]), 0.0)
                except KomorebiError as error:
                    raise self._raised(error, island, deadline) from None
                pending[island] = handles[~run._arrived(handles)]

    def inspect(self, group=None, timeout=10.0):
        """While the run goes on: what each rank of `group` (flat first, count; default
        every rank) is doing, as Run.inspect reads it, asked of each island's
        dispatcher in turn within one `timeout`; each core's dict also holds its `rank`
        and `island`."""
        group = (0, self.cores) if group is None else tuple(group)
        deadline = time.monotonic() + timeout
        out = []
        for island, first, count in self.graph.split(group):
            cores = self.graph.rank_cores[island][first : first + count]
            if cores[-1] - cores[0] + 1 != count:
                raise ValueError(
                    f"ranks {group}: not one run of island {island}'s cores"
                )
            try:
                read = self.runs[island].inspect(
                    (self.graph.first(island) + first, count), _left(deadline)
                )
            except KomorebiError as error:
                raise self._error(error, island) from None
            out.extend(self._ranked(island, read))
        return out

    def reset_core(self, core, timeout=10.0, channels=(), drops=()):
        """Resets the core of flat rank `core` and loads its kernels again while the
        others run, as Run.reset_core does on its island (`channels`, `drops` that
        island's): its result with `rank` and `island`. KomorebiError names the rank
        if the core did not come back."""
        rank = core
        island, local = self.graph.resolve(rank)
        core = self.graph.rank_cores[island][local]
        try:
            done = self.runs[island].reset_core(core, timeout, channels, drops)
        except KomorebiError as error:
            if error.args:
                where = f"rank {rank} (island {island}, core {core})"
                error.args = (f"{where}: {error.args[0]}", *error.args[1:])
            error.island = island
            raise
        return {**done, "rank": rank, "island": island}

    def reset_link(self, island, k, timeout=10.0):
        """Resets link core k of island `island` by the host and opens its connections
        again (the session's reset_link): {island, core, by, seconds, ...}."""
        if self.session is None:
            raise ValueError("a run without links has no link core to reset")
        done = self.session.reset_link(island, k, timeout)
        self._links_reset[island, k] = len(self.runs[island].link_events)
        return done

    @property
    def links_hung(self):
        """(island, link core) of each link core a give-up found hung (its dispatcher's
        KMD_EVENT_LINK) and not reset since: run.reset_link(island, k) recovers it."""
        out = []
        for island, run in enumerate(self.runs):
            for at, event in enumerate(run.link_events):
                key = (island, event["link"])
                if (
                    event["what"] in ("hung", "lost")
                    and at >= self._links_reset.get(key, 0)
                    and key not in out
                ):
                    out.append(key)
        return out

    def _far_cores(self, island, group):
        """At a time-out of island `island`'s run: every other island's part of flat
        `group` as its dispatcher's inspect of the deadline read it (a complete timeout
        of that part's cores), its cores ranked and marked `far`; [] where none is."""
        out = []
        for other, _first, _count in self.graph.split(tuple(group)):
            run = self.runs[other]
            local = run._local(group)
            if other == island or local[1] == 0 or not hasattr(run.backend, "inspect"):
                continue
            block = device_state.inspect(run.backend.inspect(local[1])["block"], run.h)
            if (
                block["complete"]
                and block["kind"] == "timeout"
                and (block["first"], block["count"]) == local
            ):
                out += [
                    {**core, "far": True}
                    for core in self._ranked(other, block["cores"])
                ]
        return out

    def _named_cores(self, island):
        """Island `island`'s own cores that a give-up of any island's run named (its
        own part's at the deadline, or another island's part read there)."""
        return {
            core["core"]
            for other, run in enumerate(self.runs)
            for _event, cores in run._given_up.values()
            for core in cores
            if core.get("island", other) == island and "core" in core
        }

    def _ranked(self, island, cores):
        """Core dicts of island `island` (its own core numbers in `core`) with their
        flat `rank` and `island`; a link core's rank is None, its `link` its index. A
        core of another island's part (`far`) is kept as it is."""
        mine, links = self.graph.rank_cores[island], self.graph.link_cores[island]
        first = self.graph.first(island)
        out = []
        for core in cores:
            if core.get("far"):
                out.append(core)
                continue
            named = {**core, "rank": None, "island": island}
            if core["core"] in mine:
                named["rank"] = first + mine.index(core["core"])
            elif core["core"] in links:
                named["link"] = links.index(core["core"])
            out.append(named)
        return out

    def _error(self, error, island):
        """`error` from island `island`'s run as this run's: named by its island, and a
        CommandTimeout by its ticket and the flat ranks of its cores."""
        if isinstance(error, CommandTimeout) and getattr(error, "island", None) is None:
            cores = self._ranked(island, error.cores)
            ranks = {
                core["core"]: core["rank"] for core in cores if not core.get("far")
            }
            tickets = self._tickets(island, [error.handle])
            CommandTimeout.__init__(
                error,
                tickets[0] if tickets else error.handle,
                error.slot,
                error.sequence,
                error.group,
                cores,
                error.result,
                error.given_up,
                [ranks.get(core, core) for core in error.recovered],
                ranks=True,
                links=error.links,
            )
            if isinstance(error, MessagesGaveUp):  # its core's rank and island
                (core,) = self._ranked(island, [{"core": error.gave["core"]}])
                error.gave |= {"rank": core["rank"], "island": island}
                error.gave |= {"link": core["link"]} if "link" in core else {}
                error.name_gave()
        return _named(error, island)

    @property
    def recovery(self):
        """The kept reset lists (Run.recovery) as one card's dict: every island's lists,
        bytes and seconds together, `cores` by flat rank; `islands` each island's own.
        None when an island kept none (recovery_refused names it)."""
        kept = [run.recovery for run in self.runs]
        if any(one is None for one in kept):
            return None
        offsets = {one["offset"] for one in kept}
        out = device_state.IslandResult(
            offset=offsets.pop() if len(offsets) == 1 else None,
            bytes=sum(one["bytes"] for one in kept),
            lists=[entry for one in kept for entry in one["lists"]],
            cores=[
                self.graph.first(island) + mine.index(core)
                for island, (one, mine) in enumerate(zip(kept, self.graph.rank_cores))
                for core in one["cores"]
                if core in mine
            ],
            seconds=sum(one["seconds"] for one in kept),
        )
        out.islands = kept
        return out

    @property
    def recoveries(self):
        """Every island's give-ups whose cores its dispatcher reset, island order:
        {island, ticket, handle, cores (its own core numbers), ranks, text}, the text
        that of the give-up, naming the flat ranks."""
        out = []
        for island, run in enumerate(self.runs):
            for entry in run.recoveries:
                tickets = self._tickets(island, [entry["handle"]])
                given_up = run.given_up(entry["handle"])
                text = f"island {island}: {entry['text']}"
                if given_up is not None:
                    text = str(self._error(given_up, island))
                ranked = self._ranked(island, [{"core": c} for c in entry["cores"]])
                out.append(
                    {"island": island, "ticket": next(iter(tickets), None)}
                    | entry
                    | {"ranks": [core["rank"] for core in ranked], "text": text}
                )
        return out

    def _tickets(self, island, handles):
        """The tickets whose part on `island` has one of these handles."""
        parts = self._parts[island][: self._next]
        return [int(t) for t in np.flatnonzero(np.isin(parts, handles)) if t]

    def drain(self, timeout=10.0):
        """Sends everything queued and waits, within one `timeout`, for every
        island's events; returns how many events it waited for, every island's
        together. `drain_given_up`: the parts of them given up, every island's."""
        started = time.perf_counter()
        try:
            return self._drain(timeout)
        finally:
            self.wait_s += time.perf_counter() - started

    def _drain(self, timeout):
        deadline = time.monotonic() + timeout
        self.send()
        waited = 0
        for island, run in enumerate(self.runs):
            try:
                waited += run.drain(_left(deadline))
            except KomorebiError as error:
                raise self._raised(error, island, deadline) from None
        self._crossed(list(self._measured))  # every command done: its events by event()
        return waited

    def _raised(self, error, island, deadline):
        """`error` of island `island`'s part as this run's (_error); a timeout first
        takes another island's named give-up, and its ticket is kept for the stop."""
        raised = self._error(*self._gave_elsewhere(error, island, deadline))
        if isinstance(raised, CommandTimeout):
            self._timed_out.add(raised.handle)
        return raised

    def _gave_elsewhere(self, error, island, deadline):
        """(error, island): another island's MessagesGaveUp when that island's
        dispatcher named who gave the run up (its drain ends at its own deadline), for
        a timeout or a give-up in a wait on another island (kWhereFar: a consequence),
        unless that one is such a wait too; else `error` as it came."""
        far = (abi.headers()["kWhereFar"], "far")  # the word, or its name
        waited = isinstance(error, MessagesGaveUp) and error.gave.get("where") in far
        if not isinstance(error, CommandTimeout) or (
            isinstance(error, MessagesGaveUp) and not waited
        ):
            return error, island
        for other, run in enumerate(self.runs):
            if other == island:
                continue
            try:
                run.drain(min(_left(deadline), GAVE_WAIT))
            except MessagesGaveUp as named:
                if not waited or named.gave.get("where") not in far:
                    return named, other
            except KomorebiError:
                continue
        return error, island

    @property
    def drain_given_up(self):
        return sum(run.drain_given_up for run in self.runs)

    def given_up(self, handle):
        """The CommandTimeout of ticket `handle`'s first given-up part, naming its
        island and the wait each rank sat in (event() after drain()), or None."""
        for island, run in enumerate(self.runs):
            part = int(self._parts[island][handle])
            error = run.given_up(part) if part else None
            if error is not None:
                return self._error(error, island)
        return None

    def event(self, handle):
        """The event of ticket `handle` as Run.event gives one: an IslandEvent (its
        parts' events, read as one card's) once every part arrived, else None."""
        parts = self.event_parts(handle)
        return IslandEvent(parts) if all(parts) else None

    def event_parts(self, handle):
        """The parts' events of ticket `handle` in island order, None for a part not
        arrived (a timeout's partial picture)."""
        ticket = handle
        return tuple(
            self.runs[island].event(int(self._parts[island][ticket]))
            for island in range(len(self.runs))
            if self._parts[island][ticket]
        )

    @property
    def stopped(self):
        return all(run.stopped for run in self.runs)

    def stop(self, timeout=10.0):
        """Stops every island's run within one `timeout`, each even if another
        failed; the first failure is raised, naming its island, but not a timeout of a
        ticket wait or drain already raised (`stop_named` keeps it). The result."""
        deadline = time.monotonic() + timeout
        failure = None
        for island, run in enumerate(self.runs):
            try:
                run.stop(_left(deadline))
            except BaseException as error:  # noqa: BLE001 (raised after the others)
                named = self._error(error, island)
                if (
                    isinstance(named, CommandTimeout)
                    and named.handle in self._timed_out
                ):
                    self.stop_named.append(str(named))  # raised before: kept, ended
                elif failure is None:
                    failure = named
        if failure is not None:
            raise failure
        return self.result()

    def abort(self):
        """Aborts every island's run, each even if another's abort raised; the first
        error is raised after the rest."""
        failure = abort_every(self.runs)
        if failure is not None:
            raise failure

    def result(self):
        """The run's result block, as one card's reads (state.island_result: its
        keys over every island); `.islands` each island's. Valid after the stop."""
        return device_state.island_result([run.result() for run in self.runs])

    def trace(self):
        """The per-command trace over islands (IslandTrace), or None if the run was not
        traced; valid after the stop."""
        traces = [run.trace() for run in self.runs]
        return (
            None if None in traces else island_records.IslandTrace(traces, self._parts)
        )

    def state(self):
        """Every rank's state words, as Run.state gives them, in flat rank order, each
        with its `rank` and `island`."""
        states = [run.state() for run in self.runs]
        return [
            {
                **states[island][core],
                "rank": self.graph.first(island) + i,
                "island": island,
            }
            for island, cores in enumerate(self.graph.rank_cores)
            for i, core in enumerate(cores)
        ]

    def crossings(self, puts=False):
        """What crossed each link since the run's start, by link and direction, read
        now from the link cores (a Crossings; none on one island); `after - before` is
        what crossed between two reads. `puts`: each row also holds its put words."""
        if self._counts is None:
            return island_records.Crossings({})
        return self._counts(puts=puts) if puts else self._counts()

    def all_gather(self, tensor, first=0, pages=None, timeout=10.0):
        """All-gather of pages first .. first + pages - 1 (default to the end) of the
        copy `tensor`, island c's share (layout.partition of the pages, c of the
        islands) written on island c: on return every island's copy holds every
        share. One island: returns at once. km::all_gather on every rank (GATHER)."""
        first, pages = self._collective(tensor, first, pages)
        if len(self.runs) < 2 or pages == 0:
            return None
        h = abi.headers()
        words = [0] * 7
        words[h["KM_COLL_TENSOR"]] = tensor.address
        words[h["KM_COLL_FIRST"]] = first
        words[h["KM_COLL_PAGES"]] = pages
        return self._served(h["KM_HANDLER_GATHER"], [words] * len(self.runs), timeout)

    def all_reduce(self, tensor, scratch, first=0, pages=None, timeout=10.0, tree=None):
        """All-reduce of pages first .. first + pages - 1 of the copy `tensor` (each
        island's copy its partial): every copy then holds the sum, in its format.
        `scratch` and `tree` as reduce_to's (a tree: every page up it, then down). One
        island: returns at once; else REDUCE on every rank (km::all_reduce)."""
        first, pages = self._collective(tensor, first, pages)
        kinds = {"int32": 0, "uint32": 0, "float32": 1, "bfloat16": 2}
        if tensor.dtype not in kinds:
            raise KomorebiError(
                f"all_reduce adds int32, uint32, float32, bfloat16: not {tensor.dtype}"
            )
        if len(self.runs) < 2 or pages == 0:
            return None
        self._collective(scratch, 0, self._ports() * pages)
        if scratch.aligned_page_bytes != tensor.aligned_page_bytes:
            raise KomorebiError("all_reduce: the scratch's pages differ in size")
        transfers = [run.backend.transfer for run in self.runs]
        if any(t is None or t.chunk < tensor.aligned_page_bytes for t in transfers):
            raise KomorebiError("all_reduce: the run has no transfer buffer of a page")
        h = abi.headers()
        words = [0] * 10
        words[h["KM_COLL_TENSOR"]] = tensor.address
        words[h["KM_COLL_FIRST"]] = first
        words[h["KM_COLL_PAGES"]] = pages
        words[h["KM_COLL_SCRATCH"]] = scratch.address
        words[h["KM_COLL_SUM"]] = kinds[tensor.dtype]
        self._tree(words, tree)
        by_island = []
        for transfer in transfers:  # the folds' L1: each island's run's own
            words[h["KM_COLL_BUFFER"]] = transfer.address
            words[h["KM_COLL_ROOM"]] = 2 * transfer.chunk
            by_island.append(list(words))
        return self._served(h["KM_HANDLER_REDUCE"], by_island, timeout)

    def _collective(self, tensor, first, pages):
        """(first, pages) of a collective on `tensor`, checked: a copy, the pages in
        it, the islands joined (the link services)."""
        pages = tensor.pages - first if pages is None else pages
        if not getattr(tensor.placement, "copy", False):
            raise KomorebiError("a collective's tensor is a copy (copy=True)")
        if first < 0 or pages < 0 or first + pages > tensor.pages:
            raise KomorebiError(
                f"pages {first} + {pages}: the tensor has {tensor.pages}"
            )
        if len(self.runs) > 1 and self._counts is None:
            raise KomorebiError("a collective crosses links: the session has none")
        return first, pages

    def _served(self, handler, words, timeout):
        """A command of the runtime's service overlay on every rank, a part on each
        island (as submit) with that island's `words`, waited for; its events, island
        order."""
        group = (0, self.cores)
        self._grow(1)
        ticket = self._next
        for island, _first, _count in self.graph.split(group):
            run = self.runs[island]
            try:
                handle = run._queue_command(
                    handler, run._group(group), words[island], False, run._flag_part
                )
            except KomorebiError as error:
                raise self._error(error, island) from None
            self._parts[island][ticket] = handle
        self._next += 1
        return self.wait(ticket, timeout)


def abort_every(runs):
    """Aborts each run of `runs`, each even if another's abort raised; the first
    error (None: none), for the caller to raise or to log beside its own."""
    failure = None
    for run in runs:
        try:
            run.abort()
        except BaseException as error:  # noqa: BLE001 (given back after the rest)
            failure = failure or error
    return failure
