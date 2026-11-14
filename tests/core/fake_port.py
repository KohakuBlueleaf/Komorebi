"""A fake dispatcher behind the native port's interface, for the Run tests. In order,
commands get the lowest free slot and the next sequence while their events fit the ring
by the read count (KM_FLAG_EXCLUSIVE: while no busy group overlaps); they complete at
once or on request."""

import time

import numpy as np

from komorebi import abi

HEADERS = abi.headers()
CONSTANTS = HEADERS.constants


class FakePort:
    """The port: enqueue, queued, wait, push, room, poll, wait_events, host_word, plus
    test controls."""

    def __init__(self, cores=120, depth=4, host_depth=8, event_depth=16):
        self.cores = cores
        self.depth = depth
        self.host_depth = host_depth
        self.event_depth = event_depth
        self.next = 1  # next command number the host stores
        self.consumed = 0
        self.seen = 0
        self.published = 0  # the read count the host stored for the kernel
        self.held_for_room = 0  # passes the kernel held a command for the read count
        self.ring = {}  # command number -> words
        self.busy = {}  # slot -> (sequence, req, group)
        self.sequence = 0
        self.events = []  # pushed, not yet read: [index, req, status, ticks]
        self.pushed_events = 0
        self.state = CONSTANTS["KMD_STATE_READY"]
        self.error = 0
        self.result = {}
        self.auto = True  # complete every busy command at each poll
        self.stall_req = None  # this command times out instead of completing
        self.exited = False  # EXIT was submitted: later commands are rejected
        self.held_exit = 0  # passes EXIT waited for busy slots
        self.slot_sequence = {}  # slot -> the last sequence it carried
        self.submitted = []  # (req, sequence, slot, handler, group) in submit order
        self.batches = []  # number of commands per push
        self.max_busy = 0
        self.max_unread = 0
        self.batch = 64
        self.queue = []  # words not stored yet
        self.waits = 0  # calls of wait()
        self.results = {}  # command number -> the leader's result word (KM_FLAG_RESULT)
        self.handlers = {}  # handler -> handler(req, words): commands a test serves
        self.programs = None  # the dispatcher's table: the program field per core
        self.give_up_req = (
            None  # this command is given up at its deadline, run going on
        )
        self.given_up = 0  # the dispatcher's count of commands it gave up
        self.wheres = {}  # core -> (role, where): a kernel's live wait in an inspect
        self.inspect_block = np.zeros(
            (CONSTANTS["KMD_INSPECT_HEAD"] + 128 * CONSTANTS["KMD_INSPECT_RECORD"])
            // 4,
            dtype=np.uint32,
        )

    # The native port's interface ---------------------------------------------

    def room(self):
        ring = self.host_depth - (self.next - 1 - self.consumed)
        events = self.event_depth - (self.next - 1 - self.seen)
        return max(0, min(ring, events))

    def push(self, rows):
        rows = np.asarray(rows, dtype=np.uint32)
        assert rows.ndim == 2 and rows.shape[1] == 16
        pushed = min(len(rows), self.room())
        for j in range(pushed):
            words = [int(word) for word in rows[j]]
            words[0] = words[4] = words[8] = words[12] = self.next & abi.M32
            self.ring[self.next] = words
            self.next += 1
        self.published = self.seen  # every batch carries the read count
        if pushed:
            self.batches.append(pushed)
        return pushed

    def enqueue(self, commands):
        rows = np.asarray(commands, dtype=np.uint32)
        assert rows.ndim == 2 and rows.shape[1] == 16
        self.queue += [[int(word) for word in row] for row in rows]
        return self.queued

    @property
    def queued(self):
        return len(self.queue)

    def wait(self, wanted, upto, timeout_s):
        """As the native loop: store a batch up to `upto`, read events, until every
        command in `wanted` has one, another kind arrives, the kernel ended, or time
        is up; with nothing wanted, once."""
        self.waits += 1
        need = {int(handle) for handle in wanted}
        deadline = time.monotonic() + timeout_s
        out = []
        while True:
            self._store_queued(int(upto))
            rows = self.poll()
            self._publish_if_held()
            other = False
            for row in rows.tolist():
                need.discard(row[1])
                other = other or (row[2] & 0xFF) != CONSTANTS["KM_EVENT_DONE"]
            out += rows.tolist()
            ended = self.state != CONSTANTS["KMD_STATE_READY"]
            if not need or other or ended or time.monotonic() > deadline:
                return np.asarray(out, dtype=np.uint32).reshape(-1, 4)

    def _store_queued(self, upto):
        due = (upto - self.next) & abi.M32
        due = (due - (1 << 32) if due >= 1 << 31 else due) + 1
        stored = max(0, min(len(self.queue), self.batch, self.room(), due))
        if stored:
            self.push(self.queue[:stored])
            del self.queue[:stored]

    def poll(self):
        self._step()
        out = self.events
        self.events = []
        self.seen += len(out)
        return np.asarray(out, dtype=np.uint32).reshape(-1, 4)

    def wait_events(self, timeout_s):
        out = self.poll()
        self._publish_if_held()
        return out

    def _publish_if_held(self):
        if self.next - 1 - self.published >= self.event_depth:
            self.published = self.seen  # the kernel may hold a command for it

    def host_word(self, word):
        return self.state if word == 0 else 0

    # The fake kernel ---------------------------------------------------------

    def _event(self, req, kind, auxiliary=0, ticks=100):
        self.pushed_events += 1
        unread = self.pushed_events - self.seen
        self.max_unread = max(self.max_unread, unread)
        assert unread <= self.event_depth, "event ring overrun"
        index = self.pushed_events & abi.M32
        self.events.append([index, req & abi.M32, kind | auxiliary << 8, ticks])

    def _finish(self, error, slot=0, sequence=0, req=0):
        self.error = error
        self.state = (
            CONSTANTS["KMD_STATE_FAILED"] if error else CONSTANTS["KMD_STATE_DONE"]
        )
        self.result = {"error": error, "slot": slot, "sequence": sequence, "req": req}

    def _step(self):
        """Takes the commands that fit, then completes the busy slots: a command in
        flight is still busy when a command behind it is taken."""
        self._take()
        if self.state == CONSTANTS["KMD_STATE_READY"]:
            self._complete()

    def _take(self):
        while (
            self.consumed + 1 in self.ring
            and self.state == CONSTANTS["KMD_STATE_READY"]
        ):
            req = self.consumed + 1
            words = self.ring[req]
            handler, flags = words[1] & 0xFF, (words[1] >> 8) & 0xFF
            first, count = words[2] & 0xFFFF, words[2] >> 16
            if handler == CONSTANTS["KM_HANDLER_STOP"]:
                self.consumed = req
                self._finish(CONSTANTS["KMD_ERROR_STOP_BUSY"] if self.busy else 0)
                return
            if handler == CONSTANTS["KM_HANDLER_PING"]:
                self.consumed = req
                self._event(req, CONSTANTS["KM_EVENT_PONG"])
                continue
            if handler == CONSTANTS["KMD_HANDLER_INSPECT"]:
                self.consumed = req
                self._inspect(CONSTANTS["KMD_INSPECT_HOST"], req, first, count)
                self._event(req, CONSTANTS["KM_EVENT_PONG"])
                continue
            if handler in self.handlers:
                self.consumed = req
                self.handlers[handler](req, words)
                continue
            reason = self._rejection(handler, flags, first, count)
            if reason:
                self.consumed = req
                self._event(req, CONSTANTS["KM_EVENT_REJECTED"], auxiliary=reason)
                continue
            if handler == CONSTANTS["KM_HANDLER_EXIT"]:
                if self._exit_waits():  # held until no slot is busy
                    self.held_exit += 1
                    return
                first, count = 0, self.cores  # to every core
                self.exited = True
            elif (
                flags & CONSTANTS["KM_FLAG_QUEUE"]
            ):  # the lowest first overlapping no busy group
                taken = [group for _, _, group in self.busy.values()]
                fits = [
                    candidate
                    for candidate in range(self.cores - count + 1)
                    if all(
                        candidate + count <= busy_first
                        or busy_first + busy_count <= candidate
                        for busy_first, busy_count in taken
                    )
                ]
                if not fits:
                    return
                first = fits[0]
            elif flags & CONSTANTS["KM_FLAG_EXCLUSIVE"] and self._overlaps_busy(
                first, count
            ):
                return  # held, and every command behind it
            elif handler == CONSTANTS["KM_HANDLER_LOAD"] and self._overlaps_busy(
                first, count
            ):
                return  # a load is exclusive
            free = [j for j in range(self.depth) if j not in self.busy]
            if not free:
                return
            unread = self.pushed_events - self.published
            if unread + len(self.busy) + 1 > self.event_depth:
                self.held_for_room += 1
                return
            self.sequence = (self.sequence + 1) & abi.M32
            while self.sequence == 0 or self.sequence == self.slot_sequence.get(
                free[0]
            ):
                self.sequence = (self.sequence + 1) & abi.M32
            self.slot_sequence[free[0]] = self.sequence
            self.busy[free[0]] = (self.sequence, req, (first, count))
            self.max_busy = max(self.max_busy, len(self.busy))
            self.submitted.append(
                (req, self.sequence, free[0], handler, (first, count))
            )
            if handler == CONSTANTS["KM_HANDLER_LOAD"]:
                self.programs[first : first + count] = [words[1] >> 24] * count
            self.consumed = req

    def _overlaps_busy(self, first, count):
        taken = [group for _, _, group in self.busy.values()]
        return any(
            busy_first < first + count and first < busy_first + busy_count
            for busy_first, busy_count in taken
        )

    def _exit_waits(self):
        return bool(self.busy)

    def _rejection(self, handler, flags, first, count):
        """The README's reasons a command is consumed with KM_EVENT_REJECTED, or 0."""
        if self.exited:
            return CONSTANTS["KMD_REJECT_EXITED"]
        load = handler == CONSTANTS["KM_HANDLER_LOAD"]
        if (
            CONSTANTS["KM_HANDLER_USER_MAX"]
            < handler
            < CONSTANTS["KM_HANDLER_DEADLINE"]
            and not load
        ):
            return CONSTANTS["KMD_REJECT_HANDLER"]
        if flags & 1:
            return CONSTANTS["KMD_REJECT_FLAGS"]
        if flags & CONSTANTS["KM_FLAG_QUEUE"]:
            first = 0  # only the count is checked
        if handler != CONSTANTS["KM_HANDLER_EXIT"] and (
            count == 0 or first + count > self.cores
        ):
            return CONSTANTS["KMD_REJECT_GROUP"]
        return self._program_rejection(handler, flags, first, count, load)

    def _program_rejection(self, handler, flags, first, count, load):
        """Reason 13 from the dispatcher's table: a group with a core that does not
        hold the command's program (queued: any core), a load where there is none."""
        if load and (self.programs is None or flags & CONSTANTS["KM_FLAG_QUEUE"]):
            return CONSTANTS["KMD_REJECT_PROGRAM"]
        if (
            self.programs is None
            or load
            or handler in (0, CONSTANTS["KM_HANDLER_EXIT"])
        ):
            return 0
        program = self.ring[self.consumed + 1][1] >> 24
        if flags & CONSTANTS["KM_FLAG_QUEUE"]:
            first, count = 0, self.cores
        if any(held != program for held in self.programs[first : first + count]):
            return CONSTANTS["KMD_REJECT_PROGRAM"]
        return 0

    def _inspect(self, kind, req, first, count, slot=0, sequence=0):
        """Writes the inspect block: each core of the group read, its state magic and
        the live wait `wheres` gives it, its last trace entry at `sequence`."""
        block, record = self.inspect_block, CONSTANTS["KMD_INSPECT_RECORD"] // 4
        block[:] = 0
        header = {"KIND": kind, "REQUEST": req, "FIRST": first, "COUNT": count}
        header |= {"DONE": count, "SLOT": slot, "SEQUENCE": sequence, "TICKS": 1350}
        for name, value in header.items():
            block[CONSTANTS[f"KMD_I_{name}"]] = value
        block[CONSTANTS["KMD_I_MARK"]] = CONSTANTS["KMD_INSPECT_MARK"]
        for index in range(count):
            line = block[CONSTANTS["KMD_INSPECT_HEAD"] // 4 + index * record :][:record]
            core = first + index
            line[CONSTANTS["KMD_IR_CORE"]] = core
            line[CONSTANTS["KMD_IR_FLAGS"]] = CONSTANTS["KMD_INSPECT_READ"]
            line[CONSTANTS["KMD_IR_SEQUENCES"]] = sequence
            state = CONSTANTS["KMD_INSPECT_STATE"] // 4
            line[state + CONSTANTS["kStMagic"]] = CONSTANTS["kStateMagic"] | core
            if core in self.wheres:
                role, where = self.wheres[core]
                live = CONSTANTS["KMD_INSPECT_LIVE"] // 4
                line[live + CONSTANTS["kLvWhere"] + role] = where
                line[live + CONSTANTS["kLvSequence"] + role] = sequence

    def _complete(self):
        for slot in sorted(self.busy):
            sequence, req, (first, _) = self.busy[slot]
            low = (sequence & CONSTANTS["KM_EVENT_SEQUENCE_MASK"]) << CONSTANTS[
                "KM_EVENT_SEQUENCE_SHIFT"
            ]
            if req == self.give_up_req:  # the inspect of the deadline, then the event
                del self.busy[slot]
                first, count = self.ring[req][2] & 0xFFFF, self.ring[req][2] >> 16
                timeout = CONSTANTS["KMD_INSPECT_TIMEOUT"]
                self._inspect(timeout, req, first, count, slot, sequence)
                self.given_up += 1
                self._event(
                    req, CONSTANTS["KM_EVENT_TIMEOUT"], auxiliary=slot | low, ticks=1
                )
                continue
            if req == self.stall_req:
                del self.busy[slot]
                self._event(
                    req,
                    CONSTANTS["KM_EVENT_TIMEOUT"],
                    auxiliary=slot | low,
                    ticks=2_500_000,
                )
                self._finish(CONSTANTS["KMD_ERROR_DEADLINE"], slot, sequence, req)
                return
            if self.auto:
                del self.busy[slot]
                ticks = self.results.get(req, 100)
                self._event(
                    req, CONSTANTS["KM_EVENT_DONE"], auxiliary=first | low, ticks=ticks
                )


class FakeBackend:
    """A Run's backend over a FakePort: finish, abort, read back."""

    def __init__(self, port, trace_cap=0, stalled_core=None):
        self.port = port
        self.mhz = 1350
        self.trace_cap = trace_cap
        self.stalled_core = stalled_core
        self.calls = []
        self.answer = {}  # finish() fields to answer instead (result_ready, ...)
        self.cleaned = True  # what abort() answers
        self.close_failed = ""  # the runtime's error message when not cleaned
        self.aborted = None  # {words, cores} the kernel left at abort()
        self.report = "the fake kernel's report"  # what kernel_report() answers

    def finish(self, timeout):
        self.calls.append("finish")
        self.port._step()  # the kernel takes what is in its ring (STOP)
        if self.answer.get("result_ready", True):
            assert (
                self.port.state != CONSTANTS["KMD_STATE_READY"]
            ), "finished while running"
        words = [0] * CONSTANTS["KMD_R_WORDS"]
        words[CONSTANTS["KMD_R_STATE"]] = self.port.state
        for name, value in self.port.result.items():
            words[CONSTANTS[f"KMD_R_{name.upper()}"]] = value
        words[CONSTANTS["KMD_R_COMMANDS"]] = len(self.port.submitted)
        if "state" in self.answer:
            words[CONSTANTS["KMD_R_STATE"]] = self.answer["state"]
        out = {
            "result_ready": True,
            "result": 0,
            "cleaned": True,
            "failed": "",
            "words": words,
        }
        out.update({key: entry for key, entry in self.answer.items() if key in out})
        return out

    def abort(self):
        self.calls.append("abort")
        return self.cleaned

    def kernel_report(self):
        self.calls.append("kernel_report")
        return self.report

    def log_state(self):
        pass

    def inspect(self, count):
        """The inspect block's header and first `count` records, the give-up count."""
        words = CONSTANTS["KMD_INSPECT_HEAD"] + count * CONSTANTS["KMD_INSPECT_RECORD"]
        block = self.port.inspect_block[: words // 4].copy()
        return {"block": block, "given_up": self.port.given_up}

    def read_back(self, stop_grid=False):
        """Control blocks whose state words say every core of the group reported its
        commands' sequences, except the stalled core; trace: seen/done/sent timestamps
        per sequence.
        """
        self.calls.append("read_back")
        self.grid_stopped = stop_grid
        cores = self.port.cores
        blocks = np.zeros((cores, CONSTANTS["KM_CONTROL_BYTES"] // 4), dtype=np.uint32)
        state_offset = CONSTANTS["KM_CONTROL_STATE"] // 4
        state_words = blocks[
            :, state_offset : state_offset + CONSTANTS["kStateWords"]
        ]  # a view: written below
        state_words[:, CONSTANTS["kStMagic"]] = CONSTANTS["kStateMagic"] | np.arange(
            cores
        )
        trace = None
        if self.trace_cap:
            trace = np.zeros((cores, self.trace_cap, 4), dtype=np.uint32)
        stalled = self.port.stall_req
        for req, sequence, slot, _, (first, count) in self.port.submitted:
            for core in range(first, first + count):
                if req == stalled and core == self.stalled_core:
                    word = (
                        CONSTANTS["kGaveStop"]
                        | CONSTANTS["kWhereStall"] << 8
                        | slot << 16
                    )
                    state_words[core, CONSTANTS["kStReader"]] = word
                    state_words[core, CONSTANTS["kStReaderSequence"]] = sequence
                    continue
                if req == stalled and core == first:
                    word = (
                        CONSTANTS["kGaveStop"]
                        | CONSTANTS["kWhereComplete"] << 8
                        | slot << 16
                    )
                    state_words[core, CONSTANTS["kStWriter"]] = word
                    state_words[core, CONSTANTS["kStWriterSequence"]] = sequence
                    continue
                state_words[core, CONSTANTS["kStReported"] + slot] = sequence
                if trace is not None and sequence <= self.trace_cap:
                    tick_base = 1000 * sequence
                    trace[core, sequence - 1] = [
                        tick_base + core,
                        tick_base + 500,
                        tick_base + 600 + core,
                        3,
                    ]
        return blocks, trace
