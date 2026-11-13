"""A fake dispatch kernel behind the native port's interface, for the Run tests.

In order, records get the lowest free slot and the next sequence, only while their
events fit the ring by the published read count; they complete at once or on request.
"""

import time

import numpy as np

from komorebi import abi

H = abi.headers()
K = H.constants


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
        self.busy = {}  # slot -> (seq, req, team)
        self.seq = 0
        self.events = []  # pushed, not yet read: [index, req, status, ticks]
        self.pushed_events = 0
        self.state = K["KMD_STATE_READY"]
        self.error = 0
        self.result = {}
        self.auto = True  # complete every busy record at each poll
        self.stall_req = None  # this command times out instead of completing
        self.exited = False  # EXIT was issued: later records are refused
        self.held_exit = 0  # passes EXIT waited for busy slots
        self.slot_seq = {}  # slot -> the last sequence it carried
        self.issued = []  # (req, seq, slot, seg, team) in issue order
        self.batches = []  # number of records per push
        self.max_busy = 0
        self.max_unread = 0
        self.batch = 64
        self.queue = []  # (words, exclusive) not stored yet
        self.flight = {}  # command number -> team word, stored without an event
        self.waits = 0  # calls of wait()
        self.results = {}  # command number -> the leader's result word (KM_F_RESULT)
        self.handlers = {}  # segment -> handler(req, words): commands a test serves

    # The native port's interface ---------------------------------------------

    def room(self):
        ring = self.host_depth - (self.next - 1 - self.consumed)
        events = self.event_depth - (self.next - 1 - self.seen)
        return max(0, min(ring, events))

    def push(self, rows):
        rows = np.asarray(rows, dtype=np.uint32)
        assert rows.ndim == 2 and rows.shape[1] == 16
        k = min(len(rows), self.room())
        for j in range(k):
            words = [int(w) for w in rows[j]]
            words[0] = words[4] = words[8] = words[12] = self.next & abi.M32
            self.ring[self.next] = words
            self.flight[self.next & abi.M32] = words[2]
            self.next += 1
        self.published = self.seen  # every batch carries the read count
        if k:
            self.batches.append(k)
        return k

    def enqueue(self, records, exclusive=None):
        rows = np.asarray(records, dtype=np.uint32)
        assert rows.ndim == 2 and rows.shape[1] == 16
        flags = [0] * len(rows) if exclusive is None else list(exclusive)
        assert len(flags) == len(rows)
        self.queue += [([int(w) for w in r], f) for r, f in zip(rows, flags)]
        return self.queued

    @property
    def queued(self):
        return len(self.queue)

    def wait(self, wanted, upto, timeout_s):
        """As the native loop: store a batch up to `upto`, read events, until every
        command in `wanted` has one, another kind arrives, the kernel ended, or time
        is up; with nothing wanted, once."""
        self.waits += 1
        need = {int(w) for w in wanted}
        by = time.monotonic() + timeout_s
        out = []
        while True:
            self._store_queued(int(upto))
            rows = self.poll()
            self._publish_if_held()
            other = False
            for row in rows.tolist():
                need.discard(row[1])
                other = other or (row[2] & 0xFF) != K["KM_EV_DONE"]
            out += rows.tolist()
            ended = self.state != K["KMD_STATE_READY"]
            if not need or other or ended or time.monotonic() > by:
                return np.asarray(out, dtype=np.uint32).reshape(-1, 4)

    def _store_queued(self, upto):
        due = (upto - self.next) & abi.M32
        due = (due - (1 << 32) if due >= 1 << 31 else due) + 1
        k = max(0, min(len(self.queue), self.batch, self.room(), due))
        n = 0
        while n < k:
            words, exclusive = self.queue[n]
            if exclusive:
                n += n == 0 and not self._overlaps(words[2])
                break
            n += 1
        if n:
            self.push([w for w, _ in self.queue[:n]])
            del self.queue[:n]

    def _overlaps(self, team):
        a0, a1 = team & 0xFFFF, (team & 0xFFFF) + (team >> 16)
        for t in self.flight.values():
            b0, b1 = t & 0xFFFF, (t & 0xFFFF) + (t >> 16)
            if t and b0 < a1 and a0 < b1:
                return True
        return False

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

    def _event(self, req, kind, aux=0, ticks=100):
        self.pushed_events += 1
        unread = self.pushed_events - self.seen
        self.max_unread = max(self.max_unread, unread)
        assert unread <= self.event_depth, "event ring overrun"
        self.flight.pop(req & abi.M32, None)
        index = self.pushed_events & abi.M32
        self.events.append([index, req & abi.M32, kind | aux << 8, ticks])

    def _finish(self, error, slot=0, seq=0, req=0):
        self.error = error
        self.state = K["KMD_STATE_FAILED"] if error else K["KMD_STATE_DONE"]
        self.result = {"error": error, "slot": slot, "seq": seq, "req": req}

    def _step(self):
        """Takes the commands that fit, then completes the busy slots: a record in
        flight is still busy when a command behind it is taken."""
        self._take()
        if self.state == K["KMD_STATE_READY"]:
            self._complete()

    def _take(self):
        while self.consumed + 1 in self.ring and self.state == K["KMD_STATE_READY"]:
            req = self.consumed + 1
            words = self.ring[req]
            seg, flags = words[1] & 0xFF, (words[1] >> 8) & 0xFF
            first, count = words[2] & 0xFFFF, words[2] >> 16
            if seg == K["KM_SEG_STOP"]:
                self.consumed = req
                self._finish(K["KMD_ERR_STOP_BUSY"] if self.busy else 0)
                return
            if seg == K["KM_SEG_PING"]:
                self.consumed = req
                self._event(req, K["KM_EV_PONG"])
                continue
            if seg in self.handlers:
                self.consumed = req
                self.handlers[seg](req, words)
                continue
            reason = self._refusal(seg, flags, first, count)
            if reason:
                self.consumed = req
                self._event(req, K["KM_EV_REFUSED"], aux=reason)
                continue
            if seg == K["KM_SEG_EXIT"]:
                if self._exit_waits():  # held until no slot is busy
                    self.held_exit += 1
                    return
                first, count = 0, self.cores  # to every core
                self.exited = True
            elif flags & K["KM_F_QUEUE"]:  # the lowest first overlapping no busy team
                taken = [t for _, _, t in self.busy.values()]
                fits = [
                    f
                    for f in range(self.cores - count + 1)
                    if all(f + count <= a or a + n <= f for a, n in taken)
                ]
                if not fits:
                    return
                first = fits[0]
            free = [j for j in range(self.depth) if j not in self.busy]
            if not free:
                return
            unread = self.pushed_events - self.published
            if unread + len(self.busy) + 1 > self.event_depth:
                self.held_for_room += 1
                return
            self.seq = (self.seq + 1) & abi.M32
            while self.seq == 0 or self.seq == self.slot_seq.get(free[0]):
                self.seq = (self.seq + 1) & abi.M32
            self.slot_seq[free[0]] = self.seq
            self.busy[free[0]] = (self.seq, req, (first, count))
            self.max_busy = max(self.max_busy, len(self.busy))
            self.issued.append((req, self.seq, free[0], seg, (first, count)))
            self.consumed = req

    def _exit_waits(self):
        return bool(self.busy)

    def _refusal(self, seg, flags, first, count):
        """The README's reasons a command is consumed with KM_EV_REFUSED, or 0."""
        if self.exited:
            return K["KMD_REFUSE_EXITED"]
        if K["KM_SEG_USER_MAX"] < seg < K["KM_SEG_DEADLINE"]:
            return K["KMD_REFUSE_SEG"]
        if flags & 1:
            return K["KMD_REFUSE_FLAGS"]
        if flags & K["KM_F_QUEUE"]:
            first = 0  # only the count is checked
        if seg != K["KM_SEG_EXIT"] and (count == 0 or first + count > self.cores):
            return K["KMD_REFUSE_TEAM"]
        return 0

    def _complete(self):
        for slot in sorted(self.busy):
            seq, req, _ = self.busy[slot]
            if req == self.stall_req:
                del self.busy[slot]
                self._event(req, K["KM_EV_TIMEOUT"], aux=slot, ticks=2_500_000)
                self._finish(K["KMD_ERR_DEADLINE"], slot, seq, req)
                return
            if self.auto:
                del self.busy[slot]
                self._event(req, K["KM_EV_DONE"], ticks=self.results.get(req, 100))


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

    def finish(self, timeout):
        self.calls.append("finish")
        self.port._step()  # the kernel takes what is in its ring (STOP)
        if self.answer.get("result_ready", True):
            assert self.port.state != K["KMD_STATE_READY"], "finished while running"
        words = [0] * K["KMD_R_WORDS"]
        words[K["KMD_R_STATE"]] = self.port.state
        for name, value in self.port.result.items():
            words[K[f"KMD_R_{name.upper()}"]] = value
        words[K["KMD_R_RECORDS"]] = len(self.port.issued)
        if "state" in self.answer:
            words[K["KMD_R_STATE"]] = self.answer["state"]
        out = {
            "result_ready": True,
            "result": 0,
            "cleaned": True,
            "failed": "",
            "words": words,
        }
        out.update({k: v for k, v in self.answer.items() if k in out})
        return out

    def abort(self):
        self.calls.append("abort")
        return self.cleaned

    def read_back(self, stop_grid=False):
        """Control blocks whose state words say every member reported its records'
        sequences, except the stalled core; trace: seen/done/sent stamps per sequence.
        """
        self.calls.append("read_back")
        self.grid_stopped = stop_grid
        cores = self.port.cores
        blocks = np.zeros((cores, K["KM_CTL_BYTES"] // 4), dtype=np.uint32)
        at = K["KM_CTL_STATE"] // 4
        st = blocks[:, at : at + K["kStateWords"]]  # a view: written below
        st[:, K["kStMagic"]] = K["kStateMagic"] | np.arange(cores)
        trace = None
        if self.trace_cap:
            trace = np.zeros((cores, self.trace_cap, 4), dtype=np.uint32)
        stalled = self.port.stall_req
        for req, seq, slot, _, (first, count) in self.port.issued:
            for core in range(first, first + count):
                if req == stalled and core == self.stalled_core:
                    word = K["kGaveStop"] | K["kWhereStall"] << 8 | slot << 16
                    st[core, K["kStReader"]] = word
                    st[core, K["kStReaderSeq"]] = seq
                    continue
                if req == stalled and core == first:
                    word = K["kGaveStop"] | K["kWhereComplete"] << 8 | slot << 16
                    st[core, K["kStWriter"]] = word
                    st[core, K["kStWriterSeq"]] = seq
                    continue
                st[core, K["kStReported"] + slot] = seq
                if trace is not None and seq <= self.trace_cap:
                    t0 = 1000 * seq
                    trace[core, seq - 1] = [t0 + core, t0 + 500, t0 + 600 + core, 3]
        return blocks, trace
