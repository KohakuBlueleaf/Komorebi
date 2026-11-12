"""A fake dispatch kernel behind the native port's interface, for the Run tests.

In order, records get the lowest free slot and the next sequence, only while their
events fit the ring by the published read count; they complete at once or on request.
"""

import numpy as np

from komorebi import abi

H = abi.headers()
K = H.constants


class FakePort:
    """The port: push, room, poll, wait_events, host_word, plus test controls."""

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
        self.refuse_bad = False  # refuse malformed records with an event
        self.issued = []  # (req, seq, slot, seg, team) in issue order
        self.batches = []  # number of records per push
        self.max_busy = 0
        self.max_unread = 0

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
            words[0] = words[4] = words[8] = words[12] = self.next
            self.ring[self.next] = words
            self.next += 1
        self.published = self.seen  # every batch carries the read count
        if k:
            self.batches.append(k)
        return k

    def poll(self):
        self._step()
        out = self.events
        self.events = []
        self.seen += len(out)
        return np.asarray(out, dtype=np.uint32).reshape(-1, 4)

    def wait_events(self, timeout_s):
        out = self.poll()
        if self.next - 1 - self.published >= self.event_depth:
            self.published = self.seen  # the kernel may hold a command for it
        return out

    def host_word(self, word):
        return self.state if word == 0 else 0

    # The fake kernel ---------------------------------------------------------

    def _event(self, req, kind, aux=0, ticks=100):
        self.pushed_events += 1
        unread = self.pushed_events - self.seen
        self.max_unread = max(self.max_unread, unread)
        assert unread <= self.event_depth, "event ring overrun"
        self.events.append([self.pushed_events, req, kind | aux << 8, ticks])

    def _finish(self, error, slot=0, seq=0, req=0):
        self.error = error
        self.state = K["KMD_STATE_FAILED"] if error else K["KMD_STATE_DONE"]
        self.result = {"error": error, "slot": slot, "seq": seq, "req": req}

    def _step(self):
        if self.state != K["KMD_STATE_READY"]:
            return
        self._complete()
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
            bad = (
                (seg > K["KM_SEG_USER_MAX"] and seg != K["KM_SEG_EXIT"])
                or flags & 1
                or count == 0
                or first + count > self.cores
            )
            if bad:
                self.consumed = req
                if self.refuse_bad:
                    self._event(req, 3, aux=1)
                    continue
                self._finish(K["KMD_ERR_COMMAND"], 0, self.seq + 1, req)
                return
            free = [j for j in range(self.depth) if j not in self.busy]
            if not free:
                return
            unread = self.pushed_events - self.published
            if unread + len(self.busy) + 1 > self.event_depth:
                self.held_for_room += 1
                return
            self.seq += 1
            self.busy[free[0]] = (self.seq, req, (first, count))
            self.max_busy = max(self.max_busy, len(self.busy))
            self.issued.append((req, self.seq, free[0], seg, (first, count)))
            self.consumed = req

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
                self._event(req, K["KM_EV_DONE"])


class FakeBackend:
    """A Run's backend over a FakePort: finish, abort, read back."""

    def __init__(self, port, trace_cap=0, stalled_core=None):
        self.port = port
        self.mhz = 1350
        self.trace_cap = trace_cap
        self.stalled_core = stalled_core
        self.calls = []

    def finish(self, timeout):
        self.calls.append("finish")
        self.port._step()  # the kernel takes what is in its ring (STOP)
        assert self.port.state != K["KMD_STATE_READY"], "finish before the kernel ended"
        words = [0] * K["KMD_R_WORDS"]
        words[K["KMD_R_STATE"]] = self.port.state
        for name, value in self.port.result.items():
            words[K[f"KMD_R_{name.upper()}"]] = value
        words[K["KMD_R_RECORDS"]] = len(self.port.issued)
        return {
            "result_ready": True,
            "result": 0,
            "cleaned": True,
            "failed": "",
            "words": words,
        }

    def abort(self):
        self.calls.append("abort")
        return True

    def read_back(self):
        """State words: every member reported its records' sequences, except the
        stalled core; trace: seen/done/sent stamps per sequence."""
        self.calls.append("read_back")
        cores = self.port.cores
        st = np.zeros((cores, K["kStateWords"]), dtype=np.uint32)
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
        return st, trace
