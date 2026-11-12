"""Decoding what a run leaves: the dispatch kernel's result block, the cores' state
words and the per-record trace. Pure functions over arrays read back after the stop.
"""

import numpy as np

from komorebi import abi

WHERES = (
    "Checkin",
    "Token",
    "Next",
    "Stall",
    "Complete",
    "Writer",
    "Data",
    "CToken",
    "CInput",
    "COutput",
)
KMD_PREFIXES = {"error": "KMD_ERR_", "where": "KMD_WHERE_", "state": "KMD_STATE_"}


def code_name(kind, value, h=None):
    """The name of a dispatch kernel code (`kind`: error, where, state), lower case."""
    h = h or abi.headers()
    prefix = KMD_PREFIXES[kind]
    for name, v in h.constants.items():
        if name.startswith(prefix) and v == value:
            return name[len(prefix) :].lower()
    return str(value)


def result_block(words, h=None):
    """The result block's words as a dict; 64-bit counts joined; codes named."""
    h = h or abi.headers()
    w = [int(x) for x in words]
    out = {}
    for name, index in h.constants.items():
        if not name.startswith("KMD_R_") or name == "KMD_R_WORDS":
            continue
        key = name[6:].lower()
        if key.endswith("_hi"):
            continue
        if key.endswith("_lo"):
            out[key[:-3]] = w[index] | w[index + 1] << 32
        else:
            out[key] = w[index]
    for kind in ("state", "error", "where"):
        out[f"{kind}_name"] = code_name(kind, out[kind], h)
    return out


def state_words(ctl_rows, ctl_offset=0, h=None):
    """[cores, kStateWords] uint32 state words from the control blocks read back.

    `ctl_offset` is the control block's word offset inside each row.
    """
    h = h or abi.headers()
    rows = np.asarray(ctl_rows).astype(np.int64) & abi.M32
    start = ctl_offset + h["KM_CTL_STATE"] // 4
    return rows[:, start : start + h["kStateWords"]].astype(np.uint32)


def gave_up(word, seq, h=None):
    """A gave-up word pair as a dict, or None when the kernel did not give up."""
    h = h or abi.headers()
    word = int(word)
    code = word & 0xFF
    if code == h["kGaveNone"]:
        return None
    wheres = {h[f"kWhere{w}"]: w.lower() for w in WHERES}
    return {
        "code": "stop" if code == h["kGaveStop"] else "limit",
        "where": wheres.get((word >> 8) & 0xFF, (word >> 8) & 0xFF),
        "slot": (word >> 16) & 0xFF,
        "seq": int(seq),
    }


def core_state(st, core, h=None):
    """One core's state words as a dict."""
    h = h or abi.headers()
    s = [int(v) for v in st[core]]
    magic, ends, wrong = s[h["kStMagic"]], s[h["kStEnds"]], s[h["kStWrong"]]
    return {
        "core": core,
        "magic_ok": magic >> 24 == h["kStateMagic"] >> 24 and magic & 0xFF == core,
        "tag": (magic >> 8) & 0xFFFF,
        "runs": s[h["kStRuns"]],
        "completed": s[h["kStCompleted"]],
        "passes": s[h["kStPasses"]],
        "next_cycles": s[h["kStNextCycles"]] * 16,
        "compute_seen": s[h["kStSeen"]],
        "compute_wrong": [(wrong >> (8 * b)) & 0xFF for b in range(3)],
        "reader_end": ends & 0xFF,
        "writer_end": (ends >> 8) & 0xFF,
        "compute_end": (ends >> 16) & 0xFF,
        "slot_wrong": s[h["kStSlotWrong"]],
        "trace_dropped": s[h["kStTraceDropped"]],
        "reader": gave_up(s[h["kStReader"]], s[h["kStReaderSeq"]], h),
        "writer": gave_up(s[h["kStWriter"]], s[h["kStWriterSeq"]], h),
        "compute": gave_up(s[h["kStCompute"]], s[h["kStComputeSeq"]], h),
        "last": s[h["kStLast"] : h["kStLast"] + 16],
        "reported": s[h["kStReported"] : h["kStReported"] + 16],
    }


def not_reported(st, team, slot, seq, h=None):
    """The members of `team` whose reported word for `slot` is not `seq`, with where
    each kernel gave up."""
    h = h or abi.headers()
    first, count = team
    out = []
    for core in range(first, first + count):
        c = core_state(st, core, h)
        if c["reported"][slot] != seq:
            out.append(
                {"core": core, **{k: c[k] for k in ("reader", "writer", "compute")}}
            )
    return out


def wrap(delta):
    """Differences of 32-bit clock words, mod 2**32, as int64."""
    return np.asarray(delta, dtype=np.int64) % (1 << 32)


class Trace:
    """The per-record trace: wall-clock words per core and sequence number.

    `rows` is [cores, cap, kTraceWords]; `seqs` maps a ticket to its sequence number
    and `teams` to its team; `mhz` is the Tensix clock.
    """

    def __init__(self, rows, seqs, teams, mhz, h=None):
        self.h = h or abi.headers()
        self.rows = np.asarray(rows).astype(np.int64) & abi.M32
        self.seqs = seqs
        self.teams = teams
        self.mhz = mhz

    @property
    def capacity(self):
        return self.rows.shape[1]

    def record(self, ticket):
        """[count, 4] words (seen, done, sent, extra) of the ticket's members, or None
        when the ticket was not traced."""
        seq = self.seqs.get(ticket)
        if seq is None or not 1 <= seq <= self.capacity:
            return None
        first, count = self.teams[ticket]
        return self.rows[first : first + count, seq - 1]

    def phases(self, ticket):
        """Microseconds after the first member took the record: last member took it,
        last outputs written, leader pushed the completion. None when not traced."""
        t = self.record(ticket)
        if t is None or not t[:, 0].any():
            return None
        h = self.h
        rel = wrap(t[:, :3] - t[:, h["kTraceSeen"]].min()) / self.mhz
        done, sent = rel[:, h["kTraceDone"]], rel[:, h["kTraceSent"]]
        return {
            "count": len(t),
            "last_seen": float(rel[:, h["kTraceSeen"]].max()),
            "last_done": float(done.max()),
            "leader_sent": float(sent[0]),
            "push_after_last_done": float(sent[0] - done.max()),
            "passes_mean": float(t[:, h["kTraceExtra"]].mean()),
        }

    def lead(self, a, b):
        """Microseconds by which a member of ticket b took its record before ticket
        a's leader pushed its completion (positive: both were in flight at once)."""
        ta, tb = self.record(a), self.record(b)
        if ta is None or tb is None:
            return None
        pushed = int(ta[0, self.h["kTraceSent"]])
        seen = int(tb[:, self.h["kTraceSeen"]].min())
        delta = (pushed - seen) % (1 << 32)
        if delta >= 1 << 31:
            delta -= 1 << 32
        return delta / self.mhz
