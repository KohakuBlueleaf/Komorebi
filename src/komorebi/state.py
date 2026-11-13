"""Decoding what a run leaves: the dispatcher's result block, the cores' state
words and the per-command trace. Pure functions over arrays read back after the stop.
"""

import numpy as np

from komorebi import abi

GAVES = {
    "Stop": "stop",
    "Limit": "limit",
    "BadItem": "bad_item",
    "BadBroadcast": "bad_broadcast",
}  # kGave*
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
    "CDone",
    "BroadcastBegin",
    "BroadcastDone",
    "BroadcastValid",
    "BroadcastFree",
    "Port",
    "Emit",
)
KMD_PREFIXES = {"error": "KMD_ERROR_", "where": "KMD_WHERE_", "state": "KMD_STATE_"}


def code_name(kind, value, h=None):
    """The name of a dispatcher code (`kind`: error, where, state), lower case."""
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


def state_words(control_rows, control_offset=0, h=None):
    """[cores, kStateWords] uint32 state words from the control blocks read back.

    `control_offset` is the control block's word offset inside each row.
    """
    h = h or abi.headers()
    rows = np.asarray(control_rows).astype(np.int64) & abi.M32
    start = control_offset + h["KM_CONTROL_STATE"] // 4
    return rows[:, start : start + h["kStateWords"]].astype(np.uint32)


def gave_up(word, sequence, h=None):
    """A gave-up word pair as a dict, or None when the kernel did not give up."""
    h = h or abi.headers()
    word = int(word)
    code = word & 0xFF
    if code == h["kGaveNone"]:
        return None
    wheres = {h[f"kWhere{w}"]: w.lower() for w in WHERES}
    gaves = {h[f"kGave{g}"]: name for g, name in GAVES.items()}
    out = {
        "code": gaves.get(code, code),
        "where": wheres.get((word >> 8) & 0xFF, (word >> 8) & 0xFF),
        "slot": (word >> 16) & 0xFF,
        "sequence": int(sequence),
    }
    if out["where"] in ("port", "emit"):
        out["port"] = word >> h["KM_GAVE_PORT_SHIFT"]  # the port a wait named
    return out


def ports(block, h=None):
    """A core's port slots from its control block words: [{count, want}] by port."""
    h = h or abi.headers()
    at, stride = h["KM_CONTROL_PORT"] // 4, h["KM_PORT_STRIDE"] // 4
    return [
        {
            "count": int(block[at + p * stride + h["KM_PORT_COUNT"]]),
            "want": int(block[at + p * stride + h["KM_PORT_WANT"]]),
        }
        for p in range(h["KM_PORTS"])
    ]


def broadcasts(block, h=None):
    """A core's broadcast areas from its control block words, by broadcast: the VALID
    word, the first give-up (where, broadcast, block, sequence; None if none), the
    blocks delivered, the wait cycles and the DONE-BY words of ranks 1 to 15."""
    h = h or abi.headers()
    wheres = {h[f"kWhere{w}"]: w.lower() for w in WHERES}
    out = []
    for broadcast in range(h["KM_BROADCASTS"]):
        a = (h["KM_CONTROL_BROADCAST"] + broadcast * h["KM_BROADCAST_BYTES"]) // 4
        gave = h["KM_BROADCAST_GAVE"] // 4 + a
        gave = [int(v) for v in block[gave : gave + h["KM_BROADCAST_GAVE_WORDS"]]]
        stats = h["KM_BROADCAST_STATS"] // 4 + a
        stats = [int(v) for v in block[stats : stats + h["KM_BROADCAST_STATS_WORDS"]]]
        done, step = a + h["KM_BROADCAST_DONE"] // 4, h["KM_BROADCAST_DONE_STRIDE"] // 4
        out.append(
            {
                "valid": int(block[a + h["KM_BROADCAST_VALID"] // 4]),
                "gave_up": (
                    {
                        "where": wheres.get(gave[0] & 0xFF, gave[0] & 0xFF),
                        "broadcast": (gave[0] >> 8) & 0xFF,
                        "block": gave[1],
                        "sequence": gave[2],
                    }
                    if gave[0]
                    else None
                ),
                "blocks": stats[0],
                "wait_cycles": stats[1] | stats[2] << 32,
                "done": [
                    int(block[done + step * r])
                    for r in range(1, h["KM_BROADCAST_RANKS"])
                ],
            }
        )
    return out


# Dynamic runs -------------------------------------------------------------------

MESSAGES_WHERES = {
    "Gap": "gap",
    "Overflow": "overflow",
    "Atomic": "atomic",
    "Flush": "flush",
    "Read": "read",
    "InPage": "in_page",
    "Line": "line",
    "BadDest": "bad_dest",
    "Next": "next",
    "OutPage": "out_page",
    "Outbox": "outbox",
    "Stopped": "stopped",
    "Align": "align",
    "Lost": "lost",
    "Fill": "fill",
    "Full": "full",
    "Counter": "counter",
}  # kWhereMessages*


def dynamic(block, h=None):
    """A core's dynamic-run words from its control block: queue head and tail, messages
    sent and consumed, quota left, the first give-up (where named, position, run, the
    word seen; None if none), and as a counter tile its last run and cores served."""
    h = h or abi.headers()
    at = h["KM_CONTROL_MESSAGES"] // 4

    def word(offset):
        return int(block[at + offset // 4])

    gave = [word(h["KM_MESSAGES_GAVE"] + 4 * i) for i in range(4)]
    ready = [
        int(block[(h["KM_CONTROL_TAILS"] + h["KM_TAILS_READY"]) // 4 + i])
        for i in range(4)
    ]
    names = {h[f"kWhereMessages{k}"]: v for k, v in MESSAGES_WHERES.items()}
    return {
        "head": word(h["KM_MESSAGES_IN_HEAD"]),
        "tail": word(h["KM_MESSAGES_IN_TAIL"]),
        "sent": word(h["KM_MESSAGES_SENT"]),
        "consumed": word(h["KM_MESSAGES_CONSUMED"]),
        "quota": word(h["KM_MESSAGES_QUOTA"]),
        "gave_up": (
            {
                "where": names.get(gave[0], gave[0]),
                "position": gave[1],
                "run": gave[2],
                "seen": gave[3],
            }
            if gave[0]
            else None
        ),
        "counter": {"run": ready[0], "serves": ready[2]} if ready[1] == 1 else None,
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
        "reader": gave_up(s[h["kStReader"]], s[h["kStReaderSequence"]], h),
        "writer": gave_up(s[h["kStWriter"]], s[h["kStWriterSequence"]], h),
        "compute": gave_up(s[h["kStCompute"]], s[h["kStComputeSequence"]], h),
        "last": s[h["kStLast"] : h["kStLast"] + 16],
        "reported": s[h["kStReported"] : h["kStReported"] + 16],
    }


def not_reported(st, group, slot, sequence, h=None):
    """The cores of `group` whose reported word for `slot` is not `sequence`, with where
    each kernel gave up."""
    h = h or abi.headers()
    first, count = group
    out = []
    for core in range(first, first + count):
        c = core_state(st, core, h)
        if c["reported"][slot] != sequence:
            out.append(
                {"core": core, **{k: c[k] for k in ("reader", "writer", "compute")}}
            )
    return out


def wrap(delta):
    """Differences of 32-bit clock words, mod 2**32, as int64."""
    return np.asarray(delta, dtype=np.int64) % (1 << 32)


class Trace:
    """The per-command trace: wall-clock words per core and sequence number.

    `rows` is [cores, cap, kTraceWords]; `seqs` and `groups` map a handle to its
    sequence number and group; `mhz`: the Tensix clock.
    """

    def __init__(self, rows, seqs, groups, mhz, h=None):
        self.h = h or abi.headers()
        self.rows = np.asarray(rows).astype(np.int64) & abi.M32
        self.seqs = seqs
        self.groups = groups
        self.mhz = mhz

    @property
    def capacity(self):
        return self.rows.shape[1]

    def command(self, handle):
        """[count, 4] words (seen, done, sent, extra) of the handle's cores, or None
        when the handle was not traced."""
        sequence = self.seqs.get(handle)
        if sequence is None or not 1 <= sequence <= self.capacity:
            return None
        first, count = self.groups[handle]
        return self.rows[first : first + count, sequence - 1]

    def phases(self, handle):
        """Microseconds after the first core took the command: last core took it,
        last outputs written, leader pushed the completion. None when not traced."""
        t = self.command(handle)
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
        """Microseconds by which a core of handle b took its command before handle
        a's leader pushed its completion (positive: both were in flight at once)."""
        ta, tb = self.command(a), self.command(b)
        if ta is None or tb is None:
            return None
        pushed = int(ta[0, self.h["kTraceSent"]])
        seen = int(tb[:, self.h["kTraceSeen"]].min())
        delta = (pushed - seen) % (1 << 32)
        if delta >= 1 << 31:
            delta -= 1 << 32
        return delta / self.mhz
