"""Decoding what a run leaves: the dispatcher's result block, the cores' state
words and the per-command trace, and an inspect's block while the run goes on. Pure
functions over arrays read back.
"""

import re

import numpy as np

from komorebi import abi

GAVES = {
    "Stop": "stop",
    "Limit": "limit",
    "BadItem": "bad_item",
    "BadBroadcast": "bad_broadcast",
    "Overlay": "overlay",
    "BadTransfer": "bad_transfer",
    "Deadline": "deadline",
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
    "OverlayFind",
    "OverlayLoad",
    "OverlayMeet",
    "ProgramSwitch",
    "ProgramStart",
    "ToHost",
    "FromHost",
    "Copy",
    "NocFlush",
    "CMailbox",
    "CConfirm",
    "Clear",
)
ROLES = ("reader", "writer", "unpack", "math", "pack")  # kRole*: the live words' order
KMD_PREFIXES = {"error": "KMD_ERROR_", "where": "KMD_WHERE_", "state": "KMD_STATE_"}


def code_name(kind, value, abi_headers=None):
    """The name of a dispatcher code (`kind`: error, where, state), lower case."""
    abi_headers = abi_headers or abi.headers()
    prefix = KMD_PREFIXES[kind]
    for name, candidate in abi_headers.constants.items():
        if name.startswith(prefix) and candidate == value:
            return name[len(prefix) :].lower()
    return str(value)


def result_block(words, abi_headers=None):
    """The result block's words as a dict; 64-bit counts joined; codes named."""
    abi_headers = abi_headers or abi.headers()
    values = [int(x) for x in words]
    out = {}
    for name, index in abi_headers.constants.items():
        if not name.startswith("KMD_R_") or name == "KMD_R_WORDS":
            continue
        key = name[6:].lower()
        if key.endswith("_hi"):
            continue
        if key.endswith("_lo"):
            out[key[:-3]] = values[index] | values[index + 1] << 32
        else:
            out[key] = values[index]
    for kind in ("state", "error", "where"):
        out[f"{kind}_name"] = code_name(kind, out[kind], abi_headers)
    return out


def state_words(control_rows, control_offset=0, abi_headers=None):
    """[cores, kStateWords] uint32 state words from the control blocks read back.

    `control_offset` is the control block's word offset inside each row.
    """
    abi_headers = abi_headers or abi.headers()
    rows = np.asarray(control_rows).astype(np.int64) & abi.M32
    start = control_offset + abi_headers["KM_CONTROL_STATE"] // 4
    return rows[:, start : start + abi_headers["kStateWords"]].astype(np.uint32)


def gave_up(word, sequence, abi_headers=None):
    """A gave-up word pair as a dict, or None when the kernel did not give up."""
    abi_headers = abi_headers or abi.headers()
    word = int(word)
    code = word & 0xFF
    if code == abi_headers["kGaveNone"]:
        return None
    wheres = {abi_headers[f"kWhere{where}"]: where.lower() for where in WHERES}
    gaves = {abi_headers[f"kGave{kind}"]: name for kind, name in GAVES.items()}
    out = {
        "code": gaves.get(code, code),
        "where": wheres.get((word >> 8) & 0xFF, (word >> 8) & 0xFF),
        "slot": (word >> 16) & 0xFF,
        "sequence": int(sequence),
    }
    if out["where"] in ("port", "emit"):
        out["port"] = word >> abi_headers["KM_GAVE_PORT_SHIFT"]  # the port a wait named
    if str(out["where"]).startswith("overlay"):
        out["overlay"] = word >> 16  # the overlay's number
    return out


def ports(block, abi_headers=None):
    """A core's port slots from its control block words: [{count, want}] by port."""
    abi_headers = abi_headers or abi.headers()
    first_word, stride = (
        abi_headers["KM_CONTROL_PORT"] // 4,
        abi_headers["KM_PORT_STRIDE"] // 4,
    )
    return [
        {
            "count": int(
                block[first_word + port * stride + abi_headers["KM_PORT_COUNT"]]
            ),
            "want": int(
                block[first_word + port * stride + abi_headers["KM_PORT_WANT"]]
            ),
        }
        for port in range(abi_headers["KM_PORTS"])
    ]


def broadcasts(block, abi_headers=None):
    """A core's broadcast areas from its control block words, by broadcast: VALID and
    its word, the first give-up (where, broadcast, block, sequence; None if none), the
    blocks, the wait cycles, ranks 1 to 15's DONE-BY words and marks of words taken."""
    abi_headers = abi_headers or abi.headers()
    wheres = {abi_headers[f"kWhere{where}"]: where.lower() for where in WHERES}
    out = []
    for broadcast in range(abi_headers["KM_BROADCASTS"]):
        area = (
            abi_headers["KM_CONTROL_BROADCAST"]
            + broadcast * abi_headers["KM_BROADCAST_BYTES"]
        ) // 4
        gave = abi_headers["KM_BROADCAST_GAVE"] // 4 + area
        gave = [
            int(word)
            for word in block[gave : gave + abi_headers["KM_BROADCAST_GAVE_WORDS"]]
        ]
        stats = abi_headers["KM_BROADCAST_STATS"] // 4 + area
        stats = [
            int(word)
            for word in block[stats : stats + abi_headers["KM_BROADCAST_STATS_WORDS"]]
        ]
        valid = area + abi_headers["KM_BROADCAST_VALID"] // 4
        done, step = (
            area + abi_headers["KM_BROADCAST_DONE"] // 4,
            abi_headers["KM_BROADCAST_DONE_STRIDE"] // 4,
        )
        out.append(
            {
                "valid": int(block[valid]),
                "word": int(block[valid + abi_headers["KM_BROADCAST_WORD"]]),
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
                    int(block[done + step * rank])
                    for rank in range(1, abi_headers["KM_BROADCAST_RANKS"])
                ],
                "arrived": [
                    int(block[done + step * rank + abi_headers["KM_BROADCAST_ARRIVED"]])
                    for rank in range(1, abi_headers["KM_BROADCAST_RANKS"])
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
    "Handler": "handler",
    "Load": "load",
    "Body": "body",
}  # kWhereMessages*


def dynamic(block, abi_headers=None):
    """A core's dynamic-run words from its control block: queue head and tail, messages
    sent and consumed, quota left, the first give-up (where named, position, run, the
    word seen; None if none), and as a counter tile its last run and cores served."""
    abi_headers = abi_headers or abi.headers()
    first_word = abi_headers["KM_CONTROL_MESSAGES"] // 4

    def word(offset):
        return int(block[first_word + offset // 4])

    gave = [word(abi_headers["KM_MESSAGES_GAVE"] + 4 * i) for i in range(4)]
    ready = [
        int(
            block[
                (abi_headers["KM_CONTROL_TAILS"] + abi_headers["KM_TAILS_READY"]) // 4
                + i
            ]
        )
        for i in range(4)
    ]
    names = {
        abi_headers[f"kWhereMessages{suffix}"]: label
        for suffix, label in MESSAGES_WHERES.items()
    }
    return {
        "head": word(abi_headers["KM_MESSAGES_IN_HEAD"]),
        "tail": word(abi_headers["KM_MESSAGES_IN_TAIL"]),
        "sent": word(abi_headers["KM_MESSAGES_SENT"]),
        "consumed": word(abi_headers["KM_MESSAGES_CONSUMED"]),
        "quota": word(abi_headers["KM_MESSAGES_QUOTA"]),
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


def core_state(state, core, abi_headers=None):
    """One core's state words as a dict."""
    abi_headers = abi_headers or abi.headers()
    words = [int(word) for word in state[core]]
    slots = abi_headers["KM_RING_SLOTS"]
    last, reported = abi_headers["kStLast"], abi_headers["kStReported"]
    magic, ends, wrong = (
        words[abi_headers["kStMagic"]],
        words[abi_headers["kStEnds"]],
        words[abi_headers["kStWrong"]],
    )
    return {
        "core": core,
        "magic_ok": magic >> 24 == abi_headers["kStateMagic"] >> 24
        and magic & 0xFF == core,
        "tag": (magic >> 8) & 0xFFFF,
        "runs": words[abi_headers["kStRuns"]],
        "completed": words[abi_headers["kStCompleted"]],
        "passes": words[abi_headers["kStPasses"]],
        "next_cycles": words[abi_headers["kStNextCycles"]] * 16,
        "compute_seen": words[abi_headers["kStSeen"]],
        "compute_wrong": [(wrong >> (8 * byte)) & 0xFF for byte in range(3)],
        "reader_end": ends & 0xFF,
        "writer_end": (ends >> 8) & 0xFF,
        "compute_end": (ends >> 16) & 0xFF,
        "slot_wrong": words[abi_headers["kStSlotWrong"]],
        "trace_dropped": words[abi_headers["kStTraceDropped"]],
        "reader": gave_up(
            words[abi_headers["kStReader"]],
            words[abi_headers["kStReaderSequence"]],
            abi_headers,
        ),
        "writer": gave_up(
            words[abi_headers["kStWriter"]],
            words[abi_headers["kStWriterSequence"]],
            abi_headers,
        ),
        "compute": gave_up(
            words[abi_headers["kStCompute"]],
            words[abi_headers["kStComputeSequence"]],
            abi_headers,
        ),
        "last": words[last : last + slots],
        "reported": words[reported : reported + slots],
    }


def where_names(abi_headers=None):
    """Every kWhere* value by name, lower case with underscores (the runtime's own
    names first, then a dynamic run's)."""
    abi_headers = abi_headers or abi.headers()
    names = {}
    for name, value in abi_headers.constants.items():
        if name.startswith("kWhere") and value not in names:
            words = re.findall(r"[A-Z0-9][a-z0-9]*", name[len("kWhere") :])
            names[value] = "_".join(words).lower()
    names.update({abi_headers[f"kWhere{where}"]: where.lower() for where in WHERES})
    return names


def live(words, abi_headers=None):
    """A core's live words (KM_LIVE_READ bytes) as a dict: by kernel, the runtime wait
    it is in (None: none), the command it runs (sequence 0: none) and the commands it
    has left; the writer's count of commands given up."""
    abi_headers = abi_headers or abi.headers()
    words = [int(word) for word in words]
    names = where_names(abi_headers)
    out = {}
    for role, name in enumerate(ROLES):
        where = words[abi_headers["kLvWhere"] + role]
        out[name] = {
            "where": names.get(where, where) if where else None,
            "sequence": words[abi_headers["kLvSequence"] + role],
            "slot": words[abi_headers["kLvSlot"] + role],
            "finished": words[abi_headers["kLvFinished"] + role],
        }
    out["given_up"] = words[abi_headers["kLvGivenUp"]]
    return out


def inspect(block, abi_headers=None):
    """An inspect block (uint32 words: the header, then a record per core) as a dict:
    its kind, whether it is complete, what it answers, and per core what core_state
    gives, its live words and its trace entries (None for a core it could not read)."""
    abi_headers = abi_headers or abi.headers()
    words = np.asarray(block, dtype=np.int64).ravel() & abi.M32
    header = [int(word) for word in words[: abi_headers["KMD_INSPECT_HEAD"] // 4]]
    kinds = {
        abi_headers["KMD_INSPECT_HOST"]: "host",
        abi_headers["KMD_INSPECT_TIMEOUT"]: "timeout",
    }
    out = {
        "complete": header[abi_headers["KMD_I_MARK"]]
        == abi_headers["KMD_INSPECT_MARK"],
        "kind": kinds.get(header[abi_headers["KMD_I_KIND"]]),
        **{
            key: header[abi_headers[f"KMD_I_{key.upper()}"]]
            for key in ("request", "first", "count", "done", "slot", "sequence")
        },
        "ticks": header[abi_headers["KMD_I_TICKS"]],
    }
    record, head = abi_headers["KMD_INSPECT_RECORD"] // 4, len(header)
    state_at, live_at, trace_at = (
        abi_headers[name] // 4
        for name in ("KMD_INSPECT_STATE", "KMD_INSPECT_LIVE", "KMD_INSPECT_TRACE")
    )
    entries, entry = abi_headers["KMD_INSPECT_ENTRIES"], abi_headers["kTraceWords"]
    cores = []
    for index in range(min(out["done"], (len(words) - head) // record)):
        line = words[head + index * record :][:record]
        core, flags = int(line[abi_headers["KMD_IR_CORE"]]), int(
            line[abi_headers["KMD_IR_FLAGS"]]
        )
        if not flags & abi_headers["KMD_INSPECT_READ"]:
            cores.append({"core": core, "read": False})
            continue
        state = {core: line[state_at : state_at + abi_headers["kStateWords"]]}
        sequences = line[abi_headers["KMD_IR_SEQUENCES"] :][:entries]
        trace = [
            {
                "sequence": int(sequence),
                **dict(
                    zip(
                        ("seen", "done", "sent", "extra"),
                        (int(word) for word in line[at : at + entry]),
                        strict=True,
                    )
                ),
            }
            for sequence, at in zip(
                sequences,
                range(trace_at, trace_at + entries * entry, entry),
                strict=True,
            )
            if flags & abi_headers["KMD_INSPECT_TRACED"] and line[at : at + entry].any()
        ]
        cores.append(
            core_state(state, core, abi_headers)
            | {
                "read": True,
                "live": live(
                    line[live_at : live_at + abi_headers["KM_LIVE_READ"] // 4],
                    abi_headers,
                ),
                "trace": trace,
            }
        )
    out["cores"] = cores
    return out


def not_reported(state, group, slot, sequence, abi_headers=None):
    """The cores of `group` whose reported word for `slot` is not `sequence`, with where
    each kernel gave up."""
    abi_headers = abi_headers or abi.headers()
    first, count = group
    out = []
    for core in range(first, first + count):
        report = core_state(state, core, abi_headers)
        if report["reported"][slot] != sequence:
            out.append(
                {
                    "core": core,
                    **{role: report[role] for role in ("reader", "writer", "compute")},
                }
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

    def __init__(self, rows, seqs, groups, mhz, abi_headers=None):
        self.h = abi_headers or abi.headers()
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
        times = self.command(handle)
        if times is None or not times[:, 0].any():
            return None
        abi_headers = self.h
        rel = wrap(times[:, :3] - times[:, abi_headers["kTraceSeen"]].min()) / self.mhz
        done, sent = (
            rel[:, abi_headers["kTraceDone"]],
            rel[:, abi_headers["kTraceSent"]],
        )
        return {
            "count": len(times),
            "last_seen": float(rel[:, abi_headers["kTraceSeen"]].max()),
            "last_done": float(done.max()),
            "leader_sent": float(sent[0]),
            "push_after_last_done": float(sent[0] - done.max()),
            "passes_mean": float(times[:, abi_headers["kTraceExtra"]].mean()),
        }

    def lead(self, earlier, later):
        """Microseconds by which a core of handle `later` took its command before handle
        `earlier`'s leader pushed its completion (positive: both were in flight)."""
        earlier_times, later_times = self.command(earlier), self.command(later)
        if earlier_times is None or later_times is None:
            return None
        pushed = int(earlier_times[0, self.h["kTraceSent"]])
        seen = int(later_times[:, self.h["kTraceSeen"]].min())
        delta = (pushed - seen) % (1 << 32)
        if delta >= 1 << 31:
            delta -= 1 << 32
        return delta / self.mhz
