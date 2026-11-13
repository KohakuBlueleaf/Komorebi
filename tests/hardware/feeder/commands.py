"""Commands, start-up groups and results of the test feeder, as pure functions.

Constants and field orders are read from the C headers, so the host and the kernels
cannot disagree on them."""

import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INCLUDE = ROOT / "src" / "komorebi_native" / "include"
ABI_H = INCLUDE / "komorebi" / "abi" / "km_abi.h"
CORE_HPP = INCLUDE / "komorebi" / "device" / "km_core.hpp"
FEEDER_H = HERE / "feeder.h"

NUMBER = r"(0x[0-9A-Fa-f]+|\d+)u?"
DEFINE = re.compile(rf"^#define\s+(KM_\w+)\s+{NUMBER}\b", re.M)
CONSTEXPR = re.compile(rf"constexpr\s+uint32_t\s+(k\w+)\s*=\s*{NUMBER}\s*;")


def constants(*paths):
    """Every `#define KM_* <n>` and `constexpr uint32_t k* = <n>;` in paths."""
    found = {}
    for path in paths:
        text = Path(path).read_text()
        for pattern in (DEFINE, CONSTEXPR):
            for name, value in pattern.findall(text):
                found[name] = int(value, 0)
    return found


def struct_fields(path, name):
    """Field names in order of `typedef struct {...} name;` or `struct name {...};`."""
    text = Path(path).read_text()
    body = re.search(rf"typedef struct \{{(.*?)\}}\s*{name};", text, re.S) or re.search(
        rf"struct {name} \{{(.*?)\}};", text, re.S
    )
    if body is None:
        raise ValueError(f"no struct {name} in {path}")
    fields = []
    for decl in re.findall(
        r"uint32_t\s+([^;]+);", re.sub(r"//.*|/\*.*?\*/", "", body[1])
    ):
        fields += [f.strip() for f in decl.split(",")]
    return fields


K = constants(ABI_H, CORE_HPP, FEEDER_H)
COMMAND_FIELDS = struct_fields(ABI_H, "km_command_t")
ARG_FIELDS = struct_fields(CORE_HPP, "Args")
ARG_WORDS = [f"a{i}" for i in range(10)]


def command(handler, first, count, args=(), flags=0):
    """The 16 words of a command; the feeder fills seq, slot and the landing words."""
    if len(args) > len(ARG_WORDS):
        raise ValueError(f"{len(args)} arguments, a command has {len(ARG_WORDS)}")
    words = dict.fromkeys(COMMAND_FIELDS, 0)
    words["hdr"] = handler | flags << 8  # flag bit 0 is reserved and zero
    words["group"] = first | count << 16
    words.update(zip(ARG_WORDS, args))
    return [words[f] & 0xFFFFFFFF for f in COMMAND_FIELDS]


def partition(n, index, count):
    """(start, length) of rank `index` of `count` in n items, as km::partition."""
    base, extra = divmod(n, count)
    return index * base + min(index, extra), base + (index < extra)


def as_int32(rows):
    """An int32 array with the bits of the uint32 values in rows."""
    return np.asarray(rows, dtype=np.uint64).astype(np.uint32).view(np.int32)


def as_uint32(array):
    """The uint32 view of an int32 array read back from the card."""
    return np.asarray(array, dtype=np.int32).view(np.uint32)


def groups(grid, workers):
    """Start-up column groups of cores 0 to workers - 1 of a (gx, gy) grid.

    Returns ({core: (leader, rank, peers, ready_slot)}, number of groups): the
    lowest core of each column leads; ready_slot is the column."""
    gx = grid[0]
    out = {}
    columns = 0
    for x in range(gx):
        column = [i for i in range(x, workers, gx)]
        if not column:
            continue
        columns += 1
        leader = column[0]
        out[leader] = (leader, 0, len(column) - 1, x)
        for rank, core in enumerate(column[1:]):
            out[core] = (leader, rank, 0, x)
    return out, columns


def start_args(values):
    """The start-up arguments in the order of km::Args, from a dict of every field."""
    missing = set(ARG_FIELDS) - set(values)
    if missing:
        raise ValueError(f"missing start-up arguments: {sorted(missing)}")
    return [values[f] & 0xFFFFFFFF for f in ARG_FIELDS]


def limit_units(seconds, mhz):
    """A timeout in the kernels' units of 1024 cycles at mhz."""
    return max(1, int(seconds * mhz * 1e6) >> 10)


def random_groups(rng, n, workers):
    """n groups (first, count) of cores 0 to workers - 1, count and first uniform."""
    groups = []
    for _ in range(n):
        count = int(rng.integers(1, workers + 1))
        first = int(rng.integers(0, workers - count + 1))
        groups.append((first, count))
    return groups


def feeder_summary(rows):
    """Row 0 of the feeder's results as a dict."""
    r = as_uint32(rows)[0]
    status = {
        K["kFsNone"]: "none",
        K["kFsDone"]: "done",
        K["kFsTimeout"]: "timeout",
        K["kFsBoot"]: "boot",
        K["kFsDynGave"]: "dyn_gave",
    }
    return {
        "magic_ok": int(r[K["kFrMagic"]]) == K["kFeederMagic"],
        "status": status.get(int(r[K["kFrStatus"]]), int(r[K["kFrStatus"]])),
        "slot": int(r[K["kFrSlot"]]),
        "seq": int(r[K["kFrSeq"]]),
        "index": int(r[K["kFrIndex"]]),
        "submitted": int(r[K["kFrSubmitted"]]),
        "done": int(r[K["kFrDone"]]),
        "boot_cycles": int(r[K["kFrBoot"]]) * 16,
        "span_cycles": int(r[K["kFrSpanLo"]]) | int(r[K["kFrSpanHi"]]) << 32,
        "submit_cycles": int(r[K["kFrSubmitCycles"]]),
        "max_busy": int(r[K["kFrMaxBusy"]]),
        "ready_mask": int(r[K["kFrReady"]]),
        "exit_acked": bool(r[K["kFrExitAcked"]]),
    }


def feeder_commands(rows):
    """Per command (rows 1 to n): seq, slot, status, submit, submitted, acked, busy,
    held, result."""
    r = as_uint32(rows)[1:]
    names = ("RecSeq", "RecSlot", "RecStatus", "CmdSubmit", "CmdSubmitted", "RecAcked")
    out = {n[3:].lower(): r[:, K[f"kFr{n}"]].astype(np.int64) for n in names}
    out["busy"] = r[:, K["kFrRecBusy"]].astype(np.int64)
    out["held"] = r[:, K["kFrRecHeld"]].astype(np.int64)
    out["result"] = r[:, K["kFrRecResult"]].astype(np.int64)
    return out


def state_words(ctl_rows):
    """[cores, kStateWords] uint32 state words from the control blocks read back."""
    start = K["KM_CTL_STATE"] // 4
    return as_uint32(ctl_rows)[:, start : start + K["kStateWords"]]


def port_words(ctl_rows):
    """[cores, KM_PORTS, 2] uint32 (count, wanted) of every port, from the blocks."""
    start, stride = K["KM_CTL_PORT"] // 4, K["KM_PORT_STRIDE"] // 4
    words = as_uint32(ctl_rows)[:, start : start + K["KM_PORTS"] * stride]
    words = words.reshape(len(words), K["KM_PORTS"], stride)
    return words[:, :, [K["KM_PORT_COUNT"], K["KM_PORT_WANT"]]]


def waiting_ports(ports, core):
    """Ports of `core` still wanting arrivals: {port: missing count}."""
    return {
        p: int(want) - int(count)
        for p, (count, want) in enumerate(ports[core])
        if want != 0
    }


GAVES = {
    K["kGaveStop"]: "stop",
    K["kGaveLimit"]: "limit",
    K["kGaveBadItem"]: "bad_item",
}


def gave_up(word, seq):
    """A gave-up word pair as a dict, or None when the kernel did not give up."""
    code = word & 0xFF
    if code == K["kGaveNone"]:
        return None
    wheres = {v: k[6:].lower() for k, v in K.items() if k.startswith("kWhere")}
    where = (word >> 8) & 0xFF
    out = {
        "code": GAVES.get(code, code),
        "where": wheres.get(where, where),
        "slot": (word >> 16) & 0xFF,
        "seq": int(seq),
    }
    if where in (K["kWherePort"], K["kWhereEmit"]):  # the top byte names the port
        out["port"] = word >> 24
    return out


def core_state(st, core, tag=None):
    """One core's state words as a dict (`tag`: the run's tag, checked with magic)."""
    s = [int(v) for v in st[core]]
    magic = s[K["kStMagic"]]
    ends, wrong = s[K["kStEnds"]], s[K["kStWrong"]]
    return {
        "core": core,
        "magic_ok": magic >> 24 == K["kStateMagic"] >> 24
        and magic & 0xFF == core
        and (tag is None or (magic >> 8) & 0xFFFF == tag),
        "runs": s[K["kStRuns"]],
        "completed": s[K["kStCompleted"]],
        "passes": s[K["kStPasses"]],
        "next_cycles": s[K["kStNextCycles"]] * 16,
        "compute_seen": s[K["kStSeen"]],
        "compute_wrong": [(wrong >> (8 * b)) & 0xFF for b in range(3)],
        "reader_end": ends & 0xFF,
        "writer_end": (ends >> 8) & 0xFF,
        "compute_end": (ends >> 16) & 0xFF,
        "slot_wrong": s[K["kStSlotWrong"]],
        "trace_dropped": s[K["kStTraceDropped"]],
        "reader": gave_up(s[K["kStReader"]], s[K["kStReaderSeq"]]),
        "writer": gave_up(s[K["kStWriter"]], s[K["kStWriterSeq"]]),
        "compute": gave_up(s[K["kStCompute"]], s[K["kStComputeSeq"]]),
        "last": s[K["kStLast"] : K["kStLast"] + 16],
        "reported": s[K["kStReported"] : K["kStReported"] + 16],
    }


def wrap(delta):
    """Differences of low clock words, mod 2**32, as int64."""
    return np.asarray(delta, dtype=np.int64) % (1 << 32)
