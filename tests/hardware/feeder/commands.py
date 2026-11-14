"""Commands, start-up groups and results of the test feeder, as pure functions.

Constants and field orders are read from the C headers, so the host and the kernels
cannot disagree on them."""

import re
from pathlib import Path

import numpy as np

from komorebi import abi

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
        fields += [field.strip() for field in decl.split(",")]
    return fields


# the ABI's and the device's constants as the package reads them, then the feeder's
CONSTANTS = {**abi.headers().constants, **constants(FEEDER_H)}
K = CONSTANTS  # the name examples/02_merge_tree and examples/03_broadcast still import
COMMAND_FIELDS = struct_fields(ABI_H, "km_command_t")
ARG_FIELDS = struct_fields(CORE_HPP, "Args")
ARG_WORDS = [f"a{i}" for i in range(10)]


def command(handler, first, count, args=(), flags=0):
    """The 16 words of a command; the feeder fills sequence, slot and landing words."""
    if len(args) > len(ARG_WORDS):
        raise ValueError(f"{len(args)} arguments, a command has {len(ARG_WORDS)}")
    words = dict.fromkeys(COMMAND_FIELDS, 0)
    words["hdr"] = handler | flags << 8  # flag bit 0 is reserved and zero
    words["group"] = first | count << 16
    words.update(zip(ARG_WORDS, args))
    return [words[field] & 0xFFFFFFFF for field in COMMAND_FIELDS]


def partition(size, index, count):
    """(start, length) of rank `index` of `count` in n items, as km::partition."""
    base, extra = divmod(size, count)
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
    grid_x = grid[0]
    out = {}
    columns = 0
    for x in range(grid_x):
        column = [i for i in range(x, workers, grid_x)]
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
    return [values[field] & 0xFFFFFFFF for field in ARG_FIELDS]


def limit_units(seconds, mhz):
    """A timeout in the kernels' units of 1024 cycles at mhz."""
    return max(1, int(seconds * mhz * 1e6) >> 10)


def random_groups(rng, jobs, workers):
    """n groups (first, count) of cores 0 to workers - 1, count and first uniform."""
    groups = []
    for _ in range(jobs):
        count = int(rng.integers(1, workers + 1))
        first = int(rng.integers(0, workers - count + 1))
        groups.append((first, count))
    return groups


def feeder_summary(rows):
    """Row 0 of the feeder's results as a dict."""
    row = as_uint32(rows)[0]
    status = {
        CONSTANTS["kFsNone"]: "none",
        CONSTANTS["kFsDone"]: "done",
        CONSTANTS["kFsTimeout"]: "timeout",
        CONSTANTS["kFsBoot"]: "boot",
        CONSTANTS["kFsMessagesGave"]: "messages_gave",
    }
    return {
        "magic_ok": int(row[CONSTANTS["kFrMagic"]]) == CONSTANTS["kFeederMagic"],
        "status": status.get(
            int(row[CONSTANTS["kFrStatus"]]), int(row[CONSTANTS["kFrStatus"]])
        ),
        "slot": int(row[CONSTANTS["kFrSlot"]]),
        "sequence": int(row[CONSTANTS["kFrSequence"]]),
        "index": int(row[CONSTANTS["kFrIndex"]]),
        "submitted": int(row[CONSTANTS["kFrSubmitted"]]),
        "done": int(row[CONSTANTS["kFrDone"]]),
        "boot_cycles": int(row[CONSTANTS["kFrBoot"]]) * 16,
        "span_cycles": int(row[CONSTANTS["kFrSpanLo"]])
        | int(row[CONSTANTS["kFrSpanHi"]]) << 32,
        "submit_cycles": int(row[CONSTANTS["kFrSubmitCycles"]]),
        "max_busy": int(row[CONSTANTS["kFrMaxBusy"]]),
        "ready_mask": int(row[CONSTANTS["kFrReady"]]),
        "exit_acked": bool(row[CONSTANTS["kFrExitAcked"]]),
    }


def feeder_commands(rows):
    """Per command (rows 1 to n): sequence, slot, status, submit, submitted, acked,
    busy, held, result."""
    records = as_uint32(rows)[1:]
    names = (
        "RecSequence",
        "RecSlot",
        "RecStatus",
        "CmdSubmit",
        "CmdSubmitted",
        "RecAcked",
    )
    out = {
        name[3:].lower(): records[:, CONSTANTS[f"kFr{name}"]].astype(np.int64)
        for name in names
    }
    out["busy"] = records[:, CONSTANTS["kFrRecBusy"]].astype(np.int64)
    out["held"] = records[:, CONSTANTS["kFrRecHeld"]].astype(np.int64)
    out["result"] = records[:, CONSTANTS["kFrRecResult"]].astype(np.int64)
    return out


def state_words(control_rows):
    """[cores, kStateWords] uint32 state words from the control blocks read back."""
    start = CONSTANTS["KM_CONTROL_STATE"] // 4
    return as_uint32(control_rows)[:, start : start + CONSTANTS["kStateWords"]]


def port_words(control_rows):
    """[cores, KM_PORTS, 2] uint32 (count, wanted) of every port, from the blocks."""
    start, stride = CONSTANTS["KM_CONTROL_PORT"] // 4, CONSTANTS["KM_PORT_STRIDE"] // 4
    words = as_uint32(control_rows)[:, start : start + CONSTANTS["KM_PORTS"] * stride]
    words = words.reshape(len(words), CONSTANTS["KM_PORTS"], stride)
    return words[:, :, [CONSTANTS["KM_PORT_COUNT"], CONSTANTS["KM_PORT_WANT"]]]


def waiting_ports(ports, core):
    """Ports of `core` still wanting arrivals: {port: missing count}."""
    return {
        port: int(want) - int(count)
        for port, (count, want) in enumerate(ports[core])
        if want != 0
    }


GAVES = {
    CONSTANTS["kGaveStop"]: "stop",
    CONSTANTS["kGaveLimit"]: "limit",
    CONSTANTS["kGaveBadItem"]: "bad_item",
}


def gave_up(word, sequence):
    """A gave-up word pair as a dict, or None when the kernel did not give up."""
    code = word & 0xFF
    if code == CONSTANTS["kGaveNone"]:
        return None
    wheres = {
        value: name[6:].lower()
        for name, value in CONSTANTS.items()
        if name.startswith("kWhere")
    }
    where = (word >> 8) & 0xFF
    out = {
        "code": GAVES.get(code, code),
        "where": wheres.get(where, where),
        "slot": (word >> 16) & 0xFF,
        "sequence": int(sequence),
    }
    if where in (
        CONSTANTS["kWherePort"],
        CONSTANTS["kWhereEmit"],
    ):  # the top byte names the port
        out["port"] = word >> 24
    return out


def core_state(words_by_core, core, tag=None):
    """One core's state words as a dict (`tag`: the run's tag, checked with magic)."""
    words = [int(value) for value in words_by_core[core]]
    magic = words[CONSTANTS["kStMagic"]]
    ends, wrong = words[CONSTANTS["kStEnds"]], words[CONSTANTS["kStWrong"]]
    return {
        "core": core,
        "magic_ok": magic >> 24 == CONSTANTS["kStateMagic"] >> 24
        and magic & 0xFF == core
        and (tag is None or (magic >> 8) & 0xFFFF == tag),
        "runs": words[CONSTANTS["kStRuns"]],
        "completed": words[CONSTANTS["kStCompleted"]],
        "passes": words[CONSTANTS["kStPasses"]],
        "next_cycles": words[CONSTANTS["kStNextCycles"]] * 16,
        "compute_seen": words[CONSTANTS["kStSeen"]],
        "compute_wrong": [(wrong >> (8 * position)) & 0xFF for position in range(3)],
        "reader_end": ends & 0xFF,
        "writer_end": (ends >> 8) & 0xFF,
        "compute_end": (ends >> 16) & 0xFF,
        "slot_wrong": words[CONSTANTS["kStSlotWrong"]],
        "trace_dropped": words[CONSTANTS["kStTraceDropped"]],
        "reader": gave_up(
            words[CONSTANTS["kStReader"]], words[CONSTANTS["kStReaderSequence"]]
        ),
        "writer": gave_up(
            words[CONSTANTS["kStWriter"]], words[CONSTANTS["kStWriterSequence"]]
        ),
        "compute": gave_up(
            words[CONSTANTS["kStCompute"]], words[CONSTANTS["kStComputeSequence"]]
        ),
        "last": words[CONSTANTS["kStLast"] : CONSTANTS["kStLast"] + 16],
        "reported": words[CONSTANTS["kStReported"] : CONSTANTS["kStReported"] + 16],
    }


def wrap(delta):
    """Differences of low clock words, mod 2**32, as int64."""
    return np.asarray(delta, dtype=np.int64) % (1 << 32)
