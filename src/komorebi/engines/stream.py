"""Micro-op streams: the builder, the upload encoding, the check and the simulator.

The constants repeat `km_stream.h`; a test compares them.
"""

import struct
from collections import namedtuple
from dataclasses import dataclass

MAGIC = 0x4B4D5354
VERSION = 1
REGS = 16
SEED_REG = 15
MAX_OPS = 65535
MAX_ACTIVE = 4
LINE = 64
HEADER_BYTES = 64
UOP_BYTES = 16
TEMPLATE_BYTES = 80
TPL_NONE = 0xFF
PATCHES = 4
ARGS = 10
STEP_OPS = 64
QUEUE_JOBS = 64
QUEUE_LOOK = 16
QUEUE_PASSES = 1024
RING_SLOTS = 16
SEG_USER_MAX = 239
F_EXCL = 2
F_RESULT = 4
F_QUEUE = 8
MASK = 0xFFFFFFFF
DEFAULT_DEADLINE_PASSES = 1_000_000

OP_INVALID = 0
OP_SET, OP_MOV, OP_ADD, OP_ADDI, OP_SUB = 1, 2, 3, 4, 5
OP_MUL, OP_AND, OP_SHR, OP_MOD, OP_RAND = 6, 7, 8, 9, 10
OP_JMP, OP_JLT, OP_JEQ, OP_JNZ, OP_HALT = 16, 17, 18, 19, 20
OP_ISSUE, OP_SUBMIT, OP_WAIT, OP_RESULT, OP_EVENT, OP_TIME = 32, 33, 40, 41, 48, 49

EV_TIMEOUT = 1
EV_STREAM, EV_STREAM_END, EV_STREAM_ERROR = 4, 5, 6

(
    ERR_HEADER,
    ERR_OPCODE,
    ERR_JUMP,
    ERR_REGISTER,
    ERR_TEMPLATE,
    ERR_STEPS,
    ERR_MOD_ZERO,
    ERR_DEADLINE,
    ERR_NEVER_FITS,
    ERR_CANCELLED,
    ERR_TEAM,
) = range(1, 12)

# opcode -> (name, register fields in use, kind of imm: None, "imm", "jump", "template")
OPS = {
    OP_SET: ("SET", "a", "imm"),
    OP_MOV: ("MOV", "ab", None),
    OP_ADD: ("ADD", "abc", None),
    OP_ADDI: ("ADDI", "ab", "imm"),
    OP_SUB: ("SUB", "abc", None),
    OP_MUL: ("MUL", "abc", None),
    OP_AND: ("AND", "abc", None),
    OP_SHR: ("SHR", "ab", "imm"),
    OP_MOD: ("MOD", "abc", None),
    OP_RAND: ("RAND", "a", None),
    OP_JMP: ("JMP", "", "jump"),
    OP_JLT: ("JLT", "ab", "jump"),
    OP_JEQ: ("JEQ", "ab", "jump"),
    OP_JNZ: ("JNZ", "a", "jump"),
    OP_HALT: ("HALT", "a", None),
    OP_ISSUE: ("ISSUE", "", "template"),
    OP_SUBMIT: ("SUBMIT", "a", "template"),
    OP_WAIT: ("WAIT", "", "imm"),
    OP_RESULT: ("RESULT", "a", None),
    OP_EVENT: ("EVENT", "a", None),
    OP_TIME: ("TIME", "a", None),
}

RECORD_WORDS = 16
# word index of a0 to a9 inside a record
ARG_WORD = (3, 5, 6, 7, 9, 10, 11, 13, 14, 15)
HEADER = struct.Struct("<8I32x")
HEADER_FIELDS = (
    "magic",
    "version",
    "n_ops",
    "n_records",
    "max_steps",
    "deadline_us",
    "req",
    "seed",
)
UOP = struct.Struct("<4B3I")
TAIL = struct.Struct("<4B4B3B5B")

StreamError = namedtuple("StreamError", "reason pc")
Uop = namedtuple("Uop", "op a b c imm imm2", defaults=(0, 0, 0, 0, 0))
Event = namedtuple("Event", "kind req aux pass_index")
IssuedRecord = namedtuple(
    "IssuedRecord", "seq slot seg first count flags args pass_index pc queued"
)
SimResult = namedtuple("SimResult", "records events status passes error halt stopped")


class StreamFormatError(ValueError):
    """An upload that `check` refuses; carries the reason and program counter."""

    def __init__(self, error):
        super().__init__(f"bad stream: reason {error.reason} at pc {error.pc}")
        self.error = error


@dataclass(frozen=True)
class Reg:
    """A register of a stream."""

    index: int

    def __post_init__(self):
        if not 0 <= self.index < REGS:
            raise ValueError(f"register {self.index} out of range")


@dataclass(eq=False)
class Label:
    """A position in the program, bound now or later with `Stream.bind`."""

    position: int | None = None


@dataclass(frozen=True)
class Template:
    """A record template: its 16 words and the register patch list."""

    rec: tuple
    arg: tuple = (0, 0, 0, 0)
    reg: tuple = (0, 0, 0, 0)
    first_reg: int = TPL_NONE
    count_reg: int = TPL_NONE
    n_patch: int = 0

    @property
    def seg(self):
        return self.rec[1] & 0xFF

    @property
    def flags(self):
        return self.rec[1] >> 8 & 0xFF


def _u32(value, what):
    if not -(1 << 31) <= value < 1 << 32:
        raise ValueError(f"{what} {value} does not fit 32 bits")
    return value & MASK


def _reg(value, what):
    if not isinstance(value, Reg):
        raise TypeError(f"{what} must be a register, got {value!r}")
    return value.index


def xorshift32(x):
    """One step of the stream's generator."""
    x ^= x << 13 & MASK
    x ^= x >> 17
    x ^= x << 5 & MASK
    return x


def _lines(n):
    return -n % LINE


class Stream:
    """A micro-op program under construction, or one decoded from an upload."""

    def __init__(self, deadline_us=0, max_steps=1 << 20, seed=1, req=1):
        self.deadline_us = deadline_us
        self.max_steps = max_steps
        self.seed = seed
        self.req = req
        self._check_fields()
        self._body = []
        self._templates = []
        self._consts = {}
        self._next_reg = 0

    def _check_fields(self):
        for name, low in (
            ("deadline_us", 0),
            ("max_steps", 1),
            ("seed", 1),
            ("req", 0),
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or not low <= value <= MASK:
                raise ValueError(f"{name} must be an integer from {low} to {MASK}")

    def regs(self, n):
        """Allocate n registers (0 to 14; register 15 holds the seed)."""
        if n < 0 or self._next_reg + n > SEED_REG:
            free = SEED_REG - self._next_reg
            raise ValueError(
                f"out of registers: {n} requested, {free} free of {SEED_REG}"
                f" (register {SEED_REG} is the seed)"
            )
        out = [Reg(self._next_reg + i) for i in range(n)]
        self._next_reg += n
        return out

    def const(self, value):
        """A register set once in the prologue; equal values share a register."""
        value = _u32(value, "constant")
        if value not in self._consts:
            self._consts[value] = self.regs(1)[0]
        return self._consts[value]

    def label(self):
        """A label bound at the next micro-op."""
        return Label(len(self._body))

    def forward(self):
        """A label to bind later with `bind`."""
        return Label()

    def bind(self, label):
        if label.position is not None:
            raise ValueError("label already bound")
        label.position = len(self._body)

    def _add(self, op, a=0, b=0, c=0, imm=0):
        self._body.append(Uop(op, a, b, c, imm, 0))

    def set(self, a, value):
        self._add(OP_SET, _reg(a, "a"), imm=_u32(value, "value"))

    def mov(self, a, b):
        self._add(OP_MOV, _reg(a, "a"), _reg(b, "b"))

    def _three(self, op, a, b, c):
        self._add(op, _reg(a, "a"), _reg(b, "b"), _reg(c, "c"))

    def add(self, a, b, c):
        self._three(OP_ADD, a, b, c)

    def sub(self, a, b, c):
        self._three(OP_SUB, a, b, c)

    def mul(self, a, b, c):
        self._three(OP_MUL, a, b, c)

    def and_(self, a, b, c):
        self._three(OP_AND, a, b, c)

    def mod(self, a, b, c):
        self._three(OP_MOD, a, b, c)

    def addi(self, a, b, value):
        self._add(OP_ADDI, _reg(a, "a"), _reg(b, "b"), imm=_u32(value, "value"))

    def shr(self, a, b, amount):
        self._add(OP_SHR, _reg(a, "a"), _reg(b, "b"), imm=_u32(amount, "amount"))

    def rand(self, a):
        self._add(OP_RAND, _reg(a, "a"))

    def jmp(self, target):
        self._add(OP_JMP, imm=self._target(target))

    def jlt(self, a, b, target):
        self._add(OP_JLT, _reg(a, "a"), _reg(b, "b"), imm=self._target(target))

    def jeq(self, a, b, target):
        self._add(OP_JEQ, _reg(a, "a"), _reg(b, "b"), imm=self._target(target))

    def jnz(self, a, target):
        self._add(OP_JNZ, _reg(a, "a"), imm=self._target(target))

    def halt(self, a):
        self._add(OP_HALT, _reg(a, "a"))

    def wait(self, pending):
        if not 0 <= pending <= MASK:
            raise ValueError(f"pending {pending} must be from 0 to {MASK}")
        self._add(OP_WAIT, imm=pending)

    def result(self, a):
        self._add(OP_RESULT, _reg(a, "a"))

    def event(self, a):
        self._add(OP_EVENT, _reg(a, "a"))

    def time(self, a):
        self._add(OP_TIME, _reg(a, "a"))

    @staticmethod
    def _target(target):
        if not isinstance(target, Label):
            raise TypeError(f"a jump needs a label, got {target!r}")
        return target

    def _template(self, seg, team, args, exclusive, result):
        if not 1 <= seg <= SEG_USER_MAX:
            raise ValueError(f"segment {seg} is not a program segment (1 to 239)")
        if len(args) > ARGS:
            raise ValueError(f"{len(args)} arguments, a record has {ARGS}")
        words = [0] * RECORD_WORDS
        words[1] = (
            seg | (F_EXCL if exclusive else 0) << 8 | (F_RESULT if result else 0) << 8
        )
        patch = []
        for i, value in enumerate(args):
            if isinstance(value, Reg):
                patch.append((i, value.index))
            else:
                words[ARG_WORD[i]] = _u32(value, "argument")
        if len(patch) > PATCHES:
            raise ValueError(f"{len(patch)} register arguments, at most {PATCHES}")
        first_reg = count_reg = TPL_NONE
        first, cnt = team
        if isinstance(first, Reg):
            first_reg = first.index
        else:
            words[2] |= _team_field(first, "first")
        if isinstance(cnt, Reg):
            count_reg = cnt.index
        else:
            words[2] |= _team_field(cnt, "count") << 16
        arg = [p[0] for p in patch] + [0] * (PATCHES - len(patch))
        reg = [p[1] for p in patch] + [0] * (PATCHES - len(patch))
        self._templates.append(
            Template(
                tuple(words), tuple(arg), tuple(reg), first_reg, count_reg, len(patch)
            )
        )
        return len(self._templates) - 1

    def issue(self, seg, team, args=(), exclusive=False, result=False):
        """ISSUE: a record on `team` = (first, count), each an int or a register."""
        index = self._template(seg, team, list(args), exclusive, result)
        self._add(OP_ISSUE, imm=index)

    def submit(self, seg, count, args=(), exclusive=False, result=False):
        """SUBMIT: a job of `count` cores (int or register); first is chosen."""
        reg = count if isinstance(count, Reg) else self.const(count)
        index = self._template(seg, (0, 0), list(args), exclusive, result)
        self._add(OP_SUBMIT, _reg(reg, "count"), imm=index)

    def uops(self):
        """The micro-ops as encoded: constant prologue, body, labels resolved."""
        prologue = [Uop(OP_SET, r.index, 0, 0, v, 0) for v, r in self._consts.items()]
        out = list(prologue)
        for op in self._body:
            if isinstance(op.imm, Label):
                if op.imm.position is None:
                    raise ValueError("a label was never bound")
                if op.imm.position >= len(self._body):
                    raise ValueError("a label is bound past the last micro-op")
                op = op._replace(imm=op.imm.position + len(prologue))
            out.append(op)
        return out

    def templates(self):
        return list(self._templates)

    def encode(self):
        """The upload: header, micro-ops, templates, each part padded to a line."""
        self._check_fields()
        ops = self.uops()
        if len(ops) > MAX_OPS:
            raise ValueError(f"{len(ops)} micro-ops, at most {MAX_OPS}")
        if len(self._templates) > MAX_OPS:
            raise ValueError(f"{len(self._templates)} templates, at most {MAX_OPS}")
        head = [MAGIC, VERSION, len(ops), len(self._templates), self.max_steps]
        head += [self.deadline_us, self.req, self.seed]
        data = HEADER.pack(*head)
        data += b"".join(UOP.pack(o.op, o.a, o.b, o.c, o.imm, o.imm2, 0) for o in ops)
        data += bytes(_lines(len(ops) * UOP_BYTES))
        for t in self._templates:
            data += struct.pack("<16I", *t.rec)
            data += TAIL.pack(
                *t.arg, *t.reg, t.first_reg, t.count_reg, t.n_patch, *[0] * 5
            )
        return data + bytes(_lines(len(self._templates) * TEMPLATE_BYTES))

    @classmethod
    def decode(cls, data):
        """The stream an upload describes; StreamFormatError if `check` refuses it."""
        error = check(data, len(data))
        if error:
            raise StreamFormatError(error)
        h = _header(data)
        s = cls(h["deadline_us"], h["max_steps"], h["seed"], h["req"])
        s._body, s._templates = _parse(data, h)
        return s

    def simulate(self, **options):
        """`simulate` on this stream's upload."""
        return simulate(self.encode(), **options)


def _team_field(value, what):
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"team {what} {value} does not fit 16 bits")
    return value


def _header(data):
    w = HEADER.unpack_from(data)
    return dict(zip(HEADER_FIELDS, w))


def _size(h):
    ops = h["n_ops"] * UOP_BYTES
    tpl = h["n_records"] * TEMPLATE_BYTES
    return HEADER_BYTES + ops + _lines(ops) + tpl + _lines(tpl)


def _parse(data, h):
    ops = []
    for i in range(h["n_ops"]):
        op, a, b, c, imm, imm2, _ = UOP.unpack_from(data, HEADER_BYTES + i * UOP_BYTES)
        ops.append(Uop(op, a, b, c, imm, imm2))
    base = HEADER_BYTES + h["n_ops"] * UOP_BYTES
    base += _lines(h["n_ops"] * UOP_BYTES)
    templates = []
    for i in range(h["n_records"]):
        at = base + i * TEMPLATE_BYTES
        rec = struct.unpack_from("<16I", data, at)
        *tail, _p0, _p1, _p2, _p3, _p4 = TAIL.unpack_from(data, at + 64)
        templates.append(
            Template(
                rec, tuple(tail[0:4]), tuple(tail[4:8]), tail[8], tail[9], tail[10]
            )
        )
    return ops, templates


def check(data, region_size):
    """None if the kernel accepts the upload, else `StreamError(reason, pc)`.

    Faults are found in the order header, micro-ops, templates; the first one wins.
    """
    if len(data) < HEADER_BYTES or len(data) > region_size or len(data) % LINE:
        return StreamError(ERR_HEADER, 0)
    h = _header(data)
    if h["magic"] != MAGIC or h["version"] != VERSION or h["seed"] == 0:
        return StreamError(ERR_HEADER, 0)
    if not 1 <= h["n_ops"] <= MAX_OPS or h["n_records"] > MAX_OPS:
        return StreamError(ERR_HEADER, 0)
    if h["max_steps"] == 0 or len(data) != _size(h):
        return StreamError(ERR_HEADER, 0)
    ops, templates = _parse(data, h)
    first_use = {}
    for pc, o in enumerate(ops):
        if o.op not in OPS:
            return StreamError(ERR_OPCODE, pc)
        _, fields, kind = OPS[o.op]
        if any(getattr(o, f) >= REGS for f in fields):
            return StreamError(ERR_REGISTER, pc)
        if kind == "jump" and o.imm >= h["n_ops"]:
            return StreamError(ERR_JUMP, pc)
        if kind == "template":
            if o.imm >= h["n_records"]:
                return StreamError(ERR_TEMPLATE, pc)
            first_use.setdefault(o.imm, pc)
    for index, t in enumerate(templates):
        if _bad_template(t):
            return StreamError(ERR_TEMPLATE, first_use.get(index, 0))
    return None


def check_run(data, offset, region_size, active=(), active_reqs=()):
    """What the kernel answers a `KM_SEG_RUN` for `data` at `offset`: (error, req).

    `error` is None to start the stream, else reason 1 for a bad range or a reused `req`
    or what `check` finds; `req` is the header's, or 0 if it is outside the region.
    """
    length = len(data)
    head_inside = offset <= region_size - HEADER_BYTES
    req = 0
    if head_inside and length >= 28:
        req = _header(data.ljust(HEADER_BYTES, b"\0"))["req"]
    refused = StreamError(ERR_HEADER, 0)
    if offset % LINE or offset > region_size or length > region_size - offset:
        return refused, req
    if any(offset < o + n and o < offset + length for o, n in active):
        return refused, req
    if req in active_reqs:
        return refused, req
    return check(data, region_size), req


def _bad_template(t):
    if not 1 <= t.seg <= SEG_USER_MAX or t.flags & ~(F_EXCL | F_RESULT):
        return True
    if t.n_patch > PATCHES:
        return True
    if any(t.arg[i] >= ARGS or t.reg[i] >= REGS for i in range(t.n_patch)):
        return True
    return any(r != TPL_NONE and r >= REGS for r in (t.first_reg, t.count_reg))


def team_outside(first, count, n_cores):
    """True if cores [first, first + count) are not in the grid; forms no sum."""
    return count == 0 or first > n_cores or count > n_cores - first


def alloc(busy, count, n_cores):
    """The lowest `first` whose cores [first, first + count) overlap no (first, count)
    interval in `busy`, or None if there is no room now."""
    for first in range(n_cores - count + 1):
        if all(first + count <= f or f + c <= first for f, c in busy):
            return first
    return None


@dataclass
class _Busy:
    record: IssuedRecord
    done_pass: int | None
    result: int
    deadline_pass: int | None


@dataclass
class _Job:
    words: list
    count: int
    pc: int
    passed: int = 0


def simulate(
    data,
    *,
    n_cores=119,
    delay=0,
    result=None,
    deadline_passes=None,
    default_deadline_passes=DEFAULT_DEADLINE_PASSES,
    cancel_pass=None,
    max_passes=100_000,
    ticks_per_pass=1000,
    region_size=1 << 20,
):
    """Run an upload as the kernel would and return a `SimResult`.

    Pass order, completion model and parameters: docs/streams.md, "The simulator".
    """
    for name, value in (
        ("deadline_passes", deadline_passes),
        ("default_deadline_passes", default_deadline_passes),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be at least 1, a deadline is never zero")
    bad = check(data, region_size)
    if bad:
        req = struct.unpack_from("<I", data, 24)[0] if len(data) >= 28 else 0
        events = [Event(EV_STREAM_ERROR, req, bad.reason | bad.pc << 8, 0)]
        return SimResult([], events, "error", 1, bad, None, False)
    h = _header(data)
    ops, templates = _parse(data, h)
    if deadline_passes is None and h["deadline_us"] == 0:
        deadline_passes = default_deadline_passes
    sim = _Sim(ops, templates, h, n_cores, delay, result, deadline_passes)
    sim.ticks = ticks_per_pass
    for p in range(max_passes):
        sim.run_pass(p, cancel_pass == p)
        if sim.error is not None or sim.halt is not None:
            if sim.error is not None and sim.error.reason != ERR_DEADLINE:
                sim.drain(p + 1, max_passes)
            status = "error" if sim.error else "ended"
            return SimResult(
                sim.records,
                sim.events,
                status,
                sim.events[-1].pass_index + 1,
                sim.error,
                sim.halt,
                sim.stopped,
            )
    return SimResult(sim.records, sim.events, "stuck", max_passes, None, None, False)


class _Sim:
    """The state of one stream, the completion table and the queue."""

    def __init__(self, ops, templates, h, n_cores, delay, result, deadline_passes):
        self.ops, self.templates, self.h = ops, templates, h
        self.n_cores, self.delay, self.result_fn = n_cores, delay, result
        self.deadline_passes = deadline_passes
        self.regs = [0] * REGS
        self.regs[SEED_REG] = h["seed"]
        self.table = [None] * RING_SLOTS
        self.queue, self.records, self.events = [], [], []
        self.pc = self.steps = self.seq = self.pending = self.last_result = 0
        self.error = self.halt = None
        self.ticks = 1000
        self.now = 0
        self.stopped = False

    def run_pass(self, p, cancel):
        """One pass. Issuers are the streams in activation order, then the queue; in
        pass p the turns start at issuer p % (streams + 1) and go on cyclically."""
        self.now = p
        self.sweep()
        self.check_deadlines()
        if cancel and self.error is None:
            self.fail(ERR_CANCELLED, self.pc)
        turns = [self.step_stream, self.step_queue]
        start = p % len(turns)
        for turn in turns[start:] + turns[:start]:
            if self.error is None:
                turn()

    def drain(self, start, limit):
        """After a stream error its records still run; a late one stops the grid."""
        for p in range(start, limit):
            self.now = p
            self.sweep()
            self.check_deadlines()
            if self.stopped or not any(self.table):
                return

    def post(self, kind, aux):
        self.events.append(Event(kind, self.h["req"], aux, self.now))

    def fail(self, reason, pc):
        self.error = StreamError(reason, pc)
        self.post(EV_STREAM_ERROR, reason | pc << 8)
        self.queue.clear()
        return "error"

    def sweep(self):
        """Free the slots whose acknowledgement is due, in slot order."""
        for slot, busy in enumerate(self.table):
            if busy and busy.done_pass is not None and busy.done_pass <= self.now:
                if busy.record.flags & F_RESULT:
                    self.last_result = busy.result
                self.table[slot] = None
                self.pending -= 1

    def check_deadlines(self):
        """The lowest-slot late record stops the grid: it ends the stream, or if the
        stream already ended, posts a timeout event for that slot."""
        for slot, busy in enumerate(self.table):
            if (
                busy
                and busy.deadline_pass is not None
                and self.now >= busy.deadline_pass
            ):
                self.stopped = True
                if self.error is None:
                    self.fail(ERR_DEADLINE, busy.record.pc)
                else:
                    self.post(EV_TIMEOUT, slot)
                return

    def busy_teams(self):
        return [(b.record.first, b.record.count) for b in self.table if b]

    def free_slot(self):
        return next((i for i, b in enumerate(self.table) if b is None), None)

    def put(self, slot, words, first, count, pc, queued):
        self.seq += 1
        args = tuple(words[i] for i in ARG_WORD)
        flags = words[1] >> 8 & 0xFF & ~F_QUEUE
        seg = words[1] & 0xFF
        rec = IssuedRecord(
            self.seq, slot, seg, first, count, flags, args, self.now, pc, queued
        )
        self.records.append(rec)
        delay = self.delay(rec) if callable(self.delay) else self.delay
        done = None if delay is None else self.now + 1 + delay
        value = self.result_fn(rec) & MASK if self.result_fn else 0
        limit = (
            None if self.deadline_passes is None else self.now + self.deadline_passes
        )
        self.table[slot] = _Busy(rec, done, value, limit)
        if not queued:
            self.pending += 1

    def step_stream(self):
        regs = self.regs
        for _ in range(STEP_OPS):
            if self.pc >= len(self.ops):
                return self.fail(ERR_JUMP, self.pc)
            if self.steps >= self.h["max_steps"]:
                return self.fail(ERR_STEPS, self.pc)
            o = self.ops[self.pc]
            nxt = self.pc + 1
            if o.op in (OP_ISSUE, OP_SUBMIT, OP_WAIT, OP_HALT):
                state = self.work(o)
                if state != "ok":
                    return state
            elif o.op == OP_SET:
                regs[o.a] = o.imm
            elif o.op == OP_MOV:
                regs[o.a] = regs[o.b]
            elif o.op == OP_ADD:
                regs[o.a] = regs[o.b] + regs[o.c] & MASK
            elif o.op == OP_ADDI:
                regs[o.a] = regs[o.b] + o.imm & MASK
            elif o.op == OP_SUB:
                regs[o.a] = regs[o.b] - regs[o.c] & MASK
            elif o.op == OP_MUL:
                regs[o.a] = regs[o.b] * regs[o.c] & MASK
            elif o.op == OP_AND:
                regs[o.a] = regs[o.b] & regs[o.c]
            elif o.op == OP_SHR:
                regs[o.a] = regs[o.b] >> (o.imm & 31)
            elif o.op == OP_MOD:
                if regs[o.c] == 0:
                    return self.fail(ERR_MOD_ZERO, self.pc)
                regs[o.a] = regs[o.b] % regs[o.c]
            elif o.op == OP_RAND:
                regs[SEED_REG] = xorshift32(regs[SEED_REG])
                regs[o.a] = regs[SEED_REG]
            elif o.op == OP_JMP:
                nxt = o.imm
            elif o.op == OP_JLT:
                nxt = o.imm if regs[o.a] < regs[o.b] else nxt
            elif o.op == OP_JEQ:
                nxt = o.imm if regs[o.a] == regs[o.b] else nxt
            elif o.op == OP_JNZ:
                nxt = o.imm if regs[o.a] != 0 else nxt
            elif o.op == OP_RESULT:
                regs[o.a] = self.last_result
            elif o.op == OP_EVENT:
                self.post(EV_STREAM, regs[o.a] & 0xFFFFFF)
            elif o.op == OP_TIME:
                regs[o.a] = self.now * self.ticks & MASK
            self.steps += 1
            self.pc = nxt
        return "ok"

    def work(self, o):
        """ISSUE, SUBMIT, WAIT and HALT: "ok" (done), "stall", "error" or "end"."""
        if o.op == OP_WAIT:
            return "stall" if self.pending > o.imm else "ok"
        if o.op == OP_HALT:
            if self.pending:
                return "stall"
            self.halt = self.regs[o.a]
            self.post(EV_STREAM_END, self.halt & 0xFFFFFF)
            return "end"
        return self.do_issue(o) if o.op == OP_ISSUE else self.do_submit(o)

    def do_issue(self, o):
        t, regs = self.templates[o.imm], self.regs
        first = t.rec[2] & 0xFFFF if t.first_reg == TPL_NONE else regs[t.first_reg]
        count = t.rec[2] >> 16 if t.count_reg == TPL_NONE else regs[t.count_reg]
        if team_outside(first, count, self.n_cores):
            return self.fail(ERR_TEAM, self.pc)
        slot = self.free_slot()
        if slot is None:
            return "stall"
        if t.flags & F_EXCL and (
            self.queue  # a job queued before it is not placed yet
            or any(first < f + c and f < first + count for f, c in self.busy_teams())
        ):
            return "stall"
        self.put(slot, self.patched(t), first, count, self.pc, False)
        return "ok"

    def do_submit(self, o):
        count = self.regs[o.a]
        if count == 0 or count > self.n_cores:
            return self.fail(ERR_NEVER_FITS, self.pc)
        if len(self.queue) >= QUEUE_JOBS:
            return "stall"
        job = _Job(self.patched(self.templates[o.imm]), count, self.pc)
        self.queue.append(job)
        self.pending += 1
        return "ok"

    def patched(self, t):
        words = list(t.rec)
        for i in range(t.n_patch):
            words[ARG_WORD[t.arg[i]]] = self.regs[t.reg[i]]
        return words

    def step_queue(self):
        """Issue what fits among the first 16 jobs; 1,024 jobs may overtake one."""
        skipped = []
        for job in list(self.queue[:QUEUE_LOOK]):
            if any(j.passed >= QUEUE_PASSES for j in skipped):
                break
            slot = self.free_slot()
            first = None
            if slot is not None:
                first = alloc(self.busy_teams(), job.count, self.n_cores)
            if first is None:
                skipped.append(job)
                continue
            for j in skipped:
                j.passed += 1
            self.queue.remove(job)
            self.put(slot, job.words, first, job.count, job.pc, True)
