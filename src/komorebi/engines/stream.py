"""Micro-op streams: the builder, the upload encoding, the check and the simulator.

The constants repeat `km_stream.h`; a test compares them.
"""

import struct
from collections import namedtuple
from dataclasses import dataclass

MAGIC = 0x4B4D5354
VERSION = 1
REGISTERS = 16
SEED_REGISTER = 15
MAX_OPS = 65535
MAX_ACTIVE = 4
LINE = 64
HEADER_BYTES = 64
UOP_BYTES = 16
TEMPLATE_BYTES = 80
TEMPLATE_NONE = 0xFF
PATCHES = 4
ARGS = 10
STEP_OPS = 64
QUEUE_JOBS = 64
QUEUE_LOOK = 16
QUEUE_PASSES = 1024
RING_SLOTS = 16
HANDLER_USER_MAX = 239
FLAG_EXCLUSIVE = 2
FLAG_RESULT = 4
FLAG_QUEUE = 8
MASK = 0xFFFFFFFF
DEFAULT_DEADLINE_PASSES = 1_000_000

OP_INVALID = 0
OP_SET, OP_MOV, OP_ADD, OP_ADDI, OP_SUB = 1, 2, 3, 4, 5
OP_MUL, OP_AND, OP_SHR, OP_MOD, OP_RAND = 6, 7, 8, 9, 10
OP_JMP, OP_JLT, OP_JEQ, OP_JNZ, OP_HALT = 16, 17, 18, 19, 20
OP_SUBMIT, OP_ENQUEUE, OP_WAIT, OP_RESULT, OP_EVENT, OP_TIME = 32, 33, 40, 41, 48, 49

EVENT_TIMEOUT = 1
EVENT_STREAM, EVENT_STREAM_END, EVENT_STREAM_ERROR = 4, 5, 6

(
    ERROR_HEADER,
    ERROR_OPCODE,
    ERROR_JUMP,
    ERROR_REGISTER,
    ERROR_TEMPLATE,
    ERROR_STEPS,
    ERROR_MOD_ZERO,
    ERROR_DEADLINE,
    ERROR_NEVER_FITS,
    ERROR_CANCELLED,
    ERROR_GROUP,
) = range(1, 12)

# opcode -> (name, register fields in use, kind of immediate: None, "immediate", "jump",
# "template")
OPS = {
    OP_SET: ("SET", "a", "immediate"),
    OP_MOV: ("MOV", "ab", None),
    OP_ADD: ("ADD", "abc", None),
    OP_ADDI: ("ADDI", "ab", "immediate"),
    OP_SUB: ("SUB", "abc", None),
    OP_MUL: ("MUL", "abc", None),
    OP_AND: ("AND", "abc", None),
    OP_SHR: ("SHR", "ab", "immediate"),
    OP_MOD: ("MOD", "abc", None),
    OP_RAND: ("RAND", "a", None),
    OP_JMP: ("JMP", "", "jump"),
    OP_JLT: ("JLT", "ab", "jump"),
    OP_JEQ: ("JEQ", "ab", "jump"),
    OP_JNZ: ("JNZ", "a", "jump"),
    OP_HALT: ("HALT", "a", None),
    OP_SUBMIT: ("SUBMIT", "", "template"),
    OP_ENQUEUE: ("ENQUEUE", "a", "template"),
    OP_WAIT: ("WAIT", "", "immediate"),
    OP_RESULT: ("RESULT", "a", None),
    OP_EVENT: ("EVENT", "a", None),
    OP_TIME: ("TIME", "a", None),
}

COMMAND_WORDS = 16
# word index of a0 to a9 inside a command
ARG_WORD = (3, 5, 6, 7, 9, 10, 11, 13, 14, 15)
HEADER = struct.Struct("<8I32x")
HEADER_FIELDS = (
    "magic",
    "version",
    "op_count",
    "command_count",
    "max_steps",
    "deadline_microseconds",
    "req",
    "seed",
)
UOP = struct.Struct("<4B3I")
TAIL = struct.Struct("<4B4B3B5B")

StreamError = namedtuple("StreamError", "reason pc")
Uop = namedtuple("Uop", "op a b c immediate imm2", defaults=(0, 0, 0, 0, 0))
Event = namedtuple("Event", "kind req auxiliary pass_index")
SubmittedCommand = namedtuple(
    "SubmittedCommand",
    "sequence slot handler first count flags args pass_index pc queued",
)
SimResult = namedtuple("SimResult", "commands events status passes error halt stopped")


class StreamFormatError(ValueError):
    """An upload that `check` rejects; carries the reason and program counter."""

    def __init__(self, error):
        super().__init__(f"bad stream: reason {error.reason} at pc {error.pc}")
        self.error = error


@dataclass(frozen=True)
class Register:
    """A register of a stream."""

    index: int

    def __post_init__(self):
        if not 0 <= self.index < REGISTERS:
            raise ValueError(f"register {self.index} out of range")


@dataclass(eq=False)
class Label:
    """A position in the program, bound now or later with `Stream.bind`."""

    position: int | None = None


@dataclass(frozen=True)
class Template:
    """A command template: its 16 words and the register patch list."""

    rec: tuple
    arg: tuple = (0, 0, 0, 0)
    register: tuple = (0, 0, 0, 0)
    first_register: int = TEMPLATE_NONE
    count_register: int = TEMPLATE_NONE
    patch_count: int = 0

    @property
    def handler(self):
        return self.rec[1] & 0xFF

    @property
    def flags(self):
        return self.rec[1] >> 8 & 0xFF


def _u32(value, what):
    if not -(1 << 31) <= value < 1 << 32:
        raise ValueError(f"{what} {value} does not fit 32 bits")
    return value & MASK


def _register(value, what):
    if not isinstance(value, Register):
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

    def __init__(self, deadline_microseconds=0, max_steps=1 << 20, seed=1, req=1):
        self.deadline_microseconds = deadline_microseconds
        self.max_steps = max_steps
        self.seed = seed
        self.req = req
        self._check_fields()
        self._body = []
        self._templates = []
        self._consts = {}
        self._next_register = 0

    def _check_fields(self):
        for name, low in (
            ("deadline_microseconds", 0),
            ("max_steps", 1),
            ("seed", 1),
            ("req", 0),
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or not low <= value <= MASK:
                raise ValueError(f"{name} must be an integer from {low} to {MASK}")

    def registers(self, n):
        """Allocate n registers (0 to 14; register 15 holds the seed)."""
        if n < 0 or self._next_register + n > SEED_REGISTER:
            free = SEED_REGISTER - self._next_register
            raise ValueError(
                f"out of registers: {n} requested, {free} free of {SEED_REGISTER}"
                f" (register {SEED_REGISTER} is the seed)"
            )
        out = [Register(self._next_register + i) for i in range(n)]
        self._next_register += n
        return out

    def const(self, value):
        """A register set once in the prologue; equal values share a register."""
        value = _u32(value, "constant")
        if value not in self._consts:
            self._consts[value] = self.registers(1)[0]
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

    def _add(self, op, a=0, b=0, c=0, immediate=0):
        self._body.append(Uop(op, a, b, c, immediate, 0))

    def set(self, a, value):
        self._add(OP_SET, _register(a, "a"), immediate=_u32(value, "value"))

    def mov(self, a, b):
        self._add(OP_MOV, _register(a, "a"), _register(b, "b"))

    def _three(self, op, a, b, c):
        self._add(op, _register(a, "a"), _register(b, "b"), _register(c, "c"))

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
        self._add(
            OP_ADDI,
            _register(a, "a"),
            _register(b, "b"),
            immediate=_u32(value, "value"),
        )

    def shr(self, a, b, amount):
        self._add(
            OP_SHR,
            _register(a, "a"),
            _register(b, "b"),
            immediate=_u32(amount, "amount"),
        )

    def rand(self, a):
        self._add(OP_RAND, _register(a, "a"))

    def jmp(self, target):
        self._add(OP_JMP, immediate=self._target(target))

    def jlt(self, a, b, target):
        self._add(
            OP_JLT, _register(a, "a"), _register(b, "b"), immediate=self._target(target)
        )

    def jeq(self, a, b, target):
        self._add(
            OP_JEQ, _register(a, "a"), _register(b, "b"), immediate=self._target(target)
        )

    def jnz(self, a, target):
        self._add(OP_JNZ, _register(a, "a"), immediate=self._target(target))

    def halt(self, a):
        self._add(OP_HALT, _register(a, "a"))

    def wait(self, pending):
        if not 0 <= pending <= MASK:
            raise ValueError(f"pending {pending} must be from 0 to {MASK}")
        self._add(OP_WAIT, immediate=pending)

    def result(self, a):
        self._add(OP_RESULT, _register(a, "a"))

    def event(self, a):
        self._add(OP_EVENT, _register(a, "a"))

    def time(self, a):
        self._add(OP_TIME, _register(a, "a"))

    @staticmethod
    def _target(target):
        if not isinstance(target, Label):
            raise TypeError(f"a jump needs a label, got {target!r}")
        return target

    def _template(self, handler, group, args, exclusive, result):
        if not 1 <= handler <= HANDLER_USER_MAX:
            raise ValueError(f"handler {handler} is not a program handler (1 to 239)")
        if len(args) > ARGS:
            raise ValueError(f"{len(args)} arguments, a command has {ARGS}")
        words = [0] * COMMAND_WORDS
        words[1] = (
            handler
            | (FLAG_EXCLUSIVE if exclusive else 0) << 8
            | (FLAG_RESULT if result else 0) << 8
        )
        patch = []
        for i, value in enumerate(args):
            if isinstance(value, Register):
                patch.append((i, value.index))
            else:
                words[ARG_WORD[i]] = _u32(value, "argument")
        if len(patch) > PATCHES:
            raise ValueError(f"{len(patch)} register arguments, at most {PATCHES}")
        first_register = count_register = TEMPLATE_NONE
        first, cnt = group
        if isinstance(first, Register):
            first_register = first.index
        else:
            words[2] |= _group_field(first, "first")
        if isinstance(cnt, Register):
            count_register = cnt.index
        else:
            words[2] |= _group_field(cnt, "count") << 16
        arg = [p[0] for p in patch] + [0] * (PATCHES - len(patch))
        register = [p[1] for p in patch] + [0] * (PATCHES - len(patch))
        self._templates.append(
            Template(
                tuple(words),
                tuple(arg),
                tuple(register),
                first_register,
                count_register,
                len(patch),
            )
        )
        return len(self._templates) - 1

    def submit(self, handler, group, args=(), exclusive=False, result=False):
        """SUBMIT: a command on `group` = (first, count), each an int or a register."""
        index = self._template(handler, group, list(args), exclusive, result)
        self._add(OP_SUBMIT, immediate=index)

    def enqueue(self, handler, count, args=(), exclusive=False, result=False):
        """ENQUEUE: a job of `count` cores (int or register); first is chosen."""
        register = count if isinstance(count, Register) else self.const(count)
        index = self._template(handler, (0, 0), list(args), exclusive, result)
        self._add(OP_ENQUEUE, _register(register, "count"), immediate=index)

    def uops(self):
        """The micro-ops as encoded: constant prologue, body, labels resolved."""
        prologue = [Uop(OP_SET, r.index, 0, 0, v, 0) for v, r in self._consts.items()]
        out = list(prologue)
        for op in self._body:
            if isinstance(op.immediate, Label):
                if op.immediate.position is None:
                    raise ValueError("a label was never bound")
                if op.immediate.position >= len(self._body):
                    raise ValueError("a label is bound past the last micro-op")
                op = op._replace(immediate=op.immediate.position + len(prologue))
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
        head += [self.deadline_microseconds, self.req, self.seed]
        data = HEADER.pack(*head)
        data += b"".join(
            UOP.pack(o.op, o.a, o.b, o.c, o.immediate, o.imm2, 0) for o in ops
        )
        data += bytes(_lines(len(ops) * UOP_BYTES))
        for t in self._templates:
            data += struct.pack("<16I", *t.rec)
            data += TAIL.pack(
                *t.arg,
                *t.register,
                t.first_register,
                t.count_register,
                t.patch_count,
                *[0] * 5,
            )
        return data + bytes(_lines(len(self._templates) * TEMPLATE_BYTES))

    @classmethod
    def decode(cls, data):
        """The stream an upload describes; StreamFormatError if `check` rejects it."""
        error = check(data, len(data))
        if error:
            raise StreamFormatError(error)
        h = _header(data)
        s = cls(h["deadline_microseconds"], h["max_steps"], h["seed"], h["req"])
        s._body, s._templates = _parse(data, h)
        return s

    def simulate(self, **options):
        """`simulate` on this stream's upload."""
        return simulate(self.encode(), **options)


def _group_field(value, what):
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"group {what} {value} does not fit 16 bits")
    return value


def _header(data):
    w = HEADER.unpack_from(data)
    return dict(zip(HEADER_FIELDS, w))


def _size(h):
    ops = h["op_count"] * UOP_BYTES
    template = h["command_count"] * TEMPLATE_BYTES
    return HEADER_BYTES + ops + _lines(ops) + template + _lines(template)


def _parse(data, h):
    ops = []
    for i in range(h["op_count"]):
        op, a, b, c, immediate, imm2, _ = UOP.unpack_from(
            data, HEADER_BYTES + i * UOP_BYTES
        )
        ops.append(Uop(op, a, b, c, immediate, imm2))
    base = HEADER_BYTES + h["op_count"] * UOP_BYTES
    base += _lines(h["op_count"] * UOP_BYTES)
    templates = []
    for i in range(h["command_count"]):
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
        return StreamError(ERROR_HEADER, 0)
    h = _header(data)
    if h["magic"] != MAGIC or h["version"] != VERSION or h["seed"] == 0:
        return StreamError(ERROR_HEADER, 0)
    if not 1 <= h["op_count"] <= MAX_OPS or h["command_count"] > MAX_OPS:
        return StreamError(ERROR_HEADER, 0)
    if h["max_steps"] == 0 or len(data) != _size(h):
        return StreamError(ERROR_HEADER, 0)
    ops, templates = _parse(data, h)
    first_use = {}
    for pc, o in enumerate(ops):
        if o.op not in OPS:
            return StreamError(ERROR_OPCODE, pc)
        _, fields, kind = OPS[o.op]
        if any(getattr(o, f) >= REGISTERS for f in fields):
            return StreamError(ERROR_REGISTER, pc)
        if kind == "jump" and o.immediate >= h["op_count"]:
            return StreamError(ERROR_JUMP, pc)
        if kind == "template":
            if o.immediate >= h["command_count"]:
                return StreamError(ERROR_TEMPLATE, pc)
            first_use.setdefault(o.immediate, pc)
    for index, t in enumerate(templates):
        if _bad_template(t):
            return StreamError(ERROR_TEMPLATE, first_use.get(index, 0))
    return None


def check_run(data, offset, region_size, active=(), active_reqs=()):
    """What the kernel answers a `KM_HANDLER_RUN` for `data` at `offset`: (error, req).

    `error` is None to start the stream, else reason 1 for a bad range or a reused `req`
    or what `check` finds; `req` is the header's, or 0 if it is outside the region.
    """
    length = len(data)
    head_inside = offset <= region_size - HEADER_BYTES
    req = 0
    if head_inside and length >= 28:
        req = _header(data.ljust(HEADER_BYTES, b"\0"))["req"]
    rejected = StreamError(ERROR_HEADER, 0)
    if offset % LINE or offset > region_size or length > region_size - offset:
        return rejected, req
    if any(offset < o + n and o < offset + length for o, n in active):
        return rejected, req
    if req in active_reqs:
        return rejected, req
    return check(data, region_size), req


def _bad_template(t):
    if not 1 <= t.handler <= HANDLER_USER_MAX or t.flags & ~(
        FLAG_EXCLUSIVE | FLAG_RESULT
    ):
        return True
    if t.patch_count > PATCHES:
        return True
    if any(
        t.arg[i] >= ARGS or t.register[i] >= REGISTERS for i in range(t.patch_count)
    ):
        return True
    return any(
        r != TEMPLATE_NONE and r >= REGISTERS
        for r in (t.first_register, t.count_register)
    )


def group_outside(first, count, core_count):
    """True if cores [first, first + count) are not in the grid; forms no sum."""
    return count == 0 or first > core_count or count > core_count - first


def alloc(busy, count, core_count):
    """The lowest `first` whose cores [first, first + count) overlap no (first, count)
    interval in `busy`, or None if there is no room now."""
    for first in range(core_count - count + 1):
        if all(first + count <= f or f + c <= first for f, c in busy):
            return first
    return None


@dataclass
class _Busy:
    command: SubmittedCommand
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
    core_count=119,
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
        events = [Event(EVENT_STREAM_ERROR, req, bad.reason | bad.pc << 8, 0)]
        return SimResult([], events, "error", 1, bad, None, False)
    h = _header(data)
    ops, templates = _parse(data, h)
    if deadline_passes is None and h["deadline_microseconds"] == 0:
        deadline_passes = default_deadline_passes
    sim = _Sim(ops, templates, h, core_count, delay, result, deadline_passes)
    sim.ticks = ticks_per_pass
    for p in range(max_passes):
        sim.run_pass(p, cancel_pass == p)
        if sim.error is not None or sim.halt is not None:
            if sim.error is not None and sim.error.reason != ERROR_DEADLINE:
                sim.drain(p + 1, max_passes)
            status = "error" if sim.error else "ended"
            return SimResult(
                sim.commands,
                sim.events,
                status,
                sim.events[-1].pass_index + 1,
                sim.error,
                sim.halt,
                sim.stopped,
            )
    return SimResult(sim.commands, sim.events, "stuck", max_passes, None, None, False)


class _Sim:
    """The state of one stream, the completion table and the queue."""

    def __init__(self, ops, templates, h, core_count, delay, result, deadline_passes):
        self.ops, self.templates, self.h = ops, templates, h
        self.core_count, self.delay, self.result_function = core_count, delay, result
        self.deadline_passes = deadline_passes
        self.registers = [0] * REGISTERS
        self.registers[SEED_REGISTER] = h["seed"]
        self.table = [None] * RING_SLOTS
        self.queue, self.commands, self.events = [], [], []
        self.pc = self.steps = self.sequence = self.pending = self.last_result = 0
        self.error = self.halt = None
        self.ticks = 1000
        self.now = 0
        self.stopped = False

    def run_pass(self, p, cancel):
        """One pass. Submitters are the streams in activation order, then the queue; in
        pass p the turns start at submitter p % (streams + 1) and go on cyclically."""
        self.now = p
        self.sweep()
        self.check_deadlines()
        if cancel and self.error is None:
            self.fail(ERROR_CANCELLED, self.pc)
        turns = [self.step_stream, self.step_queue]
        start = p % len(turns)
        for turn in turns[start:] + turns[:start]:
            if self.error is None:
                turn()

    def drain(self, start, limit):
        """After a stream error its commands still run; a late one stops the grid."""
        for p in range(start, limit):
            self.now = p
            self.sweep()
            self.check_deadlines()
            if self.stopped or not any(self.table):
                return

    def post(self, kind, auxiliary):
        self.events.append(Event(kind, self.h["req"], auxiliary, self.now))

    def fail(self, reason, pc):
        self.error = StreamError(reason, pc)
        self.post(EVENT_STREAM_ERROR, reason | pc << 8)
        self.queue.clear()
        return "error"

    def sweep(self):
        """Free the slots whose acknowledgement is due, in slot order."""
        for slot, busy in enumerate(self.table):
            if busy and busy.done_pass is not None and busy.done_pass <= self.now:
                if busy.command.flags & FLAG_RESULT:
                    self.last_result = busy.result
                self.table[slot] = None
                self.pending -= 1

    def check_deadlines(self):
        """The lowest-slot late command stops the grid: it ends the stream, or if the
        stream already ended, posts a timeout event for that slot."""
        for slot, busy in enumerate(self.table):
            if (
                busy
                and busy.deadline_pass is not None
                and self.now >= busy.deadline_pass
            ):
                self.stopped = True
                if self.error is None:
                    self.fail(ERROR_DEADLINE, busy.command.pc)
                else:
                    self.post(EVENT_TIMEOUT, slot)
                return

    def busy_groups(self):
        return [(b.command.first, b.command.count) for b in self.table if b]

    def free_slot(self):
        return next((i for i, b in enumerate(self.table) if b is None), None)

    def put(self, slot, words, first, count, pc, queued):
        self.sequence += 1
        args = tuple(words[i] for i in ARG_WORD)
        flags = words[1] >> 8 & 0xFF & ~FLAG_QUEUE
        handler = words[1] & 0xFF
        rec = SubmittedCommand(
            self.sequence,
            slot,
            handler,
            first,
            count,
            flags,
            args,
            self.now,
            pc,
            queued,
        )
        self.commands.append(rec)
        delay = self.delay(rec) if callable(self.delay) else self.delay
        done = None if delay is None else self.now + 1 + delay
        value = self.result_function(rec) & MASK if self.result_function else 0
        limit = (
            None if self.deadline_passes is None else self.now + self.deadline_passes
        )
        self.table[slot] = _Busy(rec, done, value, limit)
        if not queued:
            self.pending += 1

    def step_stream(self):
        registers = self.registers
        for _ in range(STEP_OPS):
            if self.pc >= len(self.ops):
                return self.fail(ERROR_JUMP, self.pc)
            if self.steps >= self.h["max_steps"]:
                return self.fail(ERROR_STEPS, self.pc)
            o = self.ops[self.pc]
            nxt = self.pc + 1
            if o.op in (OP_SUBMIT, OP_ENQUEUE, OP_WAIT, OP_HALT):
                state = self.work(o)
                if state != "ok":
                    return state
            elif o.op == OP_SET:
                registers[o.a] = o.immediate
            elif o.op == OP_MOV:
                registers[o.a] = registers[o.b]
            elif o.op == OP_ADD:
                registers[o.a] = registers[o.b] + registers[o.c] & MASK
            elif o.op == OP_ADDI:
                registers[o.a] = registers[o.b] + o.immediate & MASK
            elif o.op == OP_SUB:
                registers[o.a] = registers[o.b] - registers[o.c] & MASK
            elif o.op == OP_MUL:
                registers[o.a] = registers[o.b] * registers[o.c] & MASK
            elif o.op == OP_AND:
                registers[o.a] = registers[o.b] & registers[o.c]
            elif o.op == OP_SHR:
                registers[o.a] = registers[o.b] >> (o.immediate & 31)
            elif o.op == OP_MOD:
                if registers[o.c] == 0:
                    return self.fail(ERROR_MOD_ZERO, self.pc)
                registers[o.a] = registers[o.b] % registers[o.c]
            elif o.op == OP_RAND:
                registers[SEED_REGISTER] = xorshift32(registers[SEED_REGISTER])
                registers[o.a] = registers[SEED_REGISTER]
            elif o.op == OP_JMP:
                nxt = o.immediate
            elif o.op == OP_JLT:
                nxt = o.immediate if registers[o.a] < registers[o.b] else nxt
            elif o.op == OP_JEQ:
                nxt = o.immediate if registers[o.a] == registers[o.b] else nxt
            elif o.op == OP_JNZ:
                nxt = o.immediate if registers[o.a] != 0 else nxt
            elif o.op == OP_RESULT:
                registers[o.a] = self.last_result
            elif o.op == OP_EVENT:
                self.post(EVENT_STREAM, registers[o.a] & 0xFFFFFF)
            elif o.op == OP_TIME:
                registers[o.a] = self.now * self.ticks & MASK
            self.steps += 1
            self.pc = nxt
        return "ok"

    def work(self, o):
        """SUBMIT, ENQUEUE, WAIT and HALT: "ok" (done), "stall", "error" or "end"."""
        if o.op == OP_WAIT:
            return "stall" if self.pending > o.immediate else "ok"
        if o.op == OP_HALT:
            if self.pending:
                return "stall"
            self.halt = self.registers[o.a]
            self.post(EVENT_STREAM_END, self.halt & 0xFFFFFF)
            return "end"
        return self.do_submit(o) if o.op == OP_SUBMIT else self.do_enqueue(o)

    def do_submit(self, o):
        t, registers = self.templates[o.immediate], self.registers
        first = (
            t.rec[2] & 0xFFFF
            if t.first_register == TEMPLATE_NONE
            else registers[t.first_register]
        )
        count = (
            t.rec[2] >> 16
            if t.count_register == TEMPLATE_NONE
            else registers[t.count_register]
        )
        if group_outside(first, count, self.core_count):
            return self.fail(ERROR_GROUP, self.pc)
        slot = self.free_slot()
        if slot is None:
            return "stall"
        if t.flags & FLAG_EXCLUSIVE and (
            self.queue  # a job queued before it is not placed yet
            or any(first < f + c and f < first + count for f, c in self.busy_groups())
        ):
            return "stall"
        self.put(slot, self.patched(t), first, count, self.pc, False)
        return "ok"

    def do_enqueue(self, o):
        count = self.registers[o.a]
        if count == 0 or count > self.core_count:
            return self.fail(ERROR_NEVER_FITS, self.pc)
        if len(self.queue) >= QUEUE_JOBS:
            return "stall"
        job = _Job(self.patched(self.templates[o.immediate]), count, self.pc)
        self.queue.append(job)
        self.pending += 1
        return "ok"

    def patched(self, t):
        words = list(t.rec)
        for i in range(t.patch_count):
            words[ARG_WORD[t.arg[i]]] = self.registers[t.register[i]]
        return words

    def step_queue(self):
        """Submit what fits among the first 16 jobs; 1,024 jobs may overtake one."""
        skipped = []
        for job in list(self.queue[:QUEUE_LOOK]):
            if any(j.passed >= QUEUE_PASSES for j in skipped):
                break
            slot = self.free_slot()
            first = None
            if slot is not None:
                first = alloc(self.busy_groups(), job.count, self.core_count)
            if first is None:
                skipped.append(job)
                continue
            for j in skipped:
                j.passed += 1
            self.queue.remove(job)
            self.put(slot, job.words, first, job.count, job.pc, True)
