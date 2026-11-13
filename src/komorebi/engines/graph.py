"""Graphs: the builder, the upload encoding, the check and the simulator.

The constants repeat `km_graph.h`; a test compares them. The executor here follows
`km_graph_engine.c` step for step, so the two give the same commands and registers.
"""

import struct
from collections import namedtuple
from dataclasses import dataclass, field

from komorebi.engines import stream as st
from komorebi.engines.stream import Label, Register, Template, Uop, alloc

MAGIC = 0x4B4D4752
VERSION = 1
MAX_NODES = 256
MAX_EDGES = 1024
MAX_PREDECESSORS = 4
MAX_TEMPLATES = 64
MAX_OPS = 4096
MAX_STEPS = 4096
BINDINGS = 16
STEP = 16
STEP_OPS = 64
MAX_ACTIVE = 2
MAX_KEPT = 8
KEEP = 1
NONE = 0xFF
NO_NODE = 0xFFFF
LINE = st.LINE
MASK = st.MASK
REGISTERS = st.REGISTERS

NODE_COMMAND, NODE_CPU, NODE_JOIN = 1, 2, 3
GRAPH_RUN_UPLOAD, GRAPH_RUN_KEPT = 1, 2
EVENT_GRAPH, EVENT_GRAPH_END, EVENT_GRAPH_ERROR = 7, 8, 9
ERROR_NODE, ERROR_EDGE, ERROR_RUNS, ERROR_KEPT_FULL = 12, 13, 14, 15

# kmg_step's answers (RUNNING, END and ERROR are also the phase) and the node states
RUNNING, ROUND, END, ERROR = 0, 1, 2, 3
WAITING, READY, FLIGHT, DONE = 0, 1, 2, 3
SUBMITTED, BUSY = 0, 1

HEADER = struct.Struct("<12I16x")
HEADER_FIELDS = (
    "magic",
    "version",
    "node_count",
    "edge_count",
    "op_count",
    "command_count",
    "req",
    "again_register",
    "max_runs",
    "deadline_microseconds",
    "step_limit",
    "flags",
)
NODE = struct.Struct("<4B2H2B3H")
NODE_BYTES = 16
TEMPLATE_BYTES = st.TEMPLATE_BYTES

# CPU micro-ops: opcode -> (register fields in use, is a jump)
CPU_OPS = {
    st.OP_SET: ("a", False),
    st.OP_MOV: ("ab", False),
    st.OP_ADD: ("abc", False),
    st.OP_ADDI: ("ab", False),
    st.OP_SUB: ("abc", False),
    st.OP_MUL: ("abc", False),
    st.OP_AND: ("abc", False),
    st.OP_SHR: ("ab", False),
    st.OP_MOD: ("abc", False),
    st.OP_RAND: ("a", False),
    st.OP_JMP: ("", True),
    st.OP_JLT: ("ab", True),
    st.OP_JEQ: ("ab", True),
    st.OP_JNZ: ("a", True),
}

GraphError = namedtuple("GraphError", "reason node")
NodeRow = namedtuple(
    "NodeRow",
    "kind flags predecessor_count skip_register arg op_count result_register"
    " successor_offset successor_count",
)
Event = namedtuple("Event", "kind req auxiliary extra pass_index")
GraphCommand = namedtuple(
    "GraphCommand",
    "node run sequence slot handler first count flags args pass_index queued",
)
GraphSimResult = namedtuple(
    "GraphSimResult", "commands events status error registers runs passes stopped kept"
)


class GraphFormatError(ValueError):
    """An upload that `check` rejects; carries the reason and node."""

    def __init__(self, error):
        super().__init__(f"bad graph: reason {error.reason} at node {error.node}")
        self.error = error


def _line_up(n):
    return n + (-n % LINE)


def _register_or_none(value, what):
    if value is None:
        return NONE
    if not isinstance(value, Register):
        raise TypeError(f"{what} must be a register, got {value!r}")
    return value.index


# Builder -----------------------------------------------------------------------


@dataclass(eq=False)
class Node:
    """A node of a graph under construction; `index` is its number."""

    graph: "Graph"
    index: int
    kind: int
    after: tuple
    skip: int = NONE
    template: int = 0
    result: int = NONE
    ops: list = field(default_factory=list)


class CpuNode(Node):
    """A CPU node: its micro-ops are added with the methods below; labels are local."""

    def _add(self, op, a=0, b=0, c=0, immediate=0):
        self.ops.append(Uop(op, a, b, c, immediate, 0))

    def set(self, a, value):
        self._add(st.OP_SET, st._register(a, "a"), immediate=st._u32(value, "value"))

    def mov(self, a, b):
        self._add(st.OP_MOV, st._register(a, "a"), st._register(b, "b"))

    def _three(self, op, a, b, c):
        self._add(op, st._register(a, "a"), st._register(b, "b"), st._register(c, "c"))

    def add(self, a, b, c):
        self._three(st.OP_ADD, a, b, c)

    def sub(self, a, b, c):
        self._three(st.OP_SUB, a, b, c)

    def mul(self, a, b, c):
        self._three(st.OP_MUL, a, b, c)

    def and_(self, a, b, c):
        self._three(st.OP_AND, a, b, c)

    def mod(self, a, b, c):
        self._three(st.OP_MOD, a, b, c)

    def addi(self, a, b, value):
        immediate = st._u32(value, "value")
        self._add(
            st.OP_ADDI, st._register(a, "a"), st._register(b, "b"), immediate=immediate
        )

    def shr(self, a, b, amount):
        immediate = st._u32(amount, "amount")
        self._add(
            st.OP_SHR, st._register(a, "a"), st._register(b, "b"), immediate=immediate
        )

    def rand(self, a):
        self._add(st.OP_RAND, st._register(a, "a"))

    def label(self):
        """A label at the next micro-op of this node."""
        return Label(len(self.ops))

    def forward(self):
        return Label()

    def bind(self, label):
        if label.position is not None:
            raise ValueError("label already bound")
        label.position = len(self.ops)

    def jmp(self, target):
        self._add(st.OP_JMP, immediate=st.Stream._target(target))

    def jlt(self, a, b, target):
        a, b = st._register(a, "a"), st._register(b, "b")
        self._add(st.OP_JLT, a, b, immediate=st.Stream._target(target))

    def jeq(self, a, b, target):
        a, b = st._register(a, "a"), st._register(b, "b")
        self._add(st.OP_JEQ, a, b, immediate=st.Stream._target(target))

    def jnz(self, a, target):
        self._add(st.OP_JNZ, st._register(a, "a"), immediate=st.Stream._target(target))

    def lt(self, d, a, b):
        """d = 1 if a < b (unsigned), else 0."""
        self._compare(d, a, b, 1, 0)

    def ge(self, d, a, b):
        """d = 1 if a >= b (unsigned), else 0."""
        self._compare(d, a, b, 0, 1)

    def _compare(self, d, a, b, if_less, otherwise):
        less, end = self.forward(), self.forward()
        self.jlt(a, b, less)
        self.set(d, otherwise)
        self.jmp(end)
        self.bind(less)
        self.set(d, if_less)
        self.bind(end)


class Graph:
    """A graph under construction, or one decoded from an upload."""

    def __init__(
        self,
        deadline_microseconds=0,
        max_runs=1,
        step_limit=MAX_STEPS,
        req=1,
        keep=False,
    ):
        self.deadline_microseconds, self.max_runs = deadline_microseconds, max_runs
        self.step_limit, self.req, self.keep = step_limit, req, bool(keep)
        self._check_fields()
        self.nodes = []
        self._templates = []
        self._template_index = {}
        self._names = {}
        self._consts = {}
        self._next_register = 0
        self._again = None

    def _check_fields(self):
        for name, low, high in (
            ("deadline_microseconds", 0, MASK),
            ("max_runs", 1, MASK),
            ("step_limit", 1, MAX_STEPS),
            ("req", 0, MASK),
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer from {low} to {high}")

    # registers

    def registers(self, n):
        """Allocate n registers."""
        if n < 0 or self._next_register + n > REGISTERS:
            free = REGISTERS - self._next_register
            raise ValueError(
                f"out of registers: {n} requested, {free} free of {REGISTERS}"
            )
        out = [Register(self._next_register + i) for i in range(n)]
        self._next_register += n
        return out

    def register(self):
        return self.registers(1)[0]

    def bindings(self, *names):
        """Registers set from the values given with each run, by name."""
        out = []
        for name in names:
            if name in self._names:
                raise ValueError(f"binding {name!r} declared twice")
            self._names[name] = self.register()
            out.append(self._names[name])
        return out[0] if len(out) == 1 else tuple(out)

    def const(self, value):
        """A register given `value` with the bindings at each run."""
        value = st._u32(value, "constant")
        if value not in self._consts:
            self._consts[value] = self.register()
        return self._consts[value]

    @property
    def again(self):
        """The register that, nonzero when a round ends, starts another round."""
        if self._again is None:
            self._again = self.register()
        return self._again

    def binding_values(self, **values):
        """The binding block of one run: registers 0 to n - 1, n as small as can be."""
        unknown = set(values) - set(self._names)
        missing = set(self._names) - set(values)
        if unknown or missing:
            raise ValueError(
                f"bindings: unknown {sorted(unknown)}, missing {sorted(missing)}"
            )
        fixed = {r.index: st._u32(values[n], n) for n, r in self._names.items()}
        fixed.update({r.index: v for v, r in self._consts.items()})
        block = [0] * (max(fixed) + 1 if fixed else 0)
        for index, value in fixed.items():
            block[index] = value
        return block

    # nodes

    def _node(self, cls, kind, after, skip_if_zero, **fields):
        if len(self.nodes) >= MAX_NODES:
            raise ValueError(f"more than {MAX_NODES} nodes")
        after = tuple(after)
        if len(after) > MAX_PREDECESSORS:
            raise ValueError(f"{len(after)} predecessors, at most {MAX_PREDECESSORS}")
        for p in after:
            if not isinstance(p, Node) or p.graph is not self:
                raise TypeError(
                    f"a predecessor must be a node of this graph, got {p!r}"
                )
        if len({p.index for p in after}) != len(after):
            raise ValueError("a predecessor is named twice")
        skip = _register_or_none(skip_if_zero, "skip_if_zero")
        node = cls(self, len(self.nodes), kind, after, skip, **fields)
        self.nodes.append(node)
        return node

    def command(
        self,
        handler,
        group=None,
        count=None,
        args=(),
        after=(),
        result=None,
        exclusive=False,
        skip_if_zero=None,
    ):
        """A COMMAND node on `group` = (first, count), or on `count` cores the allocator
        places; `result` is a register for the record's result word."""
        if (group is None) == (count is None):
            raise ValueError("give either group=(first, count) or count")
        result_register = _register_or_none(result, "result")
        index = self._template(
            handler, group, count, list(args), exclusive, result_register
        )
        return self._node(
            Node,
            NODE_COMMAND,
            after,
            skip_if_zero,
            template=index,
            result=result_register,
        )

    def cpu(self, after=(), skip_if_zero=None):
        """A CPU node; add its micro-ops through the node."""
        return self._node(CpuNode, NODE_CPU, after, skip_if_zero)

    def join(self, after=(), skip_if_zero=None):
        """A node with no work."""
        return self._node(Node, NODE_JOIN, after, skip_if_zero)

    def _template(self, handler, group, count, args, exclusive, result_register):
        if not 1 <= handler <= st.HANDLER_USER_MAX:
            raise ValueError(f"handler {handler} is not a program handler (1 to 239)")
        if len(args) > st.ARGS:
            raise ValueError(f"{len(args)} arguments, a command has {st.ARGS}")
        flags = (st.FLAG_EXCLUSIVE if exclusive else 0) | (
            0 if group else st.FLAG_QUEUE
        )
        flags |= st.FLAG_RESULT if result_register != NONE else 0
        words = [0] * st.COMMAND_WORDS
        words[1] = handler | flags << 8
        patch = []
        for i, value in enumerate(args):
            if isinstance(value, Register):
                patch.append((i, value.index))
            else:
                words[st.ARG_WORD[i]] = st._u32(value, "argument")
        if len(patch) > st.PATCHES:
            raise ValueError(f"{len(patch)} register arguments, at most {st.PATCHES}")
        first, cnt = group if group else (0, count)
        first_register = count_register = NONE
        if isinstance(first, Register):
            first_register = first.index
        else:
            words[2] |= st._group_field(first, "first")
        if isinstance(cnt, Register):
            count_register = cnt.index
        else:
            words[2] |= st._group_field(cnt, "count") << 16
        pad = [0] * (st.PATCHES - len(patch))
        t = Template(
            tuple(words),
            tuple([p[0] for p in patch] + pad),
            tuple([p[1] for p in patch] + pad),
            first_register,
            count_register,
            len(patch),
        )
        if t not in self._template_index:
            if len(self._templates) >= MAX_TEMPLATES:
                raise ValueError(f"more than {MAX_TEMPLATES} distinct templates")
            self._template_index[t] = len(self._templates)
            self._templates.append(t)
        return self._template_index[t]

    # encoding

    def templates(self):
        return list(self._templates)

    def rows(self):
        """The node table and edge list as encoded, CPU micro-ops resolved."""
        succ = [[] for _ in self.nodes]
        for n in self.nodes:
            for p in n.after:
                succ[p.index].append(n.index)
        rows, edges, ops = [], [], []
        for n in self.nodes:
            arg, op_count = n.template if n.kind == NODE_COMMAND else 0, 0
            if n.kind == NODE_CPU:
                if not n.ops:
                    raise ValueError(f"CPU node {n.index} has no micro-ops")
                arg, op_count = len(ops), len(n.ops)
                ops += [_resolve(o, arg, op_count) for o in n.ops]
            row = NodeRow(
                n.kind, 0, len(n.after), n.skip, arg, op_count, n.result, len(edges),
                len(succ[n.index]),
            )  # fmt: skip
            rows.append(row)
            edges += sorted(succ[n.index])
        if len(edges) > MAX_EDGES or len(ops) > MAX_OPS:
            raise ValueError(
                f"{len(edges)} edges and {len(ops)} micro-ops, at most {MAX_EDGES}"
                f" and {MAX_OPS}"
            )
        return rows, edges, ops

    def encode(self):
        """The upload: header, nodes, edges, micro-ops, templates, each line-padded."""
        self._check_fields()
        rows, edges, ops = self.rows()
        again = NONE if self._again is None else self._again.index
        head = [MAGIC, VERSION, len(rows), len(edges), len(ops), len(self._templates)]
        head += [
            self.req,
            again,
            self.max_runs,
            self.deadline_microseconds,
            self.step_limit,
        ]
        head += [KEEP if self.keep else 0]
        data = HEADER.pack(*head)
        data += b"".join(
            NODE.pack(*r[:7], 0, r.successor_offset, r.successor_count, 0) for r in rows
        )
        data += bytes(-len(rows) * NODE_BYTES % LINE)
        data += struct.pack(f"<{len(edges)}H", *edges)
        data += bytes(-len(edges) * 2 % LINE)
        data += b"".join(
            st.UOP.pack(o.op, o.a, o.b, o.c, o.immediate, o.imm2, 0) for o in ops
        )
        data += bytes(-len(ops) * st.UOP_BYTES % LINE)
        for t in self._templates:
            data += struct.pack("<16I", *t.rec)
            data += st.TAIL.pack(
                *t.arg,
                *t.register,
                t.first_register,
                t.count_register,
                t.patch_count,
                *[0] * 5,
            )
        return data + bytes(-len(self._templates) * TEMPLATE_BYTES % LINE)

    @classmethod
    def decode(cls, data):
        """The graph an upload describes (registers by number, no binding names);
        GraphFormatError if `check` rejects it."""
        error = check(data)
        if error:
            raise GraphFormatError(error)
        h, rows, edges, ops, templates = _parse(data)
        g = cls(h["deadline_microseconds"], h["max_runs"], h["step_limit"], h["req"])
        g.keep = bool(h["flags"] & KEEP)
        g._next_register = REGISTERS
        if h["again_register"] != NONE:
            g._again = Register(h["again_register"])
        g._templates = list(templates)
        g._template_index = {t: i for i, t in enumerate(templates)}
        preds = [[] for _ in rows]
        for p, r in enumerate(rows):
            for s in edges[r.successor_offset : r.successor_offset + r.successor_count]:
                preds[s].append(p)
        for i, r in enumerate(rows):
            kind_cls = CpuNode if r.kind == NODE_CPU else Node
            after = tuple(g.nodes[p] for p in preds[i])
            node = kind_cls(g, i, r.kind, after, r.skip_register)
            if r.kind == NODE_COMMAND:
                node.template, node.result = r.arg, r.result_register
            if r.kind == NODE_CPU:
                node.ops = [
                    _relative(o, r.arg) for o in ops[r.arg : r.arg + r.op_count]
                ]
            g.nodes.append(node)
        return g

    def check(self):
        """`check` on this graph's upload."""
        return check(self.encode())

    def simulate(self, bindings=None, **options):
        """`simulate` on this graph's upload; `bindings` is a dict by name or a list."""
        if isinstance(bindings, dict) or bindings is None:
            bindings = self.binding_values(**(bindings or {}))
        return simulate(self.encode(), bindings=bindings, **options)


def _resolve(o, start, op_count):
    if not isinstance(o.immediate, Label):
        return o
    if o.immediate.position is None:
        raise ValueError("a label was never bound")
    return o._replace(immediate=start + o.immediate.position)


def _relative(o, start):
    if CPU_OPS.get(o.op, ("", False))[1]:
        return o._replace(immediate=Label(o.immediate - start))
    return o


# Upload --------------------------------------------------------------------------


def _layout(h):
    """Byte offsets of the edges, micro-ops and templates, and the upload's size."""
    edges = HEADER.size + _line_up(h["node_count"] * NODE_BYTES)
    ops = edges + _line_up(h["edge_count"] * 2)
    template = ops + _line_up(h["op_count"] * st.UOP_BYTES)
    return (
        edges,
        ops,
        template,
        template + _line_up(h["command_count"] * TEMPLATE_BYTES),
    )


def _parse(data):
    h = dict(zip(HEADER_FIELDS, HEADER.unpack_from(data)))
    at_edges, at_ops, at_template, _ = _layout(h)
    rows = []
    for i in range(h["node_count"]):
        w = NODE.unpack_from(data, HEADER.size + i * NODE_BYTES)
        rows.append(NodeRow(*w[:7], w[8], w[9]))
    edges = list(struct.unpack_from(f"<{h['edge_count']}H", data, at_edges))
    ops = []
    for i in range(h["op_count"]):
        op, a, b, c, immediate, imm2, _ = st.UOP.unpack_from(
            data, at_ops + i * st.UOP_BYTES
        )
        ops.append(Uop(op, a, b, c, immediate, imm2))
    templates = []
    for i in range(h["command_count"]):
        at = at_template + i * TEMPLATE_BYTES
        rec = struct.unpack_from("<16I", data, at)
        tail = st.TAIL.unpack_from(data, at + 64)
        templates.append(Template(rec, tail[0:4], tail[4:8], *tail[8:11]))
    return h, rows, edges, ops, templates


def _bad_register(r):
    return r != NONE and r >= REGISTERS


def _bad_template(t):
    if not 1 <= t.handler <= st.HANDLER_USER_MAX:
        return True
    if (
        t.flags & ~(st.FLAG_EXCLUSIVE | st.FLAG_RESULT | st.FLAG_QUEUE)
        or t.patch_count > st.PATCHES
    ):
        return True
    if any(
        t.arg[k] >= st.ARGS or t.register[k] >= REGISTERS for k in range(t.patch_count)
    ):
        return True
    return _bad_register(t.first_register) or _bad_register(t.count_register)


def _check_slice(ops, first, end):
    for o in ops[first:end]:
        if o.op not in CPU_OPS:
            return st.ERROR_OPCODE
        fields, jump = CPU_OPS[o.op]
        if any(getattr(o, f) >= REGISTERS for f in fields):
            return st.ERROR_REGISTER
        if jump and not first <= o.immediate <= end:
            return st.ERROR_JUMP
    return 0


def _check_node(h, n, ops, templates, cpu_next):
    if not NODE_COMMAND <= n.kind <= NODE_JOIN or n.flags:
        return ERROR_NODE
    if n.predecessor_count > MAX_PREDECESSORS:
        return ERROR_EDGE
    if _bad_register(n.skip_register) or _bad_register(n.result_register):
        return st.ERROR_REGISTER
    if n.kind == NODE_COMMAND:
        if n.arg >= h["command_count"] or _bad_template(templates[n.arg]):
            return st.ERROR_TEMPLATE
        no_result = not templates[n.arg].flags & st.FLAG_RESULT
        if n.op_count or (n.result_register != NONE and no_result):
            return ERROR_NODE
        return 0
    if n.result_register != NONE:
        return ERROR_NODE
    if n.kind == NODE_JOIN:
        return ERROR_NODE if n.arg or n.op_count else 0
    if n.arg != cpu_next or n.op_count == 0 or n.op_count > h["op_count"] - cpu_next:
        return ERROR_NODE
    return _check_slice(ops, n.arg, n.arg + n.op_count)


def check(data):
    """None if the kernel accepts the upload, else `GraphError(reason, node)`.

    The order of the checks is that of `kmg_check`; the first fault wins.
    """
    header = GraphError(st.ERROR_HEADER, NO_NODE)
    if len(data) < HEADER.size or len(data) % LINE:
        return header
    h = dict(zip(HEADER_FIELDS, HEADER.unpack_from(data)))
    if h["magic"] != MAGIC or h["version"] != VERSION:
        return header
    if not 1 <= h["node_count"] <= MAX_NODES or h["edge_count"] > MAX_EDGES:
        return header
    if h["op_count"] > MAX_OPS or h["command_count"] > MAX_TEMPLATES:
        return header
    if _bad_register(h["again_register"]) or h["max_runs"] == 0:
        return header
    if not 1 <= h["step_limit"] <= MAX_STEPS or h["flags"] & ~KEEP:
        return header
    if len(data) != _layout(h)[3]:
        return header
    h, rows, edges, ops, templates = _parse(data)
    seen = [0] * h["node_count"]
    cpu_next = edge_next = 0
    for i, n in enumerate(rows):
        why = _check_node(h, n, ops, templates, cpu_next)
        if why:
            return GraphError(why, i)
        if n.kind == NODE_CPU:
            cpu_next += n.op_count
        if (
            n.successor_offset != edge_next
            or n.successor_count > h["edge_count"] - edge_next
        ):
            return GraphError(ERROR_EDGE, i)
        prev = i
        for s in edges[edge_next : edge_next + n.successor_count]:
            if s <= prev or s >= h["node_count"] or seen[s] >= MAX_PREDECESSORS:
                return GraphError(ERROR_EDGE, i)
            seen[s] += 1
            prev = s
        edge_next += n.successor_count
    if edge_next != h["edge_count"]:
        return GraphError(ERROR_EDGE, NO_NODE)
    if cpu_next != h["op_count"]:
        return GraphError(ERROR_NODE, NO_NODE)
    for i, n in enumerate(rows):
        if seen[i] != n.predecessor_count:
            return GraphError(ERROR_EDGE, i)
    if any(_bad_template(t) for t in templates):
        return GraphError(st.ERROR_TEMPLATE, NO_NODE)
    return None


# Executor ------------------------------------------------------------------------


class Executor:
    """The engine of `km_graph_engine.c` in Python: start, step, complete, stop.

    `submit(node, words)` gets the command's 16 words and returns SUBMITTED or BUSY.
    """

    def start(self, data, bindings=(), core_count=119):
        if len(bindings) > BINDINGS:
            return st.ERROR_HEADER
        self.h, self.rows, self.edges, self.ops, self.template = _parse(data)
        self.registers = [b & MASK for b in bindings] + [0] * (
            REGISTERS - len(bindings)
        )
        self.runs, self.phase, self.error = 1, RUNNING, None
        self.in_flight, self.core_count = 0, core_count
        self._reset()
        return 0

    def _reset(self):
        n = self.h["node_count"]
        self.left = [r.predecessor_count for r in self.rows]
        self.status = [WAITING] * n
        self.ready, self.done = [], 0
        for i, r in enumerate(self.rows):
            if r.predecessor_count == 0:
                self._push(i)

    def _push(self, node):
        self.ready.append(node)
        self.status[node] = READY

    def _fail(self, reason, node):
        self.phase = ERROR
        self.error = GraphError(reason, node)

    def _finish(self, node):
        r = self.rows[node]
        self.status[node] = DONE
        self.done += 1
        for s in self.edges[
            r.successor_offset : r.successor_offset + r.successor_count
        ]:
            self.left[s] -= 1
            if self.left[s] == 0:
                self._push(s)

    def step(self, submit):
        """One step; returns RUNNING, ROUND, END or ERROR."""
        if self.phase != RUNNING:
            return self.phase
        taken = busy = ops = 0
        n = self.h["node_count"]
        while self.done < n and taken < STEP and self.ready and busy < len(self.ready):
            node = self.ready.pop(0)
            r = self.rows[node]
            taken += 1
            if r.skip_register != NONE and self.registers[r.skip_register] == 0:
                self._finish(node)
                busy = 0
                continue
            if r.kind == NODE_COMMAND:
                got = self._submit(node, submit)
                if got == ERROR:
                    return ERROR
                busy = busy + 1 if got == BUSY else 0
                continue
            busy = 0
            if r.kind == NODE_CPU:
                ops += self._run_cpu(node)
                if self.phase != RUNNING:
                    return ERROR
            self._finish(node)
            if ops >= STEP_OPS:
                break
        return self._round_end() if self.done == n else RUNNING

    def _round_end(self):
        again = self.h["again_register"]
        if again == NONE or self.registers[again] == 0:
            self.phase = END
            return END
        if self.runs >= self.h["max_runs"]:
            self._fail(ERROR_RUNS, NO_NODE)
            return ERROR
        self.runs += 1
        self._reset()
        return ROUND

    def _run_cpu(self, node):
        r, registers = self.rows[node], self.registers
        pc, end, steps = r.arg, r.arg + r.op_count, 0
        while pc != end:
            if steps >= self.h["step_limit"]:
                self._fail(st.ERROR_STEPS, node)
                return steps
            o = self.ops[pc]
            nxt = pc + 1
            if o.op == st.OP_SET:
                registers[o.a] = o.immediate
            elif o.op == st.OP_MOV:
                registers[o.a] = registers[o.b]
            elif o.op == st.OP_ADD:
                registers[o.a] = registers[o.b] + registers[o.c] & MASK
            elif o.op == st.OP_ADDI:
                registers[o.a] = registers[o.b] + o.immediate & MASK
            elif o.op == st.OP_SUB:
                registers[o.a] = registers[o.b] - registers[o.c] & MASK
            elif o.op == st.OP_MUL:
                registers[o.a] = registers[o.b] * registers[o.c] & MASK
            elif o.op == st.OP_AND:
                registers[o.a] = registers[o.b] & registers[o.c]
            elif o.op == st.OP_SHR:
                registers[o.a] = registers[o.b] >> (o.immediate & 31)
            elif o.op == st.OP_MOD:
                if registers[o.c] == 0:
                    self._fail(st.ERROR_MOD_ZERO, node)
                    return steps
                registers[o.a] = registers[o.b] % registers[o.c]
            elif o.op == st.OP_RAND:
                registers[st.SEED_REGISTER] = st.xorshift32(registers[st.SEED_REGISTER])
                registers[o.a] = registers[st.SEED_REGISTER]
            elif o.op == st.OP_JMP:
                nxt = o.immediate
            elif o.op == st.OP_JLT:
                nxt = o.immediate if registers[o.a] < registers[o.b] else nxt
            elif o.op == st.OP_JEQ:
                nxt = o.immediate if registers[o.a] == registers[o.b] else nxt
            elif o.op == st.OP_JNZ:
                nxt = o.immediate if registers[o.a] != 0 else nxt
            steps += 1
            pc = nxt
        return steps

    def command_words(self, node):
        """The command of a COMMAND node with the registers as they are: (words, first,
        count) before the group check."""
        t = self.template[self.rows[node].arg]
        words = list(t.rec)
        words[0] = words[4] = words[8] = words[12] = 0
        words[1] &= 0xFFFF
        for k in range(t.patch_count):
            words[st.ARG_WORD[t.arg[k]]] = self.registers[t.register[k]]
        first, count = words[2] & 0xFFFF, words[2] >> 16
        if t.first_register != NONE:
            first = self.registers[t.first_register]
        if t.count_register != NONE:
            count = self.registers[t.count_register]
        return words, first, count

    def _submit(self, node, submit):
        words, first, count = self.command_words(node)
        if words[1] >> 8 & st.FLAG_QUEUE:
            first = 0
            if count == 0 or count > self.core_count:
                self._fail(st.ERROR_NEVER_FITS, node)
                return ERROR
        elif st.group_outside(first, count, self.core_count):
            self._fail(st.ERROR_GROUP, node)
            return ERROR
        words[2] = first | count << 16
        if submit(node, words) != SUBMITTED:
            self._push(node)
            return BUSY
        self.status[node] = FLIGHT
        self.in_flight += 1
        return SUBMITTED

    def complete(self, node, result):
        """The command of `node` completed; 1 if it was not in flight, else 0."""
        if self.phase != RUNNING or not 0 <= node < self.h["node_count"]:
            return 1
        if self.status[node] != FLIGHT:
            return 1
        register = self.rows[node].result_register
        if register != NONE:
            self.registers[register] = result & MASK
        self.in_flight -= 1
        self._finish(node)
        return 0

    def retry(self, node):
        """The command of `node` was rejected after its submit: back to the ring's
        tail."""
        if self.phase != RUNNING or not 0 <= node < self.h["node_count"]:
            return 1
        if self.status[node] != FLIGHT:
            return 1
        self.in_flight -= 1
        self._push(node)
        return 0

    def stop(self, reason, node):
        if self.phase == RUNNING:
            self._fail(reason, node)


# Simulator -----------------------------------------------------------------------


@dataclass
class _Busy:
    command: GraphCommand
    done_pass: int | None
    result: int
    deadline_pass: int | None


def simulate(
    data,
    *,
    bindings=(),
    core_count=119,
    delay=0,
    result=None,
    deadline_passes=None,
    cancel_pass=None,
    max_passes=100_000,
    kept=0,
    engine=None,
):
    """Run an upload as the kernel would and return a `GraphSimResult`. `kept`: graphs
    the kernel keeps already; `engine`: the executor (default `Executor()`). Pass
    order, completion model, parameters: docs/graphs.md, "The simulator".
    """
    bad = check(data)
    req = struct.unpack_from("<I", data, 24)[0] if len(data) >= 28 else 0
    keep = not bad and bool(HEADER.unpack_from(data)[11] & KEEP)
    if not bad and keep and kept >= MAX_KEPT:
        bad = GraphError(ERROR_KEPT_FULL, NO_NODE)
    eng = engine if engine is not None else Executor()
    if not bad and eng.start(data, list(bindings), core_count):
        bad = GraphError(st.ERROR_HEADER, NO_NODE)
    if bad:
        events = [Event(EVENT_GRAPH_ERROR, req, bad.reason | bad.node << 8, 0, 0)]
        return GraphSimResult(
            [], events, "error", bad, [0] * REGISTERS, 0, 1, False, False
        )
    sim = _Sim(eng, core_count, delay, result, deadline_passes, req)
    for p in range(max_passes):
        status = sim.run_pass(p, cancel_pass == p)
        if status:
            err = GraphError(*eng.error) if status == "error" else None
            return GraphSimResult(
                sim.commands, sim.events, status, err, list(eng.registers), eng.runs,
                p + 1, sim.stopped, keep,
            )  # fmt: skip
    registers = list(eng.registers)
    return GraphSimResult(
        sim.commands,
        sim.events,
        "stuck",
        None,
        registers,
        eng.runs,
        max_passes,
        False,
        keep,
    )


class _Sim:
    """The completion table around one graph: 16 slots, groups, a model of the queue."""

    def __init__(self, eng, core_count, delay, result, deadline_passes, req):
        self.eng, self.core_count, self.delay = eng, core_count, delay
        self.result_function, self.deadline_passes, self.req = (
            result,
            deadline_passes,
            req,
        )
        self.table = [None] * st.RING_SLOTS
        self.commands, self.events = [], []
        self.sequence = self.now = 0
        self.stopped = False

    def post(self, kind, auxiliary, extra=0):
        self.events.append(Event(kind, self.req, auxiliary, extra, self.now))

    def run_pass(self, p, cancel):
        """Sweep, deadlines, cancel, one engine step; "ended", "error" or None."""
        self.now = p
        eng = self.eng
        for slot, busy in enumerate(self.table):
            if busy and busy.done_pass is not None and busy.done_pass <= p:
                self.table[slot] = None
                eng.complete(busy.command.node, busy.result)
        for busy in self.table:
            if busy and busy.deadline_pass is not None and p >= busy.deadline_pass:
                self.stopped = True
                eng.stop(st.ERROR_DEADLINE, busy.command.node)
                break
        if cancel:
            eng.stop(st.ERROR_CANCELLED, NO_NODE)
        code = eng.phase if eng.phase != RUNNING else eng.step(self.submit)
        if code == ROUND:
            self.post(EVENT_GRAPH, (eng.runs - 1) & 0xFFFFFF)
        elif code == END:
            self.post(EVENT_GRAPH_END, eng.registers[0] & 0xFFFFFF, eng.registers[1])
            return "ended"
        elif code == ERROR:
            reason, node = eng.error
            self.post(EVENT_GRAPH_ERROR, reason | node << 8)
            return "error"
        return None

    def submit(self, node, words):
        slot = next((i for i, b in enumerate(self.table) if b is None), None)
        if slot is None:
            return BUSY
        busy = [(b.command.first, b.command.count) for b in self.table if b]
        first, count = words[2] & 0xFFFF, words[2] >> 16
        flags = words[1] >> 8 & 0xFF
        queued = bool(flags & st.FLAG_QUEUE)
        if queued:
            first = alloc(busy, count, self.core_count)
            if first is None:
                return BUSY
        elif flags & st.FLAG_EXCLUSIVE and any(
            first < f + c and f < first + count for f, c in busy
        ):
            return BUSY
        self.sequence += 1
        rec = GraphCommand(
            node, self.eng.runs, self.sequence, slot, words[1] & 0xFF, first, count,
            flags & ~st.FLAG_QUEUE, tuple(words[i] for i in st.ARG_WORD), self.now,
            queued,
        )  # fmt: skip
        self.commands.append(rec)
        delay = self.delay(rec) if callable(self.delay) else self.delay
        done = None if delay is None else self.now + 1 + delay
        value = self.result_function(rec) & MASK if self.result_function else 0
        limit = (
            None if self.deadline_passes is None else self.now + self.deadline_passes
        )
        self.table[slot] = _Busy(rec, done, value, limit)
        return SUBMITTED
