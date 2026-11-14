"""Graphs: the builder, the upload encoding, the check and the simulator.

The constants repeat `km_graph.h`; a test compares them. The executor here follows
`km_graph_engine.c` step for step, so the two give the same commands and registers.
"""

import struct
from collections import namedtuple
from dataclasses import dataclass, field

from komorebi.engines import stream as stream_module
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
LINE = stream_module.LINE
MASK = stream_module.MASK
REGISTERS = stream_module.REGISTERS

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
TEMPLATE_BYTES = stream_module.TEMPLATE_BYTES

# CPU micro-ops: opcode -> (register fields in use, is a jump)
CPU_OPS = {
    stream_module.OP_SET: ("a", False),
    stream_module.OP_MOV: ("ab", False),
    stream_module.OP_ADD: ("abc", False),
    stream_module.OP_ADDI: ("ab", False),
    stream_module.OP_SUB: ("abc", False),
    stream_module.OP_MUL: ("abc", False),
    stream_module.OP_AND: ("abc", False),
    stream_module.OP_SHR: ("ab", False),
    stream_module.OP_MOD: ("abc", False),
    stream_module.OP_RAND: ("a", False),
    stream_module.OP_JMP: ("", True),
    stream_module.OP_JLT: ("ab", True),
    stream_module.OP_JEQ: ("ab", True),
    stream_module.OP_JNZ: ("a", True),
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
    "node run sequence slot handler first count flags args pass_index queued program",
    defaults=(0,),
)
GraphSimResult = namedtuple(
    "GraphSimResult", "commands events status error registers runs passes stopped kept"
)


class GraphFormatError(ValueError):
    """An upload that `check` rejects; carries the reason and node."""

    def __init__(self, error):
        super().__init__(f"bad graph: reason {error.reason} at node {error.node}")
        self.error = error


def _line_up(size):
    return size + (-size % LINE)


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
        self._add(
            stream_module.OP_SET,
            stream_module._register(a, "a"),
            immediate=stream_module._u32(value, "value"),
        )

    def mov(self, a, b):
        self._add(
            stream_module.OP_MOV,
            stream_module._register(a, "a"),
            stream_module._register(b, "b"),
        )

    def _three(self, op, a, b, c):
        self._add(
            op,
            stream_module._register(a, "a"),
            stream_module._register(b, "b"),
            stream_module._register(c, "c"),
        )

    def add(self, a, b, c):
        self._three(stream_module.OP_ADD, a, b, c)

    def sub(self, a, b, c):
        self._three(stream_module.OP_SUB, a, b, c)

    def mul(self, a, b, c):
        self._three(stream_module.OP_MUL, a, b, c)

    def and_(self, a, b, c):
        self._three(stream_module.OP_AND, a, b, c)

    def mod(self, a, b, c):
        self._three(stream_module.OP_MOD, a, b, c)

    def addi(self, a, b, value):
        immediate = stream_module._u32(value, "value")
        self._add(
            stream_module.OP_ADDI,
            stream_module._register(a, "a"),
            stream_module._register(b, "b"),
            immediate=immediate,
        )

    def shr(self, a, b, amount):
        immediate = stream_module._u32(amount, "amount")
        self._add(
            stream_module.OP_SHR,
            stream_module._register(a, "a"),
            stream_module._register(b, "b"),
            immediate=immediate,
        )

    def rand(self, a):
        self._add(stream_module.OP_RAND, stream_module._register(a, "a"))

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
        self._add(stream_module.OP_JMP, immediate=stream_module.Stream._target(target))

    def jlt(self, a, b, target):
        a, b = stream_module._register(a, "a"), stream_module._register(b, "b")
        self._add(
            stream_module.OP_JLT, a, b, immediate=stream_module.Stream._target(target)
        )

    def jeq(self, a, b, target):
        a, b = stream_module._register(a, "a"), stream_module._register(b, "b")
        self._add(
            stream_module.OP_JEQ, a, b, immediate=stream_module.Stream._target(target)
        )

    def jnz(self, a, target):
        self._add(
            stream_module.OP_JNZ,
            stream_module._register(a, "a"),
            immediate=stream_module.Stream._target(target),
        )

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

    def registers(self, count):
        """Allocate `count` registers."""
        if count < 0 or self._next_register + count > REGISTERS:
            free = REGISTERS - self._next_register
            raise ValueError(
                f"out of registers: {count} requested, {free} free of {REGISTERS}"
            )
        out = [Register(self._next_register + i) for i in range(count)]
        self._next_register += count
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
        value = stream_module._u32(value, "constant")
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
        fixed = {
            register.index: stream_module._u32(values[name], name)
            for name, register in self._names.items()
        }
        fixed.update(
            {register.index: value for value, register in self._consts.items()}
        )
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
        for predecessor in after:
            if not isinstance(predecessor, Node) or predecessor.graph is not self:
                raise TypeError(
                    f"a predecessor must be a node of this graph, got {predecessor!r}"
                )
        if len({predecessor.index for predecessor in after}) != len(after):
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
        cluster=None,
    ):
        """A COMMAND node on `group` = (first, count), on `count` cores the allocator
        places, or to the X280 thread of work cluster `cluster` (1 to 3); `result` is
        a register for the record's result word."""
        if cluster is not None:
            if count is not None:
                raise ValueError("give a count or a cluster, not both")
            group = stream_module.cluster_group(cluster, group, exclusive)
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
        if not 1 <= handler <= stream_module.HANDLER_USER_MAX:
            raise ValueError(f"handler {handler} is not a program handler (1 to 239)")
        if len(args) > stream_module.ARGS:
            raise ValueError(
                f"{len(args)} arguments, a command has {stream_module.ARGS}"
            )
        flags = (stream_module.FLAG_EXCLUSIVE if exclusive else 0) | (
            0 if group else stream_module.FLAG_QUEUE
        )
        flags |= stream_module.FLAG_RESULT if result_register != NONE else 0
        words = [0] * stream_module.COMMAND_WORDS
        words[1] = (
            int(handler) | flags << 8 | stream_module.program_field(handler) << 24
        )
        patch = []
        for i, value in enumerate(args):
            if isinstance(value, Register):
                patch.append((i, value.index))
            else:
                words[stream_module.ARG_WORD[i]] = stream_module._u32(value, "argument")
        if len(patch) > stream_module.PATCHES:
            raise ValueError(
                f"{len(patch)} register arguments, at most {stream_module.PATCHES}"
            )
        first, group_size = group if group else (0, count)
        first_register = count_register = NONE
        if isinstance(first, Register):
            first_register = first.index
        else:
            words[2] |= stream_module._group_field(first, "first")
        if isinstance(group_size, Register):
            count_register = group_size.index
        else:
            words[2] |= stream_module._group_field(group_size, "count") << 16
        pad = [0] * (stream_module.PATCHES - len(patch))
        template = Template(
            tuple(words),
            tuple([entry[0] for entry in patch] + pad),
            tuple([entry[1] for entry in patch] + pad),
            first_register,
            count_register,
            len(patch),
        )
        if template not in self._template_index:
            if len(self._templates) >= MAX_TEMPLATES:
                raise ValueError(f"more than {MAX_TEMPLATES} distinct templates")
            self._template_index[template] = len(self._templates)
            self._templates.append(template)
        return self._template_index[template]

    # encoding

    def templates(self):
        return list(self._templates)

    def rows(self):
        """The node table and edge list as encoded, CPU micro-ops resolved."""
        successors = [[] for _ in self.nodes]
        for node in self.nodes:
            for predecessor in node.after:
                successors[predecessor.index].append(node.index)
        rows, edges, ops = [], [], []
        for node in self.nodes:
            arg, op_count = node.template if node.kind == NODE_COMMAND else 0, 0
            if node.kind == NODE_CPU:
                if not node.ops:
                    raise ValueError(f"CPU node {node.index} has no micro-ops")
                arg, op_count = len(ops), len(node.ops)
                ops += [_resolve(operation, arg, op_count) for operation in node.ops]
            row = NodeRow(
                node.kind, 0, len(node.after), node.skip, arg, op_count, node.result,
                len(edges), len(successors[node.index]),
            )  # fmt: skip
            rows.append(row)
            edges += sorted(successors[node.index])
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
            NODE.pack(*row[:7], 0, row.successor_offset, row.successor_count, 0)
            for row in rows
        )
        data += bytes(-len(rows) * NODE_BYTES % LINE)
        data += struct.pack(f"<{len(edges)}H", *edges)
        data += bytes(-len(edges) * 2 % LINE)
        data += b"".join(
            stream_module.UOP.pack(
                operation.op,
                operation.a,
                operation.b,
                operation.c,
                operation.immediate,
                operation.imm2,
                0,
            )
            for operation in ops
        )
        data += bytes(-len(ops) * stream_module.UOP_BYTES % LINE)
        for template in self._templates:
            data += struct.pack("<16I", *template.rec)
            data += stream_module.TAIL.pack(
                *template.arg,
                *template.register,
                template.first_register,
                template.count_register,
                template.patch_count,
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
        header, rows, edges, ops, templates = _parse(data)
        graph = cls(
            header["deadline_microseconds"],
            header["max_runs"],
            header["step_limit"],
            header["req"],
        )
        graph.keep = bool(header["flags"] & KEEP)
        graph._next_register = REGISTERS
        if header["again_register"] != NONE:
            graph._again = Register(header["again_register"])
        graph._templates = list(templates)
        graph._template_index = {template: i for i, template in enumerate(templates)}
        predecessors = [[] for _ in rows]
        for predecessor, row in enumerate(rows):
            for successor in edges[
                row.successor_offset : row.successor_offset + row.successor_count
            ]:
                predecessors[successor].append(predecessor)
        for i, row in enumerate(rows):
            node_class = CpuNode if row.kind == NODE_CPU else Node
            after = tuple(graph.nodes[predecessor] for predecessor in predecessors[i])
            node = node_class(graph, i, row.kind, after, row.skip_register)
            if row.kind == NODE_COMMAND:
                node.template, node.result = row.arg, row.result_register
            if row.kind == NODE_CPU:
                node.ops = [
                    _relative(operation, row.arg)
                    for operation in ops[row.arg : row.arg + row.op_count]
                ]
            graph.nodes.append(node)
        return graph

    def check(self):
        """`check` on this graph's upload."""
        return check(self.encode())

    def simulate(self, bindings=None, **options):
        """`simulate` on this graph's upload; `bindings` is a dict by name or a list."""
        if isinstance(bindings, dict) or bindings is None:
            bindings = self.binding_values(**(bindings or {}))
        return simulate(self.encode(), bindings=bindings, **options)


def _resolve(operation, start, op_count):
    if not isinstance(operation.immediate, Label):
        return operation
    if operation.immediate.position is None:
        raise ValueError("a label was never bound")
    return operation._replace(immediate=start + operation.immediate.position)


def _relative(operation, start):
    if CPU_OPS.get(operation.op, ("", False))[1]:
        return operation._replace(immediate=Label(operation.immediate - start))
    return operation


# Upload --------------------------------------------------------------------------


def _layout(header):
    """Byte offsets of the edges, micro-ops and templates, and the upload's size."""
    edges = HEADER.size + _line_up(header["node_count"] * NODE_BYTES)
    ops = edges + _line_up(header["edge_count"] * 2)
    template = ops + _line_up(header["op_count"] * stream_module.UOP_BYTES)
    return (
        edges,
        ops,
        template,
        template + _line_up(header["command_count"] * TEMPLATE_BYTES),
    )


def _parse(data):
    header = dict(zip(HEADER_FIELDS, HEADER.unpack_from(data)))
    edges_offset, ops_offset, templates_offset, _ = _layout(header)
    rows = []
    for i in range(header["node_count"]):
        words = NODE.unpack_from(data, HEADER.size + i * NODE_BYTES)
        rows.append(NodeRow(*words[:7], words[8], words[9]))
    edges = list(struct.unpack_from(f"<{header['edge_count']}H", data, edges_offset))
    ops = []
    for i in range(header["op_count"]):
        opcode, a, b, c, immediate, imm2, _ = stream_module.UOP.unpack_from(
            data, ops_offset + i * stream_module.UOP_BYTES
        )
        ops.append(Uop(opcode, a, b, c, immediate, imm2))
    templates = []
    for i in range(header["command_count"]):
        offset = templates_offset + i * TEMPLATE_BYTES
        record = struct.unpack_from("<16I", data, offset)
        tail = stream_module.TAIL.unpack_from(data, offset + 64)
        templates.append(Template(record, tail[0:4], tail[4:8], *tail[8:11]))
    return header, rows, edges, ops, templates


def _bad_register(register):
    return register != NONE and register >= REGISTERS


def _bad_template(template):
    if not 1 <= template.handler <= stream_module.HANDLER_USER_MAX:
        return True
    if (
        template.flags
        & ~(
            stream_module.FLAG_EXCLUSIVE
            | stream_module.FLAG_RESULT
            | stream_module.FLAG_QUEUE
        )
        or template.patch_count > stream_module.PATCHES
    ):
        return True
    if any(
        template.arg[slot] >= stream_module.ARGS or template.register[slot] >= REGISTERS
        for slot in range(template.patch_count)
    ):
        return True
    return _bad_register(template.first_register) or _bad_register(
        template.count_register
    )


def _check_slice(ops, first, end):
    for operation in ops[first:end]:
        if operation.op not in CPU_OPS:
            return stream_module.ERROR_OPCODE
        fields, jump = CPU_OPS[operation.op]
        if any(getattr(operation, operand) >= REGISTERS for operand in fields):
            return stream_module.ERROR_REGISTER
        if jump and not first <= operation.immediate <= end:
            return stream_module.ERROR_JUMP
    return 0


def _check_node(header, node, ops, templates, cpu_next):
    if not NODE_COMMAND <= node.kind <= NODE_JOIN or node.flags:
        return ERROR_NODE
    if node.predecessor_count > MAX_PREDECESSORS:
        return ERROR_EDGE
    if _bad_register(node.skip_register) or _bad_register(node.result_register):
        return stream_module.ERROR_REGISTER
    if node.kind == NODE_COMMAND:
        if node.arg >= header["command_count"] or _bad_template(templates[node.arg]):
            return stream_module.ERROR_TEMPLATE
        no_result = not templates[node.arg].flags & stream_module.FLAG_RESULT
        if node.op_count or (node.result_register != NONE and no_result):
            return ERROR_NODE
        return 0
    if node.result_register != NONE:
        return ERROR_NODE
    if node.kind == NODE_JOIN:
        return ERROR_NODE if node.arg or node.op_count else 0
    if (
        node.arg != cpu_next
        or node.op_count == 0
        or node.op_count > header["op_count"] - cpu_next
    ):
        return ERROR_NODE
    return _check_slice(ops, node.arg, node.arg + node.op_count)


def check(data):
    """None if the kernel accepts the upload, else `GraphError(reason, node)`.

    The order of the checks is that of `kmg_check`; the first fault wins.
    """
    bad_header = GraphError(stream_module.ERROR_HEADER, NO_NODE)
    if len(data) < HEADER.size or len(data) % LINE:
        return bad_header
    header = dict(zip(HEADER_FIELDS, HEADER.unpack_from(data)))
    if header["magic"] != MAGIC or header["version"] != VERSION:
        return bad_header
    if not 1 <= header["node_count"] <= MAX_NODES or header["edge_count"] > MAX_EDGES:
        return bad_header
    if header["op_count"] > MAX_OPS or header["command_count"] > MAX_TEMPLATES:
        return bad_header
    if _bad_register(header["again_register"]) or header["max_runs"] == 0:
        return bad_header
    if not 1 <= header["step_limit"] <= MAX_STEPS or header["flags"] & ~KEEP:
        return bad_header
    if len(data) != _layout(header)[3]:
        return bad_header
    header, rows, edges, ops, templates = _parse(data)
    seen = [0] * header["node_count"]
    cpu_next = edge_next = 0
    for i, node in enumerate(rows):
        why = _check_node(header, node, ops, templates, cpu_next)
        if why:
            return GraphError(why, i)
        if node.kind == NODE_CPU:
            cpu_next += node.op_count
        if (
            node.successor_offset != edge_next
            or node.successor_count > header["edge_count"] - edge_next
        ):
            return GraphError(ERROR_EDGE, i)
        previous = i
        for successor in edges[edge_next : edge_next + node.successor_count]:
            if (
                successor <= previous
                or successor >= header["node_count"]
                or seen[successor] >= MAX_PREDECESSORS
            ):
                return GraphError(ERROR_EDGE, i)
            seen[successor] += 1
            previous = successor
        edge_next += node.successor_count
    if edge_next != header["edge_count"]:
        return GraphError(ERROR_EDGE, NO_NODE)
    if cpu_next != header["op_count"]:
        return GraphError(ERROR_NODE, NO_NODE)
    for i, node in enumerate(rows):
        if seen[i] != node.predecessor_count:
            return GraphError(ERROR_EDGE, i)
    if any(_bad_template(template) for template in templates):
        return GraphError(stream_module.ERROR_TEMPLATE, NO_NODE)
    return None


# Executor ------------------------------------------------------------------------


class Executor:
    """The engine of `km_graph_engine.c` in Python: start, step, complete, stop.

    `submit(node, words)` gets the command's 16 words and returns SUBMITTED or BUSY.
    """

    def start(self, data, bindings=(), core_count=119):
        if len(bindings) > BINDINGS:
            return stream_module.ERROR_HEADER
        self.h, self.rows, self.edges, self.ops, self.template = _parse(data)
        self.registers = [binding & MASK for binding in bindings] + [0] * (
            REGISTERS - len(bindings)
        )
        self.runs, self.phase, self.error = 1, RUNNING, None
        self.in_flight, self.core_count = 0, core_count
        self._reset()
        return 0

    def _reset(self):
        node_count = self.h["node_count"]
        self.left = [row.predecessor_count for row in self.rows]
        self.status = [WAITING] * node_count
        self.ready, self.done = [], 0
        for i, row in enumerate(self.rows):
            if row.predecessor_count == 0:
                self._push(i)

    def _push(self, node):
        self.ready.append(node)
        self.status[node] = READY

    def _fail(self, reason, node):
        self.phase = ERROR
        self.error = GraphError(reason, node)

    def _finish(self, node):
        row = self.rows[node]
        self.status[node] = DONE
        self.done += 1
        for successor in self.edges[
            row.successor_offset : row.successor_offset + row.successor_count
        ]:
            self.left[successor] -= 1
            if self.left[successor] == 0:
                self._push(successor)

    def step(self, submit):
        """One step; returns RUNNING, ROUND, END or ERROR."""
        if self.phase != RUNNING:
            return self.phase
        taken = busy = ops = 0
        node_count = self.h["node_count"]
        while (
            self.done < node_count
            and taken < STEP
            and self.ready
            and busy < len(self.ready)
        ):
            node = self.ready.pop(0)
            row = self.rows[node]
            taken += 1
            if row.skip_register != NONE and self.registers[row.skip_register] == 0:
                self._finish(node)
                busy = 0
                continue
            if row.kind == NODE_COMMAND:
                got = self._submit(node, submit)
                if got == ERROR:
                    return ERROR
                busy = busy + 1 if got == BUSY else 0
                continue
            busy = 0
            if row.kind == NODE_CPU:
                ops += self._run_cpu(node)
                if self.phase != RUNNING:
                    return ERROR
            self._finish(node)
            if ops >= STEP_OPS:
                break
        return self._round_end() if self.done == node_count else RUNNING

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
        row, registers = self.rows[node], self.registers
        pc, end, steps = row.arg, row.arg + row.op_count, 0
        while pc != end:
            if steps >= self.h["step_limit"]:
                self._fail(stream_module.ERROR_STEPS, node)
                return steps
            operation = self.ops[pc]
            next_pc = pc + 1
            if operation.op == stream_module.OP_SET:
                registers[operation.a] = operation.immediate
            elif operation.op == stream_module.OP_MOV:
                registers[operation.a] = registers[operation.b]
            elif operation.op == stream_module.OP_ADD:
                registers[operation.a] = (
                    registers[operation.b] + registers[operation.c] & MASK
                )
            elif operation.op == stream_module.OP_ADDI:
                registers[operation.a] = (
                    registers[operation.b] + operation.immediate & MASK
                )
            elif operation.op == stream_module.OP_SUB:
                registers[operation.a] = (
                    registers[operation.b] - registers[operation.c] & MASK
                )
            elif operation.op == stream_module.OP_MUL:
                registers[operation.a] = (
                    registers[operation.b] * registers[operation.c] & MASK
                )
            elif operation.op == stream_module.OP_AND:
                registers[operation.a] = registers[operation.b] & registers[operation.c]
            elif operation.op == stream_module.OP_SHR:
                registers[operation.a] = registers[operation.b] >> (
                    operation.immediate & 31
                )
            elif operation.op == stream_module.OP_MOD:
                if registers[operation.c] == 0:
                    self._fail(stream_module.ERROR_MOD_ZERO, node)
                    return steps
                registers[operation.a] = registers[operation.b] % registers[operation.c]
            elif operation.op == stream_module.OP_RAND:
                registers[stream_module.SEED_REGISTER] = stream_module.xorshift32(
                    registers[stream_module.SEED_REGISTER]
                )
                registers[operation.a] = registers[stream_module.SEED_REGISTER]
            elif operation.op == stream_module.OP_JMP:
                next_pc = operation.immediate
            elif operation.op == stream_module.OP_JLT:
                next_pc = (
                    operation.immediate
                    if registers[operation.a] < registers[operation.b]
                    else next_pc
                )
            elif operation.op == stream_module.OP_JEQ:
                next_pc = (
                    operation.immediate
                    if registers[operation.a] == registers[operation.b]
                    else next_pc
                )
            elif operation.op == stream_module.OP_JNZ:
                next_pc = (
                    operation.immediate if registers[operation.a] != 0 else next_pc
                )
            steps += 1
            pc = next_pc
        return steps

    def command_words(self, node):
        """The command of a COMMAND node with the registers as they are: (words, first,
        count) before the group check."""
        template = self.template[self.rows[node].arg]
        words = list(template.rec)
        words[0] = words[4] = words[8] = words[12] = 0
        words[1] &= 0xFFFF
        for slot in range(template.patch_count):
            words[stream_module.ARG_WORD[template.arg[slot]]] = self.registers[
                template.register[slot]
            ]
        first, count = words[2] & 0xFFFF, words[2] >> 16
        if template.first_register != NONE:
            first = self.registers[template.first_register]
        if template.count_register != NONE:
            count = self.registers[template.count_register]
        return words, first, count

    def _submit(self, node, submit):
        words, first, count = self.command_words(node)
        if words[1] >> 8 & stream_module.FLAG_QUEUE:
            first = 0
            if count == 0 or count > self.core_count:
                self._fail(stream_module.ERROR_NEVER_FITS, node)
                return ERROR
        elif stream_module.group_outside(first, count, self.core_count, words[1] >> 8):
            self._fail(stream_module.ERROR_GROUP, node)
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
    executor = engine if engine is not None else Executor()
    if not bad and executor.start(data, list(bindings), core_count):
        bad = GraphError(stream_module.ERROR_HEADER, NO_NODE)
    if bad:
        events = [Event(EVENT_GRAPH_ERROR, req, bad.reason | bad.node << 8, 0, 0)]
        return GraphSimResult(
            [], events, "error", bad, [0] * REGISTERS, 0, 1, False, False
        )
    simulation = _Sim(executor, core_count, delay, result, deadline_passes, req)
    for pass_index in range(max_passes):
        status = simulation.run_pass(pass_index, cancel_pass == pass_index)
        if status:
            graph_error = GraphError(*executor.error) if status == "error" else None
            return GraphSimResult(
                simulation.commands, simulation.events, status, graph_error,
                list(executor.registers), executor.runs, pass_index + 1,
                simulation.stopped, keep,
            )  # fmt: skip
    registers = list(executor.registers)
    return GraphSimResult(
        simulation.commands,
        simulation.events,
        "stuck",
        None,
        registers,
        executor.runs,
        max_passes,
        False,
        keep,
    )


class _Sim:
    """The completion table around one graph: ring slots, groups, a queue model."""

    def __init__(self, executor, core_count, delay, result, deadline_passes, req):
        self.executor, self.core_count, self.delay = executor, core_count, delay
        self.result_function, self.deadline_passes, self.req = (
            result,
            deadline_passes,
            req,
        )
        self.table = [None] * stream_module.ring_slots()
        self.commands, self.events = [], []
        self.sequence = self.now = 0
        self.stopped = False

    def post(self, kind, auxiliary, extra=0):
        self.events.append(Event(kind, self.req, auxiliary, extra, self.now))

    def run_pass(self, pass_index, cancel):
        """Sweep, deadlines, cancel, one engine step; "ended", "error" or None."""
        self.now = pass_index
        executor = self.executor
        for slot, busy in enumerate(self.table):
            if busy and busy.done_pass is not None and busy.done_pass <= pass_index:
                self.table[slot] = None
                executor.complete(busy.command.node, busy.result)
        for busy in self.table:
            if (
                busy
                and busy.deadline_pass is not None
                and pass_index >= busy.deadline_pass
            ):
                self.stopped = True
                executor.stop(stream_module.ERROR_DEADLINE, busy.command.node)
                break
        if cancel:
            executor.stop(stream_module.ERROR_CANCELLED, NO_NODE)
        code = (
            executor.phase if executor.phase != RUNNING else executor.step(self.submit)
        )
        if code == ROUND:
            self.post(EVENT_GRAPH, (executor.runs - 1) & 0xFFFFFF)
        elif code == END:
            self.post(
                EVENT_GRAPH_END, executor.registers[0] & 0xFFFFFF, executor.registers[1]
            )
            return "ended"
        elif code == ERROR:
            reason, node = executor.error
            self.post(EVENT_GRAPH_ERROR, reason | node << 8)
            return "error"
        return None

    def submit(self, node, words):
        slot = next((i for i, entry in enumerate(self.table) if entry is None), None)
        if slot is None:
            return BUSY
        busy = [
            (entry.command.first, entry.command.count) for entry in self.table if entry
        ]
        first, count = words[2] & 0xFFFF, words[2] >> 16
        flags = words[1] >> 8 & 0xFF
        queued = bool(flags & stream_module.FLAG_QUEUE)
        if queued:
            first = alloc(busy, count, self.core_count)
            if first is None:
                return BUSY
        elif flags & stream_module.FLAG_EXCLUSIVE and any(
            first < busy_first + busy_count and busy_first < first + count
            for busy_first, busy_count in busy
        ):
            return BUSY
        self.sequence += 1
        record = GraphCommand(
            node, self.executor.runs, self.sequence, slot, words[1] & 0xFF, first,
            count, flags & ~stream_module.FLAG_QUEUE,
            tuple(words[i] for i in stream_module.ARG_WORD), self.now, queued,
            words[1] >> 24,
        )  # fmt: skip
        self.commands.append(record)
        delay = self.delay(record) if callable(self.delay) else self.delay
        done = None if delay is None else self.now + 1 + delay
        value = self.result_function(record) & MASK if self.result_function else 0
        limit = (
            None if self.deadline_passes is None else self.now + self.deadline_passes
        )
        self.table[slot] = _Busy(record, done, value, limit)
        return SUBMITTED
