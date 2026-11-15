"""The chain compiler: the vector-unit functions (Chains) of a traced expression, each
over at most four register slots. Moved from Kohaku-TT's `dsl/tile/plan.py` (same owner,
Apache 2.0) with its names changed; Kohaku-TT's forms, plans and lowerings stay there.
"""

import os
from dataclasses import dataclass, field

from komorebi.dsl.tile import sfpu as f
from komorebi.dsl.tile.ir import BF16, Int, Node, PlanError, fail

# Destination register slots of one half with fp32 registers.
SLOTS = 4
# Row stores as one function over 2 tiles with broadcasts read in place (a knob).
BANDS = os.environ.get("KMR_TILE_BANDS", "1") == "1"
# Vector-unit instructions per element group (estimates) of each operation.
COST = {
    "add": 1, "sub": 1, "mul": 1, "neg": 1, "abs": 1, "max": 2, "min": 2,
    "relu": 2, "square": 1, "exp": 14, "exp2": 5, "sigmoid": 16, "silu": 17, "tanh": 16,
    "sqrt": 12, "rsqrt": 12, "recip": 8, "div": 9, "fill": 1, "reduce": 4,
}  # fmt: skip


@dataclass(eq=False)
class Leaf:
    """A register operand: tile `index` (a callable of the chunk context) of the
    operand `source` unpacked as `kind` ("copy", "bcast_row", "bcast_col")."""

    source: object
    kind: str
    offset: Int | None = None


@dataclass(eq=False)
class Chain:
    """One vector-unit function: its register operands (Leaves) in slots 0 .., its
    runtime scalars (`values`, Scalars passed as fp32 bits), the operations it runs."""

    operands: list
    fn: object
    out: int = 0
    pairs: bool = False
    ops: dict = field(default_factory=dict)
    values: list = field(default_factory=list)
    # The stage run before this one (its result is operand 0), or None.
    prev: object = None


@dataclass(eq=False)
class Case:
    """The chains of one store under the cat branch `bound` (None: unconditional):
    tile j takes this case when j < bound and no earlier case took it."""

    bound: Int | None
    chain: Chain
    # {G: Chain}: the stages for G tiles per register scope (operands per stage at
    # most 5 - G: the G results, the shared and the per-tile operands fit 4 slots).
    groups: dict = field(default_factory=dict)
    # One function over 2 tiles with the broadcasts read in place, or None.
    bands: object = None


@dataclass(eq=False)
class Operand:
    """A tensor view the kernel reads: per unit (`resident` False) or once per core;
    `row`: a tile row (else one tile); `held`: the whole row is kept in L1."""

    node: Node
    row: bool
    resident: bool
    held: bool = False
    random: bool = False
    passes: set = field(default_factory=set)

    @property
    def view(self):
        return self.node.attrs["view"]

    @property
    def tensor(self):
        return self.view.tensor


@dataclass(eq=False)
class Scalar:
    """A runtime scalar operand: a float value, or a size as a float, maybe its
    reciprocal; the launch gives its fp32 bits."""

    value: object
    recip: bool = False

    def key(self):
        v = self.value
        text = v.name if hasattr(v, "name") else v.text()
        return ("1/" if self.recip else "") + text


@dataclass(eq=False)
class Design:
    """A kernel class's plan before sizes: `form` "stream" or "row", the operands,
    the stores' cases (pass 2), the reduction (pass 1) and the statistic chain."""

    ir: object
    form: str
    operands: list
    stores: list
    reduce: object = None
    stat: Chain | None = None
    scalars: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)
    stat_root: Node | None = None
    dest: str = "fp32"


def _is_bf16_exact(node):
    """True when the value is a bf16 load (rounding it to bf16 changes nothing)."""
    if node.kind in ("load", "resident"):
        return node.attrs["view"].tensor.dtype == BF16
    if node.kind == "cast":
        return node.attrs["dtype"] == BF16 or _is_bf16_exact(node.args[0])
    return False


class _Shapes:
    """The form of every node: "tile", "row", "col" (one value per row), "scalar"."""

    def __init__(self, row_form):
        self.row_form, self.of = row_form, {}

    def __call__(self, node):
        known = self.of.get(id(node))
        if known is not None:
            return known
        shape = self._shape(node)
        self.of[id(node)] = shape
        return shape

    def _shape(self, node):
        k = node.kind
        if k in ("load", "resident"):
            return "row" if node.attrs["view"].kind == "row" else "tile"
        if k in ("const", "param", "size"):
            return "scalar"
        args = [self(a) for a in node.args]
        if k == "reduce":
            if node.attrs.get("dim", -1) == -2:
                fail("a sum over rows (dim=-2) reduces a column run", node.line)
            if args[0] != "row":
                fail("tile.reduce_sum reduces a row: load x.tiles[r, :]", node.line)
            return "col"
        if k == "col":
            if args[0] not in ("col", "tile"):
                fail(".col() broadcasts a reduced value or a tile", node.line)
            return "bcol" if args[0] == "col" else "tile"
        if k in ("slice", "cat"):
            if any(a != "row" for a in args):
                fail("slices and tile.cat take row blocks", node.line)
            return "row"
        if "row" in args or "bcol" in args:
            if "col" in args or "tile" in args:
                fail(
                    "a row block meets a column block: broadcast it with .col()",
                    node.line,
                )
            return "row"
        if "tile" in args:
            if "col" in args:
                fail(
                    "a tile meets a reduced value: broadcast it with .col()", node.line
                )
            return "tile"
        if "col" in args:
            return "col"
        return "scalar"


class _Compiler:
    """Builds the vector-unit functions of a kernel class."""

    def __init__(self, name, shapes, scalars):
        self.name, self.shapes, self.scalars = name, shapes, scalars
        self.count = 0

    def scalar(self, node, recip=False):
        """The Scalar of runtime scalar node `node`, registered once."""
        value = node.attrs["value"]
        s = Scalar(value, recip)
        return self.scalars.setdefault(s.key(), s)

    def chain(self, root, leaf_of, cut=None, stops=(), limit=SLOTS):
        """The Chain computing `root`; `leaf_of(node, offset)` gives the operand of a
        load (or None); `cut` maps a cat node to the piece index taken; nodes whose
        ids are in `stops` are operands too. A chain reading more blocks than `limit`
        register slots runs as stages (`Chain.prev`) in one register scope."""
        try:
            return self._single(root, leaf_of, cut, stops, limit=limit)
        except PlanError as e:
            if "register slots" not in str(e):
                raise
            top = root.args[0] if root.kind == "cast" else root
            if len(top.args) != 2 or top.kind not in (
                "add",
                "sub",
                "mul",
                "max",
                "min",
            ):
                raise
            for first in top.args:
                try:
                    prev = self.chain(first, leaf_of, cut, stops, limit)
                    out = self._single(
                        root, leaf_of, cut, stops, prev=(first, prev), limit=limit
                    )
                except PlanError as inner:
                    if "register slots" not in str(inner):
                        raise
                    continue
                return out
            raise

    def grouped(self, root, leaf_of, cut=None, stops=(), limit=SLOTS):
        """The stages computing `root` with at most `limit` operands each, in as few
        stages as found: one, two (every inner node tried as the first stage's
        result), else `chain`'s staging. None when no staging fits."""
        try:
            return self._single(root, leaf_of, cut, stops, limit=limit)
        except PlanError as e:
            if "register slots" not in str(e):
                raise
        top = root.args[0] if root.kind == "cast" else root
        for node in _inner(top, root):
            try:
                first = self._single(node, leaf_of, cut, stops, limit=limit)
                return self._single(
                    root, leaf_of, cut, stops, prev=(node, first), limit=limit
                )
            except PlanError as e:
                if "register slots" not in str(e) and "cannot be read" not in str(e):
                    raise
        try:
            return self.chain(root, leaf_of, cut, stops, limit)
        except PlanError as e:
            if "register slots" not in str(e):
                raise
            return None

    def banded(self, root, leaf_of):
        """One function computing `root` over 2 output tiles per call with broadcast
        operands read in place (sfpu along=): one tile operand, at most 2 column
        operands (one slot, two halves) and 2 row operands (one slot, two bands per
        tile); None when the chain does not have that shape."""
        try:
            chain = self._single(root, leaf_of, limit=6, bands=True)
        except PlanError:
            return None
        if sum(x.kind not in ("bcast_col", "bcast_row") for x in chain.operands) != 1:
            return None
        return chain

    def _single(
        self, root, leaf_of, cut=None, stops=(), prev=None, limit=SLOTS, bands=False
    ):
        """One generated function computing `root`; `prev` (node, Chain): that node
        is the previous stage's result, operand 0. `bands`: broadcast operands as
        `along` operands (see `banded`)."""
        operands, keys, ops, values, places = [], {}, {}, [], {}
        if prev is not None:
            keys[("prev",)] = f.operand(0)
            operands.append(Leaf("var", "prev"))

        def value(s):
            if s.key() not in places:
                if len(values) == 4:
                    fail("the chain needs more than 4 runtime scalars", root.line)
                places[s.key()] = f.scalar(len(values))
                values.append(s)
            return places[s.key()]

        def along(item):
            kinds = [x.kind for x in operands]
            if item.kind == "bcast_col" and kinds.count("bcast_col") < 2:
                return {"along": "col", "half": kinds.count("bcast_col")}
            if item.kind == "bcast_row" and kinds.count("bcast_row") < 2:
                return {"along": "row", "band": kinds.count("bcast_row"), "step": 2}
            if item.kind in ("bcast_col", "bcast_row"):
                fail("more than 2 broadcasts of one kind in a register slot pair")
            return {}

        def operand(item, key):
            if key not in keys:
                if len(operands) == limit:
                    fail(f"the chain needs more than {limit} register slots", root.line)
                keys[key] = f.operand(len(operands), **(along(item) if bands else {}))
                operands.append(item)
            return keys[key]

        def leaf(node, offset, kind):
            got = leaf_of(node, offset, kind)
            if got is None:
                fail(f"{node.kind} cannot be read here", node.line)
            key = (id(got.source), kind, None if offset is None else offset.text())
            return operand(got, key)

        built = {}

        def build(node, offset=None, kind="copy"):
            key = (id(node), None if offset is None else offset.text(), kind)
            if key not in built:
                built[key] = make(node, offset, kind)
            return built[key]

        def make(node, offset, kind):
            if prev is not None and node is prev[0] and offset is None:
                return keys[("prev",)]
            k = node.kind
            ops[k] = ops.get(k, 0) + 1
            if k in ("load", "resident", "reduce", "acc") or id(node) in stops:
                return leaf(node, offset, kind)
            if k == "const":
                return node.attrs["value"]
            if k in ("param", "size"):
                return value(self.scalar(node))
            if k == "cast":
                inner = node.args[0]
                rounds = node.attrs["dtype"] == BF16 and not _is_bf16_exact(inner)
                if rounds and node is not root:
                    # Rounded to nearest even here; at the root, the pack rounds.
                    return f.bf16(build(inner, offset, kind))
                return build(inner, offset, kind)
            if k == "row":
                return build(node.args[0], offset, "bcast_row")
            if k == "all":
                return leaf(node.args[0], None, "bcast_col")
            if k == "col":
                if self.shapes(node.args[0]) == "col":
                    return leaf(node.args[0], None, "bcast_col")
                return build(node.args[0], None, "bcast_col")
            if k == "slice":
                lo = node.attrs["lo"]
                shift = _tiles(lo, node, self.checks)
                total = shift if offset is None else offset + shift
                return build(node.args[0], total, kind)
            if k == "cat":
                if cut is None or id(node) not in cut:
                    fail("one tile.cat per stored value", node.line)
                piece, start = cut[id(node)]
                zero = Int("const", value=0)
                shifted = (
                    None
                    if start is None
                    else ((zero - start) if offset is None else offset - start)
                )
                return build(node.args[piece], shifted, kind)
            if k in ("even", "odd"):
                inner = build(node.args[0], offset, kind)
                return inner.even if k == "even" else inner.odd
            if k == "div" and self.shapes(node.args[1]) == "scalar":
                a, b = build(node.args[0], offset, kind), node.args[1]
                if b.kind == "const":
                    return a * (1.0 / b.attrs["value"])
                return a * value(self.scalar(b, recip=True))
            args = [build(a, offset, kind) for a in node.args]
            if k == "div":
                return args[0] * f.recip(args[1])
            if k == "lagmax":
                m, r = args
                return f.select(r > m + node.attrs["lag"], r, m)
            return _VECTOR[k](*args)

        self.checks = []
        result = build(root)
        pairs = isinstance(result, tuple)
        if pairs:
            result = {"even": result[0], "odd": result[1]}
        self.count += 1
        try:
            fn = f.sfpu_fn(f"{self.name}_fn{self.count}", result)
        except ValueError as e:
            fail(str(e).replace("komorebi.dsl.tile: ", ""), root.line)
        chain = Chain(operands, fn, 0, pairs, ops, values)
        chain.checks = self.checks
        if prev is not None:
            chain.prev = prev[1]
            chain.checks = prev[1].checks + chain.checks
            for k, n in prev[1].ops.items():
                chain.ops[k] = chain.ops.get(k, 0) + n
        return chain


_VECTOR = {
    "neg": lambda a: -a,
    "abs": abs,
    "exp": f.exp,
    # Kohaku-TT's flash attention 2^(u - 639) with u = x + 639, exactly 1 at x = 0 (so
    # a rescale by exp2(m - m) is none).
    "exp2": lambda a: f.select(a == 0.0, 1.0, f.flash_exp2(a + 639.0)),
    "sigmoid": f.sigmoid,
    "silu": f.silu,
    "tanh": f.tanh,
    "sqrt": f.sqrt,
    "rsqrt": f.rsqrt,
    "recip": f.recip,
    "relu": f.relu,
    "square": f.square,
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "max": f.maximum,
    "min": f.minimum,
    "pairs": lambda a, b: (a, b),
}


_LEAVES = ("load", "resident", "reduce", "acc", "const", "param", "size", "col", "row")


def _inner(top, root):
    """The inner nodes under `top` that a first stage may compute, largest first."""
    sizes, order = {}, []

    def walk(node):
        if id(node) in sizes:
            return sizes[id(node)]
        leaf = node.kind in _LEAVES
        n = 1 + (0 if leaf else sum(walk(a) for a in node.args))
        sizes[id(node)] = n
        if node is not top and node is not root and node.kind not in _LEAVES:
            order.append(node)
        return n

    walk(top)
    return sorted(order, key=lambda node: -sizes[id(node)])


def _tiles(elements, node, checks):
    """`elements` (a column offset) in tiles; the launch checks it is a multiple of
    32."""
    if elements.kind == "const":
        if elements.value % 32:
            fail("a slice of a row starts at a multiple of 32 columns", node.line)
        return Int("const", value=elements.value // 32)
    checks.append(("multiple", elements, 32, node.line))
    return elements // 32


def _width(node, shapes):
    """The tile count of row block `node` as an Int."""
    k = node.kind
    if k in ("load", "resident"):
        name = node.attrs["view"].tensor.name
        return Int("size", value=("row_tiles", name, None))
    if k == "slice":
        lo, hi = node.attrs["lo"], node.attrs["hi"]
        whole = _width(node.args[0], shapes)
        top = whole if hi is None else hi // 32
        return top - lo // 32
    if k == "cat":
        out = _width(node.args[0], shapes)
        for a in node.args[1:]:
            out = out + _width(a, shapes)
        return out
    rows = [a for a in node.args if shapes(a) == "row"]
    return _width(rows[0], shapes)


def _cats(node, found):
    if node.kind == "cat" and node not in found:
        found.append(node)
    for a in node.args:
        _cats(a, found)
    return found
