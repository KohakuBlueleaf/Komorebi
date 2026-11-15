"""torch on Komorebi-nn: an eager tensor whose ops with a descriptor are appended to the
device's open list (the rest run on the host, counted), and a torch.compile backend
("komorebi") from AOT autograd's inference graph onto the same table: fusion rules as
data, intermediates given DRAM tensors once, the chain, a closed list replayed."""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch._dynamo
from torch._functorch.aot_autograd import aot_function, aot_module_simplified
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.utils._pytree import tree_map

from komorebi import formats
from projects.decode import model as md
from projects.nn import table

aten = torch.ops.aten
# The backend's stages, an option each: fusion, DRAM tensors once, the chain, x and its
# squares as one tensor (joint), a call as one host submission (one; README).
STAGES = ("fuse", "once", "chain", "joint", "one")
DEFAULTS = {"fuse": True, "once": True, "chain": True, "joint": False, "one": False}


@torch.library.custom_op("komorebi::attend", mutates_args=())
def attend(
    qkv: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, position: int
) -> torch.Tensor:
    """Row 0's token (q, k, v side by side, q and k normed and rotated) over the cache's
    rows 0 .. position - 1 and its own k, v at `position`: softmax(q k^T / sqrt(d)) v a
    query head, its KV head's; the other rows 0. The card also writes k, v there."""
    kv_heads, _, width = keys.shape
    heads = qkv.shape[-1] // width - 2 * kv_heads
    x = qkv[0].float()
    q = x[: heads * width].reshape(heads, width)
    k = x[heads * width : (heads + kv_heads) * width].reshape(kv_heads, 1, width)
    v = x[(heads + kv_heads) * width :].reshape(kv_heads, 1, width)
    group = heads // kv_heads
    cached_k = torch.cat([keys[:, :position].float(), k], 1).repeat_interleave(group, 0)
    cached_v = torch.cat([values[:, :position].float(), v], 1).repeat_interleave(
        group, 0
    )
    scores = torch.einsum("hd,hnd->hn", q, cached_k) / math.sqrt(width)
    out = torch.zeros(qkv.shape[0], heads * width)
    out[0] = torch.einsum("hn,hnd->hd", scores.softmax(-1), cached_v).reshape(-1)
    return out.to(qkv.dtype)


@attend.register_fake
def _(qkv, keys, values, position):
    kv_heads, _, width = keys.shape
    return qkv.new_empty(qkv.shape[0], qkv.shape[-1] - 2 * kv_heads * width)


# The models' pieces, written once: the modules and the rules' patterns call them.


def rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def head_norm_rope(t, weight, cos, sin, width, eps):
    """Each head of t (width wide) normed, times `weight`, then rotated by the
    position's cos and sin (sin signed as rotate_half takes it, md.rotary)."""
    heads = t.reshape(t.shape[0], -1, width)
    normed = rms_norm(heads, weight, eps)
    turned = torch.cat([normed[..., width // 2 :], normed[..., : width // 2]], -1)
    return (normed * cos + turned * sin).reshape(t.shape)


def mlp(x, norm_weight, w_gate, w_up, eps):
    h = rms_norm(x, norm_weight, eps)
    return torch.nn.functional.silu(h @ w_gate) * (h @ w_up)


def qkv_attend(x, norm_weight, wq, wk, wv, q_norm, k_norm, cos, sin, keys, values, eps):
    """norm, q k v, the heads' norms and rotation, then komorebi::attend; the position
    is the cache's rows not yet written (a constant of the module)."""
    h = rms_norm(x, norm_weight, eps)
    width = keys.shape[-1]
    q = head_norm_rope(h @ wq, q_norm, cos, sin, width, eps)
    k = head_norm_rope(h @ wk, k_norm, cos, sin, width, eps)
    return torch.cat([q, k, h @ wv], -1)


# Eager


class CardTensor(torch.Tensor):
    """A torch tensor whose values are a Komorebi-nn tensor (`tensor`) of the device
    `nn`: ops go through __torch_dispatch__; numpy(), item() and printing read it."""

    @staticmethod
    def __new__(cls, tensor, device):
        out = torch.Tensor._make_wrapper_subclass(
            cls, tensor.shape, dtype=torch.bfloat16, device="cpu"
        )
        out.tensor, out.nn = tensor, device
        return out

    __torch_function__ = torch._C._disabled_torch_function_impl

    def numpy(self):
        return np.asarray(self.nn.read(self.tensor, "out_read"), np.float32)

    def __repr__(self):
        return f"CardTensor({self.numpy()!r})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        device = next(
            a.nn
            for a in torch.utils._pytree.tree_leaves((args, kwargs))
            if isinstance(a, CardTensor)
        )
        return eager_dispatch(device, func, args, kwargs or {})


def to_card(values, device):
    """A host tensor as a CardTensor (written to the card when it has a tile form)."""
    return CardTensor(
        device.upload(values.detach().float().numpy(), "in_write"), device
    )


def card_module(module, device):
    """The module's parameters and buffers replaced by CardTensors, in place."""
    for owner in module.modules():
        for store in (owner._parameters, owner._buffers):
            for name, value in store.items():
                if value is not None:
                    store[name] = CardTensor(
                        device.upload(value.detach().float().numpy(), "weights"),
                        device,
                    )
    return module


def unwrap_card(device, value):
    """A CardTensor argument of a card call as its Tensor, on the card."""
    if not isinstance(value, CardTensor):
        raise ValueError("an operand the host holds")
    return device.to_card(value.tensor)


def eager_matmul(device, args, kwargs):
    a, w = (unwrap_card(device, value) for value in args)
    if len(a.shape) != 2 or len(w.shape) != 2 or w.kn not in device.plans:
        raise ValueError(f"{a.shape} @ {w.shape}: no plan")
    out = device.empty((a.shape[0], w.shape[1]))
    device.append(table.TABLE["matmul"].plan(device, [a, w], out), [out])
    return out


def eager_attend(device, args, kwargs):
    qkv, keys, values = (unwrap_card(device, value) for value in args[:3])
    position = args[3]
    width = keys.shape[-1]
    out = device.empty((qkv.shape[0], qkv.shape[1] - 2 * keys.shape[0] * width))
    call = table.TABLE["attend"].plan(
        device, [qkv, keys, values], out, position=position
    )
    device.append(call, [out])
    return out


EAGER = {aten.mm.default: eager_matmul}  # the op table's eager entries, by aten op
CUSTOM = {"komorebi::attend": eager_attend}  # a custom op, by its name


def eager_dispatch(device, func, args, kwargs):
    started = time.perf_counter()
    try:
        planner = EAGER.get(func) or CUSTOM.get(func._schema.name)
        out = None
        if planner is not None:
            try:
                out = CardTensor(planner(device, args, kwargs), device)
            except ValueError as why:
                device.fallback(f"{func} ({why})")
        else:
            device.fallback(str(func))
        if out is None:
            out = host_op(device, func, args, kwargs)
        return out
    finally:
        device.clock["dispatch"] += time.perf_counter() - started


def host_op(device, func, args, kwargs):
    """An op with no descriptor (or none that fits): the open list synced, its card
    inputs read, torch on the CPU in float32, each output written back (with `mirror`
    the host keeps it until a call takes it)."""
    device.sync()

    def host(value):
        if isinstance(value, CardTensor):
            return torch.from_numpy(
                np.array(device.read(value.tensor, "fallback_read"), np.float32)
            )
        return value

    args, kwargs = tree_map(host, (args, kwargs))
    started = time.perf_counter()
    out = func(*args, **kwargs)
    device.clock["host"] += time.perf_counter() - started

    def back(value):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            return value
        values = md.bf16(value.float().numpy())
        if device.mirror or table.tile_form(values.shape) is None:
            return CardTensor(table.Tensor(values.shape, None, values), device)
        tensor = device.upload(values, "fallback_write")
        tensor.host = None  # the card holds it; a later host op reads it back
        return CardTensor(tensor, device)

    return tree_map(back, out)


# torch.compile


@dataclass(frozen=True)
class Weight:
    """A call's weight: `function` of the host values of `nodes` (weights), uploaded
    once a device (cached by `name` and the tensors)."""

    name: str
    function: Callable
    nodes: tuple


@dataclass(frozen=True)
class Inner:
    """A tensor between two calls of one rule, no node of the graph."""

    name: str
    shape: tuple


def plain(node):
    return Weight("plain", lambda value: value, (node,))


@dataclass
class Call:
    """A descriptor's call in a compiled graph: operands (graph nodes, Inner, or
    Weight), its output (a node or an Inner)."""

    descriptor: str
    operands: list
    out: object
    constants: dict = field(default_factory=dict)
    squares: bool = False  # the output keeps its rows' squares (an add's)


@dataclass(frozen=True)
class Rule:
    """A fusion rule: a pattern (a function of tensors, traced to aten) and what a
    match becomes (emit(compiled, match) -> [Call], or None when it does not fit)."""

    name: str
    pattern: Callable
    examples: Callable  # (model) -> example tensors of the pattern
    emit: Callable
    any_literal: bool = False  # literals as wildcards (emit reads the graph's)


def host_values(value):
    """A host tensor's values as float32 numpy (bf16 values)."""
    return md.bf16(value.detach().float().numpy())


def as_float(value):
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.float()
    return value


def bf16(*shape):
    return torch.zeros(shape, dtype=torch.bfloat16)


def literal_eps(compiled, match):
    """Whether every float literal of the match's add nodes is the model's epsilon."""
    for node in match.nodes_map.values():
        if node.op == "call_function" and node.target == aten.add.Tensor:
            for value in node.args:
                if isinstance(value, float) and value != compiled.model.eps:
                    return False
    return True


def kn(node):
    shape = node.meta["val"].shape
    return shape[0] // table.TILE, shape[1] // table.TILE


def emit_norm_gate(compiled, match):
    x, wn, w_gate, w_up = match.placeholder_nodes
    (out,) = match.returning_nodes
    weights = (wn, w_gate, w_up)
    if not (literal_eps(compiled, match) and compiled.normable(x)):
        return None
    if not all(compiled.is_weight(node) for node in weights):
        return None
    k, n = kn(w_gate)
    if (k, 2 * n) not in compiled.device.plans:
        return None

    def prepare(norm, gate, up):
        return table.interleave(table.fold(norm, gate), table.fold(norm, up))

    return [Call("norm_gate", [x, Weight("norm_gate", prepare, weights)], out)]


def emit_matmul_add(compiled, match):
    a, w, residual = match.placeholder_nodes
    (out,) = match.returning_nodes
    plan = compiled.device.plans.get(kn(w))
    if plan is None or plan.groups() != table.D["DC_PARTS"]:
        return None
    if len(out.meta["val"].shape) != 2 or out.meta["val"].shape[0] > table.TILE:
        return None
    weight = plain(w) if compiled.is_weight(w) else w
    return [Call("matmul_add", [a, weight, residual], out, squares=True)]


def emit_qkv_attend(compiled, match):
    x, wn, wq, wk, wv, q_norm, k_norm, cos, sin, keys, values = match.placeholder_nodes
    (out,) = match.returning_nodes
    weights = (wn, wq, wk, wv, q_norm, k_norm, cos, sin, keys, values)
    if not (literal_eps(compiled, match) and compiled.normable(x)):
        return None
    if not all(compiled.is_weight(node) for node in weights):
        return None
    if x.meta["val"].shape[0] != 1:
        return None
    attends = [
        node
        for node in match.nodes_map.values()
        if node.op == "call_function"
        and node.target == torch.ops.komorebi.attend.default
    ]
    position = attends[0].args[3]
    model = compiled.model

    def prepare(norm, q, k, v):
        return table.fold(norm, np.concatenate([q, k, v], axis=1))

    def rotation(cos_values, sin_values):
        cos_table, sin_table = md.rotary(model, position + 1)
        if not (
            np.array_equal(md.bf16(cos_table[position]), cos_values)
            and np.array_equal(md.bf16(sin_table[position]), sin_values)
        ):
            raise ValueError("the module's rotation is not the card's at its position")
        return None

    qkv = Inner("qkv", (1, model.products()["qkv"][1]))
    return [
        Call(
            "norm_qkv",
            [x, Weight("norm_qkv", prepare, (wn, wq, wk, wv))],
            qkv,
            {
                "layer": (q_norm, k_norm),
                "position": position,
                "check": (rotation, (cos, sin)),
            },
        ),
        Call("attend", [qkv, plain(keys), plain(values)], out, {"position": position}),
    ]


def qkv_attend_pattern(
    x, norm_weight, wq, wk, wv, q_norm, k_norm, cos, sin, keys, values
):
    qkv = qkv_attend(
        x, norm_weight, wq, wk, wv, q_norm, k_norm, cos, sin, keys, values, 1e-6
    )
    return torch.ops.komorebi.attend(qkv, keys, values, 0)


def qkv_examples(model):
    q, kv = model.heads * model.head_dim, model.kv_heads * model.head_dim
    hidden, width = model.hidden, model.head_dim
    cache = bf16(model.kv_heads, 64, width)
    return (
        bf16(1, hidden),
        bf16(hidden),
        bf16(hidden, q),
        bf16(hidden, kv),
        bf16(hidden, kv),
        bf16(width),
        bf16(width),
        bf16(width),
        bf16(width),
        cache,
        cache.clone(),
    )


# In order: an add's output keeps its squares, so the norm after it can fuse.
RULES = (
    Rule("norm_qkv + attend", qkv_attend_pattern, qkv_examples, emit_qkv_attend, True),
    Rule(
        "matmul_add",
        lambda a, w, residual: residual + a @ w,
        lambda m: (bf16(1, m.mlp), bf16(m.mlp, m.hidden), bf16(1, m.hidden)),
        emit_matmul_add,
    ),
    Rule(
        "norm_gate",
        lambda x, wn, w_gate, w_up: mlp(x, wn, w_gate, w_up, 1e-6),
        lambda m: (bf16(1, m.hidden), bf16(m.hidden), *[bf16(m.hidden, m.mlp)] * 2),
        emit_norm_gate,
    ),
)


def trace(function, examples):
    """A pattern's aten inference graph, as AOT autograd gives the model's."""
    graphs = []

    def capture(graph, inputs):
        graphs.append(graph)
        return graph

    with torch.no_grad():
        aot_function(function, fw_compiler=capture)(*examples)
    return graphs[0].graph


def backend(gm, example_inputs, options=None):
    """torch.compile's backend "komorebi": options `device` (table.Device) and the
    stages (STAGES, DEFAULTS) and `spin` (one: the wait's spin in seconds)."""
    options = dict(options or {})
    device = options.pop("device")
    stages = {name: bool(options.pop(name, DEFAULTS[name])) for name in STAGES}
    spin = float(options.pop("spin", table.SPIN_S))
    if options:
        raise ValueError(f"unknown options {sorted(options)}")
    names = [node.name for node in gm.graph.nodes if node.op == "placeholder"]
    weights = {i for i, name in enumerate(names) if "_parameters_" in name}
    weights |= {i for i, name in enumerate(names) if "_buffers_" in name}

    def compiler(graph, inputs):
        return Compiled(device, graph, weights, stages, spin=spin)

    return aot_module_simplified(gm, example_inputs, fw_compiler=compiler)


torch._dynamo.register_backend(backend, name="komorebi")


def bf16_bits(value):
    """A tensor's values as bf16 bits, uint16 [rows, columns]: a bf16 tensor's own
    bits, else rounded (formats.bf16_bits)."""
    if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
        bits = value.contiguous().view(torch.int16).numpy().view(np.uint16)
        return bits.reshape(-1, bits.shape[-1])
    values = host_values(value)
    return formats.bf16_bits(values.reshape(-1, values.shape[-1]))


def whole(tensor):
    """A Tensor's whole DRAM tensor (a joint part's, or its own)."""
    card = tensor.card
    return card.whole if isinstance(card, table.Part) else card


class Compiled:
    """A compiled graph: its nodes as card calls (descriptors, fused by RULES when
    `fuse`) and host ops, scheduled so that card calls run together; on the first call
    weights are prepared and uploaded, tensors given (with `once`, kept), lists written;
    each call writes the inputs, runs the lists and host ops, reads the outputs."""

    def __init__(self, device, gm, weights, stages, rules=RULES, spin=table.SPIN_S):
        self.device, self.gm, self.stages, self.spin = device, gm, stages, spin
        self.model = device.decode.model
        self.nodes = list(gm.graph.nodes)
        self.inputs = [node for node in self.nodes if node.op == "placeholder"]
        self.weight_nodes = {self.inputs[i] for i in weights}
        self.output = next(node for node in self.nodes if node.op == "output")
        self.matches = []
        self.calls = {}  # unit's last node -> [Call]
        claimed = {}
        for rule in rules if stages["fuse"] else ():
            pattern = trace(rule.pattern, rule.examples(self.model))
            matcher = SubgraphMatcher(pattern, ignore_literals=rule.any_literal)
            for match in matcher.match(gm.graph):
                inner = [
                    node
                    for pn, node in match.nodes_map.items()
                    if pn.op not in ("placeholder", "output")
                ]
                if any(node in claimed for node in inner):
                    continue
                calls = rule.emit(self, match)
                if calls is None:
                    continue
                for node in inner:
                    claimed[node] = calls
                self.matches.append((rule.name, inner, calls))
        for node in self.nodes:
            if node.op == "call_function" and node not in claimed:
                single = self.single(node)
                if single is not None:
                    claimed[node] = single
        self.claimed = claimed
        self.schedule = self.order()
        self.ready = False
        self.host_ops = [
            str(node.target) for kind, node in self.schedule if kind == "host"
        ]

    # Analysis

    def is_weight(self, node):
        return node in self.weight_nodes

    def normable(self, x):
        """Whether x's squares are known: an input the host writes, an add's output."""
        if x.op == "placeholder":
            return not self.is_weight(x)
        return any(
            call.out is x and call.squares
            for _, _, calls in self.matches
            for call in calls
        )

    def single(self, node):
        """A node's own descriptor call (no rule), or None (a host op)."""
        if node.target == aten.mm.default:
            a, w = node.args
            shape = a.meta["val"].shape
            if shape[0] > table.TILE or kn(w) not in self.device.plans:
                return None
            weight = plain(w) if self.is_weight(w) else w
            return [Call("matmul", [a, weight], node)]
        if node.target == torch.ops.komorebi.attend.default:
            qkv, keys, values, position = node.args
            if not (self.is_weight(keys) and self.is_weight(values)):
                return None
            caches = [plain(keys), plain(values)]
            return [Call("attend", [qkv, *caches], node, {"position": position})]
        return None

    def order(self):
        """The nodes as units in an order torch's allows: a card unit (a rule's match
        or one call) as soon as its inputs are made, host ops only when none is ready.
        """
        units, unit_of = [], {}
        for node in self.nodes:
            if node.op != "call_function":
                continue
            calls = self.claimed.get(node)
            key = id(calls) if calls is not None else id(node)
            if key not in unit_of:
                unit_of[key] = len(units)
                units.append(("card" if calls is not None else "host", [], calls))
            units[unit_of[key]][1].append(node)
        made = {node for node in self.nodes if node.op == "placeholder"}
        done, out = set(), []
        while len(done) < len(units):
            for kind in ("card", "host"):
                ready = [
                    i
                    for i, (k, nodes, _) in enumerate(units)
                    if i not in done
                    and k == kind
                    and all(
                        arg in made or arg in nodes
                        for node in nodes
                        for arg in node.all_input_nodes
                    )
                ]
                if ready:
                    break
            if not ready:
                raise ValueError("the graph's units do not order")
            for i in ready if kind == "card" else ready[:1]:
                kind_i, nodes, calls = units[i]
                done.add(i)
                made.update(nodes)
                out.append(("card", calls) if kind_i == "card" else ("host", nodes[0]))
        return out

    # The first call

    def prepare(self, args):
        """Weights prepared and uploaded (once a device), every tensor a card call takes
        or makes given, the inputs written, each segment's calls planned and written."""
        device = self.device
        host = {node: args[i] for i, node in enumerate(self.inputs)}
        self.tensors = {}
        self.steps = []
        card_made = set()
        needed_on_card = set()
        for kind, item in self.schedule:
            if kind == "card":
                for call in item:
                    for operand in call.operands:
                        if not isinstance(operand, Weight | Inner):
                            needed_on_card.add(operand)
                    card_made.add(call.out)
        self.needed_on_card = needed_on_card
        for node in self.inputs:
            if node in needed_on_card and not self.is_weight(node):
                shape = tuple(node.meta["val"].shape)
                if self.stages["joint"] and self.needs_squares(node):
                    self.tensors[node] = device.empty_joint(shape)
                else:
                    self.tensors[node] = device.empty(shape)
        segment = []
        for kind, item in self.schedule:
            if kind == "host":
                if segment:
                    self.steps.append(("run", segment))
                    segment = []
                for arg in item.all_input_nodes:
                    if arg in card_made and ("read", arg) not in self.steps:
                        self.steps.append(("read", arg))
                self.steps.append(("host", item))
                if any(user in needed_on_card for user in [item]):
                    self.steps.append(("write", item))
                continue
            for call in item:
                for operand in call.operands:
                    if isinstance(operand, Weight | Inner) or operand in self.tensors:
                        continue
                    self.tensors[operand] = device.empty(
                        tuple(operand.meta["val"].shape)
                    )
                shape = (
                    call.out.shape
                    if isinstance(call.out, Inner)
                    else tuple(call.out.meta["val"].shape)
                )
                if call.out not in self.tensors:
                    self.tensors[call.out] = device.empty(shape, squares=call.squares)
                segment.append(call)
        if segment:
            self.steps.append(("run", segment))
        for value in self.output.args[0]:
            if value in card_made:
                self.steps.append(("read", value))
        self.write_inputs(host)
        planned = []
        for step in self.steps:
            if step[0] == "run":
                planned.append(("run", self.plan_segment(step[1], host)))
            else:
                planned.append(step)
        self.steps = planned
        self.one = self.one_submission() if self.stages["one"] else None
        self.ready = True

    def one_submission(self):
        """The call as one host submission (table.OneSubmission): a graph of one list
        whose outputs are all the list's, its inputs pulled and outputs pushed by it."""
        kinds = [kind for kind, _ in self.steps]
        outputs = list(self.output.args[0])
        if kinds != ["run"] + ["read"] * len(outputs):
            raise ValueError(f"one submission takes one list and its reads: {kinds}")
        self.pulled = [
            node
            for node in self.inputs
            if node in self.tensors and not self.is_weight(node)
        ]
        inputs = [whole(self.tensors[node]) for node in self.pulled]
        pushed = [whole(self.tensors[node]) for node in outputs]
        one = table.OneSubmission(self.device, self.steps[0][1], inputs, pushed)
        self.writers = []
        for node, move in zip(self.pulled, one.pulls, strict=True):
            tensor = self.tensors[node]
            joint = isinstance(tensor.card, table.Part)
            columns = tensor.card.shape[1]
            self.writers.append(
                (node, move, table.TileWriter(tensor.rows, columns, joint))
            )
        self.takes = []
        for node, move in zip(outputs, one.pushes, strict=True):
            shape = tuple(node.meta["val"].shape)
            rows = math.prod(shape) // shape[-1]
            if shape[-1] != self.tensors[node].card.shape[1]:
                raise ValueError(f"{shape}: an output narrower than its tile rows")
            self.takes.append((move, rows, shape))
        return one

    def call_one(self, host):
        """A call of the one-submission form: the inputs' bf16 bits tiled into the
        region, the outputs' bits out of it as bf16 tensors, nothing else made."""
        one = self.one

        def fill():
            for node, move, writer in self.writers:
                writer(one.tiles(move), bf16_bits(host[node]))

        def take():
            return tuple(
                torch.from_numpy(table.tile_bits(one.tiles(move), rows).view(np.int16))
                .view(torch.bfloat16)
                .reshape(shape)
                for move, rows, shape in self.takes
            )

        return one(fill, take, self.spin)

    def operand(self, operand, host):
        """A call's operand as a Tensor: a node's, or a prepared weight's (cached)."""
        if not isinstance(operand, Weight):
            return self.tensors[operand]
        key = (operand.name, *(id(host[node]) for node in operand.nodes))
        if key not in self.device.prepared:
            values = [host_values(host[node]) for node in operand.nodes]
            started = time.perf_counter()
            prepared = operand.function(*values)
            self.device.prepared[key] = self.device.upload(prepared, "weights")
            self.device.clock["weights"] += time.perf_counter() - started
        return self.device.prepared[key]

    def plan_segment(self, calls, host):
        """A segment's calls planned and its closed list written: (entries, count)."""
        planned = []
        for call in calls:
            constants = {}
            for name, value in call.constants.items():
                if name == "layer":
                    norms = [host_values(host[node]) for node in value]
                    constants["layer"] = self.device.qk_layer(*norms)
                elif name == "check":
                    function, nodes = value
                    function(*(host_values(host[node]) for node in nodes))
                else:
                    constants[name] = value
            inputs = [self.operand(operand, host) for operand in call.operands]
            out = self.tensors[call.out]
            descriptor = table.TABLE[call.descriptor]
            planned.append(descriptor.plan(self.device, inputs, out, **constants))
        if self.stages["chain"]:
            items = table.chained(planned)
        else:
            items = [("command", call) for call in planned]
        count = sum(len(item) if kind == "chain" else 1 for kind, item in items)
        entries = table.entries_of(items, self.device.reserve(count))
        self.device.write_list(entries)
        return entries

    # Each call

    def write_inputs(self, host):
        for node in self.inputs:
            if node in self.tensors and not self.is_weight(node):
                values = host_values(host[node])
                tensor = self.tensors[node]
                if isinstance(tensor.card, table.Part):
                    self.device.write_joint(tensor, values)
                    continue
                self.device.write(tensor, values, "in_write")
                if self.needs_squares(node):
                    self.device.write_squares(tensor, values)

    def needs_squares(self, node):
        return any(
            call.descriptor.startswith("norm") and call.operands[0] is node
            for _, _, calls in self.matches
            for call in calls
        )

    def __call__(self, *args):
        device = self.device
        if not self.ready or not self.stages["once"]:
            if self.ready:
                self.release()
            host = {node: args[i] for i, node in enumerate(self.inputs)}
            self.prepare(args)
        else:
            if self.one is not None:
                started = time.perf_counter()
                out = self.call_one(dict(zip(self.inputs, args, strict=True)))
                device.clock["call_body"] += time.perf_counter() - started
                return out
            host = {node: args[i] for i, node in enumerate(self.inputs)}
            self.write_inputs(host)
        values = dict(host)
        for step in self.steps:
            kind, item = step
            if kind == "run":
                device.run_list(item, "compiled")
            elif kind == "read":
                values[item] = torch.from_numpy(
                    np.array(device.read(self.tensors[item], "out_read"), np.float32)
                ).to(torch.bfloat16)
            elif kind == "host":
                device.fallback(str(item.target))
                started = time.perf_counter()
                args, kwargs = (
                    tree_map(lambda a: as_float(values.get(a, a)), item.args),
                    {},
                )
                kwargs = tree_map(lambda a: as_float(values.get(a, a)), item.kwargs)
                out = item.target(*args, **kwargs)
                if isinstance(out, torch.Tensor) and out.is_floating_point():
                    out = out.to(torch.bfloat16)
                device.clock["host"] += time.perf_counter() - started
                values[item] = out
            else:
                device.write(self.tensors[item], values[item].float().numpy(), "write")
        return tuple(values[value] for value in self.output.args[0])

    def release(self):
        if self.one is not None:
            self.one.collect()
        for node, tensor in self.tensors.items():
            if node not in self.weight_nodes:
                self.device.free(tensor)
        self.ready = False

    def describe(self):
        """The compiled graph: the rules' matches, the card calls, the host ops."""
        cards = [
            call.descriptor
            for kind, item in self.schedule
            if kind == "card"
            for call in item
        ]
        return {
            "stages": self.stages,
            "matches": [name for name, _, _ in self.matches],
            "card_calls": cards,
            "host_ops": self.host_ops,
            "lists": sum(1 for kind, _ in self.schedule if kind == "card"),
        }
