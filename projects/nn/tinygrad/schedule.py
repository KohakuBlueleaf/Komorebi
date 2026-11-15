"""tinygrad's schedule of the two models as a list: each under TinyJit on tinygrad's
CPU device, called three times; the replayed calls in order, each kernel's buffers,
operations and stored expressions (from its AST before code generation). Needs
tinygrad, not a dependency of komorebi: run it where tinygrad is, with DEV=CPU."""

import argparse
import json
import math
import re
import sys

import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes, nn
from tinygrad.engine import realize
from tinygrad.uop.ops import Ops

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# Qwen3-1.7B, the decode project's shapes (projects/decode/model.py)
HIDDEN, MLP, HEADS, KV_HEADS, HEAD = 2048, 6144, 16, 8, 128
EPS, THETA, CONTEXT = 1e-6, 1e6, 4096
ROWS = 32  # model A's activation rows: one tile row
SEED = 1
# the operations a kernel's AST names, beside loads, stores, indices and constants
STRUCTURE = {
    "SINK", "STORE", "LOAD", "INDEX", "CONST", "DEFINE_VAR", "PARAM", "RANGE", "END",
    "BUFFER", "AFTER", "GROUP", "VCONST", "NOOP", "SPECIAL", "BIND", "VECTORIZE",
    "GEP", "DEFINE_REG", "DEFINE_LOCAL", "BARRIER", "UNROLL", "CONTRACT", "SHRINK",
}  # fmt: skip

ASTS = {}  # the PROGRAM's key -> the AST it was made from
_to_program = realize.to_program


def _recording(ast, renderer):
    program = _to_program(ast, renderer)
    ASTS[program.key] = ast
    return program


realize.to_program = _recording


def describe(ast):
    """The operations (beside the structure), the reductions, the casts and the
    stored expressions."""
    topo = ast.toposort()
    ops = sorted({u.op.name for u in topo} - STRUCTURE)
    reduce = sorted({str(u.arg[0]).split(".")[-1] for u in topo if u.op is Ops.REDUCE})
    casts = sorted({u.dtype.name for u in topo if u.op is Ops.CAST})
    out = {"ops": ops, "reduce": reduce, "casts": casts}
    out["epilogue"] = [epilogue(u.src[1]) for u in topo if u.op is Ops.STORE]
    return out


# tinygrad's elementwise operations as the epilogue's expression names them
EXPRESSION = {
    Ops.ADD: "add", Ops.MUL: "mul", Ops.RECIPROCAL: "recip", Ops.EXP2: "exp2",
    Ops.SQRT: "sqrt", Ops.MAX: "max", Ops.NEG: "neg", Ops.LOG2: "log2",
    Ops.SIN: "sin", Ops.WHERE: "where", Ops.CMPLT: "lt", Ops.CMPNE: "ne",
}  # fmt: skip


def epilogue(value):
    """A stored value as an expression of its inputs: each reduction and each load
    an input (numbered in order), so ["mul", ["in", 0], ["const", 2.0]]; an
    operation of no name here is ["unnamed", its name]."""
    inputs = []

    def walk(u):
        if u.op is Ops.CONST:
            return ["const", float(u.arg)]
        if u.op is Ops.CAST and u.dtype.name == u.src[0].dtype.name:
            return walk(u.src[0])
        if u.op in (Ops.REDUCE, Ops.INDEX, Ops.LOAD) or u.op is Ops.CAST:
            if u not in inputs:
                inputs.append(u)
            return ["in", inputs.index(u)]
        if u.op not in EXPRESSION:
            return ["unnamed", u.op.name]
        return [EXPRESSION[u.op], *(walk(s) for s in u.src)]

    expression = walk(value)
    names = []
    for u in inputs:
        if u.op is Ops.REDUCE:
            names.append(f"reduce {str(u.arg[0]).split('.')[-1]}")
        elif u.op is Ops.CAST:
            names.append(f"{u.src[0].op.name} {u.src[0].dtype.name} as {u.dtype.name}")
        else:
            names.append(f"load {u.dtype.name}")
    return {"expression": expression, "inputs": names}


# a call's argument: the step's input, a buffer the step holds, or a view of the
# JIT's scratch memory (an intermediate)
ROLE = {Ops.PARAM: "input", Ops.BUFFER: "held", Ops.BITCAST: "scratch"}


def buffer_of(u):
    role = ROLE.get(u.op, u.op.name)
    return {"role": role, "elements": int(u.max_shape[0]), "dtype": u.dtype.name}


def calls(linear):
    """The captured list's calls in order, each a dict."""
    out = []
    for call in linear.src:
        head = call.src[0]
        item = {"kind": head.op.name}
        if head.op is Ops.PROGRAM:
            info = head.arg
            item["name"] = ANSI.sub("", info.name)
            item["vars"] = [str(v.arg.name) for v in info.vars]
            item["outs"] = list(info.outs)
            ast = ASTS.get(head.key)
            item.update(describe(ast if ast is not None else head.src[0]))
        else:
            item["name"] = str(head.arg)[:80]
        item["buffers"] = [buffer_of(u) for u in call.src[1:]]
        out.append(item)
    return out


def weight(k, n):
    """A bf16 weight of k x n (the decode project's form), as tinygrad's Linear makes
    it (out x in), its values of scale 1 / sqrt(k)."""
    w = Tensor.uniform(n, k, low=-1 / math.sqrt(k), high=1 / math.sqrt(k))
    return w.cast(dtypes.bfloat16).contiguous().realize()


def model_a():
    """g = silu(x W1) (x W3), y = x + g W2 at 32 x 2048, W1, W3 2048 x 6144."""
    w1, w3, w2 = weight(HIDDEN, MLP), weight(HIDDEN, MLP), weight(MLP, HIDDEN)

    @TinyJit
    def step(x):
        g = x.linear(w1.T).silu() * x.linear(w3.T)
        return (x + g.linear(w2.T)).realize()

    inputs = [Tensor.randn(ROWS, HIDDEN).realize() for _ in range(3)]
    return step, [(x,) for x in inputs]


def apply_rope(x, freqs):
    # tinygrad.llm.model.apply_rope
    cos, sin = freqs.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)


class Block:
    """tinygrad.llm.model's TransformerBlock (its ops, in its order) with Qwen3's
    per-head norm, for one decode token; the cache in fp16 as tinygrad keeps it."""

    def __init__(self):
        self.attn_norm, self.ffn_norm = nn.RMSNorm(HIDDEN, EPS), nn.RMSNorm(HIDDEN, EPS)
        self.q_norm, self.k_norm = nn.RMSNorm(HEAD, EPS), nn.RMSNorm(HEAD, EPS)
        self.wq, self.wk = weight(HIDDEN, HEADS * HEAD), weight(HIDDEN, KV_HEADS * HEAD)
        self.wv, self.wo = weight(HIDDEN, KV_HEADS * HEAD), weight(HEADS * HEAD, HIDDEN)
        self.wg, self.wu = weight(HIDDEN, MLP), weight(HIDDEN, MLP)
        self.wd = weight(MLP, HIDDEN)
        shape = (2, 1, KV_HEADS, CONTEXT, HEAD)
        self.cache = Tensor.zeros(*shape, dtype=dtypes.half).contiguous().realize()
        freqs = 1.0 / (THETA ** (Tensor.arange(0, HEAD, 2)[: HEAD // 2] / HEAD))
        freqs = Tensor.arange(CONTEXT).unsqueeze(1) * freqs.unsqueeze(0)
        self.freqs = freqs.cos().cat(freqs.sin(), dim=-1).contiguous().realize()

    def attention(self, x, start):
        q, k, v = x.linear(self.wq.T), x.linear(self.wk.T), x.linear(self.wv.T)
        b, t, _ = x.shape
        q = q.reshape(b, t, HEADS, HEAD).transpose(1, 2)
        k = k.reshape(b, t, KV_HEADS, HEAD).transpose(1, 2)
        v = v.reshape(b, t, KV_HEADS, HEAD).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        freqs = self.freqs[start : start + t]
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        new = Tensor.stack(k, v).cast(dtypes.half)
        store = self.cache[:, :, :, start : start + t, :].uop.store(new.uop)
        kv = Tensor(self.cache.uop.after(store))
        keys, values = kv[0, :, :, 0 : start + t, :], kv[1, :, :, 0 : start + t, :]
        a = q.scaled_dot_product_attention(keys, values, enable_gqa=True)
        return a.transpose(1, 2).reshape(b, t, -1).linear(self.wo.T)

    def __call__(self, x, start):
        h = x + self.attention(self.attn_norm(x), start)
        f = self.ffn_norm(h)
        g = f.linear(self.wg.T).silu().contiguous() * f.linear(self.wu.T)
        return (h + g.linear(self.wd.T)).contiguous()


def model_b():
    """One decode token through the layer at cache positions 100, 101, 102 (the
    position a bound variable, as tinygrad's generate binds start_pos)."""
    block = Block()
    position = UOp.variable("start_pos", 0, CONTEXT - 2)

    @TinyJit
    def step(x, start):
        return block(x, start).realize()

    inputs = [Tensor.randn(1, 1, HIDDEN).realize() for _ in range(3)]
    return step, [(x, position.bind(100 + i)) for i, x in enumerate(inputs)]


def dump(make):
    step, args = make()
    for a in args:
        step(*a)
    return calls(step.captured.linear)


def gate_case(path, epilogue):
    """Model A's first kernel's operands for a card run of its epilogue: tinygrad's
    two products (each its own reduction kernel), the fused kernel's output, the
    epilogue; and whether tinygrad's own elementwise kernel on them gives the same."""
    Tensor.manual_seed(SEED)
    x = Tensor.randn(ROWS, HIDDEN).realize()
    w1, w3 = weight(HIDDEN, MLP), weight(HIDDEN, MLP)
    a = x.linear(w1.T).contiguous().realize()
    b = x.linear(w3.T).contiguous().realize()
    g = (x.linear(w1.T).silu() * x.linear(w3.T)).realize()
    again = (a.silu() * b).realize().numpy()
    data = {"a": a.numpy(), "b": b.numpy(), "g": g.numpy()}
    np.savez(path, **data, epilogue=json.dumps(epilogue))
    return bool((again.view(np.uint32) == data["g"].view(np.uint32)).all())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", help="write the two lists here")
    p.add_argument("--gate", help="write model A's first kernel's case here (.npz)")
    args = p.parse_args()
    Tensor.manual_seed(SEED)
    out = {"a": dump(model_a), "b": dump(model_b)}
    if args.gate:
        same = gate_case(args.gate, out["a"][0]["epilogue"][0])
        print(
            f"gate case {args.gate}: the products' own kernels give the fused "
            f"kernel's bits: {same}"
        )
    for model, items in out.items():
        print(f"model {model}: {len(items)} calls replayed")
        for i, item in enumerate(items):
            sizes = [
                f"{b['role']} {b['dtype']} {b['elements']}" for b in item["buffers"]
            ]
            print(
                f"  {i:2d} {item['kind']:8s} {item['name']:40s} {item.get('ops')} "
                f"reduce {item.get('reduce')} casts {item.get('casts')} {sizes}"
            )
            for e in item.get("epilogue", []):
                print(f"       stores {json.dumps(e['expression'])} of {e['inputs']}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
