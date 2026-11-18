"""The Dst window check on a step-form unit (a softmax step, state tile 3): it builds
and its verdict is within Dst's 8 tiles."""

import math

from komorebi.dsl.tile import trace
from komorebi.dsl.tile.forms.steps import step_program
from komorebi.dsl.tile.ir import F32, KernelIR, TensorParam

s, st = TensorParam("s", False, F32), TensorParam("st", False, F32)
p, out = TensorParam("p", True, F32), TensorParam("out", True, F32)
ir = KernelIR("softstep", [s, st, p, out], [], {})
with trace.Tracer(ir):
    ss, sts, ps, outs = (trace.TensorProxy(t) for t in (s, st, p, out))
    for t in trace.grid(trace.num_tile_rows(ss)):
        x = trace.load(ss.tiles[t, :]) * 0.18
        state = trace.load_state(sts.tiles[t])
        m_new = trace.maximum(state[0], trace.reduce_max(x, exact=True))
        prob = trace.exp2(x - m_new.col(), error=2e-3, over=(-math.inf, 0.0))
        l_new = state[1] + trace.reduce_sum(prob, exact=True)
        trace.store(ps.tiles[t, :], prob)
        moved = trace.word_max(m_new - state[0])
        trace.store_state(outs.tiles[t], [m_new, l_new], word=moved)
verdict = {c.rule: c.verdict for c in step_program(ir, 2, 3).checks}["dst-window"]
print("step form dst-window:", verdict)
assert verdict.endswith("of 8") and int(verdict.split()[0]) <= 8
print("OK")
