"""Named reference graphs, saved as files for the kernel's executor to replay.

Each graph is a `.bin` upload and a `.json` with bindings, parameters and answer.
`python graph_reference.py` rewrites graphs/; `--check` only compares (both engines).
"""

import json
import struct
import sys
from pathlib import Path

from komorebi.engines import graph as gr
from komorebi.engines import stream as st
from komorebi.engines.graph import Graph
from komorebi.engines.graph_native import NativeExecutor, native_check

DATA = Path(__file__).resolve().parent / "graphs"
SEG_A, SEG_B, SEG_C, SEG_D = 1, 2, 3, 4

NOTES = (
    "Binding: check, status, error, regs, runs, stopped_grid, the issue set and the"
    " events without their pass. The issue set lists every record issued, sorted by"
    " (run, node): seg, first (null for a queued record, the allocator places it),"
    " count, flags as the grid sees them, a0 to a9 with trailing zeros left out."
    " `order` (seq, slot, pass per record) and the event passes are the simulator's"
    " and are advisory. Completion: a record issued in pass p is acknowledged by the"
    " sweep of pass p + 1 + delay; delay is an integer, null (never) or {mod: m}"
    " (seq % m, seq counted from 1 over the whole graph). A record's result word is"
    " (a0 + count) mod 2^32. Events are [kind, req, aux, extra]: extra is register 1"
    " for KM_EV_GRAPH_END, else 0. Rules: docs/graphs.md."
)

DEFAULTS = {
    "n_cores": 119,
    "delay": 0,
    "deadline_passes": None,
    "cancel_pass": None,
    "max_passes": 100_000,
}


def result_of(record):
    return record.args[0] + record.count & st.MASK


def completion(spec):
    if isinstance(spec, dict):
        return lambda r: r.seq % spec["mod"]
    return spec


def run(upload, bindings, params, engine=None):
    p = {**DEFAULTS, **params}
    return gr.simulate(
        upload,
        bindings=bindings,
        n_cores=p["n_cores"],
        delay=completion(p["delay"]),
        result=result_of,
        deadline_passes=p["deadline_passes"],
        cancel_pass=p["cancel_pass"],
        max_passes=p["max_passes"],
        engine=engine,
    )


def trimmed(args):
    args = list(args)
    while args and args[-1] == 0:
        args.pop()
    return args


def expected(upload, bindings, params, engine=None):
    r = run(upload, bindings, params, engine)
    bad = gr.check(upload)
    records = sorted(r.records, key=lambda x: (x.run, x.node))
    return {
        "check": list(bad) if bad else None,
        "status": r.status,
        "error": list(r.error) if r.error else None,
        "regs": r.regs,
        "runs": r.runs,
        "stopped_grid": r.stopped,
        "issued": [
            {
                "run": x.run,
                "node": x.node,
                "seg": x.seg,
                "first": None if x.queued else x.first,
                "count": x.count,
                "flags": x.flags,
                "args": trimmed(x.args),
            }
            for x in records
        ],
        "events": [list(e[:4]) for e in r.events],
        "order": [[x.seq, x.slot, x.pass_index] for x in records],
        "event_passes": [e.pass_index for e in r.events],
        "passes": r.passes,
    }


# Graphs that run ----------------------------------------------------------------


def chain():
    g = Graph()
    a = g.record(SEG_A, team=(0, 4), args=[1, 2])
    b = g.record(SEG_B, team=(4, 4), args=[3], after=[a])
    g.record(SEG_C, team=(0, 119), after=[b])
    return g, [], {}


def branches():
    """a, then b and c on disjoint halves in flight together, then d."""
    g = Graph()
    a = g.record(SEG_A, team=(0, 60))
    b = g.record(SEG_B, team=(0, 30), after=[a])
    c = g.record(SEG_C, team=(30, 30), after=[a])
    g.record(SEG_D, team=(0, 60), after=[b, c])
    return g, [], {"delay": 2}


def example_loop(it=0):
    """The documented example: the loop ends on a result word computed per round."""
    g = Graph(max_runs=100)
    i, err = g.bindings("it"), g.reg()
    a = g.record(SEG_A, team=(0, 60), args=[i])
    b = g.record(SEG_B, team=(0, 30), args=[i], after=[a])
    c = g.record(SEG_C, team=(30, 30), args=[i], after=[a])
    d = g.record(SEG_D, team=(0, 60), args=[i], after=[b, c], result=err)
    e = g.cpu(after=[d])
    e.addi(i, i, 1)
    e.lt(g.again, err, g.const(65))
    return g, g.binding_values(it=it), {"delay": {"mod": 3}}


def example_loop_kept():
    """The same upload as example_loop, run again with it = 2: fewer rounds."""
    return example_loop(2)


def skip_if_zero():
    g = Graph()
    off, on = g.bindings("off", "on")
    g.record(SEG_A, team=(0, 1), skip_if_zero=off)
    g.record(SEG_B, team=(1, 1), skip_if_zero=on)
    j = g.join(skip_if_zero=off)
    g.record(SEG_C, team=(2, 1), after=[j])
    return g, g.binding_values(off=0, on=1), {}


def cpu_arith():
    g = Graph()
    r = g.regs(8)
    e = g.cpu()
    e.set(r[0], 0xFFFFFFFF)
    e.set(r[1], 3)
    e.add(r[2], r[0], r[1])
    e.sub(r[3], r[1], r[0])
    e.mul(r[4], r[0], r[0])
    e.and_(r[5], r[0], r[1])
    e.shr(r[6], r[0], 36)
    e.mod(r[7], r[0], r[1])
    e.addi(r[1], r[1], -1)
    e.rand(r[0])
    f = g.cpu(after=[e])
    top, out = f.label(), f.forward()
    f.addi(r[5], r[5], 5)
    f.addi(r[1], r[1], -1)
    f.jnz(r[1], top)
    f.jeq(r[1], r[7], out)
    f.set(r[6], 99)
    f.bind(out)
    f.mov(r[6], r[5])
    g.record(SEG_A, team=(0, 1), args=r[2:6], after=[f])
    return g, [0] * 15 + [7], {}


def args_and_team_from_registers():
    g = Graph()
    first, count, x = g.bindings("first", "count", "x")
    g.record(SEG_A, team=(first, count), args=[9, x, 8, x, 7])
    return g, g.binding_values(first=110, count=9, x=0xDEADBEEF), {}


def result_feeds_record():
    g = Graph()
    v = g.reg()
    a = g.record(SEG_A, team=(0, 2), args=[40], result=v)
    g.record(SEG_B, team=(0, 2), args=[v], after=[a])
    return g, [], {}


def queued_records():
    g = Graph()
    n = g.bindings("n")
    a = g.record(SEG_A, count=50, args=[1])
    b = g.record(SEG_B, count=n, args=[2])
    g.record(SEG_C, count=119, after=[a, b])
    return g, g.binding_values(n=60), {"delay": 1}


def exclusive_overlap():
    g = Graph()
    g.record(SEG_A, team=(0, 10), exclusive=True)
    g.record(SEG_B, team=(5, 10), exclusive=True)
    g.record(SEG_C, team=(20, 10), exclusive=True)
    return g, [], {"delay": 2}


def more_ready_than_slots():
    g = Graph()
    nodes = [g.record(SEG_A, team=(i, 1), args=[i]) for i in range(20)]
    j1 = g.join(after=nodes[:4])
    j2 = g.join(after=nodes[4:8])
    g.join(after=[j1, j2])
    return g, [], {"delay": {"mod": 4}}


def round_events():
    """Three rounds of a CPU-only graph: two KM_EV_GRAPH events, then the end."""
    g = Graph(max_runs=3)
    k = g.reg()
    e = g.cpu()
    e.addi(k, k, 1)
    e.lt(g.again, k, g.const(3))
    return g, g.binding_values(), {}


# Graphs that end in an error at run time ----------------------------------------


def team_outside_grid():
    g = Graph()
    first = g.bindings("first")
    g.record(SEG_A, team=(first, 2))
    return g, g.binding_values(first=118), {}


def never_fits():
    g = Graph()
    n = g.bindings("n")
    g.record(SEG_A, count=n)
    return g, g.binding_values(n=120), {}


def mod_zero():
    g = Graph()
    a, b = g.regs(2)
    e = g.cpu()
    e.mod(a, a, b)
    return g, [], {}


def step_limit():
    g = Graph(step_limit=10)
    a = g.reg()
    e = g.cpu()
    top = e.label()
    e.addi(a, a, 1)
    e.jmp(top)
    return g, [], {}


def max_runs():
    g = Graph(max_runs=3)
    e = g.cpu()
    e.set(g.again, 1)
    return g, [], {}


def deadline():
    g = Graph()
    a = g.record(SEG_A, team=(0, 1))
    g.record(SEG_B, team=(1, 1))
    g.join(after=[a])
    return g, [], {"delay": None, "deadline_passes": 5}


def cancelled():
    g = Graph()
    g.record(SEG_A, team=(0, 1))
    return g, [], {"delay": None, "cancel_pass": 3}


def too_many_bindings():
    g = Graph()
    g.join()
    return g, [0] * 17, {}


# Uploads the check refuses: each patches one field of a good upload ------------


def _base():
    g = Graph()
    r = g.reg()
    a = g.record(SEG_A, team=(0, 1), result=r)
    e = g.cpu(after=[a])
    e.addi(r, r, 1)
    g.join(after=[a, e])
    return g


def _patch_node(data, index, **fields):
    at = gr.HEADER.size + index * gr.NODE_BYTES
    w = list(gr.NODE.unpack_from(data, at))
    names = "kind flags n_pred skip_reg arg n_ops result_reg pad0 succ_off succ_n pad1"
    for name, value in fields.items():
        w[names.split().index(name)] = value
    return data[:at] + gr.NODE.pack(*w) + data[at + gr.NODE_BYTES :]


def _patch_word(data, at, fmt, value):
    return data[:at] + struct.pack(fmt, value) + data[at + struct.calcsize(fmt) :]


def _layout(data):
    h = dict(zip(gr.HEADER_FIELDS, gr.HEADER.unpack_from(data)))
    return gr._layout(h)


BAD = {
    "bad_magic": lambda d: _patch_word(d, 0, "<I", 0x4B4D5354),
    "bad_step_limit": lambda d: _patch_word(d, 40, "<I", gr.MAX_STEPS + 1),
    "bad_size": lambda d: d + bytes(64),
    "bad_kind": lambda d: _patch_node(d, 2, kind=0),
    "bad_flags": lambda d: _patch_node(d, 2, flags=1),
    "bad_skip_reg": lambda d: _patch_node(d, 1, skip_reg=16),
    "bad_result_on_cpu": lambda d: _patch_node(d, 1, result_reg=0),
    "bad_template_index": lambda d: _patch_node(d, 0, arg=1),
    "bad_template_segment": lambda d: _patch_word(d, _layout(d)[2] + 4, "<I", 240),
    "bad_opcode": lambda d: _patch_word(d, _layout(d)[1], "<B", st.OP_HALT),
    "bad_jump": lambda d: _patch_word(
        _patch_word(d, _layout(d)[1], "<B", st.OP_JMP), _layout(d)[1] + 4, "<I", 2
    ),
    "bad_register": lambda d: _patch_word(d, _layout(d)[1] + 1, "<B", 16),
    "bad_slice": lambda d: _patch_node(d, 1, arg=1),
    "bad_pred_count": lambda d: _patch_node(d, 2, n_pred=1),
    "bad_successor_order": lambda d: _patch_word(d, _layout(d)[0] + 2, "<H", 0),
    "bad_successor_offset": lambda d: _patch_node(d, 1, succ_off=0),
}

GOOD = [
    chain,
    branches,
    example_loop,
    example_loop_kept,
    skip_if_zero,
    cpu_arith,
    args_and_team_from_registers,
    result_feeds_record,
    queued_records,
    exclusive_overlap,
    more_ready_than_slots,
    round_events,
    team_outside_grid,
    never_fits,
    mod_zero,
    step_limit,
    max_runs,
    deadline,
    cancelled,
    too_many_bindings,
]


def references():
    """(name, description, upload, bindings, params) for every reference."""
    out = []
    for build in GOOD:
        g, bindings, params = build()
        doc = (build.__doc__ or build.__name__.replace("_", " ")).strip()
        out.append((build.__name__, doc, g.encode(), list(bindings), params))
    for name, patch in BAD.items():
        upload = patch(_base().encode())
        out.append((name, name.replace("_", " "), upload, [], {}))
    return out


def main(argv):
    check_only = "--check" in argv
    DATA.mkdir(exist_ok=True)
    differ = 0
    for name, doc, upload, bindings, params in references():
        want = expected(upload, bindings, params)
        if want != expected(upload, bindings, params, NativeExecutor()):
            print(f"{name}: the C engine differs from the Python executor")
            differ += 1
        if native_check(upload) != gr.check(upload):
            print(f"{name}: kmg_check differs from check")
            differ += 1
        entry = {
            "name": name,
            "description": doc,
            "notes": NOTES,
            "bindings": bindings,
            "params": {**DEFAULTS, **params},
            "expect": want,
        }
        text = json.dumps(entry, indent=1) + "\n"
        if check_only:
            saved = DATA / f"{name}.json"
            same = saved.is_file() and json.loads(saved.read_text()) == entry
            same = same and (DATA / f"{name}.bin").read_bytes() == upload
            if not same:
                print(f"{name}: differs from the saved reference")
                differ += 1
        else:
            (DATA / f"{name}.bin").write_bytes(upload)
            (DATA / f"{name}.json").write_text(text)
    print(f"{len(references())} references, {differ} differences")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
