"""Named reference streams, saved as files for the kernel's interpreter to replay.

Each stream is a `.bin` upload and a `.json` with the simulator's answer.
`python stream_reference.py` rewrites streams/; `--check` only compares them.
"""

import json
import struct
import sys
from pathlib import Path

from stream_reference_more import MORE

from komorebi.engines import stream as st
from komorebi.engines.stream import Stream

DATA = Path(__file__).resolve().parent / "streams"
SEG_A, SEG_B = 1, 7

NOTES = (
    "Records, events, status, error, halt and stopped_grid are the binding answer;"
    " `passes` is the simulator's own count and is advisory. Completion: a record"
    " issued in pass p is acknowledged by the sweep of pass p + 1 + delay (null:"
    " never); delay is an integer, {base, mod} (base + seq % mod) or {by_seq: {seq:"
    " delay}, rest: delay}, where seq is the stream's own count of issued records from"
    " 1 (the `seq` of each record). A KM_F_RESULT record's result is (a0 + count) mod"
    " 2^32. Records list a0 to a9 with trailing zeros left out. Events are"
    " [kind, req, aux, pass]. Pass order and the other rules: docs/streams.md."
)

DEFAULTS = {
    "n_cores": 119,
    "delay": 0,
    "deadline_passes": None,
    "default_deadline_passes": 1_000_000,
    "cancel_pass": None,
    "max_passes": 100_000,
    "ticks_per_pass": 1000,
    "region_size": 1 << 20,
}


def result_of(record):
    return record.args[0] + record.count & st.MASK


def completion(spec):
    """The delay function or number of a `delay` parameter of a reference."""
    if not isinstance(spec, dict):
        return spec
    if "by_seq" in spec:
        table = {int(k): v for k, v in spec["by_seq"].items()}
        return lambda r: table.get(r.seq, spec["rest"])
    return lambda r: spec["base"] + r.seq % spec["mod"]


def run(upload, params, module=st):
    """Simulate an upload with the saved parameters (`module` holds the simulator)."""
    p = {**DEFAULTS, **params}
    return module.simulate(
        upload,
        n_cores=p["n_cores"],
        delay=completion(p["delay"]),
        result=result_of,
        deadline_passes=p["deadline_passes"],
        default_deadline_passes=p["default_deadline_passes"],
        cancel_pass=p["cancel_pass"],
        max_passes=p["max_passes"],
        ticks_per_pass=p["ticks_per_pass"],
        region_size=p["region_size"],
    )


def trimmed(args):
    args = list(args)
    while args and args[-1] == 0:
        args.pop()
    return args


def expected(upload, params, module=st):
    p = {**DEFAULTS, **params}
    r = run(upload, params, module)
    bad = module.check(upload, p["region_size"])
    return {
        "check": list(bad) if bad else None,
        "status": r.status,
        "error": list(r.error) if r.error else None,
        "halt": r.halt,
        "stopped_grid": r.stopped,
        "passes": r.passes,
        "records": [
            {
                "seq": x.seq,
                "slot": x.slot,
                "pass": x.pass_index,
                "pc": x.pc,
                "queued": x.queued,
                "seg": x.seg,
                "first": x.first,
                "count": x.count,
                "flags": x.flags,
                "args": trimmed(x.args),
            }
            for x in r.records
        ],
        "events": [list(e) for e in r.events],
    }


def _halt_with(s, value):
    s.halt(s.const(value))
    return s


def arith_wrap():
    s = Stream(max_steps=100, seed=1)
    a, b, c = s.regs(3)
    s.set(a, 0xFFFFFFFF)
    s.set(b, 2)
    s.add(c, a, b)
    s.sub(c, c, b)
    s.mul(c, c, b)
    s.addi(c, c, 0xFFFFFFFF)
    s.halt(c)
    return s


def shifts_and_mask():
    s = Stream()
    a, b = s.regs(2)
    s.set(a, 0xF0F0F0F1)
    s.shr(b, a, 4)
    s.and_(b, b, s.const(0xFFFF))
    s.shr(a, a, 36)
    s.add(b, b, a)
    s.halt(b)
    return s


def mod_basic():
    s = Stream()
    a, b = s.regs(2)
    s.set(a, 1000)
    s.mod(b, a, s.const(7))
    s.event(b)
    s.mod(b, a, s.const(1001))
    s.halt(b)
    return s


def rand_sequence():
    s = Stream(seed=1)
    a = s.regs(1)[0]
    for _ in range(5):
        s.rand(a)
        s.event(a)
    s.halt(a)
    return s


def issue_loop_depth1():
    s = Stream()
    i = s.regs(1)[0]
    top = s.label()
    s.issue(SEG_A, team=(0, 8), args=[i, 5])
    s.wait(0)
    s.addi(i, i, 1)
    s.jlt(i, s.const(12), top)
    return _halt_with(s, 0)


def issue_computed_team():
    s = Stream(seed=9)
    i, first, count = s.regs(3)
    top = s.label()
    s.rand(count)
    s.mod(count, count, s.const(60))
    s.addi(count, count, 1)
    s.rand(first)
    s.mod(first, first, s.const(59))
    s.issue(SEG_A, team=(first, count), args=[i, count, first])
    s.wait(0)
    s.addi(i, i, 1)
    s.jlt(i, s.const(20), top)
    return _halt_with(s, 3)


def issue_pipeline():
    s = Stream()
    i = s.regs(1)[0]
    top = s.label()
    s.issue(SEG_A, team=(0, 4), args=[i])
    s.wait(3)
    s.addi(i, i, 1)
    s.jlt(i, s.const(30), top)
    s.wait(0)
    return _halt_with(s, 0)


def issue_slots_full():
    s = Stream()
    i = s.regs(1)[0]
    top = s.label()
    s.issue(SEG_B, team=(2, 3), args=[i])
    s.addi(i, i, 1)
    s.jlt(i, s.const(40), top)
    return _halt_with(s, 40)


def issue_excl():
    s = Stream()
    i = s.regs(1)[0]
    top = s.label()
    s.issue(SEG_A, team=(0, 10), args=[i], exclusive=True)
    s.issue(SEG_A, team=(5, 10), args=[i, 1], exclusive=True)
    s.issue(SEG_A, team=(20, 10), args=[i, 2])
    s.addi(i, i, 1)
    s.jlt(i, s.const(5), top)
    s.wait(0)
    return _halt_with(s, 0)


def result_chain():
    s = Stream()
    n, r = s.regs(2)
    s.set(n, 3)
    top = s.label()
    s.issue(SEG_A, team=(0, n), args=[n], result=True)
    s.wait(0)
    s.result(r)
    s.event(r)
    s.mov(n, r)
    s.jlt(n, s.const(20), top)
    s.halt(n)
    return s


def submit_random_teams():
    s = Stream(seed=12345)
    i, count = s.regs(2)
    top = s.label()
    s.rand(count)
    s.mod(count, count, s.const(119))
    s.addi(count, count, 1)
    s.submit(SEG_A, count=count, args=[i, 99])
    s.addi(i, i, 1)
    s.jlt(i, s.const(100), top)
    s.wait(0)
    return _halt_with(s, 0)


def submit_large_job_passed_over():
    s = Stream()
    i = s.regs(1)[0]
    s.submit(SEG_A, count=1, args=[1000])
    s.submit(SEG_B, count=119, args=[2000])
    top = s.label()
    s.submit(SEG_A, count=1, args=[i])
    s.addi(i, i, 1)
    s.jlt(i, s.const(300), top)
    s.wait(0)
    return _halt_with(s, 0)


def submit_queue_full():
    s = Stream()
    i = s.regs(1)[0]
    top = s.label()
    s.submit(SEG_A, count=119, args=[i])
    s.event(i)
    s.addi(i, i, 1)
    s.jlt(i, s.const(70), top)
    s.wait(0)
    return _halt_with(s, 0)


def event_progress():
    s = Stream()
    i = s.regs(1)[0]
    top = s.label()
    s.addi(i, i, 1)
    s.event(i)
    s.jlt(i, s.const(5), top)
    return _halt_with(s, 0x12345678)


def halt_waits_for_records():
    s = Stream()
    s.issue(SEG_A, team=(0, 1))
    s.issue(SEG_A, team=(1, 1))
    return _halt_with(s, 7)


def jump_forward_label():
    s = Stream()
    a, b = s.regs(2)
    end, skip = s.forward(), s.forward()
    s.set(a, 0)
    s.jnz(a, end)
    s.set(b, 5)
    s.jeq(a, a, skip)
    s.set(b, 6)
    s.bind(skip)
    s.jmp(end)
    s.set(b, 7)
    s.bind(end)
    s.event(b)
    s.halt(b)
    return s


def args_patched():
    s = Stream()
    a, b, c, d = s.regs(4)
    s.set(a, 11)
    s.set(b, 22)
    s.set(c, 33)
    s.set(d, 44)
    s.issue(SEG_A, team=(3, 2), args=[1, a, 2, b, c, 3, 4, 5, d, 6], result=True)
    s.wait(0)
    return _halt_with(s, 0)


def time_elapsed():
    s = Stream()
    t0, t1 = s.regs(2)
    moved = s.forward()
    s.time(t0)
    s.issue(SEG_A, team=(0, 1))
    s.wait(0)
    s.time(t1)
    s.sub(t1, t1, t0)
    s.jnz(t1, moved)
    s.halt(s.const(0))
    s.bind(moved)
    s.halt(s.const(1))
    return s


def step_pass_limit():
    s = Stream()
    a = s.regs(1)[0]
    for _ in range(130):
        s.addi(a, a, 1)
    s.halt(a)
    return s


def _poke(data, offset, fmt, *values):
    out = bytearray(data)
    struct.pack_into(fmt, out, offset, *values)
    return bytes(out)


def _tiny():
    s = Stream()
    s.issue(SEG_A, team=(0, 1))
    s.wait(0)
    return _halt_with(s, 0)


def _round(n):
    return n + st.LINE - 1 & -st.LINE


def _count(data, word):
    return struct.unpack_from("<I", data, word * 4)[0]


def _op_at(index):
    return st.HEADER_BYTES + index * st.UOP_BYTES


def err_bad_magic():
    return _poke(_tiny().encode(), 0, "<I", 0x12345678)


def err_seed_zero():
    return _poke(_tiny().encode(), 28, "<I", 0)


def err_truncated():
    return _tiny().encode()[:-64]


def err_bad_opcode():
    return _poke(_tiny().encode(), _op_at(1), "<B", 99)


def err_bad_jump():
    s = Stream()
    a = s.regs(1)[0]
    s.set(a, 1)
    s.jnz(a, s.label())
    s.halt(a)
    return _poke(s.encode(), _op_at(1) + 4, "<I", 3)


def err_bad_register():
    s = Stream()
    a = s.regs(1)[0]
    s.set(a, 1)
    s.halt(a)
    return _poke(s.encode(), _op_at(1) + 1, "<B", 16)


def err_bad_template():
    data = _tiny().encode()
    at = st.HEADER_BYTES + _round(st.UOP_BYTES * _count(data, 2))
    return _poke(data, at + 4, "<I", 250)


def err_step_limit():
    s = Stream(max_steps=10)
    a = s.regs(1)[0]
    top = s.label()
    s.addi(a, a, 1)
    s.jmp(top)
    return s


def err_mod_zero():
    s = Stream()
    a, b = s.regs(2)
    s.set(a, 5)
    s.mod(b, a, s.const(0))
    return _halt_with(s, 0)


def err_deadline():
    s = Stream()
    s.issue(SEG_A, team=(0, 4))
    s.wait(0)
    return _halt_with(s, 0)


def err_never_fits():
    s = Stream()
    s.submit(SEG_A, count=120)
    return _halt_with(s, 0)


def err_cancelled():
    s = Stream()
    top = s.label()
    s.issue(SEG_A, team=(0, 2))
    s.wait(0)
    s.jmp(top)
    return s


def err_team_outside():
    s = Stream()
    s.issue(SEG_A, team=(100, 20))
    return _halt_with(s, 0)


def err_run_off_end():
    s = Stream()
    a = s.regs(1)[0]
    s.set(a, 1)
    s.addi(a, a, 1)
    return s


# name -> (builder, parameters of the simulation, what it shows)
REFERENCES = {
    "arith_wrap": (arith_wrap, {}, "ADD, SUB, MUL and ADDI wrap at 32 bits"),
    "shifts_and_mask": (shifts_and_mask, {}, "SHR masks its amount to 5 bits; AND"),
    "mod_basic": (mod_basic, {}, "MOD with a nonzero divisor"),
    "rand_sequence": (rand_sequence, {}, "five steps of the generator from seed 1"),
    "issue_loop_depth1": (issue_loop_depth1, {"delay": 1}, "one record in flight"),
    "issue_computed_team": (issue_computed_team, {}, "teams from RAND and MOD"),
    "issue_pipeline": (issue_pipeline, {"delay": 3}, "at most four records in flight"),
    "issue_slots_full": (
        issue_slots_full,
        {"delay": 5},
        "all 16 slots busy: ISSUE waits",
    ),
    "issue_excl": (issue_excl, {"delay": 2}, "KM_F_EXCL waits for an overlapping team"),
    "result_chain": (result_chain, {"delay": 1}, "RESULT feeds the next team size"),
    "submit_random_teams": (
        submit_random_teams,
        {"delay": {"base": 2, "mod": 4}},
        "jobs of random size through the queue",
    ),
    "submit_large_job_passed_over": (
        submit_large_job_passed_over,
        {"delay": {"base": 3, "mod": 5}},
        "a 119-core job behind small jobs",
    ),
    "submit_queue_full": (
        submit_queue_full,
        {"delay": 4},
        "SUBMIT waits on a full queue",
    ),
    "event_progress": (event_progress, {}, "EVENT posts a progress event"),
    "halt_waits_for_records": (
        halt_waits_for_records,
        {"delay": 6},
        "HALT ends only when every record is done",
    ),
    "jump_forward_label": (jump_forward_label, {}, "forward jumps, JNZ, JEQ, JMP"),
    "args_patched": (args_patched, {}, "four registers patched into arguments"),
    "time_elapsed": (time_elapsed, {"delay": 3}, "TIME advances while a record runs"),
    "step_pass_limit": (step_pass_limit, {}, "64 micro-ops per pass"),
    "err_bad_magic": (err_bad_magic, {}, "reason 1 at pc 0"),
    "err_seed_zero": (err_seed_zero, {}, "reason 1: a zero seed"),
    "err_truncated": (err_truncated, {}, "reason 1: size does not match the header"),
    "err_bad_opcode": (err_bad_opcode, {}, "reason 2 at pc 1"),
    "err_bad_jump": (err_bad_jump, {}, "reason 3 at pc 1"),
    "err_bad_register": (err_bad_register, {}, "reason 4 at pc 1"),
    "err_bad_template": (err_bad_template, {}, "reason 5 at the issuing pc"),
    "err_step_limit": (err_step_limit, {}, "reason 6 after 10 micro-ops"),
    "err_mod_zero": (err_mod_zero, {}, "reason 7 at the MOD"),
    "err_deadline": (
        err_deadline,
        {"delay": None, "deadline_passes": 5},
        "reason 8 at the ISSUE, grid stopped",
    ),
    "err_never_fits": (err_never_fits, {}, "reason 9: a 120-core job on 119 cores"),
    "err_cancelled": (
        err_cancelled,
        {"delay": 2, "cancel_pass": 5},
        "reason 10 at the pc reached by pass 5",
    ),
    "err_team_outside": (err_team_outside, {}, "reason 11: cores 100 to 119"),
    "err_run_off_end": (err_run_off_end, {}, "reason 3 at pc 2 = n_ops"),
}


REFERENCES.update(MORE)


def build(name):
    """(upload bytes, parameters, description) of a reference."""
    builder, params, text = REFERENCES[name]
    made = builder()
    return (made.encode() if isinstance(made, Stream) else made), params, text


def document(name):
    upload, params, text = build(name)
    full = {**DEFAULTS, **params}
    doc = {"name": name, "description": text, "notes": NOTES, "params": full}
    return upload, {**doc, "expect": expected(upload, full)}


def dump(doc):
    """JSON text of a reference, with one record or event per line."""
    expect = doc["expect"]
    top = [
        f" {json.dumps(k)}: {json.dumps(v)}" for k, v in doc.items() if k != "expect"
    ]
    rest = [f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in expect.items()]
    rest = [r for r in rest if not r.startswith(('  "records"', '  "events"'))]
    for key in ("records", "events"):
        rows = ",\n".join(f"   {json.dumps(row)}" for row in expect[key])
        rest.append(f'  "{key}": [\n{rows}\n  ]' if rows else f'  "{key}": []')
    return (
        "{\n" + ",\n".join(top) + ',\n "expect": {\n' + ",\n".join(rest) + "\n }\n}\n"
    )


def write():
    DATA.mkdir(parents=True, exist_ok=True)
    for name in REFERENCES:
        upload, doc = document(name)
        (DATA / f"{name}.bin").write_bytes(upload)
        (DATA / f"{name}.json").write_text(dump(doc))


def differing():
    """The names of the references whose saved files differ from the generator's."""
    out = []
    for name in REFERENCES:
        upload, doc = document(name)
        same = (DATA / f"{name}.bin").read_bytes() == upload
        same = same and (DATA / f"{name}.json").read_text() == dump(doc)
        if not same:
            out.append(name)
    return out


if __name__ == "__main__":
    if "--check" in sys.argv:
        bad = differing()
        print(f"{len(bad)} of {len(REFERENCES)} streams differ: {bad}")
        sys.exit(bool(bad))
    write()
    print(f"wrote {len(REFERENCES)} streams to {DATA}", file=sys.stderr)
