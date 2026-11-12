"""Card cases of the host package that the example does not cover. One case per run;
prints one JSON line with `ok`.

    python session_cases.py compile-error | exclusive | read | start-error
"""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

import komorebi as km
from komorebi import session as km_session

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "01_mapcollect"
T0 = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def program(kernels):
    """The example's kernels with no tensors: only NOP and STALL records are sent."""
    return km.Program(
        "mapcollect",
        kernels=kernels,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
        ],
        compile_args={"reader": [2, 2048], "writer": [2, 2048]},
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        segments=Path(kernels) / "mapcollect.h",
    )


def case_compile_error(session):
    """A compute kernel that fails to compile: ProgramError, no run loaded; then the
    good program loads and runs in the same session (no image was left running)."""
    out, before = {}, session.run
    with tempfile.TemporaryDirectory() as tmp:
        broken = Path(tmp) / "kernels"
        shutil.copytree(EXAMPLE / "kernels", broken)
        compute = broken / "compute.cpp"
        compute.write_text(compute.read_text() + "\n#error deliberate compile error\n")
        try:
            session.load(program(broken))
            out["raised"] = None
        except km.ProgramError as err:
            out["raised"] = type(err).__name__
            out["names_error"] = "deliberate compile error" in str(err)
    out["run_after_error"] = session.run is not before
    run = session.load(program(EXAMPLE / "kernels"))
    out["nop_after"] = run.wait(run.issue(0, (0, 8))).kind
    run.stop()
    out["ok"] = (
        out["raised"] == "ProgramError"
        and out["names_error"]
        and not out["run_after_error"]
        and out["nop_after"] == "done"
        and run.result()["error"] == 0
    )
    return out


def case_exclusive(session):
    """A STALL-free check of `exclusive`: a record overlapping one in flight is held
    until that one completed (trace: its members took it after the leader pushed)."""
    run = session.load(program(EXAMPLE / "kernels"), trace=64)
    pairs = []
    for k in range(20):
        a = run.issue(0, (0, 60))
        b = run.issue(0, (30, 60), exclusive=k % 2 == 1)
        pairs.append((a, b, k % 2 == 1))
    run.drain()
    run.stop()
    trace = run.trace()
    leads = [(excl, trace.lead(a, b)) for a, b, excl in pairs]
    out = {
        "exclusive_leads_us": [round(v, 3) for e, v in leads if e],
        "plain_leads_us": [round(v, 3) for e, v in leads if not e],
    }
    out["ok"] = all(v < 0 for e, v in leads if e)
    return out


def bf16_tiles(device, x):
    return ttnn.from_torch(
        torch.from_numpy(x), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


def case_read(session):
    """Tensors read by banks equal what was written and what ttnn reads (an int32
    row-major one, two bfloat16 tile ones, one barrier), also while a run is loaded."""
    rng = np.random.default_rng(3)
    words = rng.integers(0, 1 << 32, (37, 24), dtype=np.uint64).astype(np.uint32)
    x = rng.integers(-4, 5, (96, 64)).astype(np.float32)  # exact in bfloat16
    w = km_session._int32_tensor(session.device, words)
    a, b = bf16_tiles(session.device, x), bf16_tiles(session.device, 2 * x)
    raw, got_a, got_b = session.read(w)[0], *session.read_bf16(a, b)
    out = {
        "words": bool((raw.reshape(-1).view(np.uint32) == words.reshape(-1)).all()),
        "tiles": bool((got_a == x).all() and (got_b == 2 * x).all()),
        "as_ttnn": bool((got_a == ttnn.to_torch(a).float().numpy()).all()),
    }
    run = session.load(program(EXAMPLE / "kernels"))
    run.wait(run.issue(0, (0, 8)))
    log("read while the run is loaded")
    (during,) = session.read_bf16(a)
    out["during_run"] = bool((during == x).all())
    run.stop()
    out["ok"] = all(out[k] for k in ("words", "tiles", "as_ttnn"))
    out["ok"] = out["ok"] and out["during_run"] and run.result()["error"] == 0
    return out


def case_start_error(session):
    """A record deadline under one timer tick: the kernel refuses its arguments before
    it opens a window. StartError with its result; the host stops the grid (every
    reader gave up on the stop word, not its idle limit); then a run works."""
    out = {}
    t0 = time.monotonic()
    try:
        session.load(program(EXAMPLE / "kernels"), deadline=1e-9, idle=60.0)
        out["raised"] = None
    except km.StartError as err:
        out["raised"] = type(err).__name__
        out["result"] = err.result is not None
        out["message"] = str(err)[:200]
    out["start_error_s"] = round(time.monotonic() - t0, 2)
    failed = session.run
    readers = {}
    for c in failed.state():
        r = c["reader"]
        key = f"{r['code']} in {r['where']}" if r else "none"
        readers[key] = readers.get(key, 0) + 1
    out["readers"] = readers
    run = session.load(program(EXAMPLE / "kernels"))
    out["nop_after"] = run.wait(run.issue(0, None)).kind
    run.stop()
    out["ok"] = (
        out["raised"] == "StartError"
        and out["result"]
        and readers.get("stop in next", 0) == 120
        and out["nop_after"] == "done"
        and run.result()["error"] == 0
    )
    return out


CASES = {
    "compile-error": case_compile_error,
    "exclusive": case_exclusive,
    "read": case_read,
    "start-error": case_start_error,
}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("case", choices=sorted(CASES))
    args = p.parse_args()
    with km.Session(log=log) as session:
        out = {"case": args.case, **CASES[args.case](session)}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
