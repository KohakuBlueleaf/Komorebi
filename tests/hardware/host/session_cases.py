"""Card cases of the host package that the example does not cover. One case per run;
prints one JSON line with `ok`.

    python session_cases.py compile-error | exclusive
"""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import komorebi as km

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
    out = {}
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
    out["run_after_error"] = session.run is not None
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


CASES = {"compile-error": case_compile_error, "exclusive": case_exclusive}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("case", choices=sorted(CASES))
    args = p.parse_args()
    with km.Session(log=log) as session:
        out = {"case": args.case, **CASES[args.case](session)}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
