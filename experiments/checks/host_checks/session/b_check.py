"""(b) own check: Run.reset_core with a timeout under 1 ms on a stand-in run whose
native reset_core records the call: refused before it, or passed to it."""

import sys
import types

sys.path.insert(0, sys.argv[1])
from komorebi.session.run import Run

calls = []
stand = types.SimpleNamespace(
    cores=16,
    grid=(4, 4),
    recovery=None,
    backend=types.SimpleNamespace(
        launch=types.SimpleNamespace(
            reset_core=lambda *a: calls.append(a) or (0.0, -1.0, 0.0)
        )
    ),
)
reached = []
for timeout in (0.0005, 0.0, 0.001):
    calls.clear()
    try:
        Run.reset_core(stand, 3, timeout)
        outcome = "returned"
    except Exception as error:  # noqa: BLE001
        outcome = f"{type(error).__name__}: {error}"
    reached.append(bool(calls))
    print(f"timeout {timeout}: native calls {[c[2] for c in calls]}; {outcome}")
WANTED = [False, False, True]  # under 1 ms refused before the native call
print("PASS" if reached == WANTED else "FAIL")
sys.exit(0 if reached == WANTED else 1)
