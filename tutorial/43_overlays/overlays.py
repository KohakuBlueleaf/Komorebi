"""Overlays: two pieces of work that are not in the kernels, loaded into one slot of
the writer in turn while the kernels run."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

# --8<-- [start:program]
program = km.Program(
    "overlays",
    kernels=HERE / "kernels",
    handlers=HERE / "kernels" / "handlers.h",
)
program.overlay("add", HERE / "kernels" / "add")  # add/writer.cpp, kOverlayAdd
program.overlay("multiply", HERE / "kernels" / "multiply")
program.slots(count=1)  # one slot: each load replaces the overlay held
# --8<-- [end:program]

with km.Session() as session:
    run = session.load(program)
    # --8<-- [start:submit]
    for name in ("add", "multiply", "add"):
        number = program.overlays[name].number
        handle = run.submit("apply", group=(0, 1), args=[number, 6, 7], result=True)
        answer = run.wait(handle, timeout=10.0).result
        print(f"6 {name:8} 7 = {answer} (overlay {number})")
    # --8<-- [end:submit]
    run.stop()
