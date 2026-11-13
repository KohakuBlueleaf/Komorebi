"""Placing groups: a row, two rows, a panel, and a rectangle made of several groups."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

program = km.Program(
    "placing_groups",
    kernels=HERE / "kernels",
    handlers=HERE / "kernels" / "handlers.h",
)

with km.Session() as session:
    run = session.load(program)

    # --8<-- [start:groups]
    CORES_PER_ROW = 12
    placements = {
        "row 2": [(2 * CORES_PER_ROW, CORES_PER_ROW)],
        "rows 2 and 3": [(2 * CORES_PER_ROW, 2 * CORES_PER_ROW)],
        "left panel of row 2": [(2 * CORES_PER_ROW, 6)],
        "right panel of row 2": [(2 * CORES_PER_ROW + 6, 6)],
        "right panel of rows 2 to 4": [
            (row * CORES_PER_ROW + 6, 6) for row in (2, 3, 4)
        ],
    }
    for name, groups in placements.items():
        handles = [run.submit("where", group=group, result=True) for group in groups]
        events = run.wait(handles, timeout=10.0)
        leaders = [event.result for event in events]
        print(f"{name:27} {len(groups)} group(s), leader cores {leaders}")
        assert leaders == [first for first, _ in groups]
    # --8<-- [end:groups]
    run.stop()
