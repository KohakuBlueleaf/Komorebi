"""The smallest Komorebi program: every core gets one command and reports it done."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

program = km.Program(
    "hello",
    kernels=HERE / "kernels",
    segments=HERE / "kernels" / "handlers.h",
)

with km.Session() as session:
    run = session.load(program)
    handle = run.issue("hello")
    event = run.wait(handle, timeout=10.0)
    run.stop()
    print(f"cores: {session.grid[0] * session.grid[1]}, event: {event.kind}")
