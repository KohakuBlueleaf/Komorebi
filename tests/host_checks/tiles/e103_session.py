"""Padding through a session, one card or two (argv: 0 or 0,1; run under `kmr
prebuild --cards ...` off the card): session.tensor's and session.empty's marks, a
submit refused where the padding a handler wrote enters a sum; an
empty 32 x 40 tensor held as 32 x 64 on either (written, read back padded)."""

import sys

import e103_classes as c
import numpy as np

import komorebi as km

cards = [int(k) for k in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0]
opened = {"cards": cards} if len(cards) > 1 else {}
with km.Session(log=lambda line: None, **opened) as session:
    t = session.empty((32, 40), "float32", "tile")
    assert (t.shape, t.logical, t.padding) == ((32, 64), (32, 40), "dirty"), t.shape
    t.write(np.ones((32, 40), np.float32))
    assert t.padding == "clean" and t.read().shape == (32, 64)
    print(
        f"{type(session).__name__}: empty (32, 40) is {t.shape}, written, read padded"
    )
    xs = session.tensor(np.zeros((32, 40), np.float32), "float32", "tile")
    y = session.empty((32, 40), "float32", "tile")
    out = session.empty((32, 40), "float32", "tile")
    program = c.Two(log=lambda line: None, session=session, x=xs, y=y, out=out)
    run = session.load(program)
    try:
        run.submit(c.Two.first, (0, 1), args=(xs, y))  # y marked at its submit
        try:
            run.submit(c.Two.second, (0, 1), args=(y, out))
            raise SystemExit("the second submit was not refused")
        except km.ProgramError as error:
            assert "written by first" in str(error), str(error)
            print(f"second refused: {str(error)[:90]}")
    finally:
        run.stop()
print("OK")
