"""The mock run refuses a local=True command whose groups cross the islands' edge."""

import numpy as np
from local import Sevens

import komorebi as km

with km.Session(cards=[0, 1]) as session:
    edge = session.graph.first(1)
    s = session.tensor(
        np.zeros((session.ranks * 32, 32), np.float32), "float32", "tile"
    )
    t = session.tensor(
        np.zeros((session.ranks * 32, 32), np.float32), "float32", "tile"
    )
    run = session.load(Sevens(session=session, s=s, t=t), wait=0.5)
    try:
        run.submit(Sevens.sevens, (edge - 10, 21), (s, t, 0))
        print("CROSS: not refused")
    except ValueError as error:
        print(f"CROSS: refused: {str(error)[:150]}")
    run.stop()
