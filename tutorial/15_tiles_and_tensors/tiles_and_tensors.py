"""Tiles and tensors: make a tensor on the card from an array, look at its pages."""

import numpy as np

import komorebi as km

with km.Session() as session:
    # --8<-- [start:make]
    numbers = np.arange(64 * 96, dtype=np.uint32).reshape(64, 96)
    tensor = session.tensor(numbers, "uint32", "tile")
    print(f"shape {tensor.shape}, {tensor.pages} tiles of {tensor.page_bytes} bytes")
    # --8<-- [end:make]

    # --8<-- [start:read]
    assert (tensor.read() == numbers).all()
    (pages,) = session.read(tensor)
    print("the first number of each tile:", pages.view(np.uint32)[:, 0].tolist())
    # --8<-- [end:read]

    # --8<-- [start:padding]
    small = session.tensor(np.ones((40, 40), np.uint32), "uint32", "tile")
    print(f"a 40 x 40 array is stored as shape {small.shape}, {small.pages} tiles")
    # --8<-- [end:padding]
