"""The card as a program sees it: the cores, their layout and their memory."""

import komorebi as km

with km.Session() as session:
    # --8<-- [start:grid]
    columns, rows = session.grid
    print(f"{columns * rows} Tensix cores: {columns} in a row, {rows} rows")
    # --8<-- [end:grid]
    # --8<-- [start:panels]
    panels = [f"columns {first}-{last}" for first, last in session.nocs().panels]
    print("panels: " + " and ".join(panels))
    # --8<-- [end:panels]
    # --8<-- [start:l1]
    _, bytes_for_data = session.l1()
    print(f"each core has {bytes_for_data:,} bytes of L1 for data")
    # --8<-- [end:l1]
