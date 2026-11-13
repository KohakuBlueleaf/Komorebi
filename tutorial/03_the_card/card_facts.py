"""The card as a program sees it: the cores, their layout and their memory."""

import komorebi as km

with km.Session() as session:
    # --8<-- [start:grid]
    columns, rows = session.grid
    print(f"{columns * rows} Tensix cores: {columns} in a row, {rows} rows")
    # --8<-- [end:grid]
    # --8<-- [start:panels]
    (left_first, left_last), (right_first, right_last) = session.nocs().panels
    print(f"panels: columns {left_first}-{left_last} and {right_first}-{right_last}")
    # --8<-- [end:panels]
    # --8<-- [start:l1]
    _, bytes_for_data = session.l1()
    print(f"each core has {bytes_for_data:,} bytes of L1 for data")
    # --8<-- [end:l1]
