"""What the card looks like to a program: the grid, the NoC coordinates, the L1."""

import komorebi as km

# --8<-- [start:facts]
with km.Session() as session:
    gx, gy = session.grid  # Tensix cores per row, rows
    nocs = session.nocs()  # where each core sits on the NoC
    kernel_bytes, buffer_bytes = session.l1()  # L1: the kernels' share, the rest
    print(f"{gx * gy} Tensix cores: {gx} per row, {gy} rows")
    print(f"panels (first and last column): {nocs.panels}")
    print(
        f"core 0 at NoC0 {nocs.noc0[(0, 0)]}, core 119 at {nocs.noc0[(gx - 1, gy - 1)]}"
    )
    print(f"L1 of a core: {kernel_bytes} bytes for kernels, {buffer_bytes} for buffers")
# --8<-- [end:facts]
