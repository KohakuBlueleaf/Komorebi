"""Today's matrix-unit words of a tile product (matrix.py) where the corpus does not
reach them: each fidelity, products of 1 to 3 tiles each way (`tile_products`) and
Wormhole's image (`wormhole_products`), a word a line. Usage: <this> <folder>."""

import sys
from pathlib import Path

from komorebi.dsl.tile.matrix import PHASES, tile_products, wormhole_products


def lines(pairs):
    """Each word as `op block | fields ; note`."""
    out = []
    for (op, block), insn in pairs:
        fields = ",".join(f"{k}={v}" for k, v in insn.args.items())
        out.append(f"{op} {block} | {insn.op} {fields} ; {insn.note}")
    return out


def main(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for fidelity in [*PHASES, "HiFi5"]:
        for ct, rt, dst in ((1, 1, 0), (1, 1, 2), (2, 1, 0), (1, 2, 1), (3, 2, 0)):
            out.append(f"TILE {fidelity} {ct} {rt} {dst}")
            try:
                out += lines(tile_products(fidelity, ct, rt, dst))
            except ValueError as error:
                out.append(f"refused: {error}")
        for dst in (0, 3):
            out.append(f"WORMHOLE {fidelity} {dst}")
            try:
                out += lines(wormhole_products(fidelity, dst))
            except ValueError as error:
                out.append(f"refused: {error}")
    (folder / "matrix.out").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
