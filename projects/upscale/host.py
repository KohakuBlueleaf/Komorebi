"""Checks A and B on the host (README.md, "Running it"): each frame's 2 x 2 box average
through the float network (float32 and int8-trained weights), the card's emulation and
bicubic, against the picture. Prints one JSON line; --out writes the PNGs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from projects.upscale import model  # noqa: E402
from reference import png  # noqa: E402


def frame(path):
    """A picture's display values in [0, 1]: the target, and its 2 x 2 box average."""
    target = png.read_png(path).astype(np.float32) / 255
    return target, model.box_down(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights")
    parser.add_argument("pictures", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args()
    w32 = model.load(Path(args.weights) / "float32.npz")
    w8 = model.load(Path(args.weights) / "int8.npz")
    rows = []
    for path in args.pictures:
        target, low = frame(path)
        net = model.forward(low, w32)
        net8 = model.forward(low, w8)
        emu = model.emulate(low, w32)
        cubic = model.bicubic2(low)
        diff8 = np.abs(model.to8(emu).astype(int) - model.to8(net).astype(int))
        rows.append(
            {
                "picture": Path(path).name,
                "psnr_float32": model.psnr(net, target),
                "psnr_int8_trained": model.psnr(net8, target),
                "psnr_bicubic": model.psnr(cubic, target),
                "psnr_emulation": model.psnr(emu, target),
                "psnr_emulation_vs_float": model.psnr(emu, net),
                "psnr_float32_vs_bicubic": model.psnr(net, cubic),
                "emulation_8bit_most": int(diff8.max()),
                "emulation_8bit_share_over_1": float((diff8 > 1).mean()),
                "emulation_float_most": float(np.abs(emu - net).max()),
            }
        )
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            stem = Path(path).stem
            png.write_png(out / f"{stem}_input.png", model.to8(low))
            png.write_png(out / f"{stem}_target.png", model.to8(target))
            png.write_png(out / f"{stem}_bicubic.png", model.to8(cubic))
            png.write_png(out / f"{stem}_host.png", model.to8(net))
    print(json.dumps({"frames": rows}))


if __name__ == "__main__":
    main()
