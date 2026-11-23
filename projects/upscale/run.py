"""QuickSRNet-Medium 2x on the card: each frame's 2 x 2 box average through the Upscale
class (the seven layers one chain command), checked against the host's emulation and
timed. Prints one JSON line; usage in README.md, "Running it"."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi import formats  # noqa: E402
from komorebi.bench import power  # noqa: E402
from komorebi.dsl.chain import chain_stamps  # noqa: E402
from projects.upscale import model  # noqa: E402
from projects.upscale import upscale as up  # noqa: E402
from reference import png  # noqa: E402

IMAGE = 512
CORES = -(-IMAGE // up.ROWS)  # 103: 5 rows a core (a chain's blocks are equal)
TENSOR_ROWS = CORES * up.ROWS + 2  # row 0 and the rows past the image: padding
OUT_PAGES = CORES * up.OUT_ROWS  # 3 colours of 2 output rows an image row
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", file=sys.stderr, flush=True)


def to_tensor(image):
    """An image (512, 512, C <= 32) as the activation tensor's array: tensor row R
    (32 array rows) holds image row R - 1, its tile t pixels 32 t .. 32 t + 31 (tile
    rows) by channels (tile columns)."""
    a = np.zeros((TENSOR_ROWS, 32, up.TILES, 32), np.float32)
    rows = image.reshape(IMAGE, up.TILES, 32, image.shape[2])  # R, t, p, c
    a[1 : IMAGE + 1, :, :, : image.shape[2]] = rows.transpose(0, 2, 1, 3)
    return a.reshape(TENSOR_ROWS * 32, up.TILES * 32)


def from_tensor(array, channels):
    """to_tensor's inverse: the image rows of an activation tensor's array."""
    a = np.asarray(array, np.float32).reshape(TENSOR_ROWS, 32, up.TILES, 32)
    return (
        a[1 : IMAGE + 1, :, :, :channels]
        .transpose(0, 2, 1, 3)
        .reshape(IMAGE, IMAGE, channels)
    )


def weight_array(weights):
    """The 84 weight tiles as the card multiplies them (each the turned product's
    left side: out by in): seven layers of ten, then the shuffle's P_jr and Q_hj."""
    tiles = [model.weight_tiles(w, b) for w, b in weights]
    p = np.zeros((2, up.ROWS, 32, 32), np.float32)
    for j in range(2):
        for r in range(up.ROWS):
            for i in range(2):
                for c in range(3):
                    p[j, r, 6 * r + 3 * i + c, 4 * c + 2 * i + j] = 1
    q = np.zeros((2, 2, 32, 32), np.float32)
    for h in range(2):
        for j in range(2):
            for x in range(j, 32, 2):
                q[h, j, x, 16 * h + x // 2] = 1
    out = np.concatenate(tiles + [p.reshape(-1, 32, 32), q.reshape(-1, 32, 32)])
    assert len(out) == up.WEIGHTS
    return out.reshape(-1, 32)


def frame_of(raw):
    """The result tensor's pages (row 3 Y + c: output row Y of colour c) as an image."""
    pages = np.asarray(raw).view(np.float32).reshape(-1, 2 * IMAGE)[: 6 * IMAGE]
    return pages.reshape(2 * IMAGE, 3, 2 * IMAGE).transpose(0, 2, 1).copy()


def compare(card, emulated):
    """8-bit differences of the card's frame against the emulation's."""
    d = np.abs(model.to8(card).astype(int) - model.to8(emulated).astype(int))
    return {
        "share_over_1": float((d > 1).mean()),
        "share_nonzero": float((d > 0).mean()),
        "most": int(d.max()),
        "float_most": float(np.abs(card - emulated).max()),
    }


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("weights", help="folder of float32.npz (README.md)")
    parser.add_argument("pictures", nargs="+", help="1,024 x 1,024 PNG pictures")
    parser.add_argument("--out", default=None, help="a folder for the PNG outputs")
    parser.add_argument("--calls", type=int, default=32)
    parser.add_argument("--layer", action="store_true", help="layer 1 alone first")
    parser.add_argument("--stamps", action="store_true", help="the chain's stamps")
    return parser.parse_args()


def main():
    args = parse()
    weights = model.load(Path(args.weights) / "float32.npz")
    report = {"cores": CORES, "frames": []}
    index = power.visible_card()
    with km.Session(log=log) as session:
        frame = session.tensor(np.zeros((TENSOR_ROWS * 32, IMAGE)), "bfloat16", "tile")
        a = session.tensor(np.zeros((TENSOR_ROWS * 32, IMAGE)), "bfloat16", "tile")
        b = session.tensor(np.zeros((TENSOR_ROWS * 32, IMAGE)), "bfloat16", "tile")
        w = session.tensor(weight_array(weights), "bfp8_b", "tile")
        ones = np.zeros((32, 32), np.float32)
        ones[:, 0] = 1
        o = session.tensor(ones, "bfloat16", "tile")
        result = session.empty((OUT_PAGES, 2 * IMAGE), "float32")
        options = {"chain_stamps": True} if args.stamps else {}
        stamps = chain_stamps.tensor_for(session, CORES) if args.stamps else None
        program = Upscale(
            x=frame, y=a, w=w, o=o, result=result, session=session, log=log, **options
        )
        run = session.load(program)
        event = run.wait(run.submit(Upscale.load, (0, CORES), args=(w, o)))
        if event.kind != "done":
            raise km.KomorebiError(f"load: {event}")
        chain = run.chain(program.forward(frame, a, b, result), cores=(0, CORES))
        for path in args.pictures:
            target = png.read_png(path).astype(np.float32) / 255
            low = model.box_down(target)
            keep = []
            emulated = model.emulate(low, weights, keep)
            raw, _ = formats.to_bytes(to_tensor(low), "bfloat16", "tile")
            session.write(frame, raw.reshape(-1, 2048))
            row = {"picture": Path(path).name}
            if args.layer:
                event = run.wait(
                    run.submit(Upscale.conv, (0, CORES), args=(frame, a, 0)),
                    timeout=60.0,
                )
                if event.kind != "done":
                    raise km.KomorebiError(f"conv: {event}")
                got = from_tensor(session.read_bf16(a)[0], 32)
                row["layer1"] = {
                    "most": float(np.abs(got - keep[0]).max()),
                    "share_differ": float((got != keep[0]).mean()),
                    "us": event.microseconds,
                }
                log(f"layer 1: {json.dumps(row['layer1'])}")
            event = run.wait(run.submit(chain), timeout=60.0)
            if event.kind != "done":
                raise km.KomorebiError(f"chain: {event}")
            card = frame_of(session.read(result)[0])
            row["check"] = compare(card, emulated)
            row["layers"] = []
            for k, tensor in ((4, a), (5, b)):  # layers 5 and 6: the chain's last two
                got = from_tensor(session.read_bf16(tensor)[0], 32)
                row["layers"].append(
                    {
                        "layer": k + 1,
                        "most": float(np.abs(got - keep[k]).max()),
                        "share_differ": float((got != keep[k]).mean()),
                    }
                )
            row["psnr"] = {
                "card": model.psnr(card, target),
                "emulation": model.psnr(emulated, target),
                "card_vs_emulation": model.psnr(card, emulated),
            }
            log(f"{row['picture']}: {json.dumps(row['check'] | row['psnr'])}")
            if args.out:
                out = Path(args.out)
                out.mkdir(parents=True, exist_ok=True)
                png.write_png(out / f"{Path(path).stem}_card.png", model.to8(card))
            report["frames"].append(row)
        card_us, wall_us = [], []
        before = power.aiclk_mhz(index)
        for _ in range(args.calls):
            started = time.perf_counter()
            event = run.wait(run.submit(chain), timeout=60.0)
            wall_us.append((time.perf_counter() - started) * 1e6)
            if event.kind != "done":
                raise km.KomorebiError(f"chain: {event}")
            card_us.append(event.microseconds)
        report["time"] = {
            "calls": args.calls,
            "card_us_median": float(np.median(card_us)),
            "card_us_least": float(np.min(card_us)),
            "wall_us_median": float(np.median(wall_us)),
            "mhz": [before, power.aiclk_mhz(index)],
        }
        log(f"time {json.dumps(report['time'])}")
        if args.stamps:
            rows = chain_stamps.read(session, run, program, stamps, (0, CORES))
            mhz = power.aiclk_mhz(index)
            report["stamps"] = [
                {"record": k, **terms}
                for k, terms in enumerate(chain_stamps.terms(rows, mhz, 7))
            ]
        run.stop()
        report["kernel"] = run.result()["error_name"]
    report["ok"] = report["kernel"] == "none"
    print(json.dumps(report, default=float), flush=True)
    sys.exit(0 if report["ok"] else 1)


Upscale = up.Upscale

if __name__ == "__main__":
    main()
