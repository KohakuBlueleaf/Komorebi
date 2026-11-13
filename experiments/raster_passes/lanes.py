"""Small triangles as vector lanes, on one core: tiles filled four ways (small ones by
rows or as lanes; all by rows, or the larger by rows and the small as lanes), each
against the reference, with the pack thread's cycles. One JSON line.
Example: python experiments/raster_passes/lanes.py --scene s1m --tiles 16"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from projects.raster import bench, card, lists, run  # noqa: E402
from reference import raster  # noqa: E402

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
LANES_H = KERNELS / "lanes.h"
L = {
    name: int(value, 0) for name, value in lists.CONSTEXPR.findall(LANES_H.read_text())
}
K = lists.K
PAGE_WORDS = L["kLanesPageBytes"] // 4
PROFILE = [
    "total",
    "init",
    "rows",
    "copy_in",
    "vector",
    "resolve",
    "out",
    "wait",
    "row_records",
    "row_count",
    "lane_pages",
    "offsets",
    "fragments",
    "wins",
    "window_chain",
    "window_loads",
    "window_stores",
    "l1_chain",
    "setup",
    "bits_wrong",
    "broadcast_write",
]
MODES = ["rows_small", "lanes_small", "rows_all", "mixed"]


def program(defines):
    """The probe's three kernels; one command in flight."""
    return km.Program(
        "raster_lanes",
        kernels={
            role: KERNELS / f"lanes_{role}.cpp"
            for role in ("reader", "compute", "writer")
        },
        buffers=[
            km.Buffer(
                L["kLanesCbIn"], "uint32", L["kLanesPageBytes"], L["kLanesInPages"]
            ),
            km.Buffer(
                L["kLanesCbOut"], "uint32", L["kLanesPageBytes"], L["kLanesOutPages"]
            ),
        ],
        compute={
            "math_fidelity": "HiFi4",
            "fp32_dest_acc_en": True,
            "enable_trisc2_rvv": True,
        },
        depth=1,
        defines=defines,
        handlers={"tile": L["kHandlerTile"]},
    )


def clipped_boxes(records, members, tile):
    """Box of each member clipped to the tile, in the tile: x0, x1, y0, y1."""
    ty, tx = divmod(tile, raster.TILES_X)
    box = records["box"][members].astype(np.int64)
    return np.stack(
        [
            np.maximum(box[:, 0] - 32 * tx, 0),
            np.minimum(box[:, 1] - 32 * tx, 31),
            np.maximum(box[:, 2] - 32 * ty, 0),
            np.minimum(box[:, 3] - 32 * ty, 31),
        ],
        axis=1,
    )


def lane_pages(packed, records, members, tile):
    """The lane pages of `members` (each box at most 4 x 4 in the tile), 32 a page in
    the order of their shapes."""
    ty, tx = divmod(tile, raster.TILES_X)
    clip = clipped_boxes(records, members, tile)
    columns, rows = clip[:, 1] - clip[:, 0] + 1, clip[:, 3] - clip[:, 2] + 1
    order = np.lexsort((columns, rows))
    words = np.zeros((len(members), 32), dtype=np.uint32)
    words[:, :24] = packed[members, :24]
    origin_x, origin_y = np.float32(32 * tx), np.float32(32 * ty)
    clip_float = clip.astype(np.float32)
    words[:, L["kLanePx"]] = lists.bits(origin_x + clip_float[:, 0] + np.float32(0.5))
    words[:, L["kLanePxEnd"]] = lists.bits(origin_x + clip_float[:, 1] + np.float32(1))
    words[:, L["kLanePy"]] = lists.bits(origin_y + clip_float[:, 2] + np.float32(0.5))
    words[:, L["kLanePyEnd"]] = lists.bits(origin_y + clip_float[:, 3] + np.float32(1))
    words[:, L["kLaneTri"]] = packed[members, K["kRecTri"]]
    words[:, L["kLaneCorner"]] = clip[:, 0] | clip[:, 2] << 8
    pages = []
    slot = np.arange(32)
    for start in range(0, len(members), 32):
        chosen = order[start : start + 32]
        page = np.zeros(PAGE_WORDS, dtype=np.uint32)
        for lane, index in enumerate(chosen):
            page[64 * (slot >> 1) + 2 * lane + (slot & 1)] = words[index]
        page[L["kLaneShape"]] = columns[chosen].max() | rows[chosen].max() << 4
        pages.append(page)
    return pages


def tile_input(packed, records, tile, row_members, lane_members):
    """The command's pages: header, row records 32 a page, lane pages."""
    head = np.zeros(PAGE_WORDS, dtype=np.uint32)
    lanes = lane_pages(packed, records, lane_members, tile) if len(lane_members) else []
    head[[L["kHeadTile"], L["kHeadRowRecords"], L["kHeadLanePages"]]] = [
        tile,
        len(row_members),
        len(lanes),
    ]
    row_pages = -(-len(row_members) // 32)
    rows = np.zeros((row_pages * 32, 32), dtype=np.uint32)
    rows[: len(row_members)] = packed[row_members]
    return np.concatenate(
        [
            head[None],
            rows.reshape(row_pages, PAGE_WORDS),
            np.array(lanes).reshape(-1, PAGE_WORDS),
        ]
    )


def expected_planes(records, members, tile):
    """The five planes the card writes for `members` of the tile (window order)."""
    ty, tx = divmod(tile, raster.TILES_X)
    colour, depth, winner = raster.fill_tile(records, members, tx, ty)
    triangle = np.where(winner >= 0, records["tri"][np.maximum(winner, 0)], -1)
    pixels = np.stack(
        [depth, colour[:, :, 0], colour[:, :, 1], colour[:, :, 2], triangle]
    ).astype(np.float32)
    planes = np.zeros((K["kPlanes"], 1024), dtype=np.uint32)
    planes[:, lists.TILE_ORDER] = pixels.view(np.uint32)
    return planes


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="s1m")
    parser.add_argument("--tiles", type=int, default=16)
    parser.add_argument(
        "--pick", default="random", choices=["random", "busiest"], help="which tiles"
    )
    parser.add_argument("--side", type=int, default=4, help="small: box at most side^2")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--define", action="append", default=[], help="NAME=VALUE")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    modes = args.modes.split(",")
    defines = dict(item.split("=", 1) for item in args.define)
    scene, _ = run.load(args.scene)
    records = raster.setup(scene)
    packed = lists.pack_records(records)
    counts, offsets, members = raster.bin_triangles(records)
    busy = np.flatnonzero(counts > 0)
    rng = np.random.default_rng(args.seed)
    if args.pick == "busiest":
        tiles = np.sort(np.argsort(-counts, kind="stable")[: args.tiles])
    else:
        chosen = rng.choice(busy, size=min(args.tiles, len(busy)), replace=False)
        tiles = np.sort(chosen)
    commands, inputs, page = [], [], 0
    for tile in tiles.tolist():
        everyone = members[offsets[tile] : offsets[tile + 1]]
        clip = clipped_boxes(records, everyone, tile)
        small = (clip[:, 1] - clip[:, 0] < args.side) & (
            clip[:, 3] - clip[:, 2] < args.side
        )
        parts = {
            "rows_small": (everyone[small], everyone[:0], everyone[small]),
            "lanes_small": (everyone[:0], everyone[small], everyone[small]),
            "rows_all": (everyone, everyone[:0], everyone),
            "mixed": (everyone[~small], everyone[small], everyone),
        }
        for mode in modes:
            row_members, lane_members, drawn = parts[mode]
            pages = tile_input(packed, records, tile, row_members, lane_members)
            commands.append(
                {
                    "tile": tile,
                    "mode": mode,
                    "first": page,
                    "pages": len(pages),
                    "small": int(small.sum()),
                    "entries": len(everyone),
                    "drawn": drawn,
                }
            )
            inputs.append(pages)
            page += len(pages)
    inputs = np.concatenate(inputs)
    card.log(f"{len(commands)} commands, {len(inputs)} input pages")
    report = {"scene": args.scene, "args": vars(args), "tiles": tiles.tolist()}
    index = bench.card_index()
    with km.Session(log=card.log) as session:
        source = card.tensor(session, inputs)
        out_pages = L["kLanesOutPages"]
        target = card.tensor(session, np.zeros((out_pages * len(commands), PAGE_WORDS)))
        with bench.ClockSampler(index) as clock:
            running = session.load(program(defines), deadline=1.0)
            card.log("loaded")
            for number, command in enumerate(commands):
                handle = running.submit(
                    "tile",
                    group=(0, 1),
                    args=[
                        source.address,
                        command["first"],
                        command["pages"],
                        target.address,
                        out_pages * number,
                    ],
                )
                running.wait(handle, timeout=10.0)
            running.stop()
        card.log("stopped")
        report["clock"] = clock.figures()
        report["status"] = card.check_run(running, 1, 0, [])
        (output,) = card.read(session, target)
    output = output.reshape(len(commands), out_pages, PAGE_WORDS)
    rows = []
    for number, command in enumerate(commands):
        expected = expected_planes(records, command.pop("drawn"), command["tile"])
        planes = output[number, : K["kPlanes"]]
        command["differ"] = int((planes != expected).sum())
        profile = output[number, K["kPlanes"], : len(PROFILE)].astype(np.int64)
        command.update(zip(PROFILE, profile.tolist()))
        rows.append(command)
    report["commands"] = rows
    summary = {}
    for mode in modes:
        chosen = [row for row in rows if row["mode"] == mode]
        totals = {key: sum(row[key] for row in chosen) for key in PROFILE}
        totals["differ"] = sum(row["differ"] for row in chosen)
        totals["small"] = sum(row["small"] for row in chosen)
        totals["entries"] = sum(row["entries"] for row in chosen)
        summary[mode] = totals
    report["summary"] = summary
    text = json.dumps(report)
    print(json.dumps(summary, indent=1))
    if args.json:
        Path(args.json).write_text(text + "\n")


if __name__ == "__main__":
    main()
