"""The cost of one ray (or a few) against a wide node's child boxes on one core: the
probe's forms timed by the core's own clock, the entry distances checked against numpy
(the README has the commands)."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from experiments.programs.ray_moves.paths import ray_sets  # noqa: E402
from experiments.programs.ray_wide import wide  # noqa: E402
from komorebi import abi  # noqa: E402
from reference import bvh, raytrace, scenes  # noqa: E402

KERNELS = HERE / "kernels"
LAYOUTS = {"1x32": 1, "2x16": 2, "4x8": 4}  # name: rays an operation
METHODS = {"1x32": ("window", "unpack"), "2x16": ("window", "gather")}
METHODS["4x8"] = METHODS["2x16"]
FORMS = ("check", "load", "test", "out", "stack", "ray")
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def visits_of(tree, wide_tree, origin, direction, best):
    """The wide nodes one ray visits, nearest first, walked with its final best: a
    list of (node, children)."""
    inverse = raytrace.inverse_direction(direction[None, :])[0]
    root = np.ascontiguousarray(tree["nodes"][:1, :6]).view(np.float32)
    hit, entry = wide.box_entry(root, origin[None, :], inverse[None, :], best[None])
    stack = [(0, float(entry[0]))] if hit[0] else []
    found = []
    while stack:
        node, entry = stack.pop()
        if entry > best:
            continue
        present = wide_tree["present"][node]
        found.append((node, int(present.sum())))
        hit, entries = wide.box_entry(
            wide_tree["boxes"][node], origin[None, :], inverse[None, :], best
        )
        inner = hit & present & ~wide_tree["is_leaf"][node]
        order = np.argsort(-np.where(inner, entries, -np.inf), kind="stable")
        stack.extend(
            (int(wide_tree["target"][node, slot]), float(entries[slot]))
            for slot in order
            if inner[slot]
        )
    return found


def build(tree, wide_tree, rays, groups, op_limit):
    """The probe's data for `groups` rays an operation: operation words, node records,
    ray records and the expected entry distances (NaN where a lane is not checked)."""
    constants = abi.constants(KERNELS / "probe.h")
    children = constants["kLanes"] // groups
    origins, directions, best = rays
    inverse = raytrace.inverse_direction(directions)
    sequences = [
        visits_of(tree, wide_tree, origins[r], directions[r], best[r])
        for r in range(len(best))
    ]
    queue = [r for r in range(len(best)) if sequences[r]]
    current = [None] * groups
    position = [0] * groups
    ops, records, expected = [], [], []
    while len(ops) < op_limit:
        words = []
        lanes = np.full(constants["kLanes"], np.nan, dtype=np.float32)
        for group in range(groups):
            if current[group] is None or position[group] >= len(
                sequences[current[group]]
            ):
                if not queue:
                    break
                current[group], position[group] = queue.pop(0), 0
            ray = current[group]
            node, count = sequences[ray][position[group]]
            position[group] += 1
            boxes = np.zeros((children, 6), dtype=np.float32)
            boxes[:count] = wide_tree["boxes"][node][:count]
            words += [sum(len(r) for r in records), ray, count]
            records.append(boxes.T.ravel().view(np.uint32))
            hit, entries = wide.box_entry(
                boxes[:count], origins[ray][None, :], inverse[ray][None, :], best[ray]
            )
            lanes[group * children : group * children + count] = np.where(
                hit, entries, np.inf
            )
        else:
            ops.append(words + [0] * (constants["kOpWords"] - len(words)))
            expected.append(lanes)
            continue
        break
    ray_words = np.zeros((len(best), constants["kRayWords"]), dtype=np.uint32)
    ray_words[:, 0:3] = origins.view(np.uint32)
    ray_words[:, 3:6] = inverse.view(np.uint32)
    ray_words[:, 6] = best.view(np.uint32)
    return (
        np.array(ops, dtype=np.uint32),
        np.concatenate(records),
        ray_words,
        np.array(expected),
    )


def tile_images(nodes, ops):
    """Node tiles for the unpacker: operation k's six box words (1 x 32) at the Dst
    rows and columns of slots 0 to 5 in tile k, the rest zero."""
    constants = abi.constants(KERNELS / "probe.h")
    lanes = constants["kLanes"]
    tiles = np.zeros((len(ops), constants["kTileWords"]), dtype=np.uint32)
    lane = np.arange(lanes)
    for op, words in enumerate(ops):
        record = nodes[words[0] : words[0] + 6 * lanes].reshape(6, lanes)
        for slot in range(6):
            row = 4 * (slot >> 1) + lane // 8
            column = 2 * (lane & 7) + (slot & 1)
            tiles[op, 16 * row + column] = record[slot]
    return tiles


def sharded(session, words, cores):
    """The same row of words in every core's L1."""
    rows = np.broadcast_to(
        np.asarray(words, dtype=np.uint32).ravel(), (cores, words.size)
    )
    return session.tensor(np.ascontiguousarray(rows), "uint32", shard_grid=session.grid)


def run_form(session, data, groups, form, method, passes, first, count, arguments):
    """One load and one command of the probe in `form`, boxes loaded by `method`, on
    cores first to first + count - 1; the result words of every core and the entry
    distances of core first."""
    constants = abi.constants(KERNELS / "probe.h")
    grid = session.grid
    cores = grid[0] * grid[1]
    ops, nodes, rays, expected = data
    if method == "unpack":
        ops, expected = (
            ops[: constants["kTilesMax"]],
            expected[: constants["kTilesMax"]],
        )
    tiles = sharded(
        session,
        (
            tile_images(nodes, ops)
            if method == "unpack"
            else np.zeros(constants["kTileWords"], np.uint32)
        ),
        cores,
    )
    tensors = [
        sharded(session, ops, cores),
        sharded(session, nodes, cores),
        sharded(session, rays, cores),
        sharded(session, np.zeros(len(ops) * constants["kLanes"], np.uint32), cores),
        sharded(session, np.zeros(constants["kResultWords"], np.uint32), cores),
    ]
    program = km.Program(
        "wide_probe",
        kernels=KERNELS,
        buffers=[
            km.Buffer(
                constants["kCbNodes"],
                "float32",
                4 * constants["kTileWords"],
                constants["kTilesMax"],
            )
        ],
        args=lambda core: (
            [tiles.address],
            [],
            [tensor.address for tensor in tensors],
        ),
        compute={
            "math_fidelity": "HiFi4",
            "fp32_dest_acc_en": True,
            "enable_trisc2_rvv": True,
            "dst_full_sync_en": True,
            "unpack_to_dest_mode": {constants["kCbNodes"]: "UnpackToDestFp32"},
        },
        handlers={"wide_probe": constants["kHandlerWideProbe"]},
        depth=1,
    )
    run = session.load(program, deadline=arguments.deadline, wait=1.0)
    form_number = constants[f"kForm{form.capitalize()}"]
    method_number = constants[f"kMethod{method.capitalize()}"]
    handle = run.submit(
        "wide_probe",
        (first, count),
        [form_number, groups, len(ops), passes, method_number],
        exclusive=True,
    )
    run.drain(timeout=arguments.deadline + 5.0)
    run.stop()
    event, sequence = run.event(handle), run.sequence(handle)
    results = tensors[4].read().astype(np.int64)[first : first + count]
    entries = tensors[3].read()[first].view(np.uint32).reshape(len(ops), -1)
    return {
        "done": bool(event and event.kind == "done"),
        "run": bool((results[:, constants["kResultRun"]] == sequence).all()),
        "error": int(results[:, constants["kResultError"]].sum()),
        "results": results,
        "entries": entries,
        "expected": expected,
    }


def compare(entries, expected):
    """Lanes whose entry distance differs from numpy's in any bit."""
    checked = ~np.isnan(expected)
    want = expected.astype(np.float32).view(np.uint32)
    return int((entries[checked] != want[checked]).sum()), int(checked.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="sponza_overview")
    parser.add_argument("--trace", default="bounce")
    parser.add_argument("--layouts", nargs="+", default=list(LAYOUTS))
    parser.add_argument("--forms", nargs="+", default=list(FORMS))
    parser.add_argument("--rays", type=int, default=64)
    parser.add_argument("--ops", type=int, default=256)
    parser.add_argument("--passes", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cores", type=int, default=1, help="cores 0 to N - 1")
    parser.add_argument("--data", default=os.environ.get("KR_DATA", "build/data"))
    parser.add_argument("--cache", default="build/ray_moves")
    parser.add_argument("--deadline", type=float, default=10.0)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    data_folder = Path(arguments.data)
    scene = scenes.load_scene(data_folder / f"{arguments.scene}.npz")
    tree = bvh.load_tree(data_folder / f"{arguments.scene}_tree.npz")
    cut = bvh.cut_treelets(tree, wide.BUDGET * 1024)
    sets = ray_sets(arguments.scene, scene, tree, arguments.cache, 8)
    _, origins, directions, maximum = sets[arguments.trace]
    generator = np.random.default_rng(wide.SEED)
    pick = np.sort(generator.choice(len(origins), arguments.rays, replace=False))
    any_hit = arguments.trace in wide.ANY_HIT
    hits = raytrace.trace(
        tree, origins[pick], directions[pick], maximum[pick][0], any_hit
    )
    best = np.where(any_hit & (hits["leaf"] >= 0), np.float32(-1), hits["t"]).astype(
        np.float32
    )
    card = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
    report = {
        "scene": arguments.scene,
        "trace": arguments.trace,
        "rays": arguments.rays,
        "passes": arguments.passes,
        "cores": arguments.cores,
        "load_before": Path("/proc/loadavg").read_text().split()[:3],
        "layouts": {},
    }
    with km.Session(log=log) as session:
        log("device open")
        for name in arguments.layouts:
            groups = LAYOUTS[name]
            width = 32 // groups
            wide_tree = wide.wide_tree(tree, cut, width)
            data = build(
                tree,
                wide_tree,
                (origins[pick], directions[pick], best),
                groups,
                arguments.ops,
            )
            log(f"{name}: {len(data[0])} operations of W = {width} nodes")
            row = {"width": width, "operations": len(data[0]), "methods": {}}
            for method in METHODS[name]:
                row["methods"][method] = measure(
                    session, data, name, groups, method, card, arguments
                )
            report["layouts"][name] = row
    report["load_after"] = Path("/proc/loadavg").read_text().split()[:3]
    report["ok"] = all(
        form["ok"]
        for row in report["layouts"].values()
        for method in row["methods"].values()
        for form in method.values()
    )
    print(json.dumps({"ok": report["ok"]}), flush=True)
    if arguments.json:
        Path(arguments.json).write_text(json.dumps(report, indent=1) + "\n")
    return report["ok"]


def measure(session, data, name, groups, method, card, arguments):
    """Every form of one layout and method: best of `repeats`, cycles an operation on
    the slowest core; the check form compares every lane with numpy."""
    forms = {}
    for form in arguments.forms:
        if form == "ray" and method != "window":
            continue
        cycles, entry = [], {}
        passes = 1 if form == "check" else arguments.passes
        if method == "unpack" and form != "check":
            passes *= 8
        for _ in range(arguments.repeats if form != "check" else 1):
            out = run_form(
                session,
                data,
                groups,
                form,
                method,
                passes,
                0,
                arguments.cores,
                arguments,
            )
            results = out["results"]
            ok = out["done"] and out["run"] and out["error"] == 0
            per_op = results[:, 0] / np.maximum(results[:, 1], 1)
            cycles.append(float(per_op.max()))
            entry = {
                "ok": ok,
                "cycles_per_operation": cycles,
                "operations": int(results[0, 1]),
                "visits": int(results[0, 2]),
                "entered": int(results[0, 3]),
                "ray_loads": int(results[0, 5]),
                "clock_mhz": clock(card),
            }
            if form == "check":
                differ, checked = compare(out["entries"], out["expected"])
                entry.update(differ=differ, checked=checked)
                entry["ok"] = ok and differ == 0
            log(f"{name} {method} {form}: {per_op.max():.1f} cycles an operation")
            log(f"{entry}")
        entry["best"] = min(cycles)
        forms[form] = entry
    return forms


def clock(card):
    """AICLK of the card in MHz, from the driver; None if it cannot be read."""
    path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
