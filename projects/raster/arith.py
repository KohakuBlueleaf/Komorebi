"""Arithmetic check: the fill's per-pixel expressions (a px + b py + c, each step
rounded to float32, compared in the total order) on the card against numpy, bit for bit,
on a real scene's records and edge cases. See the README's "Arithmetic check".
`python -m projects.raster.arith --n 1000000 [--scene s10k] [--json FILE]`"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from projects.raster import card, lists
from ref import native, raster, scenes

K = lists.K
F = np.float32
LANES = K["kArithLanes"]
NAMES = ("a px", "b py", "sum", "e", "e > T", "e > d")


def total_key(x):
    """Signed keys that order float32 bit patterns as the vector unit's SFPGT does."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.int64)
    return np.where(bits >> 31, -(bits & 0x7FFFFFFF) - 1, bits)


def ftz(x):
    """Denormals to zero of the same sign."""
    x = np.asarray(x, dtype=np.float32)
    tiny = (np.abs(x) < np.finfo(np.float32).tiny) & (x != 0)
    return np.where(tiny, np.copysign(F(0), x), x).astype(np.float32)


def expected(a, b, c, px, py, threshold, depth, flush=False):
    """The six outputs in numpy float32 (IEEE; with flush, denormals flushed)."""
    rounded = ftz if flush else (lambda values: np.asarray(values, dtype=np.float32))
    with np.errstate(all="ignore"):
        a_px = rounded(rounded(a) * rounded(px))
        b_py = rounded(rounded(b) * rounded(py))
        products_sum = rounded(a_px + b_py)
        edge_value = rounded(products_sum + rounded(c))
    gt_t = total_key(edge_value) > total_key(threshold)
    gt_d = total_key(edge_value) > total_key(depth)
    return [
        a_px,
        b_py,
        products_sum,
        edge_value,
        gt_t.astype(np.uint32),
        gt_d.astype(np.uint32),
    ]


def scene_inputs(generator, count, scene):
    """n inputs from the planes of a scene's records near their boxes."""
    native.ensure_scene(scene)
    records = raster.setup(scenes.load_scene(native.data_dir() / f"{scene}.npz"))
    triangles = len(records["tri"])
    picked = generator.integers(0, triangles, count)
    kind = generator.integers(
        0, 7, count
    )  # 0-2 edges, 3 the 1/z plane, 4-6 colour channels
    coefficients = np.empty((count, 3), dtype=np.float32)
    for edge in range(3):
        selected = kind == edge
        coefficients[selected] = np.stack(
            [
                records["a"][picked[selected], edge],
                records["b"][picked[selected], edge],
                records["c"][picked[selected], edge],
            ],
            axis=1,
        )
    selected = kind == 3
    coefficients[selected] = records["w_planes"][picked[selected]]
    for channel in range(3):
        selected = kind == 4 + channel
        coefficients[selected] = records["c_planes"][picked[selected], :, channel]
    box = records["box"][picked].astype(np.int64)
    col = np.clip(generator.integers(box[:, 0] - 2, box[:, 1] + 3), 0, raster.WIDTH - 1)
    row = np.clip(
        generator.integers(box[:, 2] - 2, box[:, 3] + 3), 0, raster.HEIGHT - 1
    )
    px = (col.astype(np.float32) + F(0.5)).astype(np.float32)
    py = (row.astype(np.float32) + F(0.5)).astype(np.float32)
    top_left = np.where(
        kind < 3,
        records["top_left"][picked, np.minimum(kind, 2)],
        generator.random(count) < 0.5,
    )
    thresholds = (
        np.where(top_left, lists.NEG_ZERO_NEIGHBOUR, 0)
        .astype(np.uint32)
        .view(np.float32)
    )
    return (
        coefficients[:, 0],
        coefficients[:, 1],
        coefficients[:, 2],
        px,
        py,
        thresholds,
        records,
    )


def edge_inputs(generator, count):
    """n edge cases: signed zeros, tiny, denormal and large values, cancellation."""
    special = np.array(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            1.2e-38,
            -1.2e-38,
            1e-40,
            -1e-40,
            3e-39,
            1e30,
            -1e30,
            3.4e38,
            1e-20,
            2.0**-63,
        ],
        dtype=np.float32,
    )
    pick = generator.integers(0, len(special), (count, 3))
    random_values = generator.uniform(-1e3, 1e3, (count, 3)).astype(np.float32)
    use_special = generator.random((count, 3)) < 0.5
    a, b, c = (
        np.where(use_special[:, term], special[pick[:, term]], random_values[:, term])
        for term in range(3)
    )
    px = (generator.integers(0, 1024, count) + F(0.5)).astype(np.float32)
    py = (generator.integers(0, 1024, count) + F(0.5)).astype(np.float32)
    # A third: c cancels a finite sum exactly, so e is +-0.
    with np.errstate(all="ignore"):
        partial_sum = (a * px).astype(np.float32) + (b * py).astype(np.float32)
    cancel = (generator.random(count) < 1 / 3) & np.isfinite(partial_sum)
    c = np.where(cancel, -partial_sum, c).astype(np.float32)
    thresholds = np.where(
        generator.random(count) < 0.5, lists.NEG_ZERO_NEIGHBOUR, 0
    ).astype(np.uint32)
    return a, b, c, px, py, thresholds.view(np.float32)


def depths(generator, edge_values, count):
    """d: e itself, its neighbours, zeros, or random depths."""
    kind = generator.integers(0, 5, count)
    above = np.nextafter(edge_values, np.float32(np.inf))
    below = np.nextafter(edge_values, np.float32(-np.inf))
    random_depths = generator.uniform(0, 1, count).astype(np.float32)
    zero = np.where(generator.random(count) < 0.5, F(0), F(-0.0))
    return np.choose(kind, [edge_values, above, below, zero, random_depths]).astype(
        np.float32
    )


def chunk_pages(inputs):
    """Pages of the inputs: per chunk of 512, one page per input array."""
    values = len(inputs[0])
    chunks = values // LANES
    pages = np.stack([np.asarray(x, dtype=np.float32).view(np.uint32) for x in inputs])
    pages = pages.reshape(len(inputs), chunks, LANES).transpose(1, 0, 2)
    return pages.reshape(chunks * len(inputs), LANES)


def plan(chunks, cores):
    """(list pages per core order, directory): chunks dealt round-robin to cores."""
    owner = [list(range(core, chunks, cores)) for core in range(cores)]
    return owner


def build(count, scene, edge_fraction, seed):
    """(inputs, source): n inputs (a multiple of a chunk), a fraction of them edge
    cases (source True), in a random order."""
    generator = np.random.default_rng(seed)
    count = (count + LANES - 1) // LANES * LANES
    edge_count = int(count * edge_fraction)
    a, b, c, px, py, threshold, _ = scene_inputs(generator, count - edge_count, scene)
    edge_a, edge_b, edge_c, edge_px, edge_py, edge_threshold = edge_inputs(
        generator, edge_count
    )
    a, b, c = (
        np.concatenate([a, edge_a]),
        np.concatenate([b, edge_b]),
        np.concatenate([c, edge_c]),
    )
    px, py, threshold = (
        np.concatenate([px, edge_px]),
        np.concatenate([py, edge_py]),
        np.concatenate([threshold, edge_threshold]),
    )
    reference_edge = expected(a, b, c, px, py, threshold, threshold)[3]
    depth = depths(generator, reference_edge, count)
    order = generator.permutation(count)
    inputs = [x[order] for x in (a, b, c, px, py, threshold, depth)]
    source = np.concatenate(
        [np.zeros(count - edge_count, bool), np.ones(edge_count, bool)]
    )[order]
    return inputs, source


def lay_out(inputs, workers):
    """(list pages, directory) of the chunks dealt round-robin to the workers."""
    chunks = len(inputs[0]) // LANES
    per_chunk = chunk_pages(inputs)  # (chunks * 7, 512)
    owner = plan(chunks, workers)
    pages, directory, first = [], np.zeros((workers, 1024), np.uint32), 0
    end_page = np.zeros((1, LANES), np.uint32)
    end_page[0, 0] = K["kMarkEnd"]
    for core, chunk_ids in enumerate(owner):
        own_pages = [per_chunk[chunk * 7 : chunk * 7 + 7] for chunk in chunk_ids] + [
            end_page
        ]
        block = np.concatenate(own_pages)
        directory[core, K["kDirFirst"]] = first
        directory[core, K["kDirPages"]] = len(block)
        directory[core, K["kDirItems"]] = len(chunk_ids)
        directory[core, K["kDirIds"] : K["kDirIds"] + len(chunk_ids)] = chunk_ids
        pages.append(block)
        first += len(block)
    return np.concatenate(pages), directory


def compare(output_words, inputs, source):
    """Per output and per rounding (IEEE, denormals flushed): how many differ."""
    count = len(inputs[0])
    chunks = count // LANES
    got = output_words.reshape(chunks, 6, LANES).transpose(1, 0, 2).reshape(6, count)
    report = {}
    for flush in (False, True):
        want = expected(*inputs, flush=flush)
        key = "ftz" if flush else "ieee"
        report[key] = {}
        for output_number, name in enumerate(NAMES):
            want_words = (
                np.asarray(want[output_number]).view(np.uint32)
                if output_number < 4
                else want[output_number]
            )
            differs = got[output_number] != want_words
            entry = {
                "differ": int(differs.sum()),
                "differ_scene": int((differs & ~source).sum()),
                "differ_edge": int((differs & source).sum()),
            }
            if output_number < 4:
                got_floats, want_floats = got[output_number].view(
                    np.float32
                ), np.asarray(want[output_number], np.float32)
                both_zero = (got_floats == 0) & (want_floats == 0)
                entry["differ_only_zero_sign"] = int((differs & both_zero).sum())
            first_differing = np.nonzero(differs)[0][:5]
            entry["examples"] = [
                {
                    "in": [f"{inputs[field][i]:.9g}" for field in range(7)],
                    "got": int(got[output_number][i]),
                    "want": int(want_words[i]),
                }
                for i in first_differing
            ]
            report[key][name] = entry
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--scene", default="s10k")
    parser.add_argument("--edge-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    inputs, source = build(args.n, args.scene, args.edge_fraction, args.seed)
    count = len(inputs[0])
    chunks = count // LANES
    with km.Session(log=card.log) as session:
        workers = card.workers(session)
        pages, directory = lay_out(inputs, workers)
        output, _, status, _, times = card.run_lists(
            session, K["kHandlerArith"], pages, directory, chunks * 3, K["kArithOut"]
        )
    report = {"n": count, "edge_cases": int(source.sum()), "scene": args.scene}
    report["run"] = status
    report["times"] = times
    report.update(compare(output, inputs, source))
    # Exact: no difference from the flushing model anywhere, none from IEEE on the
    # scene's own inputs (only the edge cases hold denormals).
    report["ok"] = bool(
        status["ok"]
        and all(entry["differ"] == 0 for entry in report["ftz"].values())
        and all(entry["differ_scene"] == 0 for entry in report["ieee"].values())
    )
    print(json.dumps(report), flush=True)
    if args.json:
        with open(args.json, "w") as file:
            json.dump(report, file, indent=1)
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    started = time.time()
    main()
    print(f"wall {time.time() - started:.1f}s", flush=True)
