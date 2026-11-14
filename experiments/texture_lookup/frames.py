"""Real frames for probe.py's replay: `inputs` writes the sample tiles and each view's
12 busiest cores, `project` fits a cost per item and projects each view on 120 cores.
python frames.py inputs --streams DIR --out DIR; python frames.py project --streams DIR
--inputs DIR --results DIR --out FILE"""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.texture_lookup import probe

SAMPLE_VIEWS = (
    "sponza_overview",
    "bistro_interior_overview",
    "bistro_exterior_overview",
)


def tile_words(data, tile):
    """A whole tile's reads, its items in row order."""
    items = data["items"]
    own = items[items[:, 3] == tile]
    own = own[np.argsort(own[:, 4])]
    return np.concatenate([data["words"][f : f + n] for _, f, n, *_ in own])


def samples(folder):
    words, items, labels = [], [], []
    at = 0
    for view in SAMPLE_VIEWS:
        data = dict(np.load(folder / f"{view}.npz"))
        pages, mats, reads = (
            data["tile_pages"],
            data["tile_materials"],
            data["tile_reads"],
        )
        live = np.flatnonzero(reads > 0)
        many = live[np.argmax(pages[live])]
        few = live[np.argmin(np.where(reads[live] >= 512, pages[live], 1 << 30))]
        middle = live[np.argsort(pages[live])[len(live) // 2]]
        rest = live[(live != many) & (live != few) & (live != middle)]
        edge = rest[np.argmax(mats[rest])]
        for name, tile in (
            ("many", many),
            ("few", few),
            ("median", middle),
            ("edge", edge),
        ):
            got = tile_words(data, int(tile))
            items.append([len(items), at, len(got)])
            words.append(got)
            labels.append(
                {
                    "view": view,
                    "sample": name,
                    "tile": int(tile),
                    "reads": int(reads[tile]),
                    "pages": int(pages[tile]),
                    "materials": int(mats[tile]),
                }
            )
            at += len(got)
    return np.concatenate(words), np.asarray(items, np.int64), labels


def heavy(folder, view):
    data = dict(np.load(folder / f"{view}.npz"))
    items = data["items"]
    per_core = np.bincount(items[:, 0], weights=items[:, 2], minlength=120)
    chosen = np.argsort(-per_core, kind="stable")[:12]
    words, out, labels, at = [], [], [], 0
    for slot, core in enumerate(chosen):
        for row in items[items[:, 0] == core]:
            got = data["words"][row[1] : row[1] + row[2]]
            out.append([slot, at, len(got)])
            words.append(got)
            at += len(got)
        labels.append({"view": view, "core": int(core), "reads": int(per_core[core])})
    return np.concatenate(words), np.asarray(out, np.int64), labels


MHZ = 1350.0
CORES = 120
FILL_MS = {  # the thread program's FILL a frame, projects/raster/README.md (settled)
    "sponza_overview": 1.592,
    "bistro_interior_overview": 2.095,
}


def walk(keys, slots, table):
    """Misses of keys (repeats removed: a repeat always hits and keeps the order of
    use) through the kernel's table; `table` [tag, use, clock] carries over."""
    tag, use = table[0], table[1]
    misses = 0
    for key in keys:
        first = probe.set_of(key, slots)
        ways = tag[first : first + 4]
        if key + 1 in ways:
            slot = first + ways.index(key + 1)
        else:
            slot = first
            for way in range(4):
                if tag[first + way] == 0:
                    slot = first + way
                    break
                if use[first + way] < use[slot]:
                    slot = first + way
            tag[slot] = key + 1
            misses += 1
        table[2] += 1
        use[slot] = table[2]
    return misses


def item_counts(words, slots, table):
    """(reads, changes, misses) of one item; the table carries over."""
    keys = (words >> 10).astype(np.int64)
    if len(keys) == 0:
        return 0, 0, 0
    change = np.ones(len(keys), bool)
    change[1:] = keys[1:] != keys[:-1]
    runs = keys[change].tolist()
    return len(keys), len(runs), walk(runs, slots, table)


def fresh():
    """An empty table: tags, uses, the clock."""
    return [[0] * 32, [0] * 32, 0]


def measured_rows(inputs, results):
    """Every replayed item (results/replay-<set>.json, probe.py --json): form, slots,
    reads, changes, misses (checked against the host's walk), cycles."""
    rows = []
    names = ["samples"] + [
        f"heavy_{view}"
        for view in (
            "sponza_overview",
            "bistro_interior_overview",
            "bistro_exterior_overview",
        )
    ]
    for name in names:
        data = dict(np.load(inputs / f"{name}.npz"))
        tag = "samples" if name == "samples" else name.replace("heavy_", "heavy-")
        report = json.loads((results / f"replay-{tag}.json").read_text())
        items = data["items"]
        table, table_key = None, None
        for row in report["rows"]:
            key = (row["slots"], row["form"], row["core"])
            if key != table_key:  # a command clears each core's table once
                table_key, table = key, fresh()
            own = items[items[:, 0] == row["core"]]
            _, first, count = own[row["item"]]
            reads, changes, misses = item_counts(
                data["words"][first : first + count], row["slots"], table
            )
            if misses != row["misses"]:
                raise RuntimeError(f"{name} {key}: host {misses} card {row['misses']}")
            rows.append(
                {
                    "set": name,
                    "slots": row["slots"],
                    "form": row["form"],
                    "core": row["core"],
                    "pass": row["pass"],
                    "reads": reads,
                    "changes": changes,
                    "misses": misses,
                    "cycles": row["cycles"],
                }
            )
    return rows


def fit(rows, form):
    """cycles = a reads + b changes + c misses + d, least squares over the items."""
    mine = [r for r in rows if r["form"] == form]
    x = np.array([[r["reads"], r["changes"], r["misses"], 1.0] for r in mine], float)
    y = np.array([r["cycles"] for r in mine], float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    error = (x @ coef - y) / y
    return coef, {
        "a_read": round(float(coef[0]), 2),
        "b_change": round(float(coef[1]), 2),
        "c_miss": round(float(coef[2]), 1),
        "d_item": round(float(coef[3]), 1),
        "items": len(mine),
        "error_median_pct": round(float(np.median(np.abs(error))) * 100, 2),
        "error_max_pct": round(float(np.abs(error).max()) * 100, 2),
    }


def project(streams, coefs, slots_list, forms):
    """Per view, page size, table size and form: the model's time of the slowest and
    the mean core (every core's items in order, its table cold at the frame's start),
    the misses and the DRAM bytes they read."""
    out = {}
    for path in sorted(streams.glob("*.npz")):
        view = path.stem
        data = dict(np.load(path))
        items = data["items"]
        result = {"reads": int(len(data["words"]))}
        for page, words_key in (("4KB", "words"), ("1KB", "words16")):
            for slots in slots_list:
                counts = np.zeros((CORES, 4))  # reads, changes, misses, items
                for core in range(CORES):
                    table = fresh()
                    for _, first, count, *_ in items[items[:, 0] == core]:
                        r, c, m = item_counts(
                            data[words_key][first : first + count], slots, table
                        )
                        counts[core] += (r, c, m, 1)
                for form in forms:
                    coef = coefs[form]
                    cycles = counts @ coef
                    entry = {
                        "misses": int(counts[:, 2].sum()),
                        "hit_rate": round(
                            1 - counts[:, 2].sum() / counts[:, 0].sum(), 4
                        ),
                        "ms_max_core": round(float(cycles.max()) / MHZ / 1e3, 3),
                        "ms_mean_core": round(float(cycles.mean()) / MHZ / 1e3, 3),
                        "miss_ms_max_core": round(
                            float((counts[:, 2] * coef[2]).max()) / MHZ / 1e3, 3
                        ),
                        "dram_mb": round(
                            counts[:, 2].sum()
                            * (4096 if page == "4KB" else 1024)
                            / 1e6,
                            1,
                        ),
                    }
                    entry["dram_gbs_at_max"] = round(
                        entry["dram_mb"] / entry["ms_max_core"], 1
                    )
                    result[f"{page}_{slots}_{form}"] = entry
        out[view] = result
        print(
            view,
            json.dumps({k: v for k, v in result.items() if "4KB_16" in k}),
            flush=True,
        )
    return out


def against_model(rows, coefs):
    """The busiest cores' measured time against the model's, the slowest core."""
    check = {}
    for row in rows:
        if row["set"] == "samples":
            continue
        key = f"{row['set']}_{row['slots']}_{row['form']}"
        entry = check.setdefault(key, {})
        cell = entry.setdefault(row["core"], [0, 0.0])
        cell[0] += row["cycles"]
        cell[1] += float(
            np.dot(coefs[row["form"]], [row["reads"], row["changes"], row["misses"], 1])
        )
    return {
        key: {
            "measured_ms_max": round(max(v[0] for v in cores.values()) / MHZ / 1e3, 3),
            "model_ms_max": round(max(v[1] for v in cores.values()) / MHZ / 1e3, 3),
        }
        for key, cores in check.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("inputs", "project"))
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--results", type=Path, help="project: the replays' JSON")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.what == "inputs":
        args.out.mkdir(parents=True, exist_ok=True)
        words, items, labels = samples(args.streams)
        np.savez(args.out / "samples.npz", words=words, items=items)
        (args.out / "samples.json").write_text(json.dumps(labels, indent=1))
        for view in SAMPLE_VIEWS:
            words, items, labels = heavy(args.streams, view)
            np.savez(args.out / f"heavy_{view}.npz", words=words, items=items)
            (args.out / f"heavy_{view}.json").write_text(json.dumps(labels, indent=1))
            print(view, len(words), [label["core"] for label in labels], flush=True)
        return
    rows = measured_rows(args.inputs, args.results)
    coefs, fits = {}, {}
    for form in ("hash", "hash_last"):
        coefs[form], fits[form] = fit(rows, form)
    print(json.dumps(fits), flush=True)
    heavy_check = against_model(rows, coefs)
    print(json.dumps(heavy_check), flush=True)
    projected = project(args.streams, coefs, (8, 16, 32), ("hash", "hash_last"))
    report = {"fits": fits, "heavy": heavy_check, "projected": projected}
    report["fill_ms"] = FILL_MS
    args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
