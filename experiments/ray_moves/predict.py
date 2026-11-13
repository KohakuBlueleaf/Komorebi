"""The predicted frame with resident treelets and moving rays, beside today's: part A's
walks and bundles, part B's rates, today's profile for what stays (the README has the
command)."""

import argparse
import json
from pathlib import Path

TRACES = ("primary", "shadow", "bounce", "shadow2")
KEPT = ("GEN", "SHADE1", "SHADE2", "IMAGE")
RECORD_BYTES = 48
BUNDLE_HEADER_BYTES = 16
PAGE_SIZES = (2048, 4096)


def page_for(bundle_rays):
    """The smallest measured page that holds a bundle of `bundle_rays` rays, or None."""
    need = BUNDLE_HEADER_BYTES + bundle_rays * RECORD_BYTES
    return next((size for size in PAGE_SIZES if size >= need), None)


def rates(paths):
    """{(kernels, size): moves a second} from part B's result files."""
    found = {}
    for path in paths:
        for size, row in json.loads(Path(path).read_text())["sizes"].items():
            found[(row["kernels"], int(size))] = row["moves_per_second"]
    return found


def trace_terms(trace, today, rate, bundle_rays, rule, clock_mhz):
    """The terms of one trace in milliseconds, and its prediction with and without the
    owners' walks overlapping the transport."""
    placement = trace["by_cycles"][trace["placement"]]
    row = placement["bundles"][f"{bundle_rays}/{rule}"]
    walk_max, walk_mean = row["walk"]["cycles_per_core_max_mean"]
    terms = {
        "top walk": today["phase1"],
        "owner walks": walk_max / clock_mhz / 1e3,
        "transport": row["bundles"] / rate * 1e3,
        "output": today["phase3"],
        "rest": max(today["record"] - today["total"], 0.0),
    }
    fixed = terms["top walk"] + terms["output"] + terms["rest"]
    return {
        "terms": {name: round(value, 2) for name, value in terms.items()},
        "owner_walk_mean_ms": round(walk_mean / clock_mhz / 1e3, 2),
        "bundles": row["bundles"],
        "fill": round(row["fill_mean"], 3),
        "lanes_per_inner": round(row["walk"]["lanes_per_inner"], 2),
        "fits": placement["fits"],
        "overlapped": round(fixed + max(terms["owner walks"], terms["transport"]), 2),
        "serial": round(fixed + terms["owner walks"] + terms["transport"], 2),
    }


def predict(scene, today, found, budget, clock_mhz):
    """Every measured transport and R with the "full" rule: per trace and frame."""
    options = []
    kept = sum(today["commands_ms"][name] for name in KEPT)
    for kernels in ("pages", "zero-copy"):
        for bundle_rays in (32, 64, 128):
            size = page_for(bundle_rays)
            if size is None or (kernels, size) not in found:
                continue
            traces = {}
            for name in TRACES:
                measured = today["traces"][name] | {
                    "record": today["commands_ms"][name]
                }
                traces[name] = trace_terms(
                    scene["budgets"][str(budget)]["traces"][name],
                    measured,
                    found[(kernels, size)],
                    bundle_rays,
                    "full",
                    clock_mhz,
                )
            options.append(
                {
                    "kernels": kernels,
                    "bundle_rays": bundle_rays,
                    "page_bytes": size,
                    "rule": "full",
                    "traces": traces,
                    "frame_overlapped": round(
                        kept + sum(t["overlapped"] for t in traces.values()), 2
                    ),
                    "frame_serial": round(
                        kept + sum(t["serial"] for t in traces.values()), 2
                    ),
                    "fits": all(t["fits"] for t in traces.values()),
                }
            )
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths")
    parser.add_argument("today")
    parser.add_argument("transport", nargs="+")
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    paths = json.loads(Path(arguments.paths).read_text())
    today = json.loads(Path(arguments.today).read_text())
    found = rates(arguments.transport)
    result = {}
    for name, measured in today["scenes"].items():
        if name not in paths["scenes"]:
            continue
        options = predict(
            paths["scenes"][name], measured, found, arguments.budget, today["aiclk_mhz"]
        )
        best = min(options, key=lambda option: option["frame_overlapped"])
        result[name] = {
            "today_ms": measured["commands_ms"],
            "options": options,
            "best": [best["kernels"], best["bundle_rays"]],
        }
        print(f"{name}: today {measured['commands_ms']['frame']} ms")
        for option in options:
            print(
                f"  {option['kernels']:16s} R={option['bundle_rays']:3d}"
                f" {option['page_bytes']} B: frame {option['frame_overlapped']:7.1f} ms"
                f" overlapped, {option['frame_serial']:7.1f} serial,"
                f" fits {option['fits']}"
            )
        for trace, row in best["traces"].items():
            print(
                f"    {trace:8s} {row['overlapped']:7.1f} ms"
                f" (today {measured['commands_ms'][trace]}) {row['terms']}"
            )
    if arguments.json:
        Path(arguments.json).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
