"""The bounce and second-shadow traces and the frame with one ray (or a few) against
wide nodes, beside today's and the V100's: part A's counts, part B's cycles, today's
profile for the rest of each trace (the README has the command)."""

import argparse
import json
from pathlib import Path

TRACES = ("bounce", "shadow2")
SCENES = ("sponza_overview", "bistro_interior_overview")
LAYOUTS = {"1x32": 1, "2x16": 2, "4x8": 4}
WIDTHS = ("8", "16", "32")
CLOCK_KHZ = 1350e3
POP_CYCLES = 36  # a popped entry beyond the best: today's pop
TRIANGLE_CYCLES = {"bounce": 737, "shadow2": 485}  # today's test, nearest and any
TRIANGLE_WORDS = 9  # a triangle's words into the lanes, against a box's six
SORT_COMMANDS = ("SORT_bounce", "UNSORT_bounce", "SORT_shadow2", "UNSORT_shadow2")


def operation_costs(probe):
    """Per layout, the cheapest measured way of loading boxes: cycles of an operation
    (the stack form), of its box loading, and of loading one ray into its lanes."""
    costs = {}
    for name, row in probe["layouts"].items():
        method, forms = min(
            row["methods"].items(), key=lambda item: item[1]["stack"]["best"]
        )
        ray_form = row["methods"]["window"].get("ray")
        groups = LAYOUTS[name]
        costs[name] = {
            "method": method,
            "operation": forms["stack"]["best"],
            "load": forms["load"]["best"],
            "ray": ray_form["best"] / groups if ray_form and groups > 1 else 0.0,
        }
    return costs


def ray_cycles(counts, layout, cost, trace, treelet_visits, boxes_free=False):
    """A ray's cycles in one trace: its box operations, triangle operations, popped
    entries beyond its best and its loads into the lanes at each treelet."""
    row = counts["layouts"][layout]
    operation = cost["operation"] - (cost["load"] if boxes_free else 0.0)
    triangle_load = 0.0 if boxes_free else cost["load"] * TRIANGLE_WORDS / 6
    terms = {
        "boxes": row["box_operations"] * operation,
        "triangles": row["triangle_operations"]
        * (TRIANGLE_CYCLES[trace] + triangle_load),
        "pops": (counts["pops"] - counts["visits"]) * POP_CYCLES,
        "ray loads": treelet_visits * 2 * cost["ray"],
    }
    return terms


def predict(wide, today, costs, moves, scene, boxes_free=False):
    """Predicted trace times (ms) per layout and width, and the frame's."""
    measured = today["scenes"][scene]
    commands = measured["commands_ms"]
    rays = {trace: measured["v100"]["rays"][f"rays_{trace}"] for trace in TRACES}
    rows = {}
    for width in WIDTHS:
        for layout, cost in costs.items():
            traces = {}
            for trace in TRACES:
                counts = wide["scenes"][scene]["widths"][width]["traces"][trace][
                    "nearest"
                ]
                moved = moves["scenes"][scene]["budgets"]["128"]["traces"][trace]
                terms = ray_cycles(
                    counts, layout, cost, trace, moved["visits_per_ray"][0], boxes_free
                )
                profile = measured["traces"][trace]
                walk = profile["walk_ms"]
                spread = walk["max"] / walk["mean"]
                per_core = rays[trace] / profile["cores_used"]
                walk_ms = {
                    name: value * per_core / CLOCK_KHZ * spread
                    for name, value in terms.items()
                }
                rest = commands[trace] - walk["slowest_core"]
                traces[trace] = {
                    "ms": round(rest + sum(walk_ms.values()), 2),
                    "walk_ms": {
                        name: round(value, 2) for name, value in walk_ms.items()
                    },
                    "rest_ms": round(rest, 2),
                    "cycles_per_ray": round(sum(terms.values())),
                }
            frame = commands["frame"] - sum(commands[name] for name in SORT_COMMANDS)
            frame += sum(traces[t]["ms"] - commands[t] for t in TRACES)
            rows[f"{layout}/W{width}"] = {
                "layout": layout,
                "width": int(width),
                "method": cost["method"],
                "traces": traces,
                "frame_ms": round(frame, 2),
            }
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("wide")
    parser.add_argument("today")
    parser.add_argument("probe")
    parser.add_argument("moves", help="ray_moves' paths.json (treelet visits a ray)")
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    wide = json.loads(Path(arguments.wide).read_text())
    today = json.loads(Path(arguments.today).read_text())
    probe = json.loads(Path(arguments.probe).read_text())
    moves = json.loads(Path(arguments.moves).read_text())
    costs = operation_costs(probe)
    result = {"costs": costs, "scenes": {}}
    for scene in SCENES:
        measured = today["scenes"][scene]
        v100 = measured["v100"]
        entry = {
            "today_ms": {t: measured["commands_ms"][t] for t in (*TRACES, "frame")},
            "today_cycles_per_ray": {
                t: round(
                    measured["traces"][t]["walk_ms"]["mean"]
                    * CLOCK_KHZ
                    * measured["traces"][t]["cores_used"]
                    / v100["rays"][f"rays_{t}"]
                )
                for t in TRACES
            },
            "v100_ms": {
                "bounce": round(
                    v100["rays"]["rays_bounce"]
                    / v100["mray_per_second"]["bounce"]
                    / 1e3,
                    2,
                ),
                "frame": round(v100["frame_ms"], 2),
            },
            "predicted": predict(wide, today, costs, moves, scene),
            "boxes_free": predict(wide, today, costs, moves, scene, boxes_free=True),
        }
        result["scenes"][scene] = entry
        print(f"{scene}: today {entry['today_ms']}, V100 {entry['v100_ms']}")
        print(f"  today's walk cycles a ray: {entry['today_cycles_per_ray']}")
        for kind in ("predicted", "boxes_free"):
            for name, row in entry[kind].items():
                traces = row["traces"]
                print(
                    f"  {kind:10s} {name:9s} ({row['method']}): bounce"
                    f" {traces['bounce']['ms']:7.1f} ms"
                    f" ({traces['bounce']['cycles_per_ray']} a ray), second shadow"
                    f" {traces['shadow2']['ms']:6.1f} ms, frame {row['frame_ms']:6.1f}"
                )
    if arguments.json:
        Path(arguments.json).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
