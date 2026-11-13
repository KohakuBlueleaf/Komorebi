"""The frame as one kept graph: uploaded once, run for each camera with the camera's
constant page and the colour as bindings and no host step inside a frame, in the
dispatcher and host-fed. Each camera's image is checked against numpy's and the
two runs against each other. Prints one JSON line with `ok`; exits 1 without it.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from komorebi.engines.graph import Graph  # noqa: E402
from projects.raytrace import card, frame, packets  # noqa: E402
from projects.raytrace.reference import harness  # noqa: E402
from ref import native, raytrace, scenes  # noqa: E402

K = packets.K
VEC = packets.VEC
F = np.float32
REQ = 21  # the graph's request id
MODES = ("graph", "hostfed")


def build(addr, directory, profs, workers, deadline_s):
    """The frame's commands as a kept graph, each after the commands that write its
    inputs (SHADOW and BOUNCE may run together); bindings `const` and `colour`."""
    g = Graph(deadline_microseconds=round(deadline_s * 1e6), keep=True, req=REQ)
    const, colour = g.bindings("const", "colour")
    addr = dict(addr, colour=colour)
    made = {}
    for step, prof in zip(frame.STEPS, profs):
        words = frame.step_words(step, addr, directory, const, prof)
        after = []
        for s in step[2]:
            if s in made and made[s] not in after:
                after.append(made[s])
        node = g.command(K[step[1]], group=(0, workers), args=words, after=after)
        for s in step[3]:
            made[s] = node
    return g


def cameras(scene, n):
    """n scenes whose cameras step forward along the view direction (the first is the
    scene's own)."""
    out = []
    for k in range(n):
        cam = scene["camera"].copy()
        cam[0:3] = (cam[0:3] + F(0.05 * k) * cam[9:12]).astype(F)
        out.append(dict(scene, camera=cam))
    return out


def run_modes(session, g, image, tree_t, cases, modes, deadline_s):
    """One load: LOAD, then the graph per camera in each mode. Returns ({mode: [frame
    seconds]}, the kernel check, the host-fed commands per frame)."""
    run = session.load(card.program(len(image)), deadline=deadline_s)
    try:
        handle = run.submit(K["kHandlerLoad"], args=[tree_t.address, image.nbytes])
        run.wait([handle], timeout=10.0 + deadline_s)
        seconds, submitted = {m: [] for m in modes}, []
        if "graph" in modes:
            up = run.upload(g)
            for k, b in enumerate(cases["graph"]):
                t0 = time.perf_counter()
                t = run.start(up, b) if k == 0 else run.restart(REQ, b)
                event = run.wait_engine(t, timeout=10.0 + 8 * deadline_s)
                seconds["graph"].append(time.perf_counter() - t0)
                card.log(
                    f"graph camera {k}: {event.kind} in {seconds['graph'][-1]:.4f}s"
                )
            run.release(up)
        for k, b in enumerate(cases.get("hostfed", [])):
            t0 = time.perf_counter()
            res = run.run_hostfed(g, b, timeout=10.0 + 8 * deadline_s)
            seconds["hostfed"].append(time.perf_counter() - t0)
            submitted.append(len(res.commands))
            card.log(f"hostfed camera {k}: {res.status}, {len(res.commands)} commands")
            if res.status != "ended":
                raise RuntimeError(f"host-fed graph: {res.status} {res.error}")
    finally:
        run.stop()
    ran = card.Ran([], run.result(), run.state(), 0.0)
    return seconds, card.check_run(ran, card.workers(session)), submitted


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", default="tiny")
    p.add_argument("--tiles", type=int, nargs="*", default=None, help="32 x 32 tiles")
    p.add_argument("--cameras", type=int, default=2)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--deadline", type=float, default=5.0, help="per command, seconds")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    data = native.data_dir()
    tree = harness.ensure_tree(args.scene)
    scene = scenes.load_scene(data / f"{args.scene}.npz")
    image = packets.tree_image(tree, scene)
    views = cameras(scene, args.cameras)
    report = {"scene": args.scene, "tiles": args.tiles, "cameras": args.cameras}
    with card.open_session() as session:
        workers = card.workers(session)
        pix, directory, desc = packets.frame_layout(args.tiles, workers)
        n = len(pix)
        names = {s for _, _, ins, outs in frame.STEPS for s in ins + outs} - {"colour"}
        zeros = {s: np.zeros((n, frame.vectors(s) * VEC)) for s in names}
        tensors = {s: card.tensor(session, packets.group(z)) for s, z in zeros.items()}
        tensors["desc"] = card.tensor(session, packets.group(desc))
        dir_t = card.tensor(session, directory)
        tree_t = card.tensor(session, image)
        consts = [card.tensor(session, packets.frame_constants(v)) for v in views]
        blank = packets.group(np.zeros((n, frame.vectors("colour") * VEC)))
        colours = {
            (m, k): card.tensor(session, blank)
            for m in args.modes
            for k in range(len(views))
        }
        prof_rows = np.zeros((workers, K["kOutPageBytes"] // 4), dtype=np.uint32)
        profs = [card.tensor(session, prof_rows) for _ in frame.STEPS]
        addr = {s: t.address for s, t in tensors.items()}
        g = build(
            addr,
            dir_t.address,
            [t.address for t in profs],
            workers,
            args.deadline,
        )
        cases = {
            m: [
                {"const": c.address, "colour": colours[m, k].address}
                for k, c in enumerate(consts)
            ]
            for m in args.modes
        }
        report.update(
            packets=n, aiclk_mhz=card.aiclk_mhz(), load_avg=os.getloadavg()[0]
        )
        report["upload_bytes"] = len(g.encode())
        seconds, run, submitted = run_modes(
            session, g, image, tree_t, cases, args.modes, args.deadline
        )
        report["run"], report["hostfed_commands"] = run, submitted
        report["frame_ms"] = {
            m: [round(s * 1e3, 3) for s in v] for m, v in seconds.items()
        }
        keys = sorted(colours)
        rows = card.read_each(
            session, [colours[k] for k in keys], [f"colour {m} {c}" for m, c in keys]
        )
        got = {
            k: frame.lanes_of(packets.ungroup(r, 3, n), "colour").T.view(F)
            for k, r in zip(keys, rows)
        }
    flat = pix.reshape(-1)
    checks = []
    for k, view in enumerate(views):
        if args.tiles is None and k == 0:
            want = frame.reference_frame(args.scene)[1]["image"].reshape(-1, 3)[flat]
        else:
            want = raytrace.render(view, tree, flat)["image"]
        row = {"camera": k}
        for m in args.modes:
            row[m] = frame.differ(got[m, k], want)
        if len(args.modes) == 2:
            row["graph_vs_hostfed"] = frame.differ(got["graph", k], got["hostfed", k])
        checks.append(row)
    report["image_checks"] = checks
    moved = len(views) < 2 or frame.differ(got[args.modes[0], 0], got[args.modes[0], 1])
    report["cameras_differ"] = int(moved)
    ok = run["kernel_error"] == 0 and set(run["cores_ended"].values()) == {workers}
    ok &= all(v == 0 for row in checks for key, v in row.items() if key != "camera")
    ok &= bool(moved) and all(i == len(frame.STEPS) for i in submitted)
    report["ok"] = bool(ok)
    print(json.dumps(report), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    return report["ok"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
