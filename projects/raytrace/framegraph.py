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

from komorebi import power  # noqa: E402
from komorebi.engines.graph import Graph  # noqa: E402
from projects.raytrace import card, frame, packets  # noqa: E402
from projects.raytrace.reference import harness  # noqa: E402
from reference import native, raytrace, scenes  # noqa: E402

K = packets.K
VEC = packets.VEC
F = np.float32
REQ = 21  # the graph's request id
MODES = ("graph", "hostfed")


def build(address_of, directory, profile_addresses, workers, deadline_s):
    """The frame's commands as a kept graph, each after the commands that write its
    inputs (SHADOW and BOUNCE may run together); bindings `const` and `colour`."""
    graph = Graph(deadline_microseconds=round(deadline_s * 1e6), keep=True, req=REQ)
    constants, colour = graph.bindings("const", "colour")
    address_of = dict(address_of, colour=colour)
    producer = {}
    for step, profile in zip(frame.STEPS, profile_addresses):
        words = frame.step_words(step, address_of, directory, constants, profile)
        after = []
        for stream in step[2]:
            if stream in producer and producer[stream] not in after:
                after.append(producer[stream])
        node = graph.command(K[step[1]], group=(0, workers), args=words, after=after)
        for stream in step[3]:
            producer[stream] = node
    return graph


def cameras(scene, count):
    """count scenes whose cameras step forward along the view direction (the first is
    the scene's own)."""
    views = []
    for index in range(count):
        camera = scene["camera"].copy()
        camera[0:3] = (camera[0:3] + F(0.05 * index) * camera[9:12]).astype(F)
        views.append(dict(scene, camera=camera))
    return views


def run_modes(session, graph, image, tree_tensor, cases, modes, deadline_s):
    """One load: LOAD, then the graph per camera in each mode. Returns ({mode: [frame
    seconds]}, the kernel check, the host-fed commands per frame)."""
    run = session.load(card.program(len(image)), deadline=deadline_s)
    try:
        handle = run.submit(K["kHandlerLoad"], args=[tree_tensor.address, image.nbytes])
        run.wait([handle], timeout=10.0 + deadline_s)
        seconds, submitted = {mode: [] for mode in modes}, []
        if "graph" in modes:
            uploaded = run.upload(graph)
            for camera, bindings in enumerate(cases["graph"]):
                started = time.perf_counter()
                engine_run = (
                    run.start(uploaded, bindings)
                    if camera == 0
                    else run.restart(REQ, bindings)
                )
                event = run.wait_engine(engine_run, timeout=10.0 + 8 * deadline_s)
                seconds["graph"].append(time.perf_counter() - started)
                card.log(
                    f"graph camera {camera}: {event.kind} in "
                    f"{seconds['graph'][-1]:.4f}s"
                )
            run.release(uploaded)
        for camera, bindings in enumerate(cases.get("hostfed", [])):
            started = time.perf_counter()
            result = run.run_hostfed(graph, bindings, timeout=10.0 + 8 * deadline_s)
            seconds["hostfed"].append(time.perf_counter() - started)
            submitted.append(len(result.commands))
            card.log(
                f"hostfed camera {camera}: {result.status}, "
                f"{len(result.commands)} commands"
            )
            if result.status != "ended":
                raise RuntimeError(f"host-fed graph: {result.status} {result.error}")
    finally:
        run.stop()
    ran = card.Ran([], run.result(), run.state(), 0.0)
    return seconds, card.check_run(ran, card.workers(session)), submitted


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="tiny")
    parser.add_argument(
        "--tiles", type=int, nargs="*", default=None, help="32 x 32 tiles"
    )
    parser.add_argument("--cameras", type=int, default=2)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--deadline", type=float, default=5.0, help="per command, seconds"
    )
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    data = native.data_dir()
    tree = harness.ensure_tree(arguments.scene)
    scene = scenes.load_scene(data / f"{arguments.scene}.npz")
    image = packets.tree_image(tree, scene)
    views = cameras(scene, arguments.cameras)
    report = {
        "scene": arguments.scene,
        "tiles": arguments.tiles,
        "cameras": arguments.cameras,
    }
    with card.open_session() as session:
        workers = card.workers(session)
        pixels, directory, desc = packets.frame_layout(arguments.tiles, workers)
        packet_count = len(pixels)
        stream_names = {
            stream
            for _, _, inputs, outputs in frame.STEPS
            for stream in inputs + outputs
        } - {"colour"}
        zeros = {
            stream: np.zeros((packet_count, frame.vectors(stream) * VEC))
            for stream in stream_names
        }
        tensors = {
            stream: card.tensor(session, packets.group(page))
            for stream, page in zeros.items()
        }
        tensors["desc"] = card.tensor(session, packets.group(desc))
        directory_tensor = card.tensor(session, directory)
        tree_tensor = card.tensor(session, image)
        constants_tensors = [
            card.tensor(session, packets.frame_constants(view)) for view in views
        ]
        blank = packets.group(np.zeros((packet_count, frame.vectors("colour") * VEC)))
        colours = {
            (mode, camera): card.tensor(session, blank)
            for mode in arguments.modes
            for camera in range(len(views))
        }
        profile_page = np.zeros((workers, K["kOutPageBytes"] // 4), dtype=np.uint32)
        profile_tensors = [card.tensor(session, profile_page) for _ in frame.STEPS]
        address_of = {stream: tensor.address for stream, tensor in tensors.items()}
        graph = build(
            address_of,
            directory_tensor.address,
            [tensor.address for tensor in profile_tensors],
            workers,
            arguments.deadline,
        )
        cases = {
            mode: [
                {"const": constants.address, "colour": colours[mode, camera].address}
                for camera, constants in enumerate(constants_tensors)
            ]
            for mode in arguments.modes
        }
        report.update(
            packets=packet_count,
            aiclk_mhz=card.aiclk_mhz(),
            load_avg=os.getloadavg()[0],
        )
        report["upload_bytes"] = len(graph.encode())
        seconds, run, submitted = run_modes(
            session,
            graph,
            image,
            tree_tensor,
            cases,
            arguments.modes,
            arguments.deadline,
        )
        report["run"], report["hostfed_commands"] = run, submitted
        report["frame_ms"] = {
            mode: [round(duration * 1e3, 3) for duration in durations]
            for mode, durations in seconds.items()
        }
        keys = sorted(colours)
        rows = card.read_each(
            session,
            [colours[key] for key in keys],
            [f"colour {mode} {camera}" for mode, camera in keys],
        )
        got = {
            key: frame.lanes_of(
                packets.ungroup(page, 3, packet_count), "colour"
            ).T.view(F)
            for key, page in zip(keys, rows)
        }
    flat_pixels = pixels.reshape(-1)
    checks = []
    for camera, view in enumerate(views):
        if arguments.tiles is None and camera == 0:
            want = frame.reference_frame(arguments.scene)[1]["image"].reshape(-1, 3)[
                flat_pixels
            ]
        else:
            want = raytrace.render(view, tree, flat_pixels)["image"]
        row = {"camera": camera}
        for mode in arguments.modes:
            row[mode] = frame.differ(got[mode, camera], want)
        if len(arguments.modes) == 2:
            row["graph_vs_hostfed"] = frame.differ(
                got["graph", camera], got["hostfed", camera]
            )
        checks.append(row)
    report["image_checks"] = checks
    moved = len(views) < 2 or frame.differ(
        got[arguments.modes[0], 0], got[arguments.modes[0], 1]
    )
    report["cameras_differ"] = int(moved)
    ok = run["kernel_error"] == 0 and set(run["cores_ended"].values()) == {workers}
    ok &= all(
        value == 0 for row in checks for key, value in row.items() if key != "camera"
    )
    ok &= bool(moved) and all(count == len(frame.STEPS) for count in submitted)
    report["ok"] = bool(ok)
    print(json.dumps(power.clocked(report)), flush=True)
    if arguments.json:
        with open(arguments.json, "w") as file:
            json.dump(report, file, indent=1)
    return report["ok"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
