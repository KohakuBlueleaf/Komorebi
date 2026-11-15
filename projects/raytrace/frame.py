"""A frame on the card: generation, the four traces, the shading and the colour as
commands, each core keeping its tiles' packets; the colour (--stages: every stage) is
checked against numpy, --reps reruns the frame, --pictures DIR writes PNGs and JSON.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from komorebi import power  # noqa: E402
from projects.raytrace import card, packets, planner  # noqa: E402
from projects.raytrace import ties as ties_module  # noqa: E402
from projects.raytrace.reference import harness  # noqa: E402
from projects.raytrace.run import reference_frame  # noqa: E402
from reference import bvh, native, png, raytrace, scenes  # noqa: E402

K = packets.K
VEC = packets.VEC
F = np.float32
# Each step: name, handler, input streams, output streams (in the kernels' order).
STEPS = (
    ("gen", "kHandlerGen", ("desc",), ("primary_pk",)),
    ("primary", "kHandlerTrace", ("primary_pk",), ("primary_hit",)),
    (
        "shade1",
        "kHandlerShade1",
        ("primary_pk", "primary_hit"),
        ("shadow_pk", "bounce_pk", "state1"),
    ),
    ("shadow", "kHandlerTrace", ("shadow_pk",), ("shadow_hit",)),
    ("bounce", "kHandlerTrace", ("bounce_pk",), ("bounce_hit",)),
    (
        "shade2",
        "kHandlerShade2",
        ("state1", "shadow_hit", "bounce_pk", "bounce_hit"),
        ("shadow2_pk", "state2"),
    ),
    ("shadow2", "kHandlerTrace", ("shadow2_pk",), ("shadow2_hit",)),
    ("image", "kHandlerImage", ("state2", "shadow2_hit"), ("colour",)),
)
ANY_HIT = ("shadow", "shadow2")
SORTED = ("bounce", "shadow2")  # traces whose rays --sort orders on each core first


def sorted_steps():
    """STEPS with each SORTED trace on its rays in the core's order (SORT before it,
    UNSORT of its hits after it)."""
    ordered = []
    for name, handler, inputs, outputs in STEPS:
        if name not in SORTED:
            ordered.append((name, handler, inputs, outputs))
            continue
        sorted_packets, sorted_hits = f"{name}_s_pk", f"{name}_s_hit"
        order = f"order_{name}"
        ordered.append(
            (f"sort_{name}", "kHandlerSort", inputs, (sorted_packets, order))
        )
        ordered.append((name, handler, (sorted_packets,), (sorted_hits,)))
        ordered.append(
            (f"unsort_{name}", "kHandlerSort", (sorted_hits, order), outputs)
        )
    return tuple(ordered)


def sort_key(tree):
    """SORT's words a6 to a9: the root box's low corner in fixed point (fraction_bits
    fraction bits, the largest coordinate under 2^30) and the shift to
    2^kSortCellBits cells."""
    low, high = (box[0].astype(np.float64) for box in bvh.node_boxes(tree["nodes"][:1]))
    largest = max(np.abs(low).max(), np.abs(high).max(), 1e-30)
    fraction_bits = int(np.clip(29 - np.ceil(np.log2(largest)), 0, 60))
    low_fixed = np.floor(low * 2.0**fraction_bits).astype(np.int64)
    span = int(np.ceil((high - low).max() * 2.0**fraction_bits)) + 1
    shift = max(0, span.bit_length() - K["kSortCellBits"])
    return [int(value) & 0xFFFFFFFF for value in low_fixed] + [
        fraction_bits | shift << 8
    ]


STREAM_WORDS = (1, 2, 5, 6, 7, 8, 9)  # command word of the k-th stream (stream_arg)


def vectors(stream):
    """Vectors of a stream's page per packet."""
    if stream.endswith("_pk"):
        return K["kPkVectors"]
    if stream.endswith("_hit"):
        return packets.hit_vectors(stream.split("_")[0] in ANY_HIT)
    if stream.startswith("order_"):
        return 1
    named = {"desc": "kDescVectors", "state1": "kS1Vectors", "state2": "kS2Vectors"}
    return K[named.get(stream, "kColVectors")]


def step_words(step, address_of, directory, constants, profile):
    """A step's ten command words; `address_of` maps a stream to its address (or a
    graph's register)."""
    name, handler, inputs, outputs = step
    words = [0] * 10
    words[0], words[4] = directory, profile
    if handler == "kHandlerTrace":
        words[1], words[2] = address_of[inputs[0]], address_of[outputs[0]]
        words[3] = K["kModeAny"] if name in ANY_HIT else K["kModeNearest"]
    elif handler == "kHandlerSort":
        streams = inputs + outputs
        order = next(stream for stream in streams if stream.startswith("order_"))
        source, destination = (stream for stream in streams if stream != order)
        words[1], words[2], words[3] = (
            address_of[source],
            address_of[destination],
            address_of[order],
        )
        words[5] = vectors(source) if name.startswith("unsort") else 0
    else:
        words[3] = constants
        for index, stream in enumerate(inputs + outputs):
            words[STREAM_WORDS[index]] = address_of[stream]
    return words


def commands(
    tensors,
    directory_tensors,
    constants_tensor,
    profiles,
    treelets=None,
    steps=STEPS,
    key=(),
):
    """Every step's commands (handler, ten words), in submit order. With treelets (the
    tensor and four words) a trace has a command per chunk directory_tensors[1:],
    others directory_tensors[0]."""
    address_of = {stream: tensor.address for stream, tensor in tensors.items()}
    constants_address = constants_tensor.address
    entries = []
    for step, step_profiles in zip(steps, profiles):
        chunks = (
            directory_tensors[1:]
            if treelets and step[1] == "kHandlerTrace"
            else directory_tensors[:1]
        )
        for directory_tensor, profile_tensor in zip(chunks, step_profiles):
            words = step_words(
                step,
                address_of,
                directory_tensor.address,
                constants_address,
                profile_tensor.address,
            )
            handler = K[step[1]]
            if treelets and step[1] == "kHandlerTrace":
                handler = K["kHandlerTreelets"]
                words[5:10] = [treelets[0].address, *treelets[1]]
            if step[1] == "kHandlerSort":
                words[6:10] = key
            entries.append((handler, words))
    return entries


def load_cut(scene, tree, kib):
    """The scene's treelets of at most `kib` KiB: the stored cut, else a new one."""
    path = native.data_dir() / f"{scene}_treelets_{kib}k.npz"
    if path.exists():
        with np.load(path) as archive:
            return {name: archive[name] for name in archive.files}
    return bvh.cut_treelets(tree, kib * 1024)


def lanes_of(pages, stream):
    """A stream's pages (P, vectors * 32) as (vectors, P * 32): lane values in order."""
    vector_count = vectors(stream)
    return (
        pages.reshape(-1, vector_count, VEC)
        .transpose(1, 0, 2)
        .reshape(vector_count, -1)
    )


def expected(scene, tree, pixels):
    """The reference's values of every stage at the lanes' pixels (flat)."""
    origins, directions = raytrace.camera_rays(scene["camera"], pixels)
    hits = raytrace.trace(tree, origins, directions, raytrace.TMAX)
    hit_mask = hits["leaf"] >= 0
    hit_rows = {name: rows[hit_mask] for name, rows in hits.items()}
    points = origins[hit_mask] + directions[hit_mask] * hit_rows["t"][:, None]
    normals, albedo = raytrace.surface(scene, tree, hit_rows, directions[hit_mask])
    shadow_origins = points + normals * scene["eps"][0]
    to_light = scene["light"] - shadow_origins
    cosine = raytrace.dot(normals, to_light) * (
        F(1.0) / np.sqrt(raytrace.dot(to_light, to_light))
    )
    bounce_directions = raytrace.bounce_directions(pixels[hit_mask], normals)
    colour = raytrace.render(scene, tree, pixels)["image"]
    triangles = np.where(hit_mask, tree["leaf_tri"][np.maximum(hits["leaf"], 0)], -1)
    return {
        "origins": origins,
        "directions": directions,
        "inverse": raytrace.inverse_direction(directions),
        "triangle": triangles,
        "hit": hit_mask,
    }, {
        "shadow_origin": shadow_origins,
        "to_light": to_light,
        "cosine": cosine,
        "bounce_direction": bounce_directions,
        "albedo": albedo,
        "sky": raytrace.sky(directions),
        "colour": colour,
    }


def differ(left, right):
    """Lanes whose float32 bits differ (rows of a 2-D array count once)."""
    left, right = packets.bits(left), packets.bits(right)
    differing = left != right
    return int(differing.any(axis=1).sum() if differing.ndim == 2 else differing.sum())


def pictures(folder, scene, flat_pixels, colour, want, report):
    """The card's image, the reference's and the difference as PNGs, and frame.json."""
    folder.mkdir(parents=True, exist_ok=True)
    side = scenes.WIDTH
    card_image = np.zeros((side * side, 3), dtype=F)
    reference_image = np.zeros((side * side, 3), dtype=F)
    card_image[flat_pixels], reference_image[flat_pixels] = colour, want
    card_image = card_image.reshape(side, side, 3)
    reference_image = reference_image.reshape(side, side, 3)
    mask = (packets.bits(card_image) != packets.bits(reference_image)).any(axis=2)
    png.write_png(folder / f"{scene}_card.png", png.to_srgb8(card_image))
    png.write_png(folder / f"{scene}_reference.png", png.to_srgb8(reference_image))
    png.write_png(folder / f"{scene}_diff.png", png.diff_picture(card_image, mask))
    with open(folder / f"{scene}_frame.json", "w") as file:
        json.dump(report, file, indent=1)


SAMPLE_CORES = (0, 40, 80, 119)  # cores whose trace hit pages --profile keeps


def save_profile(
    path, session, raw, command_microseconds, directory, tensors, steps=STEPS
):
    """Every command's counters per core, the command times, the directory and the hit
    pages of SAMPLE_CORES' packets for each trace (first kHitAnyVectors vectors)."""
    arrays = {"command_microseconds": command_microseconds, "directory": directory}
    for (name, *_), rows in zip(steps, raw):
        arrays[f"prof_{name}"] = np.stack(rows)
    page_packets = K["kPagePackets"]
    for name, handler, _, outputs in steps:
        if handler != "kHandlerTrace":
            continue
        vector_count = vectors(outputs[0])
        parts = []
        for core in SAMPLE_CORES:
            first, count = (
                int(word) for word in directory[core, [K["kDirFirst"], K["kDirCount"]]]
            )
            first_page = first // page_packets
            end_page = (first + count - 1) // page_packets + 1
            (pages,) = session.read(
                tensors[outputs[0]], first=first_page, pages=end_page - first_page
            )
            hit_packets = pages.view(np.uint32).reshape(-1, vector_count, VEC)
            hit_packets = hit_packets[
                first
                - first_page * page_packets : first
                - first_page * page_packets
                + count,
                : K["kHitAnyVectors"],
            ]
            parts.append(hit_packets)
        arrays[f"hits_{name}"] = np.concatenate(parts)
    np.savez_compressed(path, **arrays)


def check_stages(got, scene, tree, pixels):
    """Lanes that differ from the reference, stage by stage."""
    primary, derived = expected(scene, tree, pixels)
    hit_mask = primary["hit"]
    primary_packets = lanes_of(got["primary_pk"], "primary_pk")
    primary_hits = lanes_of(got["primary_hit"], "primary_hit")
    shadow_packets = lanes_of(got["shadow_pk"], "shadow_pk")
    bounce_packets = lanes_of(got["bounce_pk"], "bounce_pk")
    state1 = lanes_of(got["state1"], "state1")
    colour = state1[K["kS1Col"] : K["kS1Col"] + 3].T
    wanted_colour = derived["sky"].copy()
    wanted_colour[hit_mask] = derived["albedo"]
    differences = {
        "gen_o": differ(
            primary_packets[K["kPkO"] : K["kPkO"] + 3].T.view(F), primary["origins"]
        ),
        "gen_d": differ(
            primary_packets[K["kPkD"] : K["kPkD"] + 3].T.view(F), primary["directions"]
        ),
        "gen_inv": differ(
            primary_packets[K["kPkInv"] : K["kPkInv"] + 3].T.view(F),
            primary["inverse"],
        ),
        "primary_tri": int(
            (primary_hits[K["kHitTri"]].view(np.int32) != primary["triangle"]).sum()
        ),
        "shade1_hit_flag": int(((state1[K["kS1Hit"]] != 0) != hit_mask).sum()),
        "shade1_colour": differ(colour.view(F), wanted_colour),
        "shade1_cos": differ(state1[K["kS1Cos"]].view(F)[hit_mask], derived["cosine"]),
        "shadow_o": differ(
            shadow_packets[K["kPkO"] : K["kPkO"] + 3].T.view(F)[hit_mask],
            derived["shadow_origin"],
        ),
        "shadow_d": differ(
            shadow_packets[K["kPkD"] : K["kPkD"] + 3].T.view(F)[hit_mask],
            derived["to_light"],
        ),
        "bounce_d": differ(
            bounce_packets[K["kPkD"] : K["kPkD"] + 3].T.view(F)[hit_mask],
            derived["bounce_direction"],
        ),
    }
    return differences


def planned(
    arguments,
    session,
    records,
    steps,
    per_step,
    treelets,
    directory,
    profiles,
    shape,
    report,
):
    """The frames driven by the X280 thread (planner.py): command microseconds per frame
    and step, the profile tensors of the last frame, the host's seconds, power
    samples."""
    tree_pages, sizes, workers, packet_count, packets_per_command = shape
    capacity = min(packets_per_command, planner.P["RP_COST_MAX"])
    entries = planner.entries_of(records, steps, per_step, treelets)
    table_words = planner.table(
        entries,
        directory,
        [
            step_profiles
            for step, step_profiles in zip(steps, profiles)
            if step[1] == "kHandlerTrace"
        ],
        workers,
    )
    program = planner.program(
        card.program(tree_pages, sizes),
        table_words,
        arguments.plan,
        arguments.reps,
        workers,
        packet_count,
        capacity,
        len(entries),
    )
    card.log(
        f"frame: {packet_count} packets, {len(entries)} table commands, "
        f"plan {arguments.plan}"
    )
    meter = card.Power()
    frames, status, state, samples = planner.run(
        session, program, arguments.reps, meter, card.log
    )
    report["plan"] = {"mode": arguments.plan, "cap": capacity, "thread": status}
    kinds = [entry[0] & 0xFF for entry in entries]
    frame_step_times, last_chunks = [], None
    for index, frame_log in enumerate(frames):
        chunks = [frame_log["chunks"][kind - 1] if kind else 1 for kind in kinds]
        chunks[0] = 1 if index == 0 else 0  # LOAD: the cold frame only
        recorded = np.asarray(frame_log["command_microseconds"], dtype=np.float64)
        if len(recorded) != sum(chunks):
            break
        step_microseconds = np.bincount(
            np.repeat(np.arange(len(kinds)), chunks), recorded
        )[1:]
        frame_step_times.append(step_microseconds)
        last_chunks = frame_log["chunks"]
    report["frames"] = [
        {
            "frame_ms": float(step_times.sum()) / 1000,
            "step_ms": (step_times / 1000).tolist(),
            "wall_ms": (
                frame_log["wall_microseconds"]
                - (frame_log["command_microseconds"][0] if index == 0 else 0)
            )
            / 1000,
            "plan_ms": [value / 1000 for value in frame_log["plan_microseconds"]],
            "chunks": frame_log["chunks"],
            "plan_parts_ms": [
                value / 1000 for value in frame_log["plan_parts_microseconds"]
            ],
        }
        for index, (step_times, frame_log) in enumerate(zip(frame_step_times, frames))
    ]
    if len(frame_step_times) > 1:
        report["repeated_frame_ms"] = (
            float(np.median([times.sum() for times in frame_step_times[1:]])) / 1000
        )
    done = (
        len(frames) == arguments.reps
        and len(frame_step_times) == arguments.reps
        and status["code"] == "ok"
    )
    ran = card.Ran([], status["kernel"], state, status["host_s"])
    report["run"] = card.check_run(ran, workers)
    report["run"].update(done=int(done), commands=1)
    report["host"] = {"commands": 0, "command_bytes": 0, "wall_s": status["host_s"]}
    kind_of = iter(kind for kind in kinds[1:] if kind)
    profiles_read = []
    for step, step_profiles in zip(steps, profiles):
        profiles_read.append(
            step_profiles[: last_chunks[next(kind_of) - 1]]
            if step[1] == "kHandlerTrace"
            else step_profiles
        )
    frame_step_times = (
        np.stack(frame_step_times) if frame_step_times else np.zeros((1, len(steps)))
    )
    last_microseconds = (
        np.asarray(frames[-1]["command_microseconds"]) if frames else np.zeros(0)
    )
    return (
        frame_step_times,
        profiles_read,
        status["host_s"],
        samples,
        last_microseconds,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="tiny")
    parser.add_argument(
        "--tiles", type=int, nargs="*", default=None, help="32 x 32 tiles"
    )
    parser.add_argument(
        "--stages", action="store_true", help="read and check every stage"
    )
    parser.add_argument("--reps", type=int, default=1, help="frames in the one load")
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--pictures", type=Path, default=None, help="folder for the PNGs"
    )
    parser.add_argument(
        "--treelets", type=int, default=0, help="KiB: treelets from DRAM"
    )
    parser.add_argument("--slots", type=int, default=4, help="treelet slots per core")
    parser.add_argument("--profile", default=None, help="NPZ: every core's counters")
    parser.add_argument("--sort", action="store_true", help="sort rays on each core")
    parser.add_argument(
        "--plan", choices=planner.MODES, default=None, help="X280 thread cuts traces"
    )
    arguments = parser.parse_args()
    if (arguments.sort or arguments.plan) and not arguments.treelets:
        parser.error("--sort and --plan need --treelets")
    if arguments.plan and planner.same_numbers():
        parser.error(f"plan.h differs from raytrace.h: {planner.same_numbers()}")
    steps = sorted_steps() if arguments.sort else STEPS
    data = native.data_dir()
    tree = harness.ensure_tree(arguments.scene)
    scene = scenes.load_scene(data / f"{arguments.scene}.npz")
    cut = (
        load_cut(arguments.scene, tree, arguments.treelets)
        if arguments.treelets
        else None
    )
    if cut is None:
        image = packets.tree_image(tree, scene)
    else:
        treelet_pages, treelet_table = packets.treelet_images(tree, cut, scene)
        image = packets.top_image(cut, treelet_table)
        slot_bytes = int(treelet_table[:, 1].max()) * K["kTreePageBytes"]
    report = {
        "scene": arguments.scene,
        "tiles": arguments.tiles,
        "treelets_kib": arguments.treelets,
    }
    report["sort"] = arguments.sort
    with card.open_session() as session:
        workers = card.workers(session)
        pixels, directory, desc = packets.frame_layout(arguments.tiles, workers)
        packet_count = len(pixels)
        report["packets"] = packet_count
        stream_names = {
            stream for _, _, inputs, outputs in steps for stream in inputs + outputs
        }
        tensors = {
            stream: card.tensor(
                session,
                packets.group(np.zeros((packet_count, vectors(stream) * VEC))),
            )
            for stream in sorted(stream_names)
        }
        tensors["desc"] = card.tensor(session, packets.group(desc))
        directory_chunks, sizes, treelets = [directory], None, None
        if cut is not None:
            longest = int(directory[:, K["kDirCount"]].max())
            # SORT works in the slot buffer (free between treelet commands).
            sort_bytes, shortfall = (
                packets.sort_bytes(longest) if arguments.sort else 0,
                0,
            )
            for _ in range(4):
                packets_per_command, slots, sizes = packets.plan_commands(
                    longest,
                    len(treelet_table),
                    slot_bytes,
                    len(image),
                    arguments.slots,
                    budget=packets.L1_BUDGET - shortfall,
                )
                if sort_bytes - sizes["slots"] <= shortfall:
                    break
                shortfall = sort_bytes - sizes["slots"]
            sizes["slots"] = max(sizes["slots"], sort_bytes)
            directory_chunks = packets.chunks(directory, packets_per_command)
            treelet_tensor = card.tensor(session, treelet_pages)
            treelet_words = [sizes["entries"], slot_bytes, slots, K["kPolicyOnce"]]
            treelets = (treelet_tensor, treelet_words)
            report["treelets"] = {
                "count": len(treelet_table),
                "packets_per_command": packets_per_command,
            }
            report["treelets"].update(
                sizes, slot_count=slots, chunks=len(directory_chunks)
            )
        directory_tensors = [
            card.tensor(session, chunk)
            for chunk in [directory] + directory_chunks[1 if cut is None else 0 :]
        ]
        constants_tensor = card.tensor(session, packets.frame_constants(scene))
        tree_tensor = card.tensor(session, image)
        profile_page = np.zeros((workers, K["kOutPageBytes"] // 4), dtype=np.uint32)
        per_step = [
            (
                len(directory_chunks)
                if cut is not None and step[1] == "kHandlerTrace"
                else 1
            )
            for step in steps
        ]
        chunk_counts = [
            planner.P["RP_CHUNKS"] if arguments.plan else count for count in per_step
        ]
        chunk_counts = [
            count if step[1] == "kHandlerTrace" else 1
            for count, step in zip(chunk_counts, steps)
        ]
        profiles = [
            [card.tensor(session, profile_page) for _ in range(count)]
            for count in chunk_counts
        ]
        records = [(K["kHandlerLoad"], [tree_tensor.address, image.nbytes])]
        sort_words = sort_key(tree) if arguments.sort else ()
        frame_commands = commands(
            tensors,
            directory_tensors,
            constants_tensor,
            profiles,
            treelets,
            steps,
            sort_words,
        )
        meter = card.Power()
        report["power_idle_w"] = meter.idle()
        report["aiclk_mhz"] = card.aiclk_mhz()
        report["x280_mhz"] = session.pll_mhz
        report["load_avg"] = os.getloadavg()[0]
        if arguments.plan:
            frame_step_times, profiles_read, host_s, samples, command_microseconds = (
                planned(
                    arguments,
                    session,
                    records + frame_commands,
                    steps,
                    per_step,
                    treelets,
                    directory,
                    profiles,
                    (len(image), sizes, workers, packet_count, packets_per_command),
                    report,
                )
            )
        else:
            records += frame_commands * arguments.reps
            card.log(f"frame: {packet_count} packets, {len(records)} commands")
            ran, samples = card.run_commands(
                session, records, len(image), power=meter, sizes=sizes
            )
            report["run"] = card.check_run(ran, workers)
            # A step's time is its commands' (one per directory chunk) summed.
            step_of = np.repeat(np.arange(len(steps)), per_step)
            per_repetition = card.command_microseconds(ran)[1:].reshape(
                arguments.reps, len(step_of)
            )
            frame_step_times = np.stack(
                [
                    np.bincount(step_of, repetition, len(steps))
                    for repetition in per_repetition
                ]
            )
            report["host"] = {**ran.host, "wall_s": ran.host_s}
            host_s, profiles_read = ran.host_s, profiles
            command_microseconds = card.command_microseconds(ran)
        run = report["run"]
        report["frame_microseconds_median"] = float(
            np.median(frame_step_times.sum(axis=1))
        )
        report["frame_ms"] = report["frame_microseconds_median"] / 1000
        if arguments.plan:
            report["frame_ms"] = float(frame_step_times[0].sum()) / 1000
        watts = [
            sample_watts
            for sample_time, sample_watts in samples
            if sample_time >= samples[-1][0] - host_s
        ]
        report["power_mean_w"] = float(np.mean(watts)) if watts else None
        report["steps"] = []
        raw = [
            card.read_each(session, step_profiles, [step[0]] * len(step_profiles))
            for step, step_profiles in zip(steps, profiles_read)
        ]
        rows = [sum(row.astype(np.uint64) for row in per_core) for per_core in raw]
        if arguments.profile:
            save_profile(
                arguments.profile,
                session,
                raw,
                command_microseconds,
                directory,
                tensors,
                steps,
            )
        for (name, *_), event_microseconds, profile_row in zip(
            steps, frame_step_times.T, rows
        ):
            counters = card.profile_rows(profile_row.astype(np.uint64), workers)
            step = {"step": name, "microseconds": float(np.median(event_microseconds))}
            step["errors"] = int((counters["error"] != 0).sum())
            step["busy_max"] = int(counters["total"].max())
            used = counters["packets"] > 0
            for counter in ("load", "box", "out", "wait_in", "wait_out", "total"):
                per_packet = counters[counter][used].sum() / max(
                    counters["packets"][used].sum(), 1
                )
                step[f"{counter}_per_packet"] = round(float(per_packet), 1)
            report["steps"].append(step)
        report["command_ms"] = {
            entry["step"]: entry["microseconds"] / 1000 for entry in report["steps"]
        }
        wanted = sorted(stream_names) if arguments.stages else ["colour"]
        got = card.read_each(session, [tensors[stream] for stream in wanted], wanted)
        got = {
            stream: packets.ungroup(pages, vectors(stream), packet_count)
            for stream, pages in zip(wanted, got)
        }
    flat_pixels = pixels.reshape(-1)
    colour = lanes_of(got["colour"], "colour").T.view(F)
    if arguments.tiles is None:
        name, reference = reference_frame(arguments.scene)
        want = reference["image"].reshape(-1, 3)[flat_pixels]
    else:
        name, want = "numpy", raytrace.render(scene, tree, flat_pixels)["image"]
    report["image_check"] = {"against": name, "pixels": len(flat_pixels)}
    differing = (packets.bits(colour) != packets.bits(want)).any(axis=1)
    ties, wrong = ties_module.classify(
        scene, tree, flat_pixels[differing], colour[differing]
    )
    report["image_check"]["differs"], report["image_check"]["ties"] = wrong, ties
    report["image_check"]["max_abs_error"] = float(np.abs(colour - want).max())
    report["image_check"]["first_pixels"] = flat_pixels[differing][:64].tolist()
    if arguments.stages:
        report["stages"] = check_stages(got, scene, tree, flat_pixels)
    ok = card.ran_cleanly(run, workers) and report["image_check"]["differs"] == 0
    ok &= all(entry["errors"] == 0 for entry in report["steps"])
    ok &= all(value == 0 for value in report.get("stages", {}).values())
    report["ok"] = bool(ok)
    print(json.dumps(power.clocked(report)), flush=True)
    if arguments.pictures is not None:
        pictures(arguments.pictures, arguments.scene, flat_pixels, colour, want, report)
    if arguments.json:
        with open(arguments.json, "w") as file:
            json.dump(report, file, indent=1)
    return report["ok"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
