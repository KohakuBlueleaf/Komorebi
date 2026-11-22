"""The thread-driven frame as a loop a host drives: the scene loaded once; for each
camera one FRAME command, and the 8-bit image read from the display tensor or, with
`--push`, pushed by the thread into the host channel. Prints one JSON line; exits 1
unless `ok`."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.bench import power
from komorebi.program.thread import TIMER_HZ
from projects.raster import card, card_thread, lists, passes, run
from projects.raytrace import framegraph
from reference import png, raster

F, SHOW = card_thread.F, card_thread.SHOW
CLUSTER = card_thread.CLUSTER
TILE = 32
PUSH_TIMEOUT_S = 2.0  # a frame's wait for the push's sequence words
PUSH_SPIN_S = 20e-3  # its looks without a pause: longer than a frame
WIDE = raster.TILES_X  # tiles a row


def show_table():
    """SHOW's block: word k (1 to 255) the bits of the smallest float32 whose 8-bit
    value by png.to_srgb8 is k (the rule is monotonic in a float32's bits on [0, 1]);
    word 0 is never read (the bits of -0)."""
    wanted = np.arange(1, 256)
    low = np.zeros(255, dtype=np.int64)
    high = np.full(255, np.float32(1.0).view(np.uint32), dtype=np.int64)
    while (low < high).any():
        middle = (low + high) // 2
        value = png.to_srgb8(middle.astype(np.uint32).view(np.float32))
        reached = value >= wanted
        high = np.where(reached, middle, high)
        low = np.where(reached, low, middle + 1)
    table = np.zeros(SHOW["kShowTableWords"], dtype=np.uint32)
    table[0] = 0x80000000
    table[1:] = low
    if not SHOW["kShowBucketBase"] < low[0] <= low[-1] < SHOW["kShowOne"]:
        raise ValueError("the thresholds leave the writer's buckets")
    return table


def show_bytes(table, colour):
    """The writer's conversion on the host, step for step (show_writer.hpp show_byte):
    each float32's bits as a signed word; under the buckets 0, from 1.0 255, else the
    bucket's value stepped over the thresholds inside it."""
    signed = np.ascontiguousarray(colour, dtype=np.float32).view(np.int32).ravel()
    thresholds = table.view(np.int32)
    base, one = SHOW["kShowBucketBase"], SHOW["kShowOne"]
    shift, last = SHOW["kShowBucketShift"], len(table) - 1
    starts = base + (np.arange((one - base) >> shift, dtype=np.int64) << shift)
    buckets = np.searchsorted(thresholds[1:], starts, side="right")
    out = np.where(signed < base, 0, last).astype(np.int64)
    inside = np.flatnonzero((signed >= base) & (signed < one))
    index = buckets[(signed[inside].astype(np.int64) - base) >> shift]
    while True:
        step = (index < last) & (
            signed[inside] >= thresholds[np.minimum(index + 1, last)]
        )
        if not step.any():
            break
        index = index + step
    out[inside] = index
    return out.astype(np.uint8).reshape(np.shape(colour))


def table_check(table):
    """Values on both sides of every threshold, the clamp's ends and random colours:
    the table's bytes against png.to_srgb8, the number that differ."""
    bits = table[1:].astype(np.int64)
    edges = np.concatenate([bits, bits - 1, bits + 1]).astype(np.uint32)
    special = np.array(
        [0.0, -0.0, -1.0, 1.0, 1.5, 1e30, -1e30, 1e-30, 0.05, 0.08], dtype=np.float32
    )
    rng = np.random.default_rng(7)
    values = np.concatenate(
        [
            edges.view(np.float32),
            special,
            rng.random(1 << 20, dtype=np.float32),
            rng.uniform(-0.5, 1.5, 1 << 16).astype(np.float32),
        ]
    )
    return int((show_bytes(table, values) != png.to_srgb8(values)).sum())


def untile(pages, image):
    """The display's pages (tile = ty * 32 + tx; 32 rows of 32 pixels of 4 bytes) into
    `image` (1024, 1024, 4): one reshape and one transpose."""
    image.reshape(WIDE, TILE, WIDE, TILE, 4)[...] = pages.reshape(
        WIDE, WIDE, TILE, TILE, 4
    ).transpose(0, 2, 1, 3, 4)
    return image


def expected(view):
    """The reference's 8-bit image of a camera: numpy's colour by png's rule, alpha
    255."""
    colour = raster.rasterize(view)["colour"]
    image = np.full(colour.shape[:2] + (4,), 255, dtype=np.uint8)
    image[..., :3] = png.to_srgb8(colour)
    return image


def figures(values):
    """Median, 5th and 95th percentile of a list of milliseconds."""
    values = np.asarray(values, dtype=np.float64)
    return {
        "median": round(float(np.median(values)), 4),
        "p5": round(float(np.percentile(values, 5)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
    }


def times_rows(raw, frames):
    """The thread's rows of the frames served (the last RF_TIMES_ROWS), in order."""
    rows = np.frombuffer(raw, dtype=np.uint32)[
        : F["RF_TIMES_ROWS"] * F["RF_TIMES_ROW_WORDS"]
    ].reshape(F["RF_TIMES_ROWS"], F["RF_TIMES_ROW_WORDS"])
    kept = min(frames, F["RF_TIMES_ROWS"])
    numbers = np.arange(frames - kept, frames)
    return rows[numbers % F["RF_TIMES_ROWS"]], numbers


def loop(args):
    scene, _ = run.load(args.scene)
    blocks = passes.mesh_blocks(scene)
    views = framegraph.cameras(scene, args.frames)
    table = show_table()
    report = {
        "scene": args.scene,
        "frames": args.frames,
        "path": "push" if args.push else "read",
    }
    report["table_differs"] = table_check(table)
    checked = sorted({0, args.frames // 2, args.frames - 1})
    kept = {}
    with km.Session(log=card.log) as session:
        workers = card.workers(session)
        started = time.perf_counter()
        slots = args.list_slots or 2 * len(scene["indices"]) + 65536  # card_thread's
        uploaded = card_thread.tensors(session, blocks, workers, slots)
        display = card.tensor(
            session, np.zeros((lists.TILES, SHOW["kShowPageBytes"] // 4))
        )
        show_profile = card.tensor(
            session, np.zeros((workers, SHOW["kShowProfBytes"] // 4))
        )
        session.device.finish()
        words = card_thread.words(uploaded, len(blocks), workers, workers, 4)
        words[F["RF_W_REPEAT"]] = 0  # serve FRAME commands
        words[F["RF_W_DISPLAY"]] = display.address
        words[F["RF_W_SHOWPROF"]] = show_profile.address
        report["upload_s"] = round(time.perf_counter() - started, 3)
        started = time.perf_counter()
        running = session.load(
            card_thread.program(scene, words, workers, table), deadline=1.0
        )
        report["load_s"] = round(time.perf_counter() - started, 3)
        transfer = None
        if args.push:
            transfer = session.channel_transfer(display, push=True)
            entry = transfer.entry
            push_words = np.zeros(F["RF_PUSH_BYTES"] // 4, dtype=np.uint32)
            push_words[F["RF_PUSH_HEADER"]] = entry.handler | entry.field << 24
            push_words[F["RF_PUSH_GROUP"]] = entry.group[0] | entry.group[1] << 16
            push_words[F["RF_PUSH_ARGS"] : F["RF_PUSH_ARGS"] + 10] = entry.args
            running.thread.write("push", push_words)
        pages = np.empty((lists.TILES, SHOW["kShowPageBytes"]), dtype=np.uint8)
        image = np.empty((raster.HEIGHT, raster.WIDTH, 4), dtype=np.uint8)
        host, camera_ms, command_ms, read_ms, words_back = [], [], [], [], []
        collect_ms = []
        for index, view in enumerate(views):
            camera = np.asarray(view["camera"], dtype=np.float32).view(np.uint32)
            began = time.perf_counter()
            if transfer is None:
                running.thread.write("camera", camera)
                written = time.perf_counter()
                event = running.wait(
                    running.submit("frame", cluster=CLUSTER, result=True), timeout=10.0
                )
                commanded = time.perf_counter()
                session.read(display, out=pages)
                read = time.perf_counter()
                untile(pages, image)
            else:
                key = transfer.arm()
                running.thread.write("camera", camera)
                written = time.perf_counter()
                handle = running.submit(
                    "frame", cluster=CLUSTER, args=[key, 1], result=True
                )
                running.store()  # submit only queues it: the wait stores it otherwise
                arrived = transfer.wait(PUSH_TIMEOUT_S, PUSH_SPIN_S)
                commanded = read = time.perf_counter()
                if arrived:
                    untile(transfer.data, image)
            ended = time.perf_counter()
            if transfer is not None:  # FRAME's event, after the image (its word)
                event = running.wait(handle, timeout=10.0)
                collect_ms.append((time.perf_counter() - ended) * 1e3)
                if not arrived and event.kind == "done" and event.result == 0:
                    event = event._replace(result="push words missing")
            host.append((ended - began) * 1e3)
            camera_ms.append((written - began) * 1e3)
            command_ms.append((commanded - written) * 1e3)
            read_ms.append((read - commanded) * 1e3)
            word = event.result if event.kind == "done" else f"{event.kind}"
            words_back.append(word)
            if index in checked:
                kept[index] = image.copy()
            if word != 0:
                card.log(f"frame {index}: {event.kind}, word {word}")
                break
        running.stop()
        report["status"] = card.check_run(running, workers, 0, [])
        rows, numbers = times_rows(running.thread.read("times"), len(words_back))
        (profile_pages,) = card.read(session, show_profile)
    profile = profile_pages.reshape(workers, -1).astype(np.int64)
    ticks_ms = 1e3 / TIMER_HZ
    report["host_ms"] = figures(host)
    report["fps"] = round(1000.0 / report["host_ms"]["median"], 2)
    report["camera_write_ms"] = figures(camera_ms)
    report["command_ms"] = figures(command_ms)
    report["read_ms"] = figures(read_ms)
    if collect_ms:
        report["event_after_image_ms"] = figures(collect_ms)
        report["card_push_ms"] = round(
            float(np.median(rows[:, F["RF_T_PUSH"]])) * ticks_ms, 4
        )
    report["card_frame_ms"] = round(
        float(np.median(rows[:, F["RF_T_FRAME"]])) * ticks_ms, 4
    )
    report["card_show_ms"] = round(
        float(np.median(rows[:, F["RF_T_SHOW"]])) * ticks_ms, 4
    )
    report["heavy_tiles"] = rows[:, F["RF_O_HEAVY"]].tolist()[:4]
    report["heavy_tiles_max"] = int(rows[:, F["RF_O_HEAVY"]].max())
    report["items_max"] = int(rows[:, F["RF_O_ITEMS"]].max())
    report["rows_in_order"] = bool((rows[:, F["RF_T_NUMBER"]] == numbers).all())
    convert = profile[:, SHOW["kShowProfConvert"]]
    converted_rows = profile[:, SHOW["kShowProfRows"]]
    report["show_convert_cycles_per_tile"] = round(
        float(convert.sum() / max(converted_rows.sum() / TILE, 1)), 1
    )
    report["show_writer_max_cycles"] = int(profile[:, SHOW["kShowProfTotal"]].max())
    report["words_nonzero"] = sum(word != 0 for word in words_back)
    report["frames_run"] = len(words_back)
    differs = {}
    for index, got in kept.items():
        want = expected(views[index])
        differs[index] = int((got != want).any(axis=2).sum())
        if args.pictures:
            folder = Path(args.pictures)
            folder.mkdir(parents=True, exist_ok=True)
            png.write_png(folder / f"{args.scene}_{index:04d}.png", got[..., :3])
    report["pixels_differ"] = differs
    report["ok"] = bool(
        len(words_back) == args.frames
        and report["words_nonzero"] == 0
        and len(differs) == len(checked)
        and not any(differs.values())
        and report["status"]["ok"]
        and report["table_differs"] == 0
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="tiny")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument(
        "--push", action="store_true", help="the display pushed inside FRAME"
    )
    parser.add_argument("--pictures", default=None, help="PNG of the checked frames")
    parser.add_argument(
        "--list-slots", type=int, default=0, help="0: card_thread's, from the triangles"
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    report = loop(args)
    status = report.pop("status")
    report["status_ok"] = status["ok"]
    print(json.dumps(power.clocked(report)), flush=True)
    if args.json:
        with open(args.json, "w") as file:
            json.dump({**report, "status": status}, file, indent=1, default=str)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
