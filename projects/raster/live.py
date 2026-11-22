"""The thread-driven frame as a loop a host drives: a FRAME command a camera, the image
read back or (`--push`) pushed by the thread; `--shade` shaded (SHADE, masked ones a
layer), `--light` lit (SHADE's surface, LIGHT; `--sun AZ,EL`, `--sun-path N` move the
sun: a SUN pass). One JSON line; exit 1 unless ok."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.bench import power
from komorebi.program.thread import TIMER_HZ
from projects.raster import card, card_thread, light, lists, passes, run, shade
from projects.raytrace import framegraph
from reference import lights as light_data
from reference import native, png, raster

F, SHOW, SH = card_thread.F, card_thread.SHOW, card_thread.SH
CLUSTER = card_thread.CLUSTER
TILE = 32
PUSH_TIMEOUT_S = 2.0  # a frame's wait for the push's sequence words
PUSH_SPIN_S = 20e-3  # its looks without a pause: longer than a frame
WIDE = raster.TILES_X  # tiles a row
# The lists sized for the sun's view too: its entries at the scene's own sun and at
# these (azimuth offset, elevation) suns, times the margin, plus card_thread's 65,536.
SUN_SAMPLES = [
    (turn, elevation) for turn in (0, 90, 180, 270) for elevation in (25, 80)
]
SUN_SAMPLES.append((0, 90))  # the zenith (every azimuth's)
SUN_LIST_MARGIN = 1.25


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


def difference_image(got, want):
    """The absolute difference of two 8-bit images, four times, at most 255."""
    error = np.abs(got[..., :3].astype(np.int16) - want[..., :3].astype(np.int16))
    return np.minimum(4 * error, 255).astype(np.uint8)


def shade_profile(profile):
    """SHADE's profile pages (cores, 64 words) as totals: tiles, records, rows,
    passes, pages read, texels fetched; the pack thread's cycles a tile by part."""
    totals = profile[:, : SH["kShadeProfTotal"] + 1].sum(axis=0)
    tiles = max(int(totals[SH["kShadeProfTiles"]]), 1)
    names = {
        "tiles": "kShadeProfTiles",
        "records": "kShadeProfRecords",
        "record_rows": "kShadeProfRows",
        "passes": "kShadeProfPasses",
        "pages_read": "kShadeProfPages",
        "texels_fetched": "kShadeProfTexels",
    }
    report = {key: int(totals[SH[name]]) for key, name in names.items()}
    parts = {
        "wait_in": "kShadeProfWaitIn",
        "copy_in": "kShadeProfCopyIn",
        "divide": "kShadeProfDivide",
        "records": "kShadeProfRecord",
        "scan": "kShadeProfScan",
        "wait_pages": "kShadeProfWaitPages",
        "gather": "kShadeProfGather",
        "out": "kShadeProfOut",
        "total": "kShadeProfTotal",
    }
    report["cycles_per_tile"] = {
        key: round(float(totals[SH[name]]) / tiles, 1) for key, name in parts.items()
    }
    report["slowest_core_cycles"] = int(profile[:, SH["kShadeProfTotal"]].max())
    report["cores_reported"] = int((profile[:, SH["kShadeProfTotal"]] != 0).sum())
    report["item_pages_max"] = int(profile[:, SH["kShadeProfItemPages"]].max())
    report["items_in_passes"] = int(profile[:, SH["kShadeProfItemPasses"]].sum())
    return report


def light_profile(profile, lamps=0):
    """LIGHT's profile pages (cores, 64 words) as totals: tiles, rows, pages read,
    passes; the pack thread's cycles a tile by part (the colour's: shadow filter,
    lamps, the sun's specular term, the rest), and a lamp's cycles a tile."""
    totals = profile[:, : SH["kLightProfTotal"] + 1].sum(axis=0)
    tiles = max(int(totals[SH["kLightProfTiles"]]), 1)
    parts = {
        "wait": "kLightProfWait",
        "copy_in": "kLightProfCopyIn",
        "keys": "kLightProfKeys",
        "scan": "kLightProfScan",
        "wait_pages": "kLightProfWaitPages",
        "gather": "kLightProfGather",
        "shadow": "kLightProfShadow",
        "lamps": "kLightProfLamps",
        "specular": "kLightProfSpecular",
        "rest": "kLightProfMath",
        "out": "kLightProfOut",
        "total": "kLightProfTotal",
    }
    lamp_cycles = float(totals[SH["kLightProfLamps"]]) / tiles / max(lamps, 1)
    return {
        "tiles": int(totals[SH["kLightProfTiles"]]),
        "rows": int(totals[SH["kLightProfRows"]]),
        "pages_read": int(totals[SH["kLightProfPages"]]),
        "passes": int(totals[SH["kLightProfPasses"]]),
        "cycles_a_lamp_a_tile": round(lamp_cycles, 1) if lamps else None,
        "cycles_per_tile": {
            key: round(float(totals[SH[name]]) / tiles, 1)
            for key, name in parts.items()
        },
        "slowest_core_cycles": int(profile[:, SH["kLightProfTotal"]].max()),
    }


def figures(values):
    """Median, 5th and 95th percentile of a list of milliseconds."""
    values = np.asarray(values, dtype=np.float64)
    return {
        "median": round(float(np.median(values)), 4),
        "p5": round(float(np.percentile(values, 5)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
    }


def sun_slots(scene, materials, angles):
    """Each layer's list entries in the sun's view, the most over the scene's own sun
    (`angles`) and SUN_SAMPLES around it (light.sun_entries), and the slots that take
    them: {layer: (entries, slots)}; a sun beyond them ends its SUN command with the
    result word RF_LISTS_SMALL, never an overrun."""
    masked = materials["kind"][materials["material"]] == raster.MASKED
    suns = [angles] + [(angles[0] + turn, el) for turn, el in SUN_SAMPLES]
    layers = [(name, keep) for name, keep in (("opaque", ~masked), ("masked", masked))]
    layers = [(name, keep) for name, keep in layers if keep.any()]
    most = light.sun_entries(scene, [keep for _, keep in layers], suns)
    return {
        name: (entries, int(np.ceil(SUN_LIST_MARGIN * entries)) + 65536)
        for (name, _), entries in zip(layers, most)
    }


def times_rows(raw, frames):
    """The thread's rows of the frames served (the last RF_TIMES_ROWS), in order."""
    rows = np.frombuffer(raw, dtype=np.uint32)[
        : F["RF_TIMES_ROWS"] * F["RF_TIMES_ROW_WORDS"]
    ].reshape(F["RF_TIMES_ROWS"], F["RF_TIMES_ROW_WORDS"])
    kept = min(frames, F["RF_TIMES_ROWS"])
    numbers = np.arange(frames - kept, frames)
    return rows[numbers % F["RF_TIMES_ROWS"]], numbers


class Loop:
    """The scene loaded once in `session`, the thread's program running: a FRAME command
    a camera (`frame`), the image read back or pushed (`push`); `materials`: shaded
    (shade.py; masked triangles a layer of their own), `lights`: also lit (the surface,
    LIGHT's table in the sun region; `set_sun`). Fills upload_s, load_s, sun_lists."""

    def __init__(
        self,
        session,
        scene,
        push,
        list_slots,
        table,
        report,
        materials=None,
        lights=None,
        terms=raster.TERMS,
    ):
        self.session = session
        self.scene, self.materials, self.lights, self.terms = (
            scene,
            materials,
            lights,
            terms,
        )
        self.sun = None  # (azimuth, elevation) once set_sun moved it
        self.box = None if lights is None else light.scene_box(scene)
        sun_need = {}
        if lights is not None:
            started = time.perf_counter()
            angles = light_data.sun_angles(lights["sun_direction"])
            sun_need = sun_slots(scene, materials, angles)
            report["sun_lists"] = {
                "entries": {name: need[0] for name, need in sun_need.items()},
                "s": round(time.perf_counter() - started, 3),
            }
        prepared = (
            None if materials is None else shade.prepare(scene, materials, lights)
        )
        blocks = passes.mesh_blocks(scene) if prepared is None else prepared["blocks"]
        self.workers = workers = card.workers(session)
        started = time.perf_counter()
        default_slots = 2 * len(scene["indices"]) + 65536  # card_thread's
        slots = max(list_slots or default_slots, sun_need.get("opaque", (0, 0))[1])
        self.uploaded = card_thread.tensors(session, blocks, workers, slots)
        self.shade_profile = None
        if prepared is not None:
            self.shaded = {
                "shade": card.tensor(session, prepared["shade_blocks"]),
                "textures": card.tensor(session, prepared["pages"]),
                "table": card.tensor(session, prepared["table"][None, :]),
            }
            self.shade_profile = card.tensor(
                session, np.zeros((workers, SH["kShadeProfBytes"] // 4))
            )
            report["texture_pages"] = len(prepared["pages"])
        self.surface = self.light_profile = None
        if lights is not None:
            report["list_slots"] = slots
            self.surface = light.surface_tensors(
                session, workers, lights, materials, terms
            )
            self.light_profile = self.surface["profile"]
        self.masked = None
        if prepared is not None and "masked_blocks" in prepared:
            masked_slots = 2 * prepared["masked_triangles"] + 65536  # card_thread's
            if list_slots:  # the masked layer's lists grown as the others are
                masked_slots = -(-masked_slots * list_slots // default_slots)
            masked_slots = max(masked_slots, sun_need.get("masked", (0, 0))[1])
            self.masked = card_thread.layer_tensors(
                session, prepared["masked_blocks"], workers, masked_slots
            )
            self.masked["shade"] = card.tensor(session, prepared["masked_shade_blocks"])
            report["masked_triangles"] = prepared["masked_triangles"]
        self.display = card.tensor(
            session, np.zeros((lists.TILES, SHOW["kShowPageBytes"] // 4))
        )
        self.show_profile = card.tensor(
            session, np.zeros((workers, SHOW["kShowProfBytes"] // 4))
        )
        session.device.finish()
        words = card_thread.words(self.uploaded, len(blocks), workers, workers, 4)
        words[F["RF_W_REPEAT"]] = 0  # serve FRAME commands
        words[F["RF_W_DISPLAY"]] = self.display.address
        words[F["RF_W_SHOWPROF"]] = self.show_profile.address
        if prepared is not None:
            words[F["RF_W_SHADE"]] = self.shaded["shade"].address
            words[F["RF_W_TEXTURES"]] = self.shaded["textures"].address
            words[F["RF_W_SHADE_TABLE"]] = self.shaded["table"].address
            words[F["RF_W_SHADEPROF"]] = self.shade_profile.address
        sun = None
        if self.surface is not None:
            words[F["RF_W_SURFACE"]] = self.surface["dir"].address
            words[F["RF_W_SUN_PAGES"]] = self.surface["pages"].address
            sun = light.sun_region(lights, materials, terms)
        if self.masked is not None:
            card_thread.masked_words(
                words,
                self.masked,
                len(prepared["masked_blocks"]),
                self.masked["shade"].address,
            )
        report["upload_s"] = round(time.perf_counter() - started, 3)
        started = time.perf_counter()
        self.running = running = session.load(
            card_thread.program(scene, words, workers, table, sun), deadline=1.0
        )
        report["load_s"] = round(time.perf_counter() - started, 3)
        self.transfer = None
        if push:
            self.transfer = transfer = session.channel_transfer(self.display, push=True)
            entry = transfer.entry
            push_words = np.zeros(F["RF_PUSH_BYTES"] // 4, dtype=np.uint32)
            push_words[F["RF_PUSH_HEADER"]] = entry.handler | entry.field << 24
            push_words[F["RF_PUSH_GROUP"]] = entry.group[0] | entry.group[1] << 16
            push_words[F["RF_PUSH_ARGS"] : F["RF_PUSH_ARGS"] + 10] = entry.args
            running.thread.write("push", push_words)
        self.pages = np.empty((lists.TILES, SHOW["kShowPageBytes"]), dtype=np.uint8)

    def frame(self, camera, image):
        """One FRAME command for `camera` (16 float32), its 8-bit picture into `image`
        (1024, 1024, 4). Returns its event, the times (s) began, camera written,
        commanded, read and ended, and the event's wait after the image (ms; None
        without the push)."""
        running, transfer = self.running, self.transfer
        camera = np.asarray(camera, dtype=np.float32).view(np.uint32)
        collect = None
        began = time.perf_counter()
        if transfer is None:
            running.thread.write("camera", camera)
            written = time.perf_counter()
            event = running.wait(
                running.submit("frame", cluster=CLUSTER, result=True), timeout=10.0
            )
            commanded = time.perf_counter()
            self.session.read(self.display, out=self.pages)
            read = time.perf_counter()
            untile(self.pages, image)
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
            collect = (time.perf_counter() - ended) * 1e3
            if not arrived and event.kind == "done" and event.result == 0:
                event = event._replace(result="push words missing")
        return event, (began, written, commanded, read, ended, collect)

    def set_sun(self, azimuth, elevation):
        """The sun at `azimuth`, `elevation` (degrees): light.sun_lights into the sun
        region, then above the horizon the SUN command (the next frame's shadow map).
        The command's word (None: no pass), the host's ms, the lights (no map)."""
        began = time.perf_counter()
        lights = light.sun_lights(self.lights, self.box, azimuth, elevation)
        region = light.sun_region(lights, self.materials, self.terms)
        self.running.thread.write("sun", region)
        word = None
        if elevation > 0:
            event = self.running.wait(
                self.running.submit("sun", cluster=CLUSTER, result=True), timeout=10.0
            )
            word = event.result if event.kind == "done" else f"{event.kind}"
        self.sun = (float(azimuth), float(elevation))
        return word, (time.perf_counter() - began) * 1e3, lights

    def shadow_map(self):
        """The card's shadow map (1024 x 1024 float32 1/z) from the light pages."""
        side = self.lights["shadow_map"].shape[0]
        (pages,) = self.session.read(
            self.surface["pages"], first=0, pages=(side // light.PAGE) ** 2
        )
        return light.shadow_of_pages(pages.view(np.uint32))

    def stop(self):
        """Stops the program (the run's own stop)."""
        self.running.stop()


def loop(args):
    scene, _ = run.load(args.scene)
    views = framegraph.cameras(scene, args.frames)
    table = show_table()
    args.shade = args.shade or args.light
    materials = shade.load(args.scene) if args.shade else None
    lights = light_data.load(args.scene, native.data_dir()) if args.light else None
    terms = tuple(args.light_terms.split(","))
    report = {
        "scene": args.scene,
        "frames": args.frames,
        "path": "push" if args.push else "read",
        "shaded": bool(args.shade),
        "lit": bool(args.light),
    }
    if args.light:
        report["light_terms"] = list(terms)
    report["table_differs"] = table_check(table)
    sun_set = parse_sun(args.sun) if args.sun else None
    if (sun_set or args.sun_path or args.check_sun) and not args.light:
        raise SystemExit("--sun, --sun-path and --check-sun move a lit frame's sun")
    sun_at = {}  # frame index: the sun it was rendered with, once moved
    passes_run = []  # each SUN command: angles, word, host ms
    checked = sorted({0, args.frames // 2, args.frames - 1})
    surfaces = sorted({0, args.frames - 1}) if args.check_surface else []
    kept, surface_got = {}, {}
    with km.Session(log=card.log) as session:
        live = Loop(
            session,
            scene,
            args.push,
            args.list_slots,
            table,
            report,
            materials,
            lights,
            terms,
        )
        workers, running = live.workers, live.running
        sun_got = None

        def move_sun(azimuth, elevation, index):
            word, host_ms, _ = live.set_sun(azimuth, elevation)
            passes_run.append(
                {
                    "frame": index,
                    "angles": [round(azimuth, 4), round(elevation, 4)],
                    "word": word,
                    "host_ms": round(host_ms, 3),
                }
            )
            if word not in (0, None):
                card.log(f"SUN before frame {index}: word {word}")
            return word

        if sun_set:
            word = move_sun(*sun_set, 0)
            if args.check_sun and word == 0:
                sun_got = live.shadow_map()
        image = np.empty((raster.HEIGHT, raster.WIDTH, 4), dtype=np.uint8)
        host, camera_ms, command_ms, read_ms, words_back = [], [], [], [], []
        collect_ms = []
        own = light_data.sun_angles(lights["sun_direction"]) if args.light else None
        for index, view in enumerate(views):
            if args.sun_path and index % args.sun_path == 0:
                path_sun = light.sun_path(own[0], index // args.sun_path)
                if move_sun(*path_sun, index) not in (0, None):
                    break
            if live.sun is not None:
                sun_at[index] = live.sun
            event, times = live.frame(view["camera"], image)
            began, written, commanded, read, ended, collect = times
            if collect is not None:
                collect_ms.append(collect)
            host.append((ended - began) * 1e3)
            camera_ms.append((written - began) * 1e3)
            command_ms.append((commanded - written) * 1e3)
            read_ms.append((read - commanded) * 1e3)
            word = event.result if event.kind == "done" else f"{event.kind}"
            words_back.append(word)
            if index in checked:
                kept[index] = image.copy()
            if index in surfaces and word == 0:  # the planes before the next frame
                surface_got[index] = light.read_surface(
                    session, live.uploaded["frame"], live.surface
                )
            if word != 0:
                card.log(f"frame {index}: {event.kind}, word {word}")
                break
        live.stop()
        report["status"] = card.check_run(running, workers, 0, [])
        rows, numbers = times_rows(running.thread.read("times"), len(words_back))
        sun_rows = None
        if passes_run:
            sun_rows, _ = times_rows(
                running.thread.read("sun_times"),
                sum(run["word"] is not None for run in passes_run),
            )
        (profile_pages,) = card.read(session, live.show_profile)
        if live.shade_profile is not None:
            (shade_pages,) = card.read(session, live.shade_profile)
            report["shade_profile"] = shade_profile(
                shade_pages.reshape(workers, -1).astype(np.int64)
            )
        if live.light_profile is not None:
            (light_pages,) = card.read(session, live.light_profile)
            report["light_profile"] = light_profile(
                light_pages.reshape(workers, -1).astype(np.int64),
                len(lights["lamp_intensity"]) if "lamps" in terms else 0,
            )
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
    report["card_frame_max_ms"] = round(
        float(rows[:, F["RF_T_FRAME"]].max()) * ticks_ms, 4
    )
    report["host_max_ms"] = round(float(np.max(host)), 4)
    report["card_show_ms"] = round(
        float(np.median(rows[:, F["RF_T_SHOW"]])) * ticks_ms, 4
    )
    report["card_shade_ms"] = round(
        float(np.median(rows[:, F["RF_T_SHADE"]])) * ticks_ms, 4
    )
    report["card_light_ms"] = round(
        float(np.median(rows[:, F["RF_T_LIGHT"]])) * ticks_ms, 4
    )
    if live.masked is not None:
        report["card_layer_ms"] = {
            layer: round(
                float(np.median(rows[:, F[f"RF_T_{layer.upper()}"]])) * ticks_ms, 4
            )
            for layer in ("opaque", "masked")
        }
        report["card_shade_masked_ms"] = round(
            float(np.median(rows[:, F["RF_T_SHADE_MASKED"]])) * ticks_ms, 4
        )
    report["card_parts_ms"] = {
        part: round(float(np.median(rows[:, F[f"RF_O_{part.upper()}"]])) * ticks_ms, 4)
        for part in ("count", "sum", "hist", "hsum", "middle", "place", "setup", "fill")
    }
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
    if passes_run:
        report["sun"] = sun_report(passes_run, sun_rows, ticks_ms)
    moved = {}  # the reference's lights per sun (lights.moved: seconds each)

    def lights_of(index):
        sun = sun_at.get(index)
        if sun is None:
            return lights
        if sun not in moved:
            moved[sun] = light_data.moved(lights, scene, materials, *sun)
        return moved[sun]

    if sun_got is not None:
        report["sun_check"] = light.sun_check(scene, materials, lights_of(0), sun_got)
    differs, shade_checks, light_checks, surface_checks = {}, {}, {}, {}
    folder = Path(args.pictures) if args.pictures else None
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
    for index, got in surface_got.items():
        surface_checks[index], _ = light.surface_check(
            views[index], materials, lights_of(index), got
        )
    for index, got in kept.items():
        mode = "_lit" if args.light else "_shaded" if args.shade else ""
        tag = f"{args.scene}_{index:04d}{mode}"
        if folder:
            png.write_png(folder / f"{tag}.png", got[..., :3])
        if args.light:
            if not args.check_light:
                continue
            light_checks[index], want = light.light_check(
                views[index], materials, lights_of(index), got, terms
            )
            if folder:
                png.write_png(folder / f"{tag}_reference.png", want)
                png.write_png(
                    folder / f"{tag}_difference.png", difference_image(got, want)
                )
            continue
        if args.shade:
            if not args.check_shade:
                continue
            shade_checks[index], want, true = shade.check(
                views[index], materials, got, png.to_srgb8
            )
            if folder:
                png.write_png(folder / f"{tag}_reference.png", want)
                png.write_png(folder / f"{tag}_true_reference.png", true)
                png.write_png(
                    folder / f"{tag}_difference.png", difference_image(got, want)
                )
            continue
        want = expected(views[index])
        differs[index] = int((got != want).any(axis=2).sum())
    if args.light:
        report["surface_check"] = surface_checks
        report["light_check"] = light_checks
        # ok: check C's rule; the surface's numbers are reported (all_within: none off)
        pictures_ok = (
            not args.check_light
            or (
                len(light_checks) == len(checked)
                and all(check["ok"] for check in light_checks.values())
            )
        ) and (not args.check_surface or len(surface_checks) == len(surfaces))
    elif args.shade:
        report["shade_check"] = shade_checks
        pictures_ok = not args.check_shade or (
            len(shade_checks) == len(checked)
            and all(check["share"] < 0.01 for check in shade_checks.values())
        )
    else:
        report["pixels_differ"] = differs
        pictures_ok = len(differs) == len(checked) and not any(differs.values())
    sun_ok = all(run["word"] in (0, None) for run in passes_run) and (
        not args.check_sun or "sun_check" in report
    )
    report["ok"] = bool(
        len(words_back) == args.frames
        and report["words_nonzero"] == 0
        and sun_ok
        and pictures_ok
        and report["status"]["ok"]
        and report["table_differs"] == 0
    )
    return report


def parse_sun(text):
    """--sun's "AZ,EL" as two floats (degrees)."""
    azimuth, elevation = (float(part) for part in text.split(","))
    return azimuth, elevation


def sun_report(passes_run, rows, ticks_ms):
    """The SUN commands: each one's angles, word and host ms, and from the thread's sun
    rows (frame.h RF_S_*) each one's parts in ms; the medians and the worst."""
    report = {"passes": len(passes_run), "runs": passes_run}
    report["words_nonzero"] = sum(run["word"] not in (0, None) for run in passes_run)
    host = [run["host_ms"] for run in passes_run if run["word"] is not None]
    if host:
        report["host_ms_median"] = round(float(np.median(host)), 3)
        report["host_ms_max"] = round(float(np.max(host)), 3)
    if rows is None or len(rows) == 0:
        return report
    names = {"total": "TOTAL", "opaque": "OPAQUE", "masked": "MASKED"}
    names |= {"merge": "MERGE", "copy": "COPY"}
    for name, key in names.items():
        values = rows[:, F[f"RF_S_{key}"]] * ticks_ms
        report[f"{name}_ms_median"] = round(float(np.median(values)), 4)
        report[f"{name}_ms_max"] = round(float(np.max(values)), 4)
    report["parts_ms_median"] = {
        part: round(float(np.median(rows[:, F[f"RF_O_{part.upper()}"]])) * ticks_ms, 4)
        for part in ("count", "sum", "hist", "hsum", "middle", "place", "setup", "fill")
    }
    report["heavy_tiles_max"] = int(rows[:, F["RF_O_HEAVY"]].max())
    report["codes"] = rows[:, F["RF_S_CODE"]].tolist()
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
        "--shade", action="store_true", help="the frame shaded: textures (SHADE)"
    )
    parser.add_argument(
        "--check-shade",
        action="store_true",
        help="the checked frames against the reference's shaded frame",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="the frame lit (implies --shade): SHADE's surface planes, LIGHT",
    )
    parser.add_argument(
        "--check-surface",
        action="store_true",
        help="lit: the first and last frame's surface planes against the reference",
    )
    parser.add_argument(
        "--check-light",
        action="store_true",
        help="lit: the checked frames against the reference's lit frame (its terms)",
    )
    parser.add_argument(
        "--light-terms",
        default=",".join(raster.TERMS),
        help="lit: the reference's terms the card computes (sky, ambient, sun always)",
    )
    parser.add_argument(
        "--sun",
        default=None,
        help="lit: AZ,EL (degrees): the sun moved before the first frame (SUN pass)",
    )
    parser.add_argument(
        "--sun-path",
        type=int,
        default=0,
        help="lit: the sun moved every N frames (light.sun_path), a SUN pass each",
    )
    parser.add_argument(
        "--check-sun",
        action="store_true",
        help="with --sun: the card's shadow map against the reference's (check B)",
    )
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
