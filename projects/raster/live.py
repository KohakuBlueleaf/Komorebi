"""The thread-driven frame as a loop: a FRAME command a camera, the image read back or
(`--push`) pushed; `--shade`, `--light` (`--sun`, `--sun-path`: SUN); `--size WxH`
(default 1024x1024), `--max-size WxH` the capacity, `--resize` sizes while running. One
JSON line; exit 1 unless ok."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.bench import power
from komorebi.program.thread import TIMER_HZ
from projects.raster import card, card_thread, light, lists, passes, run, shade, update
from projects.raytrace import framegraph
from reference import lights as light_data
from reference import native, png, raster, scenes

F, SHOW, SH, S = card_thread.F, card_thread.SHOW, card_thread.SH, card_thread.S
M = card_thread.M
CLUSTER = card_thread.CLUSTER
TILE = 32
PUSH_TIMEOUT_S = 2.0  # a frame's wait for the push's sequence words
PUSH_SPIN_S = 20e-3  # its looks without a pause: longer than a frame
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


def untile(pages, image, grid=(raster.TILES_X, raster.TILES_Y)):
    """The display's pages (tile = ty TX + tx of `grid` (TX, TY); 32 rows of 32 pixels
    of 4 bytes) into `image` (32 TY, 32 TX, 4), the padded frame: one reshape and one
    transpose."""
    across, down = grid
    image.reshape(down, TILE, across, TILE, 4)[...] = (
        pages[: across * down]
        .reshape(down, across, TILE, TILE, 4)
        .transpose(0, 2, 1, 3, 4)
    )
    return image


def parse_size(text, option="--size"):
    """A "WxH" as (W, H), any positive size (card_thread.card_grid); exits with the
    reason otherwise."""
    try:
        width, height = (int(part) for part in text.lower().split("x"))
        card_thread.card_grid((width, height))
    except ValueError as error:
        raise SystemExit(f"{option} {text}: {error}") from None
    return width, height


def parse_resize(text):
    """--resize's "N:WxH,M:WxH" as {frame index: (W, H)}."""
    out = {}
    for part in text.split(","):
        index, _, size = part.partition(":")
        try:
            out[int(index)] = parse_size(size, "--resize")
        except ValueError:
            raise SystemExit(f"--resize {text}: N:WxH,...") from None
    return out


def expected(view):
    """The reference's 8-bit picture of a camera: numpy's colour by png's rule, alpha
    255 (the padded frame's top-left W x H)."""
    colour = raster.picture(raster.rasterize(view)["colour"], view)
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
    report["pack"] = spread(profile[:, SH["kShadeProfTotal"]])
    report["cores_reported"] = int((profile[:, SH["kShadeProfTotal"]] != 0).sum())
    report["item_pages_max"] = int(profile[:, SH["kShadeProfItemPages"]].max())
    report["items_in_passes"] = int(profile[:, SH["kShadeProfItemPasses"]].sum())
    return report


def light_profile(profile, lamps=0):
    """LIGHT's profile pages (cores, 64 words) as totals: tiles, rows, pages read,
    passes; the pack thread's cycles a tile by part (the colour's: shadow filter,
    lamps, the sun's specular term, the rest), and a lamp's cycles a tile."""
    totals = profile[:, : SH["kLightProfWords"]].sum(axis=0)
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
        "field_keys": "kLightProfFieldKeys",
        "field_scan": "kLightProfFieldScan",
        "field_wait_pages": "kLightProfFieldWait",
        "field_gather": "kLightProfFieldGather",
        "field_blend": "kLightProfFieldBlend",
        "total": "kLightProfTotal",
    }
    lamp_cycles = float(totals[SH["kLightProfLamps"]]) / tiles / max(lamps, 1)
    return {
        "tiles": int(totals[SH["kLightProfTiles"]]),
        "rows": int(totals[SH["kLightProfRows"]]),
        "pages_read": int(totals[SH["kLightProfPages"]]),
        "passes": int(totals[SH["kLightProfPasses"]]),
        "field_pages_read": int(totals[SH["kLightProfFieldPages"]]),
        "field_passes": int(totals[SH["kLightProfFieldPasses"]]),
        "field_item_pages_max": int(profile[:, SH["kLightProfFieldItemPages"]].max()),
        "cycles_a_lamp_a_tile": round(lamp_cycles, 1) if lamps else None,
        "cycles_per_tile": {
            key: round(float(totals[SH[name]]) / tiles, 1)
            for key, name in parts.items()
        },
        "slowest_core_cycles": int(profile[:, SH["kLightProfTotal"]].max()),
        "pack": spread(profile[:, SH["kLightProfTotal"]]),
    }


def spread(totals):
    """Per core cycles (cores that ran) as slowest, mean, least and the mean core's
    wait for the slowest (cycles; the stage lasts as long as the slowest core)."""
    totals = np.asarray(totals, dtype=np.float64)
    totals = totals[totals > 0]
    if len(totals) == 0:
        return None
    return {
        "cores": len(totals),
        "slowest": int(totals.max()),
        "mean": int(totals.mean()),
        "least": int(totals.min()),
        "slowest_over_mean": round(float(totals.max() / totals.mean()), 3),
        "mean_wait": int(totals.max() - totals.mean()),
    }


def passes_profile(page):
    """COUNT's and SETUP's profile pages (cores, 1,024 words; setup.h kSProf*): per pass
    the pack thread's spread and parts summed over the cores, the reader's and writer's
    waits, and the cycles a block, a kept triangle and an entry."""
    report = {}
    for name, handler in (("count", S["kHandlerCount"]), ("setup", S["kHandlerSetup"])):
        words = page[:, 32 * (handler - S["kHandlerCount"]) :][:, :32]
        pack = words[:, S["kSProfPack"] : S["kSProfPack"] + 8]
        reader = words[:, S["kSProfReader"] : S["kSProfReader"] + 5]
        writer = words[:, S["kSProfWriter"] : S["kSProfWriter"] + 6]
        blocks = int((reader[:, 3] // S["kMeshPageBytes"]).sum())
        kept, entries = int(pack[:, 4].sum()), int(pack[:, 6].sum())
        total = int(pack[:, 3].sum())
        report[name] = {
            "pack": spread(pack[:, 3]),
            "writer": spread(writer[:, 2]),
            "pack_parts": {
                part: int(pack[:, column].sum())
                for column, part in (
                    (0, "mail_wait"),
                    (1, "vector"),
                    (2, "scalar"),
                    (7, "out_wait"),
                )
            },
            "reader_wait": int(reader[:, 0].sum()),
            "writer_wait": int(writer[:, 0].sum()),
            "writer_write": int(writer[:, 1].sum()),
            "blocks": blocks,
            "kept": kept,
            "clipped": int(pack[:, 5].sum()),
            "entries": entries,
            "cycles_a_block": round(total / max(blocks, 1), 1),
            "cycles_a_kept": round(total / max(kept, 1), 1),
            "writer_cycles_an_entry": round(
                float(writer[:, 1].sum()) / max(entries, 1), 1
            ),
        }
    return report


def fill_profile(page):
    """FILL's profile pages (cores, 1,024 words; run.PROF): the pack thread's spread,
    parts summed, and the cycles a (triangle, tile) entry, a row, a tile."""
    rows = run.profile_rows(page, len(page))
    entries = int(rows["tris"].sum())
    return {
        "pack": spread(rows["total"]),
        "parts": {
            name: int(rows[name].sum())
            for name in ("wait_in", "init", "tri", "rows", "drain", "out", "wait_out")
        },
        "tiles": int(rows["tiles"].sum()),
        "entries": entries,
        "rows": int(rows["row_count"].sum()),
        "cycles_an_entry": round(float(rows["total"].sum()) / max(entries, 1), 1),
        "setup_an_entry": round(float(rows["tri"].sum()) / max(entries, 1), 1),
        "cycles_a_row": round(
            float(rows["rows"].sum()) / max(int(rows["row_count"].sum()), 1), 1
        ),
    }


def figures(values):
    """Median, 5th and 95th percentile of a list of milliseconds."""
    values = np.asarray(values, dtype=np.float64)
    return {
        "median": round(float(np.median(values)), 4),
        "p5": round(float(np.percentile(values, 5)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
    }


def sun_slots(scene, materials, angles, side):
    """Each layer's list entries in a sun's view (a quadrant of a map of `side`), the
    most over the scene's own sun (`angles`) and SUN_SAMPLES around it (sun_entries),
    and the slots that take them: {layer: (entries, slots)}; a sun beyond them ends its
    SUN command with the result word RF_LISTS_SMALL, never an overrun."""
    kind = materials["kind"][materials["material"]]
    masked = kind == raster.MASKED
    opaque = ~masked & (kind != raster.TRANSPARENT)  # glass: not in the sun's view
    suns = [angles] + [(angles[0] + turn, el) for turn, el in SUN_SAMPLES]
    layers = [(name, keep) for name, keep in (("opaque", opaque), ("masked", masked))]
    layers = [(name, keep) for name, keep in layers if keep.any()]
    most = light.sun_entries(scene, [keep for _, keep in layers], suns, side)
    return {
        name: (entries, int(np.ceil(SUN_LIST_MARGIN * entries)) + 65536)
        for (name, _), entries in zip(layers, most)
    }


def tone_of(lights):
    """The tone step's numbers of `lights` (raster.tone; saturation 1, curve 0 when
    absent)."""
    return {
        "exposure": float(lights["exposure"]),
        "saturation": float(lights.get("saturation", 1.0)),
        "curve": int(lights.get("curve", raster.CURVE_PLAIN)),
    }


def times_rows(raw, frames):
    """The thread's rows of the frames served (the last RF_TIMES_ROWS), in order."""
    rows = np.frombuffer(raw, dtype=np.uint32)[
        : F["RF_TIMES_ROWS"] * F["RF_TIMES_ROW_WORDS"]
    ].reshape(F["RF_TIMES_ROWS"], F["RF_TIMES_ROW_WORDS"])
    kept = min(frames, F["RF_TIMES_ROWS"])
    numbers = np.arange(frames - kept, frames)
    return rows[numbers % F["RF_TIMES_ROWS"]], numbers


LIST_MARGIN = 1.25  # grow_lists: the slots a frame needed, and a quarter


class ListsRefused(ValueError):
    """A frame's lists the card cannot hold: layer `name` needed `need` slots of the
    `had` its lists hold, grown to `slots`; `error`, the allocation's or the load's
    (None: past the base word's slot field, kernels/setup.h kBaseSlotBits)."""

    def __init__(self, name, need, had, slots, error):
        field = 1 << M["RM_BASE_SLOT_BITS"]
        why = (
            f"the base word's slot field holds {field:,}"
            if error is None
            else f"the card did not hold them: {error}"
        )
        super().__init__(
            f"the {name} layer's lists: the frame needs {need:,} slots, they hold "
            f"{had:,}; grown to {slots:,} slots "
            f"({slots * lists.PAGE_WORDS * 4 // lists.PAGE_SLOTS:,} B): {why}"
        )


class Loop:
    """The scene loaded in `session`, its program running: a FRAME command a camera
    (`frame`); `materials`: shaded, `lights`: lit (`set_sun`, `set_tone`); `indirect`:
    the field (`set_indirect`; `card_field`: updated by the thread, card_update); the
    size, `set_size`; the capacity `held` sizes every per-tile tensor and buffer."""

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
        held=None,
        card_field=False,
        apply=False,
    ):
        self.session = session
        self.scene, self.materials, self.lights, self.terms = (
            scene,
            materials,
            lights,
            terms,
        )
        self.size = scenes.frame_size(scene)  # (W, H), set_size
        self.grid = card_thread.card_grid(self.size)  # (TX, TY)
        self.held = card_thread.capacity([self.size], lights is not None, held)
        self.push, self.list_slots, self.table = push, list_slots, table
        self.reloads = []  # set_size's reloads: sizes, capacities, seconds
        self.list_grown = {}  # a layer's lists' slots grown past list_slots' (frame)
        self.lists_grown = []  # those growths: the frame's needs, the reload's ms
        self.padded = None  # the padded frame's picture when it is larger than W x H
        self.sun = None  # (azimuth, elevation) once set_sun moved it
        self.lit = lights  # the lights LIGHT's table is of (set_sun's once moved)
        self.tone = None if lights is None else tone_of(lights)
        self.indirect = "indirect" in terms  # the switch (set_indirect)
        self.box = None if lights is None else light.scene_box(scene)
        self.field = self.field_update = None  # the term indirect: field_of's
        self.card_field, self.apply = card_field, apply
        self.card_updates = 0  # UPDATE's updates (card_field)
        if card_field and (lights is None or "indirect" not in terms):
            raise ValueError("the field's update in the thread needs the term indirect")
        if card_field and update.same_numbers():
            raise ValueError(f"update.h differs: {update.same_numbers()}")
        if lights is not None and "indirect" in terms:
            self.field = light.field_of(lights)
            self.shadow = lights["shadow_map"]  # the map the field was made with
            report["field"] = {
                "s": round(self.field["seconds"], 3),
                "pages": len(self.field["pages"]),
                "scale": float(self.field["scale"]),
            }
        sun_need = {}
        if lights is not None:
            started = time.perf_counter()
            angles = light_data.sun_angles(lights["sun_direction"])
            side = light_data.shadow_side(lights)
            sun_need = sun_slots(scene, materials, angles, side)
            report["sun_lists"] = {
                "entries": {name: need[0] for name, need in sun_need.items()},
                "s": round(time.perf_counter() - started, 3),
            }
        prepared = (
            None if materials is None else shade.prepare(scene, materials, lights)
        )
        blocks = passes.mesh_blocks(scene) if prepared is None else prepared["blocks"]
        self.workers = card.workers(session)
        self.held = self.held._replace(cores=self.workers)  # the heavy tiles' tables
        self.prepared, self.blocks, self.sun_need = prepared, blocks, sun_need
        self.load(report)

    def load(self, report, again=False):
        """The tensors of the capacity `held` (`again`: those that follow it alone,
        made again), the program loaded and running (its thread serving FRAME
        commands); a refusal for memory as card_thread.L1Refused."""
        session, scene, held = self.session, self.scene, self.held
        card_thread.l1_check(held, scenes.frame_size(scene), session.l1()[1])
        started = time.perf_counter()
        if again:
            self.tile_tensors()
        else:
            self.make_tensors(report)
        self.start(report, started)

    def tile_tensors(self):
        """The tensors that follow the capacity made again for `held` (frames, their
        directories, lists, display, counters, slots), the old ones freed."""
        session, workers, held = self.session, self.workers, self.held
        old = [*self.uploaded["frame"], self.display]
        old += [
            self.uploaded[name] for name in ("counters", "slots", "hist", "hprefix")
        ]
        self.uploaded["frame"] = card_thread.frame_tensors(session, held)
        self.uploaded.update(card_thread.capacity_tensors(session, workers, held))
        for name, layer in self.layers():
            old.append(layer["lists"])
            slots = self.layer_slots(name, held)
            layer.update(card_thread.list_tensor(session, slots))
            if name != "opaque":
                old += layer["frame"]
                layer["frame"] = card_thread.frame_tensors(session, held)
        for surface in (self.surface, self.glass_surface):
            if surface is not None:
                light.surface_frames(session, workers, surface, held.tiles)
        for tensor in old:
            tensor.free()
        self.display = card.tensor(  # a page a tile of the capacity
            session, np.zeros((held.tiles, SHOW["kShowPageBytes"] // 4))
        )

    def layer_slots(self, name, held):
        """The slots of layer `name`'s lists at the capacity `held`: list_slots' guess
        from the scene, or what a frame needed (frame's RF_LISTS_SMALL) when more."""
        guess = card_thread.list_slots(self.list_base[name], held)
        return max(guess, self.list_grown.get(name, 0))

    def layers(self):
        """(name, tensors) of each layer loaded: opaque, then masked and glass."""
        named = (
            ("opaque", self.uploaded),
            ("masked", self.masked),
            ("glass", self.glass),
        )
        return [(name, layer) for name, layer in named if layer is not None]

    def make_tensors(self, report):
        """Every tensor of the frame, for the capacity `held`."""
        session, scene, lights = self.session, self.scene, self.lights
        materials, terms, held = self.materials, self.terms, self.held
        list_slots, sun_need = self.list_slots, self.sun_need
        prepared, blocks, workers = self.prepared, self.blocks, self.workers
        default_slots = 2 * len(scene["indices"]) + 65536  # card_thread's
        slots = max(list_slots or default_slots, sun_need.get("opaque", (0, 0))[1])
        self.list_base = {"opaque": slots}  # at the default frame's tiles
        slots = max(
            card_thread.list_slots(slots, held), self.list_grown.get("opaque", 0)
        )
        self.uploaded = card_thread.tensors(session, blocks, workers, slots, held)
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
                session,
                workers,
                lights,
                materials,
                terms,
                field=self.field,
                tiles=held.tiles,
            )
            self.light_profile = self.surface["profile"]
        self.update_tensors = None
        if self.card_field:  # the passes' x and y (update.py)
            self.update_tensors = update.tensors(session, self.field["model"])
        self.masked = self.glass = self.glass_surface = None
        for name in ("masked", "glass"):
            if prepared is None or f"{name}_blocks" not in prepared:
                continue
            layer_slots = 2 * prepared[f"{name}_triangles"] + 65536  # card_thread's
            if list_slots:  # the layer's lists grown as the others are
                layer_slots = -(-layer_slots * list_slots // default_slots)
            layer_slots = max(layer_slots, sun_need.get(name, (0, 0))[1])
            self.list_base[name] = layer_slots
            layer = card_thread.layer_tensors(
                session,
                prepared[f"{name}_blocks"],
                workers,
                self.layer_slots(name, held),
                held,
            )
            layer["shade"] = card.tensor(session, prepared[f"{name}_shade_blocks"])
            # its own profile tensors: FILL's, COUNT's and SETUP's, SHADE's
            layer["prof"] = card.tensor(session, np.zeros((workers, 1024)))
            layer["sprof"] = card.tensor(
                session, np.zeros((workers, S["kSProfPageBytes"] // 4))
            )
            layer["shade_profile"] = card.tensor(
                session, np.zeros((workers, SH["kShadeProfBytes"] // 4))
            )
            report[f"{name}_triangles"] = prepared[f"{name}_triangles"]
            setattr(self, name, layer)
        if self.glass is not None and lights is not None:
            # the glass layer's own surface and LIGHT profile; the table and pages the
            # surface's
            self.glass_surface = light.surface_tensors(
                session,
                workers,
                lights,
                materials,
                terms,
                shared=self.surface,
                tiles=held.tiles,
            )
        self.display = card.tensor(  # a page a tile of the capacity
            session, np.zeros((held.tiles, SHOW["kShowPageBytes"] // 4))
        )
        self.show_profile = card.tensor(
            session, np.zeros((workers, SHOW["kShowProfBytes"] // 4))
        )

    def start(self, report, started):
        """The thread's words from the tensors, the program built for `held` and
        loaded (with the heavy tiles counted again when the full histograms do not fit
        L1), the push armed."""
        session, scene, held = self.session, self.scene, self.held
        prepared, blocks, workers = self.prepared, self.blocks, self.workers
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
        sun = profiles = None
        if self.surface is not None:
            words[F["RF_W_SURFACE"]] = self.surface["dir"].address
            words[F["RF_W_SUN_PAGES"]] = self.surface["pages"].address
            sun = self.region(self.lit)
        glass_words = None  # frame.h RF_G_*, in area 0
        for name, prefix in (("masked", "RF_W_M_"), ("glass", "RF_G_")):
            layer = getattr(self, name)
            if layer is None:
                continue
            if name == "glass":
                glass_words = [0] * F["RF_G_WORDS"]
            card_thread.masked_words(
                words if name == "masked" else glass_words,
                layer,
                len(prepared[f"{name}_blocks"]),
                layer["shade"].address,
                prefix,
            )
        if self.glass_surface is not None:
            glass_words[F["RF_G_SURFACE"]] = self.glass_surface["dir"].address
        if self.masked is not None or self.glass is not None:
            profiles = [  # frame.h RF_P_*: the masked layer's, then the glass layer's
                0 if layer is None else layer[name].address
                for layer in (self.masked, self.glass)
                for name in ("prof", "sprof", "shade_profile")
            ]
        updating = None
        if self.card_field:  # UPDATE's regions (update.py)
            updating = update.regions(
                self.field["model"],
                self.lit,
                self.surface["pages"],
                self.update_tensors,
                workers,
                self.bounces(),
            )
        report["upload_s"] = round(time.perf_counter() - started, 3)
        started = time.perf_counter()

        def build(held):
            return card_thread.program(
                scene,
                words,
                workers,
                self.table,
                sun,
                profiles,
                glass_words,
                held,
                updating,
                self.apply,
            )

        program = build(held)
        if not held.recount and not card_thread.fits_l1(session, program):
            # the full histograms do not fit: COUNT counts the heavy tiles' again
            self.held = held = held._replace(recount=True)
            program = build(held)
        try:
            self.running = running = session.load(program, deadline=1.0)
        except km.L1BudgetError as error:
            size = scenes.frame_size(scene)
            raise card_thread.L1Refused.of_budget(held, size, error) from None
        report["load_s"] = round(time.perf_counter() - started, 3)
        report["capacity"] = card_thread.capacity_report(held, workers, self.dram())
        self.transfer = None
        if self.push:
            self.transfer = transfer = session.channel_transfer(self.display, push=True)
            entry = transfer.entry
            push_words = np.zeros(F["RF_PUSH_BYTES"] // 4, dtype=np.uint32)
            push_words[F["RF_PUSH_HEADER"]] = entry.handler | entry.field << 24
            push_words[F["RF_PUSH_GROUP"]] = entry.group[0] | entry.group[1] << 16
            push_words[F["RF_PUSH_ARGS"] : F["RF_PUSH_ARGS"] + 10] = entry.args
            running.thread.write("push", push_words)
        self.pages = np.empty((held.tiles, SHOW["kShowPageBytes"]), dtype=np.uint8)

    def tensors(self):
        """Every tensor this load made, once each (the light profile: the surface's)."""
        out, seen = [], set()

        def walk(value):
            if isinstance(value, dict):
                for each in value.values():
                    walk(each)
            elif isinstance(value, list | tuple):
                for each in value:
                    walk(each)
            elif hasattr(value, "free") and id(value) not in seen:
                seen.add(id(value))
                out.append(value)

        for name in ("uploaded", "shaded", "surface", "masked", "glass"):
            walk(getattr(self, name, None))
        walk(getattr(self, "update_tensors", None))
        walk(
            [
                getattr(self, name, None)
                for name in (
                    "glass_surface",
                    "display",
                    "show_profile",
                    "shade_profile",
                )
            ]
        )
        return out

    def dram(self):
        """The DRAM bytes of this load's tensors: those that grow with the capacity's
        tiles (frames, surfaces, lists, display, counters, slots) and the rest."""
        grows = {id(self.display)}
        grows |= {id(self.uploaded[name]) for name in ("counters", "slots", "hist")}
        for _, layer in self.layers():
            grows |= {id(t) for t in layer["frame"]} | {id(layer["lists"])}
        for surface in (self.surface, self.glass_surface):
            if surface is not None:
                grows |= {id(t) for frame in surface["frames"] for t in frame}
        by_tiles = rest = 0
        for tensor in self.tensors():
            size = tensor.pages * tensor.page_bytes
            if id(tensor) in grows:
                by_tiles += size
            else:
                rest += size
        return by_tiles, rest

    def set_size(self, size):
        """The frame's size (W, H) for the next frame, any positive size: within the
        capacity the grid word (frame.h RF_GRID_AT), nothing reloaded; past it a reload
        (card_thread.L1Refused when memory does not hold it). The next frame's camera is
        the caller's (scenes.sized_camera). Returns the host's ms."""
        began = time.perf_counter()
        width, height = (int(n) for n in size)
        grid = card_thread.card_grid((width, height))
        if not card_thread.covers(self.held, (width, height), self.lights is not None):
            self.reload((width, height))
            self.size, self.grid, self.padded = (width, height), grid, None
            ms = (time.perf_counter() - began) * 1e3
            self.reloads[-1]["ms"] = round(ms, 1)
            return ms
        word = np.array([card_thread.grid_word((width, height))], dtype=np.uint32)
        self.running.thread.write("camera", word, F["RF_GRID_AT"])
        self.size, self.grid = (width, height), grid
        self.padded = None
        return (time.perf_counter() - began) * 1e3

    def reload(self, size, stopped=False):
        """The program stopped (unless `stopped`), its tensors freed, and everything
        loaded again with a capacity that holds both the old capacity and `size`; the
        tone, the indirect switch and the sun as they were (a moved sun's SUN command
        again)."""
        began = time.perf_counter()
        held = card_thread.capacity([size], self.lights is not None, self.held)
        card_thread.l1_check(held, size, self.session.l1()[1])  # before the stop
        if not stopped:
            self.running.stop()
        if self.transfer is not None:
            self.transfer.close()
        before, self.held = self.held, held
        self.scene = scenes.with_size(self.scene, size)
        report = {}
        try:
            self.load(report, again=True)
        except card_thread.L1Refused:
            self.held = before  # the old capacity again, the size as it was
            self.scene = scenes.with_size(self.scene, self.size)
            self.load({}, again=True)
            raise
        if self.sun is not None:
            self.set_sun(*self.sun)
        elif self.card_updates:  # the field as UPDATE left it (the bounces)
            self.card_update(self.lit)
        self.reloads.append(
            {
                "size": list(size),
                "capacity": list(held),
                "upload_s": report["upload_s"],
                "load_s": report["load_s"],
                "s": round(time.perf_counter() - began, 3),
            }
        )

    def picture(self, pages, image):
        """The display's pages into `image` (H, W, 4): straight when the padded frame
        is W x H, else through the padded frame, cropped."""
        width, height = self.size
        across, down = self.grid
        if (32 * across, 32 * down) == (width, height):
            return untile(pages, image, self.grid)
        if self.padded is None:
            self.padded = np.empty((32 * down, 32 * across, 4), dtype=np.uint8)
        untile(pages, self.padded, self.grid)
        image[...] = self.padded[:height, :width]
        return image

    def frame(self, camera, image, grown=0):
        """One FRAME command for `camera` (16 float32), its picture into `image` (H, W,
        4): its event, the times (s) began, camera written, commanded, read, ended, and
        the event's wait after the image (ms; None without the push). On RF_LISTS_SMALL
        the lists grown (grow_lists) and the frame again (`grown`: times so far)."""
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
            self.picture(self.pages, image)
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
                self.picture(transfer.data, image)
        ended = time.perf_counter()
        if transfer is not None:  # FRAME's event, after the image (its word)
            event = running.wait(handle, timeout=10.0)
            collect = (time.perf_counter() - ended) * 1e3
            if not arrived and event.kind == "done" and event.result == 0:
                event = event._replace(result="push words missing")
        short = event.kind == "done" and event.result == F["RF_LISTS_SMALL"]
        if short and grown < len(self.layers()):  # a layer's lists, then the next's
            self.grow_lists()
            return self.frame(camera.view(np.float32), image, grown + 1)
        return event, (began, written, commanded, read, ended, collect)

    def grow_lists(self):
        """After a frame answered RF_LISTS_SMALL: each layer whose lists hold fewer
        slots than the frame needed (its times row) grown to LIST_MARGIN times that,
        everything loaded again (reload); its needs and ms into lists_grown.
        ListsRefused when the base word's slot field or the card's DRAM cannot."""
        began = time.perf_counter()
        self.running.stop()  # its regions are read after the stop
        rows = np.frombuffer(self.running.thread.read("times"), dtype=np.uint32)
        rows = rows[: F["RF_TIMES_ROWS"] * F["RF_TIMES_ROW_WORDS"]]
        rows = rows.reshape(F["RF_TIMES_ROWS"], F["RF_TIMES_ROW_WORDS"])
        row = rows[int(np.argmax(rows[:, F["RF_T_NUMBER"]]))]  # the last served
        at = {"opaque": 0, "masked": F["RF_T_M_PARTS"], "glass": F["RF_T_G_PARTS"]}
        grown = {}
        for name, layer in self.layers():
            need = int(row[at[name] + F["RF_O_SLOTS"]])
            if need > layer["list_slots"]:
                slots = math.ceil(need * LIST_MARGIN)
                if slots >= 1 << M["RM_BASE_SLOT_BITS"]:
                    self.reload(self.size, stopped=True)  # the frames go on as before
                    raise ListsRefused(name, need, layer["list_slots"], slots, None)
                grown[name] = [layer["list_slots"], need, slots]
        before = dict(self.list_grown)
        self.list_grown.update({name: slots for name, (_, _, slots) in grown.items()})
        try:
            self.reload(self.size, stopped=True)
        except RuntimeError as error:  # DRAM (tensors), or the load
            self.list_grown = before
            self.load({}, again=True)  # the frames go on as before
            name, (had, need, slots) = next(iter(grown.items()))
            raise ListsRefused(name, need, had, slots, error) from None
        ms = round((time.perf_counter() - began) * 1e3, 1)
        self.reloads[-1]["lists"] = grown
        self.lists_grown.append({"layers": grown, "ms": ms})

    def region(self, lights):
        """The sun region (light.sun_region) of `lights` with the tone step's numbers
        now (set_tone) and the indirect switch (set_indirect)."""
        return light.sun_region(
            {**lights, **self.tone},
            self.materials,
            self.terms,
            self.field,
            self.indirect,
        )

    def write_region(self, region):
        """The sun region `region` into the thread's; with card_field all but LIGHT's
        field unit and scale words, which UPDATE writes (the host's are the load's)."""
        if not self.card_field:
            self.running.thread.write("sun", region)
            return
        at = F["RF_SUN_TABLE_AT"] // 4 + SH["kLtFieldUnit"]
        end = F["RF_SUN_TABLE_AT"] // 4 + SH["kLtFieldScale"] + 1
        self.running.thread.write("sun", region[:at])
        self.running.thread.write("sun", region[end:], 4 * end)

    def bounces(self):
        """The field's bounce passes now (lights.bounce_count of the lights')."""
        return light_data.bounce_count(self.lit.get("bounces", light_data.BOUNCES))

    def set_tone(self, exposure, saturation, curve):
        """The tone step's numbers (raster.tone: exposure > 0, saturation >= 0, curve 0
        plain or 1 filmic) into LIGHT's table in the sun region, for the next frame;
        nothing reloaded. Returns the host's ms."""
        began = time.perf_counter()
        exposure, saturation, curve = float(exposure), float(saturation), int(curve)
        if not (exposure > 0 and saturation >= 0) or curve not in (0, 1):
            raise ValueError("a tone is exposure > 0, saturation >= 0, curve 0 or 1")
        self.tone = {"exposure": exposure, "saturation": saturation, "curve": curve}
        self.write_region(self.region(self.lit))
        return (time.perf_counter() - began) * 1e3

    def set_indirect(self, on):
        """The indirect switch (a frame loaded with the field): off, LIGHT lights as
        without the field (ambient, sky x visibility); on, from the field. Its terms
        word in the sun region, for the next frame. Returns the host's ms."""
        if self.field is None:
            raise ValueError("the indirect switch needs a frame loaded with --indirect")
        began = time.perf_counter()
        self.indirect = bool(on)
        self.write_region(self.region(self.lit))
        return (time.perf_counter() - began) * 1e3

    def set_bounces(self, bounces):
        """The indirect light's bounce passes (0 to 3, lights.bounce_count): the field
        recomputed on the host with that count and written between frames (update_field,
        as after a sun change); a later sun change keeps it. Returns the host's ms."""
        if self.field is None:
            raise ValueError("the bounce passes need a frame loaded with --indirect")
        bounces = light_data.bounce_count(bounces)
        began = time.perf_counter()
        self.lights = {**self.lights, "bounces": bounces}
        self.lit = {**self.lit, "bounces": bounces}
        if self.card_field:
            self.card_update(self.lit)
        else:
            self.update_field(self.lit, False)
        return (time.perf_counter() - began) * 1e3

    def set_sun(self, azimuth, elevation):
        """The sun at `azimuth`, `elevation` (degrees): light.sun_lights into the sun
        region, then above the horizon the SUN command (the next frame's shadow map).
        The command's word (None: no pass), the host's ms, the lights (no map)."""
        began = time.perf_counter()
        lights = light.sun_lights(self.lights, self.box, azimuth, elevation)
        self.lit = lights
        region = self.region(lights)
        self.write_region(region)
        word = None
        if elevation > 0:
            event = self.running.wait(
                self.running.submit("sun", cluster=CLUSTER, result=True), timeout=10.0
            )
            word = event.result if event.kind == "done" else f"{event.kind}"
        self.sun = (float(azimuth), float(elevation))
        if self.field is not None and word in (0, None):
            if self.card_field:  # with the card's map, the last one when not rendered
                self.card_update(lights)
            else:
                self.update_field(lights, word == 0)
        return word, (time.perf_counter() - began) * 1e3, lights

    def card_update(self, lights):
        """The field for `lights` (sun_lights') by the thread's UPDATE (update.py) with
        the card's shadow map: its inputs written, then the whole update, or without
        RF_APPLY and with passes its steps with the host's apply between them;
        field_update: its word, mode, the host's seconds by part."""
        began = time.perf_counter()
        bounces = light_data.bounce_count(lights.get("bounces", light_data.BOUNCES))
        update.write_sun(self.running, lights, bounces)
        stepped = not self.apply and bounces > 0
        word, host = update.run(
            self.running,
            self.session,
            self.update_tensors,
            self.field["model"],
            bounces,
            stepped,
        )
        self.card_updates += 1
        self.field_update = {
            "card": True,
            "mode": "stepped, the host's apply" if stepped else "thread",
            "word": word,
            "bounces": bounces,
            **host,
            "s": round(time.perf_counter() - began, 3),
        }
        if word != 0:
            card.log(f"UPDATE: word {word}")

    def update_field(self, lights, rendered):
        """The field for `lights` (sun_lights') with the card's shadow map (read back
        when the SUN pass `rendered` one, else the last), its pages written into the
        light pages and the scale with the sun region; field_update: the host's
        seconds by part and the bytes written."""
        began = time.perf_counter()
        if rendered:
            self.shadow = self.shadow_map()
        read = time.perf_counter()
        self.field = light.field_of({**lights, "shadow_map": self.shadow})
        computed = time.perf_counter()
        pages = self.field["pages"]
        first = light.light_pages_first(self.lights)
        self.session.write(
            self.surface["pages"], pages.view(np.uint8).reshape(len(pages), -1), first
        )
        region = self.region(lights)
        self.running.thread.write("sun", region)
        ended = time.perf_counter()
        self.field_update = {
            "s": round(ended - began, 3),
            "shadow_read_s": round(read - began, 3),
            "field_s": round(computed - read, 3),
            "write_s": round(ended - computed, 3),
            "bytes": int(pages.nbytes + region.nbytes),
        }

    def shadow_map(self):
        """The card's shadow map (side x side float32 1/z) from the light pages."""
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
    size = parse_size(args.size)
    resize = parse_resize(args.resize) if args.resize else {}
    sized, now = [], size  # each frame's size: --size, then --resize's
    for index in range(args.frames):
        now = resize.get(index, now)
        sized.append(now)
    views = [
        scenes.with_size(view, sized[index])
        for index, view in enumerate(framegraph.cameras(scene, args.frames))
    ]
    most = parse_size(args.max_size, "--max-size") if args.max_size else size
    table = show_table()
    args.shade = args.shade or args.light
    materials = shade.load(args.scene) if args.shade else None
    lights = light_data.load(args.scene, native.data_dir()) if args.light else None
    terms = tuple(args.light_terms.split(","))
    if args.indirect:
        if not args.light:
            raise SystemExit("--indirect lights a lit frame: it needs --light")
        terms += ("indirect",)
        lights = {**lights, "indirect_packed": True}  # the reference from the packed
    report = {
        "scene": args.scene,
        "frames": args.frames,
        "path": "push" if args.push else "read",
        "shaded": bool(args.shade),
        "lit": bool(args.light),
        "size": list(size),
        "grid": list(card_thread.card_grid(size)),
    }
    if args.light:
        report["light_terms"] = list(terms)
    report["table_differs"] = table_check(table)
    sun_set = parse_sun(args.sun) if args.sun else None
    if (sun_set or args.sun_path or args.check_sun) and not args.light:
        raise SystemExit("--sun, --sun-path and --check-sun move a lit frame's sun")
    toned = (args.exposure, args.saturation, args.curve) != (None, None, None)
    if toned and not args.light:
        raise SystemExit("--exposure, --saturation and --curve tone a lit frame")
    if args.indirect_switch and not args.indirect:
        raise SystemExit("--indirect-switch flips the field of an --indirect frame")
    if args.bounces is not None and not args.indirect:
        raise SystemExit("--bounces sets the passes of an --indirect frame's field")
    if args.card_field and not args.indirect:
        raise SystemExit("--card-field updates an --indirect frame's field")
    if args.card_field:
        report["card_field"] = {  # without RF_APPLY the host applies between steps
            "mode": "stepped: the host's float apply (indirect.apply) between "
            "UPDATE's steps, no RF_APPLY; 0 passes: the whole update in the thread"
        }
    if toned:  # set_tone before the first frame; the reference with the same numbers
        tone = live_tone(lights, args)
        lights = {**lights, **tone}
        report["tone"] = tone
    indirect_at = {}  # frame index: the indirect switch it was rendered with
    sun_at = {}  # frame index: the sun it was rendered with, once moved
    passes_run = []  # each SUN command: angles, word, host ms
    checked = {0, args.frames // 2, args.frames - 1}
    checked |= {index for index in resize if index < args.frames}  # each new size's
    if args.indirect_switch and args.indirect_switch < args.frames:
        checked.add(args.indirect_switch)  # the first frame with the switch off
    checked = sorted(checked)
    surfaces = sorted({0, args.frames - 1}) if args.check_surface else []
    kept, surface_got, glass_surface_got = {}, {}, {}
    with km.Session(log=card.log) as session:
        live = Loop(
            session,
            scenes.with_size(scene, size),
            args.push,
            args.list_slots,
            table,
            report,
            materials,
            lights,
            terms,
            card_thread.capacity([most], lights is not None)._replace(
                recount=args.recount
            ),
            card_field=args.card_field,
        )
        workers = live.workers
        sun_got = None
        if toned:
            report["tone"]["host_ms"] = round(live.set_tone(**tone), 3)
        if args.bounces is not None:  # set_bounces before the first frame
            host_ms = live.set_bounces(args.bounces)
            lights = {**lights, "bounces": args.bounces}  # the reference's the same
            report["bounces"] = {
                "n": args.bounces,
                "host_ms": round(host_ms, 3),
                "field_update": live.field_update,
            }

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
            if live.field_update is not None:
                passes_run[-1]["field_update"] = live.field_update
            if word not in (0, None):
                card.log(f"SUN before frame {index}: word {word}")
            return word

        if sun_set:
            word = move_sun(*sun_set, 0)
            if args.check_sun and word == 0:
                sun_got = live.shadow_map()
        image = np.empty((size[1], size[0], 4), dtype=np.uint8)
        resized = []  # each --resize: frame, size, host ms, reloaded
        regrown = []  # each growth of the lists (Loop.grow_lists): frame, needs, ms
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
            if args.indirect_switch and index and index % args.indirect_switch == 0:
                live.set_indirect(not live.indirect)
            if args.indirect:
                indirect_at[index] = live.indirect
            if index in resize:
                reloads = len(live.reloads)
                try:
                    ms = live.set_size(resize[index])
                except card_thread.L1Refused as error:
                    report["refused"] = str(error)
                    card.log(f"--resize at frame {index}: {error}")
                    break
                resized.append(
                    {
                        "frame": index,
                        "size": list(resize[index]),
                        "host_ms": round(ms, 1),
                        "reloaded": len(live.reloads) > reloads,
                    }
                )
                image = np.empty((sized[index][1], sized[index][0], 4), dtype=np.uint8)
            grown = len(live.lists_grown)
            try:
                event, times = live.frame(view["camera"], image)
            except ListsRefused as error:
                report["refused"] = str(error)
                card.log(f"frame {index}: {error}")
                break
            # its lists grown, the frame again
            regrown.extend({"frame": index, **g} for g in live.lists_grown[grown:])
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
                    session, live.uploaded["frame"], live.surface, live.grid
                )
                if live.glass_surface is not None:
                    glass_surface_got[index] = light.read_surface(
                        session, live.glass["frame"], live.glass_surface, live.grid
                    )
            if word != 0:
                card.log(f"frame {index}: {event.kind}, word {word}")
                break
        live.stop()
        running = live.running
        report["status"] = card.check_run(running, workers, 0, [])
        if args.card_field:  # UPDATE's times rows since the last load
            report["card_field"]["updates"] = update.times(
                running.thread.read("update")
            )
        if resize:
            report["resize"] = resized
            report["reloads"] = live.reloads
        if regrown:
            report["lists_grown"] = regrown
        # the times of the frames since the last load (a reload starts them again)
        loaded_at = max(
            [entry["frame"] for entry in resized if entry["reloaded"]]
            + [entry["frame"] for entry in regrown],
            default=0,
        )
        rows, numbers = times_rows(
            running.thread.read("times"), len(words_back) - loaded_at
        )
        sun_rows = None
        if passes_run:
            sun_commands = sum(
                run["word"] is not None and run["frame"] >= loaded_at
                for run in passes_run
            )
            if loaded_at and live.sun is not None:
                sun_commands += 1  # the reload's own SUN command
            sun_rows, _ = times_rows(running.thread.read("sun_times"), sun_commands)
        (profile_pages,) = card.read(session, live.show_profile)
        layer_pages = {"opaque": live.uploaded}
        for name in ("masked", "glass"):
            if getattr(live, name) is not None:
                layer_pages[name] = getattr(live, name)
        report["stage_profile"], raw_pages = {}, {}
        for layer, tensors in layer_pages.items():
            (fill_pages,) = card.read(session, tensors["prof"])
            (pass_pages,) = card.read(session, tensors["sprof"])
            raw_pages[f"{layer}_fill"], raw_pages[f"{layer}_passes"] = (
                fill_pages,
                pass_pages,
            )
            report["stage_profile"][layer] = {
                **passes_profile(pass_pages.view(np.uint32).reshape(workers, -1)),
                "fill": fill_profile(fill_pages.view(np.uint32).reshape(workers, -1)),
            }
        for name in ("masked", "glass"):
            if getattr(live, name) is None or live.shade_profile is None:
                continue
            (layer_shade,) = card.read(session, getattr(live, name)["shade_profile"])
            report[f"shade_profile_{name}"] = shade_profile(
                layer_shade.view(np.uint32).reshape(workers, -1).astype(np.int64)
            )
            raw_pages[f"{name}_shade"] = layer_shade
        if live.shade_profile is not None:
            (shade_pages,) = card.read(session, live.shade_profile)
            report["shade_profile"] = shade_profile(
                shade_pages.reshape(workers, -1).astype(np.int64)
            )
            raw_pages["opaque_shade"] = shade_pages
        if live.light_profile is not None:
            (light_pages,) = card.read(session, live.light_profile)
            report["light_profile"] = light_profile(
                light_pages.reshape(workers, -1).astype(np.int64),
                len(lights["lamp_intensity"]) if "lamps" in terms else 0,
            )
            raw_pages["light"] = light_pages
        if live.glass_surface is not None:
            (glass_light,) = card.read(session, live.glass_surface["profile"])
            report["light_profile_glass"] = light_profile(
                glass_light.reshape(workers, -1).astype(np.int64),
                len(lights["lamp_intensity"]) if "lamps" in terms else 0,
            )
            raw_pages["light_glass"] = glass_light
        if args.profile_pages:  # each stage's profile pages of the last frame
            np.savez(args.profile_pages, **raw_pages)
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
    layers = [name for name in ("masked", "glass") if getattr(live, name) is not None]
    if layers:
        report["card_layer_ms"] = {
            layer: round(
                float(np.median(rows[:, F[f"RF_T_{layer.upper()}"]])) * ticks_ms, 4
            )
            for layer in ["opaque"] + layers
        }
        for layer in layers:
            report[f"card_shade_{layer}_ms"] = round(
                float(np.median(rows[:, F[f"RF_T_SHADE_{layer.upper()}"]])) * ticks_ms,
                4,
            )
    if live.glass is not None:
        report["card_light_glass_ms"] = round(
            float(np.median(rows[:, F["RF_T_LIGHT_GLASS"]])) * ticks_ms, 4
        )
    report["card_parts_ms"] = {
        part: round(float(np.median(rows[:, F[f"RF_O_{part.upper()}"]])) * ticks_ms, 4)
        for part in ("count", "sum", "hist", "hsum", "middle", "place", "setup", "fill")
    }
    middle_parts = ("take", "heavy_microseconds", "cut", "sort", "assign", "compact")
    report["card_middle_parts_ms"] = {  # the middle pass's (frame.h RF_O_*)
        part: round(float(np.median(rows[:, F[f"RF_O_{part.upper()}"]])) * ticks_ms, 4)
        for part in (*middle_parts, "pages", "give")
    }
    if layers:  # each layer's parts (the other layers' own words)
        names = ("count", "sum", "hist", "hsum", "middle", "place", "setup", "fill")
        own = {
            layer: {
                part: rows[
                    :, F[f"RF_T_{layer[0].upper()}_PARTS"] + F[f"RF_O_{part.upper()}"]
                ]
                for part in names
            }
            for layer in layers
        }
        report["card_layer_parts_ms"] = {
            "opaque": {
                part: round(
                    float(
                        np.median(
                            rows[:, F[f"RF_O_{part.upper()}"]]
                            - sum(own[layer][part] for layer in layers)
                        )
                    )
                    * ticks_ms,
                    4,
                )
                for part in names
            },
        }
        for layer in layers:
            report["card_layer_parts_ms"][layer] = {
                part: round(float(np.median(values)) * ticks_ms, 4)
                for part, values in own[layer].items()
            }
    report["heavy_tiles"] = rows[:, F["RF_O_HEAVY"]].tolist()[:4]
    report["heavy_tiles_max"] = int(rows[:, F["RF_O_HEAVY"]].max())
    report["items_max"] = int(rows[:, F["RF_O_ITEMS"]].max())
    report["list_slots_used_max"] = int(rows[:, F["RF_O_SLOTS"]].max())
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
    if args.indirect_switch:
        report["indirect_switch"] = {
            "every": args.indirect_switch,
            "off": [index for index, on in indirect_at.items() if not on],
        }
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
            views[index], materials, lights_of(index), got, glass_surface_got.get(index)
        )
    for index, got in kept.items():
        mode = "_lit" if args.light else "_shaded" if args.shade else ""
        tag = f"{args.scene}_{index:04d}{mode}"
        if folder:
            png.write_png(folder / f"{tag}.png", got[..., :3])
        if args.light:
            if not args.check_light:
                continue
            frame_terms = terms
            if not indirect_at.get(index, True):  # the switch off: as without the field
                frame_terms = tuple(term for term in terms if term != "indirect")
            light_checks[index], want = light.light_check(
                views[index], materials, lights_of(index), got, frame_terms
            )
            light_checks[index]["indirect"] = "indirect" in frame_terms
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


def live_tone(lights, args):
    """The tone step's numbers of --exposure, --saturation, --curve, the scene's
    (lights) where an option is not given."""
    tone = tone_of(lights)
    for name in tone:
        if getattr(args, name) is not None:
            tone[name] = type(tone[name])(getattr(args, name))
    return tone


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
    report["quadrants"] = sorted(set(rows[:, F["RF_S_QUADRANTS"]].tolist()))
    report["total_ms_first"] = round(float(rows[0, F["RF_S_TOTAL"]] * ticks_ms), 4)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="tiny")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument(
        "--size",
        default="1024x1024",
        help="the frame's WxH in pixels, any positive size (scenes.frame_grid)",
    )
    parser.add_argument(
        "--max-size",
        default=None,
        help="the capacity loaded, as the largest WxH it renders (default --size); "
        "memory alone bounds it",
    )
    parser.add_argument(
        "--recount",
        action="store_true",
        help="COUNT counts the heavy tiles' histograms again even where they fit L1",
    )
    parser.add_argument(
        "--resize",
        default=None,
        help="N:WxH,...: set_size before frame N (past the capacity: a reload)",
    )
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
        "--indirect",
        action="store_true",
        help="lit: indirect light from the field (the reference's term indirect)",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="lit: the tone step's exposure (default the scene's), set before frame 0",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=None,
        help="lit: the tone step's saturation (default the scene's, 1: none)",
    )
    parser.add_argument(
        "--curve",
        type=int,
        choices=(0, 1),
        default=None,
        help="lit: the tone curve, 0 x / (1 + x), 1 filmic (default the scene's)",
    )
    parser.add_argument(
        "--bounces",
        type=int,
        choices=range(4),
        default=None,
        help="with --indirect: the field's bounce passes 0 to 3 (default the scene's)",
    )
    parser.add_argument(
        "--card-field",
        action="store_true",
        help="with --indirect: the field's update after a sun or bounce change by "
        "the frame thread's UPDATE (without RF_APPLY its passes' applies on the host)",
    )
    parser.add_argument(
        "--indirect-switch",
        type=int,
        default=0,
        help="with --indirect: the field switched off and on every N frames",
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
    parser.add_argument(
        "--profile-pages", default=None, help="an .npz of the stages' profile pages"
    )
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
