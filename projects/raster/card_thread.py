"""A frame driven by the work cluster's thread (`thread/frame.c`), each step it
reaches checked against the host's (`passes`) and the image against the reference;
the mesh uploaded once, the camera at the load. `python -m projects.raster.card_thread
--scene tiny [--steps N] [--no-image]` prints one JSON line; exits 1 unless exact."""

import argparse
import contextlib
import json
import math
import re
import sys
import time
from typing import NamedTuple

import numpy as np

import komorebi as km
from komorebi.bench import power
from komorebi.program.thread import BANK_OF_CLUSTER, TIMER_HZ
from projects.raster import card, card_passes, lists, passes, run
from reference import raster, scenes

THREAD = lists.HERE / "thread"
KERNELS = lists.HERE / "kernels"
FRAME_H = THREAD / "frame.h"
F = {
    name: int(value.rstrip("u"), 0)
    for name, value in re.findall(
        r"^#define (RF_\w+)\s+(\d+u?)\b", FRAME_H.read_text(), re.M
    )
}
M = {  # middle.h's numbers
    name: int(value.rstrip("u"), 0)
    for name, value in re.findall(
        r"^#define (RM_\w+)\s+(\d+u?)\b", (THREAD / "middle.h").read_text(), re.M
    )
}
# expressions in frame.h
F["RF_W_PROF"] = F["RF_W_FRAME"] + F["RF_FRAME_MAX"]
F["RF_W_REPEAT"] = F["RF_W_PROF"] + 1
F["RF_W_SPROF"] = F["RF_W_REPEAT"] + 1
for i, name in enumerate(["COUNTERS", "HIST", "SLOTS", "HPREFIX"]):
    F[f"RF_W_{name}"] = F["RF_W_SPROF"] + 1 + i
F["RF_W_BALANCE"] = F["RF_W_HPREFIX"] + 1
F["RF_W_DISPLAY"] = F["RF_W_BALANCE"] + 1
F["RF_W_SHOWPROF"] = F["RF_W_DISPLAY"] + 1
for i, name in enumerate(["SHADE", "TEXTURES", "SHADE_TABLE", "SHADEPROF"]):
    F[f"RF_W_{name}"] = F["RF_W_SHOWPROF"] + 1 + i
for i, name in enumerate(["MESH", "BLOCKS", "LISTS", "LIST_SLOTS", "DIR", "SHADE"]):
    F[f"RF_W_M_{name}"] = F["RF_W_SHADEPROF"] + 1 + i
F["RF_W_M_FRAMES"] = F["RF_W_M_SHADE"] + 1
F["RF_W_M_FRAME"] = F["RF_W_M_FRAMES"] + 1
F["RF_W_SURFACE"] = F["RF_W_M_FRAME"] + F["RF_FRAME_MAX"]
F["RF_W_SUN_PAGES"] = F["RF_W_SURFACE"] + 1
F["RF_WORDS"] = F["RF_W_SUN_PAGES"] + 1
F["RF_GLASS_WORDS_AT"] = F["RF_PROFILES_AT"] + 64
F["RF_GRID_AT"] = F["RF_GLASS_WORDS_AT"] + 128
F["RF_G_SURFACE"] = F["RF_G_FRAME"] + F["RF_FRAME_MAX"]
F["RF_G_WORDS"] = F["RF_G_SURFACE"] + 1
F["RF_AREA_GLASS"] = (
    F["RF_AREA_SUN"] + F["RF_AREAS_A_QUADRANT"] * F["RF_SUN_QUADRANTS_MAX"]
)
F["RF_AREAS"] = F["RF_AREA_GLASS"] + 1
F["RF_CAMERA_BYTES"] = F["RF_AREAS"] * F["RF_AREA_BYTES"]
S, K = card_passes.S, lists.K
SHOW_H = KERNELS / "show.h"
SHOW = {
    name: int(value, 0) for name, value in lists.CONSTEXPR.findall(SHOW_H.read_text())
}
GRID_H = KERNELS / "grid.h"
GRID = {
    name: int(value, 0) for name, value in lists.CONSTEXPR.findall(GRID_H.read_text())
}
SHADE_H = KERNELS / "shade.h"
SH = {
    name: int(value, 0) for name, value in lists.CONSTEXPR.findall(SHADE_H.read_text())
}
APPLY_H = KERNELS / "apply.h"
A = {
    name: int(value, 0) for name, value in lists.CONSTEXPR.findall(APPLY_H.read_text())
}
CLUSTER = 1
PAGES_PER_ITEM = 5  # the frame's pages per tile (card.words)
TILE_ROWS = 32  # the per-tile tables' rows: a capacity is a whole number of them
WORK_LEAST = 256 << 10  # the thread's work region at the least (work_bytes)
UPDATE_HANDLER = 3  # the thread's UPDATE (thread/update.c; update.py)
APPLY_HANDLER = 4  # the thread's apply (thread/apply.c; apply.py)


class Capacity(NamedTuple):
    """What a load sizes tables, tensors and regions from: `tiles` (TILE_ROWS each);
    `wide`, 32-bit box words past the narrow grid (kernels/grid.h); `recount`, the heavy
    tiles' histograms counted again (setup.h RASTER_RECOUNT); `cores`, the fill cores
    the heavy and band tables follow (card.heavy_capacity; None: host-stepped)."""

    tiles: int
    wide: bool
    recount: bool = False
    cores: int | None = None


def narrow(grid):
    """True when the grid (TX, TY) fits the narrow words (kernels/grid.h)."""
    across, down = grid
    return across <= GRID["kNarrowAcross"] and down <= GRID["kNarrowDown"]


# the capacity's steps (step_of) from the SUN quadrant's tiles, the default frame's
STEP_BASE = GRID["kGridDefault"] ** 2


def step_of(tiles):
    """The least step of at least `tiles`: STEP_BASE times 1, 1.5, 2, 3, 4, 6, 8, ...
    (2^k and 1.5 x 2^k), so that a new frame size rarely builds the kernels again (a
    frame's work follows its own tiles, not the capacity: kernels/grid.h)."""
    step = STEP_BASE
    while step < tiles:
        if 3 * step // 2 >= tiles:
            return 3 * step // 2
        step *= 2
    return step


def need(sizes, lit=False):
    """The tiles a load must hold to render every frame size (W, H) of `sizes`: their
    most tiles (and, lit, the SUN pass's quadrant of kGridDefault x kGridDefault tiles,
    rendered into the eye's tensors) rounded up to TILE_ROWS."""
    tiles = max(across * down for across, down in map(scenes.frame_grid, sizes))
    if lit:
        tiles = max(tiles, GRID["kGridDefault"] ** 2)
    return -(-tiles // TILE_ROWS) * TILE_ROWS


def capacity(sizes, lit=False, at_least=None):
    """The capacity that renders every frame size (W, H) of `sizes`: their need
    rounded up to its step (step_of); wide when one's grid passes the narrow words; at
    least `at_least` (a Capacity) when given: a capacity never shrinks."""
    tiles = step_of(need(sizes, lit))
    wide = not all(narrow(scenes.frame_grid(size)) for size in sizes)
    if at_least is not None:  # and its recount (live.py --recount, or L1's)
        tiles, wide = max(tiles, at_least.tiles), wide or at_least.wide
        return Capacity(tiles, wide, at_least.recount, at_least.cores)
    return Capacity(tiles, wide)


def covers(held, size, lit=False):
    """True when the capacity `held` renders a frame of size (W, H): its need (not its
    step: a capacity L1 bounded below a step holds what it holds)."""
    wide = not narrow(scenes.frame_grid(size))
    return need([size], lit) <= held.tiles and (held.wide or not wide)


def default_capacity():
    """The default frame's capacity (scenes.WIDTH x scenes.HEIGHT)."""
    return capacity([(scenes.WIDTH, scenes.HEIGHT)])


def thread_sizes(held):
    """frame.h's sizes that follow the capacity `held`: its tiles (RM_TILES), heavy
    tiles and bands' items (RM_MAX_HEAVY, RM_BAND_ITEMS: card.heavy_capacity)."""
    tiles, (heavy, band_items) = held.tiles, card.heavy_capacity(held)
    items = tiles + band_items
    out = {
        "RF_HIST_MAX": heavy,
        "RF_HIST_BYTES": 64 * heavy,
        "RF_HPREFIX_BYTES": 128 * heavy,
        "RF_COUNTER_BYTES": 8 * tiles,
        "RF_SLOTS_BYTES": 4 * items,
        "RF_ITEMS_BYTES": 4 * tiles + 2 * items,
        "RF_DISPLAY_PAGES": tiles,
        "RF_SUMS_HEAVY_AT": 8 * tiles,
    }
    out["RF_BASE_AT"] = out["RF_ITEMS_BYTES"]
    out["RF_HPREFIX_ADDR_AT"] = out["RF_BASE_AT"] + 4 * items
    out["RF_SHADE_ADDR_AT"] = out["RF_HPREFIX_ADDR_AT"] + 4
    out["RF_SUMS_BYTES"] = out["RF_SUMS_HEAVY_AT"] + 128 * heavy
    out["RF_FRAME_WORDS_AT"] = out["RF_HPREFIX_ADDR_AT"] + 64
    out["RF_ITEM_WORDS_AT"] = out["RF_FRAME_WORDS_AT"] + 64
    out["RF_CORE_WORDS_AT"] = out["RF_ITEM_WORDS_AT"] + 8 * items
    out["RF_ITEMS_REGION_BYTES"] = out["RF_CORE_WORDS_AT"] + 16 * F["RF_MAX_CORES"]
    return out


def work_bytes(held):
    """The thread's own region (frame.c work_t): 32 B a tile (entries, rows, cost, heavy
    index, first item), 60 B an item (rm_item, cost, order, spare, owner, header, next
    slot), a fixed part (heavy tiles' counts, cores, areas) and a page to spare; whole
    pages, at least WORK_LEAST (the 1,024-tile frame's region as before)."""
    tiles, (heavy, band_items) = held.tiles, card.heavy_capacity(held)
    items = tiles + band_items
    cores = F["RF_MAX_CORES"]
    fixed = (
        8 * 2 * cores  # load, heap
        + 4 * (cores + 1)  # first_item
        + 8 * heavy * 2 * 16  # histograms
        + 4 * 2 * cores  # slot_counts, item_counts
        + 4 * (1 + 2 * heavy)  # tile_item's last, heavy, bands
        + 4 * (F["RF_AREAS"] + 1) * (cores + 1)  # ranges, cuts
    )
    total = 32 * tiles + 60 * items + fixed + 4096
    return max(-(-total // 4096) * 4096, WORK_LEAST)


def same_numbers(held=None):
    """The numbers frame.h takes from the kernels' headers (at the capacity `held`, by
    default the default frame's), as pairs that differ."""
    held = held or default_capacity()
    T = thread_sizes(held)
    P = card_passes.sizes(held.tiles, False, *card.heavy_capacity(held))
    pairs = {
        "RF_HANDLER_COUNT": S["kHandlerCount"],
        "RF_HANDLER_SETUP": S["kHandlerSetup"],
        "RF_HANDLER_HIST": S["kHandlerHist"],
        "RF_HANDLER_PLACE": S["kHandlerPlace"],
        "RF_HANDLER_SUM": S["kHandlerSum"],
        "RF_HANDLER_HSUM": S["kHandlerHsum"],
        "RF_BASE_AT": (T["RF_BASE_AT"], P["kBaseAt"]),
        "RF_HPREFIX_ADDR_AT": (T["RF_HPREFIX_ADDR_AT"], P["kHprefixAddrAt"]),
        "RF_HPREFIX_BYTES": (T["RF_HPREFIX_BYTES"], P["kHprefixPageBytes"]),
        "RF_HIST_MAX": (T["RF_HIST_MAX"], P["kHeavyMax"]),
        "RF_SUMS_HEAVY_AT": (T["RF_SUMS_HEAVY_AT"], P["kSumsHeavyAt"]),
        "RM_BAND_ITEMS": (T["RF_SLOTS_BYTES"] // 4 - held.tiles, P["kBandItems"]),
        "RF_HANDLER_FILL": K["kHandlerFill"],
        "RF_PAGES_PER_ITEM": PAGES_PER_ITEM,
        "RF_FRAME_MAX": K["kDirFrameMax"],
        "RF_COUNTER_BYTES": (T["RF_COUNTER_BYTES"], P["kCountBytes"]),
        "RF_HEAVY_BYTES": S["kHeavyBytes"],
        "RF_SLOTS_BYTES": (T["RF_SLOTS_BYTES"], P["kSlotsBytes"]),
        "RF_ITEMS_BYTES": (T["RF_ITEMS_BYTES"], P["kItemsBytes"]),
        "RF_FRAME_WORDS_AT": (T["RF_FRAME_WORDS_AT"], P["kFrameWordsAt"]),
        "RF_ITEM_WORDS_AT": (T["RF_ITEM_WORDS_AT"], P["kItemWordsAt"]),
        "RF_CORE_WORDS_AT": (T["RF_CORE_WORDS_AT"], P["kCoreWordsAt"]),
        "RF_ITEMS_REGION_BYTES": (
            T["RF_ITEMS_REGION_BYTES"],
            P["kCoreWordsAt"] + 16 * S["kMaxFillCores"],
        ),
        "RF_RANGES_AT": S["kRangesAt"],
        "RF_COSTS_AT": S["kCostsAt"],
        "RF_COST_KNOTS": S["kCostKnots"],
        "RF_MAX_CORES": S["kMaxFillCores"],
        "RF_COSTS_AT + costs <= RF_PROFILES_AT": (
            min(F["RF_COSTS_AT"] + 128 * S["kMaxFillCores"], F["RF_PROFILES_AT"]),
            F["RF_COSTS_AT"] + 128 * S["kMaxFillCores"],
        ),
        "RF_PROFILES_AT + 64 <= RF_AREA_BYTES": (
            min(F["RF_PROFILES_AT"] + 64, F["RF_AREA_BYTES"]),
            F["RF_PROFILES_AT"] + 64,
        ),
        "RF_GLASS_WORDS_AT + 128 <= RF_AREA_BYTES": (
            min(F["RF_GLASS_WORDS_AT"] + 128, F["RF_AREA_BYTES"]),
            F["RF_GLASS_WORDS_AT"] + 128,
        ),
        "RF_G_WORDS in two lines": (min(4 * F["RF_G_WORDS"], 128), 4 * F["RF_G_WORDS"]),
        "RF_HANDLER_SHOW": SHOW["kHandlerShow"],
        "RF_SHOW_TABLE_BYTES": 4 * SHOW["kShowTableWords"],
        "RF_HANDLER_SHADE": SH["kHandlerShade"],
        "RF_HANDLER_LIGHT": SH["kHandlerLight"],
        "RF_HANDLER_APPLY": A["kHandlerApply"],
        "RF_SHADE_MERGED_DEPTH": SH["kShadeMergedDepth"],
        "RF_SHADE_GLASS": SH["kShadeGlass"],
        "RF_LIGHT_LINEAR": SH["kLightLinear"],
        "RF_LIGHT_GLASS": SH["kLightGlass"],
        "RF_SHOW_RAW": SHOW["kShowRaw"],
        "RF_GRID_DEFAULT": GRID["kGridDefault"],
        "RF_NARROW_ACROSS": GRID["kNarrowAcross"],
        "RF_NARROW_DOWN": GRID["kNarrowDown"],
        "RF_SUN_TILES_ACROSS": GRID["kGridDefault"],
        "kGridDefault": (GRID["kGridDefault"], raster.TILES_X),
        "RF_GRID_AT + 64 <= RF_AREA_BYTES": (
            min(F["RF_GRID_AT"] + 64, F["RF_AREA_BYTES"]),
            F["RF_GRID_AT"] + 64,
        ),
        "RF_SUN_QUADRANTS_MAX": F["RF_SUN_ACROSS_MAX"] ** 2,
        "RF_SUN_CAMERA_BYTES": 4 * SH["kCamWordsShade"],
        "RF_SUN_ACROSS_AT after the cameras": (
            max(F["RF_SUN_ACROSS_AT"], F["RF_SUN_QUADRANTS_MAX"] * 64),
            F["RF_SUN_ACROSS_AT"],
        ),
        "RF_SUN_TABLE_AT + table": (
            F["RF_SUN_TABLE_AT"] + SH["kLightTableBytes"],
            F["RF_SUN_BYTES"],
        ),
        "RF_SHADE_ADDR_AT": (T["RF_SHADE_ADDR_AT"], P["kShadeAddrAt"]),
        "kRecShade": (SH["kRecShade"], S["kRecShadeWord"]),
        # PLACE's list and directory words against the fill's
        "kLayDirIds": (S["kLayDirIds"], K["kDirIds"]),
        "kLayDirFrame": (S["kLayDirFrame"], K["kDirFrame"]),
        "kLayItemEmpty": (S["kLayItemEmpty"], K["kItemEmpty"]),
        "kLayRow0Shift": (S["kLayRow0Shift"], K["kItemRow0Shift"]),
        "kLayRow1Shift": (S["kLayRow1Shift"], K["kItemRow1Shift"]),
        "kLayMarkTile": (S["kLayMarkTile"], K["kMarkTile"]),
        "kLayMarkEnd": (S["kLayMarkEnd"], K["kMarkEnd"]),
        "kLayHdrTile": (S["kLayHdrTile"], K["kHdrTile"]),
        "kLayHdrCount": (S["kLayHdrCount"], K["kHdrCount"]),
        "kLayHdrRow0": (S["kLayHdrRow0"], K["kHdrRow0"]),
        "kLayHdrRow1": (S["kLayHdrRow1"], K["kHdrRow1"]),
        "kDirPageBytes": (S["kDirPageBytes"], K["kDirBytes"]),
    }
    pairs = {
        name: value if isinstance(value, tuple) else (F[name], value)
        for name, value in pairs.items()
    }
    return {name: pair for name, pair in pairs.items() if pair[0] != pair[1]}


def card_grid(size):
    """The grid (TX, TY) of a frame of size (W, H) on the card: scenes.frame_grid's
    rule (ValueError for a side under 1). Any grid renders; its tiles set the capacity
    a load needs (capacity), which memory alone bounds (memory_check)."""
    return scenes.frame_grid(size)


def grid_word(size):
    """The frame's grid word (frame.h RF_GRID_AT; kernels/grid.h): TX | TY << 16, 0 for
    the default 32 x 32 (every command as before)."""
    across, down = card_grid(size)
    if (across, down) == (GRID["kGridDefault"], GRID["kGridDefault"]):
        return 0
    return across | down << 16


def camera_region(scene, profiles=None, glass=None):
    """The camera's numbers, then room for COUNT's and SETUP's ranges and costs (an area
    per layer and camera, frame.h RF_AREAS); at RF_PROFILES_AT `profiles` (live.py: the
    masked and glass layers' profile tensors), at RF_GLASS_WORDS_AT `glass` (the glass
    layer's words, frame.h RF_G_*), else 0; at RF_GRID_AT the scene's grid word."""
    region = np.zeros((1, F["RF_CAMERA_BYTES"] // 4), dtype=np.uint32)
    region[0, : S["kCamWords"]] = card_passes.camera_page(scene)[0]
    region[0, F["RF_GRID_AT"] // 4] = grid_word(scenes.frame_size(scene))
    for at, values in (
        (F["RF_PROFILES_AT"], profiles),
        (F["RF_GLASS_WORDS_AT"], glass),
    ):
        if values is not None:
            region[0, at // 4 : at // 4 + len(values)] = values
    return region


def regions(
    scene,
    workers,
    show_table=None,
    sun=None,
    profiles=None,
    glass=None,
    held=None,
    update=None,
    apply_header=None,
):
    """The thread's regions in frame.h's order: the camera's page, then sizes (items,
    work, sums from the capacity `held`, by default program's); SHOW's block holds
    `show_table`; the sun region `sun` (light.sun_region); `profiles`, `glass`:
    camera_region's; the update region (update_words), then `update`'s others."""
    held = held or capacity([scenes.frame_size(scene)], at_least=default_capacity())
    sizes = thread_sizes(held)
    return {
        "camera": camera_region(scene, profiles, glass),
        "heavy": F["RF_HEAVY_BYTES"],
        "items": sizes["RF_ITEMS_REGION_BYTES"],
        "work": work_bytes(held),
        "sums": sizes["RF_SUMS_BYTES"],
        "times": F["RF_TIMES_BYTES"],
        "show": (
            F["RF_SHOW_TABLE_BYTES"]
            if show_table is None
            else np.asarray(show_table, dtype=np.uint32)
        ),
        "push": F["RF_PUSH_BYTES"],
        "sun": F["RF_SUN_BYTES"] if sun is None else np.asarray(sun, dtype=np.uint32),
        "sun_times": F["RF_TIMES_BYTES"],
        "update": update_words(update, apply_header),
        **{name: value for name, value in (update or {}).items() if name != "update"},
    }


def update_words(update=None, apply_header=None):
    """The update region (frame.h RF_R_UPDATE): words 0 to RF_U_OTHER - 1 the apply's
    (`apply_header`), the others the field's update's (`update`'s "update"); its
    bytes when neither is given."""
    if update is None and apply_header is None:
        return F["RF_UPDATE_BYTES"]
    words = np.zeros(F["RF_UPDATE_BYTES"] // 4, dtype=np.uint32)
    if update is not None:
        words[:] = np.asarray(update["update"], dtype=np.uint32)
    if apply_header is not None:
        own = F["RF_U_OTHER"]
        words[:own] = np.asarray(apply_header, dtype=np.uint32)[:own]
    return words


def frame_buffers(held, fill_program=None, passes_program=None):
    """The frame program's buffers but APPLY's, for the capacity `held`: the fill's,
    the passes', SHOW's ring, SHADE's (program's)."""
    fill_program = fill_program or card.program(held=held)
    passes_program = passes_program or card_passes.program(held=held)
    return [
        *fill_program.buffers,
        *passes_program.buffers,
        km.Buffer(
            SHOW["kCbShow"], "uint32", SHOW["kShowPageBytes"], SHOW["kShowPages"]
        ),
        *shade_buffers(),
    ]


def shade_buffers():
    """SHADE's buffers (kernels/shade.h): planes, table, asks, answers, the page slots
    and the white texel's page, the pack thread's space and its lit space."""
    return [
        km.Buffer(
            SH["kCbShadePlanes"], "uint32", K["kOutPageBytes"], SH["kShadePlanePages"]
        ),
        km.Buffer(SH["kCbShadeTable"], "uint32", SH["kShadeTableBytes"], 1),
        km.Buffer(
            SH["kCbShadeAsk"], "uint32", SH["kShadeAskBytes"], SH["kShadeAskPages"]
        ),
        km.Buffer(
            SH["kCbShadeDone"], "uint32", SH["kShadeDoneBytes"], SH["kShadeDonePages"]
        ),
        km.Buffer(
            SH["kCbShadeSlots"], "uint32", SH["kTexPageBytes"], SH["kShadeSlots"] + 1
        ),
        km.Buffer(SH["kCbShadeWork"], "uint32", SH["kShadeWorkBytes"], 1),
        km.Buffer(SH["kCbLightWork"], "uint32", SH["kLightWorkBytes"], 1),
    ]


def l1_tile_bytes(held):
    """A core's L1 buffers that follow the capacity `held` (setup.h): {name: bytes}.
    A tile: counters 8 B, histograms 64 B (recount: 64 B a heavy tile; PLACE's items
    region 4 B a tile, 14 B an item, in their space), slots 4 B an item, items page
    4 B + 2 B an item (an item a tile, and the bands' kBandItems)."""
    P = card_passes.sizes(held.tiles, held.recount, *card.heavy_capacity(held))
    return {
        "counters (kCbCount)": P["kCountBytes"],
        "histograms (kCbHist)": P["kHistBytes"],
        "slots (kCbSlots)": P["kSlotsBytes"],
        "items (kCbItems)": P["kItemsBytes"],
    }


L1_BYTES_A_TILE = 8 + 64 + 4 + 6  # l1_tile_bytes' growth per tile
L1_BYTES_A_TILE_RECOUNT = 8 + 18 + 4 + 6  # and with recount (the items region's 18)


def largest_fitting(held, fixed, free):
    """The largest capacity in TILE_ROWS below `held`'s whose L1 (`fixed` bytes that do
    not grow with tiles, and l1_tile_bytes) fits the `free` bytes; 0 when none does."""
    most = 0
    for tiles in range(TILE_ROWS, held.tiles, TILE_ROWS):
        if fixed + sum(l1_tile_bytes(held._replace(tiles=tiles)).values()) <= free:
            most = tiles
    return most


def l1_fallback(session, program, held, need):
    """When `program` at the capacity `held` (a step) does not fit a core's L1
    (fits_l1): the largest capacity that does with its buffers that do not grow with
    tiles, if it holds the `need` tiles; None otherwise (it fits, or the load's
    refusal as before)."""
    if fits_l1(session, program):
        return None
    used = sum(size for _, size in session.l1_rows(program)) + L1_SLOTS_MARGIN
    fixed = used - sum(l1_tile_bytes(held).values())
    most = largest_fitting(held, fixed, session.l1()[1])
    return held._replace(tiles=most) if need <= most < held.tiles else None


class L1Refused(ValueError):
    """A load refused for a core's L1: the frame's size (W, H) and the capacity
    `held`, the bytes L1 needs (`used`) and leaves for buffers (`free`), the buffers
    that grow with tiles, and the largest capacity that fits (`fixed`: the bytes that
    do not grow with tiles; None when not known, nothing loaded)."""

    def __init__(self, held, size, used, free, fixed=None):
        width, height = size
        across, down = scenes.frame_grid(size)
        grows = l1_tile_bytes(held)
        rows = ", ".join(f"{name} {bytes_:,} B" for name, bytes_ in grows.items())
        most = None if fixed is None else largest_fitting(held, fixed, free)
        needs = "needs" if fixed is not None else "needs at least"
        fits = f" The largest capacity that fits: {most} tiles." if most else ""
        a_tile = L1_BYTES_A_TILE_RECOUNT if held.recount else L1_BYTES_A_TILE
        super().__init__(
            f"a frame of {width} x {height} pixels is {across} x {down} = "
            f"{across * down} tiles (capacity {held.tiles}): a core's L1 {needs} "
            f"{used:,} B of {free:,} B. The buffers that grow with tiles "
            f"(about {a_tile} B a tile): {rows}.{fits}"
        )
        self.used, self.free, self.most = used, free, most

    @classmethod
    def of_budget(cls, held, size, error):
        """From the load's km.L1BudgetError: its table's total and room."""
        used, free = (
            int(n.replace(",", ""))
            for n in re.search(
                r"total\s+([\d,]+) B of ([\d,]+) B", error.table
            ).groups()
        )
        grows = sum(l1_tile_bytes(held).values())
        return cls(held, size, used, free, used - grows)


L1_SLOTS_MARGIN = 16 << 10  # the overlay slots beside l1_rows' (9,344 B measured)


def fits_l1(session, program):
    """Whether `program`'s L1 budget (session.l1_rows, and L1_SLOTS_MARGIN for the
    slots it leaves out) fits a core before a load tries it: a load refused for L1
    leaves the session's L1 held, and its retry reopened the card."""
    rows = session.l1_rows(program)
    return sum(size for _, size in rows) + L1_SLOTS_MARGIN <= session.l1()[1]


def l1_check(held, size, free):
    """Before anything is stopped or built: L1Refused when the buffers that grow with
    tiles alone (with recount, their least) take more than the `free` bytes a core's
    L1 leaves for buffers (session.l1()); the load's budget checks the rest."""
    least = held._replace(recount=True)
    grows = sum(l1_tile_bytes(least).values())
    if grows > free:
        raise L1Refused(least, size, grows, free)


def capacity_report(held, workers, dram):
    """The memory a capacity `held` takes: a core's L1 (the buffers that grow with
    tiles), the card's DRAM (`dram`: bytes of the tensors that grow, of the rest), the
    thread's regions."""
    sizes = thread_sizes(held)
    heavy, band_items = card.heavy_capacity(held)
    return {
        "tiles": held.tiles,
        "wide": held.wide,
        "recount": held.recount,
        "cores": held.cores,
        "heavy": heavy,
        "band_items": band_items,
        "l1_tile_buffers": sum(l1_tile_bytes(held).values()),
        "dram_by_tiles": dram[0],
        "dram_rest": dram[1],
        "thread_regions": (
            sizes["RF_ITEMS_REGION_BYTES"] + work_bytes(held) + sizes["RF_SUMS_BYTES"]
        ),
    }


def defines(held):
    """The kernels' build defines of the capacity `held` (kernels/grid.h)."""
    return card.capacity_defines(held)


def program(
    scene,
    words,
    workers,
    show_table=None,
    sun=None,
    profiles=None,
    glass=None,
    held=None,
    update=None,
    apply=False,
    apply_parts=None,
    apply_header=None,
):
    """The frame program: the passes' and the fill's kernels (kernels/frame_*.cpp),
    their buffers, the frame thread on cluster 1, for the capacity `held`; with
    `show_table` it serves FRAME, SUN, apply, with `update` also UPDATE (update.py;
    `apply`: RF_APPLY); `apply_parts` (apply.frame_parts) adds APPLY; else regions'."""
    held = held or capacity([scenes.frame_size(scene)], at_least=default_capacity())
    fill_program = card.program(held=held)
    passes_program = card_passes.program(held=held)
    compile_args, compute = card.compile_args(), dict(fill_program.compute)
    apply_buffers, apply_defines, apply_handlers = [], {}, {}
    if apply_parts is not None:  # its tensors after the fill's, on both movers
        compile_args = {
            role: [*given, *apply_parts["args"]] for role, given in compile_args.items()
        }
        modes = {**compute.get("unpack_to_dest_mode", {})}
        compute["unpack_to_dest_mode"] = {**modes, **apply_parts["unpack_to_dest_mode"]}
        apply_buffers, apply_defines = apply_parts["buffers"], apply_parts["defines"]
        apply_handlers = {"apply": A["kHandlerApply"]}
    frame_program = km.Program(
        "raster_frame",
        kernels={
            role: KERNELS / f"frame_{role}.cpp"
            for role in ("reader", "compute", "writer")
        },
        buffers=[*frame_buffers(held, fill_program, passes_program), *apply_buffers],
        compile_args=compile_args,  # the fill's (and APPLY's) tensors: DRAM tensors'
        compute=compute,
        defines={**card.spread_parts()[2], **defines(held), **apply_defines},
        depth=1,
        handlers={
            **fill_program.handlers,
            **passes_program.handlers,
            "show": SHOW["kHandlerShow"],
            "shade": SH["kHandlerShade"],
            "light": SH["kHandlerLight"],
            **apply_handlers,
        },
    )
    thread_defines = {
        "RM_TILES": f"{held.tiles}u",
        "RM_MAX_HEAVY": f"{card.heavy_capacity(held)[0]}u",
        "RM_BAND_ITEMS": f"{card.heavy_capacity(held)[1]}u",
        **({"RF_WIDE": "1"} if held.wide else {}),
        **({"RF_RECOUNT": "1"} if held.recount else {}),
        **({"RF_SERVE": "1"} if show_table is not None else {}),
        **({"RF_UPDATE": "1"} if update is not None else {}),
        **({"RF_APPLY": "1"} if update is not None and apply else {}),
    }
    handlers = {"frame": 1, "sun": 2, "apply": APPLY_HANDLER}
    sources = [THREAD / name for name in ("frame.c", "middle.c", "update.c", "apply.c")]
    if update is not None:
        if show_table is None:
            raise ValueError("UPDATE is served: the program needs show_table")
        handlers["update"] = UPDATE_HANDLER
    frame_program.thread(
        cluster=CLUSTER,
        sources=sources,
        includes=[THREAD],
        regions=regions(
            scene, workers, show_table, sun, profiles, glass, held, update, apply_header
        ),
        words=words,
        deadline=10.0,
        defines=thread_defines,
        **({"handlers": handlers} if show_table is not None else {}),
    )
    return frame_program


def frame_tensors(session, held):
    """A frame's tensors, five pages a tile of the capacity `held`, card.chunk_pages
    pages a tensor (the directory names at most kDirFrameMax of them)."""
    output_pages = held.tiles * PAGES_PER_ITEM
    chunk = card.chunk_pages(output_pages)
    return [
        card.tensor(session, np.zeros((min(chunk, output_pages - at), 1024)))
        for at in range(0, output_pages, chunk)
    ]


def frame_pages(frame):
    """Pages per tensor of a frame's tensors (frame_tensors): the directory's word."""
    return card.chunk_pages(sum(tensor.pages for tensor in frame))


def list_slots(slots, held):
    """The list slots of a capacity `held` for `slots` sized at the default frame's
    tiles: times sqrt(tiles / lists.TILES) past it (a triangle reaches more tiles as
    they shrink against it; RF_LISTS_SMALL when a frame needs more)."""
    return math.ceil(slots * max(1.0, math.sqrt(held.tiles / lists.TILES)))


def list_tensor(session, slots):
    """The lists tensor of at least `slots` slots: {"lists", "list_slots"}."""
    pages = -(-slots // lists.PAGE_SLOTS)
    return {
        "lists": card.tensor(session, np.zeros((pages, lists.PAGE_WORDS))),
        "list_slots": pages * lists.PAGE_SLOTS,
    }


def layer_tensors(session, blocks, workers, slots, held=None):
    """A mesh's own DRAM tensors: the mesh (once per scene), lists of `slots`,
    directory, frame (frame_tensors of the capacity `held`, by default the default
    frame's)."""
    return {
        "mesh": card.tensor(session, blocks),
        **list_tensor(session, slots),
        "dir": card.tensor(session, np.zeros((workers, K["kDirBytes"] // 4))),
        "frame": frame_tensors(session, held or default_capacity()),
    }


def capacity_tensors(session, workers, held):
    """The pages only the cores read and write whose size follows the capacity `held`:
    COUNT's counters, SUM's prefixes (SETUP's slots), HIST's histograms, HSUM's
    pages (a heavy tile's: card.heavy_capacity)."""
    sizes = thread_sizes(held)
    return {
        "counters": card.tensor(
            session, np.zeros((workers, sizes["RF_COUNTER_BYTES"] // 4))
        ),
        "slots": card.tensor(
            session, np.zeros((workers, sizes["RF_SLOTS_BYTES"] // 4))
        ),
        "hist": card.tensor(session, np.zeros((workers, sizes["RF_HIST_BYTES"] // 4))),
        "hprefix": card.tensor(
            session, np.zeros((workers, sizes["RF_HPREFIX_BYTES"] // 4))
        ),
    }


def tensors(session, blocks, workers, slots, held=None):
    """The DRAM tensors: mesh (once per scene), lists, directory, frame, profiles, and
    the pages only the cores read and write; sized for the capacity `held` (by default
    the default frame's)."""
    held = held or default_capacity()
    uploaded = {
        **layer_tensors(session, blocks, workers, slots, held),
        "prof": card.tensor(session, np.zeros((workers, 1024))),
        "sprof": card.tensor(session, np.zeros((workers, S["kSProfPageBytes"] // 4))),
        **capacity_tensors(session, workers, held),
    }
    session.device.finish()  # the uploads land before any frame runs
    return uploaded


def words(uploaded, blocks, cores, workers, steps, balance=True):
    thread_words = [0] * F["RF_WORDS"]
    thread_words[F["RF_W_BALANCE"]] = int(balance)
    thread_words[F["RF_W_CORES"]] = cores
    thread_words[F["RF_W_WORKERS"]] = workers
    thread_words[F["RF_W_BANK"]] = BANK_OF_CLUSTER[CLUSTER]
    thread_words[F["RF_W_MESH"]] = uploaded["mesh"].address
    thread_words[F["RF_W_BLOCKS"]] = blocks
    thread_words[F["RF_W_LISTS"]] = uploaded["lists"].address
    thread_words[F["RF_W_LIST_SLOTS"]] = uploaded["list_slots"]
    thread_words[F["RF_W_DIR"]] = uploaded["dir"].address
    thread_words[F["RF_W_STEPS"]] = steps
    thread_words[F["RF_W_FRAME_PAGES"]] = frame_pages(uploaded["frame"])
    thread_words[F["RF_W_FRAMES"]] = len(uploaded["frame"])
    for index, frame_tensor in enumerate(uploaded["frame"]):
        thread_words[F["RF_W_FRAME"] + index] = frame_tensor.address
    thread_words[F["RF_W_PROF"]] = uploaded["prof"].address
    thread_words[F["RF_W_SPROF"]] = uploaded["sprof"].address
    for name in ("counters", "hist", "slots", "hprefix"):
        thread_words[F[f"RF_W_{name.upper()}"]] = uploaded[name].address
    return thread_words


def masked_words(thread_words, layer, blocks, shade_address, prefix="RF_W_M_"):
    """The masked layer's words (frame.h RF_W_M_*; `prefix` RF_G_: the glass layer's
    words, area 0's) into `thread_words`: its tensors (layer_tensors), blocks and shade
    tensor."""
    thread_words[F[prefix + "MESH"]] = layer["mesh"].address
    thread_words[F[prefix + "BLOCKS"]] = blocks
    thread_words[F[prefix + "LISTS"]] = layer["lists"].address
    thread_words[F[prefix + "LIST_SLOTS"]] = layer["list_slots"]
    thread_words[F[prefix + "DIR"]] = layer["dir"].address
    thread_words[F[prefix + "SHADE"]] = shade_address
    thread_words[F[prefix + "FRAMES"]] = len(layer["frame"])
    for index, frame_tensor in enumerate(layer["frame"]):
        thread_words[F[prefix + "FRAME"] + index] = frame_tensor.address
    return thread_words


def result_dict(result):
    """The thread's result: code, what the middle pass found, each part in
    microseconds."""
    out_words = list(result.words) + [0] * (F["RF_OUT_WORDS"] - len(result.words))
    microseconds_per_tick = 1e6 / TIMER_HZ
    names = ["count", "sum", "hist", "hsum", "middle", "place", "setup", "fill"]
    return {
        "ran": result.ran,
        "state": result.state,
        "error": result.error,
        "code": result.code,
        "where": result.where,
        "thread_microseconds": result.microseconds,
        "slots": out_words[F["RF_O_SLOTS"]],
        "items": out_words[F["RF_O_ITEMS"]],
        "marks": out_words[F["RF_O_MARKS"]],
        "heavy": out_words[F["RF_O_HEAVY"]],
        "microseconds": {
            part: out_words[F[f"RF_O_{part.upper()}"]] * microseconds_per_tick
            for part in names
        },
        "middle_microseconds": {
            part: out_words[F[f"RF_O_{part}"]] * microseconds_per_tick
            for part in (
                "TAKE",
                "HEAVY_MICROSECONDS",
                "CUT",
                "SORT",
                "ASSIGN",
                "COMPACT",
                "PAGES",
                "GIVE",
            )
        },
    }


def expected(scene, records, cores, workers):
    """The host's middle pass on the reference's counts: what the thread must find."""
    triangle_count = len(scene["indices"])
    entries, rows = passes.counters(records, triangle_count, cores)
    heavy = passes.heavy_tiles(entries, rows, workers)
    bands = None
    if heavy:
        histogram_tiles = [tile for tile, _ in heavy]
        starts, ends = passes.histograms(records, triangle_count, cores)
        bands = passes.items_of(
            entries, rows, heavy, starts[:, histogram_tiles], ends[:, histogram_tiles]
        )
    middle = passes.middle(entries, rows, workers, bands=bands)
    pages = middle["pages"].reshape(-1, lists.SLOT)
    middle["found"] = {
        "slots": int(middle["slots"]),
        "items": len(middle["items"]),
        "marks": int((pages[:, 0] >= K["kMarkTile"]).sum()),
        "heavy": len(heavy),
    }
    return middle


def slots_differ(session, uploaded, records, middle):
    """Differing list slots by kind (header, end, record), for a small frame."""
    counts, offsets, members = raster.bin_triangles(records)
    packed = lists.pack_records(records)
    packed[:, K["kRecIndex"]] = 0
    want, _ = lists.build_lists(packed, offsets, members, middle["tiles_of"])
    (got,) = session.read(uploaded["lists"], first=0, pages=len(want))
    want = want.reshape(-1, lists.SLOT)
    got = got.view(np.uint32).reshape(-1, lists.SLOT)
    differs = (got != want).any(axis=1)
    kind = np.where(
        want[:, 0] == K["kMarkTile"],
        "header",
        np.where(want[:, 0] == K["kMarkEnd"], "end", "record"),
    )
    report = {
        name: int((differs & (kind == name)).sum())
        for name in ("header", "end", "record")
    }
    first_differing = np.flatnonzero(differs)[:3]
    report["first"] = [
        {
            "slot": int(slot),
            "want": want[slot, :6].tolist(),
            "got": got[slot, :6].tolist(),
        }
        for slot in first_differing
    ]
    return report


def sprof_dict(profile_pages, cores):
    """Per pass, each core's writer total and pack counters from the profile pages of
    the last frame (wall ticks), and the slowest core against the mean."""
    report = {}
    for name, handler in (("count", S["kHandlerCount"]), ("setup", S["kHandlerSetup"])):
        page = profile_pages[:cores, 32 * (handler - S["kHandlerCount"]) :][
            :, :32
        ].astype(np.int64)
        total = page[:, S["kSProfWriter"] + 2]
        pack = page[:, S["kSProfPack"] : S["kSProfPack"] + 8]
        report[name] = {
            "total": total.tolist(),
            "kept": pack[:, 4].tolist(),
            "entries": pack[:, 6].tolist(),
            "max_over_mean": float(total.max() / max(total.mean(), 1)),
        }
    return report


def frame(args, session=None):
    scene, _ = run.load(args.scene)
    blocks = passes.mesh_blocks(scene)
    records = raster.setup(scene)
    report = {"scene": args.scene, "blocks": len(blocks), "times": {}}
    times = report["times"]
    mismatches = same_numbers()
    if mismatches:
        raise SystemExit(f"frame.h differs from the kernels' headers: {mismatches}")
    session_context = (
        contextlib.nullcontext(session) if session else km.Session(log=card.log)
    )
    with session_context as session:
        workers = card.workers(session)
        cores = args.cores or workers
        started = time.perf_counter()
        slots = args.list_slots or 2 * len(scene["indices"]) + 65536
        held = capacity([scenes.frame_size(scene)], at_least=default_capacity())
        held = held._replace(cores=workers)  # the heavy tiles' tables from the cores
        uploaded = tensors(session, blocks, workers, slots, held)
        times["upload_s"] = time.perf_counter() - started
        thread_words = words(
            uploaded, len(blocks), cores, workers, args.steps, args.balance
        )
        thread_words[F["RF_W_REPEAT"]] = args.repeat
        started = time.perf_counter()
        running = session.load(
            program(scene, thread_words, workers, held=held), deadline=1.0, trace=16
        )
        times["load_s"] = time.perf_counter() - started
        started = time.perf_counter()
        result = running.thread_result(timeout=15.0)
        times["thread_wait_s"] = time.perf_counter() - started
        try:
            running.stop()
        except km.errors.KernelFailed as error:
            report["stop_error"] = str(error)
            trace = running.trace()
            headers = km.abi.headers()
            rows = trace.rows[:workers]
            seen, done = (
                rows[:, :, headers["kTraceSeen"]],
                rows[:, :, headers["kTraceDone"]],
            )
            report["trace"] = {
                "seen_cores": (seen != 0).sum(0).tolist(),
                "done_cores": (done != 0).sum(0).tolist(),
                "not_done": [
                    [
                        int(core)
                        for core in np.flatnonzero(
                            (seen[:, point] != 0) & (done[:, point] == 0)
                        )
                    ]
                    for point in range(rows.shape[1])
                ],
            }
            print(json.dumps(power.clocked(report), default=str), flush=True)
        report["thread"] = result_dict(result)
        report["x280_mhz"] = session.pll_mhz
        report["status"] = card.check_run(running, workers, 0, [])
        (profile_pages,) = card.read(session, uploaded["sprof"])
        report["sprof"] = sprof_dict(
            profile_pages.view(np.uint32).reshape(workers, -1), cores
        )
        # Regions are not read back (a region read reads every bank's pages of the
        # regions' tensor at once): the tensors the thread's commands wrote are.
        middle = expected(scene, records, cores, workers)
        found = {key: report["thread"][key] for key in middle["found"]}
        report["found_equal"] = found == middle["found"]
        succeeded = (
            result.ran
            and result.code == 0
            and report["found_equal"]
            and report["status"]["ok"]
        )
        if args.steps >= 3:
            report["lists"] = card_passes.lists_check(
                session, uploaded, scene, records, middle, args.lists
            )
            (directories,) = card.read(session, uploaded["dir"])
            directories = directories.view(np.uint32).reshape(workers, -1).copy()
            frame_words = directories[:, K["kDirFrame"] :].copy()
            directories[:, K["kDirFrame"] :] = 0
            report["dir_equal"] = bool(np.array_equal(directories, middle["directory"]))
            want = [frame_pages(uploaded["frame"])] + [
                frame_tensor.address for frame_tensor in uploaded["frame"]
            ]
            report["dir_frame_equal"] = bool(
                (frame_words[:, : len(want)] == want).all()
                and not frame_words[:, len(want) :].any()
            )
            if report["lists"]["pages_differ"] and len(middle["pages"]) <= 1024:
                report["slots_differ"] = slots_differ(
                    session, uploaded, records, middle
                )
            succeeded = succeeded and report["lists"]["pages_differ"] == 0
            succeeded = succeeded and report["dir_equal"] and report["dir_frame_equal"]
        if args.steps >= 4 and args.image:
            started = time.perf_counter()
            image = card.read_frame(session, uploaded)
            times["read_frame_s"] = time.perf_counter() - started
            counts, offsets, members = raster.bin_triangles(records)
            report["check"] = run.check(
                args.scene,
                None,
                records,
                offsets,
                members,
                list(range(lists.TILES)),
                counts,
                image,
            )
            succeeded = succeeded and run.check_ok(report["check"])
    report["ok"] = bool(succeeded)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="tiny")
    parser.add_argument("--cores", type=int, default=0, help="0: every core")
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="2 the middle pass, 3 SETUP, 4 FILL and the image",
    )
    parser.add_argument(
        "--list-slots", type=int, default=0, help="0: from the triangles"
    )
    parser.add_argument(
        "--lists", type=int, default=0, help="list pages checked (0: all, -1: none)"
    )
    parser.add_argument(
        "--no-image", dest="image", action="store_false", help="no image read-back"
    )
    parser.add_argument(
        "--no-balance",
        dest="balance",
        action="store_false",
        help="equal block partitions",
    )
    parser.add_argument(
        "--repeat", type=int, default=0, help="frames in a row (0: one)"
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    report = frame(args)
    print(json.dumps(power.clocked(report)), flush=True)
    if args.json:
        with open(args.json, "w") as file:
            json.dump(report, file, indent=1)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
