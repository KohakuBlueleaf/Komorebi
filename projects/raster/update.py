"""The indirect field's update on the frame thread (`thread/update.c`, UPDATE; README
"The field's update in the thread"): its regions, words and x, y tensors, the sun's
words, the steps with the host's apply between them, the times rows, and the
reference's numbers to check it by (`expected`)."""

import re
import time

import numpy as np

from komorebi import formats
from projects.raster import card_thread, light
from projects.transport import plan
from reference import indirect
from reference import lights as light_data

F32 = np.float32
UPDATE_H = card_thread.THREAD / "update.h"
U = {
    name: int(value.rstrip("u"), 0)
    for name, value in re.findall(
        r"^#define (RF_\w+)\s+(\d+u?)\b", UPDATE_H.read_text(), re.M
    )
}
U["RF_U_WORDS"] = U["RF_U_SKY"] + U["RF_SKY_TERMS"] * 3
HANDLER = card_thread.UPDATE_HANDLER
REGIONS = ("update", "elements", "shadow", "update_work", "fieldpages")
X_PAGE, Y_PAGE = 2048, 4096  # a bf16 tile, a float32 tile
TILE = plan.TILE
ALIGN = 4096  # each array of a region on its own pages (a fetch's range alone)
OWN_AT = 16  # the update region's words before it are the apply's
UT_NAMES = {
    "fetch": "RF_UT_FETCH",
    "sources": "RF_UT_SOURCES",
    "rows": "RF_UT_ROWS",
    "pack": "RF_UT_PACK",
    "store": "RF_UT_STORE",
    "table": "RF_UT_TABLE",
    "zero": "RF_UT_ZERO",
    "total": "RF_UT_TOTAL",
}
PASS_PARTS = ("store", "apply", "fetch", "next")


def same_numbers():
    """update.h's numbers against the ones they stand for, as pairs that differ."""
    pairs = {
        "RF_R_UPDATE": (U["RF_R_UPDATE"], card_thread.F["RF_R_UPDATE"]),
        "RF_UPDATE_REGIONS": (  # the update region is frame.h's last, the others after
            U["RF_UPDATE_REGIONS"],
            card_thread.F["RF_REGIONS"] + len(REGIONS) - 1,
        ),
        "RF_UPDATE_BYTES": (U["RF_UPDATE_BYTES"], card_thread.F["RF_UPDATE_BYTES"]),
        "RF_U_APPLY_GROUP": (U["RF_U_APPLY_GROUP"], card_thread.F["RF_U_GROUP"]),
        "OWN_AT": (OWN_AT, card_thread.F["RF_U_OTHER"]),
        "RF_NO_APPLY": (U["RF_NO_APPLY"], card_thread.F["RF_NO_APPLY"]),
        "RF_LT_FIELD_UNIT": (U["RF_LT_FIELD_UNIT"], card_thread.SH["kLtFieldUnit"]),
        "RF_LT_FIELD_SCALE": (U["RF_LT_FIELD_SCALE"], card_thread.SH["kLtFieldScale"]),
        "RF_SKY_TERMS": (U["RF_SKY_TERMS"], indirect.SKY_TERMS),
        "RF_PAGE_TEXELS": (U["RF_PAGE_TEXELS"], light.PAGE * light.PAGE),
        "RF_U_BOUNCES_MAX": (U["RF_U_BOUNCES_MAX"], light_data.MOST_BOUNCES),
        "pack bits": ((11, 11, 9), indirect.PACK_BITS),
        "pack shifts": ((20, 9, 0), indirect.PACK_SHIFTS),
        "codes after frame.h's": (
            min(U["RF_NO_APPLY"], card_thread.F["RF_BAD_GRID"] + 1),
            card_thread.F["RF_BAD_GRID"] + 1,
        ),
    }
    return {name: pair for name, pair in pairs.items() if pair[0] != pair[1]}


def aligned(at):
    return -(-at // ALIGN) * ALIGN


def operator_rows(model):
    """The field's operator by row (indirect.field_values' sparse sum): each row's
    first pair (rows + 1 starts) and the (column, weight) pairs in the reference's order
    (uint32 [weights, 2], the weight's float32 bits)."""
    rows = model["field_rows"]
    count = len(model["field_keys"]) * 6
    if len(rows) and np.any(np.diff(rows) < 0):
        raise ValueError("the field's rows are not in order")
    starts = np.searchsorted(rows, np.arange(count + 1)).astype(np.uint32)
    pairs = np.empty((len(rows), 2), dtype=np.uint32)
    pairs[:, 0] = model["field_columns"].astype(np.uint32)
    pairs[:, 1] = model["field_weights"].astype(F32).view(np.uint32)
    return starts, pairs


def row_places(model):
    """Each row's (voxel x 6 + d) place in the field's pages (light.field_pages'
    layout): page x 1,024 + texel, bit 31 where the row has data (uint32 [rows])."""
    blocks, _ = light.field_layout(model)
    dims = tuple(int(d) for d in model["field_dims"])
    voxel = np.stack(np.unravel_index(model["field_keys"], dims), -1)
    block, inside = voxel // light.FIELD_BLOCK, voxel % light.FIELD_BLOCK
    page = (block[:, 1] * blocks[2] + block[:, 2]) * blocks[0] + block[:, 0]
    local = (inside[:, 1] * light.FIELD_BLOCK[2] + inside[:, 2]) * light.FIELD_BLOCK[
        0
    ] + inside[:, 0]
    texel = np.arange(6) * light.FIELD_DIRECTION + local[:, None]
    places = (page[:, None] * U["RF_PAGE_TEXELS"] + texel).astype(np.uint64)
    if places.size and places.max() >= 1 << 31:
        raise ValueError("the field's pages pass 2^31 texels")
    has = model["field_has"].astype(np.uint64) << 31
    return (places | has).astype(np.uint32).ravel()


def element_arrays(model, lights):
    """The elements' arrays the thread reads (float32 [N, 3] or [N, 25]): position,
    normal, base = emission x emissive_scale + albedo / pi x the lamps (they do not
    move), albedo / pi, transfer; base and albedo / pi 0 on pad rows."""
    p, n = model["position"].astype(F32), model["normal"].astype(F32)
    real = model["real"][:, None]
    albedo_pi = model["albedo"] * F32(1 / np.pi)
    base = model["emission"] * F32(lights["emissive_scale"]) + albedo_pi * (
        indirect.lamp_irradiance(lights, p, n)
    )
    return {
        "position": p,
        "normal": n,
        "base": np.where(real, base, F32(0)).astype(F32),
        "albedo": np.where(real, albedo_pi, F32(0)).astype(F32),
        "transfer": model["transfer"].astype(F32),
    }


def elements_region(model, lights):
    """The elements region (uint8) and its arrays' byte offsets {word name: offset}."""
    arrays = element_arrays(model, lights)
    starts, pairs = operator_rows(model)
    parts = [
        ("RF_U_E_POSITION", arrays["position"]),
        ("RF_U_E_NORMAL", arrays["normal"]),
        ("RF_U_E_BASE", arrays["base"]),
        ("RF_U_E_ALBEDO", arrays["albedo"]),
        ("RF_U_E_TRANSFER", arrays["transfer"]),
        ("RF_U_E_STARTS", starts),
        ("RF_U_E_PAIRS", pairs),
        ("RF_U_E_PLACES", row_places(model)),
    ]
    offsets, at = {}, 0
    for name, array in parts:
        offsets[name] = at
        at = aligned(at + array.nbytes)
    region = np.zeros(at, dtype=np.uint8)
    for name, array in parts:
        raw = np.ascontiguousarray(array).view(np.uint8).ravel()
        region[offsets[name] : offsets[name] + raw.size] = raw
    return region, offsets


def counts(model):
    """N, x tiles, rows, weights, field pages of a model."""
    n = len(model["position"])
    if n % TILE:
        raise ValueError(f"{n} elements: not a multiple of {TILE}")
    _, pages = light.field_layout(model)
    return {
        "n": n,
        "tiles": n // TILE,
        "rows": len(model["field_keys"]) * 6,
        "weights": len(model["field_rows"]),
        "pages": pages,
    }


def work_layout(model):
    """The work region's arrays' byte offsets and its bytes."""
    c = counts(model)
    sizes = [
        ("RF_U_W_SKY", 12 * c["n"]),
        ("RF_U_W_L0", 12 * c["n"]),
        ("RF_U_W_TOTAL", 12 * c["n"]),
        ("RF_U_W_X", X_PAGE * c["tiles"]),
        ("RF_U_W_Y", Y_PAGE * c["tiles"]),
        ("RF_U_W_VALUES", 12 * c["rows"]),
    ]
    offsets, at = {}, 0
    for name, size in sizes:
        offsets[name] = at
        at = aligned(at + size)
    return offsets, at


def tensors(session, model):
    """The passes' x (bf16 tiles, plan.x_tiles' layout) and y (float32 tiles)."""
    c = counts(model)
    return {
        "x": session.tensor(
            plan.x_tiles(np.zeros((c["n"], 3), F32)), "bfloat16", "tile"
        ),
        "y": session.empty((TILE, TILE * c["tiles"]), "float32", "tile"),
    }


def sun_words(lights, bounces):
    """The update's inputs of `lights` (sun_lights'), words RF_U_BOUNCES on, as
    uint32: the passes, the shadow texel and bias, the sun's direction and irradiance,
    its camera, the sky's 25 x 3 coefficients."""
    words = np.zeros(U["RF_U_WORDS"] - U["RF_U_BOUNCES"], dtype=np.uint32)

    def put(word, values):
        values = np.atleast_1d(np.asarray(values, dtype=F32)).ravel()
        at = U[word] - U["RF_U_BOUNCES"]
        words[at : at + len(values)] = values.view(np.uint32)

    words[0] = light_data.bounce_count(bounces)
    put("RF_U_TEXEL", lights["shadow_texel"])
    put("RF_U_BIAS", lights["shadow_bias"])
    put("RF_U_SUN", lights["sun_direction"])
    put("RF_U_IRRADIANCE", lights["sun_irradiance"])
    put("RF_U_CAMERA", lights["shadow_camera"])
    put("RF_U_SKY", indirect.sky_coefficients(lights))
    return words


def header(model, lights, light_pages, x, y, cores, offsets, bounces):
    """The update region (uint32 [1,024]) at the load: words 0 to 15 (the apply's) 0;
    the counts, tensors, offsets; the inputs of `lights` (sun_words)."""
    c = counts(model)
    words = np.zeros(U["RF_UPDATE_BYTES"] // 4, dtype=np.uint32)
    side = light_data.shadow_side(lights)
    values = {
        "RF_U_N": c["n"],
        "RF_U_TILES": c["tiles"],
        "RF_U_ROWS": c["rows"],
        "RF_U_WEIGHTS": c["weights"],
        "RF_U_PAGES": c["pages"],
        "RF_U_LIGHT": light_pages.address,
        "RF_U_FIELD_FIRST": light.light_pages_first(lights),
        "RF_U_SHADOW_PAGES": (side // light.PAGE) ** 2,
        "RF_U_SIDE": side,
        "RF_U_X": x.address,
        "RF_U_X_PAGE": X_PAGE,
        "RF_U_Y": y.address,
        "RF_U_Y_PAGE": Y_PAGE,
        "RF_U_CORES": cores,
        **offsets,
    }
    for name, value in values.items():
        words[U[name]] = value
    words[U["RF_U_BOUNCES"] : U["RF_U_WORDS"]] = sun_words(lights, bounces)
    return words


def regions(model, lights, light_pages, update_tensors, cores, bounces):
    """The thread's update regions in update.h's order: the header (with the inputs of
    `lights`), the elements, and the sizes of the shadow map's pages, the work, the
    field's pages."""
    elements, element_offsets = elements_region(model, lights)
    work_offsets, work_bytes = work_layout(model)
    side = light_data.shadow_side(lights)
    words = header(
        model,
        lights,
        light_pages,
        update_tensors["x"],
        update_tensors["y"],
        cores,
        {**element_offsets, **work_offsets},
        bounces,
    )
    return {
        "update": words,
        "elements": elements,
        "shadow": side * side * 4,
        "update_work": work_bytes,
        "fieldpages": counts(model)["pages"] * U["RF_PAGE_TEXELS"] * 4,
    }


def write_sun(running, lights, bounces):
    """The update's inputs of `lights` into the live update region (words
    RF_U_BOUNCES on)."""
    running.thread.write("update", sun_words(lights, bounces), 4 * U["RF_U_BOUNCES"])


def x_of(session, x, tiles):
    """x's tensor read back as [N, 3] float32 (rows 0 to 2 of its bf16 tiles)."""
    (raw,) = session.read(x)
    bits = np.ascontiguousarray(raw).view(np.uint16).reshape(tiles, TILE * TILE)
    values = (bits.astype(np.uint32) << 16).view(F32)
    return formats.untilize(values, TILE, TILE * tiles)[:3].T.copy()


def y_pages(y):
    """y [N, 3] as the float32 tile tensor's pages [tiles, 4,096] uint8."""
    raw, _ = formats.to_bytes(plan.x_tiles(np.asarray(y, F32)), "float32", "tile")
    return np.asarray(raw).reshape(-1, Y_PAGE)


def host_apply(session, update_tensors, model):
    """The host in the apply's place: x read from its tensor, y = M x in float32
    (indirect.apply on the model's blocks and weights), y written to its tensor.
    Seconds: read, apply, write."""
    began = time.perf_counter()
    c = counts(model)
    x = x_of(session, update_tensors["x"], c["tiles"])
    read = time.perf_counter()
    y = indirect.apply(model["blocks"], model["weights"], x)
    applied = time.perf_counter()
    session.write(update_tensors["y"], y_pages(y))
    ended = time.perf_counter()
    return x, {
        "read_s": read - began,
        "apply_s": applied - read,
        "write_s": ended - applied,
    }


def submit(running, step, cluster=card_thread.CLUSTER):
    """One UPDATE command of `step`, waited for: its word (0, a code) or its kind."""
    event = running.wait(
        running.submit("update", cluster=cluster, args=[step], result=True),
        timeout=30.0,
    )
    return event.result if event.kind == "done" else event.kind


def run(running, session, update_tensors, model, bounces, stepped, on_x=None):
    """One update: the whole of it in one command, or (`stepped`, passes above 0)
    sources, each pass's apply on the host, finish. `on_x(p, x)`: each pass's x as
    the host read it. Returns (word, host seconds by part)."""
    began = time.perf_counter()
    host = {"commands_s": 0.0, "read_s": 0.0, "apply_s": 0.0, "write_s": 0.0}

    def command(step):
        start = time.perf_counter()
        word = submit(running, step)
        host["commands_s"] += time.perf_counter() - start
        return word

    if not stepped or bounces == 0:
        word = command(U["RF_STEP_ALL"])
    else:
        word = command(U["RF_STEP_SOURCES"])
        for p in range(bounces):
            if word != 0:
                break
            x, parts = host_apply(session, update_tensors, model)
            for key, value in parts.items():
                host[key] += value
            if on_x is not None:
                on_x(p, x)
            word = command(U["RF_STEP_PASS"])
        if word == 0:
            word = command(U["RF_STEP_FINISH"])
    host["s"] = time.perf_counter() - began
    return word, {key: round(value, 4) for key, value in host.items()}


def times(raw, ticks_ms=1e3 / 50e6):
    """The times rows of the update region's bytes (after the stop), oldest first:
    each update's parts in ms, its number, code, step, passes, scale."""
    words = np.frombuffer(np.asarray(raw, np.uint8).tobytes(), dtype=np.uint32)
    at, width = U["RF_U_TIMES"], U["RF_UT_WORDS"]
    rows = words[at : at + U["RF_U_TIMES_ROWS"] * width].reshape(-1, width)
    out = []
    for row in rows:
        if not row.any():
            continue
        entry = {
            name: round(row[U[key]] * ticks_ms, 4) for name, key in UT_NAMES.items()
        }
        entry["passes_ms"] = [
            {
                part: round(
                    row[U["RF_UT_PASS"] + U["RF_UT_PASS_WORDS"] * p + k] * ticks_ms, 4
                )
                for k, part in enumerate(PASS_PARTS)
            }
            for p in range(int(row[U["RF_UT_PASSES"]]))
        ]
        entry["number"] = int(row[U["RF_UT_NUMBER"]])
        entry["code"] = int(row[U["RF_UT_CODE"]])
        entry["step"] = int(row[U["RF_UT_STEP"]])
        entry["passes"] = int(row[U["RF_UT_PASSES"]])
        entry["scale"] = float(
            row[U["RF_UT_SCALE"] : U["RF_UT_SCALE"] + 1].view(F32)[0]
        )
        out.append(entry)
    return sorted(out, key=lambda entry: entry["number"])


def work_arrays(raw, model):
    """sky and L0 [N, 3] float32 from the work region's bytes (after the stop)."""
    offsets, _ = work_layout(model)
    raw = np.asarray(raw, np.uint8)
    n = counts(model)["n"]

    def array(word):
        at = offsets[word]
        return raw[at : at + 12 * n].view(F32).reshape(n, 3).copy()

    return {"sky": array("RF_U_W_SKY"), "l0": array("RF_U_W_L0")}


def field_texels(pages, model):
    """The field's texels (rows,) uint32 from its pages (light.field_pages' layout)."""
    places = row_places(model) & np.uint32(0x7FFFFFFF)
    return np.asarray(pages, np.uint32).ravel()[places]


def expected(lights, model, bounces):
    """The reference's numbers for `lights` (its shadow map: the card's, when checking
    the card): sky, L0, the passes (E_p), the field's values (rows, 3), its texels
    (rows,), scale and pages (indirect.element_light, field_source, field_values,
    pack_field; light.field_pages)."""
    lit = indirect.element_light(lights, model, bounces)
    values = indirect.field_values(model, indirect.field_source(lit))
    texels, scale = indirect.pack_field(values, model["field_has"])
    return {
        "sky": lit["sky"],
        "l0": lit["l0"],
        "passes": lit["passes"],
        "values": values.reshape(-1, 3),
        "texels": texels.ravel(),
        "scale": scale,
        "pages": light.field_pages(model, texels),
    }
