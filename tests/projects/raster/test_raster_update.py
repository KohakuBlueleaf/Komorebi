"""The arrays the thread's UPDATE reads (projects/raster/update.py), built on the host
from a small field: the operator by row gives indirect.field_values' sums, the rows'
places give light.field_pages' pages, the regions hold their arrays at the offsets
the header names, update.h's numbers are the ones they stand for."""

import numpy as np

from projects.raster import light, update
from reference import indirect
from reference import lights as light_data

F = np.float32


def small_model(count=96, seed=11):
    """Elements on a few walls (count a multiple of 32, the last 5 pad rows) with a
    field of spacing 1."""
    rng = np.random.default_rng(seed)
    position = rng.random((count, 3)).astype(F) * F([6, 3, 5])
    normal = rng.normal(size=(count, 3))
    normal = (normal / np.linalg.norm(normal, axis=1, keepdims=True)).astype(F)
    real = np.ones(count, bool)
    real[-5:] = False
    model = {
        "position": position,
        "normal": normal,
        "real": real,
        "albedo": rng.uniform(0.2, 0.9, (count, 3)).astype(F),
        "emission": rng.uniform(0, 0.1, (count, 3)).astype(F),
        "transfer": rng.uniform(0, 0.2, (count, indirect.SKY_TERMS)).astype(F),
    }
    return model | indirect.build_field(position, normal, 1.0)


def test_update_h_numbers_are_the_ones_they_stand_for():
    assert update.same_numbers() == {}


def test_the_operator_by_row_gives_the_field_values():
    model = small_model()
    e = np.random.default_rng(2).random((len(model["position"]), 3)).astype(F)
    starts, pairs = update.operator_rows(model)
    rows = len(model["field_keys"]) * 6
    assert len(starts) == rows + 1 and starts[-1] == len(pairs)
    columns, weights = pairs[:, 0], pairs[:, 1].view(F)
    row_of = np.repeat(np.arange(rows), np.diff(starts))
    # the thread's sum: each row's pairs in order, float32 products, a double sum
    got = np.stack(
        [np.bincount(row_of, weights * e[columns, c], rows) for c in range(3)], -1
    ).astype(F)
    want = indirect.field_values(model, e).reshape(-1, 3)
    np.testing.assert_array_equal(got, want)


def test_the_rows_places_give_the_field_pages():
    model = small_model()
    rows = len(model["field_keys"]) * 6
    texels = np.random.default_rng(4).integers(0, 1 << 31, rows, dtype=np.uint32)
    texels = np.where(model["field_has"].ravel(), texels | np.uint32(1 << 31), 0)
    places = update.row_places(model)
    np.testing.assert_array_equal((places >> 31) != 0, model["field_has"].ravel())
    _, count = light.field_layout(model)
    pages = np.zeros(count * 1024, np.uint32)
    pages[places & np.uint32(0x7FFFFFFF)] = texels
    want = light.field_pages(model, texels.reshape(-1, 6).astype(np.uint32))
    np.testing.assert_array_equal(pages.reshape(count, 1024), want)
    np.testing.assert_array_equal(update.field_texels(want, model), texels)


def test_the_regions_hold_their_arrays_where_the_header_says():
    model = small_model()
    lights = {
        "emissive_scale": 2.0,
        "lamp_position": np.zeros((1, 3), F),
        "lamp_colour": np.ones((1, 3), F),
        "lamp_intensity": np.ones(1, F),
        "size": 1.0,
        "radius": 50.0,
    }
    region, offsets = update.elements_region(model, lights)
    arrays = update.element_arrays(model, lights)
    n = len(model["position"])
    for word, name, width in (
        ("RF_U_E_POSITION", "position", 3),
        ("RF_U_E_NORMAL", "normal", 3),
        ("RF_U_E_BASE", "base", 3),
        ("RF_U_E_ALBEDO", "albedo", 3),
        ("RF_U_E_TRANSFER", "transfer", indirect.SKY_TERMS),
    ):
        at = offsets[word]
        assert at % update.ALIGN == 0
        got = region[at : at + 4 * width * n].view(F).reshape(n, width)
        np.testing.assert_array_equal(got, arrays[name])
    assert not arrays["base"][~model["real"]].any()
    assert not arrays["albedo"][~model["real"]].any()
    lamps = indirect.lamp_irradiance(lights, model["position"], model["normal"])
    albedo_pi = model["albedo"] * F(1 / np.pi)
    real = model["real"]
    np.testing.assert_array_equal(
        arrays["base"][real],
        (model["emission"] * F(2.0) + albedo_pi * lamps)[real],
    )
    work, work_bytes = update.work_layout(model)
    counts = update.counts(model)
    assert work["RF_U_W_VALUES"] + 12 * counts["rows"] <= work_bytes
    assert all(at % update.ALIGN == 0 for at in work.values())


def test_the_sun_words_and_the_tiles():
    sun = np.array([0.3, 0.8, 0.5], F)
    lights = {
        "shadow_texel": F(2.25),
        "shadow_bias": F(1e-12),
        "sun_direction": sun,
        "sun_irradiance": F([3, 2, 1]),
        "shadow_camera": np.arange(16, dtype=F),
        "turn": F(0),
        "octahedral": np.full((64, 64, 3), 0.5, F),
    }
    words = update.sun_words(lights, 2)
    base = update.U["RF_U_BOUNCES"]
    assert words[0] == 2
    assert words[update.U["RF_U_TEXEL"] - base :][:1].view(F)[0] == F(2.25)
    at = update.U["RF_U_CAMERA"] - base
    np.testing.assert_array_equal(words[at : at + 16].view(F), lights["shadow_camera"])
    at = update.U["RF_U_SKY"] - base
    np.testing.assert_array_equal(
        words[at : at + 75].view(F), indirect.sky_coefficients(lights).ravel()
    )
    assert len(words) == update.U["RF_U_WORDS"] - base
    # y's pages: element (c, i) of tile i // 32 at update.c's tile_at
    y = np.random.default_rng(1).random((64, 3)).astype(F)
    pages = update.y_pages(y).view(F).reshape(2, 1024)
    for i, c in ((0, 0), (17, 1), (40, 2), (63, 2)):
        column = i % 32
        at = (column >> 4) * 256 + c * 16 + (column & 15)
        assert pages[i // 32, at] == y[i, c]
    assert light_data.bounce_count(update.U["RF_U_BOUNCES_MAX"]) == 3
