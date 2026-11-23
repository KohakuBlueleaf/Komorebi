"""A frame loaded lit (LIGHT, kernels/light.h): the tangent frame a record's planes give
(SHADE's scalar step, here in float32 as the card does it), the lights' table, and the
checks of the surface planes and of the lit frame against the reference."""

import argparse
import json
import sys

import numpy as np

from projects.raster import card, card_thread, lists, run, shade
from reference import lights as light_data
from reference import native, png, raster

F32 = np.float32
RSQRT_MAGIC = 0x5F3759DF  # 1 / sqrt(x) estimate from the bits, then Newton steps
SH, K = card_thread.SH, lists.K
SURFACE_NAMES = ("a", "normal", "visibility", "specular")


def rsqrt(x):
    """1 / sqrt(x) (float32, x > 0) as the card's scalar code: the magic estimate and
    three Newton steps of multiplies and adds (the cores have no divide)."""
    x = np.asarray(x, dtype=F32)
    y = (np.uint32(RSQRT_MAGIC) - (x.view(np.uint32) >> np.uint32(1))).view(F32)
    half = x * F32(0.5)
    for _ in range(3):
        y = y * (F32(1.5) - half * y * y)
    return y


def unit(v):
    """v (..., 3) over its length by rsqrt; zero where the length is zero."""
    length2 = raster.dot3(v, v)
    safe = np.where(length2 > 0, length2, F32(1))
    return np.where((length2 > 0)[..., None], v * rsqrt(safe)[..., None], F32(0))


def to_world(cam, v):
    """Camera-space directions (n, 3) to scene space: right x + up y + forward z."""
    return (cam[3:6] * v[:, 0:1] + cam[6:9] * v[:, 1:2]) + cam[9:12] * v[:, 2:3]


def frame_from_planes(cam, w_plane, u_plane, v_plane):
    """The tangent frame (n, 3, 3: tangent, bitangent, eye-facing normal) and validity
    from each record's planes of 1/z, u/z, v/z (n, 3), float32. README "Surface planes
    and LIGHT": normal -(a f, -b f, c'), dP/du ~ diag(1, -1, f)(m2 x m0) sign(det M);
    the reference's (shade_setup) frame; not valid where det M or the tangent is 0."""
    # About the centre (c' = a cx + b cy + c) a pixel's camera point is (x', -y', f) /
    # (f w): 1/z = m0 . (x', y', 1) and the point is affine in (u, v) through M.
    f, cx, cy = cam[12], cam[13], cam[14]

    def centred(plane):
        p = plane.astype(F32)
        return np.stack([p[:, 0], p[:, 1], (p[:, 0] * cx + p[:, 1] * cy) + p[:, 2]], 1)

    m0, m1, m2 = centred(w_plane), centred(u_plane), centred(v_plane)
    normal = -np.stack([m0[:, 0] * f, -(m0[:, 1] * f), m0[:, 2]], 1)
    normal = unit(to_world(cam, normal))
    det = raster.dot3(m0, raster.cross3(m1, m2))
    sign = np.where(det < 0, F32(-1), F32(1))[:, None]
    scale = np.array([1, -1, f], dtype=F32)
    along_u = to_world(cam, raster.cross3(m2, m0) * scale) * sign
    along_v = to_world(cam, raster.cross3(m0, m1) * scale) * sign
    tangent = along_u - normal * raster.dot3(normal, along_u)[:, None]
    tangent = unit(tangent)
    side = raster.cross3(normal, tangent)
    side = np.where((raster.dot3(side, along_v) < 0)[:, None], -side, side)
    valid = (det != 0) & (raster.dot3(tangent, tangent) > 0)
    return np.stack([tangent, side, normal], axis=1), valid


def angle_degrees(a, b):
    """Angles (n,) in degrees between unit vectors a and b (float64)."""
    cosine = np.clip(
        (a.astype(np.float64) * b.astype(np.float64)).sum(axis=-1), -1.0, 1.0
    )
    return np.degrees(np.arccos(cosine))


def figures(values):
    """Median, 99th percentile and largest of angles (degrees), and the count."""
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "median": round(float(np.median(values)), 5),
        "p99": round(float(np.percentile(values, 99)), 5),
        "max": round(float(np.max(values)), 5),
        "over_2deg": int((values > 2).sum()),
    }


def frame_check(name, samples, seed=1):
    """Check A: the frame from each record's planes against the reference's frame
    (shade_setup) for `samples` records of the scene's first kept pieces (whole
    triangles and first pieces of cut ones, apart), angles in degrees."""
    scene, _ = run.load(name)
    materials = shade.load(name)
    extra = np.concatenate([materials["uv"], materials["light"][:, :, None]], axis=2)
    records = raster.setup(scene, extra)
    prepared = raster.shade_setup(scene, materials)
    owner = records["tri"]
    first = np.r_[True, owner[1:] != owner[:-1]]
    cut = np.zeros(len(owner), dtype=bool)
    cut[1:] |= owner[1:] == owner[:-1]
    cut[:-1] |= owner[1:] == owner[:-1]
    frame, valid = frame_from_planes(
        scene["camera"],
        records["w_planes"],
        records["extra_planes"][:, :, 0],
        records["extra_planes"][:, :, 1],
    )
    want = prepared["frame"][owner]
    want_ok = prepared["frame_ok"][owner]
    rng = np.random.default_rng(seed)
    report = {"scene": name, "records": int(len(owner))}
    for label, chosen in (("whole", first & ~cut), ("first_of_cut", first & cut)):
        index = np.flatnonzero(chosen)
        if len(index) > samples:
            index = np.sort(rng.choice(index, samples, replace=False))
        both = valid[index] & want_ok[index]
        part = {
            "records": int(len(index)),
            "valid_differs": int((valid[index] != want_ok[index]).sum()),
            "reference_not_valid": int((~want_ok[index]).sum()),
            "normal": figures(angle_degrees(frame[index, 2], want[index, 2])),
        }
        for axis, key in ((0, "tangent"), (1, "bitangent")):
            part[key] = figures(
                angle_degrees(frame[index[both], axis], want[index[both], axis])
            )
        report[label] = part
    return report


# The card's terms past part A's (sky, ambient, the sun, always on): kernels/shade.h
# kTerm* by the reference's names (raster.TERMS).
BASE_TERMS = ("sky", "ambient", "sun")
CARD_TERMS = {
    "shadow": "kTermShadow",
    "lamps": "kTermLamps",
    "specular": "kTermSpecular",
    "background": "kTermBackground",
}
PAGE = 32  # a light page: 32 x 32 values of a map (as a texture page)


def term_bits(terms):
    """The table's kLtTerms word of a set of the reference's terms (must hold
    BASE_TERMS; the sun's shadow only with the sun)."""
    terms = set(terms)
    unknown = terms - set(raster.TERMS)
    if unknown or not set(BASE_TERMS) <= terms:
        raise ValueError(
            f"the card's terms: {BASE_TERMS} and any of {list(CARD_TERMS)}"
        )
    return sum(SH[name] for term, name in CARD_TERMS.items() if term in terms)


def map_pages(values):
    """A map (side, side) of 32-bit values in pages of 32 x 32, pages row by row, a
    page's values row by row from the top: (pages, 1024) uint32."""
    side = values.shape[0]
    blocks = values.reshape(side // PAGE, PAGE, side // PAGE, PAGE).transpose(
        0, 2, 1, 3
    )
    return np.ascontiguousarray(blocks.reshape(-1, PAGE * PAGE).view(np.uint32))


def light_pages(lights):
    """LIGHT's pages: the shadow map's 1/z, then the background map's red, green and
    blue planes, each a map_pages; and each map's first page."""
    shadow = lights["shadow_map"].astype(F32)
    octa = lights["octahedral"].astype(F32)
    if shadow.shape[0] % PAGE or octa.shape[0] % PAGE:
        raise ValueError("a light map's side is not a multiple of 32")
    parts = [map_pages(shadow)] + [map_pages(octa[:, :, c]) for c in range(3)]
    first = {"shadow": 0, "octahedral": len(parts[0])}
    return np.concatenate(parts), first


def light_table(lights, materials, terms=raster.TERMS):
    """LIGHT's table (kernels/shade.h kLt*): per channel the sky's products, ambient,
    the sun; the sun's direction, exposure, background, 1 / pi, terms, exponent bits,
    lamps; the shadow texel, bias, camera; the turn; the maps' pages; each lamp."""
    words = np.zeros(SH["kLightTableBytes"] // 4, dtype=np.uint32)

    def put(at, values):
        values = np.atleast_1d(np.asarray(values, dtype=F32))
        words[at : at + len(values)] = values.view(np.uint32)

    s = lights["sh"].astype(F32)
    c1, c2, c3, c4, c5 = (
        light_data.C1,
        light_data.C2,
        light_data.C3,
        light_data.C4,
        light_data.C5,
    )
    two_c1, two_c2 = F32(2) * c1, F32(2) * c2
    for channel in range(3):
        k = s[:, channel]
        sky = [
            c4 * k[0] - c5 * k[6],
            c3 * k[6],
            c1 * k[8],
            two_c1 * k[4],
            two_c1 * k[7],
            two_c1 * k[5],
            two_c2 * k[3],
            two_c2 * k[1],
            two_c2 * k[2],
        ]
        at = SH["kLtChannel"] * channel
        put(at + SH["kLtSky"], sky)
        put(at + SH["kLtAmbient"], lights["ambient"][channel])
        put(at + SH["kLtSun"], lights["sun_irradiance"][channel])
    put(SH["kLtSunDirection"], lights["sun_direction"])
    exposure = F32(lights["exposure"])
    put(SH["kLtExposure"], exposure)
    put(SH["kLtScale"], exposure * F32(1 / np.pi))
    put(SH["kLtBackground"], raster.BACKGROUND)
    put(SH["kLtInvPi"], F32(1 / np.pi))
    words[SH["kLtTerms"]] = term_bits(terms)
    words[SH["kLtExponentBits"]] = int(
        shade.exponents(materials).max(initial=0)
    ).bit_length()
    lamps = len(lights["lamp_intensity"]) if "lamps" in terms else 0
    if lamps > SH["kLampsMax"]:
        raise ValueError(f"{lamps} lamps: LIGHT's table holds {SH['kLampsMax']}")
    words[SH["kLtLamps"]] = lamps
    put(SH["kLtShadowTexel"], lights["shadow_texel"])
    put(SH["kLtShadowBias"], lights["shadow_bias"])
    put(SH["kLtShadowCamera"], lights["shadow_camera"])
    # the background's view direction turned by -turn about y (raster.py background)
    angle = np.radians(-float(lights["turn"]))
    put(SH["kLtTurnCos"], F32(np.cos(angle)))
    put(SH["kLtTurnSin"], F32(np.sin(angle)))
    # light_pages' layout from the maps' sides
    shadow_pages = (lights["shadow_map"].shape[0] // PAGE) ** 2
    first = {"shadow": 0, "octahedral": shadow_pages}
    pages = shadow_pages + 3 * (lights["octahedral"].shape[0] // PAGE) ** 2
    for name, key in (("Shadow", "shadow_map"), ("Octa", "octahedral")):
        side = lights[key].shape[0]
        put(SH[f"kLt{name}First"], 1 + first["shadow" if name == "Shadow" else key])
        put(SH[f"kLt{name}Across"], side // PAGE)
        put(SH[f"kLt{name}Side"], side)
    put(SH["kLtOctaLast"], lights["octahedral"].shape[0] - 1)
    words[SH["kLtOctaChannel"]] = (lights["octahedral"].shape[0] // PAGE) ** 2 << 10
    words[SH["kLtPages"]] = pages
    radius, size = F32(lights["radius"]), F32(lights["size"])
    for lamp in range(lamps):
        at = SH["kLtLamp"] + SH["kLampWords"] * lamp
        put(at, lights["lamp_position"][lamp])
        put(at + SH["kLampColour"], lights["lamp_colour"][lamp])
        put(at + SH["kLampIntensity"], lights["lamp_intensity"][lamp])
        put(at + SH["kLampRadius"], radius)
        put(at + SH["kLampInvRadius"], F32(1) / radius)
        put(at + SH["kLampSize2"], size**2)
    return words


SUN_PATH_STEP = 15.0  # --sun-path: degrees of azimuth a step
SUN_PATH_LOW, SUN_PATH_HIGH = 25.0, 80.0  # its elevation's range


def scene_box(scene):
    """The scene's box as a scene of two points (its least and greatest corner): what
    sun_camera and shadow_map's texel read of the positions (their min and max), at the
    cost of two points instead of the mesh's."""
    positions = scene["positions"]
    return {"positions": np.stack([positions.min(0), positions.max(0)])}


def sun_lights(lights, scene, azimuth, elevation):
    """lights.moved's numbers from the loaded `lights` without its shadow map (the card
    renders it): turn, sh_turned, direction, sun_camera at the map's side, texel, bias;
    elevation <= 0: irradiance 0. The shadow map stays the loaded array (its shape only
    is used); `scene` may be scene_box's."""
    own_azimuth, _ = light_data.sun_angles(lights["sun_direction"])
    change = float(azimuth) - own_azimuth
    sun = light_data.sun_direction(azimuth, elevation)
    out = dict(lights)
    out["turn"] = np.float32(float(lights["turn"]) + change)
    out["sh"] = light_data.sh_turned(lights["sh"], change)
    out["sun_direction"] = sun
    distance = float(lights["shadow_texel"] * lights["shadow_camera"][12])
    texels = float(lights["shadow_bias"]) * distance**2 / float(lights["shadow_texel"])
    positions = scene["positions"]
    camera = light_data.sun_camera(positions, sun, light_data.shadow_side(lights))
    out["shadow_camera"] = camera
    if elevation <= 0:
        out["sun_irradiance"] = np.zeros(3, dtype=np.float32)
        return out
    centre = (positions.min(0) + positions.max(0)) / 2  # shadow_map's texel
    texel = F32(np.linalg.norm(camera[0:3].astype(np.float64) - centre) / camera[12])
    out["shadow_texel"] = texel
    distance = float(texel * camera[12])
    out["shadow_bias"] = F32(texels * float(texel) / distance**2)
    return out


def sun_path(azimuth, step):
    """--sun-path's sun at `step`: the azimuth from `azimuth` on, SUN_PATH_STEP degrees
    a step; the elevation SUN_PATH_LOW to SUN_PATH_HIGH, a sine over 24 steps."""
    middle = (SUN_PATH_LOW + SUN_PATH_HIGH) / 2
    swing = (SUN_PATH_HIGH - SUN_PATH_LOW) / 2
    angle = np.radians(SUN_PATH_STEP * step)
    return float(azimuth + SUN_PATH_STEP * step), float(middle + swing * np.sin(angle))


def quadrant_cameras(camera, side):
    """The sun's cameras of a shadow map of `side` from its camera: (side / 1,024)^2
    lights.quadrant_camera, quadrant (i, j) the (j n + i)-th."""
    across, most = side // light_data.FILL_SIDE, card_thread.F["RF_SUN_ACROSS_MAX"]
    if side % light_data.FILL_SIDE or not 1 <= across <= most:
        raise ValueError(f"a shadow map of side {side}: the card's are 1,024 or 2,048")
    return [
        light_data.quadrant_camera(camera, i, j)
        for j in range(across)
        for i in range(across)
    ]


def sun_region(lights, materials, terms=raster.TERMS):
    """The frame thread's sun region (thread/frame.h RF_R_SUN): the sun's camera of each
    quadrant of the shadow map as COUNT's and SETUP's camera page, the quadrants a side,
    then LIGHT's table (light_table)."""
    words = np.zeros(card_thread.F["RF_SUN_BYTES"] // 4, dtype=np.uint32)
    side = light_data.shadow_side(lights)
    cameras = quadrant_cameras(lights["shadow_camera"], side)
    step = card_thread.F["RF_SUN_CAMERA_BYTES"] // 4
    for quadrant, camera in enumerate(cameras):
        words[step * quadrant : step * quadrant + SH["kCamWordsShade"]] = camera.view(
            np.uint32
        )
    words[card_thread.F["RF_SUN_ACROSS_AT"] // 4] = side // light_data.FILL_SIDE
    at = card_thread.F["RF_SUN_TABLE_AT"] // 4
    words[at : at + SH["kLightTableBytes"] // 4] = light_table(lights, materials, terms)
    return words


def sun_entries(scene, keeps, suns, side=light_data.SHADOW_SIDE):
    """The most list entries each layer (each mask of `keeps`) makes in one of the sun's
    views (a quadrant of a shadow map of `side`) over `suns` ((azimuth, elevation)):
    raster.setup's box, bin_triangles' tiles, zero areas counted too (an upper bound).
    A list, one number a mask."""
    box = scene_box(scene)
    most = [0] * len(keeps)
    half, tile = F32(0.5), raster.TILE
    corners = [np.ascontiguousarray(scene["indices"][:, k]) for k in range(3)]

    def lowest(values):
        return np.minimum(
            np.minimum(values[corners[0]], values[corners[1]]), values[corners[2]]
        )

    def highest(values):
        return np.maximum(
            np.maximum(values[corners[0]], values[corners[1]]), values[corners[2]]
        )

    for azimuth, elevation in suns:
        sun = light_data.sun_direction(azimuth, elevation)
        view = light_data.sun_camera(box["positions"], sun, side)
        for camera in quadrant_cameras(view, side):
            xc, yc, zc = raster.camera_space({**scene, "camera": camera})
            x, y, _ = raster.to_screen(camera, xc, yc, zc)
            x0 = np.clip(np.ceil(lowest(x) - half), 0, raster.WIDTH)
            x1 = np.clip(np.floor(highest(x) - half), -1, raster.WIDTH - 1)
            y0 = np.clip(np.ceil(lowest(y) - half), 0, raster.HEIGHT)
            y1 = np.clip(np.floor(highest(y) - half), -1, raster.HEIGHT - 1)
            tiles = np.where(
                (x0 <= x1) & (y0 <= y1),
                (x1 // tile - x0 // tile + 1) * (y1 // tile - y0 // tile + 1),
                0,
            )
            for index, keep in enumerate(keeps):
                most[index] = max(most[index], int(tiles[keep].sum()))
    return most


def shadow_of_pages(pages):
    """The shadow map (side, side) float32 from its light pages (map_pages' layout)."""
    pages = np.asarray(pages, dtype=np.uint32)
    across = int(round(np.sqrt(len(pages))))
    blocks = pages.reshape(across, across, PAGE, PAGE).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(blocks.reshape(across * PAGE, across * PAGE)).view(F32)


def sun_check(view, materials, lights, got):
    """Check B: the card's shadow map `got` against rasterize_shaded at each quadrant's
    camera of `lights`' shadow camera (lights.moved's): texels over 1e-5 relative apart
    and texels empty on one side only, by the reference's layer and by quadrant; later
    pieces of cut triangles apart."""
    side = light_data.shadow_side(lights)
    fill = light_data.FILL_SIDE
    want = np.zeros((side, side), dtype=F32)
    tri = np.full((side, side), -1, dtype=np.int32)
    masked_tri = np.full((side, side), -1, dtype=np.int32)
    later = np.zeros((side, side), dtype=bool)
    corner = {**materials, "light": lights["visibility"][view["indices"]]}
    cameras = quadrant_cameras(lights["shadow_camera"], side)
    for quadrant, camera in enumerate(cameras):
        sun_view = {
            "positions": view["positions"],
            "indices": view["indices"],
            "colors": np.zeros((len(view["positions"]), 3), dtype=np.float32),
            "camera": camera,
            "light": np.zeros(3, dtype=np.float32),
        }
        shaded = raster.rasterize_shaded(sun_view, materials)
        j, i = divmod(quadrant, side // fill)
        place = (slice(fill * j, fill * (j + 1)), slice(fill * i, fill * (i + 1)))
        want[place], tri[place] = shaded["depth"], shaded["tri"]
        masked_tri[place] = shaded["masked"]["tri"]
        owner = raster.setup(sun_view)["tri"]
        if (owner[1:] == owner[:-1]).any():
            later[place] = shade.later_pieces(sun_view, corner, shaded["tri"])
    report = {
        "side": side,
        "reference_equals_moved": bool(
            np.array_equal(want.view(np.uint32), lights["shadow_map"].view(np.uint32))
        ),
    }
    taken_masked = (masked_tri >= 0) & (tri == masked_tri)
    with np.errstate(all="ignore"):
        relative = np.abs(got.astype(np.float64) - want) / np.abs(
            want.astype(np.float64)
        )
    both = (got != 0) & (want != 0)
    differs = both & (relative > 1e-5)
    one_empty = (got == 0) != (want == 0)
    report["later_pieces"] = int(later.sum())
    report["texels_covered"] = int((want != 0).sum())
    groups = {
        "all": ~later,
        "opaque_layer": ~later & (tri >= 0) & ~taken_masked,
        "masked_layer": ~later & taken_masked,
        "empty_in_reference": ~later & (tri < 0),
    }
    for name, chosen in groups.items():
        count = max(int(chosen.sum()), 1)
        report[name] = {
            "texels": int(chosen.sum()),
            "differ_over_1e-5": int((differs & chosen).sum()),
            "share_differ": round(float((differs & chosen).sum() / count), 7),
            "one_empty": int((one_empty & chosen).sum()),
            "share_one_empty": round(float((one_empty & chosen).sum() / count), 7),
        }
    report["max_relative"] = float(relative[both & ~later].max(initial=0))
    quadrant_of = (np.arange(side) // fill)[:, None] * (side // fill) + (
        np.arange(side) // fill
    )[None, :]
    report["by_quadrant"] = [
        {
            "differ_over_1e-5": int((differs & ~later & (quadrant_of == q)).sum()),
            "one_empty": int((one_empty & ~later & (quadrant_of == q)).sum()),
        }
        for q in range(len(cameras))
    ]
    return report


def frame_tensors(session):
    """A frame's tensors as the fill's (card_thread.layer_tensors): five pages a tile,
    card.CHUNK_PAGES pages a tensor."""
    pages = lists.TILES * K["kPlanes"]
    return [
        card.tensor(session, np.zeros((min(card.CHUNK_PAGES, pages - at), 1024)))
        for at in range(0, pages, card.CHUNK_PAGES)
    ]


def surface_tensors(session, workers, lights, materials, terms=raster.TERMS):
    """The surface's two frames, LIGHT's table, pages and profile tensor, and the
    directory (a page a worker): the first frame's words at kDirFrame, the second's at
    kSurfaceSecond, the table's, profile's and pages' addresses (kernels/shade.h)."""
    frames = [frame_tensors(session), frame_tensors(session)]
    table = card.tensor(session, light_table(lights, materials, terms)[None, :])
    pages = card.tensor(session, light_pages(lights)[0])
    profile = card.tensor(session, np.zeros((workers, SH["kShadeProfBytes"] // 4)))
    page = np.zeros(K["kDirBytes"] // 4, dtype=np.uint32)
    for frame, at in zip(frames, (K["kDirFrame"], SH["kSurfaceSecond"])):
        page[at] = card.CHUNK_PAGES
        page[at + 1 : at + 1 + len(frame)] = [tensor.address for tensor in frame]
    page[SH["kSurfaceLightTable"]] = table.address
    page[SH["kSurfaceLightProfile"]] = profile.address
    page[SH["kSurfaceLightPages"]] = pages.address
    directory = card.tensor(session, np.tile(page, (workers, 1)))
    return {
        "frames": frames,
        "dir": directory,
        "table": table,
        "pages": pages,
        "profile": profile,
    }


def decode_planes(pages):
    """Frame pages (tiles x 5, 1024) as five images (5, H, W) of uint32."""
    planes = np.asarray(pages, dtype=np.uint32).reshape(lists.TILES, K["kPlanes"], 1024)
    pixels = planes[:, :, lists.TILE_ORDER].reshape(
        raster.TILES_Y, raster.TILES_X, K["kPlanes"], 32, 32
    )
    return pixels.transpose(2, 0, 3, 1, 4).reshape(
        K["kPlanes"], raster.HEIGHT, raster.WIDTH
    )


def read_surface(session, frame_tensors_of_layer, surface):
    """The card's surface after a frame: the frame's planes (1/z, material word) and
    the surface's ten planes, as images."""
    frame = decode_planes(np.concatenate(card.read(session, *frame_tensors_of_layer)))
    first, second = (
        decode_planes(np.concatenate(card.read(session, *tensors)))
        for tensors in surface["frames"]
    )
    planes = np.concatenate([first, second]).view(F32)
    return {
        "depth": frame[0].view(F32),
        "material": frame[4],
        "a": np.moveaxis(planes[0:3], 0, 2),
        "normal": np.moveaxis(planes[3:6], 0, 2),
        "visibility": planes[6],
        "specular": np.moveaxis(planes[7:10], 0, 2),
    }


def reference_surface(view, materials, lights):
    """The reference's merged surface of a camera from rasterize_lighted's parts (terms:
    the specular only, so no shadow or lamp is computed): a (emissive x emissive_scale),
    normal, visibility, specular, material word, 1/z, triangle, later pieces."""
    lit = raster.rasterize_lighted(view, materials, lights, terms=("specular",))
    parts, tri = lit["parts"], lit["tri"]
    material = np.maximum(parts["material"], 0)
    kind = materials["kind"][material]
    emissive = (parts["material"] >= 0) & (kind == raster.EMISSIVE)
    scale = np.where(emissive, F32(lights["emissive_scale"]), F32(1))
    word = (
        material.astype(np.uint32)
        | kind.astype(np.uint32) << SH["kMaterialKindShift"]
        | shade.exponents(materials)[material] << SH["kMaterialExponentShift"]
    )
    corner = {**materials, "light": lights["visibility"][view["indices"]]}
    return {
        "a": parts["albedo"] * scale[..., None],
        "normal": parts["normal"],
        "visibility": parts["visibility"],
        "specular": parts["specular"],
        "material": np.where(parts["material"] >= 0, word, np.uint32(0xFFFFFFFF)),
        "depth": lit["depth"],
        "tri": tri,
        "later": shade.later_pieces(view, corner, tri),
    }


def surface_check(view, materials, lights, got):
    """Check B: the card's surface `got` (read_surface) against the reference's for the
    pixels the reference covers, later pieces counted apart."""
    want = reference_surface(view, materials, lights)
    covered = want["tri"] >= 0
    groups = {
        "checked": covered & ~want["later"],
        "later_pieces": covered & want["later"],
    }
    to8 = lambda x: np.clip(np.floor(x * F32(255) + F32(0.5)), 0, 255)  # noqa: E731
    a_error = np.abs(to8(got["a"]) - to8(want["a"])).max(axis=2)
    angle = angle_degrees(got["normal"], want["normal"])
    vis_error = np.abs(got["visibility"] - want["visibility"])
    spec_error = np.abs(to8(got["specular"]) - to8(want["specular"])).max(axis=2)
    material_differs = got["material"] != want["material"]
    empty_differs = (want["tri"] < 0) & ((got["material"] >> 31) == 0)
    report = {"empty_marked_differ": int(empty_differs.sum())}
    for name, chosen in groups.items():
        count = max(int(chosen.sum()), 1)
        report[name] = {
            "pixels": int(chosen.sum()),
            "a_over_2_of_255": int((a_error[chosen] > 2).sum()),
            "a_max_of_255": int(a_error[chosen].max(initial=0)),
            "normal_degrees": figures(angle[chosen]),
            "normal_share_over_2deg": round(
                float((angle[chosen] > 2).sum() / count), 6
            ),
            "visibility_over_1_of_255": int((vis_error[chosen] > 1 / 255).sum()),
            "specular_over_2_of_255": int((spec_error[chosen] > 2).sum()),
            "material_differ": int(material_differs[chosen].sum()),
        }
    part = report["checked"]  # the brief's tolerances, every checked pixel
    report["all_within"] = bool(
        part["a_over_2_of_255"] == 0
        and part["visibility_over_1_of_255"] == 0
        and part["material_differ"] == 0
        and report["empty_marked_differ"] == 0
    )
    return report, want


def light_reference(view, materials, lights, terms=raster.TERMS):
    """Check C's reference: rasterize_lighted with `terms` (all: the default frame),
    the 2 x 2 filter."""
    return raster.rasterize_lighted(view, materials, lights, terms=terms)


def shadow_band(view, lights, lit):
    """The pixels of the sun's shadow edge band: the reference's sun_shadow (2 x 2
    filter) strictly between 0 and 1 at the pixel's point and normal (lit's parts),
    and their neighbours (one pixel)."""
    tri = lit["tri"]
    ys, xs = np.nonzero(tri >= 0)
    px = xs.astype(F32) + F32(0.5)
    py = ys.astype(F32) + F32(0.5)
    points = raster.world_points(view["camera"], px, py, lit["depth"][ys, xs])
    share = light_data.sun_shadow(lights, points, lit["parts"]["normal"][ys, xs])
    edge = np.zeros(tri.shape, dtype=bool)
    edge[ys, xs] = (share > 0) & (share < 1)
    band = edge.copy()
    band[1:] |= edge[:-1]
    band[:-1] |= edge[1:]
    band[:, 1:] |= band[:, :-1].copy()
    band[:, :-1] |= band[:, 1:].copy()
    return band & (tri >= 0)


def light_check(view, materials, lights, got, terms=raster.TERMS):
    """Check C: the card's 8-bit picture against light_reference's by png's rule:
    shares of pixels off by more than 2, 4, 8 in a channel, later pieces apart; with
    the shadow, the same inside the sun's shadow edge band (shadow_band) and outside."""
    lit = light_reference(view, materials, lights, terms)
    want = png.to_srgb8(lit["display"])
    error = np.abs(got[..., :3].astype(np.int16) - want.astype(np.int16)).max(axis=2)
    corner = {**materials, "light": lights["visibility"][view["indices"]]}
    apart = shade.later_pieces(view, corner, lit["tri"])
    checked = ~apart
    count = max(int(checked.sum()), 1)
    report = {"checked": int(checked.sum()), "later_pieces": int(apart.sum())}
    for limit in (2, 4, 8):
        report[f"share_over_{limit}"] = round(
            float((error[checked] > limit).sum() / count), 6
        )
        report[f"later_over_{limit}"] = int((error[apart] > limit).sum())
    report["max_error"] = int(error[checked].max(initial=0))
    if "shadow" in terms:
        band = shadow_band(view, lights, lit)
        for name, chosen in (
            ("band", checked & band),
            ("outside_band", checked & ~band),
        ):
            part = {"pixels": int(chosen.sum())}
            for limit in (2, 4, 8):
                part[f"over_{limit}"] = int((error[chosen] > limit).sum())
                part[f"share_over_{limit}"] = round(
                    float((error[chosen] > limit).sum() / max(int(chosen.sum()), 1)), 6
                )
            report[name] = part
    report["ok"] = report["share_over_4"] < 0.01
    return report, want


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="sponza_overview")
    parser.add_argument("--check-frame", type=int, default=5000, help="records")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    native.data_dir()
    report = frame_check(args.scene, args.check_frame)
    print(json.dumps(report), flush=True)
    if args.json:
        with open(args.json, "w") as file:
            json.dump(report, file, indent=1)
    sys.exit(0)


if __name__ == "__main__":
    main()
