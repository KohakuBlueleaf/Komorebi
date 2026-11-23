"""A frame loaded shaded (SHADE, kernels/shade.h): what the host uploads once per scene
(the mesh with each corner's u, v and light term, the texel areas and materials, the
textures in pages, the table; a scene with masked or transparent triangles as two or
three meshes, the layers) and the check against the reference's shaded frame."""

import numpy as np

from projects.raster import lists, passes
from reference import native, public, raster

SHADE_H = lists.HERE / "kernels" / "shade.h"
SH = {
    name: int(value, 0) for name, value in lists.CONSTEXPR.findall(SHADE_H.read_text())
}
PAGE = 32  # a texture page: 32 x 32 texels of one level
# a texel's key (kernels/shade_compute.hpp record_row) is (1 + page) << 10 | texel in
# 30 bits (the masked layer tags bits 30, 31): at most 2^20 - 1 pages
KEY_PAGES = (1 << 20) - 1
F32 = np.float32


def has_materials(name):
    """True when the scene (or a view's scene) has a materials file."""
    try:
        public.load_materials(name, native.data_dir())
    except (FileNotFoundError, KeyError):
        return False
    return True


def load(name):
    """The materials of a scene or a view (reference/public.py load_materials)."""
    return public.load_materials(name, native.data_dir())[0]


LEVEL_SLOTS = (raster.ALBEDO, raster.NORMAL, raster.SPECULAR)  # the record's levels


def texel_areas(materials):
    """Per triangle the doubled area of its corners' texture coordinates in level 0
    texels of its albedo, normal and specular textures ((3, T) float32, reference
    shade_setup) and its material word, material | each one's levels << 16, 20, 24
    (levels 1 without that texture)."""
    uv = materials["uv"]
    du1, dv1 = uv[:, 1, 0] - uv[:, 0, 0], uv[:, 1, 1] - uv[:, 0, 1]
    du2, dv2 = uv[:, 2, 0] - uv[:, 0, 0], uv[:, 2, 1] - uv[:, 0, 1]
    uv_area2 = np.abs(du1 * dv2 - du2 * dv1)
    word = materials["material"].astype(np.uint32)
    areas = np.zeros((len(LEVEL_SLOTS), len(uv)), dtype=F32)
    for index, slot in enumerate(LEVEL_SLOTS):
        own = materials["textures"][materials["material"], slot]
        texture = np.maximum(own, 0)
        size = materials["mip"][texture, 0]
        texels = uv_area2 * (size[:, 1] * size[:, 2]).astype(F32)
        areas[index] = np.where(np.isfinite(texels), texels, F32(0))
        levels = np.where(own >= 0, materials["levels"][texture], 1).astype(np.uint32)
        word |= levels << (16 + 4 * index)
    return areas, word


def frame_ok(scene, materials):
    """Per triangle the reference's `frame_ok` (raster.shade_setup, the same float32
    steps): its coordinates give a tangent frame. The camera is left out: turning the
    normal toward the eye does not change the tangent."""
    p = scene["positions"][scene["indices"]]
    uv = materials["uv"]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    normal = raster.normalized(raster.cross3(e1, e2))
    du1, dv1 = uv[:, 1, 0] - uv[:, 0, 0], uv[:, 1, 1] - uv[:, 0, 1]
    du2, dv2 = uv[:, 2, 0] - uv[:, 0, 0], uv[:, 2, 1] - uv[:, 0, 1]
    det = du1 * dv2 - du2 * dv1
    sign = np.where(det < 0, F32(-1), F32(1))[:, None]
    tangent = (e1 * dv2[:, None] - e2 * dv1[:, None]) * sign
    tangent = raster.normalized(
        tangent - normal * raster.dot3(normal, tangent)[:, None]
    )
    return (det != 0) & (raster.dot3(tangent, tangent) > 0)


def mesh_blocks(scene, materials, lit=False):
    """The mesh as passes.mesh_blocks lays it out, corner colours replaced by u, v
    and light term where not flat (lit: all); shade blocks (blocks, 128): 32 albedo
    texel areas, 32 material words (lit: kShadeNoFrame where not frame_ok or
    transparent), 32 normal, 32 specular."""
    kind = materials["kind"][materials["material"]]
    values = np.concatenate(
        [materials["uv"], materials["light"][:, :, None]], axis=2
    ).astype(F32)
    shaded = np.ones(len(kind), dtype=bool) if lit else kind != raster.FLAT
    colors = np.where(shaded[:, None, None], values, scene["colors"][scene["indices"]])
    flat_scene = {
        **scene,
        "positions": scene["positions"][scene["indices"]].reshape(-1, 3),
        "colors": colors.reshape(-1, 3).astype(F32),
        "indices": np.arange(3 * len(scene["indices"])).reshape(-1, 3),
    }
    blocks = passes.mesh_blocks(flat_scene)
    areas, word = texel_areas(materials)
    if lit:
        framed = frame_ok(scene, materials) & (kind != raster.TRANSPARENT)
        word = np.where(framed, word, word | np.uint32(SH["kShadeNoFrame"]))
    count = len(blocks) * passes.LANES
    shade = np.zeros((4, count), dtype=np.uint32)
    shade[0, : len(word)] = areas[0].view(np.uint32)
    shade[1, : len(word)] = word
    shade[2:, : len(word)] = areas[1:].view(np.uint32)
    shade_blocks = shade.reshape(4, len(blocks), passes.LANES).transpose(1, 0, 2)
    return blocks, np.ascontiguousarray(shade_blocks.reshape(len(blocks), -1))


def texture_pages(materials):
    """Every level of every texture in 32 x 32 blocks of texels (a page each, 4 bytes
    a texel as stored, blocks row by row, edge blocks padded by repeating), and per
    (texture, level) its first page, width and height: ((pages, 1024), (N, 11, 3))."""
    mip, levels, texels = materials["mip"], materials["levels"], materials["texels"]
    entries = np.zeros((len(levels), mip.shape[1], 3), dtype=np.int64)
    blocks = []
    first = 0
    for texture in range(len(levels)):
        for level in range(levels[texture]):
            offset, width, height = (int(v) for v in mip[texture, level])
            words = (
                texels[offset : offset + 4 * width * height]
                .view(np.uint32)
                .reshape(height, width)
            )
            across, down = -(-width // PAGE), -(-height // PAGE)
            padded = np.pad(
                words, ((0, down * PAGE - height), (0, across * PAGE - width)), "edge"
            )
            page_words = padded.reshape(down, PAGE, across, PAGE).transpose(0, 2, 1, 3)
            blocks.append(page_words.reshape(-1, PAGE * PAGE))
            entries[texture, level] = first, width, height
            first += across * down
    return np.concatenate(blocks), entries


def specular_term(materials):
    """Per material whether it has the Blinn-Phong term: lit or transparent (the
    glass's, always) with a specular colour (raster.py light_layer)."""
    shiny = (materials["kind"] == raster.LIT) | (
        materials["kind"] == raster.TRANSPARENT
    )
    return shiny & (materials["specular"].max(axis=1) > 0)


def exponents(materials):
    """Per material its specular exponent where it has a specular term
    (specular_term), else 0: the material word's bits from kMaterialExponentShift."""
    has = specular_term(materials)
    exponent = np.where(has, materials["exponent"], 0).astype(np.int64)
    if (exponent < 0).any() or exponent.max(initial=0) > SH["kExponentMax"]:
        raise ValueError(f"a specular exponent is outside 0 to {SH['kExponentMax']}")
    return exponent.astype(np.uint32)


def table(materials, entries, pages, emissive_scale=1.0):
    """SHADE's table (kernels/shade.h): counts, the texel tables, the background, the
    materials (lit: emissive bases x emissive_scale; transparent: tau and 1 - max tau),
    the (texture, level) entries."""
    count, levels = entries.shape[:2]
    words = np.zeros(SH["kShadeTableBytes"] // 4, dtype=np.uint32)
    materials_count = len(materials["kind"])
    words[SH["kTabMaterials"]] = materials_count
    words[SH["kTabEntries"]] = count * levels
    words[SH["kTabPages"]] = pages
    for name, values in (
        ("kTabGamma", raster.TEXEL_LINEAR),
        ("kTabSigned", raster.TEXEL_SIGNED),
        ("kTabUnit", raster.TEXEL_UNIT),
    ):
        words[SH[name] : SH[name] + 256] = values.view(np.uint32)
    background = SH["kTabBackground"]
    words[background : background + 3] = raster.BACKGROUND.view(np.uint32)
    at = SH["kTabMaterial"]
    albedo = materials["textures"][:, raster.ALBEDO]
    rows = np.zeros((materials_count, SH["kMaterialWords"]), dtype=np.uint32)
    rows[:, 0] = (
        materials["kind"].astype(np.uint32)
        | np.where(albedo >= 0, albedo * levels + 1, 0).astype(np.uint32) << 8
    )
    rows[:, 1:4] = materials["base_colour"].astype(F32).view(np.uint32)
    mask = materials["textures"][:, raster.MASK]
    for material in np.flatnonzero((mask >= 0) & (albedo >= 0)):
        texture, alpha = albedo[material], mask[material]
        same = (
            materials["levels"][texture] == materials["levels"][alpha]
            and (
                materials["mip"][texture, :, 1:] == materials["mip"][alpha, :, 1:]
            ).all()
        )
        if not same:  # the mask's texel is fetched at the albedo's place and level
            raise ValueError(f"material {material}: mask and albedo sizes differ")
    rows[:, SH["kMaterialMask"]] = np.where(
        (mask >= 0) & (albedo >= 0), mask * levels + 1, 0
    ).astype(np.uint32)
    for name, slot in (
        ("kMaterialNormal", raster.NORMAL),
        ("kMaterialSpecular", raster.SPECULAR),
    ):
        own = materials["textures"][:, slot]
        rows[:, SH[name]] = np.where(own >= 0, own * levels + 1, 0).astype(np.uint32)
    emissive = materials["kind"] == raster.EMISSIVE
    base = materials["base_colour"].astype(F32)
    lit_base = np.where(emissive[:, None], base * F32(emissive_scale), base)
    at_base = SH["kMaterialLitBase"]
    rows[:, at_base : at_base + 3] = lit_base.astype(F32).view(np.uint32)
    specular = materials["specular"].astype(F32)
    has = specular_term(materials)
    at_specular = SH["kMaterialSpecularColour"]
    rows[:, at_specular : at_specular + 3] = (
        np.where(has[:, None], specular, F32(0)).astype(F32).view(np.uint32)
    )
    rows[:, SH["kMaterialExponent"]] = exponents(materials)
    tau = transmission(materials)
    at_tau = SH["kMaterialTau"]
    rows[:, at_tau : at_tau + 3] = tau.view(np.uint32)
    rows[:, SH["kMaterialWeight"]] = (
        (F32(1) - tau.max(axis=1)).astype(F32).view(np.uint32)
    )
    end = at + rows.size
    entry_words = np.zeros((count * levels, SH["kEntryWords"]), dtype=np.uint32)
    entry_words[:, 0] = entries[:, :, 0].ravel()
    entry_words[:, 1] = entries[:, :, 1].ravel() | entries[:, :, 2].ravel() << 16
    if end + entry_words.size > len(words):
        raise ValueError(
            f"SHADE's table needs {end + entry_words.size} words and holds"
            f" {len(words)} (kernels/shade.h kShadeTableBytes)"
        )
    words[at:end] = rows.ravel()
    words[end : end + entry_words.size] = entry_words.ravel()
    return words


def transmission(materials):
    """Per material its transmission tau (M, 3) float32: the materials file's where the
    kind is transparent, else 0 (a file without it: 0)."""
    tau = materials.get("transmission")
    if tau is None:
        return np.zeros((len(materials["kind"]), 3), dtype=F32)
    clear = materials["kind"] == raster.TRANSPARENT
    return np.where(clear[:, None], tau, 0).astype(F32)


def layer(scene, materials, keep):
    """The scene and the materials of the triangles `keep` (a bool per triangle), in
    their order: a layer's mesh (reference/raster.py rasterize_shaded)."""
    per_triangle = {name: materials[name][keep] for name in ("uv", "light", "material")}
    return {**scene, "indices": scene["indices"][keep]}, {**materials, **per_triangle}


def prepare(scene, materials, lights=None):
    """Everything uploaded once for a shaded frame (`lights`: corner light terms
    the baked visibility): mesh and shade blocks, pages, table; masked triangles a
    mesh of their own ("masked_blocks", "masked_shade_blocks", "masked_triangles"),
    transparent too ("glass_*"), the others ("blocks", "shade_blocks") without."""
    lit = lights is not None
    if lit:
        materials = {**materials, "light": lights["visibility"][scene["indices"]]}
    kind = materials["kind"][materials["material"]]
    prepared = {}
    others = np.ones(len(kind), dtype=bool)
    for name, value in (("masked", raster.MASKED), ("glass", raster.TRANSPARENT)):
        keep = kind == value
        if not keep.any():
            continue
        layer_scene, layer_materials = layer(scene, materials, keep)
        blocks, shade_blocks = mesh_blocks(layer_scene, layer_materials, lit)
        prepared[f"{name}_blocks"] = blocks
        prepared[f"{name}_shade_blocks"] = shade_blocks
        prepared[f"{name}_triangles"] = int(keep.sum())
        others &= ~keep
    if others.all():
        materials_kept = materials
    else:
        scene, materials_kept = layer(scene, materials, others)
    blocks, shade_blocks = mesh_blocks(scene, materials_kept, lit)
    pages, entries = texture_pages(materials)
    if len(pages) > KEY_PAGES:
        raise ValueError(
            f"the textures take {len(pages)} pages; a texel's key holds {KEY_PAGES}"
        )
    scale = lights["emissive_scale"] if lit else 1.0
    return {
        **prepared,
        "blocks": blocks,
        "shade_blocks": shade_blocks,
        "pages": pages,
        "table": table(materials, entries, len(pages), scale),
    }


def as_shaded(materials):
    """A copy of the materials with the lit kind made textured: what SHADE shades in
    this step (masked with its holes, no lighting)."""
    kind = materials["kind"].copy()
    kind[kind == raster.LIT] = raster.TEXTURED
    return {**materials, "kind": kind}


def later_pieces(view, materials, tri):
    """Pixels whose triangle `tri` (H, W) the near plane cut and which lie on a piece
    after its first kept one (that piece's own planes on the card, the first's in
    the reference: the same plane, rounded apart; README.md, SHADE)."""
    extra = np.concatenate([materials["uv"], materials["light"][:, :, None]], axis=2)
    records = raster.setup(view, extra)
    owner = records["tri"]
    later = np.flatnonzero(np.r_[False, owner[1:] == owner[:-1]])
    mask = np.zeros(tri.shape, dtype=bool)
    for piece in later:
        x0, x1, y0, y1 = records["box"][piece]
        ys, xs = np.nonzero(tri[y0 : y1 + 1, x0 : x1 + 1] == owner[piece])
        px, py = (xs + x0).astype(F32) + F32(0.5), (ys + y0).astype(F32) + F32(0.5)
        edges = (
            records["a"][piece] * px[:, None] + records["b"][piece] * py[:, None]
        ) + records["c"][piece]
        inside = (edges >= 0).all(axis=1) | (edges <= 0).all(axis=1)
        mask[ys[inside] + y0, xs[inside] + x0] = True
    return mask


def check(view, materials, got, to_srgb8):
    """The card's 8-bit image `got` (H, W, >= 3) against the reference's shaded frame
    of `view` on as_shaded(materials): pixels differing by more than 2 in a channel;
    `share` over those not on later_pieces (counted apart), also over the frame.
    Returns (report, compared reference image, true reference image)."""
    shaded = raster.rasterize_shaded(view, as_shaded(materials))
    want = to_srgb8(shaded["colour"])
    true = to_srgb8(raster.rasterize_shaded(view, materials)["colour"])
    error = np.abs(got[..., :3].astype(np.int16) - want.astype(np.int16)).max(axis=2)
    over = error > 2
    tri = shaded["tri"]
    apart = later_pieces(view, materials, tri)
    kind = np.where(tri >= 0, materials["kind"][materials["material"]][tri], -1)
    names = {-1: "background", raster.FLAT: "flat", raster.TEXTURED: "textured"}
    names.update(
        {raster.MASKED: "masked", raster.LIT: "lit", raster.EMISSIVE: "emissive"}
    )
    checked = ~apart
    # the glass in front (a kind of its own), its later pieces apart too
    front = shaded["front"]
    glass_apart = np.zeros_like(front)
    if front.any():
        glass_tri = np.where(front, shaded["glass"]["tri"], -1)
        glass_apart = later_pieces(view, materials, glass_tri)
        kind = np.where(front, raster.TRANSPARENT, kind)
        names[raster.TRANSPARENT] = "glass_in_front"
        checked = checked & ~glass_apart
    report = {
        "checked": int(checked.sum()),
        "differ": int((over & checked).sum()),
        "share": round(float((over & checked).sum() / max(checked.sum(), 1)), 6),
        "max_error": int(error[checked].max(initial=0)),
        "later_pieces": int((apart | glass_apart).sum()),
        "later_pieces_differ": int((over & (apart | glass_apart)).sum()),
        "glass_in_front": int(front.sum()),
        "frame_differ": int(over.sum()),
        "frame_share": round(float(over.mean()), 6),
        "differ_by_kind": {
            name: int((over & checked & (kind == value)).sum())
            for value, name in names.items()
        },
        "share_by_kind": {
            name: round(
                float(
                    (over & checked & (kind == value)).sum()
                    / max((checked & (kind == value)).sum(), 1)
                ),
                6,
            )
            for value, name in names.items()
        },
    }
    return report, want, true
