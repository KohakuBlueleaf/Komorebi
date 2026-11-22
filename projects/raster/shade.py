"""A frame loaded shaded (SHADE, kernels/shade.h): what the host uploads once per scene
(the mesh with each corner's u, v and light term, the texel areas and materials, the
textures in pages, the table) and the check against the reference's shaded frame."""

import numpy as np

from projects.raster import lists, passes
from reference import native, public, raster

SHADE_H = lists.HERE / "kernels" / "shade.h"
SH = {
    name: int(value, 0) for name, value in lists.CONSTEXPR.findall(SHADE_H.read_text())
}
PAGE = 32  # a texture page: 32 x 32 texels of one level
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


def texel_areas(materials):
    """Per triangle the doubled area of its corners' texture coordinates in its albedo
    texture's level 0 texels (reference/raster.py shade_setup, float32) and its
    material word, material | levels << 16 (levels 1 without an albedo texture)."""
    uv = materials["uv"]
    du1, dv1 = uv[:, 1, 0] - uv[:, 0, 0], uv[:, 1, 1] - uv[:, 0, 1]
    du2, dv2 = uv[:, 2, 0] - uv[:, 0, 0], uv[:, 2, 1] - uv[:, 0, 1]
    uv_area2 = np.abs(du1 * dv2 - du2 * dv1)
    albedo = materials["textures"][materials["material"], raster.ALBEDO]
    texture = np.maximum(albedo, 0)
    size = materials["mip"][texture, 0]
    texels = uv_area2 * (size[:, 1] * size[:, 2]).astype(F32)
    texels = np.where(np.isfinite(texels), texels, F32(0)).astype(F32)
    levels = np.where(albedo >= 0, materials["levels"][texture], 1).astype(np.uint32)
    word = materials["material"].astype(np.uint32) | levels << 16
    return texels, word


def mesh_blocks(scene, materials):
    """The mesh as passes.mesh_blocks lays it out, each corner's colour replaced by its
    u, v and light term where the material is not flat; and the shade blocks (blocks,
    64): 32 texel areas, then 32 material words (kernels/setup.h kShadeAddrAt)."""
    kind = materials["kind"][materials["material"]]
    values = np.concatenate(
        [materials["uv"], materials["light"][:, :, None]], axis=2
    ).astype(F32)
    colors = np.where(
        (kind != raster.FLAT)[:, None, None], values, scene["colors"][scene["indices"]]
    )
    flat_scene = {
        **scene,
        "positions": scene["positions"][scene["indices"]].reshape(-1, 3),
        "colors": colors.reshape(-1, 3).astype(F32),
        "indices": np.arange(3 * len(scene["indices"])).reshape(-1, 3),
    }
    blocks = passes.mesh_blocks(flat_scene)
    texels, word = texel_areas(materials)
    count = len(blocks) * passes.LANES
    shade = np.zeros((2, count), dtype=np.uint32)
    shade[0, : len(texels)] = texels.view(np.uint32)
    shade[1, : len(word)] = word
    shade_blocks = shade.reshape(2, len(blocks), passes.LANES).transpose(1, 0, 2)
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


def table(materials, entries, pages):
    """SHADE's table (kernels/shade.h): counts, the gamma table, the materials, the
    (texture, level) entries."""
    count, levels = entries.shape[:2]
    words = np.zeros(SH["kShadeTableBytes"] // 4, dtype=np.uint32)
    materials_count = len(materials["kind"])
    words[SH["kTabMaterials"]] = materials_count
    words[SH["kTabEntries"]] = count * levels
    words[SH["kTabPages"]] = pages
    gamma = SH["kTabGamma"]
    words[gamma : gamma + 256] = raster.TEXEL_LINEAR.view(np.uint32)
    at = SH["kTabMaterial"]
    albedo = materials["textures"][:, raster.ALBEDO]
    rows = np.zeros((materials_count, SH["kMaterialWords"]), dtype=np.uint32)
    rows[:, 0] = (
        materials["kind"].astype(np.uint32)
        | np.where(albedo >= 0, albedo * levels + 1, 0).astype(np.uint32) << 8
    )
    rows[:, 1:4] = materials["base_colour"].astype(F32).view(np.uint32)
    end = at + rows.size
    entry_words = np.zeros((count * levels, SH["kEntryWords"]), dtype=np.uint32)
    entry_words[:, 0] = entries[:, :, 0].ravel()
    entry_words[:, 1] = entries[:, :, 1].ravel() | entries[:, :, 2].ravel() << 16
    if end + entry_words.size > len(words):
        raise ValueError(f"SHADE's table needs {end + entry_words.size} words")
    words[at:end] = rows.ravel()
    words[end : end + entry_words.size] = entry_words.ravel()
    return words


def prepare(scene, materials):
    """Everything uploaded once for a shaded frame: mesh and shade blocks, texture
    pages, table."""
    blocks, shade_blocks = mesh_blocks(scene, materials)
    pages, entries = texture_pages(materials)
    return {
        "blocks": blocks,
        "shade_blocks": shade_blocks,
        "pages": pages,
        "table": table(materials, entries, len(pages)),
    }


def as_shaded(materials):
    """A copy of the materials with masked and lit kinds made textured: what SHADE
    shades in this step (no holes, no lighting)."""
    kind = materials["kind"].copy()
    kind[np.isin(kind, (raster.MASKED, raster.LIT))] = raster.TEXTURED
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
    report = {
        "checked": int(checked.sum()),
        "differ": int((over & checked).sum()),
        "share": round(float((over & checked).sum() / max(checked.sum(), 1)), 6),
        "max_error": int(error[checked].max(initial=0)),
        "later_pieces": int(apart.sum()),
        "later_pieces_differ": int((over & apart).sum()),
        "frame_differ": int(over.sum()),
        "frame_share": round(float(over.mean()), 6),
        "differ_by_kind": {
            name: int((over & checked & (kind == value)).sum())
            for value, name in names.items()
        },
    }
    return report, want, true
