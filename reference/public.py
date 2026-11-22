"""Public demo scenes, fetched from their sources and converted to the scene layout:
`python -m reference.public list | fetch | materials | lighting <scene> ...`. The
table (`public_scenes.json`) is data; files go to `$KR_DATA/public/`, never into the
repository. Each view is a scene of its own named `<scene>_<view>` (same triangles, its
own camera). `materials` writes `<scene>_materials.npz` beside the scene file for the
scenes whose entry has `materials` (texture coordinates, materials, textures; it
decodes the textures with Pillow, the `scenes` extra of pyproject.toml)."""

import argparse
import concurrent.futures
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from reference import meshio, scenes

ROOT = Path(__file__).resolve().parents[1]
TABLE = Path(__file__).resolve().parent / "public_scenes.json"
GROUND_CELLS = 16
GROUND_TINT = ((0.55, 0.55, 0.52), (0.42, 0.42, 0.40))
EPS_SCALE = 2e-5
NEAR_SCALE = 1e-4
CHUNK_BYTES = 1 << 20
# Baked shade: colour = diffuse * (SHADE_BASE + SHADE_GAIN * |normal . SHADE_DIR|).
SHADE_BASE, SHADE_GAIN = 0.55, 0.45
SHADE_DIR = np.array([0.3, 1.0, 0.5]) / np.sqrt(0.09 + 1.0 + 0.25)
# Materials: the kinds of shader (the number is the kind's code), a material's texture
# slots, the largest texture side kept, levels a mip chain can have (1024 to 1).
KINDS = ("flat", "textured", "masked", "lit", "emissive")
SLOTS = ("albedo", "mask", "normal", "specular")
TEXTURE_SIDE = 1024
MIP_LEVELS = 11
EMISSIVE_MIN = 0.5  # a material whose Ke reaches this in a channel glows
HEIGHT_GAIN = 2.0  # slope scale when a bump map holds heights, not normals
EXPONENT_RANGE = (1, 1024)
NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def table():
    """{scene name: entry} of `public_scenes.json`, plus the entries of the JSON file
    named by $KR_SCENES_EXTRA (same layout) when it is set; not `environments`."""
    entries = json.loads(TABLE.read_text())
    extra = os.environ.get("KR_SCENES_EXTRA")
    if extra:
        entries |= json.loads(Path(extra).read_text())
    entries.pop("environments", None)
    return entries


def environments():
    """{map name: entry} of the table's `environments` (the lighting's HDR maps)."""
    return json.loads(TABLE.read_text())["environments"]


def views():
    """{view name `<scene>_<view>`: (scene name, view name)} of every view."""
    return {
        f"{name}_{view}": (name, view)
        for name, entry in table().items()
        for view in entry["views"]
    }


def is_view(name):
    """True when `name` is a view of a public scene."""
    return name in views()


def default_data():
    """The data folder: $KR_DATA, else <repository>/build/data."""
    return Path(os.environ.get("KR_DATA", ROOT / "build" / "data"))


def public_dir(data):
    """Where downloads and converted meshes go: <data>/public."""
    path = Path(data) / "public"
    (path / "downloads").mkdir(parents=True, exist_ok=True)
    return path


def sha256(path):
    """SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def download(url, path, expected):
    """Downloads url to path unless a file with the expected SHA-256 is there."""
    if path.exists() and sha256(path) == expected:
        return path
    part = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "scene-fetch"})
    with urllib.request.urlopen(request, timeout=60) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, CHUNK_BYTES)
    got = sha256(part)
    if got != expected:
        raise ValueError(f"{url}: SHA-256 {got}, expected {expected}")
    part.replace(path)
    return path


def fetch(name, data, key="files"):
    """Downloads the files of a scene (checked) and returns their local paths; key
    "texture_files" gives its texture archives, kept in downloads/textures/. A file is
    saved under its URL's last part, or its entry's `file` when it has one."""
    folder = public_dir(data) / "downloads"
    if key == "texture_files":
        folder = folder / "textures"
        folder.mkdir(exist_ok=True)
    return [
        download(
            f["url"], folder / f.get("file", f["url"].rsplit("/", 1)[1]), f["sha256"]
        )
        for f in table()[name].get(key, [])
    ]


def fetch_environment(name, data):
    """Downloads an environment map of the table (checked) into downloads/ and
    returns its local path in a list."""
    folder = public_dir(data) / "downloads"
    return [
        download(f["url"], folder / f["url"].rsplit("/", 1)[1], f["sha256"])
        for f in environments()[name]["files"]
    ]


def ground(lo, hi, extent):
    """A checkered grid of GROUND_CELLS^2 squares at the height of lo, under lo..hi."""
    centre = (lo + hi) / 2
    half = extent * max(hi[0] - lo[0], hi[2] - lo[2])
    s = np.linspace(-half, half, GROUND_CELLS + 1)
    i, j = np.meshgrid(np.arange(GROUND_CELLS), np.arange(GROUND_CELLS), indexing="ij")
    i, j = i.ravel(), j.ravel()
    y = np.full(len(i), lo[1])
    corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
    pos = np.stack(
        [np.stack([centre[0] + s[a], y, centre[2] + s[b]], 1) for a, b in corners], 1
    ).reshape(-1, 3)
    quad = 4 * np.arange(len(i))[:, None]
    faces = np.concatenate([quad + [0, 2, 1], quad + [0, 3, 2]])
    tint = np.array(GROUND_TINT)[(i + j) % 2]
    return pos, faces, np.concatenate([tint, tint])


def shared_vertices(positions, faces, colour):
    """Vertices split by (vertex, colour), so each triangle keeps its flat colour."""
    keys = np.concatenate([faces.reshape(-1, 1), np.repeat(colour, 3, axis=0)], axis=1)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    return (
        positions[unique[:, 0].astype(np.int64)],
        unique[:, 1:],
        inverse.reshape(-1, 3),
    )


def vertex_shade(positions, faces):
    """Per-vertex shade: area-weighted mean over its triangles of a fixed-light term.

    The rasterizer has no lighting; this keeps shapes readable in its flat colours.
    """
    p = positions[faces]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = np.sqrt((n * n).sum(axis=1))
    shade = SHADE_BASE + SHADE_GAIN * np.abs(n @ SHADE_DIR) / np.maximum(area, 1e-300)
    total = np.bincount(faces.ravel(), np.repeat(area * shade, 3), len(positions))
    weight = np.bincount(faces.ravel(), np.repeat(area, 3), len(positions))
    return np.where(weight > 0, total / np.maximum(weight, 1e-300), 1.0)


def read_parts(name, data, with_uv=False):
    """Positions, faces, colours and counts of a scene's meshes (ground added), and
    the meshes as read (with texture coordinates and materials when with_uv)."""
    entry = table()[name]
    paths = fetch(name, data)
    by_name = {p.name: p for p in paths}
    pos, faces, colour, counts, meshes = [], [], [], {}, []
    offset = 0
    for part in entry["meshes"]:
        path = by_name[part["file"]]
        mesh = meshio.read_mesh(
            path, part["member"], part.get("material_colours"), with_uv
        )
        meshes.append(mesh)
        p = mesh.positions @ np.array(entry.get("axes", np.eye(3))).T
        c = mesh.colour
        if "colour" in part:
            c = np.broadcast_to(np.array(part["colour"], dtype=np.float64), c.shape)
        pos.append(p)
        faces.append(mesh.faces + offset)
        colour.append(c)
        offset += len(p)
        for k, v in mesh.counts.items():
            counts[k] = counts.get(k, 0) + v
    pos, faces = np.concatenate(pos), np.concatenate(faces)
    colour = np.concatenate(colour)
    if entry.get("ground"):
        lo, hi = pos[faces].reshape(-1, 3).min(0), pos[faces].reshape(-1, 3).max(0)
        gp, gf, gc = ground(lo, hi, entry["ground"])
        pos, faces = np.concatenate([pos, gp]), np.concatenate([faces, gf + len(pos)])
        colour = np.concatenate([colour, gc])
        counts["ground_triangles"] = len(gf)
    return pos, faces, colour, counts, meshes


def convert(name, data):
    """Reads a scene's mesh from its downloads and saves it as <public>/<name>.npz."""
    pos, faces, colour, counts, _ = read_parts(name, data)
    vp, vc, idx = shared_vertices(pos, faces, colour)
    vc = vc * vertex_shade(vp, idx)[:, None]
    counts["triangles_total"], counts["vertices"] = len(idx), len(vp)
    out = public_dir(data) / f"{name}.npz"
    np.savez(
        out,
        positions=vp.astype(np.float32),
        colors=np.clip(vc, 0, 1).astype(np.float32),
        indices=idx.astype(np.int32),
        counts=json.dumps(counts),
    )
    return out


def load_mesh(name, data):
    """The converted mesh of a scene; missing, it is an error (nothing downloads)."""
    path = public_dir(data) / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; fetch it with: python -m reference.public fetch {name}"
        )
    with np.load(path) as f:
        return {k: f[k] for k in ("positions", "colors", "indices")}, json.loads(
            str(f["counts"])
        )


def decode_image(data):
    """The pixels of an image file's bytes as uint8 (H, W, C), C = 1 (grey), 2 (grey,
    alpha), 3 (RGB) or 4 (RGBA). Decoded by Pillow, needed here only."""
    try:
        image_module = importlib.import_module("PIL.Image")
    except ImportError as error:
        raise RuntimeError(
            "the materials conversion decodes textures with Pillow, which is not "
            'installed; it is the "scenes" extra: pip install -e ".[scenes]"'
        ) from error
    with image_module.open(io.BytesIO(data)) as image:
        if image.mode in ("I;16", "I;16B", "I;16L", "I"):
            grey = np.asarray(image, dtype=np.float64)
            top = 65535.0 if grey.max() > 255 else 255.0
            return np.rint(grey * (255.0 / top)).astype(np.uint8)[:, :, None]
        mode = {"1": "L", "L": "L", "LA": "LA", "RGB": "RGB"}.get(image.mode, "RGBA")
        if image.mode == "P" and "transparency" not in image.info:
            mode = "RGB"
        pixels = np.asarray(image.convert(mode), dtype=np.uint8)
    return pixels.reshape(pixels.shape[0], pixels.shape[1], -1)


def halve(image):
    """A (H, W, 4) uint8 image box-filtered to half size (a side of 1 stays 1)."""
    height, width = image.shape[:2]
    rows, columns = (2 if height > 1 else 1), (2 if width > 1 else 1)
    h, w = height // rows, width // columns
    block = image[: h * rows, : w * columns].astype(np.uint32)
    total = block.reshape(h, rows, w, columns, 4).sum(axis=(1, 3))
    n = rows * columns
    return ((total + n // 2) // n).astype(np.uint8)


def heights_to_normals(height):
    """A tangent-space normal map (RGBA8) from a height map (H, W) in [0, 1]."""
    du = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) / 2
    # Rows run down the image, v runs up.
    dv = -(np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) / 2
    n = np.stack([-HEIGHT_GAIN * du, -HEIGHT_GAIN * dv, np.ones_like(height)], -1)
    n /= np.sqrt((n * n).sum(axis=-1, keepdims=True))
    rgb = np.rint((n * 0.5 + 0.5) * 255).astype(np.uint8)
    return np.concatenate([rgb, np.full(height.shape + (1,), 255, np.uint8)], -1)


def texture_levels(pixels, role):
    """The mip chain (list of (h, w, 4) uint8, largest first) of decoded pixels in a
    role: "colour" (as is, grey to RGB), "mask" (the alpha if it has one that is not
    all opaque, else the first channel, in all four channels) or "normal" (as is, or
    from heights when the image is grey); larger than TEXTURE_SIDE is halved first."""
    channels = pixels.shape[2]
    grey = channels <= 2 or (
        (pixels[:, :, 0] == pixels[:, :, 1]).all()
        and (pixels[:, :, 1] == pixels[:, :, 2]).all()
    )
    if role == "mask":
        has_alpha = channels in (2, 4) and pixels[:, :, -1].min() < 255
        value = pixels[:, :, -1] if has_alpha else pixels[:, :, 0]
        image = np.repeat(value[:, :, None], 4, axis=2)
    elif channels <= 2:
        alpha = pixels[:, :, 1:2] if channels == 2 else None
        if alpha is None:
            alpha = np.full(pixels.shape[:2] + (1,), 255, np.uint8)
        image = np.concatenate([np.repeat(pixels[:, :, :1], 3, axis=2), alpha], 2)
    elif channels == 3:
        alpha = np.full(pixels.shape[:2] + (1,), 255, np.uint8)
        image = np.concatenate([pixels, alpha], axis=2)
    else:
        image = pixels
    while max(image.shape[:2]) > TEXTURE_SIDE:
        image = halve(image)
    if role == "normal" and grey:
        image = heights_to_normals(image[:, :, 0].astype(np.float64) / 255.0)
    levels = [np.ascontiguousarray(image)]
    while max(levels[-1].shape[:2]) > 1:
        levels.append(halve(levels[-1]))
    return levels


def decode_levels(job):
    """The mip chain of one texture: job = (archive path, member, role)."""
    archive, member, role = job
    return texture_levels(decode_image(meshio.read_member(archive, member)), role)


class TextureFiles:
    """Finds the image an MTL statement names: in a texture archive of the table
    when the path's first folder is one of them, else beside the OBJ in its archive;
    names are matched without regard to case."""

    def __init__(self, mesh_archive, obj_member, archives):
        self.obj_folder = (
            obj_member.rsplit("/", 1)[0] + "/" if "/" in obj_member else ""
        )
        self.mesh_archive = mesh_archive
        self.archives = {k.lower(): v for k, v in archives.items()}
        self.members = {}

    def names(self, archive):
        if archive not in self.members:
            with zipfile.ZipFile(archive) as z:
                self.members[archive] = {n.lower(): n for n in z.namelist()}
        return self.members[archive]

    def find(self, text):
        """(archive, member) of a map statement's file, or None when it is missing."""
        if not text:
            return None
        path = text.split()[-1] if text.startswith("-") else text
        parts = [
            p for p in path.replace("\\", "/").split("/") if p not in ("", ".", "..")
        ]
        if not parts:
            return None
        if parts[0].lower() in self.archives and len(parts) > 1:
            archive, member = self.archives[parts[0].lower()], "/".join(parts[1:])
        else:
            archive, member = self.mesh_archive, self.obj_folder + "/".join(parts)
        found = self.names(archive).get(member.lower())
        return (archive, found) if found else None


def numbers(text, default):
    """Up to three numbers at the start of a statement's text (the last repeated to
    three; text after a number is ignored), else default (3,)."""
    values = []
    for word in (text or "").split()[:3]:
        match = NUMBER.match(word)
        if not match:
            break
        values.append(float(match.group()))
    if not values:
        return np.broadcast_to(np.asarray(default, dtype=np.float64), 3).copy()
    return np.array((values + values[-1:] * 2)[:3], dtype=np.float64)


def material_record(statements, files, texture_number):
    """(kind, base colour, texture numbers by slot, exponent, specular colour) of one
    material's MTL statements; texture_number(file, role) numbers the files."""
    found = {
        key: files.find(statements.get(key))
        for key in ("map_Kd", "map_d", "map_bump", "bump", "map_Ks", "map_Ke")
    }
    albedo, emit, mask = found["map_Kd"], found["map_Ke"], found["map_d"]
    normal, specular = found["map_bump"] or found["bump"], found["map_Ks"]
    glow = numbers(statements.get("Ke"), (0.0, 0.0, 0.0))
    slots = [-1, -1, -1, -1]
    base = numbers(statements.get("Kd"), meshio.DEFAULT_COLOUR)
    exponent = int(
        np.clip(np.rint(numbers(statements.get("Ns"), 1.0)[0]), *EXPONENT_RANGE)
    )
    specular_colour = numbers(statements.get("Ks"), (0.0, 0.0, 0.0))
    if emit or glow.max() >= EMISSIVE_MIN:
        kind = "emissive"
        if emit:
            slots[0], base = texture_number(emit, "colour"), np.ones(3)
        else:
            base = np.clip(glow, 0.0, 1.0)
    elif albedo is None:
        kind = "flat"
    else:
        slots[0] = texture_number(albedo, "colour")
        if mask:
            kind = "masked"
            if mask != albedo:
                slots[1] = texture_number(mask, "mask")
        elif normal or specular:
            kind = "lit"
            if normal:
                slots[2] = texture_number(normal, "normal")
            if specular:
                slots[3] = texture_number(specular, "colour")
        else:
            kind = "textured"
    return KINDS.index(kind), base, slots, exponent, specular_colour


def convert_materials(name, data, workers=4):
    """Writes <public>/<name>_materials.npz: per triangle of <name>.npz (same order)
    its corners' texture coordinates, its material and its corners' light term; the
    material table; the textures as mip chains in one byte array."""
    entry = table()[name]
    if "materials" not in entry:
        raise ValueError(f"{name}: the table gives it no materials")
    mesh, _ = load_mesh(name, data)
    downloads = {
        p.name: p for p in fetch(name, data) + fetch(name, data, "texture_files")
    }
    archives = {
        stem: downloads[file]
        for stem, file in entry["materials"].get("archives", {}).items()
    }
    pos, faces, colour, _, meshes = read_parts(name, data, with_uv=True)
    vp, _, idx = shared_vertices(pos, faces, colour)
    if not np.array_equal(idx, mesh["indices"]):
        raise ValueError(
            f"{name}: the scene file is not this conversion's; fetch again"
        )
    light = vertex_shade(vp, idx)[idx]
    # Material 0 is flat: the ground, triangles without coordinates or material.
    names = ["(flat)"]
    rows = [(0, np.array(meshio.DEFAULT_COLOUR), [-1] * 4, 1, np.zeros(3))]
    jobs, numbers_of = [], {}

    def texture_number(file, role):
        key = (str(file[0]), file[1], role)
        if key not in numbers_of:
            numbers_of[key] = len(jobs)
            jobs.append(key)
        return numbers_of[key]

    count = len(idx)
    uv = np.zeros((count, 3, 2), dtype=np.float32)
    material = np.zeros(count, dtype=np.int32)
    first = 0
    for part, read in zip(entry["meshes"], meshes):
        triangles = len(read.faces)
        if read.uv is None:
            first += triangles
            continue
        files = TextureFiles(downloads[part["file"]], part["member"], archives)
        number = np.zeros(len(read.material_names) + 1, dtype=np.int32)
        for k, material_name in enumerate(read.material_names):
            record = material_record(
                read.statements.get(material_name, {}), files, texture_number
            )
            number[k] = len(names)
            names.append(material_name)
            rows.append(record)
        has_uv = ~np.isnan(read.uv).any(axis=(1, 2))
        own = np.where(has_uv & (read.material >= 0), number[read.material], 0)
        span = slice(first, first + triangles)
        material[span] = own
        uv[span] = np.where(has_uv[:, None, None], np.nan_to_num(read.uv), 0)
        first += triangles
    kinds, bases, slots, exponents, speculars = zip(*rows, strict=True)
    kinds = np.array(kinds, dtype=np.int32)
    sources = [(Path(a), m, r) for a, m, r in jobs]
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(workers) as pool:
            chains = list(pool.map(decode_levels, sources))
    else:
        chains = [decode_levels(job) for job in sources]
    mip = np.zeros((len(chains), MIP_LEVELS, 3), dtype=np.int64)
    level_count = np.zeros(len(chains), dtype=np.int32)
    offset, blocks = 0, []
    for t, chain in enumerate(chains):
        level_count[t] = len(chain)
        for level, image in enumerate(chain):
            mip[t, level] = offset, image.shape[1], image.shape[0]
            blocks.append(image.reshape(-1))
            offset += image.size
    used = np.bincount(kinds[material], minlength=len(KINDS))
    counts = {
        "triangles": count,
        "materials": len(names),
        "textures": len(chains),
        "texel_bytes": offset,
        "triangles_by_kind": {k: int(n) for k, n in zip(KINDS, used)},
    }
    out = public_dir(data) / f"{name}_materials.npz"
    np.savez(
        out,
        uv=uv,
        material=material,
        light=light.astype(np.float32),
        kind=kinds,
        base_colour=np.array(bases, dtype=np.float32),
        textures=np.array(slots, dtype=np.int32),
        exponent=np.array(exponents, dtype=np.int32),
        specular=np.array(speculars, dtype=np.float32),
        texels=np.concatenate(blocks) if blocks else np.zeros(0, np.uint8),
        mip=mip,
        levels=level_count,
        names=json.dumps(names),
        texture_names=json.dumps([f"{Path(a).name}:{m} ({r})" for a, m, r in jobs]),
        counts=json.dumps(counts),
    )
    return out


MATERIAL_KEYS = (
    "uv",
    "material",
    "light",
    "kind",
    "base_colour",
    "textures",
    "exponent",
    "specular",
    "texels",
    "mip",
    "levels",
)


def load_materials(name, data):
    """The materials of a converted scene (or of a view's scene): the arrays of
    MATERIAL_KEYS and the counts; missing, an error naming the command."""
    if is_view(name):
        name = views()[name][0]
    path = public_dir(data) / f"{name}_materials.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; write it with: "
            f"python -m reference.public materials {name}"
        )
    with np.load(path) as f:
        return {k: f[k] for k in MATERIAL_KEYS}, json.loads(str(f["counts"]))


def export_materials(materials, folder):
    """Writes each array of the materials as <folder>/<key>.npy, the C++ input."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for key in MATERIAL_KEYS:
        np.save(folder / f"{key}.npy", materials[key])
    return folder


def camera(eye, target, fov_deg, near):
    """Camera array (16,) float32 looking from eye at target, square field of view."""
    eye, target = np.asarray(eye, np.float64), np.asarray(target, np.float64)
    fwd = scenes.normalize(target - eye)
    right = scenes.normalize(scenes.cross(np.array([0.0, 1.0, 0.0]), fwd))
    up = scenes.cross(fwd, right)
    focal = (scenes.WIDTH / 2) / np.tan(np.radians(fov_deg) / 2)
    tail = [focal, scenes.WIDTH / 2, scenes.HEIGHT / 2, near]
    return np.concatenate([eye, right, up, fwd, tail]).astype(np.float32)


def make_view(view, data):
    """The scene dict of view `<scene>_<view>` (the layout of reference/scenes.py).

    meta: triangles, vertices, then 0, 0, 0 (no generator parameters).
    """
    name, v = views()[view]
    entry = table()[name]
    mesh, _ = load_mesh(name, data)
    p = mesh["positions"].astype(np.float64)
    diag = float(np.sqrt(((p.max(0) - p.min(0)) ** 2).sum()))
    spec = entry["views"][v]
    near = spec.get("near", NEAR_SCALE * diag)
    meta = [len(mesh["indices"]), len(p), 0, 0, 0]
    return {
        **mesh,
        "camera": camera(spec["eye"], spec["target"], spec["fov"], near),
        "light": np.array(entry["light"], dtype=np.float32),
        "eps": np.array([EPS_SCALE * diag], dtype=np.float32),
        "meta": np.array(meta, dtype=np.int32),
    }


def triangles(view):
    """The triangle count recorded in the table for a view's scene (0 if unknown)."""
    name, _ = views()[view]
    return int(table()[name].get("triangles", 0))


def main():
    """Lists, fetches and converts the public scenes."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("command", choices=("list", "fetch", "materials", "lighting"))
    parser.add_argument("scenes", nargs="*")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4, help="processes")
    parser.add_argument("--rebake", action="store_true", help="trace visibility again")
    args = parser.parse_args()
    data = args.data or default_data()
    if args.command == "lighting":
        lights = importlib.import_module("reference.lights")
        for name in args.scenes or [n for n, e in table().items() if "lighting" in e]:
            out, counts = lights.prepare(name, data, args.workers, args.rebake)
            print(f"{name}: {out} {counts}")
        return
    if args.command == "list":
        for name, entry in table().items():
            print(f"{name}: {', '.join(entry['views'])} - {entry['attribution']}")
        return
    if args.command == "materials":
        for name in args.scenes or [n for n, e in table().items() if "materials" in e]:
            out = convert_materials(name, data, args.workers)
            _, counts = load_materials(name, data)
            print(f"{name}: {out} {counts}")
        return
    for name in args.scenes or list(table()):
        out = convert(name, data)
        _, counts = load_mesh(name, data)
        print(f"{name}: {out} {counts}")
        for view in table()[name]["views"]:
            scene = make_view(f"{name}_{view}", data)
            folder = Path(data) / f"{name}_{view}"
            scenes.save_scene(Path(data) / f"{name}_{view}.npz", scene)
            scenes.export_npy(scene, folder)
            print(f"  {name}_{view}: sha256 {scenes.checksum(scene)}")


if __name__ == "__main__":
    main()
