"""Public demo scenes, fetched from their sources and converted to the scene layout:
`python -m reference.public list | fetch <scene> ...`. The table (`public_scenes.json`)
is data; files go to `$KR_DATA/public/`, never into the repository. Each view is a scene
of its own named `<scene>_<view>` (same triangles, its own camera)."""

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
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


def table():
    """{scene name: entry} of `public_scenes.json`, plus the entries of the JSON file
    named by $KR_SCENES_EXTRA (same layout) when it is set."""
    entries = json.loads(TABLE.read_text())
    extra = os.environ.get("KR_SCENES_EXTRA")
    if extra:
        entries |= json.loads(Path(extra).read_text())
    return entries


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


def fetch(name, data):
    """Downloads the files of a scene (checked) and returns their local paths."""
    folder = public_dir(data) / "downloads"
    return [
        download(f["url"], folder / f["url"].rsplit("/", 1)[1], f["sha256"])
        for f in table()[name]["files"]
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


def convert(name, data):
    """Reads a scene's mesh from its downloads and saves it as <public>/<name>.npz."""
    entry = table()[name]
    paths = fetch(name, data)
    by_name = {p.name: p for p in paths}
    pos, faces, colour, counts = [], [], [], {}
    offset = 0
    for part in entry["meshes"]:
        path = by_name[part["file"]]
        mesh = meshio.read_mesh(path, part["member"], part.get("material_colours"))
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
    parser.add_argument("command", choices=("list", "fetch"))
    parser.add_argument("scenes", nargs="*")
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()
    data = args.data or default_data()
    if args.command == "list":
        for name, entry in table().items():
            print(f"{name}: {', '.join(entry['views'])} - {entry['attribution']}")
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
