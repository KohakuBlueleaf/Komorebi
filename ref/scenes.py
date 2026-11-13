"""Deterministic scenes: spheres of mixed size over a ground, one camera, one light; a
dict of numpy arrays (saved as .npz), equal on every host (+, -, *, / and sqrt only in
the geometry, rng.py's hash for every random number). The arrays and the camera layout
are in ref/README.md "Scenes"."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from ref import rng

WIDTH = HEIGHT = 1024
# Names of the scenes used by the benchmarks and the triangle count aimed at.
PRESETS = {"tiny": 300, "s10k": 10_000, "s100k": 100_000, "s1m": 1_000_000}
TINY_SPHERES = 5
TAN_HALF_FOV = 0.55
KEYS = ("positions", "colors", "indices", "camera", "light", "eps", "meta")


def f64(x):
    """Float64 copy of an array or scalar."""
    return np.asarray(x, dtype=np.float64)


def draw(index, stream, seed):
    """Float64 uniform numbers in [0, 1) from the hash, one per index."""
    return f64(rng.uniform(index, stream, seed))


def normalize(v):
    """Unit vector of a float64 3-vector (sqrt and division only)."""
    return v / np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def cross(a, b):
    """Cross product of two float64 3-vectors."""
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def cube_sphere(n):
    """Unit sphere from a cube with n x n quads per face: (6(n+1)^2, 3), (12n^2, 3)."""
    s = np.linspace(-1.0, 1.0, n + 1)
    a, b = np.meshgrid(s, s, indexing="ij")
    one = np.ones_like(a)
    faces = [(one, a, b), (-one, a, b), (a, one, b), (a, -one, b), (a, b, one)]
    faces.append((a, b, -one))
    verts, tris = [], []
    for f, (x, y, z) in enumerate(faces):
        p = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        verts.append(p / np.sqrt((p * p).sum(axis=1, keepdims=True)))
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        v00 = (i * (n + 1) + j).ravel() + f * (n + 1) ** 2
        v10, v01, v11 = v00 + (n + 1), v00 + 1, v00 + n + 2
        tris.append(np.stack([v00, v10, v01], axis=1))
        tris.append(np.stack([v10, v11, v01], axis=1))
    return np.concatenate(verts), np.concatenate(tris)


def ground_grid(cells, half):
    """A flat cells x cells grid on y = 0: vertices, triangles, tint in [0, 1]."""
    s = np.linspace(-half, half, cells + 1)
    x, z = np.meshgrid(s, s, indexing="ij")
    verts = np.stack([x, np.zeros_like(x), z], axis=-1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(cells), np.arange(cells), indexing="ij")
    v00 = (i * (cells + 1) + j).ravel()
    v10, v01, v11 = v00 + (cells + 1), v00 + 1, v00 + cells + 2
    tris = np.concatenate(
        [np.stack([v00, v10, v01], axis=1), np.stack([v10, v11, v01], axis=1)]
    )
    ramp_x = np.abs(np.arange(len(verts)) // (cells + 1) % 8 - 4) / 4
    ramp_z = np.abs(np.arange(len(verts)) % (cells + 1) % 8 - 4) / 4
    return verts, tris, 0.5 * (ramp_x + ramp_z)


def sphere_params(count, half, seed):
    """Centres, radii and base colours of `count` spheres; the first three are big."""
    k = np.arange(count)
    u = [draw(k, stream, seed) for stream in range(7)]
    radius = half * (0.008 + 0.1 * (u[0] * u[0] * u[0]) ** 2)
    centre = np.stack(
        [
            half * 0.9 * (2 * u[1] - 1),
            radius * (1 + 2 * u[2] * u[2]),
            half * 0.9 * (2 * u[3] - 1),
        ],
        axis=1,
    )
    base = np.stack([0.25 + 0.7 * u[4], 0.25 + 0.7 * u[5], 0.25 + 0.7 * u[6]], axis=1)
    near = [(-0.25, -0.78, 0.12), (0.2, -0.85, 0.09), (0.02, -0.55, 0.07)]
    for i, (x, z, r) in enumerate(near[: min(3, count)]):
        radius[i] = half * r
        centre[i] = [half * x, half * r, half * z]
    return centre, radius, base


def build_arrays(n_triangles, spheres, seed):
    """Positions, colours and indices (float64 / int64) of the scene."""
    half = 30.0 * np.sqrt(n_triangles / 1e4)
    cells = max(2, int(np.floor(np.sqrt(0.3 * n_triangles / 2) + 0.5)))
    centre, radius, base = sphere_params(spheres, half, seed)
    share = radius * radius / (radius * radius).sum()
    segs = np.maximum(2, np.floor(np.sqrt(0.7 * n_triangles * share / 12) + 0.5))
    gv, gt, tint = ground_grid(cells, 1.5 * half)
    p = tint[:, None]
    gc = np.hstack([0.30 + 0.15 * p, 0.38 + 0.12 * p, 0.25 + 0.05 * p])
    pos, col, idx, offset = [gv], [gc], [gt], len(gv)
    for s in range(spheres):
        unit, tris = cube_sphere(int(segs[s]))
        pos.append(centre[s] + radius[s] * unit)
        shade = 0.6 + 0.4 * (0.5 + 0.5 * unit[:, 1:2])
        col.append(base[s] * shade)
        idx.append(tris + offset)
        offset += len(unit)
    return half, np.concatenate(pos), np.concatenate(col), np.concatenate(idx)


def make_camera(half):
    """Camera arrays: looks from behind the scene's near edge toward its centre."""
    eye = np.array([0.0, 0.25 * half, -1.3 * half])
    fwd = normalize(np.array([0.0, -0.1 * half, 0.3 * half]) - eye)
    right = normalize(cross(np.array([0.0, 1.0, 0.0]), fwd))
    up = cross(fwd, right)
    focal = (WIDTH / 2) / TAN_HALF_FOV
    camera = np.concatenate([eye, right, up, fwd, [focal, WIDTH / 2, HEIGHT / 2, 0.05]])
    return camera.astype(np.float32)


def make_scene(n_triangles, spheres=None, seed=1):
    """The scene dict for about n_triangles triangles (see the module docstring)."""
    if spheres is None:
        spheres = max(4, int(np.floor(60 * np.sqrt(n_triangles / 1e4) + 0.5)))
    half, pos, col, idx = build_arrays(n_triangles, spheres, seed)
    cells = max(2, int(np.floor(np.sqrt(0.3 * n_triangles / 2) + 0.5)))
    light = np.array([-0.6 * half, 1.2 * half, -0.8 * half], dtype=np.float32)
    meta = np.array([len(idx), len(pos), spheres, cells, seed], dtype=np.int32)
    return {
        "positions": pos.astype(np.float32),
        "colors": col.astype(np.float32),
        "indices": idx.astype(np.int32),
        "camera": make_camera(half),
        "light": light,
        "eps": np.array([1e-4 * half], dtype=np.float32),
        "meta": meta,
    }


def make_preset(name):
    """One of the PRESETS by name."""
    spheres = TINY_SPHERES if name == "tiny" else None
    return make_scene(PRESETS[name], spheres=spheres)


def save_scene(path, scene):
    """Saves a scene as an uncompressed .npz."""
    np.savez(path, **scene)


def load_scene(path):
    """Loads a scene saved by `save_scene`."""
    with np.load(path) as data:
        return {key: data[key] for key in KEYS}


def export_npy(scene, folder):
    """Writes each array of a scene as <folder>/<key>.npy, the input of the C++ code."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for key in KEYS:
        np.save(folder / f"{key}.npy", scene[key])


def checksum(scene):
    """SHA-256 hex digest over the bytes of every array, in KEYS order."""
    digest = hashlib.sha256()
    for key in KEYS:
        digest.update(np.ascontiguousarray(scene[key]).tobytes())
    return digest.hexdigest()


def main():
    """Writes the preset scenes as <out>/<name>.npz and prints their sizes."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("names", nargs="*", default=list(PRESETS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        scene = make_preset(name)
        save_scene(args.out / f"{name}.npz", scene)
        t, v = scene["meta"][:2]
        print(f"{name}: {t} triangles, {v} vertices, sha256 {checksum(scene)}")


if __name__ == "__main__":
    main()
