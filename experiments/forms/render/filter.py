"""The per-pixel arithmetic of a texture filter, a depth test and a vertex stage, on
float32 tiles against numpy: bilinear as three lerps, floor (km.floor, and by the
magic constant), frac, clamp, the nearer depth (km.where, and by arithmetic), x' from
4 words of the command, and the camera as a product [x y z 1] @ M."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nn"))

import numpy as np
from forms import bf16, error, finish, log, read, tiles, timed

import komorebi as km

MAGIC = 12582912.0  # 1.5 * 2^23: (x + MAGIC) - MAGIC rounds x to an integer, |x| < 2^22


@km.tile_function
def lerp(a, b, t):
    return a + (b - a) * t


@km.tile_function
def floored(u):
    return km.floor(u)


@km.tile_function
def frac(u):
    return u - km.floor(u)


@km.tile_function
def floor_magic(u):
    r = (u + MAGIC) - MAGIC
    return r - km.minimum(km.maximum((r - u) * 18446744073709551616.0, 0.0), 1.0)


@km.tile_function
def clamp(u):
    return km.minimum(km.maximum(u, 0.0), 1.0)


@km.tile_function
def nearer(z_new, z_old):
    return km.where(z_new < z_old, z_new, z_old)


@km.tile_function
def float_step(z_new, z_old):
    mask = km.minimum(km.maximum((z_old - z_new) * 18446744073709551616.0, 0.0), 1.0)
    return z_old + mask * (z_new - z_old)


@km.tile_function
def clip_x(x, y, z, m0: float, m1: float, m2: float, m3: float):
    return x * m0 + y * m1 + z * m2 + m3


class Pixels(km.Program):
    """Core r works on the tiles r n to r n + n - 1 of each tensor."""

    @km.handler
    def lerp3(self, a: km.Tensor, b: km.Tensor, t: km.Tensor, o: km.Tensor, n: int):
        first = self.rank * n
        out = km.Stream(o.pages(first, n), write=True)
        for x, y, w in zip(
            km.Stream(a.pages(first, n)),
            km.Stream(b.pages(first, n)),
            km.Stream(t.pages(first, n)),
        ):
            out.push(lerp(x, y, w))

    @km.handler
    def unary(self, a: km.Tensor, f: km.Tensor, r: km.Tensor, c: km.Tensor, n: int):
        """floor, frac and the magic constant's rounding of a; clamp into c."""
        first = self.rank * n
        floors = km.Stream(f.pages(first, n), write=True)
        for x in km.Stream(a.pages(first, n)):
            floors.push(floored(x))
        fracs = km.Stream(r.pages(first, n), write=True)
        for x in km.Stream(a.pages(first, n)):
            fracs.push(frac(x))
        clamps = km.Stream(c.pages(first, n), write=True)
        for x in km.Stream(a.pages(first, n)):
            clamps.push(clamp(x))

    @km.handler
    def floor_round(self, a: km.Tensor, o: km.Tensor, n: int):
        """The way round km.floor: the magic constant's rounding, less 1 where above."""
        first = self.rank * n
        out = km.Stream(o.pages(first, n), write=True)
        for x in km.Stream(a.pages(first, n)):
            out.push(floor_magic(x))

    @km.handler
    def floor_only(self, a: km.Tensor, o: km.Tensor, n: int):
        first = self.rank * n
        out = km.Stream(o.pages(first, n), write=True)
        for x in km.Stream(a.pages(first, n)):
            out.push(floored(x))

    @km.handler
    def depth(self, a: km.Tensor, b: km.Tensor, w: km.Tensor, n: int):
        """The nearer of a and b by km.where."""
        first = self.rank * n
        selects = km.Stream(w.pages(first, n), write=True)
        for x, y in zip(km.Stream(a.pages(first, n)), km.Stream(b.pages(first, n))):
            selects.push(nearer(x, y))

    @km.handler
    def depth_round(self, a: km.Tensor, b: km.Tensor, s: km.Tensor, n: int):
        """The way round km.where: a 0/1 mask by arithmetic."""
        first = self.rank * n
        steps = km.Stream(s.pages(first, n), write=True)
        for x, y in zip(km.Stream(a.pages(first, n)), km.Stream(b.pages(first, n))):
            steps.push(float_step(x, y))

    @km.handler
    def vertex(
        self,
        x: km.Tensor,
        y: km.Tensor,
        z: km.Tensor,
        o: km.Tensor,
        n: int,
        m0: float,
        m1: float,
        m2: float,
        m3: float,
    ):
        first = self.rank * n
        out = km.Stream(o.pages(first, n), write=True)
        for a, b, c in zip(
            km.Stream(x.pages(first, n)),
            km.Stream(y.pages(first, n)),
            km.Stream(z.pages(first, n)),
        ):
            out.push(clip_x(a, b, c, m0, m1, m2, m3))


class Camera(km.Program):
    """Points as rows [x y z 1 0 ...] of tiles, by M (4 x 4 in a tile) held in L1."""

    m: km.L1[km.bf16, 1, 1, 32, 32]

    @km.tile_function(fidelity="HiFi4")
    def project(a, b):
        return a @ b

    @km.handler
    def load(self, matrix: km.Tensor):
        self.m.fill(km.Stream(matrix.tiles[0:1, 0:1]))

    @km.handler
    def transform(self, points: km.Tensor, clip: km.Tensor):
        r = self.rank
        acc = km.tile.zeros(km.fp32, 1)
        for pb in km.Stream(points.tiles[r : r + 1, 0:1], block=1):
            acc += self.project(pb, self.m)
        clip.tiles[r : r + 1, 0:1].write(acc)


def bilinear(t00, t01, t10, t11, fx, fy):
    top = t00 + (t01 - t00) * fx
    bottom = t10 + (t11 - t10) * fx
    return top + (bottom - top) * fy


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--tiles", type=int, default=2, help="tiles a core")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cores, n = args.cores, args.tiles
    shape = (cores * n * 32, 32)

    def f32(values):
        return np.asarray(values, np.float32).astype(np.float64)

    texels = [f32(rng.uniform(0, 1, shape)) for _ in range(4)]
    fx, fy = f32(rng.uniform(0, 1, shape)), f32(rng.uniform(0, 1, shape))
    u = f32(rng.uniform(-40, 40, shape))
    u[:4, :8] = [[-2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, -1e-7]] * 4
    z_old, z_new = f32(rng.uniform(0, 1, shape)), f32(rng.uniform(0, 1, shape))
    z_new[0, :4] = z_old[0, :4]  # ties keep z_old
    xyz = [f32(rng.uniform(-2, 2, shape)) for _ in range(3)]
    m = f32(rng.uniform(-1, 1, (4, 4)))
    points = np.zeros((cores * 32, 32))
    points[:, :3] = bf16(rng.uniform(-2, 2, (cores * 32, 3)))
    points[:, 3] = 1.0
    matrix = np.zeros((32, 32))
    matrix[:4, :4] = bf16(m)
    report = {"cores": cores, "tiles": n}
    got = {}
    with km.Session(log=log) as session:
        t = [tiles(session, v, "float32") for v in texels]
        tx, ty = tiles(session, fx, "float32"), tiles(session, fy, "float32")
        top, bottom, out = (
            tiles(session, np.zeros(shape), "float32") for _ in range(3)
        )
        tu = tiles(session, u, "float32")
        fl, fr, cl, ro = (tiles(session, np.zeros(shape), "float32") for _ in range(4))
        ta, tb = tiles(session, z_new, "float32"), tiles(session, z_old, "float32")
        sel, step = (tiles(session, np.zeros(shape), "float32") for _ in range(2))
        px, py, pz = (tiles(session, v, "float32") for v in xyz)
        clip_out = tiles(session, np.zeros(shape), "float32")
        program = Pixels(
            a=t[0], b=t[1], t=tx, o=out, f=fl, r=fr, c=cl, w=sel, s=step,
            x=px, y=py, z=pz, session=session, log=log,
        )  # fmt: skip
        run = session.load(program)
        group = (0, cores)
        try:
            us = [timed(run, Pixels.lerp3, group, (t[0], t[1], tx, top, n))]
            us.append(timed(run, Pixels.lerp3, group, (t[2], t[3], tx, bottom, n)))
            us.append(timed(run, Pixels.lerp3, group, (top, bottom, ty, out, n)))
            report["bilinear_us"] = us
            report["unary_us"] = timed(run, Pixels.unary, group, (tu, fl, fr, cl, n))
            report["floor_us"] = timed(run, Pixels.floor_only, group, (tu, fl, n))
            us = timed(run, Pixels.floor_round, group, (tu, ro, n))
            report["floor_magic_us"] = us
            report["where_us"] = timed(run, Pixels.depth, group, (ta, tb, sel, n))
            us = timed(run, Pixels.depth_round, group, (ta, tb, step, n))
            report["step_us"] = us
            words = tuple(float(v) for v in m[0])
            call = (px, py, pz, clip_out, n, *words)
            report["vertex_us"] = timed(run, Pixels.vertex, group, call)
        finally:
            run.stop()
        for name, tensor in (
            ("bilinear", out), ("floor", fl), ("frac", fr), ("clamp", cl),
            ("floor_magic", ro), ("where", sel), ("step", step), ("vertex", clip_out),
        ):  # fmt: skip
            got[name] = read(session, tensor)
        pts = tiles(session, points)
        mat = tiles(session, matrix)
        clip = tiles(session, np.zeros((cores * 32, 32)), "float32")
        program = Camera(points=pts, matrix=mat, clip=clip, session=session, log=log)
        run = session.load(program)
        try:
            timed(run, Camera.load, group, (mat,))
            report["camera_us"] = timed(run, Camera.transform, group, (pts, clip))
        finally:
            run.stop()
        got["camera"] = read(session, clip)[:, :4]
    floor = np.floor(u)
    want = {
        "bilinear": bilinear(*texels, fx, fy),
        "floor": floor,
        "frac": u - floor,
        "clamp": np.clip(u, 0, 1),
        "floor_magic": floor,
        "where": np.minimum(z_new, z_old),
        "step": np.minimum(z_new, z_old),
        "vertex": xyz[0] * m[0, 0] + xyz[1] * m[0, 1] + xyz[2] * m[0, 2] + m[0, 3],
        "camera": points[:, :4] @ matrix[:4, :4],
    }
    for name in want:
        report[name] = error(got[name], want[name])
        log(f"{name}: {report[name]}")
    finish(
        report,
        [
            ("bilinear", "rms", 1e-6),
            ("floor", "max", 0.0),
            ("frac", "max", 1e-6),
            ("clamp", "max", 0.0),
            ("floor_magic", "max", 0.0),
            ("where", "max", 0.0),
            ("step", "rms", 1e-6),
            ("vertex", "rms", 1e-6),
            ("camera", "rms", 2.9e-4),  # a sum of exact products is not exact
        ],
    )


if __name__ == "__main__":
    sys.exit(main())
