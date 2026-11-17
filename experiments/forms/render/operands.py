"""A bilinear filter of 6 operand tiles (4 texels, fx, fy) in one pass, float32, against
numpy: the tile function of 6 operands (refused when it is built: its text is reported)
and the same in an inline body that leaves the tile in DST 0. filter.py has the three
lerp passes through two scratch tensors, at the same sizes."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nn"))

import numpy as np
from forms import error, finish, log, read, tiles, timed

import komorebi as km

# Within DST 0 to 3 (fp32 DST's half): top into 0, bottom into 1, then 0 + (1 - 0) fy
BILINEAR = """
#include "api/compute/eltwise_binary_sfpu.h"
copy_init(t00_buffer);
copy_tile(t00_buffer, t00_index, 0);
copy_init(t01_buffer);
copy_tile(t01_buffer, t01_index, 1);
copy_init(wx_buffer);
copy_tile(wx_buffer, wx_index, 3);
sub_binary_tile_init();
sub_binary_tile(1, 0, 1);
mul_binary_tile_init();
mul_binary_tile(1, 3, 1);
add_binary_tile_init();
add_binary_tile(0, 1, 0);
copy_init(t10_buffer);
copy_tile(t10_buffer, t10_index, 1);
copy_init(t11_buffer);
copy_tile(t11_buffer, t11_index, 2);
sub_binary_tile_init();
sub_binary_tile(2, 1, 2);
mul_binary_tile_init();
mul_binary_tile(2, 3, 2);
add_binary_tile_init();
add_binary_tile(1, 2, 1);
copy_init(wy_buffer);
copy_tile(wy_buffer, wy_index, 3);
sub_binary_tile_init();
sub_binary_tile(1, 0, 1);
mul_binary_tile_init();
mul_binary_tile(1, 3, 1);
add_binary_tile_init();
add_binary_tile(0, 1, 0);
"""


@km.tile_function
def bilinear6(t00, t01, t10, t11, wx, wy):
    top = t00 + (t01 - t00) * wx
    bottom = t10 + (t11 - t10) * wx
    return top + (bottom - top) * wy


class Body(km.Program):
    """Core r works on the tiles r n to r n + n - 1 of each tensor."""

    @km.handler
    def bilinear(
        self,
        a: km.Tensor,
        b: km.Tensor,
        c: km.Tensor,
        d: km.Tensor,
        fx: km.Tensor,
        fy: km.Tensor,
        o: km.Tensor,
        n: int,
    ):
        first = self.rank * n
        out = km.Stream(o.pages(first, n), write=True)
        for t00, t01, t10, t11, wx, wy in zip(
            km.Stream(a.pages(first, n)),
            km.Stream(b.pages(first, n)),
            km.Stream(c.pages(first, n)),
            km.Stream(d.pages(first, n)),
            km.Stream(fx.pages(first, n)),
            km.Stream(fy.pages(first, n)),
        ):
            mixed = km.cxx(BILINEAR, takes=[t00, t01, t10, t11, wx, wy], gives=km.Tile)
            out.push(mixed)


class Direct(km.Program):
    """The tile function of 6 operands."""

    @km.handler
    def bilinear(
        self,
        a: km.Tensor,
        b: km.Tensor,
        c: km.Tensor,
        d: km.Tensor,
        fx: km.Tensor,
        fy: km.Tensor,
        o: km.Tensor,
        n: int,
    ):
        first = self.rank * n
        out = km.Stream(o.pages(first, n), write=True)
        for t00, t01, t10, t11, wx, wy in zip(
            km.Stream(a.pages(first, n)),
            km.Stream(b.pages(first, n)),
            km.Stream(c.pages(first, n)),
            km.Stream(d.pages(first, n)),
            km.Stream(fx.pages(first, n)),
            km.Stream(fy.pages(first, n)),
        ):
            out.push(bilinear6(t00, t01, t10, t11, wx, wy))


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

    t00, t01, t10, t11 = (f32(rng.uniform(0, 1, shape)) for _ in range(4))
    fx, fy = f32(rng.uniform(0, 1, shape)), f32(rng.uniform(0, 1, shape))
    top, bottom = t00 + (t01 - t00) * fx, t10 + (t11 - t10) * fx
    want = top + (bottom - top) * fy
    report = {"cores": cores, "tiles": n}
    with km.Session(log=log) as session:
        made = [tiles(session, v, "float32") for v in (t00, t01, t10, t11, fx, fy)]
        out = tiles(session, np.zeros(shape), "float32")
        names = dict(zip("abcd", made[:4]), fx=made[4], fy=made[5], o=out)
        try:
            session.load(Direct(**names, session=session, log=log)).stop()
            report["direct_refused"] = None
        except Exception as refusal:  # noqa: BLE001 - the refusal is the result
            report["direct_refused"] = f"{type(refusal).__name__}: {refusal}"
        log(f"direct: {report['direct_refused']}")
        run = session.load(Body(**names, session=session, log=log))
        try:
            call = (*made, out, n)
            report["body_us"] = timed(run, Body.bilinear, (0, cores), call)
        finally:
            run.stop()
        report["body"] = error(read(session, out), want)
    finish(report, [("body", "rms", 1e-6)])


if __name__ == "__main__":
    sys.exit(main())
