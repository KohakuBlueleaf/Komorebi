"""Packed RGBA8 texels (one uint32 a texel) shaded as floats, against numpy: a tile
function of a uint32 tile (refused when it is built: its text is reported), and the
way round, a body on mover B that unpacks each page of words into four float32 pages,
then a tile function of the four: luminance times alpha, 0.299 r + 0.587 g + 0.114 b."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nn"))

import numpy as np
from forms import error, finish, log, timed, words

import komorebi as km

# Word i of a page into word i of four float32 pages; v to float32 bits, v < 256
UNPACK = """
auto bits = [](uint32_t v) -> uint32_t {
    if (v == 0) { return 0; }
    uint32_t e = 0;
    while (v >> (e + 1)) { ++e; }
    return ((127 + e) << 23) | ((v << (23 - e)) & 0x7FFFFF);
};
volatile tt_l1_ptr uint32_t* in = (volatile tt_l1_ptr uint32_t*)get_write_ptr(packed);
volatile tt_l1_ptr uint32_t* r = (volatile tt_l1_ptr uint32_t*)get_write_ptr(red);
volatile tt_l1_ptr uint32_t* g = (volatile tt_l1_ptr uint32_t*)get_write_ptr(green);
volatile tt_l1_ptr uint32_t* b = (volatile tt_l1_ptr uint32_t*)get_write_ptr(blue);
volatile tt_l1_ptr uint32_t* a = (volatile tt_l1_ptr uint32_t*)get_write_ptr(alpha);
for (uint32_t p = first; p < first + n; ++p) {
    km::read_pages(movement, command, t, p, 1, get_write_ptr(packed));
    noc_async_read_barrier();
    for (uint32_t i = 0; i < 1024; ++i) {
        const uint32_t w = in[i];
        r[i] = bits(w & 255);
        g[i] = bits((w >> 8) & 255);
        b[i] = bits((w >> 16) & 255);
        a[i] = bits(w >> 24);
    }
    km::write_pages(movement, command, tr, p, 1, get_write_ptr(red));
    km::write_pages(movement, command, tg, p, 1, get_write_ptr(green));
    km::write_pages(movement, command, tb, p, 1, get_write_ptr(blue));
    km::write_pages(movement, command, ta, p, 1, get_write_ptr(alpha));
    noc_async_write_barrier();
}
"""


@km.tile_function(tolerance=1e-6)
def shade(r, g, b, a):
    return (0.299 * r + 0.587 * g + 0.114 * b) * (a * (1.0 / 65025.0))


@km.tile_function(tolerance=1e-6)
def shade_words(t):
    return t * (1.0 / 255.0)


class Texels(km.Program):
    """Core r works on the pages r n to r n + n - 1 of each tensor."""

    packed: km.L1[km.u32, 1, 32, 32] = km.own(push=None)
    red: km.L1[km.fp32, 1, 32, 32] = km.own(push=None)
    green: km.L1[km.fp32, 1, 32, 32] = km.own(push=None)
    blue: km.L1[km.fp32, 1, 32, 32] = km.own(push=None)
    alpha: km.L1[km.fp32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def unpack(
        self,
        t: km.Tensor,
        tr: km.Tensor,
        tg: km.Tensor,
        tb: km.Tensor,
        ta: km.Tensor,
        n: int,
    ):
        first = self.rank * n
        km.cxx(
            UNPACK,
            takes=[
                t, tr, tg, tb, ta, first, n,
                self.packed, self.red, self.green, self.blue, self.alpha,
            ],  # fmt: skip
            on="mover_b",
        )

    @km.handler
    def light(
        self,
        tr: km.Tensor,
        tg: km.Tensor,
        tb: km.Tensor,
        ta: km.Tensor,
        o: km.Tensor,
        n: int,
    ):
        first = self.rank * n
        out = km.Stream(o.pages(first, n), write=True)
        for r, g, b, a in zip(
            km.Stream(tr.pages(first, n)),
            km.Stream(tg.pages(first, n)),
            km.Stream(tb.pages(first, n)),
            km.Stream(ta.pages(first, n)),
        ):
            out.push(shade(r, g, b, a))


class Direct(km.Program):
    """A tile function of the uint32 texel tile."""

    @km.handler
    def light(self, t: km.Tensor, o: km.Tensor, n: int):
        first = self.rank * n
        out = km.Stream(o.pages(first, n), write=True)
        for x in km.Stream(t.pages(first, n)):
            out.push(shade_words(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--tiles", type=int, default=2, help="pages a core")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cores, n = args.cores, args.tiles
    pages = cores * n
    texels = rng.integers(0, 1 << 32, (pages, 1024), dtype=np.uint64).astype(np.uint32)
    texels[0, :4] = [0, 0xFFFFFFFF, 0x01010101, 0x80FF0080]  # zeros, 255s, 1s, mixed
    channels = [((texels >> s) & 255).astype(np.float64) for s in (0, 8, 16, 24)]
    r, g, b, a = channels
    want = (0.299 * r + 0.587 * g + 0.114 * b) * (a / 65025.0)
    report = {"cores": cores, "pages": pages}
    with km.Session(log=log) as session:
        t = session.tensor(texels, "uint32")
        zeros = np.zeros((pages * 32, 32), np.float32)
        made = [session.tensor(zeros, "float32", "tile") for _ in range(5)]
        tr, tg, tb, ta, o = made
        try:
            tw = session.tensor(texels.reshape(pages * 32, 32), "uint32", "tile")
            program = Direct(t=tw, o=o, session=session, log=log)
            session.load(program).stop()
            report["direct_refused"] = None
        except Exception as refusal:  # noqa: BLE001 - the refusal is the result
            report["direct_refused"] = f"{type(refusal).__name__}: {refusal}"
        log(f"direct: {report['direct_refused']}")
        names = dict(t=t, tr=tr, tg=tg, tb=tb, ta=ta, o=o)
        run = session.load(Texels(**names, session=session, log=log))
        group = (0, cores)
        try:
            call = (t, tr, tg, tb, ta, n)
            report["unpack_us"] = timed(run, Texels.unpack, group, call)
            call = (tr, tg, tb, ta, o, n)
            report["light_us"] = timed(run, Texels.light, group, call)
        finally:
            run.stop()
        got = [words(session, x).view(np.float32).astype(np.float64) for x in made]
    report["unpacked_wrong"] = {
        "count": int(sum(np.count_nonzero(x != c) for x, c in zip(got, channels)))
    }
    report["light"] = error(got[4], want)  # word i of a page is texel i: elementwise
    checks = [("unpacked_wrong", "count", 0), ("light", "rms", 1e-6)]
    finish(report, checks)


if __name__ == "__main__":
    sys.exit(main())
