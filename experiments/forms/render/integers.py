"""Integer work a rasterizer wants, against numpy: (a) bit operations on a handler's
int words, seen by the page it writes; (b) a depth comparison of two stream tiles as
a 0/1 mask by km.where; (c) a radix pass, one byte of u32 keys, in a body on mover B."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nn"))

import numpy as np
from forms import error, finish, log, read, tiles, timed, words

import komorebi as km

PAGES = 600  # pages of the tensor (a) writes one of
RADIX = """
volatile tt_l1_ptr uint32_t* keys = (volatile tt_l1_ptr uint32_t*)get_write_ptr(stage);
volatile tt_l1_ptr uint32_t* count =
    (volatile tt_l1_ptr uint32_t*)get_write_ptr(counts);
volatile tt_l1_ptr uint32_t* out =
    (volatile tt_l1_ptr uint32_t*)get_write_ptr(sorted_page);
km::read_pages(movement, command, k, page, 1, get_write_ptr(stage));
noc_async_read_barrier();
for (uint32_t i = 0; i < 256; ++i) { count[i] = 0; }
for (uint32_t i = 0; i < 1024; ++i) { count[(keys[i] >> shift) & 255]++; }
uint32_t total = 0;
for (uint32_t i = 0; i < 256; ++i) {
    const uint32_t c = count[i];
    count[i] = total;
    total += c;
}
for (uint32_t i = 0; i < 1024; ++i) {
    out[count[(keys[i] >> shift) & 255]++] = keys[i];
}
km::write_pages(movement, command, o, page, 1, get_write_ptr(sorted_page));
noc_async_write_barrier();
"""


@km.tile_function
def closer(z_new, z_old):
    return km.where(z_new < z_old, 1.0, 0.0)


@km.tile_function
def closer_round(z_new, z_old):
    return km.minimum(km.maximum((z_old - z_new) * 18446744073709551616.0, 0.0), 1.0)


class Lines(km.Program):
    @km.handler
    def bits(self, t: km.Tensor, a: int, b: int):
        m = (a & 255) | ((b >> 3) << 8)
        k = (a ^ b) % 1024
        n = min(m, k) + max(a, b)
        total = km.tile.zeros(km.fp32)
        t.pages(n, 1).write(total)

    @km.handler
    def mask(self, a: km.Tensor, b: km.Tensor, o: km.Tensor, n: int):
        first = self.rank * n
        out = km.Stream(o.pages(first, n), write=True)
        for x, y in zip(km.Stream(a.pages(first, n)), km.Stream(b.pages(first, n))):
            out.push(closer(x, y))

    @km.handler
    def mask_round(self, a: km.Tensor, b: km.Tensor, o: km.Tensor, n: int):
        """The way round km.where: the difference scaled by 2^64 and clamped."""
        first = self.rank * n
        out = km.Stream(o.pages(first, n), write=True)
        for x, y in zip(km.Stream(a.pages(first, n)), km.Stream(b.pages(first, n))):
            out.push(closer_round(x, y))


class Radix(km.Program):
    stage: km.L1[km.u32, 1, 32, 32] = km.own(push=None)
    counts: km.L1[km.u32, 1, 32, 32] = km.own(push=None)
    sorted_page: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def sort(self, k: km.Tensor, o: km.Tensor, shift: int):
        page = self.rank
        km.cxx(
            RADIX,
            takes=[k, o, page, shift, self.stage, self.counts, self.sorted_page],
            on="mover_b",
        )


def bits(a, b):
    """(a)'s page, as numpy reckons it on unsigned 32-bit words."""
    m = (a & 255) | ((b >> 3) << 8)
    k = (a ^ b) % 1024
    return min(m, k) + max(a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cores = args.cores
    pairs = [(300, 40), (77, 200)]
    shape = (cores * 2 * 32, 32)
    z_new = np.float32(rng.uniform(0, 1, shape)).astype(np.float64)
    z_old = np.float32(rng.uniform(0, 1, shape)).astype(np.float64)
    z_new[0, :4] = z_old[0, :4]  # ties: not closer
    keys = rng.integers(0, 1 << 32, (cores, 1024), dtype=np.uint64).astype(np.uint32)
    report = {"cores": cores, "pairs": pairs}
    with km.Session(log=log) as session:
        written = []
        ta, tb = tiles(session, z_new, "float32"), tiles(session, z_old, "float32")
        to = tiles(session, np.zeros(shape), "float32")
        tr = tiles(session, np.zeros(shape), "float32")
        for a, b in pairs:
            t = tiles(session, np.full((PAGES * 32, 32), 7.0), "float32")
            program = Lines(t=t, a=ta, b=tb, o=to, session=session, log=log)
            run = session.load(program)
            try:
                timed(run, Lines.bits, (0, 1), (t, a, b))
                if not written:
                    call = (ta, tb, to, 2)
                    report["mask_us"] = timed(run, Lines.mask, (0, cores), call)
                    call = (ta, tb, tr, 2)
                    us = timed(run, Lines.mask_round, (0, cores), call)
                    report["mask_round_us"] = us
            finally:
                run.stop()
            pages = read(session, t).reshape(PAGES, 32, 32)
            zero = np.flatnonzero((pages == 0).all((1, 2)))
            written.append([int(p) for p in zero[:4]])
        mask = read(session, to)
        mask_round = read(session, tr)
        k = session.tensor(keys, "uint32")
        o = session.tensor(np.zeros_like(keys), "uint32")
        program = Radix(k=k, o=o, session=session, log=log)
        run = session.load(program)
        try:
            report["radix_us"] = timed(run, Radix.sort, (0, cores), (k, o, 8))
        finally:
            run.stop()
        got = words(session, o)
    want_pages = [[bits(a, b)] for a, b in pairs]
    report["written"] = written
    report["bits_wrong"] = {"count": int(written != want_pages)}
    report["mask"] = error(mask, (z_new < z_old).astype(np.float64))
    report["mask_round"] = error(mask_round, (z_new < z_old).astype(np.float64))
    digit = (keys >> 8) & 255
    order = np.argsort(digit, axis=1, kind="stable")
    want = np.take_along_axis(keys, order, axis=1)
    report["radix_wrong"] = {"count": int(np.count_nonzero(got != want))}
    log(f"pages written {written} (numpy {want_pages}), mask {report['mask']}")
    log(f"radix: {report['radix_wrong']['count']} keys out of place")
    finish(
        report,
        [
            ("bits_wrong", "count", 0),
            ("mask", "max", 0.0),
            ("mask_round", "max", 0.0),
            ("radix_wrong", "count", 0),
        ],
    )


if __name__ == "__main__":
    sys.exit(main())
