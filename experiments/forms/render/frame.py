"""A frame's five passes as one class, against numpy: vertex, bin (a scatter of the
screen tiles into list pages by each tile's bin, a body on mover B), raster (an entries
loop over its bin's first page and count), shade and resolve; as 5 commands, split at
the scatter (a chain, two commands, a chain), and as one chain through bin."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nn"))

import numpy as np
from forms import error, finish, log, read, tiles, timed, words

import komorebi as km

BINS = 4  # bins, and the cores raster runs on; page BINS of counts: the bins in use
BIN = """
volatile tt_l1_ptr uint32_t* ids = (volatile tt_l1_ptr uint32_t*)get_write_ptr(stage);
volatile tt_l1_ptr uint32_t* pair = (volatile tt_l1_ptr uint32_t*)get_write_ptr(span);
km::read_pages(movement, command, bins, 0, 1, get_write_ptr(stage));
noc_async_read_barrier();
uint32_t first = 0;
uint32_t active = 0;
for (uint32_t b = 0; b < groups; ++b) {
    uint32_t placed = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (ids[i] != b) {
            continue;
        }
        km::read_pages(movement, command, screen, i, 1, get_write_ptr(tile_page));
        noc_async_read_barrier();
        const uint32_t at = first + placed;
        km::write_pages(movement, command, lists, at, 1, get_write_ptr(tile_page));
        noc_async_write_barrier();
        ++placed;
    }
    pair[0] = first;
    pair[1] = placed;
    km::write_pages(movement, command, counts, b, 1, get_write_ptr(span));
    noc_async_write_barrier();
    first += placed;
    active = placed ? b + 1 : active;
}
pair[0] = active;
pair[1] = 0;
km::write_pages(movement, command, counts, groups, 1, get_write_ptr(span));
noc_async_write_barrier();
"""


@km.tile_function
def transform(x, y, z):
    return x * 0.5 + y * 0.25 + z * 0.125 + 0.0625


@km.tile_function
def coverage(c):
    return km.minimum(km.maximum(c, 0.0), 1.0)


@km.tile_function
def shade_fn(zbuf, albedo, light):
    return albedo * light + zbuf * 0.25


@km.tile_function
def tone(c):
    return c * 0.5 + 0.25


class Span(km.Entry):
    first: int
    pages: int


class Frame(km.Program):
    """Core r works on tiles r n to r n + n - 1; bin runs on one core, raster on a
    core a bin."""

    stage: km.L1[km.u32, 1, 32, 32] = km.own(push=None)
    tile_page: km.L1[km.u32, 1, 32, 32] = km.own(push=None)
    span: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def vertex(
        self, x: km.Tensor, y: km.Tensor, z: km.Tensor, screen: km.Tensor, n: int
    ):
        first = self.rank * n
        out = km.Stream(screen.pages(first, n), write=True)
        for a, b, c in zip(
            km.Stream(x.pages(first, n)),
            km.Stream(y.pages(first, n)),
            km.Stream(z.pages(first, n)),
        ):
            out.push(transform(a, b, c))

    @km.handler
    def bin(
        self,
        screen: km.Tensor,
        bins: km.Tensor,
        lists: km.Tensor,
        counts: km.Tensor,
        count: int,
        groups: int,
    ):
        km.cxx(
            BIN,
            takes=[
                screen, bins, lists, counts, count, groups,
                self.stage, self.tile_page, self.span,
            ],
            reads=[screen, bins],
            writes=[lists, counts],
            on="mover_b",
        )  # fmt: skip

    @km.handler
    def raster(self, lists: km.Tensor, counts: km.Tensor, zbuf: km.Tensor):
        for e in counts.pages(self.rank, 1).entries(Span, first=0, size=2, fixed=1):
            out = km.Stream(zbuf.pages(e.first, e.pages), write=True)
            for tile in km.Stream(lists.pages(e.first, e.pages)):
                out.push(coverage(tile))

    @km.handler
    def shade(
        self,
        zbuf: km.Tensor,
        albedo: km.Tensor,
        light: km.Tensor,
        color: km.Tensor,
        n: int,
    ):
        first = self.rank * n
        out = km.Stream(color.pages(first, n), write=True)
        for d, a, li in zip(
            km.Stream(zbuf.pages(first, n)),
            km.Stream(albedo.pages(first, n)),
            km.Stream(light.pages(first, n)),
        ):
            out.push(shade_fn(d, a, li))

    @km.handler
    def resolve(self, color: km.Tensor, image: km.Tensor, n: int):
        first = self.rank * n
        out = km.Stream(image.pages(first, n), write=True)
        for c in km.Stream(color.pages(first, n)):
            out.push(tone(c))


def reference(x, y, z, ids, albedo, light):
    """numpy, tile by tile in float32: the list order is the tiles sorted by bin."""
    f = np.float32
    screen = x * f(0.5) + y * f(0.25) + z * f(0.125) + f(0.0625)
    tiled = screen.reshape(-1, 32, 32)
    order = np.argsort(ids, kind="stable")
    lists = tiled[order].reshape(screen.shape)
    zbuf = np.clip(lists, 0, 1)
    color = albedo * light + zbuf * f(0.25)
    return color * f(0.5) + f(0.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--tiles", type=int, default=2, help="tiles a core")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cores, n = args.cores, args.tiles
    count = cores * n
    shape = (count * 32, 32)
    x, y, z, albedo, light = (
        rng.uniform(-0.5, 2.0, shape).astype(np.float32) for _ in range(5)
    )
    ids = np.concatenate([np.arange(BINS), rng.integers(0, BINS, count - BINS)])
    ids = rng.permutation(ids).astype(np.uint32)  # every bin holds a tile
    page = np.zeros((1, 1024), np.uint32)
    page[0, :count] = ids
    report = {"cores": cores, "tiles": count, "bins": BINS}
    got = {}
    with km.Session(log=log) as session:
        tx, ty, tz, ta, tl = (
            tiles(session, v, "float32") for v in (x, y, z, albedo, light)
        )
        bins = session.tensor(page, "uint32")

        def scratch():
            made = [tiles(session, np.zeros(shape), "float32") for _ in range(5)]
            spans = session.tensor(np.zeros((BINS + 1, 1024), np.uint32), "uint32")
            return (*made, spans)

        sets = {"commands": scratch(), "split": scratch(), "chain": scratch()}
        screen, lists, zbuf, color, image, counts = sets["commands"]
        program = Frame(
            x=tx, y=ty, z=tz, screen=screen, bins=bins, lists=lists, counts=counts,
            zbuf=zbuf, albedo=ta, light=tl, color=color, image=image,
            chain=True, session=session, log=log,
        )  # fmt: skip
        run = session.load(program)
        group = (0, cores)
        try:
            us = [timed(run, Frame.vertex, group, (tx, ty, tz, screen, n))]
            call = (screen, bins, lists, counts, count, BINS)
            us.append(timed(run, Frame.bin, (0, 1), call))
            us.append(timed(run, Frame.raster, (0, BINS), (lists, counts, zbuf)))
            us.append(timed(run, Frame.shade, group, (zbuf, ta, tl, color, n)))
            us.append(timed(run, Frame.resolve, group, (color, image, n)))
            report["commands_us"] = us
            screen, lists, zbuf, color, image, counts = sets["split"]
            first = run.chain(
                [Frame.vertex(tx, ty, tz, screen, n, cores=group)], cores=group
            )
            last = run.chain(
                [
                    Frame.shade(zbuf, ta, tl, color, n, cores=group),
                    Frame.resolve(color, image, n, cores=group),
                ],
                cores=group,
            )
            event = run.wait(run.submit(first), timeout=30.0)
            us = [round(event.microseconds, 1)]
            call = (screen, bins, lists, counts, count, BINS)
            us.append(timed(run, Frame.bin, (0, 1), call))
            us.append(timed(run, Frame.raster, (0, BINS), (lists, counts, zbuf)))
            event = run.wait(run.submit(last), timeout=30.0)
            us.append(round(event.microseconds, 1))
            report["split_us"] = us
            screen, lists, zbuf, color, image, counts = sets["chain"]
            through = [
                Frame.vertex(tx, ty, tz, screen, n, cores=group),
                Frame.bin(screen, bins, lists, counts, count, BINS, cores=(0, 1)),
                Frame.raster(lists, counts, zbuf, cores=(0, BINS)),
                Frame.shade(zbuf, ta, tl, color, n, cores=group),
                Frame.resolve(color, image, n, cores=group),
            ]
            try:
                chain = run.chain(through, cores=group)
            except Exception as refused:  # noqa: BLE001 - the refusal is the result
                report["one_chain"] = f"{type(refused).__name__}: {refused}"[:400]
                del sets["chain"]
            else:
                event = run.wait(run.submit(chain), timeout=30.0)
                report["one_chain"] = event.kind
                report["chain_us"] = round(event.microseconds, 1)
            log(f"one chain through bin: {report['one_chain']}")
        finally:
            run.stop()
        for name, (_, _, _, _, image, counts) in sets.items():
            got[name] = (
                read(session, image),
                words(session, image),
                words(session, counts),
            )
    want = reference(x, y, z, ids, albedo, light).astype(np.float64)
    spans = np.zeros((BINS, 2), np.uint32)
    spans[:, 1] = np.bincount(ids, minlength=BINS)
    spans[1:, 0] = np.cumsum(spans[:-1, 1])
    for name, (image, _, counts) in got.items():
        report[name] = error(image, want)
        wrong = int(np.count_nonzero(counts[:BINS, :2] != spans))
        wrong += int(counts[BINS, 0] != ids.max() + 1)
        report[f"{name}_spans_wrong"] = {"count": wrong}
        log(f"{name}: {report[name]}, spans wrong {wrong}")
    checks = []
    for name in got:
        checks += [(name, "rms", 1e-6), (f"{name}_spans_wrong", "count", 0)]
        if name != "commands":
            same = np.array_equal(got["commands"][1], got[name][1])
            report[f"{name}_equals_commands"] = {"count": 0 if same else 1}
            checks.append((f"{name}_equals_commands", "count", 0))
    finish(report, checks)


if __name__ == "__main__":
    sys.exit(main())
