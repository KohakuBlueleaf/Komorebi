import numpy as np

import komorebi as km


@km.tile_function(tolerance=1e-4)
def ok(row):
    mean = km.reduce_sum(row, exact=True) * 0.015625
    return row - mean.col()


@km.tile_function(tolerance=1e-4)
def bad_mean(row):
    mean = km.reduce_sum(row, exact=True) * 0.015625
    return row - mean


@km.tile_function(tolerance=1e-4)
def bad_mix(row):
    mean = km.reduce_sum(row, exact=True) * 0.015625
    return row * mean.row()


@km.tile_function(tolerance=1e-4)
def bad_weight(row, w):
    mean = km.reduce_sum(row, exact=True) * 0.015625
    return (row - mean.col()) * w


def prog(fn, field):
    if field:

        class P(km.Program):
            w: km.L1[km.fp32, 1, 32, 32]

            @km.handler
            def go(self, x: km.Tensor, sc: km.Tensor, y: km.Tensor):
                self.w.fill(km.Stream(sc.pages(0, 1)))
                results = km.Stream(y.pages(0, 2), write=True, block=2)
                for row in km.Stream(x.pages(0, 2), block=2, row=2):
                    results.push(fn(row, self.w))

        return P

    class Q(km.Program):
        @km.handler
        def go(self, x: km.Tensor, y: km.Tensor):
            results = km.Stream(y.pages(0, 2), write=True, block=2)
            for row in km.Stream(x.pages(0, 2), block=2, row=2):
                results.push(fn(row))

    return Q


with km.Session() as s:
    x = s.tensor(np.random.rand(32, 64).astype(np.float32), "bfloat16", "tile")
    sc = s.tensor(np.random.rand(32, 32).astype(np.float32), "float32", "tile")
    y = s.empty((32, 64), "bfloat16", "tile")
    for name, fn, field in (
        ("row - sum.col()", ok, False),
        ("row - sum (no .col())", bad_mean, False),
    ):
        try:
            cls = prog(fn, field)
            run = s.load(cls(x=x, y=y, session=s, **({"sc": sc} if field else {})))
            run.stop()
            print("CASE", name, "-> loaded")
        except Exception as e:  # noqa: BLE001
            print("CASE", name, "-> REFUSED:", str(e).replace("\n", " ")[:330])
