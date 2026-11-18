import numpy as np

import komorebi as km


@km.tile_function(tolerance=1e-2)
def nocol(x, f):
    return x * f  # field with no .col()


@km.tile_function(tolerance=1e-2)
def withcol(x, f):
    return x * f.col()


@km.tile_function(tolerance=1e-2)
def sumnocol(x):
    return x - km.reduce_sum(x)  # reduced value against a tile, no .col()


@km.tile_function(tolerance=1e-2)
def sumcol(x):
    return x - km.reduce_sum(x).col()


class FieldNo(km.Program):
    f: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, x: km.Tensor, sc: km.Tensor, y: km.Tensor):
        self.f.fill(km.Stream(sc.pages(0, 1)))
        results = km.Stream(y.pages(0, 2), write=True, block=2)
        for xb in km.Stream(x.pages(0, 2), block=2):
            results.push(nocol(xb, self.f))


class FieldYes(km.Program):
    f: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, x: km.Tensor, sc: km.Tensor, y: km.Tensor):
        self.f.fill(km.Stream(sc.pages(0, 1)))
        results = km.Stream(y.pages(0, 2), write=True, block=2)
        for xb in km.Stream(x.pages(0, 2), block=2):
            results.push(withcol(xb, self.f))


class SumNo(km.Program):
    @km.handler
    def go(self, x: km.Tensor, y: km.Tensor):
        results = km.Stream(y.pages(0, 2), write=True, block=2)
        for xb in km.Stream(x.pages(0, 2), block=2):
            results.push(sumnocol(xb))


class SumYes(km.Program):
    @km.handler
    def go(self, x: km.Tensor, y: km.Tensor):
        results = km.Stream(y.pages(0, 2), write=True, block=2)
        for xb in km.Stream(x.pages(0, 2), block=2):
            results.push(sumcol(xb))


with km.Session() as s:
    x = s.tensor(np.random.rand(64, 32).astype(np.float32), "bfloat16", "tile")
    sc = s.tensor(np.random.rand(32, 32).astype(np.float32), "float32", "tile")
    y = s.empty((64, 32), "bfloat16", "tile")
    for name, cls, kw in (
        ("field no .col()", FieldNo, {"sc": sc}),
        ("field .col()", FieldYes, {"sc": sc}),
    ):
        try:
            run = s.load(cls(x=x, y=y, session=s, **kw))
            run.stop()
            print(name, "-> loaded")
        except Exception as e:  # noqa: BLE001
            print(name, "-> REFUSED:", str(e).splitlines()[0][:230])
