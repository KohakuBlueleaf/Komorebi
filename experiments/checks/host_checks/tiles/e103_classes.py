import komorebi as km


@km.tile_function
def expo(a):
    return km.exp(a)


@km.tile_function
def gelu(a):
    return a * 0.5 * (1.0 + km.erf(a * 0.7071067811865476))


@km.tile_function(tolerance=1e-5)
def row_sum(row):
    return row * 0.0 + km.reduce_sum(row, exact=True).col()


@km.tile_function(tolerance=1e-5)
def exp_sum(row):
    return row * 0.0 + km.reduce_sum(km.exp(row), exact=True).col()


@km.tile_function(tolerance=1e-5)
def row_max(row):
    return row * 0.0 + km.reduce_max(row, exact=True).col()


class Two(km.Program):
    @km.handler
    def first(self, x: km.Tensor, y: km.Tensor):
        results = km.Stream(y.tiles[0:1, 0:2], write=True)
        for t in km.Stream(x.tiles[0:1, 0:2]):
            results.push(expo(t))

    @km.handler
    def gelu_of(self, x: km.Tensor, y: km.Tensor):
        results = km.Stream(y.tiles[0:1, 0:2], write=True)
        for t in km.Stream(x.tiles[0:1, 0:2]):
            results.push(gelu(t))

    @km.handler
    def second(self, y: km.Tensor, out: km.Tensor):
        results = km.Stream(out.tiles[0:1, 0:2], write=True, block=2)
        for row in km.Stream(y.tiles[0:1, 0:2], block=2, row=2):
            results.push(row_sum(row))


class One(km.Program):
    @km.handler
    def run(self, x: km.Tensor, out: km.Tensor):
        results = km.Stream(out.tiles[0:1, 0:2], write=True, block=2)
        for row in km.Stream(x.tiles[0:1, 0:2], block=2, row=2):
            results.push(exp_sum(row))


class Max(km.Program):
    @km.handler
    def run(self, x: km.Tensor, out: km.Tensor):
        results = km.Stream(out.tiles[0:1, 0:2], write=True, block=2)
        for row in km.Stream(x.tiles[0:1, 0:2], block=2, row=2):
            results.push(row_max(row))


class Linear(km.Program):
    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def layer(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, 2)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:2], block=2),
            km.Stream(w.tiles[0:2, 0:2], block=4),
        ):
            acc += self.project(xb, wb)
        y.tiles[0:1, 0:2].write(acc)


@km.tile_function(tolerance=1e-5)
def masked_sum(row, mask):
    return row * 0.0 + km.reduce_sum(km.exp(row) * mask.row(), exact=True).col()


class Masked(km.Program):
    mask: km.L1[km.fp32, 1, 32, 64]

    @km.handler
    def run(self, x: km.Tensor, m: km.Tensor, out: km.Tensor):
        self.mask.fill(km.Stream(m.pages(0, 2)))
        results = km.Stream(out.tiles[0:1, 0:2], write=True, block=2)
        for row in km.Stream(x.tiles[0:1, 0:2], block=2, row=2):
            results.push(masked_sum(row, self.mask))
