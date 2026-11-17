"""The calls of one card's session, tensor and run, made on one card, two cards or a
split card (argv[1]: one, two, split) on the mock: a JSON line, each call's shape."""

import json
import sys

import numpy as np

import komorebi as km

OPEN = {
    "one": {"card": 0},
    "two": {"cards": [0, 1]},
    "split": {"card": 0, "islands": 2},
}


class Sums(km.Program):
    @km.handler
    def run(self, source: km.Tensor, tiles: int, output: km.Tensor):
        part = self.partition(source.pages(0, tiles))
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(source[part], block=8):
            total += tile
        output.pages(self.rank, 1).write(total)


def shape(value):
    """What a script relies on of a value: its kind; an array's type and rank."""
    if isinstance(value, np.ndarray):
        return ["array", str(value.dtype), value.ndim]
    if isinstance(value, bool | type(None)):
        return repr(value)
    if isinstance(value, int | float | str):
        return type(value).__name__
    if isinstance(value, tuple | list):
        return [type(value).__name__, [shape(one) for one in value]]
    if isinstance(value, dict):
        return ["dict", sorted(value)]
    return type(value).__name__


def main(mode):
    shapes = {}

    def call(name, read):
        try:
            shapes[name] = shape(read())
        except Exception as error:  # noqa: BLE001 (its kind is the shape)
            shapes[name] = ["refused", type(error).__name__]

    with km.Session(**OPEN[mode]) as session:
        data = np.ones((8 * 32, 32), np.float32)
        tensor = session.tensor(data, "bfloat16", "tile")
        rows = np.zeros((tensor.pages, tensor.page_bytes), np.uint8)
        call("session.grid", lambda: session.grid)
        call("session.nocs()", session.nocs)
        call("session.l1()", session.l1)
        call("session.pll_mhz", lambda: session.pll_mhz)
        call("session.read(out=)", lambda: session.read(tensor, out=rows)[0] is rows)
        call(
            "session.read(first=, pages=)",
            lambda: session.read(tensor, first=1, pages=2),
        )
        call("session.read_chunks", lambda: list(session.read_chunks(tensor, 4096)))
        call("session.write", lambda: session.write(tensor, rows))
        call("tensor.read()", tensor.read)
        call("tensor.read_bytes()", tensor.read_bytes)
        call("tensor.nbytes", lambda: tensor.nbytes)
        call("tensor.pages", lambda: tensor.pages)
        call("tensor.address", lambda: tensor.address)
        call("tensor.tensor_args()", lambda: tensor.tensor_args()[:0])
        call(
            "empty(own=True)",
            lambda: session.empty((64, 32), "bfloat16", own=True).pages,
        )
        output = session.empty((256 * 32, 32), "float32", "tile")
        run = session.load(Sums(source=tensor, output=output), 64)
        host = run.host  # the session's own run, under the mock's
        ticket = host.submit(Sums.run, (0, 8), args=(tensor, 8, output))
        call("run.alive()", host.alive)
        call("run.h", lambda: host.h)
        call("run.programs", lambda: len(host.programs))
        call("run.added", lambda: host.added)
        call("run.submit_s", lambda: host.submit_s)
        call("run.wait_s", lambda: host.wait_s)
        call("run.native_s", lambda: host.native_s)
        call("run.stalls", lambda: host.stalls)
        call("run.recovery_refused", lambda: host.recovery_refused is None)
        call("run.forget(handles=)", lambda: host.forget(handles=[ticket]))
        call("run.given_up(handle=)", lambda: host.given_up(handle=ticket))
        call("run.sequence", lambda: host.sequence)
        call("run.upload", lambda: host.upload)
        call("tensor.freed", lambda: tensor.freed)
        tensor.free()
        call("tensor.freed after free()", lambda: tensor.freed)
    print("SHAPES " + json.dumps(shapes, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
