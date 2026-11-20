"""A box piece in a process of its own, only its cards visible to it, its site calls
and its host hooks over a pipe (README.md, `driver/remote.py`)."""

import os
import pickle
import subprocess
import sys
import traceback

from .interface import HOOKS, load_factory, load_weights, set_hooks
from .run import open_session, parse_layers


def _send(f, obj):
    data = pickle.dumps(obj, protocol=5)
    f.write(len(data).to_bytes(8, "little"))
    f.write(data)
    f.flush()


def _recv(f):
    head = f.read(8)
    if len(head) < 8:
        raise EOFError("the piece's process closed its pipe")
    return pickle.loads(f.read(int.from_bytes(head, "little")))


class RemotePiece:
    """The driver's side: `piece(site, layer, *inputs)` as for a piece in the driver's
    own process; a hook the piece calls (`compress_hook`) runs here, in `hooks`."""

    def __init__(
        self, factory, cards, layers, ckpt, log, hooks, kernel_bytes=None, load=""
    ):
        down_r, down_w = os.pipe()
        up_r, up_w = os.pipe()
        cmd = [sys.executable, "-u", "-m", __name__, factory, cards, layers, ckpt]
        cmd += [str(down_r), str(up_w), str(kernel_bytes or 0), load or "-"]
        env = dict(os.environ, TT_VISIBLE_DEVICES=cards)
        with open(log, "w") as out:
            self.proc = subprocess.Popen(
                cmd,
                env=env,
                pass_fds=(down_r, up_w),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
            )
        os.close(down_r)
        os.close(up_w)
        self.tx, self.rx = os.fdopen(down_w, "wb"), os.fdopen(up_r, "rb")
        self.hooks, self.log = hooks, log
        self.sites = tuple(self._wait())

    def _wait(self):
        while True:
            kind, *rest = _recv(self.rx)
            if kind == "result":
                return rest[0]
            if kind == "hook":
                name, args = rest
                _send(self.tx, ("hook_result", self.hooks[name](*args)))
            else:
                raise RuntimeError(f"piece process ({self.log}): {rest[0]}")

    def __call__(self, site, layer, *inputs):
        _send(self.tx, ("call", site, layer, inputs))
        return self._wait()

    def close(self):
        _send(self.tx, ("close",))
        self.proc.wait(timeout=120)


def serve(argv):
    """The piece's side: open the session on the visible cards, make the piece, answer
    calls until "close"."""
    import komorebi as km  # noqa: PLC0415 (only the piece's process opens cards)

    try:  # the module name the pieces check the weights' type against
        from galaxy.reference.model_loader import Checkpoint  # noqa: PLC0415
    except ImportError:
        from ..reference.model_loader import Checkpoint  # noqa: PLC0415

    factory, cards, layers, ckpt, down, up, kernel_bytes, load = argv
    rx, tx = os.fdopen(int(down), "rb"), os.fdopen(int(up), "wb")

    def hook(name, args):
        _send(tx, ("hook", name, args))
        kind, value = _recv(rx)
        assert kind == "hook_result", kind
        return value

    try:
        session = open_session(km, parse_layers(cards, 0), int(kernel_bytes) or None)
        weights = Checkpoint(ckpt)
        if load != "-":
            weights = load_weights(session, load, parse_layers)
        piece = load_factory(factory)(session, weights, parse_layers(layers, 40))
        set_hooks(piece, {n: lambda *a, _n=n: hook(_n, a) for n in HOOKS})
        _send(tx, ("result", piece.sites))
        while True:
            msg = _recv(rx)
            if msg[0] == "close":
                break
            _, site, layer, inputs = msg
            _send(tx, ("result", piece(site, layer, *inputs)))
    except Exception:  # the driver gets the traceback as the error
        _send(tx, ("error", traceback.format_exc()))
        raise
    finally:
        print("piece process ends", flush=True)


if __name__ == "__main__":
    serve(sys.argv[1:])
