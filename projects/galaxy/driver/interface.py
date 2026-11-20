"""The one interface every box piece offers the driver (README.md,
`driver/interface.py`)."""

import importlib
from typing import Protocol

# a piece's attributes the driver sets to host functions, when the piece has them
HOOKS = ("compress_hook",)


class BoxPiece(Protocol):
    sites: tuple

    def __call__(self, site: str, layer, *inputs): ...


def set_hooks(piece, hooks):
    """Set each of HOOKS the piece declares (as None) to `hooks[name]`."""
    for name in HOOKS:
        if hasattr(piece, name):
            setattr(piece, name, hooks[name])


def load_factory(spec: str):
    """'package.module:function' -> the factory."""
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn or "make")


def load_weights(session, spec, parse_layers):
    """The loader's W on the session (`weights/README.md`). spec "all": the whole
    backbone (a session of the 31 cards); else `layers@kinds[@formats]`, e.g.
    `0-39@attn.wq_,attn.wkv,attn.wo_@bfp8_b`: only those weights (`host.select`)."""
    from projects.galaxy.weights import host  # noqa: PLC0415 (only box runs load)

    if spec == "all":
        return host.loader.load(session)
    lay, _, rest = spec.partition("@")
    kinds, _, formats = rest.partition("@")
    names = host.select(
        layers=(
            [x if x is not None else "head" for x in parse_layers(lay, 40)]
            if lay
            else None
        ),
        kinds=kinds.split(",") if kinds else None,
        formats=formats.split(",") if formats else None,
    )
    return host.loader.load(session, names)


def install(path, piece: BoxPiece, layers=None, sites=None):
    """Route the given sites (default: all the piece offers) of the given layers to the
    piece."""
    for s in sites or piece.sites:
        path.put(s, lambda layer, *inputs, _s=s: piece(_s, layer, *inputs), layers)
