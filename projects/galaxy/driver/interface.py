"""The one interface every box piece offers the driver (README.md,
`driver/interface.py`)."""

import importlib
import sys
from typing import Protocol

# a piece's attributes the driver sets to host functions, when the piece has them
HOOKS = ("compress_hook",)
# the weights a factory takes, as it or its module declares WEIGHTS (default: the first)
WEIGHTS = ("loader", "checkpoint")


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


def weights_for(factory, loader=None, checkpoint=None):
    """The weights `factory` takes, by its or its module's `WEIGHTS`: "loader" (the
    default) the loader's W, None when nothing was loaded (the piece loads its own);
    "checkpoint" the Checkpoint, refused by name when none was given."""
    name = f"{factory.__module__}:{factory.__name__}"
    declared = getattr(sys.modules[factory.__module__], "WEIGHTS", WEIGHTS[0])
    declared = getattr(factory, "WEIGHTS", declared)
    if declared not in WEIGHTS:
        raise ValueError(f"{name}: WEIGHTS = {declared!r}, not one of {WEIGHTS}")
    if declared == "loader":
        return loader
    if checkpoint is None:
        raise ValueError(
            f'{name} takes the Checkpoint (WEIGHTS = "checkpoint"): none given'
        )
    return checkpoint


def load_weights(session, spec, parse_layers):
    """The loader's W on the session (`weights/README.md`). spec "all": the whole
    backbone (a session of the 31 cards); else groups `layers@kinds[@formats]` joined by
    `+`, e.g. `0-39@attn.wq_,attn.wkv,attn.wo_@bfp8_b+0-4@ffn.` (`host.select`)."""
    from projects.galaxy.weights import host  # noqa: PLC0415 (only box runs load)

    if spec == "all":
        return host.loader.load(session)
    names = {}  # in order, once each
    for group in spec.split("+"):
        lay, _, rest = group.partition("@")
        kinds, _, formats = rest.partition("@")
        layers = [x if x is not None else "head" for x in parse_layers(lay, 40)]
        picked = host.select(
            layers=layers if lay else None,
            kinds=kinds.split(",") if kinds else None,
            formats=formats.split(",") if formats else None,
        )
        names.update(dict.fromkeys(picked))
    return host.loader.load(session, list(names))


def install(path, piece: BoxPiece, layers=None, sites=None):
    """Route the given sites (default: all the piece offers) of the given layers to the
    piece."""
    for s in sites or piece.sites:
        path.put(s, lambda layer, *inputs, _s=s: piece(_s, layer, *inputs), layers)
