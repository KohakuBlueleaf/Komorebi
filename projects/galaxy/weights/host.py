"""The converted cache on the host as the values the chip holds (`held`, `as_released`),
and weight names picked by layer and kind (`select`); usage: README.md, "On the host".
"""

import functools
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import convert  # noqa: E402
import loader  # noqa: E402

formats = convert.formats
COMPACT = ("bfp4_b", "bfp8_b", "bfloat16", "float32")  # the first held is taken


@functools.lru_cache(maxsize=4)
def _entries(placement):
    with open(placement) as f:
        p = json.load(f)
    by_name = {e["name"]: e for e in p["entries"]}
    by_source = {}
    for e in p["entries"]:
        by_source.setdefault(e["source"], []).append(e)
    return p["tile_bytes"], by_name, by_source


def held(name, placement=loader.PLACEMENT, cache=loader.CACHE):
    """Weight `name` (e.g. "layers.3.attn.wq_b@bfp8_b") as float32 in its held shape:
    its cache bytes decoded as the chip reads them (komorebi.formats)."""
    tile_bytes, by_name, _ = _entries(placement)
    e = by_name[name]
    tile = tile_bytes[e["dtype"]]
    with open(convert.arena_path(cache, e["island"], e["dtype"]), "rb") as f:
        f.seek(e["first"] * tile)
        raw = np.frombuffer(f.read(e["pages"] * tile), np.uint8)
    shape = formats.padded_shape(e["shape"], "tile")
    values = formats.from_bytes(raw, shape, e["dtype"], "tile")
    return np.ascontiguousarray(values[tuple(slice(0, n) for n in e["shape"])])


def _unheld(values, conv):
    """The checkpoint's logical layout from a held array (convert.held's inverse)."""
    if conv.startswith("T"):
        return values.T
    if conv == "groups T":
        groups, inner, out = values.shape
        return values.transpose(0, 2, 1).reshape(groups * out, inner)
    if conv == "row":
        return values.reshape(-1)
    if conv == "=":
        return values
    raise ValueError(conv)


def as_released(source, dtype=None, placement=loader.PLACEMENT, cache=loader.CACHE):
    """Checkpoint tensor `source` (e.g. "layers.3.attn.wq_b.weight", "head.weight")
    as float32 in the checkpoint's logical shape, with the values the chip holds."""
    _, _, by_source = _entries(placement)
    entries = by_source[source]
    if dtype is None:
        dtype = next(d for d in COMPACT if any(e["dtype"] == d for e in entries))
    chosen = sorted(
        (e for e in entries if e["dtype"] == dtype), key=lambda e: e["name"]
    )
    if not chosen:
        raise KeyError(f"{source} is not held as {dtype}")
    parts = [_unheld(held(e["name"], placement, cache), e["conv"]) for e in chosen]
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)


def select(layers=None, kinds=None, formats=None, placement=loader.PLACEMENT):
    """Weight names by layer (ints; "head" and "norm" for the global ones), kind
    (substrings of the name after "layers.N.", e.g. "attn.", "ffn.shared_experts",
    "ffn.experts.7.", "hc_"), and chip format; None takes every one."""
    _, by_name, _ = _entries(placement)
    out = []
    for name, e in by_name.items():
        m = re.match(r"layers\.(\d+)\.(.*)", name)
        layer, rest = (
            (int(m.group(1)), m.group(2))
            if m
            else (name.split("@")[0].split(".")[0], name)
        )
        if layers is not None and layer not in layers:
            continue
        if kinds is not None and not any(k in rest for k in kinds):
            continue
        if formats is not None and e["dtype"] not in formats:
            continue
        out.append(name)
    return out
