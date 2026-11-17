"""A merged state's token (km.reduce(..., combine=) of a state written in place): the
MERGE body packs the state and the next arrival's run unpacks it, so the body names an
own field compute pushes between pack and unpack, or it is refused at its line."""

import ast
import re

from komorebi.dsl.compile.source import refuse
from komorebi.dsl.front.names import spelling


def tokened(handler, node, field_, merge):
    """`merge`, the MERGE body's text, once it names an own field other than `field_`
    whose push is compute's (its token); else refused at `node`: unpack would
    read self.<field_> before pack wrote the merge before, and so would the first later
    line that takes it (named)."""
    if any(
        push in ("found", "pack")
        and re.search(
            rf"\bkField{''.join(p.capitalize() for p in f.split('_'))}\b", merge
        )
        for f, push in handler.compiled.own.items()
        if f != field_
    ):
        return merge
    later = sorted(
        call.lineno
        for call in ast.walk(handler.definition)
        if isinstance(call, ast.Call)
        and call.lineno > node.lineno
        and any(
            isinstance(value, ast.Attribute) and value.attr == field_
            for keyword_ in call.keywords
            if keyword_.arg == spelling("takes")
            for value in ast.walk(keyword_.value)
        )
    )
    after = f", and line {later[0]} reads it after the merge" if later else ""
    refuse(
        handler.where,
        node,
        f"the {spelling('combine')} body packs self.{field_} and the next "
        f"arrival's run unpacks it, with nothing between pack and unpack{after}: "
        "the body names no own field compute pushes as a token",
        f'declare one (token: km.L1[km.bf16, 1, 32, 32] = km.{spelling("own")}'
        '(push="pack")), push it on pack after the packs of the body, wait for it '
        f"on unpack before reading self.{field_} (kFieldToken in the body)",
    )
