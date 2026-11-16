"""Names in the generated C++: a tensor parameter's type name, and the names of the
handler (its scope, its words, the runtime's values) a piece of C++ or a range reads."""

import re

# The runtime's values a handler's lines may read (the rank, the rectangle's place).
GENERATED_VALUES = (
    "km_rank",
    "km_size",
    "km_group",
    "km_self_x",
    "km_self_y",
    "km_self_core",
    "km_rect_columns",
    "km_rect_rows",
)


def camel(name):
    """The C++ type name of a tensor parameter (source: Source)."""
    return "".join(part.capitalize() for part in name.split("_"))


def text_uses(text, compiler):
    """The names of the handler a piece of generated C++ reads."""
    found = set()
    for name in re.findall(r"[A-Za-z_]\w*", text):
        if name in compiler.scope or name in GENERATED_VALUES:
            found.add(name)
        if name in compiler.counts.values():
            found.add(name)
        if name in getattr(compiler, "column_words", {}).values():  # a width word
            found.add(name)
    return found


def range_uses(pages, compiler, tensor=True):
    """The names a range's C++ reads (`tensor`: its tensor's word too)."""
    found = {pages.tensor} if pages.tensor and tensor else set()
    if pages.rect is not None:  # a rectangle's rows and their step
        found |= text_uses(f"{pages.rect[0]} {pages.rect[2]}", compiler)
    return found | text_uses(pages.first, compiler) | text_uses(pages.count, compiler)


def pattern_uses(pattern):
    return {pattern} if pattern else set()
