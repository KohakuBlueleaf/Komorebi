"""A core's place as a handler with a rectangle given at each submit names it in C++:
the flat rank (Core::me()) on one island; on islands its place in the islands stacked
one below the other (km::rect_whole, the grid words' island table), where a rectangle
counts from the island of the command's first rank and runs on to the next."""


def grid(me, islands):
    """The C++ that reads the grid's words (km::rect_grid) from `me`'s (a Core or
    Compute's name) program arguments; on islands also their island table (kGridIslands
    words after the grid's: km::rect_islands). One island's text unchanged."""
    text = (
        "uint32_t km_grid_words[kGridWords];\n"
        "for (uint32_t i = 0; i < kGridWords; i++) {\n"
        f"    km_grid_words[i] = {me}.arg(kArgGrid + i);\n"
        "}\n"
        "const km::RectGrid km_grid = km::rect_grid(km_grid_words);"
    )
    if islands[0] <= 1:
        return text
    return (
        f"{text}\nconst km::RectIslands km_islands = km::rect_islands("
        "km_grid_words + kGridWords - kGridIslands, kGridIslands);"
    )


def place(me, islands):
    """The C++ of `me`'s place in the grid for a class built for `islands` (compiler
    Compiled.islands: count, most ranks): one island's text unchanged (its kernels
    byte-equal); on islands its core in the stack (km::rect_whole)."""
    if islands[0] <= 1:
        return f"{me}.me()"
    return f"km::rect_whole({me}.me(), km_grid.width, km_islands)"


def broadcast(args, me, islands):
    """The start of the call that fills a broadcast's arguments `args` on a rectangle
    given at each submit, up to the rectangle: km::rect_broadcast on one island (its
    text unchanged); on islands km::rect_broadcast_islands, a part an island fed
    across (the next part's first flat rank, the feeder's)."""
    if islands[0] <= 1:
        return f"km::rect_broadcast({args}, {me}.me(), km_grid, "
    return f"km::rect_broadcast_islands({args}, {me}.me(), km_grid, km_islands, "
