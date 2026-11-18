"""DST in the generated compute code: its half and its whole in fp32 tiles, tiles
packed from it, a page added into it, and the room pack holds for a field of sums."""

DST_HALF = 4  # fp32 tiles in DST's half a tile function's calls take (half sync)
DST_FULL = 8  # fp32 tiles in all of DST (full sync: sums wider than a half)


def pack(dst, half, width=1):
    """DST tiles `dst` .. `dst + width - 1` packed into pages of `half`; DST
    released."""
    packs = "".join(f"    pack_tile({dst + t}, {half});\n" for t in range(width))
    return (
        "tile_regs_commit();\n"
        "tile_regs_wait();\n"
        f"if (km_ok && compute.reserve_back({half}, {width})) {{\n"
        f"    KM_PACK_FORMAT({half});\n"
        f"{packs}"
        f"    cb_push_back({half}, {width});\n"
        "}\n"
        "tile_regs_release();"
    )


def add_tile(buffer, page, dst, scratch):
    """DST dst += page `page` of buffer (copied to DST scratch, added by the SFPU)."""
    return (
        f"KM_UNPACK_FORMAT_A({buffer});\n"
        f"copy_init({buffer});\n"
        f"copy_tile({buffer}, {page}, {scratch});\n"
        "add_binary_tile_init();\n"
        f"add_binary_tile({dst}, {scratch}, {dst});"
    )


def max_tile(buffer, page, dst, scratch):
    """DST dst = max(dst, page `page` of buffer) (copied to DST scratch, the SFPU's
    elementwise max)."""
    return (
        f"KM_UNPACK_FORMAT_A({buffer});\n"
        f"copy_init({buffer});\n"
        f"copy_tile({buffer}, {page}, {scratch});\n"
        "binary_max_tile_init();\n"
        f"binary_max_tile({dst}, {scratch}, {dst});"
    )


def sums_room(field_):
    """The name of pack's own word that it holds a field of sums' pages: only pack's
    reserve decides it (unpack's and math's is always true), so it guards the pack
    lines alone and never a branch all three compute threads take."""
    return f"km_room_sums_{field_}"


def sums_reserve(sums, room, buffer):
    """A new sum holds the field's pages: pack's reserve sets the room, and a give-up
    leaves it false for the command, so the three threads still run the sections
    and the destination's hand-over together while pack packs nothing."""
    return (
        f"if (km_ok && {sums} == 0) {{  // a new sum: the field's pages held\n"
        f"    {room} = {room} && compute.reserve_back({buffer}, {buffer}Pages);\n"
        "}\n"
    )


def sums_room_line(field_):
    """A field of sums' room, declared beside its count of sums."""
    return f"bool {sums_room(field_)} = true;  // pack holds self.{field_}'s pages"
