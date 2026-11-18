"""The dispatcher's link line and link area for commands across islands (x280 link.h):
every table sized by the session's islands, none by a constant of the cards."""

import struct

from komorebi import abi
from komorebi.errors import KomorebiError

CROSSINGS = 8  # KMD_CROSSINGS: crossings in flight a dispatcher starts, at most
SLOTS_MOST = 16  # packets an inbox part: two a crossing in flight
SLOTS_LEAST = 4  # fewer leave one crossing in flight
INBOX_BYTES = 64 * 1024  # what the inbox parts take before a part gets fewer slots


def _lines(size):
    return (size + 63) // 64 * 64


def slots(islands):
    """Packets an inbox part holds with `islands` islands: 16 while the parts take at
    most INBOX_BYTES, fewer beyond, never below 4."""
    h = abi.headers()
    part = INBOX_BYTES // max(islands - 1, 1) - h["KMD_LINK_COUNT_BYTES"]
    return max(SLOTS_LEAST, min(SLOTS_MOST, part // h["KM_LINK_PACKET_BYTES"]))


def layout(islands):
    """The link area's parts with `islands` islands: in LIM (the area) the island
    entries and the inbox parts (`inbox`, `bytes`); in the dispatcher's cached state
    tail the entries copied, the counts and the crossings' bits (`counts`, `pending`,
    `tables` its bytes), the dispatcher's alone."""
    h = abi.headers()
    words = (islands + 31) // 32
    entries = _lines(islands * h["KMD_LINK_ENTRY_BYTES"])
    pending = entries + _lines(8 * islands)
    count = slots(islands)
    part = h["KMD_LINK_COUNT_BYTES"] + count * h["KM_LINK_PACKET_BYTES"]
    return {
        "counts": entries,
        "pending": pending,
        "tables": _lines(pending + 4 * CROSSINGS * words),
        "inbox": entries,
        "slots": count,
        "bytes": _lines(entries + (islands - 1) * part),
    }


def line(graph, island, core, area, tiles, lim_base):
    """Island `island`'s link line, then its link area: link core 0 (`core`, its index
    in the island's cores) and its link area's L1 address `area`; `tiles` each
    island's dispatcher tile (x, y); `lim_base` the NoC address of every LIM."""
    h = abi.headers()
    islands = len(graph.islands)
    if islands + h["KM_EVENT_ISLAND"] > 0x100:  # a TIMEOUT names an island in a byte
        raise KomorebiError(
            f"commands across {islands} islands: a give-up names an island in "
            f"{0x100 - h['KM_EVENT_ISLAND']} values"
        )
    if len(tiles) != islands:
        raise KomorebiError(
            f"commands across: {len(tiles)} dispatchers, {islands} islands"
        )
    parts = layout(islands)
    if parts["tables"] > h["KMD_STATE_TAIL"]:
        raise KomorebiError(
            f"commands across {islands} islands: the dispatcher's tables take "
            f"{parts['tables']} bytes, its state tail {h['KMD_STATE_TAIL']}"
        )
    words = [0] * (h["KMD_LINK_BYTES"] // 4)
    words[h["KMD_L_MARK"]] = h["KMD_LINK_MARK"]
    words[h["KMD_L_ISLAND"]] = island
    words[h["KMD_L_ISLANDS"]] = islands
    words[h["KMD_L_CORE"]] = core
    words[h["KMD_L_AREA"]] = area
    words[h["KMD_L_LIM"]] = lim_base  # KMD_L_BLOCK and KMD_L_BYTES: the host module's
    words[h["KMD_L_COUNTS"]] = parts["counts"]
    words[h["KMD_L_PENDING"]] = parts["pending"]
    words[h["KMD_L_INBOX"]] = parts["inbox"]
    words[h["KMD_L_SLOTS"]] = parts["slots"]
    words[h["KMD_L_LINKS"]] = len(graph.link_cores[island])
    block = [0] * (parts["bytes"] // 4)
    entry = h["KMD_LINK_ENTRY_BYTES"] // 4
    for one, (x, y) in enumerate(tiles):
        threads = graph.thread_numbers(one)
        block[entry * one] = graph.first(one) | graph.count(one) << 16
        block[entry * one + 1] = threads.start | len(threads) << 16
        block[entry * one + 2] = x | y << 16
    return struct.pack(f"<{len(words) + len(block)}I", *words, *block)
