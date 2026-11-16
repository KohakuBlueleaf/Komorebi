"""The dispatcher's link area for commands across islands: sized by the session's
islands, every part on a line, the island entries in island order."""

import struct

import pytest

from komorebi import abi, islands, link_area
from komorebi.errors import KomorebiError


def graph_of(count, cores=6, threads=3):
    return islands.ring([islands.Island(tuple(range(cores)), cores, threads)] * count)


@pytest.mark.parametrize("count", [2, 4, 16, 32, 128])
def test_the_area_follows_the_islands(count):
    h = abi.headers()
    parts = link_area.layout(count)
    assert all(
        parts[name] % 64 == 0
        for name in ("counts", "pending", "tables", "inbox", "bytes")
    )
    assert parts["counts"] >= count * h["KMD_LINK_ENTRY_BYTES"]
    assert parts["pending"] >= parts["counts"] + 8 * count
    assert parts["tables"] >= parts["pending"] + 4 * link_area.CROSSINGS * -(
        -count // 32
    )
    assert parts["tables"] <= h["KMD_STATE_TAIL"]
    assert parts["inbox"] >= count * h["KMD_LINK_ENTRY_BYTES"]
    part = h["KMD_LINK_COUNT_BYTES"] + parts["slots"] * h["KM_LINK_PACKET_BYTES"]
    assert parts["bytes"] >= parts["inbox"] + (count - 1) * part
    assert parts["slots"] == (16 if count <= 33 else 4)


def test_the_line_and_the_entries():
    graph = graph_of(32)
    h = abi.headers()
    tiles = [(i % 8, i // 8) for i in range(32)]
    data = link_area.line(graph, 5, 4, 0x20000, tiles, 0x30000000)
    words = struct.unpack_from("<16I", data)
    assert words[h["KMD_L_MARK"]] == h["KMD_LINK_MARK"]
    assert (words[h["KMD_L_ISLAND"]], words[h["KMD_L_ISLANDS"]]) == (5, 32)
    assert words[h["KMD_L_SLOTS"]] == 16 and words[h["KMD_L_LIM"]] == 0x30000000
    assert len(data) == h["KMD_LINK_BYTES"] + link_area.layout(32)["bytes"]
    entries = struct.unpack_from("<128I", data, h["KMD_LINK_BYTES"])
    for island in (0, 17, 31):
        ranks, threads, tile = entries[4 * island : 4 * island + 3]
        assert ranks == graph.first(island) | graph.count(island) << 16
        assert threads >> 16 == 3 and tile == tiles[island][0] | tiles[island][1] << 16


def test_a_give_up_names_an_island_in_a_byte():
    graph = graph_of(129, cores=4, threads=0)
    with pytest.raises(KomorebiError, match="129 islands"):
        link_area.line(graph, 0, 2, 0x20000, [(0, 0)] * 129, 0)
