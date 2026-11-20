"""The placement "home": every weight but the routed experts on island 0, the routed
experts on islands 1-30; made from the default placement.json, its cache kept
(README.md, "Home"). python home.py LLM_DIR   (writes LLM_DIR/placement-home.json)
"""

import json
import os
import re
import sys

NAME = "home"
HOME = 0  # the island of every weight but the routed experts
DRAM_BYTES = 12 * 2**30
# a weight held in two formats is placed once: bfp8_b, these in bfloat16 (as read)
EXACT = (".engram.wkv@", ".attn.indexer.wq_b@")
EXPERT = re.compile(r"layers\.\d+\.ffn\.experts\.(\d+)\.")
# the relay's pages 0 .. 34,559 on island 0 (the experts piece reads them at weight 0)
RELAY_SOURCE = [f"layers.0.ffn.experts.0.{m}@bfp4_b" for m in ("w1", "w3", "w2")]


def island_of(name, islands):
    """Routed expert e on island 1 + e mod (islands - 1); the rest on HOME."""
    m = EXPERT.match(name)
    return 1 + int(m.group(1)) % (islands - 1) if m else HOME


def wanted(name):
    return "bfloat16" if any(k in name for k in EXACT) else "bfp8_b"


def kept(entries):
    """The default's entries less the second copy of a weight held in two formats."""
    held = {}
    for e in entries:
        held.setdefault(e["name"].split("@")[0], set()).add(e["dtype"])
    return [
        e
        for e in entries
        if len(held[e["name"].split("@")[0]]) == 1 or e["dtype"] == wanted(e["name"])
    ]


def plan(default):
    """The home placement from the default one (a dict as placement.json): each entry
    its island and first page here, `cache` its (island, first) in the default's cache;
    `relay` the run of bfp4_b pages put first on island 0."""
    cards, tile_bytes = default["cards"], default["tile_bytes"]
    islands = len(cards)
    by_name = {e["name"]: e for e in default["entries"]}
    arenas, entries = {}, []

    def add(e, island):
        at = arenas.setdefault((island, e["dtype"]), 0)
        entries.append(
            {
                **e,
                "island": island,
                "card": cards[island],
                "first": at,
                "cache": [e["island"], e["first"]],
            }
        )
        arenas[(island, e["dtype"])] = at + e["pages"]

    src = [by_name[n] for n in RELAY_SOURCE]
    assert all(
        a["island"] == src[0]["island"] and a["first"] + a["pages"] == b["first"]
        for a, b in zip(src, src[1:])
    ), "the relay's source is not one run"
    pages = sum(e["pages"] for e in src)
    add(
        {
            "name": "relay@bfp4_b",
            "island": src[0]["island"],
            "dtype": "bfp4_b",
            "shape": [pages * 32, 32],
            "first": src[0]["first"],
            "pages": pages,
            "source": "relay",
            "conv": "=",
        },
        HOME,
    )
    relay = entries.pop()
    for e in kept(default["entries"]):
        add(e, island_of(e["name"], islands))
    arena_list = [
        {
            "island": island,
            "card": cards[island],
            "dtype": dtype,
            "pages": n,
            "bytes": n * tile_bytes[dtype],
        }
        for (island, dtype), n in sorted(arenas.items())
    ]
    return {
        "version": default["version"],
        "name": NAME,
        "fold": NAME,
        "cards": cards,
        "tile_bytes": tile_bytes,
        "arenas": arena_list,
        "entries": entries,
        "relay": [relay],
    }


def budget(p):
    """{island: bytes} of the placement's arenas."""
    out = {}
    for a in p["arenas"]:
        out[a["island"]] = out.get(a["island"], 0) + a["bytes"]
    return out


def main():
    llm = sys.argv[1]
    with open(os.path.join(llm, "placement.json")) as f:
        p = plan(json.load(f))
    with open(os.path.join(llm, f"placement-{NAME}.json"), "w") as f:
        json.dump(p, f, separators=(",", ":"))
    per = budget(p)
    print("island\tcard\tbytes\tleft")
    for island in sorted(per):
        left = DRAM_BYTES - per[island]
        print(f"{island}\t{p['cards'][island]}\t{per[island]:,}\t{left:,}")
    print(f"entries {len(p['entries'])}, arenas {len(p['arenas'])}")


if __name__ == "__main__":
    main()
