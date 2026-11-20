"""The placement of the backbone's weights on the 31 islands: island, chip format and
pages of an arena for every weight (README.md, "Placement").

python placement.py CKPT_DIR OUT_DIR   (writes OUT_DIR/placement.json)
"""

import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ckpt  # noqa: E402

VERSION = 1
# island i is card CARDS[i] (chip 16 is out: a DRAM fault)
CARDS = [c for c in range(32) if c != 16]
ISLANDS = len(CARDS)
LAYERS = 40
EXPERTS = 384
# the islands holding 12 experts a layer (not 13), then two of the 13: layer L's home is
# HOMES[L % 21], so islands 12-30 hold two layers each (L and L + 21), 4 and 5 one each
HOMES = list(range(12, ISLANDS)) + [4, 5]
HEAD_ISLANDS = [0, 1, 2, 3]  # the head's 4 slices; the final norm with slice 0
ENGRAM_ISLANDS = {1: 6, 14: 7}  # the Engram modules' projections (layers 1 and 14)
TILE_BYTES = {"bfloat16": 2048, "float32": 4096, "bfp8_b": 1088, "bfp4_b": 576}
DRAM_BYTES = 12 * 2**30  # an island's DRAM (6 channels x 2 GiB)
# the put batches' landing Komorebi allocates on each island at 31 islands:
# n*n*(512 + 8) pages of 2,048 B (a formula, not measured)
LANDING_BYTES = ISLANDS * ISLANDS * (512 + 8) * 2048

SKIP = re.compile(
    r"^(vision\.|aligner\.|image_|mtp\.|embed\.weight$)"
    r"|engram\.embed\.|\.bias_vl$|\.scale$"
)


def home(layer):
    """The island holding layer `layer`'s weights other than its routed experts."""
    return HOMES[layer % len(HOMES)]


def expert_island(expert):
    return expert % ISLANDS


def chip_shape(name, stored_shape, stored):
    """The weight's shape on the chip: a product's right operand [in, out] (the
    checkpoint's [out, in] transposed); wo_a's 8 groups [8, 4096, 1024]; a vector
    [1, N]; the Engram q/k weights [4, 5120] as stored."""
    if len(stored_shape) == 1:
        return [1, stored_shape[0]]
    out, inner = stored_shape
    if stored == "I8":  # FP4 two a byte
        inner *= 2
    if name.endswith("attn.wo_a.weight"):
        return [8, inner, out // 8]
    if name.endswith(("engram.q_weight", "engram.k_weight")):
        return [out, inner]
    return [inner, out]


def conversion(name, stored_shape):
    """How the chip shape comes from the stored one: "T" transposed ([out, in] ->
    [in, out]); "groups T" each of wo_a's 8 row groups transposed; "row" a vector as
    one row; "=" as stored."""
    if len(stored_shape) == 1:
        return "row"
    if name.endswith("attn.wo_a.weight"):
        return "groups T"
    if name.endswith(("engram.q_weight", "engram.k_weight")):
        return "="
    return "T"


def tiles(shape):
    *outer, rows, cols = shape
    return math.prod(outer) * math.ceil(rows / 32) * math.ceil(cols / 32)


def formats_of(stored):
    """The chip formats a stored dtype is held in: the exact one first (none for FP4),
    then the compact one."""
    return {
        "I8": ["bfp4_b"],  # FP4 + E8M0/32: bfp4_b (lossy); bfp8_b would be exact
        "F8_E4M3": ["bfloat16", "bfp8_b"],  # bf16 exact; bfp8_b compact
        "BF16": ["bfloat16"],
        "F32": ["float32"],
    }[stored]


def short(name):
    return name[: -len(".weight")] if name.endswith(".weight") else name


def island_of(name):
    m = re.match(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.", name)
    if m:
        return expert_island(int(m.group(2)))
    m = re.match(r"layers\.(\d+)\.engram\.", name)
    if m:
        return ENGRAM_ISLANDS[int(m.group(1))]
    m = re.match(r"layers\.(\d+)\.", name)
    if m:
        return home(int(m.group(1)))
    if name == "norm.weight":
        return HEAD_ISLANDS[0]
    raise ValueError(name)


def order_key(name):
    """Within an arena: by layer, then the layer's other weights before its experts,
    the experts by number, w1 w3 w2."""
    m = re.match(r"layers\.(\d+)\.(.*)", name)
    if not m:
        return (LAYERS, 0, 0, 0, name)
    layer, rest = int(m.group(1)), m.group(2)
    e = re.match(r"ffn\.experts\.(\d+)\.(w\d)\.", rest)
    if e:
        return (layer, 1, int(e.group(1)), ["w1", "w3", "w2"].index(e.group(2)), "")
    return (layer, 0, 0, 0, rest)


def plan(root):
    c = ckpt.Checkpoint(root)
    names = sorted((n for n in c.where if not SKIP.search(n)), key=order_key)
    entries, arenas = [], {}

    def add(key, island, dtype, shape, source, conv):
        arena = arenas.setdefault((island, dtype), 0)
        pages = tiles(shape)
        entries.append(
            {
                "name": key,
                "island": island,
                "card": CARDS[island],
                "dtype": dtype,
                "shape": shape,
                "first": arena,
                "pages": pages,
                "source": source,
                "conv": conv,
            }
        )
        arenas[(island, dtype)] = arena + pages

    for name in names:
        if name == "head.weight":
            continue
        stored, shape = c.info(name)[:2]
        shape_chip = chip_shape(name, shape, stored)
        island = island_of(name)
        conv = conversion(name, shape)
        for dtype in formats_of(stored):
            add(f"{short(name)}@{dtype}", island, dtype, shape_chip, name, conv)
    # the head [5120, 129280]: 4 slices of 32,320 columns (1,010 tiles), one an island
    stored, (vocab, hidden) = c.info("head.weight")[:2]
    width = vocab // len(HEAD_ISLANDS)
    for j, island in enumerate(HEAD_ISLANDS):
        add(
            f"head.s{j}@bfloat16",
            island,
            "bfloat16",
            [hidden, width],
            "head.weight",
            f"T rows {j * width}:{(j + 1) * width}",
        )
    arena_list = [
        {
            "island": island,
            "card": CARDS[island],
            "dtype": dtype,
            "pages": pages,
            "bytes": pages * TILE_BYTES[dtype],
        }
        for (island, dtype), pages in sorted(arenas.items())
    ]
    return {
        "version": VERSION,
        "cards": CARDS,
        "tile_bytes": TILE_BYTES,
        "arenas": arena_list,
        "entries": entries,
    }


def main():
    root, out = sys.argv[1], sys.argv[2]
    p = plan(root)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "placement.json"), "w") as f:
        json.dump(p, f, separators=(",", ":"))
    per = {}
    for a in p["arenas"]:
        per.setdefault(a["island"], {})[a["dtype"]] = a["bytes"]
    print("island\tcard\tbfp4_b\tbfp8_b\tbfloat16\tfloat32\ttotal\tleft_after_landing")
    total = 0
    for island in range(ISLANDS):
        d = per.get(island, {})
        t = sum(d.values())
        total += t
        print(
            f"{island}\t{CARDS[island]}\t{d.get('bfp4_b', 0):,}\t"
            f"{d.get('bfp8_b', 0):,}\t{d.get('bfloat16', 0):,}\t"
            f"{d.get('float32', 0):,}\t{t:,}\t{DRAM_BYTES - LANDING_BYTES - t:,}"
        )
    print(f"entries {len(p['entries'])}, arenas {len(p['arenas'])}, bytes {total:,}")


if __name__ == "__main__":
    main()
