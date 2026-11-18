"""What test_class_pack_room.py and other tests share: PACKS, SHARED, blocks, check,
pack_only."""

PACKS = ("pack_tile", "pack_reconfig_l1_acc", "cb_push_back")
SHARED = ("tile_regs_", "matmul_block(", "copy_tile(")


def blocks(text):
    """Each line of C++ text with the heads of the blocks it is inside."""
    heads, out = [], []
    for line in text.splitlines():
        code = line.split("//")[0]
        out.append((line.strip(), list(heads)))
        for char in code:
            if char == "{":
                heads.append(code.strip())
            elif char == "}" and heads:
                heads.pop()
    return out


def pack_only(head):
    return "km_room" in head or "compute.reserve_back(" in head


def check(text):
    """Every pack line under a pack-only head; no section line under one; no shared
    flag set from a reserve. Returns the pack lines seen."""
    assert "km_ok = compute.reserve_back(" not in text
    assert "km_ok = km_ok && compute.reserve_back(" not in text
    seen = []
    for line, heads in blocks(text):
        if line.startswith(PACKS) and not line.startswith("cb_push_back(kHandover"):
            if line.startswith("cb_push_back") and "kBroadcast" in line:
                continue  # not compute's
            assert any(pack_only(head) for head in heads), line
            seen.append(line)
        if line.startswith(SHARED):
            assert not any(pack_only(head) for head in heads), (line, heads)
    return seen
