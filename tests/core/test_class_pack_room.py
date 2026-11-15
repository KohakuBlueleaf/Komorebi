"""Compute's reserve is decided by pack alone (true on unpack and math), so in a class's
generated compute its result guards only the pack lines; the sections, their hand-over
and the products run on all three threads. A pack's give-up then ends with the others,
not with unpack and math held inside the unit (no card)."""

from types import SimpleNamespace

from chain_fixture import quiet, tensor
from test_class_reduce_arrival import Small, tiles

import komorebi as km
from komorebi.run import Run

PACKS = ("pack_tile", "pack_reconfig_l1_acc", "cb_push_back")
SHARED = ("tile_regs_", "matmul_block(", "copy_tile(")


class Grid(km.Program):
    """A field of sums from two broadcasts, written whole (no reduce)."""

    part: km.L1[km.fp32, 2, 2, 32, 32]

    @km.tile_function
    def project(x, w):
        return x @ w

    @km.handler
    def grid(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        along = km.Broadcast(self, along="row")
        down = km.Broadcast(self, along="column")
        top = self.y * 2
        left = self.x * 2
        for chunk in range(2):
            k0 = chunk * 2
            if along.sender():
                xs = along.cast(x.tiles[top : top + 2, k0 : k0 + 2].read())
            else:
                xs = along.arrive()
            if down.sender():
                ws = down.cast(w.tiles[k0 : k0 + 2, left : left + 2].read())
            else:
                ws = down.arrive()
            self.part += self.project(xs, ws)
        y.tiles[top : top + 2, left : left + 2].write(self.part)


def compute_text(cls, **tensors):
    program = cls(l1=1 << 20, log=quiet, **tensors)
    return (program.built.folder / "compute.cpp").read_text()


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


def test_a_stream_product_and_a_field_reduce_guard_only_the_packs():
    text = compute_text(
        Small,
        x=tiles(1, 30, 0x10000),
        w=tiles(30, 8, 0x20000),
        y=tiles(1, 8, 0x90000),
        s=tensor(120, 0xA0000),
        t=tensor(1, 0x120000),
    )
    seen = check(text)
    assert "bool km_room_sums_sums = true;" in text
    held = "km_room_sums_sums = km_room_sums_sums && compute.reserve_back(kFieldSums,"
    assert held in text
    assert "if (km_ok && km_room_sums_sums) {" in text  # the reduce's push
    assert any(line.startswith("pack_tile<true>") for line in seen)
    assert any(line.startswith("pack_reconfig_l1_acc") for line in seen)


def test_broadcast_products_and_a_field_written_whole_guard_only_the_packs():
    text = compute_text(
        Grid,
        x=tiles(16, 4, 0x10000),
        w=tiles(4, 16, 0x60000),
        y=tiles(16, 16, 0xA0000),
    )
    seen = check(text)
    assert "km_room_sums_part = km_room_sums_part && compute.reserve_back(" in text
    assert "if (km_room_sums_part) {  // pack's: its pages held" in text
    repack = [line for line, _ in blocks(text) if "compute.reserve_back(kOut" in line]
    assert len(repack) == 1 and repack[0].startswith("km_room")  # the repack's loop
    assert any("kOutRows" in line for line in seen)


def test_a_timeout_says_what_the_deadline_read_before_the_stop():
    core = {"core": 3, "live": {"math": {"where": "the destination", "sequence": 9}}}
    idle = {"core": 4, "live": {}}
    run = SimpleNamespace(_timeout_cores=[core, idle], _reset_failed=(-1, 0))
    nothing = SimpleNamespace(_timeout_cores=[], _reset_failed=(-1, 0))
    seen = Run._timeout_seen(run)
    assert seen == (
        "at the deadline: core 3 math waits in the destination; "
        "1 cores read at it named no wait"
    )
    assert Run._timeout_seen(nothing) == ""
