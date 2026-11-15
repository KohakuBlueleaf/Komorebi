"""The room a program's code gets: the kernel buffer that follows the build (kept and
refused when given), the tensors made again on a reopened card, the service overlay's
own slot, slots lowered to what the L1 left allows. No card and no compiler."""

import gc

import numpy as np
import pytest

from komorebi import KomorebiError, kernel_buffer, overlay, session
from komorebi.errors import KernelSizeError, L1BudgetError
from komorebi.program import KernelSizes
from komorebi.tensor import Tensor


def test_the_buffer_is_the_build_and_the_margin_aligned():
    assert kernel_buffer.need(85968) == 87040
    assert kernel_buffer.need(64 * 1000 - 1024) == 64 * 1000


def opened(monkeypatch, buffer=70656, **options):
    made = session.Session(**options)
    monkeypatch.setattr(made, "l1", lambda: (buffer, 1461120))
    return made


def test_a_session_without_a_size_follows_its_kernels(monkeypatch):
    made = opened(monkeypatch)
    with pytest.raises(kernel_buffer.Resize) as larger:
        made._resize(92304)
    assert larger.value.size == 93376 and "92,304" in larger.value.why
    assert made._resize(60000) is None  # it fits: the card is not reopened
    with pytest.raises(kernel_buffer.Resize) as smaller:
        made._resize(60000, l1_failed=True)
    assert smaller.value.size == 61056


def test_a_given_size_is_kept_and_refused_with_both_numbers(monkeypatch):
    made = opened(monkeypatch, kernel_bytes=86016)
    refused = made._resize(92304)
    assert refused == (
        "the session was given kernel_bytes; Session(kernel_bytes=93376) holds them"
    )
    sizes = KernelSizes({"BRISC": 28288, "NCRISC": 25640}, 92304, 86016)
    error = KernelSizeError("linear", sizes, session.KERNEL_BUFFER_MAX, refused)
    assert "92304 B of the 86016 B kernel buffer" in str(error)
    assert "Session(kernel_bytes=93376)" in str(error)
    error = L1BudgetError("linear", "  table", refused="no smaller one")
    assert "a smaller kernel buffer (no smaller one)" in str(error)


def test_a_session_that_did_not_open_its_card_does_not_follow(monkeypatch):
    made = opened(monkeypatch)
    made._own_device = False
    assert made._resize(92304).startswith("it did not open its card")
    made = opened(monkeypatch)
    made._kept.stop()  # a run was loaded
    assert made._resize(92304).startswith("a run was loaded")


class Buffer:
    """A native buffer of the fake allocator: the lowest free address that fits."""

    def __init__(self, heap, size):
        self.heap, self.size, self.page_size = heap, size, 64
        self.address = heap.take(size)

    def __del__(self):
        self.heap.taken.pop(self.address, None)


class Heap:
    def __init__(self):
        self.taken = {}

    def take(self, size):
        at = 0
        for start, length in sorted(self.taken.items()):
            if at + size <= start:
                break
            at = max(at, start + length)
        self.taken[at] = size
        return at


class Module:
    def __init__(self):
        self.heap = Heap()

    def Buffer(self, device, size, page, shard_grid, address):
        return Buffer(self.heap, size)


def make(module, kept, words):
    tensor = Tensor(module.Buffer(None, 4 * words, 64, None, None), (1, words),
                    "uint32", "row")  # fmt: skip
    kept.keep(tensor)
    return tensor


def test_the_tensors_come_back_at_their_addresses():
    first, kept = Module(), kernel_buffer.Kept()
    a = make(first, kept, 64)
    b = make(first, kept, 128)
    del b
    gc.collect()
    c = make(first, kept, 32)  # in b's place
    assert [kind for kind, _, _ in kept.history] == ["make", "make", "free", "make"]
    reopened = type("Reopened", (), {"_module": Module(), "device": None})()
    kept.remake(reopened)
    assert (a.address, c.address) == (0, 256)
    assert a.buffer.heap is reopened._module.heap
    assert kept.refused() is None


def test_a_tensor_at_another_address_is_refused():
    first, kept = Module(), kernel_buffer.Kept()
    a = make(first, kept, 64)
    reopened = type("Reopened", (), {"_module": Module(), "device": None})()
    reopened._module.heap.taken[0] = 64  # something else took the bottom
    with pytest.raises(KomorebiError, match="came back at 0x40"):
        kept.remake(reopened)
    assert a.address == 0


def built(service=9088, reader=(2, 1024), compute=(1, 4096)):
    slots = {"reader": reader, "unpack": compute, "math": compute, "pack": compute}
    images = {(1, "reader"): bytes(1024), (63, "reader"): bytes(9088)}
    return overlay.Built(images, slots, service)


def test_the_service_has_its_own_slot_zero():
    c = overlay.constants()
    made = built()
    assert made.slot_bytes == 9088 + 2 * 1024 + 3 * 4096
    rows, placed = overlay.dram_rows(made)
    words = overlay.area_words(made, placed, 0x1000, 9088, 0x20000)
    assert words[c["kOverlayServiceSlot"] // 4] == 9088
    assert words[c["kOverlaySlots"]] == 3 and words[c["kOverlaySlotBytes"]] == 1024
    record = c["kOverlayRecords"] // 4 + c["kOverlayAddress"]
    addresses = [words[record + 4 * slot] for slot in range(3)]
    assert addresses == [0x20000, 0x20000 + 9088, 0x20000 + 9088 + 1024]
    unpack = record + 4 * c["kOverlaySlotsMax"] * 2
    assert words[unpack] == 0x20000 + 9088 + 2048
    assert session.slot_rows(made.slots, made.service)[0] == (
        "slots: service (reader), 1 x 9,088 B",
        9088,
    )


def test_the_reader_has_one_slot_fewer_beside_the_service():
    assert overlay.service_beside(4, None) == (3, None)
    assert overlay.service_beside(2, 512) == (2, 512)


def test_slots_not_asked_are_lowered_to_what_the_l1_left_allows():
    made = built(reader=(3, 1024), compute=(2, 4096))
    room = 9088 + 3 * 1024 + 3 * 4096 + 1000
    fitted = overlay.fitted(made, overlay.Slots(), room)
    assert fitted.slots["reader"] == (3, 1024)  # the largest parts go first
    assert {fitted.slots[p][0] for p in ("unpack", "math", "pack")} == {1}
    asked = overlay.fitted(made, overlay.Slots(count={"compute": 2}), room)
    assert asked.slots["reader"] == (1, 1024) and asked.slots["math"] == (2, 4096)
    assert overlay.fitted(made, overlay.Slots(), 0).slots["reader"][0] == 1


class Writes:
    """What a reopened session writes: (tensor, pages, first)."""

    def __init__(self):
        self.done = []

    def write(self, tensor, pages, first=0):
        self.done.append((tensor, pages, first))


def test_what_was_written_is_kept_by_reference_and_written_again():
    kept, given = kernel_buffer.Kept(), np.arange(64, dtype=np.uint32).reshape(1, 64)
    tensor = make(Module(), kept, 64)
    tensor.kept.append(("pages", given.view(np.uint8), 0))
    assert np.shares_memory(tensor.kept[0][1], given)  # the array given, no copy
    reopened = Writes()
    assert kernel_buffer.rewrite(reopened, tensor) == 256
    ((_, pages, first),) = reopened.done
    assert first == 0 and np.array_equal(pages, given.view(np.uint8))
    assert len(tensor.kept) == 1  # not kept twice
    kept.stop()
    assert tensor.kept is None and kept.history is None
