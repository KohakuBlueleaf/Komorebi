"""KM_GUARD's page bound on a card with no tensor table (km_extent.hpp): the session's
DRAM tensors as a sorted list in each core's L1, after its control block; a page call
past its tensor's last page is refused by name. Over islands the table's entry says."""

import weakref

import numpy as np

# The most tensors a list holds unless the session says (Session.page_bounds): 8 bytes
# of every core's L1 each.
MOST = 1024
LINE = 64  # a list entry counts a tensor's bytes in 64-byte lines


class Registry:
    """The DRAM tensors a session made and still holds, without keeping them alive."""

    def __init__(self):
        self._tensors = weakref.WeakValueDictionary()
        self._made = 0

    def add(self, tensor):
        if tensor.in_dram:
            self._tensors[self._made] = tensor
            self._made += 1

    def live(self):
        return list(self._tensors.values())


def listed(tensors, most):
    """(the list's words: count, then address and lines a tensor, by address, padded
    to 16 bytes; the tensors listed; those left unchecked past `most`)."""
    entries = {}
    for tensor in tensors:
        bytes_ = tensor.pages * tensor.aligned_page_bytes
        if tensor.aligned_page_bytes % LINE == 0 and bytes_ > 0 and tensor.address:
            entries[tensor.address] = (tensor, bytes_ // LINE)  # the last at an address
    ordered = sorted(entries.items())
    kept, left = ordered[:most], [tensor for _, (tensor, _) in ordered[most:]]
    words = [len(kept)]
    for address, (_, lines) in kept:
        words += [address, lines]
    words += [0] * (-len(words) % 4)
    return np.array(words, np.uint32), [tensor for _, (tensor, _) in kept], left


def unchecked_says(left, most):
    """The load's log line for tensors past the list (None: none)."""
    if not left:
        return None
    named = ", ".join(f"{t.shape} {t.dtype} at {t.address:#x}" for t in left[:4])
    more = f" and {len(left) - 4} more" if len(left) > 4 else ""
    return (
        f"page bounds: {len(left)} DRAM tensors past the list's {most} are not "
        f"checked: {named}{more} (Session.page_bounds = n lists more, 8 B of L1 each)"
    )
