"""The link cores' put queue (KM_LINK_CB) is a row of the load's L1 budget when they
also hold the program's buffers and slots (a split card's link cores run its kernels):
their buffers' region ends past the slots by it, so the slots are fitted around it and
a load that does not fit is refused by name (no card: a recording launch)."""

from types import SimpleNamespace

import pytest

from komorebi import abi
from komorebi.errors import ProgramError
from komorebi.program import overlay
from komorebi.session import kernels as kernel_module
from komorebi.session import sizes
from komorebi.session.backend import backend as card_backend

ROW = "link cores: their put queue (KM_LINK_CB)"


class Launch:
    def __init__(self):
        self.cbs = []

    def cb(self, cores, index, fmt, page, total, first=False):
        self.cbs.append((cores, index, total))


def queued(chunk=0, indices=64, arch="blackhole", taken=()):
    """A split card's load (a line of 3) after its put queue: backend, launch, the
    defines; the chip's `indices` buffer indices (Blackhole's 64), `taken` in use."""
    backend = SimpleNamespace(cb_indices=set(), split=None, link_kernels=0,
                              apart_links=0, link_cb=None, link_bytes=0)  # fmt: skip
    backend.cb_indices = set(taken)
    device = SimpleNamespace(buffer_indices=lambda: indices, arch=lambda: arch)
    backend.card = SimpleNamespace(device=device)
    backend.session = SimpleNamespace(island=None)
    rects = [(11, 3, 11, 3), (10, 6, 11, 6), (11, 9, 11, 9)]  # acc's line of 3
    backend._link_core_rects = lambda grid: rects
    launch = Launch()
    program = SimpleNamespace(name="layer", defines={"KM_PAGE_LINKS": 1})
    defines = card_backend._Backend._link_queue(
        backend, launch, program, (12, 10), chunk
    )
    return backend, launch, defines


def test_the_put_queue_on_link_cores_that_hold_the_program_is_a_budget_row():
    backend, launch, defines = queued()
    ((cores, index, total),) = launch.cbs
    assert index == abi.headers()["KM_LINK_CB"] and len(cores) == 3
    assert defines == [("KM_LINK_QUEUE", "1")]  # km_abi.h's index: no define more
    assert backend.link_bytes == total
    assert sizes.link_rows(backend) == [(ROW, total)]


def test_the_put_queue_past_a_chip_s_indices_is_refused_by_name():
    with pytest.raises(
        ProgramError, match=r"put queue \(KM_LINK_CB\): buffer index 33"
    ):
        queued(indices=32)  # Blackhole's index on a chip of 32


def test_a_wormhole_chip_s_put_queue_takes_the_highest_free_index_and_says_so():
    """Below the runtime's 29 (the chain's 28 taken: 27), the kernels told which."""
    backend, launch, defines = queued(indices=32, arch="wormhole_b0")
    ((_, index, _),) = launch.cbs
    assert index == backend.link_cb == 28
    assert defines == [("KM_LINK_QUEUE", "1"), ("KM_LINK_QUEUE_CB", "28")]
    _, _, defines = queued(indices=32, arch="wormhole_b0", taken={0, 1, 28})
    assert defines[-1] == ("KM_LINK_QUEUE_CB", "27")
    with pytest.raises(ProgramError, match="no buffer index below 29 is free"):
        queued(indices=32, arch="wormhole_b0", taken=set(range(29)))


def test_link_cores_with_kernels_of_their_own_or_apart_add_no_row():
    backend, _, _ = queued()
    backend.link_kernels = 2  # the program's buffers on the ranks alone
    assert sizes.link_rows(backend) == []
    backend.link_kernels, backend.apart_links = 0, 3
    assert sizes.link_rows(backend) == []
    assert sizes.link_rows(SimpleNamespace(link_bytes=0)) == []  # one card


def test_the_decode_layer_s_table_on_a_line_of_3_does_not_fit_with_the_queue():
    """acc's table of 2026-11-17 (the layer launched, then tt-metal's raw throw: the
    buffers' region 111,232 B past the slots on the link cores)."""
    rows = [
        ("circular buffers", 848_768), ("control block and overlay area", 17_104),
        ("slots", 335_168), ("host transfer buffer", 32_848),
        ("page bounds list", 464), ("L1 held by the session", 72_128),
    ]  # fmt: skip
    table, fits = overlay.budget(1_307_200, rows)
    assert fits and "1,306,480 B of 1,307,200 B" in table
    table, fits = overlay.budget(1_307_200, rows + [(ROW, 111_232)])
    assert not fits and ROW in table and "1,417,712 B of 1,307,200 B" in table


def test_a_load_refused_with_the_put_rings_is_built_again_without_them():
    """The second build only when the chunks lie on link cores that hold the slots."""
    backend, _, _ = queued(chunk=192 << 10)
    backend.link_queue_at = (0, 192 << 10)
    assert kernel_module.rings_dropped(backend)
    backend.link_queue_at = (0, 0)  # no ring already: the refusal stands
    assert not kernel_module.rings_dropped(backend)
    backend.link_queue_at, backend.link_kernels = (0, 192 << 10), 2  # slots elsewhere
    assert not kernel_module.rings_dropped(backend)
    backend.link_kernels, backend.apart_links = 0, 3
    assert not kernel_module.rings_dropped(backend)
