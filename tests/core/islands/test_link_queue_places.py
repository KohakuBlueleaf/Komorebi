"""The link cores' put queue sits at one place on every island: the overlay slots are
on the ranks alone when the link cores run the runtime's link kernels, and islands
whose queues would differ are refused (no card)."""

from types import SimpleNamespace

import pytest

from komorebi.errors import KomorebiError
from komorebi.islands.session import check_link_queues
from komorebi.program import several as programs
from komorebi.session.backend import backend as card_backend


def slot_cores(apart=0, link_kernels=0):
    backend = SimpleNamespace(
        port=SimpleNamespace(grid_x=10, grid_y=12),
        apart_links=apart,
        link_kernels=link_kernels,
    )
    return card_backend._Backend._slot_cores(backend)


def test_the_slots_leave_out_the_link_kernels_cores():
    assert slot_cores(link_kernels=2) == programs.rectangles(0, 118, 10)


def test_the_slots_cover_the_grid_without_link_kernels():
    assert slot_cores() == [(0, 0, 9, 11)]
    assert slot_cores(apart=2) == programs.rectangles(0, 118, 10)


def test_equal_queues_and_no_queue_pass():
    check_link_queues([(0, 0), (0, 0)])
    check_link_queues([None, None])
    check_link_queues([(4096, 0), None])


def test_queues_at_different_places_are_refused_naming_both():
    with pytest.raises(KomorebiError, match=r"\(103616, 0\), \(103872, 0\)"):
        check_link_queues([(103872, 0), (103616, 0)])
