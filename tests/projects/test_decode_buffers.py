"""Decode's buffers on a chip of 32 buffer indices: the second W format's buffer
(split.h kCbSwSecondFormat, 33) is refused by its name (no card)."""

from types import SimpleNamespace

import pytest

from komorebi.errors import ProgramError
from komorebi.session.backend import backend as card_backend
from projects.decode import card
from projects.decode import model as md


def backend(indices):
    """A backend whose chip has `indices` buffer indices and no runtime buffers."""
    device = SimpleNamespace(buffer_indices=lambda: indices)
    return SimpleNamespace(card=SimpleNamespace(device=device), _runtime_cbs=list)


def test_the_second_w_format_s_buffer_is_refused_by_its_name_on_wormhole():
    plan = next(iter(md.plans(md.MODELS["qwen3-1.7b"]).values()))
    buffers = card.buffers(plan, 32, weights="official")
    assert max(b.index for b in buffers) == 33
    assert card_backend._Backend._cbs(backend(64), buffers)
    match = r"kCbSwSecondFormat \(split.h: the second W's format\): buffer index 33"
    with pytest.raises(ProgramError, match=match):
        card_backend._Backend._cbs(backend(32), buffers)
