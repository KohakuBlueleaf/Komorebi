"""A card opened alone in a process that opened a fabric mesh, and the mirror: refused
at open, before any device call (no card needed)."""

import pytest

import komorebi as km
from komorebi.device import native
from komorebi.errors import KomorebiError
from komorebi.islands import session as island_session
from komorebi.session import process


@pytest.fixture
def opened(monkeypatch):
    """A clean record of what this process opened; no device is ever opened."""
    record = {"mesh": [], "alone": []}
    monkeypatch.setattr(process, "_OPENED", record)
    monkeypatch.setattr(process, "chip_holder", lambda card, path=None: None)
    monkeypatch.delenv("TT_VISIBLE_DEVICES", raising=False)

    def no_device(*args, **kwargs):
        raise AssertionError("the refusal must come before the device is touched")

    monkeypatch.setattr(native, "load_module", no_device)
    return record


def test_alone_after_mesh_refused(opened):
    opened["mesh"].append((0, 1))
    with pytest.raises(KomorebiError, match="new process") as said:
        km.Session(card=0).open()
    assert "card 0: this process opened cards 0,1 as a fabric mesh" in str(said.value)
    assert opened["alone"] == []


def test_other_card_alone_after_mesh_refused(opened):
    opened["mesh"].append((0, 1))
    with pytest.raises(KomorebiError, match="new process"):
        km.Session(card=1).open()  # both cards' Ethernet cores were left inactive


def test_mesh_after_alone_refused(opened):
    opened["alone"].append((0,))
    with pytest.raises(KomorebiError, match="new process") as said:
        island_session.IslandSession(cards=[0, 1]).open()
    assert "cards 0,1 as a mesh: this process opened card 0 alone" in str(said.value)
    assert opened["mesh"] == []


def test_same_kind_again_allowed(opened):
    process.refuse_reopen("alone", [0])
    process.refuse_reopen("alone", [1])
    assert opened["alone"] == [(0,), (1,)]
    opened["alone"].clear()
    process.refuse_reopen("mesh", [0, 1])
    process.refuse_reopen("mesh", [0, 1])
    assert opened["mesh"] == [(0, 1), (0, 1)]


def test_alone_records_before_the_device(opened):
    with pytest.raises(AssertionError, match="before the device"):
        km.Session(card=0).open()  # not refused: it reaches the device (stubbed)
    assert opened["alone"] == [(0,)]
