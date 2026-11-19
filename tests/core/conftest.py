"""tests/core: pytest puts this folder on sys.path when it loads this file, so the
shared fakes (fake_port, fake_session, fake_tree) import by name in every folder."""

import pytest

from komorebi.session import process


@pytest.fixture
def blackhole_card(monkeypatch):
    """A card without its driver's files (no card here, as on a hosted machine) read as
    a P150, Blackhole, as the fake session's: for tests that ask the arch of card 0."""
    monkeypatch.setattr(process, "card_arch", lambda card: "blackhole")
