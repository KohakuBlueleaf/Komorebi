"""The card tests share one device open for the whole run (shared_card.py): the card
is opened by the first test that needs it and closed when the run ends. Two-card
behaviour is card_check's cards 0,1 process."""

import pytest
import shared_card


@pytest.fixture(scope="module")
def session():
    """The shared session, open; a run the module left loaded is aborted at its end."""
    yield shared_card.host()
    shared_card.release()


def pytest_sessionfinish(session, exitstatus):
    """Closes the shared device after the last test."""
    shared_card.close_host()
