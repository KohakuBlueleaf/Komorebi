"""The card tests share one device open for the whole run (shared_card.py): the card
is opened by the first test that needs it and closed when the run ends. Mesh tests
open cards as a mesh, which a process that opened a card alone may not: they run only
in a run of mesh tests alone."""

import pytest
import shared_card

ALONE = (
    "opens cards as a mesh: run alone, KMR_CARD=1 KMR_MESH=1 "
    "python -m pytest tests/hardware/mesh/test_two_cards.py"
)


def pytest_collection_modifyitems(config, items):
    """Skips the mesh tests in a run that also has other tests, and those whose cards
    are not visible."""
    mixed = any("mesh" not in item.keywords for item in items)
    for item in items:
        mesh = item.get_closest_marker("mesh")
        if mesh is None:
            continue
        cards = mesh.kwargs.get("cards", (0, 1))
        if mixed:
            item.add_marker(pytest.mark.skip(reason=ALONE))
        elif not shared_card.visible(cards):
            listed = ",".join(map(str, cards))
            item.add_marker(pytest.mark.skip(reason=f"cards {listed} not visible"))


@pytest.fixture(scope="module")
def session():
    """The shared session, open; a run the module left loaded is aborted at its end."""
    yield shared_card.host()
    shared_card.release()


def pytest_sessionfinish(session, exitstatus):
    """Closes the shared device after the last test."""
    shared_card.close_host()
