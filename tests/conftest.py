"""Test setup: `card` tests run only with KMR_CARD=1, on a host with a card."""

import os

import pytest


@pytest.fixture(autouse=True)
def no_inherited_git_environment(monkeypatch):
    """Drops GIT_DIR and the other GIT_* variables a hook or rebase exports."""
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)


def pytest_collection_modifyitems(config, items):
    """Skips card tests unless KMR_CARD=1."""
    if os.environ.get("KMR_CARD") == "1":
        return
    skip = pytest.mark.skip(reason="needs a Tenstorrent card (KMR_CARD=1)")
    for item in items:
        if "card" in item.keywords:
            item.add_marker(skip)
