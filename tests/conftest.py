"""Test setup: `card` tests run only with KMR_CARD=1, on a host with a card."""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skips card tests unless KMR_CARD=1."""
    if os.environ.get("KMR_CARD") == "1":
        return
    skip = pytest.mark.skip(reason="needs a Tenstorrent card (KMR_CARD=1)")
    for item in items:
        if "card" in item.keywords:
            item.add_marker(skip)
