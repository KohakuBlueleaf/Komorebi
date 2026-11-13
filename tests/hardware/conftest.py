"""Card test files that open the card in the pytest process (`CARD_IN_PROCESS = True`)
and files that run a card script as a child never share one pytest run: the first kind
holds the chip's lock until the process exits, so the child would wait for the card
without a timeout. run_cards.py gives every file its own process (README.md)."""

import os

import pytest


def mixed_card_kinds(items):
    """Whether the card tests among `items` open the card here and also in a child."""
    kinds = {
        bool(getattr(item.module, "CARD_IN_PROCESS", False))
        for item in items
        if "card" in item.keywords
    }
    return len(kinds) > 1


def pytest_collection_modifyitems(config, items):
    """With KMR_CARD=1, rejects a run mixing the two kinds, before any card access."""
    if os.environ.get("KMR_CARD") == "1" and mixed_card_kinds(items):
        raise pytest.UsageError(
            "card tests that open the card in this process and card tests that run a "
            "child process cannot share one pytest run: run one file per process "
            "(tests/hardware/run_cards.py)"
        )
