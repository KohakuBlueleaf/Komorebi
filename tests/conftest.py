"""Test setup: `card` tests run only with KMR_CARD=1, on a host with a card; `sizes`
tests only with KMR_SIZES=1, on a host with a built tt-metal tree (no card)."""

import os

import pytest

from komorebi.device import remembered


@pytest.fixture(autouse=True)
def no_inherited_git_environment(monkeypatch):
    """Drops GIT_DIR and the other GIT_* variables a hook or rebase exports."""
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def remembered_apart(tmp_path_factory, monkeypatch):
    """The remembered kernel buffers (komorebi.device.remembered) in a folder of the
    test's own: no test reads or writes the build cache's file."""
    folder = tmp_path_factory.mktemp("remembered")
    monkeypatch.setattr(remembered, "_path", lambda: folder / remembered.FILE)


def pytest_collection_modifyitems(config, items):
    """Skips card tests unless KMR_CARD=1, and sizes tests unless KMR_SIZES=1."""
    gates = (
        ("card", "KMR_CARD", "needs a Tenstorrent card (KMR_CARD=1)"),
        ("sizes", "KMR_SIZES", "builds kernels on a mock device (KMR_SIZES=1)"),
    )
    for keyword, variable, reason in gates:
        if os.environ.get(variable) == "1":
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if keyword in item.keywords:
                item.add_marker(skip)
