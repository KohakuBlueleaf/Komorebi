"""Test setup: `card` tests run only with KMR_CARD=1, on a host with a card; `mesh`
tests only with KMR_MESH=1 too, in a pytest process of their own. `tree` tests (a
tt-metal tree) are selected or deselected with -m, never skipped."""

import contextlib
import os

import pytest

from komorebi.device import remembered
from komorebi.session import kernels


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


@pytest.fixture(autouse=True)
def no_build_lock(request, monkeypatch):
    """A test that stubs the build takes no lock on the real build cache (it would
    wait behind other processes' builds and say so in the log); `tree` tests, which
    build real kernels, and `build_lock` tests keep the real lock."""
    if {"tree", "build_lock"} & set(request.keywords):
        return
    monkeypatch.setattr(
        kernels, "build_lock", lambda log=print: contextlib.nullcontext()
    )


def pytest_collection_modifyitems(config, items):
    """Skips card tests unless KMR_CARD=1 and mesh tests unless KMR_MESH=1."""
    gates = (
        ("card", "KMR_CARD", "needs a Tenstorrent card (KMR_CARD=1)"),
        ("mesh", "KMR_MESH", "opens cards as a mesh, in a run of its own (KMR_MESH=1)"),
    )
    for keyword, variable, reason in gates:
        if os.environ.get(variable) == "1":
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if keyword in item.keywords:
                item.add_marker(skip)
