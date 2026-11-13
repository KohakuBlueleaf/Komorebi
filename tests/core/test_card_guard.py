"""The hardware conftest's guard: a card run never mixes a module that opens the card in
the pytest process with one that runs a card script as a child."""

import importlib.util
import types
from pathlib import Path

CONFTEST = Path(__file__).resolve().parents[1] / "hardware" / "conftest.py"


def item(in_process, card=True):
    module = types.SimpleNamespace(CARD_IN_PROCESS=True) if in_process else object()
    return types.SimpleNamespace(module=module, keywords={"card"} if card else set())


def test_a_run_of_in_process_and_child_card_modules_is_rejected():
    spec = importlib.util.spec_from_file_location("hardware_conftest", CONFTEST)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    assert guard.mixed_card_kinds([item(True), item(False)])
    assert not guard.mixed_card_kinds([item(True), item(True)])
    assert not guard.mixed_card_kinds([item(False), item(False)])
    assert not guard.mixed_card_kinds([item(True), item(False, card=False)])
