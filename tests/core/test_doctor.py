"""`kmr doctor` reports the host, with or without a tt-metal tree."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fake_tree import make_tree

from komorebi import _env, cli
from komorebi.session import process

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def no_home(monkeypatch):
    monkeypatch.delenv("TT_METAL_HOME", raising=False)


def run_doctor(capsys):
    code = cli.main(["doctor"])
    return code, capsys.readouterr().out


def test_doctor_without_a_tree(no_home, capsys):
    code, out = run_doctor(capsys)
    assert code == 0
    assert "komorebi" in out and "python" in out and "package" in out
    assert "tt-metal      not found" in out
    assert "x280 runtime  -" in out
    assert str(ROOT / "src" / "komorebi_native" / "include") in out


def test_doctor_with_a_full_tree(monkeypatch, tmp_path, capsys):
    tree = make_tree(tmp_path / "tt-metal")
    monkeypatch.setenv("TT_METAL_HOME", str(tree))
    code, out = run_doctor(capsys)
    assert code == 0
    assert f"tt-metal      {tree}\n" in out
    assert f"build         {tree / 'build'}\n" in out
    assert "x280 runtime  yes" in out
    assert f"x280 compiler {_env.x280_compiler(tree)}" in out


def test_doctor_with_a_tree_without_the_x280_runtime(monkeypatch, tmp_path, capsys):
    tree = make_tree(tmp_path / "tt-metal", x280=False, compiler=False, ninja=False)
    monkeypatch.setenv("TT_METAL_HOME", str(tree))
    code, out = run_doctor(capsys)
    assert code == 0
    assert f"tt-metal      {tree}\n" in out
    assert "build         not found" in out
    assert "x280 runtime  no" in out
    assert "x280 compiler not found" in out


def test_kmr_doctor_exits_zero_in_a_fresh_process_with_a_missing_tree():
    env = {name: value for name, value in os.environ.items() if name != "TT_METAL_HOME"}
    env["PYTHONPATH"] = str(ROOT / "src")
    env["TT_METAL_HOME"] = "/nonexistent/tt-metal"
    done = subprocess.run(
        [sys.executable, "-m", "komorebi.cli", "doctor"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert done.returncode == 0, done.stderr
    assert "komorebi" in done.stdout


def test_setup_names_what_it_needs_without_the_runtime_package(monkeypatch, capsys):
    monkeypatch.setattr(_env, "runtime_version", lambda: None)
    for command in ("sfpi", "prime", "setup"):
        assert cli.main([command]) == 2
        err = capsys.readouterr().err
        assert f"kmr {command}: needs the komorebi-runtime package" in err
        assert ("x280_prime_test" in err) == (command != "sfpi")


def test_doctor_names_the_card_setup(no_home, capsys):
    assert "card setup    once per card: kmr setup" in run_doctor(capsys)[1]


def fake_driver(tmp_path, beats):
    """A fake driver below tmp_path (sys, dev): a p150a a card of `beats`, its
    heartbeat file holding that value; returns (root, dev)."""
    for card, beat in beats.items():
        entry = tmp_path / "sys" / "class" / "tenstorrent" / f"tenstorrent!{card}"
        entry.mkdir(parents=True)
        for name, text in (("tt_card_type", "p150a"), ("tt_aiclk", "800")):
            (entry / name).write_text(text + "\n")
        (entry / "tt_heartbeat").write_text(beat)
    (tmp_path / "dev").mkdir()
    for card in beats:
        (tmp_path / "dev" / str(card)).write_text("")
    return tmp_path / "sys", tmp_path / "dev"


def test_doctor_gives_a_line_a_card_and_says_a_frozen_heartbeat(
    no_home, monkeypatch, tmp_path
):
    root, dev = fake_driver(tmp_path, {0: "7", 1: "9"})
    entry = root / "class" / "tenstorrent" / "tenstorrent!0"
    monkeypatch.setattr(  # card 0's heartbeat advances while the doctor waits
        process.time, "sleep", lambda s: (entry / "tt_heartbeat").write_text("10")
    )
    monkeypatch.setattr(process, "CHIP_LOCK", str(tmp_path / "no-lock-{card}"))
    out = cli.format_report(cli.doctor_report(root=root, dev=dev))
    lines = out.splitlines()
    assert lines[-3] == (
        "card 0        type p150a, firmware missing, AICLK 800 MHz, heartbeat "
        f"advancing, node {dev / '0'}, not held"
    )
    assert lines[-2].startswith("card 1        type p150a, firmware missing, AICLK ")
    assert "heartbeat frozen at 9," in lines[-2]
    assert lines[-1] == (
        "              NOT READY: its heartbeat stayed at 9 over 0.3 s: reset it with "
        "tt-smi -r 1 (the recovery guide, when a card reset is still the only way), "
        "then open it again"
    )
    (dev / "1").rename(tmp_path / "moved-node")
    out = cli.format_report(cli.doctor_report(root=root, dev=dev))
    assert f"node {dev / '1'} missing, not held" in out
    assert f"NOT READY: its device node {dev / '1'} is missing; its heartbeat" in out


def test_doctor_on_a_host_with_no_card_says_so_in_one_line(no_home, tmp_path):
    report = cli.doctor_report(root=tmp_path / "sys", dev=tmp_path / "dev")
    lines = cli.format_report(report).splitlines()
    assert lines[-1] == "cards         none (no driver entry, no /dev/tenstorrent node)"
    assert lines[-2].startswith("card setup")
