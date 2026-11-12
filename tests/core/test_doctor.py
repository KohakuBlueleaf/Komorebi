"""`kmr doctor` reports the host, with or without a tt-metal tree."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fake_tree import make_tree

from komorebi import _env, cli

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


def test_doctor_with_a_missing_directory(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("TT_METAL_HOME", str(tmp_path / "absent"))
    code, out = run_doctor(capsys)
    assert code == 0 and "tt-metal      not found" in out


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


def test_doctor_names_the_include_directory_of_a_checkout(no_home, capsys):
    _, out = run_doctor(capsys)
    assert str(ROOT / "src" / "komorebi_native" / "include") in out


@pytest.mark.parametrize("home", [None, "/nonexistent/tt-metal"])
def test_kmr_doctor_exits_zero_in_a_fresh_process(home):
    env = {k: v for k, v in os.environ.items() if k != "TT_METAL_HOME"}
    env["PYTHONPATH"] = str(ROOT / "src")
    if home:
        env["TT_METAL_HOME"] = home
    done = subprocess.run(
        [sys.executable, "-m", "komorebi.cli", "doctor"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert done.returncode == 0, done.stderr
    assert "komorebi" in done.stdout
