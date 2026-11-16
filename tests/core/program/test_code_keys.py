"""Builds keyed by the code that made them: the package's source hash, a first use's
folder, the cache of built parts and the instruction check's markers each change when
that code changes (no card)."""

import pytest

from komorebi import _env
from komorebi.dsl import first_use
from komorebi.program import overlay
from komorebi.program.checks import instructions as insn_check


@pytest.fixture
def package(tmp_path, monkeypatch):
    """A package of two files and a folder, standing in for komorebi's."""
    (tmp_path / "dsl").mkdir()
    (tmp_path / "dsl" / "compiler.py").write_text("A = 1\n")
    (tmp_path / "program" / "checks").mkdir(parents=True)
    (tmp_path / "program" / "overlay.py").write_text("B = 1\n")
    (tmp_path / "program" / "checks" / "instructions.py").write_text("C = 1\n")
    monkeypatch.setattr(_env, "package_directory", lambda: tmp_path)
    _env.code_key.cache_clear()
    yield tmp_path
    _env.code_key.cache_clear()


def changed(package, name, text):
    (package / name).write_text(text)
    _env.code_key.cache_clear()


def test_the_code_key_follows_the_files_named(package):
    whole, dsl, one = (
        _env.code_key(),
        _env.code_key("dsl"),
        _env.code_key("program/overlay.py"),
    )
    changed(package, "dsl/compiler.py", "A = 2\n")
    assert _env.code_key() != whole and _env.code_key("dsl") != dsl
    assert _env.code_key("program/overlay.py") == one  # a file not named
    (package / "dsl" / "notes.txt").write_text("not code")
    _env.code_key.cache_clear()
    assert _env.code_key("dsl") == _env.code_key("dsl")


def test_a_first_use_folder_follows_the_compiler(package):
    before = first_use.folder_key("gate-0", "gate", {"size": 4}, [b"class"])
    assert before == first_use.folder_key("gate-0", "gate", {"size": 4}, [b"class"])
    changed(package, "dsl/compiler.py", "A = 3\n")  # a compiler change: a new folder
    after = first_use.folder_key("gate-0", "gate", {"size": 4}, [b"class"])
    assert after != before and after.startswith("use")
    given = first_use.folder_key("gate-0", "gate", {"size": 4}, [b"class"], code="x")
    assert given != first_use.folder_key(
        "gate-0", "gate", {"size": 4}, [b"class"], code="y"
    )


def test_the_parts_cache_and_the_markers_follow_the_checks(package, tmp_path_factory):
    key = overlay.checks_key()
    changed(package, "dsl/compiler.py", "A = 4\n")
    assert overlay.checks_key() == key  # the class compiler is not the builder
    changed(package, "program/checks/instructions.py", "C = 2\n")
    assert overlay.checks_key() != key
    elf = tmp_path_factory.mktemp("elf") / "kernel.elf"
    elf.write_bytes(b"\x7fELF")
    marker = insn_check._marker(tmp_path_factory.mktemp("cache"), elf)
    changed(package, "program/checks/instructions.py", "C = 3\n")
    assert insn_check._marker(marker.parent, elf) != marker
