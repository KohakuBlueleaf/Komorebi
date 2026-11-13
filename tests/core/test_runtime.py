"""The runtime package's locator on a fake layout, and the order in which Komorebi
chooses its tt-metal tree."""

import json
import os
import tomllib
from pathlib import Path
from types import SimpleNamespace

import komorebi_runtime as kr
from komorebi import _env

ROOT = Path(__file__).resolve().parents[2]
SFPI = "sfpi_version='7.89.0'\nsfpi_repo='https://example.invalid/sfpi'\n"


def fake_package(base):
    """A package directory with a manifest, a small tree and the X280 pieces."""
    (base / "tree" / "runtime" / "hw").mkdir(parents=True)
    pieces = kr.x280(base)
    for path in [
        *pieces.objects,
        pieces.linker_script,
        kr.tree(base) / kr.SFPI_VERSION,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SFPI if path.name == "sfpi-version" else "")
    info = {"pin": "v0.81.1", "version": "0.81.1.7", "build": 7}
    (base / kr.MANIFEST).write_text(json.dumps(info))
    return base


def test_the_locator_reads_a_layout(tmp_path):
    base = fake_package(tmp_path / "pkg")
    environ = {kr.CACHE_VARIABLE: str(tmp_path / "cache")}
    assert kr.installed(base) and not kr.installed(tmp_path)
    assert (kr.pin(base), kr.version(base)) == ("v0.81.1", "0.81.1.7")
    assert kr.x280(base).include == base / "tree" / kr.X280_SOURCE / "include"
    assert kr.sfpi_dir(environ, base) == tmp_path / "cache" / "sfpi-7.89.0" / "sfpi"
    assert kr.sfpi_bin(environ, base) is None


def test_the_root_links_the_tree_and_the_fetched_compiler(tmp_path):
    base = fake_package(tmp_path / "pkg")
    environ = {kr.CACHE_VARIABLE: str(tmp_path / "cache")}
    root = kr.root(environ, base)
    assert root == tmp_path / "cache" / "runtime-0.81.1.7"
    assert (root / "runtime" / "hw").resolve() == base / "tree" / "runtime" / "hw"
    assert not (root / "runtime" / "sfpi").exists()
    gcc = kr.sfpi_dir(environ, base) / kr.SFPI_BIN / "riscv-tt-elf-gcc"
    gcc.parent.mkdir(parents=True)
    gcc.write_text("")
    root = kr.root(environ, base)
    # Komorebi's own lookups give the package's pieces through the root.
    assert _env.x280_compiler(root).resolve() == gcc
    assert _env.x280_linker_script(root).resolve() == kr.x280(base).linker_script
    assert [p.resolve() for p in _env.x280_objects(root)] == kr.x280(base).objects
    assert _env.x280_idle_image(root).resolve() == kr.x280(base).idle_image


def test_activate_sets_the_root(tmp_path):
    base = fake_package(tmp_path / "pkg")
    environ = {kr.CACHE_VARIABLE: str(tmp_path / "cache")}
    root = kr.activate(environ, base)
    assert environ[kr.ROOT_VARIABLE] == str(root)


def test_komorebi_takes_the_home_tree_first_then_the_runtime(tmp_path, monkeypatch):
    runtime = SimpleNamespace(installed=lambda: True, root=lambda env: tmp_path / "rt")
    monkeypatch.setattr(_env, "komorebi_runtime", runtime)
    tree = tmp_path / "tt-metal"
    tree.mkdir()
    home = {_env.HOME_VARIABLE: str(tree)}
    assert _env.find_tree(home) == (tree, _env.HOME_VARIABLE)
    missing = {_env.HOME_VARIABLE: str(tmp_path / "absent")}
    assert _env.find_tree(missing) == (None, _env.HOME_VARIABLE)
    assert _env.find_tree({}) == (tmp_path / "rt", _env.RUNTIME_SOURCE)
    runtime.installed = lambda: False
    assert _env.find_tree({}) == (None, None)


def test_the_runtime_version_is_its_pin_and_build_number():
    project = tomllib.loads((ROOT / "runtime" / "pyproject.toml").read_text())
    pin = (ROOT / "runtime" / "tt-metal.pin").read_text().strip()
    number = (ROOT / "runtime" / "BUILD").read_text().strip()
    assert project["project"]["version"] == f"{pin.removeprefix('v')}.{number}"
    assert os.path.samefile(
        ROOT / "packaging" / "tt-metal.pin", ROOT / "runtime" / "tt-metal.pin"
    )
