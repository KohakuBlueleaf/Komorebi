"""The runtime package's locator on a fake layout, and the order in which Komorebi
chooses its tt-metal tree."""

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import komorebi_runtime
from komorebi import _env

ROOT = Path(__file__).resolve().parents[2]
SFPI = "sfpi_version='7.89.0'\nsfpi_repo='https://example.invalid/sfpi'\n"


def fake_package(base):
    """A package directory with a manifest, a small tree and the X280 pieces."""
    (base / "tree" / "runtime" / "hw").mkdir(parents=True)
    pieces = komorebi_runtime.x280(base)
    for path in [
        *pieces.objects,
        pieces.linker_script,
        komorebi_runtime.tree(base) / komorebi_runtime.SFPI_VERSION,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SFPI if path.name == "sfpi-version" else "")
    info = {"pin": "v0.81.1", "version": "0.81.1.7", "build": 7}
    (base / komorebi_runtime.MANIFEST).write_text(json.dumps(info))
    return base


def test_the_locator_reads_a_layout(tmp_path):
    base = fake_package(tmp_path / "pkg")
    environ = {komorebi_runtime.CACHE_VARIABLE: str(tmp_path / "cache")}
    assert komorebi_runtime.installed(base) and not komorebi_runtime.installed(tmp_path)
    assert (komorebi_runtime.pin(base), komorebi_runtime.version(base)) == (
        "v0.81.1",
        "0.81.1.7",
    )
    assert (
        komorebi_runtime.x280(base).include
        == base / "tree" / komorebi_runtime.X280_SOURCE / "include"
    )
    assert (
        komorebi_runtime.sfpi_dir(environ, base)
        == tmp_path / "cache" / "sfpi-7.89.0" / "sfpi"
    )
    assert komorebi_runtime.sfpi_bin(environ, base) is None


def test_the_root_links_the_tree_and_the_fetched_compiler(tmp_path):
    base = fake_package(tmp_path / "pkg")
    environ = {komorebi_runtime.CACHE_VARIABLE: str(tmp_path / "cache")}
    root = komorebi_runtime.root(environ, base)
    assert root == tmp_path / "cache" / "runtime-0.81.1.7"
    assert (root / "runtime" / "hw").resolve() == base / "tree" / "runtime" / "hw"
    assert not (root / "runtime" / "sfpi").exists()
    gcc = (
        komorebi_runtime.sfpi_dir(environ, base)
        / komorebi_runtime.SFPI_BIN
        / "riscv-tt-elf-gcc"
    )
    gcc.parent.mkdir(parents=True)
    gcc.write_text("")
    root = komorebi_runtime.root(environ, base)
    # Komorebi's own lookups give the package's pieces through the root.
    assert _env.x280_compiler(root).resolve() == gcc
    assert (
        _env.x280_linker_script(root).resolve()
        == komorebi_runtime.x280(base).linker_script
    )
    assert [
        path.resolve() for path in _env.x280_objects(root)
    ] == komorebi_runtime.x280(base).objects
    assert (
        _env.x280_idle_image(root).resolve() == komorebi_runtime.x280(base).idle_image
    )


def test_activate_sets_the_root(tmp_path):
    base = fake_package(tmp_path / "pkg")
    environ = {komorebi_runtime.CACHE_VARIABLE: str(tmp_path / "cache")}
    root = komorebi_runtime.activate(environ, base)
    assert environ[komorebi_runtime.ROOT_VARIABLE] == str(root)


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
    runtime = ROOT / "packaging" / "runtime"
    project = tomllib.loads((runtime / "pyproject.toml").read_text())
    pin = (ROOT / "packaging" / "tt-metal.pin").read_text().strip()
    number = (runtime / "BUILD").read_text().strip()
    assert project["project"]["version"] == f"{pin.removeprefix('v')}.{number}"
