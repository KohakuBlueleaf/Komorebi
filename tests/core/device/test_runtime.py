"""The runtime package's locator on a fake layout, and the order in which Komorebi
chooses its tt-metal tree."""

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import komorebi_runtime
from komorebi import _env
from komorebi_runtime import prepare

ROOT = Path(__file__).resolve().parents[3]
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


def primed_package(tmp_path):
    """A fake package with the priming program, a fetched compiler and card 1."""
    base = fake_package(tmp_path / "pkg")
    program = komorebi_runtime.prime_program(base)
    program.parent.mkdir()
    program.write_text("")
    environ = {
        komorebi_runtime.CACHE_VARIABLE: str(tmp_path / "cache"),
        "TT_METAL_HOME": "/elsewhere",
    }
    gcc = komorebi_runtime.sfpi_dir(environ, base) / komorebi_runtime.SFPI_BIN
    gcc.mkdir(parents=True)
    (gcc / "riscv-tt-elf-gcc").write_text("")
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "1").write_text("")
    return base, environ, tmp_path / "dev"


def answer(code, text):
    """A stand-in for subprocess.run that records its call and prints `text`."""
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return SimpleNamespace(returncode=code, stdout=text, stderr="")

    return run, calls


def test_prime_reads_the_programs_three_outcomes(tmp_path):
    base, environ, devices = primed_package(tmp_path)
    cases = [
        (0, f"runtime open: x\n\n{prepare.PRIMED}: tt-smi -r <device>\n", 0, "primed"),
        (
            1,
            f"Failure\n  the first open of an unprimed chip {prepare.ALREADY}\n",
            0,
            "already",
        ),
        (1, "Failure\nchip 0 L2CPU 2: WayEnable read all ones\n", 1, "failed"),
    ]
    for status, text, code, word in cases:
        run, calls = answer(status, text)
        lines, got = prepare.prime(1, "sfpi", environ, base, devices, run)
        assert got == code and word in lines[0], lines
        ((command, env),) = calls
        assert command == [str(komorebi_runtime.prime_program(base))]
        assert (
            env["TT_VISIBLE_DEVICES"] == "1" and env["X280_ALLOW_LIM_ECC_PRIME"] == "1"
        )
        assert env[komorebi_runtime.ROOT_VARIABLE].endswith("runtime-0.81.1.7")
        assert "TT_METAL_HOME" not in env
    assert lines[1:] == ["chip 0 L2CPU 2: WayEnable read all ones"]


def test_prime_refuses_without_a_card_a_compiler_or_the_program(tmp_path):
    base, environ, devices = primed_package(tmp_path)
    run, calls = answer(0, "")
    lines, code = prepare.prime(0, "kmr sfpi", environ, base, devices, run)
    assert code == 1 and "no card 0" in lines[0]
    other = {komorebi_runtime.CACHE_VARIABLE: str(tmp_path / "empty")}
    lines, code = prepare.prime(1, "kmr sfpi", other, base, devices, run)
    assert code == 1 and "`kmr sfpi`" in lines[0]
    komorebi_runtime.prime_program(base).unlink()
    lines, code = prepare.prime(1, "kmr sfpi", environ, base, devices, run)
    assert code == 1 and "no priming program" in lines[0]
    assert not calls


def test_the_default_card_is_the_first_visible_one():
    assert prepare.default_card({}) == 0
    assert prepare.default_card({"TT_VISIBLE_DEVICES": "1,0"}) == 1
