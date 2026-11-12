"""The package imports with no tt-metal, TT-NN or X280 module, and says one version."""

import json
import os
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import komorebi

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("ttnn", "ttml", "tt_metal", "tt_x280", "_ttnn", "_ttx280")

PROBE = f"""
import json, sys
sys.path.insert(0, {str(ROOT / "src")!r})
import komorebi, komorebi.build, komorebi.cli, komorebi.layout, komorebi.stats
import komorebi.tools.cxx_format, komorebi.abi, komorebi.errors, komorebi.native
import komorebi.program, komorebi.run, komorebi.state
print(json.dumps(sorted(sys.modules)))
"""


def child_environment(**extra):
    """A copy of the current environment without TT_METAL_HOME, plus `extra`."""
    env = {k: v for k, v in os.environ.items() if k != "TT_METAL_HOME"}
    return {**env, **extra}


def imported_modules(env):
    """The modules a fresh interpreter has after importing every komorebi module."""
    done = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return json.loads(done.stdout)


def test_import_pulls_in_no_tt_module():
    env = child_environment(TT_METAL_HOME="/nonexistent/tt-metal")
    modules = imported_modules(env)
    assert "komorebi.layout" in modules
    assert [m for m in modules if m.split(".")[0] in FORBIDDEN] == []


def test_session_is_imported_only_on_use():
    modules = imported_modules(child_environment())
    assert "komorebi.session" not in modules and "komorebi.run" in modules


def test_import_works_without_tt_metal_home():
    env = child_environment()
    assert "komorebi.cli" in imported_modules(env)


def test_version_matches_pyproject_and_the_installed_distribution():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert komorebi.__version__ == project["version"]
    assert komorebi.__version__ == metadata.version("komorebi")
    assert project["name"] == "komorebi"
    assert project["scripts"]["kmr"] == "komorebi.cli:main"
