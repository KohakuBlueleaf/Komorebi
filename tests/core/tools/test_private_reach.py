"""The private-reach check: what counts as a reach, what a config exempts."""

import tomllib

from komorebi.tools import private_reach

CONFIG = {
    "exempt": {"experiments/probe.py": "probes an internal"},
    "foreign": {"_asdict": "a named tuple"},
    "internal": {"backend": "a run's device side"},
}


def found(source, path="examples/a.py"):
    return [reach.text for reach in private_reach.reaches(path, source, CONFIG)]


def test_a_private_attribute_import_or_string_is_a_reach():
    source = (
        "import komorebi._env\n"
        "from komorebi.session import _helper\n"
        "from komorebi._x import y\n"
        "run._kept\n"
        "run.island(0)._blocks\n"
        "getattr(run, '_state')\n"
        "run.backend.port\n"
    )
    assert found(source) == [
        "import komorebi._env",
        "from komorebi.session import _helper",
        "from komorebi._x import y",
        "._kept",
        "._blocks",
        'getattr(..., "_state")',
        ".backend (a run's device side)",
    ]


def test_own_names_dunders_and_other_packages_are_not_reaches():
    source = (
        "import torch\n"
        "from collections import namedtuple\n"
        "self._cache\n"
        "cls._made\n"
        "torch._C.thing\n"
        "point._asdict()\n"
        "run.__class__\n"
        "getattr(run, 'state')\n"
        "from komorebi.session import run\n"
    )
    assert found(source) == []


def test_an_exempt_file_is_skipped_and_named():
    sources = {"experiments/probe.py": "run._kept\n", "examples/a.py": "run.state()\n"}
    reaches, exempt = private_reach.check(sources, CONFIG)
    assert reaches == [] and exempt == ["experiments/probe.py"]


def test_the_shipped_config_gives_a_reason_for_every_entry():
    config = tomllib.loads(private_reach.CONFIG.read_text(encoding="utf-8"))
    assert set(config) == {"exempt", "foreign", "internal"}
    for table in config.values():
        assert all(isinstance(why, str) and why for why in table.values())
