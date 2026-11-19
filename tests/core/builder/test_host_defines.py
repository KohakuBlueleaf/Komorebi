"""The two builds of _km_host, builder/flags.py's on first use and km_host.cmake's for
the wheel, define every KM_HOST_* name km_host.cpp reads (the three files as text)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "src/komorebi_native/csrc/host/km_host.cpp"
FLAGS = ROOT / "src/komorebi/builder/flags.py"
CMAKE = ROOT / "src/komorebi_native/cmake/km_host.cmake"


def test_every_km_host_name_of_the_module_is_defined_by_both_builds():
    names = set(re.findall(r"\bKM_HOST_[A-Z0-9_]+\b", SOURCE.read_text()))
    assert names, "km_host.cpp reads no KM_HOST_* name"
    flags = FLAGS.read_text()
    cmake_definitions = " ".join(
        re.findall(
            r"target_compile_definitions\(\s*_km_host\b([^)]*)\)", CMAKE.read_text()
        )
    )
    for name in sorted(names):
        assert f"-D{name}=" in flags, f"{name}: not defined by builder/flags.py"
        assert re.search(
            rf"\b{name}=", cmake_definitions
        ), f"{name}: not defined by km_host.cmake"
