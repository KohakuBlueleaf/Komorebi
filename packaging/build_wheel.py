"""Builds the komorebi wheel from a committed tree: the package, the native sources
and the dispatch kernel image, prebuilt against a tt-metal tree with the X280 runtime.
Usage: build_wheel.py --tt-metal <tree> [--ref HEAD] [--out dist] [--isolated]
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IMAGE = "km-dispatch.bin"

# Members every wheel must carry, relative to its root.
REQUIRED = [
    "komorebi/__init__.py",
    "komorebi/cli.py",
    f"komorebi/{IMAGE}",
    "komorebi_native/include/komorebi/abi/km_abi.h",
    "komorebi_native/include/komorebi/device/km_worker.hpp",
    "komorebi_native/x280/kernel/main.c",
    "komorebi_native/x280/kernel/km_dispatch.h",
    "komorebi_native/csrc/host/km_host.cpp",
]
# Directories whose files are all carried (empty until something is written there).
CARRIED = ["komorebi_native/include", "komorebi_native/x280", "komorebi_native/kernels"]
CARRIED += ["komorebi_native/cmake", "komorebi_native/csrc"]


def run(*command, **kwargs):
    return subprocess.run(command, check=True, capture_output=True, text=True, **kwargs)


def export(ref, stage):
    """The files of commit `ref` in `stage`: nothing untracked or ignored comes."""
    archive = stage.parent / "src.tar"
    run("git", "-C", str(REPO), "archive", "--format=tar", "-o", str(archive), ref)
    with tarfile.open(archive) as tar:
        tar.extractall(stage, filter="data")
    archive.unlink()
    return run("git", "-C", str(REPO), "rev-parse", "--short", ref).stdout.strip()


def build_image(stage, tree, cache):
    """The dispatch kernel image built from the stage's sources against `tree`."""
    env = dict(os.environ, PYTHONPATH=str(stage / "src"), KMR_CACHE=str(cache))
    code = (
        "import sys; from komorebi import native; print(native.image_path(sys.argv[1]))"
    )
    out = run(sys.executable, "-c", code, str(tree), env=env, cwd=stage).stdout
    image = Path(out.strip().splitlines()[-1])
    shutil.copyfile(image, stage / "src" / "komorebi" / IMAGE)
    return image.stat().st_size


def build_wheel(stage, out, isolated):
    """setuptools' wheel of the stage, written into `out`; returns its path."""
    command = [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(out)]
    if not isolated:
        command.append("--no-build-isolation")
    before = set(out.glob("*.whl"))
    run(*command, str(stage), env=dict(os.environ, PIP_NO_CACHE_DIR="1"))
    (wheel,) = set(out.glob("*.whl")) - before
    return wheel


def check(wheel, stage):
    """Refuses a wheel missing a required member or a file of a carried folder."""
    names = set(zipfile.ZipFile(wheel).namelist())
    missing = [n for n in REQUIRED if n not in names]
    for folder in CARRIED:
        for path in (stage / "src" / folder).rglob("*"):
            name = path.relative_to(stage / "src").as_posix()
            if path.is_file() and not path.name.startswith(".") and name not in names:
                missing.append(name)
    if missing:
        raise SystemExit(f"{wheel.name} misses {missing}")
    return len(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tt-metal", required=True, type=Path)
    parser.add_argument("--ref", default="HEAD", help="commit to build (default HEAD)")
    parser.add_argument("--out", type=Path, default=REPO / "dist")
    parser.add_argument(
        "--isolated", action="store_true", help="let pip fetch the build backend"
    )
    args = parser.parse_args()
    tree = args.tt_metal.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out.resolve()
    for old in out.glob("komorebi-*.whl"):
        old.unlink()
    with tempfile.TemporaryDirectory(prefix="komorebi-wheel-", dir=out) as tmp:
        stage = Path(tmp) / "src"
        stage.mkdir()
        commit = export(args.ref, stage)
        size = build_image(stage, tree, Path(tmp) / "cache")
        wheel = build_wheel(stage, out, args.isolated)
        count = check(wheel, stage)
    print(f"{wheel} ({wheel.stat().st_size / 1e3:.0f} kB, {count} files)")
    print(f"commit {commit}; {IMAGE} {size} bytes, built against {tree}")


if __name__ == "__main__":
    main()
