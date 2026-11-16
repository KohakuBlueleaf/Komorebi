"""Builds the komorebi-tt wheel (imported as komorebi) from a committed tree: the
package, the native sources and the dispatcher's image, prebuilt against a tt-metal
tree with the X280 runtime; --host-module adds _km_host built in that tree.
Usage: build_wheel.py --tt-metal <tree> [--ref HEAD] [--out dist] [--host-module]
"""

import argparse
import base64
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IMAGE = "km-dispatch.bin"
MODULE = "_km_host.abi3.so"
# The module finds the runtime package's libraries beside its own package.
MODULE_RPATH = "$ORIGIN/../komorebi_runtime/lib"
HOOK = "src/komorebi_native/cmake/km_host.cmake"
# A stable-ABI module for CPython 3.12 and later, on this platform.
MODULE_TAG = "cp312-abi3-{platform}"

# Members every wheel must carry, relative to its root.
REQUIRED = [
    "komorebi/__init__.py",
    "komorebi/cli.py",
    f"komorebi/{IMAGE}",
    "komorebi_native/include/komorebi/abi/km_abi.h",
    "komorebi_native/include/komorebi/device/km_worker.hpp",
    "komorebi_native/include/komorebi/device/overlays/km_overlay.hpp",
    "komorebi_native/include/komorebi/device/overlays/km_overlay_part.hpp",
    "komorebi_native/include/komorebi/device/overlays/km_overlay_entry.hpp",
    "komorebi_native/include/komorebi/device/core/km_program.hpp",
    "komorebi_native/kernels/program/reader.cpp",
    "komorebi_native/kernels/program/writer.cpp",
    "komorebi_native/kernels/program/compute.cpp",
    "komorebi_native/kernels/messages/compute.cpp",
    "komorebi_native/kernels/program/service_reader.cpp",
    "komorebi/program/overlay.py",
    "komorebi_native/x280/kernel/main.c",
    "komorebi_native/x280/kernel/km_dispatch.h",
    "komorebi_native/csrc/host/km_host.cpp",
]
# Directories whose files are all carried (empty until something is written there).
CARRIED = ["komorebi_native/include", "komorebi_native/x280", "komorebi_native/kernels"]
CARRIED += ["komorebi_native/cmake", "komorebi_native/csrc"]
# Packages of src/ that belong to another wheel (komorebi-runtime's locator).
FOREIGN = ["komorebi_runtime/"]


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
    """The dispatcher image built from the stage's sources against `tree`."""
    env = dict(os.environ, PYTHONPATH=str(stage / "src"), KMR_CACHE=str(cache))
    code = (
        "import sys; from komorebi.device import native; "
        "print(native.image_path(sys.argv[1]))"
    )
    out = run(sys.executable, "-c", code, str(tree), env=env, cwd=stage).stdout
    image = Path(out.strip().splitlines()[-1])
    shutil.copyfile(image, stage / "src" / "komorebi" / IMAGE)
    return image.stat().st_size


def build_module(stage, tree, work, patchelf):
    """_km_host built in `tree`'s build with the stage's km_host.cmake (the tree's
    configuration is restored after), its rpath set to the runtime package's libraries.
    """
    build = tree / "build"
    configure = ["cmake", "-S", str(tree), "-B", str(build)]
    hook = {
        "CMAKE_PROJECT_Metalium_INCLUDE": stage / HOOK,
        "KM_NATIVE_DIR": stage / "src" / "komorebi_native",
        "KM_HOST_OUTPUT_DIR": work,
    }
    try:
        run(*configure, *[f"-D{k}={v}" for k, v in hook.items()])
        run("ninja", "-C", str(build), "_km_host")
    finally:
        run(*configure, *[f"-U{k}" for k in hook])
    module = work / MODULE
    run(patchelf, "--set-rpath", MODULE_RPATH, str(module))
    return module


def glibc_floor(paths):
    """The newest GLIBC symbol version the ELF files `paths` require, as (major, minor):
    the floor of the wheel's manylinux tag."""
    found = [(2, 17)]
    for path in paths:
        text = run("readelf", "-V", "--wide", str(path)).stdout
        found += [tuple(map(int, v)) for v in re.findall(r"GLIBC_(\d+)\.(\d+)", text)]
    return max(found)


def record_line(name, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"{name},sha256={digest.decode()},{len(data)}"


def add_module(wheel, module, platform):
    """A copy of the pure `wheel` with the module beside the package, tagged for the
    platform; the pure wheel is removed."""
    tag = MODULE_TAG.format(platform=platform)
    target = wheel.with_name(wheel.name.replace("py3-none-any", tag))
    with zipfile.ZipFile(wheel) as old, zipfile.ZipFile(target, "w") as new:
        entries = {m.filename: old.read(m) for m in old.infolist()}
        info = next(n.split("/")[0] for n in entries if n.endswith(".dist-info/WHEEL"))
        text = entries[f"{info}/WHEEL"].decode()
        text = re.sub(r"Root-Is-Purelib: \w+", "Root-Is-Purelib: false", text)
        text = re.sub(r"(?m)^Tag: .*$", f"Tag: {tag}", text)
        entries[f"{info}/WHEEL"] = text.encode()
        entries[f"komorebi/{MODULE}"] = module.read_bytes()
        record = f"{info}/RECORD"
        entries.pop(record)
        lines = [record_line(n, d) for n, d in entries.items()] + [f"{record},,"]
        entries[record] = ("\n".join(lines) + "\n").encode()
        for name, data in entries.items():
            mode = 0o755 if name.endswith(".so") else 0o644
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.external_attr = (0o100000 | mode) << 16
            new.writestr(entry, data, zipfile.ZIP_DEFLATED)
    wheel.unlink()
    return target


def build_wheel(stage, out, isolated):
    """setuptools' wheel of the stage, written into `out`; returns its path."""
    command = [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(out)]
    if not isolated:
        command.append("--no-build-isolation")
    before = set(out.glob("*.whl"))
    run(*command, str(stage), env=dict(os.environ, PIP_NO_CACHE_DIR="1"))
    (wheel,) = set(out.glob("*.whl")) - before
    return wheel


def check(wheel, stage, required):
    """Rejects a wheel missing a required member or a file of a carried folder, or
    carrying a package of another wheel."""
    names = set(zipfile.ZipFile(wheel).namelist())
    foreign = sorted(n for n in names if n.startswith(tuple(FOREIGN)))
    if foreign:
        raise SystemExit(f"{wheel.name} carries another wheel's files: {foreign}")
    missing = [n for n in required if n not in names]
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
    parser.add_argument(
        "--host-module",
        action="store_true",
        help="carry _km_host, built in --tt-metal (komorebi-runtime's build)",
    )
    parser.add_argument(
        "--platform", help="module platform tag (default: manylinux at its glibc floor)"
    )
    parser.add_argument("--patchelf", default="patchelf")
    args = parser.parse_args()
    tree = args.tt_metal.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out.resolve()
    for old in out.glob("komorebi_tt-*.whl"):
        old.unlink()
    with tempfile.TemporaryDirectory(prefix="komorebi-wheel-", dir=out) as tmp:
        stage = Path(tmp) / "src"
        stage.mkdir()
        commit = export(args.ref, stage)
        size = build_image(stage, tree, Path(tmp) / "cache")
        wheel = build_wheel(stage, out, args.isolated)
        required = REQUIRED
        if args.host_module:
            work = Path(tmp) / "module"
            module = build_module(stage, tree, work, args.patchelf)
            module_size = module.stat().st_size
            floor = glibc_floor([module])
            platform = args.platform or "manylinux_{}_{}_x86_64".format(*floor)
            wheel = add_module(wheel, module, platform)
            required = [*REQUIRED, f"komorebi/{MODULE}"]
        count = check(wheel, stage, required)
    print(f"{wheel} ({wheel.stat().st_size / 1e3:.0f} kB, {count} files)")
    print(f"commit {commit}; {IMAGE} {size} bytes, built against {tree}")
    if args.host_module:
        print(f"{MODULE} {module_size} bytes, rpath {MODULE_RPATH}")


if __name__ == "__main__":
    main()
