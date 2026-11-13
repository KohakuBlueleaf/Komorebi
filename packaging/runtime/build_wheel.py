"""Builds the komorebi-runtime wheel from a tt-metal tree built by build_tt_metal.sh:
the locator, the libraries with $ORIGIN rpaths, the kernel runtime tree, the X280
pieces.
Usage: build_wheel.py --tt-metal <tree> [--out dist] [--patchelf PATH]
"""

import argparse
import base64
import hashlib
import itertools
import json
import re
import shutil
import subprocess
import tempfile
import tomllib
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = "komorebi_runtime"
# The locator's sources and the one pin, both outside this folder.
SOURCE = HERE.parents[1] / "src" / PACKAGE
PIN = HERE.parent / "tt-metal.pin"

# The kernel runtime tree the JIT compiles against: what tt-metal's own wheel ships (its
# setup.py), by root; the same list Kohaku-TT's wheel carries.
RUNTIME = {
    "runtime": ["hw/**/*"],
    "ttnn": ["api/ttnn/tensor/enum_types.hpp"],
    "ttnn/cpp": [
        "ttnn/kernel/**/*",
        "ttnn/operations/**/kernels/**/*",
        "ttnn/operations/**/kernels_ng/**/*",
        "ttnn/operations/**/kernels_dfb/**/*",
        "ttnn/operations/**/shared_with_host/**/*",
        "ttnn/operations/**/hostdevcommon/**/*",
        "ttnn/operations/kernel_helper_functions/*",
        "ttnn/operations/ccl/**/*",
        "ttnn/operations/data_movement/**/*",
        "ttnn/operations/moreh/**/*",
        "ttnn/kernel_lib/**/*.{hpp,inl}",
        "ttnn/operations/normalization/kernel_util/**/*",
    ],
    "tt_metal": [
        "api/tt-metalium/buffer_constants.hpp",
        "api/tt-metalium/buffer_types.hpp",
        "api/tt-metalium/circular_buffer_constants.h",
        "api/tt-metalium/constants.hpp",
        "api/tt-metalium/dev_msgs.h",
        "api/tt-metalium/experimental/fabric/fabric_edm_types.hpp",
        "api/tt-metalium/experimental/fabric/edm_fabric_counters.hpp",
        "core_descriptors/*.yaml",
        "fabric/hw/**/*",
        "fabric/mesh_graph_descriptors/*.{yaml,textproto}",
        "fabric/impl/kernels/**/*",
        "hw/**/*",
        "hostdevcommon/api/hostdevcommon/**/*",
        "impl/buffers/kernels/**/*",
        "impl/buffers/dram_sender_state_block.hpp",
        "impl/buffers/tensor_prefetcher_request.hpp",
        "impl/dispatch/kernels/**/*",
        "impl/streaming_profiler/kernels/**/*",
        "include/**/*",
        "kernels/**/*",
        "tt-llk/**/*",
        "tools/profiler/**/*",
        "soc_descriptors/*.yaml",
        "sfpi-version",
    ],
    # The fused attention kernels of tt-train that Kohaku-TT's native library compiles.
    "tt-train/sources/ttml": ["metal/**/kernels/**/*", "metal/common/**/*"],
    # The X280 runtime: its linker scripts and headers; from the build, the objects
    # every image links and the idle image the tiles boot to.
    "tt_metal/fabric/experimental/x280": ["ld/*.ld", "include/**/*"],
    "build/x280": [
        "obj/entry_active.o",
        "obj/x280_active_start.o",
        "firmware-lim-idle.bin",
    ],
}
# Not needed to run kernels.
SKIP = ("/tt-llk/tests/", "/tt-llk/docs/", "/__pycache__/", "/.git")

# The libraries: _ttnncpp.so (Kohaku-TT's native library links it) and what it loads
# out of the build, libtt_metal.so among them.
TTNNCPP = "build/ttnn/_ttnncpp.so"


def expand(pattern):
    """Glob patterns of `pattern` with its {a,b} groups expanded."""
    groups = re.findall(r"\{([^}]*)\}", pattern)
    if not groups:
        return [pattern]
    template = re.sub(r"\{[^}]*\}", "{}", pattern)
    choices = [g.split(",") for g in groups]
    return [template.format(*c) for c in itertools.product(*choices)]


def run(*command, **kwargs):
    return subprocess.run(command, check=True, capture_output=True, text=True, **kwargs)


def copy_runtime(tt_metal, root):
    """Copies the RUNTIME files of `tt_metal` into `root`; returns how many (a file
    two patterns match counts once)."""
    copied = set()
    for base, patterns in RUNTIME.items():
        for pattern in itertools.chain.from_iterable(expand(p) for p in patterns):
            for path in (tt_metal / base).glob(pattern):
                text = path.as_posix()
                if not path.is_file() or any(s in text for s in SKIP):
                    continue
                target = root / path.relative_to(tt_metal)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                copied.add(target)
    return len(copied)


def build_libraries(module, build):
    """(name it is loaded as, path) of each library `module` loads from `build`, in the
    order ldd lists them."""
    out = []
    for line in run("ldd", str(module)).stdout.splitlines():
        match = re.match(r"\s*(\S+) => (\S+)", line)
        if match and Path(match[2]).resolve().is_relative_to(build):
            out.append((match[1], Path(match[2]).resolve()))
    return out


def patches():
    """{patch file name: sha256} of the patches the tree is built with."""
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((HERE / "patches").glob("*.patch"))
    }


def check_tree(tt_metal, pin):
    """Rejects a tree at another tag, or one where a patch is not applied."""
    have = run("git", "-C", str(tt_metal), "describe", "--tags", "--always").stdout
    if have.strip() != pin:
        raise SystemExit(f"{tt_metal} holds tt-metal {have.strip()}, the pin is {pin}")
    for name in patches():
        patch = HERE / "patches" / name
        try:
            run("git", "-C", str(tt_metal), "apply", "--reverse", "--check", str(patch))
        except subprocess.CalledProcessError as err:
            raise SystemExit(f"{name} is not applied to {tt_metal}") from err
    return run("git", "-C", str(tt_metal), "rev-parse", "HEAD").stdout.strip()


def glibc_floor(paths):
    """The newest GLIBC symbol version the ELF files `paths` require, as (major, minor):
    the floor of the wheel's manylinux tag."""
    found = [(2, 17)]
    for path in paths:
        text = run("readelf", "-V", "--wide", str(path)).stdout
        found += [tuple(map(int, v)) for v in re.findall(r"GLIBC_(\d+)\.(\d+)", text)]
    return max(found)


def record_line(stage, path):
    data = path.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"{path.relative_to(stage).as_posix()},sha256={digest.decode()},{len(data)}"


def metadata(project):
    """The wheel's METADATA from the [project] table of pyproject.toml."""
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
        f"Summary: {project['description']}",
        f"License-Expression: {project['license']}",
        "License-File: LICENSE",
        f"Requires-Python: {project['requires-python']}",
    ]
    for extra, requirements in project.get("optional-dependencies", {}).items():
        lines.append(f"Provides-Extra: {extra}")
        lines += [f'Requires-Dist: {r}; extra == "{extra}"' for r in requirements]
    lines.append("Description-Content-Type: text/markdown")
    readme = (HERE / project["readme"]).read_text()
    return "\n".join(lines) + "\n\n" + readme


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tt-metal", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=HERE / "dist")
    parser.add_argument(
        "--platform", help="wheel platform tag (default: manylinux at the glibc floor)"
    )
    parser.add_argument("--patchelf", default="patchelf")
    args = parser.parse_args()
    tt_metal = args.tt_metal.resolve()
    build = (tt_metal / "build").resolve()
    project = tomllib.loads((HERE / "pyproject.toml").read_text())["project"]
    pin = PIN.read_text().strip()
    number = (HERE / "BUILD").read_text().strip()
    version = f"{pin.removeprefix('v')}.{number}"
    if project["version"] != version:
        raise SystemExit(
            f"pyproject.toml says {project['version']}, pin and BUILD say {version}"
        )
    commit = check_tree(tt_metal, pin)

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out.resolve()
    with tempfile.TemporaryDirectory(prefix="komorebi-runtime-", dir=out) as tmp:
        stage = Path(tmp) / "stage"
        package = stage / PACKAGE
        shutil.copytree(
            SOURCE,
            package,
            ignore=shutil.ignore_patterns("__pycache__"),
        )

        lib = package / "lib"
        lib.mkdir()
        ttnncpp = tt_metal / TTNNCPP
        libraries = [(ttnncpp.name, ttnncpp), *build_libraries(ttnncpp, build)]
        for name, path in libraries:
            shutil.copyfile(path, lib / name)
            run(args.patchelf, "--set-rpath", "$ORIGIN", str(lib / name))
        count = copy_runtime(tt_metal, package / "tree")

        files = sorted(
            p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()
        )
        manifest = {
            "name": project["name"],
            "version": version,
            "pin": pin,
            "commit": commit,
            "build": int(number),
            "patches": list(patches()),
            "patch_sha256": patches(),
            "x280_runtime": True,
            "distributed": False,
            "libraries": [name for name, _ in libraries],
            "runtime_files": count,
            "files": files,
        }
        (package / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

        info = stage / f"{PACKAGE}-{version}.dist-info"
        (info / "licenses").mkdir(parents=True)
        shutil.copyfile(tt_metal / "LICENSE", info / "licenses" / "LICENSE")
        (info / "METADATA").write_text(metadata(project))
        elves = [p for p in package.rglob("*.so*") if p.is_file()]
        floor = glibc_floor(elves)
        platform = args.platform or "manylinux_{}_{}_x86_64".format(*floor)
        tag = f"py3-none-{platform}"
        (info / "WHEEL").write_text(
            "Wheel-Version: 1.0\nGenerator: komorebi-runtime build_wheel.py\n"
            f"Root-Is-Purelib: false\nTag: {tag}\n"
        )
        paths = sorted(p for p in stage.rglob("*") if p.is_file())
        record = info / "RECORD"
        lines = [record_line(stage, p) for p in paths]
        record.write_text("\n".join([*lines, f"{info.name}/RECORD,,"]) + "\n")

        wheel = out / f"{PACKAGE}-{version}-{tag}.whl"
        with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in [*paths, record]:
                archive.write(path, path.relative_to(stage).as_posix())
    with zipfile.ZipFile(wheel) as archive:
        members = sorted(archive.infolist(), key=lambda m: -m.compress_size)
    print(f"glibc floor {floor[0]}.{floor[1]} over {len(elves)} libraries: {tag}")
    print(
        f"{wheel} ({wheel.stat().st_size / 1e6:.1f} MB, {len(members)} members, "
        f"{count} runtime files, tt-metal {pin} at {commit[:10]})"
    )
    for member in members[:3]:
        print(
            f"  {member.compress_size / 1e6:6.1f} MB  {member.filename} "
            f"({member.file_size / 1e6:.1f} MB unpacked)"
        )


if __name__ == "__main__":
    main()
