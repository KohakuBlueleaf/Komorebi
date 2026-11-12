"""Building and running the C++ and CUDA references, and finding the data folder."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from ref import public, scenes

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "ref" / "native"
CXXFLAGS = ["-std=c++17", "-O3", "-ffp-contract=off", "-pthread"]
NVCCFLAGS = ["-std=c++17", "-O3", "--fmad=false"]
IDLE_MIB = 100
IDLE_UTIL = 10


def build_dir():
    """Where binaries go: $KR_BUILD_DIR, else <repository>/build."""
    path = Path(os.environ.get("KR_BUILD_DIR", ROOT / "build"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir():
    """Where scenes, trees and outputs go: $KR_DATA, else <repository>/build/data."""
    path = Path(os.environ.get("KR_DATA", ROOT / "build" / "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_cxx():
    """The C++ compiler: $CXX, else g++ or c++ on PATH, else None."""
    return os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")


def find_nvcc():
    """nvcc from $NVCC, PATH, $CUDA_HOME or /usr/local/cuda, else None."""
    candidates = [os.environ.get("NVCC"), shutil.which("nvcc")]
    homes = [h for h in (os.environ.get("CUDA_HOME"), "/usr/local/cuda") if h]
    candidates += [str(Path(home) / "bin" / "nvcc") for home in homes]
    return next((c for c in candidates if c and Path(c).exists()), None)


def stale(binary, sources):
    """True when the binary is missing or older than any of the source files."""
    if not binary.exists():
        return True
    return any(Path(s).stat().st_mtime > binary.stat().st_mtime for s in sources)


def build(source, name, gpu=False, includes=()):
    """Compiles one source file to <build dir>/<name>; None if there is no compiler."""
    compiler = find_nvcc() if gpu else find_cxx()
    if compiler is None:
        return None
    out = build_dir() / name
    deps = [source, *NATIVE.glob("*.hpp"), *Path(source).parents[1].glob("*.hpp")]
    deps += [h for i in includes for h in Path(i).glob("*.hpp")]
    if stale(out, deps):
        flags = NVCCFLAGS if gpu else CXXFLAGS
        arch = [f"-arch={os.environ.get('KR_CUDA_ARCH', 'sm_70')}"] if gpu else []
        inc = [f"-I{p}" for p in (NATIVE, Path(source).parents[1], *includes)]
        subprocess.run([compiler, *flags, *arch, *inc, "-o", out, source], check=True)
    return out


def idle_gpu():
    """Index of a GPU with no process, little memory and low load, else None."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    query = "--query-gpu=index,memory.used,utilization.gpu,uuid"
    gpus = subprocess.run(
        [smi, query, "--format=csv,noheader,nounits"], capture_output=True, text=True
    )
    apps = subprocess.run(
        [smi, "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if gpus.returncode or apps.returncode:
        return None
    busy = set(apps.stdout.split())
    for line in gpus.stdout.strip().splitlines():
        index, used, util, uuid = (f.strip() for f in line.split(","))
        if int(used) < IDLE_MIB and int(util) <= IDLE_UTIL and uuid not in busy:
            return int(index)
    return None


def loadavg():
    """The first three numbers of /proc/loadavg, as text."""
    return " ".join(Path("/proc/loadavg").read_text().split()[:3])


def run_json(command, gpu=None):
    """Runs a binary that prints one JSON line; adds the host load before and after."""
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    before = loadavg()
    done = subprocess.run(command, capture_output=True, text=True, env=env, check=True)
    result = json.loads(done.stdout.strip().splitlines()[-1])
    result["load_before"], result["load_after"] = before, loadavg()
    return result


def ensure_scene(name):
    """Writes <data>/<name>.npz and <name>/*.npy if missing; returns the folder.

    A generated preset is made here; a public scene's view is made from its converted
    mesh, which only `python -m ref.public fetch` downloads (missing: an error).
    """
    folder = data_dir() / name
    if not (data_dir() / f"{name}.npz").exists() or not folder.exists():
        if public.is_view(name):
            scene = public.make_view(name, data_dir())
        else:
            scene = scenes.make_preset(name)
        scenes.save_scene(data_dir() / f"{name}.npz", scene)
        scenes.export_npy(scene, folder)
    return folder


def scene_triangles(name):
    """Triangle count of a scene by name: the preset's target or the public table's."""
    return public.triangles(name) if public.is_view(name) else scenes.PRESETS[name]
