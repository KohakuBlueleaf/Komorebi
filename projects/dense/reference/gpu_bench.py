"""The matrix product with cuBLAS (fp32, and fp16 with float accumulation) on one idle
GPU, with nvidia-smi sampled every 100 ms: time per call and power. Builds
`gpu/gemm_gpu.cu`. Usage in ../README.md.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

from projects.dense import plan as pl
from ref import native, power

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "gpu" / "gemm_gpu.cu"
TYPES = ("fp32", "fp16")


def build():
    """The gemm_gpu binary, rebuilt when the source is newer; None without nvcc."""
    nvcc = native.find_nvcc()
    if nvcc is None:
        return None
    out = native.build_directory() / "gemm_gpu"
    if native.stale(out, [SOURCE]):
        arch = os.environ.get("KR_CUDA_ARCH", "sm_70")
        cmd = [nvcc, "-std=c++17", "-O3", f"-arch={arch}", "-o", out, SOURCE]
        subprocess.run([*map(str, cmd), "-lcublas"], check=True)
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--shape", choices=pl.SHAPES, required=True)
    a.add_argument("--seconds", type=float, default=4.0)
    a.add_argument("--json", default=None)
    args = a.parse_args()
    m, k, n = pl.SHAPES[args.shape]
    exe, gpu = build(), native.idle_gpu()
    if exe is None or gpu is None:
        raise SystemExit("no nvcc or no idle GPU")
    out = {"shape": args.shape, "gpu_index": gpu, "rows": []}
    for t in TYPES:
        cmd = [exe, "--m", m, "--k", k, "--n", n, "--type", t]
        cmd += ["--seconds", args.seconds]
        row = power.run_with_power([str(c) for c in cmd], gpu=gpu)
        p = row["power"]
        if "gpu_mean_w" in p:
            p["joules_per_call"] = p["gpu_mean_w"] * row["microseconds_per_call"] * 1e-6
        out["rows"].append(row)
        print(json.dumps(row), flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
