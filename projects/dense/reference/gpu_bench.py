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
    binary = native.build_directory() / "gemm_gpu"
    if native.stale(binary, [SOURCE]):
        arch = os.environ.get("KR_CUDA_ARCH", "sm_70")
        command = [nvcc, "-std=c++17", "-O3", f"-arch={arch}", "-o", binary, SOURCE]
        subprocess.run([*map(str, command), "-lcublas"], check=True)
    return binary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shape", choices=pl.SHAPES, required=True)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    m_rows, k_depth, n_columns = pl.SHAPES[args.shape]
    executable, gpu = build(), native.idle_gpu()
    if executable is None or gpu is None:
        raise SystemExit("no nvcc or no idle GPU")
    report = {"shape": args.shape, "gpu_index": gpu, "rows": []}
    for precision in TYPES:
        command = [
            executable,
            "--m",
            m_rows,
            "--k",
            k_depth,
            "--n",
            n_columns,
            "--type",
            precision,
        ]
        command += ["--seconds", args.seconds]
        row = power.run_with_power([str(argument) for argument in command], gpu=gpu)
        power_report = row["power"]
        if "gpu_mean_w" in power_report:
            power_report["joules_per_call"] = (
                power_report["gpu_mean_w"] * row["microseconds_per_call"] * 1e-6
            )
        report["rows"].append(row)
        print(json.dumps(row), flush=True)
    if args.json:
        with open(args.json, "w") as file:
            json.dump(report, file, indent=1)


if __name__ == "__main__":
    main()
