"""Mailbox depth on a Tensix core: unpack writes k words before math (and pack) read
any, for k = 1 to 8; prints the write time and the words read, one JSON line per k."""

import json
import sys
from pathlib import Path

import numpy as np
import ttnn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "feeder"))

import fed  # noqa: E402
import records as rec  # noqa: E402


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        core = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
        )
        config = ttnn.create_sharded_memory_config(
            (1, 16),
            ttnn.CoreGrid(y=1, x=1),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        mhz = fed.aiclk_mhz()
        delay = 20 * mhz  # 20 us
        for both in (0, 1):
            for k in range(1, 9):
                out = fed.int32_tensor(device, np.zeros((1, 16)), config)
                dummy = fed.int32_tensor(device, np.zeros((1, 16)))  # generic_op: 2+
                args = ttnn.RuntimeArgs()
                args[0][0] = [out.buffer_address(), k, both, delay]
                kernel = ttnn.KernelDescriptor(
                    kernel_source=str(HERE / "probe.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core,
                    compile_time_args=[],
                    runtime_args=args,
                    config=ttnn.ComputeConfigDescriptor(),
                    compiler_include_paths=[rec.INCLUDE],
                )
                desc = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[])
                ttnn.generic_op([dummy, out], desc)
                ttnn.synchronize_device(device)
                r = rec.as_uint32(fed.read_int32(out))[0]
                print(
                    json.dumps(
                        {
                            "k": k,
                            "to": "math+pack" if both else "math",
                            "write_us": round(int(r[0]) / mhz, 2),
                            "readers_done": [int(r[1]), int(r[15])],
                            "read": [int(v) for v in r[2 : 2 + k]],
                        }
                    ),
                    flush=True,
                )
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
