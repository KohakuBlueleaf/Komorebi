"""Small NoC writes whose source and destination differ in address bits [3:0]: which
word of the target arrives, and whether a non-posted one is acknowledged. JSON line.

Example: python probe.py --set controls; python probe.py --set mismatched"""

import argparse
import json
import time

import numpy as np
import torch
import ttnn

WORDS = 128  # per core: 8 target slots of 16 words on B; on A the sources and results
SRC, RESULT = 32, 64
A, B = (0, 0), (1, 0)
SETS = {  # (slot, source word, destination word, posted)
    "controls": [(0, 0, 0, 1), (1, 1, 1, 1), (2, 3, 3, 1), (7, 1, 1, 0)],
    "mismatched": [
        (3, 1, 0, 1),
        (4, 2, 1, 1),
        (5, 3, 0, 1),
        (6, 0, 3, 1),
        (7, 1, 0, 0),
    ],
}
T0 = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--set", choices=list(SETS), required=True)
    tests = SETS[p.parse_args().set]
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    log("device open")
    try:
        cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in (A, B)]
        )
        config = ttnn.create_sharded_memory_config(
            (1, WORDS),
            cores,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        zeros = torch.zeros((2, WORDS), dtype=torch.int32)
        out = ttnn.from_torch(
            zeros,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=config,
        )
        dummy = ttnn.from_torch(  # generic_op takes two tensors or more
            torch.zeros((1, 16), dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        b = device.worker_core_from_logical_core(ttnn.CoreCoord(*B))
        words = [s | sw << 8 | dw << 16 | posted << 24 for s, sw, dw, posted in tests]
        args = ttnn.RuntimeArgs()
        args[A[0]][A[1]] = [out.buffer_address(), b.x | b.y << 16, len(tests), *words]
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(__file__).replace("probe.py", "probe.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(*A), ttnn.CoreCoord(*A))]
            ),
            compile_time_args=[],
            runtime_args=args,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default,
            ),
        )
        desc = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[])
        log("launch")
        ttnn.generic_op([dummy, out], desc)
        ttnn.synchronize_device(device)
        log("kernel done")
        rows = ttnn.to_torch(out).numpy().astype(np.int64) & 0xFFFFFFFF
        log("read back")
    finally:
        ttnn.close_mesh_device(device)
    a_row, b_row = rows[0], rows[1]  # shard order follows the core set: A, then B
    results = []
    for t, (slot, sw, dw, posted) in enumerate(tests):
        slot_words = b_row[16 * slot : 16 * slot + 4]
        results.append(
            {
                "slot": slot,
                "src_word": sw,
                "dst_word": dw,
                "posted": bool(posted),
                "arrived": {
                    i: f"S[{int(v) & 0xF}]" if int(v) >> 16 == 0xA0A0 else hex(int(v))
                    for i, v in enumerate(slot_words)
                    if v
                },
                "done": {0: "no", 1: "yes", 2: "acked", 3: "no ack"}[
                    int(a_row[RESULT + t])
                ],
            }
        )
    sources_on_a = int(a_row[SRC]) == 0xA0A00000  # row 0 is A's shard
    print(json.dumps({"rows_ok": sources_on_a, "tests": results}), flush=True)


if __name__ == "__main__":
    main()
