"""A table of jobs on the movers and compute, checked against numpy: each core's page
holds its jobs (a count, then four words a job); compute adds the class's constant tile
when a job says so. README.md says what each line lowers to.
Example: python table_jobs.py --cores 8 --jobs 3"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km

STARTED = time.monotonic()
TABLE_WORDS = 1024  # a table page: 4,096 B
FIRST = 4  # the first job's word (word 0: the count)
BIAS = 0.5  # the constant tile's value (the TABLE_BIAS define)


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class Job(km.Entry):
    """A job: its source pages (first, count), its target's first page, and whether
    the constant tile is added."""

    source: int
    count: int
    target: int
    add: int


class TableJobs(km.Program, defines={"TABLE_BIAS": f"{BIAS}f"}):
    """walk: this core's jobs, each source page to its target page, plus the
    constant tile (built once at the compute kernel's start) when the job adds it."""

    bias: km.L1[km.bf16, 1, 32, 32] = km.own(push=None)  # written in place, at start

    @km.handler
    def walk(self, tables: km.Tensor, source: km.Tensor, target: km.Tensor):
        km.cxx(
            """
            #include "api/compute/eltwise_unary/fill.h"
            tile_regs_acquire();
            fill_tile_init();
            fill_tile(0, TABLE_BIAS);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(bias, 1);
            pack_tile<true>(0, bias, 0);
            tile_regs_release();
            """,
            takes=[self.bias],
            at="start",
        )
        km.cxx("noc_async_write_barrier();", on="mover_b", at="end")
        for job in tables.pages(self.rank, 1).entries(
            Job, first=FIRST, size=4, count=0
        ):
            out = km.Stream(target.pages(job.target, job.count), write=True)
            for tile in km.Stream(source.pages(job.source, job.count)):
                km.cxx(
                    """
                    #include "api/compute/eltwise_binary_sfpu.h"
                    tile_regs_acquire();
                    copy_init(tile_buffer);
                    copy_tile(tile_buffer, tile_index, 0);
                    if (job_add != 0) {
                        copy_init(bias);
                        copy_tile(bias, 0, 1);
                        add_binary_tile_init();
                        add_binary_tile(0, 1, 0);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    if (out != 0) {
                        pack_tile(0, out_buffer);
                    }
                    tile_regs_release();
                    """,
                    takes=[tile, job, out, self.bias],
                )


def tables_of(rng, cores, jobs, pages):
    """Each core's table page and its jobs: `jobs` jobs of 1 to 5 pages anywhere in
    the source, each target a page range of its own; the targets' pages in all."""
    words = np.zeros((cores, TABLE_WORDS), np.uint32)
    made, target = [], 0
    for core in range(cores):
        words[core, 0] = jobs
        for index in range(jobs):
            count = int(rng.integers(1, 6))
            first = int(rng.integers(0, pages - count + 1))
            add = int(rng.integers(0, 2))
            at = FIRST + 4 * index
            words[core, at : at + 4] = (first, count, target, add)
            made.append((first, count, target, add))
            target += count
    return words, made, target


def bf16_pages(rng, pages):
    """`pages` tiles of integers -4 to 4 as bfloat16 bits, and their values."""
    values = rng.integers(-4, 5, size=(pages, 1024)).astype(np.float32)
    return (values.view(np.uint32) >> 16).astype(np.uint16), values


def run(session, cores=8, jobs=3, pages=64, seed=1):
    """The class made, loaded and run once on `cores` cores of `session`: the report
    (the event, its time, the kernel text, the target's words that differ)."""
    rng = np.random.default_rng(seed)
    words, made, targets = tables_of(rng, cores, jobs, pages)
    bits, values = bf16_pages(rng, pages)
    report = {"cores": cores, "jobs": len(made), "target_pages": targets}
    tables = session.tensor(words, "uint32")
    source = session.empty((pages, 1024), "bfloat16")
    session.write(source, bits.view(np.uint8).reshape(pages, -1))
    target = session.empty((targets, 1024), "bfloat16")
    program = TableJobs(
        tables=tables, source=source, target=target, session=session, log=log
    )
    loaded = session.load(program)
    try:
        event = loaded.wait(
            loaded.submit(TableJobs.walk, (0, cores), args=(tables, source, target)),
            timeout=30.0,
        )
        report["event"], report["microseconds"] = event.kind, event.microseconds
    finally:
        loaded.stop()
    report["kernel_text"] = program.sizes.text
    (got,) = session.read(target)
    got = got.view(np.uint16).reshape(targets, 1024)
    want = np.zeros((targets, 1024), np.float32)
    for first, count, at, add in made:
        want[at : at + count] = values[first : first + count] + BIAS * add
    want_bits = (want.view(np.uint32) >> 16).astype(np.uint16)
    report["wrong"] = int((got != want_bits).sum())
    report["ok"] = report["wrong"] == 0 and report["event"] == "done"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=3, help="jobs a core walks")
    parser.add_argument("--pages", type=int, default=64, help="source pages")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    with km.Session(log=log) as session:
        report = run(session, args.cores, args.jobs, args.pages, args.seed)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
