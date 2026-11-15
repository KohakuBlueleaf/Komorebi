"""A table of jobs on the movers and compute, and a gather of a rectangle's ranges
(rows with a step, K transposed and V cast as one block), each checked against numpy.
README.md says what each line lowers to.
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
R, DT, DV, KB = 4, 2, 2, 3  # gather: Q's most rows, Q / K and V's columns, keys a chunk
SLOT = KB * DT + KB * DV  # a cast's block: K's most tiles transposed, then V's
QB = R * DT  # a block of Q's rows
COLUMNS = 2  # gather's rectangle: two columns of two cores
RECT = (0, 0, COLUMNS, 2)


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class Job(km.Entry):
    """A job: its source pages (first, count), its target's first page, and whether
    the constant tile is added."""

    source: int
    count: int
    target: int
    add: int


class TableJobs(km.Program, defines={"TABLE_BIAS": f"{BIAS}f", "SLOT_PAGES": SLOT}):
    """walk: each job's pages, plus the constant tile (built at compute's start) when
    it adds it; gather: Q's rows `stride` apart, then K's and V's chunks, copied."""

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

    @km.handler
    def gather(
        self, q: km.Tensor, k: km.Tensor, v: km.Tensor, qo: km.Tensor,
        slots: km.Tensor, rows: int, stride: int, keys: int, chunks: int,
    ):  # fmt: skip
        down = km.Broadcast(self, along="column")  # a core outside the rectangle skips
        r0 = self.y * COLUMNS + self.x  # this core's place in the rectangle
        copied = km.Stream(
            qo.tiles[r0 : min(r0 + R * stride, r0 + rows * stride) : stride, 0:DT],
            write=True,
            block=QB,
        )
        for tile in km.Stream(
            q.tiles[r0 : min(r0 + R * stride, r0 + rows * stride) : stride, 0:DT],
            block=QB,
        ):
            copied.push(tile)
        for chunk in range(chunks):
            k0 = chunk * KB
            if down.sender():
                kv = down.cast(
                    k.tiles[k0 : min(k0 + KB, keys), 0:DT].T,
                    v.tiles[k0 : min(k0 + KB, keys), 0:DV],
                )
            else:
                kv = down.arrive()
            seen = km.Stream(
                slots.pages((r0 * chunks + chunk) * SLOT, SLOT),
                write=True,
                block=SLOT,
            )
            km.cxx(
                """
                for (uint32_t page = 0; page < SLOT_PAGES; page++) {
                    tile_regs_acquire();
                    copy_init(kv_buffer);
                    copy_tile(kv_buffer, kv + page, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    if (seen != 0) {
                        pack_tile(0, seen_buffer);
                    }
                    tile_regs_release();
                }
                """,
                takes=[kv, seen],
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


def gather_inputs(rng, rows, stride, keys):
    """gather's tensors' pages (Q, K, V as bfloat16 bits) and its expected pages: Q's
    copy (QO) and each core's slots, with which slot pages each chunk fills."""
    q_rows = RECT[2] * RECT[3] + (rows - 1) * stride  # the last core's last row, + 1
    k_rows = -(-keys // KB) * KB
    q, _ = bf16_pages(rng, q_rows * DT)
    k, _ = bf16_pages(rng, k_rows * DT)
    v, _ = bf16_pages(rng, k_rows * DV)
    cores, chunks = RECT[2] * RECT[3], -(-keys // KB)
    qo = np.zeros_like(q)
    for rank in range(cores):
        for row in range(rank, rank + rows * stride, stride):
            qo[row * DT : (row + 1) * DT] = q[row * DT : (row + 1) * DT]
    slots = np.zeros((cores * chunks * SLOT, 1024), np.uint16)
    filled = np.zeros(cores * chunks * SLOT, bool)
    for rank in range(cores):
        for chunk in range(chunks):
            base, k0 = (rank * chunks + chunk) * SLOT, chunk * KB
            for key in range(min(KB, keys - k0)):
                for h in range(DT):  # K transposed: column h a row of the slot
                    slots[base + h * KB + key] = k[(k0 + key) * DT + h]
                    filled[base + h * KB + key] = True
                for c in range(DV):  # then V's rows, after K's most
                    at = base + KB * DT + key * DV + c
                    slots[at], filled[at] = v[(k0 + key) * DV + c], True
    return {"q": q, "k": k, "v": v}, qo, slots, filled, chunks


def slot_pages(got, want, filled, inputs):
    """Each filled slot page that differs: (core, chunk, slot page) and the input
    page it holds instead (K's or V's page, zeros, or none of them)."""
    found = {}
    for name in ("k", "v"):
        for index, page in enumerate(inputs[name]):
            found.setdefault(page.tobytes(), f"{name}{index}")
    chunks = len(got) // (RECT[2] * RECT[3] * SLOT)
    wrong = []
    for at in np.flatnonzero(filled & (got != want).any(axis=1)):
        core, rest = divmod(int(at), chunks * SLOT)
        held = found.get(got[at].tobytes(), "zeros" if not got[at].any() else "other")
        wrong.append(f"{core}/{rest // SLOT}/{rest % SLOT}={held}")
    return wrong


def submitted(loaded, handler, group, args):
    """One command of `handler` on `group`: its event's kind and microseconds."""
    event = loaded.wait(loaded.submit(handler, group, args=args), timeout=30.0)
    return event.kind, event.microseconds


def run(session, cores=8, jobs=3, pages=64, seed=1, rows=3, stride=4, keys=7):
    """The class made and loaded once; walk run on `cores` cores, gather on the
    rectangle: the report (each command's event and time, the kernel text, the
    words that differ from numpy's)."""
    rng = np.random.default_rng(seed)
    words, made, targets = tables_of(rng, cores, jobs, pages)
    bits, values = bf16_pages(rng, pages)
    inputs, qo_want, slots_want, filled, chunks = gather_inputs(rng, rows, stride, keys)
    report = {"cores": cores, "jobs": len(made), "target_pages": targets}
    tables = session.tensor(words, "uint32")
    source = session.empty((pages, 1024), "bfloat16")
    session.write(source, bits.view(np.uint8).reshape(pages, -1))
    target = session.empty((targets, 1024), "bfloat16")
    widths = {"q": DT, "k": DT, "v": DV}
    tiled = {}
    for name, pages_of in inputs.items():
        tiled[name] = session.empty(
            (len(pages_of) // widths[name] * 32, widths[name] * 32), "bfloat16", "tile"
        )
        session.write(tiled[name], pages_of.view(np.uint8).reshape(len(pages_of), -1))
    qo = session.empty(tuple(tiled["q"].shape), "bfloat16", "tile")
    session.write(qo, np.zeros((qo.pages, qo.page_bytes), np.uint8))
    slots = session.empty((len(slots_want) * 32, 32), "bfloat16", "tile")
    program = TableJobs(
        tables=tables, source=source, target=target, **tiled, qo=qo, slots=slots,
        rect=RECT, session=session, log=log,
    )  # fmt: skip
    loaded = session.load(program)
    try:
        report["event"], report["microseconds"] = submitted(
            loaded, TableJobs.walk, (0, cores), (tables, source, target)
        )
        gather = (tiled["q"], tiled["k"], tiled["v"], qo, slots)
        report["gather_event"], report["gather_microseconds"] = submitted(
            loaded, TableJobs.gather, program.rectangle["group"],
            (*gather, rows, stride, keys, chunks),
        )  # fmt: skip
    finally:
        loaded.stop()
    report["kernel_text"] = program.sizes.text
    got, got_qo, got_slots = session.read(target, qo, slots)
    got = got.view(np.uint16).reshape(targets, 1024)
    want = np.zeros((targets, 1024), np.float32)
    for first, count, at, add in made:
        want[at : at + count] = values[first : first + count] + BIAS * add
    want_bits = (want.view(np.uint32) >> 16).astype(np.uint16)
    report["wrong"] = int((got != want_bits).sum())
    got_qo = got_qo.view(np.uint16).reshape(qo_want.shape)
    report["q_rows_wrong"] = int((got_qo != qo_want).sum())
    got_slots = got_slots.view(np.uint16).reshape(slots_want.shape)
    report["slots_filled"] = int(filled.sum())
    report["slots_wrong"] = int((got_slots[filled] != slots_want[filled]).sum())
    if report["slots_wrong"]:  # each wrong page: what it holds instead
        report["slot_pages"] = slot_pages(got_slots, slots_want, filled, inputs)
    report["ok"] = (
        report["wrong"] == report["q_rows_wrong"] == report["slots_wrong"] == 0
        and report["event"] == report["gather_event"] == "done"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=3, help="jobs a core walks")
    parser.add_argument("--pages", type=int, default=64, help="source pages")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rows", type=int, default=3, help=f"gather: Q rows, <= {R}")
    parser.add_argument("--stride", type=int, default=4, help="gather: rows apart")
    parser.add_argument("--keys", type=int, default=7, help="gather: K and V rows")
    args = parser.parse_args()
    with km.Session(log=log) as session:
        report = run(
            session, args.cores, args.jobs, args.pages, args.seed, args.rows,
            args.stride, args.keys,
        )  # fmt: skip
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
