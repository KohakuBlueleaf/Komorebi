"""A body's own buffers, the two kinds the attention pass uses, each checked exactly
against numpy: `running`, a float32 tile written in place (each core's running sum of
its tiles, then written out); `delayed`, a FIFO from the pack CPU to the unpack CPU
(each core's tiles out one step late, its first tile zero). README.md says what the
own fields lower to.
Example: python own_class.py --cores 8 --tiles 40"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km

STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class OwnBuffers(km.Program):
    """running: the float32 sum of the core's tiles in its own field `total`, a step
    token from pack to unpack between its writes and reads; delayed: each tile pushed
    by pack into the FIFO `delay`, taken back by unpack one step later."""

    total: km.L1[km.fp32, 1, 32, 32] = km.own(push=None)  # written in place
    step: km.L1[km.bf16, 1, 32, 32] = km.own()  # pack's token: total is written
    delay: km.L1[km.bf16, 2, 32, 32] = km.own()  # pack to unpack, one step late

    @km.handler
    def running(self, source: km.Tensor, sums: km.Tensor, tiles: int):
        part = self.partition(tiles)
        km.cxx("""
            #include "api/compute/eltwise_binary_sfpu.h"
            uint32_t steps = 0;
            """)
        for tile in km.Stream(source.pages(part.start, part.count)):
            km.cxx(
                """
                tile_regs_acquire();
                reconfig_data_format_srca(tile_buffer);  // copy_init keeps the format
                copy_init(tile_buffer);
                copy_tile(tile_buffer, tile_index, 0);
                if (steps > 0) {  // the last sum packed: its token
                    cb_wait_front(step, 1);
                    cb_pop_front(step, 1);
                    reconfig_data_format_srca(total);
                    copy_init(total);
                    copy_tile(total, 0, 1);
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(total);
                pack_tile<true>(0, total, 0);
                cb_reserve_back(step, 1);
                cb_push_back(step, 1);
                tile_regs_release();
                steps++;
                """,
                takes=[tile, self.total, self.step],
            )
        whole = km.cxx(
            """
            cb_wait_front(step, 1);
            cb_pop_front(step, 1);
            reconfig_data_format_srca(total);
            copy_init(total);
            copy_tile(total, 0, 0);
            """,
            takes=[self.total, self.step],
            gives=km.Tile,
        )
        sums.pages(self.rank, 1).write(whole)

    @km.handler
    def delayed(self, source: km.Tensor, late: km.Tensor, tiles: int):
        part = self.partition(tiles)
        km.cxx("""
            #include "api/compute/eltwise_unary/fill.h"
            uint32_t steps = 0;
            """)
        written = km.Stream(late.pages(part.start, part.count), write=True)
        for tile in km.Stream(source.pages(part.start, part.count)):
            km.cxx(
                """
                tile_regs_acquire();
                copy_init(tile_buffer);
                copy_tile(tile_buffer, tile_index, 0);
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(delay, 1);
                pack_reconfig_data_format(delay);
                pack_tile(0, delay);
                cb_push_back(delay, 1);
                tile_regs_release();
                """,
                takes=[tile, self.delay],
            )
            earlier = km.cxx(
                """
                if (steps == 0) {
                    fill_tile_init();
                    fill_tile(0, 0.0f);
                } else {  // the tile pushed a step ago
                    cb_wait_front(delay, 1);
                    copy_init(delay);
                    copy_tile(delay, 0, 0);
                    cb_pop_front(delay, 1);
                }
                steps++;
                """,
                takes=[self.delay],
                gives=km.Tile,
            )
            written.push(earlier)
        km.cxx(
            """
            if (steps > 0) {  // the last tile, never written
                cb_wait_front(delay, 1);
                cb_pop_front(delay, 1);
            }
            """,
            takes=[self.delay],
        )


def bf16_pages(rng, pages):
    """`pages` tiles of integers -4 to 4 as bfloat16 bits, and their values."""
    values = rng.integers(-4, 5, size=(pages, 1024)).astype(np.float32)
    return (values.view(np.uint32) >> 16).astype(np.uint16), values


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--tiles", type=int, default=40, help="tiles a core takes")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    pages = args.cores * args.tiles
    bits, values = bf16_pages(np.random.default_rng(args.seed), pages)
    report = {"cores": args.cores, "tiles": args.tiles}
    with km.Session(log=log) as session:
        source = session.empty((pages, 1024), "bfloat16")
        session.write(source, bits.view(np.uint8).reshape(pages, -1))
        sums = session.empty((args.cores, 1024), "float32")
        late = session.empty((pages, 1024), "bfloat16")
        program = OwnBuffers(
            source=source, sums=sums, late=late, session=session, log=log
        )
        run = session.load(program)
        try:
            group = (0, args.cores)
            for handler, out in (
                (OwnBuffers.running, sums),
                (OwnBuffers.delayed, late),
            ):
                event = run.wait(
                    run.submit(handler, group, args=(source, out, pages)), timeout=30.0
                )
                report[f"{handler.name}_event"] = event.kind
                report[f"{handler.name}_microseconds"] = event.microseconds
        finally:
            run.stop()
        report["kernel_text"] = program.sizes.text
        (got,) = session.read(sums)
        got = got.view(np.float32).reshape(args.cores, 1024)
        want = values.reshape(args.cores, args.tiles, 1024).sum(axis=1)
        report["running_wrong"] = int((got != want).sum())
        report["running_first"] = [got.flat[:4].tolist(), want.flat[:4].tolist()]
        (got,) = session.read(late)
        got = got.view(np.uint16).reshape(args.cores, args.tiles, 1024)
        want = np.zeros_like(got)
        want[:, 1:] = bits.reshape(args.cores, args.tiles, 1024)[:, :-1]
        report["delayed_wrong"] = int((got != want).sum())
    report["ok"] = (
        report["running_wrong"] == 0
        and report["delayed_wrong"] == 0
        and report["running_event"] == report["delayed_event"] == "done"
    )
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
