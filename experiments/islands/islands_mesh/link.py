"""Both islands of one fabric mesh, crossing the link from inside a run: on each island
a link core opens a fabric connection in a handler, SEND commands write 64 bytes into
the other island's link core's L1, a COPY handler there puts them in DRAM for the
host, both ways, then the connections close. Prints one JSON line. See README.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.device import native

KERNELS = Path(__file__).resolve().parent / "kernels"
ROW_BYTES = 512  # islands.h: kRowBytes, kRowSlot, kNoLink
ROW_SLOT = 256
NO_LINK = 0xFFFFFFFF
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rounds", type=int, default=8, help="SEND then COPY")
    parser.add_argument("--sends", type=int, default=1000, help="pipelined, timed")
    parser.add_argument("--link", type=int, default=0)
    return parser.parse_args()


def slot_words(seed):
    """islands.h slot_word(): the 16 words of seed `seed`."""
    words = (seed * 0x9E3779B1 + np.arange(16, dtype=np.uint64)) % (1 << 32)
    words[0] = seed
    return words.astype(np.uint32)


class Island:
    """The experiment's program on one island, its link core under the link's
    Ethernet core in row 0, its row in every core's L1 and its DRAM copies."""

    def __init__(self, index, chip, device, link, pages):
        self.index = index
        self.session = km.Session(card=chip, device=device, log=self.log)
        self.grid = self.session.grid
        self.links, self.channel, eth_x, _ = link
        self.core = self.link_core(eth_x)
        self.noc = self.session.device.worker_core(*self.core)
        self.rank = self.core[1] * self.grid[0] + self.core[0]
        self.row = self.session.empty(
            (self.grid[0] * self.grid[1], ROW_BYTES // 4),
            "uint32",
            shard_grid=self.grid,
        )
        self.out = self.session.tensor(np.zeros((pages, 16), np.uint32), "uint32")
        row = self.row.address

        def args(core):
            if core == self.rank:
                return [], [self.channel, row, row + 16, row], []
            return [], [NO_LINK, 0, 0, row], []

        self.program = km.Program(
            "islands",
            kernels=KERNELS,
            args=args,
            compile_args={"writer": self.out.accessor_args()},
            handlers=KERNELS / "islands.h",
        )
        self.run = None

    def log(self, message):
        log(f"island {self.index}: {message}")

    def link_core(self, eth_x):
        """The row-0 Tensix core nearest at or right of the Ethernet core's x."""
        device = self.session.device
        columns = range(self.grid[0])
        right = [x for x in columns if device.worker_core(x, 0)[0] >= eth_x]
        if right:
            return min(right, key=lambda x: device.worker_core(x, 0)[0]), 0
        return max(columns, key=lambda x: device.worker_core(x, 0)[0]), 0

    def load(self):
        self.run = self.session.load(self.program, deadline=1.0)
        self.log(
            f"loaded; link core {self.core} NoC {tuple(self.noc)}, channel "
            f"{self.channel} of {self.links} links"
        )

    def command(self, handler, args=()):
        """One command to the link core, waited for; its event."""
        handle = self.run.submit(
            self.program.handler(handler), (self.rank, 1), list(args)
        )
        return self.run.wait(handle, timeout=5.0)

    def send_to(self, other, seed):
        x, y = other.noc
        return self.command("send", [x, y, other.row.address + ROW_SLOT, seed])

    def copy(self, page, seed):
        return self.command("copy", [self.out.address, page, 0, seed])


def main():
    args = parse()
    report = {"rounds": args.rounds, "sends": args.sends, "link": args.link}
    module = native.load_module()
    log("opening 2 islands as one 1 x 2 fabric mesh")
    mesh = module.Islands(2)
    chips = mesh.chips()
    log(f"mesh open: chips {chips}")
    islands = []
    try:
        mesh.open_x280()
        log("X280 runtime open over the mesh")
        links = [mesh.link(0, 1, args.link), mesh.link(1, 0, args.link)]
        report["links"] = [list(link) for link in links]
        log(f"links (count, channel, eth x, eth y): {report['links']}")
        pages = 2 * args.rounds + 2
        islands = [
            Island(index, chips[index], mesh.device(index), links[index], pages)
            for index in range(2)
        ]
        for island in islands:
            island.load()
        first, second = islands
        for island in islands:
            event = island.command("open")
            log(f"island {island.index}: OPEN {event.kind}")
        expected = {}
        kinds = []
        for r in range(args.rounds):
            for sender, receiver, base in ((first, second, 0), (second, first, 1)):
                seed = 0x10000 * (base + 1) + r + 1
                kinds.append(sender.send_to(receiver, seed).kind)
                page = base * args.rounds + r
                kinds.append(receiver.copy(page, seed).kind)
                expected[(receiver.index, page)] = seed
        log(f"{args.rounds} rounds each way: events {sorted(set(kinds))}")
        report["sent_us"] = []
        for sender, receiver, base in ((first, second, 0), (second, first, 1)):
            seeds = 0x40000 * (base + 1) + np.arange(args.sends) + 1
            x, y = receiver.noc
            words = np.stack(
                [
                    np.full(args.sends, x),
                    np.full(args.sends, y),
                    np.full(args.sends, receiver.row.address + ROW_SLOT),
                    seeds,
                ]
            ).T
            started = time.perf_counter()
            sender.run.submit_many(
                sender.program.handler("send"),
                np.full(args.sends, sender.rank),
                np.ones(args.sends, np.int64),
                words,
            )
            waited = sender.run.drain(timeout=20.0)
            spent = (time.perf_counter() - started) * 1e6 / args.sends
            report["sent_us"].append(round(spent, 3))
            if waited != args.sends:
                raise RuntimeError(f"{waited} SEND events of {args.sends}")
            page = 2 * args.rounds + base
            kinds.append(receiver.copy(page, int(seeds[-1])).kind)
            expected[(receiver.index, page)] = int(seeds[-1])
        log(f"{args.sends} SENDs each way: {report['sent_us']} us a command")
        for island in islands:
            event = island.command("close")
            log(f"island {island.index}: CLOSE {event.kind}")
            kinds.append(event.kind)
        report["events"] = sorted(set(kinds))
        results = []
        for island in islands:
            island.run.stop()
            results.append(island.run.result()["error_name"])
        report["kernel_error"] = results
        wrong = 0
        for island in islands:
            (got,) = island.session.read(island.out)
            got = got.view(np.uint32).reshape(-1, 16)
            for (index, page), seed in expected.items():
                if index == island.index:
                    wrong += int(not np.array_equal(got[page], slot_words(seed)))
        report["checked"] = len(expected)
        report["wrong"] = wrong
        log(f"checked {len(expected)} slots read back, {wrong} wrong")
    finally:
        for island in islands:
            if island.run is not None and not island.run.stopped:
                island.run.abort()
            island.session.close()
        log("closing the mesh")
        mesh.close()
        log("mesh closed")
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    ok = (
        report.get("wrong") == 0
        and report.get("events") == ["done"]
        and report.get("kernel_error") == ["none", "none"]
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
