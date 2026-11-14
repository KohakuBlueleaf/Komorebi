"""Messages across the link service between two islands: origins on island 0 send
messages to a partner on island 1 and back through each island's link core, and the
same on one island for comparison; the time a crossing and the messages a second.
Prints one JSON line. See README.md.
"""

import argparse
import json
import secrets
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import abi, native

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
DEVICE = HERE.parents[1] / "src" / "komorebi_native" / "include" / "komorebi" / "device"
CROSS = abi.constants(KERNELS / "cross.h")
LINK_ARGS = 12  # km_link.hpp: LinkArgs
LINK_TEARDOWN, LINK_CURSOR = 0x60, 0x70  # its connection's words in the region
LINK_WORDS = 52  # the region's words before its headers, copied into the results
LINK_STAMPS = 0xC0 // 4  # kLinkStamps: first send, first data, first count, up
LINK_GAVE = 0x90 // 4  # kLinkGave: the reader's, then the writer's 4 words
WHERE_LINK_UP = 60  # kWhereLinkUp
LINK_OUTBOX_AT = 0x190  # kLinkOutbox
PACKET_BYTES = 4352  # kLinkPacketBytes
LINK_TAIL = 0x20 + 2 * 16 * 256  # the reports and two passes of status lines
PREFIX = 16  # kLinkPrefix
MESSAGE_BYTES = 64
RESULT_WORDS = 68  # cross.h's 16, then the link core's region words
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="mode:origins:messages:trips:packet, mode 'across' or 'local'; repeated",
    )
    parser.add_argument("--outbox", type=int, default=128, help="entries")
    parser.add_argument("--inbox", type=int, default=4, help="packets")
    parser.add_argument("--queue", type=int, default=16384, help="queue messages")
    parser.add_argument("--idle", type=int, default=20000, help="idle cycles")
    parser.add_argument("--deadline", type=float, default=2.0, help="a RUN, seconds")
    parser.add_argument("--link", type=int, default=0)
    return parser.parse_args()


def aiclk(chip):
    """The chip's AICLK in MHz, from the driver."""
    path = Path(f"/sys/class/tenstorrent/tenstorrent!{chip}/tt_aiclk")
    return int(path.read_text().split()[0])


class Island:
    """The experiment's program on one island: its link core in row 0 under the link's
    Ethernet core, the link's region in every core's L1, the results."""

    def __init__(self, index, chip, device, link, args):
        self.index, self.chip, self.args = index, chip, args
        self.session = km.Session(card=chip, device=device, log=self.log)
        self.grid = self.session.grid
        self.cores = self.grid[0] * self.grid[1]
        self.links, self.channel, eth_x, _ = link
        self.core = self.link_core(eth_x)
        self.noc = tuple(self.session.device.worker_core(*self.core))
        self.rank = self.core[1] * self.grid[0] + self.core[0]
        entry = PREFIX + MESSAGE_BYTES
        region = LINK_OUTBOX_AT + args.outbox * entry + (args.inbox + 1) * PACKET_BYTES
        region += LINK_TAIL
        self.region = self.session.empty(
            (self.cores, (region + 63) // 64 * 16), "uint32", shard_grid=self.grid
        )
        self.results = self.session.empty(
            (self.cores, RESULT_WORDS), "uint32", shard_grid=self.grid
        )
        self.far = None
        self.program = None
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

    def link_words(self, packet, token):
        """LinkArgs of every core of this island."""
        region = self.region.address
        x, y = self.far.noc
        return [
            self.rank,
            region,
            self.args.outbox,
            self.args.inbox,
            packet,
            self.index,
            x | y << 16,
            self.far.region.address,
            self.channel,
            region + LINK_TEARDOWN,
            region + LINK_CURSOR,
            token,
        ]

    def load(self, far, packet, token):
        self.far = far
        words = self.link_words(packet, token)
        assert len(words) == LINK_ARGS
        results = self.results.address

        def args(core):
            return words, words, [results, self.index, self.region.address]

        self.program = km.Program(
            "link_service",
            kernels={
                "reader": KERNELS / "reader.cpp",
                "writer": KERNELS / "writer.cpp",
                "compute": KERNELS / "compute.cpp",
            },
            args=args,
            handlers=KERNELS / "cross.h",
        ).dynamic(
            "cross",
            message_bytes=MESSAGE_BYTES,
            queue_messages=self.args.queue,
            max_live=self.args.queue,
            idle_cycles=self.args.idle,
            counters=2,
        )
        self.run = self.session.load(self.program, deadline=self.args.deadline)
        self.log(
            f"loaded; link core {self.core} NoC {self.noc}, rank {self.rank}, channel "
            f"{self.channel} of {self.links} links, region at {self.region.address:#x}"
        )

    def submit(self, words):
        return self.run.submit("cross", (0, self.cores), words, exclusive=True)

    def link_state(self):
        """The link core's region words, read without the results' copy."""
        view = self.session.empty(
            (self.cores, LINK_WORDS),
            "uint32",
            shard_grid=self.grid,
            address=self.region.address,
        )
        got = view.read().astype(np.int64) & 0xFFFFFFFF
        return got.reshape(self.cores, LINK_WORDS)[self.rank]

    def read(self):
        got = self.results.read().astype(np.int64) & 0xFFFFFFFF
        return got.reshape(self.cores, RESULT_WORDS)


def case_words(spec, link_rank):
    """(name, words, origins, messages, trips) of a --case spec."""
    mode, origins, messages, trips, packet = spec.split(":")
    across = {"across": CROSS["kModeAcross"], "local": CROSS["kModeLocal"]}[mode]
    origins, messages, trips, packet = map(int, (origins, messages, trips, packet))
    words = [origins, messages, trips, across, link_rank, packet]
    return mode, words, origins, messages, trips


def check(rows, ranks, link_rank, origins, messages, trips, sequence):
    """The wrong origins of island 0 ([rank, its result words]) and the case's span."""
    wrong = []
    firsts, lasts = [], []
    ordinals = [rank for rank in range(ranks) if rank != link_rank][:origins]
    for ordinal, rank in enumerate(ordinals):
        row = rows[rank]
        ids = np.arange(ordinal * messages, (ordinal + 1) * messages).sum()
        got = int(row[CROSS["kResIdsLo"]]) | int(row[CROSS["kResIdsHi"]]) << 32
        if (
            row[CROSS["kResRun"]] != sequence
            or row[CROSS["kResConsumed"]] != messages
            or got != ids
            or row[CROSS["kResWrong"]] != 0
            or row[CROSS["kResRejected"]] != 0
        ):
            wrong.append([rank, row[: CROSS["kResWords"]].tolist()])
        firsts.append(int(row[CROSS["kResFirst"]]))
        lasts.append(int(row[CROSS["kResLast"]]))
    start = min(firsts)
    span = max((last - start) % (1 << 32) for last in lasts)
    return wrong, span


def main():
    args = parse()
    cases = args.case or ["across:1:1:1:54"]
    report = {"cases": [], "outbox": args.outbox, "inbox": args.inbox}
    module = native.load_module()
    log("opening 2 islands as one 1 x 2 fabric mesh")
    mesh = module.Islands(2)
    chips = mesh.chips()
    islands = []
    ok = True
    try:
        mesh.open_x280()
        links = [mesh.link(0, 1, args.link), mesh.link(1, 0, args.link)]
        log(f"links (count, channel, eth x, eth y): {[list(x) for x in links]}")
        islands = [
            Island(index, chips[index], mesh.device(index), links[index], args)
            for index in range(2)
        ]
        packet_max = PACKET_BYTES // (PREFIX + MESSAGE_BYTES)
        first, second = islands
        ranks = first.cores - 2
        for spec in cases:
            token = secrets.randbits(31) | 1  # this load's hello, both sides
            for index, island in enumerate(islands):
                island.load(islands[1 - index], packet_max, token)
            mode, words, origins, messages, trips = case_words(spec, first.rank)
            clock = aiclk(first.chip)
            log(f"case {spec}: submitting")
            started = time.perf_counter()
            handles = []
            if mode == "across":
                handles.append((second, second.submit(words)))
            handles.append((first, first.submit(words)))
            kinds, events, sequences = set(), {}, {}
            for island, handle in handles:  # both runs start before either is waited on
                island.run.store()
                island.log(f"started at {time.monotonic() - STARTED:.4f} s")
            for (
                island,
                handle,
            ) in handles:  # drain: the event and sequence stay readable
                island.run.drain(timeout=args.deadline + 5.0)
                event = island.run.event(handle)
                sequences[island.index] = island.run.sequence(handle)
                island.run.forget([handle])
                kinds.add("none" if event is None else event.kind)
            host_us = (time.perf_counter() - started) * 1e6
            kinds = sorted(kinds)
            log(f"case {spec}: events {kinds}, {host_us:.0f} us; stopping")
            errors = []
            for island in islands:
                if kinds == ["done"]:
                    island.run.stop()
                    errors.append(island.run.result()["error_name"])
                elif not island.run.stopped:
                    island.run.abort()
            log(f"case {spec}: kernels {errors}; reading results")
            for island in islands:
                state = island.link_state()
                island.log(f"stamps (send, data, count, up) {state[LINK_STAMPS:][:4]}")
                for side, name in ((0, "reader"), (1, "writer")):
                    gave = state[LINK_GAVE + 4 * side :][:4].tolist()
                    if gave[0] == WHERE_LINK_UP:
                        island.log(f"link {args.link} not up ({name})")
            if kinds != ["done"]:
                for island in islands:
                    island.log(f"link words {island.link_state().tolist()}")
                    island.log(f"rank 0 results {island.read()[0][:16].tolist()}")
                ok = False
                break
            rows = first.read()
            far_rows = second.read() if mode == "across" else None
            sequence = sequences.get(first.index)
            wrong, span = check(
                rows, ranks, first.rank, origins, messages, trips, sequence
            )
            seconds = span / (clock * 1e6)
            each_way = origins * messages * trips
            entry = {
                "case": spec,
                "events": kinds,
                "kernel_error": errors,
                "wrong_origins": len(wrong),
                "wrong_rows": wrong[:4],
                "sequence": sequence,
                "aiclk_mhz": [clock, aiclk(first.chip)],
                "span_us": round(seconds * 1e6, 2),
                "host_us": round(host_us, 1),
                "us_a_crossing": round(seconds * 1e6 / (2 * trips), 3),
                "messages_a_second_each_way": round(each_way / seconds),
                "link": [
                    rows[first.rank][16 : 16 + LINK_WORDS].tolist(),
                    (
                        far_rows[second.rank][16 : 16 + LINK_WORDS].tolist()
                        if far_rows is not None
                        else None
                    ),
                ],
            }
            if far_rows is not None:
                entry["partner_visits"] = int(far_rows[:, CROSS["kResVisits"]].sum())
                entry["far_wrong"] = int(far_rows[:, CROSS["kResWrong"]].sum())
                ok = ok and entry["partner_visits"] == each_way
                ok = ok and entry["far_wrong"] == 0
            ok = ok and not wrong and errors == ["none", "none"]
            report["cases"].append(entry)
            log(f"case {spec}: {json.dumps(entry)}")
    finally:
        for island in islands:
            if island.run is not None and not island.run.stopped:
                island.run.abort()
            island.session.close()
        log("closing the mesh")
        mesh.close()
        log("mesh closed")
    report["ok"] = ok
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
