"""What moving bundles of rays between cores costs: routed messages of 64 bytes to 4 KiB
hopping between ranks drawn from part A's destinations (the README has the commands)."""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi import abi  # noqa: E402
from komorebi.layout import counter_tiles  # noqa: E402
from komorebi.program import Buffer, Dynamic, messages_header  # noqa: E402

KERNELS = HERE / "kernels"
RUNTIME = ROOT / "src" / "komorebi_native" / "kernels"
TABLE = KERNELS / "destinations.h"
TABLE_ENTRIES = 1024
PAGE_LIMIT = 256  # larger messages take the page-sized copy of the kernels
STARTED = time.monotonic()
VARIANTS = {
    "runtime": {
        "reader": RUNTIME / "messages_reader.cpp",
        "compute": KERNELS / "compute.cpp",
        "writer": RUNTIME / "messages_writer.cpp",
    },
    "pages": {
        "reader": KERNELS / "pages_reader.cpp",
        "compute": KERNELS / "compute_pages.cpp",
        "writer": KERNELS / "pages_writer.cpp",
    },
    "zero-copy": {
        "reader": KERNELS / "zero_copy_reader.cpp",
        "compute": KERNELS / "compute_zero_copy.cpp",
        "writer": KERNELS / "zero_copy_writer.cpp",
    },
    "bodies": {
        "reader": RUNTIME / "messages_reader.cpp",
        "compute": KERNELS / "compute_bodies.cpp",
        "writer": RUNTIME / "messages_writer.cpp",
    },
}
LINE_BYTES = 64  # the bodies variant's line beside its body


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def page_constants():
    """The constants of the page-sized copy of the runtime's messages."""
    path = KERNELS / "page_messages.hpp"
    fields = abi.struct_fields(path, "MessagesArgs")
    return abi.constants(path, sizes={"MessagesArgs": 4 * len(fields)})


@dataclass(frozen=True)
class PageDynamic(Dynamic):
    """A dynamic handler on the page-sized copy: its own message limit and its own
    packet size (messages of a packet and of a bin) for the buffers."""

    def __post_init__(self):
        constants = page_constants()
        if self.message_bytes % 64 or not (
            64 <= self.message_bytes <= constants["kMessagesMessageMax"]
        ):
            raise ValueError(f"message_bytes {self.message_bytes}: not a page size")
        if not 1 <= self.bins <= constants["kMessagesBinsMax"]:
            raise ValueError(f"bins {self.bins}: not in 1..16")

    def args(self, queue, lines, body=0, x280=None):
        """The copy's own km::MessagesArgs: the runtime's words before its bodies."""
        fields = abi.struct_fields(KERNELS / "page_messages.hpp", "MessagesArgs")
        return super().args(queue, lines)[: len(fields)]

    def buffers(self):
        constants = page_constants()
        packet = constants["kMessagesPacket"]
        return [
            Buffer(
                constants["kCbMessagesIn"],
                "uint32",
                constants["kMessagesHeader"] + packet * self.message_bytes,
                2,
            ),
            Buffer(
                constants["kCbMessagesOut"], "uint32", self.message_bytes, 2 * packet
            ),
            Buffer(
                constants["kCbMessagesBins"],
                "uint32",
                self.bins * packet * self.message_bytes,
                1,
            ),
        ]


def destinations(arguments):
    """Writes kernels/destinations.h: 1024 ranks in proportion to the bundles each core
    received in one configuration of a part A run."""
    report = json.loads(Path(arguments.paths).read_text())
    trace = report["scenes"][arguments.scene]["budgets"][arguments.budget]["traces"][
        arguments.trace
    ]
    placement = trace["placement"]
    stats = trace["by_cycles"][placement]["bundles"][arguments.bundle]
    received = np.asarray(stats["received_per_core"], dtype=np.float64)
    share = received / received.sum()
    counts = np.floor(share * TABLE_ENTRIES).astype(np.int64)
    rest = np.argsort(-(share * TABLE_ENTRIES - counts), kind="stable")
    counts[rest[: TABLE_ENTRIES - counts.sum()]] += 1
    ranks = np.repeat(np.arange(len(counts)), counts)
    order = np.random.default_rng(1).permutation(TABLE_ENTRIES)
    ranks = ranks[order]
    rows = [
        "    " + ", ".join(str(rank) for rank in ranks[start : start + 16]) + ","
        for start in range(0, TABLE_ENTRIES, 16)
    ]
    source = (
        f"{arguments.scene}, {arguments.budget} KiB treelets, {arguments.trace} trace,"
        f" bundles {arguments.bundle}"
    )
    text = "\n".join(
        [
            "// The transport probe's ranks, in proportion to the bundles a core got",
            f"// in part A: {source}.",
            "#pragma once",
            "",
            "#include <stdint.h>",
            "",
            f"constexpr uint32_t kDestinationCount = {TABLE_ENTRIES};",
            "constexpr uint8_t kDestinations[kDestinationCount] = {",
            *rows,
            "};",
            "",
        ]
    )
    TABLE.write_text(text)
    log(
        f"wrote {TABLE}: busiest core {counts.max()} of {TABLE_ENTRIES} entries,"
        f" mean {counts.mean():.2f}"
    )


def packet_messages(variant, pages):
    """Messages a packet holds in a run of `variant` at a page size (`pages`)."""
    if not pages:
        return 32
    if variant == "bodies":
        return messages_header()[0]["kMessagesBodyPacket"]
    return page_constants()["kMessagesPacket"]


def clock(card):
    """AICLK of the card in MHz, from the driver; None if it cannot be read."""
    path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def load_program(session, size, arguments, bundle):
    """The probe loaded for messages of `size` bytes: (run, results tensor, program)."""
    grid = session.grid
    cores = grid[0] * grid[1]
    results = session.tensor(
        np.zeros((cores, bundle["kResultWords"]), np.uint32), "uint32", shard_grid=grid
    )
    seed = session.tensor(
        np.zeros((cores, size // 4), np.uint32), "uint32", shard_grid=grid
    )
    pages = size > PAGE_LIMIT
    variant = arguments.variant if pages else "runtime"
    kernels = VARIANTS[variant]
    program = km.Program(
        "bundles",
        kernels=kernels,
        args=lambda core: ([results.address], [], [results.address, seed.address]),
        handlers={"bundle": bundle["kHandlerBundle"]},
    )
    pages = variant != "runtime"
    queue = arguments.queue_pages if pages else arguments.queue_small
    options = {"message_bytes": size}
    if variant == "bodies":  # a line, and the bundle as its body
        options = {"message_bytes": LINE_BYTES, "body_bytes": size}
    if arguments.flush_age is not None:  # the writer's flush_age (Dynamic's default)
        options["flush_age"] = arguments.flush_age
    kind = PageDynamic if variant in ("pages", "zero-copy") else Dynamic
    program.messages = kind(
        handler=bundle["kHandlerBundle"],
        max_live=queue,
        queue_messages=queue,
        bins=arguments.bins,
        counters=2,
        **options,
    )
    program.buffers = [*program.buffers, *program.messages.buffers()]
    run = session.load(program, deadline=arguments.deadline, wait=arguments.wait)
    return run, results, program, cores


def one_run(session, size, arguments, bundle, message_count, forwards, card):
    """One load and one RUN of the probe, results read after the stop: its card time,
    the clock and the checks of every core."""
    log(f"loading the probe for {size}-byte messages")
    run, results, program, cores = load_program(session, size, arguments, bundle)
    log(f"loaded: {cores} cores, quota {program.messages.quota(cores)}")
    before = clock(card)
    log(f"submit: {message_count} messages, {forwards} forwards each")
    handle = run.submit("bundle", (0, cores), [message_count, forwards], exclusive=True)
    run.drain(timeout=arguments.deadline + 5.0)
    run.stop()
    event, sequence = run.event(handle), run.sequence(handle)
    log(f"done: {event.kind if event else None}; reading the results")
    rows = results.read().astype(np.int64)
    log("results read")
    workers = len(counter_tiles(0, cores, 2).workers)
    visits = int(rows[:workers, bundle["kResultVisits"]].sum())
    checks = {
        "event_done": bool(event and event.kind == "done"),
        "run_matches": bool((rows[:, bundle["kResultRun"]] == sequence).all()),
        "visits": visits == message_count * (forwards + 1),
        "consumed": int(rows[:workers, bundle["kResultConsumed"]].sum())
        == message_count,
        "bad": int(rows[:, bundle["kResultBad"]].sum()) == 0,
        "reader_bad": int(rows[:, bundle["kResultReaderBad"]].sum()) == 0,
        "reader_checked": arguments.variant != "zero-copy"
        or size <= PAGE_LIMIT
        or int(rows[:workers, bundle["kResultReaderChecked"]].sum()) == visits,
        "rejected": int(rows[:, bundle["kResultRejected"]].sum()) == 0,
        "counters_idle": not rows[workers:, bundle["kResultVisits"]].any(),
    }
    per_core = rows[:workers, bundle["kResultVisits"]]
    return {
        "forwards": forwards,
        "microseconds": round(event.microseconds, 1) if event else None,
        "aiclk_mhz": [before, clock(card)],
        "visits": visits,
        "visits_per_core_max_mean": [int(per_core.max()), float(per_core.mean())],
        "bad": int(rows[:, bundle["kResultBad"]].sum()),
        "pack_cycles_per_visit": {
            name: float(rows[:workers, bundle[key]].sum() / max(visits, 1))
            for name, key in (
                ("check", "kResultCheckCycles"),
                ("forward", "kResultForwardCycles"),
                ("wait", "kResultWaitCycles"),
            )
        },
        "quota": program.messages.quota(cores),
        "queue_messages": program.messages.queue_messages,
        "checks": checks,
        "ok": all(checks.values()),
    }


def measure(arguments):
    """Every size: best of `repeats` pairs of runs (forwards f1 and f2), the moves a
    second from the difference of their card times."""
    bundle = abi.constants(KERNELS / "bundle.h")
    card = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
    report = {
        "load_before": Path("/proc/loadavg").read_text().split()[:3],
        "card": card,
        "sizes": {},
    }
    with km.Session(log=log) as session:
        log("device open")
        for size in arguments.sizes:
            pages = size > PAGE_LIMIT
            message_count = (
                arguments.messages_pages if pages else arguments.messages_small
            )
            pairs = []
            for repeat in range(arguments.repeats):
                pair = [
                    one_run(
                        session, size, arguments, bundle, message_count, forwards, card
                    )
                    for forwards in arguments.forwards
                ]
                pairs.append(pair)
                times = [entry["microseconds"] for entry in pair]
                log(f"{size} B, repeat {repeat}: {times} us")
            low, high = arguments.forwards
            moves = message_count * (high - low)
            rates = [
                moves / ((pair[1]["microseconds"] - pair[0]["microseconds"]) * 1e-6)
                for pair in pairs
                if all(r["ok"] for r in pair)
            ]
            best = max(rates) if rates else None
            report["sizes"][str(size)] = {
                "kernels": arguments.variant if pages else "runtime",
                "messages": message_count,
                "packet": packet_messages(arguments.variant, pages),
                "moves_per_second": best,
                "bytes_per_second": best * size if best else None,
                "rates": rates,
                "pairs": pairs,
                "ok": len(rates) == len(pairs),
            }
            rate = f"{best / 1e6:.2f} M" if best else "no exact pair, no"
            log(f"{size} B: {rate} moves/s")
    report["load_after"] = Path("/proc/loadavg").read_text().split()[:3]
    report["ok"] = all(entry["ok"] for entry in report["sizes"].values())
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "sizes": {
                    size: entry["moves_per_second"]
                    for size, entry in report["sizes"].items()
                },
            }
        ),
        flush=True,
    )
    if arguments.json:
        Path(arguments.json).write_text(json.dumps(report, indent=1))
    return report["ok"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    table = commands.add_parser("destinations")
    table.add_argument("paths")
    table.add_argument("--scene", default="sponza_overview")
    table.add_argument("--budget", default="128")
    table.add_argument("--trace", default="bounce")
    table.add_argument("--bundle", default="32/full")
    probe = commands.add_parser("run")
    probe.add_argument("--sizes", nargs="+", type=int, default=[64, 256, 2048, 4096])
    probe.add_argument("--forwards", nargs=2, type=int, default=[100, 400])
    probe.add_argument("--repeats", type=int, default=3)
    probe.add_argument("--messages-small", type=int, default=8192)
    probe.add_argument("--messages-pages", type=int, default=2048)
    probe.add_argument("--queue-small", type=int, default=16384)
    probe.add_argument("--queue-pages", type=int, default=4096)
    probe.add_argument("--bins", type=int, default=16)
    probe.add_argument("--flush-age", type=int, default=None)
    probe.add_argument("--deadline", type=float, default=10.0)
    probe.add_argument("--wait", type=float, default=1.0)
    probe.add_argument(
        "--variant", choices=["pages", "zero-copy", "bodies"], default="pages"
    )
    probe.add_argument("--json", default=None)
    arguments = parser.parse_args()
    if arguments.command == "destinations":
        destinations(arguments)
        return True
    return measure(arguments)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
