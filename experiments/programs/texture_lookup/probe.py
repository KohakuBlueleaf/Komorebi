"""What a texture lookup costs on a core: the 1,024-pixel loop (loop), a page from DRAM
(miss), texel x light term (colour), real tiles' lookups (replay). One JSON line.

Example: python probe.py loop --cores 1; python probe.py replay --streams s.npz"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import abi
from komorebi.device import mock

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
K = abi.constants(KERNELS / "lookup.h")
SEED = 20261114
ROW = 12  # the cores of one grid row: the most this probe runs on
CPUS = {"reader": 0, "writer": 1, "unpack": 2, "pack": 3}
FORMS = {"direct": 0, "hash": 1, "hash_last": 2}
SLOTS = (8, 16, 32)
GOLDEN = 0x9E3779B1
T0 = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - T0:7.2f}] {message}", flush=True)


class Clock:
    """AICLK (MHz) and card power (W) sampled every 50 ms while it runs."""

    def __init__(self):
        card = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
        base = Path(f"/sys/class/tenstorrent/tenstorrent!{card}")
        self.aiclk = base / "tt_aiclk"
        hwmon = next(iter(sorted((base / "device" / "hwmon").glob("hwmon*"))))
        self.power = hwmon / "power1_input"
        self.samples, self.stop = [], threading.Event()

    def read(self):
        mhz = int(self.aiclk.read_text().split()[0])
        return mhz, int(self.power.read_text()) / 1e6

    def run(self):
        while not self.stop.wait(0.05):
            self.samples.append(self.read())

    def __enter__(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()

    def summary(self):
        samples = self.samples or [self.read()]
        mhz = np.array([s[0] for s in samples])
        watts = np.array([s[1] for s in samples])
        return {
            "aiclk_mhz_min": int(mhz.min()),
            "aiclk_mhz_mean": round(float(mhz.mean()), 1),
            "power_w_mean": round(float(watts.mean()), 1),
            "samples": len(samples),
        }


def texels(page, offset):
    """The texel word at `offset` of page `page`: distinct for every (page, offset)."""
    value = (
        np.asarray(page, np.uint64) * 1024 + np.asarray(offset, np.uint64)
    ) * 2654435761
    return ((value + 12345) & 0xFFFFFFFF).astype(np.uint32)


def set_of(key, slots):
    """The first slot of key's set in a table of `slots` slots (table.hpp)."""
    shift = {8: 31, 16: 30, 32: 29}[slots]
    return int(((int(key) * GOLDEN) & 0xFFFFFFFF) >> shift) * K["kWays"]


def keys_for(slots, rng):
    """`slots` keys, kWays to each set, in the order the kernel inserts them; and
    the slot each one takes."""
    sets = slots // K["kWays"]
    taken, keys, slot_of = [0] * sets, [], {}
    while len(keys) < slots:
        key = int(rng.integers(64, 1 << 20))
        first = set_of(key, slots)
        if key in slot_of or taken[first // K["kWays"]] == K["kWays"]:
            continue
        slot_of[key] = first + taken[first // K["kWays"]]
        taken[first // K["kWays"]] += 1
        keys.append(key)
    return keys, slot_of


def coherent(pages, rng):
    """A tile mapped one texel a pixel onto a 2 x 2 block of pages, at a random
    texel offset: the pixel's page (of `pages`, four) and offset."""
    u0, v0 = rng.integers(0, 32, 2)
    y, x = np.divmod(np.arange(K["kPixels"]), 32)
    tx, ty = u0 + x, v0 + y
    page = np.asarray(pages)[(ty >> 5) * 2 + (tx >> 5)]
    return page, (ty & 31) * 32 + (tx & 31)


def scattered(pages, rng):
    """Every pixel a random page of `pages` and a random texel."""
    page = np.asarray(pages)[rng.integers(0, len(pages), K["kPixels"])]
    return page, rng.integers(0, 1024, K["kPixels"])


def synthetic(rng):
    """The streams, the key lists and each stream's expected texels: stream 0 and 1
    direct (coherent, scattered), then per table size coherent and scattered."""
    streams = np.zeros((K["kStreams"], K["kPixels"]), np.uint32)
    expected = np.zeros_like(streams)
    keys = np.zeros((3, K["kSlotsMax"]), np.uint32)
    described = []
    for index, make in enumerate((coherent, scattered)):
        page, offset = make(range(4) if index == 0 else range(32), rng)
        streams[index] = page << 10 | offset
        expected[index] = texels(page, offset)
        described.append({"form": "direct", "stream": make.__name__})
    for list_index, slots in enumerate(SLOTS):
        chosen, slot_of = keys_for(slots, rng)
        keys[list_index, :slots] = chosen
        for half, make in enumerate((coherent, scattered)):
            index = 2 + 2 * list_index + half
            page, offset = make(chosen[:4] if half == 0 else chosen, rng)
            streams[index] = page << 10 | offset
            expected[index] = texels([slot_of[int(p)] for p in page], offset)
            described.append({"form": "hash", "slots": slots, "stream": make.__name__})
    return streams, keys, expected, described


def planes(rng):
    """A texel plane (RGBA8 words) and a light plane in [0, 1] (float32)."""
    texel = rng.integers(0, 1 << 32, K["kPixels"], dtype=np.uint64).astype(np.uint32)
    light = rng.random(K["kPixels"], dtype=np.float32)
    light[:4] = [0.0, 1.0, 0.5, 2.0**-20]
    return texel, light


def colour_integer(texel, light):
    """table.hpp's shade(): 0.16 fixed point from the float bits, rounded."""
    bits = light.view(np.uint32).astype(np.int64)
    exponent = bits >> 23
    mantissa = (bits & 0x7FFFFF) | 0x800000
    scale = mantissa >> np.where(exponent >= 103, 134 - exponent, 31)
    out = texel.astype(np.int64) & 0xFF000000
    for channel in range(3):
        value = (texel.astype(np.int64) >> (8 * channel)) & 0xFF
        out |= ((value * scale + 0x8000) >> 16) << (8 * channel)
    return out.astype(np.uint32)


def colour_vector(texel, light):
    """The vector unit's colour: each channel times the light, nearest (ties to
    even) of the exact product."""
    out = texel.astype(np.int64) & 0xFF000000
    for channel in range(3):
        value = (texel.astype(np.int64) >> (8 * channel)) & 0xFF
        lit = np.rint(value.astype(np.float64) * light.astype(np.float64))
        out |= lit.astype(np.int64) << (8 * channel)
    return out.astype(np.uint32)


def simulate(words, slots, keep_last, pool_pages, table=None):
    """The kernel's replay of one item through a table of `slots` slots (table.hpp's
    sets and least recently used): (misses, sum, table). `table` carries over."""
    if table is None:
        table = {"tag": [0] * K["kSlotsMax"], "use": [0] * K["kSlotsMax"], "clock": 0}
    tag, use = table["tag"], table["use"]
    misses, last = 0, None
    keys = (words >> 10).astype(np.int64)
    offsets = (words & 1023).astype(np.int64)
    for key in keys.tolist():
        if keep_last and key == last:
            continue
        first = set_of(key, slots)
        ways = tag[first : first + K["kWays"]]
        if key + 1 in ways:
            slot = first + ways.index(key + 1)
        else:
            slot = first
            for way in range(K["kWays"]):
                if tag[first + way] == 0:
                    slot = first + way
                    break
                if use[first + way] < use[slot]:
                    slot = first + way
            tag[slot] = key + 1
            misses += 1
        table["clock"] += 1
        use[slot] = table["clock"]
        last = key
    total = int(texels(keys % pool_pages, offsets).astype(np.uint64).sum())
    return misses, total & 0xFFFFFFFF, table


class Probe:
    """One loaded run on cores 0 to cores - 1 of the first grid row, its L1 tensors
    and the DRAM pool; command() submits one command and reads its results."""

    def __init__(self, session, cores, pool_pages, replay_rows=None):
        self.session, self.cores, self.pool_pages = session, cores, pool_pages
        rng = np.random.default_rng(SEED)
        page = np.arange(K["kSlotsMax"])[:, None]
        self.page_words = texels(page, np.arange(1024)[None, :])
        streams, keys, self.expected, self.described = synthetic(rng)
        self.texel, self.light = planes(rng)
        stream_row = np.concatenate([streams.ravel(), keys.ravel()])
        pool = texels(np.arange(pool_pages)[:, None], np.arange(1024)[None, :])
        log(f"tensors: pool {pool.nbytes >> 10} KiB to DRAM")
        self.pool = session.tensor(pool, "uint32")
        log("tensors: pool written")
        grid = (ROW, 1)

        def rows(words):
            words = np.asarray(words, np.uint32).ravel()
            return np.ascontiguousarray(np.broadcast_to(words, (ROW, words.size)))

        replay = replay_rows if replay_rows is not None else np.zeros((ROW, 16))
        self.tensors = {
            "pages": session.tensor(rows(self.page_words), "uint32", shard_grid=grid),
            "slots": session.tensor(
                rows(np.zeros(K["kSlotsMax"] * 1024)), "uint32", shard_grid=grid
            ),
            "streams": session.tensor(rows(stream_row), "uint32", shard_grid=grid),
            "planes": session.tensor(
                rows(np.concatenate([self.texel, self.light.view(np.uint32)])),
                "uint32",
                shard_grid=grid,
            ),
            "out": session.tensor(
                rows(np.zeros(K["kOutWords"])), "uint32", shard_grid=grid
            ),
            "results": session.tensor(
                rows(np.zeros(K["kResultWords"])), "uint32", shard_grid=grid
            ),
            "replay": session.tensor(
                np.ascontiguousarray(replay, np.uint32), "uint32", shard_grid=grid
            ),
        }
        log(f"tensors: L1 rows written, replay row {replay.shape[1]} words")
        self.report = session.tensor(
            np.zeros((K["kReportPages"], 1024), np.uint32), "uint32"
        )
        order = ("pages", "slots", "streams", "planes", "out", "results", "replay")
        addresses = [self.tensors[name].address for name in order]
        arguments = addresses + [self.pool.address, pool_pages, self.report.address]
        program = km.Program(
            "texture_lookup",
            kernels=KERNELS,
            buffers=[
                km.Buffer(K["kCbColour"], "uint32", 4096, 2),
                km.Buffer(K["kCbPacked"], "uint32", 4096, 2),
            ],
            args=lambda core: (arguments, arguments, arguments),
            compile_args={
                "reader": self.pool.accessor_args(),
                "writer": self.pool.accessor_args() + self.report.accessor_args(),
            },
            compute={
                "math_fidelity": "HiFi4",
                "fp32_dest_acc_en": True,
                "unpack_to_dest_mode": {K["kCbColour"]: "UnpackToDestFp32"},
            },
            handlers=KERNELS / "lookup.h",
        )
        self.run = session.load(program, deadline=5.0, wait=2.0)
        self.number = 0
        log("program loaded")

    def command(self, handler, args, out=False):
        """Runs one command on the cores, then REPORT; returns the cores' results
        [cores, words] and (with `out`) their out words, read from the DRAM report."""
        self.number += 1
        words = list(args) + [0] * (9 - len(args)) + [self.number]
        handles = [
            self.run.submit(handler, (0, self.cores), words),
            self.run.submit("report", (0, self.cores), [0] * 9 + [self.number]),
        ]
        for handle in handles:
            event = self.run.wait(handle, timeout=10.0)
            if event.kind != "done":
                raise RuntimeError(f"command {handler} {args}: event {event.kind}")
        (pages,) = self.session.read(self.report)
        report = pages.view(np.uint32).reshape(K["kReportPages"], -1)
        results = report[: self.cores].astype(np.int64)
        ran = results[:, K["kResultRun"]] == self.number
        errors = results[:, K["kResultError"]]
        if mock.active() and not ran.any():  # its loads are built; the words read 0
            raise SystemExit(f"command {handler}: a prebuild's reads are zeros")
        if not ran.all() or errors.any():
            raise RuntimeError(
                f"command {handler} {args}: ran {ran.tolist()} errors {errors.tolist()}"
            )
        plane = None
        if out:
            first = K["kReportOut"]
            plane = report[first : first + 2 * self.cores].reshape(self.cores, -1)
        return results, plane

    def stop(self):
        self.run.stop()


def spread(cycles, mhz, per=1):
    """min / median / max of the cores' median cycles, per `per` (and ns at mhz)."""
    medians = np.median(cycles, axis=1) / per
    return {
        "cycles": [round(float(v), 2) for v in (medians.min(), np.median(medians))]
        + [round(float(medians.max()), 2)],
        "ns_median": round(float(np.median(medians)) * 1e3 / mhz, 2),
    }


def run_loop(probe, args, mhz):
    rows = []
    for cpu in args.cpus:
        for index, described in enumerate(probe.described):
            forms = (
                ["direct"] if described["form"] == "direct" else ["hash", "hash_last"]
            )
            for form in forms:
                slots = described.get("slots", 8)
                command = [CPUS[cpu], FORMS[form], index, slots, args.reps]
                results, plane = probe.command(K["kHandlerLoop"], command, out=True)
                wrong = int((plane[:, : K["kPixels"]] != probe.expected[index]).sum())
                cycles = results[:, K["kResultReps"] : K["kResultReps"] + args.reps]
                row = {"cpu": cpu, "form": form, **described, "wrong": wrong}
                row["missing"] = int(results[:, K["kResultMisses"]].sum())
                row["pixel"] = spread(cycles[:, 1:], mhz, K["kPixels"])
                row["tile_cycles_first"] = int(np.median(cycles[:, 0]))
                rows.append(row)
                log(f"loop {cpu} {form} {described['stream']}: {row['pixel']} {wrong}")
    return {"rows": rows}


def run_miss(probe, args, mhz):
    rows = []
    for cpu in ("reader", "writer"):
        for size in args.sizes:
            for count in (1, 2, 4, 8):
                command = [CPUS[cpu], count, size, args.reps]
                results, _ = probe.command(K["kHandlerMiss"], command)
                cycles = results[:, K["kResultReps"] : K["kResultReps"] + args.reps]
                first = (np.arange(probe.cores)[:, None] * 97) + (
                    np.arange(args.reps)[None, :] * count
                )
                pages = (first[:, :, None] + np.arange(count)) % probe.pool_pages
                want = texels(pages, size // 4 - 1).astype(np.uint64).sum(axis=(1, 2))
                ok = (results[:, K["kResultSum"]] == (want & 0xFFFFFFFF)).all()
                row = {"cpu": cpu, "bytes": size, "in_flight": count, "exact": bool(ok)}
                row["batch"] = spread(cycles[:, 1:], mhz)
                row["page"] = spread(cycles[:, 1:], mhz, count)
                rows.append(row)
                log(f"miss {cpu} {size} B x {count}: {row['batch']} exact {ok}")
    return {"rows": rows}


def run_colour(probe, args, mhz):
    rows = []
    wants = {
        "integer": colour_integer(probe.texel, probe.light),
        "vector": colour_vector(probe.texel, probe.light),
        "copy": probe.texel,
    }
    for form, cpus in (
        ("integer", args.cpus),
        ("vector", ["pack"]),
        ("copy", ["pack"]),
    ):
        for cpu in cpus:
            for reps in (1, args.reps) if form != "integer" else (args.reps,):
                number = {"integer": 0, "vector": 1, "copy": 2}[form]
                command = [number, CPUS[cpu], reps]
                results, plane = probe.command(K["kHandlerColour"], command, out=True)
                wrong = int((plane[:, : K["kPixels"]] != wants[form]).sum())
                row = {"form": form, "cpu": cpu, "reps": reps, "wrong": wrong}
                if form == "integer":
                    cycles = results[:, K["kResultReps"] : K["kResultReps"] + reps]
                    row["tile"] = spread(cycles[:, 1:], mhz)
                else:
                    row["tile"] = spread(
                        results[:, K["kResultReps"] : K["kResultReps"] + 1], mhz, reps
                    )
                rows.append(row)
                log(f"colour {form} {cpu} x{reps}: {row['tile']} wrong {wrong}")
    differ = wants["integer"] != wants["vector"]
    return {"rows": rows, "integer_vs_vector_words_differ": int(differ.sum())}


def replay_rows(data, cores):
    """The replay tensor's rows: per core its item table, then its items' words."""
    items, words = data["items"], data["words"]
    per_core = [items[items[:, 0] == core] for core in range(cores)]
    longest = max(int(own[:, 2].sum()) for own in per_core)
    rows = np.zeros((ROW, K["kReplayWordsAt"] + longest), np.uint32)
    for core, own in enumerate(per_core):
        if len(own) > K["kItemsMax"]:
            raise ValueError(f"core {core}: {len(own)} items, at most kItemsMax")
        rows[core, 0] = len(own)
        at = 0
        for index, (_, first, count) in enumerate(own):
            rows[core, 1 + 2 * index] = at
            rows[core, 2 + 2 * index] = count
            rows[core, K["kReplayWordsAt"] + at : K["kReplayWordsAt"] + at + count] = (
                words[first : first + count]
            )
            at += count
    return rows, per_core


def run_replay(probe, args, mhz, data, per_core):
    rows = []
    for slots in args.slots:
        for form in args.forms:
            command = [slots, FORMS[form], args.passes]
            results, _ = probe.command(K["kHandlerReplay"], command)
            for core, own in enumerate(per_core):
                table, record = None, K["kResultItems"]
                for passed in range(args.passes):
                    for index, (_, first, count) in enumerate(own):
                        words = data["words"][first : first + count]
                        misses, total, table = simulate(
                            words, slots, form == "hash_last", probe.pool_pages, table
                        )
                        got = results[core, record : record + 3]
                        record += 3
                        rows.append(
                            {
                                "slots": slots,
                                "form": form,
                                "core": core,
                                "item": index,
                                "pass": passed,
                                "words": int(count),
                                "cycles": int(got[0]),
                                "misses": int(got[1]),
                                "misses_host": misses,
                                "sum_exact": bool(int(got[2]) == total),
                            }
                        )
            done = [
                row for row in rows if row["slots"] == slots and row["form"] == form
            ]
            exact = all(
                r["sum_exact"] and r["misses"] == r["misses_host"] for r in done
            )
            log(f"replay {slots} slots {form}: {len(done)} items, exact {exact}")
    return {"rows": rows, "pool_pages": probe.pool_pages}


def rows_right(row):
    """A row's own verdict: loop and colour rows count `wrong` words, miss rows are
    `exact`, replay rows `sum_exact` and the misses the host's simulation counts."""
    return (
        row.get("wrong", 0) == 0
        and row.get("exact", True)
        and row.get("sum_exact", True)
        and row.get("misses", 0) == row.get("misses_host", 0)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=("loop", "miss", "colour", "replay"))
    parser.add_argument("--cores", type=int, default=1, help="1 to 12, row 0")
    parser.add_argument("--reps", type=int, default=16)
    parser.add_argument("--cpus", nargs="+", default=list(CPUS), choices=list(CPUS))
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--pool-pages", type=int, default=256)
    parser.add_argument("--streams", type=Path, help="replay: the items' npz")
    parser.add_argument("--slots", type=int, nargs="+", default=list(SLOTS))
    parser.add_argument("--forms", nargs="+", default=["hash", "hash_last"])
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not 1 <= args.cores <= ROW or not 2 <= args.reps <= K["kRepsMax"]:
        parser.error(f"--cores 1 to {ROW}, --reps 2 to {K['kRepsMax']}")
    data = rows = per_core = None
    if args.what == "replay":
        data = dict(np.load(args.streams))
        args.cores = int(data["items"][:, 0].max()) + 1
        rows, per_core = replay_rows(data, args.cores)
    report = {"what": args.what, "cores": args.cores, "reps": args.reps}
    report["load_before"] = Path("/proc/loadavg").read_text().split()[:3]
    with km.Session(log=log) as session:
        log(f"device open, grid {session.grid}")
        probe = Probe(session, args.cores, args.pool_pages, rows)
        with Clock() as clock:
            mhz = clock.read()[0]
            if args.what == "loop":
                report.update(run_loop(probe, args, mhz))
            elif args.what == "miss":
                report.update(run_miss(probe, args, mhz))
            elif args.what == "colour":
                report.update(run_colour(probe, args, mhz))
            else:
                report.update(run_replay(probe, args, mhz, data, per_core))
        probe.stop()
        log("run stopped")
    report["clock"] = clock.summary()
    report["mhz_used"] = mhz
    report["load_after"] = Path("/proc/loadavg").read_text().split()[:3]
    report["ok"] = all(rows_right(row) for row in report["rows"])
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
