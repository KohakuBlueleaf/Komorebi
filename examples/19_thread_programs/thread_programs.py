"""Thread programs: an X280 thread splits the grid between spin (a unit costs what its
data says) and scale (tiles times a scalar) every round, after one command from the
host; against fixed splits run by the thread and fed by the host. Every unit checked;
one JSON line. See README.md."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.session import aiclk_mhz

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
C = km.abi.constants(HERE / "programs.h", known=km.abi.headers().constants)
CLUSTER = 1
PAGE = 1024  # values of a tile
TICKS_PER_MICROSECOND = 50  # kmx_time() counts the 50 MHz timer
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=0, help="0: the whole grid")
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--splits", type=int, default=8, help="fixed splits tried")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


class Work:
    """Each round's units, the data, and what every unit must leave."""

    def __init__(self, cores, rounds, generator):
        self.rounds = rounds
        self.spin = generator.integers(2 * cores, 8 * cores + 1, rounds)
        self.tiles = generator.integers(4 * cores, 16 * cores + 1, rounds)
        self.max_tiles = int(self.tiles.max())
        self.input_tiles = self.max_tiles + 64
        mean = 4000 + 3800 * np.sin(np.arange(rounds) * 3 * np.pi / rounds)
        costs = [
            (mean[r] * generator.uniform(0.5, 1.5, self.spin[r])).astype(np.uint32)
            for r in range(rounds)
        ]
        self.cost = np.concatenate(costs)  # cycles of each unit, all rounds in turn
        self.first_unit = np.concatenate([[0], np.cumsum(self.spin)[:-1]])
        self.first_tile = generator.integers(0, self.input_tiles - self.tiles + 1)
        values = generator.integers(-2, 3, (self.input_tiles, PAGE))
        self.input = values.astype(np.float32)  # a tile's 1024 values a page

    def input_pages(self):
        """The input as bfloat16 bits, a tile a page (SCALE is elementwise: the order
        of a tile's values does not matter)."""
        return bf16(self.input)

    def plan(self):
        """Region `rounds`: each round's line (programs.h, TP_ROUND_*)."""
        lines = np.zeros((self.rounds, C["TP_ROUND"]), np.uint32)
        lines[:, C["TP_ROUND_SPIN"]] = self.spin
        lines[:, C["TP_ROUND_TILES"]] = self.tiles
        lines[:, C["TP_ROUND_FIRST_UNIT"]] = self.first_unit
        lines[:, C["TP_ROUND_FIRST_TILE"]] = self.first_tile
        return lines.reshape(-1)

    def cost_lines(self):
        lines = np.zeros((self.cost.size, 16), np.uint32)
        lines[:, 0] = self.cost
        return lines

    def marks(self, job):
        """What SPIN leaves in each unit's out line (TP_MARK, the unit, the job)."""
        unit = np.arange(self.cost.size, dtype=np.uint64)
        mixed = unit ^ (self.cost.astype(np.uint64) << 7) ^ (job << 26)
        mark = ((mixed & 0xFFFFFFFF) * 0x9E3779B1 & 0xFFFFFFFF) ^ 0x85EBCA77
        return np.stack([mark, unit, np.full_like(unit, job)], 1).astype(np.uint32)

    def wrong(self, outs, output, job, scalar):
        """Units of every round whose result differs."""
        bad = int((outs[:, :3] != self.marks(job)).any(axis=1).sum())
        got = output.reshape(-1, PAGE)
        for r in range(self.rounds):
            first, count = int(self.first_tile[r]), int(self.tiles[r])
            want = bf16(scalar * self.input[first : first + count])
            mine = got[r * self.max_tiles : r * self.max_tiles + count]
            bad += int((mine != want).any(axis=1).sum())
        return bad


# --8<-- [start:programs]
def programs(tensors, work):
    """spin and scale in the runtime's form; the thread is scale's (any one program
    of a run may declare it)."""
    spin = km.Program(
        "spin",
        kernels=KERNELS / "spin",
        buffers=[km.Buffer(C["TP_CB_LINE"], "uint32", C["TP_LINE_BYTES"], 1)],
        compile_args={
            "reader": tensors["costs"].tensor_args() + tensors["outs"].tensor_args()
        },
        handlers={"spin": C["TP_HANDLER_SPIN"]},
    )
    tile = dict(format="bfloat16", page_bytes=C["TP_TILE_BYTES"], pages=2)
    scale = km.Program(
        "scale",
        kernels=KERNELS / "scale",
        buffers=[km.Buffer(C["TP_CB_IN"], **tile), km.Buffer(C["TP_CB_OUT"], **tile)],
        compile_args={
            "reader": tensors["input"].tensor_args(),
            "writer": tensors["output"].tensor_args(),
        },
        handlers={"scale": C["TP_HANDLER_SCALE"]},
    )
    scale.thread(
        cluster=CLUSTER,
        sources=[HERE / "thread" / "rounds.c"],
        includes=[HERE],
        regions={"rounds": work.plan(), "log": 4 * C["TP_LOG"] * work.rounds},
        words=[
            tensors["costs"].address,
            tensors["outs"].address,
            tensors["input"].address,
            tensors["output"].address,
            work.max_tiles,
        ],
        handlers={"rounds": C["TP_HANDLER_ROUNDS"]},
    )
    return [spin, scale]  # combined: every core holds both


# --8<-- [end:programs]


def bf16(values):
    """bfloat16 bits of float32 values that bfloat16 holds exactly."""
    return (np.asarray(values, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def float_bits(value):
    return int(np.float32(value).view(np.uint32))


def job(session, work, tensors, cores, number, split=None, host=False):
    """One load, one job: the thread's split (split None), a fixed split run by the
    thread, or a fixed split fed by the host one round at a time. Its rounds' log
    lines, wrong units and the host's seconds."""
    scalar = float(2 + number % 3)  # differs from the job before: none is stale
    spin, scale = programs(tensors, work)
    run = session.load([spin, scale])
    try:
        started = time.perf_counter()
        if host:
            lines = host_fed(run, spin, scale, tensors, work, cores, split, number)
        else:
            # --8<-- [start:send]
            mode = (0, 0) if split is None else (1, split)
            args = (*mode, work.rounds, cores, number, float_bits(scalar))
            handle = run.submit("rounds", cluster=CLUSTER, args=args, result=True)
            event = run.wait(handle, timeout=60.0)
            # --8<-- [end:send]
            if event.kind != "done" or event.result != 0:
                raise km.KomorebiError(f"the thread's job: {event}")
            lines = None
        seconds = time.perf_counter() - started
    finally:
        run.stop()
    if lines is None:
        raw = run.threads[CLUSTER].read("log").view(np.uint32)
        lines = raw.reshape(work.rounds, C["TP_LOG"])
    outs = session.read(tensors["outs"])[0].view(np.uint32).reshape(-1, 16)
    output = session.read(tensors["output"])[0].view(np.uint16)
    return lines, work.wrong(outs, output, number, scalar), seconds


def host_fed(run, spin, scale, tensors, work, cores, split, number):
    """The same rounds with a fixed split, each round's two commands from the host;
    log lines as the thread's (host microseconds x 50 for its round)."""
    lines = np.zeros((work.rounds, C["TP_LOG"]), np.uint32)
    scalar = float_bits(2 + number % 3)
    for r in range(work.rounds):
        started = time.perf_counter()
        spin_words = [
            int(work.first_unit[r]), int(work.spin[r]), tensors["costs"].address,
            tensors["outs"].address, number,
        ]  # fmt: skip
        scale_words = [
            tensors["input"].address, int(work.tiles[r]), tensors["output"].address,
            r * work.max_tiles, scalar, 0, int(work.first_tile[r]),
        ]  # fmt: skip
        handles = [
            run.submit(spin.handler("spin"), (0, split), spin_words),
            run.submit(scale.handler("scale"), (split, cores - split), scale_words),
        ]
        for handle in handles:
            if run.wait(handle, 10.0).kind != "done":
                raise km.KomorebiError(f"round {r}: a command did not finish")
        lines[r, C["TP_LOG_SPLIT"]] = split
        lines[r, C["TP_LOG_ROUND_TICKS"]] = int(
            (time.perf_counter() - started) * 1e6 * TICKS_PER_MICROSECOND
        )
    return lines


def summary(lines):
    """Microseconds of the whole job, and per round outside the grid's work."""
    ticks = lines.astype(np.int64)
    round_ticks = ticks[:, C["TP_LOG_ROUND_TICKS"]]
    grid = np.maximum(
        ticks[:, C["TP_LOG_SPIN_TICKS"]], ticks[:, C["TP_LOG_SCALE_TICKS"]]
    )
    return {
        "job_us": round(float(round_ticks.sum()) / TICKS_PER_MICROSECOND, 1),
        "outside_us_a_round": round(
            float(np.median(round_ticks - grid)) / TICKS_PER_MICROSECOND, 2
        ),
        "splits": ticks[:, C["TP_LOG_SPLIT"]].tolist(),
    }


def main():
    args = parse()
    with km.Session(log=log) as session:
        grid = session.grid
        cores = args.cores or grid[0] * grid[1]
        work = Work(cores, args.rounds, np.random.default_rng(args.seed))
        tensors = {  # one set for every job: each job's scalar differs
            "costs": session.tensor(work.cost_lines(), "uint32"),
            "outs": session.empty((work.cost.size, 16), "uint32"),
            "input": session.tensor(work.input_pages(), "uint16"),
            "output": session.empty((work.rounds * work.max_tiles, PAGE), "uint16"),
        }
        mhz = aiclk_mhz(session.card)
        log(f"{cores} cores, {args.rounds} rounds, aiclk {mhz} MHz")
        number, rows, wrong = 0, {}, 0

        def one(name, **options):
            nonlocal number, wrong
            lines, bad, seconds = job(session, work, tensors, cores, number, **options)
            number, wrong = number + 1, wrong + bad
            rows[name] = {**summary(lines), "wrong": bad, "host_seconds": seconds}
            log(f"{name}: {rows[name]['job_us']} us, {bad} wrong")
            return rows[name]

        one("thread")
        candidates = sorted(
            {max(1, min(cores - 1, round(cores * (i + 1) / (args.splits + 1))))
             for i in range(args.splits)}
        )  # fmt: skip
        fixed = {k: one(f"fixed {k}", split=k)["job_us"] for k in candidates}
        best = min(fixed, key=fixed.get)
        one("host fed", split=best, host=True)
    table = {
        "thread split": rows["thread"]["job_us"],
        f"best fixed split (k {best})": fixed[best],
        f"host fed (k {best})": rows["host fed"]["job_us"],
        "thread outside the grid, a round": rows["thread"]["outside_us_a_round"],
    }
    for name, value in table.items():
        print(f"{name:34}: {value:10.1f} us")
    out = {
        "cores": cores,
        "rounds": args.rounds,
        "aiclk_mhz": mhz,
        "table_us": table,
        "fixed_us": fixed,
        "rows": rows,
        "wrong": wrong,
        "ok": wrong == 0,
    }
    print(json.dumps(out), flush=True)
    if wrong:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
