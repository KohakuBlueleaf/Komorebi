"""Thread load: after one command from the host, an X280 thread moves the boundary
between the cores of sum and those of widen every round with a load, then commands
both; against the host issuing every load and command. Every unit checked; one JSON
line. See README.md."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.bench import power
from komorebi.session import aiclk_mhz

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
C = km.abi.constants(HERE / "thread_load.h", known=km.abi.headers().constants)
CLUSTER = 1
TILE = 32
INPUT_TILES = 512
TICKS_PER_MICROSECOND = 50  # kmx_time() and the dispatcher count the 50 MHz timer
M32 = 0xFFFFFFFF
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=0, help="0: the whole grid")
    parser.add_argument("--rounds", type=int, default=40, help="rounds a job")
    parser.add_argument("--jobs", type=int, default=1, help="thread jobs, a load each")
    parser.add_argument("--no-host", action="store_true", help="no host-fed job")
    parser.add_argument("--no-check", action="store_true", help="no CHECK command")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def mix(hashes, words):
    """TL_MIX of each row's running hash with one word a row (uint32, wrapping)."""
    return ((hashes << np.uint32(5)) | (hashes >> np.uint32(27))) + words


def tile_hashes(words):
    """The hash of each row of 32-bit words, as the writers take them in L1 order."""
    hashes = np.full(words.shape[0], C["TL_HASH_START"], np.uint32)
    for column in range(words.shape[1]):
        hashes = mix(hashes, words[:, column])
    return hashes


def faces(tiles):
    """Tiles [n, 32, 32] in the order of a tile in memory: four 16 x 16 faces."""
    return tiles.reshape(-1, 2, 16, 2, 16).transpose(0, 1, 3, 2, 4).reshape(-1, 1024)


def parts(total, count):
    """Each rank's (start, length) of `total` items over `count` cores, as
    km::partition: the first total % count cores take one more."""
    base, extra = divmod(int(total), int(count))
    rank = np.arange(count)
    return rank * base + np.minimum(rank, extra), base + (rank < extra)


def split(cores, sum_tiles, widen_tiles):
    """TL_SPLIT clamped to 1 .. cores - 1, as the thread computes it."""
    total = int(sum_tiles) + int(widen_tiles)
    return min(max((cores * int(sum_tiles) + total // 2) // total, 1), cores - 1)


class Work:
    """The input, each round's amounts, and what every core's line must hold."""

    def __init__(self, cores, rounds, generator):
        self.cores, self.rounds = cores, rounds
        values = generator.integers(-2, 3, (INPUT_TILES * TILE, TILE))
        self.input = values.astype(np.float32)  # every sum a core makes is exact
        self.tiles = self.input.reshape(INPUT_TILES, TILE, TILE)
        self.prefix = np.concatenate([np.zeros((1, 1024)), faces(self.tiles).cumsum(0)])
        wide = faces(self.tiles).astype(np.float32).view(np.uint32)
        hashes = tile_hashes(wide).astype(np.uint64)
        self.widen_prefix = np.concatenate([np.zeros(1, np.uint64), np.cumsum(hashes)])
        self.amounts = generator.integers(cores // 2 + 1, 3 * cores + 1, (rounds, 2))
        most = self.amounts.max(axis=1)
        self.firsts = generator.integers(
            0, INPUT_TILES - most[:, None] + 1, (rounds, 2)
        )
        self.k0 = cores // 2

    def plan(self):
        """Region `rounds`: each round's line (thread_load.h, TL_ROUND_*)."""
        lines = np.zeros((self.rounds, C["TL_ROUND"]), np.uint32)
        lines[:, C["TL_ROUND_SUM"]] = self.amounts[:, 0]
        lines[:, C["TL_ROUND_WIDEN"]] = self.amounts[:, 1]
        lines[:, C["TL_ROUND_SUM_FIRST"]] = self.firsts[:, 0]
        lines[:, C["TL_ROUND_WIDEN_FIRST"]] = self.firsts[:, 1]
        return lines.reshape(-1)

    def splits(self):
        """Each round's boundary k, as the thread decides it."""
        return [split(self.cores, *amount) for amount in self.amounts]

    def want(self, splits, job):
        """Every core's line of every round [rounds, cores, 5] (TL_LINE_* words)."""
        want = np.zeros((self.rounds, self.cores, 5), np.uint32)
        sums = []  # (round, core, first, tiles) of every SUM core
        for r, k in enumerate(splits):
            for program, first, count, column in (
                (1, 0, k, 0),
                (2, k, self.cores - k, 1),
            ):
                starts, lengths = parts(self.amounts[r, column], count)
                starts = starts + self.firsts[r, column]
                cores = slice(first, first + count)
                want[r, cores, C["TL_LINE_TILES"]] = lengths
                want[r, cores, C["TL_LINE_FIRST"]] = starts
                want[r, cores, C["TL_LINE_PROGRAM"]] = program
                if program == 2:
                    ends = self.widen_prefix[starts + lengths]
                    want[r, cores, C["TL_LINE_HASH"]] = (
                        ends - self.widen_prefix[starts]
                    ) & M32
                else:
                    sums += [
                        (r, first + i, starts[i], lengths[i]) for i in range(count)
                    ]
        want[:, :, C["TL_LINE_TAG"]] = (job << 16) | (np.arange(self.rounds) & 0xFFFF)[
            :, None
        ]
        sums = np.array(sums, np.int64).reshape(-1, 4)
        for at in range(0, len(sums), 8192):
            rows = sums[at : at + 8192]
            tiles = self.prefix[rows[:, 2] + rows[:, 3]] - self.prefix[rows[:, 2]]
            bits = (tiles.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            hashes = tile_hashes(bits.view(np.uint32))
            want[rows[:, 0], rows[:, 1], C["TL_LINE_HASH"]] = hashes
        return want


# --8<-- [start:programs]
def programs(tensors, work):
    """sum and widen in the runtime's form: other buffers, other formats (example 16's
    readers and computes); the thread is sum's."""
    compute = {"math_fidelity": "HiFi4", "fp32_dest_acc_en": True}
    line = dict(format="uint32", page_bytes=C["TL_LINE_BYTES"], pages=1)
    tensor_args = {
        "reader": tensors["input"].tensor_args(),
        "writer": tensors["outs"].tensor_args(),
    }
    sum_program = km.Program(
        "sum",
        kernels=KERNELS / "sum",
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
            km.Buffer(index=C["TL_CB_SUM_LINE"], **line),
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
        ],
        compile_args=tensor_args,
        compute=compute,
        handlers={"sum": C["TL_HANDLER_SUM"]},
    )
    widen_program = km.Program(
        "widen",  # bfloat16 in at index 1, float32 out at index 16
        kernels=KERNELS / "widen",
        buffers=[
            km.Buffer(index=1, format="bfloat16", page_bytes=2048, pages=4),
            km.Buffer(index=16, format="float32", page_bytes=4096, pages=2),
            km.Buffer(index=C["TL_CB_WIDEN_LINE"], **line),
        ],
        compile_args=tensor_args,
        compute=compute,
        handlers={"widen": C["TL_HANDLER_WIDEN"]},
    )
    sum_program.thread(
        cluster=CLUSTER,
        sources=[HERE / "thread" / "rounds.c"],
        includes=[HERE],
        regions={
            "rounds": work.plan(),
            "log": 4 * C["TL_LOG"] * work.rounds,
            "check": 4 * C["TL_CHECK"],
        },
        words=[tensors["input"].address, tensors["outs"].address, work.cores],
        handlers={"rounds": C["TL_HANDLER_ROUNDS"], "check": C["TL_HANDLER_CHECK"]},
    )
    return sum_program, widen_program


# --8<-- [end:programs]


def load(session, tensors, work):
    """One load: sum on cores 0 to k0 - 1, widen on the rest of the work's cores."""
    sum_program, widen_program = programs(tensors, work)
    cores, k0 = work.cores, work.k0
    run = session.load(
        {sum_program: (0, k0), widen_program: (k0, cores - k0)}, deadline=10.0
    )
    return run, sum_program, widen_program


def check(run, widen_program, work):
    """CHECK: the thread's findings (TL_CHECK_* words)."""
    base, step, _ = run.thread.records  # each program's record, as kmx_load has them
    record = base + run.program_field(widen_program) * step
    args = [record, 0, work.k0, work.rounds * work.cores]
    event = run.wait(run.submit("check", cluster=CLUSTER, args=args), timeout=10.0)
    if event.kind != "done":
        raise km.KomorebiError(f"CHECK: {event}")


def thread_job(session, tensors, work, job, with_check):
    """One load, one ROUNDS command; the thread's log lines and CHECK's words."""
    run, _, widen_program = load(session, tensors, work)
    try:
        if with_check:
            check(run, widen_program, work)
        # --8<-- [start:send]
        args = [0, work.rounds, job, work.k0]
        handle = run.submit("rounds", cluster=CLUSTER, args=args, result=True)
        event = run.wait(handle, timeout=60.0)
        # --8<-- [end:send]
        if event.kind != "done" or event.result != 0:
            raise km.KomorebiError(f"ROUNDS: {event}")
    finally:
        run.stop()
    thread = run.threads[CLUSTER]
    lines = thread.read("log").view(np.uint32).reshape(work.rounds, C["TL_LOG"])
    found = (
        thread.read("check").view(np.uint32)[: C["TL_CHECK"]] if with_check else None
    )
    return lines, found


def host_job(session, tensors, work, job):
    """The same rounds with the host issuing every load and both commands; log lines
    as the thread's (the host's microseconds x 50)."""
    run, sum_program, widen_program = load(session, tensors, work)
    lines = np.zeros((work.rounds, C["TL_LOG"]), np.uint32)
    k = work.k0
    try:
        for r, split_k in enumerate(work.splits()):
            started = time.perf_counter()
            if split_k != k:
                group, program = (
                    ((split_k, k - split_k), widen_program)
                    if split_k < k
                    else ((k, split_k - k), sum_program)
                )
                asked = time.perf_counter()
                event = run.wait(run.load(program, group=group), 10.0)
                lines[r, C["TL_LOG_LOADS"]] = 1
                lines[r, C["TL_LOG_LOAD_TICKS"]] = ticks(time.perf_counter() - asked)
                lines[r, C["TL_LOG_LOAD_EVENT"]] = event.ticks
                k = split_k
            tag = job << 16 | (r & 0xFFFF)
            handles = [
                run.submit(handler, group, words)
                for handler, group, words in (
                    (
                        sum_program.handler("sum"),
                        (0, k),
                        command(tensors, work, r, 0, tag),
                    ),
                    (
                        widen_program.handler("widen"),
                        (k, work.cores - k),
                        command(tensors, work, r, 1, tag),
                    ),
                )
            ]
            for handle in handles:
                if run.wait(handle, 10.0).kind != "done":
                    raise km.KomorebiError(f"round {r}: a command did not finish")
            lines[r, C["TL_LOG_SPLIT"]] = k
            lines[r, C["TL_LOG_ROUND_TICKS"]] = ticks(time.perf_counter() - started)
    finally:
        run.stop()
    return lines


def command(tensors, work, r, column, tag):
    """The words of round r's SUM (column 0) or WIDEN (1), as the thread's."""
    return [
        tensors["input"].address, int(work.amounts[r, column]), tensors["outs"].address,
        r * work.cores, tag, 0, int(work.firsts[r, column]),
    ]  # fmt: skip


def ticks(seconds):
    return int(seconds * 1e6 * TICKS_PER_MICROSECOND)


def wrong_lines(session, tensors, work, splits, job):
    """Lines of the job's rounds that differ from what their cores must write."""
    outs = tensors["outs"]
    got = []
    for first, pages in session.read_chunks(outs):
        got.append(pages.view(np.uint32).reshape(-1, C["TL_LINE_WORDS"]))
        log(f"read lines {first} to {first + len(got[-1])} of {outs.pages}")
    lines = np.concatenate(got)[: work.rounds * work.cores, :5]
    want = work.want(splits, job).reshape(-1, 5)
    return int((lines != want).any(axis=1).sum())


def summary(lines):
    """Microseconds of the job; a load's median from issue to done and in the
    dispatcher (0 without loads); loads a round."""
    ticks_ = lines.astype(np.int64)
    loads = ticks_[ticks_[:, C["TL_LOG_LOADS"]] == 1]

    def median_us(column):
        return round(float(np.median(loads[:, column])) / 50, 2) if len(loads) else 0.0

    return {
        "job_us": round(float(ticks_[:, C["TL_LOG_ROUND_TICKS"]].sum()) / 50, 1),
        "load_us": median_us(C["TL_LOG_LOAD_TICKS"]),
        "load_dispatcher_us": median_us(C["TL_LOG_LOAD_EVENT"]),
        "loads": len(loads),
        "loads_a_round": round(len(loads) / len(lines), 3),
    }


def main():
    args = parse()
    with km.Session(log=log) as session:
        grid = session.grid
        cores = args.cores or grid[0] * grid[1]
        work = Work(cores, args.rounds, np.random.default_rng(args.seed))
        tensors = {
            "input": session.tensor(work.input, "bfloat16", "tile"),
            "outs": session.tensor(
                np.zeros(((args.rounds + 1) * cores, C["TL_LINE_WORDS"])), "uint32"
            ),
        }
        mhz = [aiclk_mhz(session.card)]
        log(f"{cores} cores, {args.rounds} rounds, aiclk {mhz[0]} MHz")
        splits, rows, wrong, found = work.splits(), {}, 0, None
        for job in range(args.jobs):
            lines, found_now = thread_job(
                session, tensors, work, job, not args.no_check and job == 0
            )
            found = found if found_now is None else found_now
            bad = wrong_lines(session, tensors, work, splits, job)
            bad += int((lines[:, C["TL_LOG_SPLIT"]] != splits).sum())
            bad += int((lines[:, C["TL_LOG_STATUS"]] != 0).sum())
            wrong += bad
            rows.setdefault("thread", []).append({**summary(lines), "wrong": bad})
            log(f"thread job {job}: {rows['thread'][-1]}")
        if not args.no_host:
            job = args.jobs
            lines = host_job(session, tensors, work, job)
            bad = wrong_lines(session, tensors, work, splits, job)
            wrong += bad
            rows["host"] = {**summary(lines), "wrong": bad}
            log(f"host job: {rows['host']}")
        mhz.append(aiclk_mhz(session.card))
    first = rows["thread"][0]
    table = {
        "job_us": [first["job_us"], rows.get("host", {}).get("job_us")],
        "load_us": [first["load_us"], rows.get("host", {}).get("load_us")],
        "load_dispatcher_us": [
            first["load_dispatcher_us"],
            rows.get("host", {}).get("load_dispatcher_us"),
        ],
        "loads_a_round": [
            first["loads_a_round"],
            rows.get("host", {}).get("loads_a_round"),
        ],
    }
    for name, (thread, host) in table.items():
        print(f"{name:20}: thread {thread}, host {host}")
    checks = None
    if found is not None:
        checks = {name: int(found[C[f"TL_CHECK_{name.upper()}"]]) for name in CHECKS}
    ok = wrong == 0 and (checks is None or expected(checks))
    out = {
        "cores": cores,
        "rounds": args.rounds,
        "jobs": args.jobs,
        "aiclk_mhz": mhz,
        "table": table,
        "rows": rows,
        "loads": sum(row["loads"] for row in rows["thread"]),
        "check": checks,
        "wrong": wrong,
        "ok": ok,
    }
    print(json.dumps(power.clocked(out)), flush=True)
    if not ok:
        raise SystemExit(1)


CHECKS = (
    "refused", "busy", "load_status", "sum_ticks", "load_ticks", "back_status", "after",
)  # fmt: skip


def expected(checks):
    """CHECK as it must be: the SUM to widen's core refused (reason 13), the load to a
    busy core rejected busy (11) and then done by kmx_load after the long SUM."""
    rejected = km.abi.headers().constants["KM_EVENT_REJECTED"]
    return (
        checks["refused"] == rejected << 8 | 13
        and checks["busy"] == rejected << 8 | 11
        and checks["load_status"] == 0
        and checks["back_status"] == 0
        and checks["after"] == 0
        and checks["load_ticks"] >= checks["sum_ticks"]
    )


if __name__ == "__main__":
    main()
