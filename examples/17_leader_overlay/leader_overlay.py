"""A leader casts which overlay its group runs: the first core of each broadcast reads a
value, picks one of four overlays and casts its number; every core of the broadcast
loads and calls it. Every core's data is checked against numpy, sampled rounds timed;
--compiled builds the operations into the kernels instead. See README.md."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import layout
from komorebi.bench import power
from komorebi.session.process import aiclk_mhz

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
HEADER = KERNELS / "leader_overlay.h"
CONSTANTS = km.abi.constants(HEADER, known=km.abi.headers().constants)
OVERLAYS = ("mix", "turn", "scale", "shift")  # the leader's choice c names c + 1
WORDS = CONSTANTS["kDataWords"]
VALUES = CONSTANTS["kTileValues"]
TIMED_ROUNDS = CONSTANTS["kRecordRounds"]
M32 = 0xFFFFFFFF
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--rect", default="0,0,1,10", help="first_x,first_y,columns,rows"
    )
    parser.add_argument("--along", choices=("column", "row"), default="column")
    parser.add_argument("--rounds", type=int, default=100000, help="rounds in all")
    parser.add_argument("--per", type=int, default=2000, help="rounds per command")
    parser.add_argument(
        "--choices",
        choices=("cycle", "blocks", "random"),
        default="cycle",
        help="cycle: 0 1 2 3 (every round loads); blocks: three of the four for "
        "--period rounds, then another three (loads are rare); random",
    )
    parser.add_argument("--period", type=int, default=1000)
    parser.add_argument("--timed", type=int, default=10, help="commands timed")
    parser.add_argument(
        "--compiled", action="store_true", help="the operations compiled in"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--addon",
        default="",
        help="overlays not loaded (a comma list, e.g. shift): run.addon adds each to "
        "the live run, with its number, before the first command",
    )
    return parser.parse_args()


def choices_of(kind, count, period, generator):
    """The leader's choice for every round."""
    index = np.arange(count)
    if kind == "cycle":
        return index % 4
    if kind == "random":
        return generator.integers(0, 4, count)
    third = 2 + (index // period) % 2  # mix, turn and one of scale and shift
    return np.choose(generator.integers(0, 3, count), [index * 0, index * 0 + 1, third])


# --8<-- [start:replay]
def bf16_bits(values):
    """bfloat16 bits of float32 values that bfloat16 holds exactly."""
    return (np.asarray(values, np.float32).view(np.uint32) >> 16).astype(np.uint64)


SMALL = np.arange(8)
# out bits of scale (choice 2) and shift (choice 3) by scalar 1 to 4 and value 0 to 7
TILE_OUT = {
    2: [bf16_bits(SMALL * scalar) for scalar in range(1, 5)],
    3: [bf16_bits(SMALL + scalar) for scalar in range(1, 5)],
}
COLUMN = np.arange(VALUES) & (WORDS - 1)
SHIFT = (4 * ((np.arange(VALUES) >> 5) & 7)).astype(np.uint32)
FOLD = np.array([pow(3, WORDS - 1 - m, 1 << 32) for m in range(WORDS)], np.uint64)


def data_start(cores, first):
    """Each core's data at the start of a command (ops.h, data_start)."""
    core = np.asarray(cores, np.uint64)[:, None]
    k = np.arange(WORDS, dtype=np.uint64)[None, :]
    return ((first * 2654435761 + core * 40503 + k * 97) & M32).astype(np.uint32)


def step(data, choice, index):
    """One round of the overlay `choice` on every core's data (ops.h)."""
    if choice == 0:
        return (
            data * np.uint32(33) + np.uint32(index) + np.arange(WORDS, dtype=np.uint32)
        )
    if choice == 1:
        old = np.roll(data, -(1 + (index & 15)), axis=1)
        return (old << np.uint32(5) | old >> np.uint32(27)) ^ np.uint32(index)
    small = (data[:, COLUMN] >> SHIFT) & np.uint32(7)
    out = TILE_OUT[choice][index & 3][small].reshape(len(data), WORDS, WORDS)
    folded = (out * FOLD[None, :, None]).sum(axis=1)
    return ((data.astype(np.uint64) * pow(3, WORDS, 1 << 32) + folded) & M32).astype(
        np.uint32
    )


# --8<-- [end:replay]


def plan(nocs, rect, along):
    """The group covering `rect`, each core's program arguments (reader: the
    broadcast's words, without a buffer; compute: included), the cores of the
    rectangle and each one's broadcast sender."""
    first_x, first_y, columns, rows = rect
    width = nocs.grid[0]
    first = first_y * width + first_x
    group = layout.Group(
        first, (first_y + rows - 1) * width + first_x + columns - first
    )
    common = (nocs.grid, nocs.panels, group, rect, along, 0, nocs.noc0, nocs.noc_xy)
    words = layout.broadcast_args(*common, (0, 0, 0, 0))
    fields = km.abi.struct_fields(km.abi.headers().device[-1], "BroadcastArgs")
    cells = [
        y * width + x
        for y in range(first_y, first_y + rows)
        for x in range(first_x, first_x + columns)
    ]

    def arguments_of(core):
        if core not in words:
            return [0] * len(fields), [0], [0]
        return [words[core][field] for field in fields], [0], [int(core in cells)]

    senders = {core: words[core]["sender"] for core in cells}
    return group, arguments_of, cells, senders


# --8<-- [start:program]
def program(tensors, arguments_of, compiled, later=()):
    """Three kernels of the kernel form; four overlays and 3 reader slots, 1 compute
    slot, or (compiled) the four operations in the kernels."""
    values, results, times = tensors
    tile = dict(format="bfloat16", page_bytes=2048, pages=2)
    made = km.Program(
        "leader_overlay",
        kernels=KERNELS,
        handlers=HEADER,  # kHandlerLead; kOverlayMix = 1 ... kOverlayShift = 4
        buffers=[
            km.Buffer(CONSTANTS["kCbIn"], **tile),
            km.Buffer(CONSTANTS["kCbOut"], **tile),
            km.Buffer(CONSTANTS["kCbChoice"], "uint32", 64, 2),
            km.Buffer(CONSTANTS["kCbValues"], "uint32", values.page_bytes, 1),
            km.Buffer(CONSTANTS["kCbData"], "uint32", 4 * WORDS, 1),
            km.Buffer(CONSTANTS["kCbResult"], "uint32", results.page_bytes, 1),
            km.Buffer(CONSTANTS["kCbRecords"], "uint32", times.page_bytes, 1),
        ],
        args=arguments_of,
        compile_args={
            "reader": values.tensor_args() + results.tensor_args() + times.tensor_args()
        },
        defines={"LEADER_COMPILED": 1} if compiled else {},
    )
    if compiled:
        return made
    for name in OVERLAYS:  # mix/reader.cpp ... shift/reader.cpp and compute.cpp
        if name not in later:  # added to the live run instead (--addon)
            made.overlay(name, KERNELS / name)
    # slot bytes named (empty: the sizes stay) give the kernels the path for run.addon
    return made.slots(count={"reader": 3, "compute": 1}, bytes={} if later else None)


# --8<-- [end:program]


def check(rows, choices, args, group, cells):
    """The replay of every command on every core; the first wrong cores."""
    bad = []
    for job in range(rows.shape[0]):
        first = job * args.per
        data = data_start(cells, first)
        for offset in range(args.per):
            data = step(data, int(choices[first + offset]), first + offset)
        got = rows[job, [core - group.first for core in cells]]
        for slot, core in enumerate(cells):
            right = np.array_equal(got[slot, :WORDS], data[slot])
            wrong, _, done = got[slot, WORDS : WORDS + 3]
            if not (right and wrong == 0 and done == args.per):
                bad.append({"job": job, "core": core, "wrong": int(wrong)})
        if job % 10 == 9:
            log(f"replayed {job + 1} of {rows.shape[0]} commands")
    return bad


def median(values):
    return float(np.median(values)) if len(values) else None


def timings(records, cells, senders, group, mhz):
    """Medians of the timed rounds, cycles and µs: the leader's cast, load() loading
    (reader part only, with compute's parts) and held, the call by overlay, and from
    the leader's cast to the last call done in its broadcast."""
    slot = {core: core - group.first for core in cells}
    leaders = sorted(set(senders.values()))
    taken = records[:, [slot[core] for core in cells]]  # (timed, cores, rounds, 4)
    used = taken[..., 3] != 0
    load = taken[..., 2] & 0xFFFFFF
    loaded = (taken[..., 2] >> 28) & 3
    number = (taken[..., 2] >> 24) & 0xF
    call = taken[..., 1] >> 16
    lead = records[:, [slot[core] for core in leaders]]
    spans, kinds, loads = [], [], []
    for leader_slot, leader in enumerate(leaders):
        members = [i for i, core in enumerate(cells) if senders[core] == leader]
        begun = lead[:, leader_slot, :, 0].astype(np.int64)
        done = taken[:, members][..., 3].astype(np.int64)
        span = (done - begun[:, None, :]) % (1 << 32)
        ok = used[:, members].all(axis=1) & (lead[:, leader_slot, :, 3] != 0)
        spans.append(span.max(axis=1)[ok])
        kinds.append(((lead[:, leader_slot, :, 2] >> 24) & 0xF)[ok])
        loads.append((loaded[:, members] != 0).any(axis=1)[ok])
    spans, kinds, loads = (np.concatenate(items) for items in (spans, kinds, loads))
    out = {
        "cast": median((lead[..., 1] & 0xFFFF)[lead[..., 3] != 0]),
        "load loading reader": median(load[used & (loaded == 1)]),
        "load loading with compute": median(load[used & (loaded >= 2)]),
        "load held": median(load[used & (loaded == 0)]),
        "cast to last call": median(spans),
        "rounds timed": int(used.sum()),
        "loading share": float((loaded[used] != 0).mean()) if used.any() else None,
    }
    for choice, name in enumerate(OVERLAYS):
        out[f"call {name}"] = median(call[used & (number == choice + 1)])
    for choice, name in enumerate(OVERLAYS):
        for how, which in (("loading", loads), ("held", ~loads)):
            chosen = spans[(kinds == choice + 1) & which]
            out[f"cast to last call {name} {how}"] = median(chosen)
    return {
        key: (
            {"cycles": value, "us": round(value / mhz, 3)}
            if value is not None and key.split()[0] in ("cast", "load", "call")
            else value
        )
        for key, value in out.items()
    }


def main():
    args = parse()
    if args.rounds % args.per or not 1 <= args.per <= 32767:
        raise SystemExit("--rounds must be a multiple of --per, --per 1 to 32767")
    rect = tuple(int(field) for field in args.rect.split(","))
    generator = np.random.default_rng(args.seed)
    jobs = args.rounds // args.per
    choices = choices_of(args.choices, args.rounds, args.period, generator)
    values_flat = (choices.astype(np.uint64) << 30) | generator.integers(
        0, 1 << 30, args.rounds, dtype=np.uint64
    )
    pages = -(-args.rounds // CONSTANTS["kValuesPage"])
    values = np.zeros(pages * CONSTANTS["kValuesPage"], np.uint32)
    values[: args.rounds] = values_flat.astype(np.uint32)
    timed = sorted({job * jobs // args.timed for job in range(min(args.timed, jobs))})
    stride = max(1, args.per // TIMED_ROUNDS) | 1
    with km.Session(log=log) as session:
        nocs = session.nocs()
        group, arguments_of, cells, senders = plan(nocs, rect, args.along)
        log(f"group {group.first}:{group.count}, {len(cells)} cores along {args.along}")
        values_tensor = session.tensor(values.reshape(pages, -1), "uint32")
        results = session.empty(
            (jobs * group.count, CONSTANTS["kResultWords"]), "uint32"
        )
        times = session.empty(
            (max(1, len(timed)) * group.count, 4 * TIMED_ROUNDS), "uint32"
        )
        later = [name for name in args.addon.split(",") if name]
        made = program(
            (values_tensor, results, times), arguments_of, args.compiled, later
        )
        run = session.load(made)
        for name in later:  # the leader casts its number as any overlay's
            number = OVERLAYS.index(name) + 1
            run.addon(name, KERNELS / name, number=number)
        sizes = dict(made.sizes.text)
        built = made.overlays_built
        mhz = [aiclk_mhz(session.card)]
        log(f"program loaded, AICLK {mhz[0]} MHz, kernel text {sizes}")
        microseconds = []
        for job in range(jobs):
            # --8<-- [start:submit]
            timing = timed.index(job) if job in timed else None
            job_args = [
                values_tensor.address,
                args.per,
                results.address,
                job * args.per,
                job * group.count,
                times.address if timing is not None else 0,
                (timing or 0) * group.count,
                stride,
            ]
            handle = run.submit(
                "lead", group=(group.first, group.count), args=job_args, exclusive=True
            )
            microseconds.append(run.wait(handle, timeout=10.0).microseconds)
            # --8<-- [end:submit]
            if job % 10 == 9 or job == jobs - 1:
                log(f"command {job + 1} of {jobs} done")
        mhz.append(aiclk_mhz(session.card))
        run.stop()
        rows = []
        for first, chunk in session.read_chunks(results):
            rows.append(chunk.view(np.uint32))
            log(f"results read from page {first}")
        rows = np.concatenate(rows).reshape(jobs, group.count, -1)
        records = []
        for first, chunk in session.read_chunks(times):
            records.append(chunk.view(np.uint32))
            log(f"times read from page {first}")
        records = np.concatenate(records).reshape(-1, group.count, TIMED_ROUNDS, 4)
    bad = check(rows, choices, args, group, cells)
    per_command = float(np.median(microseconds))
    loads = rows[:, [core - group.first for core in cells], WORDS + 3 : WORDS + 5]
    out = {
        "rect": rect,
        "along": args.along,
        "compiled": args.compiled,
        "addon": args.addon,
        "added": {name: each.parts for name, each in run.added.items()},
        "choices": args.choices,
        "cores": len(cells),
        "broadcasts": len(set(senders.values())),
        "rounds": args.rounds,
        "per_command": args.per,
        "cores_bad": len(bad),
        "first_bad": bad[:8],
        "loads_per_core": {
            "reader": float(loads[..., 0].sum() / len(cells)),
            "with_compute": float(loads[..., 1].sum() / len(cells)),
        },
        "us_per_round": round(per_command / args.per, 3),
        "aiclk_mhz": mhz,
        "medians": timings(records[: len(timed)], cells, senders, group, min(mhz)),
        "kernel_text": sizes,
        "overlay_bytes": (
            {f"{n}-{p}": len(i) for (n, p), i in sorted(built.images.items())}
            if built
            else {}
        ),
        "slots": (
            {part: list(value) for part, value in built.slots.items()} if built else {}
        ),
        "ok": not bad,
    }
    print(
        f"{args.rounds} rounds on {len(cells)} cores, {len(bad)} wrong, "
        f"{out['us_per_round']} us a round"
    )
    print(json.dumps(power.clocked(out), default=float), flush=True)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
