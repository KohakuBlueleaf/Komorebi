"""Two programs in one run: sum (each core adds its tiles into one) and scale (each
core doubles its tiles), loaded on disjoint groups and combined on every core, commands
to both interleaved, every result checked; then the price of the runtime's form. One
JSON line. Example: python two_programs.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.layout import partition
from komorebi.session import aiclk_mhz

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
TODAY = HERE.parent / "01_mapcollect" / "kernels"  # the same sum in today's form
TILE = 32
INPUT_TILES = 512
TILES = 16  # tiles a command reads
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=0, help="0: the whole grid")
    parser.add_argument("--rounds", type=int, default=40, help="commands a phase")
    parser.add_argument(
        "--phase",
        choices=["disjoint", "combined", "price", "switch", "all"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


# --8<-- [start:programs]
def programs(inputs, sums, doubled, slots=None):
    """sum and scale in the runtime's form: buffers 0 and 16 declared identically,
    so combined they are one shared buffer each; handler 1 in each."""
    buffers = [
        km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
        km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
    ]
    compute = {"math_fidelity": "HiFi4", "fp32_dest_acc_en": True}
    sum_program = km.Program(
        "sum",
        kernels=KERNELS / "sum",  # reader.cpp, compute.cpp, writer.cpp: km_program
        buffers=buffers,
        compile_args={"reader": inputs.accessor_args(), "writer": sums.accessor_args()},
        compute=compute,
        handlers={"sum": 1, "empty": 2},
    )
    scale_program = km.Program(
        "scale",
        kernels=KERNELS / "scale",
        buffers=buffers,
        compile_args={
            "reader": inputs.accessor_args(),
            "writer": doubled.accessor_args(),
        },
        compute=compute,
        handlers={"scale": 1, "empty": 2},
    )
    if slots is not None:
        sum_program.slots(count=slots)
    return sum_program, scale_program


# --8<-- [end:programs]


def today(inputs, sums):
    """The sum of 01_mapcollect in today's form (its MAP takes the same words)."""
    return km.Program(
        "mapcollect",
        kernels=TODAY,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
        ],
        compile_args={"reader": inputs.accessor_args(), "writer": sums.accessor_args()},
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        handlers=TODAY / "mapcollect.h",
    )


class Checked:
    """The tensors, the commands sent and what their results must be."""

    def __init__(self, session, cores, rounds, generator):
        self.session, self.cores, self.generator = session, cores, generator
        values = generator.integers(-2, 3, (INPUT_TILES * TILE, TILE))
        self.input = values.astype(np.float32)  # every sum of 16 tiles exact in bf16
        self.inputs = session.tensor(self.input, "bfloat16", "tile")
        pages = 2 * rounds * cores
        self.sums = session.tensor(np.zeros((pages * TILE, TILE)), "bfloat16", "tile")
        zeros = np.zeros((INPUT_TILES * TILE, TILE))
        self.doubled = session.tensor(zeros, "bfloat16", "tile")
        self.tiles = self.input.reshape(INPUT_TILES, TILE, TILE)
        self.prefix = np.concatenate([np.zeros((1, TILE, TILE)), self.tiles.cumsum(0)])
        self.want_sums = {}  # output page -> tile
        self.scaled = np.zeros(INPUT_TILES, bool)  # input tiles some SCALE doubled
        self.page = 0

    def sum_words(self, group):
        first_tile = int(self.generator.integers(0, INPUT_TILES - TILES + 1))
        page, self.page = self.page, self.page + group[1]
        for rank in range(group[1]):
            share = partition(range(first_tile, first_tile + TILES), rank, group[1])
            self.want_sums[page + rank] = (
                self.prefix[share.stop] - self.prefix[share.start]
            )
        words = [self.inputs.address, TILES, self.sums.address, page, 0, 0xFFFFFFFF]
        return [*words, first_tile]

    def scale_words(self, group):
        first_tile = int(self.generator.integers(0, INPUT_TILES - TILES + 1))
        self.scaled[first_tile : first_tile + TILES] = True
        return [self.inputs.address, TILES, self.doubled.address, 0, 0, 0, first_tile]

    def wrong(self):
        """Values that differ from what the commands must have written."""
        sums, doubled = self.session.read_bf16(self.sums, self.doubled)
        sums = sums.reshape(-1, TILE, TILE)
        wrong = sum(
            int((sums[page] != tile).sum()) for page, tile in self.want_sums.items()
        )
        want = np.where(self.scaled[:, None, None], 2 * self.tiles, 0)
        return wrong + int((doubled.reshape(-1, TILE, TILE) != want).sum())


def group_in(generator, first, count):
    """A random group inside cores [first, first + count)."""
    size = int(generator.integers(1, count + 1))
    return (first + int(generator.integers(0, count - size + 1)), size)


def interleaved(run, checked, groups, rounds, sum_handler, scale_handler):
    """SUM and SCALE in turn, each on a group `groups(program)` draws; all waited."""
    handles = []
    for _ in range(rounds):
        group = groups("sum")
        handles.append(run.submit(sum_handler, group, checked.sum_words(group)))
        group = groups("scale")
        handles.append(run.submit(scale_handler, group, checked.scale_words(group)))
    run.drain(timeout=60.0)
    return len(handles)


def checked_phase(session, args, cores, combined):
    """Disjoint groups ({sum: first part, scale: the rest}) or combined ([sum,
    scale] on every core): interleaved commands, every result checked."""
    generator = np.random.default_rng(args.seed)
    checked = Checked(session, cores, args.rounds, generator)
    sum_program, scale_program = programs(checked.inputs, checked.sums, checked.doubled)
    grid = session.grid
    every = grid[0] * grid[1]
    split = max(1, cores * 11 // 15)
    # --8<-- [start:load]
    if combined:
        run = session.load([sum_program, scale_program])  # every core holds both
    else:
        run = session.load(
            {sum_program: (0, split), scale_program: (split, every - split)}
        )
    # --8<-- [end:load]

    def groups(name):
        if combined:
            return group_in(generator, 0, cores)
        if name == "sum":
            return group_in(generator, 0, split)
        return group_in(generator, split, max(1, cores - split))

    try:
        sent = interleaved(
            run,
            checked,
            groups,
            args.rounds,
            sum_program.handler("sum"),  # handler 1 of sum
            scale_program.handler("scale"),  # handler 1 of scale
        )
        refused = refusal(run, sum_program, split, every, combined)
    finally:
        run.stop()
    wrong = checked.wrong()
    log(f"{'combined' if combined else 'disjoint'}: {sent} commands, {wrong} wrong")
    return {"commands": sent, "wrong": wrong, "refused": refused}


def refusal(run, sum_program, split, every, combined):
    """Whether the host refuses SUM to cores that do not hold sum (disjoint)."""
    if combined:
        return None
    try:
        run.submit(sum_program.handler("empty"), (split, every - split))
    except ValueError:
        return True
    return False


def one_at_a_time(run, commands, count):
    """Each (key, handler, group, words) `count` times, one command at a time; then
    the run stops. {key: median over commands of the median core's cycles from the
    reader taking the command to its completion}, the first two of each left out."""
    handles = {}
    for key, handler, group, words in commands:
        for _ in range(count):
            handle = run.submit(handler, group, words)
            run.wait(handle, 10.0)
            handles.setdefault(key, []).append(handle)
    run.stop()
    trace = run.trace()
    out = {}
    for key, made in handles.items():
        cycles = []
        for handle in made[2:]:
            times = trace.command(handle).astype(np.int64)
            cycles.append(np.median((times[:, 2] - times[:, 0]) % (1 << 32)))
        out[key] = float(np.median(cycles))
        log(f"{key}: {out[key]:.0f} cycles a command")
    return out


def price(session, args, cores):
    """The same sum in today's form (01_mapcollect) and in the runtime's form: cycles
    per command, an empty one and a sum of 16 tiles, on one core and on `cores`."""
    generator = np.random.default_rng(args.seed)
    checked = Checked(session, cores, args.rounds, generator)
    sum_program, _ = programs(checked.inputs, checked.sums, checked.doubled)
    forms = {
        "today": (today(checked.inputs, checked.sums), 7),  # 7: MAP's default branch
        "program": (sum_program, sum_program.handler("empty")),
    }
    out = {}
    for form, (program, empty) in forms.items():
        commands = []
        for group in ((0, 1), (0, cores)):
            words = checked.sum_words(group)
            commands.append((f"{form} empty {group[1]}", empty, group, words))
            commands.append(
                (f"{form} sum {group[1]}", program.handler(1), group, words)
            )
        run = session.load(program, trace=4 * args.rounds + 2)
        out.update(one_at_a_time(run, commands, args.rounds))
    return out


def switch(session, args, cores):
    """Combined with one slot a kernel: commands to sum and scale in turn load the
    other program's five parts each time; the same program again is held."""
    generator = np.random.default_rng(args.seed)
    checked = Checked(session, cores, args.rounds, generator)
    sum_program, scale_program = programs(
        checked.inputs, checked.sums, checked.doubled, slots=1
    )
    run = session.load([sum_program, scale_program], trace=6 * args.rounds + 2)
    commands = []
    for group in ((0, 1), (0, cores)):
        empty = sum_program.handler("empty")
        commands += [(f"held {group[1]}", empty, group, [])] * args.rounds
        for turn in range(args.rounds):  # in turn: each command loads
            program = scale_program if turn % 2 else sum_program
            commands.append((f"switch {group[1]}", program.handler("empty"), group, []))
    out = one_at_a_time(run, commands, 1)
    parts = {
        f"{number}-{part}": len(image)
        for (number, part), image in sorted(sum_program.overlays_built.images.items())
    }
    return out, parts


def main():
    args = parse()
    report = {"phase": args.phase}
    with km.Session(log=log) as session:
        grid = session.grid
        cores = args.cores or grid[0] * grid[1]
        log(f"device open, grid {grid}; commands to {cores} cores")
        report["cores"] = cores
        report["aiclk_mhz"] = [aiclk_mhz(session.card)]
        if args.phase in ("disjoint", "all"):
            report["disjoint"] = checked_phase(session, args, cores, False)
        if args.phase in ("combined", "all"):
            report["combined"] = checked_phase(session, args, cores, True)
        if args.phase in ("price", "all"):
            report["price_cycles"] = price(session, args, cores)
        if args.phase in ("switch", "all"):
            report["switch_cycles"], report["part_bytes"] = switch(session, args, cores)
        report["aiclk_mhz"].append(aiclk_mhz(session.card))
    wrong = sum(
        report[key]["wrong"] for key in ("disjoint", "combined") if key in report
    )
    refused = report.get("disjoint", {}).get("refused", True)
    report["ok"] = wrong == 0 and refused
    print(json.dumps(report, default=float), flush=True)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
