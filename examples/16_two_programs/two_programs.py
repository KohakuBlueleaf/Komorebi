"""Two programs in one run, sum and scale, on disjoint groups and combined, every result
checked; the price of the runtime's form; a group that switches program on the live run
(with widen, a third program of other buffers and formats); --cards 0,1: the same on
two cards, groups over flat ranks. One JSON line. Example: python two_programs.py"""

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
KERNEL_FORM = HERE.parent / "01_mapcollect" / "kernels"  # the sum, kernel form
TILE = 32
INPUT_TILES = 512
TILES = 16  # tiles a command reads
REJECT_PROGRAM = 13  # the dispatcher: a core of the group does not hold it
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
        choices=[
            "disjoint",
            "combined",
            "price",
            "switch",
            "live",
            "load",
            "merged",
            "all",
        ],
        default="all",
    )
    parser.add_argument(
        "--switches", type=int, default=200, help="live: loads into the part"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cards", default="", help="e.g. 0,1: a session of islands")
    return parser.parse_args()


# --8<-- [start:programs]
def programs(inputs, sums, doubled, slots=None, wide=None):
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
        compile_args={"reader": inputs.tensor_args(), "writer": sums.tensor_args()},
        compute=compute,
        handlers={"sum": 1, "empty": 2},
    )
    scale_program = km.Program(
        "scale",
        kernels=KERNELS / "scale",
        buffers=buffers,
        compile_args={
            "reader": inputs.tensor_args(),
            "writer": doubled.tensor_args(),
        },
        compute=compute,
        handlers={"scale": 1, "empty": 2},
    )
    if slots is not None:
        sum_program.slots(count=slots)
    if wide is None:
        return sum_program, scale_program
    widen_program = km.Program(
        "widen",  # three buffers, bfloat16 in at index 1, float32 out at index 16
        kernels=KERNELS / "widen",
        buffers=[
            km.Buffer(index=1, format="bfloat16", page_bytes=2048, pages=4),
            km.Buffer(index=16, format="float32", page_bytes=4096, pages=2),
            km.Buffer(index=24, format="uint32", page_bytes=64, pages=1),
        ],
        compile_args={"reader": inputs.tensor_args(), "writer": wide.tensor_args()},
        compute=compute,
        handlers={"widen": 1},
    )
    return sum_program, scale_program, widen_program


# --8<-- [end:programs]


def kernel_form(inputs, sums):
    """The sum of 01_mapcollect in the kernel form (its MAP takes the same words)."""
    return km.Program(
        "mapcollect",
        kernels=KERNEL_FORM,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
        ],
        compile_args={"reader": inputs.tensor_args(), "writer": sums.tensor_args()},
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        handlers=KERNEL_FORM / "mapcollect.h",
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
        self.wide = session.tensor(zeros.astype(np.float32), "float32", "tile")
        self.tiles = self.input.reshape(INPUT_TILES, TILE, TILE)
        self.prefix = np.concatenate([np.zeros((1, TILE, TILE)), self.tiles.cumsum(0)])
        self.want_sums = {}  # output page -> tile
        self.scaled = np.zeros(INPUT_TILES, bool)  # input tiles some SCALE doubled
        self.widened = np.zeros(INPUT_TILES, bool)  # input tiles some WIDEN wrote
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
        words = [self.inputs.address, TILES, self.doubled.address, first_tile, 0, 0]
        return [*words, first_tile]

    def widen_words(self, group):
        first_tile = int(self.generator.integers(0, INPUT_TILES - TILES + 1))
        self.widened[first_tile : first_tile + TILES] = True
        words = [self.inputs.address, TILES, self.wide.address, first_tile, 0, 0]
        return [*words, first_tile]

    def words(self, name, group):
        """The words of a command of program `name` on `group`."""
        return getattr(self, f"{name}_words")(group)

    def wrong(self):
        """Values that differ from what the commands must have written."""
        sums, doubled = self.session.read_bf16(self.sums, self.doubled)
        sums = sums.reshape(-1, TILE, TILE)
        wrong = sum(
            int((sums[page] != tile).sum()) for page, tile in self.want_sums.items()
        )
        want = np.where(self.scaled[:, None, None], 2 * self.tiles, 0)
        wrong += int((doubled.reshape(-1, TILE, TILE) != want).sum())
        if self.widened.any():
            (pages,) = self.session.read(self.wide)  # float32 tiles, four faces each
            faces = pages.reshape(-1).view(np.float32).reshape(-1, 2, 2, 16, 16)
            wide = faces.transpose(0, 1, 3, 2, 4).reshape(-1, TILE, TILE)
            want = np.where(self.widened[:, None, None], self.tiles, 0)
            wrong += int((wide != want).sum())
        return wrong

    def next_block(self):
        """Output pages of SUM are used again: what was checked is forgotten."""
        self.want_sums, self.page = {}, 0


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
    every = ranks_of(session)
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
    """The same sum in the kernel form (01_mapcollect) and in the runtime's form: cycles
    per command, an empty one and a sum of 16 tiles, on one core and on `cores`."""
    generator = np.random.default_rng(args.seed)
    checked = Checked(session, cores, args.rounds, generator)
    sum_program, _ = programs(checked.inputs, checked.sums, checked.doubled)
    forms = {
        "kernel": (
            kernel_form(checked.inputs, checked.sums),
            7,
        ),  # 7: MAP's default branch
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


def live_groups(cores, boundary=0):
    """{name: group} of the live phase on `cores` (at least 2), and the part of sum's
    group that switches: half of sum's cores; scale and widen share the rest. With an
    island `boundary`, sum is cut at it and its part is the far island's half, where
    scale and widen have groups (a part switches only to a program its island holds)."""
    held = max(1, cores // 2) if not boundary else boundary + (cores - boundary) // 2
    others = cores - held
    wide = max(1, others // 5)
    groups = {"sum": (0, held)}
    if others - wide > 0:
        groups["scale"] = (held, others - wide)
    groups["widen"] = (cores - wide, wide)
    part = max(1, held // 2) if not boundary else held - boundary
    return groups, (held - part, part)


def live(session, args, cores):
    """A part of sum's group switches to scale or widen and back while commands for
    every program run; one SUM to the part while it holds another program is sent
    unchecked by the host and must be refused by the dispatcher (reason 13)."""
    cores = max(2, cores)
    generator = np.random.default_rng(args.seed)
    block = 50  # rounds between two checks of every result
    checked = Checked(session, cores, block, generator)
    made = programs(checked.inputs, checked.sums, checked.doubled, wide=checked.wide)
    by_name = {program.name: program for program in made}
    boundary = session.graph.first(1) if hasattr(session, "graph") else 0
    groups, part = live_groups(cores, boundary)
    # --8<-- [start:live]
    run = session.load({by_name[name]: group for name, group in groups.items()})
    run.load(by_name["widen"], group=part)  # part's cores now hold widen
    # --8<-- [end:live]
    out_refused = None
    if boundary:  # widen has no group on island 0: a load there is refused
        try:
            run.load(by_name["widen"], group=(0, 1))
            out_refused = False
        except km.ProgramError as error:
            out_refused = "no group on this island" in str(error)
    run.wait(run.load(by_name["sum"], group=part), 10.0)
    others = [name for name in groups if name != "sum"]
    rest = (0, part[0])  # the sum cores that never switch
    out = {"switches": 0, "commands": 0, "refused": 0, "refused_other": 0, "wrong": 0}
    if out_refused is not None:
        out["load_refused"] = out_refused
        out["wrong"] += not out_refused
    refused = []

    def send(name, group):
        handler = by_name[name].handler(name)
        out["commands"] += 1
        return run.submit(handler, group, checked.words(name, group))

    try:
        for start in range(0, args.switches // 2, block):
            handles = []
            for turn in range(start, min(start + block, args.switches // 2)):
                other = others[turn % len(others)]
                handles.append(run.load(by_name[other], group=part))
                handles.append(send(other, part))
                if rest[1]:
                    handles.append(send("sum", rest))
                handles.append(send(other, groups[other]))
                run.check_programs = False  # the dispatcher's own refusal
                refused.append(run.submit(by_name["sum"].handler("sum"), part))
                run.check_programs = True
                handles.append(run.load(by_name["sum"], group=part))
                handles.append(send("sum", groups["sum"]))
                out["switches"] += 2
            run.drain(timeout=60.0)
            for handle in refused:
                event = event_of(run, handle)
                right = event.kind == "rejected" and event.auxiliary == REJECT_PROGRAM
                out["refused" if right else "refused_other"] += 1
            kinds = {event_of(run, handle).kind for handle in handles}
            out["wrong"] += checked.wrong() + (kinds != {"done"})
            run.forget([*handles, *refused])
            refused.clear()
            checked.next_block()
            log(f"live: {out['switches']} switches, {out['wrong']} wrong")
    finally:
        run.stop()
    out["commands"] += out["switches"]  # a load is a command too
    return out


def load_time(session, args, cores):
    """Cycles of a load command (the reader taking it to its completion, median
    core): to another program and to the program held, on one core and on `cores`."""
    cores = max(2, cores)
    generator = np.random.default_rng(args.seed)
    checked = Checked(session, cores, 1, generator)
    sum_program, _, widen_program = programs(
        checked.inputs, checked.sums, checked.doubled, wide=checked.wide
    )
    run = session.load(
        {sum_program: (0, cores - 1), widen_program: (cores - 1, 1)},
        trace=8 * args.rounds + 4,
    )
    handles = {}
    for group in ((0, 1), (0, cores)):
        for key, program in (("other", widen_program), ("other", sum_program)) * (
            args.rounds // 2
        ) + (("held", sum_program),) * args.rounds:
            handle = run.load(program, group=group)
            run.wait(handle, 10.0)
            handles.setdefault(f"load {key} {group[1]}", []).append(handle)
    run.stop()
    trace = run.trace()
    out = {}
    for key, made in handles.items():
        cycles = []
        for handle in made[2:]:
            times = trace.command(handle).astype(np.int64)
            cycles.append(np.median((times[:, 2] - times[:, 0]) % (1 << 32)))
        out[key] = float(np.median(cycles))
        log(f"{key}: {out[key]:.0f} cycles a load")
    return out


def merged_phase(session, args, cores):
    """sum and scale merged into ONE program (km.merge): commands in turn on random
    groups, every result checked; then the cycles a command of the two as a list and
    merged: an empty command and a sum of 16 tiles, on one core and on `cores`."""
    generator = np.random.default_rng(args.seed)
    checked = Checked(session, cores, args.rounds, generator)
    sum_program, scale_program = programs(checked.inputs, checked.sums, checked.doubled)
    # --8<-- [start:merged]
    merged = km.merge(sum_program, scale_program, log=log)  # handlers 1, 2 and 3, 4
    run = session.load(merged)
    sum_handler = merged.handler("sum")  # sum's alone: 1
    scale_handler = scale_program.handler("scale")  # scale's own 1, mapped: 3
    # --8<-- [end:merged]
    try:
        sent = interleaved(
            run,
            checked,
            lambda name: group_in(generator, 0, cores),
            args.rounds,
            sum_handler,
            scale_handler,
        )
    finally:
        run.stop()
    wrong = checked.wrong()
    log(f"merged: {sent} commands, {wrong} wrong")
    out = {"commands": sent, "wrong": wrong, "cycles": {}}
    forms = {
        "list": (
            [sum_program, scale_program],
            sum_program.handler("empty"),
            sum_program.handler("sum"),
        ),
        "merged": (merged, merged.handler(sum_program, "empty"), sum_handler),
    }
    for form, (loaded, empty, sums) in forms.items():
        commands = []
        for group in ((0, 1), (0, cores)):
            words = checked.sum_words(group)
            commands.append((f"{form} empty {group[1]}", empty, group, words))
            commands.append((f"{form} sum {group[1]}", sums, group, words))
        run = session.load(loaded, trace=4 * args.rounds + 2)
        out["cycles"].update(one_at_a_time(run, commands, args.rounds))
    return out


def ranks_of(session):
    """Every core of the grid; on a session of islands, every flat rank."""
    if hasattr(session, "graph"):
        return session.ranks
    grid = session.grid
    return grid[0] * grid[1]


def event_of(run, handle):
    """A command's event; over islands (a part an island) the first part not done,
    else the first."""
    event = run.event(handle)
    if hasattr(event, "kind"):  # one Event (a NamedTuple), not a tuple of parts
        return event
    return next((part for part in event if part.kind != "done"), event[0])


def main():
    args = parse()
    report = {"phase": args.phase}
    cards = [int(card) for card in args.cards.split(",")] if args.cards else []
    opened = {"cards": cards, "links": 1} if len(cards) > 1 else {}
    with km.Session(log=log, **opened) as session:
        card = session.sessions[0].card if opened else session.card
        cores = args.cores or ranks_of(session)
        log(f"device open; commands to {cores} cores, cards {cards or 'one'}")
        report["cores"] = cores
        report["cards"] = cards
        report["aiclk_mhz"] = [aiclk_mhz(card)]
        if args.phase in ("disjoint", "all"):
            report["disjoint"] = checked_phase(session, args, cores, False)
        if args.phase in ("combined", "all"):
            report["combined"] = checked_phase(session, args, cores, True)
        if args.phase in ("price", "all"):
            report["price_cycles"] = price(session, args, cores)
        if args.phase in ("switch", "all"):
            report["switch_cycles"], report["part_bytes"] = switch(session, args, cores)
        if args.phase in ("live", "all"):
            report["live"] = live(session, args, cores)
        if args.phase in ("load", "all"):
            report["load_cycles"] = load_time(session, args, cores)
        if args.phase in ("merged", "all"):
            report["merged"] = merged_phase(session, args, cores)
        report["aiclk_mhz"].append(aiclk_mhz(card))
    wrong = sum(
        report[key]["wrong"]
        for key in ("disjoint", "combined", "live", "merged")
        if key in report
    )
    refused = report.get("disjoint", {}).get("refused", True)
    refused &= report.get("live", {}).get("refused_other", 0) == 0
    report["ok"] = wrong == 0 and refused
    print(json.dumps(report, default=float), flush=True)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
