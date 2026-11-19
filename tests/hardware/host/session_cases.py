"""Card cases of the host package that the example does not cover, one per run
(`python session_cases.py CASE`, CASES below); prints one JSON line with `ok`."""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.engines import stream as micro_ops
from komorebi.engines.graph import Graph

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples" / "01_mapcollect"
ROUTED = ROOT / "tutorial" / "28_routed_message" / "kernels"  # rank 0 sends a0 messages
RUNTIME = ROOT / "src" / "komorebi_native" / "kernels"
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def program(kernels):
    """The example's kernels with no tensors: only NOP and STALL commands are sent."""
    return km.Program(
        "mapcollect",
        kernels=kernels,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
        ],
        compile_args={"reader": [2, 2048], "writer": [2, 2048]},
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        handlers=Path(kernels) / "mapcollect.h",
    )


def case_compile_error(session):
    """A compute kernel that fails to compile: ProgramError, no run loaded; then the
    good program loads and runs in the same session (no image was left running)."""
    out, before = {}, session.run
    with tempfile.TemporaryDirectory() as directory:
        broken = Path(directory) / "kernels"
        shutil.copytree(EXAMPLE / "kernels", broken)
        compute = broken / "compute.cpp"
        compute.write_text(compute.read_text() + "\n#error deliberate compile error\n")
        try:
            session.load(program(broken))
            out["raised"] = None
        except km.ProgramError as error:
            out["raised"] = type(error).__name__
            out["names_error"] = "deliberate compile error" in str(error)
    out["run_after_error"] = session.run is not before
    run = session.load(program(EXAMPLE / "kernels"))
    out["nop_after"] = run.wait(run.submit(0, (0, 8))).kind
    run.stop()
    out["ok"] = (
        out["raised"] == "ProgramError"
        and out["names_error"]
        and not out["run_after_error"]
        and out["nop_after"] == "done"
        and run.result()["error"] == 0
    )
    return out


def case_exclusive(session):
    """A STALL-free check of `exclusive`: a command overlapping one in flight is held
    until that one completed (trace: its cores took it after the leader pushed)."""
    run = session.load(program(EXAMPLE / "kernels"), trace=64)
    pairs = []
    for index in range(20):
        first = run.submit(0, (0, 60))
        second = run.submit(0, (30, 60), exclusive=index % 2 == 1)
        pairs.append((first, second, index % 2 == 1))
    run.drain()
    run.stop()
    trace = run.trace()
    leads = [
        (exclusive, trace.lead(first, second)) for first, second, exclusive in pairs
    ]
    out = {
        "exclusive_leads_microseconds": [
            round(lead, 3) for is_exclusive, lead in leads if is_exclusive
        ],
        "plain_leads_microseconds": [
            round(lead, 3) for is_exclusive, lead in leads if not is_exclusive
        ],
    }
    out["ok"] = all(lead < 0 for is_exclusive, lead in leads if is_exclusive)
    return out


def case_load_busy(session, tiles=128):
    """A host LOAD to the core of a long SUM (`tiles` tiles, example 21's programs)
    waits for it, under either dispatcher: both done, the SUM's event first."""
    sys.path.insert(0, str(ROOT / "examples" / "21_thread_load"))
    import thread_load as example  # noqa: PLC0415 (only this case needs it)

    work = example.Work(4, 2, np.random.default_rng(1))
    tensors = {
        "input": session.tensor(work.input, "bfloat16", "tile"),
        "outs": session.tensor(np.zeros((3 * 4, example.C["TL_LINE_WORDS"])), "uint32"),
    }
    run, sum_program, widen_program = example.load(session, tensors, work)
    words = example.command(tensors, work, 0, 0, 0)
    words[1], words[6] = tiles, 0
    long_sum = run.submit(sum_program.handler("sum"), (0, 1), args=words)
    load = run.load(widen_program, group=(0, 1))
    first = run.wait_any([long_sum, load], timeout=10.0)  # the lower if both came
    later = run.wait(load if first.handle == long_sum else long_sum, timeout=10.0)
    run.stop()
    events = {"sum": first, "load": later} if first.handle == long_sum else {}
    out = {
        name: {"kind": event.kind, "ticks": int(event.ticks)}
        for name, event in events.items()
    }
    out["first"] = "sum" if first.handle == long_sum else "load"
    out["ok"] = bool(
        out["first"] == "sum"
        and out["sum"]["kind"] == "done"
        and out["load"]["kind"] == "done"
    )
    return out


def case_queue_order(session, rounds=1000, jobs=32, count=8, busy=200):
    """Each round: BUSY on every core (its leader reads a page `busy` times), then
    `jobs` queued PUTs of `count` cores, which wait in the kernel's queue, each writing
    the round's value into its page, then an exclusive SUM on every core counting the
    pages that hold it. SUM must wait for every queued job: `jobs` each round."""
    prog, pages = queue_program(session)
    run = session.load(prog)
    address, counts, started = pages.address, [], time.monotonic()
    for round_index in range(rounds):
        run.submit(3, None, args=[address, busy])
        for j in range(jobs):
            run.submit(1, count, args=[address, j, round_index + 1], queued=True)
        total = run.submit(
            2, None, args=[address, jobs, round_index + 1], exclusive=True, result=True
        )
        counts.append(run.wait(total).result)
    run.drain()
    took = time.monotonic() - started
    run.stop()
    short = [
        (round_index, done_count)
        for round_index, done_count in enumerate(counts)
        if done_count != jobs
    ]
    out = {"rounds": rounds, "rounds_short": len(short), "first_short": short[:8]}
    out["ms_per_round"] = round(took / rounds * 1e3, 3)
    out["ok"] = not short and run.result()["error"] == 0
    return out


def case_bursts(session, rounds=200, burst=40, reads=100, depth=64):
    """Rounds of `burst` BUSY commands on cores 0 to burst - 1 at depth `depth`, then
    NOPs: the slots used go up and down, and a Tensix dispatcher's ring bound with them
    (its writes and range in the result's tail). Every event must come done."""
    prog, pages = queue_program(session)
    run = session.load(prog, depth=depth)
    kinds = {}
    for _ in range(rounds):
        handles = [
            run.submit(3, (j, 1), args=[pages.address, reads]) for j in range(burst)
        ]
        handles += [run.submit(0, (0, 1)) for _ in range(4)]
        for event in run.wait(handles, timeout=20.0):
            kinds[event.kind] = kinds.get(event.kind, 0) + 1
    run.stop()
    result = run.result()
    out = {"events": kinds, "max_busy": result["max_busy"]}
    out |= getattr(run.port, "tail", {})
    out["ok"] = kinds == {"done": rounds * (burst + 4)} and result["error"] == 0
    return out


def queue_program(session):
    """The queue cases' program and its tensor: 64 pages of 64 bytes, uint32."""
    pages = session.tensor(np.zeros((64, 16), np.uint32), "uint32")
    prog = km.Program(
        "queue_order",
        kernels=Path(__file__).resolve().parent / "queue_kernels",
        buffers=[km.Buffer(index=16, format="uint32", page_bytes=64, pages=1)],
        compile_args={"writer": pages.accessor_args()},
    )
    return prog, pages


def case_queue_cost(session, windows=10, window=48, count=8, reads=(0, 100)):
    """Where a queued command's time goes: `windows` x `window` commands of `count`
    cores, queued or on fixed disjoint groups, each window submitted and waited for at
    once: PUTs, or BUSY commands whose leader reads a page `reads` times."""
    prog, pages = queue_program(session)
    address, out = pages.address, {}
    for mode in [
        (mode_name, reader) for reader in reads for mode_name in ("plain", "queued")
    ]:
        run = session.load(prog)
        events, started = [], time.monotonic()
        for window_index in range(windows):
            handles = []
            for j in range(window):
                if mode[0] == "queued":
                    group, queued = count, True
                else:
                    group, queued = (count * (j % (run.cores // count)), count), False
                handler, args = (
                    (3, [address, mode[1]])
                    if mode[1]
                    else (1, [address, j, window_index + 1])
                )
                handles.append(run.submit(handler, group, args=args, queued=queued))
            events += run.wait(handles)
        wall = time.monotonic() - started
        run.stop()
        result, commands = run.result(), windows * window
        out[f"{mode[0]}_{mode[1]}_reads"] = {
            "host_microseconds_per_command": round(wall / commands * 1e6, 2),
            "host_submit_microseconds": round(run.submit_s / commands * 1e6, 2),
            "kernel_microseconds_per_command": round(
                result["last"] / run.tick_mhz / commands, 2
            ),
            "submit_to_done_microseconds": round(
                float(np.mean([event.microseconds for event in events])), 2
            ),
            "passes_per_command": round(result["passes"] / commands, 1),
            "max_busy": result["max_busy"],
            "all_done": all(event.kind == "done" for event in events),
        }
    out["ok"] = all(lead["all_done"] for lead in out.values())
    return out


def bf16_tiles(session, x):
    return session.tensor(x, "bfloat16", "tile")


def case_read(session):
    """Tensors read by banks equal what was written and a whole-buffer read (a uint32
    row-major one of 37 pages over every bank, two bfloat16 ones of 6 tiles, one
    barrier), also while a run is loaded. The read's speed is an experiment
    (experiments/host/dram_read)."""
    rng = np.random.default_rng(3)
    words = rng.integers(0, 1 << 32, (37, 24), dtype=np.uint64).astype(np.uint32)
    x = rng.integers(-4, 5, (96, 64)).astype(np.float32)  # exact in bfloat16
    words_tensor = session.tensor(words, "uint32")
    first_tiles, second_tiles = bf16_tiles(session, x), bf16_tiles(session, 2 * x)
    raw, got_a, got_b = session.read(words_tensor)[0], *session.read_bf16(
        first_tiles, second_tiles
    )
    out = {
        "words": bool((raw.reshape(-1).view(np.uint32) == words.reshape(-1)).all()),
        "tiles": bool((got_a == x).all() and (got_b == 2 * x).all()),
        "as_whole": bool((got_a == first_tiles.read()).all()),
    }
    run = session.load(program(EXAMPLE / "kernels"))
    run.wait(run.submit(0, (0, 8)))
    log("read while the run is loaded")
    (during,) = session.read_bf16(first_tiles)
    out["during_run"] = bool((during == x).all())
    run.stop()
    out["ok"] = all(out[name] for name in ("words", "tiles", "as_whole"))
    out["ok"] = out["ok"] and out["during_run"] and run.result()["error"] == 0
    return out


def case_start_error(session):
    """A command deadline under one timer tick: the kernel rejects its arguments before
    it opens a window. StartError with its result; the host stops the grid (every
    reader gave up on the stop flag, not its idle limit); then a run works."""
    out = {}
    started = time.monotonic()
    try:
        session.load(program(EXAMPLE / "kernels"), deadline=1e-9, idle=60.0, start=1.0)
        out["raised"] = None
    except km.StartError as error:
        out["raised"] = type(error).__name__
        out["result"] = error.result is not None
        out["message"] = str(error)[:200]
    out["start_error_s"] = round(time.monotonic() - started, 2)
    failed = session.run
    readers = {}
    for core in failed.state():
        reader = core["reader"]
        key = f"{reader['code']} in {reader['where']}" if reader else "none"
        readers[key] = readers.get(key, 0) + 1
    out["readers"] = readers
    run = session.load(program(EXAMPLE / "kernels"))
    out["nop_after"] = run.wait(run.submit(0, None)).kind
    run.stop()
    out["ok"] = (
        out["raised"] == "StartError"
        and out["result"]
        and readers.get("stop in next", 0) == 120
        and out["nop_after"] == "done"
        and run.result()["error"] == 0
    )
    return out


def routed(session, max_live):
    """Tutorial 28's dynamic program (handler "route": rank 0 sends a0 messages, each
    to rank value * value % the ranks taking messages), with `max_live` live sends."""
    program = km.Program(
        "routed_message",
        kernels={
            "reader": RUNTIME / "messages/reader.cpp",
            "compute": ROUTED / "compute.cpp",
            "writer": RUNTIME / "messages/writer.cpp",
        },
        handlers=ROUTED / "handlers.h",
    ).dynamic("route", message_bytes=64, max_live=max_live, queue_messages=4096)
    notes = session.tensor(
        np.zeros((session.grid[0] * session.grid[1], 4), np.int32),
        "int32",
        shard_grid=session.grid,
    )
    return program, notes


def case_dynamic_gave_up(session, group=8):
    """A dynamic run given up on purpose: max_live the session's ranks (the least a load
    takes), and rank 0 sends twice that, past its quota: MessagesGaveUp naming core 0.
    """
    program, notes = routed(session, max_live=session.ranks)
    run = session.load(program, deadline=5.0)
    out = {"raised": None}
    try:
        run.wait(
            run.submit(
                "route",
                (0, group),
                args=[2 * session.ranks, notes.address],
                exclusive=True,
            ),
            timeout=10.0,
        )
    except km.MessagesGaveUp as error:
        out["raised"] = type(error).__name__
        out["core"], out["where"] = error.gave.get("core"), error.gave.get("where")
    out["ok"] = (
        out["raised"] == "MessagesGaveUp"
        and out["core"] == 0
        and out["where"] == "quota"
    )
    return out


def case_dynamic_queued(session, group=8, messages=12):
    """A dynamic command sent queued is rejected (reason 12, messages) and the run goes
    on: the same command sent plainly then completes."""
    program, notes = routed(session, max_live=4096)
    run = session.load(program)
    out = {"reason": None}
    try:
        run.wait(
            run.submit("route", group, args=[messages, notes.address], queued=True)
        )
    except km.CommandRejected as error:
        out["reason"], out["name"] = error.reason, error.reason_name
    done = run.submit(
        "route", (0, group), args=[messages, notes.address], exclusive=True
    )
    out["after"] = run.wait(done, timeout=10.0).kind
    run.stop()
    out["ok"] = (
        out["reason"] == 12 and out["after"] == "done" and run.result()["error"] == 0
    )
    return out


def case_reset_past_ranks(session):
    """run.reset_core of the first core past the session's ranks (a Tensix
    dispatcher's own core; past the grid under the X280's) is refused; core 0's reset
    and a NOP after it complete."""
    run = session.load(program(EXAMPLE / "kernels"))
    out = {"raised": None}
    try:
        run.reset_core(session.ranks)
    except ValueError as error:
        out["raised"], out["text"] = type(error).__name__, str(error)
    out["by"] = run.reset_core(0)["by"]
    out["nop_after"] = run.wait(run.submit(0, (0, 8))).kind
    run.stop()
    out["ok"] = (
        out["raised"] == "ValueError"
        and out["nop_after"] == "done"
        and run.result()["error"] == 0
    )
    return out


def case_hostfed_never_fits(session):
    """A host-fed stream's ENQUEUE and a graph's command of one core more than the
    ranks end with the error never-fits; a NOP after them completes."""
    run = session.load(program(EXAMPLE / "kernels"))
    stream = micro_ops.Stream(req=50)  # handler 1 (MAP): never sent, it never fits
    stream.enqueue(1, session.ranks + 1)
    stream.halt(stream.const(0))
    graph = Graph()
    graph.command(1, count=session.ranks + 1)
    out = {}
    for name, made in (("stream", stream), ("graph", graph)):
        try:
            result = run.run_hostfed(made, timeout=10.0)
            reason = getattr(result.error, "reason", None)
            out[name] = {"status": result.status, "reason": reason}
        except (ValueError, km.KomorebiError) as error:
            out[name] = {"raised": f"{type(error).__name__}: {error}"}
    out["nop_after"] = run.wait(run.submit(0, (0, 8))).kind
    run.stop()
    never = micro_ops.ERROR_NEVER_FITS
    out["ok"] = (
        all(out[name].get("reason") == never for name in ("stream", "graph"))
        and out["nop_after"] == "done"
        and run.result()["error"] == 0
    )
    return out


class RowCast(km.Program):
    """Each core of its rectangle adds its row's page, cast along the row, into one
    tile at output page rank: made with no rect=, the default rectangle."""

    @km.handler
    def cast(self, rows: km.Tensor, output: km.Tensor):
        row = km.Broadcast(self, along="row")
        total = km.tile.zeros(km.fp32)
        if row.sender():  # noqa: SIM108 (the compiler's form)
            along = row.cast(rows.pages(self.y, 1).read())
        else:
            along = row.arrive()
        for tile in along:
            total += tile
        output.pages(self.rank, 1).write(total)


def case_default_rect(session):
    """A broadcast class with no rect= is submitted to its default rectangle's group
    (the whole grid; under a Tensix dispatcher its whole rows of ranks) and each core's
    tile is its row's page."""
    columns, rows = session.grid
    pages = np.repeat(np.arange(1, rows + 1, dtype=np.float32), 32 * 32)
    source = session.tensor(pages.reshape(-1, 32), "bfloat16", "tile")
    output = session.tensor(
        np.zeros((columns * rows * 32, 32), np.float32), "float32", "tile"
    )
    program = RowCast(rows=source, output=output, log=log)
    run = session.load(program)
    first, count = program.rectangle["group"]
    handle = run.submit(RowCast.cast, (first, count), args=(source, output))
    kind = run.wait(handle).kind
    run.stop()
    tiles = output.read().reshape(-1, 32, 32)[:count]
    want = (np.arange(count) // columns + 1).astype(np.float32)
    out = {
        "group": [first, count],
        "ranks": session.ranks,
        "rect": program.rectangle["rect"],
        "kind": kind,
    }
    out["tiles_wrong"] = int(np.sum(np.any(tiles != want[:, None, None], axis=(1, 2))))
    out["ok"] = (
        kind == "done"
        and first == 0
        and count <= session.ranks
        and out["tiles_wrong"] == 0
        and run.result()["error"] == 0
    )
    return out


CASES = {
    "default-rect": case_default_rect,
    "bursts": case_bursts,
    "compile-error": case_compile_error,
    "dynamic-gave-up": case_dynamic_gave_up,
    "dynamic-queued": case_dynamic_queued,
    "exclusive": case_exclusive,
    "hostfed-never-fits": case_hostfed_never_fits,
    "load-busy": case_load_busy,
    "queue-cost": case_queue_cost,
    "queue-order": case_queue_order,
    "read": case_read,
    "reset-past-ranks": case_reset_past_ranks,
    "start-error": case_start_error,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", choices=sorted(CASES))
    args = parser.parse_args()
    with km.Session(log=log) as session:
        out = {"case": args.case, **CASES[args.case](session)}
    print(json.dumps(out), flush=True)
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
