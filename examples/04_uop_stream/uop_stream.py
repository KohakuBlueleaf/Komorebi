"""MAP jobs with random group sizes decided by a micro-op stream: run on the host, on
the X280 submitting each command, and on the X280 enqueuing them to the job queue, each
bit-exact against numpy; then three streams that end in their error events. Prints one
JSON line with `ok` and exits non-zero without it. See README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi.bench import power
from komorebi.device import native
from komorebi.engines import stream as micro_ops
from komorebi.layout import partition
from komorebi.session.state import wrap

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
TILE = 32
STEP = 61  # the input window moves this many tiles per job
AFTER = 20  # jobs of the clean stream after the error cases
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def read_bf16_chunked(session, tensor, name):
    """A bfloat16 tile tensor as session.read_bf16 gives it, read in chunks of 2 MB
    with a line per chunk: a whole read of an output takes a minute or more."""
    total, parts = tensor.pages, []
    for first, pages in session.read_chunks(tensor):
        parts.append(pages)
        log(f"{name}: {first + len(pages)} of {total} pages read")
    rows, columns = tensor.shape[-2:]
    bits = np.concatenate(parts).reshape(-1).view(np.uint16)
    return formats.untilize(formats.bf16_values(bits), rows, columns)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", type=int, default=1000)
    parser.add_argument("--tiles", type=int, default=64, help="input tiles T per job")
    parser.add_argument("--in-tiles", type=int, default=512, help="tiles of the input")
    parser.add_argument(
        "--ways",
        default="hostfed,submit,enqueue,errors",
        help="comma-separated: hostfed, submit (X280, depth 1), enqueue (X280, "
        "queued), errors (three streams that end in an error, then a clean one)",
    )
    parser.add_argument("--depth", type=int, default=16, help="ring slots used")
    parser.add_argument(
        "--deadline", type=float, default=1.0, help="per command, seconds"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="the stream's xorshift seed"
    )
    parser.add_argument("--card", type=int, default=None)
    parser.add_argument("--no-trace", action="store_true", help="no per-command trace")
    return parser.parse_args(argv)


def job_stream(form, jobs, cores, window, addresses, tiles, seed):
    """The stream of every way: per job a group size from RAND, the output page after
    the previous job's, the input window moved by STEP. `form` "submit": a computed
    first core, SUBMIT and WAIT 0; "enqueue": ENQUEUE, the queue picks the group."""
    stream = micro_ops.Stream(req=40 if form == "submit" else 41, seed=seed)
    job, random_word, count, first, room, base, page = stream.registers(7)
    core_count, job_count, window_limit = (
        stream.const(cores),
        stream.const(jobs),
        stream.const(window),
    )
    for register in (job, base, page):
        stream.set(register, 0)
    top = stream.label()
    stream.rand(random_word)
    stream.mod(count, random_word, core_count)
    stream.addi(count, count, 1)  # 1 to every core
    map_args = [addresses["in"], tiles, addresses["out"], base, page]
    if form == "submit":
        stream.sub(room, core_count, count)
        stream.addi(room, room, 1)
        stream.shr(first, random_word, 8)
        stream.mod(first, first, room)
        stream.submit(addresses["map"], (first, count), args=map_args)
        stream.wait(0)
    else:
        stream.enqueue(addresses["map"], count, args=map_args)
    stream.add(base, base, count)
    stream.addi(page, page, STEP)
    inside = stream.forward()
    stream.jlt(page, window_limit, inside)
    stream.sub(page, page, window_limit)
    stream.bind(inside)
    stream.addi(job, job, 1)
    stream.jlt(job, job_count, top)
    stream.wait(0)
    stream.halt(job)
    return stream


def simulated(stream, cores):
    """The commands the kernel submits for `stream`, from the package's simulator:
    [(first, count, output page, first input tile)]."""
    simulation = stream.simulate(core_count=cores, max_passes=1_000_000)
    if simulation.status != "ended":
        raise SystemExit(f"the simulator ends the stream with {simulation.status}")
    return [
        (command.first, command.count, command.args[3], command.args[4])
        for command in simulation.commands
    ]


def expected(values, jobs, tiles):
    """Every output tile: each rank of a job adds its partition of the job's input tiles
    (float64 over integer inputs, exact in bf16)."""
    sums = np.cumsum(values.reshape(-1, TILE, TILE).astype(np.float64), 0)
    prefix = np.concatenate([np.zeros((1, TILE, TILE)), sums])
    pages = sum(count for _, count, _, _ in jobs)
    result = np.zeros((pages, TILE, TILE))
    for _, count, base, start in jobs:
        for rank in range(count):
            tile_range = partition(range(start, start + tiles), rank, count)
            result[base + rank] = prefix[tile_range.stop] - prefix[tile_range.start]
    return result


def ranks_of(jobs, cores):
    """Commands each core is in."""
    edge = np.zeros(cores + 1, np.int64)
    for first, count, _, _ in jobs:
        edge[first] += 1
        edge[first + count] -= 1
    return np.cumsum(edge)[:cores]


def traced(trace):
    """Each traced command in sequence order: (first, count, took, done) with took the
    first core's reader time and done the last core's writer time, in us."""
    offsets = trace.h
    seen = trace.rows[:, :, offsets["kTraceSeen"]]
    done = trace.rows[:, :, offsets["kTraceDone"]]
    slots = np.flatnonzero(seen.any(0))
    if not slots.size:
        return []
    zero = seen[:, slots][seen[:, slots] != 0].min()
    records = []
    for slot in slots:
        cores = np.flatnonzero(seen[:, slot])
        took = wrap(seen[cores, slot] - zero).min() / trace.mhz
        end = wrap(done[cores, slot] - zero).max() / trace.mhz
        records.append((int(cores[0]), len(cores), float(took), float(end)))
    return records


def most_in_flight(commands):
    """The most commands whose (took, done) intervals overlap at one time."""
    edges = sorted(
        [(command[2], 1) for command in commands]
        + [(command[3], -1) for command in commands]
    )
    now = most = 0
    for _, step in edges:
        now += step
        most = max(most, now)
    return most


def main():
    args = parse()
    report = {"jobs": args.jobs, "tiles": args.tiles, "seed": args.seed}
    with km.Session(card=args.card, log=log) as session:
        log("device open")
        if session.dispatcher == "x280":  # a Tensix dispatcher has no X280 image
            report["image_bytes"] = native.image_path().stat().st_size
        report.update(run_ways(session, args))
    report["ok"] = verdict(report, args)
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(power.clocked(report)), flush=True)
    sys.exit(0 if report["ok"] else 1)


def verdict(report, args):
    """Every way exact with every core's count right, as simulated and ended with
    its halt value; the queue with more than one command in flight; each error stream
    ended with its reason and the jobs after them exact."""
    ok = report["same_jobs_submit_enqueue"]
    for way, got in report["ways"].items():
        ok &= got["tiles_wrong"] == 0 and got["cores_bad"] == 0
        ok &= got["status"] == ("ended" if way == "hostfed" else "end")
        ok &= got.get("halt", args.jobs) == args.jobs
        if way != "enqueue":
            ok &= got.get("as_simulated", True)
        elif "traced" in got:
            ok &= got["most_in_flight"] > 1
    if "errors" in report:
        errors = report["errors"]
        reasons = [
            errors[name].get("reason") for name in ("bad_opcode", "steps", "never_fits")
        ]
        after = errors["after"]
        ok &= reasons == ["opcode", "steps", "never_fits"]
        ok &= after["halt"] == min(AFTER, args.jobs) and after["kernel_error"] == 0
        ok &= after["cores_bad"] == 0 and after["tiles_wrong"] == 0
    return bool(ok)


def run_ways(session, args):
    grid_width, grid_height = session.grid
    cores = grid_width * grid_height
    ways = args.ways.split(",")
    tiles, input_tiles = args.tiles, max(args.in_tiles, args.tiles + STEP)
    window = input_tiles - tiles + 1
    generator = np.random.default_rng(args.seed)
    # Integers in [-2, 2]: every partial sum of up to 64 tiles is exact in bf16.
    values = generator.integers(-2, 3, (input_tiles * TILE, TILE)).astype(np.float32)

    def tile_tensor(array):
        return session.tensor(array, "bfloat16", "tile")

    # The jobs, from the simulator of the SUBMIT stream: the same for every way.
    probe = {"in": 0, "out": 0, "map": 1}
    submit_jobs = simulated(
        job_stream("submit", args.jobs, cores, window, probe, tiles, args.seed), cores
    )
    enqueue_jobs = simulated(
        job_stream("enqueue", args.jobs, cores, window, probe, tiles, args.seed), cores
    )
    # The queue may submit a later, smaller job first: the same jobs, in another order.
    same = sorted(job[1:] for job in submit_jobs) == sorted(
        job[1:] for job in enqueue_jobs
    )
    want = expected(values, submit_jobs, tiles)
    pages = len(want)
    log(f"{len(submit_jobs)} jobs, {pages} output tiles, {cores} cores")

    # Tensors before any load: a tensor write waits for idle cores.
    input_tensor = tile_tensor(values)
    size = {way: pages for way in ways}
    if "errors" in ways:  # only the clean stream's pages
        size["errors"] = sum(
            count for _, count, _, _ in submit_jobs[: min(AFTER, args.jobs)]
        )
    output_tensors = {
        way: tile_tensor(np.zeros((size[way] * TILE, TILE), np.float32)) for way in ways
    }
    program = km.Program(
        "uop_stream",
        kernels=KERNELS,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),  # kCbIn
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),  # kCbOut
        ],
        compile_args={
            "reader": input_tensor.tensor_args(),
            "writer": output_tensors[ways[0]].tensor_args(),
        },
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        depth=args.depth,
        handlers=KERNELS / "uop_stream.h",
    )
    result = {"cores": cores, "same_jobs_submit_enqueue": same, "ways": {}}
    for way in ways:
        if way == "errors":
            result["errors"] = error_cases(
                session,
                program,
                args,
                input_tensor,
                output_tensors[way],
                want,
                submit_jobs,
                cores,
                window,
            )
            continue
        addresses = {
            "in": input_tensor.address,
            "out": output_tensors[way].address,
            "map": program.handler("map"),
        }
        result["ways"][way] = one_way(
            session, program, args, way, addresses, submit_jobs, cores, window
        )
        got = read_bf16_chunked(session, output_tensors[way], way).reshape(
            -1, TILE, TILE
        )
        wrong = int(np.count_nonzero((got.astype(np.float64) != want).any((1, 2))))
        result["ways"][way]["tiles_wrong"] = wrong
        log(f"{way}: {wrong} tiles wrong of {pages}")
    return result


def one_way(session, program, args, way, addresses, jobs, cores, window):
    """Loads the program, runs the jobs one way, stops; times and core counts."""
    trace = 0 if args.no_trace else len(jobs)
    run = session.load(program, deadline=args.deadline, trace=trace)
    log(f"{way}: loaded")
    job_count = len(jobs)
    report = {}
    form = "enqueue" if way == "enqueue" else "submit"
    stream = job_stream(
        form, args.jobs, cores, window, addresses, args.tiles, args.seed
    )
    start = time.perf_counter()
    if way == "hostfed":
        hostfed = run.run_hostfed(stream, timeout=60.0)
        report["status"] = hostfed.status
        sent = [
            (command.first, command.count, command.args[3], command.args[4])
            for command in hostfed.commands
        ]
        report["as_simulated"] = sent == jobs
    else:
        upload = run.upload(stream)
        report["upload_bytes"] = upload.bytes
        handle = run.start(upload)
        end = run.wait_engine(handle, timeout=60.0)
        report["status"], report["halt"] = end.kind, end.value
        run.release(upload)
    report["microseconds_per_job"] = round(
        (time.perf_counter() - start) * 1e6 / job_count, 3
    )
    run.stop()
    log(f"{way}: {report['microseconds_per_job']} us per job")
    if trace:
        records = traced(run.trace())[
            :job_count
        ]  # the stop's EXIT may follow in the trace
        report["traced"] = len(records)
        if way == "submit":  # one command in flight: the order is the simulator's
            report["as_simulated"] = [command[:2] for command in records] == [
                job[:2] for job in jobs
            ]
        report["most_in_flight"] = most_in_flight(records)
    states = run.state()
    runs = np.array([state["runs"] for state in states])
    done = np.array([state["completed"] for state in states])
    report["commands"] = run.result()["commands"]
    report["rank_commands"] = int(done.sum())
    if way != "enqueue":  # the queue picks the groups: only the total is known
        rank = ranks_of(jobs, cores)
        report["cores_bad"] = int(np.count_nonzero((runs != rank + 1) | (done != rank)))
    else:
        report["cores_bad"] = int(done.sum() != sum(count for _, count, _, _ in jobs))
    return report


def error_cases(
    session, program, args, input_tensor, output_tensor, want, jobs, cores, window
):
    """An upload with a bad opcode, a stream past its step limit and a job larger than
    the grid, each ending in its error event; then the first jobs of the SUBMIT stream
    on the same run, exact, and a clean stop."""
    addresses = {"in": input_tensor.address, "out": output_tensor.address}
    addresses["map"] = program.handler("map")
    run = session.load(program, deadline=args.deadline)
    bad = bytearray(
        job_stream("submit", 4, cores, window, addresses, args.tiles, 1).encode()
    )
    bad[64] = 0xEE  # the first micro-op's opcode
    steps = micro_ops.Stream(req=42, max_steps=1000)
    (counter,) = steps.registers(1)
    top = steps.label()
    steps.addi(counter, counter, 1)
    steps.jmp(top)
    big = micro_ops.Stream(req=43)
    (group_size,) = big.registers(1)
    big.set(group_size, cores + 1)
    big.enqueue(
        addresses["map"],
        group_size,
        args=[addresses["in"], args.tiles, addresses["out"], 0, 0],
    )
    big.wait(0)
    big.halt(group_size)
    report = {}
    for name, payload in (
        ("bad_opcode", bytes(bad)),
        ("steps", steps),
        ("never_fits", big),
    ):
        upload = run.upload(payload, kind="stream")
        try:
            end = run.wait_engine(run.start(upload), timeout=10.0)
            report[name] = {"ended": end.kind}
        except km.EngineError as error:
            report[name] = {"reason": error.event.reason_name, "pc": error.event.pc}
        except km.EngineRejected as error:
            report[name] = {"rejected": error.reason_name}
        run.release(upload)
        log(f"errors: {name} {report[name]}")
    clean_jobs = min(AFTER, len(jobs))
    clean = job_stream(
        "submit", clean_jobs, cores, window, addresses, args.tiles, args.seed
    )
    upload = run.upload(clean)
    report["after"] = {"halt": run.wait_engine(run.start(upload), timeout=10.0).value}
    run.release(upload)
    result = run.stop()
    report["after"]["kernel_error"] = result["error"]
    done = np.array([state["completed"] for state in run.state()])
    report["after"]["cores_bad"] = int(
        np.count_nonzero(done != ranks_of(jobs[:clean_jobs], cores))
    )
    pages = sum(count for _, count, _, _ in jobs[:clean_jobs])
    got = read_bf16_chunked(session, output_tensor, "errors").reshape(-1, TILE, TILE)
    wrong = (got.astype(np.float64) != want[:pages]).any((1, 2))
    report["after"]["tiles_wrong"] = int(np.count_nonzero(wrong))
    return report


if __name__ == "__main__":
    main()
