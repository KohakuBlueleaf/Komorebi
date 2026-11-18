"""The dispatcher on one Tensix core, fed live: the host writes command lines into a
ring in core 119's L1 while the program runs and reads one event a command from the
host channel. One JSON line a run, e.g. `python live.py bench --n 100000 --depth 16`."""

import argparse
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import komorebi as km
from komorebi.device import mock, native
from komorebi.tools.feeder import commands as feeder_commands
from komorebi.tools.feeder import fed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DISPATCHER = HERE / "kernels" / "dispatcher.cpp"
HELPER = HERE / "kernels" / "helper.cpp"
C = {**feeder_commands.CONSTANTS, **feeder_commands.constants(HERE / "kernels/live.h")}
MAPCOLLECT = ROOT / "experiments" / "checks" / "feeder_runs" / "mapcollect_fed.py"
EXAMPLE = ROOT / "examples" / "01_mapcollect" / "kernels"
HANDLERS = feeder_commands.constants(EXAMPLE / "mapcollect.h")
STATUS = {0: "none", 1: "done", 2: "timeout", 3: "boot", 4: "idle"}
CHANNEL_OFFSET = 0  # the host lines' offset in the session's region
FORMS = ("unicast", "multicast", "loopback", "apart")


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("case", choices=["bench", "preload", "latency", "map"])
    parser.add_argument("--n", type=int, default=1000, help="commands (jobs for map)")
    parser.add_argument("--runs", type=int, default=1, help="runs in one device open")
    parser.add_argument("--depth", type=int, default=16, help="ring slots in use")
    parser.add_argument("--group", type=int, default=1, help="cores (0: every worker)")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--batch", type=int, default=64, help="lines a batch, at most")
    parser.add_argument("--host-depth", type=int, default=64)
    parser.add_argument("--event-depth", type=int, default=128)
    parser.add_argument("--exclusive", action="store_true")
    parser.add_argument("--single", action="store_true", help="one line a write call")
    parser.add_argument("--form", choices=FORMS, default="unicast")
    parser.add_argument(
        "--mover",
        choices=["brisc", "ncrisc", "both", "both-ncrisc"],
        default="brisc",
        help="the loop's processor; both: the loop on BRISC and the helper on NCRISC, "
        "both-ncrisc: the other way",
    )
    parser.add_argument("--pace", type=int, default=0, help="nops between polls")
    parser.add_argument("--parts", action="store_true", help="time each loop part")
    parser.add_argument(
        "--barrier", action="store_true", help="acks of every counts line awaited"
    )
    parser.add_argument("--deadline", type=float, default=0.05, help="seconds")
    parser.add_argument(
        "--wait", type=float, default=2.0, help="workers' wait, seconds"
    )
    parser.add_argument("--idle", type=float, default=3.0, help="dispatcher idle limit")
    parser.add_argument("--tiles", type=int, default=64, help="map: input tiles a job")
    parser.add_argument("--out-pages", type=int, default=2048)
    parser.add_argument("--out-tensors", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def mapcollect_module():
    """experiments/checks/feeder_runs/mapcollect_fed.py, for its checks."""
    spec = importlib.util.spec_from_file_location("mapcollect_fed", MAPCOLLECT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rectangles(session, runner, form):
    """[(NoC 0 start, end, receivers)] covering workers 0 to n - 1, one panel or part
    of one each (the dispatcher is the grid's last core)."""
    if form == "unicast":
        return []
    nocs = session.nocs()
    grid_x, grid_y = runner.grid
    cut = []
    for x0, x1 in nocs.panels:
        holds = x0 <= grid_x - 1 <= x1  # the dispatcher's panel
        if holds and form == "apart":
            cut.append((x0, 0, x1, grid_y - 2))
            if x1 > x0:
                cut.append((x0, grid_y - 1, x1 - 1, grid_y - 1))
            else:
                raise ValueError("a one-column panel holds the dispatcher")
        else:
            cut.append((x0, 0, x1, grid_y - 1))
    out = []
    for x0, y0, x1, y1 in cut:
        a, b = nocs.noc_xy[(x0, y0)], nocs.noc_xy[(x1, y1)]
        cores = (x1 - x0 + 1) * (y1 - y0 + 1)
        holds = x0 <= grid_x - 1 <= x1 and y0 <= grid_y - 1 <= y1
        receivers = cores - 1 if holds and form == "multicast" else cores
        out.append((a[0] | a[1] << 16, b[0] | b[1] << 16, receivers))
    if len(out) > C["kLvRectsMax"]:
        raise ValueError(f"{len(out)} rectangles, at most {C['kLvRectsMax']}")
    return out


class Live:
    """Fed's program with the live dispatcher on the last core; run() feeds it."""

    def __init__(self, session, args):
        self.session, self.args = session, args
        self.runner = fed.Fed(session)
        self.module = session._module or native.load_module(session.tree)
        self.mock = mock.active() or session.channel is None
        if args.host_depth & (args.host_depth - 1) or not 1 <= args.host_depth <= 64:
            raise ValueError("--host-depth: a power of two, 1 to 64")
        if (
            args.event_depth & (args.event_depth - 1)
            or not 1 <= args.event_depth <= 128
        ):
            raise ValueError("--event-depth: a power of two, 1 to 128")
        if args.mover.startswith("both") and args.form == "unicast":
            raise ValueError(
                "--mover both: the helper multicasts rectangle 1 (a --form)"
            )

    def program(self, kernels, trace_cap, rects):
        """The launch: workers as Fed's, the dispatcher (and its helper) on the last
        core. Returns (launch, tensors kept alive, control, results, block address)."""
        runner, session, args = self.runner, self.session, self.args
        mhz = runner.mhz
        control = runner.control_blocks()
        block = runner.l1_blocks(C["kLbBytes"] // 4)
        core_table = np.zeros((1, C["KM_CORES_MAX"]), dtype=np.uint64)
        core_table[0, : runner.core_count] = runner.noc
        cores_t = fed.int32_tensor(session, core_table)
        result_words = C["kLsRows"] * 16 + trace_cap * C["kLtWords"]
        results_t = fed.int32_tensor(session, np.zeros((-(-result_words // 16), 16)))
        grp, n_groups = feeder_commands.groups(runner.grid, runner.workers)
        dispatcher = runner.workers
        base = {
            "control": control.address,
            "cores": cores_t.address,
            "dispatcher_coordinates": runner.noc[dispatcher],
            "ack_base": control.address + C["KM_CONTROL_DONE"],
            "ready_base": control.address + C["KM_CONTROL_BOOT"],
            "tag": 1,
            "depth": args.depth,
            "workers": runner.workers,
            "row": runner.grid[0],
            "wait_limit": feeder_commands.limit_units(args.wait, mhz),
            "idle_limit": 0,
            "trace": 0,
            "trace_cap": 0,
        }
        mailbox = control.address + C["KM_CONTROL_JOB"]
        reader_args, writer_args, compute_args = [], [], []
        for core in range(runner.workers):
            x, y = runner.logical[core]
            leader, rank, peers, ready_slot = grp[core]
            start = feeder_commands.start_args(
                base
                | {
                    "me": core,
                    "mailbox": mailbox if core == 0 else 0,
                    "leader": leader,
                    "rank": rank,
                    "peers": peers,
                    "ready_slot": ready_slot,
                }
            )
            reader_args.append((x, y, start))
            writer_args.append((x, y, start))
            compute = [control.address, core, base["wait_limit"], 0]
            compute_args.append((x, y, compute))
        words = [0] * C["kLvArgs"]
        values = {
            "kLvControl": control.address,
            "kLvWorkers": runner.workers,
            "kLvCores": cores_t.address,
            "kLvBlock": block.address,
            "kLvHostDepth": args.host_depth,
            "kLvEventDepth": args.event_depth,
            "kLvChannel": CHANNEL_OFFSET,
            "kLvDepth": args.depth,
            "kLvDeadline": feeder_commands.limit_units(args.deadline, mhz),
            "kLvResults": results_t.address,
            "kLvGroups": n_groups,
            "kLvBootLimit": feeder_commands.limit_units(args.wait, mhz),
            "kLvIdleLimit": feeder_commands.limit_units(args.idle, mhz),
            "kLvTraceCap": trace_cap,
            "kLvRects": len(rects),
            "kLvHelper": int(args.mover.startswith("both")),
        }
        for name, value in values.items():
            words[C[name]] = value
        for r, rect in enumerate(rects):
            words[C["kLvRect"] + 3 * r : C["kLvRect"] + 3 * r + 3] = rect
        x, y = runner.logical[dispatcher]
        include = [str(feeder_commands.INCLUDE)]
        launch = session.launch()
        tok = C["kTokenBytes"]
        cbs = [
            (i, data_format, page_bytes, page_count, runner.worker_set)
            for i, data_format, page_bytes, page_count in kernels.cbs
        ]
        cbs += [
            (C["kCbTokenCompute"], "uint32", tok, 1, runner.worker_set),
            (C["kCbTokenWriter"], "uint32", tok, 1, runner.worker_set),
            (0, "uint32", -(-result_words // 16) * 64, 1, runner.feeder_set),
        ]
        for index, fmt, page, pages, core_set in cbs:
            launch.cb(core_set, index, fmt, page, page * pages)
        worker = [
            (kernels.reader, reader_args, 0, kernels.reader_ct, {}),
            (kernels.writer, writer_args, 1, kernels.writer_ct, {}),
            (
                kernels.compute,
                compute_args,
                2,
                [],
                fed.compute_config(kernels.compute_options),
            ),
        ]
        for source, kernel_args, processor, compile_arguments, config in worker:
            launch.kernel(
                str(source), runner.worker_set, processor, list(compile_arguments),
                [], include, kernel_args, **config,
            )  # fmt: skip
        defines = []
        if args.parts:
            defines.append(("LIVE_PARTS", "1"))
        if args.barrier:
            defines.append(("LIVE_BARRIER", "1"))
        if args.pace:
            defines.append(("LIVE_PACE", str(args.pace)))
        if args.form == "loopback":
            defines.append(("LIVE_LOOPBACK", "1"))
        loop_processor = 1 if args.mover in ("ncrisc", "both-ncrisc") else 0
        launch.kernel(
            str(DISPATCHER), runner.feeder_set, loop_processor, [], defines, include,
            [(x, y, words)],
        )  # fmt: skip
        if args.mover.startswith("both"):
            launch.kernel(
                str(HELPER), runner.feeder_set, 1 - loop_processor, [], defines,
                include, [(x, y, words)],
            )  # fmt: skip
        kept = (control, block, cores_t, results_t)
        return launch, kept, control, results_t, block.address

    def run(self, kernels, lines, trace_cap=0, preload=0, latency=False):
        """Launches the program and feeds `lines` (16 words each, the last an EXIT):
        the first `preload` written before the launch. Returns a dict."""
        args, runner, session = self.args, self.runner, self.session
        rects = rectangles(session, runner, args.form)
        launch, kept, control, results_t, block = self.program(
            kernels, trace_cap, rects
        )
        x, y = runner.logical[runner.workers]
        n = len(lines)
        stamped = np.asarray(lines, dtype=np.uint64).astype(np.uint32).copy()
        for i in range(n):  # command i + 1: its number in the four landing words
            stamped[i, [0, 4, 8, 12]] = i + 1
        feed = Feed(self.module, session, x, y, block, args, stamped)
        if not self.mock:
            feed.clear()
            for i in range(preload):
                feed.write(i + 1, 1)
        fed.log(f"launch: {n} commands, {len(rects)} rectangles, {args.mover}")
        started = time.perf_counter()
        launch.run()
        if not self.mock:
            feed.next = preload + 1
            feed.loop(latency)
        session.device.finish()
        host_s = time.perf_counter() - started
        fed.log(f"program done in {host_s:.2f}s")
        rows = feeder_commands.as_uint32(fed.read_int32(results_t)).reshape(-1)
        control_words = fed.read_int32(control)
        del kept
        return feed, rows, control_words, host_s, rects


class Feed:
    """The host's side: the ring in the dispatcher's L1 (write_l1), the counts line and
    the event ring in the host channel (read in place)."""

    def __init__(self, module, session, x, y, block, args, lines):
        self.module, self.device = module, session.device
        self.x, self.y, self.block, self.args, self.lines = x, y, block, args, lines
        self.n = len(lines)
        self.next = 1  # the next command to write
        self.seen = 0  # events read
        self.published = 0  # the read count last written
        self.write_calls = []  # (lines, ns)
        self.events = np.zeros((self.n + 1, 4), np.uint32)  # by event index
        self.retries = 0  # polls where told was ahead of the events landed
        self.timeouts = 0  # TIMEOUT events
        self.steps = np.arange(args.event_depth, dtype=np.uint32)
        self.wrap = np.concatenate([self.steps, self.steps]).astype(np.int64)
        self.sent_ns = np.zeros(self.n + 1, np.int64)  # host time of the write
        self.seen_ns = np.zeros(self.n + 1, np.int64)  # when its event was read
        self.taken_ns = np.zeros(self.n + 1, np.int64)  # latency: consumed showed it
        self.told_ns = np.zeros(self.n + 1, np.int64)  # latency: told showed its event
        self.end = None
        if session.channel is not None:
            words = (C["kLhEvents"] + args.event_depth * 16) // 4
            raw = session.channel[CHANNEL_OFFSET : CHANNEL_OFFSET + words * 4]
            self.host = raw.view("<u4")
            self.ring = self.host[C["kLhEvents"] // 4 :].reshape(args.event_depth, 4)

    def clear(self):
        self.host[:] = 0

    def write(self, first, count):
        """Commands first to first + count - 1 into their ring lines: one call for each
        run of lines that does not wrap."""
        depth = self.args.host_depth
        done = 0
        while done < count:
            number = first + done
            slot = number % depth
            run = min(count - done, depth - slot)
            if self.args.single:
                run = 1
            words = self.lines[number - 1 : number - 1 + run].reshape(-1).tolist()
            t0 = time.perf_counter_ns()
            self.module.write_l1(
                self.device,
                self.x,
                self.y,
                self.block + C["kLbRing"] + slot * 64,
                words,
            )
            t1 = time.perf_counter_ns()
            self.write_calls.append((run, t1 - t0))
            self.sent_ns[number : number + run] = t1
            done += run

    def publish(self):
        if self.seen != self.published:
            self.module.write_l1(
                self.device, self.x, self.y, self.block + C["kLbRead"], [self.seen]
            )
            self.published = self.seen

    def poll(self):
        """Reads every event that landed: the ring (each event's index before its
        fields), then the index column again; keeps the events up to the first whose
        two index reads do not both give its number. Returns how many were read."""
        depth = self.args.event_depth
        told = int(self.host[C["kLhTold"]])
        count = told - self.seen
        if count <= 0:
            return 0
        block = self.ring.copy()
        second = self.ring[:, 0].copy()
        start = (self.seen + 1) & (depth - 1)
        at = self.wrap[start : start + count]
        expect = self.steps[:count] + np.uint32(self.seen + 1)
        landed = (block[at, 0] == expect) & (second[at] == expect)
        if not landed.all():
            count = int(landed.argmin())
            self.retries += 1
            if count == 0:
                return 0
            at = at[:count]
        got = block[at]
        self.events[self.seen + 1 : self.seen + 1 + count] = got
        self.seen_ns[np.minimum(got[:, 1], self.n)] = time.perf_counter_ns()
        if (got[:, 2] & 0xFF).any():  # a kind other than DONE: is it a timeout?
            kinds = got[:, 2] & 0xFF
            self.timeouts += int((kinds == C["KM_EVENT_TIMEOUT"]).sum())
        self.seen += count
        return count

    def loop(self, latency):
        """Feeds every command: a batch while the rings have room, then the events;
        `latency`: one command at a time. Ends at the EXIT's event or a final one."""
        args, n = self.args, self.n
        give_up = time.monotonic() + args.idle + 30.0
        while self.seen < n:
            consumed = int(self.host[C["kLhConsumed"]])
            room = min(
                args.host_depth - (self.next - 1 - consumed),
                args.event_depth - (self.next - 1 - self.seen),
                n + 1 - self.next,
                1 if latency else args.batch,
            )
            if latency and self.next - 1 != self.seen:
                room = 0
            if room > 0:
                self.write(self.next, room)
                self.next += room
                if latency:  # when the consumed count shows it taken
                    until = time.perf_counter_ns() + 10**9
                    while int(self.host[C["kLhConsumed"]]) < self.next - 1:
                        if time.perf_counter_ns() > until:
                            break
                    self.taken_ns[self.next - 1] = time.perf_counter_ns()
                    while int(self.host[C["kLhTold"]]) < self.next - 1:
                        if time.perf_counter_ns() > until:
                            break
                    self.told_ns[self.next - 1] = time.perf_counter_ns()
            self.poll()
            self.publish()
            if self.timeouts:
                self.end = "timeout"
                break
            if time.monotonic() > give_up:
                self.end = "host gave up"
                break
        self.poll()

    def checks(self):
        """Events by request: lost, doubled; stale (an event's fields those of the
        event a ring before it) and inconsistent (a DONE whose sequence bits are not its
        request's), as counts."""
        depth = self.args.event_depth
        events = self.events[1 : self.seen + 1].astype(np.int64)
        requests, status = events[:, 1], events[:, 2]
        counts = np.bincount(requests, minlength=self.n + 1)[1 : self.n + 1]
        stale = (events[depth:, 1:] == events[:-depth, 1:]).all(axis=1).sum()
        done = status & 0xFF == C["KM_EVENT_DONE"]
        inconsistent = done & (status >> 16 != requests & 0xFFFF)
        return {
            "events": self.seen,
            "lost": int((counts == 0).sum()),
            "doubled": int((counts > 1).sum()),
            "stale": int(stale),
            "inconsistent": int(inconsistent.sum()),
            "told_ahead_polls": self.retries,
            "timeouts": self.timeouts,
            "rejected": int((status & 0xFF == C["KM_EVENT_REJECTED"]).sum()),
        }

    def write_cost(self):
        """Host microseconds a write call by its lines (median), and the calls."""
        by = {}
        for lines, ns in self.write_calls:
            by.setdefault(lines, []).append(ns)
        return {
            str(lines): {
                "calls": len(times),
                "microseconds_median": round(float(np.median(times)) / 1e3, 2),
            }
            for lines, times in sorted(by.items())
        }


def summary_of(rows, mhz):
    """The dispatcher's two summary rows as a dict."""
    s = rows

    def wide(lo, hi):
        return int(s[C[lo]]) | int(s[C[hi]]) << 32

    span = wide("kLsLastLo", "kLsLastHi") - wide("kLsFirstLo", "kLsFirstHi")
    parts = {
        name: int(s[C[f"kLp{name.capitalize()}"]]) * 16
        for name in ("take", "slot", "write", "sweep", "event", "idle")
    }
    return {
        "magic_ok": int(s[C["kLsMagic"]]) == C["kLiveMagic"],
        "status": STATUS.get(int(s[C["kLsStatus"]]), int(s[C["kLsStatus"]])),
        "taken": int(s[C["kLsTaken"]]),
        "done": int(s[C["kLsDone"]]),
        "pushed": int(s[C["kLsPushed"]]),
        "timed_out": [
            int(s[C["kLsSlot"]]),
            int(s[C["kLsSequence"]]),
            int(s[C["kLsRequest"]]),
        ],
        "boot_microseconds": round(int(s[C["kLsBoot"]]) * 16 / mhz, 1),
        "span_ticks": span,
        "max_busy": int(s[C["kLsMaxBusy"]]),
        "exit_acked": bool(s[C["kLsExitAcked"]]),
        "ready_mask": int(s[C["kLsReady"]]),
        "passes": int(s[C["kLpPasses"]]),
        "empty_passes": int(s[C["kLpEmpty"]]),
        "held_passes": int(s[C["kLpHeld"]]),
        "room_passes": int(s[C["kLpRoom"]]),
        "counts_lines": int(s[C["kLpCounts"]]),
        "rejected": int(s[C["kLpRejected"]]),
        "helper_wait_ticks": int(s[C["kLpHelperWait"]]) * 16,
        "parts_ticks": parts,
        "helper_issue_ticks": int(s[C["kLpHelperIssue"]]) * 16,
        "helper_polls": int(s[C["kLpHelperPolls"]]),
        "helper_lines": int(s[C["kLpHelperLines"]]),
        "hand_over_ticks": int(s[C["kLpHandOver"]]) * 16,
        "hand_back_ticks": int(s[C["kLpHandBack"]]) * 16,
        "loop_polls": int(s[C["kLpLoopPolls"]]),
    }


def trace_of(rows, trace_cap):
    """[cap, 4] trace rows: sequence, slot | status << 8 | held << 16, submit, ack."""
    start = C["kLsRows"] * 16
    return rows[start : start + trace_cap * C["kLtWords"]].reshape(trace_cap, 4)


def nop_kernels(accessor_args=()):
    """The map/collect example's kernels, which serve NOP like every program; their
    reader and writer take a tensor's accessor arguments."""
    return fed.Kernels(
        reader=EXAMPLE / "reader.cpp",
        compute=EXAMPLE / "compute.cpp",
        writer=EXAMPLE / "writer.cpp",
        reader_ct=list(accessor_args),
        writer_ct=list(accessor_args),
        compute_options={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        cbs=[
            (HANDLERS["kCbIn"], "bfloat16", 2048, HANDLERS["kInPages"]),
            (HANDLERS["kCbOut"], "bfloat16", 2048, 2),
        ],
    )


def exit_line():
    return feeder_commands.command(C["KM_HANDLER_EXIT"], 0, 1)


def nop_lines(args, workers):
    count = workers if args.group == 0 else args.group
    flags = C["KM_FLAG_EXCLUSIVE"] if args.exclusive else 0
    nop = feeder_commands.command(C["KM_HANDLER_NOP"], args.first, count, (), flags)
    return [nop] * args.n + [exit_line()]


def timing(out, live, feed, rows, host_s, n):
    mhz = live.runner.mhz
    summary = summary_of(rows, mhz)
    out["dispatcher"] = summary
    out["microseconds_per_command_device"] = round(
        summary["span_ticks"] / mhz / max(n, 1), 3
    )
    out["host_program_s"] = round(host_s, 3)
    if feed.seen:
        sent = feed.sent_ns[1 : n + 1]
        seen = feed.seen_ns[1 : n + 1]
        if (sent > 0).all() and (seen > 0).all():
            out["microseconds_per_command_host"] = round(
                (seen.max() - sent.min()) / 1e3 / n, 3
            )
    out["write_cost"] = feed.write_cost()
    out["events"] = feed.checks()
    out["end_event"] = feed.end
    return summary


def case_nop(session, args, out):
    live = Live(session, args)
    lines = nop_lines(args, live.runner.workers)
    n = args.n
    preload = n + 1 if args.case == "preload" else 0
    if preload > args.host_depth:
        raise ValueError(f"preload: {preload} lines in a ring of {args.host_depth}")
    trace_cap = min(n + 1, 4096) // 4 * 4 if args.case == "latency" else 0
    tile = np.zeros((32, 32), dtype=np.float32)
    unused = session.tensor(tile, "bfloat16", "tile")  # the accessor's tensor only
    feed, rows, control_words, host_s, rects = live.run(
        nop_kernels(unused.accessor_args()),
        lines,
        trace_cap,
        preload,
        latency=args.case == "latency",
    )
    if live.mock:
        return out
    out["rectangles"] = [[hex(a), hex(b), r] for a, b, r in rects]
    summary = timing(out, live, feed, rows, host_s, n)
    ok = summary["status"] == "done" and summary["exit_acked"]
    ok &= summary["done"] == n + 1 and summary["pushed"] == n + 1
    events = out["events"]
    ok &= events["lost"] == 0 and events["doubled"] == 0 and events["stale"] == 0
    ok &= events["inconsistent"] == 0
    count = live.runner.workers if args.group == 0 else args.group
    states = [
        feeder_commands.core_state(feeder_commands.state_words(control_words), core)
        for core in range(live.runner.workers)
    ]
    runs = np.array([state["runs"] for state in states])
    want = np.ones(live.runner.workers, int)  # the EXIT
    want[args.first : args.first + count] += n
    out["cores_runs_wrong"] = int((runs != want).sum())
    ok &= out["cores_runs_wrong"] == 0
    if args.case == "latency":
        mhz = live.runner.mhz
        ticks = feed.events[1 : n + 1, 3].astype(np.int64)
        host = (feed.seen_ns[1 : n + 1] - feed.sent_ns[1 : n + 1]) / 1e3
        trace = trace_of(rows, trace_cap)
        out["latency"] = {
            "device_submit_to_ack_microseconds_median": round(
                float(np.median(ticks)) / mhz, 3
            ),
            "host_write_to_event_microseconds_median": round(float(np.median(host)), 2),
            "host_write_to_event_microseconds_p10_p90": [
                round(float(np.percentile(host, q)), 2) for q in (10, 90)
            ],
            "host_write_to_told_microseconds_median": round(
                float(np.median(feed.told_ns[1 : n + 1] - feed.sent_ns[1 : n + 1]))
                / 1e3,
                2,
            ),
            "host_write_to_consumed_microseconds_median": round(
                float(np.median(feed.taken_ns[1 : n + 1] - feed.sent_ns[1 : n + 1]))
                / 1e3,
                2,
            ),
            "traced": int(trace_cap),
            "device_ack_to_next_submit_microseconds_median": round(
                float(
                    np.median(
                        feeder_commands.wrap(
                            trace[1 : n - 1, 2].astype(np.int64)
                            - trace[: n - 2, 3].astype(np.int64)
                        )
                    )
                )
                / mhz,
                2,
            ),
            "host_event_to_next_write_microseconds_median": round(
                float(np.median(feed.sent_ns[2:n] - feed.seen_ns[1 : n - 1])) / 1e3, 2
            ),
        }
    out["ok"] = bool(ok)
    return out


def case_map(session, args, out):
    """1,000 MAP jobs of the map/collect example on random groups, fed live; the
    example's own checks (mapcollect_fed.py): jobs_bad and ranks_bad."""
    mc = mapcollect_module()
    live = Live(session, args)
    runner = live.runner
    rng = np.random.default_rng(args.seed)
    jobs = args.n
    margs = SimpleNamespace(
        groups="random", group_list=None, jobs=jobs, exit_at=-1, raw=-1, raw_slot=0,
        stall=-1, late=-1, bad_item=-1, result=-1, out_pages=args.out_pages,
        tiles=args.tiles,
    )  # fmt: skip
    groups = mc.make_groups(margs, runner.workers, rng)
    tile = mc.TILE
    tiles = args.tiles
    x_exact = rng.integers(-2, 3, ((tiles + jobs) * tile, tile)).astype(np.float32)
    x_rand = rng.uniform(-1, 1, ((tiles + jobs) * tile, tile)).astype(np.float32)
    x_rand = mc.formats.bf16_values(mc.formats.bf16_bits(x_rand))

    def tile_tensor(array):
        return session.tensor(array, "bfloat16", "tile")

    inputs = [tile_tensor(x_exact), tile_tensor(x_rand)]
    pages = args.out_pages
    zeros = np.zeros((pages * tile, tile), dtype=np.float32)
    outs = [tile_tensor(zeros) for _ in range(args.out_tensors)]
    base = mc.place_outputs(groups, pages, args.out_tensors)
    lines = []
    for j, (first, count) in enumerate(groups):
        words = [
            inputs[j % 2].address,
            tiles,
            outs[base[j] // pages].address,
            base[j] % pages,
            0,
            0xFFFFFFFF,
            j,
        ]
        lines.append(feeder_commands.command(mc.HANDLER_MAP, first, count, words))
    lines.append(exit_line())
    kernels = nop_kernels()
    kernels.reader_ct = inputs[0].accessor_args()
    kernels.writer_ct = outs[0].accessor_args()
    trace_cap = (jobs + 1 + 3) // 4 * 4
    feed, rows, control_words, host_s, rects = live.run(kernels, lines, trace_cap)
    if live.mock:
        return out
    summary = timing(out, live, feed, rows, host_s, jobs)
    trace = trace_of(rows, trace_cap)[:jobs].astype(np.int64)
    result = SimpleNamespace(
        commands={
            "sequence": trace[:, 0],
            "slot": trace[:, 1] & 0xFF,
            "status": (trace[:, 1] >> 8) & 0xFF,
        },
        summary={
            "status": summary["status"],
            "exit_acked": summary["exit_acked"],
            "submitted": jobs,
        },
    )
    states = [
        feeder_commands.core_state(feeder_commands.state_words(control_words), core)
        for core in range(runner.workers)
    ]
    out["ranks_bad"] = mc.rank_errors(margs, groups, result, states)[:16]
    out["compute_wrong"] = sum(sum(state["compute_wrong"]) for state in states)
    out["max_busy"] = summary["max_busy"]
    if summary["status"] == "done":
        out.update(mc.check(margs, groups, base, outs, [x_exact, x_rand]))
    events = out["events"]
    out["ok"] = bool(
        summary["status"] == "done"
        and summary["exit_acked"]
        and not out["ranks_bad"]
        and out["compute_wrong"] == 0
        and out.get("jobs_bad", 1) == 0
        and events["lost"] == 0
        and events["doubled"] == 0
        and events["stale"] == 0
    )
    return out


def main():
    args = parse()
    t_start = time.time()
    with km.Session(log=fed.log) as session:
        fed.log(f"device open {time.time() - t_start:.1f}s")
        for run in range(args.runs):
            out = {
                "case": args.case,
                "run": run,
                "n": args.n,
                "depth": args.depth,
                "group": args.group,
                "batch": args.batch,
                "host_depth": args.host_depth,
                "event_depth": args.event_depth,
                "exclusive": args.exclusive,
                "single": args.single,
                "form": args.form,
                "mover": args.mover,
                "pace": args.pace,
                "parts": args.parts,
                "barrier": args.barrier,
                "aiclk_mhz": fed.aiclk_mhz(),
            }
            if args.case == "map":
                case_map(session, args, out)
            else:
                case_nop(session, args, out)
            out["wall_s"] = round(time.time() - t_start, 1)
            print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
