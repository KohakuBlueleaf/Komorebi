"""Graph executor cases on a card, on the harness of dispatch_card.py: the saved
reference graphs, the RUN forms, the kept table and CANCEL; a command past its deadline.
python graph_card.py <case> [options] prints one JSON line, as dispatch_card.py does.
"""

import argparse
import contextlib
import json
import os
import struct
import time

import dispatch_card

from komorebi.engines import graph as graph_engine
from komorebi.engines import stream as stream_engine
from komorebi.engines.graph import Graph

GRAPHS = os.path.join(
    dispatch_card.HERE,
    "..",
    "..",
    "..",
    "experiments",
    "checks",
    "graph_checks",
    "graphs",
)
HANDLER_SUM = 5  # kernels/dk_test.h DK_HANDLER_SUM: result a0 + count
REJECT_HANDLER = dispatch_card.REJECT_HANDLER
GRAPH_ENDS = (graph_engine.EVENT_GRAPH_END, graph_engine.EVENT_GRAPH_ERROR)


def graph_run(offset, size, bind_off=0, n_bind=0):
    """KM_HANDLER_RUN of an upload: a0 offset, a1 bytes, a2 = 1, a3 / a4 the
    bindings."""
    words = dispatch_card.run_command(offset, size)
    words[6], words[7], words[9] = graph_engine.GRAPH_RUN_UPLOAD, bind_off, n_bind
    return words


def kept_run(req, bind_off=0, n_bind=0):
    """KM_HANDLER_RUN of a kept graph: a0 its req, a2 = 2, a3 / a4 the bindings."""
    words = dispatch_card.kernel_command(dispatch_card.HANDLER_RUN, req)
    words[6], words[7], words[9] = graph_engine.GRAPH_RUN_KEPT, bind_off, n_bind
    return words


def on_sum(data):
    """The upload with every template's handler that is a program's (1 to 239) made
    DK_HANDLER_SUM, so the stand-in worker answers a0 + count; other handlers stay."""
    data = bytearray(data)
    if len(data) < graph_engine.HEADER.size:
        return bytes(data)
    header = dict(
        zip(graph_engine.HEADER_FIELDS, graph_engine.HEADER.unpack_from(data))
    )
    template = graph_engine._layout(header)[2]
    for i in range(min(header["command_count"], graph_engine.MAX_TEMPLATES)):
        offset = template + i * graph_engine.TEMPLATE_BYTES + 4
        if offset < len(data) and 1 <= data[offset] <= 239:
            data[offset] = HANDLER_SUM
    return bytes(data)


def a0_plus_count(command):
    return command.args[0] + command.count


def references():
    """The saved reference graphs: [(name, upload, saved json)]."""
    out = []
    for name in sorted(os.listdir(GRAPHS)):
        if name.endswith(".json"):
            stem = name[:-5]
            with open(os.path.join(GRAPHS, stem + ".bin"), "rb") as file:
                data = file.read()
            with open(os.path.join(GRAPHS, name)) as file:
                out.append((stem, data, json.load(file)))
    return out


def graph_events(plan, reqs, timeout_s=20.0):
    """Every event until each graph in `reqs` has ended (KM_EVENT_GRAPH_END or _ERROR),
    the kernel finished or timeout_s passed; a line per batch and every 5 s."""
    reqs = {reqs} if isinstance(reqs, int) else set(reqs)
    out, start = [], time.time()
    ended, said = set(), start
    while ended != reqs and time.time() < start + timeout_s:
        batch = []
        with contextlib.suppress(RuntimeError):
            batch = plan.take(plan.seen + 1, 1.0)
        out += batch
        if batch:
            dispatch_card.log(f"{len(batch)} events, {len(out)} in all")
        ended = {
            event[1]
            for event in out
            if event[1] in reqs and dispatch_card.kind(event[2])[0] in GRAPH_ENDS
        }
        if plan.host_word(0) >= dispatch_card.STATE_DONE:
            break
        if time.time() - said > 5.0:
            said = time.time()
            dispatch_card.log(
                f"waiting for graphs {sorted(reqs - ended)}: {len(out)} events"
            )
    return out


def as_tuples(events, req):
    """(kind, req, auxiliary, ticks of an end event else 0) of the events with `req`."""
    out = []
    for event in events:
        kind, auxiliary = dispatch_card.kind(event[2])
        if event[1] == req:
            out.append(
                (
                    kind,
                    event[1],
                    auxiliary,
                    event[3] if kind == graph_engine.EVENT_GRAPH_END else 0,
                )
            )
    return out


def wanted(simulation):
    return [
        (event.kind, event.req, event.auxiliary, event.extra)
        for event in simulation.events
    ]


def put_bindings(plan, offset, values):
    data = struct.pack(
        f"<{len(values)}I", *[value & dispatch_card.M32 for value in values]
    )
    dispatch_card.put_upload(plan, offset, data)


def filler(req):
    """A kept graph of one JOIN, `req` its own."""
    graph = Graph(req=req, keep=True)
    graph.join()
    return graph.encode()


def delay_graph(req, cycles):
    """One DK_HANDLER_DELAY command on core 0 for `cycles` Tensix cycles."""
    graph = Graph(req=req)
    graph.command(dispatch_card.HANDLER_DELAY, group=(0, 1), args=[req, cycles])
    return graph.encode()


def loop_graph(req, keep):
    """The documented loop (docs/graphs.md) on DK_HANDLER_SUM: rounds while r < 65."""
    graph = Graph(max_runs=100, req=req, keep=keep)
    i, result_register = graph.bindings("it"), graph.register()
    root = graph.command(HANDLER_SUM, group=(0, 60), args=[i])
    left = graph.command(HANDLER_SUM, group=(0, 30), args=[i], after=[root])
    right = graph.command(HANDLER_SUM, group=(30, 30), args=[i], after=[root])
    merge = graph.command(
        HANDLER_SUM,
        group=(0, 60),
        args=[i],
        after=[left, right],
        result=result_register,
    )
    update = graph.cpu(after=[merge])
    update.addi(i, i, 1)
    update.lt(graph.again, result_register, graph.const(65))
    return graph


def match_groups(entries, commands):
    """Trace entries (kind, slot, req, first, count, ..) of one graph against its
    simulated commands: the counts are the same, every command with a group of its own
    has it; a queued command takes the next entry of its count left. Returns (fault or
    None, [(first, count, a0)] as the grid ran them)."""
    left = [(entry[3], entry[4]) for entry in entries]
    groups = []
    for command in commands:
        if not command.queued and (command.first, command.count) in left:
            left.remove((command.first, command.count))
            groups.append((command.first, command.count, command.args[0]))
    for command in sorted(
        (command for command in commands if command.queued),
        key=lambda command: command.sequence,
    ):
        first = next(
            (
                candidate_first
                for candidate_first, count in left
                if count == command.count
            ),
            None,
        )
        if first is not None:
            left.remove((first, command.count))
            groups.append((first, command.count, command.args[0]))
    if len(groups) == len(commands) and not left:
        return None, groups
    want = sorted(
        (command.count, None if command.queued else command.first)
        for command in commands
    )
    got = sorted((entry[4], entry[3]) for entry in entries)
    return {"got": got[:8], "want": want[:8]}, groups


class Layout:
    """Places in the upload region above the trace part, 4 KiB each."""

    def __init__(self, plan):
        self.base = dispatch_card.stream_base(plan)

    def slot_address(self, slot):
        return self.base + 4096 * slot


def case_graphs(session, args):
    """The test image: RUN forms, rejections and range errors, the kept table and
    CANCEL, every saved reference against the simulator on this grid (DK_HANDLER_SUM),
    the kept loop again with another binding; a PING answers after each part. The
    engine's commands reach the grid with the simulated groups, every core ran them."""
    if args.image == dispatch_card.IMAGE:
        args.image = dispatch_card.VERIFY_IMAGE
    core_count, trace, submitted = 0, {}, []

    def ping(plan, name, out):
        i = plan.submit(dispatch_card.kernel_command(dispatch_card.HANDLER_PING))
        event = [record for record in plan.take(plan.seen + 1, 5.0) if record[1] == i]
        out["pings"][name] = bool(
            event and dispatch_card.kind(event[0][2])[0] == dispatch_card.EVENT_PONG
        )

    def expect(plan, out, name, words, want):
        """One command and the events it gives at once (rejections, drops, errors)."""
        i = plan.submit(words)
        event = plan.take(plan.seen + 1, 5.0)
        got = [
            (
                dispatch_card.kind(record[2])[0],
                record[1],
                dispatch_card.kind(record[2])[1],
            )
            for record in event
        ]
        out["run"].append({"name": name, "got": got, "ok": got == [want(i)]})

    def one(plan, out, name, data, bindings, lay, kept=0, words=None):
        """Upload (unless `words` names the RUN), run, compare with the simulator."""
        simulation = graph_engine.simulate(
            data,
            bindings=bindings,
            core_count=core_count,
            result=a0_plus_count,
            kept=kept,
        )
        req = simulation.events[0].req
        put_bindings(plan, lay.slot_address(1), bindings or [0])
        if words is None:
            dispatch_card.put_upload(plan, lay.slot_address(2), data)
            words = graph_run(
                lay.slot_address(2), len(data), lay.slot_address(1), len(bindings)
            )
        plan.submit(words)
        got = as_tuples(graph_events(plan, req), req)
        matched = got == wanted(simulation)
        out["graphs"][name] = matched
        if not matched:
            out["wrong"].append(
                {"name": name, "got": got[:6], "want": wanted(simulation)[:6]}
            )
        submitted.append((name, simulation))
        dispatch_card.log(f"graph {name}: {'ok' if matched else 'WRONG'}")

    def body(plan):
        nonlocal core_count
        core_count = plan.grid_x * plan.grid_y
        lay = Layout(plan)
        out = {"tag": plan.tag, "pings": {}, "graphs": {}, "wrong": [], "run": []}
        ping(plan, "start", out)
        error_hdr = stream_engine.ERROR_HEADER | graph_engine.NO_NODE << 8
        cancelled = stream_engine.ERROR_CANCELLED | graph_engine.NO_NODE << 8
        cycles = int(0.3 * dispatch_card.AICLK_HZ)

        # Forms, rejections and range errors, with two graphs running (long DELAYs).
        for j, req in enumerate((300, 301)):
            data = delay_graph(req, cycles)
            dispatch_card.put_upload(plan, lay.slot_address(10 + j), data)
            plan.submit(graph_run(lay.slot_address(10 + j), len(data)))
        data = delay_graph(302, cycles)
        dispatch_card.put_upload(plan, lay.slot_address(12), data)
        expect(
            plan,
            out,
            "third graph",
            graph_run(lay.slot_address(12), len(data)),
            lambda i: (dispatch_card.EVENT_REJECTED, i, dispatch_card.REJECT_STREAMS),
        )
        words = graph_run(lay.slot_address(12), len(data))
        words[6] = 3
        expect(
            plan,
            out,
            "form 3",
            words,
            lambda i: (dispatch_card.EVENT_REJECTED, i, REJECT_HANDLER),
        )
        for req in (300, 301):
            plan.submit(dispatch_card.kernel_command(dispatch_card.HANDLER_CANCEL, req))
            got = as_tuples(graph_events(plan, req), req)
            out["run"].append(
                {
                    "name": f"cancel {req}",
                    "got": got,
                    "ok": got == [(graph_engine.EVENT_GRAPH_ERROR, req, cancelled, 0)],
                }
            )
        time.sleep(0.5)  # the cancelled graphs' commands complete meanwhile
        expect(
            plan,
            out,
            "kept unknown",
            kept_run(999),
            lambda i: (dispatch_card.EVENT_REJECTED, i, dispatch_card.REJECT_UNKNOWN),
        )
        stream = dispatch_card.loop_stream(400).encode()
        dispatch_card.put_upload(plan, lay.slot_address(13), stream)
        plan.submit(dispatch_card.run_command(lay.slot_address(13), len(stream)))
        expect(
            plan,
            out,
            "overlaps a stream",
            graph_run(lay.slot_address(13), len(stream)),
            lambda i: (graph_engine.EVENT_GRAPH_ERROR, 400, error_hdr),
        )
        expect(
            plan,
            out,
            "unaligned",
            graph_run(lay.slot_address(12) + 32, len(data)),
            lambda i: (graph_engine.EVENT_GRAPH_ERROR, 0, error_hdr),
        )
        expect(
            plan,
            out,
            "cancel the stream",
            dispatch_card.kernel_command(dispatch_card.HANDLER_CANCEL, 400),
            lambda i: (
                dispatch_card.EVENT_STREAM_ERROR,
                400,
                stream_engine.ERROR_CANCELLED,
            ),
        )
        ping(plan, "after forms", out)

        # The kept table: seven kept, the reference kept_graph is the eighth (then
        # dropped), eight kept again and kept_table_full is rejected; all dropped.
        fill = {700 + j: filler(700 + j) for j in range(8)}
        for j, data in enumerate(fill.values()):
            dispatch_card.put_upload(plan, lay.slot_address(20 + j), data)
        for j, (req, data) in enumerate(list(fill.items())[:7]):
            plan.submit(graph_run(lay.slot_address(20 + j), len(data)))
            graph_events(plan, req)
        refs = {name: (data, saved) for name, data, saved in references()}
        data, saved = refs.pop("kept_graph")
        one(plan, out, "kept_graph", on_sum(data), saved["bindings"], lay, kept=7)
        req = graph_engine.HEADER.unpack_from(data)[6]
        expect(
            plan,
            out,
            "drop kept_graph",
            dispatch_card.kernel_command(dispatch_card.HANDLER_CANCEL, req),
            lambda i: (graph_engine.EVENT_GRAPH_ERROR, req, cancelled),
        )
        plan.submit(graph_run(lay.slot_address(27), len(fill[707])))
        graph_events(plan, 707)
        data, saved = refs.pop("kept_table_full")
        one(plan, out, "kept_table_full", on_sum(data), saved["bindings"], lay, kept=8)
        for req in fill:
            expect(
                plan,
                out,
                f"drop {req}",
                dispatch_card.kernel_command(dispatch_card.HANDLER_CANCEL, req),
                lambda i, req=req: (graph_engine.EVENT_GRAPH_ERROR, req, cancelled),
            )
        ping(plan, "after kept", out)

        # Every other saved reference that terminates, on this grid.
        for name, (data, saved) in refs.items():
            if saved["params"]["delay"] is None:
                continue
            one(plan, out, name, on_sum(data), saved["bindings"], lay)
        ping(plan, "after references", out)

        # The loop kept, run again by its req with another binding, then dropped.
        graph = loop_graph(500, keep=True)
        data = graph.encode()
        one(plan, out, "loop it=0", data, graph.binding_values(it=0), lay)
        bindings = graph.binding_values(it=2)
        words = kept_run(500, lay.slot_address(1), len(bindings))
        one(plan, out, "loop kept it=2", data, bindings, lay, words=words)
        expect(
            plan,
            out,
            "drop the loop",
            dispatch_card.kernel_command(dispatch_card.HANDLER_CANCEL, 500),
            lambda i: (graph_engine.EVENT_GRAPH_ERROR, 500, cancelled),
        )
        ping(plan, "after the loop", out)
        out["exit"] = dispatch_card.exit_and_stop(plan)
        return out

    def after(plan, fin):
        trace["head"], trace["entries"] = dispatch_card.read_trace(plan)

    out, fin, counters = dispatch_card.with_kernel(
        session, args, body, depth=16, after=after
    )
    out["finish"] = fin
    out["cores"] = dispatch_card.core_report(counters, core_count, out["tag"])
    submits = (dispatch_card.TRACE_SUBMIT, dispatch_card.TRACE_QUEUED)
    engine = [
        record for record in trace["entries"] if record[0] in submits and record[6] == 1
    ]
    # The two cancelled DELAY graphs' commands come first, then each graph's in turn.
    ran = [(record[3], record[4], 300 + j) for j, record in enumerate(engine[:2])]
    next_command, faults = 2, []
    for name, simulation in submitted:
        mine = engine[next_command : next_command + len(simulation.commands)]
        next_command += len(simulation.commands)
        fault, groups = match_groups(mine, simulation.commands)
        if fault:
            faults.append({"name": name, **fault})
        ran += groups
    out["engine_commands"] = [len(engine), next_command]
    out["group_faults"] = faults
    out["group_mismatch"], out["first_mismatch"] = dispatch_card.check_groups(
        counters, core_count, ran
    )
    out["ok"] = bool(
        dispatch_card.finished_clean(fin, core_count, out["cores"])
        and all(case_result["ok"] for case_result in out["run"])
        and all(out["pings"].values())
        and all(out["graphs"].values())
        and len(engine) == next_command
        and not faults
        and out["group_mismatch"] == 0
        and trace["head"][2] == 0
    )
    return out


def case_graph_deadline(session, args):
    """A graph's second command has rank --core waiting for the stop flag, header
    deadline --deadline-microseconds: KM_EVENT_GRAPH_ERROR reason 8 at that node, the
    kernel ends with the deadline error, every core gives up on the stop flag."""
    core_count = 0

    def body(plan):
        nonlocal core_count
        core_count = plan.grid_x * plan.grid_y
        graph = Graph(req=56, deadline_microseconds=args.deadline_microseconds)
        root = graph.command(HANDLER_SUM, group=(0, core_count), args=[3])
        graph.command(
            dispatch_card.HANDLER_STALL,
            group=(0, core_count),
            args=[9, args.core],
            after=[root],
        )
        data = graph.encode()
        dispatch_card.put_upload(plan, dispatch_card.stream_base(plan), data)
        plan.submit(graph_run(dispatch_card.stream_base(plan), len(data)))
        dispatch_card.log("graph started")
        out = {"tag": plan.tag, "events": graph_events(plan, 56)}
        dispatch_card.log("graph ended")
        return out

    out, fin, counters = dispatch_card.with_kernel(session, args, body, depth=4)
    out["finish"] = fin
    out["cores"] = dispatch_card.core_report(counters, core_count, out["tag"])
    blk = fin["block"]
    got = as_tuples(out["events"], 56)
    out["ok"] = bool(
        got
        == [
            (
                graph_engine.EVENT_GRAPH_ERROR,
                56,
                stream_engine.ERROR_DEADLINE | 1 << 8,
                0,
            )
        ]
        and blk["error"] == dispatch_card.ERROR_DEADLINE
        and blk["source"] == dispatch_card.SOURCE_ENGINE
        and blk["engine"] == dispatch_card.ENGINE_STOPPED
        and out["cores"]["cores_finished"] == core_count
        and out["cores"]["gave"] == {"stop": core_count}
    )
    return out


CASES = {"graphs": case_graphs, "graph-deadline": case_graph_deadline}


def case_set(session, args):
    """Both cases in one device open; a JSON line per case."""
    oks = {}
    for name in CASES:
        dispatch_card.log(f"case {name}")
        out = CASES[name](session, argparse.Namespace(**vars(args)))
        print(json.dumps({"case": name, **out}, default=str), flush=True)
        oks[name] = bool(out.get("ok"))
    return {"cases": oks, "ok": all(oks.values())}


if __name__ == "__main__":
    ALL = {**CASES, "set": case_set}
    dispatch_card.main(ALL, dispatch_card.parser(ALL).parse_args())
