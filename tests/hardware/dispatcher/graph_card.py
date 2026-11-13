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

import dispatch_card as dc

from komorebi.engines import graph as gr
from komorebi.engines import stream as st
from komorebi.engines.graph import Graph

GRAPHS = os.path.join(
    dc.HERE, "..", "..", "..", "experiments", "graph_checks", "graphs"
)
HANDLER_SUM = 5  # kernels/dk_test.h DK_HANDLER_SUM: result a0 + count
REJECT_HANDLER = dc.REJECT_HANDLER
GRAPH_ENDS = (gr.EV_GRAPH_END, gr.EV_GRAPH_ERROR)


def graph_run(offset, size, bind_off=0, n_bind=0):
    """KM_HANDLER_RUN of an upload: a0 offset, a1 bytes, a2 = 1, a3 / a4 the
    bindings."""
    words = dc.run_command(offset, size)
    words[6], words[7], words[9] = gr.GRUN_UPLOAD, bind_off, n_bind
    return words


def kept_run(req, bind_off=0, n_bind=0):
    """KM_HANDLER_RUN of a kept graph: a0 its req, a2 = 2, a3 / a4 the bindings."""
    words = dc.kernel_command(dc.HANDLER_RUN, req)
    words[6], words[7], words[9] = gr.GRUN_KEPT, bind_off, n_bind
    return words


def on_sum(data):
    """The upload with every template's handler that is a program's (1 to 239) made
    DK_HANDLER_SUM, so the stand-in worker answers a0 + count; other handlers stay."""
    data = bytearray(data)
    if len(data) < gr.HEADER.size:
        return bytes(data)
    h = dict(zip(gr.HEADER_FIELDS, gr.HEADER.unpack_from(data)))
    tpl = gr._layout(h)[2]
    for i in range(min(h["n_commands"], gr.MAX_TEMPLATES)):
        at = tpl + i * gr.TEMPLATE_BYTES + 4
        if at < len(data) and 1 <= data[at] <= 239:
            data[at] = HANDLER_SUM
    return bytes(data)


def a0_plus_count(command):
    return command.args[0] + command.count


def references():
    """The saved reference graphs: [(name, upload, saved json)]."""
    out = []
    for name in sorted(os.listdir(GRAPHS)):
        if name.endswith(".json"):
            stem = name[:-5]
            with open(os.path.join(GRAPHS, stem + ".bin"), "rb") as f:
                data = f.read()
            with open(os.path.join(GRAPHS, name)) as f:
                out.append((stem, data, json.load(f)))
    return out


def graph_events(plan, reqs, timeout_s=20.0):
    """Every event until each graph in `reqs` has ended (KM_EV_GRAPH_END or _ERROR),
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
            dc.log(f"{len(batch)} events, {len(out)} in all")
        ended = {e[1] for e in out if e[1] in reqs and dc.kind(e[2])[0] in GRAPH_ENDS}
        if plan.host_word(0) >= dc.STATE_DONE:
            break
        if time.time() - said > 5.0:
            said = time.time()
            dc.log(f"waiting for graphs {sorted(reqs - ended)}: {len(out)} events")
    return out


def as_tuples(events, req):
    """(kind, req, aux, ticks of an end event else 0) of the events with `req`."""
    out = []
    for e in events:
        k, aux = dc.kind(e[2])
        if e[1] == req:
            out.append((k, e[1], aux, e[3] if k == gr.EV_GRAPH_END else 0))
    return out


def wanted(sim):
    return [(e.kind, e.req, e.aux, e.extra) for e in sim.events]


def put_bindings(plan, offset, values):
    data = struct.pack(f"<{len(values)}I", *[v & dc.M32 for v in values])
    dc.put_upload(plan, offset, data)


def filler(req):
    """A kept graph of one JOIN, `req` its own."""
    g = Graph(req=req, keep=True)
    g.join()
    return g.encode()


def delay_graph(req, cycles):
    """One DK_HANDLER_DELAY command on core 0 for `cycles` Tensix cycles."""
    g = Graph(req=req)
    g.command(dc.HANDLER_DELAY, group=(0, 1), args=[req, cycles])
    return g.encode()


def loop_graph(req, keep):
    """The documented loop (docs/graphs.md) on DK_HANDLER_SUM: rounds while r < 65."""
    g = Graph(max_runs=100, req=req, keep=keep)
    i, err = g.bindings("it"), g.reg()
    a = g.command(HANDLER_SUM, group=(0, 60), args=[i])
    b = g.command(HANDLER_SUM, group=(0, 30), args=[i], after=[a])
    c = g.command(HANDLER_SUM, group=(30, 30), args=[i], after=[a])
    d = g.command(HANDLER_SUM, group=(0, 60), args=[i], after=[b, c], result=err)
    e = g.cpu(after=[d])
    e.addi(i, i, 1)
    e.lt(g.again, err, g.const(65))
    return g


def match_groups(entries, commands):
    """Trace entries (kind, slot, req, first, count, ..) of one graph against its
    simulated commands: the counts are the same, every command with a group of its own
    has it; a queued command takes the next entry of its count left. Returns (fault or
    None, [(first, count, a0)] as the grid ran them)."""
    left = [(e[3], e[4]) for e in entries]
    groups = []
    for r in commands:
        if not r.queued and (r.first, r.count) in left:
            left.remove((r.first, r.count))
            groups.append((r.first, r.count, r.args[0]))
    for r in sorted((r for r in commands if r.queued), key=lambda r: r.seq):
        first = next((f for f, c in left if c == r.count), None)
        if first is not None:
            left.remove((first, r.count))
            groups.append((first, r.count, r.args[0]))
    if len(groups) == len(commands) and not left:
        return None, groups
    want = sorted((r.count, None if r.queued else r.first) for r in commands)
    got = sorted((e[4], e[3]) for e in entries)
    return {"got": got[:8], "want": want[:8]}, groups


class Layout:
    """Places in the upload region above the trace part, 4 KiB each."""

    def __init__(self, plan):
        self.base = dc.stream_base(plan)

    def at(self, k):
        return self.base + 4096 * k


def case_graphs(session, args):
    """The test image: RUN forms, rejections and range errors, the kept table and
    CANCEL, every saved reference against the simulator on this grid (DK_HANDLER_SUM),
    the kept loop again with another binding; a PING answers after each part. The
    engine's commands reach the grid with the simulated groups, every core ran them."""
    if args.image == dc.IMAGE:
        args.image = dc.VERIFY_IMAGE
    n_cores, trace, submitted = 0, {}, []

    def ping(plan, name, out):
        i = plan.submit(dc.kernel_command(dc.HANDLER_PING))
        ev = [e for e in plan.take(plan.seen + 1, 5.0) if e[1] == i]
        out["pings"][name] = bool(ev and dc.kind(ev[0][2])[0] == dc.EV_PONG)

    def expect(plan, out, name, words, want):
        """One command and the events it gives at once (rejections, drops, errors)."""
        i = plan.submit(words)
        ev = plan.take(plan.seen + 1, 5.0)
        got = [(dc.kind(e[2])[0], e[1], dc.kind(e[2])[1]) for e in ev]
        out["run"].append({"name": name, "got": got, "ok": got == [want(i)]})

    def one(plan, out, name, data, bindings, lay, kept=0, words=None):
        """Upload (unless `words` names the RUN), run, compare with the simulator."""
        sim = gr.simulate(
            data, bindings=bindings, n_cores=n_cores, result=a0_plus_count, kept=kept
        )
        req = sim.events[0].req
        put_bindings(plan, lay.at(1), bindings or [0])
        if words is None:
            dc.put_upload(plan, lay.at(2), data)
            words = graph_run(lay.at(2), len(data), lay.at(1), len(bindings))
        plan.submit(words)
        got = as_tuples(graph_events(plan, req), req)
        ok = got == wanted(sim)
        out["graphs"][name] = ok
        if not ok:
            out["wrong"].append({"name": name, "got": got[:6], "want": wanted(sim)[:6]})
        submitted.append((name, sim))
        dc.log(f"graph {name}: {'ok' if ok else 'WRONG'}")

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        lay = Layout(plan)
        out = {"tag": plan.tag, "pings": {}, "graphs": {}, "wrong": [], "run": []}
        ping(plan, "start", out)
        err_hdr = st.ERR_HEADER | gr.NO_NODE << 8
        cancelled = st.ERR_CANCELLED | gr.NO_NODE << 8
        cycles = int(0.3 * dc.AICLK_HZ)

        # Forms, rejections and range errors, with two graphs running (long DELAYs).
        for j, req in enumerate((300, 301)):
            data = delay_graph(req, cycles)
            dc.put_upload(plan, lay.at(10 + j), data)
            plan.submit(graph_run(lay.at(10 + j), len(data)))
        data = delay_graph(302, cycles)
        dc.put_upload(plan, lay.at(12), data)
        expect(
            plan,
            out,
            "third graph",
            graph_run(lay.at(12), len(data)),
            lambda i: (dc.EV_REJECTED, i, dc.REJECT_STREAMS),
        )
        words = graph_run(lay.at(12), len(data))
        words[6] = 3
        expect(
            plan, out, "form 3", words, lambda i: (dc.EV_REJECTED, i, REJECT_HANDLER)
        )
        for req in (300, 301):
            plan.submit(dc.kernel_command(dc.HANDLER_CANCEL, req))
            got = as_tuples(graph_events(plan, req), req)
            out["run"].append(
                {
                    "name": f"cancel {req}",
                    "got": got,
                    "ok": got == [(gr.EV_GRAPH_ERROR, req, cancelled, 0)],
                }
            )
        time.sleep(0.5)  # the cancelled graphs' commands complete meanwhile
        expect(
            plan,
            out,
            "kept unknown",
            kept_run(999),
            lambda i: (dc.EV_REJECTED, i, dc.REJECT_UNKNOWN),
        )
        stream = dc.loop_stream(400).encode()
        dc.put_upload(plan, lay.at(13), stream)
        plan.submit(dc.run_command(lay.at(13), len(stream)))
        expect(
            plan,
            out,
            "overlaps a stream",
            graph_run(lay.at(13), len(stream)),
            lambda i: (gr.EV_GRAPH_ERROR, 400, err_hdr),
        )
        expect(
            plan,
            out,
            "unaligned",
            graph_run(lay.at(12) + 32, len(data)),
            lambda i: (gr.EV_GRAPH_ERROR, 0, err_hdr),
        )
        expect(
            plan,
            out,
            "cancel the stream",
            dc.kernel_command(dc.HANDLER_CANCEL, 400),
            lambda i: (dc.EV_STREAM_ERROR, 400, st.ERR_CANCELLED),
        )
        ping(plan, "after forms", out)

        # The kept table: seven kept, the reference kept_graph is the eighth (then
        # dropped), eight kept again and kept_table_full is rejected; all dropped.
        fill = {700 + j: filler(700 + j) for j in range(8)}
        for j, data in enumerate(fill.values()):
            dc.put_upload(plan, lay.at(20 + j), data)
        for j, (req, data) in enumerate(list(fill.items())[:7]):
            plan.submit(graph_run(lay.at(20 + j), len(data)))
            graph_events(plan, req)
        refs = {name: (data, saved) for name, data, saved in references()}
        data, saved = refs.pop("kept_graph")
        one(plan, out, "kept_graph", on_sum(data), saved["bindings"], lay, kept=7)
        req = gr.HEADER.unpack_from(data)[6]
        expect(
            plan,
            out,
            "drop kept_graph",
            dc.kernel_command(dc.HANDLER_CANCEL, req),
            lambda i: (gr.EV_GRAPH_ERROR, req, cancelled),
        )
        plan.submit(graph_run(lay.at(27), len(fill[707])))
        graph_events(plan, 707)
        data, saved = refs.pop("kept_table_full")
        one(plan, out, "kept_table_full", on_sum(data), saved["bindings"], lay, kept=8)
        for req in fill:
            expect(
                plan,
                out,
                f"drop {req}",
                dc.kernel_command(dc.HANDLER_CANCEL, req),
                lambda i, req=req: (gr.EV_GRAPH_ERROR, req, cancelled),
            )
        ping(plan, "after kept", out)

        # Every other saved reference that ends by itself, on this grid.
        for name, (data, saved) in refs.items():
            if saved["params"]["delay"] is None:
                continue
            one(plan, out, name, on_sum(data), saved["bindings"], lay)
        ping(plan, "after references", out)

        # The loop kept, run again by its req with another binding, then dropped.
        g = loop_graph(500, keep=True)
        data = g.encode()
        one(plan, out, "loop it=0", data, g.binding_values(it=0), lay)
        bindings = g.binding_values(it=2)
        words = kept_run(500, lay.at(1), len(bindings))
        one(plan, out, "loop kept it=2", data, bindings, lay, words=words)
        expect(
            plan,
            out,
            "drop the loop",
            dc.kernel_command(dc.HANDLER_CANCEL, 500),
            lambda i: (gr.EV_GRAPH_ERROR, 500, cancelled),
        )
        ping(plan, "after the loop", out)
        out["exit"] = dc.exit_and_stop(plan)
        return out

    def after(plan, fin):
        trace["head"], trace["entries"] = dc.read_trace(plan)

    out, fin, cnt = dc.with_kernel(session, args, body, depth=16, after=after)
    out["finish"] = fin
    out["cores"] = dc.core_report(cnt, n_cores, out["tag"])
    submits = (dc.TRACE_SUBMIT, dc.TRACE_QUEUED)
    engine = [e for e in trace["entries"] if e[0] in submits and e[6] == 1]
    # The two cancelled DELAY graphs' commands come first, then each graph's in turn.
    ran = [(e[3], e[4], 300 + j) for j, e in enumerate(engine[:2])]
    k, faults = 2, []
    for name, sim in submitted:
        mine = engine[k : k + len(sim.commands)]
        k += len(sim.commands)
        fault, groups = match_groups(mine, sim.commands)
        if fault:
            faults.append({"name": name, **fault})
        ran += groups
    out["engine_commands"] = [len(engine), k]
    out["group_faults"] = faults
    out["group_mismatch"], out["first_mismatch"] = dc.check_groups(cnt, n_cores, ran)
    out["ok"] = bool(
        dc.finished_clean(fin, n_cores, out["cores"])
        and all(c["ok"] for c in out["run"])
        and all(out["pings"].values())
        and all(out["graphs"].values())
        and len(engine) == k
        and not faults
        and out["group_mismatch"] == 0
        and trace["head"][2] == 0
    )
    return out


def case_graph_deadline(session, args):
    """A graph's second command has member --core waiting for the stop flag, header
    deadline --deadline-us: KM_EV_GRAPH_ERROR reason 8 at that node, the kernel ends
    with the deadline error, every core gives up on the stop flag."""
    n_cores = 0

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        g = Graph(req=56, deadline_us=args.deadline_us)
        a = g.command(HANDLER_SUM, group=(0, n_cores), args=[3])
        g.command(dc.HANDLER_STALL, group=(0, n_cores), args=[9, args.core], after=[a])
        data = g.encode()
        dc.put_upload(plan, dc.stream_base(plan), data)
        plan.submit(graph_run(dc.stream_base(plan), len(data)))
        dc.log("graph started")
        out = {"tag": plan.tag, "events": graph_events(plan, 56)}
        dc.log("graph ended")
        return out

    out, fin, cnt = dc.with_kernel(session, args, body, depth=4)
    out["finish"] = fin
    out["cores"] = dc.core_report(cnt, n_cores, out["tag"])
    blk = fin["block"]
    got = as_tuples(out["events"], 56)
    out["ok"] = bool(
        got == [(gr.EV_GRAPH_ERROR, 56, st.ERR_DEADLINE | 1 << 8, 0)]
        and blk["error"] == dc.ERR_DEADLINE
        and blk["source"] == dc.SOURCE_ENGINE
        and blk["engine"] == dc.ENGINE_STOPPED
        and out["cores"]["cores_finished"] == n_cores
        and out["cores"]["gave"] == {"stop": n_cores}
    )
    return out


CASES = {"graphs": case_graphs, "graph-deadline": case_graph_deadline}


def case_set(session, args):
    """Both cases in one device open; a JSON line per case."""
    oks = {}
    for name in CASES:
        dc.log(f"case {name}")
        out = CASES[name](session, argparse.Namespace(**vars(args)))
        print(json.dumps({"case": name, **out}, default=str), flush=True)
        oks[name] = bool(out.get("ok"))
    return {"cases": oks, "ok": all(oks.values())}


if __name__ == "__main__":
    ALL = {**CASES, "set": case_set}
    dc.main(ALL, dc.parser(ALL).parse_args())
