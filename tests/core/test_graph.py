"""Graphs: header constants, encoding, check, the executor's semantics, and the C engine
against the saved reference graphs."""

import functools
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from komorebi.engines import graph as gr
from komorebi.engines import graph_native as gn
from komorebi.engines import stream as st
from komorebi.engines.graph import Graph, check

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = ROOT / "src" / "komorebi_native" / "include"
ENGINE = ROOT / "src" / "komorebi_native" / "x280" / "engines" / "km_graph_engine.c"
REFERENCES = ROOT / "experiments" / "graph_checks" / "graphs"
DEFINE = re.compile(r"^#define\s+(KM_\w+)\s+(0x[0-9A-Fa-f]+|\d+)u?\b", re.M)
H = {
    k: int(v, 0)
    for k, v in DEFINE.findall((INCLUDE / "komorebi/abi/km_graph.h").read_text())
}
needs_cc = pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")


def loop(max_runs=100):
    """The documented example: round after round while the result word is under 65."""
    g = Graph(max_runs=max_runs)
    it, err = g.bindings("it"), g.register()
    a = g.command(1, group=(0, 60), args=[it])
    b = g.command(2, group=(0, 30), args=[it], after=[a])
    c = g.command(3, group=(30, 30), args=[it], after=[a])
    d = g.command(4, group=(0, 60), args=[it], after=[b, c], result=err)
    e = g.cpu(after=[d])
    e.addi(it, it, 1)
    e.lt(g.again, err, g.const(65))
    return g


KEYS = ("run", "node", "handler", "first", "count", "flags")


def trimmed(args):
    args = list(args)
    while args and args[-1] == 0:
        args.pop()
    return args


def sequence_mod(command, m):
    return command.sequence % m


def a0_plus_count(command):
    return command.args[0] + command.count


def test_constants_match_the_header():
    pairs = {
        "MAGIC": "KM_GRAPH_MAGIC",
        "VERSION": "KM_GRAPH_VERSION",
        "MAX_NODES": "KM_GRAPH_MAX_NODES",
        "MAX_EDGES": "KM_GRAPH_MAX_EDGES",
        "MAX_PREDECESSORS": "KM_GRAPH_MAX_PREDECESSORS",
        "MAX_TEMPLATES": "KM_GRAPH_MAX_TEMPLATES",
        "MAX_OPS": "KM_GRAPH_MAX_OPS",
        "MAX_STEPS": "KM_GRAPH_MAX_STEPS",
        "BINDINGS": "KM_GRAPH_BINDINGS",
        "STEP": "KM_GRAPH_STEP",
        "STEP_OPS": "KM_GRAPH_STEP_OPS",
        "MAX_ACTIVE": "KM_GRAPH_MAX_ACTIVE",
        "MAX_KEPT": "KM_GRAPH_MAX_KEPT",
        "NONE": "KM_GRAPH_NONE",
        "NO_NODE": "KM_GRAPH_NO_NODE",
        "NODE_COMMAND": "KM_NODE_COMMAND",
        "NODE_CPU": "KM_NODE_CPU",
        "NODE_JOIN": "KM_NODE_JOIN",
        "GRAPH_RUN_UPLOAD": "KM_GRAPH_RUN_UPLOAD",
        "GRAPH_RUN_KEPT": "KM_GRAPH_RUN_KEPT",
        "EVENT_GRAPH": "KM_EVENT_GRAPH",
        "EVENT_GRAPH_END": "KM_EVENT_GRAPH_END",
        "EVENT_GRAPH_ERROR": "KM_EVENT_GRAPH_ERROR",
        "ERROR_NODE": "KM_STREAM_ERROR_NODE",
        "ERROR_EDGE": "KM_STREAM_ERROR_EDGE",
        "ERROR_RUNS": "KM_STREAM_ERROR_RUNS",
        "ERROR_KEPT_FULL": "KM_STREAM_ERROR_KEPT_FULL",
        "KEEP": "KM_GRAPH_KEEP",
    }
    for name, header_name in pairs.items():
        assert getattr(gr, name) == H[header_name], name
    assert (gr.HEADER.size, gr.NODE.size) == (64, 16)


def test_decode_is_the_inverse_of_encode():
    data = loop().encode()
    assert Graph.decode(data).encode() == data
    with pytest.raises(gr.GraphFormatError):
        Graph.decode(data[:-64])


def test_the_builder_rejects_what_the_check_would():
    g = Graph()
    nodes = [g.join() for _ in range(5)]
    with pytest.raises(ValueError, match="5 predecessors"):
        g.join(after=nodes)
    with pytest.raises(TypeError):
        g.join(after=[Graph().join()])
    with pytest.raises(ValueError, match="either group"):
        g.command(1, group=(0, 1), count=1)
    g.bindings("x")
    with pytest.raises(ValueError, match="missing"):
        g.binding_values()
    with pytest.raises(ValueError):
        Graph(step_limit=gr.MAX_STEPS + 1)


def test_bindings_fill_registers_from_zero_constants_included():
    g = Graph()
    x = g.bindings("x")
    g.register()
    k = g.const(7)
    assert (x.index, k.index) == (0, 2)
    assert g.binding_values(x=5) == [5, 0, 7]


def test_independent_branches_are_submitted_in_the_same_pass():
    r = loop(max_runs=1).simulate({"it": 70}, delay=2, result=a0_plus_count)
    passes = {x.node: x.pass_index for x in r.commands}
    assert passes[1] == passes[2] < passes[3]
    assert r.status == "ended" and r.runs == 1


def test_the_loop_ends_on_the_result_word_and_a_kept_graph_takes_new_bindings():
    g = loop()
    first = g.simulate({"it": 0}, result=a0_plus_count)
    again = g.simulate({"it": 2}, result=a0_plus_count)
    assert (first.runs, first.registers[0]) == (6, 6)
    assert (again.runs, again.registers[0]) == (4, 6)
    rounds = [e.auxiliary for e in first.events if e.kind == gr.EVENT_GRAPH]
    assert rounds == [1, 2, 3, 4, 5]
    assert first.events[-1][:4] == (gr.EVENT_GRAPH_END, 1, 6, 65)


def test_a_skipped_node_counts_as_done_and_submits_nothing():
    g = Graph()
    off = g.bindings("off")
    a = g.command(1, group=(0, 1), skip_if_zero=off)
    g.command(2, group=(1, 1), after=[a])
    r = g.simulate({"off": 0})
    assert [x.node for x in r.commands] == [1] and r.status == "ended"


def test_again_still_set_after_max_runs_is_an_error():
    g = Graph(max_runs=3)
    g.cpu().set(g.again, 1)
    r = g.simulate()
    assert r.error == (gr.ERROR_RUNS, gr.NO_NODE) and r.runs == 3


def test_more_ready_commands_than_slots_wait_their_turn():
    g = Graph()
    for i in range(20):
        g.command(1, group=(i, 1))
    r = g.simulate(delay=3)
    assert len(r.commands) == 20 and max(x.slot for x in r.commands) == 15
    assert r.commands[16].pass_index > r.commands[15].pass_index


def test_check_names_the_node_of_a_slice_with_a_non_compute_opcode():
    g = Graph()
    g.join()
    g.cpu().set(g.register(), 1)
    data = bytearray(g.encode())
    at = gr._layout(dict(zip(gr.HEADER_FIELDS, gr.HEADER.unpack_from(data))))[1]
    data[at] = st.OP_HALT
    assert check(bytes(data)) == (st.ERROR_OPCODE, 1)


@needs_cc
def test_a_command_rejected_after_its_submit_goes_back_behind_the_others():
    g = Graph()
    for i in range(3):
        g.command(1, group=(i, 1))
    data = g.encode()
    for engine in (gr.Executor(), gn.NativeExecutor()):
        submitted = []

        def submit(node, words, submitted=submitted):
            submitted.append(node)
            return gr.SUBMITTED

        engine.start(data)
        engine.step(submit)
        assert (engine.retry(0), engine.retry(0)) == (0, 1)
        engine.step(submit)
        assert submitted == [0, 1, 2, 0]
        assert [engine.complete(n, 0) for n in range(3)] == [0, 0, 0]
        assert engine.step(submit) == gr.END


@needs_cc
def test_the_engine_is_freestanding_with_no_data_of_its_own(tmp_path):
    obj = tmp_path / "engine.o"
    cmd = ["cc", "-std=c11", "-O2", "-ffreestanding", "-Wall", "-Wextra", "-Werror"]
    subprocess.run([*cmd, f"-I{INCLUDE}", "-c", ENGINE, "-o", obj], check=True)
    symbols = subprocess.run(["nm", obj], capture_output=True, text=True).stdout
    kinds = {line.split()[-2] for line in symbols.splitlines()}
    assert not kinds & {"U", "b", "B", "d", "D", "C"}, symbols


@needs_cc
def test_both_executors_give_every_saved_reference_answer():
    lib = gn.load()
    for path in sorted(REFERENCES.glob("*.json")):
        ref = json.loads(path.read_text())
        upload = path.with_suffix(".bin").read_bytes()
        p, want = ref["params"], ref["expect"]
        delay = p["delay"]
        if isinstance(delay, dict):
            delay = functools.partial(sequence_mod, m=delay["mod"])
        assert gn.native_check(upload, lib) == check(upload), path.name
        for engine in (gr.Executor(), gn.NativeExecutor(lib)):
            r = gr.simulate(
                upload,
                bindings=ref["bindings"],
                core_count=p["core_count"],
                delay=delay,
                result=a0_plus_count,
                deadline_passes=p["deadline_passes"],
                cancel_pass=p["cancel_pass"],
                max_passes=p["max_passes"],
                kept=p["kept"],
                engine=engine,
            )
            submitted = [
                dict(zip(KEYS, (x.run, x.node, x.handler, x.first, x.count, x.flags)))
                | {"first": None if x.queued else x.first, "args": trimmed(x.args)}
                for x in sorted(r.commands, key=lambda x: (x.run, x.node))
            ]
            assert submitted == want["submitted"], path.name
            assert [list(e[:4]) for e in r.events] == want["events"], path.name
            assert [r.status, r.registers, r.runs, r.kept] == [
                want["status"],
                want["registers"],
                want["runs"],
                want["kept"],
            ], path.name
