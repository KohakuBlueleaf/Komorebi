"""Graphs: header constants, encoding, check, the executor's semantics, and the C engine
against the saved reference graphs."""

import functools
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from komorebi.engines import graph as graph_module
from komorebi.engines import graph_native as graph_native_module
from komorebi.engines import stream as stream_module
from komorebi.engines.graph import Graph, check

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = ROOT / "src" / "komorebi_native" / "include"
ENGINE = ROOT / "src" / "komorebi_native" / "x280" / "engines" / "km_graph_engine.c"
REFERENCES = ROOT / "experiments" / "checks" / "graph_checks" / "graphs"
DEFINE = re.compile(r"^#define\s+(KM_\w+)\s+(0x[0-9A-Fa-f]+|\d+)u?\b", re.M)
GRAPH_CONSTANTS = {
    k: int(v, 0)
    for k, v in DEFINE.findall((INCLUDE / "komorebi/abi/km_graph.h").read_text())
}
needs_cc = pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")


def loop(max_runs=100):
    """The documented example: round after round while the result word is under 65."""
    graph = Graph(max_runs=max_runs)
    binding, result_register = graph.bindings("it"), graph.register()
    root = graph.command(1, group=(0, 60), args=[binding])
    left = graph.command(2, group=(0, 30), args=[binding], after=[root])
    right = graph.command(3, group=(30, 30), args=[binding], after=[root])
    merge = graph.command(
        4, group=(0, 60), args=[binding], after=[left, right], result=result_register
    )
    update = graph.cpu(after=[merge])
    update.addi(binding, binding, 1)
    update.lt(graph.again, result_register, graph.const(65))
    return graph


KEYS = ("run", "node", "handler", "first", "count", "flags")


def trimmed(args):
    args = list(args)
    while args and args[-1] == 0:
        args.pop()
    return args


def sequence_mod(command, modulus):
    return command.sequence % modulus


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
        assert getattr(graph_module, name) == GRAPH_CONSTANTS[header_name], name
    assert (graph_module.HEADER.size, graph_module.NODE.size) == (64, 16)


def test_decode_is_the_inverse_of_encode():
    data = loop().encode()
    assert Graph.decode(data).encode() == data
    with pytest.raises(graph_module.GraphFormatError):
        Graph.decode(data[:-64])


def test_the_builder_rejects_what_the_check_would():
    graph = Graph()
    nodes = [graph.join() for _ in range(5)]
    with pytest.raises(ValueError, match="5 predecessors"):
        graph.join(after=nodes)
    with pytest.raises(TypeError):
        graph.join(after=[Graph().join()])
    with pytest.raises(ValueError, match="either group"):
        graph.command(1, group=(0, 1), count=1)
    graph.bindings("x")
    with pytest.raises(ValueError, match="missing"):
        graph.binding_values()
    with pytest.raises(ValueError):
        Graph(step_limit=graph_module.MAX_STEPS + 1)


def test_bindings_fill_registers_from_zero_constants_included():
    graph = Graph()
    x = graph.bindings("x")
    graph.register()
    constant = graph.const(7)
    assert (x.index, constant.index) == (0, 2)
    assert graph.binding_values(x=5) == [5, 0, 7]


def test_independent_branches_are_submitted_in_the_same_pass():
    result = loop(max_runs=1).simulate({"it": 70}, delay=2, result=a0_plus_count)
    passes = {x.node: x.pass_index for x in result.commands}
    assert passes[1] == passes[2] < passes[3]
    assert result.status == "ended" and result.runs == 1


def test_the_loop_ends_on_the_result_word_and_a_kept_graph_takes_new_bindings():
    graph = loop()
    first = graph.simulate({"it": 0}, result=a0_plus_count)
    again = graph.simulate({"it": 2}, result=a0_plus_count)
    assert (first.runs, first.registers[0]) == (6, 6)
    assert (again.runs, again.registers[0]) == (4, 6)
    rounds = [
        event.auxiliary
        for event in first.events
        if event.kind == graph_module.EVENT_GRAPH
    ]
    assert rounds == [1, 2, 3, 4, 5]
    assert first.events[-1][:4] == (graph_module.EVENT_GRAPH_END, 1, 6, 65)


def test_a_skipped_node_counts_as_done_and_submits_nothing():
    graph = Graph()
    off = graph.bindings("off")
    first = graph.command(1, group=(0, 1), skip_if_zero=off)
    graph.command(2, group=(1, 1), after=[first])
    result = graph.simulate({"off": 0})
    assert [x.node for x in result.commands] == [1] and result.status == "ended"


def test_again_still_set_after_max_runs_is_an_error():
    graph = Graph(max_runs=3)
    graph.cpu().set(graph.again, 1)
    result = graph.simulate()
    assert (
        result.error == (graph_module.ERROR_RUNS, graph_module.NO_NODE)
        and result.runs == 3
    )


def test_more_ready_commands_than_slots_wait_their_turn():
    graph = Graph()
    slots = stream_module.ring_slots()
    for _ in range(slots + 4):
        graph.command(1, group=(0, 1))
    result = graph.simulate(delay=3)
    assert len(result.commands) == slots + 4
    assert max(x.slot for x in result.commands) == slots - 1
    assert result.commands[slots].pass_index > result.commands[slots - 1].pass_index


def test_check_names_the_node_of_a_slice_with_a_non_compute_opcode():
    graph = Graph()
    graph.join()
    graph.cpu().set(graph.register(), 1)
    data = bytearray(graph.encode())
    offset = graph_module._layout(
        dict(zip(graph_module.HEADER_FIELDS, graph_module.HEADER.unpack_from(data)))
    )[1]
    data[offset] = stream_module.OP_HALT
    assert check(bytes(data)) == (stream_module.ERROR_OPCODE, 1)


@needs_cc
def test_a_command_rejected_after_its_submit_goes_back_behind_the_others():
    graph = Graph()
    for i in range(3):
        graph.command(1, group=(i, 1))
    data = graph.encode()
    for engine in (graph_module.Executor(), graph_native_module.NativeExecutor()):
        submitted = []

        def submit(node, words, submitted=submitted):
            submitted.append(node)
            return graph_module.SUBMITTED

        engine.start(data)
        engine.step(submit)
        assert (engine.retry(0), engine.retry(0)) == (0, 1)
        engine.step(submit)
        assert submitted == [0, 1, 2, 0]
        assert [engine.complete(node_number, 0) for node_number in range(3)] == [
            0,
            0,
            0,
        ]
        assert engine.step(submit) == graph_module.END


def a_thread_between_two_groups(exclusive_first=None):
    """A thread's word (a0 * 10) feeds a grid command; or, with `exclusive_first`, an
    exclusive command on a group from registers, which names a thread."""
    graph = Graph()
    word = graph.register()
    if exclusive_first is not None:
        first = graph.bindings("first")
        graph.command(1, group=(first, graph.const(1)), exclusive=True)
        return graph.encode(), graph.binding_values(first=exclusive_first)
    root = graph.command(1, group=(0, 8), args=[2])
    thread = graph.command(9, cluster=3, args=[5], after=[root], result=word)
    graph.command(2, group=(8, 8), args=[word], after=[thread])
    return graph.encode(), []


@needs_cc
def test_both_executors_take_a_thread_as_a_commands_target():
    for engine in (graph_module.Executor(), graph_native_module.NativeExecutor()):
        upload, bindings = a_thread_between_two_groups()
        done = graph_module.simulate(
            upload, result=lambda command: command.args[0] * 10, engine=engine
        )
        assert done.status == "ended"
        assert [(c.handler, c.first, c.count) for c in done.commands] == [
            (1, 0, 8),
            (9, 0xFF03, 1),
            (2, 8, 8),
        ]
        assert done.commands[2].args[0] == 50
        upload, bindings = a_thread_between_two_groups(exclusive_first=0xFF02)
        done = graph_module.simulate(upload, bindings=bindings, engine=engine)
        assert done.error == (stream_module.ERROR_GROUP, 0) and not done.commands
    graph = Graph()
    with pytest.raises(ValueError, match="exclusive"):
        graph.command(9, cluster=1, exclusive=True)
    with pytest.raises(ValueError, match="not both"):
        graph.command(9, cluster=1, count=1)


@needs_cc
def test_the_engine_is_freestanding_with_no_data_of_its_own(tmp_path):
    object_file = tmp_path / "engine.o"
    command_line = [
        "cc",
        "-std=c11",
        "-O2",
        "-ffreestanding",
        "-Wall",
        "-Wextra",
        "-Werror",
    ]
    subprocess.run(
        [*command_line, f"-I{INCLUDE}", "-c", ENGINE, "-o", object_file], check=True
    )
    symbols = subprocess.run(["nm", object_file], capture_output=True, text=True).stdout
    kinds = {line.split()[-2] for line in symbols.splitlines()}
    assert not kinds & {"U", "b", "B", "d", "D", "C"}, symbols


@needs_cc
def test_both_executors_give_every_saved_reference_answer():
    lib = graph_native_module.load()
    for path in sorted(REFERENCES.glob("*.json")):
        ref = json.loads(path.read_text())
        upload = path.with_suffix(".bin").read_bytes()
        parameters, want = ref["params"], ref["expect"]
        delay = parameters["delay"]
        if isinstance(delay, dict):
            delay = functools.partial(sequence_mod, modulus=delay["mod"])
        assert graph_native_module.native_check(upload, lib) == check(upload), path.name
        for engine in (
            graph_module.Executor(),
            graph_native_module.NativeExecutor(lib),
        ):
            result = graph_module.simulate(
                upload,
                bindings=ref["bindings"],
                core_count=parameters["core_count"],
                delay=delay,
                result=a0_plus_count,
                deadline_passes=parameters["deadline_passes"],
                cancel_pass=parameters["cancel_pass"],
                max_passes=parameters["max_passes"],
                kept=parameters["kept"],
                engine=engine,
            )
            submitted = [
                dict(zip(KEYS, (x.run, x.node, x.handler, x.first, x.count, x.flags)))
                | {"first": None if x.queued else x.first, "args": trimmed(x.args)}
                for x in sorted(result.commands, key=lambda x: (x.run, x.node))
            ]
            assert submitted == want["submitted"], path.name
            assert [list(event[:4]) for event in result.events] == want[
                "events"
            ], path.name
            assert [result.status, result.registers, result.runs, result.kept] == [
                want["status"],
                want["registers"],
                want["runs"],
                want["kept"],
            ], path.name
